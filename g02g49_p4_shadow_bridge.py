#!/usr/bin/env python3
"""G02-G49/P4 : pont de collecte isole, sans reseau ni sauvegarde."""

from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from g02g49_prospective_contract import (
    FEATURES_PRE87,
    make_capture,
    parse_utc,
)


def stamp(value):
    return parse_utc(value).isoformat().replace("+00:00", "Z")


class ShadowBridge:
    def __init__(self, mode="off"):
        if mode not in ("off", "dry_run"):
            raise ValueError("Mode non autorise")

        self.mode = mode
        self._cycles = {}
        self._lock = Lock()
        self._ctx = ContextVar(
            "g02g49_shadow_context",
            default=None,
        )

    def register_listing(
        self, cycle_id, started_utc, received_utc, matches
    ):
        if self.mode == "off":
            return

        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("Identifiant du cycle invalide")

        started = parse_utc(started_utc)
        received = parse_utc(received_utc)

        if received < started:
            raise ValueError("Horaires de liste inverses")

        safe_matches = {}

        for item in matches:
            mid = item["match_id"]
            minute = item["minute"]

            if not isinstance(mid, str) or not mid:
                raise ValueError("ID match invalide")

            if mid in safe_matches:
                raise ValueError("ID duplique dans la liste")

            if (
                isinstance(minute, bool)
                or not isinstance(minute, int)
            ):
                raise ValueError("Minute invalide")

            if not 80 <= minute < 87:
                continue

            updated = item.get("provider_updated_at")

            safe_matches[mid] = {
                "minute": minute,
                "provider_updated_at": (
                    stamp(updated) if updated is not None else None
                ),
            }

        with self._lock:
            if cycle_id in self._cycles:
                raise ValueError("Cycle deja enregistre")

            self._cycles[cycle_id] = {
                "started": stamp(started_utc),
                "received": stamp(received_utc),
                "matches": safe_matches,
            }

    @contextmanager
    def match_scope(self, cycle_id, match_id):
        if self.mode == "off":
            yield None
            return

        if self._ctx.get() is not None:
            raise ValueError("Contexte imbrique interdit")

        with self._lock:
            cycle = self._cycles.get(cycle_id)
            match = (
                cycle["matches"].get(match_id)
                if cycle else None
            )

        if match is None:
            raise ValueError(
                "Match absent ou minute non autorisee"
            )

        trace = {
            "match_id": match_id,
            "minute": match["minute"],
            "listing_received": cycle["received"],
            "requests": [{
                "source": "live_matches",
                "started_utc": cycle["started"],
                "received_utc": cycle["received"],
                "provider_updated_at": match[
                    "provider_updated_at"
                ],
            }],
            "attempts": [],
        }

        token = self._ctx.set(trace)

        try:
            yield trace
        finally:
            self._ctx.reset(token)

    def observe_live_stats(
        self, match_id, started_utc,
        received_utc, http_status
    ):
        trace = self._ctx.get()

        if trace is None or trace["match_id"] != match_id:
            raise ValueError(
                "Requete hors contexte ou mauvais match"
            )

        started = parse_utc(started_utc)
        received = parse_utc(received_utc)

        if (
            started < parse_utc(trace["listing_received"])
            or received < started
        ):
            raise ValueError(
                "Chronologie des requetes incoherente"
            )

        if (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
        ):
            raise ValueError("Statut HTTP invalide")

        trace["attempts"].append(http_status)

        # Seules les reponses HTTP reussies peuvent
        # devenir une source des features.
        if 200 <= http_status < 300:
            trace["requests"].append({
                "source": "live_stats",
                "started_utc": stamp(started_utc),
                "received_utc": stamp(received_utc),
                "provider_updated_at": None,
            })

    def build(self, match_id, captured_utc, features):
        trace = self._ctx.get()

        if trace is None or trace["match_id"] != match_id:
            raise ValueError(
                "Capture hors contexte ou mauvais match"
            )

        return make_capture(
            match_id,
            trace["minute"],
            captured_utc,
            trace["requests"],
            features,
        )


def selftest():
    print(
        "=== G02-G49/P4 | PONT PROSPECTIF SHADOW | SELFTEST ==="
    )

    inactive = ShadowBridge()

    inactive.register_listing(
        "off", "nonsense", "nonsense", []
    )

    with inactive.match_scope("off", "any") as value:
        assert value is None

    print("PASS | mode OFF par defaut")

    bridge = ShadowBridge(mode="dry_run")

    start = "2026-11-08T12:00:00Z"
    received = "2026-11-08T12:00:01Z"

    matches = [
        {
            "match_id": f"mt_SYNTH_{i:03d}",
            "minute": 85,
            "provider_updated_at":
                "2026-11-08T11:59:59Z",
        }
        for i in range(24)
    ]

    bridge.register_listing(
        "cycle_A", start, received, matches
    )

    try:
        bridge.register_listing(
            "cycle_A", start, received, matches
        )
    except ValueError:
        print("PASS | reutilisation du cycle refusee")
    else:
        raise AssertionError("Cycle duplique accepte")

    def worker(i):
        mid = f"mt_SYNTH_{i:03d}"

        with bridge.match_scope("cycle_A", mid) as trace:

            if i == 0:
                try:
                    bridge.observe_live_stats(
                        "mt_WRONG",
                        "2026-11-08T12:00:02Z",
                        "2026-11-08T12:00:03Z",
                        200,
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "Mauvais ID accepte"
                    )

            if i % 6 == 0:
                bridge.observe_live_stats(
                    mid,
                    "2026-11-08T12:00:02Z",
                    "2026-11-08T12:00:03Z",
                    429,
                )

            bridge.observe_live_stats(
                mid,
                "2026-11-08T12:00:03Z",
                "2026-11-08T12:00:04Z",
                200,
            )

            capture = bridge.build(
                mid,
                "2026-11-08T12:00:05Z",
                dict.fromkeys(FEATURES_PRE87, 0),
            )

            assert len(capture["requests"]) == 2

            assert (
                capture["requests"][0]["provider_updated_at"]
                == "2026-11-08T11:59:59Z"
            )

            assert (
                capture["requests"][1]["provider_updated_at"]
                is None
            )

            return capture, list(trace["attempts"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(24)))

    assert len({
        capture["match_id"]
        for capture, _ in results
    }) == 24

    assert sum(
        status == 429
        for _, attempts in results
        for status in attempts
    ) == 4

    assert all(
        not capture["asof_certified"]
        and not capture["labels_included"]
        for capture, _ in results
    )

    assert bridge._ctx.get() is None

    print("PASS | 24 matchs / 8 workers sans contamination")
    print("PASS | 4 erreurs HTTP 429 exclues des captures")
    print("PASS | timestamp fournisseur non certifie")
    print("PASS | aucun appel API et aucune sauvegarde")
    print("G02-G49/P4 SHADOW_BRIDGE_SELFTEST PASS")
    print(
        "SCANNER INCHANGE | 0 LABEL | 0 FIT | "
        "AS-OF NON CERTIFIE"
    )


if __name__ == "__main__":
    selftest()
