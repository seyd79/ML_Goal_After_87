#!/usr/bin/env python3
"""G02-G49 : contrat de capture prospectif, indépendant du scanner."""

from datetime import datetime, timezone
from pathlib import Path
from math import isfinite
import json
import os
import sys
import tempfile

FEATURES_PRE87 = (
    "score_home_snapshot",
    "score_away_snapshot",
    "shots_80_to_capture",
    "sot_80_to_capture",
    "corners_80_to_capture",
    "home_sot_cumulative_snapshot",
    "away_sot_cumulative_snapshot",
    "minutes_since_last_sot_snapshot",
)

SOURCES = (
    "live_matches",
    "live_stats",
    "match_detail",
    "match_stats",
)


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("Horodatage non textuel")

    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Fuseau horaire manquant")

    return dt.astimezone(timezone.utc)


def as_iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def make_capture(match_id, minute, captured_utc, requests, features):
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("Identifiant absent")

    if (
        isinstance(minute, bool)
        or not isinstance(minute, int)
        or not 80 <= minute < 87
    ):
        raise ValueError("Minute hors fenetre [80,87[")

    captured = parse_utc(captured_utc)

    if not isinstance(requests, list) or not requests:
        raise ValueError("Sources absentes")

    if (
        not isinstance(features, dict)
        or set(features) != set(FEATURES_PRE87)
    ):
        raise ValueError("Features manquantes ou non autorisees")

    clean_features = {}

    for key in FEATURES_PRE87:
        value = features[key]

        if value is None:
            clean_features[key] = None

        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError("Feature non numerique : " + key)

        else:
            clean_features[key] = float(value)

    clean_requests = []

    for request in requests:
        if (
            not isinstance(request, dict)
            or request.get("source") not in SOURCES
        ):
            raise ValueError("Source non autorisee")

        started = parse_utc(request["started_utc"])
        received = parse_utc(request["received_utc"])

        if not started <= received <= captured:
            raise ValueError("Chronologie incoherente")

        provider_time = request.get("provider_updated_at")

        provider_iso = (
            as_iso(parse_utc(provider_time))
            if provider_time is not None
            else None
        )

        clean_requests.append({
            "source": request["source"],
            "started_utc": as_iso(started),
            "received_utc": as_iso(received),
            "latency_ms": round(
                (received - started).total_seconds() * 1000, 3
            ),
            "age_at_snapshot_ms": round(
                (captured - received).total_seconds() * 1000, 3
            ),
            "provider_updated_at": provider_iso,
            "provider_time_semantics": "UNVERIFIED",
        })

    source_names = {
        request["source"]
        for request in clean_requests
    }

    if not {"live_matches", "live_stats"}.issubset(source_names):
        raise ValueError("Sources live obligatoires absentes")

    return {
        "schema": "G02G49_V1",
        "match_id": match_id,
        "minute": minute,
        "captured_utc": as_iso(captured),
        "requests": clean_requests,
        "features_pre87": clean_features,
        "asof_certified": False,
        "labels_included": False,
        "research_only": True,
    }


def append_new_log(path, record):
    """Ecriture JSONL pour une future source prospective distincte."""
    destination = Path(path)

    if destination.exists() and not destination.is_file():
        raise ValueError("Destination invalide")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def selftest():
    print("=== G02-G49/P1 | TEST DU CONTRAT PROSPECTIF ===")

    requests = [
        {
            "source": "live_matches",
            "started_utc": "2026-11-08T12:00:00Z",
            "received_utc": "2026-11-08T12:00:01Z",
            "provider_updated_at": "2026-11-08T11:59:59Z",
        },
        {
            "source": "live_stats",
            "started_utc": "2026-11-08T12:00:01Z",
            "received_utc": "2026-11-08T12:00:02Z",
            "provider_updated_at": None,
        },
    ]

    features = dict.fromkeys(FEATURES_PRE87, 0)

    record = make_capture(
        "SYNTHETIC_ONLY",
        85,
        "2026-11-08T12:00:03Z",
        requests,
        features,
    )

    assert record["requests"][1]["age_at_snapshot_ms"] == 1000
    assert record["asof_certified"] is False
    assert "goal_after_87" not in json.dumps(record)

    print("PASS | capture 85 et horodatages")

    def must_fail(name, function):
        try:
            function()
        except (ValueError, KeyError):
            print("PASS | rejet :", name)
        else:
            raise AssertionError("Echec du controle : " + name)

    must_fail(
        "minute 87",
        lambda: make_capture(
            "X", 87, "2026-11-08T12:00:03Z",
            requests, features
        )
    )

    must_fail(
        "horloge inversee",
        lambda: make_capture(
            "X", 85, "2026-11-08T12:00:00Z",
            requests, features
        )
    )

    must_fail(
        "injection d'un label",
        lambda: make_capture(
            "X", 85, "2026-11-08T12:00:03Z",
            requests, {**features, "goal_after_87": 1}
        )
    )

    must_fail(
        "source live absente",
        lambda: make_capture(
            "X", 85, "2026-11-08T12:00:03Z",
            requests[:1], features
        )
    )

    must_fail(
        "fuseau horaire absent",
        lambda: make_capture(
            "X", 85, "2026-11-08T12:00:03",
            requests, features
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.jsonl"

        append_new_log(path, record)

        loaded = json.loads(
            path.read_text(encoding="utf-8").strip()
        )

        assert loaded == record
        print("PASS | journal temporaire synthetique")

    print("G02-G49/P1 PROSPECTIVE_CONTRACT_SELFTEST PASS")
    print("0 API | 0 LABEL REEL | 0 FIT")
    print("SCANNER LIVE NON MODIFIE")


if __name__ == "__main__":
    if sys.argv[1:] != ["--selftest"]:
        raise SystemExit(
            "Usage : python3 -B "
            "g02g49_prospective_contract.py --selftest"
        )

    selftest()
