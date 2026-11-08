#!/usr/bin/env python3
"""G02-G49/P2 : test de traçabilité HTTP sans réseau."""

from datetime import datetime, timezone
from urllib.parse import urlsplit
import json
import re

from g02g49_prospective_contract import (
    FEATURES_PRE87,
    make_capture,
)


def iso_utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Horloge avec fuseau horaire requise")

    return value.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def classify_endpoint(url):
    """Ne conserve ni URL complète ni paramètres sensibles."""
    path = urlsplit(url).path.rstrip("/")

    if path == "/football/matches":
        return "live_matches", None

    match = re.fullmatch(
        r"/football/matches/([A-Za-z0-9_-]+)"
        r"(?:/(live-stats|stats))?",
        path,
    )

    if not match:
        return None, None

    match_id, suffix = match.groups()

    source = {
        "live-stats": "live_stats",
        "stats": "match_stats",
        None: "match_detail",
    }[suffix]

    return source, match_id


class HttpTimingProbe:
    def __init__(self, transport, utc_now=None):
        self._transport = transport
        self._utc_now = utc_now or (
            lambda: datetime.now(timezone.utc)
        )
        self.records = []

    def request(self, method, url, **kwargs):
        """Mesure un appel HTTP sans inspecter son contenu."""
        source, match_id = classify_endpoint(url)
        started = self._utc_now()

        try:
            response = self._transport(
                method=method,
                url=url,
                **kwargs,
            )
        except Exception:
            received = self._utc_now()

            if source:
                self.records.append({
                    "source": source,
                    "match_id": match_id,
                    "started_utc": iso_utc(started),
                    "received_utc": iso_utc(received),
                    "http_status": None,
                    "success": False,
                    "provider_updated_at": None,
                })
            raise

        received = self._utc_now()

        if source:
            status = getattr(response, "status_code", None)

            self.records.append({
                "source": source,
                "match_id": match_id,
                "started_utc": iso_utc(started),
                "received_utc": iso_utc(received),
                "http_status": status if isinstance(status, int) else None,
                "success": (
                    isinstance(status, int)
                    and 200 <= status < 300
                ),
                "provider_updated_at": None,
            })

        return response


def selftest():
    print("=== G02-G49/P2 | TRACE HTTP | SELFTEST SANS RESEAU ===")

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status

    calls = []
    fake_response = FakeResponse(200)

    def fake_transport(method, url, **kwargs):
        calls.append((method, url))
        return fake_response

    timestamps = iter(
        datetime.fromisoformat(value)
        for value in (
            "2026-11-08T12:00:00+00:00",
            "2026-11-08T12:00:01+00:00",
            "2026-11-08T12:00:01+00:00",
            "2026-11-08T12:00:02+00:00",
            "2026-11-08T12:00:02+00:00",
            "2026-11-08T12:00:03+00:00",
        )
    )

    probe = HttpTimingProbe(
        fake_transport,
        utc_now=lambda: next(timestamps),
    )

    response1 = probe.request(
        "GET",
        "https://test.invalid/football/matches"
        "?secret=DO_NOT_LOG",
    )

    response2 = probe.request(
        "GET",
        "https://test.invalid/football/matches/"
        "mt_123/live-stats",
        params={"token": "DO_NOT_LOG"},
    )

    probe.request(
        "GET",
        "https://test.invalid/untracked?token=DO_NOT_LOG",
    )

    assert response1 is fake_response
    assert response2 is fake_response
    assert len(calls) == 3
    assert len(probe.records) == 2

    assert [
        record["source"]
        for record in probe.records
    ] == ["live_matches", "live_stats"]

    assert probe.records[1]["match_id"] == "mt_123"
    assert "DO_NOT_LOG" not in json.dumps(probe.records)
    assert all(r["success"] for r in probe.records)

    print("PASS | requetes simulees tracees sans secret")

    requests = [
        {
            key: record[key]
            for key in (
                "source",
                "started_utc",
                "received_utc",
                "provider_updated_at",
            )
        }
        for record in probe.records
    ]

    capture = make_capture(
        "mt_123",
        85,
        "2026-11-08T12:00:04Z",
        requests,
        dict.fromkeys(FEATURES_PRE87, 0),
    )

    assert capture["asof_certified"] is False
    assert capture["labels_included"] is False
    assert len(capture["requests"]) == 2

    print("PASS | compatibilite contrat G02-G49/P1")

    statuses = iter([429, 200])

    def fake_retry(method, url, **kwargs):
        return FakeResponse(next(statuses))

    retry_times = iter(
        datetime.fromisoformat(value)
        for value in (
            "2026-11-08T12:01:00+00:00",
            "2026-11-08T12:01:01+00:00",
            "2026-11-08T12:01:02+00:00",
            "2026-11-08T12:01:03+00:00",
        )
    )

    retry_probe = HttpTimingProbe(
        fake_retry,
        utc_now=lambda: next(retry_times),
    )

    url = (
        "https://test.invalid/football/matches/"
        "mt_123/live-stats"
    )

    retry_probe.request("GET", url)
    retry_probe.request("GET", url)

    assert [
        record["http_status"]
        for record in retry_probe.records
    ] == [429, 200]

    assert [
        record["success"]
        for record in retry_probe.records
    ] == [False, True]

    print("PASS | HTTP 429 identifie comme echec")

    print("G02-G49/P2 HTTP_TIMING_PROBE_SELFTEST PASS")
    print("0 API REELLE | 0 LABEL | 0 FIT")
    print("SCANNER INCHANGE | AS-OF NON CERTIFIE")


if __name__ == "__main__":
    selftest()
