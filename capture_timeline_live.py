#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot, append-only live TheStatsAPI timeline capture at minutes 85/86/87.

Call while the match is actually LIVE. DO NOT use this to fetch old timelines
and call them live: past timelines are hindsight and are not as-of data.

The single HTTP GET is deliberately explicit, not a 10-second background poll.
Python 3.9+ stdlib only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import build_features


def env_file(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in ("STATS_API_KEY", "STATS_API_BASE_URL") and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--minute", type=int, required=True, choices=(85, 86, 87))
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home-id")
    ap.add_argument("--away-id")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("--output", type=Path, default=Path("outputs/history/v3c_pressure_captures.jsonl"))
    args = ap.parse_args()
    env_file(args.env)
    key = (os.environ.get("STATS_API_KEY") or "").strip()
    if not key:
        ap.error("STATS_API_KEY manquante: définir la variable ou un .env")
    if key.lower().startswith("bearer "):
        key = key[7:]
    base = (os.environ.get("STATS_API_BASE_URL") or "https://api.thestatsapi.com/api").rstrip("/")
    url = base + "/football/matches/" + args.match_id + "/timeline"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        print(f"API HTTP {e.code}")
        print(f"Détails : {details[:1000]}")
        raise SystemExit("Capture non enregistrée")
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise SystemExit("API timeline indisponible: %s" % e)
    captured = datetime.now(timezone.utc).isoformat()
    record = dict(
        match_id=args.match_id,
        minute=args.minute,
        home_team=args.home,
        away_team=args.away,
        home_id=args.home_id,
        away_id=args.away_id,
        source="LIVE_ASOF",
        captured_at_utc=captured,
        live_status_verified=False,
        shots_corners_schema_verified=False,
        timeline_payload=payload,
    )
    computed = build_features(
        payload,
        args.match_id,
        args.minute,
        args.home,
        args.away,
        source="LIVE_ASOF",
        home_id=args.home_id,
        away_id=args.away_id,
        schema_confirmed=False,
    )
    record["computed_features"] = computed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"capture_file": str(args.output), "captured_at_utc": captured, **computed},
                     ensure_ascii=False, indent=2))
    if computed["pressure_status"] != "VALID_LIVE_ASOF":
        print("ATTENTION: inconnu/non classable: ne pas remplacer les NULL par 0.", file=sys.stderr)
    print("REMARQUE: live_status_verified=False; vérifier que le match était live à la capture.", file=sys.stderr)


if __name__ == "__main__":
    main()
