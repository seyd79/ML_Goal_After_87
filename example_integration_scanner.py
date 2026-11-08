# -*- coding: utf-8 -*-
"""Example only: adapt in the real scanner after inspecting its event API format.

This reuses the timeline payload ALREADY fetched by the scanner; no new API GET.
Call ONLY during a live match, when minute is 85/86/87, before making the
snapshot decision. The caller must record the timestamp AFTER the timeline is
received. Nothing here changes the production model or its 96-feature schema.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from v3c_pressure_features import build_features


def add_pressure_sidecar(match_id, minute, home_team, away_team,
                         timeline_payload, out_file="outputs/history/v3c_pressure_captures.jsonl",
                         home_id=None, away_id=None, match_is_live=False,
                         shots_corners_schema_verified=False):
    # Caller is responsible for deciding whether the match is genuinely live.
    if minute not in (85, 86, 87):
        return None
    if not match_is_live:
        # Never log historical results as a valid real-time record.
        return None
    capture_time = datetime.now(timezone.utc).isoformat()
    pressure = build_features(timeline_payload, match_id, minute, home_team,
                              away_team, source="LIVE_ASOF",
                              home_id=home_id, away_id=away_id,
                              schema_confirmed=shots_corners_schema_verified)
    record = {
        "match_id": match_id, "minute": minute, "home_team": home_team,
        "away_team": away_team, "home_id": home_id, "away_id": away_id,
        "source": "LIVE_ASOF", "captured_at_utc": capture_time,
        "live_status_verified": True, # requires existing scanner live status
        "shots_corners_schema_verified": shots_corners_schema_verified,
        "timeline_payload": timeline_payload,
        "computed_features": pressure,
    }
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Integrate new columns into your snapshot only after confirming exact
    # compatibility. Until then, keep pressure as a parallel (shadow) log.
    return pressure
