#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect whether a real API timeline includes shot/corner events.
Does not certify completeness or change schema verification flags.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from v3c_pressure_features import get_events, event_category, event_minute, team_side, capture_to_row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=8)
    args = ap.parse_args()
    summaries = []
    with args.captures.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            cap = json.loads(line)
            evts = get_events(cap.get("timeline_payload"))
            kinds = Counter()
            types = Counter()
            unassigned = 0
            if evts is not None:
                for e in evts:
                    kinds[event_category(e) or "other_or_unknown"] += 1
                    raw = e.get("type") or e.get("event_type") or e.get("name") or "<missing>"
                    if isinstance(raw, dict):
                        raw = raw.get("name") or str(raw)
                    types[str(raw)] += 1
                    if event_category(e) and team_side(e, cap.get("home_team",""), cap.get("away_team",""), cap.get("home_id"), cap.get("away_id")) is None:
                        unassigned += 1
            features = capture_to_row(cap)
            summaries.append(dict(match_id=cap.get("match_id"), minute=cap.get("minute"),
                                  timeline_present=evts is not None,
                                  timeline_events=len(evts) if evts is not None else None,
                                  recognized_event_kinds=dict(kinds),
                                  top_raw_event_types=types.most_common(12),
                                  offensive_events_without_team=unassigned,
                                  features_status=features["pressure_status"],
                                  live_status_verified=cap.get("live_status_verified"),
                                  shots_corners_schema_verified=cap.get("shots_corners_schema_verified")))
    print(json.dumps(dict(n_captures=len(summaries), examples=summaries[:args.max_examples],
                          note="Counted event types alone do not prove completeness. Compare with real match statistics before enabling schema verification."), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
