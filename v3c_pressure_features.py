#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3-C pressure features v1, Python >=3.9, stdlib only.

Strict timeline policy:
- Five COMPLETED minutes: [snapshot_minute - 5, snapshot_minute).
- No current-minute or future-minute events.
- Events only from an actual timeline response; no imputation from totals.
- Match/team ambiguity -> six NULL features, never fabricated zeroes.
- Historical/full-time timelines have provenance HISTORICAL_DIAGNOSTIC_ONLY.
- Live timeline fetched at capture time is LIVE_ASOF and may be used for later
  training ONLY when the collector truly ran at that live minute.
- The source API's event schema is not guaranteed; inspect unknown-type counts.

Usage:
    python v3c_pressure_features.py inspect --timeline sample.json --minute 87 \
       --match-id mt_x --home 'Team A' --away 'Team B'
    python v3c_pressure_features.py audit --snapshots snapshots.csv
    python v3c_pressure_features.py merge --snapshots snapshots.csv \
       --captures outputs/history/v3c_pressure_captures.jsonl \
       --output outputs/history/snapshots_pressure_v1.csv
"""
import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

VERSION = "pressure_asof_v1_completed_minutes"
FEATURES = (
    "shots_home_last5", "shots_away_last5", "sot_home_last5",
    "sot_away_last5", "corners_home_last5", "corners_away_last5",
)


def normalized(value: Any) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s.casefold()).strip("_")


def parse_minute(value: Any) -> Optional[int]:
    """Accept 87, '90+4', {'elapsed': 90,'extra':4}; reject unspecified."""
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        base = next((value[k] for k in ("elapsed", "regular", "minute", "minutes")
                     if value.get(k) is not None), None)
        extra = next((value[k] for k in ("extra", "added", "stoppage")
                      if value.get(k) is not None), 0)
        if base is None:
            return None
        base_min = parse_minute(base)
        extra_min = parse_minute(extra)
        return (base_min + extra_min) if base_min is not None and extra_min is not None else None
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value < 0 or value != int(value):
            return None
        return int(value)
    s = str(value or "").strip()
    m = re.fullmatch(r"(\d{1,3})\s*(?:\+\s*(\d{1,2}))?\s*(?:'|min)?", s, re.I)
    return int(m.group(1)) + int(m.group(2) or 0) if m else None


def event_minute(event: Mapping[str, Any]) -> Optional[int]:
    for k in ("time", "minute"):
        if k in event:
            m = parse_minute(event[k])
            if m is not None:
                return m
    for k in ("elapsed", "elapsed_minutes", "match_minute"):
        if k in event:
            m = parse_minute(event[k])
            extra = next((event[kk] for kk in ("extra", "added_time", "stoppage_time", "extra_minute")
                         if event.get(kk) is not None), 0)
            x = parse_minute(extra)
            if m is not None and x is not None:
                return m + x
    return None


def get_events(payload: Any) -> Optional[List[Mapping[str, Any]]]:
    """None means no recognized event list; [] is a valid empty event list."""
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if not isinstance(payload, dict):
        return None
    for k in ("events", "timeline", "data", "result", "response"):
        v = payload.get(k)
        if isinstance(v, list):
            return [e for e in v if isinstance(e, dict)]
        if isinstance(v, dict):
            sub = get_events(v)
            if sub is not None:
                return sub
    return None


def event_category(event: Mapping[str, Any]) -> Optional[str]:
    """Classify positively identified events. Do not infer goals as shots.

    Unknown event schemas should be audited, not mapped to zeros silently.
    """
    def field_tokens(keys):
        tokens = []
        for key in keys:
            v = event.get(key)
            if isinstance(v, dict):
                tokens.extend(normalized(w) for w in v.values() if not isinstance(w, (dict, list)))
            elif isinstance(v, (str, int)):
                tokens.append(normalized(v))
        return "_".join(t for t in tokens if t)
    kind = field_tokens(("type", "event_type", "event", "name", "category"))
    details = field_tokens(("detail", "outcome", "result", "shot_type", "status"))
    joined = "_".join((kind, details))
    # Disallowed, penalty shootout, extra-time and VAR are excluded.
    if any(t in joined for t in ("disallowed", "cancelled", "canceled", "shootout", "penalty_shootout", "extra_time", "aet", "var_review")):
        return None
    if "corner" in kind or "corner_kick" in kind or "corner" in details:
        return "corner"
    is_shot = any(t in kind for t in ("shot", "attempt", "tir", "on_target", "off_target"))
    if not is_shot:
        return None
    targeted = (event.get("on_target") is True or event.get("is_on_target") is True
                or any(t in joined for t in ("on_target", "on_goal", "sot", "shot_saved", "saved_shot")))
    if targeted:
        return "sot"
    return "shot"


def team_side(event: Mapping[str, Any], home_team: str, away_team: str,
              home_id: Optional[str] = None, away_id: Optional[str] = None) -> Optional[str]:
    """Only explicit team IDs/names/side indicators; never assume home/away."""
    team = next((event[k] for k in ("team", "team_id", "teamId", "team_name", "teamName")
                 if event.get(k) is not None), None)
    if isinstance(team, dict):
        team_id = str(team.get("id") or team.get("team_id") or "")
        name = team.get("name") or team.get("team_name")
        side = team.get("side") or team.get("location")
    else:
        team_id, name, side = str(team or ""), team, None
    side = side or event.get("side") or event.get("team_side")
    if normalized(side) in ("home", "away"):
        return normalized(side)
    if home_id and team_id == str(home_id):
        return "home"
    if away_id and team_id == str(away_id):
        return "away"
    value = normalized(name)
    if value and value == normalized(home_team) and normalized(home_team) != normalized(away_team):
        return "home"
    if value and value == normalized(away_team) and normalized(home_team) != normalized(away_team):
        return "away"
    return None


def base_row(match_id: str, minute: int, source: str) -> Dict[str, Any]:
    return dict(match_id=match_id, minute=minute, feature_logic_version=VERSION,
                pressure_source=source, events_available_at_snapshot=False,
                pressure_status="UNKNOWN", events_timeline_count=None,
                events_in_window=None, events_unknown_team=None,
                events_unknown_type=None, events_future_filtered=None,
                events_invalid_minute=None, **{k: None for k in FEATURES})


def build_features(payload: Any, match_id: str, snapshot_minute: int,
                   home_team: str, away_team: str,
                   source: str = "HISTORICAL_DIAGNOSTIC_ONLY",
                   home_id: Optional[str] = None,
                   away_id: Optional[str] = None,
                   schema_confirmed: bool = False) -> Dict[str, Any]:
    minute = parse_minute(snapshot_minute)
    if minute not in (85, 86, 87):
        raise ValueError("snapshot_minute doit être 85, 86 ou 87")
    row = base_row(str(match_id), minute, source)
    evts = get_events(payload)
    if evts is None:
        row["pressure_status"] = "NO_TIMELINE_LIST"
        return row
    row["events_timeline_count"] = len(evts)
    row["events_in_window"] = 0
    row["events_unknown_team"] = 0
    row["events_unknown_type"] = 0
    row["events_future_filtered"] = 0
    row["events_invalid_minute"] = 0
    counts = Counter()
    # Unique provider event ID: duplicates of SAME ID in payload ignored.
    seen_ids = set()
    for e in evts:
        identifier = e.get("id") or e.get("event_id")
        if identifier is not None:
            if str(identifier) in seen_ids:
                continue
            seen_ids.add(str(identifier))
        # No extra-time / shootout periods.
        period = normalized(e.get("period") or e.get("match_period") or e.get("phase") or "")
        if any(x in period for x in ("extra_time", "shootout", "penalties", "aet")):
            continue
        m = event_minute(e)
        if m is None:
            row["events_invalid_minute"] += 1
            continue
        if m >= minute:
            row["events_future_filtered"] += 1
            continue
        if m < minute - 5:
            continue
        kind = event_category(e)
        if kind is None:
            row["events_unknown_type"] += 1
            continue
        side = team_side(e, home_team, away_team, home_id, away_id)
        if side is None:
            row["events_unknown_team"] += 1
            continue
        row["events_in_window"] += 1
        if kind == "sot":
            counts[f"sot_{side}_last5"] += 1
            counts[f"shots_{side}_last5"] += 1
        elif kind == "shot":
            counts[f"shots_{side}_last5"] += 1
        elif kind == "corner":
            counts[f"corners_{side}_last5"] += 1
    # Fail closed: if potentially relevant events cannot be attributed, all
    # features stay NULL to avoid deceptively low zeros.
    if row["events_unknown_team"] or row["events_invalid_minute"]:
        row["pressure_status"] = "INCOMPLETE_UNKNOWN_TEAM_OR_TIME"
        return row
    if source == "LIVE_ASOF" and not schema_confirmed:
        # The provider may expose only goals/cards, not shot/corner events.
        # Do not interpret zero observed shots as zero actual shots.
        row["pressure_status"] = "SHOT_CORNER_EVENT_COVERAGE_UNVERIFIED"
        return row
    if source == "LIVE_ASOF" and row["events_future_filtered"]:
        # A real live snapshot should not contain events in later minutes.
        # Treat clock mismatch or accidentally fetched full-time timeline as unsafe.
        row["pressure_status"] = "SUSPECT_LIVE_TIMELINE_CONTAINS_FUTURE"
        return row
    for k in FEATURES:
        row[k] = counts[k]
    row["pressure_status"] = "VALID_" + source
    row["events_available_at_snapshot"] = source == "LIVE_ASOF"
    return row


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def capture_to_row(capture: Dict[str, Any]) -> Dict[str, Any]:
    source = capture.get("source", "HISTORICAL_DIAGNOSTIC_ONLY")
    if source == "LIVE_ASOF" and capture.get("live_status_verified") is not True:
        source = "LIVE_UNVERIFIED"
    row = build_features(capture.get("timeline_payload"), capture["match_id"],
                         capture["minute"], capture["home_team"], capture["away_team"],
                         source=source, home_id=capture.get("home_id"),
                         away_id=capture.get("away_id"),
                         schema_confirmed=capture.get("shots_corners_schema_verified") is True)
    if source == "LIVE_UNVERIFIED":
        # The CLI captures data but does NOT independently verify that the
        # game is live. Never pass that data on as valid training inputs.
        row["pressure_status"] = "MATCH_LIVE_STATUS_NOT_VERIFIED"
        for k in FEATURES:
            row[k] = None
    return row


def to_datetime(value: Any) -> Optional[datetime]:
    try:
        if not value:
            return None
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if d.tzinfo is None:
            return None
        return d.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def merge_captures(snapshots: Path, captures: Path, output: Path) -> Dict[str, int]:
    src = read_csv(snapshots)
    by_key = defaultdict(list)
    with captures.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            cap = json.loads(line)
            if cap.get("source") != "LIVE_ASOF":
                continue
            at = to_datetime(cap.get("captured_at_utc"))
            if at is None:
                raise ValueError("Capture ligne %s: captured_at_utc invalide" % line_num)
            by_key[(str(cap["match_id"]), int(cap["minute"]))].append((at, cap))
    for seq in by_key.values():
        seq.sort(key=lambda x: x[0])
    merged = []
    found = 0
    for snapshot in src:
        key = (str(snapshot["match_id"]), int(snapshot["minute"]))
        snapshot_at = to_datetime(snapshot.get("snapshot_timestamp_utc"))
        selected = None
        # Only captures made BEFORE snapshot was recorded (strict as-of).
        if snapshot_at is not None:
            for at, cap in by_key.get(key, ()):
                if at <= snapshot_at:
                    selected = cap
                else:
                    break
        if selected is not None:
            pressure = capture_to_row(selected)
            found += int(pressure["events_available_at_snapshot"] is True)
        else:
            pressure = base_row(key[0], key[1], "NO_LIVE_CAPTURE")
            pressure["pressure_status"] = "NO_ASOF_CAPTURE"
        row = dict(snapshot)
        row.update({k: v for k, v in pressure.items() if k not in ("match_id", "minute")})
        merged.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(merged[0]) if merged else []
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    return {"snapshots": len(merged), "with_valid_asof": found,
            "without_asof": len(merged) - found}


def audit_snapshots(path: Path) -> Dict[str, Any]:
    d = read_csv(path)
    minutes = Counter(r.get("minute") for r in d)
    n_match = len({r.get("match_id") for r in d})
    dup = len(d) - len({(r.get("match_id"), r.get("minute")) for r in d})
    suspect = Counter()
    for r in d:
        m = parse_minute(r.get("minute"))
        if m is None:
            continue
        for field in ("last_goal_minute", "last_shot_minute", "last_shot_on_target_minute", "last_corner_minute"):
            em = parse_minute(r.get(field))
            if em is not None and em > m:
                suspect[field] += 1
    return dict(rows=len(d), unique_matches=n_match, duplicate_match_minute=dup,
                minutes=dict(minutes), future_event_minutes=dict(suspect),
                new_features_present=all(f in d[0] for f in FEATURES) if d else False)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Inspect a JSON timeline, diagnostic only")
    inspect.add_argument("--timeline", type=Path, required=True)
    inspect.add_argument("--match-id", required=True)
    inspect.add_argument("--minute", type=int, required=True)
    inspect.add_argument("--home", required=True)
    inspect.add_argument("--away", required=True)
    inspect.add_argument("--home-id", default=None)
    inspect.add_argument("--away-id", default=None)
    audit = sub.add_parser("audit", help="Audit snapshot CSV; no six invented features")
    audit.add_argument("--snapshots", type=Path, required=True)
    merge = sub.add_parser("merge", help="Merge only proven live-asof timeline captures")
    merge.add_argument("--snapshots", type=Path, required=True)
    merge.add_argument("--captures", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "inspect":
        data = json.loads(args.timeline.read_text(encoding="utf-8"))
        r = build_features(data, args.match_id, args.minute, args.home, args.away,
                           home_id=args.home_id, away_id=args.away_id)
    elif args.command == "audit":
        r = audit_snapshots(args.snapshots)
    else:
        r = merge_captures(args.snapshots, args.captures, args.output)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
