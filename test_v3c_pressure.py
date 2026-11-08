# -*- coding: utf-8 -*-
import csv
import json
import tempfile
import unittest
from pathlib import Path

from v3c_pressure_features import build_features, event_category, event_minute, merge_captures, parse_minute

class PressureTests(unittest.TestCase):
    def build(self, events, minute=87, source="LIVE_ASOF"):
        return build_features({"data": {"events": events}}, "mt_1", minute, "Team Home", "Team Away", source, schema_confirmed=True)

    def test_minute_and_stoppage(self):
        self.assertEqual(parse_minute("90+4"), 94)
        self.assertEqual(event_minute({"time": {"elapsed": 90, "extra": 4}}), 94)

    def test_five_completed_minutes_and_future_protection(self):
        events = [
            {"id": "a", "minute": 81, "type": "Shot", "team": "Team Home"},
            {"id": "b", "minute": 82, "type": "Shot", "team": "Team Home"},
            {"id": "c", "minute": 83, "type": "Shot on Target", "team": "Team Away"},
            {"id": "d", "minute": 85, "type": "Corner", "team": "Team Home"},
            {"id": "e", "minute": 86, "type": "Shot", "detail": "On Target", "team": "Team Home"},
            {"id": "f", "minute": 87, "type": "Shot", "team": "Team Away"},
            {"id": "g", "minute": 89, "type": "Shot", "team": "Team Away"},
        ]
        r = self.build(events, 87, source="HISTORICAL_DIAGNOSTIC_ONLY")
        self.assertEqual([r[k] for k in ("shots_home_last5","shots_away_last5","sot_home_last5", "sot_away_last5", "corners_home_last5", "corners_away_last5")], [2,1,1,1,1,0])
        self.assertEqual(r["events_future_filtered"], 2)
        self.assertFalse(r["events_available_at_snapshot"])
        early = self.build(events, 85, source="HISTORICAL_DIAGNOSTIC_ONLY")
        self.assertEqual(early["shots_home_last5"], 2)
        self.assertEqual(early["shots_away_last5"], 1)
        self.assertEqual(early["corners_home_last5"], 0)

    def test_schema_coverage_unverified_is_quarantined(self):
        r = build_features({"events": []}, "mt_1", 87, "Team Home", "Team Away", source="LIVE_ASOF")
        self.assertEqual(r["pressure_status"], "SHOT_CORNER_EVENT_COVERAGE_UNVERIFIED")
        self.assertIsNone(r["shots_home_last5"])

    def test_live_future_events_quarantined(self):
        r = self.build([{"minute": 89,"type":"Shot","team":"Team Home"}])
        self.assertEqual(r["pressure_status"], "SUSPECT_LIVE_TIMELINE_CONTAINS_FUTURE")
        self.assertIsNone(r["shots_home_last5"])
        self.assertFalse(r["events_available_at_snapshot"])

    def test_unknown_team_fail_closed(self):
        r = self.build([{"minute": 85, "type":"Shot", "team": "Somebody else"}])
        self.assertEqual(r["pressure_status"], "INCOMPLETE_UNKNOWN_TEAM_OR_TIME")
        self.assertIsNone(r["shots_home_last5"])

    def test_empty_list_is_zero_but_missing_list_is_not(self):
        r = self.build([])
        self.assertEqual(r["shots_home_last5"], 0)
        self.assertEqual(r["pressure_status"], "VALID_LIVE_ASOF")
        x = build_features({"data":{}}, "mt_1", 87, "Home", "Away", "LIVE_ASOF")
        self.assertIsNone(x["shots_home_last5"])
        self.assertEqual(x["pressure_status"], "NO_TIMELINE_LIST")

    def test_goal_does_not_count_as_shot_by_default(self):
        r = self.build([{"minute": 84, "type": "Goal", "team": "Team Home"}])
        self.assertEqual(r["shots_home_last5"], 0)

    def test_duplicates_one_id(self):
        e = {"id": 53, "minute": 86, "type": "Corner", "team": "Team Away"}
        r = self.build([e, e])
        self.assertEqual(r["corners_away_last5"], 1)

    def test_extra_time_excluded(self):
        r = self.build([{"minute": 86, "period":"extra_time", "type":"Shot", "team":"Team Home"}])
        self.assertEqual(r["shots_home_last5"], 0)

    def test_nested_team(self):
        r = self.build([{"minute": 86, "type": {"name":"Shot"}, "detail":"Saved", "team":{"name":"Team Home"}}])
        self.assertEqual(r["shots_home_last5"], 1)

    def test_historical_is_not_live(self):
        r = self.build([{"minute": 84, "type":"Shot", "team":"Team Home"}], source="HISTORICAL_DIAGNOSTIC_ONLY")
        self.assertFalse(r["events_available_at_snapshot"])
        self.assertEqual(r["shots_home_last5"], 1)

    def test_merge_strict_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            s = p/"snapshot.csv"
            s.write_text("match_id,minute,snapshot_timestamp_utc\nmt_1,87,2026-10-25T15:00:00Z\nmt_2,87,2026-10-25T15:00:00Z\n", encoding="utf8")
            c = p/"captures.jsonl"
            rows = [
                dict(match_id="mt_1", minute=87, captured_at_utc="2026-10-25T14:59:59Z", home_team="Team Home", away_team="Team Away", source="LIVE_ASOF", live_status_verified=True, shots_corners_schema_verified=True, timeline_payload={"events":[{"minute":86,"type":"Shot","team":"Team Home"}]}),
                dict(match_id="mt_2", minute=87, captured_at_utc="2026-10-25T15:00:01Z", home_team="Team Home", away_team="Team Away", source="LIVE_ASOF", live_status_verified=True, shots_corners_schema_verified=True, timeline_payload={"events":[{"minute":86,"type":"Shot","team":"Team Home"}]})
            ]
            c.write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf8")
            o = p/"out.csv"
            d = merge_captures(s, c, o)
            self.assertEqual(d["with_valid_asof"], 1)
            with o.open() as f:
                merged = list(csv.DictReader(f))
            self.assertEqual(merged[0]["shots_home_last5"], "1")
            self.assertEqual(merged[1]["shots_home_last5"], "")

if __name__ == '__main__':
    unittest.main(verbosity=2)
