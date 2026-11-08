#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from rebuild_historical_pressure import reconstructed_rows, read_rows, prepare, raw_path, rebuild, import_captures, compare_stats

MATCH = {"match_id": "mt_378787130", "home_team": "Bari", "away_team": "Crotone",
         "home_id": "tm_98418", "away_id": "tm_89602", "kickoff_utc": "2026-10-19T00:00:00+00:00"}
EVENTS = [
    {"sequence": 1, "minute": 80, "type": "shot_on_target", "team": {"id": "tm_98418", "name": "Bari"}},
    {"sequence": 2, "minute": 82, "type": "corner_kick", "team": {"id": "tm_89602", "name": "Crotone"}},
    {"sequence": 3, "minute": 83, "type": "shot_off_target", "team": {"id": "tm_89602", "name": "Crotone"}},
    {"sequence": 4, "minute": 84, "type": "shot_blocked", "team": {"id": "tm_98418", "name": "Bari"}},
    {"sequence": 5, "minute": 86, "type": "shot_on_target", "team": {"id": "tm_89602", "name": "Crotone"}},
    {"sequence": 6, "minute": 88, "type": "corner_kick", "team": {"id": "tm_89602", "name": "Crotone"}},
    {"sequence": 7, "minute": 90, "type": "shot_on_target", "team": {"id": "tm_98418", "name": "Bari"}},
]
PAYLOAD = {"data": {"match_id": MATCH["match_id"], "coverage": "full", "events": EVENTS},
           "meta": {"coverage": "full", "last_updated": "2026-10-19T13:00:05Z"}}

class HistoricalTests(unittest.TestCase):
    def test_completed_minutes(self):
        rows = reconstructed_rows(PAYLOAD, MATCH)
        self.assertEqual([r["minute"] for r in rows], [85,86,87])
        self.assertEqual([r["shots_home_last5"] for r in rows], [2,1,1])
        self.assertEqual([r["shots_away_last5"] for r in rows], [1,1,2])
        self.assertEqual([r["sot_away_last5"] for r in rows], [0,0,1])
        self.assertEqual([r["corners_away_last5"] for r in rows], [1,1,1])
        self.assertEqual([r["events_future_filtered"] for r in rows], [3,3,2])
        self.assertFalse(any(r["training_ready"] for r in rows))
    def test_wrong_id_all_null(self):
        payload = json.loads(json.dumps(PAYLOAD))
        payload["data"]["match_id"] = "mt_OTHER"
        rows = reconstructed_rows(payload, MATCH)
        self.assertEqual(rows[0]["historical_reconstruction_status"], "MISMATCH_OR_MISSING_RESPONSE_MATCH_ID")
        self.assertIsNone(rows[0]["shots_home_last5"])
    def test_partial_coverage_no_zero(self):
        payload = json.loads(json.dumps(PAYLOAD))
        payload["data"]["coverage"] = "partial"
        self.assertIsNone(reconstructed_rows(payload, MATCH)[0]["corners_home_last5"])
    def test_no_offensive_schema(self):
        payload = {"data":{"match_id": MATCH["match_id"], "coverage": "full", "events": [
            {"minute": 82,"type":"foul","team":{"name":"Bari"}}]}}
        rows = reconstructed_rows(payload, MATCH)
        self.assertEqual(rows[0]["historical_reconstruction_status"], "NO_OFFENSIVE_EVENT_TYPES_VERIFY_PROVIDER")
    def test_unknown_team(self):
        payload = json.loads(json.dumps(PAYLOAD))
        payload["data"]["events"][0]["team"] = {"id": "tm_unknown", "name": "Unknown"}
        self.assertEqual(reconstructed_rows(payload, MATCH)[0]["historical_reconstruction_status"], "BAD_EVENT_TIME_OR_TEAM")
    def test_duplicates_quarantine(self):
        payload = json.loads(json.dumps(PAYLOAD))
        payload["data"]["events"].append(dict(payload["data"]["events"][1]))
        self.assertEqual(reconstructed_rows(payload, MATCH)[0]["historical_reconstruction_status"], "DUPLICATE_SEQUENCE_AUDIT_REQUIRED")
    def test_stats_crosscheck(self):
        ref = dict(zip(("home_shots_ft","away_shots_ft","home_sot_ft","away_sot_ft","home_corners_ft","away_corners_ft"),
                       (3,2,2,1,0,2)))
        rows = reconstructed_rows(PAYLOAD, MATCH, official_stats=ref)
        self.assertEqual(rows[0]["stats_crosscheck_status"], "MATCH_6_FT_COUNTS")
        ref["home_shots_ft"] = 100
        self.assertEqual(reconstructed_rows(PAYLOAD, MATCH, official_stats=ref)[0]["stats_crosscheck_status"], "MISMATCH_FT_COUNTS")
    def test_prepare_manifest_keeps_labels_separate(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            s = p/"source.tsv"
            s.write_text("match_id\thome_team\taway_team\tgoal_after_87\tkickoff_utc\n"
                         "mt_1\tBari\tCrotone\t1.0\t46278.5\n"
                         "mt_1\tBari\tCrotone\t1.0\t46278.5\n",encoding="utf-8")
            prepare(Namespace(input=s,outdir=p/"out"))
            mf = read_rows(p/"out"/"manifest.csv")
            self.assertEqual(len(mf),1)
            self.assertNotIn("goal_after_87",mf[0])
            labels = read_rows(p/"out"/"reference_labels_SEPARATE.csv")
            self.assertEqual(labels[0]["reference_goal_after_87"],"1.0")
    def test_import_captures_without_api(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            cap = {"match_id": MATCH["match_id"], "home_team":"Bari", "away_team":"Crotone",
                   "minute":87, "source":"LIVE_ASOF", "live_status_verified":False,
                   "timeline_payload":PAYLOAD}
            source = p/"captures.jsonl"
            source.write_text(json.dumps(cap)+"\n", encoding="utf-8")
            import_captures(Namespace(captures=source,outdir=p/"demo"))
            self.assertTrue(raw_path(p/"demo",MATCH["match_id"]).is_file())
            rebuilt = reconstructed_rows(json.loads(raw_path(p/"demo", MATCH["match_id"]).read_text()), MATCH)
            self.assertFalse(rebuilt[0]["training_ready"])
            self.assertEqual(len(read_rows(p/"demo"/"manifest.csv")), 1)

    def test_rebuild_cache_only(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            inp = p/"manifest.csv"
            with inp.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=MATCH)
                w.writeheader();w.writerow(MATCH)
            cache=raw_path(p, MATCH["match_id"])
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(PAYLOAD),encoding="utf8")
            rebuild(Namespace(manifest=inp,outdir=p,official_stats=None))
            result=read_rows(p/"historical_pressure_85_86_87.csv")
            self.assertEqual(len(result),3)
            self.assertEqual(result[-1]["shots_away_last5"],"2")
            self.assertEqual(result[-1]["training_ready"],"False")

if __name__ == "__main__":
    unittest.main(verbosity=2)
