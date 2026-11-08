#!/usr/bin/env python3
"""Suite de tests V2 entièrement hors ligne, Python stdlib."""
import json
import tempfile
import unittest
from pathlib import Path
from v3c_pressure_features_v2 import compute, full_time_counts
from rebuild_pressure_v2_experimental import run

MID='mt_example'
MANIFEST={'match_id':MID,'home_team':'Alpha','away_team':'Beta',
          'home_id':'tm_h','away_id':'tm_a','kickoff_utc':'2026-01-01T20:00:00Z'}


def ev(seq,typ,minute,side='home',extra=0,player=None,period='second_half'):
    d={'sequence':seq,'type':typ,'minute':minute,'extra_time':extra,'period':period,
       'team':{'id':'tm_h' if side=='home' else 'tm_a',
               'name':'Alpha' if side=='home' else 'Beta'}}
    if player:
        d['player']={'id':player}
    return d


def payload(events,mid=MID,coverage='full'):
    return {'data':{'match_id':mid,'coverage':coverage,'events':events},'meta':{}}


class V2Tests(unittest.TestCase):
    def test_five_completed_minutes_no_87(self):
        rows=[ev(1,'shot_on_target',81),ev(2,'shot_off_target',82),
              ev(3,'shot_blocked',86),ev(4,'goal',87),ev(5,'corner_kick',88,'away')]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],2)
        self.assertEqual(r['sot_home_last5'],0)
        self.assertEqual(r['corners_home_last5'],0)
        self.assertEqual(r['events_after_cutoff_excluded'],2)

    def test_corners_globally_swapped(self):
        r=compute(payload([ev(1,'corner_kick',83,'home'),ev(2,'corner_kick',84,'away')]),MANIFEST,87)
        self.assertEqual((r['corners_home_last5'],r['corners_away_last5']),(1,1))
        r=compute(payload([ev(1,'corner_kick',83,'home')]),MANIFEST,87)
        self.assertEqual((r['corners_home_last5'],r['corners_away_last5']),(0,1))

    def test_goals_penalty_and_woodwork(self):
        rows=[ev(1,'goal',82),ev(2,'shot_woodwork',83),
              ev(3,'penalty_scored',84),ev(4,'penalty_missed',85),
              ev(5,'shot_on_target',86)]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],5)
        self.assertEqual(r['sot_home_last5'],3)

    def test_penalty_goal_strong_dedup(self):
        rows=[ev(1,'goal',84,player='P'),ev(2,'penalty_scored',84,player='P')]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],1)
        self.assertEqual(r['sot_home_last5'],1)
        self.assertIn('PENALTY_GOAL_DUPLICATE_MERGED',r['diagnostic_issues'])

    def test_not_dedup_other_players(self):
        rows=[ev(1,'goal',84,player='P'),ev(2,'penalty_scored',84,player='Q')]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],2)

    def test_own_goal_not_counted_as_attempt(self):
        r=compute(payload([ev(1,'own_goal',85)]),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],0)
        self.assertEqual(r['sot_home_last5'],0)

    def test_penalty_awarded_is_not_a_shot(self):
        rows=[ev(1,'penalty_awarded',83), ev(2,'penalty_scored',84)]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],1)
        self.assertEqual(r['sot_home_last5'],1)
        self.assertEqual(r['pressure_status'],'V2_EXPERIMENTAL_NOT_CERTIFIED')

    def test_shot_unknown_type_fails_closed(self):
        r=compute(payload([ev(1,'shot_saved_new_schema',84)]),MANIFEST,87)
        self.assertEqual(r['pressure_status'],'QUARANTINED_OFFENSIVE_EVENT_AMBIGUITY')
        self.assertIsNone(r['shots_home_last5'])

    def test_full_time_future_does_not_affect_snapshot_features(self):
        a=payload([ev(1,'shot_on_target',84,'home')])
        b=payload([ev(1,'shot_on_target',84,'home'),ev(2,'goal',89,'away'),
                   ev(3,'corner_kick',90,'away')])
        v1=compute(a,MANIFEST,87)
        v2=compute(b,MANIFEST,87)
        for key in ('shots_home_last5','shots_away_last5', 'sot_home_last5',
                    'sot_away_last5','corners_home_last5','corners_away_last5'):
            self.assertEqual(v1[key],v2[key])

    def test_duplicate_identical_sequence(self):
        x=ev(1,'corner_kick',83,'home')
        r=compute(payload([x,x.copy()]),MANIFEST,87)
        self.assertEqual(r['corners_away_last5'],1)
        self.assertEqual(r['pressure_status'],'V2_EXPERIMENTAL_NOT_CERTIFIED')

    def test_bad_match_fail_closed(self):
        r=compute(payload([ev(1,'goal',85)],mid='other'),MANIFEST,87)
        self.assertIsNone(r['shots_home_last5'])
        self.assertEqual(r['pressure_status'],'TIMELINE_MATCH_ID_MISMATCH')

    def test_missing_coverage_not_zero(self):
        r=compute(payload([],coverage='partial'),MANIFEST,87)
        self.assertIsNone(r['shots_home_last5'])

    def test_unknown_team_not_zero(self):
        rows=[ev(1,'corner_kick',84)]
        rows[0]['team']={'id':'tm_other','name':'Other'}
        r=compute(payload(rows),MANIFEST,87)
        self.assertIsNone(r['corners_home_last5'])
        self.assertEqual(r['pressure_status'],'QUARANTINED_OFFENSIVE_EVENT_AMBIGUITY')

    def test_conflicting_sequence_quarantine(self):
        rows=[ev(1,'corner_kick',83),ev(1,'shot_on_target',84)]
        r=compute(payload(rows),MANIFEST,87)
        self.assertIsNone(r['shots_home_last5'])

    def test_extra_time_outside_window(self):
        rows=[ev(1,'goal',90,'home',extra=4),ev(2,'goal',86)]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],1)

    def test_ft_and_snapshot_independent_of_final_stats(self):
        source=payload([ev(1,'goal',84),ev(2,'corner_kick',83,'away')])
        r=compute(source,MANIFEST,87)
        ft=full_time_counts(source,MANIFEST)
        self.assertEqual(ft[('home','shots')],1)
        self.assertEqual(ft[('home','corners')],1)
        self.assertFalse(r['training_ready'])

    def test_regulation_only(self):
        rows=[ev(1,'goal',84,period='extra_time'),ev(2,'corner_kick',85,period='second_half')]
        r=compute(payload(rows),MANIFEST,87)
        self.assertEqual(r['shots_home_last5'],0)
        self.assertEqual(r['corners_away_last5'],1)

    def test_bad_clock_quarantine(self):
        rows=[ev(1,'goal','bad')]
        r=compute(payload(rows),MANIFEST,87)
        self.assertIsNone(r['shots_home_last5'])

    def test_rebuild_offline_does_not_touch_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'timeline_cache').mkdir()
            (p/'stats_cache').mkdir()
            original=json.dumps(payload([ev(1,'corner_kick',83,'home'),ev(2,'goal',86,'away')]))
            cache=p/'timeline_cache'/(MID+'.json')
            cache.write_text(original)
            manifest=p/'manifest.csv'
            manifest.write_text('match_id,home_team,away_team,home_id,away_id,kickoff_utc\n'
                                + MID + ',Alpha,Beta,tm_h,tm_a,\n')
            summary=run(manifest,p,p/'pressure_v2_experimental')
            self.assertEqual(summary['snapshots'],3)
            self.assertEqual(summary['api_requests'],0)
            self.assertEqual(cache.read_text(),original)
            self.assertTrue((p/'pressure_v2_experimental'/'pressure_v2_85_86_87_EXPERIMENTAL.csv').exists())
            self.assertEqual(summary['training_ready_matches'],0)

if __name__=='__main__':
    unittest.main(verbosity=2)
