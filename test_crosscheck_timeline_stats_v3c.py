import csv
import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import crosscheck_timeline_stats_v3c as mod

class CrosscheckTests(unittest.TestCase):
    def setUp(self):
        self.mid='mt_10101'
        self.manifest={'match_id':self.mid,'home_team':'Bari','away_team':'Crotone','home_id':'tm_98418','away_id':'tm_89602'}
        self.evs=[
            {'sequence':1,'minute':82,'period':'second_half','type':'shot_on_target','team':{'id':'tm_98418','name':'Bari'}},
            {'sequence':2,'minute':83,'period':'second_half','type':'shot_blocked','team':{'id':'tm_98418','name':'Bari'}},
            {'sequence':3,'minute':84,'period':'second_half','type':'shot_off_target','team':{'id':'tm_89602','name':'Crotone'}},
            {'sequence':4,'minute':86,'period':'second_half','type':'corner_kick','team':{'id':'tm_89602','name':'Crotone'}}]
        self.timeline={'data':{'match_id':self.mid,'coverage':'full','events':self.evs},'meta':{'last_updated':'2026-10-24T10:00Z'}}
        def field(h,a): return {'all':{'home':h,'away':a}}
        self.stats={'data':{'match_id':self.mid,'overview':{'total_shots':field(2,1),'shots_on_target':field(1,0),'corner_kicks':field(0,1)}, 'shots':{'total_shots':field(2,1),'shots_on_target':field(1,0)}}}

    def test_six_match(self):
        counts,issues,cov,_,n=mod.extract_timeline(self.timeline,self.manifest)
        stated,internal=mod.extract_stats(self.stats,self.mid)
        self.assertEqual(counts,stated)
        self.assertEqual(internal,'OK')
        self.assertEqual(n,4)
        self.assertEqual(cov,'full')
        self.assertFalse(issues)
    def test_missing_stat_never_zero(self):
        del self.stats['data']['overview']['corner_kicks']
        with self.assertRaises(ValueError):mod.extract_stats(self.stats,self.mid)
    def test_team_unknown_issue(self):
        self.evs[1]['team']={'id':'wrong','name':'wrong'}
        _,issues,_,_,_=mod.extract_timeline(self.timeline,self.manifest)
        self.assertEqual(issues['unknown_offensive_team'],1)
    def test_duplicate_sequence(self):
        self.evs.append(dict(self.evs[0]))
        _,issues,_,_,_=mod.extract_timeline(self.timeline,self.manifest)
        self.assertEqual(issues['duplicate_sequences'],1)
    def test_wrong_match_id(self):
        with self.assertRaises(ValueError):mod.extract_stats(self.stats,'mt_999')
    def test_offline_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            mod.write_csv(p/'manifest.csv',list(self.manifest),[self.manifest])
            (p/'timeline_cache').mkdir()
            (p/'stats_cache').mkdir()
            (p/'timeline_cache'/(self.mid+'.json')).write_text(json.dumps(self.timeline))
            (p/'stats_cache'/(self.mid+'.json')).write_text(json.dumps(self.stats))
            args=Namespace(manifest=p/'manifest.csv',outdir=p,env=p/'.env',limit=0,delay=0,timeout=20,offline=True,force=False)
            mod.run(args)
            with (p/'timeline_stats_crosscheck.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['status'],'MATCH_6_FT_COUNTS')
            self.assertEqual(rows[0]['diff_home_shots_ft'],'0')

if __name__=='__main__':unittest.main(verbosity=2)
