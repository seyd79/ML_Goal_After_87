import csv
import json
import tempfile
import unittest
from pathlib import Path
from merge_v3c_pressure_v2 import build_dataset, read_table, PRESSURE_FIELDS


def csvwrite(p, names, rows):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=names)
        w.writeheader();w.writerows(rows)

class TestMerge(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d=Path(self.tmp.name)
        self.paths={n:d/(n+'.csv') for n in ['pressure','manifest','labels','baseline']}
        self.pnames=['match_id','minute','home_team','away_team','kickoff_utc','training_ready','pressure_status','temporal_provenance']+list(PRESSURE_FIELDS)
        self.prows=[]
        for minute in (85,86,87):
            a={'match_id':'mt_123','minute':str(minute),'home_team':'A','away_team':'B','kickoff_utc':'2026-09-01T10:00:00+00:00','training_ready':'False','pressure_status':'V2_EXPERIMENTAL_NOT_CERTIFIED','temporal_provenance':'HISTORICAL_HINDSIGHT_RECONSTRUCTION'}
            a.update({k: '1' for k in PRESSURE_FIELDS})
            self.prows.append(a)
        csvwrite(self.paths['pressure'],self.pnames,self.prows)
        csvwrite(self.paths['manifest'],['match_id','home_team','away_team','kickoff_utc'],[{'match_id':'mt_123','home_team':'A','away_team':'B','kickoff_utc':'2026-09-01T10:00:00+00:00'}])
        csvwrite(self.paths['labels'],['match_id','reference_goal_after_87','reference_target_status','reference_target_warning'],[{'match_id':'mt_123','reference_goal_after_87':'1.0','reference_target_status':'LABELED_V2','reference_target_warning':''}])
        csvwrite(self.paths['baseline'],['match_id','snapshot_minute','score_87_home','final_score_home','goal_after_87','model_probability'],[{'match_id':'mt_123','snapshot_minute':'87','score_87_home':'0','final_score_home':'3','goal_after_87':'1','model_probability':'0.91'}])
        self.output=d/'joined'
    def test_pair_87_only_and_label_separation(self):
        summary=build_dataset(*[self.paths[x] for x in ('pressure','manifest','labels')],self.output,self.paths['baseline'])
        self.assertEqual(summary['pressure_rows'],3)
        self.assertEqual(summary['paired_87_A_and_B'],1)
        self.assertEqual(summary['certified_training_rows'],0)
        a_cols, aa=read_table(self.output/'MODEL_A_BASELINE_87_REVIEW.csv')
        b_cols, bb=read_table(self.output/'MODEL_B_ENRICHED_87_REVIEW.csv')
        self.assertEqual(len(bb),1)
        self.assertEqual(bb[0]['shots_home_last5'],'1')
        for cols in (a_cols,b_cols):
            for forbidden in ('final_score_home','goal_after_87','model_probability','target_status'):
                self.assertNotIn(forbidden,cols)
        _,labels=read_table(self.output/'LABELS_87_SEPARATE_NOT_REVALIDATED.csv')
        self.assertEqual(labels[0]['goal_after_87'],'1.0')
    def test_pressure_only_without_baseline(self):
        s=build_dataset(*[self.paths[x] for x in ('pressure','manifest','labels')],self.output)
        self.assertEqual(s['paired_87_A_and_B'],0)
        _,pp=read_table(self.output/'PRESSURE_ONLY_85_86_87_REVIEW.csv')
        self.assertEqual(len(pp),3)
    def test_duplicate_fails(self):
        csvwrite(self.paths['pressure'],self.pnames,self.prows+[self.prows[-1]])
        with self.assertRaisesRegex(ValueError,'doublon'):
            build_dataset(*[self.paths[x] for x in ('pressure','manifest','labels')],self.output)
    def test_bad_label_not_feature(self):
        csvwrite(self.paths['labels'],['match_id','reference_goal_after_87','reference_target_status','reference_target_warning'],[{'match_id':'mt_123','reference_goal_after_87':'2','reference_target_status':'LABELED_V2','reference_target_warning':'bad'}])
        s=build_dataset(*[self.paths[x] for x in ('pressure','manifest','labels')],self.output,self.paths['baseline'])
        self.assertEqual(s['issue_counts']['REFERENCE_LABEL_INVALID'],3)
        self.assertEqual(s['issue_counts']['REFERENCE_LABEL_WARNING'],3)
    def test_kickoff_inconsistency(self):
        self.prows[0]['kickoff_utc']='2024-01-01'
        csvwrite(self.paths['pressure'],self.pnames,self.prows)
        s=build_dataset(*[self.paths[x] for x in ('pressure','manifest','labels')],self.output)
        self.assertEqual(s['issue_counts']['KICKOFF_MISMATCH'],1)

if __name__=='__main__':unittest.main(verbosity=2)
