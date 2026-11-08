#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from collections import Counter
from audit_conventions_v3c import (metric_events, metric_stats, score_variant, comparison)

ROW={'match_id':'mt_123','home_team':'Home','away_team':'Away','home_id':'h','away_id':'a'}

def ev(t,team='h',period='second_half',seq=1,minute=87):
    return {'type':t,'team':{'id':team,'name':'Home' if team=='h' else 'Away'},
            'period':period,'sequence':seq,'minute':minute,'extra_time':0}

class Tests(unittest.TestCase):
    def test_goal_and_penalty_separate(self):
        p={'data':{'match_id':'mt_123','coverage':'full','events':[
            ev('shot_on_target',seq=1),ev('goal',seq=2),ev('penalty_scored',seq=3),
            ev('shot_woodwork',team='a',seq=4),ev('own_goal',team='a',seq=5)]}}
        c,issues,cover,n=metric_events(p,ROW,'mt_123')
        self.assertEqual(c[('home','goal')],1)
        self.assertEqual(c[('home','penalty_scored')],1)
        self.assertEqual(c[('away','own_goal')],1)
        self.assertEqual(cover,'full')
        self.assertEqual(issues,{})
    def test_duplicate_sequence_removed(self):
        p={'data':{'match_id':'mt_123','events':[ev('corner_kick',seq=1),ev('corner_kick',seq=1)]}}
        c,issues,_,_=metric_events(p,ROW,'mt_123')
        self.assertEqual(c[('home','corner_kick')],1)
        self.assertEqual(issues.get('duplicate_sequence'),1)
    def test_corner_swap_global(self):
        rec={'counts':Counter({('home','corner_kick'):1,('away','corner_kick'):4}),
             'stats':{('home','corner'):4,('away','corner'):1}}
        self.assertEqual(score_variant([rec],'corner',frozenset(),swapped=True)['abs_error'],0)
        self.assertEqual(score_variant([rec],'corner',frozenset(),swapped=False)['abs_error'],6)
    def test_stats_not_imputed(self):
        with self.assertRaises(ValueError):
            metric_stats({'data':{'match_id':'mt_123','overview':{'total_shots':{'all':{'home':0,'away':1}}}}},'mt_123')
    def test_wrong_match_id(self):
        with self.assertRaises(ValueError):
            metric_events({'data':{'match_id':'mt_other','events':[]}},ROW,'mt_123')
    def test_period_and_team(self):
        p={'data':{'match_id':'mt_123','events':[ev('goal',period='extra_time',seq=1),dict(ev('corner_kick',team='wrong',seq=2), team={'id':'wrong','name':'Unknown'})]}}
        c,issues,_,_=metric_events(p,ROW,'mt_123')
        self.assertEqual(c[('home','goal')],0)
        self.assertEqual(issues['offensive_non_regulation'],1)
        self.assertEqual(issues['unknown_offensive_team'],1)
    def test_hypothesis_selection_does_not_follow_holdout(self):
        rec=lambda e: {'counts':Counter({('home','goal'):e}),
                      'stats':{('home','sot'):e,('away','sot'):0}}
        # first 5 prefer goals; holdout deviates from them.
        records=[rec(1) for _ in range(5)]+[{'counts':Counter({('home','goal'):1}),
                      'stats':{('home','sot'):0,('away','sot'):0}} for _ in range(10)]
        table=comparison(records,'sot',{'BASE':frozenset(),'GOALS':frozenset({'goal'})})
        self.assertEqual(table[0]['name'],'GOALS')
        self.assertEqual(table[0]['holdout']['abs_error'],10)

if __name__=='__main__':
    unittest.main(verbosity=2)
