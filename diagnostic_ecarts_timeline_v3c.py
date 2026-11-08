#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit hors ligne des désaccords /timeline vs /stats de V3-C.

Aucune requête réseau. Lecture seule des caches du dossier --outdir.
Ce script n'applique AUCUNE correction aux six variables.
Python 3.9+ stdlib.
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def norm(s):
    s = unicodedata.normalize('NFKD', str(s or '')).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')


def side_of(ev, home, away):
    team = ev.get('team')
    if not isinstance(team, dict):
        return None
    ident = str(team.get('id') or '')
    if home.get('id') and ident == home['id']:
        return 'home'
    if away.get('id') and ident == away['id']:
        return 'away'
    name = norm(team.get('name'))
    if name and name == norm(home['name']):
        return 'home'
    if name and name == norm(away['name']):
        return 'away'
    return None


def stats_counts(stats):
    data = stats['data']
    overview = data['overview']
    result = {}
    for label, field in (('shot', 'total_shots'), ('sot', 'shots_on_target'), ('corner', 'corner_kicks')):
        for side in ('home', 'away'):
            result[(side, label)] = overview[field]['all'][side]
    return result


def diagnostics(events, row, stats):
    counts = Counter()
    types = {'home': Counter(), 'away': Counter(), 'unknown': Counter()}
    teams = {'home': {'id': str(row.get('home_id') or ''), 'name': row['home_team']},
             'away': {'id': str(row.get('away_id') or ''), 'name': row['away_team']}}
    seen = set()
    examples = {'corner': {'home': [], 'away': []}, 'goal': {'home': [], 'away': []}}
    other_period = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        seq = ev.get('sequence')
        if seq is not None:
            if str(seq) in seen:
                continue
            seen.add(str(seq))
        period = str(ev.get('period') or '').lower()
        if period and period not in ('first_half','second_half'):
            other_period += 1
            continue
        typ = str(ev.get('type') or '')
        side = side_of(ev, teams['home'], teams['away']) or 'unknown'
        types[side][typ] += 1
        if side == 'unknown':
            continue
        if typ in ('shot_on_target','shot_off_target','shot_blocked'):
            counts[(side,'shot')] += 1
        if typ == 'shot_on_target':
            counts[(side,'sot')] += 1
        if typ == 'corner_kick':
            counts[(side,'corner')] += 1
            if len(examples['corner'][side]) < 4:
                examples['corner'][side].append({'minute':ev.get('minute'), 'team':ev.get('team',{}).get('name'), 'type':typ})
        if 'goal' in typ or typ in ('penalty_scored','penalty_goal'):
            counts[(side,'goal_event')] += 1
            if len(examples['goal'][side]) < 4:
                examples['goal'][side].append({'minute':ev.get('minute'), 'team':ev.get('team',{}).get('name'), 'type':typ})
    result = {'match_id':row['match_id'], 'home_team':row['home_team'], 'away_team':row['away_team'],
              'unknown_types':dict(types['unknown']), 'all_event_types':dict(types['home']+types['away']+types['unknown']),
              'non_regulation_events':other_period, 'corner_examples':examples['corner'],
              'goal_examples':examples['goal']}
    for label in ('shot','sot','corner'):
        for side in ('home','away'):
            result['%s_%s_timeline'%(side,label)] = counts[(side,label)]
            result['%s_%s_stats'%(side,label)] = stats[(side,label)]
            result['%s_%s_diff'%(side,label)] = counts[(side,label)]-stats[(side,label)]
            if label in ('shot','sot'):
                result['%s_%s_goal_events'%(side,label)] = counts[(side,'goal_event')]
                result['%s_%s_if_goals_added'%(side,label)] = counts[(side,label)]+counts[(side,'goal_event')]
    result['corners_exact_if_swapped'] = (counts[('away','corner')]==stats[('home','corner')] and
                                          counts[('home','corner')]==stats[('away','corner')])
    result['corner_error_normal'] = (abs(counts[('home','corner')]-stats[('home','corner')]) +
                                     abs(counts[('away','corner')]-stats[('away','corner')]))
    result['corner_error_if_swapped'] = (abs(counts[('away','corner')]-stats[('home','corner')]) +
                                         abs(counts[('home','corner')]-stats[('away','corner')]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('historical_1710_manifest.csv'))
    parser.add_argument('--outdir',type=Path,default=Path('outputs/pressure_historical'))
    parser.add_argument('--output',type=Path,default=None)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8-sig',newline='') as f:
        manifest = list(csv.DictReader(f))
    report = []
    for row in manifest:
        mid = row.get('match_id','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',mid):
            continue
        timeline_file = args.outdir/'timeline_cache'/(mid+'.json')
        stats_file = args.outdir/'stats_cache'/(mid+'.json')
        if not timeline_file.exists() or not stats_file.exists():
            continue
        try:
            timeline = json.loads(timeline_file.read_text(encoding='utf-8'))
            stats = json.loads(stats_file.read_text(encoding='utf-8'))
            if timeline.get('data',{}).get('match_id')!=mid or stats.get('data',{}).get('match_id')!=mid:
                print('IGNORE : ID incoherent',mid)
                continue
            r = diagnostics(timeline['data']['events'],row,stats_counts(stats))
            report.append(r)
        except (ValueError,KeyError,TypeError,OSError) as e:
            print('IGNORE : schema/cache',mid,type(e).__name__)
    print('\n=== DIAGNOSTIC PAR MATCH (FT) ===')
    for r in report:
        print('\n%s | %s vs %s'%(r['match_id'],r['home_team'],r['away_team']))
        for label in ('shot','sot','corner'):
            print('  %-8s timeline %2s-%-2s | stats %2s-%-2s | ecarts %+d / %+d'%(
                label,r['home_%s_timeline'%label],r['away_%s_timeline'%label],
                r['home_%s_stats'%label],r['away_%s_stats'%label],
                r['home_%s_diff'%label],r['away_%s_diff'%label]))
        print('  EVENEMENTS BUT: dom=%s ext=%s (hypothese a verifier)'%(
            r['home_shot_goal_events'],r['away_shot_goal_events']))
        print('  CORNERS: erreur normale=%s | erreur si cotes inverses=%s | inversion exacte=%s'%(
            r['corner_error_normal'],r['corner_error_if_swapped'],r['corners_exact_if_swapped']))
        print('  Types d\'evenements (tous):',json.dumps(r['all_event_types'],ensure_ascii=False))
        print('  Echantillon corners bruts:',json.dumps(r['corner_examples'],ensure_ascii=False))
    print('\n=== SYNTHESE ===')
    print('Matchs compares :',len(report))
    if report:
        print('Inversion exacte des corners :',sum(r['corners_exact_if_swapped'] for r in report),'/',len(report))
        print('Erreur absolue corners : avant=%s | inversion hypothétique=%s'%(
            sum(r['corner_error_normal'] for r in report),sum(r['corner_error_if_swapped'] for r in report)))
    out = args.output or (args.outdir/'diagnostic_ecarts_detail.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rapport detaille :',out)
    print('LECTURE SEULE : aucune correction appliquee aux caches ou features.')

if __name__=='__main__':
    main()
