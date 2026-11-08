#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruit les 6 features V2 sur timelines en cache. OFFLINE, stdlib Python 3.9+.

Commande :
 python rebuild_pressure_v2_experimental.py \
  --manifest historical_1710_manifest.csv --outdir outputs/pressure_historical

Lit : outdir/timeline_cache/{match_id}.json, outdir/stats_cache/{match_id}.json
Ecrit UNIQUEMENT : outdir/pressure_v2_experimental/*
NE MODIFIE PAS les caches, les features V1, les labels ou le scanner.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from v3c_pressure_features_v2 import FEATURES, MINUTES, VERSION, compute, full_time_counts, stats_counts

META = ('match_id','minute','home_team','away_team','kickoff_utc','feature_logic_version',
        'temporal_provenance','training_ready','pressure_status','timeline_coverage',
        'event_count','diagnostic_issues','events_in_window','events_after_cutoff_excluded',
        'corner_policy','shot_policy','sot_policy')
AUDIT = ('match_id','home_team','away_team','ft_crosscheck_status','ft_exact_team_counts',
         'ft_abs_error','ft_shots_exact','ft_sot_exact','ft_corners_exact',
         'timeline_home_shots','stats_home_shots','diff_home_shots',
         'timeline_away_shots','stats_away_shots','diff_away_shots',
         'timeline_home_sot','stats_home_sot','diff_home_sot',
         'timeline_away_sot','stats_away_sot','diff_away_sot',
         'timeline_home_corners','stats_home_corners','diff_home_corners',
         'timeline_away_corners','stats_away_corners','diff_away_corners')


def write_csv(path, headers, data):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def load_manifest(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        records = list(csv.DictReader(f))
    if not records or not {'match_id','home_team','away_team'}.issubset(records[0]):
        raise ValueError('Manifest requis avec match_id,home_team,away_team')
    seen = set()
    result = []
    for r in records:
        mid = (r.get('match_id') or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{3,100}', mid):
            raise ValueError('Identifiant de match invalide dans manifest : ' + repr(mid))
        if mid in seen:
            raise ValueError('Doublon match_id dans manifest : ' + mid)
        if not (r.get('home_team') or '').strip() or not (r.get('away_team') or '').strip():
            raise ValueError('Equipe manquante pour ' + mid)
        seen.add(mid)
        result.append(r)
    return result


def check_ft(timeline, stats, row):
    r = {'match_id':row['match_id'], 'home_team':row['home_team'],
         'away_team':row['away_team'], 'ft_crosscheck_status':'',
         'ft_exact_team_counts':'', 'ft_abs_error':'',
         'ft_shots_exact':'', 'ft_sot_exact':'', 'ft_corners_exact':''}
    if stats is None:
        r['ft_crosscheck_status'] = 'STATS_CACHE_MISSING'
        return r
    try:
        left = full_time_counts(timeline, row)
        right = stats_counts(stats, row['match_id'])
    except (TypeError, ValueError, KeyError) as e:
        r['ft_crosscheck_status'] = 'CANNOT_CROSSCHECK_' + str(e)
        return r
    exact = 0
    error = 0
    types = Counter()
    for side in ('home','away'):
        for metric in ('shots','sot','corners'):
            a,b = left[(side, metric)],right[(side, metric)]
            r['timeline_'+side+'_'+metric] = a
            r['stats_'+side+'_'+metric] = b
            r['diff_'+side+'_'+metric] = a-b
            exact += int(a == b)
            types[metric] += int(a == b)
            error += abs(a-b)
    r['ft_exact_team_counts'] = exact
    r['ft_abs_error'] = error
    for metric in ('shots','sot','corners'):
        r['ft_'+metric+'_exact'] = types[metric]
    r['ft_crosscheck_status'] = 'MATCH_6_FT_COUNTS' if exact == 6 else 'MISMATCH_6_FT_COUNTS'
    return r


def run(manifest, outdir, output_dir, limit=0):
    entries = load_manifest(manifest)
    source = (outdir/'timeline_cache').resolve()
    stats_root = outdir/'stats_cache'
    output = output_dir.resolve()
    if output == outdir.resolve() or output == source or output == stats_root.resolve() or \
       (source in output.parents) or (stats_root.resolve() in output.parents):
        raise ValueError('Refus : dossier de sortie dans un cache source ou dossier source')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    checks = []
    failures = []
    missing = 0
    cached = 0
    valid_candidates = 0
    for r in entries:
        mid = r['match_id']
        path = outdir/'timeline_cache'/(mid+'.json')
        if not path.is_file():
            missing += 1
            continue
        if limit and cached >= limit:
            break
        cached += 1
        try:
            timeline = json.loads(path.read_text(encoding='utf-8'))
            computed = [compute(timeline, r, minute) for minute in MINUTES]
            for f in computed:
                f.update({'home_team':r['home_team'],'away_team':r['away_team'],
                          'kickoff_utc':r.get('kickoff_utc','')})
            results.extend(computed)
            if all(f['pressure_status']=='V2_EXPERIMENTAL_NOT_CERTIFIED' for f in computed):
                valid_candidates += 1
            stats_path = stats_root/(mid+'.json')
            stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.is_file() else None
            checks.append(check_ft(timeline,stats,r))
        except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as e:
            failures.append({'match_id':mid, 'error_type':type(e).__name__})
    write_csv(output/'pressure_v2_85_86_87_EXPERIMENTAL.csv',META+FEATURES, results)
    write_csv(output/'pressure_v2_ft_crosscheck_same_provider.csv',AUDIT, checks)
    write_csv(output/'pressure_v2_failed_cache.csv',('match_id','error_type'),failures)
    statuses=Counter(row['pressure_status'] for row in results)
    ft_statuses=Counter(row['ft_crosscheck_status'] for row in checks)
    summary={'version':VERSION,'manifest_matches':len(entries),'timeline_cached_processed':cached,
             'snapshots':len(results),'missing_timeline_cache':missing,
             'feature_candidate_matches':valid_candidates,
             'crosschecked_matches':sum(s in ('MATCH_6_FT_COUNTS','MISMATCH_6_FT_COUNTS') for s in
                                       (r['ft_crosscheck_status'] for r in checks)),
             'ft_statuses':dict(ft_statuses),'snapshot_statuses':dict(statuses),
             'read_errors':len(failures),'api_requests':0,
             'training_ready_matches':0,
             'warning':'EXPERIMENTAL: no training-ready certification; same-provider FT counts are only internal consistency checks; no future match data used in snapshot calculation.',
             'output_dir':str(output)}
    (output/'pressure_v2_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True,help='Racine existante contenant timeline_cache et stats_cache')
    p.add_argument('--output-dir',type=Path,default=None,
                   help='Dossier distinct, défaut OUTDIR/pressure_v2_experimental')
    p.add_argument('--limit',type=int,default=0,help='0 = tous les matchs en cache; >0 limite de tests')
    a=p.parse_args()
    if a.limit<0:
        p.error('--limit doit être >= 0')
    run(a.manifest,a.outdir,a.output_dir or a.outdir/'pressure_v2_experimental',a.limit)

if __name__=='__main__':
    main()
