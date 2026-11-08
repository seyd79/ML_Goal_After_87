#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3-C: comparer timelines historisees avec /football/matches/{id}/stats.

Python 3.9+, stdlib, aucune modification du scanner ni de ses modeles.
API /stats et /timeline viennent du MEME fournisseur : controle de coherence
interne, PAS validation independante ni preuve d'exhaustivite.

Exemple :
 python crosscheck_timeline_stats_v3c.py --manifest historical_1710_manifest.csv \
  --outdir outputs/pressure_historical --limit 5 --delay 1

 python rebuild_historical_pressure.py rebuild --manifest historical_1710_manifest.csv \
  --outdir outputs/pressure_historical \
  --official-stats outputs/pressure_historical/official_stats_api.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from collections import Counter
from pathlib import Path

METRICS = ('home_shots_ft','away_shots_ft','home_sot_ft','away_sot_ft',
           'home_corners_ft','away_corners_ft')
TYPES = {'shot_off_target':'shots', 'shot_on_target':'sot', 'shot_blocked':'shots',
         'corner_kick':'corners'}


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        if k in ('STATS_API_KEY','STATS_API_BASE_URL') and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as fd:
        return list(csv.DictReader(fd))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fd:
        w = csv.DictWriter(fd, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def safe_id(value):
    value = str(value or '').strip()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{3,100}', value):
        raise ValueError('Identifiant de match non valide')
    return value


def extract_stats(payload, expected_id):
    """Six totaux FT extraits de /stats ; aucune imputation de valeurs nulles."""
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get('match_id') != expected_id:
        raise ValueError('Mismatch match_id de /stats')
    overview = data.get('overview')
    if not isinstance(overview, dict):
        raise ValueError('Pas de bloc overview dans /stats')
    result = {}
    for field, category in [('total_shots','shots'),('shots_on_target','sot'),('corner_kicks','corners')]:
        all_counts = overview.get(field, {}).get('all') if isinstance(overview.get(field), dict) else None
        if not isinstance(all_counts, dict):
            raise ValueError('Champ stats manquant : '+field)
        for side in ('home','away'):
            val = all_counts.get(side)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError('Compte invalide : %s.%s' % (field, side))
            result['%s_%s_ft' % (side, category)] = val
    # Controle de concordance interne /stats : overview vs shots.
    internals = []
    shots_branch = data.get('shots')
    if isinstance(shots_branch, dict):
        for name, category in [('total_shots','shots'),('shots_on_target','sot')]:
            branch = shots_branch.get(name)
            check = branch.get('all') if isinstance(branch, dict) else None
            if isinstance(check, dict):
                for side in ('home','away'):
                    if check.get(side) != result['%s_%s_ft' % (side,category)]:
                        internals.append('%s.%s'%(name,side))
    return result, ('INTERNAL_STATS_MISMATCH:' + ','.join(internals) if internals else 'OK')


def norm_name(value):
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '_', value.casefold()).strip('_')


def extract_timeline(payload, manifest):
    data = payload.get('data') if isinstance(payload,dict) else None
    meta = payload.get('meta') if isinstance(payload,dict) else None
    if not isinstance(data,dict) or data.get('match_id') != manifest['match_id']:
        raise ValueError('Mismatch match_id de /timeline')
    events = data.get('events')
    if not isinstance(events,list):
        raise ValueError('Pas de liste d\'evenements')
    home = norm_name(manifest.get('home_team'))
    away = norm_name(manifest.get('away_team'))
    home_id = manifest.get('home_id') or ''
    away_id = manifest.get('away_id') or ''
    if not home or not away or home == away:
        raise ValueError('Noms des equipes absents/ambigus')
    counters = Counter()
    issues = Counter()
    seen_seq = set()
    for ev in events:
        if not isinstance(ev,dict):
            issues['invalid_record'] += 1
            continue
        seq = ev.get('sequence')
        if seq is not None:
            seq = str(seq)
            if seq in seen_seq:
                issues['duplicate_sequences'] += 1
                continue
            seen_seq.add(seq)
        typ = ev.get('type')
        if typ not in TYPES:
            continue
        period = str(ev.get('period') or '').casefold()
        if period and period not in ('first_half','second_half'):
            issues['non_regulation_offensive_events'] += 1
            continue
        minute = ev.get('minute')
        if isinstance(minute,bool) or not isinstance(minute,int) or minute < 0:
            issues['invalid_offensive_minutes'] += 1
            continue
        team = ev.get('team')
        if not isinstance(team,dict):
            issues['unknown_offensive_team'] += 1
            continue
        team_id = str(team.get('id') or '')
        name = norm_name(team.get('name'))
        if home_id and team_id == str(home_id):
            side = 'home'
        elif away_id and team_id == str(away_id):
            side = 'away'
        elif name == home:
            side = 'home'
        elif name == away:
            side = 'away'
        else:
            issues['unknown_offensive_team'] += 1
            continue
        cls = TYPES[typ]
        counters[side+'_'+cls+'_ft'] += 1
        if cls == 'sot':
            counters[side+'_shots_ft'] += 1
    return {k:counters[k] for k in METRICS}, issues, data.get('coverage'), (meta or {}).get('last_updated',''),len(events)


def get_stats(mid, key, base, timeout):
    if not base.startswith('https://'):
        raise ValueError('Base URL non HTTPS')
    url = base.rstrip('/') + '/football/matches/' + urllib.parse.quote(mid) + '/stats'
    req = urllib.request.Request(url, headers={
        'Authorization':'Bearer '+key, 'Accept':'application/json',
        'User-Agent':'v3c-crosscheck/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def run(args):
    manifest = read_csv(args.manifest)
    if not manifest:
        raise ValueError('Manifest vide')
    load_env(args.env)
    key = (os.environ.get('STATS_API_KEY') or '').strip()
    if key.lower().startswith('bearer '):
        key = key[7:]
    if not args.offline and not key:
        raise ValueError('STATS_API_KEY introuvable')
    base = (os.environ.get('STATS_API_BASE_URL') or 'https://api.thestatsapi.com/api').rstrip('/')
    stats_dir = args.outdir/'stats_cache'
    stats_dir.mkdir(parents=True,exist_ok=True)
    fetched = skipped = failures = 0
    eligible = []
    for entry in manifest:
        mid = safe_id(entry.get('match_id'))
        timeline_file = args.outdir/'timeline_cache'/(mid+'.json')
        if not timeline_file.is_file():
            continue
        eligible.append(entry)
        cache = stats_dir/(mid+'.json')
        if cache.is_file() and not args.force:
            skipped += 1
            continue
        if args.offline or fetched + failures >= args.limit:
            continue
        try:
            payload = get_stats(mid,key,base,args.timeout)
            # Protect cache: only validated stats with expected match id.
            extract_stats(payload,mid)
            temp = cache.with_suffix('.json.tmp')
            temp.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            temp.replace(cache)
            fetched += 1
            print('OK /stats',mid,'(%s/%s nouvelles)'%(fetched,args.limit),flush=True)
        except (ValueError,KeyError,TypeError,urllib.error.HTTPError,urllib.error.URLError,OSError,TimeoutError) as exc:
            failures += 1
            print('ECHEC /stats',mid,type(exc).__name__,file=sys.stderr,flush=True)
        if args.delay and fetched + failures < args.limit:
            time.sleep(args.delay)
    official = []
    reports = []
    statuses = Counter()
    for row in eligible:
        mid = safe_id(row['match_id'])
        timeline_file = args.outdir/'timeline_cache'/(mid+'.json')
        stats_file = stats_dir/(mid+'.json')
        rep = {'match_id':mid, 'home_team':row.get('home_team',''), 'away_team':row.get('away_team','')}
        for m in METRICS:
            rep['timeline_'+m] = ''
            rep['stats_'+m] = ''
            rep['diff_'+m] = ''
        if not stats_file.exists():
            rep['status'] = 'STATS_CACHE_MISSING'
            reports.append(rep)
            statuses[rep['status']] += 1
            continue
        try:
            t_payload = json.loads(timeline_file.read_text(encoding='utf-8'))
            s_payload = json.loads(stats_file.read_text(encoding='utf-8'))
            t_counts,issues,coverage,updated,n_events = extract_timeline(t_payload,row)
            s_counts,internal = extract_stats(s_payload,mid)
            rep.update(timeline_coverage=coverage,timeline_updated=updated,timeline_n_events=n_events,
                       timeline_issues=';'.join('%s=%s'%(k,v) for k,v in sorted(issues.items())),
                       stats_internal_check=internal)
            for m in METRICS:
                rep['timeline_'+m] = t_counts[m]
                rep['stats_'+m] = s_counts[m]
                rep['diff_'+m] = t_counts[m] - s_counts[m]
            # Export /stats values as independent of timeline only (same provider).
            official.append({'match_id':mid, **s_counts})
            if internal != 'OK':
                status = 'STATS_INTERNAL_MISMATCH'
            elif str(coverage).lower() != 'full':
                status = 'TIMELINE_COVERAGE_NOT_FULL'
            elif issues:
                status = 'TIMELINE_ISSUES_REVIEW'
            elif any(t_counts[m] != s_counts[m] for m in METRICS):
                status = 'MISMATCH_6_FT_COUNTS'
            else:
                status = 'MATCH_6_FT_COUNTS'
            rep['status'] = status
        except (ValueError,TypeError,KeyError,OSError) as exc:
            rep['status'] = 'INVALID_CACHE_OR_SCHEMA'
            rep['error_type'] = type(exc).__name__
        statuses[rep['status']] += 1
        reports.append(rep)
    report_columns = ['match_id','home_team','away_team','status','timeline_coverage','timeline_updated',
                      'timeline_n_events','timeline_issues','stats_internal_check','error_type']
    for m in METRICS:
        report_columns += ['timeline_'+m,'stats_'+m,'diff_'+m]
    write_csv(args.outdir/'official_stats_api.csv',['match_id',*METRICS],official)
    write_csv(args.outdir/'timeline_stats_crosscheck.csv',report_columns,reports)
    summary = {'timeline_cached_matches':len(eligible),'new_stats_requests_ok':fetched,
               'new_stats_requests_failed':failures,'already_cached_stats':skipped,
               'compared_matches':len(reports),'statuses':dict(statuses),
               'official_stats_csv':str(args.outdir/'official_stats_api.csv'),
               'crosscheck_report':str(args.outdir/'timeline_stats_crosscheck.csv'),
               'interpretation':'Same-provider crosscheck only; NOT independent validation or training-ready certification'}
    (args.outdir/'timeline_stats_crosscheck_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,default=Path('outputs/pressure_historical'))
    ap.add_argument('--env',type=Path,default=Path('.env'))
    ap.add_argument('--limit',type=int,default=5,help='Maximum de NOUVELLES requetes /stats, defaut 5')
    ap.add_argument('--delay',type=float,default=1.0)
    ap.add_argument('--timeout',type=int,default=20)
    ap.add_argument('--offline',action='store_true',help='Recompare les caches, aucune requete API')
    ap.add_argument('--force',action='store_true',help='Requete meme si /stats deja en cache')
    args=ap.parse_args()
    if args.limit<0 or args.delay<0 or args.timeout<=0:
        ap.error('limit >= 0, delay >= 0 et timeout > 0 requis')
    run(args)

if __name__=='__main__':
    main()
