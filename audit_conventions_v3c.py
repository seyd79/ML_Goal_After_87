#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit OFFLINE des conventions TheStatsAPI: /timeline vs /stats.

Python 3.9+, bibliotheque standard. AUCUNE requete reseau.
Ne modifie JAMAIS les caches, le scanner ou les features d'entrainement.

Compare des regles GLOBALES sur les 5 premiers matchs (diagnostic initial),
puis leur tenue sur les 10 matchs suivants (verification hors echantillon).
Ce classement n'autorise PAS a qualifier les reconstructions comme training-ready.

python audit_conventions_v3c.py --manifest historical_1710_manifest.csv \
  --outdir outputs/pressure_historical
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

BASE_SHOTS = frozenset(('shot_on_target', 'shot_off_target', 'shot_blocked'))
SPECIAL = frozenset(('goal', 'shot_woodwork', 'penalty_scored', 'penalty_missed', 'own_goal', 'penalty_saved'))
SOT_BASE = frozenset(('shot_on_target',))
CORNER = 'corner_kick'
# Explicit, global hypotheses. Do not tune on each game or create corrections via /stats FT.
SHOT_RULES = {
    'BASE': frozenset(),
    'GOALS': frozenset(('goal',)),
    'GOALS_WOODWORK': frozenset(('goal', 'shot_woodwork')),
    'GOALS_WOODWORK_PEN_SCORED': frozenset(('goal', 'shot_woodwork', 'penalty_scored')),
    'GOALS_WOODWORK_ALL_PEN': frozenset(('goal', 'shot_woodwork', 'penalty_scored', 'penalty_missed')),
    'GOALS_PEN_SCORED': frozenset(('goal', 'penalty_scored')),
}
SOT_RULES = {
    'BASE': frozenset(),
    'GOALS': frozenset(('goal',)),
    'GOALS_PEN_SCORED': frozenset(('goal', 'penalty_scored')),
}


def norm(value):
    text = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return re.sub('[^a-z0-9]+', '_', ''.join(c for c in text if not unicodedata.combining(c))).strip('_')


def read_manifest(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def team_side(event, row):
    team = event.get('team')
    if not isinstance(team, dict):
        return None
    tid = str(team.get('id') or '')
    hid = str(row.get('home_id') or '')
    aid = str(row.get('away_id') or '')
    if hid and tid == hid:
        return 'home'
    if aid and tid == aid:
        return 'away'
    label = norm(team.get('name'))
    if label and label == norm(row.get('home_team')) and norm(row.get('home_team')) != norm(row.get('away_team')):
        return 'home'
    if label and label == norm(row.get('away_team')) and norm(row.get('home_team')) != norm(row.get('away_team')):
        return 'away'
    return None


def metric_stats(payload, mid):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get('match_id') != mid:
        raise ValueError('stats match_id absent ou incoherent')
    overview = data.get('overview')
    if not isinstance(overview, dict):
        raise ValueError('stats overview absent')
    output = {}
    for typ, field in [('shots', 'total_shots'), ('sot', 'shots_on_target'), ('corner', 'corner_kicks')]:
        m = overview.get(field)
        both = m.get('all') if isinstance(m, dict) else None
        if not isinstance(both, dict):
            raise ValueError('stats champ non disponible: ' + field)
        for side in ('home', 'away'):
            v = both.get(side)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError('stats valeur non valide %s %s' % (field, side))
            output[(side, typ)] = v
    return output


def metric_events(payload, row, mid):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get('match_id') != mid:
        raise ValueError('timeline match_id absent ou incoherent')
    events = data.get('events')
    if not isinstance(events, list):
        raise ValueError('timeline events absents')
    counts = Counter()
    issues = Counter()
    seen = set()
    for event in events:
        if not isinstance(event, dict):
            issues['non_dict_event'] += 1
            continue
        seq = event.get('sequence')
        if seq is not None:
            seq = str(seq)
            if seq in seen:
                issues['duplicate_sequence'] += 1
                continue
            seen.add(seq)
        typ = str(event.get('type') or '')
        # Include only regulation halves. Event in another period is not FT regulation.
        period = str(event.get('period') or '').lower()
        if period and period not in ('first_half', 'second_half'):
            if typ in BASE_SHOTS or typ in SPECIAL or typ == CORNER:
                issues['offensive_non_regulation'] += 1
            continue
        if typ not in BASE_SHOTS and typ not in SPECIAL and typ != CORNER:
            continue
        minute = event.get('minute')
        if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
            issues['bad_offensive_minute'] += 1
            continue
        side = team_side(event, row)
        if side is None:
            issues['unknown_offensive_team'] += 1
            continue
        counts[(side, typ)] += 1
    return counts, dict(issues), data.get('coverage'), len(events)


def score_variant(records, metric, extra, swapped=False):
    err = 0
    exact_pairs = 0
    exact_matches = 0
    for rec in records:
        pair_ok = 0
        for side in ('home','away'):
            if metric == 'shots':
                predicted = sum(rec['counts'][(side, t)] for t in BASE_SHOTS | extra)
            elif metric == 'sot':
                predicted = sum(rec['counts'][(side, t)] for t in SOT_BASE | extra)
            else:
                count_side = ('away' if side == 'home' else 'home') if swapped else side
                predicted = rec['counts'][(count_side, CORNER)]
            expected = rec['stats'][(side, metric)]
            err += abs(predicted - expected)
            if predicted == expected:
                exact_pairs += 1
                pair_ok += 1
        if pair_ok == 2:
            exact_matches += 1
    return {'abs_error': err, 'exact_team_counts':exact_pairs, 'exact_matches':exact_matches, 'n_team_counts':len(records)*2}


def comparison(records, metric, variants):
    train = records[:5]
    holdout = records[5:]
    all_r = []
    for name, rules in variants.items():
        swap = bool(rules) if metric == 'corner' else False
        extra = frozenset() if metric == 'corner' else rules
        tr = score_variant(train, metric, extra, swap) if train else None
        ho = score_variant(holdout, metric, extra, swap) if holdout else None
        all_r.append({'name': name, 'train': tr, 'holdout': ho,
                      'complexity': int(swap) if metric == 'corner' else len(extra)})
    if train:
        all_r.sort(key=lambda v: (v['train']['abs_error'], -v['train']['exact_team_counts'], v['complexity'], v['name']))
    return all_r


def run(manifest, outdir):
    manifest_rows = read_manifest(manifest)
    rows = []
    not_cached = 0
    skipped = []
    for entry in manifest_rows:
        mid = str(entry.get('match_id') or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{3,100}', mid):
            continue
        timeline_file = outdir/'timeline_cache'/(mid+'.json')
        stats_file = outdir/'stats_cache'/(mid+'.json')
        if not timeline_file.exists() or not stats_file.exists():
            not_cached += 1
            continue
        try:
            timeline = json.loads(timeline_file.read_text(encoding='utf-8'))
            stats = json.loads(stats_file.read_text(encoding='utf-8'))
            counts, issues, coverage, n_events = metric_events(timeline, entry, mid)
            expectation = metric_stats(stats, mid)
            if issues.get('unknown_offensive_team') or issues.get('bad_offensive_minute'):
                skipped.append({'match_id':mid,'reason':'TEAM_OR_MINUTE_AMBIGUITY','issues':issues})
                continue
            rows.append({'match_id':mid,'counts':counts,'stats':expectation,'issues':issues,
                         'coverage':coverage,'n_events':n_events})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skipped.append({'match_id':mid,'reason':type(exc).__name__})
    # Preserves the original manifest order: 5 used for hypothesis selection, remainder for verification.
    selections = {
        'shots': comparison(rows, 'shots', SHOT_RULES),
        'sot': comparison(rows, 'sot', SOT_RULES),
        'corner': comparison(rows, 'corner', {'AS_IS':False,'SWAPPED':True}),
    }
    print('\n=== HYPOTHESES GLOBALes (aucune correction appliquee) ===')
    print('Matches with both caches:',len(rows),' | first 5 initial | next %s verification' % max(0,len(rows)-5))
    print('Potentially ambiguous and skipped:',len(skipped),' | missing a cache:',not_cached)
    for metric, variants in selections.items():
        print('\n',metric.upper())
        print('  Hypothese                         initial err/exact    verification err/exact')
        for variant in variants:
            tr,ho = variant['train'],variant['holdout']
            fmt=lambda x: ('%s / %s-%s'% (x['abs_error'],x['exact_team_counts'],x['n_team_counts'])) if x else 'sans donnees'
            print('  %-34s %-20s %s' % (variant['name'],fmt(tr),fmt(ho)))
        best = variants[0]
        print('  Meilleure sur les 5 premiers (PAS validation):',best['name'])
    # Report includes *none* of the training labels or model probabilities.
    detailed = []
    for r in rows:
        o={'match_id':r['match_id'],'coverage':r['coverage'],'event_count':r['n_events'],
           'issues':json.dumps(r['issues'],ensure_ascii=False)}
        for side in ('home','away'):
            for ev in sorted(BASE_SHOTS | SPECIAL | {CORNER}):
                o[side+'_evt_'+ev] = r['counts'][(side,ev)]
            for typ in ('shots','sot','corner'):
                o[side+'_stats_'+typ] = r['stats'][(side,typ)]
        detailed.append(o)
    outdir.mkdir(parents=True,exist_ok=True)
    output = outdir/'audit_conventions_v3c_detail.csv'
    if detailed:
        with output.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(detailed[0]))
            writer.writeheader();writer.writerows(detailed)
    summary = {'n_comparable':len(rows),'n_skipped_ambiguous':len(skipped),
               'n_manifest_without_complete_cache':not_cached,
               'fit_first_n':min(5,len(rows)),'verify_next_n':max(0,len(rows)-5),
               'variants':selections,'skipped':skipped,
               'IMPORTANT':'Diagnostic interne meme fournisseur; aucune correction automatique, aucune garantie as-of ni training-ready.'}
    json_output=outdir/'audit_conventions_v3c_summary.json'
    json_output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nRapports:',output,' | ',json_output)
    print('LECTURE SEULE: caches et fichiers V3-C non modifies. 0 appel API.')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('historical_1710_manifest.csv'))
    ap.add_argument('--outdir',type=Path,default=Path('outputs/pressure_historical'))
    args=ap.parse_args()
    run(args.manifest,args.outdir)

if __name__=='__main__':
    main()
