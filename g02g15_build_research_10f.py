#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G15 | Assemble research-only 10-feature dataset, with NO labels.

Run from project root (where g02g13_collect_100.py and g02g10_short.py exist):
    python3 -B g02g15_build_research_10f.py             # strict READ-ONLY preflight
    python3 -B g02g15_build_research_10f.py --freeze    # write NEW research files only

No API, no access to target labels/results, no training. Historical as-of and
live event-publication timing are NOT certified. No imputation or row deletion.
Python 3.9+; standard library only.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import runpy
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import g02g13_collect_100 as g

BASE = Path('outputs/v4_research')
ALL = BASE / 'v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
LIVE = BASE / 'v42_g02f6_live8_adapter/V42_LIVE8_ADAPTED_RESEARCH.csv'
QUALITY = BASE / 'v42_g02f6_live8_adapter/V42_LIVE8_ADAPTER_QUALITY.csv'
LIVE_REPORT = BASE / 'v42_g02f6_live8_adapter/G02F6_REPORT.json'
REGISTRY = BASE / 'v42_g02f2_quality_pre87/G02F2_PRE87_QUALITY_REGISTRY.csv'
OUT = BASE / 'v42_g02g15_dataset_research_v1'

EXPECTED = {
    ALL: '4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
    LIVE: '2656ff84f9096b7ec8a1f7b8517b1c9c70d8f339f210a2d20ad2e4375c91b146',
    REGISTRY: '4cfe6146db28a4f05837bd4b66673eb146099bf7363197e5096a51de3cdd1b17',
}
# Independent G02-G14/P3 result observed before this stage; fail closed if changed.
EXPECTED_PREMATCH_FINGERPRINT = '8700d33a615044b866a1082c78e0cdc9ca992a2d64d0367193ca94f9f96e2917'
EXPECTED_STATUSES = {'COMPLETE20': 2894, 'PARTIAL10_19': 60, 'MISSING_LT10': 46}

LIVE_FIELDS = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target',
)
PRE_FIELDS = ('home_avg_goals_scored', 'away_avg_goals_scored')
META_FIELDS = (
    'match_id', 'kickoff_utc', 'home_team_id', 'away_team_id',
)
EXTRA_FIELDS = (
    'home_history_n', 'away_history_n',
    'home_history_status', 'away_history_status',
    'home_prematch_source_sha256', 'away_prematch_source_sha256',
    'live_feature_status', 'live_risk_reasons', 'live_source_sha256',
    'prematch_asof_certified', 'live_publication_timing_certified',
    'dataset_status',
)
DATA_FIELDS = META_FIELDS + LIVE_FIELDS + PRE_FIELDS + EXTRA_FIELDS
OBS_FIELDS = (
    'match_id', 'side', 'team_id', 'safe_cutoff_utc', 'history_n',
    'status', 'avg_goals_scored', 'asof_certified', 'source_sha256',
)


def stop(msg):
    raise SystemExit('G02-G15 STOP : ' + str(msg))


def check(ok, msg):
    if not ok:
        stop(msg)


def read_csv_strict(path):
    check(path.is_file(), 'source absente : ' + str(path))
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
              'entetes CSV invalides : ' + str(path))
        rows = list(reader)
    check(all(None not in r for r in rows), 'colonnes supplementaires : ' + str(path))
    return rows


def positive_int(value, name):
    try:
        check(value != '' and str(value).strip() == str(int(value)), name + ' non entier : ' + repr(value))
        n = int(value)
        check(n >= 0, name + ' negatif')
        return n
    except (ValueError, TypeError):
        stop(name + ' non entier : ' + repr(value))


def numeric_feature(value, field, mid):
    if value is None or str(value).strip() == '':
        check(field == 'minutes_since_last_shot_on_target',
              'live feature manquante ' + mid + '/' + field)
        return ''
    try:
        f = float(value)
    except (TypeError, ValueError):
        stop('live feature non numerique : ' + mid + '/' + field)
    check(math.isfinite(f) and f >= 0, 'live feature negative/non-finie : ' + mid + '/' + field)
    if field == 'minutes_since_last_shot_on_target':
        check(f <= 87, 'minute depuis dernier SOT > 87 : ' + mid)
    else:
        check(f.is_integer(), 'compte live non entier : ' + mid + '/' + field)
    return str(value).strip()


def validate_live(row):
    mid = row['match_id']
    clean = {field: numeric_feature(row.get(field), field, mid) for field in LIVE_FIELDS}
    check(int(float(clean['sot_80_87'])) <= int(float(clean['shots_80_87'])),
          'SOT 80-87 > tirs 80-87 : ' + mid)
    check(int(float(clean['sot_80_87'])) <= (
          int(float(clean['home_shots_on_target_87'])) +
          int(float(clean['away_shots_on_target_87']))),
          'SOT 80-87 > total SOT <87 : ' + mid)
    return clean


def canonical_csv(rows, fields):
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n', extrasaction='raise')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode('utf-8')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pilot_calculations():
    path = Path('g02g10_short.py')
    check(path.is_file(), 'script pilote absent : g02g10_short.py')
    previous_argv = sys.argv[:]
    buffer = io.StringIO()
    try:
        sys.argv = [str(path)]  # MUST NOT set --freeze
        with contextlib.redirect_stdout(buffer):
            namespace = runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = previous_argv
    log = buffer.getvalue()
    check('G02-G10 PLAN OK | AUCUNE API | AUCUNE ECRITURE | AUCUN ENTRAINEMENT' in log,
          'audit pilote non valide')
    output = namespace['output']
    check(len(output) == 52 and namespace['counts']['score_uses'] == 1039,
          'bilan pilote different')
    return output


def assemble_prematch():
    """Recompute 3000 team-match rows exactly like independently passed P3."""
    for path, expected in {**g.HASHES, ALL: EXPECTED[ALL]}.items():
        check(path.is_file() and g.sha(path) == expected,
              'source pre-match gelee absente/modifiee : ' + str(path))
    contract = json.loads(g.C.read_text(encoding='utf-8'))
    check(contract.get('historical_status_asof_certified') is False
          and contract.get('training_allowed') is False
          and contract.get('last_n_historical_matches') == 20,
          'contrat G02-G11 invalide')

    targets = read_csv_strict(ALL)
    general = read_csv_strict(g.T)
    check(len(targets) == 3000 and len(general) == 2948,
          'nombre observations prematch incorrect')
    full_keys = [(r['match_id'], r['side'], r['team_id']) for r in targets]
    general_keys = {(r['match_id'], r['side'], r['team_id']) for r in general}
    check(len(set(full_keys)) == 3000 and len(general_keys) == 2948,
          'observations dupliquees')
    by = defaultdict(list)
    for row in general:
        by[row['team_id']].append(row)
    check(len(by) == 1098, 'equipes generales != 1098')

    recomputed = {}
    for pilot in pilot_calculations():
        k = (pilot['match_id'], pilot['side'], pilot['team_id'])
        check(k not in recomputed and k not in general_keys,
              'chevauchement pilote/general : ' + repr(k))
        recomputed[k] = dict(pilot)

    score_uses = 0
    for i, tid in enumerate(sorted(by), 1):
        team_targets = by[tid]
        state = g.scan(tid, [
            {'match_id': r['match_id'], 'cutoff': g.utc(r['safe_cutoff_utc'])}
            for r in team_targets
        ])
        check(state['state'] == 'COVERED', 'cache general non couvert : ' + tid)
        dated, page_shas = [], []
        for page in range(1, state['cached_pages'] + 1):
            p = g.cache_path(tid, page)
            wrapper = json.loads(p.read_text(encoding='utf-8'))
            check(wrapper.get('status') == 'RAW_NOT_ASOF_CERTIFIED',
                  'statut de source inattendu : ' + str(p))
            d, _ = g.inspect_response(wrapper, tid, page)
            dated.extend(d)
            page_shas.append(g.sha(p))
        for target in team_targets:
            k = (target['match_id'], target['side'], tid)
            check(k not in recomputed, 'doublon pre-match : ' + repr(k))
            cutoff = g.utc(target['safe_cutoff_utc'])
            eligible = [x for x in dated if x[0] < cutoff and x[1] != target['match_id']]
            check(not (len(eligible) > 20 and eligible[19][0] == eligible[20][0]),
                  'egalite non resolue au rang20 : ' + repr(k))
            picked = eligible[:20]
            values = []
            for dt, match_id, match in picked:
                side = 'home' if match['home_team']['id'] == tid else 'away'
                regulation = (match.get('score') or {}).get('regulation')
                check(isinstance(regulation, dict) and g.goal_valid(regulation.get(side)),
                      'score historique invalide : ' + tid + '/' + match_id)
                values.append(int(float(regulation[side])))
            n = len(values)
            status = 'COMPLETE20' if n == 20 else (
                'PARTIAL10_19' if n >= 10 else 'MISSING_LT10')
            recomputed[k] = {
                'match_id': target['match_id'], 'side': target['side'],
                'team_id': tid, 'safe_cutoff_utc': target['safe_cutoff_utc'],
                'history_n': n, 'status': status,
                'avg_goals_scored': format(sum(values)/n, '.12f') if n >= 10 else '',
                'asof_certified': '0', 'source_sha256': ','.join(page_shas),
            }
            score_uses += n
        if i % 300 == 0:
            print('  Moyennes recalculees :', i, '/1098 equipes')

    check(len(recomputed) == 3000 and set(recomputed) == set(full_keys),
          'jointure prematch incomplete/dupliquee')
    counts = Counter(r['status'] for r in recomputed.values())
    check(dict(counts) == EXPECTED_STATUSES, 'statuts non conformes : ' + repr(counts))
    check(score_uses == 57905, 'usages historiques G02-G13 differents')
    for k, row in recomputed.items():
        n = positive_int(row['history_n'], 'history_n')
        check(0 <= n <= 20, 'history_n > 20 : ' + repr(k))
        check((row['avg_goals_scored'] == '') == (n < 10),
              'moyenne absente/inattendue : ' + repr(k))
        if n >= 10:
            try:
                avg = float(row['avg_goals_scored'])
            except (TypeError, ValueError):
                stop('moyenne non numerique : ' + repr(k))
            check(math.isfinite(avg) and 0 <= avg <= 100,
                  'moyenne non-finie/hors bornes : ' + repr(k))

    fingerprint = hashlib.sha256()
    for k in sorted(recomputed):
        r = recomputed[k]
        fingerprint.update(('\t'.join((*k, str(r['history_n']), r['status'],
                                       r['avg_goals_scored'], r['source_sha256'])) + '\n').encode())
    digest = fingerprint.hexdigest()
    check(digest == EXPECTED_PREMATCH_FINGERPRINT,
          'G02-G14/P3 empreinte prematch differente : ' + digest)
    print('  Empreinte G02-G14/P3 :', digest, '| MATCH')

    # Match-level organization: two target sides per match, same kickoff, each
    # strictly 4 hours before its own match. NO target labels are ever loaded.
    match = defaultdict(dict)
    for row in targets:
        mid, side, tid = row['match_id'], row['side'], row['team_id']
        check(side in ('home', 'away') and side not in match[mid],
              'home/away duplique ou invalide : ' + mid)
        check(g.utc(row['kickoff_utc']) - g.utc(row['safe_cutoff_utc']) == timedelta(hours=4),
              'cutoff != kickoff -4h : ' + mid)
        match[mid][side] = row
    check(len(match) == 1500 and all(set(d) == {'home','away'} for d in match.values()),
          'il manque le cote home ou away de certains matchs')
    check(len({r['team_id'] for r in targets}) == 1116,
          'nombre equipes uniques != 1116')
    for mid, d in match.items():
        check(g.utc(d['home']['kickoff_utc']) == g.utc(d['away']['kickoff_utc']),
              'kickoff home != away : ' + mid)
        check(d['home']['team_id'] != d['away']['team_id'],
              'meme equipe de chaque cote : ' + mid)
    ordered_obs = sorted(recomputed.values(), key=lambda r:
                         (r['match_id'], r['side'], r['team_id']))
    return recomputed, match, ordered_obs, counts, digest


def assemble_live(match):
    for path, expected in EXPECTED.items():
        check(path.is_file() and g.sha(path) == expected,
              'live source manquante ou SHA divergent : ' + str(path))
    check(QUALITY.is_file() and LIVE_REPORT.is_file(), 'qualite/rapport G02-F6 manquant')
    report = json.loads(LIVE_REPORT.read_text(encoding='utf-8'))
    check(report.get('records') == 1500, 'rapport live != 1500')
    check(report.get('feature_output_sha256') == EXPECTED[LIVE],
          'F6 rapport vs live feature incorrect')
    check(report.get('quality_output_sha256') == g.sha(QUALITY),
          'qualite G02-F6 modifiee')
    check(report.get('source_registry_sha256') == EXPECTED[REGISTRY],
          'source registre live modifiee')
    check(report.get('independently_certified') is False,
          'rapport live signale certification inattendue')
    features, qual = read_csv_strict(LIVE), read_csv_strict(QUALITY)
    check(len(features) == len(qual) == 1500, 'effectif F6 different de 1500')
    fl, ql = {}, {}
    for row in features:
        mid = row['match_id']
        check(mid and mid not in fl and mid in match, 'live match inconnu/duplique : ' + mid)
        fl[mid] = row
    for row in qual:
        mid = row['match_id']
        check(mid and mid not in ql and mid in match, 'qualite match inconnu/duplique : ' + mid)
        check(row.get('feature_status') in ('FEATURE_CANDIDATE', 'REVIEW'),
              'statut qualite non reconnu : ' + mid)
        ql[mid] = row
    check(set(fl) == set(ql) == set(match), 'jointure live/qualite/prematch incorrecte')
    statuses = Counter()
    for mid in match:
        row, qr = fl[mid], ql[mid]
        kickoff = g.utc(match[mid]['home']['kickoff_utc'])
        check(g.utc(row['kickoff_utc']) == kickoff and
              g.utc(qr['kickoff_utc']) == kickoff,
              'kickoff live/prematch different : ' + mid)
        validate_live(row)
        statuses[qr['feature_status']] += 1
    return fl, ql, statuses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze', action='store_true',
                    help='ecrit des fichiers de recherche dans un NOUVEAU dossier (sans overwrite)')
    args = ap.parse_args()
    print('=== G02-G15 | 10 FEATURES | RESEARCH ONLY | 0 API | 0 LABEL | 0 TRAIN ===')
    print('MODE :', 'FREEZE (nouvelles sorties seulement)' if args.freeze else 'DRY RUN (lecture seule)')
    prem, match, observations, statuses, fp = assemble_prematch()
    live, quality, qstatus = assemble_live(match)
    data = []
    missing_last_sot = 0
    both_pre_valid = 0
    for mid in sorted(match, key=lambda m: (g.utc(match[m]['home']['kickoff_utc']), m)):
        h, a = match[mid]['home'], match[mid]['away']
        hk = (mid, 'home', h['team_id'])
        ak = (mid, 'away', a['team_id'])
        hp, ap = prem[hk], prem[ak]
        features = validate_live(live[mid])
        if features['minutes_since_last_shot_on_target'] == '':
            missing_last_sot += 1
        if hp['avg_goals_scored'] and ap['avg_goals_scored']:
            both_pre_valid += 1
        r = {
            'match_id': mid, 'kickoff_utc': h['kickoff_utc'],
            'home_team_id': h['team_id'], 'away_team_id': a['team_id'],
            **features,
            'home_avg_goals_scored': hp['avg_goals_scored'],
            'away_avg_goals_scored': ap['avg_goals_scored'],
            'home_history_n': hp['history_n'], 'away_history_n': ap['history_n'],
            'home_history_status': hp['status'], 'away_history_status': ap['status'],
            'home_prematch_source_sha256': hp['source_sha256'],
            'away_prematch_source_sha256': ap['source_sha256'],
            'live_feature_status': quality[mid]['feature_status'],
            'live_risk_reasons': quality[mid].get('risk_reasons') or '',
            'live_source_sha256': quality[mid]['source_sha256'],
            'prematch_asof_certified': '0',
            'live_publication_timing_certified': '0',
            'dataset_status': 'RESEARCH_RETROSPECTIVE_NOT_ASOF_CERTIFIED',
        }
        data.append(r)
    check(len(data) == 1500 and len({r['match_id'] for r in data}) == 1500,
          'dataset final != 1500 matchs uniques')
    check(sum(1 for r in observations if r['avg_goals_scored'] != '') == 2954,
          'compte valeurs prematch non manquantes incorrect')
    # Structural checks on every row/column. No label/target/outcome columns.
    check(len(DATA_FIELDS) == len(set(DATA_FIELDS)) and
          len(LIVE_FIELDS) == 8 and len(PRE_FIELDS) == 2,
          'schema non conforme')
    for row in data:
        check(set(row) == set(DATA_FIELDS), 'colonnes incoherentes : ' + row['match_id'])
        for f in PRE_FIELDS:
            v = row[f]
            check(v == '' or (math.isfinite(float(v)) and float(v) >= 0),
                  'valeur prematch invalide : ' + row['match_id'])
    data_bytes = canonical_csv(data, DATA_FIELDS)
    obs_bytes = canonical_csv(observations, OBS_FIELDS)
    report_obj = {
        'id': 'G02G15_DATASET_RESEARCH_V1',
        'mode': 'RESEARCH_RETROSPECTIVE_NOT_ASOF_CERTIFIED',
        'training_authorized_by_this_stage': False,
        'label_or_outcome_columns_included': False,
        'labels_opened': False,
        'api_calls': 0,
        'matches': 1500, 'teams': 1116, 'prematch_observations': 3000,
        'live_features': list(LIVE_FIELDS), 'prematch_features': list(PRE_FIELDS),
        'feature_count': 10,
        'prematch_status_counts': dict(sorted(statuses.items())),
        'live_quality_status_counts': dict(sorted(qstatus.items())),
        'matches_with_both_prematch_values_present': both_pre_valid,
        'matches_with_missing_last_sot': missing_last_sot,
        'prematch_g02g14p3_fingerprint_sha256': fp,
        'input_sha256': {
            'G02G5_TARGET_OBSERVATIONS_FROZEN.csv': EXPECTED[ALL],
            'G02G11_PREMATCH_CONTRACT_FROZEN.json': g.HASHES[g.C],
            'G02G12_REMAINING_TARGETS_FROZEN.csv': g.HASHES[g.T],
            'V42_LIVE8_ADAPTED_RESEARCH.csv': EXPECTED[LIVE],
            'V42_LIVE8_ADAPTER_QUALITY.csv': g.sha(QUALITY),
            'G02F6_REPORT.json': g.sha(LIVE_REPORT),
        },
        'output_sha256': {
            'V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv': sha_bytes(data_bytes),
            'G02G15_PREMATCH3000_RESEARCH_AUDITED.csv': sha_bytes(obs_bytes),
        },
        'limitations': [
            'Past match records downloaded retrospectively; historical results as-of cutoff unverified.',
            'Live-event publication timing as-of minute 87 unverified.',
            'Cutoff is target kickoff minus four hours; finished match publication at cutoff not demonstrated.',
            'No future-label leakage tested through evaluation because outcomes are not accessed here.',
            'Do not treat dataset as training-approved or deployment-ready.',
            'Missing prematch values and last SOT left empty; any imputation must be learned within train only.',
            'Research-quality flags preserved; no silent row exclusion.',
        ],
    }
    report_bytes = (json.dumps(report_obj, ensure_ascii=False, sort_keys=True,
                               indent=2) + '\n').encode('utf-8')
    print('MATCHS :', len(data), '| EQUIPES : 1116 | OBSERVATIONS PREMATCH :', len(observations))
    print('STATUTS PREMATCH :', dict(sorted(statuses.items())))
    print('QUALITE LIVE :', dict(sorted(qstatus.items())))
    print('MATCHS AVEC 2 MOYENNES PREMATCH DISPONIBLES :', both_pre_valid)
    print('MATCHS SANS DERNIER SOT OBSERVE :', missing_last_sot)
    print('SHA DATASET 1500 :', sha_bytes(data_bytes))
    print('SHA PROVENANCE 3000 :', sha_bytes(obs_bytes))
    print('AS-OF : NON CERTIFIE | 0 API | 0 LABEL | 0 TRAIN')
    if not args.freeze:
        print('G02-G15 PRECHECK PASS | 0 ECRITURE')
        print('Pour figer ensuite : python3 -B g02g15_build_research_10f.py --freeze')
        return
    check(not OUT.exists(), 'dossier sortie existe deja : ' + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    for filename, payload in (
        ('V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv', data_bytes),
        ('G02G15_PREMATCH3000_RESEARCH_AUDITED.csv', obs_bytes),
        ('G02G15_REPORT.json', report_bytes),
    ):
        path = OUT / filename
        with path.open('xb') as handle:
            handle.write(payload)
        check(g.sha(path) == sha_bytes(payload), 'ecriture divergente : ' + filename)
        print('GEL', path, '| SHA256', g.sha(path))
    print('G02-G15 FREEZE PASS | 1500 MATCHS | 10 FEATURES | 0 API | 0 LABEL | 0 TRAIN')
    print('IMPORTANT : RESEARCH ONLY, NON CERTIFIE AS-OF, PAS DE GO ENTRAINEMENT')


if __name__ == '__main__':
    main()
