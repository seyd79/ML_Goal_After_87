#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G19/P1 — TRAIN-only feature scaling preparation and label-access allowlist.

From project root (Python >=3.9, stdlib only):
  python3 -B g02g19_train_preprocessing_guard.py
  python3 -B g02g19_train_preprocessing_guard.py --freeze
  python3 -B g02g19_train_preprocessing_guard.py --verify

No outcome, label file, external connection, model fit or TEST feature transformation.
Does NOT authorize training or certify historical as-of availability. The proposed
TRAIN-only mean/scale reproduces the population-variance convention (ddof=0) of
StandardScaler; exact library round-off equivalence has not been checked.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('outputs/v4_research')
DATA = BASE / 'v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
PROTOCOL = BASE / 'v42_g02g18_protocol_research_only_v1/G02G18_EXPERIMENT_PROTOCOL_FROZEN.json'

EXPECTED_SHA = {
    DATA: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    PROTOCOL: '2607dedecb886baaaf7d63be60e3da0d212749fd9671ba463be7cb0b8db05373',
}
FEATURES = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target', 'home_avg_goals_scored',
    'away_avg_goals_scored',
)
PARTS = ('TRAIN', 'VALIDATION', 'TEST')
EXPECTED_ALL = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
EXPECTED_ELIG = {'TRAIN': 845, 'VALIDATION': 269, 'TEST': 286}
OUT = BASE / 'v42_g02g19_train_preprocessing_guard_v1'
ALLOW = OUT / 'G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
REPORT = OUT / 'G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'
ACCESS_FIELDS = ('match_id', 'kickoff_utc', 'partition')


def stop(message):
    raise SystemExit('G02-G19 STOP : ' + str(message))


def check(ok, message):
    if not ok:
        stop(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_sha():
    for path, expected in EXPECTED_SHA.items():
        check(path.is_file(), 'source gelee absente : ' + str(path))
        actual = digest(path.read_bytes())
        check(actual == expected, 'empreinte SHA256 differente : ' + str(path) + ' | ' + actual)
        print('SHA OK :', path.name)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        check(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
              'en-tetes invalides : ' + str(path))
        columns = set(reader.fieldnames)
        rows = list(reader)
    check(all(None not in r for r in rows), 'lignes avec champs en trop : ' + str(path))
    check(not ({'label', 'target', 'target_status', 'target_label',
                'goal_after_87', 'y_true'} & columns), 'colonne de label interdite : ' + str(path))
    return rows, columns


def index_by_match(rows, name):
    ans = {}
    for row in rows:
        mid = row.get('match_id', '')
        check(mid and mid not in ans, 'match_id vide ou double : ' + name + '/' + repr(mid))
        ans[mid] = row
    return ans


def parse_utc(s):
    try:
        val = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        stop('date invalide : ' + repr(s))
    check(val.tzinfo is not None, 'date sans fuseau : ' + repr(s))
    return val.astimezone(timezone.utc)


def numerical(s, field, mid):
    check(isinstance(s, str) and s.strip() != '', 'valeur vide ' + field + ' ' + mid)
    try:
        n = float(s)
    except (TypeError, ValueError):
        stop('valeur non numerique : ' + field + ' / ' + mid)
    check(math.isfinite(n) and n >= 0, 'valeur non finie ou negative : ' + field + ' / ' + mid)
    return n


def train_only_parameters(training_rows, features=FEATURES):
    """Pure function. No validation or TEST samples are passed in here."""
    check(training_rows, 'aucune ligne TRAIN')
    stats = {}
    for field in features:
        xs = [numerical(row[field], field, row['match_id']) for row in training_rows]
        mean = math.fsum(xs) / len(xs)
        variance = math.fsum((v - mean) ** 2 for v in xs) / len(xs)
        check(math.isfinite(mean) and math.isfinite(variance) and variance >= 0,
              'variance non finie : ' + field)
        stddev = math.sqrt(variance)
        scale = stddev if stddev != 0.0 else 1.0
        stats[field] = {
            'n_train': len(xs), 'mean_train': mean,
            'variance_train_ddof0': variance, 'scale_train': scale,
            'minimum_train': min(xs), 'maximum_train': max(xs),
            'zero_variance': stddev == 0.0,
        }
    return stats


def canonical_access_csv(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=ACCESS_FIELDS, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def make_artifacts():
    print('=== G02-G19/P1 | TRAIN-ONLY PREPROCESSING | 0 API/LABEL/MODEL ===')
    verify_sha()
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    check(p.get('status') == 'PROTOCOL_ONLY_NOT_ASOF_CERTIFIED_NOT_TRAINING_AUTHORIZATION',
          'statut protocole G02-G18 invalide')
    spec = p.get('candidate_model_for_later_scientific_go_nogo', {})
    cohort = p.get('cohort', {})
    check(spec.get('scaler') == 'StandardScaler' and
          spec.get('scaler_fit_on') == 'TRAIN_ELIGIBLE_ONLY' and
          spec.get('input') == 'all_original_10_features_in_order' and
          spec.get('fixed_alert_probability_threshold') == 0.60,
          'politique du pretraitement G02-G18 differente')
    check(cohort.get('original_feature_names_in_order') == list(FEATURES) and
          cohort.get('partition_eligible') == EXPECTED_ELIG and
          p.get('labels_read') is False and p.get('test_outcomes_read') is False and
          p.get('asof_certified') is False,
          'contrat de recherche G02-G18 non respecte')

    data, dcols = read_csv(DATA)
    splits, scols = read_csv(SPLIT)
    elig, ecols = read_csv(ELIG)
    check({'match_id', 'kickoff_utc'} | set(FEATURES) <= dcols,
          'colonnes dataset manquantes')
    check({'match_id', 'kickoff_utc', 'partition'} <= scols, 'colonnes split manquantes')
    check({'match_id', 'kickoff_utc', 'partition', 'research_eligibility'} <= ecols,
          'colonnes eligibilite manquantes')
    check(len(data) == len(splits) == len(elig) == 1500, 'effectifs sources != 1500')
    dmap, smap, emap = (index_by_match(v, name) for v, name in
                        ((data, 'DATA'), (splits, 'SPLIT'), (elig, 'ELIG')))
    check(set(dmap) == set(smap) == set(emap), 'IDs divergeants')

    counts, admiss = Counter(), Counter()
    fit_rows, allowed_rows = [], []
    time_ranges = {k: [] for k in PARTS}
    for mid, s in smap.items():
        d, e = dmap[mid], emap[mid]
        check(d['kickoff_utc'] == e['kickoff_utc'] == s['kickoff_utc'],
              'kickoff divergent : ' + mid)
        part = s['partition']
        check(part in PARTS and part == e['partition'], 'partition non concordante : ' + mid)
        counts[part] += 1
        t = parse_utc(s['kickoff_utc'])
        time_ranges[part].append(t)
        if 'utc_day' in s:
            check(t.date().isoformat() == s['utc_day'], 'jour UTC errone : ' + mid)
        status = e['research_eligibility']
        check(status in ('RESEARCH_ELIGIBLE', 'NOT_ELIGIBLE'),
              'admissibilite inconnue : ' + mid)
        if status == 'RESEARCH_ELIGIBLE':
            admiss[part] += 1
            if part == 'TRAIN':
                fit_rows.append(d)  # The ONLY rows whose feature values are processed.
            elif part == 'VALIDATION':
                pass  # No features from VALIDATION are used to compute any statistic.
            else:
                pass  # TEST: no feature value consumed, no label or output access.
            if part != 'TEST':
                allowed_rows.append({
                    'match_id': mid, 'kickoff_utc': s['kickoff_utc'],
                    'partition': part,
                })
    check(dict(counts) == EXPECTED_ALL, 'repartition non conforme : ' + repr(counts))
    check(dict(admiss) == EXPECTED_ELIG, 'cohortes eligibles non conformes : ' + repr(admiss))
    check(max(time_ranges['TRAIN']) < min(time_ranges['VALIDATION']) <
          max(time_ranges['VALIDATION']) < min(time_ranges['TEST']),
          'ordre chronologique impossible')
    days = [{t.date() for t in time_ranges[k]} for k in PARTS]
    check(not (days[0] & days[1]) and not (days[1] & days[2]) and
          not (days[0] & days[2]), 'jour UTC partage')
    check(len(fit_rows) == 845 and len(allowed_rows) == 1114, 'effectifs LABEL ACCESS incorrects')
    check(all(r['partition'] != 'TEST' for r in allowed_rows), 'TEST dans acces labels')
    allowed_rows.sort(key=lambda r: (PARTS.index(r['partition']), parse_utc(r['kickoff_utc']), r['match_id']))
    fit_rows.sort(key=lambda r: (parse_utc(r['kickoff_utc']), r['match_id']))

    param = train_only_parameters(fit_rows)
    raw_ids = canonical_access_csv(allowed_rows)
    assert len({r['match_id'] for r in allowed_rows}) == 1114
    report = {
        'stage': 'G02-G19/P1',
        'status': 'PREPROCESSING_RESEARCH_ONLY_NOT_TRAINING_AUTHORIZATION',
        'source_frozen_sha256': {path.name: h for path, h in EXPECTED_SHA.items()},
        'train_only_allowlist_sha256': digest(raw_ids),
        'partitions_total': EXPECTED_ALL,
        'partitions_research_eligible': EXPECTED_ELIG,
        'allowed_future_label_access_partitions': ['TRAIN', 'VALIDATION'],
        'allowed_future_label_access_count': 1114,
        'labels_accessed_in_this_stage': False,
        'test_labels_or_outcomes_accessed_in_this_stage': False,
        'test_ids_in_allowlist': False,
        'feature_names_in_order': list(FEATURES),
        'normalization_design': {
            'algorithm': 'StandardScaler-compatible population mean and standard deviation',
            'variance_ddof': 0,
            'fit_partition': 'TRAIN_ELIGIBLE_ONLY',
            'fit_n': 845,
            'zero_variance_scale': 1.0,
            'no_imputation': True,
            'test_features_used_for_scaler': False,
            'validation_features_used_for_scaler': False,
            'statistics': param,
        },
        'model_training_performed': False,
        'model_hyperparameters_modified': False,
        'historical_asof_certified': False,
        'live_publication_timing_certified': False,
        'no_authorization_to_open_labels_or_fit_model': True,
        'limitations': [
            'Only retrospective research; historical publication times unavailable.',
            'This report describes numerical scaler statistics, not trained model weights.',
            'TEST outcomes must remain sealed for one final evaluation.',
            'Future label source must be independently verified before use.',
        ],
    }
    raw_report = (json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
                  + '\n').encode('utf-8')
    print('TOTALS :', dict(counts), '| ELIGIBLES :', dict(admiss))
    print('SCALER FIT : TRAIN admissibles =', len(fit_rows))
    print('COLONNES :', len(param), '| VARIANCE NULLE :',
          [k for k, v in param.items() if v['zero_variance']])
    print('AUTORISATIONS LABEL FUTURES : TRAIN 845 + VALIDATION 269 = 1114 | TEST 0')
    print('SHA ALLOWLIST :', digest(raw_ids))
    print('SHA PROTOCOLE PRETRAITEMENT :', digest(raw_report))
    print('AS-OF NON CERTIFIE | 0 API | 0 LABEL | 0 MODELE ENTRAINE')
    return raw_ids, raw_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    raw_ids, raw_report = make_artifacts()
    if args.verify:
        for path, expected in ((ALLOW, raw_ids), (REPORT, raw_report)):
            check(path.is_file(), 'sortie gelee manquante : ' + str(path))
            check(path.read_bytes() == expected, 'fichier gele different : ' + str(path))
            print('VERIFY SHA OK :', path.name, digest(expected))
        print('G02-G19 VERIFY PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 MODEL')
        return
    if not args.freeze:
        print('G02-G19 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 MODEL')
        print('Pour figer : python3 -B g02g19_train_preprocessing_guard.py --freeze')
        return
    check(not OUT.exists(), 'dossier gel deja present : pas d ecrasement ' + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    for path, raw in ((ALLOW, raw_ids), (REPORT, raw_report)):
        with path.open('xb') as fh:
            fh.write(raw)
        check(path.read_bytes() == raw, 'erreur verification ecriture ' + str(path))
        print('GEL :', path, '| SHA256', digest(raw))
    print('G02-G19 FREEZE PASS | 845 TRAIN | 269 VALIDATION | 0 TEST LABEL | 0 MODEL')
    print('IMPORTANT : AUCUN GO POUR OUVRIR LES LABELS OU ENTRAINER | AS-OF NON CERTIFIE')


if __name__ == '__main__':
    main()
