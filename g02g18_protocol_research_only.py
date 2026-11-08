#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G18/P1 — pre-register an experimental RESEARCH protocol (no labels).

From the ML_Goal_After_87 project root, with frozen G02-G15/16/17 files:
    python3 -B g02g18_protocol_research_only.py          # READ ONLY
    python3 -B g02g18_protocol_research_only.py --freeze # create NEW JSON only
    python3 -B g02g18_protocol_research_only.py --verify # READ ONLY

This is a planning gate only, not authorization to fit a model or open labels.
Uses Python 3.9+ standard library; 0 API, 0 labels, 0 training.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('outputs/v4_research')
DATA = BASE / 'v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
OBS = BASE / 'v42_g02g15_dataset_research_v1/G02G15_PREMATCH3000_RESEARCH_AUDITED.csv'
DATA_REPORT = BASE / 'v42_g02g15_dataset_research_v1/G02G15_REPORT.json'
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
SPLIT_REPORT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLIT_REPORT.json'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
ELIG_REPORT = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_ELIGIBILITY_POLICY_FROZEN.json'
OUT = BASE / 'v42_g02g18_protocol_research_only_v1'
POLICY = OUT / 'G02G18_EXPERIMENT_PROTOCOL_FROZEN.json'

EXPECTED = {
    DATA: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    OBS: '8937d22ba2c12390ed979ad88be2c2eb97a4373aab3bd6c2290f7ebf270283e3',
    DATA_REPORT: 'bf9a85d0a122bc3b9ba290b29327f805c0d4c8fe1c34ace7e2fe0bd431d26a30',
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    SPLIT_REPORT: '0ff5a497c0453ce89f12d426d9a8e571422e4c495742876660a770559bfb0dfe',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    ELIG_REPORT: '839886c31dd2895c83b751fcbc213ef6cef9d3e7b5622012db8f9810756ee970',
}
FEATURES = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target', 'home_avg_goals_scored',
    'away_avg_goals_scored',
)
PARTS = ('TRAIN', 'VALIDATION', 'TEST')
TOTAL = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
ELIGIBLE = {'TRAIN': 845, 'VALIDATION': 269, 'TEST': 286}


def stop(msg):
    raise SystemExit('G02-G18 STOP : ' + str(msg))


def check(test, msg):
    if not test:
        stop(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
              'en-tetes invalides : ' + str(path))
        rows = list(reader)
    check(all(None not in r for r in rows), 'colonnes supplementaires : ' + str(path))
    return rows, set(reader.fieldnames)


def utc_date(text):
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        stop('date UTC illisible : ' + repr(text))
    check(dt.tzinfo is not None, 'date sans fuseau : ' + repr(text))
    return dt.astimezone(timezone.utc)


def build():
    print('=== G02-G18/P1 | PROTOCOLE EXPERIMENTAL | RESEARCH ONLY ===')
    for path, expected in EXPECTED.items():
        check(path.is_file(), 'source gelee absente : ' + str(path))
        actual = sha(path.read_bytes())
        check(actual == expected, 'SHA incorrect pour ' + path.name + ' : ' + actual)
        print('SHA OK :', path.name)

    data, data_fields = readcsv(DATA)
    split, split_fields = readcsv(SPLIT)
    elig, elig_fields = readcsv(ELIG)
    check(set(FEATURES) | {'match_id', 'kickoff_utc'} <= data_fields,
          'colonnes features absentes')
    check({'match_id', 'kickoff_utc', 'partition', 'utc_day'} <= split_fields,
          'colonnes split absentes')
    check({'match_id', 'kickoff_utc', 'partition', 'research_eligibility',
           'exclusion_reasons'} <= elig_fields, 'colonnes eligibilite absentes')
    forbidden = {'label', 'target', 'target_status', 'target_label', 'goal_after_87', 'y_true'}
    check(not forbidden.intersection(data_fields | split_fields | elig_fields),
          'colonne label/target dans les sources non-label')
    check(len(data) == len(split) == len(elig) == 1500, 'nombre de matchs != 1500')

    def index(rows, name):
        ans = {}
        for row in rows:
            mid = row.get('match_id')
            check(mid and mid not in ans, 'match vide/double dans ' + name)
            ans[mid] = row
        return ans

    dmap = index(data, 'dataset')
    smap = index(split, 'partitions')
    emap = index(elig, 'eligibilite')
    check(set(dmap) == set(smap) == set(emap), 'IDs differents entre les 3 sources')

    totals, good = Counter(), Counter()
    by_dates = defaultdict(list)
    missing = Counter()
    for mid, r in dmap.items():
        s, e = smap[mid], emap[mid]
        check(r['kickoff_utc'] == s['kickoff_utc'] == e['kickoff_utc'],
              'kickoff different : ' + mid)
        part = s['partition']
        check(part in PARTS and part == e['partition'], 'partition differente : ' + mid)
        dt = utc_date(r['kickoff_utc'])
        check(dt.date().isoformat() == s['utc_day'], 'jour UTC incorrect : ' + mid)
        totals[part] += 1
        by_dates[part].append(dt)
        status = e['research_eligibility']
        check(status in ('RESEARCH_ELIGIBLE', 'NOT_ELIGIBLE'),
              'statut eligibilite inconnu : ' + mid)
        if status == 'RESEARCH_ELIGIBLE':
            check(not e['exclusion_reasons'], 'motif sur match admissible : ' + mid)
            good[part] += 1
            for field in FEATURES:
                val = r[field]
                check(val is not None and val.strip() != '',
                      'feature manquante sur candidat : ' + mid + '/' + field)
                try:
                    n = float(val)
                except (TypeError, ValueError):
                    stop('feature non numerique : ' + mid + '/' + field)
                check(math.isfinite(n) and n >= 0,
                      'feature negative/non finie : ' + mid + '/' + field)
        else:
            check(bool(e['exclusion_reasons']), 'exclusion sans motif : ' + mid)
            missing[part] += 1
    check(dict(totals) == TOTAL, 'effectifs partitions differents : ' + str(dict(totals)))
    check(dict(good) == ELIGIBLE, 'effectifs admissibles differents : ' + str(dict(good)))
    check(len({x.date() for x in sum(by_dates.values(), [])}) ==
          sum(len({x.date() for x in by_dates[p]}) for p in PARTS),
          'jour UTC partage entre partitions')
    check(max(by_dates['TRAIN']) < min(by_dates['VALIDATION']) and
          max(by_dates['VALIDATION']) < min(by_dates['TEST']),
          'ordre chronologique invalide')

    p = {
        'stage': 'G02-G18/P1',
        'name': 'V4.2 retrospective research experimental protocol',
        'status': 'PROTOCOL_ONLY_NOT_ASOF_CERTIFIED_NOT_TRAINING_AUTHORIZATION',
        'source_frozen_sha256': {path.name: expected for path, expected in EXPECTED.items()},
        'cohort': {
            'eligible_only': True,
            'original_feature_names_in_order': list(FEATURES),
            'feature_count': 10,
            'partition_total': TOTAL,
            'partition_eligible': ELIGIBLE,
            'excluded_rows_retained_in_g02g17_manifest': True,
            'no_missing_value_imputation': True,
            'no_reassignment_or_shuffle': True,
        },
        'outcome_definition_for_later_separate_review': {
            'event': 'at_least_one_goal_strictly_after_minute_87',
            'minute_87_itself_positive': False,
            'not_opened_in_this_stage': True,
            'target_file_path': None,
        },
        'candidate_model_for_later_scientific_go_nogo': {
            'type': 'LogisticRegression',
            'implementation': 'scikit-learn',
            'penalty': 'l2',
            'C': 0.1,
            'solver': 'liblinear',
            'max_iter': 2000,
            'class_weight': None,
            'input': 'all_original_10_features_in_order',
            'scaler': 'StandardScaler',
            'scaler_fit_on': 'TRAIN_ELIGIBLE_ONLY',
            'fixed_alert_probability_threshold': 0.60,
            'no_threshold_selection_on_TEST': True,
            'status': 'DESIGN_ONLY_NOT_FIT',
        },
        'evaluation_plan': {
            'TRAIN': 'Model fitting only after a separate GO decision and verified label provenance.',
            'VALIDATION': 'One chronological retrospective evaluation; no post-hoc optimization using TEST.',
            'TEST': 'SEALED. Do not open outcomes until entire modeling protocol is frozen and authorized.',
            'metrics': ['alert_count', 'positive_predictions', 'true_positives',
                        'precision_at_0.60', 'recall', 'roc_auc_if_defined',
                        'brier_score', 'log_loss', 'wilson_95pct_for_precision'],
            'undefined_precision_on_zero_alerts': True,
            'no_claim_of_60pct_or_profitable_betting_without_evidence': True,
            'no_test_iterative_reuse': True,
            'no_live_deployment_from_retrospective_evaluation': True,
        },
        'limitations': [
            'As-of historical result publication not certified.',
            'Live event publication at the prediction instant not certified.',
            'Research features retrospectively collected, NOT prospectively verified.',
            'Independent chronological TEST is reserved, not yet evaluated.',
        ],
        'api_calls': 0,
        'labels_read': False,
        'test_outcomes_read': False,
        'training_performed': False,
        'asof_certified': False,
    }
    raw = (json.dumps(p, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    print('PARTITIONS :', dict(totals))
    print('ADMISSIBLES :', dict(good))
    print('EXCLUS :', dict(missing))
    print('CANDIDATS AVEC 10 VALEURS NUMERIQUES :', sum(good.values()))
    print('SHA PROTOCOLE :', sha(raw))
    print('AS-OF NON CERTIFIE | 0 API | 0 LABEL | 0 TRAIN')
    return raw


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    m = arg.add_mutually_exclusive_group()
    m.add_argument('--freeze', action='store_true')
    m.add_argument('--verify', action='store_true')
    opt = arg.parse_args()
    raw = build()
    if opt.verify:
        check(POLICY.is_file(), 'fichier de protocole gele absent')
        check(POLICY.read_bytes() == raw, 'le protocole gele ne correspond pas au recalcul')
        print('G02-G18 VERIFY PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
    elif opt.freeze:
        check(not OUT.exists(), 'dossier existe deja : refus ecrasement')
        OUT.mkdir(parents=True, exist_ok=False)
        with POLICY.open('xb') as fh:
            fh.write(raw)
        check(POLICY.read_bytes() == raw, 'ecriture differente du precheck')
        print('G02-G18 FREEZE PASS | PROTOCOLE SEUL | 0 API | 0 LABEL | 0 TRAIN')
        print('IMPORTANT : PAS DE GO ENTRAINEMENT | AS-OF NON CERTIFIE')
    else:
        print('G02-G18 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
        print('Pour figer ensuite : python3 -B g02g18_protocol_research_only.py --freeze')


if __name__ == '__main__':
    main()

