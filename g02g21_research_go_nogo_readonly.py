#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G21/P1 — decision gate for V4.2 retrospective research baseline.

Run from the ML_Goal_After_87 project root:
    python3 -B g02g21_research_go_nogo_readonly.py

Reads only FROZEN non-label CSV manifests and JSON audit/protocol reports.
Does NOT open the labels CSV, any raw timeline, or any TEST outcome.
Does NOT train, save, make API calls, or write files. Python 3.9+ stdlib.

A PASS establishes authorization eligibility for a limited exploratory research
experiment, not a claim of independent certification, as-of availability,
performance, betting profitability, or production readiness.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BASE = Path('outputs/v4_research')
DATA = BASE / 'v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
PROTOCOL = BASE / 'v42_g02g18_protocol_research_only_v1/G02G18_EXPERIMENT_PROTOCOL_FROZEN.json'
ALLOW = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
SCALER = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'
AUDIT = BASE / 'v42_g02g20_p2h_internal_label_reconciliation_v1/G02G20_AUTHORIZED_LABEL_RECONCILIATION_FROZEN.json'
EXPECTED = {
    DATA: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    PROTOCOL: '2607dedecb886baaaf7d63be60e3da0d212749fd9671ba463be7cb0b8db05373',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    SCALER: '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
    AUDIT: '179eda9311c681c63dbe28b09281365c030f65f923dfdb95ebae881edf428bce',
}
FEATURES = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target', 'home_avg_goals_scored',
    'away_avg_goals_scored',
)
TOTAL = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
ELIGIBLE = {'TRAIN': 845, 'VALIDATION': 269, 'TEST': 286}


def require(condition, explanation):
    if not condition:
        raise SystemExit('G02-G21 STOP : ' + explanation)


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as fd:
        for block in iter(lambda: fd.read(131072), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as fd:
        reader = csv.DictReader(fd)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                'en-tete invalide : ' + path.name)
        require(set(fields) <= set(reader.fieldnames), 'colonnes absentes : ' + path.name)
        require(not {'goal_after_87', 'y_true', 'target_label', 'label'} & set(reader.fieldnames),
                'colonne de resultat interdite : ' + path.name)
        out = {}
        for r in reader:
            mid = r['match_id']
            require(mid and mid not in out and None not in r,
                    'match_id duplique/invalide : ' + path.name)
            out[mid] = r
    return out


def verify_protocol(proto, scaler, audit):
    require(proto.get('status') == 'PROTOCOL_ONLY_NOT_ASOF_CERTIFIED_NOT_TRAINING_AUTHORIZATION',
            'protocole G02-G18 de type inattendu')
    require(proto.get('labels_read') is False and proto.get('test_outcomes_read') is False
            and proto.get('asof_certified') is False, 'G02-G18 limites inconsistantes')
    spec = proto.get('candidate_model_for_later_scientific_go_nogo', {})
    require(spec.get('type') == 'LogisticRegression' and spec.get('penalty') == 'l2'
            and spec.get('C') == 0.1 and spec.get('solver') == 'liblinear'
            and spec.get('max_iter') == 2000 and spec.get('class_weight') is None
            and spec.get('scaler') == 'StandardScaler'
            and spec.get('scaler_fit_on') == 'TRAIN_ELIGIBLE_ONLY'
            and spec.get('fixed_alert_probability_threshold') == 0.60,
            'configuration baseline differente du protocole gele')
    require(proto.get('cohort', {}).get('original_feature_names_in_order') == list(FEATURES)
            and proto.get('cohort', {}).get('partition_eligible') == ELIGIBLE,
            'liste features/cohorte G02-G18 differente')
    require(scaler.get('status') == 'PREPROCESSING_RESEARCH_ONLY_NOT_TRAINING_AUTHORIZATION',
            'rapport scaler G02-G19 non conforme')
    require(scaler.get('feature_names_in_order') == list(FEATURES)
            and scaler.get('partitions_total') == TOTAL
            and scaler.get('partitions_research_eligible') == ELIGIBLE,
            'statistiques/partitions G02-G19 incoherentes')
    require(scaler.get('train_only_allowlist_sha256') == EXPECTED[ALLOW],
            'fingerprint allowlist differente')
    norm = scaler.get('normalization_design', {})
    require(norm.get('fit_partition') == 'TRAIN_ELIGIBLE_ONLY'
            and norm.get('fit_n') == 845
            and norm.get('variance_ddof') == 0 and norm.get('no_imputation') is True
            and norm.get('test_features_used_for_scaler') is False
            and norm.get('validation_features_used_for_scaler') is False,
            'normalisation TRAIN-only non demontree dans le rapport')
    stats = norm.get('statistics', {})
    require(set(stats) == set(FEATURES), 'nombre/nom statistiques scalaire incorrect')
    for feat in FEATURES:
        x = stats[feat]
        require(x.get('n_train') == 845 and x.get('zero_variance') is False,
                'variance/size incoherente : ' + feat)
        for f in ('mean_train', 'variance_train_ddof0', 'scale_train'):
            v = x.get(f)
            require(type(v) in (float, int) and math.isfinite(v),
                    'statistique non finie : ' + feat + '/' + f)
        require(x['variance_train_ddof0'] > 0 and x['scale_train'] > 0,
                'ecart-type/variance non positive : ' + feat)
    require(scaler.get('test_labels_or_outcomes_accessed_in_this_stage') is False
            and scaler.get('model_training_performed') is False,
            'G02-G19 a deja lu resultats ou entraine')
    require(audit.get('status') == 'INTERNAL_RECONCILIATION_ONLY_NOT_INDEPENDENT_NOT_ASOF',
            'statut audit labels inconnu')
    require(audit.get('authorized_csv_records_sha256') ==
            'b5d3aee84eb2211c1ec1c59f3479ed03f7601c49f3b627583bfc37d088920153',
            'fingerprint labels autorises ne correspond pas au gel')
    require(audit.get('authorized_timeline_sha_manifest_sha256') ==
            '92a1aba20c5345822ff7efb7469c834b3349d24476535556334e423e794d3e98',
            'fingerprint timelines autorisees ne correspond pas au gel')
    require(audit.get('asof_certified') is False and
            audit.get('training_authorized_by_this_report') is False and
            audit.get('model_trained') is False, 'audit G02-G20 incoherent')
    bypart = audit.get('counts', {})
    require(bypart.get('TRAIN', {}).get('MATCHS') == 845 and
            bypart.get('TRAIN', {}).get('POSITIFS') == 211 and
            bypart.get('VALIDATION', {}).get('MATCHS') == 269 and
            bypart.get('VALIDATION', {}).get('POSITIFS') == 90,
            'effectifs de labels audites incorrects')
    require(bypart['TRAIN'].get('INDEPENDENT_CERTIFIED_FLAG_1') == 0 and
            bypart['VALIDATION'].get('INDEPENDENT_CERTIFIED_FLAG_1') == 0,
            'flags certification ont change : reviser politique')
    c = audit.get('counts_total', {})
    require(c.get('authorized_matches') == 1114 and c.get('positive_labels') == 301
            and c.get('independently_certified_flags_1') == 0
            and c.get('differences') == 0 and c.get('TEST_timelines_read') == 0
            and c.get('TEST_label_values_interpreted') == 0,
            'totaux et protection TEST non conformes')


def main():
    print('=== G02-G21/P1 | DECISION SCIENTIFIQUE BASELINE RECHERCHE | READONLY ===')
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), 'source introuvable / lien symbolique : ' + str(path))
        current = checksum(path)
        require(current == expected, 'SHA incorrect : ' + path.name + ' ' + current)
        print('SHA OK :', path.name)
    split = read_rows(SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = read_rows(ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    allow = read_rows(ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    require(len(split) == len(elig) == 1500 and set(split) == set(elig),
            'split/elig differents')
    require(dict(Counter(row['partition'] for row in split.values())) == TOTAL,
            'effectifs chronologiques incorrects')
    expected_allowed = set()
    good = Counter()
    for mid, row in elig.items():
        require(row['partition'] == split[mid]['partition']
                and row['kickoff_utc'] == split[mid]['kickoff_utc'],
                'partition/jour divergent : ' + mid)
        if row['research_eligibility'] == 'RESEARCH_ELIGIBLE':
            good[row['partition']] += 1
            if row['partition'] in ('TRAIN', 'VALIDATION'):
                expected_allowed.add(mid)
        else:
            require(row['research_eligibility'] == 'NOT_ELIGIBLE',
                    'statut autre : ' + mid)
    require(dict(good) == ELIGIBLE and set(allow) == expected_allowed and len(allow) == 1114,
            'ensemble de labels autorises ne correspond pas a la cohorte')
    require(dict(Counter(r['partition'] for r in allow.values())) ==
            {'TRAIN': 845, 'VALIDATION': 269}, 'repartition allowlist incorrecte')
    require(all(r['partition'] == split[mid]['partition'] and
                r['kickoff_utc'] == split[mid]['kickoff_utc']
                for mid, r in allow.items()), 'allowlist non concordante')
    require(not set(allow) & {mid for mid, r in split.items() if r['partition'] == 'TEST'},
            'TEST dans autorisations labels')
    # No content access to the label CSV, raw timelines, or TEST features.
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    s = json.loads(SCALER.read_text(encoding='utf-8'))
    a = json.loads(AUDIT.read_text(encoding='utf-8'))
    verify_protocol(p, s, a)
    print('COHORTE : TRAIN 845 | VALIDATION 269 | TEST reserve 286')
    print('TRAIN LABELS AUDITES : 211 positifs / 845 | VALIDATION : 90 / 269')
    print('PRETRAITEMENT : 10 features | TRAIN seulement | aucune variance nulle')
    print('MODELE PRE-ENREGISTRE : LogisticRegression L2 C=0.1 solver=liblinear, seuil=0.60')
    print('PREUVES : concordance interne seulement | certifies independamment=0')
    print('LIMITES : as-of NON certifie | TEST ferme | performances inconnues')
    print('DECISION : GO CONDITIONNEL pour baseline de RECHERCHE RETROSPECTIVE uniquement')
    print('DECISION : NO-GO pour validation independante certifiee et deploiement LIVE')
    print('0 API | 0 LABEL LU | 0 TIMELINE OUVERTE | 0 ECRITURE | 0 MODEL')
    print('G02-G21/P1 RESEARCH_GATE PASS | ENTRAINEMENT NON EXECUTE | TEST FERME')


if __name__ == '__main__':
    main()
