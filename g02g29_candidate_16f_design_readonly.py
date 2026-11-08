#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G29 | Contrat CANDIDAT 16 features, TRAIN uniquement, lecture seule.

Depuis la racine du projet ML_Goal_After_87 :
  python3 -B g02g29_candidate_16f_design_readonly.py
  python3 -B g02g29_candidate_16f_design_readonly.py --audit-train

Le mode standard controle les sources gelees et la selection conceptuelle,
sans ouvrir les timelines. Le mode --audit-train verifie les 845 timelines
TRAIN admissibles et les 16 colonnes candidates sans lire les labels.

AUCUN entrainement, aucun fit, aucune API, aucun fichier cree, aucune lecture
VALIDATION ou TEST, aucune certification de disponibilite live/as-of.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

G27_PATH = Path('g02g27_feature_feasibility_train_readonly.py')
G27_SHA = 'fbf1f86ef60f48ee4d33e1dd534335b05854e3a50f37fa571edef399d7bcb3bd'
G28_PATH = Path('g02g28_train_feature_contract_readonly.py')
G28_SHA = '66145298ec86deccfea9098253f89e9416ca848c3c73645be9e7ab8a5c10f6ce'

# Six features anciennes conservees (recherche). Toutes sans label.
FROZEN_6 = (
    'score_87_home', 'score_87_away',
    'home_shots_on_target_87', 'away_shots_on_target_87',
    'home_avg_goals_scored', 'away_avg_goals_scored',
)
# Dix nouvelles features, extraites des seuls evenements < 87 minutes.
NEW_10 = (
    'home_shots_80_87', 'away_shots_80_87',
    'home_sot_80_87', 'away_sot_80_87',
    'home_corners_80_87', 'away_corners_80_87',
    'shots_84_87', 'sot_84_87',
    'home_goals_80_87', 'away_goals_80_87',
)
FEATURES = FROZEN_6 + NEW_10

# Distinguer ce qui est omis POUR EVITER LES ADDITIONS EXACTES.
DROPPED_OLD = (
    'shots_80_87', 'sot_80_87', 'corners_80_87',
    'minutes_since_last_shot_on_target',
)
DROPPED_CANDIDATES = (
    'shots_80_84', 'score_80_home', 'score_80_away',
    'home_minutes_since_last_sot', 'away_minutes_since_last_sot',
)


def stop(message):
    raise SystemExit('G02-G29 STOP : ' + str(message))


def require(test, message):
    if not test:
        stop(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fd:
        for part in iter(lambda: fd.read(131072), b''):
            h.update(part)
    return h.hexdigest()


def load_preflight():
    print('=== G02-G29/P1 | CONTRAT CANDIDAT 16F | TRAIN ONLY ===')
    for path, expected in ((G27_PATH, G27_SHA), (G28_PATH, G28_SHA)):
        require(path.is_file() and not path.is_symlink(), 'source absente/lien symbolique : ' + str(path))
        require(sha(path) == expected, 'SHA source incorrect : ' + str(path))
        print('SHA SCRIPT OK :', path.name)

    import g02g28_train_feature_contract_readonly as g28
    g27 = g28.guarded_g27()  # inclut le test synthetique G27
    train_ids, dataset, registry = g27.load_guards()  # SHA + TEST et VAL interdits

    require(len(FEATURES) == len(set(FEATURES)) == 16, 'schema non unique/16')
    require(len(FROZEN_6) == 6 and len(NEW_10) == 10, 'schema 6+10 invalide')
    require(set(NEW_10) <= set(g27.CANDIDATES), 'feature absente du reconstructeur G27')
    require(not (set(FEATURES) & set(DROPPED_OLD)), 'ancienne feature redondante conservee')
    require(not (set(FEATURES) & set(DROPPED_CANDIDATES)), 'candidate redondante retenue')
    require(len(train_ids) == 845, 'cohorte TRAIN non 845')

    # Test d'absence de fuite dans le code de reconstruction, deja teste par G27.
    print('PLAN : 6 features historiques + 10 candidates = 16F, sans labels')
    print('CONSERVEES :', ', '.join(FROZEN_6))
    print('NOUVELLES :', ', '.join(NEW_10))
    print('OMISES (exactes additions / couverture) :', ', '.join(DROPPED_OLD + DROPPED_CANDIDATES))
    print('ATTENTION : 16F = hypothese de representation, NON modele selectionne')
    print('ATTENTION : absence de timestamp de publication; as-of NON certifie')
    print('G02-G29/P1 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TIMELINE | 0 FIT')
    return g27, g28, train_ids, dataset, registry


def audit_train(g27, g28, train_ids, dataset, registry):
    print('=== G02-G29/P2 | CONSTRUCTION EN MEMOIRE | 845 TRAIN | NO LABEL ===')
    require(all(p.is_dir() and not p.is_symlink() for p in g27.RAW_DIRS),
            'dossier timeline brut absent/lien symbolique')
    raw_dirs = {p.resolve() for p in g27.RAW_DIRS}
    columns = {field: [] for field in FEATURES}
    observations = 0
    timestamp_matches = 0
    for mid in train_ids:
        row, reg = dataset[mid], registry[mid]
        p = Path(reg['source_path'])
        require(p.is_file() and not p.is_symlink() and p.suffix == '.json' and
                p.stem == mid and p.resolve().parent in raw_dirs,
                'chemin brut non autorise : ' + mid)
        require(sha(p) == reg['source_sha256'], 'SHA timeline modifie : ' + mid)
        with p.open('r', encoding='utf-8') as fd:
            wrapper = json.load(fd)
        require(isinstance(wrapper, dict) and wrapper.get('match_id') == mid and
                str(wrapper.get('home_id')) == str(row['home_team_id']) and
                str(wrapper.get('away_id')) == str(row['away_team_id']) and
                wrapper.get('kickoff_utc') == row['kickoff_utc'],
                'metadata match incorrect : ' + mid)
        response = wrapper.get('timeline_response')
        require(isinstance(response, dict) and isinstance(response.get('data'), dict),
                'timeline_response.data absent : ' + mid)
        result, issues, _, has_ts, _ = g27.reconstruct(response['data'].get('events'), wrapper, row)
        require(not issues, 'anomalie reconstruction : ' + mid + ' | ' + str(dict(issues)))
        g28.validate_invariants(result, row)  # 6 identities et inclusions, sans labels
        timestamp_matches += int(bool(has_ts))
        for col in FEATURES:
            raw = row[col] if col in FROZEN_6 else result[col]
            require(raw is not None and str(raw) != '', 'feature absente : ' + mid + '/' + col)
            x = float(raw)
            require(math.isfinite(x) and x >= 0, 'valeur non finie/negative : ' + mid + '/' + col)
            columns[col].append(x)
        observations += 1

    require(observations == 845 and all(len(v) == 845 for v in columns.values()),
            '845 valeurs par feature non obtenues')
    print('MATCHS CONFORMES :', observations, '/845 | VAL : 0 | TEST : 0')
    print('COLONNES | distinctes | moyenne | ecart-type (TRAIN) | zero sur TRAIN')
    flat = []
    for col, values in columns.items():
        mu = sum(values) / len(values)
        variance = sum((x-mu)*(x-mu) for x in values) / len(values)
        sd = math.sqrt(variance)
        if sd < 1e-12:
            flat.append(col)
        print('%-31s | %3d | %9.4f | %9.4f | %d' %
              (col, len(set(values)), mu, sd, sum(x == 0 for x in values)))
    require(not flat, 'colonnes constantes TRAIN : ' + ','.join(flat))

    print('CONTROLE CORRELATIONS LINEAIRES ENTRE FEATURES (TRAIN, SANS LABEL) :')
    high = []
    mean = {k: sum(v)/len(v) for k,v in columns.items()}
    sd = {k: math.sqrt(sum((x-mean[k])**2 for x in v)/len(v)) for k,v in columns.items()}
    for i, left in enumerate(FEATURES):
        for right in FEATURES[i+1:]:
            cov = sum((a-mean[left])*(b-mean[right])
                      for a,b in zip(columns[left], columns[right])) / 845
            corr = cov / (sd[left]*sd[right])
            if abs(corr) >= 0.95:
                high.append((left, right, round(corr, 4)))
    print('PAIRES |corr| >= 0.95 :', high if high else 'aucune')

    # Rang du schema + constante : optionnel pour ne pas imposer numpy.
    try:
        import numpy as np
    except ImportError:
        print('RANG MATRICIEL : non calcule (numpy absent)')
    else:
        mat = np.column_stack([columns[k] for k in FEATURES])
        matrix = np.column_stack([np.ones(845), (mat-mat.mean(axis=0)) / mat.std(axis=0)])
        rank = int(np.linalg.matrix_rank(matrix, tol=1e-9))
        print('RANG [CONSTANTE + 16 FEATURES] :', rank, '/17')
        require(rank == 17, 'dependance lineaire exacte detectee sur TRAIN')

    print('TIMESTAMPS PUBLICATION EVENEMENT :', timestamp_matches, '/845 matchs')
    print('AUCUN LABEL | AUCUN REENTRAINEMENT | VAL/TEST NON OUVERTS')
    print('LES 16F NE SONT PAS CERTIFIEES LIVE, NI PROUVEES PREDICTIVES')
    print('0 API | 0 ECRITURE | 0 LABEL | 0 FIT')
    print('G02-G29/P2 CANDIDATE_16F_AUDIT PASS | TRAIN ONLY | RESEARCH ONLY')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-train', action='store_true',
                        help='reconstitue 16F sur 845 TRAIN seulement, en memoire')
    args = parser.parse_args()
    g27, g28, train_ids, dataset, registry = load_preflight()
    if args.audit_train:
        audit_train(g27, g28, train_ids, dataset, registry)
    else:
        print('Puis : python3 -B g02g29_candidate_16f_design_readonly.py --audit-train')


if __name__ == '__main__':
    main()
