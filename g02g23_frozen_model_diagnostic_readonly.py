#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G23/P1: diagnostic du modele V4.2 G02-G22 FROZEN (read only).

Depuis le dossier ML_Goal_After_87:
    python3 -B g02g23_frozen_model_diagnostic_readonly.py

NE LIT AUCUN CSV DE LABELS / TIMELINE. NE REENTRAINE RIEN.
Calcule uniquement la distribution des scores en TRAIN depuis les coefficients
et le scaler du modele fige. N'utilise pas VALIDATION pour choisir un seuil.
0 API, 0 ecriture de fichiers, TEST outcomes fermes. Python >= 3.9 stdlib.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('outputs/v4_research')
REPORT = ROOT / 'v42_g02g22_baseline_research_v1/G02G22_BASELINE_L2_C01_RESEARCH_FROZEN.json'
DIGEST = ROOT / 'v42_g02g22_baseline_research_v1/G02G22_BASELINE_L2_C01_RESEARCH_FROZEN.sha256'
DATASET = ROOT / 'v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
ALLOW = ROOT / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
SPLIT = ROOT / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
CHECKSUMS = {
    REPORT: '34c3dfce3ca63e88e80225d95b1f7117722096f0de7a432c914b2948bd5f4eaa',
    DATASET: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
}
FEATURES = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target', 'home_avg_goals_scored',
    'away_avg_goals_scored',
)


def insist(cond, msg):
    if not cond:
        raise SystemExit('G02-G23 STOP : ' + str(msg))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 128), b''):
            h.update(b)
    return h.hexdigest()


def numeric(v, context):
    insist(type(v) in (float, int) and math.isfinite(float(v)),
           'nombre non fini ou invalide : ' + context)
    return float(v)


def logistic(t):
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    ex = math.exp(t)
    return ex / (1.0 + ex)


def percentile_sorted(a, q):
    insist(len(a) > 0 and 0 <= q <= 1, 'percentile invalide')
    t = (len(a) - 1) * q
    lo = int(math.floor(t))
    hi = int(math.ceil(t))
    return a[lo] * (hi - t) + a[hi] * (t - lo) if hi != lo else a[lo]


def read_csv_index(path, required):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        insist(len(headers) == len(set(headers)) and set(required) <= set(headers),
               'colonnes invalides : ' + path.name)
        insist(not {'goal_after_87', 'target', 'label', 'y_true'} & set(headers),
               'un label apparait dans une source sans labels')
        out = {}
        for row in reader:
            mid = row['match_id']
            insist(mid and mid not in out and None not in row,
                   'match_id invalide/duplique dans ' + path.name)
            out[mid] = row
    return out


def main():
    print('=== G02-G23/P1 | DIAGNOSTIC MODELE DEJA FIGE | TRAIN UNIQUEMENT ===')
    for path, expected in CHECKSUMS.items():
        insist(path.is_file() and not path.is_symlink(), 'source absente/symlink : ' + str(path))
        val = digest(path)
        insist(val == expected, 'SHA different : ' + path.name + ' ' + val)
        print('SHA OK :', path.name)
    insist(DIGEST.is_file() and not DIGEST.is_symlink(), 'empreinte G02-G22 manquante')
    insist(DIGEST.read_text(encoding='ascii').strip() == CHECKSUMS[REPORT],
           'empreinte sauvegardee G02-G22 differente')
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    insist(report.get('status') ==
           'RETROSPECTIVE_EXPLORATORY_NOT_INDEPENDENTLY_CERTIFIED_NOT_ASOF',
           'rapport inconnu')
    insist(report.get('test_outcomes_read') is False and
           report.get('test', {}).get('outcome_labels_parsed') == 0 and
           report.get('test', {}).get('timelines_opened') == 0,
           'TEST non protege')
    insist(report.get('feature_names_in_order') == list(FEATURES),
           'ordre features modifie')
    for path in (DATASET, ALLOW, SPLIT):
        insist(report.get('frozen_input_sha256', {}).get(path.name) == CHECKSUMS[path],
               'reference SHA du rapport differente : ' + path.name)
    model = report['model']
    scaler = report['scaler']
    insist(model.get('name') == 'LogisticRegression' and model.get('C') == 0.1
           and model.get('solver') == 'liblinear', 'modele different')
    insist(scaler.get('fit_rows') == 845, 'scaler pas TRAIN-only')
    coef = [numeric(x, 'coef') for x in model['coefficients_in_feature_order']]
    means = [numeric(x, 'mean') for x in scaler['mean']]
    scales = [numeric(x, 'scale') for x in scaler['scale']]
    intercept = model.get('intercept', [])
    insist(len(coef) == len(means) == len(scales) == 10 and len(intercept) == 1
           and all(x > 0 for x in scales), 'vecteurs modele/scaler invalides')
    offset = numeric(intercept[0], 'intercept')

    access = read_csv_index(ALLOW, ('match_id', 'partition'))
    insist(len(access) == 1114, 'allowlist taille incorrecte')
    train_ids = {mid for mid, row in access.items() if row['partition'] == 'TRAIN'}
    val_ids = {mid for mid, row in access.items() if row['partition'] == 'VALIDATION'}
    insist(len(train_ids) == 845 and len(val_ids) == 269 and
           not train_ids.intersection(val_ids) and
           not {row['partition'] for row in access.values()} - {'TRAIN', 'VALIDATION'},
           'partition allowlist invalide')
    split = read_csv_index(SPLIT, ('match_id', 'partition'))
    test_ids = {mid for mid, row in split.items() if row['partition'] == 'TEST'}
    insist(len(split) == 1500 and len(test_ids) == 302 and
           not test_ids.intersection(access) and
           all(split[mid]['partition'] == row['partition'] for mid, row in access.items()),
           'TEST dans allowlist ou partition different')
    # CSV source SANS LABEL. Aucun score/feature TEST n'est converti en nombre.
    with DATASET.open(encoding='utf-8-sig', newline='') as fd:
        reader = csv.DictReader(fd)
        headers = reader.fieldnames or []
        insist(set(('match_id',) + FEATURES) <= set(headers) and
               len(headers) == len(set(headers)), 'colonnes dataset invalides')
        insist(not {'goal_after_87', 'label', 'y_true', 'target'} & set(headers),
               'label detecte dans dataset')
        seen = set()
        scores = []
        for row in reader:
            mid = row['match_id']
            insist(mid and mid not in seen and None not in row, 'match invalide/duplique')
            seen.add(mid)
            if mid not in train_ids:
                continue
            xx = []
            for f in FEATURES:
                raw = row.get(f, '')
                try:
                    x = float(raw)
                except (ValueError, TypeError):
                    raise SystemExit('G02-G23 STOP : feature invalide ' + mid + '/' + f)
                insist(math.isfinite(x) and x >= 0, 'feature non conforme ' + mid + '/' + f)
                xx.append(x)
            logit = offset + sum(c * (x - mu) / s
                                 for c, x, mu, s in zip(coef, xx, means, scales))
            scores.append(logistic(logit))
    insist(len(seen) == 1500 and len(scores) == 845 and train_ids <= seen,
           'dataset/ TRAIN incoherent')
    scores.sort()
    vm = report['validation_metrics']
    insist(vm['alert_count_at_0_60'] == 0 and vm['validation_count'] == 269,
           'rapport de validation different')
    n_train_over = sum(p >= 0.60 for p in scores)
    p_train = 211 / 845
    valid_pos = 90
    valid_neg = 179
    const_brier = (valid_pos * (1-p_train)**2 + valid_neg * p_train**2)/269
    print('=== DIAGNOSTIC TRAIN (PROBAS RECONSTRUITES, SANS LABELS) ===')
    print('TRAIN : 845 | probabilites calculees uniquement depuis le modele fige')
    print('SCORES TRAIN min / p25 / mediane / p75 / p90 / p95 / p99 / max :')
    print(' | '.join('%.4f' % percentile_sorted(scores, t)
                   for t in (0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1)))
    print('TRAIN nombre p>=0.60 :', n_train_over, '/ 845 (diagnostic descriptif, NON performance)')
    print('=== COEFFICIENTS ABSOLUS STANDARDISES (NON CAUSAUX) ===')
    for name, val in sorted(zip(FEATURES, coef), key=lambda x: abs(x[1]), reverse=True):
        print('  %+.6f  %s' % (val, name))
    print('=== COMPARAISONS DEJA FIGEES (PAS DE NOUVELLE VALIDATION) ===')
    print('VALIDATION G02-G22 : alertes au seuil 0.60 = 0 | ROC AUC = %.4f | BRIER = %.4f'
          % (vm['roc_auc'], vm['brier_score']))
    print('BRIER REFERENCE PROBABILITE CONSTANTE TRAIN (211/845) SUR VALIDATION : %.4f'
          % const_brier)
    print('REFERENCE : fixee a partir du taux TRAIN ; ne predit pas a 60% et ne prouve pas as-of')
    print('0 OUVERTURE DE LABEL | 0 TIMELINE | 0 TEST OUTCOME | 0 API | 0 ECRITURE | 0 FIT')
    print('G02-G23/P1 DIAGNOSTIC READONLY PASS | NE PAS CHOISIR SEUIL SUR VALIDATION')


if __name__ == '__main__':
    main()
