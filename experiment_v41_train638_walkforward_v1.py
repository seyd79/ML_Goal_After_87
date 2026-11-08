#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etape 254 : SIMPLE6 walk-forward sur TRAIN638 uniquement.

Aucun acces a VALIDATION154, TEST_FUTUR, API, V3-C ou modeles LIVE.
Evaluation exploratoire : labels et snapshots historiques RESEARCH_ONLY.
"""
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('outputs/v4_research')
SOURCE = ROOT / 'V41_TRAIN638_B29_RESEARCH_V1.csv'
OUTPUT = ROOT / 'v41_walkforward_simple6_v1'
PRED = OUTPUT / 'TRAIN638_SIMPLE6_WALKFORWARD_PREDICTIONS_V1.csv'
SUMMARY = OUTPUT / 'TRAIN638_SIMPLE6_WALKFORWARD_SUMMARY_V1.json'

FEATURES = [
    'score_total_87', 'score_diff_abs_87', 'is_draw_87',
    'shots_total_last5', 'sot_total_last5', 'corners_total_last5',
]
# Definitions fixees par date, sans regarder les resultats.
FOLDS = [
    ('2026-05', '2026-05-01', '2026-06-01'),
    ('2026-06', '2026-06-01', '2026-07-01'),
    ('2026-07', '2026-07-01', '2026-08-01'),
    ('2026-08-01_19', '2026-08-01', '2026-08-20'),
    ('2026-08-20_24', '2026-08-20', '2026-08-25'),
]
# Parametres simples bloques avant evaluation, pas de recherche de seuil ici.
THRESHOLD = 0.30
REGULARIZATION_C = 0.1


def read_data(path):
    if not path.is_file():
        raise SystemExit('Dataset introuvable : {}'.format(path))
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        need = {'match_id', 'kickoff_utc', 'goal_after_87'} | set(FEATURES)
        missing = sorted(need - set(names))
        if missing:
            raise SystemExit('Colonnes absentes : {}'.format(missing))
        rows = list(reader)
    if len(rows) != 638:
        raise SystemExit('TRAIN638 attendu, {} lignes trouvees'.format(len(rows)))
    ids = [r['match_id'].strip() for r in rows]
    if any(not mid for mid in ids) or len(set(ids)) != len(ids):
        raise SystemExit('match_id vides ou dupliques')
    for row in rows:
        raw = row['kickoff_utc'].strip()
        try:
            dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        except ValueError:
            raise SystemExit('Date illisible pour {} : {}'.format(row['match_id'], raw))
        if dt.tzinfo is None:
            raise SystemExit('Date sans fuseau : {}'.format(row['match_id']))
        row['_dt'] = dt.astimezone(timezone.utc)
        row['_date'] = row['_dt'].strftime('%Y-%m-%d')
        if row['goal_after_87'].strip() not in ('0', '1', '0.0', '1.0'):
            raise SystemExit('Label invalide : {}'.format(row['match_id']))
        row['_y'] = int(float(row['goal_after_87']))
        for c in FEATURES:
            val = row[c].strip()
            if val in ('', 'NA', 'NaN', 'nan', 'None', 'null'):
                row['_'+c] = np.nan
            else:
                try:
                    num = float(val)
                except ValueError:
                    raise SystemExit('Feature illisible : {} {}'.format(row['match_id'], c))
                if not math.isfinite(num) or num < 0:
                    raise SystemExit('Feature invalide : {} {}'.format(row['match_id'], c))
                row['_'+c] = num
    rows.sort(key=lambda r: (r['_dt'], r['match_id']))
    if sum(r['_y'] for r in rows) != 162:
        raise SystemExit('Nombre de positifs inattendu')
    return rows


def matrix(rows):
    return np.array([[r['_'+f] for f in FEATURES] for r in rows], dtype=float)


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(int(a) for a in y)) == 2 else None


def metrics(items):
    if not items:
        return None
    ys = np.array([int(r['label']) for r in items], dtype=int)
    ps = np.array([float(r['probability']) for r in items], dtype=float)
    selected = ps >= THRESHOLD
    result = {
        'n': len(items), 'positives': int(ys.sum()), 'base_rate': float(ys.mean()),
        'auc': safe_auc(ys, ps),
        'average_precision': float(average_precision_score(ys, ps)) if ys.sum() else None,
        'brier': float(brier_score_loss(ys, ps)),
        'threshold': THRESHOLD, 'alerts': int(selected.sum()),
        'alert_wins': int(ys[selected].sum()),
        'precision_alerts': float(ys[selected].mean()) if selected.any() else None,
    }
    # Les tops sont locaux a chaque fenetre pour eviter la domination du dernier bloc.
    order = np.argsort(-ps, kind='stable')
    for k in (10, 20):
        actual = min(k, len(order))
        result['top{}_n'.format(k)] = actual
        result['top{}_wins'.format(k)] = int(ys[order[:actual]].sum())
        result['top{}_precision'.format(k)] = float(ys[order[:actual]].mean())
    return result


def fmt(value):
    return 'N/A' if value is None else '{:.1%}'.format(value)


def main():
    print('='*74)
    print('ETAPE 254 — SIMPLE6 WALK-FORWARD TRAIN638 (RESEARCH_ONLY)')
    print('='*74)
    if OUTPUT.exists() and (PRED.exists() or SUMMARY.exists()):
        raise SystemExit('Sorties deja presentes ; aucun ecrasement : {}'.format(OUTPUT))
    data = read_data(SOURCE)
    print('TRAIN638: {} matchs / {} positifs / {} journees'.format(
        len(data), sum(r['_y'] for r in data), len({r['_date'] for r in data})))
    all_pred = []
    details = []
    for name, start, end in FOLDS:
        train = [r for r in data if r['_date'] < start]
        valid = [r for r in data if start <= r['_date'] < end]
        if not train or not valid:
            raise SystemExit('Fenetre invalide : {}'.format(name))
        if max(r['_dt'] for r in train) >= min(r['_dt'] for r in valid):
            raise SystemExit('FUITE CHRONOLOGIQUE : {}'.format(name))
        x_train = matrix(train)
        x_valid = matrix(valid)
        y_train = np.array([r['_y'] for r in train], dtype=int)
        if len(set(y_train.tolist())) < 2:
            raise SystemExit('TRAIN a une seule classe : {}'.format(name))
        if np.any(np.all(np.isnan(x_train), axis=0)):
            raise SystemExit('Colonne de features entierement absente dans TRAIN : {}'.format(name))
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=REGULARIZATION_C, max_iter=2000, random_state=42)),
        ])
        model.fit(x_train, y_train)
        predictions = model.predict_proba(x_valid)[:, 1]
        this = []
        for row, p in zip(valid, predictions):
            this.append({
                'fold': name,
                'match_id': row['match_id'],
                'kickoff_utc': row['kickoff_utc'],
                'research_origin': row.get('research_origin', ''),
                'label': row['_y'],
                'probability': float(p),
                'alert_ge_030': int(p >= THRESHOLD),
                'train_n': len(train),
            })
        report = metrics(this)
        report['fold'] = name
        report['train_n'] = len(train)
        report['train_positives'] = int(y_train.sum())
        report['start'] = start
        report['end_exclusive'] = end
        details.append(report)
        all_pred.extend(this)
        print('{} | train={:3d} | test={:3d} ({:2d}+) | AUC={} | AP={} | alert >=.30: {}/{} = {} | top10 {}/10'.format(
            name, len(train), len(valid), report['positives'],
            fmt(report['auc']), fmt(report['average_precision']),
            report['alert_wins'], report['alerts'], fmt(report['precision_alerts']),
            report['top10_wins']))
    pred_ids = [r['match_id'] for r in all_pred]
    if len(pred_ids) != len(set(pred_ids)):
        raise SystemExit('DOUBLONS entre fenetres')
    global_report = metrics(all_pred)
    early = [r for r in all_pred if r['fold'] != '2026-08-20_24']
    early_report = metrics(early)
    base_by_origin = Counter(r['research_origin'] for r in all_pred)
    print('\n=== BILANS, NE PAS CONFONDRE AVEC TEST INDEPENDANT ===')
    for label, report in [('GLOBAL OOF', global_report), ('SANS 20-24 AOUT', early_report)]:
        print('{} | n={} | positifs={} | AUC={} | AP={} | seuil .30 : {}/{} = {}'.format(
            label, report['n'], report['positives'], fmt(report['auc']),
            fmt(report['average_precision']), report['alert_wins'], report['alerts'],
            fmt(report['precision_alerts'])))
    print('Origines dans les predictions:', dict(base_by_origin))
    print('AVERTISSEMENT : scores historiques non verifies independamment ; ancienne recherche sur aout 20-24.')
    print('Aucune optimisation de seuil. Aucun acces a VALIDATION154 ou TEST_FUTUR.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PRED.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'fold', 'match_id', 'kickoff_utc', 'research_origin', 'label',
            'probability', 'alert_ge_030', 'train_n'])
        writer.writeheader()
        writer.writerows(all_pred)
    payload = {
        'purpose': 'research_only_walkforward',
        'source': str(SOURCE), 'features': FEATURES,
        'logistic_regression_C': REGULARIZATION_C,
        'threshold_fixed': THRESHOLD,
        'folds': details, 'overall': global_report,
        'without_aug20_24': early_report,
        'research_warning': 'Not independent: old dates/model choices previously explored. Labels not externally verified.',
    }
    with SUMMARY.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('\nPREDICTIONS :', PRED)
    print('RAPPORT :', SUMMARY)
    print('ETAPE 254 TERMINEE | V3-C inchange')


if __name__ == '__main__':
    main()
