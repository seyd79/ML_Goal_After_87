#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETAPE 255 - Comparaison exploratoire fixe SIMPLE6 vs B29 en walk-forward.

Lecture locale de TRAIN638 uniquement. Validation et TEST_FUTUR non consultes.
Aucun appel reseau; aucun changement de fichiers historiques ou V3-C.
Ne pas utiliser ces resultats comme confirmation independante des 60%.
"""
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = Path('outputs/v4_research/V41_TRAIN638_B29_RESEARCH_V1.csv')
OUT = Path('outputs/v4_research/v41_b29_walkforward_v1')
PRED = OUT / 'V41_B29_VS_SIMPLE6_OOF_V1.csv'
REPORT = OUT / 'V41_B29_VS_SIMPLE6_SUMMARY_V1.json'

SIMPLE6 = [
    'score_total_87', 'score_diff_abs_87', 'is_draw_87',
    'shots_total_last5', 'sot_total_last5', 'corners_total_last5'
]
B29 = [
    'score_87_home', 'score_87_away', 'goal_diff_87', 'is_draw_87',
    'home_shots_87', 'away_shots_87',
    'home_shots_on_target_87', 'away_shots_on_target_87',
    'home_shots_off_target_87', 'away_shots_off_target_87',
    'home_shots_blocked_87', 'away_shots_blocked_87',
    'home_corners_87', 'away_corners_87',
    'home_fouls_87', 'away_fouls_87',
    'home_yellow_cards_87', 'away_yellow_cards_87',
    'home_red_cards_87', 'away_red_cards_87',
    'home_offsides_87', 'away_offsides_87',
    'goal_diff_signed_87',
    'shots_home_last5', 'shots_away_last5',
    'sot_home_last5', 'sot_away_last5',
    'corners_home_last5', 'corners_away_last5',
]
FEATURES = sorted(set(SIMPLE6) | set(B29))
FOLDS = [
    ('2026-05', '2026-05-01', '2026-06-01'),
    ('2026-06', '2026-06-01', '2026-07-01'),
    ('2026-07', '2026-07-01', '2026-08-01'),
    ('2026-08-01_19', '2026-08-01', '2026-08-20'),
    ('2026-08-20_24', '2026-08-20', '2026-08-25'),
]
THRESHOLD = .30
MODELS = ('SIMPLE6_LR', 'B29_LR', 'B29_RF')


def load():
    with SOURCE.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        absent = sorted((set(FEATURES) | {'match_id', 'kickoff_utc', 'goal_after_87'}) - set(reader.fieldnames or []))
        if absent:
            raise SystemExit('Colonnes absentes: ' + str(absent))
        rows = list(reader)
    if len(rows) != 638:
        raise SystemExit('TRAIN638: 638 lignes attendues, recu {}'.format(len(rows)))
    if len(set(r['match_id'] for r in rows)) != len(rows):
        raise SystemExit('IDs dupliques')
    for r in rows:
        try:
            dt = datetime.fromisoformat(r['kickoff_utc'].replace('Z', '+00:00'))
            if dt.tzinfo is None:
                raise ValueError('date sans fuseau')
            r['_dt'] = dt.astimezone(timezone.utc)
            r['_date'] = r['_dt'].date().isoformat()
            y = float(r['goal_after_87'])
            if y not in (0., 1.):
                raise ValueError('label non binaire')
            r['_y'] = int(y)
            for name in FEATURES:
                raw = str(r.get(name, '')).strip()
                if raw.lower() in ('', 'na', 'nan', 'none', 'null'):
                    r['_' + name] = np.nan
                else:
                    n = float(raw)
                    if not math.isfinite(n) or n < 0 and name not in ('goal_diff_signed_87', 'goal_diff_87'):
                        raise ValueError('feature non finie ou negative: ' + name)
                    r['_' + name] = n
        except Exception as exc:
            raise SystemExit('Ligne invalide {}: {}'.format(r.get('match_id'), exc))
    rows.sort(key=lambda r: (r['_dt'], r['match_id']))
    if sum(r['_y'] for r in rows) != 162:
        raise SystemExit('Positifs inattendus : ' + str(sum(r['_y'] for r in rows)))
    return rows


def matrix(rows, names):
    return np.asarray([[r['_' + col] for col in names] for r in rows], dtype=float)


def make_model(name):
    if name == 'B29_RF':
        classifier = RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=15,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        return Pipeline([('imputer', SimpleImputer(strategy='median')), ('rf', classifier)])
    classifier = LogisticRegression(C=.1, max_iter=2000, random_state=42)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()), ('lr', classifier),
    ])


def metric(items):
    if not items:
        return None
    yy = np.asarray([a['label'] for a in items], dtype=int)
    pp = np.asarray([a['probability'] for a in items], dtype=float)
    mask = pp >= THRESHOLD
    order = np.argsort(-pp, kind='stable')
    wins = int(yy[mask].sum())
    return {
        'n': len(yy), 'positives': int(yy.sum()), 'base_rate': float(yy.mean()),
        'auc': float(roc_auc_score(yy, pp)) if len(set(yy.tolist())) == 2 else None,
        'ap': float(average_precision_score(yy, pp)) if yy.sum() else None,
        'alerts': int(mask.sum()), 'wins': wins,
        'precision': float(wins / int(mask.sum())) if mask.any() else None,
        'top10_n': min(10, len(yy)),
        'top10_wins': int(yy[order[:10]].sum()),
    }


def pct(x):
    return 'N/A' if x is None else '{:.1%}'.format(x)


def main():
    print('=' * 74)
    print('ETAPE 255 — COMPARAISON WALK-FORWARD SIMPLE6 vs B29 (RESEARCH_ONLY)')
    print('=' * 74)
    if PRED.exists() or REPORT.exists():
        raise SystemExit('Sorties existantes: aucun ecrasement. ' + str(OUT))
    rows = load()
    print('TRAIN638: {} matchs, {} positifs, {} journees'.format(
        len(rows), sum(r['_y'] for r in rows), len({r['_date'] for r in rows})
    ))
    print('B29: {} features fixes | seuil: {:.2f}'.format(len(B29), THRESHOLD))
    all_pred = []
    details = []
    for fold, start, end in FOLDS:
        training = [r for r in rows if r['_date'] < start]
        testing = [r for r in rows if start <= r['_date'] < end]
        if not training or not testing:
            raise SystemExit('Fenetre vide: ' + fold)
        if max(r['_dt'] for r in training) >= min(r['_dt'] for r in testing):
            raise SystemExit('FUITE CHRONO: ' + fold)
        ytr = np.asarray([r['_y'] for r in training], dtype=int)
        if len(set(ytr.tolist())) != 2:
            raise SystemExit('Une seule classe sur: ' + fold)
        for model_name in MODELS:
            names = SIMPLE6 if model_name == 'SIMPLE6_LR' else B29
            xtr = matrix(training, names)
            xte = matrix(testing, names)
            # Selection basee uniquement sur les donnees d'entrainement :
            # les colonnes entierement vides dans TRAIN sont ignorees dans ce fold.
            active = ~np.all(np.isnan(xtr), axis=0)
            if not active.any():
                raise SystemExit('Toutes les features vides: {} {}'.format(fold, model_name))
            xtr = xtr[:, active]
            xte = xte[:, active]
            fitted = make_model(model_name)
            fitted.fit(xtr, ytr)
            prob = fitted.predict_proba(xte)[:, 1]
            current = []
            for r, p in zip(testing, prob):
                item = {
                    'model': model_name, 'fold': fold, 'match_id': r['match_id'],
                    'kickoff_utc': r['kickoff_utc'], 'research_origin': r.get('research_origin', ''),
                    'label': r['_y'], 'probability': float(p),
                    'alert_ge_030': int(p >= THRESHOLD), 'train_n': len(training),
                }
                all_pred.append(item)
                current.append(item)
            m = metric(current)
            m.update({'fold': fold, 'model': model_name, 'train_n': len(training), 'features_active': int(active.sum())})
            details.append(m)
            print('{:<14} {:<15} train={:3d} test={:3d} | AUC={} AP={} | alert={}/{} ({}) | top10={}/10'.format(
                fold, model_name, len(training), len(testing), pct(m['auc']), pct(m['ap']),
                m['wins'], m['alerts'], pct(m['precision']), m['top10_wins'],
            ))
    print('\n=== BILAN OOF (exploratoire, non independant) ===')
    summary = {}
    for model_name in MODELS:
        mine = [r for r in all_pred if r['model'] == model_name]
        total = metric(mine)
        early = metric([r for r in mine if r['fold'] != '2026-08-20_24'])
        summary[model_name] = {'global': total, 'excluding_aug20_24': early}
        print('{:<15} GLOBAL: {}/{}={} | AUC={} AP={} | sans aout20-24: {}/{}={} AUC={}'.format(
            model_name, total['wins'], total['alerts'], pct(total['precision']),
            pct(total['auc']), pct(total['ap']), early['wins'], early['alerts'],
            pct(early['precision']), pct(early['auc'])
        ))
    # Reproduction exacte attendue de l'etape 254 sur les memes dates/folds.
    ref = summary['SIMPLE6_LR']['global']
    ref_early = summary['SIMPLE6_LR']['excluding_aug20_24']
    if (ref['n'], ref['positives'], ref['alerts'], ref['wins']) != (464, 117, 76, 26):
        raise SystemExit('BLOCAGE: SIMPLE6 ne reproduit pas 254: {}'.format(ref))
    if (ref_early['n'], ref_early['positives'], ref_early['alerts'], ref_early['wins']) != (194, 47, 42, 13):
        raise SystemExit('BLOCAGE: SIMPLE6 sans bloc aout differe de 254: {}'.format(ref_early))
    OUT.mkdir(parents=True, exist_ok=True)
    with PRED.open('x', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'model', 'fold', 'match_id', 'kickoff_utc', 'research_origin',
            'label', 'probability', 'alert_ge_030', 'train_n'
        ])
        writer.writeheader()
        writer.writerows(all_pred)
    with REPORT.open('x', encoding='utf-8') as fh:
        json.dump({
            'purpose': 'research_only_fixed_comparison',
            'models': list(MODELS), 'features_simple6': SIMPLE6, 'features_b29': B29,
            'folds': FOLDS, 'threshold': THRESHOLD, 'fold_metrics': details,
            'summary': summary,
            'warnings': [
                'Non independent: research folds previously explored.',
                'Historical labels not independently verified.',
                'No use of VALIDATION154 or TEST_FUTUR.',
                'Feature availability in actual LIVE snapshots not yet certified.',
            ]
        }, fh, indent=2, ensure_ascii=False)
    print('\nPREDICTIONS:', PRED)
    print('RAPPORT:', REPORT)
    print('Aucun appel API | VALIDATION154 et TEST_FUTUR non consultes | V3-C inchange')
    print('ETAPE 255 TERMINEE')


if __name__ == '__main__':
    main()
