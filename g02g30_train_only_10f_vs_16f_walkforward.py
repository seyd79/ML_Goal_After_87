#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G30 | Comparaison 10F vs 16F strictement sur TRAIN (recherche).

Depuis la racine ML_Goal_After_87, environnement virtuel actif :
  python3 -B g02g30_train_only_10f_vs_16f_walkforward.py
  python3 -B g02g30_train_only_10f_vs_16f_walkforward.py --compare-train

Sans argument : garde SHA, provenance, partitions, schema; aucune timeline,
aucun label ouvert et aucun entrainement.
--compare-train : 845 timelines TRAIN uniquement (evenements <87) et 845 labels
TRAIN deja audites. Quatre plis expanding-window identiques a G02-G24,
StandardScaler ajuste exclusivement au passe de chaque pli, regression
logistique L2 C=0.1 liblinear, seuil FIXE 0.60. Comparaison appairee 10F/16F.
Aucune lecture des labels VALIDATION / TEST, aucune timeline VAL / TEST,
aucune API, aucune ecriture, aucun ajustement de seuil. Preuves as-of et
certification independante des labels NON acquises. Diagnostic exploratoire,
PAS de selection/gel d'un nouveau modele et PAS de deployment live.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import g02g24_train_only_walkforward_diagnostic as g24
import g02g29_candidate_16f_design_readonly as g29

EXPECTED_SCRIPTS = {
    Path('g02g24_train_only_walkforward_diagnostic.py'):
        'eb835dd7aafc9e43862516390bc81a0f36c763d1360601907f9e5bd45c5dace2',
    Path('g02g29_candidate_16f_design_readonly.py'):
        'f9af2b54e1147d98fe0fb1b98c8b906c8a9810bcd0f9f7afebfe3bd7bab94e91',
}
EXPECTED_G22_REPORT_SHA = '34c3dfce3ca63e88e80225d95b1f7117722096f0de7a432c914b2948bd5f4eaa'
THRESHOLD = 0.60


def require(condition, why):
    if not condition:
        raise SystemExit('G02-G30 STOP : ' + why)


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(131072), b''):
            h.update(part)
    return h.hexdigest()


def precheck():
    print('=== G02-G30/P1 | COMPARAISON 10F vs 16F | GARDE TRAIN ONLY ===')
    for path, digest in EXPECTED_SCRIPTS.items():
        require(path.is_file() and not path.is_symlink(),
                'script dependance absent/symlink : ' + str(path))
        require(checksum(path) == digest, 'script source modifie : ' + str(path))
        print('SHA SCRIPT OK :', path.name)
    report = g24.base.RESULT
    require(report.is_file() and not report.is_symlink(), 'rapport G02-G22 absent')
    require(checksum(report) == EXPECTED_G22_REPORT_SHA,
            'baseline G02-G22 non conforme')
    print('SHA BASELINE 10F GELÉE : OK')
    g27, g28, train_ids, dataset, registry = g29.load_preflight()
    require(len(train_ids) == 845 and len(registry) == 1500,
            'cohorte TRAIN ou registre incorrect')
    require(len(g24.gate.FEATURES) == 10 and len(g29.FEATURES) == 16,
            'schemas 10F/16F inattendus')
    require(g24.N_FOLDS == 4 and g24.INITIAL_DAY_RATIO == 0.5,
            'fenetres G02-G24 modifiees')
    print('PROTOCOLE FIXE : 4 plis G02-G24 | L2 C=0.1 liblinear | seuil=0.60')
    print('TEST PROTEGE : 302 matchs | VALIDATION : 269 labels non ouverts')
    print('G02-G30/P1 PRECHECK PASS | 0 LABEL | 0 TIMELINE | 0 FIT | 0 ECRITURE | 0 API')
    return g27, g28, train_ids, dataset, registry


def build_train_16f(g27, g28, train_ids, dataset, registry):
    """Ne lit que les JSON autorises TRAIN. Reconstruction strictement pre-87."""
    paths = {p.resolve() for p in g27.RAW_DIRS}
    require(all(p.is_dir() and not p.is_symlink() for p in g27.RAW_DIRS),
            'repertoires timelines manquants/symlink')
    values = {}
    for mid in train_ids:
        frozen = dataset[mid]
        reg = registry[mid]
        p = Path(reg['source_path'])
        require(p.is_file() and not p.is_symlink() and p.suffix == '.json'
                and p.stem == mid and p.resolve().parent in paths,
                'timeline hors garde TRAIN : ' + mid)
        require(checksum(p) == reg['source_sha256'], 'timeline SHA different : ' + mid)
        with p.open('r', encoding='utf-8') as f:
            wrapper = json.load(f)
        require(isinstance(wrapper, dict) and wrapper.get('match_id') == mid
                and str(wrapper.get('home_id')) == str(frozen['home_team_id'])
                and str(wrapper.get('away_id')) == str(frozen['away_team_id'])
                and wrapper.get('kickoff_utc') == frozen['kickoff_utc'],
                'metadonnees timeline differents : ' + mid)
        response = wrapper.get('timeline_response')
        require(isinstance(response, dict) and isinstance(response.get('data'), dict),
                'timeline_response.data absente : ' + mid)
        features, issues, _, _, _ = g27.reconstruct(response['data'].get('events'), wrapper, frozen)
        require(not issues, 'anomalies de reconstruction pre87 : ' + mid + '/' + str(dict(issues)))
        g28.validate_invariants(features, frozen)
        x = []
        for col in g29.FEATURES:
            raw = frozen[col] if col in g29.FROZEN_6 else features[col]
            require(raw is not None and str(raw).strip() != '',
                    'feature absente : ' + mid + '/' + col)
            try:
                number = float(raw)
            except (ValueError, TypeError, OverflowError):
                raise SystemExit('G02-G30 STOP : feature non numerique : ' + mid + '/' + col)
            require(math.isfinite(number) and number >= 0,
                    'feature non finie/negative : ' + mid + '/' + col)
            x.append(number)
        values[mid] = x
    require(len(values) == 845, 'reconstruction 16F non 845')
    print('RECONSTRUCTION 16F : 845 timelines TRAIN conformes | VAL 0 | TEST 0')
    return values


def fit_fold(lib, history, future, ys, columns):
    np, sklearn, LogisticRegression, StandardScaler, roc_auc_score, brier_score_loss, _, _ = lib
    X_hist = np.asarray([columns[r[2]] for r in history], dtype=np.float64)
    X_future = np.asarray([columns[r[2]] for r in future], dtype=np.float64)
    y_hist = np.asarray([ys[r[2]] for r in history], dtype=np.int64)
    y_future = np.asarray([ys[r[2]] for r in future], dtype=np.int64)
    require(set(np.unique(y_hist)) == {0, 1} and set(np.unique(y_future)) == {0, 1},
            'une classe absente dans un pli')
    scaling = StandardScaler()
    XX = scaling.fit_transform(X_hist)
    ZZ = scaling.transform(X_future)
    model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear',
                               max_iter=2000, class_weight=None, random_state=42)
    model.fit(XX, y_hist)
    require(model.n_iter_.max() < 2000, 'convergence non etablie')
    proba = model.predict_proba(ZZ)[:, 1]
    require(bool(np.all(np.isfinite(proba))), 'probabilites non finies')
    alerts = proba >= THRESHOLD
    return {
        'p': proba,
        'y': y_future,
        'auc': float(roc_auc_score(y_future, proba)),
        'brier': float(brier_score_loss(y_future, proba)),
        'alerts': int(alerts.sum()),
        'wins': int(np.logical_and(alerts, y_future == 1).sum()),
        'max': float(proba.max()),
    }


def compare(lib, data):
    g27, g28, train_ids, dataset, registry = data
    rec10, split, train_set = g24.read_train_only_features()
    require(train_set == set(train_ids), 'differents matchs entre 10F et 16F')
    folds, days = g24.construct_folds(rec10)
    require(len(folds) == 4, 'nombre de plis incorrect')
    xx10 = {r[2]: r[3] for r in rec10}
    xx16 = build_train_16f(g27, g28, train_ids, dataset, registry)
    ys = g24.read_train_only_labels(split, train_set)  # exclusivement 845 TRAIN
    np, sklearn, LogisticRegression, StandardScaler, roc_auc_score, brier_score_loss, _, _ = lib
    y_all, p10_all, p16_all, pref_all = [], [], [], []
    a10, w10, a16, w16 = 0, 0, 0, 0
    print('=== G02-G30/P2 | 10F vs 16F | 4 PLIS CHRONOLOGIQUES IDENTIQUES ===')
    for i, (hist, future, from_day, to_day) in enumerate(folds, 1):
        s10 = fit_fold(lib, hist, future, ys, xx10)
        s16 = fit_fold(lib, hist, future, ys, xx16)
        require(bool(np.array_equal(s10['y'], s16['y'])), 'labels futurs non alignes')
        y_eval = s10['y']
        y_hist = np.asarray([ys[r[2]] for r in hist], dtype=np.int64)
        base = float(y_hist.mean())
        prior = np.full(len(y_eval), base)
        print('FOLD %d | %s a %s | FIT %d | OOF %d | POS %d' %
              (i, from_day, to_day, len(hist), len(future), int(y_eval.sum())))
        print('  10F : AUC %.4f | Brier %.4f | pmax %.4f | alertes %d / gains %d' %
              (s10['auc'], s10['brier'], s10['max'], s10['alerts'], s10['wins']))
        print('  16F : AUC %.4f | Brier %.4f | pmax %.4f | alertes %d / gains %d' %
              (s16['auc'], s16['brier'], s16['max'], s16['alerts'], s16['wins']))
        print('  Delta Brier (16F - 10F) : %+.4f | REF HIST %.4f' %
              (s16['brier'] - s10['brier'], float(brier_score_loss(y_eval, prior))))
        y_all.extend(y_eval.tolist())
        p10_all.extend(s10['p'].tolist())
        p16_all.extend(s16['p'].tolist())
        pref_all.extend(prior.tolist())
        a10 += s10['alerts']; w10 += s10['wins']
        a16 += s16['alerts']; w16 += s16['wins']
    y, p10, p16, prior = map(np.asarray, (y_all, p10_all, p16_all, pref_all))
    require(len(y) == len(p10) == len(p16) == len(prior) == 472 and int(y.sum()) == 114,
            'OOF ne correspond pas a G02-G24')
    b10 = float(brier_score_loss(y, p10)); b16 = float(brier_score_loss(y, p16))
    b_ref = float(brier_score_loss(y, prior))
    auc10 = float(roc_auc_score(y, p10)); auc16 = float(roc_auc_score(y, p16))
    # Garantir une reproduction numerique effective de l'experience historique.
    require(abs(b10 - 0.1867) < 0.00006 and abs(b_ref - 0.1835) < 0.00006
            and abs(auc10 - 0.5270) < 0.00006 and a10 == w10 == 1,
            '10F ne reproduit pas G02-G24 : STOP comparaison')
    print('=== SYNTHESE OOF TRAIN — MÊMES 472 MATCHS ===')
    print('REFERENCE passée : Brier %.4f' % b_ref)
    print('10F | AUC %.4f | Brier %.4f | alertes @0.60 %d | victoires %d' %
          (auc10, b10, a10, w10))
    print('16F | AUC %.4f | Brier %.4f | alertes @0.60 %d | victoires %d' %
          (auc16, b16, a16, w16))
    print('DELTA 16F-10F | AUC %+.4f | Brier %+.4f (Brier plus faible = meilleur)' %
          (auc16 - auc10, b16 - b10))
    print('DELTA 16F-reference | Brier %+.4f' % (b16 - b_ref))
    print('IMPORTANT : evaluation exploratoire sur TRAIN apres avoir vu la premiere VALIDATION.')
    print('IMPORTANT : AUC OOF globale melange des modeles ajustes sur des histories differentes.')
    print('IMPORTANT : ne pas traiter 1 ou quelques alertes comme preuve de 60%% de precision.')
    print('IMPORTANT : labels non certifies independamment, as-of NON certifie.')
    print('VALIDATION/TEST : 0 valeur outcome interpretee, 0 timeline ouverte')
    print('0 API | 0 ECRITURE | 0 CHANGEMENT DE SEUIL | 0 ARTEFACT MODIFIE')
    print('G02-G30/P2 TRAIN_ONLY_PAIRED_COMPARISON COMPLETE | AUCUN MODELE 16F GELE')


def synthetic_unit_tests():
    """Teste le calcul sur donnees factices; aucune source projet n'est lue."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.exceptions import ConvergenceWarning
    import sklearn
    lib = (np, sklearn, LogisticRegression, StandardScaler, roc_auc_score,
           brier_score_loss, log_loss, ConvergenceWarning)
    # Une valeur informative mais non parfaite avec 2 features; 2 classes presentes.
    class Rec(tuple):
        pass
    hist = [(None, None, 'm%d' % i, None) for i in range(50)]
    future = [(None, None, 'm%d' % i, None) for i in range(50, 70)]
    ys = {'m%d' % i: int(i % 4 == 0) for i in range(70)}
    xx = {'m%d' % i: [float(i % 7), float((i * 3) % 11)] for i in range(70)}
    result = fit_fold(lib, hist, future, ys, xx)
    assert len(result['p']) == 20 and 0 <= result['auc'] <= 1
    assert 0 <= result['brier'] <= 1 and 0 <= result['wins'] <= result['alerts']
    print('TEST SYNTHÉTIQUE CALCUL : PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-train', action='store_true',
                        help='4 fits 10F + 4 fits 16F uniquement a l interieur de TRAIN')
    args = parser.parse_args()
    guarded = precheck()
    if not args.compare_train:
        print('Puis (apres PASS) : python3 -B g02g30_train_only_10f_vs_16f_walkforward.py --compare-train')
        return
    lib = g24.base.check_libraries()
    compare(lib, guarded)


if __name__ == '__main__':
    main()
