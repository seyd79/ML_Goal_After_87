#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G25/P1 : diagnostic exploratoire de stabilite des features sur TRAIN seulement.

Depuis la racine du projet ML_Goal_After_87 (Python >= 3.9):
  python3 -B g02g25_train_only_signal_stability.py
  python3 -B g02g25_train_only_signal_stability.py --diagnose-train

Sans argument, controles SHA/read-only, 0 label, 0 fit.
Avec --diagnose-train, compare 2 sous-ensembles chrono TRAIN (premier bloc
historique du walk-forward G02-G24 vs 4 blocs suivants). Seules les 845
etiquettes TRAIN sont lues; VALIDATION et TEST sont ignores. Aucun entrainement,
aucune modification, aucun seuil ou hyperparametre selectionne, aucune API.

Analyse descriptive: taux positifs, deplacement de moyenne standardise par la
variance du premier bloc, AUC univariees signees par feature (toutes 10,
sans selection). Ne demontre ni l'as-of, ni une certification independante.
"""
import argparse
from pathlib import Path
import math
import statistics

import g02g24_train_only_walkforward_diagnostic as wf
import g02g21_research_go_nogo_readonly as gate

WF_SCRIPT = Path('g02g24_train_only_walkforward_diagnostic.py')
EXPECTED_WF_SHA = 'eb835dd7aafc9e43862516390bc81a0f36c763d1360601907f9e5bd45c5dace2'


def require(condition, reason):
    if not condition:
        raise SystemExit('G02-G25 STOP : ' + reason)


def guard():
    print('=== G02-G25/P1 | STABILITE DES FEATURES | TRAIN SEULEMENT ===')
    require(WF_SCRIPT.is_file() and not WF_SCRIPT.is_symlink(),
            'script G02-G24 absent ou lien symbolique')
    require(wf.base.sha_file(WF_SCRIPT) == EXPECTED_WF_SHA,
            'SHA G02-G24 non conforme; arret avant diagnostic')
    require(Path(wf.__file__).resolve() == (Path.cwd() / WF_SCRIPT).resolve(),
            'module G02-G24 importe hors projet')
    require(Path(gate.__file__).resolve() ==
            (Path.cwd() / 'g02g21_research_go_nogo_readonly.py').resolve(),
            'module G02-G21 importe hors projet')
    print('SHA SCRIPT G02-G24 : OK')
    wf.verify_precheck()
    print('G02-G25/P1 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 FIT')


def auc_signed(rows, labels, ix, roc_auc_score):
    outcomes = [labels[r[2]] for r in rows]
    if len(set(outcomes)) < 2:
        return float('nan')
    return float(roc_auc_score(outcomes, [r[3][ix] for r in rows]))


def summarize_block(rows, labels, tag):
    positive = sum(labels[r[2]] for r in rows)
    count = len(rows)
    require(count > 0, 'sous-ensemble vide: ' + tag)
    print('%-14s | %4d matchs | %3d positifs | taux %.2f%% | %s a %s'
          % (tag, count, positive, positive * 100.0 / count,
             min(r[0] for r in rows), max(r[0] for r in rows)))


def diagnostic():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as ex:
        raise SystemExit('G02-G25 STOP : sklearn absent : ' + str(ex))
    rows, split, train_ids = wf.read_train_only_features()
    folds, days = wf.construct_folds(rows)
    labels = wf.read_train_only_labels(split, train_ids)

    first_train = folds[0][0]
    subsequent = [r for _, evaluations, _, _ in folds for r in evaluations]
    require(len(first_train) + len(subsequent) == 845,
            'la premiere histoire et les 4 fenetres doivent couvrir 845 TRAIN')
    require(len({r[2] for r in first_train + subsequent}) == 845,
            'doublon ou match TRAIN absent')
    require(max(r[1] for r in first_train) < min(r[1] for r in subsequent),
            'separation chrono invalide')
    print('=== G02-G25/P2 | STABILITE INTERNE | SANS FIT | TRAIN LABELS UNIQUEMENT ===')
    print('JOURS UTC TRAIN :', len(days), '| 1er bloc histoire =', len(first_train),
          '| 4 blocs suivants =', len(subsequent))
    summarize_block(first_train, labels, 'HISTORIQUE 1/2')
    for i, (_, future, _, _) in enumerate(folds, 1):
        summarize_block(future, labels, 'FOLD ' + str(i))
    summarize_block(subsequent, labels, 'FOLDS 1-4')

    print('=== 10 FEATURES | AUC UNIVARIEE BRUTE / DEPLACEMENT DES MOYENNES ===')
    print('AUC : > 0.5 association positive, < 0.5 association inverse; NON causal, NON independant')
    print('SHIFT : (moyenne FOLDS - moyenne HISTORIQUE) / ecart-type HISTORIQUE')
    print('FEATURE | AUC HIST | AUC FUTUR | SHIFT SD | MEAN HIST | MEAN FUTUR')
    for ix, name in enumerate(gate.FEATURES):
        past = [r[3][ix] for r in first_train]
        later = [r[3][ix] for r in subsequent]
        a = auc_signed(first_train, labels, ix, roc_auc_score)
        b = auc_signed(subsequent, labels, ix, roc_auc_score)
        pm = statistics.fmean(past)
        fm = statistics.fmean(later)
        psd = statistics.pstdev(past)
        if psd == 0:
            shift = 'N/A (sd zero)'
        else:
            shift = '%+.3f' % ((fm - pm) / psd)
        print('%s | %.4f | %.4f | %s | %.3f | %.3f' %
              (name, a, b, shift, pm, fm))

    print('IMPORTANT : ces correlations sont exploratoires et examinees apres un echec sur VALIDATION.')
    print('IMPORTANT : ne pas choisir ici une feature / un seuil puis revendiquer VALIDATION independante.')
    print('IMPORTANT : labels TRAIN non independamment certifies; as-of NON certifie.')
    print('VALIDATION/TEST OUTCOME : 0 interprete | 0 TIMELINE | 0 API | 0 ECRITURE | 0 FIT')
    print('G02-G25/P2 TRAIN_ONLY_STABILITY_DIAGNOSTIC COMPLETE | 0 NOUVEAU MODELE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose-train', action='store_true',
                        help='lit uniquement les labels TRAIN et effectue diagnostic descriptif sans entrainement')
    args = parser.parse_args()
    guard()
    if args.diagnose_train:
        diagnostic()
    else:
        print('Apres PASS : python3 -B g02g25_train_only_signal_stability.py --diagnose-train')


if __name__ == '__main__':
    main()
