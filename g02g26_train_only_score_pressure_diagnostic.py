#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G26/P1 : exploratoire descriptif SCORE x PRESSION sur TRAIN (V4.2).

Depuis la racine ML_Goal_After_87 (Python 3.9+, venv) :
    python3 -B g02g26_train_only_score_pressure_diagnostic.py
    python3 -B g02g26_train_only_score_pressure_diagnostic.py --diagnose-train

Sans argument : verifications SHA des scripts et sources gelee, 0 label, 0 fit.
Avec --diagnose-train : lit uniquement les 845 labels TRAIN autorises, compare
les frequences de but >87 dans des groupes definis A L'AVANCE :
  - score a 87' : 0-0 ; autres nuls ; ecart 1 ; ecart >=2
  - activite shots_80_87 : 0 ; 1 ; >=2
  - tableau croise de ces categories avec taille et taux par bloc chrono.

Ce script ne selectionne PAS de segment, ne modifie PAS de feature/seuil,
n'entraine PAS de modele, ne relit PAS VALIDATION/TEST, n'ouvre pas de timeline,
n'appelle aucune API et ne cree aucun fichier. Analyse exploratoire seulement :
aucun de ces groupes ne peut etre presente comme une regle fiable ou validee.
Les labels ne sont pas certifies independamment; as-of non certifie.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import g02g25_train_only_signal_stability as stability
import g02g24_train_only_walkforward_diagnostic as walk
import g02g21_research_go_nogo_readonly as gate

STABILITY_FILE = Path('g02g25_train_only_signal_stability.py')
EXPECTED_STABILITY_SHA = '38e6fcfa6c3c8c44c19d18977d93a2de27c39eed3523167d1e2ef98d7120add3'

SCORE_GROUPS = ('0-0', 'Nul non nul', 'Ecart 1', 'Ecart >=2')
SHOT_GROUPS = ('0 tir', '1 tir', '2+ tirs')


def require(condition, detail):
    if not condition:
        raise SystemExit('G02-G26 STOP : ' + detail)


def precheck():
    print('=== G02-G26/P1 | SCORE x PRESSION | TRAIN ONLY | PRECHECK ===')
    require(STABILITY_FILE.is_file() and not STABILITY_FILE.is_symlink(),
            'script G02-G25 absent ou lien symbolique')
    require(walk.base.sha_file(STABILITY_FILE) == EXPECTED_STABILITY_SHA,
            'script G02-G25 modifie : SHA different')
    require(Path(stability.__file__).resolve() == (Path.cwd() / STABILITY_FILE).resolve(),
            'module G02-G25 importe depuis un autre dossier')
    print('SHA SCRIPT G02-G25 : OK')
    stability.guard()
    require(tuple(gate.FEATURES[:3]) == ('score_87_home', 'score_87_away', 'shots_80_87'),
            'ordre des features different du contrat attendu')
    print('GROUPES FIGES POUR DIAGNOSTIC : score (4) x tirs 80-87 (3)')
    print('G02-G26/P1 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 FIT')


def parse_nonneg_count(x, title):
    require(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 and float(x).is_integer(),
            'valeur non-entier/negative : ' + title + ' = ' + repr(x))
    return int(x)


def group_for_features(xx):
    home = parse_nonneg_count(xx[0], 'score_87_home')
    away = parse_nonneg_count(xx[1], 'score_87_away')
    shots = parse_nonneg_count(xx[2], 'shots_80_87')
    if home == away == 0:
        score = SCORE_GROUPS[0]
    elif home == away:
        score = SCORE_GROUPS[1]
    elif abs(home - away) == 1:
        score = SCORE_GROUPS[2]
    else:
        score = SCORE_GROUPS[3]
    activity = SHOT_GROUPS[min(shots, 2)]
    return score, activity


def wilson_95(n, positive):
    if n == 0:
        return None
    z = 1.959963984540054
    p = positive / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt((p*(1-p)+z*z/(4*n))/n) / d
    return 100*(c-h), 100*(c+h)


def format_cell(n, positives, ci=True):
    if n == 0:
        return 'n=0 | NA'
    p = 100 * positives / n
    if ci:
        lo, hi = wilson_95(n, positives)
        return f'n={n:3d} | +={positives:3d} | {p:5.1f}% | IC95 [{lo:4.1f},{hi:4.1f}]' + (' | PETIT N' if n < 30 else '')
    return f'n={n:3d} | +={positives:3d} | {p:5.1f}%'


def counts(rows, labels):
    score = defaultdict(lambda: [0, 0])
    activity = defaultdict(lambda: [0, 0])
    both = defaultdict(lambda: [0, 0])
    for r in rows:
        s, a = group_for_features(r[3])
        mid = r[2]
        require(mid in labels, 'label TRAIN absent : ' + mid)
        y = labels[mid]
        require(y in (0, 1), 'label TRAIN non binaire : ' + mid)
        for target, key in ((score, s), (activity, a), (both, (s, a))):
            target[key][0] += 1
            target[key][1] += y
    return score, activity, both


def diagnose():
    rows, split, train_ids = walk.read_train_only_features()
    folds, days = walk.construct_folds(rows)
    labels = walk.read_train_only_labels(split, train_ids)
    history = folds[0][0]
    future = [r for _, fold_rows, _, _ in folds for r in fold_rows]
    require(len(history) + len(future) == len(rows) == 845,
            'nombre de matchs TRAIN incoherent')
    require({r[2] for r in rows} == {r[2] for r in history + future},
            'matchs TRAIN incoherents avec la partition chrono')
    require(max(r[1] for r in history) < min(r[1] for r in future),
            'aucune separation chronologique')
    require(len(labels) == 845 and sum(labels.values()) == 211,
            'labels TRAIN en desaccord avec rapport gele')

    all_score, all_activity, all_joint = counts(rows, labels)
    hist_score, hist_activity, hist_joint = counts(history, labels)
    fut_score, fut_activity, fut_joint = counts(future, labels)
    require(sum(v[0] for v in all_joint.values()) == 845, 'partition score/activite incomplete')
    require(sum(v[1] for v in all_joint.values()) == 211, 'compte positifs incorrect')

    print('=== G02-G26/P2 | GROUPES PREDEFINIS, TRAIN UNIQUEMENT | SANS FIT ===')
    print('TRAIN : 845 | positifs : 211 | base : 24.97%')
    print('HISTORIQUE INITIAL : %d | FOLDS CHRONO : %d | jours TRAIN : %d'
          % (len(history), len(future), len(days)))
    print('=== CONTEXTE SCORE A 87 MINUTES ===')
    for s in SCORE_GROUPS:
        n, p = all_score[s]
        print('%-12s | %s' % (s, format_cell(n, p)))
    print('=== TIRS 80-87 MINUTES ===')
    for a in SHOT_GROUPS:
        n, p = all_activity[a]
        print('%-12s | %s' % (a, format_cell(n, p)))
    print('=== SCORE x TIRS | TOUT TRAIN (groupes descriptifs) ===')
    for s in SCORE_GROUPS:
        for a in SHOT_GROUPS:
            n, p = all_joint[(s, a)]
            print('%-12s | %-7s | %s' % (s, a, format_cell(n, p)))
    print('=== CONTROLE DES GROUPES ENTRE PERIODES TRAIN ===')
    print('SCORE | TIRS | HISTORIQUE | FOLDS SUIVANTS')
    for s in SCORE_GROUPS:
        for a in SHOT_GROUPS:
            hn, hp = hist_joint[(s, a)]
            fn, fp = fut_joint[(s, a)]
            print('%-12s | %-7s | %-24s | %s' %
                  (s, a, format_cell(hn, hp, ci=False), format_cell(fn, fp, ci=False)))
    print('ATTENTION : groupements descriptifs non pre-enregistres avant le resultat de VALIDATION.')
    print('ATTENTION : ne PAS selectionner un groupe/seuil puis revendiquer VALIDATION independante.')
    print('ATTENTION : IC Wilson descriptifs, cellules non independantes temporellement; pas de preuve live.')
    print('LABELS TRAIN seulement | VALIDATION/TEST OUTCOME 0 interprete | 0 timeline')
    print('0 API | 0 ECRITURE | 0 FIT | as-of NON certifie | certification independante 0')
    print('G02-G26/P2 TRAIN_ONLY_SCORE_PRESSURE_DIAGNOSTIC COMPLETE | 0 NOUVEAU MODELE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose-train', action='store_true',
                        help='analyse descriptive de categories sur 845 matchs TRAIN uniquement')
    args = parser.parse_args()
    precheck()
    if args.diagnose_train:
        diagnose()
    else:
        print('Apres PASS : python3 -B g02g26_train_only_score_pressure_diagnostic.py --diagnose-train')


if __name__ == '__main__':
    main()
