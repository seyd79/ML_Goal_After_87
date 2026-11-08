#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G24 — diagnostic chronologique sur TRAIN uniquement (recherche retrospective).

Depuis la racine ML_Goal_After_87, venv actif :
    python3 -B g02g24_train_only_walkforward_diagnostic.py
    python3 -B g02g24_train_only_walkforward_diagnostic.py --diagnose-train

Sans argument : PRECHECK. Aucune lecture de valeur de label / aucun fit.
--diagnose-train : lit et parse exclusivement les 845 labels TRAIN deja audites,
forme 4 plis a origine progressive SANS JOUR UTC PARTAGE (premiere moitie
chronologique des jours reservee aux premiers fits), entraine uniquement
la configuration preenregistree L2 C=0.1 / StandardScaler fit sur le passe,
compare au benchmark de prevalence du passe. Aucune consultation de labels
VALIDATION ou TEST, aucune timeline, aucun fichier ecrit, aucun appel API.

DIAGNOSTIC UNIQUEMENT : les performances OOF TRAIN ne certifient ni
la validite externe des labels, ni l'as-of historique, ni le live.
Ne pas optimiser un seuil/choisir de features sur VALIDATION. 0.60 reste fixe.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import g02g22_baseline_research_guarded as base
import g02g21_research_go_nogo_readonly as gate

BASELINE_SCRIPT = Path('g02g22_baseline_research_guarded.py')
EXPECTED_BASELINE_SCRIPT_SHA = '7fdc177d305a467e27feef64d1c98c5803f8d31222b4b108c9f799720dd985a5'
EXPECTED_REPORT_SHA = '34c3dfce3ca63e88e80225d95b1f7117722096f0de7a432c914b2948bd5f4eaa'
N_FOLDS = 4
INITIAL_DAY_RATIO = 0.5


def require(ok, why):
    if not ok:
        raise SystemExit('G02-G24 STOP : ' + why)


def parse_utc(s):
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except (TypeError, ValueError, AttributeError):
        raise SystemExit('G02-G24 STOP : date invalide : ' + repr(s))
    require(d.tzinfo is not None, 'kickoff sans fuseau UTC : ' + repr(s))
    return d.astimezone(timezone.utc)


def verify_precheck():
    print('=== G02-G24/P1 | DIAGNOSTIC TRAIN CHRONOLOGIQUE | PROTECTIONS ===')
    require(BASELINE_SCRIPT.is_file() and not BASELINE_SCRIPT.is_symlink(),
            'script G02-G22 manquant')
    require(base.sha_file(BASELINE_SCRIPT) == EXPECTED_BASELINE_SCRIPT_SHA,
            'script G02-G22 different de la version de reference')
    print('SHA SCRIPT G02-G22 : OK')
    base.precheck()  # SOURCE SHA G02-G15..G02-G20 + env + aucun label ouvert
    require(base.RESULT.is_file() and base.DIGEST.is_file() and
            not base.RESULT.is_symlink() and not base.DIGEST.is_symlink(),
            'rapport baseline gele absent')
    require(base.sha_file(base.RESULT) == EXPECTED_REPORT_SHA and
            base.DIGEST.read_text(encoding='ascii').strip() == EXPECTED_REPORT_SHA,
            'rapport G02-G22 non conforme')
    base.do_verify()
    print('GARDE TEST : 302 matchs hors allowlist, aucun label TEST accessible au diagnostic')
    print('POLITIQUE : 4 plis chronologiques sur TRAIN ; premier fit apres 50% des jours TRAIN')
    print('MODELE FIXE : L2 C=0.1 liblinear | StandardScaler passe seulement | seuil 0.60')
    print('G02-G24/P1 PRECHECK PASS | 0 API | 0 LABEL | 0 TIMELINE | 0 ECRITURE | 0 FIT')


def read_train_only_features():
    split = gate.read_rows(gate.SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    allow = gate.read_rows(gate.ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    data = gate.read_rows(gate.DATA, ('match_id', 'kickoff_utc') + gate.FEATURES)
    require(len(split) == 1500 and len(data) == 1500 and len(allow) == 1114,
            'tailles sources changees')
    require(set(split) == set(data), 'IDs dataset differents du split')
    require(set(allow) <= set(split), 'allowlist inconnue')
    train_ids = {mid for mid, row in allow.items() if row['partition'] == 'TRAIN'}
    val_ids = {mid for mid, row in allow.items() if row['partition'] == 'VALIDATION'}
    test_ids = {mid for mid, row in split.items() if row['partition'] == 'TEST'}
    require(len(train_ids) == 845 and len(val_ids) == 269 and len(test_ids) == 302 and
            not test_ids.intersection(allow) and not train_ids.intersection(val_ids),
            'comptes / garde TEST invalides')
    require(all(split[mid]['partition'] == allow[mid]['partition'] for mid in allow),
            'partition divergente')
    observations = []
    for mid in train_ids:
        a = allow[mid]
        require(a['kickoff_utc'] == split[mid]['kickoff_utc'] == data[mid]['kickoff_utc'],
                'kickoff divergent : ' + mid)
        dt = parse_utc(a['kickoff_utc'])
        xx = []
        for feat in gate.FEATURES:
            raw = data[mid].get(feat, '')
            try:
                val = float(raw)
            except (ValueError, TypeError):
                raise SystemExit('G02-G24 STOP : valeur feature non numerique : ' + mid + '/' + feat)
            require(math.isfinite(val) and val >= 0,
                    'feature invalide ' + mid + '/' + feat)
            xx.append(val)
        observations.append((dt.date(), dt, mid, xx))
    observations.sort(key=lambda r: (r[1], r[2]))
    require(len(observations) == 845 and len({r[2] for r in observations}) == 845,
            'TRAIN incomplet / doublons')
    return observations, split, train_ids


def read_train_only_labels(split, train_ids):
    """Pas de decoding des colonnes outcome sur VALIDATION ou TEST."""
    path = base.LABEL_CSV
    require(path.is_file() and not path.is_symlink(), 'CSV labels absent / symlink')
    base.label_guard.read_header_only(path)
    all_seen = set()
    ys = {}
    certified = 0
    with path.open('rb') as f:
        header = f.readline(65536)
        require(header.endswith(b'\n'), 'header labels non termine')
        for line in f:
            require(line.endswith(b'\n'), 'ligne labels sans terminateur')
            # Seul l'ID initial des lignes non TRAIN est interprete.
            raw_mid = line.partition(b',')[0].strip(b'"')
            try:
                mid = raw_mid.decode('ascii')
            except UnicodeDecodeError:
                raise SystemExit('G02-G24 STOP : ID non ascii')
            require(mid in split and mid not in all_seen, 'ID invalide / duplique CSV labels')
            all_seen.add(mid)
            if mid not in train_ids:
                continue
            try:
                row = next(csv.reader([line.decode('utf-8')]))
            except (UnicodeDecodeError, csv.Error, StopIteration):
                raise SystemExit('G02-G24 STOP : ligne label TRAIN malformee : ' + mid)
            require(len(row) == 5 and row[0] == mid, 'format TRAIN label invalide : ' + mid)
            require(row[1] in ('0', '1') and row[4] in ('0', '1'),
                    'label TRAIN non binaire : ' + mid)
            ys[mid] = int(row[1])
            certified += int(row[4])
    require(len(all_seen) == len(split) == 1500 and all_seen == set(split),
            'couverture labels commune incomplete')
    require(set(ys) == train_ids and len(ys) == 845 and sum(ys.values()) == 211,
            'labels TRAIN incompatibles avec audit gele')
    require(certified == 0, 'certification TRAIN changee : revoir protocole')
    print('LABELS TRAIN : 845 lus | 211 positifs | certification independante=0')
    print('LABELS VALIDATION/TEST : 0 valeur interpretee')
    return ys


def construct_folds(records, n_folds=N_FOLDS):
    days = sorted({r[0] for r in records})
    require(len(days) >= 16, 'pas assez de jours TRAIN distincts')
    init = max(1, int(len(days) * INITIAL_DAY_RATIO))
    remaining = len(days) - init
    require(remaining >= n_folds, 'pas assez de jours apres premiere fenetre')
    partitions = []
    seen_val = set()
    for fold in range(n_folds):
        lo = init + (remaining * fold) // n_folds
        hi = init + (remaining * (fold + 1)) // n_folds
        train_days = set(days[:lo])
        eval_days = set(days[lo:hi])
        require(eval_days and not train_days.intersection(eval_days), 'jour partage')
        history = [r for r in records if r[0] in train_days]
        evaluate = [r for r in records if r[0] in eval_days]
        require(history and evaluate and max(r[1] for r in history) < min(r[1] for r in evaluate),
                'fuite chronologique entre historique et fold')
        require(not seen_val.intersection(r[2] for r in evaluate), 'evaluation repete une ligne')
        seen_val.update(r[2] for r in evaluate)
        partitions.append((history, evaluate, min(eval_days), max(eval_days)))
    require(len(partitions) == n_folds, 'nombre plis incorrect')
    require(len(seen_val) == sum(len(f[1]) for f in partitions), 'doublon OOF')
    return partitions, days


def diagnose(lib):
    np, sklearn, LogisticRegression, StandardScaler, roc_auc_score, brier_score_loss, log_loss, ConvergenceWarning = lib
    records, split, train_ids = read_train_only_features()
    folds, days = construct_folds(records)
    print('DATES TRAIN :', days[0], 'a', days[-1], '| jours UTC distincts', len(days))
    print('PREPARE :', len(folds), 'plis expanding-window sur TRAIN uniquement')
    ys = read_train_only_labels(split, train_ids)
    all_y, all_p, all_p_ref = [], [], []
    all_alerts, all_wins = 0, 0
    print('=== G02-G24/P2 | DIAGNOSTIC TRAIN WALK-FORWARD (NON GELANT) ===')
    for i, (hist, future, start_day, end_day) in enumerate(folds, 1):
        x_hist = np.asarray([r[3] for r in hist], dtype=np.float64)
        y_hist = np.asarray([ys[r[2]] for r in hist], dtype=np.int64)
        x_future = np.asarray([r[3] for r in future], dtype=np.float64)
        y_future = np.asarray([ys[r[2]] for r in future], dtype=np.int64)
        require(set(np.unique(y_hist)) == {0, 1}, 'une classe absente dans historique fold ' + str(i))
        require(set(np.unique(y_future)) == {0, 1}, 'une classe absente dans eval fold ' + str(i))
        scale = StandardScaler()
        xh = scale.fit_transform(x_hist)
        xf = scale.transform(x_future)
        model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear',
                                   max_iter=2000, class_weight=None, random_state=42)
        model.fit(xh, y_hist)
        require(model.n_iter_.max() < 2000, 'optimiseur potentiellement non converge')
        p = model.predict_proba(xf)[:, 1]
        require(bool(np.all(np.isfinite(p))), 'probabilites non finies fold')
        prior = float(y_hist.mean())
        ref = np.repeat(prior, len(y_future))
        alerts = p >= .60
        a = int(alerts.sum())
        w = int(np.logical_and(alerts, y_future == 1).sum())
        auc = float(roc_auc_score(y_future, p))
        bs = float(brier_score_loss(y_future, p))
        bs_ref = float(brier_score_loss(y_future, ref))
        print('FOLD %d | %s a %s | fit %d | eval %d | positifs eval %d'
              % (i, start_day, end_day, len(hist), len(future), int(y_future.sum())))
        print('  AUC %.4f | Brier %.4f | Brier ref past %.4f | pmax %.4f | alertes p>=0.60: %d | victoires %d'
              % (auc, bs, bs_ref, float(p.max()), a, w))
        all_y.extend(y_future.tolist())
        all_p.extend(p.tolist())
        all_p_ref.extend(ref.tolist())
        all_alerts += a
        all_wins += w
    yy = np.asarray(all_y)
    pp = np.asarray(all_p)
    rr = np.asarray(all_p_ref)
    auc_global = float(roc_auc_score(yy, pp)) if len(set(yy.tolist())) == 2 else float('nan')
    bs = float(brier_score_loss(yy, pp))
    bs_ref = float(brier_score_loss(yy, rr))
    print('=== SYNTHÈSE OOF TRAIN CHRONOLOGIQUE ===')
    print('OBSERVATIONS OOF : %d / 845 | positifs %d' % (len(yy), int(yy.sum())))
    print('AUC OOF globale (probabilites issues de fits differents) : %.4f' % auc_global)
    print('BRIER OOF : %.4f | BRIER REFERENCE PASSEE : %.4f | DELTA (modele - ref) : %+.4f'
          % (bs, bs_ref, bs - bs_ref))
    print('ALERTES OOF p>=0.60 : %d | positifs parmi alertes : %d' % (all_alerts, all_wins))
    print('IMPORTANT : analyse TRAIN-only descriptive, pas selection de seuil/feature/hyperparametre')
    print('IMPORTANT : AUC OOF agreguee sur plis a modeles differents; voir les AUC par pli')
    print('IMPORTANT : labels non independamment certifies; as-of NON certifie')
    print('VALIDATION : 0 outcome interprete | TEST : 0 outcome interprete / 0 timeline')
    print('0 API | 0 ECRITURE | aucun artefact precedent modifie')
    print('G02-G24/P2 TRAIN_ONLY_WALKFORWARD_DIAGNOSTIC COMPLETE | PAS DE DEPLOIEMENT LIVE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnose-train', action='store_true',
                   help='ouvre exclusivement les labels TRAIN et ajuste 4 modeles consecutifs sur TRAIN')
    args = p.parse_args()
    verify_precheck()
    if not args.diagnose_train:
        print('Pour lancer le diagnostic TRAIN uniquement, si le precheck est PASS :')
        print('  python3 -B g02g24_train_only_walkforward_diagnostic.py --diagnose-train')
        return
    lib = base.check_libraries()
    diagnose(lib)


if __name__ == '__main__':
    main()
