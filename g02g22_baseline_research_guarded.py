#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G22/P1 — baseline V4.2 de recherche retrospective, TEST reste ferme.

Depuis la racine ML_Goal_After_87 (Python >= 3.9):
    python3 -B g02g22_baseline_research_guarded.py                 # PRECHECK, sans labels/train/ecriture
    python3 -B g02g22_baseline_research_guarded.py --train-freeze  # fit TRAIN et evaluation VALIDATION unique, gel
    python3 -B g02g22_baseline_research_guarded.py --verify        # verification des fichiers geles, sans labels/train

Modele strictement pre-enregistre: LogisticRegression(penalty='l2', C=0.1,
solver='liblinear', max_iter=2000, class_weight=None), StandardScaler fit
sur les 845 TRAIN admissibles seulement, seuil 0.60. Les 269 VALIDATION
n'entrent pas dans l'ajustement. Aucune lecture du TEST outcome; les
lignes du CSV commun hors allowlist ne sont pas interpretees au-dela de
leur identifiant. Aucune timeline n'est ouverte. Aucun acces API.

Ceci n'etablit ni certification independante des labels, ni disponibilite
historique as-of, ni performance du modele live, ni rentabilite de paris.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import warnings

import g02g21_research_go_nogo_readonly as gate
import g02g20_p2g_authorized_label_reconcile_readonly as label_guard

ROOT = Path('outputs/v4_research')
LABEL_CSV = ROOT / 'v42_g02f4_reconstruction/V42_LABELS_RESEARCH.csv'
OUTDIR = ROOT / 'v42_g02g22_baseline_research_v1'
RESULT = OUTDIR / 'G02G22_BASELINE_L2_C01_RESEARCH_FROZEN.json'
DIGEST = OUTDIR / 'G02G22_BASELINE_L2_C01_RESEARCH_FROZEN.sha256'
REQUIRED_SCRIPTS = {
    Path('g02g21_research_go_nogo_readonly.py'):
        '0b31a948837d4b23e99d1790756554920a9059678ab6d95e6c27b5a861660ef3',
    Path('g02g20_p2g_authorized_label_reconcile_readonly.py'):
        '71b86098dbb7f12725fa41a08413a3893c31f98f8d89449f75aea57fa0408cc6',
}
EXPECTED_AUTHORIZED_LABEL_SHA = 'b5d3aee84eb2211c1ec1c59f3479ed03f7601c49f3b627583bfc37d088920153'


def require(ok, msg):
    if not ok:
        raise SystemExit('G02-G22 STOP : ' + msg)


def sha_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(131072), b''):
            hasher.update(block)
    return hasher.hexdigest()


def source_guards():
    for path, expected in REQUIRED_SCRIPTS.items():
        require(path.is_file() and not path.is_symlink(), 'script de garde absent : ' + str(path))
        current = sha_file(path)
        require(current == expected, 'script de garde modifie : ' + str(path) + ' | ' + current)
        print('SCRIPT SHA OK :', path.name)
    require(Path(gate.__file__).resolve() == (Path.cwd() / 'g02g21_research_go_nogo_readonly.py').resolve(),
            'module gate importe hors du projet')
    require(Path(label_guard.__file__).resolve() == (Path.cwd() / 'g02g20_p2g_authorized_label_reconcile_readonly.py').resolve(),
            'module labels importe hors du projet')
    gate.main()  # Verification sans valeur de label; 7 SHA geles et gate scientifique.
    label_guard.read_header_only(LABEL_CSV)  # Uniquement l'en-tete.
    require(LABEL_CSV.is_file() and not LABEL_CSV.is_symlink(), 'CSV labels non regulier')
    print('CSV LABELS : en-tete seul confirme, valeurs non ouvertes')


def check_libraries():
    try:
        import numpy as np
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:
        raise SystemExit('G02-G22 STOP : scikit-learn/numpy absent dans le venv : ' + str(exc))
    print('ENVIRONNEMENT : numpy', np.__version__, '| sklearn', sklearn.__version__)
    return (np, sklearn, LogisticRegression, StandardScaler,
            roc_auc_score, brier_score_loss, log_loss, ConvergenceWarning)


def precheck():
    print('=== G02-G22/P1 | PRECHECK BASELINE RETROSPECTIVE | TEST FERME ===')
    source_guards()
    lib = check_libraries()
    print('MODELE : LogisticRegression L2 C=0.1 liblinear max_iter=2000 class_weight=None')
    print('SCALER : fit sur 845 TRAIN uniquement | 10 features | seuil=0.60')
    print('LABELS : 0 ouvert | TIMELINES : 0 ouverte | TEST : ferme')
    print('G02-G22/P1 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
    return lib


def select_features():
    # Dataset non-label: all 1500 entries may be indexed by ID, but only allowed
    # TRAIN+VALIDATION entries have their 10 features converted into numbers.
    data = gate.read_rows(gate.DATA, ('match_id', 'kickoff_utc') + gate.FEATURES)
    allow = gate.read_rows(gate.ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    split = gate.read_rows(gate.SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = gate.read_rows(gate.ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    require(len(data) == len(split) == len(elig) == 1500 and len(allow) == 1114,
            'effectifs sources modifies')
    test_ids = {mid for mid, row in split.items() if row['partition'] == 'TEST'}
    require(len(test_ids) == 302 and test_ids.isdisjoint(allow), 'TEST dans allowlist')
    require(set(allow) == {mid for mid, row in elig.items()
                           if row['partition'] in ('TRAIN', 'VALIDATION')
                           and row['research_eligibility'] == 'RESEARCH_ELIGIBLE'},
            'allowlist differente de l eligibilite')
    out = {'TRAIN': [], 'VALIDATION': []}
    for mid, access in allow.items():
        part = access['partition']
        require(part in out and split[mid]['partition'] == part
                and elig[mid]['partition'] == part and mid in data,
                'partition non conforme : ' + mid)
        require(access['kickoff_utc'] == split[mid]['kickoff_utc'] ==
                elig[mid]['kickoff_utc'] == data[mid]['kickoff_utc'],
                'kickoff different : ' + mid)
        values = []
        for field in gate.FEATURES:
            raw = data[mid][field]
            require(isinstance(raw, str) and raw.strip() != '', 'feature vide : ' + mid + '/' + field)
            try:
                val = float(raw)
            except ValueError:
                raise SystemExit('G02-G22 STOP : feature non numerique : ' + mid + '/' + field)
            require(math.isfinite(val) and val >= 0, 'feature incorrecte : ' + mid + '/' + field)
            values.append(val)
        out[part].append((access['kickoff_utc'], mid, values))
    for part, expected in [('TRAIN', 845), ('VALIDATION', 269)]:
        out[part].sort(key=lambda x: (x[0], x[1]))
        require(len(out[part]) == expected, 'taille ' + part + ' incorrecte')
    require(max(x[0] for x in out['TRAIN']) < min(x[0] for x in out['VALIDATION']),
            'ordre temporel TRAIN / VALIDATION invalide')
    return out, allow, split


def authorized_labels(allow, split):
    # Uses the frozen audited P2G CSV reader: for TEST and excluded IDs only
    # the match_id prefix is decoded; outcome values are NEVER parsed.
    rows = label_guard.read_authorized_csv_rows(LABEL_CSV, allow, split)
    require(set(rows) == set(allow) and len(rows) == 1114,
            'ensemble labels autorises incomplet')
    h = hashlib.sha256()
    positives, independently = Counter(), Counter()
    ys = {}
    for mid in sorted(allow):
        part = allow[mid]['partition']
        require(part in ('TRAIN', 'VALIDATION'), 'label TEST bloque')
        row = rows[mid]
        normalized = {'match_id': mid, 'partition': part}
        for field in ('goal_after_87', 'goal_at_87', 'extra_time_goals_excluded'):
            val = label_guard.parse_count(row[field], field)
            if field != 'extra_time_goals_excluded':
                require(val in (0, 1), 'valeur binaire invalide : ' + field + '/' + mid)
            normalized[field] = val
        cert = label_guard.parse_binary(row['label_certified_independently'],
                                        'label_certified_independently')
        normalized['label_certified_independently'] = cert
        h.update((json.dumps(normalized, ensure_ascii=True, separators=(',', ':')) + '\n').encode('ascii'))
        positives[part] += normalized['goal_after_87']
        independently[part] += cert
        ys[mid] = normalized['goal_after_87']
    require(h.hexdigest() == EXPECTED_AUTHORIZED_LABEL_SHA,
            'empreinte des 1114 labels autorises differente du gel G02-G20')
    require(positives == Counter({'TRAIN': 211, 'VALIDATION': 90}),
            'distribution de labels changee')
    require(not any(independently.values()),
            'certification independante a change; revoir decision GO/NO-GO')
    print('LABELS AUTORISES SHA :', h.hexdigest())
    print('TRAIN : 211 positifs/845 | VALIDATION : 90 positifs/269 | TEST : 0 interprete')
    return ys


def wilson(successes, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    div = 1 + z*z/n
    center = (p + z*z/(2*n))/div
    margin = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/div
    return [max(0.0, center - margin), min(1.0, center + margin)]


def train_and_report(lib):
    (np, sklearn, LogisticRegression, StandardScaler,
     roc_auc_score, brier_score_loss, log_loss, ConvergenceWarning) = lib
    features, allow, split = select_features()
    ys = authorized_labels(allow, split)
    train = features['TRAIN']
    valid = features['VALIDATION']
    X_train = np.asarray([r[2] for r in train], dtype=np.float64)
    X_val = np.asarray([r[2] for r in valid], dtype=np.float64)
    y_train = np.asarray([ys[r[1]] for r in train], dtype=np.int64)
    y_val = np.asarray([ys[r[1]] for r in valid], dtype=np.int64)
    require(X_train.shape == (845, 10) and X_val.shape == (269, 10), 'dimensions incorrectes')
    require(int(y_train.sum()) == 211 and int(y_val.sum()) == 90, 'positifs invalides')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # UNIQUE fit du scaler, TRAIN.
    frozen = json.loads(gate.SCALER.read_text(encoding='utf-8'))
    stats = frozen['normalization_design']['statistics']
    for j, field in enumerate(gate.FEATURES):
        ref = stats[field]
        require(math.isclose(float(scaler.mean_[j]), float(ref['mean_train']), rel_tol=1e-9, abs_tol=1e-9),
                'moyenne scaler differente du gel G02-G19 : ' + field)
        require(math.isclose(float(scaler.scale_[j]), float(ref['scale_train']), rel_tol=1e-9, abs_tol=1e-9),
                'ecart-type scaler different du gel G02-G19 : ' + field)
    X_val_scaled = scaler.transform(X_val)  # VALIDATION : transform uniquement.

    model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear',
                               max_iter=2000, class_weight=None, random_state=42)
    with warnings.catch_warnings(record=True) as all_warnings:
        warnings.simplefilter('always')
        model.fit(X_train_scaled, y_train)
    require(not any(issubclass(w.category, ConvergenceWarning) for w in all_warnings),
            'optimiseur non converge')
    require(model.classes_.tolist() == [0, 1], 'classes de probabilite inattendues')
    probabilities = model.predict_proba(X_val_scaled)[:, 1]
    require(bool(np.all(np.isfinite(probabilities))), 'probabilites non finies')
    predict = probabilities >= 0.60  # Seuil fixe, jamais selectionne sur VALIDATION.
    alert_count = int(predict.sum())
    true_pos = int(np.logical_and(predict, y_val == 1).sum())
    false_pos = alert_count - true_pos
    rec = true_pos / int(y_val.sum())
    prec = true_pos / alert_count if alert_count else None
    auc = float(roc_auc_score(y_val, probabilities))
    brier = float(brier_score_loss(y_val, probabilities))
    loss = float(log_loss(y_val, probabilities, labels=[0, 1]))
    metrics = {
        'threshold_pre_registered': 0.60,
        'validation_count': 269,
        'validation_positive_count': 90,
        'validation_positive_rate': 90/269,
        'alert_count_at_0_60': alert_count,
        'true_positives_at_0_60': true_pos,
        'false_positives_at_0_60': false_pos,
        'precision_at_0_60': prec,
        'precision_wilson95': wilson(true_pos, alert_count),
        'recall_at_0_60': rec,
        'roc_auc': auc,
        'brier_score': brier,
        'log_loss': loss,
    }
    report = {
        'stage': 'G02-G22/P1',
        'status': 'RETROSPECTIVE_EXPLORATORY_NOT_INDEPENDENTLY_CERTIFIED_NOT_ASOF',
        'frozen_input_sha256': {p.name: h for p, h in gate.EXPECTED.items()},
        'authorized_labels_sha256': EXPECTED_AUTHORIZED_LABEL_SHA,
        'source_script_sha256': {p.name: h for p, h in REQUIRED_SCRIPTS.items()},
        'training_script_sha256': sha_file(Path(__file__)),
        'libraries': {'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'feature_names_in_order': list(gate.FEATURES),
        'train': {'rows': 845, 'positives': 211, 'labels_independently_certified': 0},
        'validation': {'rows': 269, 'positives': 90, 'labels_independently_certified': 0},
        'test': {'total_matches': 302, 'eligible_matches': 286,
                 'outcome_labels_parsed': 0, 'timelines_opened': 0, 'model_fit_or_tuned_on_test': False},
        'scaler': {'name': 'StandardScaler', 'fit_rows': 845,
                   'mean': [float(x) for x in scaler.mean_],
                   'scale': [float(x) for x in scaler.scale_],
                   'variance': [float(x) for x in scaler.var_]},
        'model': {'name': 'LogisticRegression', 'penalty': 'l2', 'C': 0.1,
                  'solver': 'liblinear', 'max_iter': 2000, 'class_weight': None,
                  'random_state': 42, 'fitted_iterations': model.n_iter_.tolist(),
                  'intercept': [float(x) for x in model.intercept_],
                  'coefficients_in_feature_order': [float(x) for x in model.coef_[0]]},
        'validation_metrics': metrics,
        'limitations': [
            'Labels internally reconciled but none independently certified.',
            'Prematch and live features not historically as-of certified.',
            'One retrospective VALIDATION evaluation; future tuning here reduces independence.',
            'TEST outcomes remain sealed and cannot be used until separate final authorization.',
            'Not a calibrated live probability or evidence of betting profitability.',
        ],
        'api_calls': 0,
        'test_outcomes_read': False,
        'live_deployment_authorized': False,
    }
    print('=== G02-G22/P1 | RESULTATS DE LA BASELINE RETROSPECTIVE ===')
    print('TRAIN : 845 matchs | 211 positifs | model fit L2 C=0.1')
    print('VALIDATION : 269 matchs | 90 positifs')
    print('ALERTES (p >= 0.60) :', alert_count)
    print('VICTOIRES :', true_pos, '| FAUSSES ALERTES :', false_pos)
    print('PRECISION :', 'NON DEFINIE (0 alerte)' if prec is None else ('%.4f (%.2f%%)' % (prec, 100*prec)))
    print('WILSON 95% :', 'NON DEFINI' if prec is None else
          ('[%.2f%%, %.2f%%]' % tuple(100*x for x in metrics['precision_wilson95'])))
    print('RAPPEL : %.4f | ROC AUC : %.4f | BRIER : %.4f | LOG LOSS : %.4f' % (rec, auc, brier, loss))
    print('STATUT : RECHERCHE RETROSPECTIVE | AS-OF NON CERTIFIE | CERTIFICATION INDEPENDANTE 0')
    print('TEST FERME | 0 API | AUCUNE MODIFICATION DES SOURCES')
    return report


def bytes_report(report):
    return (json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def do_verify():
    print('=== G02-G22/P1 | VERIFICATION DU RAPPORT FROZEN ===')
    for path in (RESULT, DIGEST):
        require(path.is_file() and not path.is_symlink(), 'fichier resultat absent : ' + str(path))
    expected = DIGEST.read_text(encoding='ascii').strip()
    require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected),
            'empreinte enregistree invalide')
    actual = sha_file(RESULT)
    require(actual == expected, 'rapport modifie ou altere')
    report = json.loads(RESULT.read_text(encoding='utf-8'))
    require(report.get('stage') == 'G02-G22/P1' and
            report.get('status') == 'RETROSPECTIVE_EXPLORATORY_NOT_INDEPENDENTLY_CERTIFIED_NOT_ASOF',
            'statut resultat inattendu')
    require(report.get('test', {}).get('outcome_labels_parsed') == 0 and
            report.get('test', {}).get('timelines_opened') == 0 and
            report.get('test_outcomes_read') is False and
            report.get('live_deployment_authorized') is False,
            'protection TEST / politique live invalide')
    print('SHA RESULTAT :', actual)
    print('VALIDATION :', report['validation_metrics'])
    print('G02-G22/P1 VERIFY PASS | 0 API | 0 ECRITURE | 0 LABEL OUVERT | 0 TRAIN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--train-freeze', action='store_true',
                       help='ouvre les labels autorises, ajuste 845 TRAIN, evalue 269 VALIDATION et ecrit un nouveau rapport')
    modes.add_argument('--verify', action='store_true', help='verifie uniquement le rapport et sa SHA')
    args = parser.parse_args()
    lib = precheck()
    if args.verify:
        do_verify()
        return
    if not args.train_freeze:
        print('Pour entrainer et figer apres accord : python3 -B g02g22_baseline_research_guarded.py --train-freeze')
        return
    require(not OUTDIR.exists(), 'dossier resultat deja present : pas d ecrasement ' + str(OUTDIR))
    report = train_and_report(lib)
    blob = bytes_report(report)
    fingerprint = hashlib.sha256(blob).hexdigest()
    require(not OUTDIR.exists(), 'dossier cree entre temps')
    OUTDIR.mkdir(parents=True, exist_ok=False)
    with RESULT.open('xb') as f:
        f.write(blob)
    with DIGEST.open('x', encoding='ascii') as f:
        f.write(fingerprint + '\n')
    print('RAPPORT FIGE :', RESULT)
    print('SHA RAPPORT :', fingerprint)
    print('G02-G22/P1 TRAIN_FREEZE PASS | TRAIN 845 | VALIDATION 269 | TEST 0 | 0 API')


if __name__ == '__main__':
    main()
