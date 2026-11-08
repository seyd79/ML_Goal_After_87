#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2D -- Controle statique de la provenance F4, SANS ouvrir les labels.

Depuis ML_Goal_After_87:
    python3 -B g02g20_p2d_f4_provenance_static_readonly.py

Lit : 4 artefacts de garde (SHA), le RAPPORT de reconstruction F4 (metadonnees),
le code SOURCE de l'adaptateur et du constructeur (aucun module importe).
NE LIT PAS V42_LABELS_RESEARCH.csv, ni aucune timeline, ni les resultats TEST.
N'appelle aucune API, n'ecrit aucun fichier, n'entraine aucun modele.
Python 3.9+ / bibliotheque standard.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

GUARDS = {
    'outputs/v4_research/v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv':
        'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    'outputs/v4_research/v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv':
        '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    'outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv':
        '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    'outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json':
        '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
}
BASE = Path('outputs/v4_research/v42_g02f4_reconstruction')
REPORT = BASE / 'V42_RECONSTRUCTION_REPORT.json'
LABELS = BASE / 'V42_LABELS_RESEARCH.csv'
BUILDER = Path('build_historical_ab_research_v3.py')
ADAPTER = Path('g02f6_adapter_v42.py')


def stop_if(condition, msg):
    if condition:
        raise SystemExit('G02-G20/P2D STOP : ' + msg)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 256), b''):
            h.update(chunk)
    return h.hexdigest()


def protected_file(path):
    stop_if(not path.is_file() or path.is_symlink(), 'introuvable / lien symbolique : ' + str(path))


def show_context(lines, needles, before=4, after=5, cap=110):
    ranges = []
    for i, line in enumerate(lines):
        if any(tok in line.lower() for tok in needles):
            ranges.append((max(0, i-before), min(len(lines), i+after+1)))
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    n = 0
    for a, b in merged:
        if n >= cap:
            break
        print('--- LIGNES {}-{} ---'.format(a + 1, min(b, a + cap - n)))
        for idx in range(a, min(b, a + cap - n)):
            print('{:4d}: {}'.format(idx + 1, lines[idx][:220]))
            n += 1
    if n >= cap:
        print('... extraits limites a {} lignes de code, demander autres sections si necessaire'.format(cap))


def main():
    print('=== G02-G20/P2D | TRACE F4 + SHA BUILDER | READONLY ===')
    for name, expected in GUARDS.items():
        path = Path(name)
        protected_file(path)
        stop_if(digest(path) != expected, 'SHA garde divergent : ' + path.name)
        print('GARDE SHA OK :', path.name)

    protected_file(LABELS)
    print('FICHIER LABELS :', LABELS)
    print('TAILLE (octets, sans lire son contenu) :', LABELS.stat().st_size)
    protected_file(REPORT)
    protected_file(BUILDER)
    protected_file(ADAPTER)

    # Le rapport est un artefact de provenance distinct du CSV des labels.
    # N'afficher AUCUNE statistique / cible / valeur de label de ce rapport.
    with REPORT.open('r', encoding='utf-8-sig') as f:
        report = json.load(f)
    stop_if(not isinstance(report, dict), 'rapport F4 non-objet JSON')
    print('RAPPORT SOURCE F4 :', REPORT)
    builder_expected = report.get('builder_sha256')
    stop_if(not isinstance(builder_expected, str) or not re.fullmatch(r'[0-9a-f]{64}', builder_expected),
            'builder_sha256 manquant/invalide dans rapport F4')
    builder_actual = digest(BUILDER)
    print('BUILDER DANS RAPPORT SHA256 :', builder_expected)
    print('BUILDER ACTUEL SHA256 :     ', builder_actual)
    stop_if(builder_actual != builder_expected, 'builder different du SHA enregistre dans F4')
    print('BUILDER_SHA_MATCH : PASS')
    print('RAPPORT CONTIENT CHAMP labels_sha256 :', isinstance(report.get('labels_sha256'), str))
    print('ATTENTION : SHA du fichier CSV de labels NON recalcule pour garder le CSV ferme')

    adapter_src = ADAPTER.read_text(encoding='utf-8-sig')
    adapter_ast = ast.parse(adapter_src, filename=str(ADAPTER))
    del adapter_ast
    builder_src = BUILDER.read_text(encoding='utf-8-sig')
    builder_ast = ast.parse(builder_src, filename=str(BUILDER))
    del builder_ast
    print('SOURCES PYTHON : AST PASS (sans execution)')
    checks = ('V42_RECONSTRUCTION_REPORT.json' in adapter_src and
              'build_historical_ab_research_v3.py' in adapter_src and
              'builder_sha256' in adapter_src)
    print('LIEN ADAPTATEUR -> RAPPORT -> CONSTRUCTEUR :', 'DETECTE' if checks else 'NON DETECTE')
    print('=== EXTRAITS DU CODE DE CALCUL (pas de resultats de matchs) ===')
    show_context(builder_src.splitlines(), (
        'late_goals', 'goal_at_87', 'extra_time_goals_excluded',
        'event_minute', 'extra_time', 'label_certified_independently',
    ), before=2, after=3, cap=95)
    print('SOURCE DE RECONSTRUCTION : BUILDER RELIE AU RAPPORT F4, SI RAPPORT FIABLE')
    print('LABELS INDIVIDUELS : NON VERIFIES | TEST FERME | AS-OF NON CERTIFIE')
    print('0 API | 0 ECRITURE | 0 VALEUR DE LABEL LUE | 0 TRAIN')
    print('G02-G20/P2D STATIC PASS | BUILDER SHA VERIFIE | LABEL CSV NON OUVERT')


if __name__ == '__main__':
    main()
