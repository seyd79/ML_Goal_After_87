#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2C -- localisation statique du code producteur de labels.

Depuis la racine ML_Goal_After_87 :
    python3 -B g02g20_p2c_label_builder_static_trace.py

Lit seulement des fichiers .py et deux artefacts de garde G02-G19 pour verifier
leurs empreintes. NE LIT JAMAIS V42_LABELS_RESEARCH.csv ni les raw_timelines.
Aucune API, aucune ecriture, aucune ouverture des valeurs des labels, aucun train.
Python 3.9+, stdlib uniquement.
"""
import hashlib
import os
from pathlib import Path

GUARD_FILES = {
    Path('outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/'
         'G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'):
        '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    Path('outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/'
         'G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'):
        '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
}
LABEL_FILE = Path('outputs/v4_research/v42_g02f4_reconstruction/V42_LABELS_RESEARCH.csv')
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules',
             'raw_timelines', 'raw_pages', 'raw_shotmaps',
             'rejected_timelines', 'quarantine', 'site-packages'}
SEARCH_TOKENS = (
    'V42_LABELS_RESEARCH', 'label_certified_independently',
    'extra_time_goals_excluded', 'goal_at_87', 'goal_after_87',
)
MAX_PY_FILES = 5000
MAX_BYTES_PER_SCRIPT = 2000000
MAX_RESULTS = 12
MAX_LINES_PER_RESULT = 18


def check(condition, msg):
    if not condition:
        raise SystemExit('G02-G20/P2C STOP : ' + msg)


def iter_python_sources():
    n = 0
    for current, dirs, filenames in os.walk('.', followlinks=False):
        dirs[:] = sorted(name for name in dirs
                         if name not in SKIP_DIRS and
                         not (Path(current) / name).is_symlink())
        for name in sorted(filenames):
            if not name.endswith('.py'):
                continue
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            n += 1
            check(n <= MAX_PY_FILES, 'plus de 5000 scripts Python; limiter la recherche')
            yield path


def main():
    print('=== G02-G20/P2C | TRACE STATIQUE DU PRODUCTEUR DE LABELS | READONLY ===')
    for path, expected in GUARD_FILES.items():
        check(path.is_file() and not path.is_symlink(), 'garde G02-G19 manquante : ' + str(path))
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()
        check(actual == expected, 'SHA garde G02-G19 different : ' + path.name)
        print('GARDE SHA OK :', path.name)

    check(LABEL_FILE.is_file() and not LABEL_FILE.is_symlink(),
          'fichier des labels absent ou lien symbolique : ' + str(LABEL_FILE))
    print('FICHIER LABELS REPERE (NOM + TAILLE UNIQUEMENT) :', LABEL_FILE)
    print('OCTETS :', LABEL_FILE.stat().st_size)
    print('IMPORTANT : les valeurs des 1500 labels ne sont pas lues par ce script')

    found, scanned, skipped = [], 0, 0
    for path in iter_python_sources():
        if path.stat().st_size > MAX_BYTES_PER_SCRIPT:
            skipped += 1
            continue
        try:
            raw = path.read_bytes()
            src = raw.decode('utf-8-sig')
        except (OSError, UnicodeError):
            skipped += 1
            continue
        scanned += 1
        hits = [(i, line.rstrip()) for i, line in enumerate(src.splitlines(), 1)
                if any(tok in line for tok in SEARCH_TOKENS)]
        if not hits:
            continue
        filename_hits = sum('V42_LABELS_RESEARCH' in line for _, line in hits)
        certification_hits = sum('label_certified_independently' in line for _, line in hits)
        score = 100 * filename_hits + 15 * certification_hits + len(hits)
        found.append((score, str(path), hashlib.sha256(raw).hexdigest(), hits))
    found.sort(key=lambda x: (-x[0], x[1]))

    print('SCRIPTS PYTHON PARCOURUS :', scanned, '| ignores/non lisibles :', skipped)
    print('SCRIPTS AVEC REFERENCES AUX CHAMPS LABELS :', len(found))
    if not found:
        print('AUCUNE REFERENCE : le producteur ne peut pas etre identifie depuis les scripts disponibles')
    for score, path, digest, hits in found[:MAX_RESULTS]:
        print('--- SOURCE CANDIDATE (code seulement) ---')
        print('CHEMIN :', path)
        print('SHA256 :', digest)
        print('LIGNES CORRESPONDANTES :', len(hits))
        for line_number, line in hits[:MAX_LINES_PER_RESULT]:
            print(f'  {line_number}: {line[:230]}')
        if len(hits) > MAX_LINES_PER_RESULT:
            print('  ... autres lignes non affichees :', len(hits) - MAX_LINES_PER_RESULT)
    if len(found) > MAX_RESULTS:
        print('AUTRES SCRIPTS NON AFFICHES :', len(found) - MAX_RESULTS)

    exact = [path for _, path, _, hits in found
             if any('V42_LABELS_RESEARCH' in line for _, line in hits)]
    print('SCRIPTS REFERENCANT LE CSV PAR NOM :', len(exact))
    for path in exact[:8]:
        print('  ', path)
    print('CERTIFICATION DES LABELS : NON DEMONTREE PAR CETTE TRACE STATIQUE')
    print('AS-OF NON CERTIFIE | 0 API | 0 ECRITURE | 0 VALEUR DE LABEL LUE | 0 TRAIN')
    print('G02-G20/P2C STATIC PASS | SOURCE(S) CANDIDATE(S) A EXAMINER | TEST FERME')


if __name__ == '__main__':
    main()
