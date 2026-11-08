#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G32 - Inventaire local de sources candidates d'etats observes avant 87'.

Execution depuis la racine du projet ML_Goal_After_87 :
    python3 -B g02g32_asof_snapshot_inventory_readonly.py
    python3 -B g02g32_asof_snapshot_inventory_readonly.py --inventory

Ce script ne lit AUCUN fichier de donnees de candidats, aucun label,
aucune timeline, aucun score, et n'entraine aucun modele. Il verifie seulement
quatre fichiers FROZEN sans labels avec SHA256, puis, en --inventory,
parcourt les NOMS des fichiers sous le repertoire courant et leurs tailles.
Il evite notamment tous les dossiers de raw_timelines et de labels.

La presence de noms 'snapshot' ou 'live' ne prouve aucune disponibilite as-of.
Les mtime de fichiers n'ont aucune valeur de certification de capture.
"""

import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path

EXPECTED = {
    'outputs/v4_research/v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv':
        '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    'outputs/v4_research/v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv':
        'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    'outputs/v4_research/v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv':
        '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    'outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv':
        '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
}

SKIP_DIR = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache',
            '.mypy_cache', '.ruff_cache', 'raw_timelines', '.idea', '.vscode'}
PROHIBITED_SUBSTRINGS = ('label', 'target', 'outcome', 'result', 'final_score',
                         'reconstruction', 'predictions', 'prediction_results',
                         'postmatch', 'post_match', 'ground_truth')
FILE_EXT = {'.json', '.jsonl', '.csv', '.parquet', '.sqlite', '.db', '.log', '.txt', '.ndjson'}
STRONG = ('snapshot', 'snapshots', 'pre87', 'pre_87', 'pre-87', 'at_87', 'at87',
          '87_min', '87min', 'asof', 'as_of', 'as-of')
MEDIUM = ('live', 'scanner', 'capture', 'captured', 'journal', 'alerts', 'alert',
          'polling', 'replay', 'feed', 'telemetry', 'stream', 'tick')


def require(condition, reason):
    if not condition:
        raise SystemExit('G02-G32 STOP : ' + reason)


def sha256_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def precheck(root):
    require((root / 'outputs' / 'v4_research').is_dir(),
            'executer a la racine du projet (outputs/v4_research manquant)')
    for rel, expected in EXPECTED.items():
        p = root / rel
        require(p.is_file() and not p.is_symlink(), 'fichier gelé manquant/lien : ' + rel)
        require(sha256_file(p) == expected, 'SHA incorrect pour ' + rel)
        print('SHA OK : ' + p.name)
    print('GARDE : fichiers FROZEN sans labels conformes | aucun outcome ouvert')
    print('G02-G32/P1 PRECHECK PASS | 0 API | 0 LABEL | 0 TIMELINE | 0 ECRITURE | 0 FIT')


def inventory(root):
    findings = {'SNAPSHOT_CANDIDATE': [], 'OTHER_LIVE_CANDIDATE': []}
    counts = Counter()
    max_entries = 150000
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=True, followlinks=False):
        # Omettre les dossiers dont le nom suggere resultats/targets et tous raw_timelines.
        filtered = []
        for d in dirnames:
            if d in SKIP_DIR or any(k in d.lower() for k in PROHIBITED_SUBSTRINGS):
                counts['directories_skipped'] += 1
            else:
                filtered.append(d)
        dirnames[:] = filtered
        counts['directories_seen'] += 1
        for name in filenames:
            counts['files_seen'] += 1
            require(counts['files_seen'] <= max_entries,
                    'trop de fichiers; scan interrompu sans lire aucun contenu')
            if name.startswith('.'):
                continue
            p = Path(dirpath) / name
            suffix = p.suffix.lower()
            if suffix not in FILE_EXT:
                continue
            rel = p.relative_to(root).as_posix()
            low = rel.lower()
            if any(k in low for k in PROHIBITED_SUBSTRINGS):
                counts['sensitive_paths_skipped'] += 1
                continue
            # Ne jamais traiter les timelines historiques completes comme snapshots.
            if 'raw_timelines' in low:
                continue
            category = ('SNAPSHOT_CANDIDATE' if any(k in low for k in STRONG)
                        else 'OTHER_LIVE_CANDIDATE' if any(k in low for k in MEDIUM)
                        else None)
            if category is None:
                continue
            if p.is_symlink():
                counts['symlinks_skipped'] += 1
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                counts['stat_errors'] += 1
                continue
            findings[category].append((rel, sz))
    print('=== G02-G32/P2 | INVENTAIRE NOMS/TAILLES UNIQUEMENT ===')
    print('DOSSIERS PARCOURUS :', counts['directories_seen'],
          '| FICHIERS VUS :', counts['files_seen'],
          '| DOSSIERS IGNORES :', counts['directories_skipped'])
    for category, items in findings.items():
        print(category + ' :', len(items), 'fichiers candidats (NOM seulement)')
        for rel, sz in sorted(items)[:35]:
            print('  NOM CANDIDAT :', rel, '| octets :', sz)
        if len(items) > 35:
            print('  ...', len(items) - 35, 'autres noms omis')
    print('NOTE : noms/tailles ne prouvent PAS qu un fichier a ete capture avant 87 minutes.')
    print('NOTE : 0 candidat ne prouverait pas qu il n existe aucune source externe.')
    print('TEST FERME | 0 CONTENU DE FICHIER CANDIDAT | 0 LABEL | 0 TIMELINE')
    print('0 API | 0 ECRITURE | 0 FIT | AS-OF TOUJOURS NON CERTIFIE')
    print('G02-G32/P2 METADATA_INVENTORY COMPLETE | NOMS SEULEMENT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true', help='inventaire des seuls noms et tailles')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print('=== G02-G32/P1 | INVENTAIRE SOURCES AS-OF | READONLY ===')
    precheck(root)
    if args.inventory:
        inventory(root)
    else:
        print('Pour inventorier : python3 -B g02g32_asof_snapshot_inventory_readonly.py --inventory')


if __name__ == '__main__':
    main()
