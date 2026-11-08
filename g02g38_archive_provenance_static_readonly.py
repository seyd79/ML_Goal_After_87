#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G38 | Cohorte d'archives V3C : provenance et garde du holdout.

Utilisation depuis la racine du projet ML_Goal_After_87 :
    python3 -B g02g38_archive_provenance_static_readonly.py
    python3 -B g02g38_archive_provenance_static_readonly.py --audit-provenance

P1: SHA du script G02-G37, prechecks transitifs de la chaine G02-G33..G37,
    et selftests sans ouvrir les archives.
P2: 1) lit uniquement les IDs et partitions d'un CSV V4.2 gelé SANS labels,
    2) lit uniquement le prefixe autorise (ID, horodatage UTC, minute)
       des archives V3C, via le parseur audite de G02-G37,
    3) expose seulement les metadonnees des captures dont la minute recule,
    4) recherche STATIQUEMENT les emplacements dans le code Python ou le
       champ de capture et les noms des archives sont mentionnes.

IMPORTANT : aucun score, label, resultat, probabilite, feature, timeline,
source JSON de match, API, reseau, modele, ecriture sur disque n'est utilise.
Seul le CSV de partition, qui ne contient aucun outcome, est lu en entier.
Le balayage de code ne l'execute jamais et ne publie aucune ligne de code.
Un match_id commun ne suffit pas pour prouver l'identite des evenements,
ni pour certifier l'authenticite du timestamp. Le zero intersection ne
prouve pas l'independance statistique de nouvelles donnees.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
from collections import Counter, defaultdict

G37_NAME = 'g02g37_archive_capture_internal_consistency_readonly.py'
G37_SHA256 = 'b426860e6af63496a3cba3b6af2b11e609858fa485900bb44b6bca065419751e'
SPLIT_REL = ('outputs/v4_research/v42_g02g16_chrono_split_research_v1/'
             'G02G16_CHRONO_SPLITS_1500_FROZEN.csv')
EXPECTED_COUNTS = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
EXPECTED_SPLIT_COLS = ['match_id', 'kickoff_utc', 'utc_day', 'partition']
RELEVANT_MARKERS = ('snapshot_timestamp_utc', 'live_feature_snapshots.csv',
                    'live_feature_snapshots_v3c.csv', 'v3c_127_alertes_snapshots.csv')
WRITER_MARKERS = ('writerow(', 'writerows(', 'dictwriter(', 'to_csv(',
                  'open(', '.write(', 'append(', 'csv.writer(', 'write_csv(')
CLOCK_MARKERS = ('datetime.now(', 'datetime.utcnow(', 'time.time(',
                 'datetime.datetime.now(', 'utcnow(', 'strftime(')
EXCLUDED_DIRS = {
    'outputs', '.venv', 'venv', 'env', '.git', '.svn', '__pycache__',
    'node_modules', '.mypy_cache', '.pytest_cache', 'site-packages',
    'build', 'dist', 'cache', 'logs', '.tox', '.idea', '.vscode',
}
MAX_SOURCE_FILES = 10000
MAX_SOURCE_BYTES = 800000
MAX_MATCHED_FILES = 30


def require(cond, reason):
    if not cond:
        raise SystemExit('G02-G38 STOP : ' + str(reason))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def decreasing_pairs(records):
    """Retourne couples metadata seulement, le temps doit strictement croitre."""
    ordered = sorted(set(records))
    return [((t1, m1), (t2, m2))
            for (t1, m1), (t2, m2) in zip(ordered, ordered[1:])
            if t2 > t1 and m2 < m1]


def selftest(g34):
    from datetime import datetime, timezone, timedelta
    t0 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
    require(not decreasing_pairs([(t0, 81), (t0 + timedelta(minutes=1), 82)]),
            'selftest ordre chronologique')
    a = decreasing_pairs([(t0, 84), (t0 + timedelta(minutes=1), 83)])
    require(len(a) == 1 and a[0][0][1] == 84 and a[0][1][1] == 83,
            'selftest minute decroissante')
    # Prove that untrusted columns beyond the three authorized prefix fields
    # are never fed to a decoder. Invalid bytes would crash a full decode.
    raw = (b'2026-10-18T15:30:00Z,v3,mt_123,1,2,home,away,86,'
           b'unsafe_score,\xff\xfe,unsafe_label\n')
    fields = g34.parse_csv_initial_fields(raw, 8)
    require(fields[0].endswith('Z') and fields[2] == 'mt_123' and fields[7] == '86',
            'selftest parseur de prefixe')


def precheck():
    p = Path(G37_NAME)
    require(p.is_file() and not p.is_symlink(), 'script G02-G37 absent/lien symbolique')
    require(sha256(p) == G37_SHA256, 'empreinte G02-G37 differente')
    import g02g37_archive_capture_internal_consistency_readonly as g37
    g34, root, train_ids = g37.precheck()
    selftest(g34)
    print('=== G02-G38/P1 | PROVENANCE ARCHIVES | PRECHECK ===')
    print('SHA SCRIPT G02-G37 : OK | SELFTEST PREFIXE/CHRONOLOGIE : PASS')
    print('GARDE : partitions V4.2 pour match_id uniquement; aucun outcome VAL/TEST')
    print('GARDE : snapshots = prefixe (ID, date UTC, minute), aucune autre colonne')
    print('GARDE : recherche statique dans les .py hors outputs/.venv; aucun code execute')
    print('G02-G38/P1 PRECHECK PASS | 0 ARCHIVE | 0 LABEL | 0 TIMELINE | 0 API | 0 ECRITURE | 0 FIT')
    return g37, g34, root, train_ids


def load_split_ids(g34, root):
    import g02g33_snapshot_csv_schema_readonly as g33
    p = g33.secure_file(root, SPLIT_REL)
    cols = g33.read_one_csv_header(p)
    require(cols == EXPECTED_SPLIT_COLS, 'schema split non reconnu: stop')
    sets = defaultdict(set)
    unique = set()
    with p.open('r', encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        require(rdr.fieldnames == EXPECTED_SPLIT_COLS, 'colonnes split incorrectes')
        for row in rdr:
            require(None not in row and None not in row.values(),
                    'ligne split incomplete')
            mid = row['match_id'].strip()
            part = row['partition'].strip()
            require(mid and mid not in unique, 'ID vide ou doublon split')
            require(part in EXPECTED_COUNTS, 'partition inconnue')
            unique.add(mid)
            sets[part].add(mid)
    require({part: len(sets[part]) for part in EXPECTED_COUNTS} == EXPECTED_COUNTS,
            'nombre de matchs par partition incorrect')
    require(len(unique) == 1500, '1500 IDs attendus')
    return sets


def source_code_inventory(root):
    """Lecture statique de scripts seulement, pas d'execution, pas de snippets."""
    hits = []
    examined = 0
    skipped_large = 0
    for here, dirs, files in os.walk(str(root), followlinks=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS
                   and not (Path(here) / d).is_symlink()]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            examined += 1
            require(examined <= MAX_SOURCE_FILES, 'trop de scripts : stop')
            src = Path(here) / fname
            if src.is_symlink() or not src.is_file():
                continue
            if src.stat().st_size > MAX_SOURCE_BYTES:
                skipped_large += 1
                continue
            rel = str(src.relative_to(root))
            # Audits already provided are consumers rather than collector code.
            if fname.startswith('g02g3') or fname.startswith('g02g2'):
                continue
            try:
                body = src.read_text(encoding='utf-8')
            except (UnicodeDecodeError, OSError):
                continue
            lines = body.splitlines()
            target_lines = [i + 1 for i, line in enumerate(lines)
                            if any(tok in line.lower() for tok in RELEVANT_MARKERS)]
            if not target_lines:
                continue
            lower_body = body.lower()
            writer_possible = any(tok in lower_body for tok in WRITER_MARKERS)
            clock_possible = any(tok in lower_body for tok in CLOCK_MARKERS)
            hits.append((bool(writer_possible), bool(clock_possible), rel,
                         target_lines[:8], len(target_lines)))
    hits.sort(key=lambda x: (-int(x[0]), -int(x[1]), x[2]))
    print('\n=== CODE PYTHON | INVENTAIRE DE PROVENANCE STATIQUE ===')
    print('FICHIERS .py vus :', examined, '| > limite taille ignores :', skipped_large)
    print('SCRIPTS avec reference aux champs/fichiers des captures :', len(hits))
    for potential_writer, clock, rel, lines, total in hits[:MAX_MATCHED_FILES]:
        print('  CODE :', rel, '| lignes marqueurs :', ','.join(map(str, lines)),
              '| total :', total,
              '| indices ecriture meme fichier :', 'OUI' if potential_writer else 'NON',
              '| indices horloge meme fichier :', 'OUI' if clock else 'NON')
    if len(hits) > MAX_MATCHED_FILES:
        print('  ...', len(hits) - MAX_MATCHED_FILES, 'autres fichiers omis')
    print('ATTENTION : ces indices de code ne prouvent pas ou/quand le timestamp a ete produit.')


def audit(g37, g34, root, train_ids):
    import g02g33_snapshot_csv_schema_readonly as g33
    sets = load_split_ids(g34, root)
    require(set(train_ids).issubset(sets['TRAIN']),
            'les 845 TRAIN admissibles ne sont pas dans la partition TRAIN')
    print('=== G02-G38/P2 | IDS V4.2 + ANOMALIES + CODE | METADONNEES SEULEMENT ===')
    print('V4.2 geles (ID + partition) : TRAIN 912 | VALIDATION 286 | TEST 302')
    all_records = defaultdict(list)
    all_ids = set()
    archive_counts = Counter()
    for config in g34.SOURCES:
        res = g37.scan_one(g34, root, config)
        archive_counts[res['rel']] = res['rows']
        for mid, captures in res['records'].items():
            all_ids.add(mid)
            all_records[mid].extend(captures)
    print('IDS distincts des 4 archives :', len(all_ids))
    print('LIGNES par archive :', dict(archive_counts))
    for part in ('TRAIN', 'VALIDATION', 'TEST'):
        print('INTERSECTION IDs archives x', part, ':', len(all_ids & sets[part]),
              '/', len(sets[part]), '| IDENTIFIANTS UNIQUEMENT, 0 OUTCOME')
    print('\n=== ANOMALIES DE MINUTE DECROISSANTE (UTC + MINUTE SEULEMENT) ===')
    count_ids = 0
    count_pairs = 0
    for mid, samples in sorted(all_records.items()):
        anomalies = decreasing_pairs(samples)
        if not anomalies:
            continue
        count_ids += 1
        count_pairs += len(anomalies)
        print('  ID :', mid, '| transitions en baisse :', len(anomalies))
        for (t1, m1), (t2, m2) in anomalies[:5]:
            print('    ', t1.isoformat(), 'minute', m1, '->',
                  t2.isoformat(), 'minute', m2,
                  '| ecart UTC sec', round((t2 - t1).total_seconds(), 3))
    print('TOTAL anomalies :', count_ids, 'identifiants |', count_pairs, 'transitions')
    source_code_inventory(root)
    print('\nLIMITES : aucune heure de kickoff verifiee pour les archives V3C;')
    print('          aucun timestamp fournisseur as-of certifie;')
    print('          independance prouvee seulement au niveau de l ID, si intersection nulle;')
    print('          aucun label/outcome consulte, aucune performance calculee.')
    print('0 API | 0 ECRITURE | 0 FIT | TEST OUTCOMES FERMES')
    print('G02-G38/P2 PROVENANCE_STATIC_TRIAGE COMPLETE | RESEARCH ONLY | AS-OF NON CERTIFIE')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-provenance', action='store_true',
                    help='IDs seuls, anomalies horodatages+minutes, provenance code statique')
    args = ap.parse_args()
    g37, g34, root, train_ids = precheck()
    if args.audit_provenance:
        audit(g37, g34, root, train_ids)
    else:
        print('Puis : python3 -B g02g38_archive_provenance_static_readonly.py --audit-provenance')


if __name__ == '__main__':
    main()
