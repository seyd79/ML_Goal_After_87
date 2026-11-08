#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G35 | Audit de convention match_id entre TRAIN V4.2 et snapshots.

Run from ML_Goal_After_87 project root:
  python3 -B g02g35_snapshot_id_namespace_readonly.py
  python3 -B g02g35_snapshot_id_namespace_readonly.py --audit-ids

Read-only, no API, no training, no timelines. The archive reader only decodes
match_id, stopping before all other fields including outcomes and predictions.
It never decodes an outcome from any cohort. Id format comparisons are ONLY
heuristics, NOT verified joins; never use them to attach features or labels.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re

G34_NAME = 'g02g34_snapshot_train_capture_overlap_readonly.py'
G34_SHA256 = 'f37995bc10aef0151c97e196908e73dfc4f2bb05f20679c36f4218e0302e0aae'
ID_PATTERN = re.compile(r'^(?:(mt|match|fixture)[_-])?(\d{4,16})$', re.IGNORECASE)
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)


def require(cond, reason):
    if not cond:
        raise SystemExit('G02-G35 STOP : ' + str(reason))


def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def format_class(mid):
    if UUID_PATTERN.fullmatch(mid):
        return 'uuid'
    m = ID_PATTERN.fullmatch(mid)
    if m:
        return (m.group(1).lower() + '_<digits>') if m.group(1) else 'digits_only'
    if not mid:
        return 'empty'
    if mid.isascii() and mid.isalnum():
        return 'alphanumeric_other'
    return 'other'


def loose_numeric_core(mid):
    """Diagnostic ONLY: removing prefixes/leading zeros is NOT an identity join."""
    m = ID_PATTERN.fullmatch(mid)
    return (m.group(2).lstrip('0') or '0') if m else None


def selftest():
    assert format_class('mt_00123456') == 'mt_<digits>'
    assert format_class('123456') == 'digits_only'
    assert format_class('fixture-1234') == 'fixture_<digits>'
    assert loose_numeric_core('mt_001234') == '1234'
    assert loose_numeric_core('1234') == '1234'
    assert loose_numeric_core('different_1234') is None
    assert format_class('team-v-team') == 'other'


def precheck():
    p = Path(G34_NAME)
    require(p.is_file() and not p.is_symlink(), 'script G02-G34 absent')
    require(file_sha(p) == G34_SHA256, 'empreinte G02-G34 modifiee; ne pas contourner')
    import g02g34_snapshot_train_capture_overlap_readonly as g34
    root = Path.cwd().resolve()
    _, train = g34.precheck(root)
    require(len(train) == 845, 'TRAIN attendus = 845')
    selftest()
    print('=== G02-G35/P1 | CONVENTIONS match_id | PRECHECK ===')
    print('GARDE : 845 IDs TRAIN | seules colonnes match_id d archives lues durant P2')
    print('AUCUNE NORMALISATION NE SERA UTILISEE POUR JOINDRE DES MATCHS')
    print('G02-G35/P1 PRECHECK PASS | 0 ARCHIVE PARCOURUE | 0 LABEL | 0 API | 0 ECRITURE | 0 FIT')
    return g34, root, train


def scan_id_only(g34, g33, root, cfg):
    rel, expected_prefix, id_idx, *_ = cfg
    path = g33.secure_file(root, rel)
    header = g33.read_one_csv_header(path)
    require(tuple(header[:len(expected_prefix)]) == expected_prefix,
            'schema archive change : ' + rel)
    ids = set()
    counts = Counter()
    with path.open('rb') as fh:
        head = fh.readline(65536)
        require(head.endswith(b'\n'), 'en-tete invalide')
        for line in fh:
            counts['rows'] += 1
            require(counts['rows'] <= g34.MAX_ROWS_PER_FILE, 'nombre lignes excessif')
            mid = g34.parse_csv_initial_fields(line, id_idx + 1)[id_idx].strip()
            require(mid, 'match_id vide')
            ids.add(mid)
            counts['formats_' + format_class(mid)] += 1
    return ids, counts


def audit(g34, root, train):
    import g02g33_snapshot_csv_schema_readonly as g33
    print('=== G02-G35/P2 | FORMATS IDs | PAS DE RESULTATS, PAS DE TIMESTAMPS ===')
    train_ids = set(train)
    train_format = Counter(format_class(m) for m in train_ids)
    train_numeric = {v for m in train_ids if (v := loose_numeric_core(m)) is not None}
    print('TRAIN IDs :', len(train_ids), '| formats :', dict(sorted(train_format.items())))
    print('TRAIN avec noyau numerique lisible :', sum(loose_numeric_core(m) is not None for m in train_ids))
    all_ids = set()
    for cfg in g34.SOURCES:
        rel = cfg[0]
        ids, stats = scan_id_only(g34, g33, root, cfg)
        all_ids.update(ids)
        exact = len(ids & train_ids)
        core = {v for m in ids if (v := loose_numeric_core(m)) is not None}
        approximate = len(core & train_numeric)
        formats = {k[len('formats_'):]: v for k, v in stats.items() if k.startswith('formats_')}
        print('\nSOURCE :', rel)
        print('  LIGNES :', stats['rows'], '| IDs distincts :', len(ids))
        print('  FORMATS :', dict(sorted(formats.items())))
        print('  INTERSECTION IDs EXACTS TRAIN :', exact)
        print('  NOYAUX NUMERIQUES COMMUNS (INDICE NON VERIFIE) :', approximate)
    print('\nTOTAL de-duplique :', len(all_ids), 'IDs d archives')
    print('INTERSECTION EXACTE TRAIN :', len(all_ids & train_ids), '/845')
    archive_numeric = {v for m in all_ids if (v := loose_numeric_core(m)) is not None}
    print('NOYAUX NUMERIQUES COMMUNS (NE PAS JOINDRE) :', len(archive_numeric & train_numeric))
    print('NOTE : meme des noyaux numeriques egaux ne prouvent PAS qu il s agit du meme match.')
    print('NOTE : un format partage ne prouve pas la provenance live ni la couverture temporelle.')
    print('AUCUN contenu hors match_id decode | AUCUNE valeur de resultat | 0 LABEL | 0 TIMELINE')
    print('0 API | 0 ECRITURE | 0 FIT | TEST outcome ferme | AS-OF NON CERTIFIE')
    print('G02-G35/P2 ID_NAMESPACE_DIAGNOSTIC COMPLETE | RESEARCH ONLY')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-ids', action='store_true', help='only match_id fields, no outcomes')
    args = ap.parse_args()
    g34, root, train = precheck()
    if args.audit_ids:
        audit(g34, root, train)
    else:
        print('Puis : python3 -B g02g35_snapshot_id_namespace_readonly.py --audit-ids')


if __name__ == '__main__':
    main()
