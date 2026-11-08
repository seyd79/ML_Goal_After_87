#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P1 -- audit d'autorisations et inventaire des sources de labels.

Usage depuis la racine ML_Goal_After_87 :
    python3 -B g02g20_p1_label_source_inventory_readonly.py

Zero API, zero ecriture, zero ouverture de labels / resultats / timelines,
zero entrainement. Les fichiers de source possibles ne sont decouverts
QUE par leur NOM. Ce controle ne certifie aucune provenance de label.
Python >= 3.9, bibliotheque standard uniquement.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('outputs/v4_research')
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
ALLOW = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
REPORT = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'
EXPECTED = {
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    REPORT: '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
}
TOTALS = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
ELIGIBLE = {'TRAIN': 845, 'VALIDATION': 269, 'TEST': 286}
RAW_DIRS = [
    BASE / 'v41_mass350/raw_timelines',
    BASE / 'v42_g02d2_pilot5/raw_timelines',
    BASE / 'v42_g02d3_coverage24/raw_timelines',
    BASE / 'v42_g02d5_collection/raw_timelines',
]
# Ne jamais rechercher a l'interieur des caches ou timelines d'une equipe.
SKIP_DIRS = {'raw_timelines', 'raw_pages', 'rejected_timelines',
             'quarantine', 'raw_shotmaps', '__pycache__', '.venv'}
KEYWORDS = ('label', 'target', 'outcome', 'ground_truth', 'goal_after', 'result')


def check(ok, msg):
    if not ok:
        raise SystemExit('G02-G20/P1 STOP : ' + msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(text):
    try:
        t = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        raise SystemExit('G02-G20/P1 STOP : date UTC invalide')
    check(t.tzinfo is not None, 'date sans fuseau UTC')
    return t.astimezone(timezone.utc)


def rows(path, expected_fields):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        check(fields is not None and len(fields) == len(set(fields)),
              'entetes invalides dans ' + path.name)
        check(set(expected_fields) <= set(fields), 'colonnes absentes dans ' + path.name)
        # CSV autorise est volontairement strict: pas de source de label possible.
        if path == ALLOW:
            check(tuple(fields) == ('match_id', 'kickoff_utc', 'partition'),
                  'colonnes interdites dans la liste d autorisations')
        result = list(reader)
    check(all(None not in r for r in result), 'ligne trop longue : ' + path.name)
    check(all(r.get('match_id') for r in result), 'match_id vide : ' + path.name)
    ids = [r['match_id'] for r in result]
    check(len(ids) == len(set(ids)), 'IDs dupliques : ' + path.name)
    return {r['match_id']: r for r in result}


def discover_names_only(root):
    """Inventaire par nom/chemin, SANS ouvrir aucun fichier candidat."""
    found = []
    checked = 0
    for where, folders, files in os.walk(str(root), followlinks=False):
        folders[:] = sorted(d for d in folders if d not in SKIP_DIRS and
                            not (Path(where) / d).is_symlink())
        for name in sorted(files):
            checked += 1
            if checked > 100000:
                raise SystemExit('G02-G20/P1 STOP : inventaire trop volumineux')
            if name.startswith('.') or not name.lower().endswith(('.csv', '.json', '.jsonl', '.parquet')):
                continue
            if any(k in name.lower() for k in KEYWORDS):
                path = Path(where) / name
                if not path.is_symlink():
                    found.append(path)
    return sorted(found)


def count_directory_names_only(folder):
    if not folder.is_dir():
        return 'ABSENT'
    # noms seulement ; aucun JSON brut charge, aucun resultat lu
    return sum(1 for p in folder.iterdir() if p.is_file() and not p.is_symlink()
               and p.name.startswith('mt_') and p.suffix == '.json')


def main():
    print('=== G02-G20/P1 | GARDE LABELS + INVENTAIRE SOURCES | READ ONLY ===')
    for path, expected in EXPECTED.items():
        check(path.is_file(), 'fichier gele absent : ' + str(path))
        actual = sha(path)
        check(actual == expected, 'SHA256 different : ' + path.name + ' | ' + actual)
        print('SHA OK :', path.name, actual)
    policy = json.loads(REPORT.read_text(encoding='utf-8'))
    check(policy.get('train_only_allowlist_sha256') == EXPECTED[ALLOW],
          'SHA autorisations incoherent avec G02-G19')
    check(policy.get('labels_accessed_in_this_stage') is False and
          policy.get('test_labels_or_outcomes_accessed_in_this_stage') is False and
          policy.get('historical_asof_certified') is False,
          'rapport G02-G19 incompatible')

    split = rows(SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = rows(ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    allowed = rows(ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    check(len(split) == len(elig) == 1500 and set(split) == set(elig),
          'effectifs/jointures des 1500 differents')
    check(len(allowed) == 1114, 'liste autorisee != 1114')
    check(set(allowed) <= set(split), 'match autorise absent du split')
    counts = Counter()
    admiss = Counter()
    expected_allowed = set()
    test_ids = set()
    for mid, s in split.items():
        e = elig[mid]
        part = s['partition']
        check(part in TOTALS and e['partition'] == part,
              'partition incoherente : ' + mid)
        check(s['kickoff_utc'] == e['kickoff_utc'], 'date incoherente : ' + mid)
        check(utc(s['kickoff_utc']) == utc(e['kickoff_utc']), 'heure UTC differente : ' + mid)
        counts[part] += 1
        status = e['research_eligibility']
        check(status in ('RESEARCH_ELIGIBLE', 'NOT_ELIGIBLE'),
              'statut inattendu : ' + mid)
        if part == 'TEST':
            test_ids.add(mid)
        if status == 'RESEARCH_ELIGIBLE':
            admiss[part] += 1
            if part in ('TRAIN', 'VALIDATION'):
                expected_allowed.add(mid)
    check(dict(counts) == TOTALS, 'partition totale incorrecte : ' + repr(counts))
    check(dict(admiss) == ELIGIBLE, 'admissibilite incorrecte : ' + repr(admiss))
    check(set(allowed) == expected_allowed, 'autorisation non equivalente a TRAIN+VALIDATION')
    check(not (set(allowed) & test_ids), 'TEST PRESENT DANS LA LISTE AUTORISEE')
    check(dict(Counter(r['partition'] for r in allowed.values())) ==
          {'TRAIN': 845, 'VALIDATION': 269}, 'comptages autorisations invalides')
    for mid, r in allowed.items():
        check(r['partition'] == split[mid]['partition'] and
              utc(r['kickoff_utc']) == utc(split[mid]['kickoff_utc']),
              'date ou partition autorisation invalide : ' + mid)

    print('TOTAL :', dict(counts))
    print('ELIGIBLES :', dict(admiss))
    print('ALLOWLIST : 1114 MATCHS | TRAIN 845 | VALIDATION 269 | TEST 0')
    print('TEST PROTEGE : 302 IDs hors liste autorisee dont 286 eligibles')
    print('=== INVENTAIRE DES NOMS (AUCUN CONTENU DE LABEL LU) ===')
    print('REPERTOIRES TIMELINES (noms des fichiers uniquement) :')
    for d in RAW_DIRS:
        print('  ', str(d), '| fichiers mt_*.json :', count_directory_names_only(d))
    print('FICHIERS DONT LE NOM SUGGERE DES LABELS/CIBLES/RESULTATS :')
    candidates = discover_names_only(BASE) if BASE.is_dir() else []
    print('  NOMBRE DE CANDIDATS :', len(candidates))
    for p in candidates[:45]:
        print('  CANDIDAT NOM SEULEMENT :', str(p))
    if len(candidates) > 45:
        print('  ...', len(candidates) - 45, 'autres noms omis')
    print('IMPORTANT : aucun candidat de cette liste n est encore certifie source de labels')
    print('IMPORTANT : si les labels sont dans un CSV commun TRAIN/VAL/TEST,')
    print('            ne PAS le lire entierement pendant le developpement.')
    print('AS-OF NON CERTIFIE | 0 API | 0 ECRITURE | 0 LABEL OUVERT | 0 MODEL')
    print('G02-G20/P1 PREFLIGHT PASS | SOURCE DE LABELS A IDENTIFIER | TEST FERME')


if __name__ == '__main__':
    main()
