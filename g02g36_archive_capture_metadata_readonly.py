#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G36 | Audit des captures d'archives comme cohorte DISTINCTE, sans labels.

Depuis la racine ML_Goal_After_87 :
    python3 -B g02g36_archive_capture_metadata_readonly.py
    python3 -B g02g36_archive_capture_metadata_readonly.py --audit-archives

Garde : scripts et artefacts G02-G35 verifies; aucun fichier modifie.
--audit-archives lit uniquement le prefixe de chaque enregistrement CSV :
  - historiques V3C : snapshot_timestamp_utc, match_id, minute (et les
    champs de metadata intermediaires necessaires pour atteindre minute)
  - prospectif V4 : match_id, kickoff_utc, captured_at_utc, snapshot_minute.
Aucun resultat, label, prediction, score, feature ou timeline n'est interprete.
Les lignes sont lues en octets bruts et tronquees AVANT les colonnes interdites.

Ne certifie NI l'origine reelle des horodatages, NI le kickoff de V3C,
NI la disponibilite des evenements en temps reel, NI l'independance
contre les partitions VALIDATION/TEST de V4.2.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import statistics
from pathlib import Path

G35_NAME = 'g02g35_snapshot_id_namespace_readonly.py'
G35_SHA256 = '71e0bd67a861efe9eff6e24cc3a1f7944102e95d60ffa2083a9088fb5e63471b'


def require(condition, message):
    if not condition:
        raise SystemExit('G02-G36 STOP : ' + message)


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def selftest(g34):
    # An intentionally invalid UTF8 byte is situated AFTER the authorized
    # prefix.  The parser must not touch it (or any outcome column).
    historical = (b'2026-03-01T19:30:00Z,v3,mt_12345,1,2,Home,Away,86,'
                  b'bad_prediction,\xff\xfe,unsafe_label\n')
    fields = g34.parse_csv_initial_fields(historical, 8)
    assert len(fields) == 8
    assert fields[0] == '2026-03-01T19:30:00Z'
    assert fields[2] == 'mt_12345' and fields[7] == '86'
    prospective = (b'mt_99,2026-03-01T17:00:00Z,'
                   b'2026-03-01T19:30:00Z,86,\xff\xfe\n')
    fields = g34.parse_csv_initial_fields(prospective, 4)
    assert len(fields) == 4 and fields[0] == 'mt_99' and fields[3] == '86'
    assert minute_class(79.99) == 'before80'
    assert minute_class(80) == '80_to_under87'
    assert minute_class(86.99) == '80_to_under87'
    assert minute_class(87) == 'exactly87'
    assert minute_class(87.01) == 'after87'


def minute_class(minute):
    if minute < 80:
        return 'before80'
    if minute < 87:
        return '80_to_under87'
    if minute == 87:
        return 'exactly87'
    return 'after87'


def precheck():
    path = Path(G35_NAME)
    require(path.is_file() and not path.is_symlink(), 'script G02-G35 absent')
    require(file_sha(path) == G35_SHA256, 'empreinte G02-G35 modifiee')
    import g02g35_snapshot_id_namespace_readonly as g35
    g34, root, train = g35.precheck()
    selftest(g34)
    print('=== G02-G36/P1 | METADONNEES ARCHIVES | PRECHECK ===')
    print('SHA G02-G35 : OK | SELFTEST ISO8601/CSV/BORNE 87 : PASS')
    print('GARDE : 4 archives et schemas fixes | TRAIN 845 pour garde d intersection seulement')
    print('GARDE : aucune ligne de score/label/prediction/feature ne sera decodee')
    print('G02-G36/P1 PRECHECK PASS | 0 ARCHIVE PARCOURUE | 0 LABEL | 0 API | 0 ECRITURE | 0 FIT')
    return g34, root, set(train)


def scan_archive(g34, g33, root, config):
    rel, expected_prefix, id_idx, timestamp_idx, minute_idx, kickoff_idx = config
    path = g33.secure_file(root, rel)
    header = g33.read_one_csv_header(path)
    require(tuple(header[:len(expected_prefix)]) == expected_prefix,
            'colonnes initiales differentes : ' + rel)
    take = 1 + max(id_idx, timestamp_idx, minute_idx,
                   kickoff_idx if kickoff_idx is not None else 0)
    require(take <= len(expected_prefix), 'prefixe non autorise')
    counts = Counter()
    ids = set()
    ids_80_to_87 = set()
    ids_87 = set()
    ids_valid_date = set()
    first_timestamp = None
    last_timestamp = None
    per_match = defaultdict(list)  # metadata only (timestamp, minute)
    unique_samples = set()  # only match_id, timestamp, minute
    with path.open('rb') as fh:
        header_line = fh.readline(65536)
        require(header_line.endswith(b'\n'), 'en-tete non termine : ' + rel)
        for raw in fh:
            counts['rows'] += 1
            require(counts['rows'] <= g34.MAX_ROWS_PER_FILE,
                    'limite lignes depassee : ' + rel)
            fields = g34.parse_csv_initial_fields(raw, take)
            mid = fields[id_idx].strip()
            require(mid, 'match_id vide dans ' + rel)
            ids.add(mid)
            ts = g34.parse_utc(fields[timestamp_idx])
            minute = g34.parse_minute(fields[minute_idx])
            if ts is None:
                counts['invalid_or_naive_timestamp'] += 1
            else:
                counts['timestamp_valid'] += 1
                ids_valid_date.add(mid)
                if first_timestamp is None or ts < first_timestamp:
                    first_timestamp = ts
                if last_timestamp is None or ts > last_timestamp:
                    last_timestamp = ts
            if minute is None:
                counts['invalid_minute'] += 1
            else:
                counts['minute_valid'] += 1
                counts['minute_' + minute_class(minute)] += 1
            if kickoff_idx is not None:
                ko = g34.parse_utc(fields[kickoff_idx])
                if ko is None:
                    counts['prospective_kickoff_invalid'] += 1
                elif ts is not None:
                    elapsed = (ts - ko).total_seconds() / 60
                    if elapsed < 0:
                        counts['prospective_capture_before_kickoff'] += 1
                    elif elapsed <= 240:
                        counts['prospective_capture_within_4h_after_kickoff'] += 1
                    else:
                        counts['prospective_capture_over_4h_after_kickoff'] += 1
            # Do not label a minute-only record as genuine capture evidence.
            if ts is not None and minute is not None:
                per_match[mid].append((ts, minute))
                unique_samples.add((mid, ts, minute))
                if 80 <= minute < 87:
                    ids_80_to_87.add(mid)
                elif minute == 87:
                    ids_87.add(mid)
    counts['unique_timestamp_minute_triplets'] = len(unique_samples)
    counts['repeated_identical_triplets'] = sum(len(v) for v in per_match.values()) - len(unique_samples)
    # A decrease in reported match minute across INCREASING capture times
    # merits review, but is not sufficient evidence of any specific cause.
    ids_2plus = 0
    ids_backward = set()
    for mid, samples in per_match.items():
        if len(samples) >= 2:
            ids_2plus += 1
        sorted_samples = sorted(set(samples))
        if any(t2 > t1 and minute2 < minute1 for
               (t1, minute1), (t2, minute2) in zip(sorted_samples, sorted_samples[1:])):
            ids_backward.add(mid)
    counts['matches_2plus_valid_captures'] = ids_2plus
    counts['matches_backward_minute_in_timestamp_order'] = len(ids_backward)
    return {
        'rel': rel, 'counts': counts, 'ids': ids,
        'ids_80_to_87': ids_80_to_87,
        'ids_87': ids_87,
        'ids_valid_date': ids_valid_date,
        'earliest': first_timestamp,
        'latest': last_timestamp,
    }


def audit(g34, root, train_ids):
    import g02g33_snapshot_csv_schema_readonly as g33
    print('=== G02-G36/P2 | COHORTE ARCHIVES | METADONNEES SEULEMENT ===')
    union = set()
    all_80_87 = set()
    all_87 = set()
    total_rows = 0
    for config in g34.SOURCES:
        result = scan_archive(g34, g33, root, config)
        c = result['counts']
        union.update(result['ids'])
        all_80_87.update(result['ids_80_to_87'])
        all_87.update(result['ids_87'])
        total_rows += c['rows']
        print('\nSOURCE :', result['rel'])
        print('  LIGNES :', c['rows'], '| IDs uniques :', len(result['ids']))
        print('  DATE UTC VALIDE :', c['timestamp_valid'],
              '| DATE INVALIDE :', c['invalid_or_naive_timestamp'])
        print('  MINUTE VALIDE :', c['minute_valid'],
              '| MINUTE INVALIDE :', c['invalid_minute'])
        print('  MINUTES : <80', c['minute_before80'],
              '| [80,87[', c['minute_80_to_under87'],
              '| =87', c['minute_exactly87'],
              '| >87', c['minute_after87'])
        print('  IDs avec timestamp+minute [80,87[ :', len(result['ids_80_to_87']))
        print('  IDs avec timestamp+minute exactement 87 :', len(result['ids_87']))
        print('  >=2 captures datees par ID :', c['matches_2plus_valid_captures'])
        print('  Triplets identiques repetes :', c['repeated_identical_triplets'])
        print('  IDs minute decroissante selon horodatage :',
              c['matches_backward_minute_in_timestamp_order'])
        print('  PREMIERE DATE de capture declaree :', result['earliest'].isoformat()
              if result['earliest'] else 'AUCUNE')
        print('  DERNIERE DATE de capture declaree :', result['latest'].isoformat()
              if result['latest'] else 'AUCUNE')
        if config[5] is not None:
            print('  KICKOFF V4 (seul fichier qui possede le champ) :'
                  ' invalides', c['prospective_kickoff_invalid'],
                  '| avant', c['prospective_capture_before_kickoff'],
                  '| 0-4h apres', c['prospective_capture_within_4h_after_kickoff'],
                  '| >4h apres', c['prospective_capture_over_4h_after_kickoff'])
    print('\n=== SYNTHESE DEDUPLIQUEE ARCHIVES ===')
    print('LIGNES PHYSIQUES :', total_rows)
    print('IDs ARCHIVES DISTINCTS :', len(union))
    print('IDs avec >=1 (timestamp UTC parseable + minute [80,87[) :', len(all_80_87))
    print('IDs avec >=1 (timestamp UTC parseable + minute exactement 87) :', len(all_87))
    print('CORRESPONDANCE TRAIN V4.2 :', len(union & train_ids), '/845')
    print('IMPORTANT : les archives historiques n ont PAS kickoff_utc; impossible de verifier'
          ' que le timestamp tombe dans la fenetre reelle du match pour ces archives.')
    print('IMPORTANT : minutes [80,87[ et timestamps parseables ne prouvent pas la'
          ' capture genuine ni la disponibilite des features avant 87.')
    print('IMPORTANT : independance par rapport a VALIDATION/TEST non examinee.')
    print('AUCUN resultat/label/prediction/score/feature interprete | 0 TIMELINE')
    print('0 API | 0 ECRITURE | 0 FIT | TEST outcome ferme')
    print('G02-G36/P2 ARCHIVE_CAPTURE_METADATA_AUDIT COMPLETE | RESEARCH ONLY')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-archives', action='store_true',
                        help='analyser timestamps et minutes des snapshots, sans outcomes')
    args = parser.parse_args()
    g34, root, train_ids = precheck()
    if args.audit_archives:
        audit(g34, root, train_ids)
    else:
        print('Puis : python3 -B g02g36_archive_capture_metadata_readonly.py --audit-archives')


if __name__ == '__main__':
    main()
