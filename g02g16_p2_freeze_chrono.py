
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G16/P2 — Freeze chronological match partitions, research-only.

From ML_Goal_After_87 project root:
    python3 -B g02g16_p2_freeze_chrono.py           # dry run; zero writes
    python3 -B g02g16_p2_freeze_chrono.py --freeze  # new-only partition files
    python3 -B g02g16_p2_freeze_chrono.py --verify  # read-only verification of frozen output

Uses G02-G16/P1's audited, frozen 1500-row data and exactly the same
UTC-day-aware 60/20/20 nearest-count method. NO labels, targets or outcomes
are read; no API calls, model fitting, filtering or quality-based selection.
NEVER interpret the splits as temporally certified: historical data were
captured retrospectively and publication timing is unverified.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import g02g16_preflight_chrono_readonly as pre

OUT = Path('outputs/v4_research/v42_g02g16_chrono_split_research_v1')
SPLIT = OUT / 'G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
REPORT = OUT / 'G02G16_CHRONO_SPLIT_REPORT.json'
FIELDS = ('match_id', 'kickoff_utc', 'utc_day', 'partition')
PARTITIONS = ('TRAIN', 'VALIDATION', 'TEST')
EXPECTED_COUNTS = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
EXPECTED_DATE_RANGES = {
    'TRAIN': ('2026-03-01', '2026-05-09'),
    'VALIDATION': ('2026-05-10', '2026-07-17'),
    'TEST': ('2026-07-18', '2026-08-19'),
}

def stop(msg):
    raise SystemExit('G02-G16/P2 STOP : ' + msg)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical_csv(rows):
    b = io.StringIO(newline='')
    w = csv.DictWriter(b, fieldnames=FIELDS, lineterminator='\n', extrasaction='raise')
    w.writeheader()
    w.writerows(rows)
    return b.getvalue().encode('utf-8')

def make_plan(data):
    # Explicitly disregard every column except match id and kickoff.
    days = defaultdict(list)
    seen = set()
    for row in data:
        mid = row['match_id']
        if not mid or mid in seen:
            stop('ID vide/duplique : ' + repr(mid))
        seen.add(mid)
        t = pre.utc(row['kickoff_utc'])
        days[t.date().isoformat()].append((t, mid, row['kickoff_utc']))
    if len(data) != 1500 or len(days) < 3:
        stop('nombre de matchs ou jours invalide')

    ordered_days = sorted(days)
    prefix = [0]
    for day in ordered_days:
        prefix.append(prefix[-1] + len(days[day]))

    # EXACT same tie-breaking as G02-G16/P1; only UTC-day boundaries.
    best = None
    for i in range(1, len(ordered_days) - 1):
        for j in range(i + 1, len(ordered_days)):
            candidate = (abs(prefix[i] - 900) + abs(prefix[j] - 1200), i, j)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        stop('partition impossible')
    _, i, j = best
    assigned_days = {
        'TRAIN': ordered_days[:i],
        'VALIDATION': ordered_days[i:j],
        'TEST': ordered_days[j:],
    }

    entries = []
    info = {}
    assigned = set()
    for name in PARTITIONS:
        block = assigned_days[name]
        if not block:
            stop('partition vide : ' + name)
        mids = set()
        for day in block:
            if day in assigned:
                stop('jour UTC partage : ' + day)
            assigned.add(day)
            for dt, mid, raw in sorted(days[day], key=lambda x: (x[0], x[1])):
                entries.append({
                    'match_id': mid,
                    'kickoff_utc': raw,
                    'utc_day': day,
                    'partition': name,
                })
                mids.add(mid)
        info[name] = {
            'matches': len(mids),
            'first_utc_day': block[0],
            'last_utc_day': block[-1],
            'utc_days': len(block),
        }
        if len(mids) != EXPECTED_COUNTS[name]:
            stop('nombre inattendu pour ' + name + ' : ' + str(len(mids)))
        dates = (block[0], block[-1])
        if dates != EXPECTED_DATE_RANGES[name]:
            stop('dates inattendues pour ' + name + ' : ' + str(dates))
    if len(entries) != 1500 or len(assigned) != len(ordered_days):
        stop('partition incomplète')
    if len(set(x['match_id'] for x in entries)) != 1500:
        stop('match duplique dans les partitions')
    return entries, info

def build_artifacts():
    # Re-audit the original 3 files and every relationship first.
    pre.main()
    data, _ = pre.readcsv(pre.DATA)
    entries, info = make_plan(data)
    csv_bytes = canonical_csv(entries)
    counts = dict(Counter(x['partition'] for x in entries))
    if counts != EXPECTED_COUNTS:
        stop('comptages du CSV differents : ' + repr(counts))
    metadata = {
        'stage': 'G02-G16/P2',
        'mode': 'RESEARCH_RETROSPECTIVE_NOT_ASOF_CERTIFIED',
        'policy': 'NEAREST_60_20_20_COUNTS_WITH_WHOLE_UTC_DAYS',
        'tie_break': 'EARLIEST_UTC_DAY_BOUNDARIES',
        'target_counts': {'train': 900, 'validation_cumulative': 1200},
        'source_sha256': {path.name: checksum for path, checksum in pre.EXPECTED.items()},
        'matches': 1500,
        'partitions': info,
        'partition_csv_sha256': digest(csv_bytes),
        'labels_read': False,
        'labels_included': False,
        'api_calls': 0,
        'training_performed': False,
        'training_eligibility_selected': False,
        'quality_used_to_select_split': False,
        'split_frozen_only_not_training_authorization': True,
        'prematch_asof_certified': False,
        'live_publication_timing_certified': False,
        'test_outcomes_consulted': False,
    }
    report_bytes = (json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    print('=== G02-G16/P2 | PLAN CHRONOLOGIQUE = PARTITIONS DE MATCHS ===')
    for name in PARTITIONS:
        r = info[name]
        print(name, '|', r['matches'], 'matchs |', r['first_utc_day'], 'a', r['last_utc_day'], '| jours UTC', r['utc_days'])
    print('SOURCE G02-G15 SHA :', pre.EXPECTED[pre.DATA])
    print('SHA SPLIT :', digest(csv_bytes))
    print('SHA RAPPORT :', digest(report_bytes))
    print('AVERTISSEMENT : cohortes NON filtrees, labels NON ouverts, as-of NON certifie')
    return csv_bytes, report_bytes

def main():
    parser = argparse.ArgumentParser(description='G02-G16/P2 partitions chrono pour recherche, sans labels')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true', help='ecrire de nouveaux fichiers sans ecraser')
    mode.add_argument('--verify', action='store_true', help='verifier les fichiers figes sans ecriture')
    args = parser.parse_args()
    raw_csv, raw_report = build_artifacts()
    if args.verify:
        for path, expected in ((SPLIT, raw_csv), (REPORT, raw_report)):
            if not path.is_file():
                stop('fichier absent : ' + str(path))
            if path.read_bytes() != expected:
                stop('fichier ne correspond plus au calcul : ' + str(path))
            print('VERIFICATION OK :', path, '| SHA256', digest(expected))
        print('G02-G16/P2 VERIFY PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
        return
    if not args.freeze:
        print('G02-G16/P2 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
        print('Pour figer : python3 -B g02g16_p2_freeze_chrono.py --freeze')
        return
    if OUT.exists():
        stop('dossier de gel existe deja, aucun ecrasement : ' + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    SPLIT.write_bytes(raw_csv)
    REPORT.write_bytes(raw_report)
    for path, expected in ((SPLIT, raw_csv), (REPORT, raw_report)):
        if path.read_bytes() != expected:
            stop('verification apres ecriture impossible : ' + str(path))
        print('GEL :', path, '| SHA256', digest(expected))
    print('G02-G16/P2 FREEZE PASS | 1500 MATCHS | 0 API | 0 LABEL | 0 TRAIN')
    print('IMPORTANT : AS-OF NON CERTIFIE ; PAS DE GO ENTRAINEMENT')

if __name__ == '__main__':
    main()
0
