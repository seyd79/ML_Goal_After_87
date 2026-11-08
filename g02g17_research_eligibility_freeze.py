#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G17 | Pre-register research eligibility, 10 original features, NO labels.

Run from the ML_Goal_After_87 project root, with G02-G15 and G02-G16
frozen inputs and the two G02-G16 Python scripts in the same directory:
    python3 -B g02g17_research_eligibility_freeze.py           # read-only preflight
    python3 -B g02g17_research_eligibility_freeze.py --freeze  # new-only freeze
    python3 -B g02g17_research_eligibility_freeze.py --verify  # read-only recheck

ZERO API, ZERO labels/targets, ZERO training. Retrospective research ONLY.
Eligibility rules are predeclared here before any outcome access:
    - live_feature_status must equal FEATURE_CANDIDATE
    - both prematch averages present (>=10 admissible histories per team)
    - minutes_since_last_shot_on_target must be present
Never include REVIEW or impute anything in this initial 10-feature cohort.
Every match stays in the manifest even when ineligible; locked TEST partition
is never used for fitting or tuning. Not an authorization to train/deploy.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import g02g16_preflight_chrono_readonly as p1
import g02g16_p2_freeze_chrono as p2

OUT = Path('outputs/v4_research/v42_g02g17_eligibility_research_v1')
MANIFEST = OUT / 'G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
REPORT = OUT / 'G02G17_ELIGIBILITY_POLICY_FROZEN.json'
FIELDS = (
    'match_id', 'kickoff_utc', 'partition', 'research_eligibility',
    'exclusion_reasons', 'live_feature_status',
    'home_prematch_present', 'away_prematch_present', 'last_sot_present',
)
PARTITIONS = ('TRAIN', 'VALIDATION', 'TEST')
EXPECTED_ALL = {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}
EXPECTED_ELIGIBLE = {'TRAIN': 845, 'VALIDATION': 269, 'TEST': 286}
EXPECTED_RESEARCH_ELIGIBLE_TOTAL = 1400
EXPECTED_SPLIT_SHA256 = 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877'
EXPECTED_SPLIT_REPORT_SHA256 = '0ff5a497c0453ce89f12d426d9a8e571422e4c495742876660a770559bfb0dfe'


def stop(message):
    raise SystemExit('G02-G17 STOP : ' + str(message))


def check(condition, message):
    if not condition:
        stop(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    check(path.is_file(), 'fichier absent : ' + str(path))
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
              'en-tetes invalides : ' + str(path))
        rows = list(reader)
    check(all(None not in r for r in rows), 'colonnes en trop : ' + str(path))
    return rows


def canonical_csv(rows):
    buff = io.StringIO(newline='')
    writer = csv.DictWriter(buff, fieldnames=FIELDS, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buff.getvalue().encode('utf-8')


def evaluate_rows(data_rows, split_rows):
    by_id = {}
    for s in split_rows:
        mid = s.get('match_id', '')
        check(mid and mid not in by_id, 'doublon/ID vide dans split : ' + repr(mid))
        check(s.get('partition') in PARTITIONS, 'partition non reconnue : ' + repr(mid))
        by_id[mid] = s
    check(len(by_id) == len(data_rows), 'effectifs split/data differents')

    manifest = []
    total_part = Counter()
    eligible_part = Counter()
    reason_counts = Counter()
    overlap_counts = Counter()
    seen = set()
    for r in data_rows:
        mid = r.get('match_id', '')
        check(mid and mid not in seen, 'doublon/ID vide dataset : ' + repr(mid))
        seen.add(mid)
        s = by_id.get(mid)
        check(s is not None and s['kickoff_utc'] == r['kickoff_utc'],
              'match/kickoff non concordant : ' + mid)
        partition = s['partition']
        total_part[partition] += 1

        status = r['live_feature_status']
        check(status in ('FEATURE_CANDIDATE', 'REVIEW'), 'qualite inconnue : ' + mid)
        home = r['home_avg_goals_scored'] != ''
        away = r['away_avg_goals_scored'] != ''
        last = r['minutes_since_last_shot_on_target'] != ''
        reasons = []
        if status != 'FEATURE_CANDIDATE':
            reasons.append('LIVE_REVIEW')
        if not home or not away:
            reasons.append('PREMATCH_MISSING')
        if not last:
            reasons.append('LAST_SOT_MISSING')
        eligibility = 'RESEARCH_ELIGIBLE' if not reasons else 'NOT_ELIGIBLE'
        if not reasons:
            eligible_part[partition] += 1
        else:
            overlap_counts[len(reasons)] += 1
            for reason in reasons:
                reason_counts[reason] += 1
        manifest.append({
            'match_id': mid,
            'kickoff_utc': r['kickoff_utc'],
            'partition': partition,
            'research_eligibility': eligibility,
            'exclusion_reasons': ';'.join(reasons),
            'live_feature_status': status,
            'home_prematch_present': '1' if home else '0',
            'away_prematch_present': '1' if away else '0',
            'last_sot_present': '1' if last else '0',
        })
    check(set(by_id) == seen, 'les matchs split/data ne correspondent pas')
    check(len(manifest) == 1500, 'dataset doit contenir 1500 matchs')
    check(dict(total_part) == EXPECTED_ALL, 'effectifs chrono alteres : ' + str(dict(total_part)))
    check(dict(eligible_part) == EXPECTED_ELIGIBLE,
          'effectifs admissibilite inattendus : ' + str(dict(eligible_part)))
    check(sum(eligible_part.values()) == EXPECTED_RESEARCH_ELIGIBLE_TOTAL,
          'candidats research != 1400')
    check(reason_counts['LIVE_REVIEW'] == 62, 'REVIEW !=62')
    check(reason_counts['PREMATCH_MISSING'] == 44, 'sans deux moyennes !=44')
    check(reason_counts['LAST_SOT_MISSING'] == 5, 'dernier SOT vide !=5')
    check(sum(overlap_counts.values()) == 100, 'cohorte non eligible != 100')
    check(sum(count * n for count, n in overlap_counts.items()) == sum(reason_counts.values()),
          'incoherence intersections des motifs')
    return manifest, total_part, eligible_part, reason_counts, overlap_counts


def build_artifacts():
    # Reverify every frozen feature, observation, and split input, plus chronology.
    # Both checks are read-only. No labels, no API or model packages are touched.
    raw_csv, raw_report = p2.build_artifacts()
    check(p2.SPLIT.is_file() and p2.REPORT.is_file(), 'gel G02-G16/P2 absent')
    check(sha256(p2.SPLIT.read_bytes()) == EXPECTED_SPLIT_SHA256, 'SHA split different')
    check(sha256(p2.REPORT.read_bytes()) == EXPECTED_SPLIT_REPORT_SHA256, 'SHA rapport split different')
    check(p2.SPLIT.read_bytes() == raw_csv and p2.REPORT.read_bytes() == raw_report,
          'gel split diverge du calcul independant')

    data = read_csv(p1.DATA)
    split = read_csv(p2.SPLIT)
    check(len(data) == 1500 and len(split) == 1500, 'nombre de matchs different')
    manifest, totals, selected, reasons, overlaps = evaluate_rows(data, split)
    payload = canonical_csv(manifest)
    report = {
        'stage': 'G02-G17',
        'status': 'RESEARCH_ELIGIBILITY_FROZEN_ONLY_NOT_TRAINING_APPROVAL',
        'source_sha256': {p.name: h for p, h in p1.EXPECTED.items()},
        'chrono_split_sha256': EXPECTED_SPLIT_SHA256,
        'chrono_report_sha256': EXPECTED_SPLIT_REPORT_SHA256,
        'policy_pre_registered_without_labels': {
            'version': 'v1_COMPLETE_CASE_10F',
            'live_quality': 'FEATURE_CANDIDATE_ONLY',
            'prematch': 'BOTH_AVERAGES_PRESENT_ONLY',
            'last_sot': 'PRESENT_ONLY',
            'imputation': 'NONE',
            'feature_count': 10,
            'partitions': 'REUSE_FROZEN_G02G16_WITHOUT_REASSIGNMENT',
            'exclusion_reasons_order': ['LIVE_REVIEW', 'PREMATCH_MISSING', 'LAST_SOT_MISSING'],
            'no_outcome_based_selection': True,
        },
        'total_matches': len(manifest),
        'research_eligible_matches': sum(selected.values()),
        'not_eligible_matches': len(manifest) - sum(selected.values()),
        'partition_totals': {p: totals[p] for p in PARTITIONS},
        'partition_research_eligible': {p: selected[p] for p in PARTITIONS},
        'partition_not_eligible': {p: totals[p] - selected[p] for p in PARTITIONS},
        'exclusion_reason_incidence_nonexclusive': dict(sorted(reasons.items())),
        'excluded_match_reason_overlap': {str(k): v for k, v in sorted(overlaps.items())},
        'eligibility_manifest_sha256': sha256(payload),
        'labels_read': False,
        'labels_included': False,
        'api_calls': 0,
        'training_performed': False,
        'test_outcomes_consulted': False,
        'asof_certified': False,
        'live_publication_timing_certified': False,
        'restrictions': [
            'Research-only retrospective features; no historical as-of certification.',
            'Frozen TEST partition is reserved for one independent future evaluation.',
            'This stage does not authorize training, hyperparameter selection, or live deployment.',
            'No exclusion may be revised using labels or TEST outcomes.',
        ],
    }
    report_bytes = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    print('=== G02-G17 | ELIGIBILITE RECHERCHE SANS LABELS ===')
    for p in PARTITIONS:
        print(p, '| total', totals[p], '| eligibles', selected[p], '| exclus', totals[p] - selected[p])
    print('MOTIFS D EXCLUSION (non exclusifs) :', dict(sorted(reasons.items())))
    print('MATCHS EXCLUS DISTINCTS :', len(manifest) - sum(selected.values()))
    print('SHA MANIFESTE :', sha256(payload))
    print('SHA RAPPORT :', sha256(report_bytes))
    print('AS-OF NON CERTIFIE | 0 API | 0 LABEL | 0 TRAIN')
    return payload, report_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true', help='creer nouveaux fichiers sans ecraser')
    mode.add_argument('--verify', action='store_true', help='verifier exactement les fichiers geles')
    args = parser.parse_args()
    raw_csv, raw_report = build_artifacts()
    if args.verify:
        for path, expected in ((MANIFEST, raw_csv), (REPORT, raw_report)):
            check(path.is_file(), 'sortie gelee absente : ' + str(path))
            check(path.read_bytes() == expected,
                  'sortie gelee differente du calcul : ' + str(path))
            print('VERIFY SHA OK :', path.name, sha256(expected))
        print('G02-G17 VERIFY PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
        return
    if not args.freeze:
        print('G02-G17 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TRAIN')
        print('Pour figer : python3 -B g02g17_research_eligibility_freeze.py --freeze')
        return
    check(not OUT.exists(), 'dossier de gel existe deja, refus ecrasement : ' + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    for path, expected in ((MANIFEST, raw_csv), (REPORT, raw_report)):
        with path.open('xb') as f:
            f.write(expected)
        check(path.read_bytes() == expected, 'ecriture non conforme : ' + str(path))
        print('GEL :', path, '| SHA256', sha256(expected))
    print('G02-G17 FREEZE PASS | 1500 MATCHS | 1400 RESEARCH ELIGIBLE | 0 API | 0 LABEL | 0 TRAIN')
    print('IMPORTANT : NO-GO TRAINING ET LIVE DEPLOYMENT | AS-OF NON CERTIFIE')


if __name__ == '__main__':
    main()
