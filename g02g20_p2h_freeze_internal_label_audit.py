#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2H: freeze reproducible evidence of TRAIN+VALIDATION label consistency.

Usage from ML_Goal_After_87 root, with the unchanged P2G audit script alongside:
    python3 -B g02g20_p2h_freeze_internal_label_audit.py
    python3 -B g02g20_p2h_freeze_internal_label_audit.py --freeze
    python3 -B g02g20_p2h_freeze_internal_label_audit.py --verify

PRECHECK/VERIFY: 0 writes. FREEZE: creates only one new JSON file.
Reads only authorized TRAIN 845 and VALIDATION 269 outcome records and timelines.
For non-allowlisted CSV rows only the match ID prefix is decoded by the P2G
routine; TEST outcome values are never parsed and TEST timelines never opened.
An internal consistency report is NOT independent source certification or
historical as-of certification, and does NOT authorize training or betting.
No API, no model fitting. Python 3.9+ stdlib.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

AUDITOR_NAME = 'g02g20_p2g_authorized_label_reconcile_readonly.py'
AUDITOR_SHA256 = '71b86098dbb7f12725fa41a08413a3893c31f98f8d89449f75aea57fa0408cc6'
OUT = Path('outputs/v4_research/v42_g02g20_p2h_internal_label_reconciliation_v1')
REPORT = OUT / 'G02G20_AUTHORIZED_LABEL_RECONCILIATION_FROZEN.json'


def require(ok, why):
    if not ok:
        raise SystemExit('G02-G20/P2H STOP : ' + why)


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for buf in iter(lambda: fh.read(128 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def sha_record(h, values):
    h.update((json.dumps(values, ensure_ascii=True, separators=(',', ':')) + '\n').encode('ascii'))


def check_and_import():
    source = Path.cwd() / AUDITOR_NAME
    require(source.is_file() and not source.is_symlink(), 'script P2G introuvable')
    actual = sha_file(source)
    require(actual == AUDITOR_SHA256, 'script P2G modifie; SHA actuel=' + actual)
    import g02g20_p2g_authorized_label_reconcile_readonly as p2g
    require(Path(p2g.__file__).resolve() == source.resolve(), 'module P2G importe ailleurs')
    print('SHA AUDITEUR P2G : OK', actual)
    return p2g


def tally(allow, labels, timelines, p2g):
    """All operations with outcome values are gated to authorized match IDs."""
    require(set(allow) == set(labels) and len(allow) == 1114,
            'ensemble TRAIN+VALIDATION non conforme')
    by_part = {'TRAIN': Counter(), 'VALIDATION': Counter()}
    label_proof = hashlib.sha256()
    timeline_proof = hashlib.sha256()
    differences = Counter()
    fields = ('goal_after_87', 'goal_at_87', 'extra_time_goals_excluded')
    for mid in sorted(allow):
        part = allow[mid]['partition']
        require(part in by_part, 'identifiant TEST/part non autorise : ' + mid)
        require(mid in timelines, 'timeline manquante : ' + mid)
        r = labels[mid]
        observed = p2g.outcome_from_timeline(timelines[mid], mid)
        c = by_part[part]
        c['MATCHS'] += 1
        c['POSITIFS'] += observed['goal_after_87']
        c['GOAL_AT_87_MATCHES'] += observed['goal_at_87']
        c['EXTRA_TIME_GOALS_EXCLUDED'] += observed['extra_time_goals_excluded']
        c['UNKNOWN_PERIOD_EVENTS'] += observed['unknown_periods']
        c['MALFORMED_EVENTS'] += observed['invalid_events']
        normalized = {'match_id': mid, 'partition': part}
        for fld in fields:
            val = p2g.parse_count(r[fld], fld)
            if fld != 'extra_time_goals_excluded':
                require(val in (0, 1), 'label binaire invalide : ' + mid)
            normalized[fld] = val
            if observed[fld] != val:
                differences[fld] += 1
        cert = p2g.parse_binary(r['label_certified_independently'],
                                'label_certified_independently')
        normalized['label_certified_independently'] = cert
        c['INDEPENDENT_CERTIFIED_FLAG_1'] += cert
        sha_record(label_proof, normalized)
        sha_record(timeline_proof, [mid, sha_file(timelines[mid])])
    require(by_part['TRAIN']['MATCHS'] == 845 and
            by_part['VALIDATION']['MATCHS'] == 269,
            'effectifs partition non conformes')
    require(not differences, 'divergences CSV / timeline : ' + repr(dict(differences)))
    require(all(c['UNKNOWN_PERIOD_EVENTS'] == 0 and c['MALFORMED_EVENTS'] == 0
                for c in by_part.values()), 'evenement invalide dans audit')
    return by_part, label_proof.hexdigest(), timeline_proof.hexdigest()


def build(p2g):
    print('=== G02-G20/P2H | RECONCILIATION AUTORISEE | TEST FERME ===')
    p2g.check_sources()
    split, allow = p2g.check_access()
    timelines = p2g.map_timelines(split)
    labels = p2g.read_authorized_csv_rows(p2g.CSV_LABELS, allow, split)
    require(set(allow).isdisjoint({mid for mid, r in split.items()
                                   if r['partition'] == 'TEST'}), 'TEST present dans allowlist')
    counts, label_digest, timeline_digest = tally(allow, labels, timelines, p2g)
    # Deliberately no SHA of entire label CSV: it would read TEST outcomes.
    report = {
        'stage': 'G02-G20/P2H',
        'status': 'INTERNAL_RECONCILIATION_ONLY_NOT_INDEPENDENT_NOT_ASOF',
        'audit_script_name': AUDITOR_NAME,
        'audit_script_sha256': AUDITOR_SHA256,
        'guard_sha256': {path.name: val for path, val in p2g.GUARDS.items()},
        'source_report_f4_sha256': sha_file(p2g.REPORT_F4),
        'builder_sha256': sha_file(p2g.BUILDER),
        'authorized_csv_records_sha256': label_digest,
        'authorized_timeline_sha_manifest_sha256': timeline_digest,
        'counts': {part: dict(counts[part]) for part in ('TRAIN', 'VALIDATION')},
        'counts_total': {
            'authorized_matches': 1114,
            'positive_labels': sum(counts[part]['POSITIFS'] for part in counts),
            'independently_certified_flags_1': sum(
                counts[part]['INDEPENDENT_CERTIFIED_FLAG_1'] for part in counts),
            'differences': 0,
            'TEST_timelines_read': 0,
            'TEST_label_values_interpreted': 0,
        },
        'label_definition': 'at least one GOALS event in second_half with numeric minute >87; extra_time periods excluded',
        'limitation': 'Agreement with original timeline is not independent sports result certification; as-of feature availability not established.',
        'api_calls': 0,
        'model_trained': False,
        'asof_certified': False,
        'training_authorized_by_this_report': False,
    }
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    for part in ('TRAIN', 'VALIDATION'):
        c = counts[part]
        print('%s | matchs %d | positifs %d | certifies independamment %d | prolongations exclus %d' % (
            part, c['MATCHS'], c['POSITIFS'], c['INDEPENDENT_CERTIFIED_FLAG_1'],
            c['EXTRA_TIME_GOALS_EXCLUDED']))
    print('DIVERGENCES : 0 | TEST 0 label lu | TEST 0 timeline ouverte')
    print('SHA AUTORISES (1114 LIGNES) :', label_digest)
    print('SHA TIMELINES AUTORISEES (MANIFESTE SHA) :', timeline_digest)
    print('SHA RAPPORT :', digest_bytes(raw))
    print('NON INDEPENDANT | AS-OF NON CERTIFIE | 0 API | 0 ENTRAINEMENT')
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    p2g = check_and_import()
    raw = build(p2g)
    if args.verify:
        require(REPORT.is_file() and not REPORT.is_symlink(), 'rapport gele absent')
        require(REPORT.read_bytes() == raw, 'rapport gele different du recalcul')
        print('G02-G20/P2H VERIFY PASS | rapport conforme | TEST FERME | 0 ECRITURE')
    elif args.freeze:
        require(not OUT.exists(), 'dossier de gel deja existant : ' + str(OUT))
        OUT.mkdir(parents=True, exist_ok=False)
        with REPORT.open('xb') as fh:
            fh.write(raw)
        print('FROZEN :', REPORT, '| SHA256', digest_bytes(raw))
        print('G02-G20/P2H FREEZE PASS | 1 FICHIER | TEST FERME | 0 MODEL')
    else:
        print('G02-G20/P2H PRECHECK PASS | 0 ECRITURE | TEST FERME')
        print('Pour figer : python3 -B g02g20_p2h_freeze_internal_label_audit.py --freeze')


if __name__ == '__main__':
    main()
