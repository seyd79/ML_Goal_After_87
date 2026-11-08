#!/usr/bin/env python3
"""Goal After 87 / V4.2 — G02-D5: deterministic reserve + resumable API collection.

Run from the root of ML_Goal_After_87:
  python3 collect_v42_g02d5.py --budget 0
  python3 collect_v42_g02d5.py --budget 20
  python3 collect_v42_g02d5.py --budget 100

No predictions, training, target labels or modification of V4.1 files.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('outputs/v4_research')
FROZEN = BASE / 'v42_g02b_selection_1150/G02B_CANDIDATES_1150_FROZEN.csv'
FREEZE = BASE / 'v42_g02b_selection_1150/G02B_SELECTION_FREEZE.json'
MANIFEST = BASE / 'v41_mass350/manifest.csv'
OLD_RAW = BASE / 'v41_mass350/raw_timelines'
OLD_BAD = BASE / 'v41_mass350/rejected_timelines'
OUT = BASE / 'v42_g02d5_collection'
RESERVE = OUT / 'G02D5_RESERVE_FROZEN.csv'
RESERVE_REPORT = OUT / 'G02D5_RESERVE_FREEZE.json'
GOOD = OUT / 'raw_timelines'
BAD = OUT / 'quarantine'
PILOTS = [BASE / 'v42_g02d2_pilot5', BASE / 'v42_g02d3_coverage24']
PROTECTED = {
    'TRAIN112': BASE / 'v41_train_dataset_corrected_366/V41_TRAIN112_OWN_GOAL_CORRECTED.csv',
    'VALIDATION46': BASE / 'v41_validation_own_goal_385/V41_A2_VALIDATION46_FEATURES_BLIND.csv',
    'TRAIN294': BASE / 'V41_TRAIN294_B29_RESEARCH_V1.csv',
    'TRAIN638': BASE / 'V41_TRAIN638_B29_RESEARCH_V1.csv',
    'DIVERSE16': BASE / 'V41_DIVERSE16_B29_RESEARCH_V1.csv',
    'SPLIT320': Path('outputs/pressure_historical/pressure_v2_dataset_join/ALL_320_CHRONO_SPLIT_PLAN_V1.csv'),
}
EXPECTED_SHA = 'd28ab757a1311aa6b9b449dbf403319f79e2c69b76efdd3f382c8fd4ef1af0bc'
SEED = 'V42_G02D5_RESERVE_DETERMINISTIC_V1'
COLS = ('match_id','home_team','away_team','home_id','away_id','kickoff_utc','sampling_month')
MONTHS = tuple(f'2026-{m:02d}' for m in range(3,9))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p, required=('match_id',)):
    with p.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Invalid CSV header: ' + str(p))
        if not set(required).issubset(reader.fieldnames):
            raise ValueError('Missing fields in: ' + str(p))
        rows = list(reader)
    ids = [r['match_id'].strip() for r in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError('Empty/duplicate match_id: ' + str(p))
    return rows


def utc_month(text):
    d = datetime.fromisoformat(str(text).strip().replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timezone missing')
    return d.astimezone(timezone.utc).strftime('%Y-%m')


def rank(r):
    return (hashlib.sha256((SEED+'|'+r['match_id']).encode()).hexdigest(), r['match_id'])


def get_ids_from_files(folder):
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob('mt_*.json')}


def index_saved(folder, authorized):
    out = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob('mt_*.json')):
        if p.stem not in authorized:
            raise ValueError('Unexpected saved match: ' + str(p))
        obj = json.loads(p.read_text(encoding='utf-8'))
        if obj.get('match_id') != p.stem:
            raise ValueError('Saved match_id mismatch: '+ str(p))
        out[p.stem] = p
    return out


def prepare():
    if digest(FROZEN) != EXPECTED_SHA:
        raise ValueError('STOP: G02-B selection changed')
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if freeze.get('selected_csv_sha256') != EXPECTED_SHA:
        raise ValueError('STOP: G02-B freeze mismatch')
    selected = read_csv(FROZEN, COLS)
    original = read_csv(MANIFEST, COLS[:-1])
    if len(selected) != 1150 or len(original) != 12415 or len(get_ids_from_files(OLD_RAW)) != 350:
        raise ValueError('Unexpected sizes in G02-B inputs')

    protected = set(get_ids_from_files(OLD_RAW))
    for name, path in PROTECTED.items():
        if not path.is_file():
            raise ValueError(f'Protected source missing: {name}: {path}')
        protected.update(r['match_id'] for r in read_csv(path))

    selected_ids = {r['match_id'] for r in selected}
    if selected_ids & protected:
        raise ValueError('STOP: selection overlaps protected historical data')
    original_by_id = {r['match_id']: r for r in original}
    for r in selected:
        source = original_by_id.get(r['match_id'])
        if not source or source.get('research_status') != 'DISCOVERED_ONLY' or str(source.get('label_used_for_selection')) != '0':
            raise ValueError('Selected source not label-blind/discovered-only: '+r['match_id'])
    by_month = {m: [] for m in MONTHS}
    for r in selected:
        if r['sampling_month'] != utc_month(r['kickoff_utc']) or r['sampling_month'] not in MONTHS:
            raise ValueError('Invalid selected month: '+r['match_id'])
        by_month[r['sampling_month']].append(r)
    quotas = {m: len(by_month[m]) for m in MONTHS}
    if sum(quotas.values()) != 1150:
        raise ValueError('Invalid quotas')

    reserved = []
    for r in original:
        mid = r['match_id']
        if mid in protected or mid in selected_ids:
            continue
        # Reserve eligibility is frozen using only pre-existing provenance flags.
        # Never add records that have been selected using target labels.
        if r.get('research_status') != 'DISCOVERED_ONLY' or str(r.get('label_used_for_selection')) != '0':
            continue
        month = utc_month(r['kickoff_utc'])
        if month in MONTHS:
            reserved.append({**{k: r[k] for k in COLS[:-1]}, 'sampling_month': month})
    reserved.sort(key=lambda r: (r['sampling_month'],) + rank(r))
    if len({r['match_id'] for r in reserved}) != len(reserved):
        raise ValueError('Duplicate reserve')

    OUT.mkdir(parents=True, exist_ok=True)
    if not RESERVE.exists() and not RESERVE_REPORT.exists():
        with RESERVE.open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLS)
            writer.writeheader()
            writer.writerows(reserved)
        report = {
            'stage': 'G02-D5', 'seed': SEED,
            'main_selection_sha256': EXPECTED_SHA,
            'manifest_sha256': digest(MANIFEST),
            'protected_sha256': {name:digest(path) for name,path in PROTECTED.items()},
            'reserve_sha256': digest(RESERVE),
            'reserve_count':len(reserved),
            'quotas_by_month':quotas,
            'selection_by_label':False,
            'reserve_status':'FROZEN',
            'note':'This reserve is ordered by SHA256 within month, not by outcomes or API availability.'
        }
        with RESERVE_REPORT.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    if not RESERVE.exists() or not RESERVE_REPORT.exists():
        raise ValueError('STOP: partially created reserve plan, inspect manually')
    report = json.loads(RESERVE_REPORT.read_text(encoding='utf-8'))
    if (report.get('seed') != SEED or report.get('main_selection_sha256') != EXPECTED_SHA
        or report.get('manifest_sha256') != digest(MANIFEST)
        or report.get('protected_sha256') != {name:digest(path) for name,path in PROTECTED.items()}
        or report.get('reserve_sha256') != digest(RESERVE)
        or report.get('quotas_by_month') != quotas):
        raise ValueError('STOP: reserve freeze changed')
    committed = read_csv(RESERVE, COLS)
    if committed != reserved:
        raise ValueError('STOP: reserve plan differs from regenerated deterministic reserve')

    GOOD.mkdir(exist_ok=True)
    BAD.mkdir(exist_ok=True)
    authorized = selected_ids | {r['match_id'] for r in committed}
    saved = {}
    for folder in [p / 'raw_timelines' for p in PILOTS] + [GOOD]:
        for mid, path in index_saved(folder, authorized).items():
            if mid in saved:
                raise ValueError('Duplicate saved match: '+mid)
            saved[mid] = path
    old_bad = get_ids_from_files(OLD_BAD)
    bad = {}
    for folder in [p / 'quarantine' for p in PILOTS] + [BAD]:
        for mid,path in index_saved(folder, authorized).items():
            if mid in bad:
                raise ValueError('Duplicate quarantine match: '+mid)
            bad[mid] = path
    if set(saved) & (set(bad)|old_bad):
        raise ValueError('Conflicting good/bad records')
    known_bad = set(bad)|old_bad
    saved_counts = Counter()
    for mid in saved:
        if mid in selected_ids:
            month = next(r['sampling_month'] for r in selected if r['match_id']==mid)
        else:
            month = next(r['sampling_month'] for r in committed if r['match_id']==mid)
        saved_counts[month] += 1
    return by_month, quotas, committed, saved, bad, known_bad, saved_counts, old_bad


def status(by_month, quotas, reserve, saved, bad, known_bad, saved_counts, old_bad):
    print('='*76)
    print('G02-D5 — COLLECTION V4.2 | frozen selection + deterministic reserve')
    print('='*76)
    print('Selected:', sum(quotas.values()), '| Reserve:',len(reserve),
          '| Saved:',len(saved), '| Pilot/new quarantine:',len(bad),
          '| Historical rejected IDs:',len(old_bad))
    print('Reserve SHA256:', digest(RESERVE))
    print('Month | goal | SAVED | remaining')
    for m in MONTHS:
        print(m, '|',quotas[m], '|',saved_counts[m], '|',max(0,quotas[m]-saved_counts[m]))
    print('New usable timelines needed:',sum(max(0,quotas[m]-saved_counts[m]) for m in MONTHS))
    print('Known bad are skipped, not deleted; good files are reused; no labels are read.')


def run(budget, delay):
    by_month, quotas, reserve, saved, bad, known_bad, counts, old_bad = prepare()
    status(by_month,quotas,reserve,saved,bad,known_bad,counts,old_bad)
    if budget == 0:
        print('\nPLAN_ONLY: no API calls made.')
        return
    import collect_v41_mass_350_v1 as collector
    getter = collector.api()
    if not callable(getter):
        raise ValueError('API getter not callable')
    # Build queues once; after each response decide whether to update month counter.
    res_by_month = defaultdict(list)
    for r in reserve:
        res_by_month[r['sampling_month']].append(r)
    queue = {}
    for m in MONTHS:
        queue[m] = [r for r in sorted(by_month[m],key=lambda x:(x['kickoff_utc'],x['match_id'])) + res_by_month[m]
                    if r['match_id'] not in saved and r['match_id'] not in known_bad]
    calls = 0
    last = None
    while calls < budget:
        progressed = False
        for m in MONTHS:
            if calls >= budget:
                break
            if counts[m] >= quotas[m] or not queue[m]:
                continue
            r = queue[m].pop(0)
            mid = r['match_id']
            progressed = True
            if last is not None:
                remaining = delay - (time.monotonic()-last)
                if remaining > 0:
                    time.sleep(remaining)
            last = time.monotonic()
            calls += 1
            print('CALL', calls,'/',budget,'|',m,'|',mid, flush=True)
            try:
                payload = getter('/football/matches/{}/timeline'.format(mid))
                decision,reason,n = collector.api_timeline_check(payload,mid)
            except Exception as exc:
                print('SAFE STOP: request/validation error:', mid, type(exc).__name__, str(exc)[:160])
                print('Completed files are durable; rerun command after investigating.')
                raise SystemExit(2)
            if decision not in ('SAVED','QUARANTINE'):
                raise ValueError('Unknown timeline decision: '+str(decision))
            rec = {
                'match_id':mid,'home_team':r['home_team'],'away_team':r['away_team'],
                'home_id':r['home_id'],'away_id':r['away_id'],'kickoff_utc':r['kickoff_utc'],
                'retrieved_at_utc':collector.utc_now(),
                'research_status':'RAW_TIMELINE_AUDIT_REQUIRED' if decision=='SAVED' else 'QUARANTINED',
                'reason':reason, 'training_certified':False, 'label_certified':False,
                'live_certified':False, 'label_used_for_selection':False,
                'diagnostic': {'events_total':n,'coverage':payload['data'].get('coverage')},
                'timeline_response':payload,
            }
            dest = (GOOD if decision=='SAVED' else BAD)/(mid+'.json')
            collector.exclusive_json(dest,rec)
            if decision=='SAVED':
                saved[mid] = dest
                counts[m]+=1
            else:
                bad[mid] = dest
                known_bad.add(mid)
            print(' ->',decision,'| events=',n,'| reason=',reason, flush=True)
        if not progressed:
            break
    print('\n=== END-OF-RUN G02-D5 ===')
    print('New API requests:',calls)
    print('Total SAVED including pilots:',len(saved))
    print('New QUARANTINE:',len(list(BAD.glob('mt_*.json'))))
    print('Counts per month:',dict(sorted(counts.items())))
    print('Target:',sum(quotas.values()),'| remaining:',sum(max(0,quotas[m]-counts[m]) for m in MONTHS))
    print('No labels, model changes, or training. Original frozen files untouched.')


def main():
    parser = argparse.ArgumentParser(description='G02-D5 frozen and resumable timeline collection')
    parser.add_argument('--budget',type=int,default=0,help='Maximum new API timeline requests for this run (default: zero)')
    parser.add_argument('--delay',type=float,default=2.5,help='Minimum seconds between requests')
    args = parser.parse_args()
    if not (0 <= args.budget <= 1000) or not math.isfinite(args.delay) or args.delay < 2.5:
        raise ValueError('budget must be 0..1000; delay >=2.5 seconds')
    run(args.budget,args.delay)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print('SAFE STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(1)
