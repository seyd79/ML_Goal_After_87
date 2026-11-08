#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4.1 MASS-350: historic discovery + timeline acquisition, RESEARCH_ONLY.

Run inside ML_Goal_After_87 where diagnostic_live_api.py is installed.
No label creation, model training, or changes to V3-C.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE = Path('outputs/v4_research/v41_mass350')
MANIFEST = BASE / 'manifest.csv'
PROGRESS = BASE / 'discovery_progress.json'
RAW = BASE / 'raw_timelines'
REJECTED = BASE / 'rejected_timelines'
CSV_ROOTS = [Path('outputs/v4_research'), Path('outputs/pressure_historical')]
FIELDS = ['match_id', 'home_team', 'away_team', 'home_id', 'away_id',
          'kickoff_utc', 'discovery_day', 'research_status', 'label_used_for_selection']
START = date(2026, 3, 1)
END = date(2026, 8, 20)
PER_PAGE = 100
MAX_PAGES = 4
TARGET = 350                 # Additional usable timelines, separate from existing 16.
MAX_FULL_PER_DAY = 6         # At least 59 different days for 350 full timelines.
MAX_TRIES_PER_DAY = 30
MIN_DAYS = 50


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(item, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def exclusive_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(item, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def read_manifest():
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('Manifest schema mismatch; refuse to overwrite')
        rows = list(reader)
    ids = [r['match_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in mass manifest')
    return rows


def append_manifest(rows):
    if not rows:
        return
    BASE.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST.exists()
    with MANIFEST.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def protected_ids():
    """Fail closed if known split or training safeguards are absent."""
    expected = [
        Path('outputs/v4_research/V4_TRAIN278_B29_RESEARCH_V1.csv'),
        Path('outputs/v4_research/V41_DIVERSE_DATES_MANIFEST_V1.csv'),
        Path('outputs/pressure_historical/pressure_v2_dataset_join/ALL_320_CHRONO_SPLIT_PLAN_V1.csv'),
    ]
    for p in expected:
        if not p.is_file():
            raise FileNotFoundError('Missing essential protection list: ' + str(p))
    ids = set()
    nfiles = 0
    for root in CSV_ROOTS:
        if not root.exists():
            raise FileNotFoundError('Missing source directory: ' + str(root))
        for path in root.rglob('*.csv'):
            if path == MANIFEST:
                continue
            try:
                with path.open(encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or 'match_id' not in reader.fieldnames:
                        continue
                    nfiles += 1
                    for row in reader:
                        mid = (row.get('match_id') or '').strip()
                        if mid:
                            ids.add(mid)
            except (UnicodeError, csv.Error) as exc:
                raise ValueError('Cannot audit protected CSV ' + str(path) + ': ' + str(exc))
    if nfiles < 3 or len(ids) < 520:
        raise ValueError('Protected ID inventory unexpectedly small: {} IDs in {} CSVs'.format(len(ids), nfiles))
    return ids, nfiles


def api():
    from diagnostic_live_api import api_get
    return api_get


def list_items(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError('Unrecognized listing response: ' + type(payload).__name__)
    data = payload.get('data', payload)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ('matches', 'items', 'results', 'fixtures', 'data'):
            if isinstance(data.get(k), list):
                return data[k]
    raise ValueError('No match list found. Response keys: ' + str(list(payload.keys())))


def team_info(match, side):
    obj = match.get(side) or match.get(side + '_team')
    if isinstance(obj, dict):
        name = obj.get('name') or obj.get('short_name') or obj.get('display_name')
        tid = obj.get('id') or obj.get('team_id')
        return str(name or '').strip(), str(tid or '').strip()
    name = obj if isinstance(obj, str) else match.get(side + '_team_name', '')
    tid = match.get(side + '_team_id', '')
    return str(name or '').strip(), str(tid or '').strip()


def kickoff_iso(match):
    candidates = ('kickoff_utc', 'kickoff', 'start_time', 'starts_at',
                  'scheduled_at', 'start_at', 'datetime', 'date_time',
                  'startTime', 'utc_date', 'match_date', 'date')
    for k in candidates:
        v = match.get(k)
        if isinstance(v, dict):
            v = v.get('utc') or v.get('datetime')
        if not isinstance(v, str) or not v.strip() or len(v.strip()) <= 10:
            continue
        try:
            dt = datetime.fromisoformat(v.strip().replace('Z', '+00:00'))
        except ValueError:
            continue
        if dt.tzinfo is None:
            continue   # Never fabricate the UTC kickoff instant.
        return dt.astimezone(timezone.utc).isoformat()
    return None


def normalize_match(match, day):
    if not isinstance(match, dict):
        return None
    mid = match.get('match_id') or match.get('id')
    if not isinstance(mid, str) or not mid.startswith('mt_'):
        return None
    home, hid = team_info(match, 'home')
    away, aid = team_info(match, 'away')
    kickoff = kickoff_iso(match)
    if not (home and away and hid and aid and kickoff):
        return None
    # UTC day may differ from the query-local date; do not invent a date.
    if not START <= datetime.fromisoformat(kickoff).date() <= END:
        return None
    return dict(match_id=mid, home_team=home, away_team=away,
                home_id=hid, away_id=aid, kickoff_utc=kickoff,
                discovery_day=day, research_status='DISCOVERED_ONLY',
                label_used_for_selection='0')


def api_timeline_check(payload, mid):
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), dict):
        raise ValueError('Timeline response/data not a dict')
    data = payload['data']
    if str(data.get('match_id')) != mid:
        raise ValueError('Timeline ID differs from requested match')
    events = data.get('events')
    coverage = data.get('coverage')
    if not isinstance(events, list):
        raise ValueError('Events not a list')
    if coverage == 'none' and len(events) == 0:
        return 'QUARANTINE', 'NO_COVERAGE', len(events)
    if coverage != 'full':
        return 'QUARANTINE', 'COVERAGE_NOT_FULL', len(events)
    if not events:
        return 'QUARANTINE', 'FULL_BUT_EMPTY', 0
    if any(not isinstance(e, dict) for e in events):
        raise ValueError('Malformed event')
    has_half = any(e.get('period') == 'second_half' for e in events)
    has_end = any(e.get('period') == 'second_half' and e.get('type') == 'period_end'
                  for e in events)
    if not has_half or not has_end:
        return 'QUARANTINE', 'SECOND_HALF_INCOMPLETE', len(events)
    return 'SAVED', 'FULL_RESEARCH_AUDIT_PENDING', len(events)


def wait_for_rate(last, delay):
    if last is not None:
        remaining = delay - (time.monotonic() - last)
        if remaining > 0:
            time.sleep(remaining)
    return time.monotonic()


def discover(args):
    protected, nfiles = protected_ids()
    rows = read_manifest()
    known = protected | {r['match_id'] for r in rows}
    progress = load_json(PROGRESS, {'finished_days': [], 'next_page': {}, 'api_calls_total': 0})
    done = set(progress['finished_days'])
    next_page = progress['next_page']
    getter = api()
    calls = 0
    new = 0
    # Interleave months. Even a small initial discovery budget spans March–August.
    months = defaultdict(list)
    for i in range((END - START).days + 1):
        day = START + timedelta(days=i)
        months[day.strftime('%Y-%m')].append(day)
    month_keys = sorted(months)
    days = [months[m][i] for i in range(max(map(len, months.values())))
            for m in month_keys if i < len(months[m])]
    print('Protected IDs:', len(protected), 'CSV sources:', nfiles)
    print('Discovering from', START, 'to', END, '| days:', len(days))
    print('New listing request budget:', args.budget, '| stored candidates:', len(rows), flush=True)
    last = None
    for d in days:
        key = d.isoformat()
        if key in done:
            continue
        page = int(next_page.get(key, 1))
        while page <= MAX_PAGES and calls < args.budget:
            last = wait_for_rate(last, args.delay)
            print('LIST', key, 'page', page, '| request', calls + 1, '/', args.budget, flush=True)
            try:
                payload = getter('/football/matches', params=dict(
                    status='finished', date_from=key, date_to=key,
                    per_page=PER_PAGE, page=page))
                calls += 1
                items = list_items(payload)
                if not isinstance(items, list):
                    raise ValueError('List parse failed')
                normalized = [normalize_match(m, key) for m in items]
                valid = [x for x in normalized if x is not None]
                if items and not valid:
                    raise ValueError('No usable records on {}: unsupported metadata schema; '
                                     'check listing payload without spending more requests'.format(key))
                novel = [x for x in valid if x['match_id'] not in known]
                append_manifest(novel)
                for item in novel:
                    known.add(item['match_id'])
                new += len(novel)
                print('  listed:', len(items), 'usable:', len(valid),
                      'new:', len(novel), 'total new this run:', new, flush=True)
                progress['api_calls_total'] += 1
                if len(items) < PER_PAGE:
                    done.add(key)
                    progress['finished_days'] = sorted(done)
                    next_page.pop(key, None)
                    break
                page += 1
                next_page[key] = page
                if page > MAX_PAGES:
                    print('  PAGE CAP reached; selection from this date may be incomplete')
                    done.add(key)
                    progress['finished_days'] = sorted(done)
                    next_page.pop(key, None)
                    break
                # Each successfully processed page is durable before next request.
            except Exception as exc:
                print('SAFE STOP: discovery error:', type(exc).__name__, str(exc), flush=True)
                atomic_json(PROGRESS, progress)
                return 1
            atomic_json(PROGRESS, progress)
        atomic_json(PROGRESS, progress)
        if calls >= args.budget:
            break
    print('DISCOVER REPORT: calls=', calls, 'new_candidates=', new,
          'total_manifest=', len(read_manifest()), 'finished_days=', len(done),
          'of', len(days))
    return 0


def status():
    rows = read_manifest()
    byday = Counter()
    tried = Counter()
    saved = Counter()
    rejected = Counter()
    for row in rows:
        mid, d = row['match_id'], row['kickoff_utc'][:10]
        byday[d] += 1
        good, bad = RAW / (mid + '.json'), REJECTED / (mid + '.json')
        if good.exists() and bad.exists():
            raise ValueError('Conflicting files: ' + mid)
        if good.exists():
            saved[d] += 1
            tried[d] += 1
        elif bad.exists():
            rejected[d] += 1
            tried[d] += 1
    total_full, total_bad = sum(saved.values()), sum(rejected.values())
    print('MASS350 STATUS | known=', len(rows), '| full=', total_full,
          '| quarantined=', total_bad, '| pending=', len(rows) - total_full - total_bad)
    print('Full-covered UTC dates=', len(saved), '| target=', TARGET,
          '| max_full_per_day=', MAX_FULL_PER_DAY, '| min days=', MIN_DAYS)
    for d in sorted(byday):
        print('{}  candidates={:3d} full={:2d} quarantine={:2d} pending={:3d}'.format(
            d, byday[d], saved[d], rejected[d], byday[d] - tried[d]))
    return rows, byday, tried, saved, rejected


def collect(args):
    protected, _ = protected_ids()
    rows, _, tried, saved, rejected = status()
    if not rows:
        raise ValueError('Manifest empty: run discover first')
    if any(row['match_id'] in protected for row in rows):
        raise ValueError('Protected ID detected in mass manifest; safe stop')
    full_total = sum(saved.values())
    days_full = sum(v > 0 for v in saved.values())
    candidates = defaultdict(list)
    for row in rows:
        mid, day = row['match_id'], row['kickoff_utc'][:10]
        if (RAW / (mid + '.json')).exists() or (REJECTED / (mid + '.json')).exists():
            continue
        candidates[day].append(row)
    for day in candidates:
        candidates[day].sort(key=lambda r: hashlib.sha256(r['match_id'].encode()).hexdigest())
    getter = api()
    calls = 0
    last = None
    days = sorted(candidates)
    print('COLLECT: target={} NEW full timelines, >= {} UTC days, <= {} full/day'.format(
        TARGET, MIN_DAYS, MAX_FULL_PER_DAY))
    print('API budget this run:', args.budget, '| delay:', args.delay, 'seconds', flush=True)
    # Round robin: one pending game per date, then circle through dates.
    while calls < args.budget and candidates:
        if full_total >= TARGET and days_full >= MIN_DAYS:
            print('TARGET REACHED', flush=True)
            break
        progress = False
        for day in days:
            if calls >= args.budget or (full_total >= TARGET and days_full >= MIN_DAYS):
                break
            if saved[day] >= MAX_FULL_PER_DAY or tried[day] >= MAX_TRIES_PER_DAY:
                continue
            if not candidates[day]:
                continue
            row = candidates[day].pop(0)
            mid = row['match_id']
            good, bad = RAW / (mid + '.json'), REJECTED / (mid + '.json')
            if good.exists() or bad.exists():
                raise ValueError('Unexpected candidate status: ' + mid)
            last = wait_for_rate(last, args.delay)
            print('TIMELINE', mid, day, '| request', calls + 1, '/', args.budget, flush=True)
            try:
                payload = getter('/football/matches/{}/timeline'.format(mid))
                calls += 1
                decision, reason, n = api_timeline_check(payload, mid)
                record = {
                    'match_id': mid, 'home_team': row['home_team'],
                    'away_team': row['away_team'], 'home_id': row['home_id'],
                    'away_id': row['away_id'], 'kickoff_utc': row['kickoff_utc'],
                    'retrieved_at_utc': utc_now(), 'research_status':
                    ('RAW_TIMELINE_AUDIT_REQUIRED' if decision == 'SAVED' else 'QUARANTINED'),
                    'reason': reason, 'training_certified': False,
                    'label_certified': False, 'live_certified': False,
                    'label_used_for_selection': False,
                    'diagnostic': {'events_total': n, 'coverage': payload['data']['coverage']},
                    'timeline_response': payload,
                }
                exclusive_json(good if decision == 'SAVED' else bad, record)
            except Exception as exc:
                print('SAFE STOP:', type(exc).__name__, str(exc), flush=True)
                print('No further API calls; inspect the failed match', mid, flush=True)
                return 1
            tried[day] += 1
            progress = True
            if decision == 'SAVED':
                if saved[day] == 0:
                    days_full += 1
                saved[day] += 1
                full_total += 1
            else:
                rejected[day] += 1
            print(' ', decision, reason, 'events=', n, '| MASS full=', full_total,
                  'days=', days_full, '| quarantined=', sum(rejected.values()), flush=True)
        if not progress:
            print('No eligible pending candidates under day limits. Run discovery or widen date range.')
            break
    print('COLLECT REPORT: calls_this_run=', calls, 'mass_full=', full_total,
          'mass_quarantined=', sum(rejected.values()), 'full_days=', days_full,
          'target_full=', TARGET)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['discover', 'collect', 'status'])
    parser.add_argument('--budget', type=int, default=25,
                        help='Maximum listing or timeline calls for this execution')
    parser.add_argument('--delay', type=float, default=2.5)
    args = parser.parse_args()
    if not 0 <= args.budget <= 500:
        parser.error('--budget must be 0..500')
    if args.delay < 2:
        parser.error('--delay must be >= 2.0 seconds')
    if args.phase == 'discover':
        return discover(args)
    if args.phase == 'collect':
        return collect(args)
    status()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('SAFE STOP:', type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(1)
