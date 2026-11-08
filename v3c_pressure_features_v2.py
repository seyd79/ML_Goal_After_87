#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3-C PRESSURE FEATURES V2 - reconstruction EXPERIMENTALE, Python >= 3.9.

Règles GLOBALES figées (ne jamais adapter un match à son /stats final) :
* 5 minutes TERMINÉES : [minute_snapshot-5, minute_snapshot), soit 82..86 à 87'.
* corner_kick.team est inversé (convention expérimentale constatée dans les caches).
* tirs = shot_on_target + shot_off_target + shot_blocked + shot_woodwork
       + goal + penalty_scored + penalty_missed.
* tirs cadrés = shot_on_target + goal + penalty_scored.
* les paires goal / penalty_scored pour le MEME joueur/equipe/minute
  sont fusionnées, sous réserve de validation; pas de correction au cas par cas.
* own_goal et penalty_saved sont EXCLUS des comptes (convention non validée).
* couverture/match/equipe/temps ambigus -> 6 valeurs nulles.

Ces features NE SONT PAS encore certifiées pour apprentissage. Une concordance de
/ timeline avec /stats est seulement un contrôle interne au même fournisseur.
"""
import json
import re
import unicodedata
from collections import Counter

VERSION = 'pressure_v2_experimental_2026_10_25'
MINUTES = (85, 86, 87)
FEATURES = ('shots_home_last5', 'shots_away_last5',
            'sot_home_last5', 'sot_away_last5',
            'corners_home_last5', 'corners_away_last5')
BASE_SHOTS = frozenset(('shot_on_target', 'shot_off_target', 'shot_blocked'))
SHOT_EXTRAS = frozenset(('goal', 'shot_woodwork', 'penalty_scored', 'penalty_missed'))
SOT_TYPES = frozenset(('shot_on_target', 'goal', 'penalty_scored'))
CORNER_TYPE = 'corner_kick'
RELEVANT = BASE_SHOTS | SHOT_EXTRAS | {'penalty_saved', 'own_goal', CORNER_TYPE}


def norm(value):
    s = unicodedata.normalize('NFKD', str(value or '')).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')


def as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 150 else None
    if isinstance(value, str) and re.fullmatch(r'\d{1,3}', value):
        return int(value)
    return None


def event_clock(e):
    minute = as_int(e.get('minute'))
    extra = as_int(e.get('extra_time', 0))
    if minute is None or extra is None:
        return None
    if minute not in range(0, 131) or extra > 40:
        return None
    # TheStatsAPI convention: minute=90, extra_time=4 means 90+4, not 90.
    return minute + extra


def side_for(e, manifest):
    team = e.get('team')
    if not isinstance(team, dict):
        return None
    eid = str(team.get('id') or '').strip()
    ename = norm(team.get('name'))
    hid = str(manifest.get('home_id') or '').strip()
    aid = str(manifest.get('away_id') or '').strip()
    home = norm(manifest.get('home_team'))
    away = norm(manifest.get('away_team'))
    if not home or not away or home == away:
        return None
    id_side = 'home' if hid and eid == hid else ('away' if aid and eid == aid else None)
    name_side = 'home' if ename and ename == home else ('away' if ename and ename == away else None)
    # Explicit ID and name must not contradict each other.
    if id_side and name_side and id_side != name_side:
        return None
    return id_side or name_side


def event_list(payload, match_id):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or str(data.get('match_id') or '') != str(match_id):
        raise ValueError('TIMELINE_MATCH_ID_MISMATCH')
    events = data.get('events')
    if not isinstance(events, list):
        raise ValueError('EVENT_LIST_MISSING')
    meta = payload.get('meta') or {}
    return events, str(data.get('coverage') or (meta.get('coverage') if isinstance(meta, dict) else '')).lower()


def dedup_and_validate(events, manifest):
    """Resolve team/clock, reject ambiguous offense, dedup identical provider sequences.

    Returns (ordered offensive events, diagnostics). All timestamps are regulation.
    """
    clean = []
    issues = Counter()
    seen = {}
    for e in events:
        if not isinstance(e, dict):
            issues['NON_DICT_EVENT'] += 1
            continue
        sequence = e.get('sequence', e.get('id'))
        if sequence is not None:
            seq = str(sequence)
            if seq in seen:
                if seen[seq] != e:
                    issues['CONFLICTING_DUPLICATE_SEQUENCE'] += 1
                else:
                    issues['IDENTICAL_DUPLICATE_SEQUENCE'] += 1
                continue
            seen[seq] = e
        typ = str(e.get('type') or '').strip().lower()
        if typ not in RELEVANT:
            # Do not silently regard unknown shot/corner/penalty types as non-events.
            if typ in ('penalty_awarded', 'penalty_cancelled', 'penalty_canceled',
                       'penalty_review', 'penalty_shootout'):
                continue  # administrative events; not shot attempts
            if any(t in typ for t in ('shot', 'corner', 'penalty')):
                issues['UNKNOWN_OFFENSIVE_TYPE'] += 1
            continue
        period = str(e.get('period') or '').strip().lower()
        if period not in ('first_half', 'second_half'):
            issues['OFFENSIVE_OUTSIDE_REGULATION_OR_UNKNOWN_PERIOD'] += 1
            # For shootouts it is safe to ignore, but missing period is ambiguous.
            if not period:
                issues['UNKNOWN_PERIOD'] += 1
            continue
        clock = event_clock(e)
        if clock is None:
            issues['BAD_OFFENSIVE_CLOCK'] += 1
            continue
        side = side_for(e, manifest)
        if side is None:
            issues['UNKNOWN_OFFENSIVE_TEAM'] += 1
            continue
        player = e.get('player')
        pid = str(player.get('id') or '').strip() if isinstance(player, dict) else ''
        clean.append({'type': typ, 'minute': clock, 'period': period, 'side': side,
                      'player_id': pid, 'sequence': str(sequence or '')})
    # Coalescing only strongly evidenced representations of the SAME penalty goal.
    # If no player ID, do not guess from same minute/team: can be two distinct events.
    goal_keys = {(e['minute'], e['side'], e['player_id']) for e in clean
                 if e['type'] == 'goal' and e['player_id']}
    filtered = []
    for e in clean:
        key = (e['minute'], e['side'], e['player_id'])
        if e['type'] == 'penalty_scored' and e['player_id'] and key in goal_keys:
            issues['PENALTY_GOAL_DUPLICATE_MERGED'] += 1
            continue
        filtered.append(e)
    return filtered, dict(issues)


def count_events(events, minute, swap_corners=True):
    counters = Counter()
    for event in events:
        if event['period'] != 'second_half' or not (minute - 5 <= event['minute'] < minute):
            continue
        kind = event['type']
        side = event['side']
        if kind == CORNER_TYPE:
            recipient = ('away' if side == 'home' else 'home') if swap_corners else side
            counters['corners_' + recipient + '_last5'] += 1
        elif kind in BASE_SHOTS or kind in SHOT_EXTRAS:
            counters['shots_' + side + '_last5'] += 1
            if kind in SOT_TYPES:
                counters['sot_' + side + '_last5'] += 1
    return {f: counters[f] for f in FEATURES}


def compute(payload, manifest, minute):
    if minute not in MINUTES:
        raise ValueError('minute doit être 85, 86 ou 87')
    mid = str(manifest.get('match_id') or '')
    out = {'match_id': mid, 'minute': minute, 'feature_logic_version': VERSION,
           'temporal_provenance': 'HISTORICAL_HINDSIGHT_RECONSTRUCTION',
           'training_ready': False, 'pressure_status': 'UNKNOWN',
           'timeline_coverage': '', 'event_count': '', 'diagnostic_issues': '{}',
           'events_in_window': '', 'events_after_cutoff_excluded': '',
           'corner_policy': 'SWAPPED_EXPERIMENTAL',
           'shot_policy': 'BASE_GOAL_WOODWORK_PEN_SCORED_PEN_MISSED',
           'sot_policy': 'SHOT_ON_TARGET_GOAL_PEN_SCORED',
           **{key: None for key in FEATURES}}
    try:
        events, coverage = event_list(payload, mid)
    except ValueError as ex:
        out['pressure_status'] = str(ex)
        return out
    out['timeline_coverage'] = coverage
    out['event_count'] = len(events)
    if coverage != 'full':
        out['pressure_status'] = 'TIMELINE_COVERAGE_NOT_FULL'
        return out
    clean, issues = dedup_and_validate(events, manifest)
    out['diagnostic_issues'] = json.dumps(issues, ensure_ascii=False, sort_keys=True)
    bad = {'NON_DICT_EVENT', 'CONFLICTING_DUPLICATE_SEQUENCE', 'UNKNOWN_OFFENSIVE_TYPE',
           'UNKNOWN_PERIOD', 'BAD_OFFENSIVE_CLOCK', 'UNKNOWN_OFFENSIVE_TEAM'}
    if any(issues.get(k, 0) for k in bad):
        out['pressure_status'] = 'QUARANTINED_OFFENSIVE_EVENT_AMBIGUITY'
        return out
    out['events_in_window'] = sum(e['period'] == 'second_half' and minute - 5 <= e['minute'] < minute for e in clean)
    out['events_after_cutoff_excluded'] = sum(e['minute'] >= minute for e in clean)
    out.update(count_events(clean, minute))
    out['pressure_status'] = 'V2_EXPERIMENTAL_NOT_CERTIFIED'
    return out


def full_time_counts(payload, manifest):
    """FT diagnostic on the SAME event rules; never passed into compute()."""
    events, coverage = event_list(payload, manifest['match_id'])
    if coverage != 'full':
        raise ValueError('TIMELINE_COVERAGE_NOT_FULL')
    clean, issues = dedup_and_validate(events, manifest)
    bad = {'NON_DICT_EVENT', 'CONFLICTING_DUPLICATE_SEQUENCE', 'UNKNOWN_OFFENSIVE_TYPE',
           'UNKNOWN_PERIOD', 'BAD_OFFENSIVE_CLOCK', 'UNKNOWN_OFFENSIVE_TEAM'}
    if any(issues.get(k, 0) for k in bad):
        raise ValueError('QUARANTINED_OFFENSIVE_EVENT_AMBIGUITY')
    counts = Counter()
    for e in clean:
        kind = e['type']
        side = e['side']
        if kind == CORNER_TYPE:
            other = 'away' if side == 'home' else 'home'
            counts[(other, 'corners')] += 1
        elif kind in BASE_SHOTS or kind in SHOT_EXTRAS:
            counts[(side, 'shots')] += 1
            if kind in SOT_TYPES:
                counts[(side, 'sot')] += 1
    return {(side, metric): counts[(side, metric)]
            for side in ('home', 'away') for metric in ('shots', 'sot', 'corners')}


def stats_counts(payload, mid):
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or str(data.get('match_id') or '') != str(mid):
        raise ValueError('STATS_MATCH_ID_MISMATCH')
    overview = data.get('overview')
    if not isinstance(overview, dict):
        raise ValueError('STATS_OVERVIEW_MISSING')
    d = {}
    for f, metric in (('total_shots', 'shots'), ('shots_on_target', 'sot'), ('corner_kicks', 'corners')):
        both = overview.get(f, {}).get('all') if isinstance(overview.get(f), dict) else None
        for side in ('home', 'away'):
            v = both.get(side) if isinstance(both, dict) else None
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError('STATS_FIELD_INVALID_' + f + '_' + side)
            d[(side, metric)] = v
    return d
