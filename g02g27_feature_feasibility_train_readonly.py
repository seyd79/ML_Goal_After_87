#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G27 | Faisabilite de nouvelles features depuis les timelines TRAIN seules.

Depuis la RACINE du projet ML_Goal_After_87 (Python >= 3.9):
  python3 -B g02g27_feature_feasibility_train_readonly.py
  python3 -B g02g27_feature_feasibility_train_readonly.py --audit-train

Mode normal : verifie les empreintes des fichiers geles et l'allowlist ;
              n'ouvre aucune timeline, aucun label, aucun resultat.
Mode --audit-train : ouvre UNIQUEMENT les timelines des 845 matchs TRAIN
              admissibles, compare les evenements a la minute <87 aux features
              V4.2 gelées, et mesure la disponibilite de NOUVELLES features.
              N'ouvre aucune timeline VALIDATION/TEST ni aucun fichier de labels.

Aucune ecriture, aucun appel API, aucun fit, aucune selection de features.
Les timelines sont RETROSPECTIVES. Meme si une feature est calculable depuis
une timeline historisee, la disponibilite telle qu'observee a la 87e minute
(AS-OF) N'EST PAS certifiee. Les etiquettes ne sont pas independamment verifiees.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('outputs/v4_research')
DATA = BASE / 'v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
ALLOW = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
PROTO = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'
REG = BASE / 'v42_g02f2_quality_pre87/G02F2_PRE87_QUALITY_REGISTRY.csv'
EXPECTED = {
    DATA: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    PROTO: '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
    REG: '4cfe6146db28a4f05837bd4b66673eb146099bf7363197e5096a51de3cdd1b17',
}
RAW_DIRS = (
    BASE / 'v41_mass350/raw_timelines',
    BASE / 'v42_g02d2_pilot5/raw_timelines',
    BASE / 'v42_g02d3_coverage24/raw_timelines',
    BASE / 'v42_g02d5_collection/raw_timelines',
)
REGULATION = {'first_half', 'second_half'}
GOALS = {'goal', 'own_goal', 'penalty_scored'}
SHOTS = {'shot_on_target', 'shot_off_target', 'shot_blocked', 'shot_woodwork',
         'goal', 'penalty_scored', 'penalty_saved', 'penalty_missed'}
SOT = {'shot_on_target', 'goal', 'penalty_scored', 'penalty_saved'}
CORNERS = {'corner_kick'}
TIMING_KEYS = frozenset({'published_at', 'published_utc', 'received_at',
    'ingested_at', 'created_at', 'observed_at', 'event_timestamp',
    'event_received_at', 'provider_timestamp', 'available_at'})
CANDIDATES = (
    'home_shots_80_87', 'away_shots_80_87',
    'home_sot_80_87', 'away_sot_80_87',
    'home_corners_80_87', 'away_corners_80_87',
    'shots_80_84', 'shots_84_87', 'sot_84_87',
    'home_minutes_since_last_sot', 'away_minutes_since_last_sot',
    'home_goals_80_87', 'away_goals_80_87',
    'score_80_home', 'score_80_away',
)


def require(condition, message):
    if not condition:
        raise SystemExit('G02-G27 STOP : ' + message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fd:
        for chunk in iter(lambda: fd.read(128 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path, required):
    with path.open('r', encoding='utf-8-sig', newline='') as fd:
        rd = csv.DictReader(fd)
        require(rd.fieldnames is not None and
                len(rd.fieldnames) == len(set(rd.fieldnames)) and
                set(required) <= set(rd.fieldnames),
                'colonnes source invalides : ' + path.name)
        items = {}
        for row in rd:
            mid = row.get('match_id', '')
            require(mid and mid not in items and None not in row,
                    'ID/ligne incorrect dans ' + path.name)
            items[mid] = row
    return items


def plain(value):
    return ' '.join(str(value or '').strip().casefold().split())


def side_of(raw_team, wrapper):
    """Resolution conservatrice : correspondance non ambigue ID/nom/cote."""
    if isinstance(raw_team, dict):
        values = [raw_team.get(k) for k in ('id','team_id','name','team_name','side')]
    else:
        values = [raw_team]
    homes = {plain(wrapper.get('home_id')), plain(wrapper.get('home_team')), 'home', 'h'}
    aways = {plain(wrapper.get('away_id')), plain(wrapper.get('away_team')), 'away', 'a'}
    homes.discard('')
    aways.discard('')
    hits = set()
    for item in values:
        if item is None:
            continue
        if plain(item) in homes:
            hits.add('home')
        if plain(item) in aways:
            hits.add('away')
    return next(iter(hits)) if len(hits) == 1 else None


def number_minute(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def numeric_count(value):
    n = number_minute(value)
    require(n is not None and n.is_integer(), 'score/donnee de reference non entiere')
    return int(n)


def reconstruct(events, wrapper, reference):
    """Ne consulte JAMAIS le type/equipe d'un evenement minute >= 87.

    Les champs period/minute des evenements tardifs ne servent qu'au filtrage.
    Les valeurs de type et team ne sont lues que pour des evenements avant 87.
    """
    require(isinstance(events, list), 'liste events absente')
    counts = Counter()
    issues = Counter()
    seen_types = Counter()
    last_sot_side = {'home': None, 'away': None}
    last_sot_global = None
    total_goals = Counter()
    recent_goals = Counter()
    timed_match = False
    for ev in events:
        if not isinstance(ev, dict):
            issues['EVENT_NON_DICT'] += 1
            continue
        period = plain(ev.get('period'))
        if period not in REGULATION:
            # Ignore extras (120 min), shootouts, autres periodes.
            if period not in {'extra_time_first_half', 'extra_time_second_half',
                              'penalties', 'shootout', 'penalty_shootout',
                              'penalty_shoot-out'}:
                issues['PERIOD_INCONNUE'] += 1
            continue
        minute = number_minute(ev.get('minute'))
        if minute is None:
            issues['MINUTE_INVALIDE'] += 1
            continue
        if minute >= 87:
            continue
        typ = plain(ev.get('type'))
        seen_types[typ if typ else '<TYPE_ABSENT>'] += 1
        if any(k in ev for k in TIMING_KEYS):
            timed_match = True
            counts['EVENTS_WITH_TIMING_KEY'] += 1
        # Comptages globaux independants de l'attribution a une equipe.
        if typ in SOT:
            last_sot_global = minute if last_sot_global is None else max(last_sot_global, minute)
        if period == 'second_half' and 80 <= minute < 87:
            if typ in SHOTS:
                counts['shots_80_87'] += 1
                if minute < 84:
                    counts['shots_80_84'] += 1
                else:
                    counts['shots_84_87'] += 1
            if typ in SOT:
                counts['sot_80_87'] += 1
                if minute >= 84:
                    counts['sot_84_87'] += 1
            if typ in CORNERS:
                counts['corners_80_87'] += 1
        # Si l'evenement concerne attaque/score, l'equipe doit etre reconnue.
        if typ not in SHOTS and typ not in CORNERS and typ not in GOALS:
            continue
        side = side_of(ev.get('team'), wrapper)
        if side is None:
            if typ in GOALS:
                issues['GOAL_SIDE_UNRESOLVED'] += 1
            if typ in SHOTS or typ in CORNERS:
                issues['OFFENSIVE_SIDE_UNRESOLVED'] += 1
            continue
        if typ in SOT:
            prev = last_sot_side[side]
            last_sot_side[side] = minute if prev is None else max(prev, minute)
        if period == 'second_half' and 80 <= minute < 87:
            if typ in SHOTS:
                counts[side + '_shots_80_87'] += 1
            if typ in SOT:
                counts[side + '_sot_80_87'] += 1
            if typ in CORNERS:
                counts[side + '_corners_80_87'] += 1
        if typ in GOALS:
            credit = ('away' if side == 'home' else 'home') if typ == 'own_goal' else side
            total_goals[credit] += 1
            if period == 'second_half' and 80 <= minute < 87:
                recent_goals[credit] += 1

    # Verifications structurelles contre les 8 features V4.2 gelees.
    discrepancies = []
    for field in ('shots_80_87', 'sot_80_87', 'corners_80_87'):
        if counts[field] != numeric_count(reference[field]):
            discrepancies.append(field)
    expected_last = reference.get('minutes_since_last_shot_on_target', '')
    measured_last = None if last_sot_global is None else 87 - last_sot_global
    if expected_last == '':
        if measured_last is not None:
            discrepancies.append('minutes_since_last_shot_on_target')
    elif measured_last is None or abs(float(expected_last) - measured_last) > 1e-5:
        discrepancies.append('minutes_since_last_shot_on_target')
    if not issues['GOAL_SIDE_UNRESOLVED'] and not issues['MINUTE_INVALIDE']:
        for side in ('home', 'away'):
            if total_goals[side] != numeric_count(reference['score_87_' + side]):
                discrepancies.append('score_87_' + side)
    if discrepancies:
        issues['REFERENCE_MISMATCH'] = len(discrepancies)

    result = {name: None for name in CANDIDATES}
    for name in ('shots_80_84', 'shots_84_87', 'sot_84_87'):
        result[name] = counts[name]
    clean_offensive = not any(issues[k] for k in ('EVENT_NON_DICT', 'PERIOD_INCONNUE',
        'MINUTE_INVALIDE', 'OFFENSIVE_SIDE_UNRESOLVED', 'REFERENCE_MISMATCH'))
    if clean_offensive:
        for side in ('home', 'away'):
            for category in ('shots','sot','corners'):
                name = side + '_' + category + '_80_87'
                result[name] = counts[name]
    for side in ('home', 'away'):
        if clean_offensive and last_sot_side[side] is not None:
            result[side + '_minutes_since_last_sot'] = 87 - last_sot_side[side]
    clean_score = not any(issues[k] for k in ('EVENT_NON_DICT', 'PERIOD_INCONNUE',
        'MINUTE_INVALIDE', 'GOAL_SIDE_UNRESOLVED', 'REFERENCE_MISMATCH'))
    if clean_score:
        for side in ('home', 'away'):
            result[side + '_goals_80_87'] = recent_goals[side]
            score_at80 = total_goals[side] - recent_goals[side]
            require(score_at80 >= 0, 'score a 80 negatif')
            result['score_80_' + side] = score_at80
    return result, issues, seen_types, timed_match, counts['EVENTS_WITH_TIMING_KEY']


def selftest():
    w = {'home_id':'tm_1', 'away_id':'tm_2','home_team':'Alpha','away_team':'Beta'}
    events = [
        {'minute': 12, 'period':'first_half', 'type':'goal', 'team':'tm_1'},
        {'minute': 40, 'period':'first_half', 'type':'own_goal', 'team':'tm_2'},
        {'minute': 82, 'period':'second_half', 'type':'shot_on_target', 'team':'tm_2'},
        {'minute': 85, 'period':'second_half', 'type':'corner_kick', 'team':'tm_1'},
        {'minute': 86, 'period':'second_half', 'type':'goal', 'team':'tm_2'},
        {'minute': 87, 'period':'second_half', 'type':'goal', 'team':'tm_1'},
        {'minute': 92, 'period':'second_half', 'type':'goal', 'team':'tm_2'},
        {'minute': 114, 'period':'extra_time_second_half','type':'goal','team':'tm_1'},
    ]
    ref = {'shots_80_87':'2', 'sot_80_87':'2', 'corners_80_87':'1',
           'minutes_since_last_shot_on_target':'1',
           'score_87_home':'2', 'score_87_away':'1'}
    a, issues, _, _, _ = reconstruct(events, w, ref)
    require(not issues and a['score_80_home'] == 2 and a['score_80_away'] == 0 and
            a['away_shots_80_87'] == 2 and a['home_corners_80_87'] == 1 and
            a['away_goals_80_87'] == 1 and
            a['away_minutes_since_last_sot'] == 1,
            'test synthetique cas normal incorrect : ' + str(dict(issues)))
    changed = [dict(e) for e in events]
    changed[-2]['type'] = 'shot_blocked'
    changed[-2]['team'] = 'tm_1'
    b, bi, _, _, _ = reconstruct(changed, w, ref)
    require(a == b and not bi, 'fuite d evenement >87 dans features')
    unresolved = [dict(e) for e in events]
    unresolved[2]['team'] = 'tm_inconnue'
    c, ci, _, _, _ = reconstruct(unresolved, w, ref)
    require(ci['OFFENSIVE_SIDE_UNRESOLVED'] and c['away_shots_80_87'] is None,
            'absence equipe mal detectee')


def load_guards():
    print('=== G02-G27/P1 | FEASIBILITE NOUVELLES FEATURES | PRECHECK ===')
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), 'source absente/lien : ' + str(path))
        require(sha(path) == expected, 'SHA modifie : ' + path.name)
        print('SHA OK :', path.name)
    split = read_csv(SPLIT, ('match_id','partition','kickoff_utc'))
    elig = read_csv(ELIG, ('match_id','partition','kickoff_utc','research_eligibility'))
    allow = read_csv(ALLOW, ('match_id','partition','kickoff_utc'))
    dataset = read_csv(DATA, ('match_id','kickoff_utc','home_team_id','away_team_id',
        'score_87_home','score_87_away','shots_80_87','sot_80_87','corners_80_87',
        'minutes_since_last_shot_on_target'))
    registry = read_csv(REG, ('match_id','source_path','source_sha256','kickoff_utc'))
    require(len(split) == len(elig) == len(dataset) == len(registry) == 1500,
            'dataset/split/elig/registre non 1500')
    require(set(split) == set(elig) == set(dataset) == set(registry),
            'les 1500 identifiants ne correspondent pas')
    require(dict(Counter(v['partition'] for v in split.values())) ==
            {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}, 'split modifie')
    allowed = {mid for mid, r in elig.items() if r['partition'] in {'TRAIN','VALIDATION'}
               and r['research_eligibility'] == 'RESEARCH_ELIGIBLE'}
    require(len(allow) == 1114 and set(allow) == allowed and
            dict(Counter(r['partition'] for r in allow.values())) ==
            {'TRAIN':845,'VALIDATION':269}, 'allowlist non conforme')
    for mid in split:
        a, e, d, r = split[mid], elig[mid], dataset[mid], registry[mid]
        require(a['kickoff_utc'] == e['kickoff_utc'] == d['kickoff_utc'] == r['kickoff_utc'],
                'date mismatchee dans les sources')
        require(a['partition'] == e['partition'], 'partitions divergentes')
    for mid in allow:
        require(allow[mid]['kickoff_utc'] == split[mid]['kickoff_utc'] and
                allow[mid]['partition'] == split[mid]['partition'],
                'allowlist partitions diverge')
    train_ids = sorted(mid for mid, row in allow.items() if row['partition'] == 'TRAIN')
    test_ids = {mid for mid, row in split.items() if row['partition'] == 'TEST'}
    require(len(train_ids) == 845 and len(test_ids) == 302 and
            not test_ids.intersection(allow), 'TEST dans allowlist')
    selftest()
    print('GARDE : TRAIN 845 | VALIDATION 269 non ouverts | TEST 302 non ouverts')
    print('TESTS SYNTHETIQUES : FRONTIERE 87 + OWN_GOAL + EQUIPE MANQUANTE PASS')
    print('G02-G27/P1 PRECHECK PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 TIMELINE | 0 FIT')
    return train_ids, dataset, registry


def audit(train_ids, dataset, registry):
    print('=== G02-G27/P2 | 845 TIMELINES TRAIN | AUCUN LABEL ===')
    stats = Counter()
    reasons = Counter()
    event_types = Counter()
    samples = defaultdict(list)
    unique_scores = {name: set() for name in CANDIDATES}
    seen = Counter()
    positive = Counter()
    available = Counter()
    sums = Counter()
    accepted_dirs = {folder.resolve() for folder in RAW_DIRS}
    require(all(folder.is_dir() and not folder.is_symlink() for folder in RAW_DIRS),
            'un repertoire RAW est absent ou lien symbolique')
    for mid in train_ids:
        row = dataset[mid]
        reg = registry[mid]
        path = Path(reg['source_path'])
        require(path.stem == mid and path.suffix == '.json' and
                path.is_file() and not path.is_symlink() and
                path.resolve().parent in accepted_dirs,
                'chemin timeline TRAIN incorrect : ' + mid)
        require(sha(path) == reg['source_sha256'], 'hash de timeline TRAIN modifie : ' + mid)
        with path.open('r', encoding='utf-8') as fd:
            wrapper = json.load(fd)
        require(isinstance(wrapper, dict) and wrapper.get('match_id') == mid,
                'ID du JSON incorrect : ' + mid)
        require(str(wrapper.get('home_id')) == str(row['home_team_id']) and
                str(wrapper.get('away_id')) == str(row['away_team_id']) and
                wrapper.get('kickoff_utc') == row['kickoff_utc'],
                'metadonnees teams/date incorrectes : ' + mid)
        response = wrapper.get('timeline_response')
        require(isinstance(response, dict) and isinstance(response.get('data'), dict),
                'timeline_response.data absent')
        events = response['data'].get('events')
        f, issues, types, has_keys, num_keys = reconstruct(events, wrapper, row)
        stats['MATCHS_TRAIN_OUVERTS'] += 1
        stats['EVENTS_META_TIMING_KEYS'] += num_keys
        stats['MATCHS_WITH_TIMING_KEYS'] += bool(has_keys)
        event_types.update(types)
        for reason, count in issues.items():
            reasons[reason] += count
            if len(samples[reason]) < 5:
                samples[reason].append(mid)
        for name in CANDIDATES:
            value = f[name]
            if value is None:
                continue
            require(math.isfinite(float(value)) and float(value) >= 0,
                    'feature invalide : ' + name)
            available[name] += 1
            sums[name] += value
            positive[name] += int(value > 0)
            unique_scores[name].add(value)
        if not issues:
            stats['MATCHS_SANS_ANOMALIE'] += 1
    require(stats['MATCHS_TRAIN_OUVERTS'] == 845, 'nombre de timelines TRAIN different')
    print('TIMELINES TRAIN OUVERTES : 845 / 845 | VALIDATION : 0 | TEST : 0')
    print('ANOMALIES DE RECONSTRUCTION (sans valeur de label) :', dict(sorted(reasons.items())))
    if samples:
        print('EXEMPLES IDs TRAIN, max 5/motif :', dict(samples))
    print('MATCHS SANS ANOMALIE :', stats['MATCHS_SANS_ANOMALIE'])
    print('=== NOUVELLES FEATURES CANDIDATES | COUVERTURE TRAIN ===')
    print('NOM | calculable/845 | >0 parmi calculables | moyenne | distinctes')
    for name in CANDIDATES:
        n = available[name]
        rate = '%5.1f%%' % (100*n/845)
        count_positive = str(positive[name]) if n else 'N/A'
        mean = ('%.3f' % (sums[name]/n)) if n else 'N/A'
        print('%-32s | %3d/845 (%s) | %5s | %7s | %d' %
              (name, n, rate, count_positive, mean, len(unique_scores[name])))
    print('=== EVENEMENTS PRE-87 PAR TYPE (15 PLUS FREQUENTS; PAS DE LABEL) ===')
    for typ, cnt in event_types.most_common(15):
        print('%-30s : %d' % (typ, cnt))
    print('METADONNEES HORODATAGE EVENEMENT (presence de cles seulement) :',
          stats['MATCHS_WITH_TIMING_KEYS'], '/845 matchs ;',
          stats['EVENTS_META_TIMING_KEYS'], 'evenements')
    print('IMPORTANT : meme si des cles de dates sont presentes, aucun AS-OF n est demontre.')
    print('IMPORTANT : calculable retrospectivement != observable en live a la minute 87.')
    print('IMPORTANT : aucune correlation/performance estimee; aucune feature selectionnee.')
    print('0 LABEL | 0 TIMELINE VALIDATION | 0 TIMELINE TEST | 0 API | 0 ECRITURE | 0 FIT')
    print('G02-G27/P2 TRAIN_ONLY_FEATURE_FEASIBILITY COMPLETE | AS-OF NON CERTIFIE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-train', action='store_true',
                        help='examine les evenements PRE87 des seules 845 timelines TRAIN')
    args = parser.parse_args()
    train_ids, dataset, registry = load_guards()
    if args.audit_train:
        audit(train_ids, dataset, registry)
    else:
        print('Apres PASS : python3 -B g02g27_feature_feasibility_train_readonly.py --audit-train')


if __name__ == '__main__':
    main()
