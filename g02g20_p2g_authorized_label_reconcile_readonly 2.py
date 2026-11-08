#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2G | Audit controle des labels TRAIN+VALIDATION, TEST ferme.

PRECHECK : python3 -B g02g20_p2g_authorized_label_reconcile_readonly.py
AUDIT   : python3 -B g02g20_p2g_authorized_label_reconcile_readonly.py --audit-authorized

PRECHECK lit uniquement les artefacts de garde, le rapport F4, le code builder,
le nom/taille et l'EN-TETE du CSV labels, et les NOMS des timelines.
AUDIT ouvre exclusivement les 1114 timelines autorisees (845 TRAIN, 269 VALIDATION),
recalcule la cible selon la regle du constructeur F4 et lit/parse exclusivement
les CINQ colonnes des 1114 lignes autorisees du CSV des labels. Les lignes CSV
TEST et les lignes exclues ne sont PAS interpretees au-dela de leur match_id;
les 302 timelines TEST ne sont jamais ouvertes. Pas de SHA sur tout le CSV labels
car cela imposerait de lire les octets des lignes TEST.

Ceci est un controle de COHERENCE avec une meme source retrospective, PAS une
certification independante des resultats, ni une certification as-of.
0 API | 0 ecriture | 0 entrainement. Python 3.9+, bibliotheque standard.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

BASE = Path('outputs/v4_research')
SPLIT = BASE / 'v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv'
ELIG = BASE / 'v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv'
ALLOW = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv'
PROTO = BASE / 'v42_g02g19_train_preprocessing_guard_v1/G02G19_TRAIN_ONLY_STANDARD_SCALER_PROTOCOL_FROZEN.json'
GUARDS = {
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    PROTO: '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
}
SOURCE_DIR = BASE / 'v42_g02f4_reconstruction'
CSV_LABELS = SOURCE_DIR / 'V42_LABELS_RESEARCH.csv'
REPORT_F4 = SOURCE_DIR / 'V42_RECONSTRUCTION_REPORT.json'
BUILDER = Path('build_historical_ab_research_v3.py')
DIRS = (
    BASE / 'v41_mass350/raw_timelines',
    BASE / 'v42_g02d2_pilot5/raw_timelines',
    BASE / 'v42_g02d3_coverage24/raw_timelines',
    BASE / 'v42_g02d5_collection/raw_timelines',
)
COLUMNS = (
    'match_id', 'goal_after_87', 'goal_at_87',
    'extra_time_goals_excluded', 'label_certified_independently',
)
GOALS = {'goal', 'own_goal', 'penalty_scored'}
EXTRA_TIME = {'extra_time_first_half', 'extra_time_second_half'}
VALID_PERIODS = {'first_half', 'second_half'} | EXTRA_TIME | {
    'penalties', 'shootout', 'penalty_shootout', 'penalty_shoot-out',
}
MID_RE = re.compile(r'^mt_[A-Za-z0-9_-]+$')


def require(flag, reason):
    if not flag:
        raise SystemExit('G02-G20/P2G STOP : ' + reason)


def file_ok(p):
    require(p.is_file() and not p.is_symlink(), 'fichier inexistant ou lien symbolique : ' + str(p))


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(128 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def index_csv(path, required):
    file_ok(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and
                len(reader.fieldnames) == len(set(reader.fieldnames)) and
                set(required) <= set(reader.fieldnames), 'colonnes garde invalides : ' + path.name)
        rows = list(reader)
    out = {}
    for row in rows:
        mid = row.get('match_id', '')
        require(mid and mid not in out and None not in row, 'IDs/lignes garde invalides : ' + path.name)
        out[mid] = row
    return out


def read_header_only(path):
    file_ok(path)
    with path.open('rb') as f:
        header = f.readline(65536)
    require(header.endswith(b'\n'), 'en-tete CSV absent ou trop long')
    try:
        columns = next(csv.reader([header.decode('utf-8-sig')]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise SystemExit('G02-G20/P2G STOP : en-tete illisible : ' + repr(exc))
    require(tuple(columns) == COLUMNS, 'structure CSV labels inattendue : ' + repr(columns))
    return columns


def check_sources():
    for path, want in GUARDS.items():
        file_ok(path)
        require(sha256(path) == want, 'SHA garde incompatible : ' + path.name)
        print('GARDE SHA OK :', path.name)
    file_ok(REPORT_F4)
    file_ok(BUILDER)
    with REPORT_F4.open(encoding='utf-8-sig') as f:
        report = json.load(f)
    require(isinstance(report, dict), 'rapport F4 non-objet')
    want = report.get('builder_sha256')
    require(isinstance(want, str) and re.fullmatch('[a-f0-9]{64}', want) is not None,
            'builder_sha256 absent ou invalide')
    require(sha256(BUILDER) == want, 'constructeur modifie depuis rapport F4')
    print('BUILDER SHA : MATCH F4')
    read_header_only(CSV_LABELS)
    print('CSV LABELS : en-tete confirme (aucune valeur lue) | octets :', CSV_LABELS.stat().st_size)


def check_access():
    split = index_csv(SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = index_csv(ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    allow = index_csv(ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    require(len(split) == 1500 and len(elig) == 1500 and set(split) == set(elig),
            'split/elig pas 1500 coherents')
    require(dict(Counter(r['partition'] for r in split.values())) ==
            {'TRAIN': 912, 'VALIDATION': 286, 'TEST': 302}, 'effectifs split changes')
    allowed_expected = {mid for mid, r in elig.items()
                        if r['research_eligibility'] == 'RESEARCH_ELIGIBLE'
                        and r['partition'] in {'TRAIN', 'VALIDATION'}}
    require(set(allow) == allowed_expected and len(allow) == 1114,
            'allowlist non conforme aux eligibility')
    require(dict(Counter(r['partition'] for r in allow.values())) ==
            {'TRAIN': 845, 'VALIDATION': 269}, 'repartition allowlist incorrecte')
    for mid, row in allow.items():
        require(row['partition'] == split[mid]['partition'] == elig[mid]['partition']
                and row['kickoff_utc'] == split[mid]['kickoff_utc'] == elig[mid]['kickoff_utc'],
                'jointure incoherente : ' + mid)
    test_ids = {mid for mid, r in split.items() if r['partition'] == 'TEST'}
    require(len(test_ids) == 302 and not test_ids.intersection(allow),
            'TEST dans la liste autorisee')
    print('ALLOWLIST : TRAIN 845 | VALIDATION 269 | TEST 0')
    return split, allow


def map_timelines(split):
    found = {}
    for folder in DIRS:
        require(folder.is_dir() and not folder.is_symlink(), 'dossier timeline absent : ' + str(folder))
        for p in folder.iterdir():
            if not (p.name.startswith('mt_') and p.suffix == '.json'):
                continue
            file_ok(p)
            require(p.stem not in found, 'nom timeline en double : ' + p.stem)
            found[p.stem] = p
    require(len(found) == 1500 and set(found) == set(split),
            'identifiants timelines ne correspondent pas au split')
    print('TIMELINES (NOMS SEULEMENT) : 1500 / 1500')
    return found


def parse_minute(v):
    require(not isinstance(v, bool), 'minute booleenne')
    try:
        x = float(v)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('minute non numerique')
    if not math.isfinite(x) or x < 0:
        raise ValueError('minute invalide')
    return x


def outcome_from_timeline(path, mid):
    """Miroir des regles label F4; ne lit que les timelines autorisees."""
    file_ok(path)
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    require(isinstance(obj, dict) and obj.get('match_id') == mid,
            'ID timeline invalide : ' + mid)
    try:
        data = obj['timeline_response']['data']
        events = data['events']
    except (TypeError, KeyError):
        raise SystemExit('G02-G20/P2G STOP : structure timeline invalide : ' + mid)
    require(isinstance(events, list), 'events non liste : ' + mid)
    late = 0
    at87 = 0
    extra_goals = 0
    unknown_periods = 0
    invalid_events = 0
    for event in events:
        if not isinstance(event, dict):
            invalid_events += 1
            continue
        typ = str(event.get('type') or '').lower()
        period = str(event.get('period') or '').lower()
        if period in EXTRA_TIME:
            if typ in GOALS:
                extra_goals += 1
            continue
        if period not in VALID_PERIODS:
            unknown_periods += 1
            continue
        if period != 'second_half' or typ not in GOALS:
            continue
        try:
            minute = parse_minute(event.get('minute'))
        except ValueError:
            raise SystemExit('G02-G20/P2G STOP : minute but invalide : ' + mid)
        if minute == 87:
            at87 += 1
        elif minute > 87:
            late += 1
    return {
        'goal_after_87': int(late > 0),
        'goal_at_87': int(at87 > 0),
        'extra_time_goals_excluded': extra_goals,
        'unknown_periods': unknown_periods,
        'invalid_events': invalid_events,
    }


def read_authorized_csv_rows(path, allow, split):
    """Parse seulement les lignes dont la PREMIERE cellule est un ID autorise.

    Pour les lignes hors allowlist, seul le prefixe match_id est decode.
    La ligne physique doit etre simple (un enregistrement par ligne), comme
    dans le CSV de labels de recherche observe. Pas de SHA global du fichier.
    """
    authorized = {}
    seen = set()
    with path.open('rb') as fd:
        header = fd.readline(65536)
        require(header.endswith(b'\n'), 'en-tete absent')
        for line in fd:
            require(line.endswith(b'\n'), 'ligne CSV sans terminateur')
            raw_mid = line.partition(b',')[0].strip(b'"')
            try:
                mid = raw_mid.decode('ascii')
            except UnicodeDecodeError:
                require(False, 'match_id non ascii dans fichier labels')
            require(MID_RE.fullmatch(mid) is not None and mid in split,
                    'match_id inattendu dans CSV labels')
            require(mid not in seen, 'match_id double dans CSV labels : ' + mid)
            seen.add(mid)
            if mid not in allow:
                # Aucun decodage/parsing des colonnes contenant les labels TEST.
                continue
            try:
                row = next(csv.reader([line.decode('utf-8')]))
            except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
                raise SystemExit('G02-G20/P2G STOP : ligne autorisee invalide : ' + repr(exc))
            require(len(row) == len(COLUMNS) and row[0] == mid,
                    'ligne autorisee CSV non conforme : ' + mid)
            authorized[mid] = dict(zip(COLUMNS, row))
    require(len(seen) == 1500 and seen == set(split), 'CSV de labels sans couverture 1500')
    require(len(authorized) == 1114 and set(authorized) == set(allow),
            'coverage CSV autorisee != 1114')
    return authorized


def parse_binary(value, name):
    require(value in {'0', '1'}, 'valeur binaire incorrecte : ' + name)
    return int(value)


def parse_count(value, name):
    require(value.isdigit(), 'nombre entier invalide : ' + name)
    return int(value)


def audit_allowed(timelines, split, allow):
    print('=== G02-G20/P2G | AUDIT LABELS AUTORISES | TEST FERME ===')
    labels = read_authorized_csv_rows(CSV_LABELS, allow, split)
    counts = {'TRAIN': Counter(), 'VALIDATION': Counter()}
    mismatch = Counter()
    for mid in sorted(allow):
        require(split[mid]['partition'] in {'TRAIN', 'VALIDATION'}, 'ouverture TEST refusee')
        observed = outcome_from_timeline(timelines[mid], mid)
        csvrow = labels[mid]
        p = allow[mid]['partition']
        counts[p]['MATCHS'] += 1
        counts[p]['POSITIFS_RECALCULES'] += observed['goal_after_87']
        counts[p]['GOALS_AT_87_MATCHS_RECALCULES'] += observed['goal_at_87']
        counts[p]['EXTRA_TIME_GOALS_EXCLUS_RECALCULES'] += observed['extra_time_goals_excluded']
        counts[p]['UNKNOWN_PERIOD_EVENTS'] += observed['unknown_periods']
        counts[p]['INVALID_EVENT_SHAPES'] += observed['invalid_events']
        for field in ('goal_after_87', 'goal_at_87', 'extra_time_goals_excluded'):
            val = parse_count(csvrow[field], field)
            if field != 'extra_time_goals_excluded':
                require(val in {0, 1}, 'valeur label binaire invalide')
            if observed[field] != val:
                mismatch[field] += 1
        cert = parse_binary(csvrow['label_certified_independently'],
                            'label_certified_independently')
        counts[p]['FLAG_INDEPENDENT_CERTIFIED_1'] += cert
    for part in ('TRAIN', 'VALIDATION'):
        c = counts[part]
        print(part, '| matchs', c['MATCHS'], '| positifs recalcules',
              c['POSITIFS_RECALCULES'], '| flag certification=1',
              c['FLAG_INDEPENDENT_CERTIFIED_1'])
        print(part, '| buts prolongation exclus', c['EXTRA_TIME_GOALS_EXCLUS_RECALCULES'],
              '| evenements periode inconnue', c['UNKNOWN_PERIOD_EVENTS'],
              '| evenements malformes', c['INVALID_EVENT_SHAPES'])
    print('DIVERGENCES RECALCUL F4 vs CSV :', dict(mismatch))
    require(not mismatch, 'des labels autorises different du recalcul des timelines')
    require(counts['TRAIN']['MATCHS'] == 845 and counts['VALIDATION']['MATCHS'] == 269,
            'effectifs audites incorrects')
    print('IMPORTANT : accord CSV/timelines != certification independante ni as-of')
    print('TEST FERME : 302 timelines non ouvertes, 0 label TEST interprete')
    print('0 API | 0 ECRITURE | 0 MODELE')
    print('G02-G20/P2G AUTHORIZED_LABEL_RECONCILIATION PASS | TRAIN+VALIDATION 1114')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-authorized', action='store_true',
                        help='ouvrir seulement les labels TRAIN 845 + VALIDATION 269; jamais TEST')
    args = parser.parse_args()
    print('=== G02-G20/P2G | GARDE SOURCES | READONLY ===')
    check_sources()
    split, allow = check_access()
    timeline_paths = map_timelines(split)
    if not args.audit_authorized:
        print('TEST FERME | 0 VALEUR DE LABEL LUE | 0 TIMELINE OUVERTE | 0 API | 0 ECRITURE | 0 TRAIN')
        print('G02-G20/P2G PRECHECK PASS | labels non ouverts')
        print('Pour auditer explicitement 1114 labels TRAIN+VALIDATION :')
        print('  python3 -B g02g20_p2g_authorized_label_reconcile_readonly.py --audit-authorized')
        return
    audit_allowed(timeline_paths, split, allow)


if __name__ == '__main__':
    main()
