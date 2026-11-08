#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2F: audit des FORMATS de minutes dans TRAIN+VALIDATION SEULEMENT.

Execution depuis la racine ML_Goal_After_87:
    python3 -B g02g20_p2f_train_val_minute_schema_readonly.py

Protections:
 - Controle 4 empreintes SHA256 des artefacts geles.
 - Autorise exclusivement les 1114 identifiants TRAIN/VALIDATION.
 - Aucune timeline TEST ouverte, aucune lecture du CSV labels.
 - Ne consulte dans les evenements que period, minute, extra_time.
 - Ne consulte JAMAIS type d'evenement, buts, scores, labels, autres donnees.
 - Aucune ecriture, aucun appel API, aucun entrainement.

Note: parser un fichier JSON charge bien ses evenements en memoire, mais le
programme ne consulte et ne rapporte aucune valeur de resultat ni de label.
Les compteurs sont exclusivement des diagnostics de format, pas des labels.
"""
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
EXPECTED = {
    SPLIT: 'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    ELIG: '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    ALLOW: '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
    PROTO: '139d42354487a52a38ce058e70545b15131ef00892a348402497efb0abbaf73c',
}
RAW_DIRS = (
    BASE / 'v41_mass350/raw_timelines',
    BASE / 'v42_g02d2_pilot5/raw_timelines',
    BASE / 'v42_g02d3_coverage24/raw_timelines',
    BASE / 'v42_g02d5_collection/raw_timelines',
)
PERIODS = {'first_half', 'second_half', 'extra_time_first_half',
           'extra_time_second_half', 'penalties', 'shootout',
           'penalty_shootout', 'penalty_shoot-out'}


def require(ok, message):
    if not ok:
        raise SystemExit('G02-G20/P2F STOP : ' + message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fd:
        for block in iter(lambda: fd.read(256 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path, required_fields):
    with path.open('r', encoding='utf-8-sig', newline='') as fd:
        r = csv.DictReader(fd)
        require(r.fieldnames is not None and len(r.fieldnames) == len(set(r.fieldnames)),
                'colonnes CSV invalides : ' + path.name)
        require(set(required_fields) <= set(r.fieldnames),
                'colonnes requises absentes : ' + path.name)
        vals = list(r)
    require(all(None not in row and row['match_id'] for row in vals),
            'ligne invalide dans ' + path.name)
    ids = [x['match_id'] for x in vals]
    require(len(ids) == len(set(ids)), 'doublons dans ' + path.name)
    return {x['match_id']: x for x in vals}


def number(value):
    """Miroir strict de float(value) + garde finie pour diagnostic seulement."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def scan_event_metadata(events):
    """Accede UNIQUEMENT aux trois champs temporels (jamais type/goal/score)."""
    c = Counter()
    require(isinstance(events, list), 'events n est pas une liste')
    c['EVENTS'] += len(events)
    for e in events:
        if not isinstance(e, dict):
            c['INVALID_EVENT_SHAPE'] += 1
            continue
        period = str(e.get('period') or '').lower()
        if period not in PERIODS:
            c['PERIOD_UNKNOWN'] += 1
        if period == 'second_half':
            c['SECOND_HALF_EVENTS'] += 1
        raw_minute = e.get('minute')
        if raw_minute is None:
            c['MINUTE_MISSING'] += 1
        elif isinstance(raw_minute, str) and re.search(r'\d\s*\+\s*\d', raw_minute):
            c['MINUTE_PLUS_STRING'] += 1
        m = number(raw_minute)
        if m is None:
            c['MINUTE_UNPARSEABLE_OR_MISSING'] += 1
            continue
        c['MINUTE_NUMERIC_PARSEABLE'] += 1
        if period != 'second_half':
            continue
        raw_extra = e.get('extra_time')
        ext = 0.0 if raw_extra is None or raw_extra == '' else number(raw_extra)
        if ext is None:
            c['SECOND_HALF_EXTRA_TIME_UNPARSEABLE'] += 1
            continue
        if ext > 0:
            c['SECOND_HALF_EXTRA_TIME_POSITIVE'] += 1
        if m == 87:
            c['SECOND_HALF_MINUTE_EQ87'] += 1
            if ext > 0:
                c['SECOND_HALF_MINUTE_EQ87_EXTRA_POSITIVE'] += 1
        if m > 87:
            c['SECOND_HALF_MINUTE_GT87'] += 1
        if m < 87 and ext > 0 and m + ext >= 87:
            c['SECOND_HALF_MINUTE_LT87_EXTRA_REACHES87'] += 1
        if m < 87 and ext > 0 and m + ext > 87:
            c['SECOND_HALF_MINUTE_LT87_EXTRA_EXCEEDS87'] += 1
    return c


def main():
    print('=== G02-G20/P2F | FORMATS MINUTE TRAIN+VALIDATION | READONLY ===')
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), 'garde manquante/lien : ' + str(path))
        require(sha(path) == expected, 'SHA garde non conforme : ' + path.name)
        print('GARDE SHA OK :', path.name)

    split = csv_rows(SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = csv_rows(ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    allow = csv_rows(ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    require(len(split) == 1500 and len(elig) == 1500 and set(split) == set(elig),
            'split/eligibilite incompatibles')
    require(len(allow) == 1114, 'allowlist pas 1114')
    require(sum(x['partition'] == 'TRAIN' for x in split.values()) == 912 and
            sum(x['partition'] == 'VALIDATION' for x in split.values()) == 286 and
            sum(x['partition'] == 'TEST' for x in split.values()) == 302,
            'effectif split incorrect')
    require(sum(x['partition'] == 'TRAIN' for x in allow.values()) == 845 and
            sum(x['partition'] == 'VALIDATION' for x in allow.values()) == 269,
            'effectif allowlist incorrect')
    expected_allowed = {
        mid for mid, row in elig.items()
        if row['research_eligibility'] == 'RESEARCH_ELIGIBLE'
        and row['partition'] in {'TRAIN', 'VALIDATION'}
    }
    require(set(allow) == expected_allowed, 'IDs autorises != eligibles TRAIN+VALIDATION')
    for mid in allow:
        require(split[mid]['partition'] == allow[mid]['partition'] == elig[mid]['partition'] and
                split[mid]['kickoff_utc'] == allow[mid]['kickoff_utc'] == elig[mid]['kickoff_utc'],
                'jointure incoherente : ' + mid)
    test_ids = {mid for mid, row in split.items() if row['partition'] == 'TEST'}
    require(len(test_ids) == 302 and not set(allow).intersection(test_ids),
            'ERREUR : identifiant TEST autorise')
    print('ALLOWLIST OK : TRAIN 845 | VALIDATION 269 | TEST 0')

    # Noms uniquement pour les 1500 timelines ; seuls les 1114 IDs autorises
    # peuvent ensuite etre OUVERTS. Aucune lecture de fichier TEST.
    located = {}
    for folder in RAW_DIRS:
        require(folder.is_dir() and not folder.is_symlink(), 'dossier absent/lien : ' + str(folder))
        for p in folder.iterdir():
            if not (p.name.startswith('mt_') and p.suffix == '.json'):
                continue
            require(not p.is_symlink() and p.is_file(), 'lien/non fichier timeline : ' + str(p))
            require(p.stem not in located, 'timeline nom duplique : ' + p.stem)
            located[p.stem] = p
    require(len(located) == 1500 and set(located) == set(split),
            'couverture timelines != split des 1500')
    total = Counter()
    by_partition = {'TRAIN': Counter(), 'VALIDATION': Counter()}
    for mid in sorted(allow):
        require(mid not in test_ids, 'tentative ouverture TEST')
        path = located[mid]
        require(not path.is_symlink(), 'timeline lien symbolique : ' + mid)
        with path.open('r', encoding='utf-8') as fd:
            wrapper = json.load(fd)
        require(isinstance(wrapper, dict) and wrapper.get('match_id') == mid,
                'timeline identifiant/schema invalide : ' + mid)
        response = wrapper.get('timeline_response')
        require(isinstance(response, dict), 'timeline_response absent : ' + mid)
        data = response.get('data')
        require(isinstance(data, dict), 'timeline.data absent : ' + mid)
        values = scan_event_metadata(data.get('events'))
        by_partition[allow[mid]['partition']].update(values)
        total.update(values)
        total['MATCHES_READ_TRAIN_VAL'] += 1
    print('TIMELINES OUVERTES :', total['MATCHES_READ_TRAIN_VAL'], '| TEST OUVERT : 0')
    fields = [
        'MATCHES_READ_TRAIN_VAL', 'EVENTS', 'INVALID_EVENT_SHAPE',
        'MINUTE_MISSING', 'MINUTE_PLUS_STRING',
        'MINUTE_UNPARSEABLE_OR_MISSING', 'MINUTE_NUMERIC_PARSEABLE',
        'PERIOD_UNKNOWN', 'SECOND_HALF_EVENTS',
        'SECOND_HALF_EXTRA_TIME_UNPARSEABLE',
        'SECOND_HALF_EXTRA_TIME_POSITIVE',
        'SECOND_HALF_MINUTE_EQ87',
        'SECOND_HALF_MINUTE_EQ87_EXTRA_POSITIVE',
        'SECOND_HALF_MINUTE_GT87',
        'SECOND_HALF_MINUTE_LT87_EXTRA_REACHES87',
        'SECOND_HALF_MINUTE_LT87_EXTRA_EXCEEDS87',
    ]
    for name in ('TRAIN', 'VALIDATION', 'TOTAL'):
        c = total if name == 'TOTAL' else by_partition[name]
        print('---', name, '| FORMATS TEMPORELS (PAS DE BUTS NI LABELS) ---')
        for key in fields:
            print(key, ':', c[key])
    print('NOTE : somme minute+extra_time = DIAGNOSTIC de frontiere, pas temps reel certifie')
    print('NOTE : meme si 0 notation +, la provenance des labels reste non certifiee')
    print('TEST FERME | 0 CSV LABEL | 0 TYPE EVENEMENT | 0 SCORE | 0 API | 0 ECRITURE | 0 TRAIN')
    print('G02-G20/P2F MINUTE_FORMAT_AUDIT PASS | TRAIN+VALIDATION UNIQUEMENT')


if __name__ == '__main__':
    main()
