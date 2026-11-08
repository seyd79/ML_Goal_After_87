#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G31 | Inventaire du temps de recuperation des timelines TRAIN.

Ce script NE certifie PAS des donnees as-of. Il cherche simplement une trace
au niveau du fichier historique qui permet de caracteriser sa collecte.

Depuis la racine ML_Goal_After_87 :
  python3 -B g02g31_train_retrieval_provenance_readonly.py
  python3 -B g02g31_train_retrieval_provenance_readonly.py --audit-train

Mode normal: verifie les SHA geles + allowlist, aucune timeline ouverte.
Mode audit: 845 timelines TRAIN seulement; verifie SHA256 puis lit uniquement
le prefixe du JSON jusqu'a la cle top-level timeline_response. Le contenu
'events' et les labels ne sont jamais examines, les timelines VALIDATION/TEST
ne sont pas ouvertes. Aucun appel API, ecriture, entrainement ni modele.

' retrieved_at_utc ' est la date d'une RECUPERATION retrospective d'un
fichier complet, pas l'horodatage de publication de chaque evenement.
Une recuperation pre/coincidant avec la minute 87 ne prouverait rien a elle
seule : il faut des journaux d'observations live horodates et verifiables.
"""
import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import g02g27_feature_feasibility_train_readonly as g27

EXPECTED_G27 = 'fbf1f86ef60f48ee4d33e1dd534335b05854e3a50f37fa571edef399d7bcb3bd'
MARKER = b'"timeline_response"'
MAX_PREFIX = 32768
STAMP_KEYS = ('retrieved_at_utc', 'fetched_at_utc', 'collected_at_utc', 'captured_at_utc')


def require(ok, reason):
    if not ok:
        raise SystemExit('G02-G31 STOP : ' + str(reason))


def parse_utc(value):
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None
    if d.tzinfo is None or d.utcoffset() is None:
        return None
    return d.astimezone(timezone.utc)


def get_literal(prefix, key):
    pat = rb'"' + re.escape(key.encode('ascii')) + rb'"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|\d+)'
    m = re.search(pat, prefix)
    if m is None:
        return None
    raw = m.group(1)
    if raw in (b'true', b'false', b'null'):
        return raw.decode('ascii')
    if raw.startswith(b'"'):
        import json
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if isinstance(value, str) else None
    return raw.decode('ascii')


def header_prefix(path):
    """Jamais lire au-dela de la cle timeline_response (notamment events)."""
    buf = bytearray()
    with path.open('rb') as fd:
        while len(buf) < MAX_PREFIX:
            chunk = fd.read(min(1024, MAX_PREFIX - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
            pos = bytes(buf).find(MARKER)
            if pos >= 0:
                return bytes(buf[:pos]), True
    return b'', False  # fail closed: ne pas analyser un prefixe incertain


def selftest():
    fake = (b'{"match_id":"mt_1","kickoff_utc":"2026-04-01T12:00:00Z",'
            b'"retrieved_at_utc":"2026-09-01T15:00:00+00:00",'
            b'"timeline_response":{"data":{"events":[{"type":"goal"}]}}}')
    prefix = fake.split(MARKER, 1)[0]
    require(get_literal(prefix, 'retrieved_at_utc') == '2026-09-01T15:00:00+00:00',
            'test header timestamp')
    require(b'goal' not in prefix and get_literal(prefix, 'match_id') == 'mt_1',
            'test isolation evenements')
    require(parse_utc('2026-09-01T15:00:00+00:00') is not None,
            'test parsing temps')
    require(parse_utc('2026-09-01T15:00:00') is None,
            'fuseau requis')


def precheck():
    print('=== G02-G31/P1 | PROVENANCE DE RECUPERATION | TRAIN ONLY ===')
    script = Path('g02g27_feature_feasibility_train_readonly.py')
    require(script.is_file() and not script.is_symlink() and g27.sha(script) == EXPECTED_G27,
            'script G02-G27 modifie/absent')
    print('SHA SCRIPT G02-G27 : OK')
    train_ids, dataset, registry = g27.load_guards()
    require(len(train_ids) == 845 and len(dataset) == 1500 and len(registry) == 1500,
            'cohortes inattendues')
    selftest()
    flags = Counter()
    for mid in train_ids:
        row = dataset[mid]
        flags['prematch_asof_certified=' + str(row.get('prematch_asof_certified', '<absent>'))] += 1
        flags['live_publication_timing_certified=' + str(row.get('live_publication_timing_certified', '<absent>'))] += 1
    print('FLAGS DATASET TRAIN :', dict(sorted(flags.items())))
    print('SELFTEST : lecture du prefixe sans evenements PASS')
    print('G02-G31/P1 PRECHECK PASS | 0 LABEL | 0 TIMELINE | 0 API | 0 ECRITURE | 0 FIT')
    return train_ids, dataset, registry


def audit(train_ids, dataset, registry):
    print('=== G02-G31/P2 | TRACES RECUPERATION JSON | TRAIN 845 ===')
    accepted_dirs = {p.resolve() for p in g27.RAW_DIRS}
    require(all(p.is_dir() and not p.is_symlink() for p in g27.RAW_DIRS),
            'repertoires RAW invalides')
    counts, stamp_field, boolean_flags, sample = Counter(), Counter(), Counter(), []
    lags = []
    for mid in train_ids:
        p = Path(registry[mid]['source_path'])
        require(p.is_file() and not p.is_symlink() and p.stem == mid and
                p.suffix == '.json' and p.resolve().parent in accepted_dirs,
                'fichier TRAIN hors contrat : ' + mid)
        require(g27.sha(p) == registry[mid]['source_sha256'],
                'SHA timeline incorrect : ' + mid)
        prefix, found = header_prefix(p)
        counts['SHA_VERIFIED'] += 1
        if not found:
            counts['HEADER_MARKER_ABSENT_OR_TOO_LATE'] += 1
            continue
        counts['HEADER_ONLY_PARSED'] += 1
        require(get_literal(prefix, 'match_id') == mid,
                'ID incoherent dans prefixe : ' + mid)
        k = get_literal(prefix, 'kickoff_utc')
        kickoff = parse_utc(k)
        require(kickoff is not None and kickoff == parse_utc(dataset[mid]['kickoff_utc']),
                'date kickoff incoherente : ' + mid)
        stamps = [(name, get_literal(prefix, name)) for name in STAMP_KEYS]
        usable = [(name, val) for name, val in stamps if val is not None]
        if not usable:
            counts['NO_COLLECTION_TIMESTAMP'] += 1
            continue
        counts['HAS_COLLECTION_TIMESTAMP'] += 1
        for name, _ in usable:
            stamp_field[name] += 1
        first_key, first_value = usable[0]  # sans pretendre equivalence des champs
        retrieved = parse_utc(first_value)
        if retrieved is None:
            counts['TIMESTAMP_INVALID_OR_NO_TIMEZONE'] += 1
            continue
        lag_h = (retrieved - kickoff).total_seconds() / 3600
        lags.append(lag_h)
        if lag_h < 0:
            counts['RETRIEVED_BEFORE_KICKOFF'] += 1
        elif lag_h <= 3:
            counts['RETRIEVED_KICKOFF_PLUS_0_TO_3H'] += 1
        elif lag_h <= 6:
            counts['RETRIEVED_KICKOFF_PLUS_3_TO_6H'] += 1
        elif lag_h <= 24:
            counts['RETRIEVED_KICKOFF_PLUS_6_TO_24H'] += 1
        else:
            counts['RETRIEVED_OVER_24H_AFTER_KICKOFF'] += 1
        for key in ('training_certified', 'label_certified', 'live_certified', 'research_status'):
            val = get_literal(prefix, key)
            if val is not None:
                boolean_flags[key + '=' + val] += 1
    require(counts['SHA_VERIFIED'] == 845, 'timelines TRAIN != 845')
    print('TIMELINES TRAIN SHA CONFORMES :', counts['SHA_VERIFIED'], '| VALIDATION 0 | TEST 0')
    print('PREFIXES INTERPRETABLES :', counts['HEADER_ONLY_PARSED'])
    print('CHAMPS DE CAPTURE PRESENTS :', dict(sorted(stamp_field.items())))
    for k in ('HEADER_MARKER_ABSENT_OR_TOO_LATE', 'NO_COLLECTION_TIMESTAMP',
              'HAS_COLLECTION_TIMESTAMP', 'TIMESTAMP_INVALID_OR_NO_TIMEZONE',
              'RETRIEVED_BEFORE_KICKOFF', 'RETRIEVED_KICKOFF_PLUS_0_TO_3H',
              'RETRIEVED_KICKOFF_PLUS_3_TO_6H', 'RETRIEVED_KICKOFF_PLUS_6_TO_24H',
              'RETRIEVED_OVER_24H_AFTER_KICKOFF'):
        print('%-42s : %d' % (k, counts[k]))
    if lags:
        lags.sort()
        print('DELAI collecte - kickoff (heures) : min %.2f | mediane %.2f | max %.2f' %
              (lags[0], lags[len(lags)//2], lags[-1]))
    print('DECLARATIONS METADATA SOURCE :', dict(sorted(boolean_flags.items())))
    print('INTERPRETATION : retrieved_at_utc = date de recuperation du fichier complet;')
    print('                   ce n est PAS une preuve de publication des evenements pre-87.')
    print('CONCLUSION : AS-OF NON CERTIFIE, y compris si un horodatage existe.')
    print('AUCUN resultat de match, type evenements, label ou modele analyse.')
    print('0 API | 0 ECRITURE | 0 FIT | AUCUNE TIMELINE VAL/TEST')
    print('G02-G31/P2 RETRIEVAL_PROVENANCE_AUDIT COMPLETE | RESEARCH ONLY')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-train', action='store_true')
    args = parser.parse_args()
    train_ids, dataset, registry = precheck()
    if args.audit_train:
        audit(train_ids, dataset, registry)
    else:
        print('Puis : python3 -B g02g31_train_retrieval_provenance_readonly.py --audit-train')


if __name__ == '__main__':
    main()
