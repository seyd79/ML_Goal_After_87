#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G37 | Coherence des captures des archives V3C (recherche seulement).

Depuis la racine du projet ML_Goal_After_87 :
  python3 -B g02g37_archive_capture_internal_consistency_readonly.py
  python3 -B g02g37_archive_capture_internal_consistency_readonly.py --audit-archives

PRECHECK : SHA256 G02-G36, garde fichiers V4.2 geles, selftests synthetiques.
AUDIT : trois CSV historiques + CSV prospectif V4 autorises exclusivement.

Lit seulement les champs du PREFIXE fixe : match_id, snapshot_timestamp_utc,
minute (pour V4 : match_id, kickoff_utc, captured_at_utc, snapshot_minute).
Ne decode jamais score, label, prediction, odds ni feature. Pas de timeline,
pas de fichier de resultat, pas de modele, pas de reseau et pas d'ecriture.

Ce diagnostic identifie la duplication des archives, les captures distinctes
avant 87 et les incoherences d'horloge interne. Il ne certifie PAS : heure
reelle de kickoff, authenticite de capture, disponibilite fournisseur as-of,
independance contre VAL/TEST, fiabilite des labels ou precision du modele.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
from itertools import combinations
from pathlib import Path

G36_FILE = Path('g02g36_archive_capture_metadata_readonly.py')
G36_SHA256 = '5263861f93428e4b071521993a08136ae13173b69662c37f6483055424b05ee2'


def require(ok, msg):
    if not ok:
        raise SystemExit('G02-G37 STOP : ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_sequence(records):
    """Compare les captures d'un match, uniquement (UTC, minute).

    Intervalles consécutifs par horodatage, sans dédupliquer deux archives
    par hypothèse : les triplets strictement identiques sont dédupliqués.
    Une anomalie de cadence reste un INDICE, jamais une preuve de fraude.
    """
    ordered = sorted(set(records))
    out = Counter()
    pre_87 = {(t, m) for t, m in ordered if 80 <= m < 87}
    out['pre87_samples'] = len(pre_87)
    if len(pre_87) >= 2:
        out['multi_pre87_match'] = 1
    if any(m == 87 for _, m in ordered):
        out['has_exactly87'] = 1
    for (t1, m1), (t2, m2) in zip(ordered, ordered[1:]):
        secs = (t2 - t1).total_seconds()
        dmin = m2 - m1
        out['pairs_total'] += 1
        if secs == 0 and dmin != 0:
            out['same_time_different_minute'] += 1
        if secs > 0:
            if dmin < 0:
                out['minute_backward_pairs'] += 1
            elif dmin == 0:
                out['minute_same_pairs'] += 1
            else:
                out['minute_forward_pairs'] += 1
            # Strictly advisory flags; slow update/corrections/paused play possible.
            if dmin >= 2 and secs < 10:
                out['fast_minute_jump_2plus_in_under10s'] += 1
            if secs > 1800 and dmin <= 7:
                out['over30m_between_nearby_minutes'] += 1
    return out


def selftest():
    base = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
    a = compare_sequence([(base, 82), (base + timedelta(minutes=3), 85),
                          (base + timedelta(minutes=4), 86),
                          (base + timedelta(minutes=4), 86)])
    require(a['pre87_samples'] == 3 and a['multi_pre87_match'] == 1,
            'selftest captures multiples')
    require(a['minute_backward_pairs'] == 0 and a['pairs_total'] == 2,
            'selftest chronologie saine')
    b = compare_sequence([(base, 86), (base + timedelta(minutes=1), 85)])
    require(b['minute_backward_pairs'] == 1, 'selftest minute decroissante')
    c = compare_sequence([(base, 84), (base, 86),
                          (base + timedelta(seconds=3), 88)])
    require(c['same_time_different_minute'] == 1, 'selftest timestamps egaux')
    require(c['fast_minute_jump_2plus_in_under10s'] == 1,
            'selftest saut de minute rapide')
    d = compare_sequence([(base, 86), (base + timedelta(minutes=40), 87)])
    require(d['over30m_between_nearby_minutes'] == 1, 'selftest long intervalle')


def precheck():
    require(G36_FILE.is_file() and not G36_FILE.is_symlink(),
            'G02-G36 absent ou lien symbolique')
    require(sha256(G36_FILE) == G36_SHA256,
            'G02-G36 a change ; ne pas contourner la chaine SHA')
    import g02g36_archive_capture_metadata_readonly as g36
    g34, root, train_ids = g36.precheck()
    selftest()
    print('=== G02-G37/P1 | COHERENCE DES CAPTURES | PRECHECK ===')
    print('SHA G02-G36 : OK | TESTS SYNTHETIQUES cadence/duplication : PASS')
    print('SOURCES : 3 archives V3C + prospectif vide potentiel; aucune source externe')
    print('GARDE : TRAIN 845 (intersections IDs seulement), VAL/TEST non consultes')
    print('COLONNES AUTORISEES : match_id + horodatage UTC + minute; kickoff si present')
    print('G02-G37/P1 PRECHECK PASS | 0 ARCHIVE PARCOURUE | 0 LABEL | 0 TIMELINE | 0 ECRITURE | 0 FIT | 0 API')
    return g34, root, train_ids


def scan_one(g34, root, config):
    import g02g33_snapshot_csv_schema_readonly as g33
    rel, prefix, idx_mid, idx_ts, idx_min, idx_ko = config
    path = g33.secure_file(root, rel)
    header = g33.read_one_csv_header(path)
    require(tuple(header[:len(prefix)]) == tuple(prefix), 'schema change : ' + rel)
    take = max(idx_mid, idx_ts, idx_min, (idx_ko if idx_ko is not None else -1)) + 1
    require(take <= len(prefix), 'prefixe de lecture trop large')
    records = defaultdict(list)
    rows = 0
    invalid_date = 0
    invalid_minute = 0
    with path.open('rb') as f:
        first_line = f.readline(65536)
        require(first_line.endswith(b'\n'), 'en-tete non termine')
        for line in f:
            rows += 1
            require(rows <= g34.MAX_ROWS_PER_FILE, 'limite de lignes : ' + rel)
            fields = g34.parse_csv_initial_fields(line, take)
            mid = fields[idx_mid].strip()
            require(mid, 'ID vide : ' + rel)
            ts = g34.parse_utc(fields[idx_ts])
            minute = g34.parse_minute(fields[idx_min])
            if ts is None:
                invalid_date += 1
            if minute is None:
                invalid_minute += 1
            if ts is not None and minute is not None:
                records[mid].append((ts, minute))
    return {'rel': rel, 'rows': rows, 'records': records,
            'bad_date': invalid_date, 'bad_minute': invalid_minute}


def audit(g34, root, train_ids):
    require(len(g34.SOURCES) == 4, 'liste des quatre archives modifiee')
    print('=== G02-G37/P2 | COHERENCE INTERNE ET RECOUVREMENT | ARCHIVES SEULEMENT ===')
    sources = [scan_one(g34, root, cfg) for cfg in g34.SOURCES]
    all_records = defaultdict(list)
    all_ids = set()
    unique_records_by_source = {}
    for s in sources:
        rec = s['records']
        unique_records_by_source[s['rel']] = set(
            (mid, ts, minute)
            for mid, entries in rec.items()
            for ts, minute in entries
        )
        c = Counter()
        for mid, captures in rec.items():
            metric = compare_sequence(captures)
            if metric['minute_backward_pairs']:
                c['ids_backward'] += 1
            if metric['same_time_different_minute']:
                c['ids_same_timestamp_conflict'] += 1
            if metric['fast_minute_jump_2plus_in_under10s']:
                c['ids_fast_jump'] += 1
            if metric['over30m_between_nearby_minutes']:
                c['ids_large_gap'] += 1
            c.update(metric)
            all_records[mid].extend(captures)
        all_ids.update(rec)
        print('\nSOURCE :', s['rel'])
        print('  LIGNES :', s['rows'], '| IDS avec captures valides :', len(rec))
        print('  HORODATAGES INVALIDES :', s['bad_date'], '| MINUTES INVALIDES :', s['bad_minute'])
        print('  IDS avec >=2 observations DISTINCTES [80,87[ :', c['multi_pre87_match'])
        print('  INTERVALLES :', c['pairs_total'], '| minute en hausse :',
              c['minute_forward_pairs'], '| minute identique :', c['minute_same_pairs'],
              '| minute en baisse :', c['minute_backward_pairs'])
        print('  IDS baisse de minute :', c['ids_backward'],
              '| IDS meme timestamp avec minutes differentes :', c['ids_same_timestamp_conflict'])
        print('  INDICES A EXAMINER (pas preuves) : saut >=2 min en <10 s',
              c['fast_minute_jump_2plus_in_under10s'],
              '| intervalle >30 min avec delta minute <=7',
              c['over30m_between_nearby_minutes'])
    print('\n=== RECOUVREMENT DES SOURCES : IDENTIFIANTS ET CAPTURES EXACTES ===')
    for a, b in combinations(sources, 2):
        a_ids, b_ids = set(a['records']), set(b['records'])
        a_triplets = unique_records_by_source[a['rel']]
        b_triplets = unique_records_by_source[b['rel']]
        print('  ', Path(a['rel']).name, '<->', Path(b['rel']).name,
              '| IDs communs', len(a_ids & b_ids),
              '| triplets (ID,UTC,minute) strictement identiques', len(a_triplets & b_triplets))
    c_all = Counter()
    ids_80_before87 = set()
    ids_87 = set()
    ids_2_pre = set()
    ids_backward_merged = set()
    for mid, captures in all_records.items():
        metric = compare_sequence(captures)
        c_all.update(metric)
        if metric['pre87_samples']:
            ids_80_before87.add(mid)
        if metric['has_exactly87']:
            ids_87.add(mid)
        if metric['multi_pre87_match']:
            ids_2_pre.add(mid)
        if metric['minute_backward_pairs']:
            ids_backward_merged.add(mid)
    union_unique_triplets = set().union(*unique_records_by_source.values())
    print('\n=== COHORTE ARCHIVES DE-DUPLIQUEE ===')
    print('IDS distincts :', len(all_ids))
    print('Captures distinctes ID+UTC+minute :', len(union_unique_triplets))
    print('IDS >=1 capture horodatee [80,87[ :', len(ids_80_before87))
    print('IDS >=2 captures DISTINCTES horodatees [80,87[ :', len(ids_2_pre))
    print('IDS >=1 capture horodatee exactement a 87 :', len(ids_87))
    print('IDS minute decroissante sur horodatage (sources fusionnees) :',
          len(ids_backward_merged))
    print('INTERSECTION exacte IDs avec TRAIN V4.2 :', len(all_ids & set(train_ids)))
    print('ATTENTION : le recouvrement de sources ne constitue pas des matchs supplementaires.')
    print('ATTENTION : des chronologies internes coherentes ne prouvent PAS l authenticite du timestamp ni le kickoff.')
    print('ATTENTION : minute =87 est AMBIGUE pour les features strictement AVANT 87.')
    print('ATTENTION : independance avec VALIDATION/TEST et labels certifies = NON VERIFIES.')
    print('AUCUN LABEL/SCORE/PROBABILITE/FEATURE DECODE | 0 API | 0 ECRITURE | 0 FIT | TEST FERME')
    print('G02-G37/P2 ARCHIVE_CAPTURE_CONSISTENCY COMPLETE | RESEARCH ONLY | AS-OF NON CERTIFIE')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-archives', action='store_true', help='audit horodatages+minute archives seulement')
    args = ap.parse_args()
    g34, root, train_ids = precheck()
    if args.audit_archives:
        audit(g34, root, train_ids)
    else:
        print('Puis : python3 -B g02g37_archive_capture_internal_consistency_readonly.py --audit-archives')


if __name__ == '__main__':
    main()
