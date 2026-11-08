#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G34 | Audit de couverture temporelle des snapshots, TRAIN seulement.

Depuis la racine ML_Goal_After_87 :
  python3 -B g02g34_snapshot_train_capture_overlap_readonly.py
  python3 -B g02g34_snapshot_train_capture_overlap_readonly.py --audit-train

PRECHECK: verifie SHA source G02-G33 et artefacts geles sans labels,
           controle l'allowlist 845 TRAIN / 269 VALIDATION sans outcome.
--audit-train: lit quatre archives snapshot comme octets, et ne decode
           que la portion initiale contenant match_id (toutes les lignes).
           UNIQUEMENT pour les IDs parmi 845 TRAIN, decode aussi minute,
           horodatage de capture, et, si disponible, kickoff du CSV prospectif.
           Jamais de valeurs goal_after_87, target_status, predictions, scores,
           ou d'autres colonnes apres le prefixe autorise.

IMPORTANT: la lecture brute d'une ligne CSV ne constitue pas l'analyse
            des colonnes interdites. Aucune valeur de resultats n'est decodee.
            Ce diagnostic n'atteste PAS de la fiabilite de la date de capture,
            de l'heure de publication chez le fournisseur ou de l'AS-OF live.
            L'inventaire des snapshots peut contenir des IDs hors TRAIN :
            seul leur champ match_id est examine pour les ignorer.
Aucun label de TEST, aucun modele, aucune API, aucune ecriture.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path

EXPECTED_G33_SHA = 'a525f3072d17794215ef9d1cf3116f048c4cce56b23ea4a01648311f15be0d23'
G33_FILE = Path('g02g33_snapshot_csv_schema_readonly.py')
SOURCES = (
    ('outputs/history/live_feature_snapshots.csv',
     ('snapshot_timestamp_utc', 'schema_version', 'match_id', 'competition_id',
      'season_id', 'home_team', 'away_team', 'minute'), 2, 0, 7, None),
    ('outputs/history/live_feature_snapshots_v3c.csv',
     ('snapshot_timestamp_utc', 'schema_version', 'match_id', 'competition_id',
      'season_id', 'home_team', 'away_team', 'minute'), 2, 0, 7, None),
    ('outputs/history/v3c_127_alertes_snapshots.csv',
     ('snapshot_timestamp_utc', 'schema_version', 'match_id', 'competition_id',
      'season_id', 'home_team', 'away_team', 'minute'), 2, 0, 7, None),
    ('outputs/v4_research/V4_PROSPECTIVE_SNAPSHOTS_87_V1.csv',
     ('match_id', 'kickoff_utc', 'captured_at_utc', 'snapshot_minute'),
     0, 2, 3, 1),
)
MAX_ROW_BYTES = 262144
MAX_ROWS_PER_FILE = 300000


def require(cond, reason):
    if not cond:
        raise SystemExit('G02-G34 STOP : ' + str(reason))


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def load_allowlist(g33, root):
    """Allowlist deja gelee; aucune colonne de resultat dans ce fichier."""
    relative = ('outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/'
                'G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv')
    path = g33.secure_file(root, relative)
    train = {}
    seen = set()
    val_count = 0
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and {'match_id', 'kickoff_utc', 'partition'}
                <= set(reader.fieldnames), 'schema allowlist non reconnu')
        require(not {'goal_after_87', 'label', 'y_true', 'target_status'}
                & set(reader.fieldnames), 'allowlist contient outcome: ABORT')
        for rec in reader:
            mid = rec['match_id']
            require(mid and mid not in seen, 'ID allowlist duplique/vide')
            seen.add(mid)
            part = rec['partition']
            if part == 'TRAIN':
                kickoff = parse_utc(rec['kickoff_utc'])
                require(kickoff is not None, 'kickoff TRAIN invalide')
                train[mid] = kickoff
            elif part == 'VALIDATION':
                val_count += 1  # nombre seulement, aucune feature ni outcome
            else:
                raise SystemExit('G02-G34 STOP : partition interdite dans allowlist')
    require(len(train) == 845 and val_count == 269 and len(seen) == 1114,
            'taille des partitions non conforme')
    return train


def parse_utc(value):
    try:
        dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        if dt.tzinfo is None or dt.utcoffset() is None:
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def parse_minute(value):
    try:
        v = float(value.strip())
        if not math.isfinite(v) or v < 0 or v > 150:
            return None
        return v
    except (ValueError, TypeError, OverflowError, AttributeError):
        return None


def parse_csv_initial_fields(raw_line, n_fields):
    """N'analyse que N champs initiaux, jamais les colonnes suivantes.

    Le separateur apres le Neme champ sert de coupe; les champs restants ne
    sont ni decodes ni passes au parseur CSV. Format = une ligne physique par
    enregistrement ; les lignes malformees provoquent STOP plutot qu'un score.
    """
    require(0 < len(raw_line) <= MAX_ROW_BYTES and raw_line.endswith(b'\n'),
            'ligne CSV tronquee/trop grande')
    in_quotes = False
    i = 0
    found = 0
    end = None
    while i < len(raw_line):
        b = raw_line[i]
        if b == 34:
            if in_quotes and i + 1 < len(raw_line) and raw_line[i + 1] == 34:
                i += 2
                continue
            in_quotes = not in_quotes
        elif b == 44 and not in_quotes:
            found += 1
            if found == n_fields:
                end = i
                break
        elif b in (10, 13) and not in_quotes:
            break
        i += 1
    require(end is not None and not in_quotes,
            'prefixe CSV incomplet/format non conforme')
    try:
        portion = raw_line[:end].decode('utf-8-sig')
        parsed = next(csv.reader(io.StringIO(portion), strict=True))
    except (UnicodeDecodeError, StopIteration, csv.Error):
        raise SystemExit('G02-G34 STOP : prefixe CSV invalide')
    require(len(parsed) == n_fields, 'nombre champs prefixe non conforme')
    return parsed


def selftest():
    # Les colonnes interdites portent des sequences UTF-8 invalides :
    # toute tentative de les decoder echouerait. Aucun de leurs octets
    # ne doit passer a csv.reader, y compris sur les lignes hors TRAIN.
    line = (b'2026-04-07T17:20:00Z,v3,mt_1,1,2,"Equipe, A","Equipe ""B""",86,'
            b'not_allowed,\xff\xfe,secret\n')
    assert parse_csv_initial_fields(line, 3) == [
        '2026-04-07T17:20:00Z', 'v3', 'mt_1']
    assert parse_csv_initial_fields(line, 8)[7] == '86'
    assert parse_csv_initial_fields(
        b'mt_1,2026-04-07T15:50:00Z,2026-04-07T17:20:00Z,86,\xff\n', 4
    ) == ['mt_1', '2026-04-07T15:50:00Z', '2026-04-07T17:20:00Z', '86']
    assert parse_utc('2026-04-07T17:20:00Z') is not None
    assert parse_utc('2026-04-07T17:20:00') is None
    assert parse_minute('86.0') == 86
    assert parse_minute('90+2') is None


def precheck(root):
    print('=== G02-G34/P1 | SOURCES DE CAPTURE x TRAIN | PRECHECK ===')
    require(G33_FILE.is_file() and not G33_FILE.is_symlink(),
            'script G02-G33 absent ou lien symbolique')
    require(file_sha(G33_FILE) == EXPECTED_G33_SHA,
            'SHA de G02-G33 different; ne pas contourner le garde')
    import g02g33_snapshot_csv_schema_readonly as g33
    g33.precheck(root)
    selftest()
    train = load_allowlist(g33, root)
    print('GARDE : TRAIN', len(train), '| VALIDATION 269 en allowlist, non evaluee'
          ' | TEST non autorise')
    print('ARCHIVES CIBLES : 4 (exclut explicitement v3c_external_test et pressure_historical)')
    print('SCHEMA : decode seulement match_id pour hors TRAIN; pour TRAIN :'
          ' date de capture + minute + eventuel kickoff')
    print('COLONNES goal_after_87 / target_status / probabilites : jamais decodees')
    print('G02-G34/P1 PRECHECK PASS | 0 ARCHIVE PARCOURUE | 0 LABEL INTERPRETE'
          ' | 0 TIMELINE | 0 API | 0 ECRITURE | 0 FIT')
    return g33, train


def scan_file(g33, root, train, config):
    rel, expected_prefix, mid_index, ts_index, minute_index, ko_index = config
    path = g33.secure_file(root, rel)
    head = g33.read_one_csv_header(path)
    require(tuple(head[:len(expected_prefix)]) == expected_prefix,
            'schema initial de l archive modifie : ' + rel)
    require(not ('match_id' in head and head.count('match_id') != 1),
            'match_id duplique dans en-tete : ' + rel)
    stats = Counter()
    mids_any = set()
    mids_plausible_80_86 = set()
    mids_at_87 = set()
    mids_in_80_87 = set()
    mids_any_train = set()
    elapsed_samples = []
    with path.open('rb') as f:
        header_line = f.readline(65536)
        require(header_line.endswith(b'\n'), 'en-tete physique sans fin de ligne')
        for raw in f:
            stats['rows_total'] += 1
            require(stats['rows_total'] <= MAX_ROWS_PER_FILE, 'limite lignes atteinte')
            fields_mid = parse_csv_initial_fields(raw, mid_index + 1)
            mid = fields_mid[mid_index].strip()
            require(mid, 'match_id snapshot vide : ' + rel)
            if mid not in train:
                stats['rows_outside_train_ignored'] += 1
                continue
            mids_any.add(mid)
            mids_any_train.add(mid)
            stats['rows_train'] += 1
            take = max(mid_index, ts_index, minute_index,
                       ko_index if ko_index is not None else 0) + 1
            fields = parse_csv_initial_fields(raw, take)
            if fields[mid_index].strip() != mid:
                raise SystemExit('G02-G34 STOP : match_id instable')
            captured = parse_utc(fields[ts_index])
            minute = parse_minute(fields[minute_index])
            if captured is None:
                stats['train_invalid_timestamp'] += 1
            if minute is None:
                stats['train_invalid_minute'] += 1
            if captured is None or minute is None:
                continue
            stats['train_ts_minute_parsable'] += 1
            if ko_index is not None:
                ko_csv = parse_utc(fields[ko_index])
                if ko_csv is None or ko_csv != train[mid]:
                    stats['train_kickoff_mismatch'] += 1
                    continue
            elapsed = (captured - train[mid]).total_seconds() / 60
            if elapsed < 0:
                stats['train_capture_before_kickoff'] += 1
                continue
            if elapsed > 240:
                stats['train_capture_over_4h_after_kickoff'] += 1
                continue
            stats['train_capture_0_to_4h_after_kickoff'] += 1
            elapsed_samples.append(elapsed)
            if minute < 80:
                stats['train_minute_before_80'] += 1
            elif minute < 87:
                stats['train_minute_80_to_under_87'] += 1
                mids_plausible_80_86.add(mid)
                mids_in_80_87.add(mid)
            elif minute == 87:
                stats['train_minute_exactly_87_boundary_uncertain'] += 1
                mids_at_87.add(mid)
                mids_in_80_87.add(mid)
            else:
                stats['train_minute_over_87'] += 1
    print('\nSOURCE :', rel)
    print('  LIGNES PHYSIQUES :', stats['rows_total'], '| ID hors TRAIN ignores :',
          stats['rows_outside_train_ignored'])
    print('  MATCHS TRAIN presents :', len(mids_any_train), '/845 | lignes TRAIN :',
          stats['rows_train'])
    print('  TIMESTAMP invalide TRAIN :', stats['train_invalid_timestamp'],
          '| minute invalide TRAIN :', stats['train_invalid_minute'])
    print('  DELAI CAPTURE : avant coup envoi', stats['train_capture_before_kickoff'],
          '| 0-4h apres', stats['train_capture_0_to_4h_after_kickoff'],
          '| >4h apres', stats['train_capture_over_4h_after_kickoff'])
    if ko_index is not None:
        print('  kickoff_utc different de la source gelee :', stats['train_kickoff_mismatch'])
    print('  MINUTES (uniquement captures entre coup envoi et +4h) :'
          ' <80', stats['train_minute_before_80'],
          '| [80,87[', stats['train_minute_80_to_under_87'],
          '| =87', stats['train_minute_exactly_87_boundary_uncertain'],
          '| >87', stats['train_minute_over_87'])
    print('  MATCHS TRAIN avec >=1 capture plausible 80<=minute<87 :',
          len(mids_plausible_80_86), '/845')
    print('  MATCHS TRAIN avec >=1 capture a minute exactement 87 (frontiere ambigue) :',
          len(mids_at_87), '/845')
    return mids_any, mids_plausible_80_86, mids_at_87


def audit(g33, root, train):
    print('=== G02-G34/P2 | COUVERTURE DE CAPTURES | 4 CSV | TRAIN ONLY ===')
    all_matches = set()
    valid_early = set()
    boundary = set()
    for config in SOURCES:
        any_m, early, bdy = scan_file(g33, root, train, config)
        all_matches.update(any_m)
        valid_early.update(early)
        boundary.update(bdy)
    print('\n=== SYNTHESE, DE-DUPLIQUEE ENTRE ARCHIVES ===')
    print('MATCHS TRAIN presentes dans au moins une archive :', len(all_matches), '/845')
    print('MATCHS TRAIN avec une capture plausible [80,87[ :', len(valid_early), '/845')
    print('MATCHS TRAIN avec une capture minute =87 :', len(boundary), '/845')
    print('INTERPRETATION : correspondance ID + timestamp plausible = piste d audit,'
          ' pas preuve de capture genuine/temps-reel.')
    print('ATENTION : minute 87 ambiguë; ne pas assimiler snapshot 87 aux features strictement avant 87.')
    print('AUCUN label/prediction/score interprete | AUCUNE timeline | AUCUNE API | 0 ECRITURE | 0 FIT')
    print('G02-G34/P2 SNAPSHOT_CAPTURE_OVERLAP COMPLETE | AS-OF NON CERTIFIE | TEST PROTEGE')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-train', action='store_true',
                    help='auditer uniquement les champs ID, minute, capture des archives')
    args = ap.parse_args()
    root = Path.cwd().resolve()
    g33, train = precheck(root)
    if args.audit_train:
        audit(g33, root, train)
    else:
        print('Puis : python3 -B g02g34_snapshot_train_capture_overlap_readonly.py --audit-train')


if __name__ == '__main__':
    main()
