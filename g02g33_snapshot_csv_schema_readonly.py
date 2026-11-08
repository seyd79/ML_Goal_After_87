#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G33 | Inspection des EN-TETES CSV de sources potentielles AS-OF.

Depuis la racine de ML_Goal_After_87 :
  python3 -B g02g33_snapshot_csv_schema_readonly.py
  python3 -B g02g33_snapshot_csv_schema_readonly.py --headers

Mode normal : SHA256 de quatre artefacts FROZEN SANS LABELS seulement.
--headers : ouverture UNIQUEMENT de la premiere ligne de six CSV nommes.
           Ne lit JAMAIS la deuxieme ligne ni une ligne de donnees.
           Les JSON ne sont PAS ouverts.
Aucun label, evenement, prediction, score ou resultat n'est interprete.
Aucun appel API, aucune ecriture, aucun entrainement, aucun test consulte.

Le nom d'une colonne de timestamp n'etablit PAS une preuve de capture
avant 87' ni de correspondance aux 1500 matchs du dataset V4.2.
"""
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path

EXPECTED = {
    'outputs/v4_research/v42_g02g15_dataset_research_v1/V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv':
        '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    'outputs/v4_research/v42_g02g16_chrono_split_research_v1/G02G16_CHRONO_SPLITS_1500_FROZEN.csv':
        'd03ce65aab41b88e777b28321c5c032cb912779b2823f700037ac5d911075877',
    'outputs/v4_research/v42_g02g17_eligibility_research_v1/G02G17_RESEARCH_ELIGIBILITY_1500_FROZEN.csv':
        '3af8ebb87b08a90bc1812b35709c2a6565c8b16205cc60676c1cad035f85ac5e',
    'outputs/v4_research/v42_g02g19_train_preprocessing_guard_v1/G02G19_LABEL_ACCESS_TRAIN_VALIDATION_ONLY_1114_FROZEN.csv':
        '4ce470604a2f06f1a8e3eb9b633e9abebd984ddfafaf852355da04c5e282b459',
}

CSV_CANDIDATES = (
    'outputs/history/live_feature_snapshots.csv',
    'outputs/history/live_feature_snapshots_v3c.csv',
    'outputs/history/v3c_127_alertes_snapshots.csv',
    'outputs/v4_research/V4_PROSPECTIVE_SNAPSHOTS_87_V1.csv',
    'outputs/pressure_historical/pressure_v2_dataset_join/ALL_320_ASOF87_FEATURE_INVARIANCE_V1.csv',
    'outputs/v3c_external_test/v3c_scored_snapshots_full.csv',
)

JSON_ONLY_METADATA = (
    'audit_2526_snapshots.json',
    'outputs/first_half_37/shadow_live/FIRST_HALF37_LIVE_STATS_SNAPSHOTS.json',
)

MAX_HEADER = 16384
MAX_COLUMNS = 300
TIMESTAMP_TOKENS = ('captur', 'retriev', 'fetch', 'collect', 'observ',
                    'receive', 'recorded', 'ingest', 'created_at',
                    'timestamp', 'time_utc', 'snapshot_at', 'snapshot_time')
ID_NAMES = {'match_id', 'fixture_id', 'game_id', 'event_id', 'id_match'}
MINUTE_NAMES = {'minute', 'match_minute', 'elapsed_minute', 'minute_alerte',
                'alert_minute', 'snapshot_minute', 'current_minute'}


def require(condition, message):
    if not condition:
        raise SystemExit('G02-G33 STOP : ' + message)


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        while True:
            data = fh.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def secure_file(root, relative):
    p = root / relative
    require(p.is_file() and not p.is_symlink(), 'fichier absent/lien symbolique : ' + relative)
    # Le chemin et ses parents doivent rester sous la racine du projet.
    require(p.resolve().is_relative_to(root.resolve()) if hasattr(p.resolve(), 'is_relative_to')
            else os.path.commonpath([str(p.resolve()), str(root.resolve())]) == str(root.resolve()),
            'chemin en dehors du projet : ' + relative)
    require(p.stat().st_size > 0, 'fichier vide : ' + relative)
    return p


def read_one_csv_header(p):
    """Lecture octet par octet. Jamais de byte de la 2e ligne."""
    header = bytearray()
    found_newline = False
    with p.open('rb') as fh:
        while len(header) < MAX_HEADER:
            byte = fh.read(1)
            if not byte:
                break
            if byte == b'\n':
                found_newline = True
                break
            header.extend(byte)
    require(found_newline, 'en-tete trop long / sans fin de ligne : ' + str(p))
    require(header.strip(), 'en-tete vide : ' + str(p))
    try:
        raw = header.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise SystemExit('G02-G33 STOP : en-tete non UTF-8, aucune ligne de donnees lue : ' + str(p))
    try:
        cols = next(csv.reader(io.StringIO(raw), strict=True))
    except (csv.Error, StopIteration):
        raise SystemExit('G02-G33 STOP : en-tete CSV non analysable : ' + str(p))
    require(0 < len(cols) <= MAX_COLUMNS and all(c.strip() for c in cols),
            'colonnes vides / trop nombreuses : ' + str(p))
    return cols


def selftest():
    # Test du parseur sans ouvrir aucun fichier du projet.
    raw = 'match_id,captured_at_utc,minute,score_87_home'
    parsed = next(csv.reader(io.StringIO(raw), strict=True))
    require(parsed[1] == 'captured_at_utc' and parsed[2] == 'minute', 'selftest CSV')
    # Par conception, la fonction de lecture s'arrete sur la premiere ligne ;
    # le test d'integration exerce cette barriere avec un faux fichier.


def precheck(root):
    print('=== G02-G33/P1 | GARDE SCHEMA SNAPSHOTS | PRECHECK ===')
    require((root / 'outputs' / 'v4_research').is_dir(),
            'executer depuis la racine ML_Goal_After_87')
    for rel, expected in EXPECTED.items():
        p = secure_file(root, rel)
        require(sha_file(p) == expected, 'empreinte SHA incorrecte : ' + rel)
        print('SHA OK :', p.name)
    selftest()
    print('CSV CIBLES :', len(CSV_CANDIDATES), '| JSON non ouverts :', len(JSON_ONLY_METADATA))
    print('G02-G33/P1 PRECHECK PASS | 0 CONTENU CANDIDAT | 0 LABEL | 0 API | 0 ECRITURE | 0 FIT')


def audit_headers(root):
    print('=== G02-G33/P2 | EN-TETES CSV UNIQUEMENT | AUCUNE LIGNE DE DONNEES ===')
    count = 0
    missing = 0
    for rel in CSV_CANDIDATES:
        p = root / rel
        if not p.exists():
            print('ABSENT :', rel)
            missing += 1
            continue
        p = secure_file(root, rel)
        names = read_one_csv_header(p)
        print('\nCSV :', rel, '| octets', p.stat().st_size, '| colonnes', len(names))
        for i, name in enumerate(names, start=1):
            print('  %02d | %s' % (i, name))
        candidates = [name for name in names if any(t in name.lower() for t in TIMESTAMP_TOKENS)]
        ids = [name for name in names if name.strip().lower() in ID_NAMES]
        minutes = [name for name in names if name.strip().lower() in MINUTE_NAMES]
        print('  CLES HORODATAGE POTENTIELLES :', candidates if candidates else '(aucune evidente)')
        print('  CLES IDENTIFIANT POTENTIELLES :', ids if ids else '(aucune evidente)')
        print('  CLES MINUTE POTENTIELLES :', minutes if minutes else '(aucune evidente)')
        count += 1
    print('\n=== JSON : NOMS ET TAILLES UNIQUEMENT (AUCUN CONTENU) ===')
    for rel in JSON_ONLY_METADATA:
        p = root / rel
        if p.exists():
            p = secure_file(root, rel)
            print('JSON SANS LECTURE :', rel, '| octets', p.stat().st_size)
        else:
            print('JSON ABSENT :', rel)
    print('CSV EN-TETES LUS :', count, '| CSV ABSENTS :', missing)
    print('IMPORTANT : meme un champ capture_at / time n est PAS une certification as-of.')
    print('IMPORTANT : aucune preuve de couverture des matchs TRAIN sans audit ulterieur des IDs.')
    print('AUCUNE LIGNE CSV DE DONNEES | 0 JSON LU | 0 RESULTAT | 0 LABEL | 0 TIMELINE')
    print('0 API | 0 ECRITURE | 0 FIT | TEST FERME')
    print('G02-G33/P2 CSV_SCHEMA_INVENTORY COMPLETE | AS-OF NON CERTIFIE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers', action='store_true', help='ouvrir seulement la premiere ligne des six CSV')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    precheck(root)
    if args.headers:
        audit_headers(root)
    else:
        print('Puis : python3 -B g02g33_snapshot_csv_schema_readonly.py --headers')


if __name__ == '__main__':
    main()
