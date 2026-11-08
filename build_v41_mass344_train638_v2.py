#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etape 251 V2: reconstruire MASS344 puis fusionner TRAIN638; recherche seulement.
Aucune requete API, aucun entrainement et aucun ecrasement des fichiers.
A executer depuis la racine du projet ML_Goal_After_87.
"""
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_historical_ab_research_v3 import build_match
from v3c_pressure_features import get_events, event_minute

R = Path('outputs/v4_research')
M = R / 'v41_mass350'
MANIFEST = M / 'manifest.csv'
RAW = M / 'raw_timelines'
OLD = R / 'V41_TRAIN294_B29_RESEARCH_V1.csv'
REFERENCE = R / 'V41_DIVERSE16_B29_RESEARCH_V1.csv'
SPLIT = Path('outputs/pressure_historical/pressure_v2_dataset_join/ALL_320_CHRONO_SPLIT_PLAN_V1.csv')
OUT_MASS = R / 'V41_MASS344_B29_RESEARCH_V1.csv'
OUT_TRAIN = R / 'V41_TRAIN638_B29_RESEARCH_V1.csv'
REPORT = R / 'V41_MASS344_TRAIN638_AUDIT_V1.json'
GOALS = {'goal', 'own_goal', 'penalty_scored'}
AMBIGUOUS = {
    'mt_159950436', 'mt_376663150', 'mt_376828810',
    'mt_517343998', 'mt_585261954', 'mt_747913790',
}
META_OVERRIDES = {
    'research_origin': 'V41_MASS344',
    'research_status': 'RESEARCH_ONLY',
    'training_certified': '0',
    'live_certified': '0',
    'label_used_for_selection': '0',
}
FEATURES = [
    'score_total_87', 'score_diff_abs_87', 'is_draw_87',
    'shots_total_last5', 'sot_total_last5', 'corners_total_last5',
]
B6 = [
    'shots_home_last5', 'shots_away_last5',
    'sot_home_last5', 'sot_away_last5',
    'corners_home_last5', 'corners_away_last5',
]


def read_csv(path):
    if not path.is_file():
        raise ValueError('Fichier introuvable: ' + str(path))
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        rows = list(rd)
    if not cols or len(cols) != len(set(cols)):
        raise ValueError('En-tete manquant/duplique: ' + str(path))
    if 'match_id' not in cols:
        raise ValueError('Colonne match_id absente: ' + str(path))
    ids = [r['match_id'] for r in rows]
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('Identifiants absents/dupliques: ' + str(path))
    return cols, rows


def number(v):
    n = float(v)
    if not math.isfinite(n):
        raise ValueError('Nombre non fini: ' + str(v))
    return n


def nonneg_int(v):
    n = number(v)
    if n < 0 or not n.is_integer():
        raise ValueError('Entier positif attendu: ' + str(v))
    return int(n)


def binary(v):
    n = nonneg_int(v)
    if n not in (0, 1):
        raise ValueError('Binaire attendu: ' + str(v))
    return n


def date_utc(v):
    d = datetime.fromisoformat(str(v).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Date sans fuseau: ' + str(v))
    return d.astimezone(timezone.utc)


def derived_simple6(row):
    home = nonneg_int(row['score_87_home'])
    away = nonneg_int(row['score_87_away'])
    return {
        'score_total_87': home + away,
        'score_diff_abs_87': abs(home - away),
        'is_draw_87': int(home == away),
        'shots_total_last5': nonneg_int(row[B6[0]]) + nonneg_int(row[B6[1]]),
        'sot_total_last5': nonneg_int(row[B6[2]]) + nonneg_int(row[B6[3]]),
        'corners_total_last5': nonneg_int(row[B6[4]]) + nonneg_int(row[B6[5]]),
    }


def check_features(row, mid):
    # Les colonnes SIMPLE6 sont des sommes et differences derivables.
    # Si elles existent deja, on les controle. Sinon on les calcule
    # seulement pour le dataset final, jamais pour le diagnostic B29 brut.
    for col, expected in derived_simple6(row).items():
        if col in row and str(row[col]).strip() != '':
            if number(row[col]) != expected:
                raise ValueError('{}: {}={} != {}'.format(mid, col, row[col], expected))
    return binary(row['goal_after_87'])


def save_csv_new(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    print('=' * 74)
    print('ETAPE 251 V2 — MASS344 B29 + TRAIN638 (RESEARCH_ONLY)')
    print('=' * 74)
    for path in (OUT_MASS, OUT_TRAIN, REPORT):
        if path.exists():
            raise SystemExit('SORTIE DEJA EXISTANTE, aucun ecrasement: ' + str(path))

    mf, manifest = read_csv(MANIFEST)
    old_fields, old_rows = read_csv(OLD)
    reference_fields, reference_rows = read_csv(REFERENCE)
    _, split_rows = read_csv(SPLIT)
    if len(manifest) != 12415 or len(old_rows) != 294 or len(reference_rows) != 16 or len(split_rows) != 320:
        raise ValueError('Effectifs de reference inattendus')
    lookup = {r['match_id']: r for r in manifest}
    old_ids = {r['match_id'] for r in old_rows}
    protected = old_ids | {r['match_id'] for r in split_rows}
    files = sorted(RAW.glob('mt_*.json'))
    if len(files) != 350:
        raise ValueError('350 timelines FULL attendues, trouvees: ' + str(len(files)))
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Noms de fichiers dupliques')
    if protected & {p.stem for p in files}:
        raise ValueError('CHEVAUCHEMENT AVEC TRAIN294 OU SPLIT PROTEGE')

    raw_cols = set(reference_fields) - set(META_OVERRIDES)
    required_b29 = set(B6 + ['goal_after_87', 'score_87_home',
                              'score_87_away', 'is_draw_87',
                              'match_id', 'kickoff_utc'])
    missing_b29 = required_b29 - raw_cols
    if missing_b29:
        raise ValueError('Schema B29 incomplet : {}'.format(sorted(missing_b29)))
    # IMPORTANT : on n'exige PAS les totaux SIMPLE6 dans la reference B29.
    # Ils seront derives de score_87 et des six compteurs B6.
    derived_columns = [col for col in FEATURES if col not in reference_fields]
    train_derived_columns = [col for col in FEATURES if col not in old_fields]
    print('SIMPLE6 derivees dans MASS :', derived_columns)
    print('SIMPLE6 derivees dans TRAIN638 :', train_derived_columns)

    accepted = []
    excluded = []
    months = Counter()
    days = Counter()
    positives = 0
    for path in files:
        mid = path.stem
        if mid not in lookup:
            raise ValueError('Match hors manifeste: ' + mid)
        row = lookup[mid]
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('match_id') != mid:
            raise ValueError('ID local incorrect: ' + mid)
        response = record.get('timeline_response')
        if not isinstance(response, dict) or not isinstance(response.get('data'), dict):
            raise ValueError('Payload non reconnu: ' + mid)
        data = response['data']
        if data.get('match_id') != mid or data.get('coverage') != 'full':
            raise ValueError('ID/couverture incoherents: ' + mid)
        events = get_events(response)
        if not isinstance(events, list) or not events or events != data.get('events'):
            raise ValueError('Evenements absents ou parser divergent: ' + mid)
        if not any(isinstance(e, dict) and e.get('period') == 'second_half'
                   and e.get('type') == 'period_end' for e in events):
            raise ValueError('Fin seconde periode absente: ' + mid)

        late_goals = 0
        boundary = False
        for e in events:
            if not isinstance(e, dict):
                raise ValueError('Evenement non structure: ' + mid)
            if e.get('period') != 'second_half' or e.get('type') not in GOALS:
                continue
            minute = event_minute(e)
            if minute is None:
                raise ValueError('But sans minute: ' + mid)
            if minute == 87:
                boundary = True
            elif minute > 87:
                late_goals += 1
        if boundary:
            excluded.append(mid)
            continue

        info = {k: '' if v is None else str(v) for k, v in row.items()}
        a, b, quality = build_match(mid, info, data)
        if a is None or not isinstance(b, dict) or not isinstance(quality, dict):
            raise ValueError('Reconstruction rejetee {}: {}'.format(mid, quality))
        if quality.get('status') != 'RESEARCH_ONLY':
            raise ValueError('Statut de reconstruction inattendu: ' + mid)
        if not raw_cols.issubset(set(b)) or set(b) - set(reference_fields):
            raise ValueError('SCHEMA B29 DIFFERENT {} | absent={} surplus={}'.format(
                mid, sorted(raw_cols - set(b)), sorted(set(b) - set(reference_fields))))
        if 'research_origin' in b:
            raise ValueError('Le builder a defini research_origin: ' + mid)
        if binary(b['goal_after_87']) != int(late_goals > 0):
            raise ValueError('Label divergent des buts second_half apres 87: ' + mid)
        check_features(b, mid)
        dt = date_utc(row['kickoff_utc'])
        if not (datetime(2026, 3, 1, tzinfo=timezone.utc) <= dt < datetime(2026, 8, 21, tzinfo=timezone.utc)):
            raise ValueError('Date hors fenetre: ' + mid)
        output = {field: META_OVERRIDES[field] if field in META_OVERRIDES else b[field]
                  for field in reference_fields}
        # Derivation deterministe a partir des colonnes pre-87 uniquement.
        for col in derived_columns:
            output[col] = derived_simple6(b)[col]
        check_features(output, mid)
        accepted.append(output)
        days[dt.strftime('%Y-%m-%d')] += 1
        months[dt.strftime('%Y-%m')] += 1
        positives += binary(output['goal_after_87'])

    if set(excluded) != AMBIGUOUS:
        raise ValueError('Liste des 6 buts a 87 differente. Observe: ' + str(sorted(excluded)))
    if len(accepted) != 344 or positives != 87 or len(days) != 100:
        raise ValueError('Attendus: 344 matches, 87 positifs, 100 dates; obtenus: {} {} {}'.format(
            len(accepted), positives, len(days)))

    if len(set(reference_fields) | set(old_fields)) != 43:
        raise ValueError('Union de schemas inattendue (43 colonnes attendues)')
    missing_old = set(old_fields) - set(reference_fields)
    missing_new = set(reference_fields) - set(old_fields)
    if missing_old != {'independent_verified'} or missing_new:
        raise ValueError('Schemas modifiees: old_only={} new_only={}'.format(
            sorted(missing_old), sorted(missing_new)))
    if any(r.get('research_origin') != 'V41_MASS344' for r in accepted):
        raise ValueError('Provenance MASS344 inattendue')
    old_positive = sum(check_features(r, r['match_id']) for r in old_rows)
    if old_positive != 75:
        raise ValueError('75 positifs TRAIN294 attendus, trouves: ' + str(old_positive))
    if any(date_utc(r['kickoff_utc']) >= datetime(2026, 8, 25, tzinfo=timezone.utc) for r in old_rows):
        raise ValueError('Date TRAIN294 hors limite')

    # Pas de remplissage artificiel des metadonnees: independent_verified = 0
    # est explicite, car aucune verification externe n'a ete faite.
    new_rows = []
    for row in accepted:
        item = {k: row.get(k, '') for k in old_fields}
        item['independent_verified'] = '0'
        for col in train_derived_columns:
            item[col] = derived_simple6(row)[col]
        new_rows.append(item)
    previous_rows = []
    for row in old_rows:
        item = dict(row)
        for col in train_derived_columns:
            item[col] = derived_simple6(row)[col]
        check_features(item, item['match_id'])
        previous_rows.append(item)
    train = previous_rows + new_rows
    if len(train) != 638 or len({r['match_id'] for r in train}) != 638:
        raise ValueError('Effectif final ou identifiants invalides')
    if sum(binary(r['goal_after_87']) for r in train) != 162:
        raise ValueError('162 labels positifs attendus')
    train.sort(key=lambda r: (date_utc(r['kickoff_utc']), r['match_id']))

    report = {
        'simple6_derived_mass': derived_columns,
        'simple6_derived_train': train_derived_columns,
        'simple6_formula_checked': True,
        'status': 'RESEARCH_ONLY', 'mass_full': 350,
        'mass_accepted': len(accepted), 'mass_excluded_boundary87': sorted(excluded),
        'mass_positive_provisional': positives, 'mass_negative_provisional': len(accepted) - positives,
        'mass_utc_days': len(days), 'mass_by_month': dict(sorted(months.items())),
        'train294': 294, 'train638': len(train), 'train638_positives': 162,
        'independent_label_verified': False,
        'independent_score_verified': False,
        'live_equivalence_verified': False,
        'model_trained': False, 'test_futur_evaluated': False,
    }
    # N'ecrire les fichiers qu'apres TOUS les controles.
    save_csv_new(OUT_MASS, reference_fields + derived_columns, accepted)
    save_csv_new(OUT_TRAIN, old_fields + train_derived_columns, train)
    with REPORT.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('MASS344   :', OUT_MASS)
    print('TRAIN638  :', OUT_TRAIN)
    print('RAPPORT   :', REPORT)
    print('Reconstruits :', len(accepted), '| positifs provisoires :', positives)
    print('Ambigus exclus :', len(excluded), '| dates :', len(days))
    print('TRAIN638 :', len(train), '| positifs :', 162)
    print('RESEARCH_ONLY | Aucun appel API | Aucun entrainement | TEST_FUTUR non evalue')
    print('ETAPE 251 V2 TERMINEE')


if __name__ == '__main__':
    main()
