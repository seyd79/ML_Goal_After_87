#!/usr/bin/env python3
"""ETAPE 261 — audit exploratoire, strictement en lecture seule.

Compare les nouveaux candidats de RAW4204 a TRAIN638, sans consulter
aucun ancien holdout, reserve, VALIDATION154 ou TEST_FUTUR.
"""
import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RAW = Path('outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv')
TRAIN = Path('outputs/v4_research/V41_TRAIN638_B29_RESEARCH_V1.csv')
CUTOFF = '2026-08-25'


def load(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def number(x):
    try:
        n = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def binary(x):
    n = number(x)
    return int(n) if n in (0, 1) else None


def valid_values(row, cols, lower=0, strict=False):
    nums = [number(row.get(c)) for c in cols]
    return all(n is not None and (n > lower if strict else n >= lower) for n in nums)


def month_of(row):
    s = str(row.get('kickoff_utc') or '').strip()
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if d.tzinfo is None:
            return None
        return d.astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return None


def where_day(d):
    if d < '2026-08-01':
        return 'AVANT_01_AOUT'
    if d < '2026-08-20':
        return '01_19_AOUT'
    return '20_24_AOUT'


def unique_counter(rows, key):
    return Counter(str(r.get(key, '')).strip() or '(vide)' for r in rows)


def sample_stats(tag, rows):
    print('\n=== ' + tag + ' ===')
    print('Matchs :', len(rows))
    print('Positifs :', sum(binary(r.get('goal_after_87')) == 1 for r in rows))
    by_month = Counter()
    by_period = Counter()
    days = Counter()
    for r in rows:
        day = month_of(r)
        if day:
            by_month[day[:7]] += 1
            by_period[where_day(day)] += 1
            days[day] += 1
    print('Mois :', dict(sorted(by_month.items())))
    print('Periodes :', dict(sorted(by_period.items())))
    print('Jours distincts :', len(days))
    print('Top 8 jours :', days.most_common(8))
    groups = {
        'score': ['score_87_home', 'score_87_away'],
        'cotes': ['odds_home', 'odds_draw', 'odds_away'],
        'pression_80_87': ['shots_80_87', 'sot_80_87', 'corners_80_87'],
        'cartons_rouges': ['home_red_cards_87', 'away_red_cards_87'],
    }
    for name, cols in groups.items():
        n = sum(valid_values(r, cols, lower=1, strict=True) if name == 'cotes'
                else valid_values(r, cols) for r in rows)
        print(f'{name} disponibles numeriquement : {n}/{len(rows)}')
    if rows:
        for col in ('target_warning', 'target_resolution_class', 'target_confidence',
                    'source', 'schema_version'):
            if col in rows[0]:
                print(f'{col} (top 8) :', unique_counter(rows, col).most_common(8))
        print('Matchs avec cotes+pression :', sum(
            valid_values(r, groups['cotes'], 1, True) and
            valid_values(r, groups['pression_80_87']) for r in rows))
        print('Matchs avec cotes+pression+cartons :', sum(
            valid_values(r, groups['cotes'], 1, True) and
            valid_values(r, groups['pression_80_87']) and
            valid_values(r, groups['cartons_rouges']) for r in rows))


def main():
    print('=' * 76)
    print('ETAPE 261 - AUDIT CANDIDATS HISTORIQUES SUPPLEMENTAIRES')
    print('=' * 76)
    raw_cols, raw = load(RAW)
    tr_cols, train = load(TRAIN)
    required = {'match_id', 'kickoff_utc', 'goal_after_87', 'target_status',
                'target_confidence', 'score_87_home', 'score_87_away'}
    if not required.issubset(raw_cols) or not {'match_id', 'goal_after_87'}.issubset(tr_cols):
        raise SystemExit('Colonnes requises absentes : audit interrompu')
    train_by_id = {str(r['match_id']).strip(): r for r in train}
    if len(train_by_id) != len(train):
        raise SystemExit('Doublons dans TRAIN638')

    eligible = []
    rejected_reasons = Counter()
    for r in raw:
        d = month_of(r)
        if d is None:
            rejected_reasons['date_invalide'] += 1
            continue
        if d >= CUTOFF:
            rejected_reasons['apres_cutoff'] += 1
            continue
        y = binary(r.get('goal_after_87'))
        if y is None or r.get('target_status') != 'LABELED_V2' or r.get('target_confidence') != 'HIGH':
            rejected_reasons['label_non_admissible'] += 1
            continue
        if not valid_values(r, ['score_87_home', 'score_87_away']):
            rejected_reasons['score_non_numerique'] += 1
            continue
        if (number(r['score_87_home']) % 1 != 0 or
                number(r['score_87_away']) % 1 != 0):
            rejected_reasons['score_non_entier'] += 1
            continue
        eligible.append(r)

    new = [r for r in eligible if str(r['match_id']).strip() not in train_by_id]
    overlap = [r for r in eligible if str(r['match_id']).strip() in train_by_id]
    mismatches = []
    for r in overlap:
        mid = str(r['match_id']).strip()
        y0 = binary(r.get('goal_after_87'))
        y1 = binary(train_by_id[mid].get('goal_after_87'))
        if y0 != y1:
            mismatches.append((mid, y0, y1, month_of(r)))

    print('\nRAW:', len(raw), '| TRAIN638:', len(train))
    print('Causes exclus RAW :', dict(rejected_reasons))
    print('RAW eligible avant cutoff :', len(eligible))
    print('Nouveaux candidats vs TRAIN638 :', len(new))
    print('Chevauchements avec TRAIN638 :', len(overlap))
    print('Conflits de labels RAW/TRAIN638 :', len(mismatches))
    for row in mismatches[:30]:
        print(' CONFLIT :', row)

    sample_stats('NOUVEAUX CANDIDATS', new)
    sample_stats('RAW DEJA DANS TRAIN638', overlap)

    # La distribution de TRAIN638 est utile pour anticiper les doublons par dates.
    sample_stats('TRAIN638 EXISTANT', train)

    print('\n=== INTERPRETATION ===')
    print('Total combine maximal provisoire :', len(train) + len(new))
    print('Conflits deja presents dans TRAIN638 (a mettre en quarantaine) :', len(mismatches))
    print('Total provisoire apres retrait de ces conflits des deux sources :',
          len(train) + len(new) - len(mismatches))
    print('ATTENTION: ces nombres NE filtrent PAS les anciens holdouts,')
    print('les reserves, les labels ambigus non reperes, ni la provenance temporelle.')
    print('Aucun fichier ecrit. Aucun fichier protege ouvert.')
    print('Aucun appel API. Aucun entrainement. V3-C inchange.')
    print('ETAPE 261 TERMINEE.')


if __name__ == '__main__':
    main()
