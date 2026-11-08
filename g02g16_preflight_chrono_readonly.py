#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G16/P1: read-only integrity, research-cohort and chronological-split preview.

Usage (from ML_Goal_After_87 root):
    python3 -B g02g16_preflight_chrono_readonly.py

0 API, 0 writes, 0 labels, 0 training. Splits are a PREVIEW, NOT frozen.
The 'strict cohort' below is a diagnostic intersection, NOT training eligibility.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path('outputs/v4_research/v42_g02g15_dataset_research_v1')
DATA = BASE / 'V42_DATASET_1500_RESEARCH_10F_NO_LABEL.csv'
OBS = BASE / 'G02G15_PREMATCH3000_RESEARCH_AUDITED.csv'
REPORT = BASE / 'G02G15_REPORT.json'
EXPECTED = {
    DATA: '032b8717ab5e9078d08437fc63882905fa7d6d67816e6e5fb556da2f90875a91',
    OBS: '8937d22ba2c12390ed979ad88be2c2eb97a4373aab3bd6c2290f7ebf270283e3',
    REPORT: 'bf9a85d0a122bc3b9ba290b29327f805c0d4c8fe1c34ace7e2fe0bd431d26a30',
}
LIVE = (
    'score_87_home', 'score_87_away', 'shots_80_87', 'sot_80_87',
    'corners_80_87', 'home_shots_on_target_87', 'away_shots_on_target_87',
    'minutes_since_last_shot_on_target',
)
PRE = ('home_avg_goals_scored', 'away_avg_goals_scored')
SCORES = {'COMPLETE20': 2894, 'PARTIAL10_19': 60, 'MISSING_LT10': 46}
QUALITY = {'FEATURE_CANDIDATE': 1438, 'REVIEW': 62}

def check(condition, message):
    if not condition:
        raise SystemExit('G02-G16 STOP : ' + message)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readcsv(p):
    with p.open(newline='', encoding='utf-8-sig') as fh:
        r = csv.DictReader(fh)
        check(r.fieldnames is not None and len(set(r.fieldnames)) == len(r.fieldnames),
              'entetes CSV invalides : ' + str(p))
        rows = list(r)
    check(all(None not in x for x in rows), 'ligne CSV avec colonnes supplementaires : ' + str(p))
    return rows, set(r.fieldnames)

def utc(v):
    try:
        d = datetime.fromisoformat(v.replace('Z', '+00:00'))
    except (TypeError, ValueError, AttributeError):
        raise SystemExit('G02-G16 STOP : date UTC invalide : ' + repr(v))
    check(d.tzinfo is not None, 'date sans fuseau UTC : ' + str(v))
    return d.astimezone(timezone.utc)

def numeric(v, name, allow_empty=False):
    if v == '' and allow_empty:
        return None
    try:
        n = float(v)
    except (ValueError, TypeError):
        raise SystemExit('G02-G16 STOP : valeur non numerique ' + name + ' : ' + repr(v))
    check(math.isfinite(n) and n >= 0, 'valeur negative/non finie : ' + name)
    return n

def main():
    print('=== G02-G16/P1 | INTEGRITE + COHORTES + CHRONO | READ ONLY ===')
    for path, digest in EXPECTED.items():
        check(path.is_file(), 'fichier gele absent : ' + str(path))
        found = sha(path)
        check(found == digest, 'SHA256 different : ' + str(path) + ' ' + found)
        print('SHA OK :', path.name, found)

    report = json.loads(REPORT.read_text(encoding='utf-8'))
    check(report.get('matches') == 1500 and report.get('prematch_observations') == 3000,
          'effectifs rapport incoherents')
    check(report.get('feature_count') == 10 and report.get('training_authorized_by_this_stage') is False,
          'contrat de recherche invalide')
    check(report.get('label_or_outcome_columns_included') is False and
          report.get('labels_opened') is False,
          'rapport de recherche ne garantit pas 0 label')
    check(report.get('mode') == 'RESEARCH_RETROSPECTIVE_NOT_ASOF_CERTIFIED',
          'mode du rapport inattendu')
    check(report.get('output_sha256', {}).get(DATA.name) == EXPECTED[DATA] and
          report.get('output_sha256', {}).get(OBS.name) == EXPECTED[OBS],
          'SHA sortie different dans le rapport')

    rows, fields = readcsv(DATA)
    prov, pfields = readcsv(OBS)
    check(len(rows) == 1500 and len(prov) == 3000, 'effectifs CSV differents')
    mandatory = {'match_id','kickoff_utc','home_team_id','away_team_id',
                 'home_history_n','away_history_n','home_history_status',
                 'away_history_status','live_feature_status',
                 'prematch_asof_certified','live_publication_timing_certified','dataset_status'}
    check(set(LIVE+PRE) | mandatory <= fields, 'colonnes du dataset manquantes')
    check({'match_id','side','team_id','safe_cutoff_utc','history_n',
           'status','avg_goals_scored','asof_certified'} <= pfields,
          'colonnes provenance manquantes')
    forbidden = {'label','target','target_status','target_label','goal_after_87','y_true'}
    check(not (fields & forbidden) and not (pfields & forbidden),
          'colonne label/target non autorisee')

    bymatch, byobs, status_counts = {}, {}, Counter()
    for r in prov:
        mid, side, tid = r['match_id'], r['side'], r['team_id']
        check(mid and side in ('home','away') and tid, 'cle provenance invalide')
        key = mid, side
        check(key not in byobs, 'doublon observation : ' + repr(key))
        byobs[key] = r
        n = r['history_n']
        check(n.isdigit() and 0 <= int(n) <= 20, 'history_n invalide : ' + repr(key))
        expected_status = ('COMPLETE20' if int(n) == 20 else
                           'PARTIAL10_19' if int(n) >= 10 else 'MISSING_LT10')
        check(r['status'] == expected_status, 'statut incoherent : ' + repr(key))
        check((r['avg_goals_scored'] == '') == (int(n) < 10),
              'moyenne manquante incoherente : ' + repr(key))
        if int(n) >= 10:
            numeric(r['avg_goals_scored'], 'avg_goals_scored')
        check(r['asof_certified'] == '0', 'certification asof inattendue')
        status_counts[expected_status] += 1
    check(len(byobs) == 3000 and dict(status_counts) == SCORES,
          'statuts/observations incoherents : ' + str(dict(status_counts)))

    quality = Counter()
    reason_counts = Counter()
    missing_both, missing_last = 0, 0
    strict_cohort = []
    dates = defaultdict(list)
    all_teams = set()
    for r in rows:
        mid = r['match_id']
        check(mid and mid not in bymatch, 'match_id vide ou duplique : ' + mid)
        bymatch[mid] = r
        ko = utc(r['kickoff_utc'])
        dates[ko.date().isoformat()].append(mid)
        all_teams.update((r['home_team_id'], r['away_team_id']))
        check(r['home_team_id'] != r['away_team_id'], 'equipe identique : ' + mid)
        check(r['dataset_status'] == 'RESEARCH_RETROSPECTIVE_NOT_ASOF_CERTIFIED' and
              r['prematch_asof_certified'] == '0' and
              r['live_publication_timing_certified'] == '0',
              'statut de certification inattendu : ' + mid)
        for f in LIVE:
            val = numeric(r[f], mid + '/' + f, allow_empty=(f == LIVE[-1]))
            if f != LIVE[-1]:
                check(val.is_integer(), 'compte live non entier : ' + mid + '/' + f)
        for f in PRE:
            numeric(r[f], mid + '/' + f, allow_empty=True)
        status = r['live_feature_status']
        check(status in QUALITY, 'statut live inconnu : ' + mid)
        quality[status] += 1
        if r.get('live_risk_reasons'):
            reason_counts[r['live_risk_reasons']] += 1
        has_both = all(r[f] != '' for f in PRE)
        has_last = (r[LIVE[-1]] != '')
        missing_both += not has_both
        missing_last += not has_last
        if status == 'FEATURE_CANDIDATE' and has_both and has_last:
            strict_cohort.append(mid)
        for side in ('home', 'away'):
            p = byobs.get((mid, side))
            check(p is not None, 'observation orpheline : ' + mid + '/' + side)
            check(utc(p['safe_cutoff_utc']) == ko - timedelta(hours=4),
                  'date limite incorrecte : ' + mid + '/' + side)
            check(p['team_id'] == r[side + '_team_id'],
                  'equipe non concordante : ' + mid + '/' + side)
            check(p['status'] == r[side + '_history_status'] and
                  p['history_n'] == r[side + '_history_n'],
                  'historique non concordant : ' + mid + '/' + side)
            pre_value = r[side + '_avg_goals_scored']
            obs_value = p['avg_goals_scored']
            if obs_value == '':
                check(pre_value == '', 'moyenne non manquante non coherente : ' + mid)
            else:
                check(pre_value != '' and abs(float(pre_value) - float(obs_value)) < 1e-10,
                      'moyenne non concordante : ' + mid + '/' + side)

    check(len(bymatch) == 1500 and len(all_teams) == 1116,
          'unicite equipe/match invalide')
    check(set(mid for mid, _ in byobs) == set(bymatch),
          'match provenance absent du dataset')
    check(dict(quality) == QUALITY, 'qualite live != rapport : ' + str(dict(quality)))
    check(1500 - missing_both == 1456 and missing_last == 5,
          'manquants != rapport G02-G15')

    print('EQUIPES :', len(all_teams), '| MATCHS :', len(bymatch),
          '| OBSERVATIONS :', len(byobs))
    print('PREMATCH :', dict(status_counts), '| LIVE :', dict(quality))
    print('SANS DEUX MOYENNES :', missing_both,
          '| SANS DERNIER SOT :', missing_last)
    print('INTERSECTION DIAGNOSTIQUE CANDIDATE+2PRE+LAST_SOT :', len(strict_cohort))
    print('ATTENTION : intersection diagnostique, PAS criteres d entrainement geles')

    # Dates grouped strictly by UTC calendar day. Preview alone; never writes.
    days = sorted(dates)
    check(len(days) >= 3, 'moins de trois jours distincts pour la partition')
    prefix = [0]
    for day in days:
        prefix.append(prefix[-1] + len(dates[day]))
    target_train, target_val_end = 900, 1200
    # Nearest candidate to each desired count, with nonempty consecutive segments.
    best = None
    for i in range(1, len(days)-1):
        for j in range(i+1, len(days)):
            mismatch = abs(prefix[i] - target_train) + abs(prefix[j] - target_val_end)
            candidate = (mismatch, i, j)
            if best is None or candidate < best:
                best = candidate
    _, i, j = best
    groups = [('TRAIN_PREVIEW', days[:i]),
              ('VALIDATION_PREVIEW', days[i:j]),
              ('TEST_PREVIEW', days[j:])]
    print('=== APERCU CHRONOLOGIQUE 60/20/20 | JOURS UTC NON PARTAGES ===')
    for name, groupdays in groups:
        mids = [mid for day in groupdays for mid in dates[day]]
        c = Counter(bymatch[mid]['live_feature_status'] for mid in mids)
        candidate = sum(mid in set(strict_cohort) for mid in mids)
        print(name, '|', len(mids), 'matchs |',
              groupdays[0], 'a', groupdays[-1],
              '|', dict(c), '| intersection diagnostique', candidate)
    print('AS-OF : NON CERTIFIE | 0 API | 0 ECRITURE | 0 LABEL | 0 ENTRAINEMENT')
    print('G02-G16/P1 PRECHECK PASS | SPLIT APERCU NON GELE')

if __name__ == '__main__':
    main()
