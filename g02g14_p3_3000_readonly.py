#!/usr/bin/env python3
"""G02-G14/P3: independent read-only consolidation of all 3,000 prematch observations.

Run from the ML_Goal_After_87 project root:
    python3 -B g02g14_p3_3000_readonly.py

Requires g02g13_collect_100.py, g02g14_audit_g13_readonly.py,
and g02g10_short.py in the project root. No API, no labels,
no output files, no training. Historical as-of remains NOT certified.
"""
import contextlib
import csv
import hashlib
import io
import runpy
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import g02g13_collect_100 as g
import g02g14_audit_g13_readonly as p1

BASE = Path('outputs/v4_research')
ALL_PATH = BASE / 'v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
MAN_PATH = BASE / 'v41_mass350/manifest.csv'
EXPECTED_ALL_HASH = '4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2'
EXPECTED_MAN_HASH = '61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e'
EXPECTED_COUNTS = {'COMPLETE20': 2894, 'PARTIAL10_19': 60, 'MISSING_LT10': 46}


def check(ok, message):
    if not ok:
        raise SystemExit('G02-G14/P3 STOP : ' + message)


def key(row):
    return (row['match_id'], row['side'], row['team_id'])


def main():
    print('=== G02-G14/P3 | CONTROLE GLOBAL 3000 | READ ONLY | ZERO API/LABEL/WRITE/TRAIN ===')
    check(ALL_PATH.is_file() and g.sha(ALL_PATH) == EXPECTED_ALL_HASH,
          'cibles globales absentes ou SHA different')
    check(MAN_PATH.is_file() and g.sha(MAN_PATH) == EXPECTED_MAN_HASH,
          'manifest historique absent ou SHA different')

    # Full existing G13/P1 audit covers hash frozen sources, cache completeness,
    # lack of orphan files, ordering, metadata, regulation scores, and counts.
    p1.main()

    # Execute the actual frozen pilot verifier (not --freeze) and reuse its
    # computed rows; it verifies pilot cache SHA, joins, scores and 4h cutoffs.
    script = Path('g02g10_short.py')
    check(script.is_file(), 'script pilote absent : g02g10_short.py')
    old_args = sys.argv[:]
    captured = io.StringIO()
    try:
        sys.argv = [str(script)]
        with contextlib.redirect_stdout(captured):
            pilot = runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = old_args
    pilot_log = captured.getvalue()
    check('G02-G10 PLAN OK | AUCUNE API | AUCUNE ECRITURE | AUCUN ENTRAINEMENT'
          in pilot_log, 'audit pilote non confirme')
    pilot_rows = pilot['output']
    check(len(pilot_rows) == 52, 'pilotes != 52')
    print('G02-G14/P2 RAPPEL : 18 EQUIPES / 52 OBS | PILOTE CONTROLE')
    print('G02-G10 :', pilot_log.strip().splitlines()[-1])

    all_rows = g.read_csv(ALL_PATH)
    g13_rows = g.read_csv(g.T)
    manifest_rows = g.read_csv(MAN_PATH)
    check(len(all_rows) == 3000, 'cibles globales != 3000')
    check(len(g13_rows) == 2948, 'cibles generales != 2948')
    check(len(manifest_rows) == 12415, 'effectif manifest inattendu')
    manifest = {r['match_id']: r for r in manifest_rows}
    check(len(manifest) == len(manifest_rows), 'doublons d ID dans manifest')

    # Check exact membership and unique home/away per target match: no implicit
    # dropping, duplicate or cross-join among pilots and general collection.
    global_keys = [key(r) for r in all_rows]
    pilot_keys = [key(r) for r in pilot_rows]
    general_keys = [key(r) for r in g13_rows]
    check(len(set(global_keys)) == 3000, 'cibles originales dupliquees')
    check(len(set(pilot_keys)) == 52, 'cibles pilote dupliquees')
    check(len(set(general_keys)) == 2948, 'cibles generales dupliquees')
    check(not (set(pilot_keys) & set(general_keys)), 'pilotes chevauchent general')
    check(set(pilot_keys) | set(general_keys) == set(global_keys),
          'partition 52 + 2948 non egale aux 3000 cibles gelees')
    check(len({r['team_id'] for r in all_rows}) == 1116, 'nombre equipes global != 1116')
    match_sides = defaultdict(set)
    for r in all_rows:
        mid, side, tid = key(r)
        check(mid in manifest, 'match absent du manifeste : ' + mid)
        check(side in ('home', 'away'), 'side invalide : ' + mid)
        check(manifest[mid][side + '_id'] == tid, 'mauvaise equipe : ' + mid)
        cutoff = g.utc(r['safe_cutoff_utc'])
        check(g.utc(manifest[mid]['kickoff_utc']) - cutoff == timedelta(hours=4),
              'cutoff -4h non respecte : ' + mid)
        check(side not in match_sides[mid], 'cote duplique : ' + mid)
        match_sides[mid].add(side)
    check(len(match_sides) == 1500, 'nombre matchs cibles != 1500')
    check(all(sides == {'home', 'away'} for sides in match_sides.values()),
          'match cible sans deux cotes')

    combined = {}
    for r in pilot_rows:
        k = key(r)
        check(k not in combined, 'doublon pilote : ' + repr(k))
        combined[k] = (int(r['history_n']), r['status'], r['avg_goals_scored'],
                       r['source_sha256'])

    by = defaultdict(list)
    for r in g13_rows:
        by[r['team_id']].append(r)
    check(len(by) == 1098, 'jointure collecte generale incorrecte')
    total_prior_goals = 0
    for i, tid in enumerate(sorted(by), 1):
        state = g.scan(tid, [
            {'match_id': r['match_id'], 'cutoff': g.utc(r['safe_cutoff_utc'])}
            for r in by[tid]
        ])
        check(state['state'] == 'COVERED', 'collecte non couverte : ' + tid)
        dated = []
        page_shas = []
        for page in range(1, state['cached_pages'] + 1):
            p = g.cache_path(tid, page)
            import json
            wrapper = json.loads(p.read_text(encoding='utf-8'))
            items, _ = g.inspect_response(wrapper, tid, page)
            dated.extend(items)
            page_shas.append(g.sha(p))
        for r in by[tid]:
            k = key(r)
            cutoff = g.utc(r['safe_cutoff_utc'])
            eligible = [entry for entry in dated
                        if entry[0] < cutoff and entry[1] != r['match_id']]
            check(not (len(eligible) > 20 and eligible[19][0] == eligible[20][0]),
                  'egalite rang20 : ' + repr(k))
            picked = eligible[:20]
            values = []
            for dt, mid, match in picked:
                side = 'home' if match['home_team']['id'] == tid else 'away'
                score = match.get('score') or {}
                regulation = score.get('regulation')
                check(isinstance(regulation, dict) and
                      g.goal_valid(regulation.get(side)),
                      'score regulation invalide : ' + repr(k) + '/' + mid)
                values.append(int(float(regulation[side])))
            n = len(values)
            status = 'COMPLETE20' if n == 20 else (
                'PARTIAL10_19' if n >= 10 else 'MISSING_LT10')
            avg = format(sum(values) / n, '.12f') if n >= 10 else ''
            combined[k] = (n, status, avg, ','.join(page_shas))
            total_prior_goals += n
        if i % 250 == 0:
            print('RECALCUL MOYENNES', i, '/ 1098 equipes')

    check(len(combined) == 3000 and set(combined) == set(global_keys),
          'nombre/identite des valeurs recalculees incorrect')
    counts = Counter(value[1] for value in combined.values())
    check(dict(counts) == EXPECTED_COUNTS,
          'repartition des 3000 observations differente : ' + repr(dict(counts)))
    check(all((not val[2]) == (val[0] < 10) for val in combined.values()),
          'moyenne numerique generee pour historique insuffisant')
    fingerprint = hashlib.sha256()
    for k in sorted(combined):
        n, status, avg, source = combined[k]
        fingerprint.update(('\t'.join((*k, str(n), status, avg, source)) + '\n').encode())
    print('MATCHS UNIQUES :', len(match_sides))
    print('EQUIPES UNIQUES :', len({r['team_id'] for r in all_rows}))
    print('OBSERVATIONS PRISES EN COMPTE :', len(combined))
    print('STATUTS :', dict(counts))
    print('UTILISATIONS SCORES PILOTES :', pilot['counts']['score_uses'])
    print('UTILISATIONS SCORES G02-G13 :', total_prior_goals)
    print('EMPREINTE REPRODUCTIBLE SHA256 EN MEMOIRE :', fingerprint.hexdigest())
    print('AS-OF : NON CERTIFIE (historique recupere a posteriori)')
    print('G02-G14/P3 PASS | 3000 OBS | 0 API | 0 ECRITURE | 0 LABEL | 0 ENTRAINEMENT')


if __name__ == '__main__':
    main()
