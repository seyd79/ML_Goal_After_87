#!/usr/bin/env python3
"""G02-G14/P1: read-only audit of G02-G13's 2,948 team-match observations.

No API, no training, no labels, no writes. Run from ML_Goal_After_87:
    python3 -B g02g14_audit_g13_readonly.py
Does not independently certify historical as-of availability or audit G02-G6..G9 pilots.
"""
from collections import Counter, defaultdict
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path

import g02g13_collect_100 as g

EXPECTED_TEAMS = 1098
EXPECTED_TARGETS = 2948
EXPECTED_STATUSES = {'COMPLETE20': 2843, 'PARTIAL10_19': 59, 'MISSING_LT10': 46}


def require(condition, message):
    if not condition:
        raise SystemExit('G02-G14/P1 STOP : ' + message)


def main():
    print('G02-G14/P1 | audit G02-G13 | READ ONLY | 0 API | 0 LABEL | 0 TRAIN')
    for path, expected_hash in g.HASHES.items():
        require(path.is_file(), 'source gelee absente : ' + str(path))
        require(g.sha(path) == expected_hash, 'SHA gelee incorrect : ' + str(path))
        print('SHA SOURCE OK :', path.name)

    contract = json.loads(g.C.read_text(encoding='utf-8'))
    require(contract.get('training_allowed') is False, 'entraînement permis : contrat inattendu')
    require(contract.get('last_n_historical_matches') == 20, 'fenetre historique inattendue')
    require(contract.get('historical_status_asof_certified') is False,
            'certification AS-OF inattendue : verifier le contrat')

    queue, targets = g.read_csv(g.Q), g.read_csv(g.T)
    require(len(queue) == EXPECTED_TEAMS, 'nombre equipes non conforme')
    require(len(targets) == EXPECTED_TARGETS, 'nombre observations non conforme')
    by = defaultdict(list)
    rank_to_team = {}
    queue_counts = {}
    seen = set()
    for i, row in enumerate(queue, start=1):
        tid = row['team_id']
        require(tid not in queue_counts and int(row['queue_rank']) == i,
                'doublon/rang equipe : ' + tid)
        queue_counts[tid] = int(row['target_observations'])
        rank_to_team[i] = tid
    for row in targets:
        tid = row['team_id']
        key = (row['match_id'], row['side'])
        require(tid in queue_counts and key not in seen, 'cible inconnue/dupliquee : ' + str(key))
        require(row['side'] in ('home', 'away'), 'cote invalide : ' + str(key))
        require(rank_to_team.get(int(row['queue_rank'])) == tid, 'rang cible incorrect : ' + tid)
        kickoff, cutoff = g.utc(row['kickoff_utc']), g.utc(row['safe_cutoff_utc'])
        require(kickoff - cutoff == timedelta(hours=4), 'cutoff != kickoff -4h : ' + str(key))
        seen.add(key)
        by[tid].append({'match_id': row['match_id'], 'cutoff': cutoff, 'side': row['side']})
    require(len(by) == EXPECTED_TEAMS, 'cibles ne couvrent pas toutes les equipes')
    require(all(len(by[t]) == n for t, n in queue_counts.items()), 'jointure cible/equipe incorrecte')

    statuses = Counter()
    page_count = 0
    fingerprints = sha256()
    observed_filenames = set()
    retrieved_times = []

    for i, row in enumerate(queue, start=1):
        tid = row['team_id']
        state = g.scan(tid, by[tid])
        require(state['state'] == 'COVERED', 'equipe non couverte ou en REVIEW : ' + tid + ': ' + str(state))
        count = state['cached_pages']
        require(count >= 1, 'pas de page cache : ' + tid)
        dated = []
        for page in range(1, count + 1):
            p = g.cache_path(tid, page)
            observed_filenames.add(p.resolve())
            require(p.is_file(), 'page manquante : ' + str(p))
            wrapper = json.loads(p.read_text(encoding='utf-8'))
            require(wrapper.get('status') == 'RAW_NOT_ASOF_CERTIFIED', 'statut source inattendu : ' + str(p))
            retrieved_times.append(g.utc(wrapper['retrieved_at_utc']))
            items, _ = g.inspect_response(wrapper, tid, page)
            dated.extend(items)
            relative = p.relative_to(g.B).as_posix()
            fingerprints.update((relative + '\t' + g.sha(p) + '\n').encode('utf-8'))
            page_count += 1
        # Independent recount by cutoff and selection order; do not read any outcomes/labels.
        team_statuses = Counter()
        for target in by[tid]:
            prior = [item for item in dated
                     if item[0] < target['cutoff'] and item[1] != target['match_id']]
            first20 = prior[:20]
            if len(prior) > 20:
                require(prior[19][0] != prior[20][0],
                        'egalite au rang 20 : ' + tid + '/' + target['match_id'])
            for _, _, match in first20:
                side = 'home' if match['home_team']['id'] == tid else 'away'
                regulation = match.get('score', {}).get('regulation')
                require(isinstance(regulation, dict) and g.goal_valid(regulation.get(side)),
                        'score regulation invalide : ' + tid + '/' + target['match_id'])
            n = len(first20)
            status = 'COMPLETE20' if n == 20 else ('PARTIAL10_19' if n >= 10 else 'MISSING_LT10')
            statuses[status] += 1
            team_statuses[status] += 1
        require(state['statuses'] == dict(team_statuses),
                'statuts incompatibles avec le recalcul : ' + tid)
        if i % 200 == 0:
            print('CONTROLE', i, '/', EXPECTED_TEAMS, 'equipes')

    actual_filenames = {p.resolve() for p in g.OUT.rglob('page_*.json')}
    require(actual_filenames == observed_filenames,
            'pages supplementaires/orphelines ou non prises en compte : ' +
            str(len(actual_filenames.symmetric_difference(observed_filenames))))
    require(dict(statuses) == EXPECTED_STATUSES, 'statuts inattendus : ' + str(dict(statuses)))
    require(sum(statuses.values()) == EXPECTED_TARGETS, 'somme observations incorrecte')
    print('EQUIPES AUDITEES :', EXPECTED_TEAMS)
    print('OBSERVATIONS AUDITEES :', EXPECTED_TARGETS)
    print('STATUTS :', dict(statuses))
    print('PAGES VERIFIEES :', page_count)
    print('MANIFESTE SHA256 CALCULE EN MEMOIRE :', fingerprints.hexdigest())
    print('DATE PREMIERE CAPTURE UTC :', min(retrieved_times).isoformat())
    print('DATE DERNIERE CAPTURE UTC :', max(retrieved_times).isoformat())
    print('AS-OF : NON CERTIFIE (capture retrospective)')
    print('PILOTES 52 OBS : HORS PERIMETRE DE CET AUDIT')
    print('G02-G14/P1 PASS | 0 API | 0 ECRITURE | 0 LABEL | 0 ENTRAINEMENT')


if __name__ == '__main__':
    main()
