#!/usr/bin/env python3
"""G02-G13 - frozen team queue; budget capped in actual HTTP GET calls.
Never trains or reads research labels. Keeps raw current-day API responses, NOT as-of proof.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

B = Path('outputs/v4_research')
G12 = B / 'v42_g02g12_remaining_prematch_plan'
OUT = B / 'v42_g02g13_remaining_prematch_history/raw_pages'
Q = G12 / 'G02G12_REMAINING_TEAMS_FROZEN.csv'
T = G12 / 'G02G12_REMAINING_TARGETS_FROZEN.csv'
R = G12 / 'G02G12_PLAN_REPORT.json'
C = B / 'v42_g02g11_prematch_contract/G02G11_PREMATCH_CONTRACT_FROZEN.json'
HASHES = {
    Q: 'ab4cdbf3d6231ce7514efdb252b7dfd8874676fc5d920941256a3fef05a5e593',
    T: 'd3b1d970ebaed15cf798ba4b0c44840d2318adc3e595e31903dba110f7921de7',
    R: '27940d3a8a805d4c40be7a525b2b7111113b2a0ce40d9ce131412872fb697405',
    C: '07ff1e2c3191008861134ee5ab6eabcd76b7a44c05995d5273df4518c5e89984',
}


def stop(s):
    raise SystemExit('STOP : ' + str(s))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc(s):
    try:
        d = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if d.tzinfo is None:
            raise ValueError('missing timezone')
        return d.astimezone(timezone.utc)
    except (TypeError, ValueError) as e:
        stop('date invalide : ' + repr(s) + ' : ' + str(e))


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def goal_valid(v):
    if isinstance(v, bool) or v is None:
        return False
    try:
        n = float(v)
        return math.isfinite(n) and n >= 0 and n.is_integer()
    except (ValueError, TypeError):
        return False


def cache_path(tid, page):
    return OUT / tid / ('page_%04d.json' % page)


def inspect_response(wrapper, tid, page):
    if not isinstance(wrapper, dict) or wrapper.get('team_id') != tid:
        stop('enveloppe cache/API invalide : ' + tid)
    if wrapper.get('endpoint') != '/football/matches':
        stop('endpoint invalide : ' + tid)
    prm = wrapper.get('params')
    if not isinstance(prm, dict) or (
        prm.get('team_id'), prm.get('status'), prm.get('per_page'), prm.get('page')
    ) != (tid, 'finished', 100, page):
        stop('params incorrects : ' + tid + ' p' + str(page))
    response = wrapper.get('response')
    if not isinstance(response, dict) or response.get('error'):
        stop('reponse API non exploitable : ' + tid + ' p' + str(page))
    meta, items = response.get('meta'), response.get('data')
    if not isinstance(meta, dict) or not isinstance(items, list):
        stop('pagination/reponse absente : ' + tid)
    try:
        mp, pages, limit, total = (int(meta[k]) for k in
                                   ('page', 'total_pages', 'per_page', 'total'))
    except (KeyError, ValueError, TypeError):
        stop('meta incomplete : ' + tid)
    if (mp != page or limit != 100 or total < len(items) or len(items) > 100
        or pages < 0 or (pages == 0 and (total != 0 or items))
        or (pages > 0 and page > pages)):
        stop('meta incoherente : ' + tid + ' p' + str(page))
    dated = []
    for m in items:
        if not isinstance(m, dict) or not m.get('id') or m.get('status') != 'finished':
            stop('match invalide : ' + tid + ' p' + str(page))
        h, a = m.get('home_team'), m.get('away_team')
        if not isinstance(h, dict) or not isinstance(a, dict):
            stop('identites manquantes : ' + tid)
        if (h.get('id') == tid) == (a.get('id') == tid):
            stop('equipe absente/dupliquee dans match : ' + tid)
        dated.append((utc(m.get('utc_date')), str(m['id']), m))
    if any(dated[i][0] < dated[i+1][0] for i in range(len(dated)-1)):
        stop('ordre decroissant non respecte : ' + tid + ' p' + str(page))
    if len({x[1] for x in dated}) != len(dated):
        stop('doublons sur page : ' + tid)
    return dated, (pages, total)


def scan(tid, entries):
    earliest = min(x['cutoff'] for x in entries)
    all_items = []
    known_meta = None
    page = 1
    while cache_path(tid, page).exists():
        path = cache_path(tid, page)
        try:
            wrapper = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            stop('cache corrompu : ' + str(path) + ' : ' + str(e))
        items, meta = inspect_response(wrapper, tid, page)
        if known_meta is not None and meta != known_meta:
            stop('meta modifiee entre pages : ' + tid)
        known_meta = meta
        if all_items and items and all_items[-1][0] < items[0][0]:
            stop('ordre entre pages incorrect : ' + tid)
        all_items.extend(items)
        if len({i for _, i, _ in all_items}) != len(all_items):
            stop('match duplique entre pages : ' + tid)
        page += 1
        if known_meta[0] == 0 or page > known_meta[0]:
            break
    if known_meta is None:
        return {'state': 'NEED_PAGE', 'page': 1, 'cached_pages': 0,
                'observations': len(entries), 'n_min': None}

    eligible = [x for x in all_items if x[0] < earliest]
    total_pages = known_meta[0]
    exhausted = (total_pages == 0 or page > total_pages)
    # When rank 20 lies at the last cached date, a future page may contain
    # the same kickoff time: do not silently choose an arbitrary tied match.
    boundary_unknown = (len(eligible) >= 20 and not exhausted
                        and eligible[19][0] == all_items[-1][0])
    if (len(eligible) < 20 or boundary_unknown) and not exhausted:
        return {'state': 'NEED_PAGE', 'page': page,
                'cached_pages': page-1, 'observations': len(entries),
                'n_min': min(len(eligible), 20)}

    statuses = Counter()
    for target in entries:
        prior = [x for x in all_items
                 if x[0] < target['cutoff'] and x[1] != target['match_id']]
        top = prior[:20]
        if len(prior) > 20 and prior[19][0] == prior[20][0]:
            statuses['TIE_AT_20'] += 1
            continue
        if (len(top) == 20 and not exhausted and
            top[19][0] == all_items[-1][0]):
            return {'state': 'NEED_PAGE', 'page': page,
                    'cached_pages': page-1, 'observations': len(entries),
                    'n_min': min(len(eligible), 20)}
        scores_ok = True
        for _, _, m in top:
            pos = 'home' if m['home_team']['id'] == tid else 'away'
            score = m.get('score')
            reg = score.get('regulation') if isinstance(score, dict) else None
            if not isinstance(reg, dict) or not goal_valid(reg.get(pos)):
                scores_ok = False
                break
        if not scores_ok:
            statuses['SCORE_MISSING'] += 1
        elif len(top) == 20:
            statuses['COMPLETE20'] += 1
        elif len(top) >= 10:
            statuses['PARTIAL10_19'] += 1
        else:
            statuses['MISSING_LT10'] += 1
    return {'state': 'COVERED' if not statuses['TIE_AT_20'] and
            not statuses['SCORE_MISSING'] else 'REVIEW',
            'cached_pages': page-1, 'observations': len(entries),
            'n_min': min(len(eligible), 20), 'statuses': dict(statuses)}


def save_atomic(path, wrapper):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    if path.exists() or tmp.exists():
        stop('refus ecrasement : ' + str(path))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(str(tmp), str(path))  # atomic no-clobber on macOS
    except FileExistsError:
        stop('concurrence, fichier deja existant : ' + str(path))
    finally:
        tmp.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=0,
                   help='nombre maximal de requetes GET, entre 0 et 100')
    args = p.parse_args()
    if not 0 <= args.budget <= 100:
        stop('budget autorise entre 0 et 100 requetes')
    print('G02-G13 | collecte generale | BUDGET API =', args.budget)
    for path, digest in HASHES.items():
        if not path.is_file() or sha(path) != digest:
            stop('SHA incorrect : ' + str(path))
        print('SHA OK :', path.name)
    contract = json.loads(C.read_text(encoding='utf-8'))
    if (contract.get('id') != 'G02G11_PREMATCH_V1'
        or contract.get('training_allowed') is not False
        or contract.get('last_n_historical_matches') != 20
        or contract.get('historical_status_asof_certified') is not False):
        stop('contrat G02-G11 non conforme')
    report = json.loads(R.read_text(encoding='utf-8'))
    if (report.get('teams_queue_sha256') != HASHES[Q]
        or report.get('targets_queue_sha256') != HASHES[T]
        or report.get('contract_sha256') != HASHES[C]):
        stop('rapport G02-G12 incoherent')
    rows = read_csv(Q)
    targets = read_csv(T)
    if len(rows) != 1098 or len(targets) != 2948:
        stop('effectifs de file incorrects')
    ids = set()
    by = defaultdict(list)
    ranks = {}
    for n, r in enumerate(rows, 1):
        tid = r['team_id']
        if tid in ids or int(r['queue_rank']) != n:
            stop('file non unique ou mal ordonnee')
        if r['stratum'] not in ('SINGLE', 'REPEATED', 'FREQUENT'):
            stop('strate inconnue')
        ids.add(tid)
        ranks[tid] = (n, int(r['target_observations']))
    seen = set()
    for r in targets:
        tid = r['team_id']
        key = (r['match_id'], r['side'])
        if (tid not in ranks or key in seen
            or int(r['queue_rank']) != ranks[tid][0]
            or r['side'] not in ('home', 'away')):
            stop('observation incoherente')
        seen.add(key)
        cutoff = utc(r['safe_cutoff_utc'])
        if utc(r['kickoff_utc']) - cutoff != timedelta(hours=4):
            stop('cutoff non conforme')
        by[tid].append({'match_id': r['match_id'],
                        'cutoff': cutoff})
    if len(by) != 1098 or any(len(by[tid]) != ranks[tid][1] for tid in ids):
        stop('jointure observations / equipes incorrecte')

    # Planning always scans saved pages first. No filesystem changes for budget 0.
    before = Counter()
    for r in rows:
        state = scan(r['team_id'], by[r['team_id']])
        before[state['state']] += 1
    print('ETAT INITIAL :', dict(before))
    if args.budget == 0:
        print('PREMIERES ACTIONS :')
        shown = 0
        for r in rows:
            state = scan(r['team_id'], by[r['team_id']])
            if state['state'] == 'NEED_PAGE':
                print(' ', r['team_id'], '| rang', r['queue_rank'],
                      '| prochaine page', state['page'],
                      '| observations', len(by[r['team_id']]))
                shown += 1
                if shown == 6:
                    break
        print('G02-G13 PLAN OK | 0 API | 0 ECRITURE | 0 ENTRAINEMENT')
        return

    import rebuild_clean_prematch_v2 as pre
    lab = pre.load_lab()
    client = pre.build_rate_limited_client(lab, max_retries=1, min_interval=2.5)
    count = 0
    for r in rows:
        tid = r['team_id']
        while count < args.budget:
            state = scan(tid, by[tid])
            if state['state'] != 'NEED_PAGE':
                break
            page = state['page']
            params = {'team_id': tid, 'status': 'finished',
                      'per_page': 100, 'page': page}
            print('APPEL', count+1, '/', args.budget, '|', tid, '| page', page)
            response = client.get('/football/matches', params=params)
            wrapper = {
                'team_id': tid,
                'endpoint': '/football/matches',
                'params': params,
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'RAW_NOT_ASOF_CERTIFIED',
                'response': response,
            }
            # Validate before writing, but do not pretend current data is historical as-of.
            items, meta = inspect_response(wrapper, tid, page)
            dest = cache_path(tid, page)
            save_atomic(dest, wrapper)
            count += 1
            print('SAUVE', dest, '| lignes', len(items),
                  '| total', meta[1], '| pages', meta[0], '| SHA256', sha(dest))
        if count >= args.budget:
            break

    after = Counter()
    detail = Counter()
    for r in rows:
        state = scan(r['team_id'], by[r['team_id']])
        after[state['state']] += 1
        if 'statuses' in state:
            detail.update(state['statuses'])
    print('ETAT FINAL :', dict(after))
    print('OBSERVATIONS EVALUEES :', dict(detail))
    print('REQUETES EFFECTUEES :', count, '/', args.budget)
    print('ATTENTION : pas de certification des resultats disponibles AS-OF.')
    print('AUCUN FICHIER GELE MODIFIE | AUCUN ENTRAINEMENT')
    print('G02-G13 LOT TERMINE')


if __name__ == '__main__':
    main()
