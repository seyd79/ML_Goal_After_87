#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G28 | Audit d'identites entre 15 features candidates et dataset 10F gele.

Depuis la racine de ML_Goal_After_87 :
  python3 -B g02g28_train_feature_contract_readonly.py
  python3 -B g02g28_train_feature_contract_readonly.py --audit-train

LECTURE SEULE : aucun label, aucun resultat >87, aucune timeline VALIDATION/TEST,
aucun fit, aucun API, aucune ecriture. Ne certifie PAS la disponibilite as-of.
Utilise le code de reconstruction G02-G27 avec SHA fixe, sans le modifier.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

G27_SCRIPT = Path('g02g27_feature_feasibility_train_readonly.py')
G27_SHA256 = 'fbf1f86ef60f48ee4d33e1dd534335b05854e3a50f37fa571edef399d7bcb3bd'


def stop(msg):
    raise SystemExit('G02-G28 STOP : ' + str(msg))


def check(ok, msg):
    if not ok:
        stop(msg)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(131072), b''):
            h.update(chunk)
    return h.hexdigest()


def guarded_g27():
    check(G27_SCRIPT.is_file() and not G27_SCRIPT.is_symlink(),
          'script G02-G27 absent ou lien symbolique')
    check(sha256(G27_SCRIPT) == G27_SHA256,
          'SHA du script G02-G27 different : STOP; ne pas modifier les scripts geles')
    import g02g27_feature_feasibility_train_readonly as g27
    g27.selftest()  # Donnees synthetiques seulement, aucune timeline reelle
    print('SHA SCRIPT G02-G27 : OK | tests synthetiques : PASS')
    return g27


def validate_invariants(feature, ref):
    """Verifications d'identites structurelles, sans jamais lire de label."""
    def f(name):
        v = feature[name]
        check(v is not None, 'feature candidate absente pour audit : ' + name)
        return float(v)

    def frozen(name):
        val = ref[name]
        check(val != '', 'feature ancienne absente : ' + name)
        return float(val)

    # Decompositions par equipe et par sous-fenetres des compteurs existants.
    rules = {
        'shots_par_equipe': f('home_shots_80_87') + f('away_shots_80_87') - frozen('shots_80_87'),
        'sot_par_equipe': f('home_sot_80_87') + f('away_sot_80_87') - frozen('sot_80_87'),
        'corners_par_equipe': f('home_corners_80_87') + f('away_corners_80_87') - frozen('corners_80_87'),
        'shots_par_fenetre': f('shots_80_84') + f('shots_84_87') - frozen('shots_80_87'),
        'score_80_plus_buts_home': f('score_80_home') + f('home_goals_80_87') - frozen('score_87_home'),
        'score_80_plus_buts_away': f('score_80_away') + f('away_goals_80_87') - frozen('score_87_away'),
    }
    for key, diff in rules.items():
        check(math.isfinite(diff) and abs(diff) < 1e-9, key + ' : identite brisee')

    # Inclusions des mesures : un SOT est un tir, un SOT recent est un SOT total.
    check(f('home_sot_80_87') <= f('home_shots_80_87') and
          f('away_sot_80_87') <= f('away_shots_80_87'),
          'SOT > tirs par equipe')
    check(f('sot_84_87') <= f('shots_84_87') and
          f('sot_84_87') <= frozen('sot_80_87'),
          'SOT tardifs > tirs tardifs / SOT sur 80-87')

    # Le dernier SOT global est celui de l'equipe la plus recente.
    side_ages = [float(feature[name]) for name in (
        'home_minutes_since_last_sot', 'away_minutes_since_last_sot'
    ) if feature[name] is not None]
    check(side_ages, 'aucun SOT par equipe sur ligne admissible')
    check(abs(min(side_ages) - frozen('minutes_since_last_shot_on_target')) < 1e-9,
          'age du dernier SOT global != min(ages equipe)')
    check(all(0 <= a <= 87 for a in side_ages), 'age SOT hors bornes')
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-train', action='store_true', help='lit les 845 timelines TRAIN admissibles, sans labels')
    args = ap.parse_args()

    print('=== G02-G28/P1 | INVARIANTS DE FEATURES | TRAIN ONLY ===')
    g27 = guarded_g27()
    train_ids, dataset, registry = g27.load_guards()
    check(len(train_ids) == 845, '845 matchs TRAIN attendus')
    print('CONTRAT : 6 identites additives + inclusions SOT + age dernier SOT')
    if not args.audit_train:
        print('0 TIMELINE | 0 LABEL | 0 API | 0 ECRITURE | 0 FIT')
        print('G02-G28/P1 PRECHECK PASS | TEST FERME')
        print('Pour auditer : python3 -B g02g28_train_feature_contract_readonly.py --audit-train')
        return

    print('=== G02-G28/P2 | VERIFICATIONS SUR 845 TIMELINES TRAIN ===')
    raw_dirs = {p.resolve() for p in g27.RAW_DIRS}
    check(all(p.is_dir() and not p.is_symlink() for p in g27.RAW_DIRS),
          'repertoire source absent ou lien symbolique')
    counters = Counter()
    fail_samples = []
    age_missing = Counter()
    for mid in train_ids:
        ref = dataset[mid]
        row = registry[mid]
        path = Path(row['source_path'])
        check(path.suffix == '.json' and path.stem == mid and
              path.is_file() and not path.is_symlink() and
              path.resolve().parent in raw_dirs,
              'chemin source TRAIN non conforme : ' + mid)
        check(sha256(path) == row['source_sha256'],
              'SHA timeline different : ' + mid)
        with path.open('r', encoding='utf-8') as f:
            wrapper = json.load(f)
        check(isinstance(wrapper, dict) and wrapper.get('match_id') == mid and
              str(wrapper.get('home_id')) == str(ref['home_team_id']) and
              str(wrapper.get('away_id')) == str(ref['away_team_id']) and
              wrapper.get('kickoff_utc') == ref['kickoff_utc'],
              'metadata timeline different : ' + mid)
        response = wrapper.get('timeline_response')
        check(isinstance(response, dict) and isinstance(response.get('data'), dict),
              'timeline_response.data absent : ' + mid)
        f, issues, _, has_ts, _ = g27.reconstruct(response['data'].get('events'), wrapper, ref)
        check(not issues, 'anomalies de reconstruction PRE87 : ' + mid + ' / ' + str(dict(issues)))
        check(all(f[x] is not None for x in g27.CANDIDATES if x not in (
            'home_minutes_since_last_sot', 'away_minutes_since_last_sot')),
            'feature candidate attendue mais manquante : ' + mid)
        validate_invariants(f, ref)
        counters['train_checked'] += 1
        counters['matches_with_event_publish_timestamp_key'] += bool(has_ts)
        for name in ('home_minutes_since_last_sot', 'away_minutes_since_last_sot'):
            if f[name] is None:
                age_missing[name] += 1
        if f['home_minutes_since_last_sot'] is None and f['away_minutes_since_last_sot'] is None:
            age_missing['BOTH'] += 1

    check(counters['train_checked'] == 845, 'nombre TRAIN non 845')
    print('TIMELINES TRAIN :', counters['train_checked'], '| VALIDATION : 0 | TEST : 0')
    print('IDENTITES : 6 / 6 PASS pour chaque match |', counters['train_checked']*6, 'comparaisons')
    print('INCLUSIONS SOT + COHERENCE DERNIER SOT : PASS sur 845 matchs')
    print('AGES DERNIER SOT MANQUANTS :', dict(age_missing))
    print('TIMESTAMPS DE PUBLICATION AU NIVEAU EVENEMENT :',
          counters['matches_with_event_publish_timestamp_key'], '/845 matchs')
    print('REDONDANCE : les 15 features ne sont PAS 15 signaux independants; plusieurs egalites exactes avec les 10F')
    print('ATTENTION : absence de SOT par equipe = valeur manquante, PAS automatiquement 0')
    print('AS-OF NON CERTIFIE : les timelines retrospectivement reconstruites ne prouvent pas un etat observable en live')
    print('0 LABEL | 0 VALIDATION | 0 TEST | 0 API | 0 ECRITURE | 0 FIT')
    print('G02-G28/P2 CONTRACT_AUDIT PASS | TRAIN ONLY | RESEARCH ONLY')


if __name__ == '__main__':
    main()
