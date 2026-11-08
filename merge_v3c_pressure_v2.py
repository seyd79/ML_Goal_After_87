#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fusion expérimentale V3C 87' + Pressure V2, 100% hors ligne, Python 3.9+.

Ne modifie aucune entrée. Conserve systématiquement les cibles dans un CSV séparé.
Attention: l'historique reconstruit n'est PAS certifié as-of LIVE.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

PRESSURE_FIELDS = (
    'shots_home_last5', 'shots_away_last5',
    'sot_home_last5', 'sot_away_last5',
    'corners_home_last5', 'corners_away_last5',
)
# Liste blanche: colonnes explicites, zéro cible ou champ post-match autorisé.
# Toutes restent candidates / à auditer avant modélisation.
BASELINE_CANDIDATE_FIELDS = (
    'score_87_home', 'score_87_away', 'goal_diff_87', 'is_draw_87',
    'home_shots_87', 'away_shots_87',
    'home_shots_on_target_87', 'away_shots_on_target_87',
    'home_shots_off_target_87', 'away_shots_off_target_87',
    'home_shots_blocked_87', 'away_shots_blocked_87',
    'home_corners_87', 'away_corners_87',
    'home_fouls_87', 'away_fouls_87',
    'home_yellow_cards_87', 'away_yellow_cards_87',
    'home_red_cards_87', 'away_red_cards_87',
    'home_offsides_87', 'away_offsides_87',
)
# Aucun champ basé sur l'issue postérieure (final_score, target*, label, model_probability, etc).
ID_FIELDS = ('match_id', 'minute')


def read_table(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as fd:
        first = fd.readline()
        fd.seek(0)
        delim = '\t' if first.count('\t') > first.count(',') else ','
        reader = csv.DictReader(fd, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError('Aucune colonne dans %s' % path)
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Colonnes dupliquées dans %s' % path)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fd:
        w = csv.DictWriter(fd, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def minute_int(value):
    try:
        f = float(str(value).strip())
        return int(f) if f == int(f) else None
    except (ValueError, TypeError):
        return None


def require_columns(names, required, path):
    missing = [field for field in required if field not in names]
    if missing:
        raise ValueError('%s: colonnes manquantes: %s' % (path, ', '.join(missing)))


def make_unique(rows, names, path, minute_field='minute'):
    require_columns(names, ['match_id', minute_field], path)
    result = {}
    for row in rows:
        match_id = row['match_id'].strip()
        minute = minute_int(row.get(minute_field))
        if not match_id or minute not in (85, 86, 87):
            raise ValueError('%s: identifiant/minute invalide: %s, %s' % (path, match_id, row.get(minute_field)))
        key = (match_id, minute)
        if key in result:
            raise ValueError('%s: doublon match-minute: %r' % (path, key))
        result[key] = row
    return result


def build_dataset(pressure_file, manifest_file, labels_file, outdir, baseline_file=None):
    pcols, prow = read_table(pressure_file)
    require_columns(pcols, list(ID_FIELDS) + list(PRESSURE_FIELDS) + ['training_ready','pressure_status'], pressure_file)
    pressures = make_unique(prow, pcols, pressure_file)
    mcols, mrows = read_table(manifest_file)
    require_columns(mcols, ['match_id', 'kickoff_utc','home_team','away_team'], manifest_file)
    manifest = {}
    for r in mrows:
        k = r['match_id'].strip()
        if not k or k in manifest:
            raise ValueError('Manifest: ID vide ou dupliqué: '+k)
        manifest[k] = r
    lcols, lrows = read_table(labels_file)
    require_columns(lcols, ['match_id', 'reference_goal_after_87', 'reference_target_status', 'reference_target_warning'], labels_file)
    labels = {}
    for r in lrows:
        k = r['match_id'].strip()
        if not k or k in labels:
            raise ValueError('Labels: ID vide ou dupliqué: '+k)
        labels[k] = r
    baselines = None
    bfields = []
    if baseline_file:
        bcols, brows = read_table(baseline_file)
        mfield = 'minute' if 'minute' in bcols else 'snapshot_minute'
        baselines = make_unique(brows, bcols, baseline_file, minute_field=mfield)
        bfields = [k for k in BASELINE_CANDIDATE_FIELDS if k in bcols]
        if not bfields:
            raise ValueError('Pas de variable baseline reconnue: choisir un snapshot historique à 87 minutes')

    outdir = Path(outdir)
    if outdir.resolve() in {Path('.').resolve(), Path(pressure_file).parent.resolve()}:
        raise ValueError('Choisir un dossier de sortie dédié pour éviter tout écrasement')
    outdir.mkdir(parents=True, exist_ok=True)
    outA, outB, outLabels, outQA, outP = [], [], [], [], []
    counts = Counter()
    chosen_keys = sorted(pressures, key=lambda k: (manifest.get(k[0], {}).get('kickoff_utc', ''), k[0], k[1]))
    for key in chosen_keys:
        match_id, minute = key
        r = pressures[key]
        m = manifest.get(match_id)
        l = labels.get(match_id)
        b = baselines.get(key) if baselines is not None else None
        issues = []
        if m is None:
            issues.append('MISSING_MANIFEST')
        if l is None:
            issues.append('MISSING_REFERENCE_LABEL')
        if any(r.get(f, '').strip() == '' for f in PRESSURE_FIELDS):
            issues.append('MISSING_PRESSURE_VALUES')
        try:
            pv = {field:float(r[field]) for field in PRESSURE_FIELDS}
            if any(value < 0 or value != value for value in pv.values()):
                issues.append('INVALID_PRESSURE_NUMBER')
            if (pv['sot_home_last5'] > pv['shots_home_last5'] or
                    pv['sot_away_last5'] > pv['shots_away_last5']):
                issues.append('SOT_GREATER_THAN_SHOTS')
        except (ValueError, KeyError, TypeError):
            issues.append('INVALID_PRESSURE_NUMBER')
        if r.get('training_ready', '').strip().lower() not in ('true','1','yes'):
            issues.append('PRESSURE_NOT_CERTIFIED')
        if r.get('temporal_provenance', '') != 'HISTORICAL_HINDSIGHT_RECONSTRUCTION':
            issues.append('UNEXPECTED_TEMPORAL_PROVENANCE')
        if m and (r.get('home_team','').casefold().strip() != m['home_team'].casefold().strip()
                  or r.get('away_team','').casefold().strip() != m['away_team'].casefold().strip()):
            issues.append('TEAM_NAME_MISMATCH')
        if m and r.get('kickoff_utc', '').strip() and m['kickoff_utc'].strip() != r['kickoff_utc'].strip():
            issues.append('KICKOFF_MISMATCH')
        if baselines is not None and minute == 87 and b is None:
            issues.append('MISSING_BASELINE_87')
        if l and l.get('reference_target_warning','').strip():
            issues.append('REFERENCE_LABEL_WARNING')
        if l and l.get('reference_goal_after_87','').strip() not in ('0','0.0','1','1.0'):
            issues.append('REFERENCE_LABEL_INVALID')
        for issue in issues:
            counts[issue] += 1
        rowp = {
            'match_id': match_id, 'minute': minute,
            'kickoff_utc': m['kickoff_utc'] if m else '',
            'pressure_status': r.get('pressure_status', ''),
            'temporal_provenance': r.get('temporal_provenance',''),
        }
        rowp.update({f: r.get(f,'') for f in PRESSURE_FIELDS})
        outP.append(rowp)
        outQA.append({
            'match_id':match_id, 'minute':minute,
            'baseline_matched':int(b is not None),
            'manifest_matched':int(m is not None),
            'reference_label_matched':int(l is not None),
            'label_revalidated':0, 'training_ready':0,
            'issues':'|'.join(issues),
            'pressure_status':r.get('pressure_status',''),
            'source_ft_crosscheck_status':'NOT_IN_FEATURE_DATASET',
        })
        if minute != 87 or b is None:
            continue
        # Le modèle A et B ont strictement les mêmes matchs et mêmes colonnes de référence.
        common = {'match_id':match_id, 'minute':minute,'kickoff_utc':m['kickoff_utc'] if m else ''}
        common.update({f:b.get(f,'') for f in bfields})
        outA.append(common)
        outB.append({**common, **{f:r.get(f,'') for f in PRESSURE_FIELDS}})
        outLabels.append({
            'match_id':match_id,'minute':minute,
            'goal_after_87':l.get('reference_goal_after_87','') if l else '',
            'reference_status':l.get('reference_target_status','') if l else '',
            'reference_confidence':l.get('reference_target_confidence','') if l else '',
            'reference_warning':l.get('reference_target_warning','') if l else '',
            'label_revalidated':0,
        })
        counts['paired_87'] += 1
    paths = {
        'pressure_85_86_87':outdir/'PRESSURE_ONLY_85_86_87_REVIEW.csv',
        'model_A':outdir/'MODEL_A_BASELINE_87_REVIEW.csv',
        'model_B':outdir/'MODEL_B_ENRICHED_87_REVIEW.csv',
        'labels':outdir/'LABELS_87_SEPARATE_NOT_REVALIDATED.csv',
        'audit':outdir/'JOIN_QUALITY_AUDIT.csv',
    }
    write_csv(paths['pressure_85_86_87'], ('match_id','minute','kickoff_utc','pressure_status','temporal_provenance')+PRESSURE_FIELDS, outP)
    write_csv(paths['model_A'], ('match_id','minute','kickoff_utc')+tuple(bfields), outA)
    write_csv(paths['model_B'], ('match_id','minute','kickoff_utc')+tuple(bfields)+PRESSURE_FIELDS, outB)
    write_csv(paths['labels'], ('match_id','minute','goal_after_87','reference_status','reference_confidence','reference_warning','label_revalidated'), outLabels)
    write_csv(paths['audit'], ('match_id','minute','baseline_matched','manifest_matched','reference_label_matched','label_revalidated','training_ready','issues','pressure_status','source_ft_crosscheck_status'),outQA)
    summary = {
        'status':'DATASET_PREPARATION_ONLY_NOT_TRAINING_READY',
        'pressure_rows':len(pressures),'pressure_unique_matches':len(set(k[0] for k in pressures)),
        'minute_distribution':dict(Counter(str(k[1]) for k in pressures)),
        'baseline_provided':bool(baseline_file),'baseline_candidate_features':bfields,
        'paired_87_A_and_B':len(outA),'label_rows_separate':len(outLabels),
        'unvalidated_reference_label_rows':len(outLabels),'certified_training_rows':0,
        'issue_counts':dict(counts),
        'warnings':[
            'Les features historiques sont reconstruites apres le match. As-of LIVE non certifie.',
            'Les labels historiques de reference ne sont pas revalides; certains ont deja presente des incoherences.',
            'Les totaux /stats de fin de match ne figurent dans aucun fichier de features.',
            'Les 22 features baseline sont une SELECTION CANDIDATE (pas le modele V3C complet).',
            'Ne pas entrainer ou annoncer un score de prediction avant audit de provenance et validation chronologique.',
        ],
        'files':{k:str(v) for k,v in paths.items()},
    }
    (outdir/'MERGE_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pressure', default='outputs/pressure_historical/pressure_v2_experimental/pressure_v2_85_86_87_EXPERIMENTAL.csv')
    ap.add_argument('--manifest', default='historical_1710_manifest.csv')
    ap.add_argument('--labels', default='historical_1710_reference_labels_SEPARATE.csv')
    ap.add_argument('--baseline', default=None,help='Fichier historique avec match_id et minute/snapshot_minute, a 87\' (optionnel)')
    ap.add_argument('--outdir', default='outputs/pressure_historical/pressure_v2_dataset_join')
    args=ap.parse_args()
    print(json.dumps(build_dataset(args.pressure,args.manifest,args.labels,args.outdir,args.baseline),indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
