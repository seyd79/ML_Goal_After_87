#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G02-G20/P2A -- audit des NOMS de timelines, sans ouvrir leur contenu.

Depuis la racine du projet ML_Goal_After_87 :
    python3 -B g02g20_p2a_timeline_filename_coverage_readonly.py

Cette etape ne valide PAS les labels. Elle valide uniquement que chaque match
gele possede un nom de fichier timeline dans les 4 repertoires historiques.
0 API, 0 lecture JSON timeline, 0 label, 0 ecriture, 0 entrainement.
Python 3.9+, bibliotheque standard uniquement.
"""
from collections import Counter, defaultdict
from pathlib import Path
import g02g20_p1_label_source_inventory_readonly as p1


def check(condition, message):
    if not condition:
        raise SystemExit('G02-G20/P2A STOP : ' + message)


def main():
    print('=== G02-G20/P2A | COUVERTURE TIMELINES PAR NOM | READONLY ===')
    for path, expected in p1.EXPECTED.items():
        check(path.is_file() and not path.is_symlink(), 'fichier gele manquant ou lien : ' + str(path))
        check(p1.sha(path) == expected, 'SHA different : ' + str(path))
        print('SHA OK : ' + path.name)

    split = p1.rows(p1.SPLIT, ('match_id', 'kickoff_utc', 'partition'))
    elig = p1.rows(p1.ELIG, ('match_id', 'kickoff_utc', 'partition', 'research_eligibility'))
    allowed = p1.rows(p1.ALLOW, ('match_id', 'kickoff_utc', 'partition'))
    check(len(split) == len(elig) == 1500 and set(split) == set(elig), 'les 1500 IDs ne correspondent pas')
    check(len(allowed) == 1114, 'allowlist != 1114')

    all_counts = Counter()
    elig_counts = Counter()
    eligible_allowed = set()
    test_ids = set()
    for mid, row in split.items():
        partition = row['partition']
        check(partition in p1.TOTALS, 'partition non reconnue')
        check(elig[mid]['partition'] == partition, 'partition incompatible : ' + mid)
        check(row['kickoff_utc'] == elig[mid]['kickoff_utc'], 'date incompatible : ' + mid)
        all_counts[partition] += 1
        if partition == 'TEST':
            test_ids.add(mid)
        if elig[mid]['research_eligibility'] == 'RESEARCH_ELIGIBLE':
            elig_counts[partition] += 1
            if partition != 'TEST':
                eligible_allowed.add(mid)
        else:
            check(elig[mid]['research_eligibility'] == 'NOT_ELIGIBLE', 'statut non reconnu : ' + mid)
    check(dict(all_counts) == p1.TOTALS, 'repartition chronologique incorrecte')
    check(dict(elig_counts) == p1.ELIGIBLE, 'admissibilite incorrecte')
    check(set(allowed) == eligible_allowed, 'allowlist != eligibles TRAIN/VALIDATION')
    check(not set(allowed).intersection(test_ids), 'acces TEST dans allowlist')
    check(dict(Counter(row['partition'] for row in allowed.values())) == {'TRAIN': 845, 'VALIDATION': 269},
          'repartition allowlist invalide')
    print('ALLOWLIST OK : TRAIN 845 | VALIDATION 269 | TEST 0')
    print('TEST GARDE : 302 matchs, dont 286 admissibles; aucun label ouvert')

    located = {}
    by_directory = defaultdict(Counter)
    untracked = []
    for directory in p1.RAW_DIRS:
        check(directory.is_dir() and not directory.is_symlink(), 'dossier raw absent / lien symbolique : ' + str(directory))
        for path in directory.iterdir():
            # L'inventaire ne touche jamais au contenu du fichier.
            if path.is_symlink() or not path.is_file() or not (path.name.startswith('mt_') and path.suffix == '.json'):
                continue
            mid = path.stem
            check(mid not in located, 'timeline filename en doublon : ' + mid)
            located[mid] = str(path)
            if mid in split:
                part = split[mid]['partition']
                by_directory[directory.name + ' @ ' + directory.parent.name][part] += 1
            else:
                if len(untracked) < 5:
                    untracked.append(mid)
    missing = set(split) - set(located)
    extra = set(located) - set(split)
    print('NOMS TIMELINES TROUVES :', len(located))
    print('MATCHS DU SPLIT SANS NOM TIMELINE :', len(missing))
    print('NOMS TIMELINES HORS SPLIT :', len(extra))
    for name, counts in by_directory.items():
        print('DOSSIER :', name, '|', dict(counts), '| total', sum(counts.values()))
    if missing:
        print('EXEMPLES IDs MANQUANTS :', sorted(missing)[:5])
    if extra:
        print('EXEMPLES IDs HORS SPLIT :', sorted(extra)[:5])
    check(not missing and not extra and len(located) == 1500, 'couverture exacte 1500 non satisfaite')
    print('COUVERTURE PAR PARTITION :', dict(Counter(split[mid]['partition'] for mid in located)))
    print('COUVERTURE ALLOWLIST :', len(set(located).intersection(allowed)), '/ 1114')
    print('COUVERTURE TEST (NOMS UNIQUEMENT) :', len(set(located).intersection(test_ids)), '/ 302')

    # Les noms des fichiers CSV/JSON des sorties V4.2 restent des hypotheses,
    # jamais des preuves qu'un label est correct. Aucun de ces fichiers n'est lu.
    names = p1.discover_names_only(p1.BASE)
    likely = []
    for path in names:
        filename = path.name.lower()
        full = str(path).lower()
        if ('v42' in full or 'g02' in full or '1500' in filename) and not any(
            k in filename for k in ('no_label', 'no_target', 'no_outcome', 'quarantine')
        ):
            likely.append(path)
    print('AUTRES SOURCES POSSIBLES (NOMS SEULEMENT) :', len(likely))
    for path in sorted(likely)[:30]:
        print('CANDIDAT NOM SEULEMENT :', path)
    if len(likely) > 30:
        print('...', len(likely) - 30, 'noms supplementaires non affiches')

    print('ATTENTION : presence des timelines != provenance, contenu et certification des labels')
    print('ATTENTION : si une timeline contient des buts apres 87, elle n a PAS ete ouverte')
    print('AS-OF NON CERTIFIE | 0 API | 0 ECRITURE | 0 LABEL OUVERT | 0 TRAIN')
    print('G02-G20/P2A METADATA PASS | SOURCE PHYSIQUE COUVERTE | LABELS NON CERTIFIES | TEST FERME')


if __name__ == '__main__':
    main()
