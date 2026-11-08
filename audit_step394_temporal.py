#!/usr/bin/env python3
"""
GOAL AFTER 87 — ETAPE 394
Audit statique temporel, lecture seule.
Aucun import des modules du projet.
Aucun appel API, aucun acces aux datasets/labels.
"""

import ast
import re
from pathlib import Path

FILES = [
    "collect_historical_dataset_v2.py",
    "v41_live87_own_goal_364.py",
    "v41_prematch_temporal_adapter_313.py",
    "rebuild_clean_prematch_v2.py",
    "v41_prematch_safe_bridge_318.py",
]

PATTERNS = {
    "MINUTES ET PROLONGATIONS": [
        r"extract_event_minute",
        r"snapshot_minute",
        r"added_time",
        r"stoppage",
        r"extra_time",
        r"injury_time",
        r"elapsed",
    ],
    "RECONSTRUCTION LIVE": [
        r"reconstruct_from_events",
        r"shotmap_usable",
        r"timeline_before",
        r"shots_before",
        r"last_sot",
        r"corners_80_87",
    ],
    "PREMATCH ET BRIDGE": [
        r"build_prematch_asof",
        r"ResearchPrematchProxy",
        r"build_isolated_adapter",
        r"filter_before_cutoff",
        r"safe_cutoff",
        r"MARGIN_HOURS",
    ],
}

def lines_for_node(source, node):
    lines = source.splitlines()
    return "\n".join(
        f"{i:5d} | {lines[i-1]}"
        for i in range(node.lineno, node.end_lineno + 1)
    )

def show_function(tree, source, function_name):
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == function_name
    ]
    for node in matches:
        print(f"\n--- {function_name} "
              f"(lignes {node.lineno}-{node.end_lineno}) ---")
        print(lines_for_node(source, node))
    if not matches:
        print(f"\nNON TROUVEE : {function_name}")

print("=" * 78)
print("GOAL AFTER 87 — ETAPE 394")
print("AUDIT STATIQUE TEMPOREL — LECTURE SEULE")
print("=" * 78)

sources = {}
trees = {}

for filename in FILES:
    path = Path(filename)
    if not path.is_file():
        print(f"[ABSENT] {filename}")
        continue
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=filename)
    except (OSError, UnicodeError, SyntaxError) as exc:
        print(f"[ERREUR] {filename}: {exc}")
        continue

    sources[filename] = source
    trees[filename] = tree
    print(f"[OK] {filename} | {len(source.splitlines())} lignes")

print("\n" + "=" * 78)
print("1 — EXTRACTION EXACTE DES MINUTES")
print("=" * 78)

filename = "v41_live87_own_goal_364.py"
if filename in trees:
    show_function(
        trees[filename],
        sources[filename],
        "extract_event_minute",
    )

print("\n" + "=" * 78)
print("2 — FONCTIONS DE RECONSTRUCTION ET DE COUPURE")
print("=" * 78)

filename = "v41_live87_own_goal_364.py"
if filename in trees:
    tree = trees[filename]
    source_lines = sources[filename].splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "reconstruct_from_events":
            continue
        print(f"Fonction : lignes {node.lineno}-{node.end_lineno}")

        markers = (
            "timeline_before",
            "shots_before",
            "minute > snapshot_minute",
            "80 <= minute",
            "last_sot",
            "shotmap_usable",
            "timeline_shot_events",
        )
        selected = set()
        for number in range(node.lineno, node.end_lineno + 1):
            if any(m in source_lines[number-1] for m in markers):
                selected.update(
                    range(
                        max(node.lineno, number - 3),
                        min(node.end_lineno, number + 5) + 1,
                    )
                )
        for number in sorted(selected):
            print(f"{number:5d} | {source_lines[number-1]}")

print("\n" + "=" * 78)
print("3 — BRANCHEMENT EFFECTIF DU PREMATCH")
print("=" * 78)

for filename, source in sources.items():
    print(f"\n--- {filename} ---")
    lines = source.splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if any(
            re.search(pattern, line)
            for pattern in (
                r"ResearchPrematchProxy",
                r"build_isolated_adapter",
                r"build_prematch_asof",
                r"filter_before_cutoff",
                r"prematch\s*=",
                r"safe_bridge",
            )
        ):
            selected.update(
                range(max(0, i-2), min(len(lines), i+4))
            )
    if not selected:
        print("Aucune reference directe trouvee.")
    for i in sorted(selected):
        print(f"{i+1:5d} | {lines[i]}")

print("\n" + "=" * 78)
print("4 — INVENTAIRE DES REFERENCES TEMPORELLES")
print("=" * 78)

for category, patterns in PATTERNS.items():
    print(f"\n[{category}]")
    for filename, source in sources.items():
        counts = {
            pattern: len(re.findall(pattern, source, re.IGNORECASE))
            for pattern in patterns
        }
        found = {k: v for k, v in counts.items() if v}
        if found:
            print(f"{filename}: {found}")

print("\n" + "=" * 78)
print("5 — RISQUES A VERIFIER")
print("=" * 78)

checks = [
    "87+1 / 90+ : representation et parsing exacts",
    "Evenements de minute 87 : ordre et heure reelle",
    "Tirs et corners : absence de valeurs finales reutilisees",
    "Dernier tir cadre : minute valide et source temporelle",
    "PREMATCH : utilisation effective du bridge",
    "Historiques : disponibilite avant kickoff",
    "LIVE : disponibilite reelle des informations a 87'",
]
for i, check in enumerate(checks, 1):
    print(f"[A VERIFIER] {i}. {check}")

print("\nAucun label consulte.")
print("Aucune prediction modifiee.")
print("Aucun appel API effectue par ce script.")
print("Aucune certification automatique.")
print("ETAPE 394 — INSPECTION TERMINEE.")
