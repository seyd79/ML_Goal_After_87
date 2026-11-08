import ast
from pathlib import Path

FILES = [
    "collect_historical_dataset_v2.py",
    "v41_live87_own_goal_364.py",
    "v41_prematch_temporal_adapter_313.py",
    "v41_prematch_safe_bridge_318.py",
    "rebuild_clean_prematch_v2.py",
    "v41_targeted_collector_328.py",
]

FEATURES = [
    "score_87_home",
    "score_87_away",
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "minutes_since_last_shot_on_target",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
]

CRITICAL = [
    "reconstruct_live87",
    "fetch_timeline_and_shotmap",
    "extract_event_minute",
    "build_prematch_asof",
    "filter_before_cutoff",
    "get_team_history",
    "get_league_history",
    "own_goal",
]

MAX_HITS_PER_FEATURE = 6
MAX_HITS_PER_CONTROL = 5

sources = {}
functions = {}
missing_files = []

for filename in FILES:
    path = Path(filename)

    if not path.is_file():
        missing_files.append(filename)
        continue

    lines = path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()

    sources[filename] = lines

    try:
        tree = ast.parse("\n".join(lines), filename)
    except SyntaxError as exc:
        print("ERREUR SYNTAXE :", filename, exc)
        continue

    definitions = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            definitions.append((
                node.lineno,
                node.end_lineno,
                node.name
            ))

    functions[filename] = definitions


def containing_function(filename, line_no):
    matches = [
        (end - start, name)
        for start, end, name in functions.get(
            filename, []
        )
        if start <= line_no <= end
    ]

    if not matches:
        return "MODULE"

    return min(matches)[1]


def search(term, limit):
    matches = []

    for filename, lines in sources.items():
        for lineno, line in enumerate(lines, 1):
            if term.lower() in line.lower():
                matches.append((
                    filename,
                    lineno,
                    containing_function(filename, lineno),
                    line.strip()
                ))

    print(f"\n--- {term} ---")
    print("Occurrences totales :", len(matches))

    for filename, lineno, func, content in matches[:limit]:
        print(f"{filename}:{lineno}")
        print("  Fonction :", func)
        print("  Code :", content[:190])

    remaining = len(matches) - limit

    if remaining > 0:
        print("  ... autres occurrences :", remaining)

    if not matches:
        print("  NON LOCALISE dans les fichiers examines")


print("=" * 72)
print("ETAPE 392 - AUDIT TEMPOREL DU CODE V4.1-A2")
print("=" * 72)

print("\n=== FICHIERS ===")

for filename in FILES:
    status = (
        "TROUVE"
        if filename in sources
        else "ABSENT"
    )
    print(f"{status:7s} | {filename}")

print("\n=== TRACABILITE DES 10 FEATURES ===")

for feature in FEATURES:
    search(feature, MAX_HITS_PER_FEATURE)

print("\n=== FONCTIONS ET CONTROLES CRITIQUES ===")

for term in CRITICAL:
    search(term, MAX_HITS_PER_CONTROL)

print("\n=== BILAN ===")
print("Fichiers demandes :", len(FILES))
print("Fichiers lus :", len(sources))
print("Fichiers absents :", len(missing_files))

print("\nIMPORTANT :")
print("Presence d'un filtre != preuve de son application.")
print("Les fonctions et appels doivent etre verifies.")
print("Disponibilite temporelle : NON ENCORE CERTIFIEE")
print("Labels validation : NON CONSULTES")
print("Predictions et seuils : INCHANGES")
print("Aucun fichier de donnees ouvert")
print("Aucun fichier ecrit")
print("Aucun appel API")
print("ETAPE 392 TERMINEE")
