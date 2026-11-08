from pathlib import Path
import ast

FILES = {
    "capture_timeline_live.py": None,
    "live_feature_engineering.py": None,
    "diagnostic_live_api.py": None,
    "live_scanner_engine.py": [
        "get_live_matches",
        "get_timeline",
        "extract_events",
        "extract_event_minute",
        "extract_event_type",
        "extract_event_team",
        "normalize_event_type",
        "save_history_snapshot",
    ],
}

SENSITIVE = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "bearer ",
    "access_token",
    "os.environ",
    "load_dot_env",
)

def safe_line(line):
    if any(term in line.lower() for term in SENSITIVE):
        return "[ligne masquée : configuration sensible]"

    return line.rstrip()[:160]


print("=" * 72)
print("ÉTAPE 209 — ARCHITECTURE DE CAPTURE LIVE V4")
print("=" * 72)

for filename, selected in FILES.items():
    path = Path(filename)

    print("\n" + "=" * 72)
    print("FICHIER :", filename)
    print("=" * 72)

    if not path.is_file():
        print("ABSENT")
        continue

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print("Erreur syntaxique :", exc)
        continue

    functions = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            functions.append(node)

    functions.sort(key=lambda n: n.lineno)

    print("\n=== FONCTIONS ===")

    for node in functions:
        if selected is None or node.name in selected:
            print(
                f"L{node.lineno}-{node.end_lineno} "
                f"| {node.name}()"
            )

    print("\n=== EXTRAITS DU CODE ===")

    if selected is None:
        terms = (
            "timeline",
            "minute",
            "event",
            "fetch",
            "snapshot",
            "requests",
            "api_get",
        )

        targets = [
            node for node in functions
            if any(
                term in node.name.lower()
                for term in terms
            )
        ][:7]
    else:
        targets = [
            node for node in functions
            if node.name in selected
        ]

    for node in targets:
        print(
            f"\n--- {node.name} "
            f"(L{node.lineno}-{node.end_lineno}) ---"
        )

        # Extrait limité pour éviter un affichage massif.
        end = min(node.end_lineno, node.lineno + 27)

        for i in range(node.lineno - 1, end):
            print(
                f"{i+1:4d} | {safe_line(lines[i])}"
            )

        if node.end_lineno > end:
            print(
                f"... {node.end_lineno-end} "
                "lignes supplémentaires"
            )

    print("\n=== POINTS D'ENTRÉE ===")

    for node in tree.body:
        if isinstance(node, ast.If):
            snippet = ast.get_source_segment(
                source, node.test
            ) or ""

            if "__name__" in snippet:
                print(
                    f"Bloc d'exécution à la ligne "
                    f"{node.lineno}"
                )

print("\n=== BILAN ===")
print("Analyse statique uniquement.")
print("Aucun module du scanner importé.")
print("Aucun appel API.")
print("Aucun snapshot créé.")
print("Aucun fichier historique modifié.")
print("Scanner V3-C inchangé.")
