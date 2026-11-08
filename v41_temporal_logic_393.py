import ast
from pathlib import Path

SECTIONS = {
    "collect_historical_dataset_v2.py": [
        ("Chargement timeline / shotmap", 1111, 1155),
        ("Appel reconstruction live87", 1155, 1185),
        ("Assemblage prematch et live", 1305, 1355),
    ],
    "v41_live87_own_goal_364.py": [
        ("Extraction de la minute", 870, 920),
        ("Filtrage initial des evenements", 1585, 1635),
        ("Traitement des buts contre leur camp", 1705, 1745),
        ("Comptage des corners", 1775, 1810),
        ("Score a 87 minutes", 1815, 1850),
        ("Tirs recents et tirs cadres", 2185, 2245),
        ("Recence du dernier tir cadre", 2315, 2360),
    ],
    "v41_prematch_temporal_adapter_313.py": [
        ("Filtre chronologique isole", 55, 100),
        ("Branchement de l'adaptateur", 100, 145),
    ],
    "rebuild_clean_prematch_v2.py": [
        ("Historiques des equipes", 1705, 1760),
        ("Historique du championnat", 1795, 1835),
        ("Moyennes de buts", 2010, 2045),
    ],
    "v41_prematch_safe_bridge_318.py": [
        ("Debut du bridge", 1, 75),
    ],
}

def show_section(path, lines, title, start, end):
    print("\n" + "=" * 72)
    print(title)
    print(f"{path}:{start}-{end}")
    print("=" * 72)

    for index in range(start - 1, min(end, len(lines))):
        print(f"{index + 1:5d} | {lines[index]}")

def inspect_file(filename):
    path = Path(filename)

    if not path.is_file():
        raise RuntimeError(
            "STOP : fichier absent : " + filename
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )
    lines = text.splitlines()

    try:
        ast.parse(text, filename=filename)
    except SyntaxError as exc:
        raise RuntimeError(
            f"STOP : syntaxe incorrecte : {filename}"
        ) from exc

    print(f"\nFICHIER VERIFIE : {filename}")
    print("Nombre de lignes :", len(lines))

    for title, start, end in SECTIONS[filename]:
        show_section(
            filename, lines, title, start, end
        )

print("=" * 72)
print("ETAPE 393 - INSPECTION LOGIQUE TEMPORELLE")
print("=" * 72)

for filename in SECTIONS:
    inspect_file(filename)

print("\n" + "=" * 72)
print("POINTS A CERTIFIER")
print("=" * 72)

print("1. Filtrage strict des evenements a 87 minutes")
print("2. Traitement correct des minutes 87+ / 90+")
print("3. Aucun tir ou corner futur dans les cumuls")
print("4. Dernier tir cadre limite a la coupure")
print("5. Historiques prematch anterieurs au coup d'envoi")
print("6. Bridge et adaptateur effectivement utilises")
print("7. Disponibilite reelle des donnees dans le live")

print("\nAucune certification automatique.")
print("Aucun label de validation consulte.")
print("Aucun fichier de donnees ouvert.")
print("Aucune prediction modifiee.")
print("Aucun appel API.")
print("ETAPE 393 TERMINEE.")
