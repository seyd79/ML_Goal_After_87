import json
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

DIAGNOSTIC_DIR = PROJECT_DIR / "outputs" / "api_diagnostic"


# Champs particulièrement intéressants pour notre scanner LIVE
INTERESTING_KEYWORDS = [
    "minute",
    "minutes",
    "elapsed",
    "clock",
    "time",
    "period",
    "status",
    "score",
    "home_score",
    "away_score",
    "shots",
    "shot",
    "shots_on_target",
    "shots_off_target",
    "blocked",
    "corners",
    "corner",
    "fouls",
    "foul",
    "yellow",
    "red",
    "card",
    "cards",
    "offsides",
    "offside",
    "substitution",
    "substitutions",
    "subs",
    "goal",
    "goals",
    "event",
    "timeline",
    "possession",
    "xg",
    "expected",
    "odds",
]


# ============================================================
# UTILITAIRES
# ============================================================

def load_json(path: Path):
    """Charge un fichier JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shorten(value: Any, max_length: int = 250) -> str:
    """Transforme une valeur en texte court."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":")
        )
    except Exception:
        text = str(value)

    if len(text) > max_length:
        return text[:max_length] + "..."

    return text


def is_interesting_key(key: str) -> bool:
    """Détermine si une clé est intéressante."""
    key_lower = key.lower()

    return any(
        keyword in key_lower
        for keyword in INTERESTING_KEYWORDS
    )


def print_structure(
    obj: Any,
    path: str = "root",
    depth: int = 0,
    max_depth: int = 8
):
    """
    Affiche récursivement la structure du JSON.
    """

    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}... profondeur maximale atteinte")
        return

    if isinstance(obj, dict):

        for key, value in obj.items():

            current_path = f"{path}.{key}"

            if isinstance(value, dict):
                print(
                    f"{indent}{current_path} = "
                    f"dict ({len(value)} clés)"
                )

                if is_interesting_key(key):
                    print(
                        f"{indent}  VALEUR = "
                        f"{shorten(value)}"
                    )

                print_structure(
                    value,
                    current_path,
                    depth + 1,
                    max_depth
                )

            elif isinstance(value, list):

                print(
                    f"{indent}{current_path} = "
                    f"list ({len(value)} éléments)"
                )

                if is_interesting_key(key):
                    print(
                        f"{indent}  VALEUR = "
                        f"{shorten(value)}"
                    )

                # On inspecte seulement les premiers éléments
                for i, item in enumerate(value[:3]):

                    item_path = f"{current_path}[{i}]"

                    if isinstance(item, (dict, list)):
                        print_structure(
                            item,
                            item_path,
                            depth + 1,
                            max_depth
                        )
                    else:
                        print(
                            f"{indent}  "
                            f"{item_path} = {item}"
                        )

            else:

                if is_interesting_key(key):

                    print(
                        f"{indent}{current_path} = "
                        f"{repr(value)} "
                        f"[{type(value).__name__}]"
                    )

    elif isinstance(obj, list):

        print(
            f"{indent}{path} = "
            f"list ({len(obj)} éléments)"
        )

        for i, item in enumerate(obj[:3]):

            item_path = f"{path}[{i}]"

            if isinstance(item, (dict, list)):
                print_structure(
                    item,
                    item_path,
                    depth + 1,
                    max_depth
                )
            else:
                print(
                    f"{indent}{item_path} = "
                    f"{repr(item)}"
                )

    else:

        print(
            f"{indent}{path} = "
            f"{repr(obj)}"
        )


def find_interesting_fields(
    obj: Any,
    path: str = "root"
):
    """
    Recherche récursivement tous les champs
    dont le nom contient un mot-clé intéressant.
    """

    results = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            current_path = f"{path}.{key}"

            if is_interesting_key(key):

                results.append(
                    (
                        current_path,
                        value
                    )
                )

            results.extend(
                find_interesting_fields(
                    value,
                    current_path
                )
            )

    elif isinstance(obj, list):

        for i, item in enumerate(obj):

            current_path = f"{path}[{i}]"

            results.extend(
                find_interesting_fields(
                    item,
                    current_path
                )
            )

    return results


# ============================================================
# ANALYSE D'UN FICHIER
# ============================================================

def analyze_file(path: Path):

    print()
    print("=" * 80)
    print(f"FICHIER : {path.name}")
    print("=" * 80)

    try:
        data = load_json(path)

    except Exception as e:

        print(f"❌ Impossible de lire le JSON : {e}")
        return

    print()
    print("TYPE RACINE")
    print("-" * 80)

    print(type(data).__name__)

    if isinstance(data, dict):

        print()
        print("CLÉS RACINE")
        print("-" * 80)

        for key in data.keys():
            print(f"  - {key}")

    elif isinstance(data, list):

        print()
        print(f"Nombre d'éléments : {len(data)}")

    print()
    print("CHAMPS IMPORTANTS DÉTECTÉS")
    print("-" * 80)

    fields = find_interesting_fields(data)

    if not fields:

        print("Aucun champ intéressant détecté.")

    else:

        # Évite d'afficher exactement le même chemin plusieurs fois
        seen = set()

        for path_str, value in fields:

            if path_str in seen:
                continue

            seen.add(path_str)

            print()
            print(f"PATH : {path_str}")
            print(f"TYPE : {type(value).__name__}")
            print(f"VALUE: {shorten(value, 500)}")

    print()
    print("STRUCTURE COMPLÈTE")
    print("-" * 80)

    print_structure(data)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LECTEUR DES JSON - DIAGNOSTIC THESTATSAPI")
    print("=" * 80)

    print()
    print(f"Dossier : {DIAGNOSTIC_DIR}")

    if not DIAGNOSTIC_DIR.exists():

        print()
        print("❌ Le dossier n'existe pas :")
        print(DIAGNOSTIC_DIR)

        return

    json_files = sorted(
        DIAGNOSTIC_DIR.glob("*.json")
    )

    if not json_files:

        print()
        print("❌ Aucun fichier JSON trouvé.")

        return

    print()
    print(
        f"✓ {len(json_files)} fichier(s) JSON trouvé(s)"
    )

    print()

    for path in json_files:

        analyze_file(path)

    print()
    print("=" * 80)
    print("LECTURE TERMINÉE")
    print("=" * 80)

    print()
    print("Les informations les plus importantes à vérifier sont :")
    print()
    print("  1. minute / elapsed / clock")
    print("  2. score")
    print("  3. shots")
    print("  4. shots_on_target")
    print("  5. shots_off_target")
    print("  6. corners")
    print("  7. fouls")
    print("  8. yellow_cards / red_cards")
    print("  9. substitutions")
    print(" 10. goals / events")
    print(" 11. possession")
    print(" 12. xG")
    print(" 13. odds")
    print()
    print("IMPORTANT : aucune clé API n'est affichée par ce script.")


if __name__ == "__main__":
    main()