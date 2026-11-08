from pathlib import Path
import ast
import json

from v3c_pressure_features import (
    get_events,
    event_minute,
    event_category,
)

CACHE = Path("outputs/pressure_historical/timeline_cache")
BUILDER = Path("build_historical_ab_research_v3.py")

SHOTS = [
    "mt_200803929",
    "mt_028974518",
]

CORNERS = [
    "mt_642130904",
    "mt_022398845",
    "mt_022910230",
]

print("=" * 72)
print("ÉTAPE 213 — DIAGNOSTIC CIBLÉ DES ÉCARTS V4")
print("=" * 72)


def load_events(mid):
    path = CACHE / f"{mid}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    events = get_events(payload)

    if events is None:
        raise ValueError(f"Timeline non reconnue : {mid}")

    return events


def safe_value(event, key):
    value = event.get(key)

    if isinstance(value, dict):
        return json.dumps(
            {
                k: v
                for k, v in value.items()
                if k in {
                    "id", "name", "type",
                    "side", "minute", "elapsed",
                    "team_id", "outcome",
                }
                and not isinstance(v, (dict, list))
            },
            ensure_ascii=False,
        )[:150]

    return str(value)[:100] if value is not None else "-"


def inspect_match(mid, only_corners=False):
    print("\n" + "-" * 72)
    print("MATCH :", mid)
    print("-" * 72)

    count = 0

    for event in load_events(mid):
        minute = event_minute(event)

        if minute is None or not (82 <= minute < 87):
            continue

        category = event_category(event)

        if only_corners:
            text = " ".join(
                str(event.get(k, ""))
                for k in ("type", "event_type", "name", "detail")
            ).lower()

            if "corner" not in text:
                continue

        count += 1

        print(
            f"\nÉVÉNEMENT {count} | minute={minute} "
            f"| catégorie ancienne={category}"
        )

        for key in (
            "id", "period", "time", "minute",
            "type", "event_type", "name",
            "detail", "outcome", "result",
            "shot_type", "on_target",
            "is_on_target", "team", "team_id",
        ):
            if key in event:
                print(
                    f"  {key:15s}: "
                    f"{safe_value(event, key)}"
                )

    print("\nÉvénements affichés :", count)


print("\n=== ÉCARTS DE TIRS ===")

for mid in SHOTS:
    inspect_match(mid)


print("\n=== ÉCARTS DE CORNERS ===")

for mid in CORNERS:
    inspect_match(mid, only_corners=True)


print("\n=== RÈGLES DU CONSTRUCTEUR HISTORIQUE ===")

if not BUILDER.is_file():
    print("Constructeur historique introuvable")
else:
    source = BUILDER.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    matching = set()

    terms = (
        "shot", "sot", "corner",
        "goal", "penalty",
        "second_half", "last5",
    )

    # Chercher uniquement dans build_match et ses fonctions voisines.
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        if node.name == "main":
            continue

        for n in range(node.lineno, node.end_lineno + 1):
            line = lines[n - 1].lower()

            if any(term in line for term in terms):
                matching.update(
                    range(
                        max(node.lineno, n - 1),
                        min(node.end_lineno, n + 1) + 1,
                    )
                )

    for n in sorted(matching)[:110]:
        line = lines[n - 1]

        if any(
            secret in line.lower()
            for secret in (
                "api_key", "authorization",
                "password", "secret", "token",
            )
        ):
            line = "[ligne sensible masquée]"

        print(f"L{n:4d} | {line[:160]}")

    if len(matching) > 110:
        print(
            f"... {len(matching)-110} "
            "lignes supplémentaires non affichées"
        )


print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucun label consulté.")
print("Aucun entraînement.")
print("Aucun dataset modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 213 TERMINÉE.")
