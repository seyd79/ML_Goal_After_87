import ast
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

SOURCE = Path("live_scanner_engine_shadow_v1.py")

tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "analyze_timeline"
]

if len(functions) != 1:
    raise RuntimeError(
        "Fonction analyze_timeline introuvable ou dupliquée"
    )

# Extraire seulement la fonction.
# Aucun import du scanner complet.
module = ast.Module(
    body=functions,
    type_ignores=[]
)
ast.fix_missing_locations(module)

def minute(event):
    return event.get("minute")

def event_type(event):
    return event.get("type", "")

def normalize(value):
    return str(value).lower().strip()

def belongs_to_home(event, *args):
    side = event.get("side")
    if side == "home":
        return True
    if side == "away":
        return False
    return None

namespace = {
    "Any": Any,
    "Dict": Dict,
    "Mapping": Mapping,
    "Optional": Optional,
    "Sequence": Sequence,
    "extract_event_minute": minute,
    "extract_event_type": event_type,
    "normalize_event_type": normalize,
    "event_belongs_to_home": belongs_to_home,
}

exec(
    compile(module, str(SOURCE), "exec"),
    namespace
)

analyze = namespace["analyze_timeline"]

events = [
    {"minute": 82, "type": "shot", "side": "home"},
    {"minute": 83, "type": "corner_kick", "side": "away"},
    {"minute": 84, "type": "shot_on_target", "side": "home"},
    {"minute": 85, "type": "shot", "side": "away"},
    {"minute": 86, "type": "goal", "side": "home"},
    {"minute": 87, "type": "corner_kick", "side": "home"},
    {"minute": 88, "type": "yellow_card", "side": "away"},
    {"minute": 90, "type": "shot_on_target", "side": "away"},
]

print("=== ÉTAPE 126 : TEST TEMPOREL SHADOW ===")

for current_minute in (85, 86, 87):
    full = analyze(
        events,
        current_minute=current_minute
    )

    historical_snapshot = [
        event for event in events
        if event["minute"] < current_minute
    ]

    expected = analyze(
        historical_snapshot,
        current_minute=current_minute
    )

    if full != expected:
        differing = sorted(
            key for key in set(full) | set(expected)
            if full.get(key) != expected.get(key)
        )

        raise AssertionError(
            f"FUITE À {current_minute}' : {differing}"
        )

    print(
        f"PASS {current_minute}' | "
        f"événements autorisés="
        f"{len(historical_snapshot)}/{len(events)}"
    )

# Vérification spécifique de la frontière 87'.
result_87 = analyze(events, current_minute=87)

assert result_87["last_shot_minute"] == 85
assert result_87["shots_80_87"] == 3
assert result_87["corners_80_87"] == 1

print("PASS frontière 87'")
print("PASS exclusion des événements futurs")

print("\n=== BILAN ===")
print("Tests temporels : 3/3")
print("Test frontière 87' : PASS")
print("Aucune fuite détectée sur les événements simulés")
print("Scanner original non modifié")
print("Aucun appel API")
print("Aucun modèle entraîné")
