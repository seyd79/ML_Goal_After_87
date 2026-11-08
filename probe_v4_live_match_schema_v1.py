#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 215 — Diagnostic de la structure des matchs LIVE.
Un seul appel API.
Aucune prédiction, aucun snapshot et aucune modification V3-C.
"""

import importlib.util
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("diagnostic_live_api.py")

print("=" * 72)
print("ÉTAPE 215 — SONDE LIVE V4")
print("=" * 72)

if not SOURCE.is_file():
    raise SystemExit(
        "ERREUR : diagnostic_live_api.py introuvable"
    )

spec = importlib.util.spec_from_file_location(
    "v4_api_readonly",
    SOURCE,
)

module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
except Exception as exc:
    raise SystemExit(
        f"Impossible de charger le client API : {exc}"
    )

if not hasattr(module, "api_get"):
    raise SystemExit(
        "Fonction api_get introuvable"
    )

print(
    "Horodatage UTC :",
    datetime.now(timezone.utc).isoformat(),
)
print("Endpoint : /football/matches")
print("Paramètre : status=live")
print("Nombre d'appels API prévus : 1")
print()

try:
    response = module.api_get(
        "/football/matches",
        params={
            "status": "live",
            "page": 1,
            "per_page": 100,
        },
    )
except Exception as exc:
    raise SystemExit(
        f"Appel API impossible : "
        f"{type(exc).__name__}: {exc}"
    )


def find_matches(obj, depth=0):
    if depth > 6:
        return None

    if isinstance(obj, list):
        if all(isinstance(x, Mapping) for x in obj):
            return obj
        return None

    if isinstance(obj, Mapping):
        for key in (
            "matches",
            "data",
            "results",
            "response",
            "items",
            "result",
        ):
            if key in obj:
                found = find_matches(
                    obj[key],
                    depth + 1,
                )
                if found is not None:
                    return found

    return None


def extract_minute(match):
    live = match.get("live")

    if isinstance(live, Mapping):
        for key in ("elapsed_minutes", "minute"):
            value = live.get(key)

            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

    value = match.get("minute")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return None


def safe_display(value):
    blocked = {
        "token",
        "api_key",
        "authorization",
        "password",
        "secret",
        "access_token",
    }

    if isinstance(value, Mapping):
        return {
            str(key): safe_display(item)
            for key, item in value.items()
            if str(key).lower() not in blocked
        }

    if isinstance(value, list):
        return f"[liste de {len(value)} éléments]"

    if isinstance(value, (str, int, float, bool)):
        return value

    return None


matches = find_matches(response)

print("\n=== RÉSULTAT API ===")

if matches is None:
    print("Structure des matchs non reconnue.")

    if isinstance(response, Mapping):
        print(
            "Clés disponibles :",
            list(response.keys()),
        )

    raise SystemExit(
        "Diagnostic arrêté : schéma inconnu."
    )

print(
    "Matchs LIVE retournés :",
    len(matches),
)

if not matches:
    print(
        "Aucun match LIVE sur cette première page."
    )
    print("Aucune prédiction créée.")
    raise SystemExit(0)


near_87 = []
unknown = 0

for match in matches:
    minute = extract_minute(match)

    if minute is None:
        unknown += 1

    elif 84 <= minute <= 90:
        near_87.append((minute, match))

near_87.sort(
    key=lambda x: abs(x[0] - 87)
)

print(
    "Matchs entre 84' et 90' :",
    len(near_87),
)
print(
    "Matchs sans minute reconnue :",
    unknown,
)

selected = [
    match
    for minute, match in near_87[:5]
]

if not selected:
    selected = matches[:2]

    print(
        "\nAucun match proche de 87'. "
        "Affichage de deux exemples."
    )


for index, match in enumerate(selected, 1):

    print("\n" + "-" * 65)
    print(f"MATCH EXEMPLE {index}")
    print("-" * 65)

    print(
        "Clés principales :",
        sorted(match.keys()),
    )

    print(
        "Minute extraite :",
        extract_minute(match),
    )

    for field in (
        "id",
        "match_id",
        "status",
        "live",
        "score",
        "home",
        "away",
        "home_team",
        "away_team",
        "teams",
        "kickoff",
        "starting_at",
    ):
        if field in match:
            print(
                f"{field}: "
                f"{str(safe_display(match[field]))[:650]}"
            )


print("\n=== GARANTIES ===")
print("Un seul appel API prévu par le diagnostic.")
print("Aucune timeline téléchargée.")
print("Aucun modèle entraîné.")
print("Aucune prédiction enregistrée.")
print("Aucun snapshot enregistré.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 215 TERMINÉE.")
