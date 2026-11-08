#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 228 — Vérification des filtres historiques.

Un seul appel API.
Aucune écriture.
Aucun label analysé.
Aucun modèle entraîné.
"""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

START = datetime(2026, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 7, 8, tzinfo=timezone.utc)

MODULE = Path("diagnostic_live_api.py")

if not MODULE.is_file():
    raise SystemExit("diagnostic_live_api.py introuvable")

spec = importlib.util.spec_from_file_location(
    "v41_api_diagnostic", MODULE
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

if not hasattr(api, "api_get"):
    raise SystemExit("Fonction api_get introuvable")


def extract_matches(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    for key in ("matches", "results", "items", "data"):
        value = payload.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            found = extract_matches(value)
            if found is not None:
                return found

    return None


def parse_date(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


print("=" * 72)
print("ÉTAPE 228 — TEST DE L'API HISTORIQUE")
print("=" * 72)

params = {
    "status": "finished",
    "date_from": "2026-07-01",
    "date_to": "2026-07-07",
    "page": 1,
    "per_page": 100,
}

print("\nPériode demandée : 1er au 7 juillet 2026")
print("Budget : 1 appel API")
print("Aucune écriture sur disque")

payload = api.api_get(
    "/football/matches",
    params,
)

matches = extract_matches(payload)

if matches is None:
    print("\nSTRUCTURE NON RECONNUE")
    print("Type de réponse :", type(payload).__name__)

    if isinstance(payload, dict):
        print("Clés principales :", list(payload.keys()))

    raise SystemExit(
        "Impossible de vérifier le filtre historique."
    )

print("\n=== RÉSULTAT DE LA REQUÊTE ===")
print("Matchs retournés :", len(matches))

in_range = 0
out_of_range = 0
unknown = 0
records = []

for match in matches:
    if not isinstance(match, dict):
        unknown += 1
        continue

    mid = match.get("id") or match.get("match_id")

    raw_date = (
        match.get("utc_date")
        or match.get("kickoff_utc")
        or match.get("start_time")
    )

    dt = parse_date(raw_date)

    status = match.get("status", "")

    if dt is None:
        unknown += 1
        category = "DATE_INCONNUE"

    elif START <= dt < END:
        in_range += 1
        category = "DANS_PERIODE"

    else:
        out_of_range += 1
        category = "HORS_PERIODE"

    records.append((
        category, str(mid), str(raw_date), str(status)
    ))

print("\n=== CONTRÔLE DES DATES ===")
print("Dans la période :", in_range)
print("Hors période :", out_of_range)
print("Dates non vérifiables :", unknown)

print("\n=== EXEMPLES ===")
for category, mid, dt, status in records[:12]:
    print(
        f"{category:14s} | "
        f"{mid:18s} | "
        f"{dt:27s} | "
        f"{status}"
    )

print("\n=== DIAGNOSTIC ===")

if not matches:
    print(
        "Aucun match retourné. "
        "Impossible de confirmer le filtrage."
    )

elif out_of_range:
    print(
        "ATTENTION : l'API a retourné des matchs "
        "hors période. Le filtrage n'est pas fiable "
        "pour cette requête."
    )

elif unknown:
    print(
        "Certaines dates sont absentes ou illisibles. "
        "Vérification incomplète."
    )

else:
    print(
        "Toutes les dates retournées correspondent "
        "à la période demandée."
    )
    print(
        "Résultat compatible avec un filtrage correct, "
        "mais une seule page ne suffit pas "
        "à le certifier définitivement."
    )

print("\n=== GARANTIES ===")
print("Aucun label consulté.")
print("Aucun nouveau cache créé.")
print("Aucun modèle modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 228 TERMINÉE.")
