#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from v3c_pressure_features import get_events, event_minute

BASE = Path(
    "outputs/v4_research/v41_diverse/raw_timelines"
)

MATCHES = {
    "mt_376037363": "Juventude vs Cuiabá",
    "mt_899705727": "Middlesbrough vs Wrexham",
}

GOAL_TYPES = {"goal", "own_goal", "penalty_scored"}

print("=" * 72)
print("ÉTAPE 240 — FRONTIÈRE 87' + RECONSTRUCTEUR B29")
print("=" * 72)

for mid, name in MATCHES.items():
    path = BASE / f"{mid}.json"

    if not path.is_file():
        raise SystemExit(f"Timeline introuvable : {path}")

    record = json.loads(
        path.read_text(encoding="utf-8")
    )

    if record.get("match_id") != mid:
        raise SystemExit(f"ID incohérent : {mid}")

    events = get_events(record["timeline_response"])

    if not isinstance(events, list):
        raise SystemExit(f"Timeline non reconnue : {mid}")

    boundary = []

    for event in events:
        if not isinstance(event, dict):
            continue

        period = str(
            event.get("period") or ""
        ).strip().lower()

        if period != "second_half":
            continue

        minute = event_minute(event)

        if minute == 87:
            boundary.append(event)

    print("\n" + "=" * 72)
    print("MATCH :", name)
    print("ID :", mid)
    print("Événements à 87' :", len(boundary))

    for i, event in enumerate(boundary, start=1):
        typ = str(
            event.get("type") or ""
        ).strip().lower()

        print(f"\nÉvénement {i}")
        print("Type :", typ)
        print("Minute calculée :", event_minute(event))
        print(
            "Données brutes :",
            json.dumps(event, ensure_ascii=False)
        )

        if typ in GOAL_TYPES:
            print(
                "ATTENTION : BUT À LA FRONTIÈRE 87'. "
                "Vérification temporelle nécessaire."
            )

    print(
        "\nConvention features : seuls les événements "
        "strictement avant 87' sont admissibles."
    )


print("\n" + "=" * 72)
print("STRUCTURE DE RECONSTRUCTION HISTORIQUE")
print("=" * 72)

builder = Path("build_v4_fetch83_full_v1.py")

if not builder.is_file():
    raise SystemExit("Reconstructeur FETCH83 introuvable")

lines = builder.read_text(
    encoding="utf-8"
).splitlines()

print("\n=== PRÉPARATION DES INFOS POUR build_match ===")

for i in range(122, min(158, len(lines))):
    print(f"{i+1:4d}: {lines[i]}")

print("\n=== SIGNATURE DE build_match ===")

source = Path("build_historical_ab_research_v3.py")

if not source.is_file():
    raise SystemExit("Builder historique introuvable")

source_lines = source.read_text(
    encoding="utf-8"
).splitlines()

for i, line in enumerate(source_lines):
    if line.startswith("def build_match("):
        for j in range(i, min(i+18, len(source_lines))):
            print(f"{j+1:4d}: {source_lines[j]}")
        break
else:
    raise SystemExit("Fonction build_match introuvable")


print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucun label créé.")
print("Aucun modèle entraîné.")
print("Aucun fichier modifié.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 240 TERMINÉE.")
