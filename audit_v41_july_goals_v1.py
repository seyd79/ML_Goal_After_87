#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
from pathlib import Path

from v3c_pressure_features import (
    get_events,
    event_minute,
)

ROOT = Path(
    "outputs/v4_research/v41_july_pilot/raw_timelines"
)

IDS = [
    "mt_155864505",
    "mt_159315811",
    "mt_258322938",
]

GOAL_TYPES = {
    "goal",
    "own_goal",
    "penalty_scored",
}

print("=" * 72)
print("ÉTAPE 232 — AUDIT DES BUTS HISTORIQUES")
print("=" * 72)

for mid in IDS:
    path = ROOT / f"{mid}.json"

    if not path.is_file():
        raise SystemExit(f"Fichier manquant : {path}")

    record = json.loads(
        path.read_text(encoding="utf-8")
    )

    if record.get("match_id") != mid:
        raise SystemExit("ID du fichier incohérent")

    events = get_events(
        record["timeline_response"]
    )

    if not isinstance(events, list):
        raise SystemExit("Timeline non reconnue")

    goals = []
    periods = Counter()

    for event in events:
        if not isinstance(event, dict):
            continue

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        if typ not in GOAL_TYPES:
            continue

        period = str(
            event.get("period") or ""
        ).strip().lower()

        minute = event_minute(event)

        periods[period] += 1

        goals.append({
            "period": period,
            "type": typ,
            "minute_parsed": minute,
            "minute_raw": event.get("minute"),
            "extra_minute": event.get("extra_minute"),
            "team": event.get("team"),
            "detail": event.get("detail"),
        })

    goals_second_half_after87 = [
        g for g in goals
        if g["period"] == "second_half"
        and g["minute_parsed"] is not None
        and g["minute_parsed"] > 87
    ]

    goals_at87 = [
        g for g in goals
        if g["period"] == "second_half"
        and g["minute_parsed"] == 87
    ]

    unknown = [
        g for g in goals
        if g["period"] == "second_half"
        and g["minute_parsed"] is None
    ]

    print("\n" + "=" * 72)
    print("MATCH :", mid)
    print(
        record.get("home_team"),
        "vs",
        record.get("away_team"),
    )

    print("\nButs par période :", dict(periods))
    print(
        "Événements de but après 87' "
        "en deuxième mi-temps :",
        len(goals_second_half_after87),
    )
    print(
        "Événements de but à la minute 87 :",
        len(goals_at87),
    )
    print(
        "Événements de but sans minute :",
        len(unknown),
    )

    print("\n=== DÉTAIL DES ÉVÉNEMENTS DE BUT ===")

    if not goals:
        print("Aucun événement de but reconnu")

    for g in goals:
        print(
            f"periode={g['period']:<23} "
            f"type={g['type']:<17} "
            f"minute={str(g['minute_parsed']):<5} "
            f"raw={g['minute_raw']} "
            f"extra={g['extra_minute']} "
            f"team={g['team']}"
        )

    if goals_at87 or unknown:
        print(
            "\nATTENTION : minute 87 ou minute "
            "inconnue à examiner manuellement."
        )

    if any(
        g["period"] not in (
            "first_half", "second_half"
        )
        for g in goals
    ):
        print(
            "\nPériodes supplémentaires présentes : "
            "exclues de la cible réglementaire."
        )

    print(
        "\nStatut du diagnostic : "
        "RESEARCH_ONLY — LABEL_NON_CERTIFIÉ"
    )

print("\n" + "=" * 72)
print("BILAN")
print("=" * 72)
print("Aucun label créé.")
print("Aucune prédiction recalculée.")
print("Aucun appel API.")
print("Aucun fichier modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 232 TERMINÉE.")
