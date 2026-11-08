#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
from pathlib import Path

from build_historical_ab_research_v3 import SHOTS, SOT
from v3c_pressure_features import get_events, event_minute

ROOT = Path(
    "outputs/v4_research/v41_july_pilot/raw_timelines"
)

IDS = [
    "mt_155864505",
    "mt_159315811",
    "mt_258322938",
]

print("=" * 72)
print("ÉTAPE 231 — AUDIT DES PÉRIODES HISTORIQUES")
print("=" * 72)

for mid in IDS:
    path = ROOT / (mid + ".json")

    if not path.is_file():
        raise SystemExit(f"Fichier manquant : {path}")

    record = json.loads(path.read_text(encoding="utf-8"))

    if record.get("match_id") != mid:
        raise SystemExit(f"Identifiant incorrect : {mid}")

    payload = record.get("timeline_response")
    events = get_events(payload)

    if not isinstance(events, list):
        raise SystemExit(f"Timeline invalide : {mid}")

    periods = Counter()
    periods_by_type = {}
    window_early = Counter()
    window_recent = Counter()

    unknown_minutes = 0
    excluded_from_features = Counter()
    examples = []

    for e in events:
        if not isinstance(e, dict):
            continue

        period = str(
            e.get("period") or "MISSING"
        ).strip().lower()

        typ = str(
            e.get("type") or "MISSING"
        ).strip().lower()

        periods[period] += 1

        periods_by_type.setdefault(
            period, Counter()
        )[typ] += 1

        minute = event_minute(e)

        if period != "second_half":
            if typ in {
                "penalty_scored",
                "penalty_missed",
                "penalty_saved",
                "goal",
                "own_goal",
            }:
                excluded_from_features[
                    (period, typ)
                ] += 1

                if len(examples) < 8:
                    examples.append({
                        "period": period,
                        "type": typ,
                        "minute": e.get("minute"),
                        "extra_minute": e.get("extra_minute"),
                        "elapsed": e.get("elapsed"),
                    })

            continue

        relevant = (
            typ in SHOTS
            or typ in SOT
            or typ == "corner_kick"
        )

        if relevant and minute is None:
            unknown_minutes += 1
            continue

        if minute is None:
            continue

        if 77 <= minute < 82:
            target = window_early
        elif 82 <= minute < 87:
            target = window_recent
        else:
            continue

        if typ in SHOTS:
            target["shots"] += 1

        if typ in SOT:
            target["sot"] += 1

        if typ == "corner_kick":
            target["corners"] += 1

    print("\n" + "=" * 72)
    print("MATCH :", mid)
    print("Événements :", len(events))
    print("Périodes :", dict(periods))

    print("\nTypes par période :")
    for period, counts in periods_by_type.items():
        print(
            f"  {period}: "
            f"{dict(counts)}"
        )

    print("\nPression 77'–81' :")
    print(dict(window_early))

    print("\nPression 82'–86' :")
    print(dict(window_recent))

    print(
        "\nÉvénements offensifs sans minute :",
        unknown_minutes
    )

    print(
        "\nÉvénements de but ou penalty "
        "hors deuxième mi-temps :"
    )
    print(dict(excluded_from_features))

    if examples:
        print("\nExemples de structure :")
        for e in examples:
            print(" ", e)

    if unknown_minutes:
        print(
            "ATTENTION : minuterie incomplète."
        )

print("\n" + "=" * 72)
print("BILAN")
print("=" * 72)
print("Trois timelines inspectées.")
print("Aucun label généré.")
print("Aucun modèle entraîné.")
print("Aucun appel API.")
print("Aucun fichier modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 231 TERMINÉE.")
