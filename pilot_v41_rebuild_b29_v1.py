#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import get_events, event_minute

ROOT = Path("outputs/v4_research")

MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"
RAW = ROOT / "v41_diverse" / "raw_timelines"

BUILDER = Path("build_historical_ab_research_v3.py")

GOAL_TYPES = {"goal", "own_goal", "penalty_scored"}

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

print("=" * 72)
print("ÉTAPE 241 — RECONSTRUCTION PILOTE B29 V4.1")
print("=" * 72)

if not MANIFEST.is_file() or not BUILDER.is_file():
    raise SystemExit("Manifeste ou reconstructeur absent")

with MANIFEST.open(
    encoding="utf-8-sig",
    newline=""
) as f:
    manifest = list(csv.DictReader(f))

if len(manifest) != 124:
    raise SystemExit("124 candidats attendus")

ids = [r["match_id"] for r in manifest]

if len(set(ids)) != len(ids):
    raise SystemExit("Doublons dans le manifeste")

lookup = {
    r["match_id"]: r
    for r in manifest
}

files = sorted(RAW.glob("mt_*.json"))

if len(files) != 9:
    raise SystemExit(
        f"9 timelines attendues, trouvées : {len(files)}"
    )

spec = importlib.util.spec_from_file_location(
    "v41_historical_builder",
    BUILDER
)

if spec is None or spec.loader is None:
    raise SystemExit("Chargement du builder impossible")

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

if not callable(getattr(builder, "build_match", None)):
    raise SystemExit("Fonction build_match absente")

statuses = Counter()
successful_days = Counter()

for path in files:
    mid = path.stem

    print("\n" + "=" * 68)
    print("MATCH :", mid)
    print("=" * 68)

    if mid not in lookup:
        raise SystemExit(
            f"Match absent du manifeste : {mid}"
        )

    row = lookup[mid]

    try:
        record = json.loads(
            path.read_text(encoding="utf-8")
        )

        if record.get("match_id") != mid:
            raise ValueError("ID du fichier incorrect")

        raw = record.get("timeline_response")

        if not isinstance(raw, dict):
            raise ValueError("Réponse non structurée")

        data = raw.get("data")

        if not isinstance(data, dict):
            raise ValueError("Bloc data invalide")

        if data.get("match_id") != mid:
            raise ValueError("ID API différent")

        if data.get("coverage") != "full":
            raise ValueError("Couverture non FULL")

        events = get_events(raw)

        if not isinstance(events, list) or not events:
            raise ValueError("Événements absents")

        period_end = any(
            isinstance(e, dict)
            and e.get("period") == "second_half"
            and e.get("type") == "period_end"
            for e in events
        )

        if not period_end:
            raise ValueError(
                "Fin de deuxième mi-temps absente"
            )

        # Vérifier que la frontière 87' ne comporte
        # pas de but dont l'ordre serait ambigu.
        goals_at_87 = [
            e for e in events
            if isinstance(e, dict)
            and str(
                e.get("period") or ""
            ).lower() == "second_half"
            and str(
                e.get("type") or ""
            ).lower() in GOAL_TYPES
            and event_minute(e) == 87
        ]

        if goals_at_87:
            raise ValueError(
                "But à 87' : examen temporel requis"
            )

        # Reproduire la structure d'information
        # du constructeur FETCH83.
        info = {
            key: "" if value is None else str(value)
            for key, value in row.items()
        }

        a, b, quality = builder.build_match(
            mid,
            info,
            data,
        )

        if a is None or b is None:
            print("RECONSTRUCTION REJETÉE")
            print("Diagnostic :", quality)
            statuses["REJECTED"] += 1
            continue

        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError(
                "Sorties du builder non structurées"
            )

        if not isinstance(quality, dict):
            raise ValueError(
                "Diagnostic du builder non structuré"
            )

        print(
            "Statut reconstructeur :",
            quality.get("status")
        )
        print(
            "Problèmes signalés :",
            quality.get("issues", "")
        )

        if quality.get("status") != "RESEARCH_ONLY":
            print("NON ACCEPTÉ : statut inattendu")
            statuses["STATUS_REVIEW"] += 1
            continue

        missing = [
            name for name in B6
            if name not in b
        ]

        if missing:
            print("B6 absentes :", missing)
            statuses["MISSING_B6"] += 1
            continue

        print("\n=== CARACTÉRISTIQUES B6 ===")
        for name in B6:
            print(
                f"{name:26s} : {b[name]}"
            )

        print("\nColonnes A :", len(a))
        print("Colonnes B :", len(b))

        print(
            "Score domicile à 87' :",
            b.get("score_87_home", "NON IDENTIFIÉ")
        )
        print(
            "Score extérieur à 87' :",
            b.get("score_87_away", "NON IDENTIFIÉ")
        )

        print(
            "\nRÉSULTAT : "
            "RECONSTRUCTION RESEARCH_ONLY"
        )

        statuses["RECONSTRUCTED"] += 1
        successful_days[row["kickoff_utc"][:10]] += 1

    except Exception as exc:
        print(
            "ERREUR :",
            type(exc).__name__,
            str(exc),
        )
        statuses["ERROR"] += 1


print("\n" + "=" * 72)
print("BILAN — ÉTAPE 241")
print("=" * 72)

print("Timelines examinées :", len(files))

for name in (
    "RECONSTRUCTED",
    "REJECTED",
    "STATUS_REVIEW",
    "MISSING_B6",
    "ERROR",
):
    print(f"{name:18s} : {statuses[name]}")

print("\nJournées avec reconstruction :")
for day, count in sorted(successful_days.items()):
    print(day, ":", count)

print("\n=== LIMITES ===")
print("Reconstruction exploratoire uniquement.")
print("Aucune certification des labels.")
print("Aucune certification de la complétude.")
print("Aucune nouvelle donnée TRAIN créée.")

print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucun résultat enregistré.")
print("Aucun modèle entraîné.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 241 TERMINÉE.")
