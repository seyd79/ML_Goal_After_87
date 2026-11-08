#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path

from build_historical_ab_research_v3 import build_match
from v3c_pressure_features import get_events, event_minute

BASE = Path("outputs/v4_research/v41_mass350")
RAW = BASE / "raw_timelines"
MANIFEST = BASE / "manifest.csv"

GOALS = {"goal", "own_goal", "penalty_scored"}

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

lookup = {r["match_id"]: r for r in rows}

files = sorted(RAW.glob("mt_*.json"))

stats = Counter()
months = Counter()
days = set()
rejected = Counter()
positive = 0

print("=" * 70)
print("ÉTAPE 248 — AUDIT INTERMÉDIAIRE V4.1 MASS350")
print("=" * 70)
print("Timelines à examiner :", len(files))

for path in files:
    mid = path.stem

    try:
        if mid not in lookup:
            raise ValueError("ID absent du manifeste")

        record = json.loads(
            path.read_text(encoding="utf-8")
        )

        if record.get("match_id") != mid:
            raise ValueError("ID local différent")

        payload = record["timeline_response"]
        data = payload["data"]

        if data.get("match_id") != mid:
            raise ValueError("ID API différent")

        if data.get("coverage") != "full":
            raise ValueError("Couverture non complète")

        events = get_events(payload)

        if not isinstance(events, list) or not events:
            raise ValueError("Événements manquants")

        if len(events) != len(data.get("events", [])):
            raise ValueError("Nombre d'événements incohérent")

        if not any(
            isinstance(e, dict)
            and e.get("period") == "second_half"
            and e.get("type") == "period_end"
            for e in events
        ):
            raise ValueError("Fin de deuxième mi-temps absente")

        late_goals = 0

        for e in events:
            if not isinstance(e, dict):
                raise ValueError("Événement invalide")

            if e.get("period") != "second_half":
                continue

            if e.get("type") not in GOALS:
                continue

            minute = event_minute(e)

            if minute is None:
                raise ValueError("But sans minute")

            if minute == 87:
                raise ValueError("But à 87' ambigu")

            if minute > 87:
                late_goals += 1

        info = {
            k: "" if v is None else str(v)
            for k, v in lookup[mid].items()
        }

        a, b, quality = build_match(
            mid, info, data
        )

        if a is None or b is None:
            raise ValueError(
                "Reconstruction rejetée : " + str(quality)
            )

        if quality.get("status") != "RESEARCH_ONLY":
            raise ValueError("Statut inattendu")

        label = int(float(b["goal_after_87"]))

        if label != int(late_goals > 0):
            raise ValueError("Label incohérent")

        date = lookup[mid]["kickoff_utc"][:10]

        days.add(date)
        months[date[:7]] += 1
        positive += label
        stats["reconstructed"] += 1

    except Exception as exc:
        stats["rejected"] += 1
        rejected[str(exc)] += 1

        print("À EXAMINER :", mid, "|", str(exc))

print("\n" + "=" * 70)
print("BILAN")
print("=" * 70)

print("Timelines :", len(files))
print("Reconstruites :", stats["reconstructed"])
print("À examiner :", stats["rejected"])
print("Journées reconstruites :", len(days))
print("Labels positifs provisoires :", positive)

print("\n=== RÉPARTITION MENSUELLE ===")

for month, count in sorted(months.items()):
    print(month, ":", count)

print("\n=== MOTIFS DE REJET ===")

if not rejected:
    print("Aucun rejet")

for reason, count in rejected.most_common():
    print(count, "|", reason)

print("\nAucun appel API.")
print("Aucun fichier modifié.")
print("Aucun entraînement.")
print("Labels non certifiés indépendamment.")
print("V3-C inchangé.")
print("ÉTAPE 248 TERMINÉE.")
