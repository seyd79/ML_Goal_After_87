#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path

from v3c_pressure_features import (
    get_events,
    event_minute,
)

ROOT = Path("outputs/v4_research")
BASE = ROOT / "v41_diverse"

VALID = BASE / "raw_timelines"
REJECTED = BASE / "rejected_timelines"

MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"

print("=" * 72)
print("ÉTAPE 236 — AUDIT DE COUVERTURE DES TIMELINES")
print("=" * 72)

with MANIFEST.open(
    encoding="utf-8-sig",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 124:
    raise SystemExit("124 candidats attendus")

if len({r["match_id"] for r in rows}) != 124:
    raise SystemExit("Doublons dans le manifeste")

by_day = Counter()
statuses = Counter()
coverages = Counter()

print("\n=== AUDIT DES RÉPONSES CONSERVÉES ===")

for row in rows:
    mid = row["match_id"]
    day = row["kickoff_utc"][:10]

    by_day[day] += 1

    good = VALID / f"{mid}.json"
    bad = REJECTED / f"{mid}.json"

    if good.exists() and bad.exists():
        raise SystemExit(
            f"Deux statuts contradictoires : {mid}"
        )

    if not good.exists() and not bad.exists():
        statuses[(day, "PENDING")] += 1
        continue

    path = good if good.exists() else bad
    state = "SAVED" if good.exists() else "QUARANTINE"

    record = json.loads(
        path.read_text(encoding="utf-8")
    )

    if record.get("match_id") != mid:
        raise SystemExit(f"ID incompatible : {mid}")

    payload = record.get("timeline_response")

    if not isinstance(payload, dict):
        raise SystemExit(
            f"Réponse non structurée : {mid}"
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise SystemExit(
            f"Bloc data inattendu : {mid}"
        )

    if str(data.get("match_id")) != mid:
        raise SystemExit(
            f"ID API incompatible : {mid}"
        )

    coverage = data.get("coverage")
    raw_events = data.get("events")
    events = get_events(payload)

    coverage_label = str(coverage)
    coverages[(state, coverage_label)] += 1
    statuses[(day, state)] += 1

    raw_count = (
        len(raw_events)
        if isinstance(raw_events, list)
        else None
    )

    parsed_count = (
        len(events)
        if isinstance(events, list)
        else None
    )

    periods = Counter()
    unknown_minutes = 0

    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue

            period = str(
                event.get("period") or ""
            ).strip().lower()

            periods[period] += 1

            typ = str(
                event.get("type") or ""
            ).strip().lower()

            if (
                period == "second_half"
                and typ in {
                    "goal", "own_goal",
                    "shot_on_target",
                    "shot_off_target",
                    "shot_blocked",
                    "shot_woodwork",
                    "corner_kick",
                }
                and event_minute(event) is None
            ):
                unknown_minutes += 1

    print("\nMatch :", mid)
    print("Journée :", day)
    print("Statut :", state)
    print("Coverage API :", repr(coverage))
    print("Type de events :", type(raw_events).__name__)
    print("Événements bruts :", raw_count)
    print("Événements reconnus :", parsed_count)
    print("Périodes :", dict(periods))
    print(
        "Événements pertinents sans minute :",
        unknown_minutes,
    )

    if state == "QUARANTINE":
        if isinstance(raw_events, list):
            print(
                "Events réellement vide :",
                len(raw_events) == 0,
            )
        else:
            print(
                "ATTENTION : events n'est pas une liste"
            )

    if state == "SAVED":
        if coverage != "full":
            print(
                "ATTENTION : couverture non déclarée full"
            )
        if not periods.get("second_half"):
            print(
                "ATTENTION : deuxième mi-temps absente"
            )

print("\n=== RÉPARTITION DES 124 CANDIDATS ===")

for day in sorted(by_day):
    print(
        day,
        "| total=", by_day[day],
        "| saved=", statuses[(day, "SAVED")],
        "| quarantaine=", statuses[(day, "QUARANTINE")],
        "| pending=", statuses[(day, "PENDING")],
    )

print("\n=== COUVERTURES RENCONTRÉES ===")

for (state, coverage), count in sorted(
    coverages.items()
):
    print(
        f"{state:12s} | "
        f"coverage={coverage!r} | n={count}"
    )

print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucun label calculé.")
print("Aucun fichier modifié.")
print("Aucun entraînement.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 236 TERMINÉE.")
