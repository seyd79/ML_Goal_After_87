#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path

from v3c_pressure_features import get_events, event_minute
from build_historical_ab_research_v3 import SHOTS, SOT

ROOT = Path("outputs/v4_research")
MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"

BASE = ROOT / "v41_diverse"
VALID = BASE / "raw_timelines"
REJECTED = BASE / "rejected_timelines"

GOALS = {"goal", "own_goal", "penalty_scored"}
RELEVANT = set(SHOTS) | set(SOT) | {"corner_kick"}

print("=" * 72)
print("ÉTAPE 239 — QUALITÉ DES TIMELINES V4.1")
print("=" * 72)

with MANIFEST.open(
    encoding="utf-8-sig", newline=""
) as f:
    manifest = list(csv.DictReader(f))

if len(manifest) != 124:
    raise SystemExit("Manifeste inattendu")

ids = [r["match_id"] for r in manifest]

if len(set(ids)) != 124:
    raise SystemExit("Doublons dans le manifeste")

stats = Counter()
days = Counter()

for row in manifest:
    mid = row["match_id"]
    day = row["kickoff_utc"][:10]

    good = VALID / f"{mid}.json"
    bad = REJECTED / f"{mid}.json"

    if good.exists() and bad.exists():
        raise SystemExit(
            f"Fichiers contradictoires : {mid}"
        )

    if bad.exists():
        stats["quarantine"] += 1
        continue

    if not good.exists():
        stats["pending"] += 1
        continue

    stats["saved"] += 1
    days[day] += 1

    record = json.loads(
        good.read_text(encoding="utf-8")
    )

    if record.get("match_id") != mid:
        raise SystemExit(f"ID incorrect : {mid}")

    payload = record.get("timeline_response")

    if not isinstance(payload, dict):
        raise SystemExit(f"Réponse invalide : {mid}")

    data = payload.get("data")

    if not isinstance(data, dict):
        raise SystemExit(f"Bloc data invalide : {mid}")

    if str(data.get("match_id")) != mid:
        raise SystemExit(f"ID API différent : {mid}")

    raw_events = data.get("events")
    events = get_events(payload)
    coverage = data.get("coverage")

    issues = []

    if coverage != "full":
        issues.append("COVERAGE_NOT_FULL")

    if not isinstance(raw_events, list):
        issues.append("RAW_EVENTS_NOT_LIST")

    if not isinstance(events, list):
        issues.append("EVENTS_NOT_RECOGNIZED")
        events = []

    if not events:
        issues.append("EMPTY_EVENTS")

    if isinstance(raw_events, list):
        if len(events) != len(raw_events):
            issues.append("PARSER_COUNT_MISMATCH")

    periods = Counter()
    types = Counter()
    recent = Counter()

    unknown_minutes = 0
    at_87 = 0
    post87_goals = 0
    other_period_goals = 0

    for event in events:
        if not isinstance(event, dict):
            issues.append("INVALID_EVENT")
            continue

        period = str(
            event.get("period") or ""
        ).strip().lower()

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        periods[period] += 1
        types[typ] += 1

        if period != "second_half":
            if typ in GOALS:
                other_period_goals += 1
            continue

        minute = event_minute(event)

        if minute is None and (
            typ in RELEVANT or typ in GOALS
        ):
            unknown_minutes += 1
            continue

        if minute is None:
            continue

        if minute == 87 and (
            typ in RELEVANT or typ in GOALS
        ):
            at_87 += 1

        if typ in GOALS and minute > 87:
            post87_goals += 1

        if 82 <= minute < 87:
            if typ in SHOTS:
                recent["shots"] += 1
            if typ in SOT:
                recent["sot"] += 1
            if typ == "corner_kick":
                recent["corners"] += 1

    if not periods["second_half"]:
        issues.append("SECOND_HALF_MISSING")

    if unknown_minutes:
        issues.append("RELEVANT_MINUTES_UNKNOWN")

    if at_87:
        issues.append("BOUNDARY_87_REVIEW")

    if any(
        period in periods
        for period in (
            "extra_time_first_half",
            "extra_time_second_half",
            "penalties",
        )
    ):
        stats["with_extra_periods"] += 1

    if issues:
        stats["needs_review"] += 1
    else:
        stats["no_structural_issue"] += 1

    print("\n" + "-" * 65)
    print("Match :", mid, "| Date :", day)
    print("Coverage :", coverage)
    print("Événements :", len(events))
    print("Périodes :", dict(periods))
    print("Tirs 82-86 :", recent["shots"])
    print("Tirs cadrés 82-86 :", recent["sot"])
    print("Corners 82-86 :", recent["corners"])
    print("Buts 2e mi-temps >87 :", post87_goals)
    print("Buts autres périodes :", other_period_goals)
    print("Événements pertinents à 87 :", at_87)
    print("Minutes inconnues :", unknown_minutes)
    print("Anomalies :", issues or "Aucune détectée")

print("\n" + "=" * 72)
print("BILAN — ÉTAPE 239")
print("=" * 72)

for name in (
    "saved",
    "quarantine",
    "pending",
    "no_structural_issue",
    "needs_review",
    "with_extra_periods",
):
    print(f"{name:26s}: {stats[name]}")

print("\nTimelines enregistrées par journée :")
for day, count in sorted(days.items()):
    print(day, ":", count)

print("\nIMPORTANT :")
print("Absence d'anomalie != certification des données.")
print("Aucun label créé.")
print("Aucun appel API.")
print("Aucun modèle entraîné.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 239 TERMINÉE.")
