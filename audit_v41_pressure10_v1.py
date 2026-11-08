#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 220 — AUDIT V4.1 PRESSION 77'-87'

Compatible Python 3.9.
Lecture des seules données TRAIN.
Aucun appel API.
Aucun entraînement.
Aucun label utilisé dans les calculs.
"""

from pathlib import Path
from collections import Counter
import csv
import json

from build_historical_ab_research_v3 import SHOTS, SOT
from v3c_pressure_features import (
    get_events,
    event_minute,
)

ROOT = Path("outputs/pressure_historical")
CACHE = ROOT / "timeline_cache"

OUT = Path("outputs/v4_research")

SOURCE = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
DEST = OUT / "V41_PRESSURE10_AUDIT_V1.csv"

if DEST.exists():
    raise SystemExit(
        "Le fichier d'audit existe déjà : "
        f"{DEST}\nAucun remplacement effectué."
    )

if not SOURCE.is_file():
    raise SystemExit("TRAIN278 introuvable")

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

FIELDS = [
    "match_id",
    "kickoff_utc",
    "research_origin",
    "status",
    "events_total",
    "shots_77_81",
    "sot_77_81",
    "corners_77_81",
    "shots_82_86",
    "sot_82_86",
    "corners_82_86",
    "shots_acceleration",
    "sot_acceleration",
    "corners_acceleration",
    "shots_total_10min",
    "sot_total_10min",
    "corners_total_10min",
    "unknown_minutes",
]

print("=" * 72)
print("ÉTAPE 220 — AUDIT HISTORIQUE V4.1")
print("=" * 72)


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def totals_from_events(payload):
    events = get_events(payload)

    if events is None:
        raise ValueError("TIMELINE_NOT_RECOGNIZED")

    old = Counter()
    recent = Counter()
    unknown_minutes = 0

    for event in events:
        if not isinstance(event, dict):
            continue

        period = str(
            event.get("period") or ""
        ).strip().lower()

        if period != "second_half":
            continue

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        relevant = (
            typ in SHOTS
            or typ in SOT
            or typ == "corner_kick"
        )

        if not relevant:
            continue

        minute = event_minute(event)

        if minute is None:
            unknown_minutes += 1
            continue

        if 77 <= minute < 82:
            counts = old
        elif 82 <= minute < 87:
            counts = recent
        else:
            continue

        if typ in SHOTS:
            counts["shots"] += 1

        if typ in SOT:
            counts["sot"] += 1

        if typ == "corner_kick":
            counts["corners"] += 1

    if unknown_minutes:
        raise ValueError(
            f"RELEVANT_EVENTS_WITHOUT_MINUTE:{unknown_minutes}"
        )

    return old, recent, len(events)


rows = read_csv(SOURCE)

if len(rows) != 278:
    raise SystemExit(
        f"278 matchs attendus, obtenus : {len(rows)}"
    )

ids = [row["match_id"] for row in rows]

if len(set(ids)) != 278:
    raise SystemExit("Doublons dans TRAIN278")

results = []
statuses = Counter()
by_origin = Counter()
valid_by_origin = Counter()
examples = []

for row in rows:
    mid = row["match_id"]
    origin = row["research_origin"]

    by_origin[origin] += 1

    result = {
        name: "" for name in FIELDS
    }

    result.update({
        "match_id": mid,
        "kickoff_utc": row["kickoff_utc"],
        "research_origin": origin,
    })

    path = CACHE / f"{mid}.json"

    if not path.is_file():
        result["status"] = "MISSING_CACHE"

    else:
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            early, recent, event_count = (
                totals_from_events(payload)
            )

            # Contrôle indépendant des totaux B6
            # déjà enregistrés dans TRAIN278.
            for name, home, away in [
                (
                    "shots",
                    "shots_home_last5",
                    "shots_away_last5",
                ),
                (
                    "sot",
                    "sot_home_last5",
                    "sot_away_last5",
                ),
                (
                    "corners",
                    "corners_home_last5",
                    "corners_away_last5",
                ),
            ]:
                expected = (
                    int(float(row[home]))
                    + int(float(row[away]))
                )

                if recent[name] != expected:
                    raise ValueError(
                        f"B6_MISMATCH_{name}:"
                        f"expected={expected},"
                        f"actual={recent[name]}"
                    )

            result["events_total"] = event_count
            result["unknown_minutes"] = 0

            for name in (
                "shots", "sot", "corners"
            ):
                previous = early[name]
                current = recent[name]

                result[f"{name}_77_81"] = previous
                result[f"{name}_82_86"] = current

                result[
                    f"{name}_acceleration"
                ] = current - previous

                result[
                    f"{name}_total_10min"
                ] = current + previous

            result["status"] = (
                "RECONSTRUCTED_RESEARCH_ONLY"
            )

            valid_by_origin[origin] += 1

        except Exception as exc:
            result["status"] = (
                "ERROR:" + str(exc)[:110]
            )

    statuses[result["status"]] += 1
    results.append(result)

    if (
        result["status"]
        == "RECONSTRUCTED_RESEARCH_ONLY"
        and len(examples) < 10
    ):
        examples.append(result)


print("\n=== DISPONIBILITÉ DES DONNÉES ===")

print("Matchs TRAIN :", len(rows))

for origin in sorted(by_origin):
    print(
        f"{origin:17s} | "
        f"disponibles={valid_by_origin[origin]:3d}/"
        f"{by_origin[origin]:3d}"
    )

print("\n=== STATUTS ===")

for name, count in statuses.most_common():
    print(f"{name:55s} : {count}")

valid = statuses["RECONSTRUCTED_RESEARCH_ONLY"]

print("\nMatchs reconstruits :", valid)
print("Matchs manquants ou invalides :", 278-valid)

print("\n=== EXEMPLES DE NOUVELLES FEATURES ===")

for r in examples:
    print(
        f"{r['match_id']} | "
        f"tirs 77-81={r['shots_77_81']} | "
        f"tirs 82-86={r['shots_82_86']} | "
        f"accélération={r['shots_acceleration']} | "
        f"SOT 10min={r['sot_total_10min']} | "
        f"corners 10min={r['corners_total_10min']}"
    )

OUT.mkdir(parents=True, exist_ok=True)

with DEST.open(
    "x",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=FIELDS,
    )
    writer.writeheader()
    writer.writerows(results)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== CONCLUSION ===")

if valid == 278:
    print("278/278 matchs reconstruits.")
    print("Prêt pour un audit approfondi de qualité.")
else:
    print(
        "Données incomplètes pour les 278 matchs."
    )
    print(
        "Ne pas entraîner V4.1 sur le sous-ensemble "
        "disponible sans étudier les exclusions."
    )

print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucune donnée VALIDATION consultée.")
print("Aucune donnée TEST_FUTUR consultée.")
print("Aucun label exploité.")
print("Aucun modèle entraîné.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 220 TERMINÉE.")
