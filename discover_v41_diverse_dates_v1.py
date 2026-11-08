#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import importlib.util
import re
import time

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("outputs/v4_research")
JOIN = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join"
)

DEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"

DATES = [
    "2026-07-03",
    "2026-07-10",
    "2026-07-17",
    "2026-07-24",
    "2026-07-31",
    "2026-08-07",
]

MAX_PER_DATE = 25

FIELDS = [
    "match_id",
    "home_team",
    "away_team",
    "home_id",
    "away_id",
    "kickoff_utc",
    "research_status",
    "label_used_for_selection",
]

print("=" * 72)
print("ÉTAPE 233 — DÉCOUVERTE V4.1 MULTI-JOURNÉES")
print("=" * 72)

if DEST.exists():
    raise SystemExit(
        "Le manifeste existe déjà. "
        "Aucun écrasement autorisé."
    )


def read_rows(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


split_path = JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
train_path = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"

split = read_rows(split_path)
train = read_rows(train_path)

if len(split) != 320 or len(train) != 278:
    raise SystemExit("Population historique inattendue")

test_ids = {
    r["match_id"] for r in split
    if r["split"] == "TEST_FUTUR"
}

if len(test_ids) != 183:
    raise SystemExit("TEST_FUTUR inattendu")

known = {
    r["match_id"] for r in split
} | {
    r["match_id"] for r in train
}

# Inventaire des IDs utilisés dans les CSV de recherche.
# Les labels ne servent jamais à la sélection.
for folder in (JOIN, ROOT):
    for path in folder.rglob("*.csv"):
        if path == DEST:
            continue

        with path.open(
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                continue

            if "match_id" not in reader.fieldnames:
                continue

            for row in reader:
                mid = (row.get("match_id") or "").strip()

                if mid:
                    known.add(mid)

print("\nIDs historiques protégés :", len(known))
print("TEST_FUTUR réservés :", len(test_ids))


spec = importlib.util.spec_from_file_location(
    "v41_api",
    "diagnostic_live_api.py"
)

if spec is None or spec.loader is None:
    raise SystemExit("Client API introuvable")

api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def extract(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "matches", "results", "items", "data"
        ):
            if key in payload:
                result = extract(payload[key])

                if result is not None:
                    return result

    return None


def team(value):
    if isinstance(value, dict):
        return (
            str(value.get("name") or ""),
            str(value.get("id") or ""),
        )

    if isinstance(value, str):
        return value, ""

    return "", ""


def parse_date(value):
    if not value:
        raise ValueError("Date absente")

    dt = datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError("Date sans fuseau")

    return dt.astimezone(timezone.utc)


candidates = []
seen = set()
report = []

print("\n=== DÉCOUVERTE ===")
print("Budget maximal :", len(DATES), "appels API")

for i, day in enumerate(DATES, 1):

    if i > 1:
        time.sleep(2)

    print(f"\nJOURNÉE {i}/{len(DATES)} : {day}")

    payload = api.api_get(
        "/football/matches",
        {
            "status": "finished",
            "date_from": day,
            "date_to": day,
            "page": 1,
            "per_page": 100,
        },
    )

    matches = extract(payload)

    if matches is None:
        raise SystemExit(
            f"Structure de réponse invalide : {day}"
        )

    available = []

    for match in matches:
        if not isinstance(match, dict):
            raise SystemExit(
                f"Enregistrement invalide : {day}"
            )

        mid = str(
            match.get("id")
            or match.get("match_id")
            or ""
        )

        if not re.fullmatch(r"mt_[0-9]+", mid):
            raise SystemExit(
                f"ID incorrect : {mid}"
            )

        raw_date = (
            match.get("utc_date")
            or match.get("kickoff_utc")
            or match.get("start_time")
        )

        kickoff = parse_date(raw_date)

        if kickoff.strftime("%Y-%m-%d") != day:
            raise SystemExit(
                f"Date hors journée : {mid}, {kickoff}"
            )

        if match.get("status") != "finished":
            raise SystemExit(
                f"Statut non terminé : {mid}"
            )

        if mid in seen:
            raise SystemExit(
                f"ID répété dans la découverte : {mid}"
            )

        seen.add(mid)

        if mid in known:
            continue

        home, home_id = team(
            match.get("home_team")
        )
        away, away_id = team(
            match.get("away_team")
        )

        if not home or not away:
            raise SystemExit(
                f"Équipes manquantes : {mid}"
            )

        available.append({
            "match_id": mid,
            "home_team": home,
            "away_team": away,
            "home_id": home_id,
            "away_id": away_id,
            "kickoff_utc": kickoff.isoformat(),
            "research_status": "DISCOVERED_ONLY",
            "label_used_for_selection": 0,
        })

    # Ordre déterministe indépendant du score final.
    # Sélection limitée aux résultats de la première page.
    available.sort(
        key=lambda row: hashlib.sha256(
            ("V41_DATE_V1|" + row["match_id"]).encode()
        ).hexdigest()
    )

    selected = available[:MAX_PER_DATE]
    candidates.extend(selected)

    report.append({
        "date": day,
        "returned": len(matches),
        "eligible": len(available),
        "selected": len(selected),
    })

    print("Matchs retournés :", len(matches))
    print("Nouveaux admissibles :", len(available))
    print("Candidats sélectionnés :", len(selected))

    if len(matches) == 100:
        print(
            "ATTENTION : la page est pleine ; "
            "la journée n'est pas inventoriée "
            "exhaustivement."
        )

ids = [r["match_id"] for r in candidates]

if len(ids) != len(set(ids)):
    raise SystemExit("Doublons dans le résultat")

if set(ids) & known:
    raise SystemExit("Chevauchement historique")

if not candidates:
    raise SystemExit(
        "Aucun candidat : aucun manifeste créé."
    )

ROOT.mkdir(parents=True, exist_ok=True)

with DEST.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=FIELDS
    )
    writer.writeheader()
    writer.writerows(candidates)

print("\n=== BILAN PAR JOURNÉE ===")

for r in report:
    print(
        r["date"],
        "| retournés :", r["returned"],
        "| admissibles :", r["eligible"],
        "| retenus :", r["selected"],
    )

print("\nTOTAL CANDIDATS :", len(candidates))
print("Journées représentées :", sum(
    r["selected"] > 0 for r in report
))
print("Manifeste :", DEST)

print("\n=== GARANTIES ===")
print("Aucun label utilisé pour la sélection.")
print("Aucune timeline téléchargée.")
print("Aucune nouvelle donnée d'entraînement.")
print("Aucun modèle modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 233 TERMINÉE.")
