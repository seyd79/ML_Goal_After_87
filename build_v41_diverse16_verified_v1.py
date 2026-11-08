#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

from v3c_pressure_features import get_events, event_minute

ROOT = Path("outputs/v4_research")
RAW = ROOT / "v41_diverse" / "raw_timelines"

MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"
PREVIOUS = ROOT / "V41_DIVERSE9_B29_RESEARCH_V1.csv"
TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"

SPLIT = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

DEST = ROOT / "V41_DIVERSE16_B29_RESEARCH_V1.csv"

GOAL_TYPES = {
    "goal", "own_goal", "penalty_scored"
}

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

def read_rows(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def comparable(a, b):
    if str(a) == str(b):
        return True
    try:
        x, y = float(a), float(b)
        return math.isfinite(x) and math.isfinite(y) and x == y
    except (TypeError, ValueError):
        return False

print("=" * 72)
print("ÉTAPE 244 — RECONSTRUCTION B29 V4.1 / 16 MATCHS")
print("=" * 72)

if DEST.exists():
    raise SystemExit(
        "Sortie déjà existante : aucun écrasement."
    )

manifest = read_rows(MANIFEST)
previous = read_rows(PREVIOUS)
train = read_rows(TRAIN)
split = read_rows(SPLIT)

if (
    len(manifest) != 124
    or len(previous) != 9
    or len(train) != 278
    or len(split) != 320
):
    raise SystemExit("Population historique inattendue")

for name, rows in [
    ("MANIFEST", manifest),
    ("PREVIOUS", previous),
    ("TRAIN", train),
    ("SPLIT", split),
]:
    ids = [r["match_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Doublons dans {name}")

lookup = {
    row["match_id"]: row
    for row in manifest
}

old_lookup = {
    row["match_id"]: row
    for row in previous
}

protected = {
    row["match_id"] for row in train
} | {
    row["match_id"] for row in split
}

files = sorted(RAW.glob("mt_*.json"))

print("Timelines trouvées :", len(files))

if len(files) != 16:
    raise SystemExit(
        "16 timelines attendues. "
        "Vérifier les fichiers avant de poursuivre."
    )

spec = importlib.util.spec_from_file_location(
    "v41_builder_244",
    "build_historical_ab_research_v3.py"
)

if spec is None or spec.loader is None:
    raise SystemExit("Builder introuvable")

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

results = []
days = Counter()
failures = []
parity_checked = 0
positives = 0

for path in files:
    mid = path.stem

    try:
        if mid not in lookup:
            raise ValueError("Absent du manifeste")

        if mid in protected:
            raise ValueError("Chevauchement protégé")

        info = lookup[mid]

        record = json.loads(
            path.read_text(encoding="utf-8")
        )

        if record.get("match_id") != mid:
            raise ValueError("ID local incorrect")

        payload = record.get("timeline_response")

        if not isinstance(payload, dict):
            raise ValueError("Payload invalide")

        data = payload.get("data")

        if not isinstance(data, dict):
            raise ValueError("Bloc data invalide")

        if data.get("match_id") != mid:
            raise ValueError("ID API incorrect")

        if data.get("coverage") != "full":
            raise ValueError("Couverture non full")

        events = get_events(payload)

        if not isinstance(events, list) or not events:
            raise ValueError("Timeline vide")

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
                raise ValueError("Événement non structuré")

            period = str(
                e.get("period") or ""
            ).strip().lower()

            typ = str(
                e.get("type") or ""
            ).strip().lower()

            if period != "second_half":
                continue

            if typ not in GOAL_TYPES:
                continue

            minute = event_minute(e)

            if minute is None:
                raise ValueError("But sans minute")

            if minute == 87:
                raise ValueError("But ambigu à 87'")

            if minute > 87:
                late_goals += 1

        builder_info = {
            k: "" if v is None else str(v)
            for k, v in info.items()
        }

        a, b, quality = builder.build_match(
            mid, builder_info, data
        )

        if a is None or b is None:
            raise ValueError(
                f"Reconstruction rejetée : {quality}"
            )

        if not isinstance(b, dict):
            raise ValueError("B29 non structuré")

        if quality.get("status") != "RESEARCH_ONLY":
            raise ValueError("Statut qualité incorrect")

        required = B6 + [
            "score_87_home",
            "score_87_away",
            "goal_after_87",
        ]

        missing = [c for c in required if c not in b]

        if missing:
            raise ValueError(f"Colonnes absentes : {missing}")

        for col in required:
            value = float(b[col])

            if not math.isfinite(value):
                raise ValueError(f"Valeur non finie : {col}")

            if value < 0 or value != int(value):
                raise ValueError(f"Valeur invalide : {col}")

        label = int(float(b["goal_after_87"]))

        if label not in (0, 1):
            raise ValueError("Label hors domaine")

        if label != int(late_goals > 0):
            raise ValueError("Label incohérent avec timeline")

        # Contrôle des 9 reconstructions précédentes.
        if mid in old_lookup:
            old = old_lookup[mid]

            for col in b:
                if col not in old:
                    raise ValueError(
                        f"Colonne absente du précédent : {col}"
                    )

                if not comparable(b[col], old[col]):
                    raise ValueError(
                        f"PARITY_MISMATCH : {col} "
                        f"ancien={old[col]} nouveau={b[col]}"
                    )

            parity_checked += 1

        output = dict(b)

        if "research_origin" in output:
            raise ValueError(
                "Champ research_origin déjà présent"
            )

        output.update({
            "research_origin": "V41_DIVERSE16",
            "research_status": "RESEARCH_ONLY",
            "training_certified": 0,
            "live_certified": 0,
            "label_used_for_selection": 0,
        })

        results.append(output)

        day = info["kickoff_utc"][:10]
        days[day] += 1
        positives += label

        print(
            f"OK {mid} | {day} "
            f"| score87={b['score_87_home']}-"
            f"{b['score_87_away']} "
            f"| label={label}"
        )

    except Exception as exc:
        failures.append((mid, str(exc)))
        print(f"REJET {mid} : {exc}")

print("\n=== BILAN ===")
print("Timelines examinées :", len(files))
print("Reconstructions acceptées :", len(results))
print("Rejets :", len(failures))
print("Parité avec ancien lot :", parity_checked, "/9")
print("Positifs :", positives)
print("Journées :", len(days))

for day, n in sorted(days.items()):
    print(day, ":", n)

if failures:
    print("\nÉCHEC DE VALIDATION : aucun CSV créé.")
    raise SystemExit(1)

if len(results) != 16 or parity_checked != 9:
    raise SystemExit(
        "Effectifs ou parité incorrects : aucun CSV créé."
    )

expected_keys = list(results[0])

if any(list(r) != expected_keys for r in results):
    raise SystemExit(
        "Schémas différents : aucun CSV créé."
    )

with DEST.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=expected_keys,
    )
    writer.writeheader()
    writer.writerows(results)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("Reconstruction RESEARCH_ONLY.")
print("Labels cohérents avec les timelines disponibles.")
print("Aucune certification indépendante.")
print("Aucun appel API.")
print("TRAIN278 inchangé.")
print("TEST_FUTUR non évalué.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 244 TERMINÉE.")
