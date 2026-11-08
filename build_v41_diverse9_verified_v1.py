#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

from v3c_pressure_features import get_events, event_minute

ROOT = Path("outputs/v4_research")
RAW = ROOT / "v41_diverse" / "raw_timelines"
MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"

TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
SPLIT = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

DEST = ROOT / "V41_DIVERSE9_B29_RESEARCH_V1.csv"

GOALS = {"goal", "own_goal", "penalty_scored"}

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]


def read_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def numeric(value):
    return float(value)


print("=" * 72)
print("ÉTAPE 242 — CONSTRUCTION B29 V4.1")
print("=" * 72)

if DEST.exists():
    raise SystemExit(
        "Fichier déjà existant : aucun écrasement."
    )

manifest = read_csv(MANIFEST)
train = read_csv(TRAIN)
split = read_csv(SPLIT)

if (
    len(manifest) != 124
    or len(train) != 278
    or len(split) != 320
):
    raise SystemExit("Effectifs historiques incorrects")

if len({r["match_id"] for r in manifest}) != 124:
    raise SystemExit("Doublons dans le manifeste")

protected = {
    r["match_id"] for r in train
} | {
    r["match_id"] for r in split
}

lookup = {
    r["match_id"]: r
    for r in manifest
}

files = sorted(RAW.glob("mt_*.json"))

if len(files) != 9:
    raise SystemExit(
        f"9 fichiers attendus, trouvés : {len(files)}"
    )

spec = importlib.util.spec_from_file_location(
    "v41_builder",
    "build_historical_ab_research_v3.py"
)

if spec is None or spec.loader is None:
    raise SystemExit("Reconstructeur introuvable")

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

results = []
days = Counter()
positive = 0

expected_columns = None

for path in files:
    mid = path.stem

    if mid not in lookup:
        raise SystemExit(
            f"Match absent du manifeste : {mid}"
        )

    if mid in protected:
        raise SystemExit(
            f"Chevauchement historique : {mid}"
        )

    info = lookup[mid]

    record = json.loads(
        path.read_text(encoding="utf-8")
    )

    if record.get("match_id") != mid:
        raise SystemExit(f"ID local incohérent : {mid}")

    payload = record.get("timeline_response")

    if not isinstance(payload, dict):
        raise SystemExit(f"Payload invalide : {mid}")

    data = payload.get("data")

    if not isinstance(data, dict):
        raise SystemExit(f"Bloc data absent : {mid}")

    if data.get("match_id") != mid:
        raise SystemExit(f"ID API incompatible : {mid}")

    if data.get("coverage") != "full":
        raise SystemExit(
            f"Couverture insuffisante : {mid}"
        )

    events = get_events(payload)

    if not isinstance(events, list) or not events:
        raise SystemExit(
            f"Événements invalides : {mid}"
        )

    if not any(
        isinstance(e, dict)
        and e.get("period") == "second_half"
        and e.get("type") == "period_end"
        for e in events
    ):
        raise SystemExit(
            f"Fin de deuxième mi-temps absente : {mid}"
        )

    late_goals = 0

    for event in events:
        if not isinstance(event, dict):
            raise SystemExit(
                f"Événement invalide : {mid}"
            )

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        period = str(
            event.get("period") or ""
        ).strip().lower()

        if period != "second_half" or typ not in GOALS:
            continue

        minute = event_minute(event)

        if minute is None:
            raise SystemExit(
                f"But sans minute : {mid}"
            )

        if minute == 87:
            raise SystemExit(
                f"But ambigu à 87' : {mid}"
            )

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
        raise SystemExit(
            f"Reconstruction rejetée : {mid} / {quality}"
        )

    if not isinstance(b, dict):
        raise SystemExit(
            f"Format B29 inattendu : {mid}"
        )

    if not isinstance(quality, dict):
        raise SystemExit(
            f"Qualité invalide : {mid}"
        )

    if quality.get("status") != "RESEARCH_ONLY":
        raise SystemExit(
            f"Statut non conforme : {mid}"
        )

    required = B6 + [
        "score_87_home",
        "score_87_away",
        "goal_after_87",
    ]

    missing = [
        col for col in required
        if col not in b
    ]

    if missing:
        raise SystemExit(
            f"Colonnes B absentes pour {mid}: {missing}"
        )

    for col in B6 + [
        "score_87_home",
        "score_87_away",
    ]:
        value = numeric(b[col])

        if value < 0 or value != int(value):
            raise SystemExit(
                f"Valeur B29 incorrecte : {mid} / {col}"
            )

    label_value = numeric(b["goal_after_87"])

    if label_value not in (0.0, 1.0):
        raise SystemExit(
            f"Label B29 invalide : {mid}"
        )

    reconstructed_label = int(label_value)
    expected_label = int(late_goals > 0)

    if reconstructed_label != expected_label:
        raise SystemExit(
            f"LABEL_MISMATCH : {mid}, "
            f"builder={reconstructed_label}, "
            f"events={expected_label}"
        )

    # Garantir un schéma B29 homogène.
    columns = tuple(b.keys())

    if expected_columns is None:
        expected_columns = columns
    elif columns != expected_columns:
        raise SystemExit(
            f"Schéma B29 différent : {mid}"
        )

    output = dict(b)

    if "research_origin" in output:
        raise SystemExit(
            "Champ research_origin déjà présent : "
            "vérification du schéma nécessaire"
        )

    output["research_origin"] = "V41_DIVERSE9"
    output["research_status"] = "RESEARCH_ONLY"
    output["training_certified"] = 0
    output["live_certified"] = 0
    output["label_used_for_selection"] = 0

    results.append(output)

    day = info["kickoff_utc"][:10]
    days[day] += 1
    positive += reconstructed_label

    print(
        f"OK {mid} | {day} "
        f"| score87={b['score_87_home']}-"
        f"{b['score_87_away']} "
        f"| label={reconstructed_label} "
        f"| buts_apres87={late_goals}"
    )

if len(results) != 9:
    raise SystemExit("Neuf reconstructions attendues")

if not expected_columns:
    raise SystemExit("Schéma B29 absent")

fieldnames = list(results[0].keys())

with DEST.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 72)
print("BILAN — ÉTAPE 242")
print("=" * 72)
print("Matchs reconstruits :", len(results))
print("Labels positifs :", positive)
print("Labels négatifs :", len(results) - positive)
print("Journées :", len(days))
print("Colonnes enregistrées :", len(fieldnames))

for day, n in sorted(days.items()):
    print(day, ":", n)

print("\nFichier :", DEST)
print("\nStatut : RESEARCH_ONLY")
print("Labels cohérents avec les timelines disponibles.")
print("Aucune validation externe des labels.")
print("Aucun appel API.")
print("TRAIN278 inchangé.")
print("TEST_FUTUR non évalué.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 242 TERMINÉE.")
