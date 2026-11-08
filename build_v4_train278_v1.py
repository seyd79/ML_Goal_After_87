from pathlib import Path
import csv
import math
from collections import Counter

import pandas as pd

JOIN = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)
OUT = Path("outputs/v4_research")

DEST = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"

A23 = [
    "score_87_home", "score_87_away",
    "goal_diff_87", "is_draw_87",
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",
    "home_corners_87", "away_corners_87",
    "home_fouls_87", "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87", "away_red_cards_87",
    "home_offsides_87", "away_offsides_87",
    "goal_diff_signed_87",
]

B6 = [
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

FEATURES = A23 + B6

assert len(FEATURES) == 29

if DEST.exists():
    raise SystemExit(
        f"Sortie existante, remplacement interdit : {DEST}"
    )

print("=" * 72)
print("ÉTAPE 203 — FUSION CONTRÔLÉE DU TRAIN V4")
print("=" * 72)

# Lire uniquement les informations de partition.
with (
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
).open(encoding="utf-8-sig", newline="") as f:
    split_rows = list(csv.DictReader(f))

if len(split_rows) != 320:
    raise SystemExit("Plan ALL_320 inattendu")

train_ids = {
    r["match_id"]
    for r in split_rows
    if r["split"] == "TRAIN"
}

if len(train_ids) != 93:
    raise SystemExit("93 IDs TRAIN ALL_320 attendus")

rows = []
seen = set()
counts = Counter()
positives = Counter()

cutoff = pd.Timestamp("2026-08-25T00:00:00Z")


def add_source(path, source, allowed_ids=None, split=None):
    if not path.is_file():
        raise ValueError(f"Fichier absent : {path}")

    added = 0

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = set(
            ["match_id", "kickoff_utc", "goal_after_87"]
            + FEATURES
        )

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Colonnes manquantes dans {path.name}: "
                f"{sorted(required-set(reader.fieldnames or []))}"
            )

        for raw in reader:
            mid = raw["match_id"]

            # Filtrer AVANT de lire les labels et caractéristiques.
            if allowed_ids is not None and mid not in allowed_ids:
                continue

            if split is not None:
                if raw.get("research_split") != split:
                    continue

            if mid in seen:
                raise ValueError(f"Doublon entre sources : {mid}")

            kickoff = pd.to_datetime(
                raw["kickoff_utc"],
                utc=True,
                errors="raise",
            )

            if kickoff >= cutoff:
                raise ValueError(
                    f"Match hors période TRAIN : {mid}"
                )

            label_number = float(raw["goal_after_87"])

            if label_number not in (0.0, 1.0):
                raise ValueError(f"Label invalide : {mid}")

            label = int(label_number)

            values = {}

            for col in FEATURES:
                value = float(raw[col])

                if not math.isfinite(value):
                    raise ValueError(
                        f"Variable non finie : {mid} / {col}"
                    )

                values[col] = value

            for side in ("home", "away"):
                if (
                    values[f"sot_{side}_last5"]
                    > values[f"shots_{side}_last5"]
                ):
                    raise ValueError(
                        f"Tirs cadrés > tirs : {mid}"
                    )

            rows.append({
                "match_id": mid,
                "kickoff_utc": raw["kickoff_utc"],
                "goal_after_87": label,
                "research_origin": source,
                **values,
                "training_certified": 0,
                "live_certified": 0,
                "independent_verified": 0,
            })

            seen.add(mid)
            counts[source] += 1
            positives[source] += label
            added += 1

    print(
        f"{source:15s} | matchs={added:3d} "
        f"| positifs={positives[source]}"
    )


print("\n=== CHARGEMENT DES SOURCES ===")

add_source(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    "ALL_320",
    allowed_ids=train_ids,
)

for n in range(2, 12):
    name = f"BATCH{n:02d}"

    add_source(
        JOIN / f"{name}_B_RESEARCH_V1.csv",
        name,
        split="TRAIN",
    )

add_source(
    OUT / "V4_CACHED12_B29_RESEARCH_V1.csv",
    "V4_CACHED12",
)

add_source(
    OUT / "V4_FETCH83_B29_RESEARCH_V1.csv",
    "V4_FETCH83",
)

print("\n=== CONTRÔLES ===")

historical = (
    counts["ALL_320"]
    + sum(counts[f"BATCH{n:02d}"] for n in range(2, 12))
)

historical_pos = (
    positives["ALL_320"]
    + sum(
        positives[f"BATCH{n:02d}"]
        for n in range(2, 12)
    )
)

print("TRAIN historique :", historical)
print("Positifs historiques :", historical_pos)
print("CACHED12 :", counts["V4_CACHED12"])
print("FETCH83 :", counts["V4_FETCH83"])

total_pos = sum(r["goal_after_87"] for r in rows)

print("Total matchs :", len(rows))
print("Total positifs :", total_pos)
print("Total négatifs :", len(rows)-total_pos)

if not (
    historical == 183
    and historical_pos == 49
    and counts["V4_CACHED12"] == 12
    and positives["V4_CACHED12"] == 4
    and counts["V4_FETCH83"] == 83
    and positives["V4_FETCH83"] == 19
    and len(rows) == 278
    and len(seen) == 278
    and total_pos == 72
):
    raise SystemExit(
        "Effectifs ou labels inattendus. Fusion refusée."
    )

rows.sort(
    key=lambda r: (r["kickoff_utc"], r["match_id"])
)

print("\n=== RÉPARTITION CHRONOLOGIQUE ===")

dates = pd.to_datetime(
    [r["kickoff_utc"] for r in rows],
    utc=True,
)

days = pd.Series(dates).dt.strftime("%Y-%m-%d")
print(days.value_counts().sort_index().to_string())

print("\n=== ENREGISTREMENT ===")

OUT.mkdir(parents=True, exist_ok=True)

with DEST.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("Fichier créé :", DEST)
print("Matchs :", len(rows))
print("Positifs :", total_pos)
print("Taux positif :", f"{total_pos/len(rows):.2%}")
print("Caractéristiques :", len(FEATURES))

print("\n=== GARANTIES ===")
print("TRAIN uniquement.")
print("Statut RESEARCH_ONLY.")
print("Aucune donnée TEST_FUTUR utilisée.")
print("VALIDATION non utilisée.")
print("Aucun modèle entraîné.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 203 TERMINÉE.")
