#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("outputs/v4_research")
JOIN = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join"
)

OLD = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
NEW = ROOT / "V41_DIVERSE16_B29_RESEARCH_V1.csv"
SPLIT = JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"

DEST = ROOT / "V41_TRAIN294_B29_RESEARCH_V1.csv"

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

CORE = [
    "match_id",
    "kickoff_utc",
    "goal_after_87",
    "research_origin",
    "score_87_home",
    "score_87_away",
    "is_draw_87",
] + B6

# Ces champs sont uniquement des métadonnées
# explicites ; jamais des features prédictives.
OPTIONAL_METADATA = {
    "research_status",
    "training_certified",
    "live_certified",
    "label_used_for_selection",
}

CUTOFF = datetime(
    2026, 8, 25, tzinfo=timezone.utc
)

print("=" * 72)
print("ÉTAPE 245 — CONSTRUCTION TRAIN294 V4.1")
print("=" * 72)


def read_table(path):
    if not path.is_file():
        raise SystemExit(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        raise SystemExit(f"En-tête absent : {path}")

    if len(fields) != len(set(fields)):
        raise SystemExit(f"Colonnes dupliquées : {path}")

    return fields, rows


def parse_kickoff(value):
    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError("Date sans fuseau")

    return dt.astimezone(timezone.utc)


def binary(value):
    number = float(value)

    if number not in (0.0, 1.0):
        raise ValueError(f"Label invalide : {value}")

    return int(number)


if DEST.exists():
    raise SystemExit(
        f"Fichier existant : {DEST}\n"
        "Aucun remplacement effectué."
    )

old_fields, old_rows = read_table(OLD)
new_fields, new_rows = read_table(NEW)
_, split_rows = read_table(SPLIT)

print("\n=== EFFECTIFS ===")
print("TRAIN ancien :", len(old_rows))
print("Nouveau lot :", len(new_rows))
print("Plan SPLIT :", len(split_rows))

if (
    len(old_rows) != 278
    or len(new_rows) != 16
    or len(split_rows) != 320
):
    raise SystemExit("Effectifs inattendus")


print("\n=== COMPATIBILITÉ DES SCHÉMAS ===")

old_set = set(old_fields)
new_set = set(new_fields)

missing_core = set(CORE) - old_set
missing_core |= set(CORE) - new_set

if missing_core:
    raise SystemExit(
        "Colonnes essentielles manquantes : "
        + str(sorted(missing_core))
    )

missing_new = old_set - new_set
extra_new = new_set - old_set

print("Colonnes TRAIN278 :", len(old_fields))
print("Colonnes V4.1 :", len(new_fields))
print("Absentes du nouveau lot :", sorted(missing_new))
print("Supplémentaires dans V4.1 :", sorted(extra_new))

# Refus de perdre silencieusement des caractéristiques.
if missing_new - OPTIONAL_METADATA:
    raise SystemExit(
        "SCHÉMA INCOMPATIBLE : certaines colonnes "
        "du TRAIN278 manquent dans V4.1.\n"
        "Aucun fichier créé."
    )

if extra_new - OPTIONAL_METADATA:
    raise SystemExit(
        "SCHÉMA INCOMPATIBLE : nouvelles colonnes "
        "non reconnues.\n"
        "Aucun fichier créé."
    )


print("\n=== PROTECTION DES IDENTIFIANTS ===")

def ids(rows):
    return [r["match_id"] for r in rows]


old_ids = ids(old_rows)
new_ids = ids(new_rows)
split_ids = ids(split_rows)

for name, population in (
    ("TRAIN278", old_ids),
    ("V41_DIVERSE16", new_ids),
    ("SPLIT", split_ids),
):
    if len(population) != len(set(population)):
        raise SystemExit(
            f"Doublons détectés : {name}"
        )

if set(old_ids) & set(new_ids):
    raise SystemExit(
        "Chevauchement entre les deux entraînements"
    )

if set(new_ids) & set(split_ids):
    raise SystemExit(
        "Match V4.1 présent dans le plan réservé"
    )

print("Nouveaux IDs distincts :", len(new_ids))
print("Chevauchement TRAIN278 :", 0)
print("Chevauchement SPLIT :", 0)


print("\n=== VÉRIFICATION CHRONOLOGIQUE ===")

combined = []
positives = 0

for source, population in (
    ("OLD278", old_rows),
    ("NEW16", new_rows),
):
    for row in population:
        kickoff = parse_kickoff(
            row["kickoff_utc"]
        )

        if kickoff >= CUTOFF:
            raise SystemExit(
                "Match postérieur à la limite TRAIN : "
                + row["match_id"]
            )

        label = binary(
            row["goal_after_87"]
        )

        if source == "NEW16":
            if row["research_origin"] != "V41_DIVERSE16":
                raise SystemExit(
                    "Origine inattendue : "
                    + row["match_id"]
                )

        for col in B6 + [
            "score_87_home",
            "score_87_away",
        ]:
            value = float(row[col])

            if (
                value < 0
                or not value.is_integer()
            ):
                raise SystemExit(
                    f"Valeur invalide : "
                    f"{row['match_id']} / {col}"
                )

        normalized = dict(row)

        # Seules les métadonnées facultatives
        # peuvent recevoir une valeur par défaut.
        for col in missing_new:
            if col == "research_status":
                normalized[col] = "RESEARCH_ONLY"
            elif col in (
                "training_certified",
                "live_certified",
                "label_used_for_selection",
            ):
                normalized[col] = "0"

        normalized["_sort_kickoff"] = kickoff
        combined.append(normalized)
        positives += label

if sum(binary(r["goal_after_87"]) for r in old_rows) != 72:
    raise SystemExit("Positifs TRAIN278 inattendus")

if sum(binary(r["goal_after_87"]) for r in new_rows) != 3:
    raise SystemExit("Positifs V4.1 inattendus")

if len(combined) != 294 or positives != 75:
    raise SystemExit("Population combinée invalide")

combined.sort(
    key=lambda r: (
        r["_sort_kickoff"],
        r["match_id"],
    )
)

for row in combined:
    del row["_sort_kickoff"]

# Conserver exactement le schéma du TRAIN278.
for row in combined:
    if set(row) - set(old_fields) != extra_new:
        # Les éventuelles colonnes V4.1 supplémentaires
        # doivent exclusivement être des métadonnées.
        raise SystemExit("Schéma fusionné inattendu")

DEST.parent.mkdir(parents=True, exist_ok=True)

with DEST.open(
    "x",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=old_fields,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(combined)

print("\n=== BILAN ===")
print("Matchs combinés :", len(combined))
print("Positifs :", positives)
print("Négatifs :", len(combined) - positives)
print(
    "Taux de base :",
    f"{positives / len(combined):.2%}"
)
print(
    "Journées différentes :",
    len(set(
        r["kickoff_utc"][:10]
        for r in combined
    ))
)

print("\nFICHIER CRÉÉ :", DEST)
print("\nStatut : RESEARCH_ONLY")
print("Ancien TRAIN278 inchangé.")
print("Modèles figés inchangés.")
print("Aucun entraînement effectué.")
print("TEST_FUTUR non évalué.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 245 TERMINÉE.")
