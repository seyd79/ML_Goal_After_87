#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("outputs/v4_research")

OLD = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
NEW = ROOT / "V41_DIVERSE16_B29_RESEARCH_V1.csv"

SPLIT = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

DEST = ROOT / "V41_TRAIN294_B29_RESEARCH_V1.csv"

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

CORE = {
    "match_id",
    "kickoff_utc",
    "goal_after_87",
    "research_origin",
    "score_87_home",
    "score_87_away",
    "is_draw_87",
    *B6,
}

EXPECTED_OLD_ONLY = {
    "independent_verified",
}

EXPECTED_NEW_ONLY = {
    "extra_time_events_excluded",
    "extra_time_goals_excluded",
    "goal_at_87",
    "label_used_for_selection",
    "live_data_certified",
    "research_status",
    "score_independently_verified",
}

CUTOFF = datetime(
    2026, 8, 25, tzinfo=timezone.utc
)


def read_csv(path):
    if not path.is_file():
        raise ValueError(f"Fichier absent : {path}")

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        raise ValueError("En-tête absent")

    if len(fields) != len(set(fields)):
        raise ValueError("Colonnes dupliquées")

    return fields, rows


def parse_date(value):
    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError(
            "Date sans fuseau horaire"
        )

    return dt.astimezone(timezone.utc)


def integer_nonnegative(value):
    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Nombre non fini")

    if number < 0 or not number.is_integer():
        raise ValueError("Nombre entier positif attendu")

    return int(number)


def binary(value):
    number = integer_nonnegative(value)

    if number not in (0, 1):
        raise ValueError("Valeur binaire attendue")

    return number


def validate_rows(rows, name):
    ids = [r["match_id"] for r in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"IDs dupliqués dans {name}"
        )

    positives = 0

    for row in rows:
        mid = row["match_id"]

        if not mid:
            raise ValueError(
                f"ID manquant dans {name}"
            )

        kickoff = parse_date(
            row["kickoff_utc"]
        )

        if kickoff >= CUTOFF:
            raise ValueError(
                f"Date hors limite : {mid}"
            )

        for col in B6:
            integer_nonnegative(row[col])

        home = integer_nonnegative(
            row["score_87_home"]
        )

        away = integer_nonnegative(
            row["score_87_away"]
        )

        draw = binary(
            row["is_draw_87"]
        )

        if draw != int(home == away):
            raise ValueError(
                f"Score/nul incohérent : {mid}"
            )

        positives += binary(
            row["goal_after_87"]
        )

    return positives


def main():
    print("=" * 72)
    print("ÉTAPE 245 V2 — FUSION B29 SÉCURISÉE")
    print("=" * 72)

    if DEST.exists():
        raise SystemExit(
            "Sortie déjà existante : aucun écrasement"
        )

    old_fields, old_rows = read_csv(OLD)
    new_fields, new_rows = read_csv(NEW)
    _, split_rows = read_csv(SPLIT)

    print("\n=== POPULATIONS ===")
    print("Ancien TRAIN :", len(old_rows))
    print("Nouveau lot :", len(new_rows))
    print("Plan SPLIT :", len(split_rows))

    if (
        len(old_rows) != 278
        or len(new_rows) != 16
        or len(split_rows) != 320
    ):
        raise ValueError(
            "Effectifs historiques inattendus"
        )

    old_set = set(old_fields)
    new_set = set(new_fields)

    if not CORE.issubset(old_set & new_set):
        raise ValueError(
            "Colonnes essentielles manquantes"
        )

    old_only = old_set - new_set
    new_only = new_set - old_set

    print("\n=== DIFFÉRENCES DE SCHÉMAS ===")
    print("Ancien uniquement :", sorted(old_only))
    print("Nouveau uniquement :", sorted(new_only))

    if old_only != EXPECTED_OLD_ONLY:
        raise ValueError(
            "Différences inattendues côté TRAIN278"
        )

    if new_only != EXPECTED_NEW_ONLY:
        raise ValueError(
            "Différences inattendues côté V4.1"
        )

    # Une seule copie de chaque colonne.
    # Les colonnes absentes restent explicites.
    final_fields = old_fields + [
        col for col in new_fields
        if col not in old_set
    ]

    print(
        "Colonnes finales :",
        len(final_fields)
    )

    if len(final_fields) != 43:
        raise ValueError(
            "43 colonnes distinctes attendues"
        )

    old_ids = {
        r["match_id"] for r in old_rows
    }

    new_ids = {
        r["match_id"] for r in new_rows
    }

    split_ids = {
        r["match_id"] for r in split_rows
    }

    if len(split_ids) != 320:
        raise ValueError(
            "Doublons dans le plan SPLIT"
        )

    if old_ids & new_ids:
        raise ValueError(
            "Chevauchement TRAIN278 / V4.1"
        )

    if new_ids & split_ids:
        raise ValueError(
            "Chevauchement avec les partitions protégées"
        )

    old_positive = validate_rows(
        old_rows, "TRAIN278"
    )

    new_positive = validate_rows(
        new_rows, "V4.1"
    )

    if old_positive != 72:
        raise ValueError(
            "Positifs TRAIN278 inattendus"
        )

    if new_positive != 3:
        raise ValueError(
            "Positifs V4.1 inattendus"
        )

    for row in new_rows:
        if row["research_origin"] != "V41_DIVERSE16":
            raise ValueError(
                "Origine V4.1 inattendue"
            )

    combined = []

    for source, rows in (
        ("OLD278", old_rows),
        ("NEW16", new_rows),
    ):
        for row in rows:

            # Union des colonnes, sans écraser
            # les valeurs originales.
            output = {
                col: row.get(col, "")
                for col in final_fields
            }

            if source == "NEW16":
                # Non certifié indépendamment.
                output["independent_verified"] = "0"

            # Pour OLD278, les métadonnées
            # propres à V4.1 restent vides :
            # elles n'ont pas été mesurées.
            combined.append(output)

    combined.sort(
        key=lambda r: (
            parse_date(r["kickoff_utc"]),
            r["match_id"],
        )
    )

    if len(combined) != 294:
        raise ValueError(
            "Fusion incomplète"
        )

    total_positive = sum(
        binary(r["goal_after_87"])
        for r in combined
    )

    if total_positive != 75:
        raise ValueError(
            "Nombre de positifs incorrect"
        )

    # Aucun ajout de nouveau champ prédictif.
    # Métadonnées conservées séparément.
    with DEST.open(
        "x",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=final_fields
        )

        writer.writeheader()
        writer.writerows(combined)

    print("\n=== BILAN ===")
    print("Matchs combinés :", len(combined))
    print("Labels positifs :", total_positive)
    print(
        "Labels négatifs :",
        len(combined) - total_positive
    )
    print(
        "Prévalence :",
        f"{total_positive / len(combined):.2%}"
    )
    print("Colonnes :", len(final_fields))
    print(
        "Journées distinctes :",
        len({
            r["kickoff_utc"][:10]
            for r in combined
        })
    )

    print("\nFichier créé :", DEST)
    print("\nRESEARCH_ONLY")
    print("Métadonnées non confondues.")
    print("Aucun entraînement.")
    print("TRAIN278 inchangé.")
    print("TEST_FUTUR non évalué.")
    print("Scanner V3-C inchangé.")
    print("ÉTAPE 245 V2 TERMINÉE.")


if __name__ == "__main__":
    main()
