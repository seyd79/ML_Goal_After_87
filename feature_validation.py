#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_validation.py

Validation complète de la cohérence entre :
1. Le dataset ML
2. Les features attendues par le modèle de production
3. Les features PRE-MATCH
4. Les features LIVE

Modèle :
    outputs/production/ET_500_d15_73_27.joblib

Features :
    outputs/production/ET_500_d15_73_27_features.json

Dataset :
    ML_Dataset_PreMatch_Live87.csv
"""

import os
import json
import sys

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "ML_Dataset_PreMatch_Live87.csv"
)

FEATURES_JSON_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "production",
    "ET_500_d15_73_27_features.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "production",
    "ET_500_d15_73_27.joblib"
)

TARGET_COLUMN = "goal_after_87"

ID_COLUMN = "match_id"


# ============================================================
# 59 FEATURES PRE-MATCH
# ============================================================

PRE_MATCH_FEATURES = [
    "home_team_rank",
    "away_team_rank",
    "home_team_points_standing",
    "away_team_points_standing",
    "home_team_matches_played",
    "away_team_matches_played",
    "home_team_wins",
    "away_team_wins",
    "home_team_draws",
    "away_team_draws",
    "home_team_losses",
    "away_team_losses",
    "home_team_goals_for",
    "away_team_goals_for",
    "home_team_goals_against",
    "away_team_goals_against",
    "home_team_goal_difference",
    "away_team_goal_difference",
    "home_position",
    "away_position",
    "home_points",
    "away_points",
    "home_form5",
    "away_form5",
    "home_form10",
    "away_form10",
    "home_scored_freq",
    "away_scored_freq",
    "home_wins_freq",
    "away_wins_freq",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",
    "home_avg_xg",
    "away_avg_xg",
    "home_btts_freq",
    "away_btts_freq",
    "home_over25_freq",
    "away_over25_freq",
    "h2h5_over25_freq",
    "h2h5_btts_freq",
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "league_2nd_half_over15_freq",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
]


# ============================================================
# 37 FEATURES LIVE
# ============================================================

LIVE_FEATURES = [
    "score_87_home",
    "score_87_away",
    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",
    "home_corners_87",
    "away_corners_87",
    "home_fouls_87",
    "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87",
    "away_red_cards_87",
    "home_offsides_87",
    "away_offsides_87",
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",
    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",
    "minutes_since_last_goal",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "minutes_since_last_corner",
]


# ============================================================
# 96 FEATURES TOTALES
# ============================================================

EXPECTED_FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES


# ============================================================
# UTILITAIRES
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0
WARNING_COUNT = 0


def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_subheader(title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


def print_status(label, condition):
    global PASS_COUNT, FAIL_COUNT

    if condition:
        print(f"✅ {label}")
        PASS_COUNT += 1
    else:
        print(f"❌ {label}")
        FAIL_COUNT += 1


def print_warning(label):
    global WARNING_COUNT

    print(f"⚠️  {label}")
    WARNING_COUNT += 1


def normalize_feature_list(obj):
    """
    Essaie de récupérer une liste de features depuis plusieurs formats JSON.
    """

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):

        possible_keys = [
            "features",
            "feature_names",
            "feature_columns",
            "columns",
            "model_features",
        ]

        for key in possible_keys:
            value = obj.get(key)

            if isinstance(value, list):
                return value

    return None


def load_features_json(path):
    if not os.path.exists(path):
        print(f"❌ Fichier features introuvable : {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = normalize_feature_list(data)

        if features is None:
            print("❌ Impossible de trouver la liste des features dans le JSON.")
            print(f"Structure JSON détectée : {type(data).__name__}")
            return None

        return features

    except Exception as e:
        print(f"❌ Erreur lecture JSON : {e}")
        return None


def duplicates(items):
    seen = set()
    duplicated = []

    for item in items:
        if item in seen and item not in duplicated:
            duplicated.append(item)

        seen.add(item)

    return duplicated


# ============================================================
# VALIDATION FICHIERS
# ============================================================

def validate_files():

    print_header("1. VALIDATION DES FICHIERS")

    print(f"Dataset :")
    print(DATASET_PATH)

    print()
    print("Features JSON :")
    print(FEATURES_JSON_PATH)

    print()
    print("Modèle :")
    print(MODEL_PATH)

    print()

    print_status(
        "Dataset présent",
        os.path.exists(DATASET_PATH)
    )

    print_status(
        "Features JSON présent",
        os.path.exists(FEATURES_JSON_PATH)
    )

    print_status(
        "Modèle de production présent",
        os.path.exists(MODEL_PATH)
    )


# ============================================================
# VALIDATION STRUCTURE FEATURES
# ============================================================

def validate_feature_definitions():

    print_header("2. VALIDATION DES FEATURES")

    print(f"Features PRE-MATCH attendues : {len(PRE_MATCH_FEATURES)}")
    print(f"Features LIVE attendues      : {len(LIVE_FEATURES)}")
    print(f"TOTAL attendu                : {len(EXPECTED_FEATURES)}")

    print_status(
        "59 features PRE-MATCH",
        len(PRE_MATCH_FEATURES) == 59
    )

    print_status(
        "37 features LIVE",
        len(LIVE_FEATURES) == 37
    )

    print_status(
        "96 features totales",
        len(EXPECTED_FEATURES) == 96
    )

    duplicate_expected = duplicates(EXPECTED_FEATURES)

    if duplicate_expected:
        print_status(
            f"Aucun doublon dans les 96 features — doublons : {duplicate_expected}",
            False
        )
    else:
        print_status(
            "Aucun doublon dans les 96 features",
            True
        )

    print_subheader("Liste des 59 features PRE-MATCH")

    for i, feature in enumerate(PRE_MATCH_FEATURES, 1):
        print(f"{i:02d}. {feature}")

    print_subheader("Liste des 37 features LIVE")

    for i, feature in enumerate(LIVE_FEATURES, 1):
        print(f"{i:02d}. {feature}")


# ============================================================
# VALIDATION MODEL JSON
# ============================================================

def validate_model_features():

    print_header("3. VALIDATION DU JSON DU MODÈLE")

    model_features = load_features_json(FEATURES_JSON_PATH)

    if model_features is None:
        return None

    print(f"Features présentes dans le JSON : {len(model_features)}")
    print()

    print_status(
        "Le JSON contient exactement 96 features",
        len(model_features) == 96
    )

    duplicate_model = duplicates(model_features)

    if duplicate_model:
        print_status(
            f"Aucun doublon dans le JSON — doublons : {duplicate_model}",
            False
        )
    else:
        print_status(
            "Aucun doublon dans le JSON",
            True
        )

    expected_set = set(EXPECTED_FEATURES)
    model_set = set(model_features)

    missing_from_model = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in model_set
    ]

    extra_in_model = [
        feature
        for feature in model_features
        if feature not in expected_set
    ]

    print_subheader("Comparaison des noms")

    if missing_from_model:
        print("❌ Features attendues mais absentes du JSON :")

        for feature in missing_from_model:
            print(f"   - {feature}")

    else:
        print("✅ Aucune feature attendue ne manque du JSON.")

    if extra_in_model:
        print()
        print("❌ Features présentes dans le JSON mais non attendues :")

        for feature in extra_in_model:
            print(f"   - {feature}")

    else:
        print("✅ Aucune feature supplémentaire dans le JSON.")

    print_status(
        "Même ensemble de 96 features",
        set(model_features) == set(EXPECTED_FEATURES)
    )

    print_status(
        "Ordre exact des 96 features identique",
        model_features == EXPECTED_FEATURES
    )

    if model_features != EXPECTED_FEATURES:

        print_subheader("DIFFÉRENCES D'ORDRE")

        max_len = max(
            len(model_features),
            len(EXPECTED_FEATURES)
        )

        differences = 0

        for i in range(max_len):

            expected = (
                EXPECTED_FEATURES[i]
                if i < len(EXPECTED_FEATURES)
                else "<absent>"
            )

            actual = (
                model_features[i]
                if i < len(model_features)
                else "<absent>"
            )

            if expected != actual:

                print(
                    f"Position {i + 1}: "
                    f"attendu='{expected}' | "
                    f"JSON='{actual}'"
                )

                differences += 1

                if differences >= 20:
                    print("... différences supplémentaires non affichées.")
                    break

    return model_features


# ============================================================
# CHARGEMENT DATASET
# ============================================================

def load_dataset():

    print_header("4. CHARGEMENT DU DATASET")

    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset introuvable.")
        return None

    try:

        df = pd.read_csv(DATASET_PATH)

    except Exception as e:

        print(f"❌ Erreur lecture dataset : {e}")
        return None

    print(f"Nombre de lignes : {len(df):,}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    print()

    print("Colonnes principales :")

    for i, column in enumerate(df.columns, 1):
        print(f"{i:03d}. {column}")

    return df


# ============================================================
# VALIDATION STRUCTURE DATASET
# ============================================================

def validate_dataset_structure(df):

    print_header("5. VALIDATION STRUCTURE DATASET")

    if ID_COLUMN not in df.columns:
        print_status(
            f"Colonne '{ID_COLUMN}' présente",
            False
        )
    else:
        print_status(
            f"Colonne '{ID_COLUMN}' présente",
            True
        )

    if TARGET_COLUMN not in df.columns:
        print_status(
            f"Colonne target '{TARGET_COLUMN}' présente",
            False
        )

        return

    print_status(
        f"Colonne target '{TARGET_COLUMN}' présente",
        True
    )

    dataset_feature_columns = [
        column
        for column in df.columns
        if column != ID_COLUMN
        and column != TARGET_COLUMN
    ]

    print()
    print(f"Features détectées dans le dataset : {len(dataset_feature_columns)}")
    print(f"Features attendues                  : {len(EXPECTED_FEATURES)}")

    print_status(
        "Le dataset contient exactement 96 features",
        len(dataset_feature_columns) == 96
    )

    expected_set = set(EXPECTED_FEATURES)
    dataset_set = set(dataset_feature_columns)

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in dataset_set
    ]

    extra = [
        feature
        for feature in dataset_feature_columns
        if feature not in expected_set
    ]

    print_subheader("Features manquantes du dataset")

    if missing:

        for feature in missing:
            print(f"❌ {feature}")

    else:

        print("✅ Aucune feature manquante.")

    print_subheader("Features supplémentaires du dataset")

    if extra:

        for feature in extra:
            print(f"❌ {feature}")

    else:

        print("✅ Aucune feature supplémentaire.")

    print_status(
        "Même ensemble de features dataset / modèle",
        dataset_set == expected_set
    )

    print_status(
        "Ordre dataset identique aux features attendues",
        dataset_feature_columns == EXPECTED_FEATURES
    )


# ============================================================
# VALIDATION TARGET
# ============================================================

def validate_target(df):

    print_header("6. VALIDATION DE LA TARGET")

    if TARGET_COLUMN not in df.columns:

        print_status(
            f"Target '{TARGET_COLUMN}' disponible",
            False
        )

        return

    target = df[TARGET_COLUMN]

    print(f"Target : {TARGET_COLUMN}")
    print(f"Type : {target.dtype}")
    print()

    print("Distribution :")
    print(target.value_counts(dropna=False).sort_index())

    print()

    unique_values = set(
        target.dropna().unique()
    )

    valid_binary = unique_values.issubset({0, 1})

    print_status(
        "Target binaire 0/1",
        valid_binary
    )

    missing_target = int(target.isna().sum())

    print_status(
        "Aucune valeur target manquante",
        missing_target == 0
    )

    if len(target) > 0:

        positive_rate = float(target.mean()) * 100

        print()
        print(
            f"Taux target=1 : {positive_rate:.2f}%"
        )

        print(
            f"Taux target=0 : {100 - positive_rate:.2f}%"
        )


# ============================================================
# VALIDATION NOMS / DOUBLONS
# ============================================================

def validate_dataset_columns(df):

    print_header("7. VALIDATION DES COLONNES")

    duplicates_columns = duplicates(list(df.columns))

    if duplicates_columns:

        print_status(
            f"Aucun doublon de colonne — doublons : {duplicates_columns}",
            False
        )

    else:

        print_status(
            "Aucun doublon de colonne",
            True
        )

    expected_columns = (
        [ID_COLUMN]
        + EXPECTED_FEATURES
        + [TARGET_COLUMN]
    )

    print()

    print(f"Colonnes attendues au total : {len(expected_columns)}")
    print(f"Colonnes réelles              : {len(df.columns)}")

    print_status(
        "Dataset contient 98 colonnes",
        len(df.columns) == 98
    )

    expected_set = set(expected_columns)
    actual_set = set(df.columns)

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:

        print()
        print("❌ Colonnes manquantes :")

        for column in sorted(missing):
            print(f"   - {column}")

    if extra:

        print()
        print("❌ Colonnes supplémentaires :")

        for column in sorted(extra):
            print(f"   - {column}")

    print_status(
        "Structure complète 98 colonnes conforme",
        expected_set == actual_set
    )


# ============================================================
# VALIDATION VALEURS MANQUANTES
# ============================================================

def validate_missing_values(df):

    print_header("8. VALIDATION DES VALEURS MANQUANTES")

    available_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature in df.columns
    ]

    missing_counts = df[available_features].isna().sum()

    total_missing = int(missing_counts.sum())

    print(
        f"Nombre total de valeurs manquantes dans les 96 features : "
        f"{total_missing}"
    )

    print_status(
        "Aucune valeur manquante dans les 96 features",
        total_missing == 0
    )

    if total_missing > 0:

        print()
        print("Features avec valeurs manquantes :")

        for feature, count in missing_counts.items():

            if count > 0:

                percentage = (
                    count / len(df) * 100
                    if len(df) > 0
                    else 0
                )

                print(
                    f"   {feature}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )


# ============================================================
# VALIDATION TYPES NUMÉRIQUES
# ============================================================

def validate_numeric_types(df):

    print_header("9. VALIDATION DES TYPES")

    available_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature in df.columns
    ]

    non_numeric = []

    for feature in available_features:

        if not pd.api.types.is_numeric_dtype(
            df[feature]
        ):

            non_numeric.append(
                (
                    feature,
                    str(df[feature].dtype)
                )
            )

    if non_numeric:

        print("❌ Features non numériques :")

        for feature, dtype in non_numeric:

            print(
                f"   - {feature}: {dtype}"
            )

    else:

        print("✅ Toutes les 96 features sont numériques.")

    print_status(
        "Toutes les features sont numériques",
        len(non_numeric) == 0
    )


# ============================================================
# VALIDATION LEADING TEAM
# ============================================================

def validate_leading_team(df):

    print_header("10. VALIDATION leading_team_87")

    column = "leading_team_87"

    if column not in df.columns:

        print_status(
            "leading_team_87 présent",
            False
        )

        return

    series = df[column]

    print(f"Type : {series.dtype}")
    print()

    print("Valeurs observées :")

    counts = series.value_counts(
        dropna=False
    ).sort_index()

    for value, count in counts.items():

        percentage = (
            count / len(series) * 100
            if len(series) > 0
            else 0
        )

        print(
            f"   {value}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()

    unique_values = set(
        series.dropna().unique()
    )

    print(
        f"Valeurs uniques : {sorted(unique_values)}"
    )

    valid_values = unique_values.issubset(
        {0, 1, 2}
    )

    print_status(
        "leading_team_87 utilise uniquement 0, 1, 2",
        valid_values
    )

    missing = int(series.isna().sum())

    print_status(
        "leading_team_87 sans valeur manquante",
        missing == 0
    )

    print()
    print(
        "⚠️ IMPORTANT : "
        "ce script vérifie uniquement que les valeurs sont 0/1/2."
    )

    print(
        "⚠️ Le mapping exact doit être identique entre "
        "le dataset et le live feature engineering."
    )

    print(
        "   Exemple possible : "
        "0 = nul, 1 = domicile mène, 2 = extérieur mène."
    )


# ============================================================
# VALIDATION FEATURES SENSIBLES
# ============================================================

def validate_sensitive_time_features(df):

    print_header("11. VALIDATION FEATURES TEMPORELLES")

    sensitive_features = [
        "last_goal_minute",
        "minutes_since_last_goal",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ]

    for feature in sensitive_features:

        if feature not in df.columns:
            print(f"⚠️ {feature} absent")
            continue

        missing = int(
            df[feature].isna().sum()
        )

        unique = df[feature].nunique(
            dropna=True
        )

        print(
            f"{feature:<35} "
            f"missing={missing:<5} "
            f"unique={unique}"
        )

    print()

    print(
        "⚠️ Ces variables sont potentiellement très prédictives."
    )

    print(
        "⚠️ Elles doivent être calculées uniquement avec "
        "des informations disponibles AVANT la prédiction."
    )

    print(
        "⚠️ Le live scanner ne doit jamais utiliser une "
        "information postérieure à la minute de prédiction."
    )


# ============================================================
# STATISTIQUES FEATURES
# ============================================================

def validate_feature_statistics(df):

    print_header("12. STATISTIQUES DES FEATURES")

    interesting_features = [
        "home_avg_xg",
        "away_avg_xg",
        "league_2nd_half_over15_freq",
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_over05",
        "odds_over15",
        "odds_over25",
        "odds_over35",
        "odds_under25",
        "odds_btts_yes",
        "odds_btts_no",
        "score_87_home",
        "score_87_away",
        "goal_diff_87",
        "home_shots_87",
        "away_shots_87",
        "home_shots_on_target_87",
        "away_shots_on_target_87",
        "shots_80_87",
        "sot_80_87",
        "corners_80_87",
        "subs_80_87",
    ]

    available = [
        feature
        for feature in interesting_features
        if feature in df.columns
    ]

    if not available:

        print("Aucune feature disponible.")
        return

    stats = df[available].describe().T

    print(
        stats[
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ].to_string()
    )


# ============================================================
# VALIDATION ORDRE COMPLET DATASET
# ============================================================

def validate_exact_feature_order(df):

    print_header("13. VALIDATION ORDRE FINAL")

    expected_order = (
        [ID_COLUMN]
        + EXPECTED_FEATURES
        + [TARGET_COLUMN]
    )

    actual_order = list(df.columns)

    if actual_order == expected_order:

        print(
            "✅ Ordre exact du dataset :"
        )

        print(
            "match_id → "
            "59 PRE-MATCH → "
            "37 LIVE → "
            "goal_after_87"
        )

        print_status(
            "Ordre complet du dataset conforme",
            True
        )

    else:

        print(
            "❌ L'ordre complet du dataset diffère."
        )

        print()
        print(
            "Premières différences :"
        )

        max_len = max(
            len(expected_order),
            len(actual_order)
        )

        differences = 0

        for i in range(max_len):

            expected = (
                expected_order[i]
                if i < len(expected_order)
                else "<absent>"
            )

            actual = (
                actual_order[i]
                if i < len(actual_order)
                else "<absent>"
            )

            if expected != actual:

                print(
                    f"Position {i + 1}: "
                    f"attendu='{expected}' | "
                    f"dataset='{actual}'"
                )

                differences += 1

                if differences >= 20:

                    print(
                        "... différences supplémentaires "
                        "non affichées."
                    )

                    break

        print_status(
            "Ordre complet du dataset conforme",
            False
        )


# ============================================================
# RAPPORT FINAL
# ============================================================

def print_final_report():

    print()
    print("=" * 90)
    print("RAPPORT FINAL")
    print("=" * 90)

    print()
    print(f"Tests réussis : {PASS_COUNT}")
    print(f"Tests échoués : {FAIL_COUNT}")
    print(f"Avertissements : {WARNING_COUNT}")

    print()

    if FAIL_COUNT == 0:

        print(
            "🎯 VALIDATION RÉUSSIE"
        )

        print()
        print(
            "Le dataset, le JSON des features et la structure "
            "attendue du modèle sont cohérents."
        )

        print()
        print(
            "PROCHAINE ÉTAPE :"
        )

        print(
            "→ vérifier le mapping exact de leading_team_87"
        )

        print(
            "→ puis connecter les 59 features PRE-MATCH "
            "aux 37 features LIVE"
        )

        print(
            "→ puis envoyer les 96 features au modèle "
            "ET_500_d15_73_27.joblib"
        )

    else:

        print(
            "❌ VALIDATION NON RÉUSSIE"
        )

        print()
        print(
            "Il faut corriger les erreurs ❌ avant "
            "de connecter le modèle au scanner live."
        )

    print()
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("FEATURE VALIDATION — GOAL AFTER 87")
    print("=" * 90)

    print()
    print(
        "Projet : ML_Goal_After_87"
    )

    print(
        "Modèle : ET_500_d15_73_27"
    )

    print(
        "Dataset : ML_Dataset_PreMatch_Live87.csv"
    )

    print(
        "Target : goal_after_87"
    )

    print(
        "Features : 96"
    )

    # --------------------------------------------------------
    # 1. FICHIERS
    # --------------------------------------------------------

    validate_files()

    # --------------------------------------------------------
    # 2. DEFINITIONS FEATURES
    # --------------------------------------------------------

    validate_feature_definitions()

    # --------------------------------------------------------
    # 3. JSON MODELE
    # --------------------------------------------------------

    model_features = validate_model_features()

    # --------------------------------------------------------
    # 4. DATASET
    # --------------------------------------------------------

    df = load_dataset()

    if df is None:

        print_final_report()
        sys.exit(1)

    # --------------------------------------------------------
    # 5. STRUCTURE DATASET
    # --------------------------------------------------------

    validate_dataset_structure(df)

    # --------------------------------------------------------
    # 6. TARGET
    # --------------------------------------------------------

    validate_target(df)

    # --------------------------------------------------------
    # 7. COLONNES
    # --------------------------------------------------------

    validate_dataset_columns(df)

    # --------------------------------------------------------
    # 8. MISSING
    # --------------------------------------------------------

    validate_missing_values(df)

    # --------------------------------------------------------
    # 9. TYPES
    # --------------------------------------------------------

    validate_numeric_types(df)

    # --------------------------------------------------------
    # 10. LEADING TEAM
    # --------------------------------------------------------

    validate_leading_team(df)

    # --------------------------------------------------------
    # 11. FEATURES TEMPORELLES
    # --------------------------------------------------------

    validate_sensitive_time_features(df)

    # --------------------------------------------------------
    # 12. STATISTIQUES
    # --------------------------------------------------------

    validate_feature_statistics(df)

    # --------------------------------------------------------
    # 13. ORDRE FINAL
    # --------------------------------------------------------

    validate_exact_feature_order(df)

    # --------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------

    print_final_report()

    # Code retour shell
    if FAIL_COUNT > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()