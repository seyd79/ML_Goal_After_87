#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Permutation Importance - Validation Temporelle
Projet : ML_Goal_After_87

Objectif
--------
Identifier les variables qui apportent réellement du signal
sur la prédiction de goal_after_87.

Contrairement à feature_importances_ du Random Forest,
la permutation importance mesure la baisse de performance
lorsqu'une variable est mélangée sur le jeu de test.

Validation
----------
5 folds temporels expansifs :

Fold 1 : train 50% -> test 10%
Fold 2 : train 60% -> test 10%
Fold 3 : train 70% -> test 10%
Fold 4 : train 80% -> test 10%
Fold 5 : train 90% -> test 10%

Modèles testés
--------------
- 73/27
- 67/33
- 50/50

Sorties
-------
outputs/permutation_importance/
    permutation_importance_by_feature.csv
    permutation_importance_summary.csv
    permutation_importance_ranking.csv
    permutation_importance_report.txt

IMPORTANT
---------
La validation temporelle suppose que l'ordre des lignes du CSV
correspond bien à l'ordre chronologique des matchs.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "permutation_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "goal_after_87"

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 10

# Nombre de répétitions de permutation par feature.
# 5 = bon compromis temps / stabilité.
N_REPEATS = 5

# Nombre de workers pour permutation importance.
N_JOBS = -1

# Pour le Random Forest
RF_N_JOBS = -1

# Les 3 distributions utilisées précédemment
TRAINING_DISTRIBUTIONS = {
    "73_27": 0.0,
    "67_33": 0.33,
    "50_50": 0.50,
}

# Les pourcentages utilisés dans l'analyse précédente
TOP_PERCENTAGES = [5, 10, 15, 20, 25, 30]


# ============================================================
# FEATURES SUSPECTES
# ============================================================

EXPLICIT_SUSPECT_FEATURES = [
    "last_goal_minute",
    "minutes_since_last_goal",
]


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_numeric_dataframe(df, columns):
    """
    Convertit les features en numérique lorsque nécessaire.
    """
    X = df[columns].copy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def detect_suspect_features(columns):
    """
    Détecte les variables potentiellement liées directement
    au target / au timing du but.
    """

    suspects = set(EXPLICIT_SUSPECT_FEATURES)

    keywords = [
        "last_goal",
        "goal_minute",
        "minutes_since_goal",
        "goal_after",
        "late_goal",
        "final_goal",
        "goal_87",
        "goal_88",
        "goal_89",
        "goal_90",
        "goal_91",
        "goal_92",
        "goal_93",
        "goal_94",
        "goal_95",
        "goal_96",
        "goal_97",
        "goal_98",
        "goal_99",
    ]

    for col in columns:
        col_lower = col.lower()

        for keyword in keywords:
            if keyword in col_lower:
                suspects.add(col)
                break

    return sorted(suspects)


def create_temporal_folds(n_rows):
    """
    Crée les 5 folds temporels expansifs.
    """

    folds = []

    train_percentages = [50, 60, 70, 80, 90]

    for fold_number, train_pct in enumerate(train_percentages, start=1):

        train_end = int(n_rows * train_pct / 100)

        test_start = train_end
        test_end = int(n_rows * (train_pct + 10) / 100)

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        folds.append(
            {
                "fold": fold_number,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )

    return folds


def balance_training_data(
    X_train,
    y_train,
    target_positive_ratio,
    random_state=42,
):
    """
    Construit les distributions d'entraînement :

    73/27 : distribution réelle
    67/33 : sur-échantillonnage de la classe positive
    50/50 : équilibrage complet

    Le jeu de test n'est JAMAIS modifié.
    """

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    if target_positive_ratio <= 0:
        return X_train, y_train

    positive_indices = np.where(y_train.values == 1)[0]
    negative_indices = np.where(y_train.values == 0)[0]

    n_positive = len(positive_indices)
    n_negative = len(negative_indices)

    if n_positive == 0 or n_negative == 0:
        return X_train, y_train

    rng = np.random.default_rng(random_state)

    current_positive_ratio = n_positive / (n_positive + n_negative)

    # Si la distribution réelle est déjà au-dessus
    # de la cible, on ne fait pas de sous-échantillonnage.
    if current_positive_ratio >= target_positive_ratio:
        return X_train, y_train

    # --------------------------------------------------------
    # Exemple :
    # 73/27 -> cible ~33% positif
    # 67/33 -> cible ~33% positif
    # 50/50 -> cible 50% positif
    # --------------------------------------------------------

    desired_positive_count = int(
        target_positive_ratio
        * n_negative
        / (1 - target_positive_ratio)
    )

    desired_positive_count = max(
        desired_positive_count,
        n_positive,
    )

    additional_needed = desired_positive_count - n_positive

    if additional_needed <= 0:
        return X_train, y_train

    sampled_positive = rng.choice(
        positive_indices,
        size=additional_needed,
        replace=True,
    )

    selected_indices = np.concatenate(
        [
            np.arange(len(X_train)),
            sampled_positive,
        ]
    )

    X_balanced = X_train.iloc[selected_indices].reset_index(drop=True)
    y_balanced = y_train.iloc[selected_indices].reset_index(drop=True)

    return X_balanced, y_balanced


def calculate_top_percentage_metrics(
    y_true,
    probabilities,
    percentages,
):
    """
    Calcule le taux de buts dans les meilleurs X% des scores.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    order = np.argsort(-probabilities)

    results = []

    baseline = float(np.mean(y_true))

    for percentage in percentages:

        n_selected = max(
            1,
            int(len(y_true) * percentage / 100),
        )

        selected_indices = order[:n_selected]

        selected_actual = y_true[selected_indices]

        goal_rate = float(np.mean(selected_actual))

        lift = (
            goal_rate / baseline
            if baseline > 0
            else np.nan
        )

        results.append(
            {
                "top_percentage": percentage,
                "n_selected": n_selected,
                "goals": int(np.sum(selected_actual)),
                "goal_rate": goal_rate,
                "baseline_rate": baseline,
                "lift": lift,
            }
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings("ignore")

    print_header(
        "PERMUTATION IMPORTANCE - VALIDATION TEMPORELLE"
    )

    print(f"Projet       : {PROJECT_DIR}")
    print(f"Dataset      : {DATASET_PATH}")
    print(f"Target       : {TARGET}")
    print(f"N repeats    : {N_REPEATS}")

    # ========================================================
    # 1. CHARGEMENT
    # ========================================================

    if not DATASET_PATH.exists():

        print()
        print("ERREUR : dataset introuvable.")
        print(f"Chemin attendu : {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    print()
    print(f"Dataset chargé : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    if TARGET not in df.columns:

        print()
        print(f"ERREUR : colonne cible '{TARGET}' absente.")
        return

    # ========================================================
    # 2. TARGET
    # ========================================================

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if y.isna().any():

        print("ERREUR : valeurs manquantes dans le target.")
        return

    y = y.astype(int)

    print()
    print("Distribution du target :")

    target_counts = y.value_counts().sort_index()

    for cls, count in target_counts.items():

        pct = count / len(y) * 100

        print(
            f"  Classe {cls}: "
            f"{count:,} "
            f"({pct:.2f}%)"
        )

    # ========================================================
    # 3. FEATURES
    # ========================================================

    excluded_columns = {
        TARGET,
        "match_id",
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
    ]

    X = safe_numeric_dataframe(
        df,
        feature_columns,
    )

    print()
    print(f"Features numériques : {len(feature_columns)}")

    # ========================================================
    # 4. FEATURES SUSPECTES
    # ========================================================

    suspects = detect_suspect_features(
        feature_columns
    )

    print()
    print("Variables suspectes détectées :")

    if suspects:

        for feature in suspects:
            print(f"  - {feature}")

    else:

        print("  Aucune")

    # ========================================================
    # 5. FOLDS TEMPORELS
    # ========================================================

    folds = create_temporal_folds(
        len(df)
    )

    print()
    print("Folds temporels :")

    for fold_info in folds:

        fold = fold_info["fold"]
        train_idx = fold_info["train_indices"]
        test_idx = fold_info["test_indices"]

        print(
            f"  Fold {fold}: "
            f"train={len(train_idx):,} "
            f"test={len(test_idx):,}"
        )

    # ========================================================
    # LISTES DE RESULTATS
    # ========================================================

    all_permutation_results = []
    all_summary_results = []
    all_top_results = []

    # ========================================================
    # 6. BOUCLE MODELES
    # ========================================================

    for distribution_name, positive_ratio in TRAINING_DISTRIBUTIONS.items():

        print_header(
            f"MODELE {distribution_name}"
        )

        print(
            f"Distribution cible d'entraînement : "
            f"{distribution_name}"
        )

        for fold_info in folds:

            fold = fold_info["fold"]

            train_idx = fold_info["train_indices"]
            test_idx = fold_info["test_indices"]

            print()
            print(
                f"[{distribution_name}] "
                f"Fold {fold}/5"
            )

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()

            X_test = X.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # ------------------------------------------------
            # Distribution training
            # ------------------------------------------------

            X_train_balanced, y_train_balanced = (
                balance_training_data(
                    X_train,
                    y_train,
                    target_positive_ratio=positive_ratio,
                    random_state=RANDOM_STATE + fold,
                )
            )

            train_positive_rate = (
                y_train_balanced.mean()
            )

            print(
                f"  Train : {len(X_train_balanced):,} "
                f"| positif : "
                f"{train_positive_rate:.2%}"
            )

            print(
                f"  Test  : {len(X_test):,} "
                f"| positif : "
                f"{y_test.mean():.2%}"
            )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=N_ESTIMATORS,
                            max_depth=MAX_DEPTH,
                            random_state=RANDOM_STATE,
                            n_jobs=RF_N_JOBS,
                        ),
                    ),
                ]
            )

            # ------------------------------------------------
            # FIT
            # ------------------------------------------------

            model.fit(
                X_train_balanced,
                y_train_balanced,
            )

            # ------------------------------------------------
            # BASELINE TEST SCORE
            # ------------------------------------------------

            test_probabilities = model.predict_proba(
                X_test
            )[:, 1]

            baseline_roc_auc = roc_auc_score(
                y_test,
                test_probabilities,
            )

            baseline_pr_auc = average_precision_score(
                y_test,
                test_probabilities,
            )

            print(
                f"  ROC-AUC : {baseline_roc_auc:.4f}"
            )

            print(
                f"  PR-AUC  : {baseline_pr_auc:.4f}"
            )

            # ------------------------------------------------
            # TOP %
            # ------------------------------------------------

            top_results = calculate_top_percentage_metrics(
                y_test.values,
                test_probabilities,
                TOP_PERCENTAGES,
            )

            for top_result in top_results:

                all_top_results.append(
                    {
                        "model": distribution_name,
                        "fold": fold,
                        "baseline_roc_auc": baseline_roc_auc,
                        "baseline_pr_auc": baseline_pr_auc,
                        **top_result,
                    }
                )

            # ------------------------------------------------
            # PERMUTATION IMPORTANCE
            # ------------------------------------------------

            print(
                "  Calcul de la permutation importance..."
            )

            permutation = permutation_importance(
                estimator=model,
                X=X_test,
                y=y_test,
                scoring="roc_auc",
                n_repeats=N_REPEATS,
                random_state=RANDOM_STATE + fold,
                n_jobs=N_JOBS,
            )

            importances_mean = (
                permutation.importances_mean
            )

            importances_std = (
                permutation.importances_std
            )

            importances_all = (
                permutation.importances
            )

            # ------------------------------------------------
            # RESULTATS PAR FEATURE
            # ------------------------------------------------

            fold_results = []

            for feature_index, feature in enumerate(
                feature_columns
            ):

                mean_importance = float(
                    importances_mean[feature_index]
                )

                std_importance = float(
                    importances_std[feature_index]
                )

                repeated_values = (
                    importances_all[feature_index]
                )

                positive_repetitions = int(
                    np.sum(
                        repeated_values > 0
                    )
                )

                negative_repetitions = int(
                    np.sum(
                        repeated_values < 0
                    )
                )

                zero_repetitions = int(
                    np.sum(
                        repeated_values == 0
                    )
                )

                fold_result = {
                    "model": distribution_name,
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": mean_importance,
                    "importance_std": std_importance,
                    "importance_min": float(
                        np.min(repeated_values)
                    ),
                    "importance_max": float(
                        np.max(repeated_values)
                    ),
                    "positive_repetitions": positive_repetitions,
                    "negative_repetitions": negative_repetitions,
                    "zero_repetitions": zero_repetitions,
                    "positive_repetition_rate": (
                        positive_repetitions
                        / N_REPEATS
                    ),
                    "baseline_roc_auc": baseline_roc_auc,
                    "baseline_pr_auc": baseline_pr_auc,
                    "is_suspect": (
                        feature in suspects
                    ),
                }

                fold_results.append(
                    fold_result
                )

                all_permutation_results.append(
                    fold_result
                )

            # ------------------------------------------------
            # TOP FEATURES DU FOLD
            # ------------------------------------------------

            fold_results_sorted = sorted(
                fold_results,
                key=lambda x: x["importance_mean"],
                reverse=True,
            )

            print()
            print("  Top 10 features du fold :")

            for rank, item in enumerate(
                fold_results_sorted[:10],
                start=1,
            ):

                print(
                    f"    {rank:2d}. "
                    f"{item['feature']:<45} "
                    f"{item['importance_mean']:+.6f}"
                )

            # ------------------------------------------------
            # RESUME DU FOLD
            # ------------------------------------------------

            for item in fold_results:

                all_summary_results.append(
                    {
                        "model": distribution_name,
                        "fold": fold,
                        "feature": item["feature"],
                        "importance_mean": item[
                            "importance_mean"
                        ],
                        "importance_std": item[
                            "importance_std"
                        ],
                        "positive_repetition_rate": item[
                            "positive_repetition_rate"
                        ],
                        "is_suspect": item[
                            "is_suspect"
                        ],
                    }
                )

    # ========================================================
    # 7. DATAFRAMES
    # ========================================================

    permutation_df = pd.DataFrame(
        all_permutation_results
    )

    summary_df = pd.DataFrame(
        all_summary_results
    )

    top_df = pd.DataFrame(
        all_top_results
    )

    # ========================================================
    # 8. AGREGER LES FEATURES
    # ========================================================

    ranking_rows = []

    for model_name in TRAINING_DISTRIBUTIONS.keys():

        model_data = summary_df[
            summary_df["model"] == model_name
        ].copy()

        if model_data.empty:
            continue

        grouped = (
            model_data
            .groupby("feature")
            .agg(
                mean_importance=(
                    "importance_mean",
                    "mean",
                ),
                std_importance_across_folds=(
                    "importance_mean",
                    "std",
                ),
                min_importance=(
                    "importance_mean",
                    "min",
                ),
                max_importance=(
                    "importance_mean",
                    "max",
                ),
                mean_positive_repetition_rate=(
                    "positive_repetition_rate",
                    "mean",
                ),
                folds_positive=(
                    "importance_mean",
                    lambda x: int(
                        np.sum(x > 0)
                    ),
                ),
                folds_total=(
                    "importance_mean",
                    "count",
                ),
                is_suspect=(
                    "is_suspect",
                    "max",
                ),
            )
            .reset_index()
        )

        grouped["stability"] = (
            grouped["folds_positive"]
            / grouped["folds_total"]
        )

        # ----------------------------------------------------
        # Score de robustesse
        #
        # On ne cherche pas uniquement la plus grande
        # importance moyenne.
        #
        # Une feature doit également :
        # - rester positive dans plusieurs folds
        # - avoir une faible variabilité
        # ----------------------------------------------------

        grouped["robust_importance"] = (
            grouped["mean_importance"]
            * grouped["stability"]
        )

        grouped["model"] = model_name

        grouped = grouped.sort_values(
            by=[
                "robust_importance",
                "mean_importance",
            ],
            ascending=False,
        ).reset_index(drop=True)

        grouped["rank"] = (
            np.arange(len(grouped)) + 1
        )

        ranking_rows.append(
            grouped
        )

    ranking_df = pd.concat(
        ranking_rows,
        ignore_index=True,
    )

    # ========================================================
    # 9. RANKING GLOBAL
    # ========================================================

    global_ranking = (
        ranking_df
        .groupby("feature")
        .agg(
            models_tested=(
                "model",
                "nunique",
            ),
            mean_importance_global=(
                "mean_importance",
                "mean",
            ),
            mean_robust_importance=(
                "robust_importance",
                "mean",
            ),
            min_mean_importance=(
                "mean_importance",
                "min",
            ),
            max_mean_importance=(
                "mean_importance",
                "max",
            ),
            mean_stability=(
                "stability",
                "mean",
            ),
            models_positive=(
                "mean_importance",
                lambda x: int(
                    np.sum(x > 0)
                ),
            ),
            is_suspect=(
                "is_suspect",
                "max",
            ),
        )
        .reset_index()
    )

    global_ranking = global_ranking.sort_values(
        by=[
            "mean_robust_importance",
            "mean_importance_global",
        ],
        ascending=False,
    ).reset_index(drop=True)

    global_ranking["global_rank"] = (
        np.arange(
            len(global_ranking)
        ) + 1
    )

    # ========================================================
    # 10. SAUVEGARDE CSV
    # ========================================================

    permutation_path = (
        OUTPUT_DIR
        / "permutation_importance_by_feature.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "permutation_importance_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "permutation_importance_ranking.csv"
    )

    top_path = (
        OUTPUT_DIR
        / "permutation_top_percentages.csv"
    )

    permutation_df.to_csv(
        permutation_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    top_df.to_csv(
        top_path,
        index=False,
    )

    # ========================================================
    # 11. RAPPORT TEXTE
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "permutation_importance_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "PERMUTATION IMPORTANCE - RAPPORT\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        report.write(
            f"Dataset : {DATASET_PATH}\n"
        )

        report.write(
            f"Rows : {len(df):,}\n"
        )

        report.write(
            f"Features : {len(feature_columns)}\n"
        )

        report.write(
            f"Target : {TARGET}\n"
        )

        report.write(
            f"N repeats : {N_REPEATS}\n\n"
        )

        report.write(
            "FEATURES SUSPECTES\n"
        )

        report.write(
            "-" * 80 + "\n"
        )

        for suspect in suspects:
            report.write(
                f"- {suspect}\n"
            )

        report.write("\n")

        # ----------------------------------------------------
        # Ranking par modèle
        # ----------------------------------------------------

        for model_name in TRAINING_DISTRIBUTIONS.keys():

            report.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            report.write(
                f"MODELE {model_name}\n"
            )

            report.write(
                "=" * 80
                + "\n\n"
            )

            model_ranking = ranking_df[
                ranking_df["model"] == model_name
            ].sort_values(
                "robust_importance",
                ascending=False,
            )

            report.write(
                "TOP 30 FEATURES\n\n"
            )

            for _, row in model_ranking.head(30).iterrows():

                report.write(
                    f"{int(row['rank']):3d}. "
                    f"{row['feature']:<45} "
                    f"mean={row['mean_importance']:+.6f} "
                    f"robust={row['robust_importance']:+.6f} "
                    f"stability={row['stability']:.0%} "
                    f"folds={int(row['folds_positive'])}/"
                    f"{int(row['folds_total'])}\n"
                )

        # ----------------------------------------------------
        # Global
        # ----------------------------------------------------

        report.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        report.write(
            "TOP 30 GLOBAL FEATURES\n"
        )

        report.write(
            "=" * 80
            + "\n\n"
        )

        for _, row in global_ranking.head(30).iterrows():

            report.write(
                f"{int(row['global_rank']):3d}. "
                f"{row['feature']:<45} "
                f"mean={row['mean_importance_global']:+.6f} "
                f"robust={row['mean_robust_importance']:+.6f} "
                f"stability={row['mean_stability']:.0%} "
                f"models={int(row['models_positive'])}/"
                f"{int(row['models_tested'])}\n"
            )

        # ----------------------------------------------------
        # Top percentages
        # ----------------------------------------------------

        report.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        report.write(
            "TOP PERCENTAGES - MOYENNE PAR MODELE\n"
        )

        report.write(
            "=" * 80
            + "\n\n"
        )

        if not top_df.empty:

            top_summary = (
                top_df
                .groupby(
                    [
                        "model",
                        "top_percentage",
                    ]
                )
                .agg(
                    mean_goal_rate=(
                        "goal_rate",
                        "mean",
                    ),
                    std_goal_rate=(
                        "goal_rate",
                        "std",
                    ),
                    mean_baseline=(
                        "baseline_rate",
                        "mean",
                    ),
                    mean_lift=(
                        "lift",
                        "mean",
                    ),
                )
                .reset_index()
            )

            for model_name in TRAINING_DISTRIBUTIONS.keys():

                report.write(
                    f"\n{model_name}\n"
                )

                model_top = top_summary[
                    top_summary["model"]
                    == model_name
                ]

                for _, row in model_top.iterrows():

                    report.write(
                        f"Top {int(row['top_percentage']):2d}% | "
                        f"goal_rate="
                        f"{row['mean_goal_rate']:.2%} | "
                        f"baseline="
                        f"{row['mean_baseline']:.2%} | "
                        f"lift="
                        f"{row['mean_lift']:.2f}x\n"
                    )

        # ----------------------------------------------------
        # Conclusion
        # ----------------------------------------------------

        report.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        report.write(
            "INTERPRETATION\n"
        )

        report.write(
            "=" * 80
            + "\n\n"
        )

        report.write(
            "La permutation importance mesure la baisse de "
            "ROC-AUC lorsque chaque feature est permutée "
            "sur le jeu de test temporel.\n\n"
        )

        report.write(
            "Une feature robuste doit idéalement avoir :\n"
            "- une importance moyenne positive ;\n"
            "- une importance positive dans plusieurs folds ;\n"
            "- une stabilité élevée ;\n"
            "- une importance qui ne dépend pas uniquement "
            "d'un modèle ou d'un fold.\n\n"
        )

        report.write(
            "ATTENTION : une importance élevée ne prouve pas "
            "qu'une variable est causalement responsable "
            "du but après la 87e minute.\n"
        )

        report.write(
            "Les variables suspectes doivent rester "
            "interprétées avec prudence.\n"
        )

    # ========================================================
    # 12. AFFICHAGE FINAL
    # ========================================================

    print_header(
        "TOP 30 GLOBAL FEATURES"
    )

    print(
        global_ranking[
            [
                "global_rank",
                "feature",
                "mean_importance_global",
                "mean_robust_importance",
                "mean_stability",
                "models_positive",
                "models_tested",
                "is_suspect",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print_header(
        "FICHIERS GENERES"
    )

    print(
        f"1. {permutation_path}"
    )

    print(
        f"2. {summary_path}"
    )

    print(
        f"3. {ranking_path}"
    )

    print(
        f"4. {top_path}"
    )

    print(
        f"5. {report_path}"
    )

    print()
    print(
        "Analyse terminée."
    )

    print()
    print(
        "PROCHAINE ETAPE : utiliser le ranking pour "
        "construire les modèles 30 features / 15 features "
        "et les comparer sur exactement les mêmes folds."
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()