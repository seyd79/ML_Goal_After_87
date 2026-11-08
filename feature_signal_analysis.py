#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FEATURE SIGNAL ANALYSIS - GOAL AFTER 87

Objectifs :
1. Charger le dataset ML
2. Détecter les variables potentiellement suspectes
3. Comparer :
   - BASELINE : toutes les variables numériques
   - NO_SUSPECT : sans les variables potentiellement suspectes
4. Effectuer une validation chronologique en 5 folds
5. Mesurer :
   - ROC-AUC
   - PR-AUC
   - Top 5 %
   - Top 10 %
   - Top 15 %
   - Top 20 %
   - Top 25 %
   - Top 30 %
6. Calculer la stabilité du signal
7. Calculer les importances des variables
8. Générer plusieurs fichiers CSV + un rapport TXT

IMPORTANT :
Le dossier du projet possède réellement un ":" à la fin :
    ML_Goal_After_87:

Le script utilise donc volontairement ce chemin.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "feature_signal_analysis"

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
N_FOLDS = 5

TARGET = "goal_after_87"
MATCH_ID = "match_id"

TOP_PERCENTAGES = [5, 10, 15, 20, 25, 30]


# =============================================================================
# VARIABLES POTENTIELLEMENT SUSPECTES
# =============================================================================

EXPLICIT_SUSPECT_FEATURES = [
    "last_goal_minute",
    "minutes_since_last_goal",
]

SUSPECT_KEYWORDS = [
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


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# WILSON CONFIDENCE INTERVAL
# =============================================================================

def wilson_interval(successes, total, confidence=0.95):

    if total == 0:
        return np.nan, np.nan

    z = 1.959963984540054

    p = successes / total

    denominator = 1 + (z ** 2 / total)

    center = (
        p
        + (z ** 2 / (2 * total))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p) / total
            )
            + (
                z ** 2 / (4 * total ** 2)
            )
        )
        / denominator
    )

    return max(0, center - margin), min(1, center + margin)


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_dataset():

    print_header("CHARGEMENT DES DONNÉES")

    print(f"Dataset : {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"""
Dataset introuvable :

{DATASET_PATH}

Vérifie le chemin dans DATASET_PATH.
"""
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return df


# =============================================================================
# VALIDATION DU DATASET
# =============================================================================

def validate_dataset(df):

    print_subheader("VALIDATION DU DATASET")

    if TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET}' est absente du dataset."
        )

    if MATCH_ID not in df.columns:
        print(
            f"⚠️ Colonne '{MATCH_ID}' absente. "
            "La validation continuera sans elle."
        )

    print(f"Cible : {TARGET}")

    target_counts = df[TARGET].value_counts().sort_index()

    print("\nDistribution de la cible :")

    for value, count in target_counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"  Classe {value}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    missing_total = df.isna().sum().sum()

    print(
        f"\nValeurs manquantes totales : "
        f"{missing_total:,}"
    )


# =============================================================================
# CONVERSION NUMÉRIQUE
# =============================================================================

def safe_numeric_dataframe(df):

    numeric_df = df.copy()

    for column in numeric_df.columns:

        if not pd.api.types.is_numeric_dtype(
            numeric_df[column]
        ):
            numeric_df[column] = pd.to_numeric(
                numeric_df[column],
                errors="coerce"
            )

    return numeric_df


# =============================================================================
# DÉTECTION DES VARIABLES SUSPECTES
# =============================================================================

def detect_suspect_features(feature_columns):

    suspects = []

    details = []

    for feature in feature_columns:

        feature_lower = feature.lower()

        explicit = (
            feature in EXPLICIT_SUSPECT_FEATURES
        )

        keyword_matches = [
            keyword
            for keyword in SUSPECT_KEYWORDS
            if keyword in feature_lower
        ]

        if explicit or keyword_matches:

            suspects.append(feature)

            reason_parts = []

            if explicit:
                reason_parts.append(
                    "explicit_suspect"
                )

            if keyword_matches:
                reason_parts.append(
                    "keyword:"
                    + ",".join(keyword_matches)
                )

            details.append(
                {
                    "feature": feature,
                    "reason": "; ".join(
                        reason_parts
                    ),
                }
            )

    return suspects, pd.DataFrame(details)


# =============================================================================
# CRÉATION DU MODÈLE
# =============================================================================

def build_model():

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
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )

    return model


# =============================================================================
# CRÉATION DES FOLDS CHRONOLOGIQUES
# =============================================================================

def create_temporal_folds(n_samples):

    folds = []

    for fold_number in range(N_FOLDS):

        train_end = int(
            n_samples
            * (
                0.50
                + fold_number * 0.10
            )
        )

        test_end = int(
            n_samples
            * (
                0.60
                + fold_number * 0.10
            )
        )

        train_indices = np.arange(
            0,
            train_end
        )

        test_indices = np.arange(
            train_end,
            test_end
        )

        folds.append(
            (
                fold_number + 1,
                train_indices,
                test_indices,
            )
        )

    return folds


# =============================================================================
# MÉTRIQUES TOP %
# =============================================================================

def calculate_top_metrics(
    probabilities,
    actual,
    baseline_rate,
    fold,
    model_name,
):

    rows = []

    probabilities = np.asarray(
        probabilities
    )

    actual = np.asarray(actual)

    order = np.argsort(
        probabilities
    )[::-1]

    for top_percentage in TOP_PERCENTAGES:

        n_selected = max(
            1,
            int(
                len(actual)
                * top_percentage
                / 100
            ),
        )

        selected_indices = order[
            :n_selected
        ]

        selected_actual = actual[
            selected_indices
        ]

        goals = int(
            selected_actual.sum()
        )

        goal_rate = (
            goals
            / n_selected
        )

        if baseline_rate > 0:
            lift = (
                goal_rate
                / baseline_rate
            )
        else:
            lift = np.nan

        ci_low, ci_high = wilson_interval(
            goals,
            n_selected
        )

        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "top_percentage": top_percentage,
                "n_selected": n_selected,
                "goals": goals,
                "goal_rate": goal_rate,
                "baseline_rate": baseline_rate,
                "lift": lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "above_baseline": (
                    goal_rate
                    > baseline_rate
                ),
            }
        )

    return rows


# =============================================================================
# ÉVALUATION D'UN MODÈLE
# =============================================================================

def evaluate_model(
    X,
    y,
    folds,
    model_name,
):

    print_subheader(
        f"ÉVALUATION : {model_name}"
    )

    all_rows = []
    importance_rows = []

    for fold, train_idx, test_idx in folds:

        print(
            f"\nFold {fold}/{N_FOLDS}"
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        baseline_rate = (
            y_test.mean()
        )

        model = build_model()

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        try:
            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = average_precision_score(
                y_test,
                probabilities
            )
        except ValueError:
            pr_auc = np.nan

        print(
            f"  ROC-AUC : {roc_auc:.4f}"
        )

        print(
            f"  PR-AUC  : {pr_auc:.4f}"
        )

        top_rows = calculate_top_metrics(
            probabilities=probabilities,
            actual=y_test.values,
            baseline_rate=baseline_rate,
            fold=fold,
            model_name=model_name,
        )

        for row in top_rows:
            all_rows.append(
                {
                    **row,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )

        rf_model = model.named_steps[
            "model"
        ]

        importances = (
            rf_model.feature_importances_
        )

        for feature, importance in zip(
            X.columns,
            importances
        ):

            importance_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "feature": feature,
                    "importance": importance,
                }
            )

    return (
        pd.DataFrame(all_rows),
        pd.DataFrame(importance_rows),
    )


# =============================================================================
# RÉSUMÉ DES RÉSULTATS
# =============================================================================

def summarize_results(
    results_df
):

    summary_rows = []

    grouped = results_df.groupby(
        [
            "model",
            "top_percentage",
        ]
    )

    for (
        model_name,
        top_percentage,
    ), group in grouped:

        mean_goal_rate = (
            group["goal_rate"].mean()
        )

        std_goal_rate = (
            group["goal_rate"].std(
                ddof=0
            )
        )

        min_goal_rate = (
            group["goal_rate"].min()
        )

        max_goal_rate = (
            group["goal_rate"].max()
        )

        mean_baseline = (
            group["baseline_rate"].mean()
        )

        mean_lift = (
            group["lift"].mean()
        )

        min_lift = (
            group["lift"].min()
        )

        max_lift = (
            group["lift"].max()
        )

        folds_above_baseline = int(
            group["above_baseline"].sum()
        )

        total_folds = len(group)

        stability = (
            folds_above_baseline
            / total_folds
            if total_folds > 0
            else np.nan
        )

        mean_selected = (
            group["n_selected"].mean()
        )

        pooled_goals = (
            group["goals"].sum()
        )

        pooled_selected = (
            group["n_selected"].sum()
        )

        pooled_rate = (
            pooled_goals
            / pooled_selected
            if pooled_selected > 0
            else np.nan
        )

        pooled_lift = (
            pooled_rate
            / mean_baseline
            if mean_baseline > 0
            else np.nan
        )

        worst_margin = (
            group["goal_rate"]
            - group["baseline_rate"]
        ).min()

        best_margin = (
            group["goal_rate"]
            - group["baseline_rate"]
        ).max()

        summary_rows.append(
            {
                "model": model_name,
                "top_percentage": top_percentage,
                "mean_goal_rate": mean_goal_rate,
                "std_goal_rate": std_goal_rate,
                "min_goal_rate": min_goal_rate,
                "max_goal_rate": max_goal_rate,
                "mean_baseline": mean_baseline,
                "mean_lift": mean_lift,
                "min_lift": min_lift,
                "max_lift": max_lift,
                "folds_above_baseline": folds_above_baseline,
                "total_folds": total_folds,
                "stability": stability,
                "mean_selected": mean_selected,
                "pooled_rate": pooled_rate,
                "pooled_lift": pooled_lift,
                "worst_margin_vs_baseline": worst_margin,
                "best_margin_vs_baseline": best_margin,
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# =============================================================================
# IMPORTANCE DES VARIABLES
# =============================================================================

def summarize_feature_importance(
    importance_df
):

    if importance_df.empty:
        return pd.DataFrame()

    summary = (
        importance_df
        .groupby("feature")
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),
            std_importance=(
                "importance",
                "std",
            ),
            min_importance=(
                "importance",
                "min",
            ),
            max_importance=(
                "importance",
                "max",
            ),
            folds_present=(
                "fold",
                "nunique",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    )

    return summary


# =============================================================================
# COMPARAISON BASELINE / NO SUSPECT
# =============================================================================

def compare_models(
    summary_df
):

    print_subheader(
        "COMPARAISON BASELINE VS NO_SUSPECT"
    )

    comparison = summary_df[
        summary_df["top_percentage"].isin(
            [10, 15, 20]
        )
    ].copy()

    if comparison.empty:
        return pd.DataFrame()

    comparison = comparison[
        [
            "model",
            "top_percentage",
            "mean_goal_rate",
            "std_goal_rate",
            "mean_baseline",
            "mean_lift",
            "stability",
            "worst_margin_vs_baseline",
            "best_margin_vs_baseline",
        ]
    ].sort_values(
        [
            "top_percentage",
            "model",
        ]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


# =============================================================================
# VERDICT
# =============================================================================

def generate_verdict(
    summary_df,
    comparison_df,
    suspects,
):

    print_subheader(
        "VERDICT AUTOMATIQUE"
    )

    if summary_df.empty:

        return (
            "IMPOSSIBLE : aucun résultat."
        )

    baseline = summary_df[
        summary_df["model"]
        == "BASELINE"
    ]

    no_suspect = summary_df[
        summary_df["model"]
        == "NO_SUSPECT"
    ]

    if baseline.empty:

        return (
            "IMPOSSIBLE : résultats BASELINE absents."
        )

    baseline_top10 = baseline[
        baseline["top_percentage"]
        == 10
    ]

    if baseline_top10.empty:

        return (
            "IMPOSSIBLE : Top 10 BASELINE absent."
        )

    baseline_top10 = baseline_top10.iloc[0]

    print(
        f"\nVariables suspectes détectées : "
        f"{len(suspects)}"
    )

    for feature in suspects:

        print(
            f"  ⚠️ {feature}"
        )

    print(
        "\nBASELINE Top 10 % :"
    )

    print(
        f"  Goal rate : "
        f"{baseline_top10['mean_goal_rate']:.2%}"
    )

    print(
        f"  Lift      : "
        f"{baseline_top10['mean_lift']:.2f}x"
    )

    print(
        f"  Stability : "
        f"{baseline_top10['stability']:.0%}"
    )

    if no_suspect.empty:

        verdict = (
            "NO_SUSPECT non disponible."
        )

        print(
            f"\nVERDICT : {verdict}"
        )

        return verdict

    no_suspect_top10 = no_suspect[
        no_suspect["top_percentage"]
        == 10
    ]

    if no_suspect_top10.empty:

        verdict = (
            "NO_SUSPECT Top 10 % absent."
        )

        print(
            f"\nVERDICT : {verdict}"
        )

        return verdict

    no_suspect_top10 = (
        no_suspect_top10.iloc[0]
    )

    baseline_rate = (
        baseline_top10["mean_goal_rate"]
    )

    no_suspect_rate = (
        no_suspect_top10["mean_goal_rate"]
    )

    difference = (
        no_suspect_rate
        - baseline_rate
    )

    print(
        "\nNO_SUSPECT Top 10 % :"
    )

    print(
        f"  Goal rate : "
        f"{no_suspect_rate:.2%}"
    )

    print(
        f"  Lift      : "
        f"{no_suspect_top10['mean_lift']:.2f}x"
    )

    print(
        f"  Stability : "
        f"{no_suspect_top10['stability']:.0%}"
    )

    print(
        f"\nDifférence NO_SUSPECT - BASELINE : "
        f"{difference:+.2%}"
    )

    if no_suspect_rate >= (
        baseline_rate - 0.02
    ):

        verdict = (
            "SIGNAL ROBUSTE : "
            "le signal reste largement présent "
            "après retrait des variables suspectes."
        )

    elif no_suspect_rate >= (
        baseline_rate - 0.05
    ):

        verdict = (
            "SIGNAL PARTIELLEMENT ROBUSTE : "
            "le signal baisse mais reste exploitable "
            "pour une analyse approfondie."
        )

    else:

        verdict = (
            "SIGNAL À RISQUE : "
            "le signal chute fortement après retrait "
            "des variables suspectes. "
            "Une analyse de fuite de données est nécessaire."
        )

    print(
        f"\nVERDICT : {verdict}"
    )

    return verdict


# =============================================================================
# RAPPORT TXT
# =============================================================================

def generate_report(
    df,
    suspects,
    summary_df,
    comparison_df,
    importance_summary,
    verdict,
):

    report_path = (
        OUTPUT_DIR
        / "feature_signal_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FEATURE SIGNAL ANALYSIS - "
            "GOAL AFTER 87\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            f"Dataset : {DATASET_PATH}\n"
        )

        file.write(
            f"Rows : {len(df):,}\n"
        )

        file.write(
            f"Columns : {len(df.columns):,}\n\n"
        )

        file.write(
            "VARIABLES SUSPECTES\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        if suspects:

            for feature in suspects:

                file.write(
                    f"- {feature}\n"
                )

        else:

            file.write(
                "Aucune variable suspecte détectée.\n"
            )

        file.write(
            "\nRÉSUMÉ\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        if not summary_df.empty:

            file.write(
                summary_df.to_string(
                    index=False
                )
            )

            file.write("\n")

        file.write(
            "\nCOMPARAISON BASELINE / NO_SUSPECT\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        if (
            comparison_df is not None
            and not comparison_df.empty
        ):

            file.write(
                comparison_df.to_string(
                    index=False
                )
            )

            file.write("\n")

        file.write(
            "\nTOP FEATURES\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        if (
            importance_summary is not None
            and not importance_summary.empty
        ):

            file.write(
                importance_summary
                .head(30)
                .to_string(
                    index=False
                )
            )

            file.write("\n")

        file.write(
            "\nVERDICT\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            verdict
            + "\n"
        )

    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "FEATURE SIGNAL ANALYSIS - GOAL AFTER 87"
    )

    # -------------------------------------------------------------------------
    # 1. Chargement
    # -------------------------------------------------------------------------

    df = load_dataset()

    # -------------------------------------------------------------------------
    # 2. Validation
    # -------------------------------------------------------------------------

    validate_dataset(df)

    # -------------------------------------------------------------------------
    # 3. Préparation des features
    # -------------------------------------------------------------------------

    print_subheader(
        "PRÉPARATION DES FEATURES"
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            TARGET,
            MATCH_ID,
        ]
    ]

    X_raw = df[
        feature_columns
    ].copy()

    X_numeric = safe_numeric_dataframe(
        X_raw
    )

    # Retirer les colonnes entièrement non numériques
    all_nan_columns = [
        column
        for column in X_numeric.columns
        if X_numeric[column].isna().all()
    ]

    if all_nan_columns:

        print(
            "\nColonnes retirées "
            "(non numériques / entièrement NaN) :"
        )

        for column in all_nan_columns:

            print(
                f"  - {column}"
            )

        X_numeric = X_numeric.drop(
            columns=all_nan_columns
        )

    y = pd.to_numeric(
        df[TARGET],
        errors="raise"
    ).astype(int)

    print(
        f"\nFeatures utilisées : "
        f"{X_numeric.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # 4. Détection des variables suspectes
    # -------------------------------------------------------------------------

    print_subheader(
        "DÉTECTION DES VARIABLES SUSPECTES"
    )

    suspects, suspect_details = (
        detect_suspect_features(
            X_numeric.columns
        )
    )

    if suspects:

        print(
            f"\n{len(suspects)} variable(s) suspecte(s) détectée(s) :"
        )

        for feature in suspects:

            print(
                f"  ⚠️ {feature}"
            )

    else:

        print(
            "\nAucune variable suspecte détectée."
        )

    # -------------------------------------------------------------------------
    # 5. Création des folds
    # -------------------------------------------------------------------------

    print_subheader(
        "VALIDATION CHRONOLOGIQUE"
    )

    folds = create_temporal_folds(
        len(df)
    )

    for (
        fold,
        train_idx,
        test_idx,
    ) in folds:

        print(
            f"Fold {fold}: "
            f"train={len(train_idx):,} | "
            f"test={len(test_idx):,}"
        )

    # -------------------------------------------------------------------------
    # 6. BASELINE
    # -------------------------------------------------------------------------

    baseline_results, baseline_importance = (
        evaluate_model(
            X=X_numeric,
            y=y,
            folds=folds,
            model_name="BASELINE",
        )
    )

    # -------------------------------------------------------------------------
    # 7. NO SUSPECT
    # -------------------------------------------------------------------------

    no_suspect_columns = [
        column
        for column in X_numeric.columns
        if column not in suspects
    ]

    X_no_suspect = X_numeric[
        no_suspect_columns
    ].copy()

    print_subheader(
        "MODÈLE SANS VARIABLES SUSPECTES"
    )

    print(
        f"Features restantes : "
        f"{X_no_suspect.shape[1]}"
    )

    no_suspect_results, no_suspect_importance = (
        evaluate_model(
            X=X_no_suspect,
            y=y,
            folds=folds,
            model_name="NO_SUSPECT",
        )
    )

    # -------------------------------------------------------------------------
    # 8. Combinaison
    # -------------------------------------------------------------------------

    all_results = pd.concat(
        [
            baseline_results,
            no_suspect_results,
        ],
        ignore_index=True,
    )

    all_importance = pd.concat(
        [
            baseline_importance,
            no_suspect_importance,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # 9. Résumé
    # -------------------------------------------------------------------------

    summary_df = summarize_results(
        all_results
    )

    # -------------------------------------------------------------------------
    # 10. Importance des variables
    # -------------------------------------------------------------------------

    importance_summary = (
        summarize_feature_importance(
            all_importance
        )
    )

    # -------------------------------------------------------------------------
    # 11. Comparaison
    # -------------------------------------------------------------------------

    comparison_df = compare_models(
        summary_df
    )

    # -------------------------------------------------------------------------
    # 12. Verdict
    # -------------------------------------------------------------------------

    verdict = generate_verdict(
        summary_df=summary_df,
        comparison_df=comparison_df,
        suspects=suspects,
    )

    # -------------------------------------------------------------------------
    # 13. Création du dossier
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # 14. Sauvegarde CSV
    # -------------------------------------------------------------------------

    by_fold_path = (
        OUTPUT_DIR
        / "feature_signal_by_fold.csv"
    )

    top_percentages_path = (
        OUTPUT_DIR
        / "feature_signal_top_percentages.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_signal_summary.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "baseline_vs_no_suspect.csv"
    )

    importance_by_fold_path = (
        OUTPUT_DIR
        / "feature_importance_by_fold.csv"
    )

    importance_summary_path = (
        OUTPUT_DIR
        / "feature_importance_summary.csv"
    )

    suspect_path = (
        OUTPUT_DIR
        / "suspect_features.csv"
    )

    all_results.to_csv(
        by_fold_path,
        index=False
    )

    all_results[
        [
            "model",
            "fold",
            "top_percentage",
            "n_selected",
            "goals",
            "goal_rate",
            "baseline_rate",
            "lift",
            "ci_low",
            "ci_high",
            "above_baseline",
        ]
    ].to_csv(
        top_percentages_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    all_importance.to_csv(
        importance_by_fold_path,
        index=False
    )

    importance_summary.to_csv(
        importance_summary_path,
        index=False
    )

    if suspect_details.empty:

        pd.DataFrame(
            columns=[
                "feature",
                "reason",
            ]
        ).to_csv(
            suspect_path,
            index=False
        )

    else:

        suspect_details.to_csv(
            suspect_path,
            index=False
        )

    # -------------------------------------------------------------------------
    # 15. Rapport
    # -------------------------------------------------------------------------

    report_path = generate_report(
        df=df,
        suspects=suspects,
        summary_df=summary_df,
        comparison_df=comparison_df,
        importance_summary=importance_summary,
        verdict=verdict,
    )

    # -------------------------------------------------------------------------
    # 16. Affichage final
    # -------------------------------------------------------------------------

    print_header(
        "FICHIERS GÉNÉRÉS"
    )

    generated_files = [
        by_fold_path,
        top_percentages_path,
        summary_path,
        comparison_path,
        importance_by_fold_path,
        importance_summary_path,
        suspect_path,
        report_path,
    ]

    for path in generated_files:

        print(
            f"✓ {path}"
        )

    # -------------------------------------------------------------------------
    # 17. Top features
    # -------------------------------------------------------------------------

    print_subheader(
        "TOP 20 FEATURES - BASELINE"
    )

    baseline_importance_summary = (
        summarize_feature_importance(
            baseline_importance
        )
    )

    if not baseline_importance_summary.empty:

        print(
            baseline_importance_summary
            .head(20)
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # 18. Résumé final
    # -------------------------------------------------------------------------

    print_header(
        "ANALYSE TERMINÉE"
    )

    print(
        f"Dataset : {len(df):,} matchs"
    )

    print(
        f"Features BASELINE : "
        f"{X_numeric.shape[1]}"
    )

    print(
        f"Features NO_SUSPECT : "
        f"{X_no_suspect.shape[1]}"
    )

    print(
        f"Variables suspectes : "
        f"{len(suspects)}"
    )

    print(
        f"\nVerdict : {verdict}"
    )


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()