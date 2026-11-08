from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "nested_feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 10

N_PERMUTATION_REPEATS = 5

TOP_K_VALUES = [10, 15, 20, 30, 50]

DISTRIBUTIONS = {
    "73_27": 0.0,
    "67_33": 0.33,
    "50_50": 0.50,
}

# 5 temporal folds :
# train 50% -> test 10%
# train 60% -> test 10%
# train 70% -> test 10%
# train 80% -> test 10%
# train 90% -> test 10%

TEMPORAL_FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]


# ============================================================
# FEATURES SUSPECTES
# ============================================================

EXPLICIT_SUSPECT_FEATURES = [
    "last_goal_minute",
    "minutes_since_last_goal",
]


# ============================================================
# UTILITAIRES
# ============================================================

def detect_suspect_features(features):
    """
    Détecte automatiquement les variables pouvant être liées
    directement à l'événement cible.
    """

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

    suspects = set(EXPLICIT_SUSPECT_FEATURES)

    for feature in features:
        feature_lower = feature.lower()

        for keyword in keywords:
            if keyword in feature_lower:
                suspects.add(feature)
                break

    return sorted(suspects)


def build_model():
    """
    Modèle identique à celui utilisé dans les étapes précédentes.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None
                )
            )
        ]
    )


def balance_training_data(X, y, target_positive_ratio):
    """
    Rééquilibre uniquement les données d'entraînement.

    IMPORTANT :
    - jamais sur validation interne
    - jamais sur test externe
    """

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if target_positive_ratio <= 0:
        return X, y

    positive_idx = np.where(y.values == 1)[0]
    negative_idx = np.where(y.values == 0)[0]

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return X, y

    current_ratio = len(positive_idx) / len(y)

    if current_ratio >= target_positive_ratio:
        return X, y

    # Nombre de positifs nécessaire pour atteindre le ratio demandé
    target_positive_count = int(
        len(negative_idx)
        * target_positive_ratio
        / (1 - target_positive_ratio)
    )

    additional_needed = target_positive_count - len(positive_idx)

    if additional_needed <= 0:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)

    sampled_positive = rng.choice(
        positive_idx,
        size=additional_needed,
        replace=True
    )

    selected_indices = np.concatenate(
        [
            negative_idx,
            positive_idx,
            sampled_positive,
        ]
    )

    rng.shuffle(selected_indices)

    X_balanced = X.iloc[selected_indices].reset_index(drop=True)
    y_balanced = y.iloc[selected_indices].reset_index(drop=True)

    return X_balanced, y_balanced


def top_percentage_metrics(y_true, probabilities):
    """
    Calcule les performances dans les zones Top 5/10/15/20.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    baseline = np.mean(y_true)

    results = []

    for percentage in [5, 10, 15, 20]:
        n_selected = max(
            1,
            int(np.ceil(len(y_true) * percentage / 100))
        )

        order = np.argsort(-probabilities)

        selected_indices = order[:n_selected]

        selected_actual = y_true[selected_indices]

        goals = int(np.sum(selected_actual))

        goal_rate = (
            goals / n_selected
            if n_selected > 0
            else np.nan
        )

        lift = (
            goal_rate / baseline
            if baseline > 0
            else np.nan
        )

        results.append(
            {
                "top_percentage": percentage,
                "n_selected": n_selected,
                "goals": goals,
                "goal_rate": goal_rate,
                "baseline_rate": baseline,
                "lift": lift,
            }
        )

    return results


def wilson_interval(successes, n, z=1.96):
    """
    Intervalle de confiance de Wilson.
    """

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1 + (z ** 2 / n)

    centre = (
        p
        + (z ** 2 / (2 * n))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p) / n
            )
            + (
                z ** 2 / (4 * n ** 2)
            )
        )
        / denominator
    )

    return centre - margin, centre + margin


# ============================================================
# TEMPORAL FOLD
# ============================================================

def get_temporal_split(n_rows, train_fraction, test_end_fraction):

    train_end = int(n_rows * train_fraction)
    test_end = int(n_rows * test_end_fraction)

    train_indices = np.arange(0, train_end)
    test_indices = np.arange(train_end, test_end)

    return train_indices, test_indices


# ============================================================
# FEATURE SELECTION NESTED
# ============================================================

def select_features_nested(
    X_outer_train,
    y_outer_train,
    distribution_name,
    fold_number
):
    """
    Sélection des features sans utiliser le test externe.

    Étapes :

    OUTER TRAIN
        |
        +--> INNER TRAIN
        |
        +--> INNER VALIDATION
                 |
                 v
        permutation importance
                 |
                 v
            TOP FEATURES

    Le test externe n'est jamais utilisé pour sélectionner
    les features.
    """

    n_outer = len(X_outer_train)

    # --------------------------------------------------------
    # Inner split chronologique
    # --------------------------------------------------------

    inner_train_end = int(n_outer * 0.80)

    inner_train_idx = np.arange(
        0,
        inner_train_end
    )

    inner_valid_idx = np.arange(
        inner_train_end,
        n_outer
    )

    X_inner_train = X_outer_train.iloc[
        inner_train_idx
    ].copy()

    y_inner_train = y_outer_train.iloc[
        inner_train_idx
    ].copy()

    X_inner_valid = X_outer_train.iloc[
        inner_valid_idx
    ].copy()

    y_inner_valid = y_outer_train.iloc[
        inner_valid_idx
    ].copy()

    # --------------------------------------------------------
    # Rééquilibrage UNIQUEMENT inner train
    # --------------------------------------------------------

    target_ratio = DISTRIBUTIONS[distribution_name]

    X_inner_train_balanced, y_inner_train_balanced = (
        balance_training_data(
            X_inner_train,
            y_inner_train,
            target_ratio
        )
    )

    # --------------------------------------------------------
    # Modèle de sélection
    # --------------------------------------------------------

    selector_model = build_model()

    selector_model.fit(
        X_inner_train_balanced,
        y_inner_train_balanced
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    permutation = permutation_importance(
        selector_model,
        X_inner_valid,
        y_inner_valid,
        scoring="roc_auc",
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_outer_train.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    ).reset_index(drop=True)

    importance_df["fold"] = fold_number
    importance_df["distribution"] = distribution_name

    return (
        importance_df,
        inner_train_idx,
        inner_valid_idx
    )


# ============================================================
# ÉVALUATION D'UN FEATURE SET
# ============================================================

def evaluate_feature_set(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_columns,
    distribution_name
):

    X_train_selected = X_train[
        feature_columns
    ].copy()

    X_test_selected = X_test[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Balance uniquement du TRAIN
    # --------------------------------------------------------

    target_ratio = DISTRIBUTIONS[distribution_name]

    X_train_balanced, y_train_balanced = (
        balance_training_data(
            X_train_selected,
            y_train,
            target_ratio
        )
    )

    # --------------------------------------------------------
    # Modèle final
    # --------------------------------------------------------

    model = build_model()

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    probabilities = model.predict_proba(
        X_test_selected
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Top percentages
    # --------------------------------------------------------

    top_metrics = top_percentage_metrics(
        y_test.values,
        probabilities
    )

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prediction_positive_rate": np.mean(predictions),
    }

    for item in top_metrics:

        percentage = item["top_percentage"]

        metrics[
            f"top{percentage}_goal_rate"
        ] = item["goal_rate"]

        metrics[
            f"top{percentage}_lift"
        ] = item["lift"]

        metrics[
            f"top{percentage}_n"
        ] = item["n_selected"]

    return (
        metrics,
        probabilities,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NESTED FEATURE SELECTION")
    print("=" * 80)

    print()

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Dataset : {df.shape[0]:,} lignes x "
        f"{df.shape[1]} colonnes"
    )

    if "goal_after_87" not in df.columns:

        raise ValueError(
            "La colonne 'goal_after_87' est absente."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = df["goal_after_87"].astype(int)

    print(
        f"Target : {target.mean():.2%} positif / "
        f"{1 - target.mean():.2%} négatif"
    )

    # --------------------------------------------------------
    # Features numériques
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        column
        for column in numeric_columns
        if column != "goal_after_87"
    ]

    # match_id ne doit jamais être utilisé
    features = [
        column
        for column in features
        if column != "match_id"
    ]

    print(
        f"Features numériques : {len(features)}"
    )

    # --------------------------------------------------------
    # Suspects
    # --------------------------------------------------------

    suspects = detect_suspect_features(
        features
    )

    print()

    print(
        f"Variables suspectes détectées : "
        f"{len(suspects)}"
    )

    for feature in suspects:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X = df[features].copy()
    y = target.copy()

    n_rows = len(df)

    # --------------------------------------------------------
    # Stockage résultats
    # --------------------------------------------------------

    fold_results = []

    prediction_rows = []

    selected_feature_rows = []

    all_importance_rows = []

    # --------------------------------------------------------
    # Boucle distributions
    # --------------------------------------------------------

    for distribution_name in DISTRIBUTIONS:

        print()
        print("=" * 80)
        print(
            f"DISTRIBUTION : {distribution_name}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Boucle temporal folds
        # ----------------------------------------------------

        for fold_number, (
            train_fraction,
            test_end_fraction
        ) in enumerate(
            TEMPORAL_FOLDS,
            start=1
        ):

            train_idx, test_idx = get_temporal_split(
                n_rows,
                train_fraction,
                test_end_fraction
            )

            X_train = X.iloc[
                train_idx
            ].copy()

            y_train = y.iloc[
                train_idx
            ].copy()

            X_test = X.iloc[
                test_idx
            ].copy()

            y_test = y.iloc[
                test_idx
            ].copy()

            print()
            print(
                f"Fold {fold_number}"
            )

            print(
                f"  Train : {len(X_train):,}"
            )

            print(
                f"  Test  : {len(X_test):,}"
            )

            # =================================================
            # NESTED FEATURE SELECTION
            # =================================================

            (
                importance_df,
                inner_train_idx,
                inner_valid_idx
            ) = select_features_nested(
                X_train,
                y_train,
                distribution_name,
                fold_number
            )

            all_importance_rows.append(
                importance_df
            )

            # -------------------------------------------------
            # Sauvegarde importance
            # -------------------------------------------------

            importance_df.to_csv(
                OUTPUT_DIR
                / (
                    f"importance_"
                    f"{distribution_name}_"
                    f"fold{fold_number}.csv"
                ),
                index=False
            )

            # -------------------------------------------------
            # Sélection des TOP K
            # -------------------------------------------------

            ranked_features = (
                importance_df["feature"]
                .tolist()
            )

            feature_sets = {
                "FULL": features
            }

            for top_k in TOP_K_VALUES:

                feature_sets[
                    f"TOP{top_k}"
                ] = ranked_features[:top_k]

            # -------------------------------------------------
            # Évaluation FULL + TOP
            # -------------------------------------------------

            for feature_set_name, feature_columns in (
                feature_sets.items()
            ):

                print(
                    f"    Evaluation {feature_set_name} "
                    f"({len(feature_columns)} features)"
                )

                (
                    metrics,
                    probabilities,
                    predictions
                ) = evaluate_feature_set(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    feature_columns,
                    distribution_name
                )

                result = {
                    "model": feature_set_name,
                    "distribution": distribution_name,
                    "fold": fold_number,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "train_positive_rate": y_train.mean(),
                    "test_positive_rate": y_test.mean(),
                    "n_features": len(feature_columns),
                    **metrics,
                }

                fold_results.append(result)

                # ------------------------------------------------
                # Features sélectionnées
                # ------------------------------------------------

                if feature_set_name != "FULL":

                    for rank, feature in enumerate(
                        feature_columns,
                        start=1
                    ):

                        selected_feature_rows.append(
                            {
                                "distribution":
                                    distribution_name,
                                "fold":
                                    fold_number,
                                "feature_set":
                                    feature_set_name,
                                "rank":
                                    rank,
                                "feature":
                                    feature,
                                "is_suspect":
                                    feature in suspects,
                            }
                        )

                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                for local_index, (
                    probability,
                    prediction,
                    actual
                ) in enumerate(
                    zip(
                        probabilities,
                        predictions,
                        y_test.values
                    )
                ):

                    prediction_rows.append(
                        {
                            "model":
                                feature_set_name,
                            "distribution":
                                distribution_name,
                            "fold":
                                fold_number,
                            "dataset_index":
                                int(
                                    test_idx[
                                        local_index
                                    ]
                                ),
                            "probability":
                                float(
                                    probability
                                ),
                            "prediction":
                                int(
                                    prediction
                                ),
                            "actual":
                                int(
                                    actual
                                ),
                        }
                    )

    # ============================================================
    # DATAFRAMES
    # ============================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    selected_features_df = pd.DataFrame(
        selected_feature_rows
    )

    importance_all_df = pd.concat(
        all_importance_rows,
        ignore_index=True
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    summary_rows = []

    group_columns = [
        "model",
        "distribution",
    ]

    for (
        model_name,
        distribution_name
    ), group in fold_results_df.groupby(
        group_columns
    ):

        row = {
            "model":
                model_name,
            "distribution":
                distribution_name,
            "n_features":
                group["n_features"].iloc[0],
            "folds":
                len(group),

            "mean_roc_auc":
                group["roc_auc"].mean(),

            "std_roc_auc":
                group["roc_auc"].std(
                    ddof=0
                ),

            "min_roc_auc":
                group["roc_auc"].min(),

            "max_roc_auc":
                group["roc_auc"].max(),

            "mean_pr_auc":
                group["pr_auc"].mean(),

            "std_pr_auc":
                group["pr_auc"].std(
                    ddof=0
                ),

            "min_pr_auc":
                group["pr_auc"].min(),

            "max_pr_auc":
                group["pr_auc"].max(),
        }

        for percentage in [
            5,
            10,
            15,
            20
        ]:

            rate_column = (
                f"top{percentage}_goal_rate"
            )

            lift_column = (
                f"top{percentage}_lift"
            )

            rates = group[
                rate_column
            ]

            lifts = group[
                lift_column
            ]

            baseline = group[
                "test_positive_rate"
            ]

            margins = (
                rates.values
                - baseline.values
            )

            row[
                f"top{percentage}_mean_goal_rate"
            ] = rates.mean()

            row[
                f"top{percentage}_std_goal_rate"
            ] = rates.std(
                ddof=0
            )

            row[
                f"top{percentage}_min_goal_rate"
            ] = rates.min()

            row[
                f"top{percentage}_max_goal_rate"
            ] = rates.max()

            row[
                f"top{percentage}_mean_lift"
            ] = lifts.mean()

            row[
                f"top{percentage}_min_lift"
            ] = lifts.min()

            row[
                f"top{percentage}_max_lift"
            ] = lifts.max()

            row[
                f"top{percentage}_folds_above_baseline"
            ] = int(
                np.sum(
                    rates.values
                    > baseline.values
                )
            )

            row[
                f"top{percentage}_stability"
            ] = (
                np.mean(
                    rates.values
                    > baseline.values
                )
            )

            row[
                f"top{percentage}_worst_margin"
            ] = margins.min()

            row[
                f"top{percentage}_best_margin"
            ] = margins.max()

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ============================================================
    # FEATURE IMPORTANCE GLOBAL
    # ============================================================

    importance_summary = (
        importance_all_df
        .groupby("feature")
        .agg(
            mean_importance=(
                "importance_mean",
                "mean"
            ),
            std_importance=(
                "importance_mean",
                "std"
            ),
            min_importance=(
                "importance_mean",
                "min"
            ),
            max_importance=(
                "importance_mean",
                "max"
            ),
            mean_std_permutation=(
                "importance_std",
                "mean"
            ),
            positive_repetitions=(
                "importance_mean",
                lambda x: np.mean(
                    x > 0
                )
            ),
            observations=(
                "importance_mean",
                "count"
            ),
        )
        .reset_index()
    )

    importance_summary[
        "is_suspect"
    ] = importance_summary[
        "feature"
    ].isin(suspects)

    importance_summary[
        "stability"
    ] = (
        importance_summary[
            "positive_repetitions"
        ]
    )

    importance_summary[
        "robust_importance"
    ] = (
        importance_summary[
            "mean_importance"
        ]
        *
        importance_summary[
            "stability"
        ]
    )

    importance_summary = (
        importance_summary
        .sort_values(
            "robust_importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_summary[
        "global_rank"
    ] = np.arange(
        1,
        len(importance_summary) + 1
    )

    # ============================================================
    # NESTED FEATURE FREQUENCY
    # ============================================================

    if not selected_features_df.empty:

        feature_frequency = (
            selected_features_df
            .groupby("feature")
            .agg(
                selections=(
                    "feature_set",
                    "count"
                ),
                feature_sets=(
                    "feature_set",
                    "nunique"
                ),
                mean_rank=(
                    "rank",
                    "mean"
                ),
                min_rank=(
                    "rank",
                    "min"
                ),
                max_rank=(
                    "rank",
                    "max"
                ),
            )
            .reset_index()
        )

        feature_frequency[
            "selection_rate"
        ] = (
            feature_frequency[
                "selections"
            ]
            /
            selected_features_df[
                [
                    "distribution",
                    "fold"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        feature_frequency[
            "is_suspect"
        ] = feature_frequency[
            "feature"
        ].isin(suspects)

        feature_frequency = (
            feature_frequency
            .sort_values(
                [
                    "selection_rate",
                    "mean_rank",
                ],
                ascending=[
                    False,
                    True,
                ]
            )
            .reset_index(drop=True)
        )

    else:

        feature_frequency = pd.DataFrame()

    # ============================================================
    # RANKING DES CONFIGURATIONS
    # ============================================================

    ranking_df = summary_df.copy()

    ranking_df[
        "score_top10"
    ] = (
        ranking_df[
            "top10_mean_goal_rate"
        ]
        *
        ranking_df[
            "top10_stability"
        ]
    )

    ranking_df[
        "score_top15"
    ] = (
        ranking_df[
            "top15_mean_goal_rate"
        ]
        *
        ranking_df[
            "top15_stability"
        ]
    )

    ranking_df[
        "score_top20"
    ] = (
        ranking_df[
            "top20_mean_goal_rate"
        ]
        *
        ranking_df[
            "top20_stability"
        ]
    )

    ranking_df = ranking_df.sort_values(
        [
            "score_top15",
            "mean_pr_auc",
            "mean_roc_auc",
        ],
        ascending=False
    ).reset_index(drop=True)

    ranking_df[
        "rank"
    ] = np.arange(
        1,
        len(ranking_df) + 1
    )

    # ============================================================
    # SAUVEGARDE CSV
    # ============================================================

    fold_results_df.to_csv(
        OUTPUT_DIR
        / "nested_feature_selection_by_fold.csv",
        index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "nested_feature_selection_summary.csv",
        index=False
    )

    ranking_df.to_csv(
        OUTPUT_DIR
        / "nested_feature_selection_ranking.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR
        / "nested_feature_selection_predictions.csv",
        index=False
    )

    selected_features_df.to_csv(
        OUTPUT_DIR
        / "nested_selected_features.csv",
        index=False
    )

    importance_all_df.to_csv(
        OUTPUT_DIR
        / "nested_importance_by_fold.csv",
        index=False
    )

    importance_summary.to_csv(
        OUTPUT_DIR
        / "nested_importance_summary.csv",
        index=False
    )

    if not feature_frequency.empty:

        feature_frequency.to_csv(
            OUTPUT_DIR
            / "nested_feature_frequency.csv",
            index=False
        )

    # ============================================================
    # RAPPORT TEXTE
    # ============================================================

    report_path = (
        OUTPUT_DIR
        / "nested_feature_selection_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "NESTED FEATURE SELECTION REPORT\n"
        )

        report.write(
            "=" * 80
            + "\n\n"
        )

        report.write(
            f"Dataset : {DATASET_PATH}\n"
        )

        report.write(
            f"Rows : {len(df):,}\n"
        )

        report.write(
            f"Features : {len(features)}\n"
        )

        report.write(
            f"Target rate : {target.mean():.4%}\n\n"
        )

        report.write(
            "SUSPECT FEATURES\n"
        )

        report.write(
            "-" * 80
            + "\n"
        )

        for suspect in suspects:

            report.write(
                f"- {suspect}\n"
            )

        report.write(
            "\n"
        )

        report.write(
            "METHODOLOGY\n"
        )

        report.write(
            "-" * 80
            + "\n"
        )

        report.write(
            "Feature selection is performed inside "
            "each outer temporal training set.\n"
        )

        report.write(
            "The outer test set is never used for "
            "feature selection.\n"
        )

        report.write(
            "Inner split = first 80% of outer training "
            "data for model training, final 20% for "
            "permutation importance.\n"
        )

        report.write(
            "Permutation importance scoring = ROC-AUC.\n"
        )

        report.write(
            f"Permutation repetitions = "
            f"{N_PERMUTATION_REPEATS}\n"
        )

        report.write(
            "\n"
        )

        report.write(
            "CONFIGURATION RANKING\n"
        )

        report.write(
            "-" * 80
            + "\n\n"
        )

        for _, row in ranking_df.iterrows():

            report.write(
                f"{int(row['rank'])}. "
                f"{row['model']} / "
                f"{row['distribution']} | "
                f"features={int(row['n_features'])} | "
                f"ROC-AUC="
                f"{row['mean_roc_auc']:.4f} | "
                f"PR-AUC="
                f"{row['mean_pr_auc']:.4f} | "
                f"Top10="
                f"{row['top10_mean_goal_rate']:.2%} | "
                f"Top10 lift="
                f"{row['top10_mean_lift']:.2f}x | "
                f"Top15="
                f"{row['top15_mean_goal_rate']:.2%} | "
                f"Top15 lift="
                f"{row['top15_mean_lift']:.2f}x | "
                f"Top20="
                f"{row['top20_mean_goal_rate']:.2%} | "
                f"Top20 lift="
                f"{row['top20_mean_lift']:.2f}x\n"
            )

        report.write(
            "\n"
        )

        report.write(
            "TOP GLOBAL FEATURES BY NESTED IMPORTANCE\n"
        )

        report.write(
            "-" * 80
            + "\n\n"
        )

        for _, row in (
            importance_summary
            .head(30)
            .iterrows()
        ):

            suspect_label = (
                " [SUSPECT]"
                if row["is_suspect"]
                else ""
            )

            report.write(
                f"{int(row['global_rank']):02d}. "
                f"{row['feature']}"
                f"{suspect_label} | "
                f"importance="
                f"{row['mean_importance']:.6f} | "
                f"stability="
                f"{row['stability']:.2%} | "
                f"robust="
                f"{row['robust_importance']:.6f}\n"
            )

    # ============================================================
    # AFFICHAGE FINAL
    # ============================================================

    print()
    print("=" * 80)
    print("RÉSULTATS NESTED FEATURE SELECTION")
    print("=" * 80)

    display_columns = [
        "model",
        "distribution",
        "n_features",
        "mean_roc_auc",
        "mean_pr_auc",
        "top10_mean_goal_rate",
        "top10_mean_lift",
        "top15_mean_goal_rate",
        "top15_mean_lift",
        "top20_mean_goal_rate",
        "top20_mean_lift",
    ]

    display_df = ranking_df[
        display_columns
    ].copy()

    print(
        display_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Fichiers sauvegardés dans :"
    )
    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Terminé."
    )


if __name__ == "__main__":
    main()