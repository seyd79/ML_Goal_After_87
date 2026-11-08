from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "goal_after_87"

RANDOM_STATE = 42

TOP_PERCENTAGES = [5, 10, 15, 20, 25, 30]

DISTRIBUTIONS = {
    "73_27": 0.0,
    "67_33": 0.33,
    "50_50": 0.50,
}


# ============================================================
# TEMPORAL FOLDS
# ============================================================

N_FOLDS = 5
TRAIN_FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]
TEST_FRACTION = 0.10


# ============================================================
# SUSPECT FEATURES
# ============================================================

EXPLICIT_SUSPECT_FEATURES = [
    "last_goal_minute",
    "minutes_since_last_goal",
]


# ============================================================
# DATA
# ============================================================

def load_dataset():
    print("=" * 70)
    print("CHARGEMENT DU DATASET")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    if TARGET not in df.columns:
        raise ValueError(f"Target absente : {TARGET}")

    y = df[TARGET].astype(int)

    feature_columns = [
        col
        for col in df.columns
        if col not in [TARGET, "match_id"]
    ]

    X = df[feature_columns].copy()

    numeric_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    X = X[numeric_columns].copy()

    print(f"Features numériques : {len(numeric_columns)}")
    print(
        f"Target positive : "
        f"{y.mean() * 100:.2f}%"
    )

    return X, y


# ============================================================
# BALANCING
# ============================================================

def balance_training_data(X, y, target_ratio):
    """
    Sur-échantillonne uniquement la classe positive
    jusqu'au ratio demandé.

    Le test n'est jamais modifié.
    """

    if target_ratio <= 0:
        return X.copy(), y.copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    positive_idx = np.where(y.values == 1)[0]
    negative_idx = np.where(y.values == 0)[0]

    n_positive = len(positive_idx)
    n_negative = len(negative_idx)

    current_ratio = n_positive / (n_positive + n_negative)

    if current_ratio >= target_ratio:
        return X.copy(), y.copy()

    required_positive = int(
        target_ratio * n_negative / (1 - target_ratio)
    )

    additional = required_positive - n_positive

    if additional <= 0:
        return X.copy(), y.copy()

    rng = np.random.RandomState(RANDOM_STATE)

    sampled_positive = rng.choice(
        positive_idx,
        size=additional,
        replace=True,
    )

    X_extra = X.iloc[sampled_positive]
    y_extra = y.iloc[sampled_positive]

    X_balanced = pd.concat(
        [X, X_extra],
        ignore_index=True,
    )

    y_balanced = pd.concat(
        [y, y_extra],
        ignore_index=True,
    )

    shuffle_idx = rng.permutation(len(X_balanced))

    X_balanced = X_balanced.iloc[shuffle_idx].reset_index(drop=True)
    y_balanced = y_balanced.iloc[shuffle_idx].reset_index(drop=True)

    return X_balanced, y_balanced


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    models = {}

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_configs = [
        {
            "name": "RF_200_d8",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "RF_300_d10",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "RF_400_d12",
            "model": RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "RF_500_d15",
            "model": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "RF_300_unlimited",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
    ]

    for config in rf_configs:
        models[config["name"]] = config["model"]

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    et_configs = [
        {
            "name": "ET_200_d8",
            "model": ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "ET_300_d10",
            "model": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "ET_400_d12",
            "model": ExtraTreesClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "ET_500_d15",
            "model": ExtraTreesClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
    ]

    for config in et_configs:
        models[config["name"]] = config["model"]

    # --------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # --------------------------------------------------------

    hgb_configs = [
        {
            "name": "HGB_100_d5",
            "model": HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=15,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "HGB_200_d5",
            "model": HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=15,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "HGB_200_d7",
            "model": HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=31,
                learning_rate=0.05,
                max_depth=7,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "HGB_300_d7",
            "model": HistGradientBoostingClassifier(
                max_iter=300,
                max_leaf_nodes=31,
                learning_rate=0.03,
                max_depth=7,
                l2_regularization=2.0,
                random_state=RANDOM_STATE,
            ),
        },
    ]

    for config in hgb_configs:
        models[config["name"]] = config["model"]

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    gb_configs = [
        {
            "name": "GB_100_d2",
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "GB_200_d2",
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "GB_200_d3",
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
        },
    ]

    for config in gb_configs:
        models[config["name"]] = config["model"]

    return models


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# TOP PERCENTAGE ANALYSIS
# ============================================================

def calculate_top_metrics(y_test, probabilities):
    results = []

    y_test = np.asarray(y_test)
    probabilities = np.asarray(probabilities)

    baseline = y_test.mean()

    order = np.argsort(-probabilities)

    for percentage in TOP_PERCENTAGES:

        n_selected = max(
            1,
            int(len(y_test) * percentage / 100),
        )

        selected_idx = order[:n_selected]

        selected_actual = y_test[selected_idx]

        goal_rate = selected_actual.mean()

        lift = (
            goal_rate / baseline
            if baseline > 0
            else np.nan
        )

        results.append(
            {
                "top_percentage": percentage,
                "n_selected": n_selected,
                "goals": int(selected_actual.sum()),
                "goal_rate": goal_rate,
                "baseline_rate": baseline,
                "lift": lift,
            }
        )

    return results


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

def create_temporal_folds(n_samples):
    folds = []

    for fold_number, train_fraction in enumerate(
        TRAIN_FRACTIONS,
        start=1,
    ):

        train_end = int(
            n_samples * train_fraction
        )

        test_size = int(
            n_samples * TEST_FRACTION
        )

        test_start = train_end
        test_end = min(
            test_start + test_size,
            n_samples,
        )

        if test_end <= test_start:
            continue

        folds.append(
            {
                "fold": fold_number,
                "train_start": 0,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    return folds


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_all_models(X, y):

    models = build_models()

    folds = create_temporal_folds(len(X))

    print()
    print("=" * 70)
    print("MODÈLES À TESTER")
    print("=" * 70)

    for name in models:
        print(f" - {name}")

    print(f"\nNombre de modèles : {len(models)}")
    print(f"Nombre de distributions : {len(DISTRIBUTIONS)}")
    print(f"Nombre de folds : {len(folds)}")

    all_results = []
    all_predictions = []

    total_runs = (
        len(models)
        * len(DISTRIBUTIONS)
        * len(folds)
    )

    current_run = 0

    for model_name, model in models.items():

        for distribution_name, target_ratio in DISTRIBUTIONS.items():

            print()
            print("=" * 70)
            print(
                f"MODÈLE : {model_name} | "
                f"DISTRIBUTION : {distribution_name}"
            )
            print("=" * 70)

            for fold_info in folds:

                current_run += 1

                fold = fold_info["fold"]

                train_start = fold_info["train_start"]
                train_end = fold_info["train_end"]
                test_start = fold_info["test_start"]
                test_end = fold_info["test_end"]

                X_train = X.iloc[
                    train_start:train_end
                ].copy()

                y_train = y.iloc[
                    train_start:train_end
                ].copy()

                X_test = X.iloc[
                    test_start:test_end
                ].copy()

                y_test = y.iloc[
                    test_start:test_end
                ].copy()

                # ------------------------------------------------
                # BALANCING TRAIN ONLY
                # ------------------------------------------------

                X_train_balanced, y_train_balanced = (
                    balance_training_data(
                        X_train,
                        y_train,
                        target_ratio,
                    )
                )

                pipeline = build_pipeline(model)

                pipeline.fit(
                    X_train_balanced,
                    y_train_balanced,
                )

                probabilities = pipeline.predict_proba(
                    X_test
                )[:, 1]

                # ------------------------------------------------
                # GLOBAL METRICS
                # ------------------------------------------------

                roc_auc = roc_auc_score(
                    y_test,
                    probabilities,
                )

                pr_auc = average_precision_score(
                    y_test,
                    probabilities,
                )

                # ------------------------------------------------
                # TOP PERCENTAGES
                # ------------------------------------------------

                top_metrics = calculate_top_metrics(
                    y_test,
                    probabilities,
                )

                for metric in top_metrics:

                    all_results.append(
                        {
                            "model": model_name,
                            "distribution": distribution_name,
                            "fold": fold,
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                            **metric,
                        }
                    )

                # ------------------------------------------------
                # PREDICTIONS
                # ------------------------------------------------

                for local_index, (
                    probability,
                    actual,
                ) in enumerate(
                    zip(
                        probabilities,
                        y_test.values,
                    )
                ):

                    all_predictions.append(
                        {
                            "model": model_name,
                            "distribution": distribution_name,
                            "fold": fold,
                            "dataset_index": test_start
                            + local_index,
                            "probability": probability,
                            "actual": int(actual),
                        }
                    )

                print(
                    f"[{current_run}/{total_runs}] "
                    f"Fold {fold} | "
                    f"ROC-AUC={roc_auc:.4f} | "
                    f"PR-AUC={pr_auc:.4f} | "
                    f"Top10={top_metrics[1]['goal_rate']:.2%}"
                )

    results_df = pd.DataFrame(all_results)

    predictions_df = pd.DataFrame(
        all_predictions
    )

    return results_df, predictions_df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results_df):

    grouped = (
        results_df
        .groupby(
            [
                "model",
                "distribution",
                "top_percentage",
            ]
        )
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_goal_rate=("goal_rate", "mean"),
            std_goal_rate=("goal_rate", "std"),
            min_goal_rate=("goal_rate", "min"),
            max_goal_rate=("goal_rate", "max"),
            mean_baseline=("baseline_rate", "mean"),
            mean_lift=("lift", "mean"),
            min_lift=("lift", "min"),
            max_lift=("lift", "max"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )

    stability_rows = []

    for (
        model,
        distribution,
        top_percentage,
    ), group in results_df.groupby(
        [
            "model",
            "distribution",
            "top_percentage",
        ]
    ):

        above_baseline = (
            group["goal_rate"]
            > group["baseline_rate"]
        )

        stability = above_baseline.mean()

        stability_rows.append(
            {
                "model": model,
                "distribution": distribution,
                "top_percentage": top_percentage,
                "stability": stability,
                "folds_above_baseline": int(
                    above_baseline.sum()
                ),
            }
        )

    stability_df = pd.DataFrame(
        stability_rows
    )

    summary = grouped.merge(
        stability_df,
        on=[
            "model",
            "distribution",
            "top_percentage",
        ],
        how="left",
    )

    return summary


# ============================================================
# RANKING
# ============================================================

def create_ranking(summary):

    ranking = summary.copy()

    # Score principal :
    # ROC-AUC + importance du Top10 + stabilité.
    #
    # Le but n'est PAS de choisir uniquement
    # le plus gros ROC-AUC.

    ranking["ranking_score"] = (
        ranking["mean_roc_auc"]
        + (
            ranking["mean_lift"]
            * 0.05
        )
        + (
            ranking["stability"]
            * 0.02
        )
    )

    ranking = ranking.sort_values(
        [
            "ranking_score",
            "mean_roc_auc",
            "mean_lift",
        ],
        ascending=False,
    )

    return ranking


# ============================================================
# BEST CONFIGURATION
# ============================================================

def select_best_candidates(summary):

    candidates = summary[
        summary["top_percentage"].isin(
            [10, 15, 20]
        )
    ].copy()

    candidates = candidates[
        candidates["stability"] >= 0.80
    ]

    candidates = candidates[
        candidates["mean_lift"] > 1.05
    ]

    if candidates.empty:
        return pd.DataFrame()

    # On privilégie Top10/15/20,
    # mais avec une forte stabilité.

    candidates["candidate_score"] = (
        candidates["mean_roc_auc"]
        + candidates["mean_lift"] * 0.05
        + candidates["stability"] * 0.03
        + candidates["mean_goal_rate"] * 0.02
    )

    candidates = candidates.sort_values(
        "candidate_score",
        ascending=False,
    )

    return candidates


# ============================================================
# REPORT
# ============================================================

def write_report(
    summary,
    ranking,
    best_candidates,
):

    report_path = (
        OUTPUT_DIR
        / "model_optimization_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "MODEL OPTIMIZATION REPORT\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            "Objectif : identifier un modèle "
            "plus performant que le Random Forest "
            "baseline tout en conservant un signal "
            "temporel robuste.\n\n"
        )

        f.write(
            "IMPORTANT : les résultats restent "
            "des performances de validation temporelle "
            "et ne constituent pas encore un backtest "
            "de rentabilité.\n\n"
        )

        f.write(
            "TOP CONFIGURATIONS\n"
        )
        f.write("-" * 70 + "\n")

        top = ranking.head(20)

        for _, row in top.iterrows():

            f.write(
                f"{row['model']} | "
                f"{row['distribution']} | "
                f"Top {int(row['top_percentage'])}% | "
                f"ROC-AUC={row['mean_roc_auc']:.4f} | "
                f"PR-AUC={row['mean_pr_auc']:.4f} | "
                f"GoalRate={row['mean_goal_rate']:.2%} | "
                f"Lift={row['mean_lift']:.2f}x | "
                f"Stability={row['stability']:.0%}\n"
            )

        f.write("\n")
        f.write(
            "BEST CANDIDATES\n"
        )
        f.write("-" * 70 + "\n")

        if best_candidates.empty:

            f.write(
                "Aucune configuration ne respecte "
                "les critères minimaux.\n"
            )

        else:

            for _, row in best_candidates.head(10).iterrows():

                f.write(
                    f"{row['model']} | "
                    f"{row['distribution']} | "
                    f"Top {int(row['top_percentage'])}% | "
                    f"ROC-AUC={row['mean_roc_auc']:.4f} | "
                    f"PR-AUC={row['mean_pr_auc']:.4f} | "
                    f"Lift={row['mean_lift']:.2f}x | "
                    f"Stability={row['stability']:.0%}\n"
                )

        f.write("\n")
        f.write(
            "INTERPRETATION\n"
        )
        f.write("-" * 70 + "\n")

        if not best_candidates.empty:

            best = best_candidates.iloc[0]

            f.write(
                f"Meilleur candidat automatique : "
                f"{best['model']} / "
                f"{best['distribution']} / "
                f"Top {int(best['top_percentage'])}%.\n"
            )

            f.write(
                f"ROC-AUC moyen : "
                f"{best['mean_roc_auc']:.4f}\n"
            )

            f.write(
                f"PR-AUC moyen : "
                f"{best['mean_pr_auc']:.4f}\n"
            )

            f.write(
                f"Goal rate : "
                f"{best['mean_goal_rate']:.2%}\n"
            )

            f.write(
                f"Lift : "
                f"{best['mean_lift']:.2f}x\n"
            )

            f.write(
                f"Stabilité : "
                f"{best['stability']:.0%}\n"
            )

        f.write("\n")
        f.write(
            "PROCHAINE ETAPE : calibration "
            "des scores puis comparaison avec "
            "les cotes réelles à 87' afin de "
            "calculer l'edge et le ROI.\n"
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MODEL OPTIMIZATION")
    print("=" * 70)

    X, y = load_dataset()

    results_df, predictions_df = evaluate_all_models(
        X,
        y,
    )

    summary = create_summary(
        results_df
    )

    ranking = create_ranking(
        summary
    )

    best_candidates = select_best_candidates(
        summary
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "model_optimization_by_fold.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "model_optimization_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "model_optimization_ranking.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "model_optimization_predictions.csv"
    )

    candidates_path = (
        OUTPUT_DIR
        / "best_model_candidates.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    best_candidates.to_csv(
        candidates_path,
        index=False,
    )

    report_path = write_report(
        summary,
        ranking,
        best_candidates,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 15 CONFIGURATIONS")
    print("=" * 70)

    display_columns = [
        "model",
        "distribution",
        "top_percentage",
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_goal_rate",
        "mean_lift",
        "stability",
    ]

    print(
        ranking[
            display_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("MEILLEURS CANDIDATS")
    print("=" * 70)

    if best_candidates.empty:

        print(
            "Aucun candidat ne respecte "
            "les critères."
        )

    else:

        print(
            best_candidates[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("=" * 70)
    print("FICHIERS CRÉÉS")
    print("=" * 70)

    print(results_path)
    print(summary_path)
    print(ranking_path)
    print(predictions_path)
    print(candidates_path)
    print(report_path)

    print()
    print(
        "OPTIMISATION TERMINÉE."
    )


if __name__ == "__main__":
    main()