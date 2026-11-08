from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

PERMUTATION_RANKING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "permutation_importance"
    / "permutation_importance_ranking.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "goal_after_87"
ID_COLUMN = "match_id"


# ============================================================
# MODELS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 10

TOP_PERCENTAGES = [10, 15, 20]

TRAINING_DISTRIBUTIONS = {
    "73_27": 0.27,
    "67_33": 0.33,
    "50_50": 0.50,
}


# ============================================================
# TEMPORAL FOLDS
# ============================================================

N_FOLDS = 5

TRAIN_RATIOS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

TEST_RATIO = 0.10


# ============================================================
# SUSPECT FEATURES
# ============================================================

EXPLICIT_SUSPECT_FEATURES = [
    "last_goal_minute",
    "minutes_since_last_goal",
]


# ============================================================
# UTILITAIRES
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_dataset():
    print_header("CHARGEMENT DU DATASET")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset : {DATASET_PATH}")
    print(f"Shape   : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    if TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET}' est absente du dataset."
        )

    print()
    print("Distribution cible :")

    target_counts = df[TARGET].value_counts().sort_index()
    target_rates = df[TARGET].value_counts(normalize=True).sort_index()

    for cls in target_counts.index:
        print(
            f"  Classe {cls}: "
            f"{target_counts[cls]:,} "
            f"({target_rates[cls] * 100:.2f}%)"
        )

    return df


def get_numeric_features(df):
    """
    Retourne toutes les features numériques utilisables
    en excluant la cible et match_id.
    """

    numeric_features = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_features = [
        col
        for col in numeric_features
        if col not in [TARGET, ID_COLUMN]
    ]

    return numeric_features


def detect_live87_features(features):
    """
    Détection des variables issues de l'état du match
    autour de la 87e minute.

    On cherche volontairement plusieurs familles de noms :
    - *_87
    - *_80_87
    - score_87_*
    - leading_team_87
    - is_draw_87
    - goal_diff_87
    - minutes_since_*
    - last_*_minute

    Les noms sont filtrés à partir des colonnes réellement
    présentes dans le dataset.
    """

    live_features = []

    for feature in features:

        name = feature.lower()

        is_live = False

        # Variables explicitement liées à la 87e minute
        if "_87" in name:
            is_live = True

        if "87_" in name:
            is_live = True

        # Fenêtre 80-87
        if "_80_87" in name:
            is_live = True

        # Score / état du match à 87'
        if "score_87" in name:
            is_live = True

        if "leading_team_87" in name:
            is_live = True

        if "is_draw_87" in name:
            is_live = True

        if "goal_diff_87" in name:
            is_live = True

        # Variables temporelles live
        if name.startswith("minutes_since_"):
            is_live = True

        if name.startswith("last_") and name.endswith("_minute"):
            is_live = True

        if is_live:
            live_features.append(feature)

    return sorted(set(live_features))


def load_permutation_ranking():
    print_header("CHARGEMENT DU RANKING PERMUTATION IMPORTANCE")

    if not PERMUTATION_RANKING_PATH.exists():
        raise FileNotFoundError(
            "Fichier de ranking introuvable :\n"
            f"{PERMUTATION_RANKING_PATH}\n\n"
            "Lance d'abord permutation_importance_temporal.py."
        )

    ranking = pd.read_csv(PERMUTATION_RANKING_PATH)

    print(f"Fichier : {PERMUTATION_RANKING_PATH}")
    print(f"Shape   : {ranking.shape}")

    print()
    print("Colonnes disponibles :")
    print(list(ranking.columns))

    # --------------------------------------------------------
    # Détection automatique de la colonne feature
    # --------------------------------------------------------

    possible_feature_columns = [
        "feature",
        "feature_name",
        "variable",
        "name",
    ]

    feature_column = None

    for column in possible_feature_columns:
        if column in ranking.columns:
            feature_column = column
            break

    if feature_column is None:
        raise ValueError(
            "Impossible de trouver la colonne contenant les noms "
            "de features dans permutation_importance_ranking.csv."
        )

    # --------------------------------------------------------
    # Détection automatique de la colonne de ranking
    # --------------------------------------------------------

    possible_score_columns = [
        "mean_robust_importance",
        "mean_importance_global",
        "mean_importance",
        "robust_importance",
        "importance",
    ]

    score_column = None

    for column in possible_score_columns:
        if column in ranking.columns:
            score_column = column
            break

    if score_column is None:
        raise ValueError(
            "Impossible de trouver une colonne de score "
            "dans permutation_importance_ranking.csv."
        )

    ranking = ranking.copy()

    ranking[score_column] = pd.to_numeric(
        ranking[score_column],
        errors="coerce"
    )

    ranking = ranking.dropna(
        subset=[feature_column, score_column]
    )

    ranking = ranking.sort_values(
        score_column,
        ascending=False
    )

    ranking_features = ranking[feature_column].astype(str).tolist()

    print()
    print(f"Feature column : {feature_column}")
    print(f"Score column   : {score_column}")
    print(f"Features dans ranking : {len(ranking_features)}")

    return ranking, feature_column, score_column


def balance_training_data(X, y, target_positive_ratio):
    """
    Ajuste uniquement le train.

    Le test n'est JAMAIS modifié.

    target_positive_ratio :
        0.27 -> distribution naturelle proche de 73/27
        0.33 -> environ 67/33
        0.50 -> 50/50
    """

    y_array = np.asarray(y)

    positive_indices = np.where(y_array == 1)[0]
    negative_indices = np.where(y_array == 0)[0]

    n_positive = len(positive_indices)
    n_negative = len(negative_indices)

    if n_positive == 0 or n_negative == 0:
        return X, y

    current_ratio = n_positive / len(y_array)

    # Déjà suffisamment proche
    if current_ratio >= target_positive_ratio:
        return X, y

    # Nombre de positifs nécessaire
    target_total = int(
        n_negative / (1 - target_positive_ratio)
    )

    target_positive = target_total - n_negative

    additional_needed = target_positive - n_positive

    if additional_needed <= 0:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)

    additional_indices = rng.choice(
        positive_indices,
        size=additional_needed,
        replace=True,
    )

    selected_indices = np.concatenate(
        [
            np.arange(len(y_array)),
            additional_indices,
        ]
    )

    rng.shuffle(selected_indices)

    X_balanced = X.iloc[selected_indices].reset_index(drop=True)
    y_balanced = y.iloc[selected_indices].reset_index(drop=True)

    return X_balanced, y_balanced


def build_model():
    """
    Pipeline propre :
    Imputation -> Random Forest

    L'imputation est apprise uniquement sur le train.
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def create_temporal_folds(n_rows):
    folds = []

    for fold_number, train_ratio in enumerate(
        TRAIN_RATIOS,
        start=1,
    ):

        train_end = int(n_rows * train_ratio)

        test_start = train_end
        test_end = int(
            n_rows * (train_ratio + TEST_RATIO)
        )

        if test_end > n_rows:
            test_end = n_rows

        train_indices = np.arange(
            0,
            train_end,
        )

        test_indices = np.arange(
            test_start,
            test_end,
        )

        folds.append(
            {
                "fold": fold_number,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )

    return folds


def calculate_top_percentage_metrics(
    probabilities,
    actual,
    top_percentage,
):
    """
    Calcule le taux de buts dans le Top X%.
    """

    probabilities = np.asarray(probabilities)
    actual = np.asarray(actual)

    n = len(actual)

    if n == 0:
        return {
            "top_percentage": top_percentage,
            "n_selected": 0,
            "goals": 0,
            "goal_rate": np.nan,
            "baseline_rate": np.nan,
            "lift": np.nan,
        }

    n_selected = max(
        1,
        int(np.ceil(n * top_percentage / 100)),
    )

    ranking = np.argsort(
        probabilities
    )[::-1]

    selected_indices = ranking[:n_selected]

    selected_actual = actual[selected_indices]

    goals = int(selected_actual.sum())

    goal_rate = goals / n_selected

    baseline_rate = actual.mean()

    if baseline_rate > 0:
        lift = goal_rate / baseline_rate
    else:
        lift = np.nan

    return {
        "top_percentage": top_percentage,
        "n_selected": n_selected,
        "goals": goals,
        "goal_rate": goal_rate,
        "baseline_rate": baseline_rate,
        "lift": lift,
    }


def evaluate_predictions(
    probabilities,
    actual,
):
    """
    Retourne les métriques principales.
    """

    probabilities = np.asarray(probabilities)
    actual = np.asarray(actual)

    result = {}

    try:
        result["roc_auc"] = roc_auc_score(
            actual,
            probabilities,
        )
    except Exception:
        result["roc_auc"] = np.nan

    try:
        result["pr_auc"] = average_precision_score(
            actual,
            probabilities,
        )
    except Exception:
        result["pr_auc"] = np.nan

    for percentage in TOP_PERCENTAGES:

        metrics = calculate_top_percentage_metrics(
            probabilities,
            actual,
            percentage,
        )

        prefix = f"top{percentage}"

        result[f"{prefix}_n"] = metrics["n_selected"]
        result[f"{prefix}_goals"] = metrics["goals"]
        result[f"{prefix}_goal_rate"] = metrics["goal_rate"]
        result[f"{prefix}_baseline"] = metrics["baseline_rate"]
        result[f"{prefix}_lift"] = metrics["lift"]

    return result


# ============================================================
# MAIN COMPARISON
# ============================================================

def run_comparison(
    df,
    feature_sets,
    numeric_features,
):
    print_header("COMPARAISON DES SÉLECTIONS DE FEATURES")

    X_all = df[numeric_features].copy()
    y_all = df[TARGET].astype(int).copy()

    folds = create_temporal_folds(
        len(df)
    )

    all_fold_results = []
    all_predictions = []

    print(
        f"Nombre de folds temporels : {len(folds)}"
    )

    print()

    for distribution_name, target_ratio in TRAINING_DISTRIBUTIONS.items():

        print()
        print(
            f"\n{'#' * 70}"
        )
        print(
            f"TRAINING DISTRIBUTION : {distribution_name}"
        )
        print(
            f"{'#' * 70}"
        )

        for feature_set_name, selected_features in feature_sets.items():

            print()
            print(
                f"--- {feature_set_name} "
                f"({len(selected_features)} features) ---"
            )

            # Vérification
            missing_features = [
                f
                for f in selected_features
                if f not in X_all.columns
            ]

            if missing_features:
                print(
                    f"ATTENTION : "
                    f"{len(missing_features)} features absentes."
                )

                selected_features = [
                    f
                    for f in selected_features
                    if f in X_all.columns
                ]

            if len(selected_features) == 0:
                print(
                    "Aucune feature disponible. Skip."
                )
                continue

            X = X_all[selected_features].copy()

            for fold_info in folds:

                fold_number = fold_info["fold"]

                train_indices = fold_info[
                    "train_indices"
                ]

                test_indices = fold_info[
                    "test_indices"
                ]

                X_train = X.iloc[
                    train_indices
                ].copy()

                y_train = y_all.iloc[
                    train_indices
                ].copy()

                X_test = X.iloc[
                    test_indices
                ].copy()

                y_test = y_all.iloc[
                    test_indices
                ].copy()

                # ------------------------------------------------
                # Balance uniquement sur le TRAIN
                # ------------------------------------------------

                X_train_balanced, y_train_balanced = (
                    balance_training_data(
                        X_train,
                        y_train,
                        target_ratio,
                    )
                )

                train_positive_rate = (
                    y_train_balanced.mean()
                )

                test_positive_rate = y_test.mean()

                print(
                    f"  Fold {fold_number} | "
                    f"train={len(X_train_balanced):,} "
                    f"({train_positive_rate * 100:.2f}% +) | "
                    f"test={len(X_test):,} "
                    f"({test_positive_rate * 100:.2f}% +)"
                )

                model = build_model()

                model.fit(
                    X_train_balanced,
                    y_train_balanced,
                )

                probabilities = model.predict_proba(
                    X_test
                )[:, 1]

                predictions = (
                    probabilities >= 0.50
                ).astype(int)

                metrics = evaluate_predictions(
                    probabilities,
                    y_test,
                )

                row = {
                    "model": feature_set_name,
                    "training_distribution": distribution_name,
                    "fold": fold_number,
                    "n_features": len(selected_features),
                    "n_train": len(X_train_balanced),
                    "n_test": len(X_test),
                    "train_positive_rate": train_positive_rate,
                    "test_positive_rate": test_positive_rate,
                    "threshold_0_50_positive_rate": predictions.mean(),
                    **metrics,
                }

                all_fold_results.append(row)

                # ------------------------------------------------
                # Sauvegarde des prédictions
                # ------------------------------------------------

                for local_index, (
                    dataset_index,
                    probability,
                    prediction,
                    actual,
                ) in enumerate(
                    zip(
                        test_indices,
                        probabilities,
                        predictions,
                        y_test.to_numpy(),
                    )
                ):

                    all_predictions.append(
                        {
                            "model": feature_set_name,
                            "training_distribution": distribution_name,
                            "fold": fold_number,
                            "dataset_index": int(
                                dataset_index
                            ),
                            "probability": float(
                                probability
                            ),
                            "prediction": int(
                                prediction
                            ),
                            "actual": int(
                                actual
                            ),
                        }
                    )

    return (
        pd.DataFrame(all_fold_results),
        pd.DataFrame(all_predictions),
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results):
    summary_rows = []

    group_columns = [
        "model",
        "training_distribution",
    ]

    grouped = results.groupby(
        group_columns,
        dropna=False,
    )

    for (
        model_name,
        distribution_name,
    ), group in grouped:

        row = {
            "model": model_name,
            "training_distribution": distribution_name,
            "n_features": int(
                group["n_features"].iloc[0]
            ),
            "folds": int(
                group["fold"].nunique()
            ),
        }

        # ------------------------------------------------------
        # ROC / PR
        # ------------------------------------------------------

        row["mean_roc_auc"] = group[
            "roc_auc"
        ].mean()

        row["std_roc_auc"] = group[
            "roc_auc"
        ].std()

        row["min_roc_auc"] = group[
            "roc_auc"
        ].min()

        row["mean_pr_auc"] = group[
            "pr_auc"
        ].mean()

        row["std_pr_auc"] = group[
            "pr_auc"
        ].std()

        row["min_pr_auc"] = group[
            "pr_auc"
        ].min()

        # ------------------------------------------------------
        # Top percentages
        # ------------------------------------------------------

        for percentage in TOP_PERCENTAGES:

            prefix = f"top{percentage}"

            goal_rates = group[
                f"{prefix}_goal_rate"
            ]

            baselines = group[
                f"{prefix}_baseline"
            ]

            lifts = group[
                f"{prefix}_lift"
            ]

            row[
                f"mean_{prefix}_goal_rate"
            ] = goal_rates.mean()

            row[
                f"std_{prefix}_goal_rate"
            ] = goal_rates.std()

            row[
                f"min_{prefix}_goal_rate"
            ] = goal_rates.min()

            row[
                f"mean_{prefix}_baseline"
            ] = baselines.mean()

            row[
                f"mean_{prefix}_lift"
            ] = lifts.mean()

            row[
                f"min_{prefix}_lift"
            ] = lifts.min()

            row[
                f"max_{prefix}_lift"
            ] = lifts.max()

            row[
                f"{prefix}_folds_above_baseline"
            ] = int(
                (
                    goal_rates
                    > baselines
                ).sum()
            )

            row[
                f"{prefix}_stability"
            ] = (
                row[
                    f"{prefix}_folds_above_baseline"
                ]
                / len(group)
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_ranking(summary):
    """
    Classe les modèles principalement sur :
    - Top15 goal rate
    - Top15 lift
    - Top15 stability
    - ROC-AUC

    Le Top15 est privilégié car il représente
    un compromis volume / sélectivité.
    """

    ranking = summary.copy()

    ranking["score"] = (
        ranking["mean_top15_goal_rate"] * 0.40
        + ranking["mean_top15_lift"] * 0.30
        + ranking["top15_stability"] * 0.20
        + ranking["mean_roc_auc"] * 0.10
    )

    ranking = ranking.sort_values(
        "score",
        ascending=False,
    )

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1,
    )

    return ranking


# ============================================================
# REPORT
# ============================================================

def create_report(
    df,
    feature_sets,
    summary,
    ranking,
    live87_features,
):
    report_path = (
        OUTPUT_DIR
        / "feature_selection_comparison_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FEATURE SELECTION COMPARISON REPORT\n"
        )
        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Dataset rows       : {len(df):,}\n"
        )

        f.write(
            f"Dataset columns    : {len(df.columns)}\n"
        )

        f.write(
            f"Target             : {TARGET}\n"
        )

        f.write(
            f"Numeric features   : "
            f"{len(get_numeric_features(df))}\n"
        )

        f.write(
            f"Live87 features    : "
            f"{len(live87_features)}\n\n"
        )

        f.write(
            "FEATURE SETS\n"
        )
        f.write(
            "-" * 80
            + "\n"
        )

        for name, features in feature_sets.items():

            f.write(
                f"{name}: {len(features)} features\n"
            )

        f.write("\n")

        # ------------------------------------------------------
        # Résultats
        # ------------------------------------------------------

        f.write(
            "SUMMARY\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        display_columns = [
            "model",
            "training_distribution",
            "n_features",
            "mean_roc_auc",
            "mean_pr_auc",
            "mean_top10_goal_rate",
            "mean_top10_lift",
            "top10_stability",
            "mean_top15_goal_rate",
            "mean_top15_lift",
            "top15_stability",
            "mean_top20_goal_rate",
            "mean_top20_lift",
            "top20_stability",
        ]

        available_columns = [
            c
            for c in display_columns
            if c in summary.columns
        ]

        f.write(
            summary[
                available_columns
            ].to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "RANKING\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        ranking_columns = [
            "rank",
            "model",
            "training_distribution",
            "n_features",
            "mean_roc_auc",
            "mean_pr_auc",
            "mean_top15_goal_rate",
            "mean_top15_lift",
            "top15_stability",
            "score",
        ]

        ranking_columns = [
            c
            for c in ranking_columns
            if c in ranking.columns
        ]

        f.write(
            ranking[
                ranking_columns
            ].to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ------------------------------------------------------
        # Interprétation
        # ------------------------------------------------------

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            "1. FULL utilise toutes les variables numériques "
            "disponibles.\n"
        )

        f.write(
            "2. TOP30 et TOP15 utilisent le ranking issu de "
            "la permutation importance.\n"
        )

        f.write(
            "3. LIVE87 utilise uniquement les variables "
            "détectées comme liées à l'état live.\n"
        )

        f.write(
            "4. Le test reste chronologique et n'est jamais "
            "rééquilibré.\n"
        )

        f.write(
            "5. Les probabilités sont utilisées principalement "
            "comme scores de ranking.\n"
        )

        f.write(
            "6. Cette comparaison TOP15/TOP30 est exploratoire "
            "car le ranking global a été obtenu à partir des "
            "folds temporels précédemment évalués.\n"
        )

        f.write(
            "7. Une validation finale devra effectuer la "
            "sélection des features uniquement sur les données "
            "d'entraînement de chaque fold.\n"
        )

        f.write("\n")

        f.write(
            "LIVE87 FEATURES\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for feature in live87_features:
            f.write(
                f"- {feature}\n"
            )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "FEATURE SELECTION COMPARISON"
    )

    print(
        "Objectif : comparer FULL vs TOP30 vs TOP15 vs LIVE87"
    )

    print(
        "Validation : 5 folds temporels"
    )

    print(
        "Training distributions : 73/27, 67/33, 50/50"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    numeric_features = get_numeric_features(
        df
    )

    print()
    print(
        f"Features numériques disponibles : "
        f"{len(numeric_features)}"
    )

    # --------------------------------------------------------
    # Live87
    # --------------------------------------------------------

    live87_features = detect_live87_features(
        numeric_features
    )

    print()
    print(
        f"Features LIVE87 détectées : "
        f"{len(live87_features)}"
    )

    for feature in live87_features:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Permutation ranking
    # --------------------------------------------------------

    ranking, feature_column, score_column = (
        load_permutation_ranking()
    )

    ranking_features = [
        feature
        for feature in ranking[
            feature_column
        ].astype(str)
        if feature in numeric_features
    ]

    # --------------------------------------------------------
    # TOP30 / TOP15
    # --------------------------------------------------------

    top30_features = ranking_features[:30]

    top15_features = ranking_features[:15]

    # --------------------------------------------------------
    # Feature sets
    # --------------------------------------------------------

    feature_sets = {
        "FULL": numeric_features,
        "TOP30": top30_features,
        "TOP15": top15_features,
        "LIVE87": live87_features,
    }

    print_header(
        "RÉSUMÉ DES FEATURE SETS"
    )

    for name, features in feature_sets.items():

        print(
            f"{name:<10} : "
            f"{len(features):>3} features"
        )

    # --------------------------------------------------------
    # Sauvegarde des feature sets
    # --------------------------------------------------------

    feature_sets_rows = []

    for name, features in feature_sets.items():

        for rank, feature in enumerate(
            features,
            start=1,
        ):

            feature_sets_rows.append(
                {
                    "model": name,
                    "rank": rank,
                    "feature": feature,
                }
            )

    feature_sets_df = pd.DataFrame(
        feature_sets_rows
    )

    feature_sets_df.to_csv(
        OUTPUT_DIR
        / "feature_sets.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparaison
    # --------------------------------------------------------

    results, predictions = run_comparison(
        df,
        feature_sets,
        numeric_features,
    )

    if results.empty:
        raise RuntimeError(
            "Aucun résultat généré."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        results
    )

    ranking_result = build_ranking(
        summary
    )

    # --------------------------------------------------------
    # Sauvegardes
    # --------------------------------------------------------

    fold_results_path = (
        OUTPUT_DIR
        / "feature_selection_by_fold.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_selection_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "feature_selection_ranking.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "feature_selection_predictions.csv"
    )

    results.to_csv(
        fold_results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    ranking_result.to_csv(
        ranking_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = create_report(
        df=df,
        feature_sets=feature_sets,
        summary=summary,
        ranking=ranking_result,
        live87_features=live87_features,
    )

    # --------------------------------------------------------
    # Affichage final
    # --------------------------------------------------------

    print_header(
        "RÉSULTATS FINAUX"
    )

    display_columns = [
        "model",
        "training_distribution",
        "n_features",
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_top10_goal_rate",
        "mean_top10_lift",
        "top10_stability",
        "mean_top15_goal_rate",
        "mean_top15_lift",
        "top15_stability",
        "mean_top20_goal_rate",
        "mean_top20_lift",
        "top20_stability",
    ]

    available_columns = [
        c
        for c in display_columns
        if c in summary.columns
    ]

    display_df = summary[
        available_columns
    ].copy()

    percentage_columns = [
        c
        for c in display_df.columns
        if (
            "goal_rate" in c
            or "stability" in c
        )
    ]

    for column in percentage_columns:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    numeric_columns = [
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_top10_lift",
        "mean_top15_lift",
        "mean_top20_lift",
    ]

    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = (
                display_df[column]
                .round(4)
            )

    print(
        display_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "FICHIERS GÉNÉRÉS"
    )

    print(
        f"  {fold_results_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {ranking_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"  {OUTPUT_DIR / 'feature_sets.csv'}"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "IMPORTANT :"
    )

    print(
        "TOP15/TOP30 utilisent le ranking global "
        "de permutation importance."
    )

    print(
        "Ils sont donc utiles pour comparer les "
        "architectures, mais cette étape n'est "
        "pas encore la validation finale."
    )

    print(
        "La prochaine validation devra effectuer "
        "la sélection des features uniquement "
        "sur le TRAIN de chaque fold."
    )


if __name__ == "__main__":
    main()