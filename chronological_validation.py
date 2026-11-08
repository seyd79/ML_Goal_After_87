"""
chronological_validation.py

Validation chronologique du modèle ML pour prédire goal_after_87.

Objectif :
- Éviter le random split.
- Entraîner uniquement sur des matchs passés.
- Tester uniquement sur des matchs futurs.
- Comparer les distributions d'entraînement :
    73/27
    67/33
    50/50
- Mesurer :
    ROC-AUC
    PR-AUC
    Brier Score
    Log Loss
    Precision
    Recall
    F1
    taux réel de goal_after_87
    lift
    top 20% vs bottom 20%
    analyse par bins de probabilité
    intervalles de confiance Wilson

IMPORTANT :
- Le test n'est jamais resamplé.
- Le resampling est appliqué uniquement au train.
- L'imputation est faite dans le Pipeline pour éviter la fuite de données.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("ML_Dataset_PreMatch_Live87.csv")
OUTPUT_DIR = Path("outputs/chronological_validation")

RANDOM_STATE = 42

# Nombre de folds temporels.
#
# Exemple :
# Fold 1 :
#   train = période ancienne
#   test  = période suivante
#
# Fold 2 :
#   train = période encore plus large
#   test  = période suivante
#
# etc.
N_FOLDS = 5

# Pourcentage du dataset utilisé comme première période d'entraînement.
INITIAL_TRAIN_SIZE = 0.50

# Taille de chaque période de test.
TEST_SIZE = 0.10

# Nombre minimum de matchs dans un test.
MIN_TEST_SIZE = 100

# Pour éviter qu'un fold de test contienne une seule classe.
REQUIRE_BOTH_CLASSES = True

# Bins utilisés pour l'analyse des probabilités.
PROBABILITY_BINS = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]


# ============================================================
# FEATURES
# ============================================================

TARGET = "goal_after_87"
MATCH_ID = "match_id"

CATEGORICAL_FEATURES = [
    "leading_team_87",
    "is_draw_87",
]


# ============================================================
# OUTILS
# ============================================================

def print_section(title):
    """Affiche une section lisible dans le terminal."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def wilson_confidence_interval(successes, total, z=1.96):
    """
    Intervalle de confiance Wilson à 95 % pour une proportion.
    """
    if total == 0:
        return np.nan, np.nan

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
            + (z ** 2 / (4 * total ** 2))
        )
    ) / denominator

    return center - margin, center + margin


def safe_roc_auc(y_true, probabilities):
    """ROC-AUC sans crash si une seule classe est présente."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probabilities)


def safe_pr_auc(y_true, probabilities):
    """PR-AUC sans crash si une seule classe est présente."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, probabilities)


def safe_log_loss(y_true, probabilities):
    """Log loss sécurisé."""
    try:
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return log_loss(y_true, probabilities, labels=[0, 1])
    except Exception:
        return np.nan


def resample_training_data(X, y, target_positive_ratio, random_state):
    """
    Modifie uniquement la distribution du TRAIN.

    target_positive_ratio :
        0.27 = environ 73/27
        0.33 = environ 67/33
        0.50 = 50/50

    On conserve toutes les observations de la classe minoritaire
    et on sur/sous-échantillonne la classe majoritaire.

    IMPORTANT :
    aucun test n'est touché.
    """

    train_df = X.copy()
    train_df["_target_"] = y.values

    positive = train_df[train_df["_target_"] == 1]
    negative = train_df[train_df["_target_"] == 0]

    n_positive = len(positive)
    n_negative = len(negative)

    if n_positive == 0 or n_negative == 0:
        return X.copy(), y.copy()

    rng = np.random.RandomState(random_state)

    # --------------------------------------------------------
    # 50/50
    # --------------------------------------------------------
    if target_positive_ratio == 0.50:

        n = min(n_positive, n_negative)

        positive_sample = positive.sample(
            n=n,
            random_state=random_state,
            replace=False,
        )

        negative_sample = negative.sample(
            n=n,
            random_state=random_state,
            replace=False,
        )

        balanced = pd.concat(
            [positive_sample, negative_sample],
            axis=0,
        )

    # --------------------------------------------------------
    # 67/33
    # --------------------------------------------------------
    elif target_positive_ratio == 0.33:

        # On garde tous les positifs.
        # Nombre de négatifs nécessaire :
        #
        # positive / total = 0.33
        #
        # negative = positive * (0.67 / 0.33)
        target_negative = int(
            round(
                n_positive
                * ((1 - target_positive_ratio) / target_positive_ratio)
            )
        )

        target_negative = min(
            target_negative,
            n_negative,
        )

        positive_sample = positive.copy()

        negative_sample = negative.sample(
            n=target_negative,
            random_state=random_state,
            replace=False,
        )

        balanced = pd.concat(
            [positive_sample, negative_sample],
            axis=0,
        )

    # --------------------------------------------------------
    # 73/27
    # --------------------------------------------------------
    else:

        # On vise environ 27 % de positifs.
        #
        # positive / total = 0.27
        #
        # negative = positive * (0.73 / 0.27)
        target_negative = int(
            round(
                n_positive
                * ((1 - target_positive_ratio) / target_positive_ratio)
            )
        )

        target_negative = min(
            target_negative,
            n_negative,
        )

        positive_sample = positive.copy()

        negative_sample = negative.sample(
            n=target_negative,
            random_state=random_state,
            replace=False,
        )

        balanced = pd.concat(
            [positive_sample, negative_sample],
            axis=0,
        )

    balanced = balanced.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    y_balanced = balanced["_target_"].astype(int)

    X_balanced = balanced.drop(
        columns=["_target_"]
    )

    return X_balanced, y_balanced


def calculate_probability_bins(
    y_true,
    probabilities,
    fold_name,
    model_name,
):
    """
    Analyse le taux réel de buts par intervalle de probabilité.
    """

    rows = []

    baseline = np.mean(y_true)

    for lower, upper in PROBABILITY_BINS:

        if upper == 1.00:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        goals = int(
            np.sum(
                y_true[mask]
            )
        )

        rate = goals / count

        lift = (
            rate / baseline
            if baseline > 0
            else np.nan
        )

        ci_low, ci_high = wilson_confidence_interval(
            goals,
            count,
        )

        rows.append(
            {
                "model": model_name,
                "fold": fold_name,
                "probability_min": lower,
                "probability_max": upper,
                "matches": count,
                "goals_after_87": goals,
                "goal_rate": rate,
                "baseline_rate": baseline,
                "lift": lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return rows


def calculate_top_bottom_20(
    y_true,
    probabilities,
    fold_name,
    model_name,
):
    """
    Compare les 20 % de probabilités les plus élevées
    aux 20 % les plus faibles.
    """

    n = len(probabilities)

    if n < 10:
        return None

    order = np.argsort(probabilities)

    bottom_n = max(1, int(n * 0.20))

    bottom_idx = order[:bottom_n]
    top_idx = order[-bottom_n:]

    bottom_y = y_true[bottom_idx]
    top_y = y_true[top_idx]

    bottom_rate = float(np.mean(bottom_y))
    top_rate = float(np.mean(top_y))

    baseline = float(np.mean(y_true))

    bottom_lift = (
        bottom_rate / baseline
        if baseline > 0
        else np.nan
    )

    top_lift = (
        top_rate / baseline
        if baseline > 0
        else np.nan
    )

    ratio = (
        top_rate / bottom_rate
        if bottom_rate > 0
        else np.nan
    )

    return {
        "model": model_name,
        "fold": fold_name,
        "bottom_20_matches": len(bottom_y),
        "bottom_20_goal_rate": bottom_rate,
        "bottom_20_lift": bottom_lift,
        "top_20_matches": len(top_y),
        "top_20_goal_rate": top_rate,
        "top_20_lift": top_lift,
        "top_bottom_ratio": ratio,
        "baseline_rate": baseline,
    }


# ============================================================
# CHARGEMENT DATASET
# ============================================================

def load_dataset():
    print_section("CHARGEMENT DU DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset : {DATA_PATH}")
    print(f"Lignes  : {len(df):,}")
    print(f"Colonnes: {len(df.columns)}")

    if TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET}' est absente."
        )

    if MATCH_ID not in df.columns:
        print(
            f"ATTENTION : '{MATCH_ID}' absent. "
            "Le dataset sera quand même traité."
        )

    print(
        f"Taux global goal_after_87 : "
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    return df


# ============================================================
# DÉTECTION DE LA COLONNE DATE
# ============================================================

def detect_date_column(df):
    """
    Essaie de trouver automatiquement une colonne de date.

    Priorité :
    date
    match_date
    game_date
    fixture_date
    kickoff
    datetime
    timestamp
    """

    candidates = [
        "date",
        "match_date",
        "game_date",
        "fixture_date",
        "kickoff",
        "datetime",
        "timestamp",
        "match_datetime",
        "fixture_datetime",
    ]

    for column in candidates:

        if column in df.columns:

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            valid = converted.notna().sum()

            if valid >= len(df) * 0.80:

                print(
                    f"Colonne temporelle détectée : {column}"
                )

                return column, converted

    # Recherche secondaire :
    # colonnes contenant date/time
    for column in df.columns:

        lower = column.lower()

        if (
            "date" in lower
            or "datetime" in lower
            or "timestamp" in lower
            or "kickoff" in lower
        ):

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            valid = converted.notna().sum()

            if valid >= len(df) * 0.80:

                print(
                    f"Colonne temporelle détectée : {column}"
                )

                return column, converted

    return None, None


# ============================================================
# PRÉPARATION FEATURES
# ============================================================

def prepare_features(df):
    """
    Prépare X et y.
    """

    columns_to_drop = [
        TARGET,
        MATCH_ID,
    ]

    X = df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in df.columns
        ],
        errors="ignore",
    ).copy()

    y = df[TARGET].astype(int).copy()

    # --------------------------------------------------------
    # Supprimer les colonnes qui sont entièrement vides
    # --------------------------------------------------------

    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        print(
            f"Colonnes entièrement vides supprimées : "
            f"{len(empty_columns)}"
        )

        X = X.drop(
            columns=empty_columns
        )

    # --------------------------------------------------------
    # Détection automatique des colonnes catégorielles
    # --------------------------------------------------------

    categorical_columns = []

    for column in X.columns:

        if (
            X[column].dtype == "object"
            or str(X[column].dtype).startswith("category")
            or X[column].dtype == "bool"
        ):
            categorical_columns.append(column)

    # Ajouter les colonnes catégorielles connues si présentes.
    for column in CATEGORICAL_FEATURES:

        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    print(
        f"Features numériques   : {len(numerical_columns)}"
    )

    print(
        f"Features catégorielles: {len(categorical_columns)}"
    )

    return (
        X,
        y,
        numerical_columns,
        categorical_columns,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(
    numerical_columns,
    categorical_columns,
):
    """
    Pipeline sans fuite :
    imputation + encodage + RandomForest.
    """

    transformers = []

    if numerical_columns:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            )
        )

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# CRÉATION DES FOLDS TEMPORELS
# ============================================================

def create_temporal_folds(df):
    """
    Crée des folds chronologiques expansifs.

    Exemple avec 5 folds :

    Fold 1:
        TRAIN : 0 -> 50%
        TEST  : 50% -> 60%

    Fold 2:
        TRAIN : 0 -> 60%
        TEST  : 60% -> 70%

    Fold 3:
        TRAIN : 0 -> 70%
        TEST  : 70% -> 80%

    Fold 4:
        TRAIN : 0 -> 80%
        TEST  : 80% -> 90%

    Fold 5:
        TRAIN : 0 -> 90%
        TEST  : 90% -> 100%
    """

    n = len(df)

    initial_train = int(
        n * INITIAL_TRAIN_SIZE
    )

    test_size = int(
        n * TEST_SIZE
    )

    folds = []

    for fold_number in range(N_FOLDS):

        train_end = (
            initial_train
            + fold_number * test_size
        )

        test_start = train_end
        test_end = min(
            test_start + test_size,
            n,
        )

        if test_start >= n:
            break

        train_indices = np.arange(
            0,
            train_end,
        )

        test_indices = np.arange(
            test_start,
            test_end,
        )

        if len(test_indices) < MIN_TEST_SIZE:
            continue

        folds.append(
            {
                "fold": fold_number + 1,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )

    return folds


# ============================================================
# VALIDATION D'UN MODÈLE
# ============================================================

def validate_model(
    X,
    y,
    folds,
    model_name,
    target_positive_ratio,
):
    """
    Effectue toute la validation chronologique pour un modèle.
    """

    print_section(
        f"VALIDATION CHRONOLOGIQUE — {model_name}"
    )

    fold_metrics = []
    bin_rows = []
    top_bottom_rows = []
    prediction_rows = []

    for fold_info in folds:

        fold_number = fold_info["fold"]

        train_indices = fold_info["train_indices"]
        test_indices = fold_info["test_indices"]

        X_train = X.iloc[train_indices].copy()
        y_train = y.iloc[train_indices].copy()

        X_test = X.iloc[test_indices].copy()
        y_test = y.iloc[test_indices].copy()

        train_original_positive_rate = float(
            y_train.mean()
        )

        test_positive_rate = float(
            y_test.mean()
        )

        # ----------------------------------------------------
        # RESAMPLING UNIQUEMENT DU TRAIN
        # ----------------------------------------------------

        X_train_resampled, y_train_resampled = (
            resample_training_data(
                X_train,
                y_train,
                target_positive_ratio,
                RANDOM_STATE + fold_number,
            )
        )

        train_resampled_positive_rate = float(
            y_train_resampled.mean()
        )

        # ----------------------------------------------------
        # Vérification du fold
        # ----------------------------------------------------

        print()
        print(
            f"Fold {fold_number}/{len(folds)}"
        )

        print(
            f"  Train original : "
            f"{len(X_train):,} "
            f"({train_original_positive_rate * 100:.2f}% positives)"
        )

        print(
            f"  Train utilisé   : "
            f"{len(X_train_resampled):,} "
            f"({train_resampled_positive_rate * 100:.2f}% positives)"
        )

        print(
            f"  Test futur      : "
            f"{len(X_test):,} "
            f"({test_positive_rate * 100:.2f}% positives)"
        )

        # ----------------------------------------------------
        # Sécurité : test avec deux classes
        # ----------------------------------------------------

        if (
            REQUIRE_BOTH_CLASSES
            and len(np.unique(y_test)) < 2
        ):

            print(
                "  ⚠️ Fold ignoré : une seule classe dans le test."
            )

            continue

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        numerical_columns = [
            column
            for column in X_train.columns
            if column not in CATEGORICAL_FEATURES
            and (
                X_train[column].dtype != "object"
                and str(X_train[column].dtype)
                != "category"
                and X_train[column].dtype != "bool"
            )
        ]

        categorical_columns = [
            column
            for column in X_train.columns
            if column not in numerical_columns
        ]

        pipeline = build_pipeline(
            numerical_columns,
            categorical_columns,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        pipeline.fit(
            X_train_resampled,
            y_train_resampled,
        )

        # ----------------------------------------------------
        # PREDICTION FUTURE
        # ----------------------------------------------------

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        roc_auc = safe_roc_auc(
            y_test,
            probabilities,
        )

        pr_auc = safe_pr_auc(
            y_test,
            probabilities,
        )

        brier = brier_score_loss(
            y_test,
            probabilities,
        )

        logloss = safe_log_loss(
            y_test,
            probabilities,
        )

        # ----------------------------------------------------
        # TOP / BOTTOM 20%
        # ----------------------------------------------------

        top_bottom = calculate_top_bottom_20(
            y_test.to_numpy(),
            probabilities,
            f"fold_{fold_number}",
            model_name,
        )

        if top_bottom is not None:
            top_bottom_rows.append(
                top_bottom
            )

        # ----------------------------------------------------
        # BINS
        # ----------------------------------------------------

        fold_bin_rows = calculate_probability_bins(
            y_test.to_numpy(),
            probabilities,
            f"fold_{fold_number}",
            model_name,
        )

        bin_rows.extend(
            fold_bin_rows
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        for local_index, (
            original_index,
            probability,
            actual,
        ) in enumerate(
            zip(
                X_test.index,
                probabilities,
                y_test,
            )
        ):

            prediction_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "dataset_index": int(
                        original_index
                    ),
                    "probability": float(
                        probability
                    ),
                    "prediction": int(
                        probability >= 0.50
                    ),
                    "actual": int(
                        actual
                    ),
                }
            )

        # ----------------------------------------------------
        # METRICS FOLD
        # ----------------------------------------------------

        fold_metrics.append(
            {
                "model": model_name,
                "fold": fold_number,
                "train_original_size": len(X_train),
                "train_original_positive_rate": train_original_positive_rate,
                "train_resampled_size": len(
                    X_train_resampled
                ),
                "train_resampled_positive_rate": train_resampled_positive_rate,
                "test_size": len(X_test),
                "test_positive_rate": test_positive_rate,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "log_loss": logloss,
                "probability_mean": float(
                    np.mean(probabilities)
                ),
                "probability_median": float(
                    np.median(probabilities)
                ),
                "probability_min": float(
                    np.min(probabilities)
                ),
                "probability_max": float(
                    np.max(probabilities)
                ),
            }
        )

        print(
            f"  ROC-AUC : {roc_auc:.4f}"
        )

        print(
            f"  PR-AUC   : {pr_auc:.4f}"
        )

        print(
            f"  Brier    : {brier:.4f}"
        )

        print(
            f"  F1       : {f1:.4f}"
        )

        if top_bottom is not None:

            print(
                f"  Top 20%  : "
                f"{top_bottom['top_20_goal_rate'] * 100:.2f}%"
            )

            print(
                f"  Bottom20%: "
                f"{top_bottom['bottom_20_goal_rate'] * 100:.2f}%"
            )

            print(
                f"  Ratio    : "
                f"{top_bottom['top_bottom_ratio']:.2f}x"
            )

    return (
        fold_metrics,
        bin_rows,
        top_bottom_rows,
        prediction_rows,
    )


# ============================================================
# AGRÉGATION
# ============================================================

def aggregate_results(
    fold_metrics,
    top_bottom_rows,
):
    """
    Agrège les résultats sur tous les folds.
    """

    if not fold_metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "test_positive_rate",
        "probability_mean",
        "probability_median",
    ]

    summary_rows = []

    model_name = metrics_df[
        "model"
    ].iloc[0]

    row = {
        "model": model_name,
        "folds": len(metrics_df),
    }

    for column in numeric_columns:

        values = metrics_df[column].dropna()

        if len(values) == 0:
            row[f"{column}_mean"] = np.nan
            row[f"{column}_std"] = np.nan
            row[f"{column}_min"] = np.nan
            row[f"{column}_max"] = np.nan

        else:

            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(
                ddof=1
            ) if len(values) > 1 else 0

            row[f"{column}_min"] = values.min()
            row[f"{column}_max"] = values.max()

    # --------------------------------------------------------
    # Top / bottom
    # --------------------------------------------------------

    top_bottom_df = pd.DataFrame(
        top_bottom_rows
    )

    if not top_bottom_df.empty:

        row[
            "top_20_goal_rate_mean"
        ] = top_bottom_df[
            "top_20_goal_rate"
        ].mean()

        row[
            "bottom_20_goal_rate_mean"
        ] = top_bottom_df[
            "bottom_20_goal_rate"
        ].mean()

        row[
            "top_bottom_ratio_mean"
        ] = top_bottom_df[
            "top_bottom_ratio"
        ].mean()

        row[
            "top_20_lift_mean"
        ] = top_bottom_df[
            "top_20_lift"
        ].mean()

        row[
            "bottom_20_lift_mean"
        ] = top_bottom_df[
            "bottom_20_lift"
        ].mean()

        row[
            "folds_top20_better"
        ] = int(
            (
                top_bottom_df[
                    "top_20_goal_rate"
                ]
                >
                top_bottom_df[
                    "bottom_20_goal_rate"
                ]
            ).sum()
        )

        row[
            "top20_better_rate"
        ] = (
            row["folds_top20_better"]
            / len(top_bottom_df)
        )

    else:

        row[
            "top_20_goal_rate_mean"
        ] = np.nan

        row[
            "bottom_20_goal_rate_mean"
        ] = np.nan

        row[
            "top_bottom_ratio_mean"
        ] = np.nan

        row[
            "top_20_lift_mean"
        ] = np.nan

        row[
            "bottom_20_lift_mean"
        ] = np.nan

        row[
            "folds_top20_better"
        ] = 0

        row[
            "top20_better_rate"
        ] = np.nan

    summary_rows.append(row)

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# RAPPORT AUTOMATIQUE
# ============================================================

def create_report(
    summary_df,
    metrics_df,
    top_bottom_df,
):
    """
    Génère un rapport texte interprétable.
    """

    report_lines = []

    report_lines.append(
        "RAPPORT — VALIDATION CHRONOLOGIQUE"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        "Objectif : vérifier si le signal ML survit "
        "lorsque le modèle prédit uniquement des matchs futurs."
    )

    report_lines.append("")

    if summary_df.empty:

        report_lines.append(
            "Aucun résultat exploitable."
        )

        return "\n".join(
            report_lines
        )

    for _, row in summary_df.iterrows():

        model = row["model"]

        report_lines.append(
            f"MODÈLE : {model}"
        )

        report_lines.append(
            "-" * 60
        )

        report_lines.append(
            f"Folds : {int(row['folds'])}"
        )

        report_lines.append(
            f"ROC-AUC moyen : "
            f"{row['roc_auc_mean']:.4f}"
        )

        report_lines.append(
            f"ROC-AUC std   : "
            f"{row['roc_auc_std']:.4f}"
        )

        report_lines.append(
            f"PR-AUC moyen  : "
            f"{row['pr_auc_mean']:.4f}"
        )

        report_lines.append(
            f"PR-AUC std    : "
            f"{row['pr_auc_std']:.4f}"
        )

        report_lines.append(
            f"Brier moyen   : "
            f"{row['brier_score_mean']:.4f}"
        )

        report_lines.append(
            f"Log loss moyen : "
            f"{row['log_loss_mean']:.4f}"
        )

        report_lines.append(
            f"F1 moyen      : "
            f"{row['f1_mean']:.4f}"
        )

        report_lines.append("")

        report_lines.append(
            f"Top 20% goal rate : "
            f"{row['top_20_goal_rate_mean'] * 100:.2f}%"
        )

        report_lines.append(
            f"Bottom 20% goal rate : "
            f"{row['bottom_20_goal_rate_mean'] * 100:.2f}%"
        )

        report_lines.append(
            f"Top/Bottom ratio : "
            f"{row['top_bottom_ratio_mean']:.2f}x"
        )

        report_lines.append(
            f"Top 20% meilleur dans "
            f"{int(row['folds_top20_better'])}/"
            f"{int(row['folds'])} folds"
        )

        report_lines.append(
            f"Stabilité top20 : "
            f"{row['top20_better_rate'] * 100:.1f}%"
        )

        report_lines.append("")

        # ----------------------------------------------------
        # Diagnostic
        # ----------------------------------------------------

        roc = row["roc_auc_mean"]
        top20_rate = row[
            "top20_better_rate"
        ]
        ratio = row[
            "top_bottom_ratio_mean"
        ]

        if (
            roc >= 0.60
            and top20_rate >= 0.80
            and ratio >= 1.30
        ):

            assessment = (
                "SIGNAL PROMETTEUR"
            )

        elif (
            roc >= 0.55
            and top20_rate >= 0.60
            and ratio >= 1.15
        ):

            assessment = (
                "SIGNAL FAIBLE MAIS POTENTIELLEMENT STABLE"
            )

        else:

            assessment = (
                "SIGNAL FAIBLE / INSTABLE"
            )

        report_lines.append(
            f"ASSESSMENT : {assessment}"
        )

        report_lines.append("")
        report_lines.append("")

    report_lines.append(
        "INTERPRÉTATION"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        "1. La validation chronologique est plus importante "
        "que le random split pour mesurer la robustesse temporelle."
    )

    report_lines.append(
        "2. Un ROC-AUC proche de 0.50 signifie que le modèle "
        "classe peu mieux que le hasard."
    )

    report_lines.append(
        "3. Le Top 20% doit idéalement avoir un taux de buts "
        "nettement supérieur au Bottom 20% sur plusieurs folds."
    )

    report_lines.append(
        "4. Le signal doit être présent de manière répétée, "
        "et pas uniquement sur un seul fold."
    )

    report_lines.append(
        "5. Cette étape ne constitue PAS encore un backtest "
        "de paris ni une preuve de rentabilité."
    )

    report_lines.append("")

    return "\n".join(
        report_lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "CHRONOLOGICAL VALIDATION — ML GOAL AFTER 87"
    )

    print(
        "Objectif : tester le modèle sur le futur "
        "sans utiliser d'information future."
    )

    print(
        f"Dataset : {DATA_PATH}"
    )

    print(
        f"Folds : {N_FOLDS}"
    )

    print(
        f"Train initial : {INITIAL_TRAIN_SIZE * 100:.0f}%"
    )

    print(
        f"Test par fold : {TEST_SIZE * 100:.0f}%"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Détection date
    # --------------------------------------------------------

    date_column, date_values = detect_date_column(
        df
    )

    if date_values is None:

        print()
        print(
            "⚠️ Aucune colonne date détectée."
        )

        print(
            "Le dataset sera traité dans son ordre actuel."
        )

        print(
            "VÉRIFIE BIEN que le CSV est déjà trié "
            "du plus ancien au plus récent."
        )

    else:

        df = df.copy()

        df["_chronological_date_"] = date_values

        invalid_dates = df[
            "_chronological_date_"
        ].isna().sum()

        if invalid_dates > 0:

            print(
                f"Dates invalides ignorées : "
                f"{invalid_dates}"
            )

            df = df[
                df["_chronological_date_"].notna()
            ].copy()

        df = df.sort_values(
            "_chronological_date_"
        ).reset_index(
            drop=True
        )

        print(
            f"Première date : "
            f"{df['_chronological_date_'].min()}"
        )

        print(
            f"Dernière date  : "
            f"{df['_chronological_date_'].max()}"
        )

        df = df.drop(
            columns=["_chronological_date_"]
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    (
        X,
        y,
        numerical_columns,
        categorical_columns,
    ) = prepare_features(df)

    print()
    print(
        f"X shape : {X.shape}"
    )

    print(
        f"y shape : {y.shape}"
    )

    print(
        f"Goal rate : {y.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    folds = create_temporal_folds(
        df
    )

    if not folds:
        raise RuntimeError(
            "Aucun fold temporel valide n'a pu être créé."
        )

    print_section(
        "FOLDS CHRONOLOGIQUES"
    )

    for fold in folds:

        train_idx = fold[
            "train_indices"
        ]

        test_idx = fold[
            "test_indices"
        ]

        print(
            f"Fold {fold['fold']}: "
            f"train={len(train_idx):,} | "
            f"test={len(test_idx):,}"
        )

    # --------------------------------------------------------
    # Modèles
    # --------------------------------------------------------

    models = {
        "73_27": 0.27,
        "67_33": 0.33,
        "50_50": 0.50,
    }

    all_fold_metrics = []
    all_bin_rows = []
    all_top_bottom_rows = []
    all_prediction_rows = []
    all_summaries = []

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    for model_name, target_positive_ratio in models.items():

        (
            fold_metrics,
            bin_rows,
            top_bottom_rows,
            prediction_rows,
        ) = validate_model(
            X,
            y,
            folds,
            model_name,
            target_positive_ratio,
        )

        all_fold_metrics.extend(
            fold_metrics
        )

        all_bin_rows.extend(
            bin_rows
        )

        all_top_bottom_rows.extend(
            top_bottom_rows
        )

        all_prediction_rows.extend(
            prediction_rows
        )

        # Summary
        metrics_df_model = pd.DataFrame(
            fold_metrics
        )

        top_bottom_df_model = pd.DataFrame(
            top_bottom_rows
        )

        summary_model = aggregate_results(
            fold_metrics,
            top_bottom_rows,
        )

        if not summary_model.empty:

            all_summaries.append(
                summary_model
            )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_fold_metrics
    )

    bins_df = pd.DataFrame(
        all_bin_rows
    )

    top_bottom_df = pd.DataFrame(
        all_top_bottom_rows
    )

    predictions_df = pd.DataFrame(
        all_prediction_rows
    )

    if all_summaries:

        summary_df = pd.concat(
            all_summaries,
            ignore_index=True,
        )

    else:

        summary_df = pd.DataFrame()

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "chronological_fold_metrics.csv"
    )

    bins_path = (
        OUTPUT_DIR
        / "chronological_probability_bins.csv"
    )

    top_bottom_path = (
        OUTPUT_DIR
        / "chronological_top_bottom_20.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "chronological_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "chronological_summary.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    bins_df.to_csv(
        bins_path,
        index=False,
    )

    top_bottom_df.to_csv(
        top_bottom_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    report = create_report(
        summary_df,
        metrics_df,
        top_bottom_df,
    )

    report_path = (
        OUTPUT_DIR
        / "chronological_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "chronological_summary.json"
    )

    json_data = {
        "configuration": {
            "dataset": str(DATA_PATH),
            "n_folds": N_FOLDS,
            "initial_train_size": INITIAL_TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "models": models,
        },
        "summary": summary_df.to_dict(
            orient="records"
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # Résultat final terminal
    # --------------------------------------------------------

    print_section(
        "RÉSULTATS FINAUX"
    )

    if summary_df.empty:

        print(
            "Aucun résultat disponible."
        )

    else:

        display_columns = [
            "model",
            "folds",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "f1_mean",
            "brier_score_mean",
            "top_20_goal_rate_mean",
            "bottom_20_goal_rate_mean",
            "top_bottom_ratio_mean",
            "top20_better_rate",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in summary_df.columns
        ]

        display_df = summary_df[
            display_columns
        ].copy()

        print(
            display_df.to_string(
                index=False
            )
        )

    print()
    print(
        "Fichiers générés :"
    )

    print(
        f"  → {metrics_path}"
    )

    print(
        f"  → {bins_path}"
    )

    print(
        f"  → {top_bottom_path}"
    )

    print(
        f"  → {predictions_path}"
    )

    print(
        f"  → {summary_path}"
    )

    print(
        f"  → {report_path}"
    )

    print(
        f"  → {json_path}"
    )

    print()

    print(
        "IMPORTANT :"
    )

    print(
        "Si le Top 20% reste supérieur au Bottom 20% "
        "sur la majorité des folds ET que le ROC-AUC reste "
        "au-dessus de 0.50 de manière stable, le signal "
        "devient beaucoup plus crédible."
    )

    print()

    print(
        "Prochaine étape après cette validation : "
        "interpréter les résultats avant de modifier "
        "les features ou le modèle."
    )


if __name__ == "__main__":
    main()