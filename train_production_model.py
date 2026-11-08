# ============================================================
# train_production_model.py
# ============================================================
# Entraînement du modèle de production pour Goal After 87
#
# Modèle :
#   ExtraTreesClassifier
#   n_estimators = 500
#   max_depth = 15
#   min_samples_leaf = 3
#   max_features = "sqrt"
#
# Distribution :
#   73/27 = dataset original, sans oversampling
#
# Dataset :
#   ML_Dataset_PreMatch_Live87.csv
#
# Sorties :
#   outputs/production/
#       ET_500_d15_73_27.joblib
#       ET_500_d15_73_27_features.json
#       ET_500_d15_73_27_metadata.json
#
# ============================================================

from pathlib import Path
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

DATASET_PATH = PROJECT_DIR / "ML_Dataset_PreMatch_Live87.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "production"

MODEL_NAME = "ET_500_d15_73_27"

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"

RANDOM_STATE = 42


# ============================================================
# MODELE EXACT RETENU
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def print_separator():
    print("=" * 70)


def load_dataset():
    """Charge le dataset CSV."""

    print_separator()
    print("CHARGEMENT DU DATASET")
    print_separator()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset introuvable :\n{DATASET_PATH}\n"
        )

    print(f"Dataset : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"Nombre de lignes : {len(df):,}")
    print(f"Nombre de colonnes : {len(df.columns):,}")

    return df


def prepare_features(df):
    """
    Prépare exactement les features utilisées par le pipeline ML.

    Règles :
    - exclusion de la target
    - exclusion de match_id
    - conservation des colonnes numériques
    """

    print_separator()
    print("PREPARATION DES FEATURES")
    print_separator()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"La colonne target '{TARGET_COLUMN}' est absente du dataset."
        )

    # Colonnes candidates
    feature_columns = [
        column
        for column in df.columns
        if column not in [TARGET_COLUMN, ID_COLUMN]
    ]

    # Sélection uniquement des colonnes numériques
    numeric_features = df[feature_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    non_numeric_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    print(f"Features candidates : {len(feature_columns)}")
    print(f"Features numériques : {len(numeric_features)}")

    if non_numeric_features:
        print(
            f"Features non numériques ignorées : "
            f"{len(non_numeric_features)}"
        )

        for column in non_numeric_features:
            print(f"  - {column}")

    X = df[numeric_features].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y, numeric_features


def validate_dataset(df, X, y, feature_names):
    """Effectue les contrôles avant entraînement."""

    print_separator()
    print("VALIDATION DU DATASET")
    print_separator()

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print(f"Rows                : {len(df):,}")
    print(f"Features             : {len(feature_names)}")
    print(f"Target               : {TARGET_COLUMN}")

    # --------------------------------------------------------
    # Valeurs target
    # --------------------------------------------------------

    if y.isna().any():
        raise ValueError(
            "La target contient des valeurs manquantes."
        )

    unique_target = sorted(y.unique().tolist())

    print(f"Valeurs target       : {unique_target}")

    if not set(unique_target).issubset({0, 1}):
        raise ValueError(
            "La target doit contenir uniquement 0 et 1."
        )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    class_counts = y.value_counts().sort_index()

    total = len(y)

    class_0_count = int(class_counts.get(0, 0))
    class_1_count = int(class_counts.get(1, 0))

    class_0_rate = class_0_count / total
    class_1_rate = class_1_count / total

    print()
    print("Distribution target :")
    print(
        f"  Classe 0 : {class_0_count:,} "
        f"({class_0_rate * 100:.2f}%)"
    )
    print(
        f"  Classe 1 : {class_1_count:,} "
        f"({class_1_rate * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Match ID
    # --------------------------------------------------------

    if ID_COLUMN in df.columns:
        print()
        print(f"ID présent           : {ID_COLUMN}")

        unique_ids = df[ID_COLUMN].nunique()

        print(f"IDs uniques          : {unique_ids:,}")

        if unique_ids != len(df):
            print(
                "ATTENTION : match_id ne semble pas unique."
            )

    # --------------------------------------------------------
    # Valeurs manquantes
    # --------------------------------------------------------

    missing_total = int(X.isna().sum().sum())

    print()
    print(f"Valeurs manquantes   : {missing_total:,}")

    # Les valeurs manquantes sont autorisées car le pipeline
    # contient un SimpleImputer.
    if missing_total > 0:
        print(
            "→ Elles seront traitées par SimpleImputer."
        )
    else:
        print("→ Aucune valeur manquante.")

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    print(f"Doublons             : {duplicate_count:,}")

    if duplicate_count > 0:
        print(
            "ATTENTION : des lignes dupliquées existent."
        )

    # --------------------------------------------------------
    # Nombre de features
    # --------------------------------------------------------

    if len(feature_names) != 96:
        raise ValueError(
            f"\nATTENTION : le nombre de features est "
            f"{len(feature_names)} au lieu de 96.\n"
            "Le modèle de production attendu doit recevoir "
            "exactement 96 features."
        )

    print()
    print("✓ Validation des 96 features réussie.")


def build_pipeline():
    """Construit le pipeline exact du modèle."""

    print_separator()
    print("CONSTRUCTION DU MODELE")
    print_separator()

    classifier = ExtraTreesClassifier(
        **MODEL_PARAMS
    )

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    print("Pipeline :")
    print("  1. SimpleImputer(strategy='median')")
    print("  2. ExtraTreesClassifier")

    print()
    print("Paramètres ExtraTrees :")

    for key, value in MODEL_PARAMS.items():
        print(f"  {key}: {value}")

    return pipeline


def train_model(pipeline, X, y):
    """Entraîne le modèle sur l'ensemble du dataset."""

    print_separator()
    print("ENTRAINEMENT DU MODELE")
    print_separator()

    print(
        f"Entraînement sur {len(X):,} matchs..."
    )

    print(
        "Distribution : 73/27 originale "
        "(aucun oversampling)"
    )

    pipeline.fit(X, y)

    print()
    print("✓ Entraînement terminé.")

    return pipeline


def validate_model(pipeline, X):
    """Vérifie que le modèle entraîné fonctionne."""

    print_separator()
    print("SANITY CHECK DU MODELE")
    print_separator()

    # Vérification du nombre de features
    n_features_in = getattr(
        pipeline,
        "n_features_in_",
        None,
    )

    print(
        f"Features attendues par le pipeline : "
        f"{n_features_in}"
    )

    if n_features_in != 96:
        raise ValueError(
            f"Le pipeline attend {n_features_in} features "
            f"au lieu de 96."
        )

    # Prédiction sur la première ligne
    sample = X.iloc[[0]]

    probabilities = pipeline.predict_proba(sample)

    probability_0 = float(probabilities[0][0])
    probability_1 = float(probabilities[0][1])

    print()
    print("Test prediction :")
    print(
        f"  P(goal_after_87 = 0) : "
        f"{probability_0:.6f}"
    )
    print(
        f"  P(goal_after_87 = 1) : "
        f"{probability_1:.6f}"
    )

    if not np.isfinite(probability_0):
        raise ValueError(
            "Probabilité classe 0 invalide."
        )

    if not np.isfinite(probability_1):
        raise ValueError(
            "Probabilité classe 1 invalide."
        )

    if not (0 <= probability_0 <= 1):
        raise ValueError(
            "Probabilité classe 0 hors intervalle [0,1]."
        )

    if not (0 <= probability_1 <= 1):
        raise ValueError(
            "Probabilité classe 1 hors intervalle [0,1]."
        )

    probability_sum = probability_0 + probability_1

    if not np.isclose(probability_sum, 1.0):
        raise ValueError(
            "Les probabilités ne totalisent pas 1."
        )

    print()
    print("✓ Sanity check réussi.")


def save_features(feature_names):
    """Sauvegarde l'ordre exact des features."""

    path = OUTPUT_DIR / (
        f"{MODEL_NAME}_features.json"
    )

    data = {
        "model_name": MODEL_NAME,
        "feature_count": len(feature_names),
        "features": feature_names,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✓ Features sauvegardées : {path}")

    return path


def save_metadata(
    df,
    y,
    feature_names,
):
    """Sauvegarde les métadonnées du modèle."""

    class_counts = y.value_counts().sort_index()

    total_rows = len(df)

    class_0_count = int(
        class_counts.get(0, 0)
    )

    class_1_count = int(
        class_counts.get(1, 0)
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "ExtraTreesClassifier",
        "training_mode": "full_dataset_production",

        "project": "ML Goal After 87",

        "target": TARGET_COLUMN,

        "distribution": {
            "name": "73_27",
            "method": "original_dataset",
            "oversampling": False,
            "class_0_rate": class_0_count / total_rows,
            "class_1_rate": class_1_count / total_rows,
        },

        "dataset": {
            "path": str(DATASET_PATH),
            "rows": total_rows,
            "columns": len(df.columns),
        },

        "features": {
            "count": len(feature_names),
            "id_column_excluded": ID_COLUMN,
            "target_excluded": TARGET_COLUMN,
            "feature_names": feature_names,
        },

        "model_parameters": MODEL_PARAMS,

        "pipeline": {
            "steps": [
                "SimpleImputer(strategy='median')",
                "ExtraTreesClassifier",
            ],
            "imputer_strategy": "median",
        },

        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },

        "notes": [
            "This model is trained on the complete dataset.",
            "This artifact is intended for production deployment.",
            "The 73/27 distribution uses the original dataset without oversampling.",
            "The model expects exactly 96 numerical features in the saved order.",
            "Live feature values must represent information available at approximately 86 minutes.",
            "Columns named with _87 correspond to the historical model schema and must be populated with the live state available before the 87th minute.",
            "This artifact does not constitute evidence of real-world betting profitability.",
        ],
    }

    path = OUTPUT_DIR / (
        f"{MODEL_NAME}_metadata.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"✓ Metadata sauvegardées : {path}")

    return path


def save_model(pipeline):
    """Sauvegarde le pipeline complet avec joblib."""

    path = OUTPUT_DIR / (
        f"{MODEL_NAME}.joblib"
    )

    joblib.dump(
        pipeline,
        path,
        compress=3,
    )

    print(f"✓ Modèle sauvegardé : {path}")

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_separator()
    print("PRODUCTION MODEL TRAINING")
    print("GOAL AFTER 87")
    print_separator()
    print()

    # --------------------------------------------------------
    # Création dossier output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Output directory : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Chargement dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Préparation
    # --------------------------------------------------------

    X, y, feature_names = prepare_features(df)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_dataset(
        df=df,
        X=X,
        y=y,
        feature_names=feature_names,
    )

    # --------------------------------------------------------
    # Construction pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline()

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    pipeline = train_model(
        pipeline=pipeline,
        X=X,
        y=y,
    )

    # --------------------------------------------------------
    # Validation modèle
    # --------------------------------------------------------

    validate_model(
        pipeline=pipeline,
        X=X,
    )

    # --------------------------------------------------------
    # Sauvegarde modèle
    # --------------------------------------------------------

    print_separator()
    print("SAUVEGARDE")
    print_separator()

    model_path = save_model(
        pipeline
    )

    features_path = save_features(
        feature_names
    )

    metadata_path = save_metadata(
        df=df,
        y=y,
        feature_names=feature_names,
    )

    # --------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------

    print()
    print_separator()
    print("MODELE DE PRODUCTION TERMINE")
    print_separator()

    print()
    print("Modèle :")
    print(f"  {MODEL_NAME}")

    print()
    print("Dataset :")
    print(f"  {len(df):,} matchs")

    print()
    print("Features :")
    print(f"  {len(feature_names)}")

    print()
    print("Architecture :")
    print("  SimpleImputer → ExtraTreesClassifier")

    print()
    print("ExtraTrees :")
    print("  n_estimators   = 500")
    print("  max_depth      = 15")
    print("  min_samples_leaf = 3")
    print("  max_features   = sqrt")
    print("  random_state   = 42")

    print()
    print("Distribution :")
    print("  73/27")
    print("  aucun oversampling")

    print()
    print("Fichiers créés :")
    print(f"  1. {model_path}")
    print(f"  2. {features_path}")
    print(f"  3. {metadata_path}")

    print()
    print("✓ Le modèle est prêt pour l'intégration au scanner LIVE.")
    print()


if __name__ == "__main__":
    main()