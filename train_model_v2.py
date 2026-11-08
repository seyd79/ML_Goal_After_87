#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model_v2.py

ENTRAÎNEMENT MODÈLE V2 — GOAL AFTER 87
======================================

OBJECTIF
--------
Entraîner un premier modèle V2 propre avec le nouveau dataset historique V2.

Ce script utilise par défaut :

    outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
    outputs/model_schema_v2/model_schema_v2.json

Il ne réutilise PAS l'ancien dataset historique de 13 800 lignes.
Il ne modifie PAS le scanner de production.
Il ne modifie PAS les collecteurs.
Il ne remplace PAS automatiquement l'ancien modèle.

PRINCIPE
--------
Le dataset CORE_READY contient uniquement les lignes qui ont toutes les features
CORE disponibles selon `model_schema_v2.json`.

Le script :

    1) charge le dataset CORE_READY
    2) charge la liste des features CORE
    3) vérifie la structure
    4) trie temporellement si `kickoff_utc` existe
    5) fait un split train/test
    6) entraîne plusieurs modèles
    7) compare ROC-AUC, PR-AUC, précision, recall, F1, Brier
    8) analyse les seuils de probabilité
    9) exporte le meilleur modèle
   10) écrit un rapport complet

SORTIES
-------
outputs/model_v2/
├── best_model_v2.joblib
├── best_model_v2_refit_all.joblib
├── model_v2_schema_used.json
├── model_v2_metrics.csv
├── model_v2_thresholds.csv
├── model_v2_predictions_test.csv
├── model_v2_feature_importance.csv
├── model_v2_manifest.json
└── model_v2_report.md

UTILISATION
-----------
Entraînement standard :

    python3 train_model_v2.py

Avec caffeinate sur Mac :

    caffeinate -dimsu python3 train_model_v2.py

Avec cotes 1X2 dans le schéma, assure-toi d'avoir construit le schéma avec :

    python3 build_model_schema_v2.py --include-main-odds

Puis :

    python3 audit_historical_dataset_v2.py
    caffeinate -dimsu python3 train_model_v2.py

IMPORTANT
---------
Ce modèle est un BASELINE V2.

Avant d'être utilisé pour envoyer des alertes réelles, il devra être validé en
shadow mode sur des matchs futurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_PATH = (
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
)

DEFAULT_SCHEMA_PATH = (
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
    / "model_schema_v2.json"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "model_v2"
)

TARGET_COLUMN = "goal_after_87"

DEFAULT_THRESHOLDS = [
    0.25,
    0.30,
    0.34,
    0.37,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

MODEL_VERSION = "model_v2_goal_after_87_2026_09_13"


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass

    return str(value).strip()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(
    value: Optional[str],
    default: Path,
) -> Path:
    if value:
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = BASE_DIR / path

        return path.resolve()

    return default.resolve()


def parse_thresholds(text: str) -> List[float]:
    if not clean_text(text):
        return list(DEFAULT_THRESHOLDS)

    thresholds = []

    for part in clean_text(text).replace(";", ",").split(","):
        part = part.strip()

        if not part:
            continue

        try:
            value = float(part)
        except Exception:
            continue

        if 0.0 <= value <= 1.0:
            thresholds.append(value)

    if not thresholds:
        return list(DEFAULT_THRESHOLDS)

    return sorted(set(thresholds))


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_auc(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    y = np.asarray(y_true)

    if len(np.unique(y)) < 2:
        return float("nan")

    try:
        return float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )
    except Exception:
        return float("nan")


def safe_pr_auc(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    y = np.asarray(y_true)

    if len(np.unique(y)) < 2:
        return float("nan")

    try:
        return float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )
    except Exception:
        return float("nan")


def safe_log_loss(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    try:
        clipped = np.clip(
            np.asarray(y_prob, dtype=float),
            1e-6,
            1.0 - 1e-6,
        )

        return float(
            log_loss(
                y_true,
                clipped,
                labels=[0, 1],
            )
        )
    except Exception:
        return float("nan")


def safe_brier(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    try:
        return float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        )
    except Exception:
        return float("nan")


def get_positive_probability(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(
                proba[:, 1],
                dtype=float,
            )

        return np.asarray(
            proba.ravel(),
            dtype=float,
        )

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)

        return 1.0 / (1.0 + np.exp(-scores))

    preds = model.predict(X)

    return np.asarray(
        preds,
        dtype=float,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# =============================================================================
# LOAD DATA / SCHEMA
# =============================================================================

def load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema introuvable : {schema_path}"
        )

    payload = json.loads(
        schema_path.read_text(
            encoding="utf-8",
        )
    )

    core_features = payload.get("core_features")

    if not isinstance(core_features, list) or not core_features:
        raise ValueError(
            "Le schema ne contient pas `core_features`."
        )

    payload["core_features"] = [
        clean_text(feature)
        for feature in core_features
        if clean_text(feature)
    ]

    if not payload["core_features"]:
        raise ValueError(
            "`core_features` est vide après nettoyage."
        )

    return payload


def load_dataset(
    dataset_path: Path,
    features: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            f"Dataset vide : {dataset_path}"
        )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Features CORE manquantes dans le dataset : "
            + ", ".join(missing_features)
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Colonne target manquante : {target_column}"
        )

    for feature in features:
        df[feature] = safe_float_series(
            df[feature]
        )

    df[target_column] = safe_float_series(
        df[target_column]
    )

    df = df[
        df[target_column].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError(
            "Aucune ligne avec target binaire 0/1."
        )

    df[target_column] = df[target_column].astype(int)

    before = len(df)

    if "match_id" in df.columns:
        df = df.drop_duplicates(
            subset=["match_id"],
            keep="last",
        ).copy()

    duplicates_removed = before - len(df)

    if duplicates_removed:
        print(
            f"Doublons match_id retirés : {duplicates_removed}"
        )

    return df


def build_training_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[list(features)].copy()

    for feature in features:
        X[feature] = safe_float_series(
            X[feature]
        )

    y = df[target_column].astype(int).copy()

    return X, y


# =============================================================================
# SPLIT
# =============================================================================

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_mode: str


def make_split(
    df: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    test_size: float,
    random_state: int,
    temporal_split: bool,
) -> SplitResult:
    work = df.copy()

    if (
        temporal_split
        and "kickoff_utc" in work.columns
    ):
        work["_kickoff_dt"] = pd.to_datetime(
            work["kickoff_utc"],
            utc=True,
            errors="coerce",
        )

        non_null = work["_kickoff_dt"].notna().sum()

        if non_null >= max(20, int(len(work) * 0.70)):
            work = work.sort_values(
                by=[
                    "_kickoff_dt",
                    "match_id" if "match_id" in work.columns else target_column,
                ]
            ).reset_index(drop=True)

            n_test = max(
                1,
                int(round(len(work) * test_size)),
            )

            n_train = len(work) - n_test

            if n_train >= 20 and n_test >= 10:
                train_df = work.iloc[:n_train].copy()
                test_df = work.iloc[n_train:].copy()

                X_train, y_train = build_training_frame(
                    train_df,
                    features,
                    target_column,
                )

                X_test, y_test = build_training_frame(
                    test_df,
                    features,
                    target_column,
                )

                return SplitResult(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    train_df=train_df,
                    test_df=test_df,
                    split_mode="temporal_by_kickoff_utc",
                )

    X, y = build_training_frame(
        work,
        features,
        target_column,
    )

    stratify = y if len(y.unique()) == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = work.loc[X_train.index].copy()
    test_df = work.loc[X_test.index].copy()

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        split_mode="stratified_random",
    )


# =============================================================================
# MODELS
# =============================================================================

def build_models(
    random_state: int,
    n_jobs: int,
    class_weight: str,
) -> Dict[str, Pipeline]:
    if class_weight == "none":
        cw = None
    else:
        cw = class_weight

    models: Dict[str, Pipeline] = {}

    models["logistic_l2_balanced"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    class_weight=cw,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["random_forest_500"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight=cw,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    models["extra_trees_700"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=700,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight=cw,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    models["gradient_boosting"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=0.035,
                    max_depth=3,
                    min_samples_leaf=8,
                    subsample=0.85,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return models


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    prob = get_positive_probability(
        model,
        X_test,
    )

    pred_050 = (
        prob >= 0.50
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred_050,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": model_name,
        "test_rows": int(len(y_test)),
        "test_positive_rate": float(np.mean(y_test)),
        "roc_auc": safe_auc(y_test, prob),
        "pr_auc": safe_pr_auc(y_test, prob),
        "brier": safe_brier(y_test, prob),
        "log_loss": safe_log_loss(y_test, prob),
        "accuracy_at_050": float(
            accuracy_score(
                y_test,
                pred_050,
            )
        ),
        "precision_at_050": float(
            precision_score(
                y_test,
                pred_050,
                zero_division=0,
            )
        ),
        "recall_at_050": float(
            recall_score(
                y_test,
                pred_050,
                zero_division=0,
            )
        ),
        "f1_at_050": float(
            f1_score(
                y_test,
                pred_050,
                zero_division=0,
            )
        ),
        "tn_at_050": int(tn),
        "fp_at_050": int(fp),
        "fn_at_050": int(fn),
        "tp_at_050": int(tp),
        "mean_probability": float(np.mean(prob)),
        "median_probability": float(np.median(prob)),
        "max_probability": float(np.max(prob)),
    }

    return metrics


def threshold_table(
    model_name: str,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    y = np.asarray(
        y_true,
        dtype=int,
    )

    prob = np.asarray(
        y_prob,
        dtype=float,
    )

    base_rate = float(
        np.mean(y)
    ) if len(y) else 0.0

    rows = []

    for threshold in thresholds:
        selected = prob >= threshold
        n_selected = int(selected.sum())

        if n_selected > 0:
            selected_y = y[selected]
            positives = int(selected_y.sum())
            negatives = int(n_selected - positives)
            precision = positives / n_selected
            coverage = n_selected / len(y)
            lift = precision / base_rate if base_rate > 0 else np.nan
        else:
            positives = 0
            negatives = 0
            precision = np.nan
            coverage = 0.0
            lift = np.nan

        rows.append({
            "model": model_name,
            "threshold": float(threshold),
            "selected": n_selected,
            "selected_rate": float(coverage),
            "positives": positives,
            "negatives": negatives,
            "precision": float(precision) if not pd.isna(precision) else np.nan,
            "late_goal_rate": float(precision) if not pd.isna(precision) else np.nan,
            "base_rate": base_rate,
            "lift_vs_base": float(lift) if not pd.isna(lift) else np.nan,
        })

    return pd.DataFrame(rows)


def choose_best_model(
    metrics_df: pd.DataFrame,
    selection_metric: str,
) -> str:
    if selection_metric not in metrics_df.columns:
        raise ValueError(
            f"Métrique de sélection inconnue : {selection_metric}"
        )

    work = metrics_df.copy()

    work[selection_metric] = pd.to_numeric(
        work[selection_metric],
        errors="coerce",
    )

    if work[selection_metric].notna().sum() == 0:
        selection_metric = "pr_auc"

    ascending_for_metric = selection_metric in (
        "brier",
        "log_loss",
    )

    work = work.sort_values(
        by=[
            selection_metric,
            "roc_auc",
        ],
        ascending=[
            ascending_for_metric,
            False,
        ],
        na_position="last",
    )

    return clean_text(
        work.iloc[0]["model"]
    )


def prediction_frame(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    cols = [
        col
        for col in [
            "match_id",
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            "kickoff_utc",
            "snapshot_minute",
            "source",
        ]
        if col in test_df.columns
    ]

    out = test_df[cols].copy()

    out["y_true"] = np.asarray(
        y_test,
        dtype=int,
    )

    out["predicted_probability"] = np.asarray(
        y_prob,
        dtype=float,
    )

    out = out.sort_values(
        by="predicted_probability",
        ascending=False,
    )

    return out


def feature_importance_frame(
    model: Pipeline,
    features: Sequence[str],
) -> pd.DataFrame:
    classifier = model.named_steps.get(
        "classifier"
    )

    if classifier is None:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_type",
            ]
        )

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(
            classifier.feature_importances_,
            dtype=float,
        )

        importance_type = "feature_importances_"

    elif hasattr(classifier, "coef_"):
        coef = np.asarray(
            classifier.coef_,
            dtype=float,
        )

        if coef.ndim == 2:
            coef = coef[0]

        values = np.abs(coef)

        importance_type = "abs_coef"

    else:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_type",
            ]
        )

    df = pd.DataFrame({
        "feature": list(features),
        "importance": values,
        "importance_type": importance_type,
    })

    df = df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return df



def dataframe_to_markdown_safe(
    df: pd.DataFrame,
) -> str:
    """
    Convertit un DataFrame en table Markdown sans dépendre de `tabulate`.

    Pandas `.to_markdown()` demande le paquet optionnel `tabulate`.
    Pour éviter que l'entraînement plante à la fin uniquement à cause du rapport,
    on génère la table Markdown manuellement.
    """

    if df is None or df.empty:
        return "_Aucune donnée._"

    work = df.copy()

    for col in work.columns:
        work[col] = work[col].apply(
            lambda value: (
                ""
                if pd.isna(value)
                else (
                    f"{value:.6f}"
                    if isinstance(value, float)
                    else str(value)
                )
            )
        )

    headers = [str(col) for col in work.columns]

    rows = [
        [str(value) for value in row]
        for row in work.values.tolist()
    ]

    widths = []

    for idx, header in enumerate(headers):
        max_width = len(header)

        for row in rows:
            if idx < len(row):
                max_width = max(
                    max_width,
                    len(row[idx]),
                )

        widths.append(max_width)

    def fmt_row(values):
        return (
            "| "
            + " | ".join(
                str(value).ljust(widths[idx])
                for idx, value in enumerate(values)
            )
            + " |"
        )

    header_line = fmt_row(headers)

    separator_line = (
        "| "
        + " | ".join(
            "-" * width
            for width in widths
        )
        + " |"
    )

    body_lines = [
        fmt_row(row)
        for row in rows
    ]

    return "\n".join(
        [header_line, separator_line] + body_lines
    )

# =============================================================================
# REPORT
# =============================================================================

def make_report(
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    features: Sequence[str],
    split: SplitResult,
    metrics_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    best_model_name: str,
    selection_metric: str,
) -> str:
    total = len(df)
    positives = int(df[TARGET_COLUMN].sum())
    negatives = int(total - positives)
    target_rate = positives / total if total else 0.0

    train_pos = int(split.y_train.sum())
    train_neg = int(len(split.y_train) - train_pos)
    test_pos = int(split.y_test.sum())
    test_neg = int(len(split.y_test) - test_pos)

    best_metrics = metrics_df[
        metrics_df["model"] == best_model_name
    ].iloc[0].to_dict()

    best_thresholds = thresholds_df[
        thresholds_df["model"] == best_model_name
    ].copy()

    lines = [
        "# Train Model V2 — Goal After 87",
        "",
        "## Objectif",
        "",
        "Entraîner un premier modèle V2 propre avec le dataset historique V2 CORE_READY.",
        "",
        "## Sources",
        "",
        f"- Dataset : `{dataset_path}`",
        f"- Schema : `{schema_path}`",
        f"- Output dir : `{output_dir}`",
        "",
        "## Dataset",
        "",
        f"- Lignes utilisées : **{total}**",
        f"- Features CORE : **{len(features)}**",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Target rate : **{target_rate:.2%}**",
        "",
        "## Split",
        "",
        f"- Mode : `{split.split_mode}`",
        f"- Train : **{len(split.y_train)}** lignes",
        f"- Train positifs/négatifs : **{train_pos}/{train_neg}**",
        f"- Test : **{len(split.y_test)}** lignes",
        f"- Test positifs/négatifs : **{test_pos}/{test_neg}**",
        "",
        "## Meilleur modèle",
        "",
        f"- Modèle sélectionné : **{best_model_name}**",
        f"- Métrique de sélection : `{selection_metric}`",
        f"- ROC-AUC : **{best_metrics.get('roc_auc', np.nan):.4f}**",
        f"- PR-AUC : **{best_metrics.get('pr_auc', np.nan):.4f}**",
        f"- Brier : **{best_metrics.get('brier', np.nan):.4f}**",
        f"- Log loss : **{best_metrics.get('log_loss', np.nan):.4f}**",
        "",
        "## Métriques par modèle",
        "",
        dataframe_to_markdown_safe(metrics_df),
        "",
        "## Seuils — meilleur modèle",
        "",
        dataframe_to_markdown_safe(best_thresholds),
        "",
        "## Important",
        "",
        "Ce modèle est un baseline V2. Il ne doit pas remplacer directement le scanner actuel.",
        "",
        "Avant toute utilisation réelle, il faut faire une validation en shadow mode sur des matchs futurs.",
    ]

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle V2 pour Goal After 87."
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset CORE_READY. Défaut: "
            "outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
        ),
    )

    parser.add_argument(
        "--schema",
        default=None,
        help=(
            "Schema model_schema_v2.json. Défaut: "
            "outputs/model_schema_v2/model_schema_v2.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Défaut: outputs/model_v2.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Taille du test set. Défaut: 0.20.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed aléatoire. Défaut: 42.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Nombre de jobs parallèles. Défaut: -1.",
    )

    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=[
            "balanced",
            "none",
        ],
        help="Gestion déséquilibre de classes. Défaut: balanced.",
    )

    parser.add_argument(
        "--selection-metric",
        default="pr_auc",
        choices=[
            "pr_auc",
            "roc_auc",
            "brier",
            "log_loss",
        ],
        help="Métrique pour choisir le meilleur modèle. Défaut: pr_auc.",
    )

    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Seuils de probabilité séparés par virgule.",
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=500,
        help="Minimum de lignes exigé pour entraîner. Défaut: 500.",
    )

    parser.add_argument(
        "--no-temporal-split",
        action="store_true",
        help="Désactive le split temporel et utilise un split stratifié aléatoire.",
    )

    args = parser.parse_args()

    dataset_path = resolve_path(
        args.dataset,
        DEFAULT_DATASET_PATH,
    )

    schema_path = resolve_path(
        args.schema,
        DEFAULT_SCHEMA_PATH,
    )

    output_dir = resolve_path(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    ensure_dir(
        output_dir
    )

    args.test_size = max(
        0.05,
        min(
            0.40,
            float(args.test_size),
        ),
    )

    thresholds = parse_thresholds(
        args.thresholds
    )

    print()
    print("=" * 100)
    print("TRAIN MODEL V2 — GOAL AFTER 87")
    print("=" * 100)
    print(f"Dataset : {dataset_path}")
    print(f"Schema : {schema_path}")
    print(f"Output : {output_dir}")

    schema = load_schema(
        schema_path
    )

    features = list(
        schema["core_features"]
    )

    print(f"Features CORE : {len(features)}")

    df = load_dataset(
        dataset_path=dataset_path,
        features=features,
        target_column=TARGET_COLUMN,
    )

    if len(df) < args.min_rows:
        raise ValueError(
            f"Dataset trop petit : {len(df)} lignes < min_rows={args.min_rows}"
        )

    positives = int(
        df[TARGET_COLUMN].sum()
    )

    negatives = int(
        len(df) - positives
    )

    target_rate = (
        positives / len(df)
        if len(df)
        else 0.0
    )

    print(f"Lignes utilisées : {len(df)}")
    print(f"Positifs : {positives}")
    print(f"Négatifs : {negatives}")
    print(f"Target rate : {target_rate:.2%}")

    missing_count = int(
        df[features].isna().sum().sum()
    )

    print(f"NaN dans features CORE : {missing_count}")

    split = make_split(
        df=df,
        features=features,
        target_column=TARGET_COLUMN,
        test_size=args.test_size,
        random_state=args.random_state,
        temporal_split=not args.no_temporal_split,
    )

    print(f"Split : {split.split_mode}")
    print(
        f"Train : {len(split.y_train)} | "
        f"Test : {len(split.y_test)}"
    )

    models = build_models(
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight=args.class_weight,
    )

    fitted_models: Dict[str, Pipeline] = {}
    metrics_rows: List[Dict[str, Any]] = []
    threshold_frames: List[pd.DataFrame] = []

    print()
    print("-" * 100)
    print("Entraînement des modèles")
    print("-" * 100)

    for model_name, model in models.items():
        print(f"Training : {model_name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                split.X_train,
                split.y_train,
            )

        fitted_models[model_name] = model

        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            X_test=split.X_test,
            y_test=split.y_test,
        )

        metrics_rows.append(
            metrics
        )

        prob = get_positive_probability(
            model,
            split.X_test,
        )

        threshold_frames.append(
            threshold_table(
                model_name=model_name,
                y_true=split.y_test,
                y_prob=prob,
                thresholds=thresholds,
            )
        )

        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Brier={metrics['brier']:.4f}"
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    best_model_name = choose_best_model(
        metrics_df,
        args.selection_metric,
    )

    best_model = fitted_models[
        best_model_name
    ]

    best_prob = get_positive_probability(
        best_model,
        split.X_test,
    )

    thresholds_df = pd.concat(
        threshold_frames,
        ignore_index=True,
    )

    predictions_df = prediction_frame(
        test_df=split.test_df,
        y_test=split.y_test,
        y_prob=best_prob,
    )

    importance_df = feature_importance_frame(
        best_model,
        features,
    )

    # Refit du meilleur modèle sur tout le dataset pour futur shadow mode.
    X_all, y_all = build_training_frame(
        df,
        features,
        TARGET_COLUMN,
    )

    best_model_refit_all = build_models(
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight=args.class_weight,
    )[best_model_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model_refit_all.fit(
            X_all,
            y_all,
        )

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------

    best_model_path = (
        output_dir
        / "best_model_v2.joblib"
    )

    best_model_refit_path = (
        output_dir
        / "best_model_v2_refit_all.joblib"
    )

    schema_used_path = (
        output_dir
        / "model_v2_schema_used.json"
    )

    metrics_path = (
        output_dir
        / "model_v2_metrics.csv"
    )

    thresholds_path = (
        output_dir
        / "model_v2_thresholds.csv"
    )

    predictions_path = (
        output_dir
        / "model_v2_predictions_test.csv"
    )

    importance_path = (
        output_dir
        / "model_v2_feature_importance.csv"
    )

    manifest_path = (
        output_dir
        / "model_v2_manifest.json"
    )

    report_path = (
        output_dir
        / "model_v2_report.md"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    joblib.dump(
        best_model_refit_all,
        best_model_refit_path,
    )

    schema_used = dict(
        schema
    )

    schema_used["model_version"] = MODEL_VERSION
    schema_used["features_used"] = features
    schema_used["target_column"] = TARGET_COLUMN
    schema_used["dataset_path"] = str(dataset_path)
    schema_used["trained_at_utc"] = now_utc_iso()

    write_json(
        schema_used_path,
        schema_used,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    thresholds_df.to_csv(
        thresholds_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    importance_df.to_csv(
        importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "model_version": MODEL_VERSION,
        "created_at_utc": now_utc_iso(),
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "output_dir": str(output_dir),
        "target_column": TARGET_COLUMN,
        "feature_count": len(features),
        "features": features,
        "rows_used": int(len(df)),
        "positives": positives,
        "negatives": negatives,
        "target_rate": target_rate,
        "split_mode": split.split_mode,
        "train_rows": int(len(split.y_train)),
        "test_rows": int(len(split.y_test)),
        "best_model_name": best_model_name,
        "selection_metric": args.selection_metric,
        "class_weight": args.class_weight,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "thresholds": thresholds,
        "artifacts": {
            "best_model_v2": str(best_model_path),
            "best_model_v2_refit_all": str(best_model_refit_path),
            "schema_used": str(schema_used_path),
            "metrics": str(metrics_path),
            "thresholds": str(thresholds_path),
            "predictions_test": str(predictions_path),
            "feature_importance": str(importance_path),
            "report": str(report_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "warning": (
            "Baseline V2. Ne pas remplacer le scanner de production sans "
            "validation shadow mode."
        ),
    }

    write_json(
        manifest_path,
        manifest,
    )

    report = make_report(
        dataset_path=dataset_path,
        schema_path=schema_path,
        output_dir=output_dir,
        df=df,
        features=features,
        split=split,
        metrics_df=metrics_df,
        thresholds_df=thresholds_df,
        best_model_name=best_model_name,
        selection_metric=args.selection_metric,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 100)
    print(f"Meilleur modèle : {best_model_name}")
    print(f"Modèle test : {best_model_path}")
    print(f"Modèle refit all : {best_model_refit_path}")
    print(f"Métriques : {metrics_path}")
    print(f"Seuils : {thresholds_path}")
    print(f"Prédictions test : {predictions_path}")
    print(f"Importance features : {importance_path}")
    print(f"Rapport : {report_path}")
    print()
    print("Ce modèle est un baseline V2. Shadow mode obligatoire avant production.")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
