#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_model_v2_results.py

ANALYSE MODÈLE V2 — GOAL AFTER 87
=================================

OBJECTIF
--------
Analyser les résultats générés par `train_model_v2.py`.

Ce script lit par défaut :

    outputs/model_v2/model_v2_metrics.csv
    outputs/model_v2/model_v2_thresholds.csv
    outputs/model_v2/model_v2_predictions_test.csv
    outputs/model_v2/model_v2_feature_importance.csv
    outputs/model_v2/model_v2_manifest.json
    outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
    outputs/model_schema_v2/model_schema_v2.json

Il produit une analyse plus utile que le tableau brut :

    1) meilleur modèle selon PR-AUC / ROC-AUC / Brier
    2) seuils exploitables
    3) bandes de probabilité : 0.20-0.25, 0.25-0.30, etc.
    4) top prédictions les plus confiantes
    5) performance par compétition
    6) calibration approximative
    7) importance des features
    8) rapport final lisible

SORTIES
-------
outputs/model_v2/analysis/
├── analysis_summary_v2.csv
├── analysis_best_thresholds_v2.csv
├── analysis_probability_bands_v2.csv
├── analysis_competition_performance_v2.csv
├── analysis_top_predictions_v2.csv
├── analysis_calibration_v2.csv
├── analysis_feature_importance_top_v2.csv
└── analysis_model_v2_report.md

UTILISATION
-----------
Commande standard :

    python3 analyze_model_v2_results.py

Avec un autre dossier modèle :

    python3 analyze_model_v2_results.py --model-dir outputs/model_v2

Avec un modèle précis :

    python3 analyze_model_v2_results.py --model logistic_l2_balanced

IMPORTANT
---------
Cette analyse ne valide pas encore le modèle pour la production.
Elle sert à décider :
    - s'il y a un signal exploitable ;
    - quel seuil tester en shadow mode ;
    - quelles compétitions sont bonnes ou mauvaises ;
    - quelles features dominent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "model_v2"
)

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

TARGET_COLUMN = "goal_after_87"

DEFAULT_BANDS = [
    0.00,
    0.20,
    0.25,
    0.30,
    0.34,
    0.37,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    1.01,
]


# =============================================================================
# BASIC HELPERS
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


def safe_float(value: Any, default: float = np.nan) -> float:
    text = clean_text(value)

    if not text:
        return default

    text = (
        text
        .replace("*", "")
        .replace("%", "")
        .replace(",", ".")
        .strip()
    )

    try:
        number = float(text)
    except Exception:
        return default

    if not math.isfinite(number):
        return default

    return number


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(value, np.nan)

    if pd.isna(number):
        return default

    return int(number)


def safe_rate(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return np.nan

    return float(numerator / denominator)


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


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        if isinstance(payload, Mapping):
            return dict(payload)

    except Exception:
        pass

    return {}


def dataframe_to_markdown_safe(
    df: pd.DataFrame,
    max_rows: int = 50,
) -> str:
    """
    Table Markdown sans dépendance `tabulate`.
    """

    if df is None or df.empty:
        return "_Aucune donnée._"

    work = df.head(max_rows).copy()

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

    def fmt_row(values: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(
                str(value).ljust(widths[idx])
                for idx, value in enumerate(values)
            )
            + " |"
        )

    separator = (
        "| "
        + " | ".join("-" * width for width in widths)
        + " |"
    )

    lines = [
        fmt_row(headers),
        separator,
    ]

    lines.extend(
        fmt_row(row)
        for row in rows
    )

    if len(df) > max_rows:
        lines.append(
            f"\n_Seulement {max_rows}/{len(df)} lignes affichées._"
        )

    return "\n".join(lines)


# =============================================================================
# LOAD FILES
# =============================================================================

def load_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} introuvable : {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            f"{label} vide : {path}"
        )

    return df


def load_inputs(
    model_dir: Path,
    dataset_path: Path,
    schema_path: Path,
) -> Dict[str, Any]:
    metrics_path = model_dir / "model_v2_metrics.csv"
    thresholds_path = model_dir / "model_v2_thresholds.csv"
    predictions_path = model_dir / "model_v2_predictions_test.csv"
    importance_path = model_dir / "model_v2_feature_importance.csv"
    manifest_path = model_dir / "model_v2_manifest.json"
    report_path = model_dir / "model_v2_report.md"

    inputs = {
        "metrics_path": metrics_path,
        "thresholds_path": thresholds_path,
        "predictions_path": predictions_path,
        "importance_path": importance_path,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "dataset_path": dataset_path,
        "schema_path": schema_path,
        "metrics": load_csv_required(metrics_path, "Metrics"),
        "thresholds": load_csv_required(thresholds_path, "Thresholds"),
        "predictions": load_csv_required(predictions_path, "Predictions test"),
        "importance": (
            pd.read_csv(importance_path, low_memory=False)
            if importance_path.exists()
            else pd.DataFrame()
        ),
        "manifest": read_json_if_exists(manifest_path),
        "schema": read_json_if_exists(schema_path),
        "dataset": (
            pd.read_csv(dataset_path, low_memory=False)
            if dataset_path.exists()
            else pd.DataFrame()
        ),
    }

    return inputs


# =============================================================================
# MODEL SELECTION / SUMMARY
# =============================================================================

def clean_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    return out


def select_best_model(
    metrics_df: pd.DataFrame,
    requested_model: Optional[str],
) -> str:
    metrics = clean_numeric_columns(
        metrics_df,
        [
            "pr_auc",
            "roc_auc",
            "brier",
            "log_loss",
        ],
    )

    if requested_model:
        if requested_model not in set(metrics["model"].astype(str)):
            raise ValueError(
                f"Modèle demandé introuvable : {requested_model}"
            )

        return requested_model

    work = metrics.copy()

    if "pr_auc" not in work.columns:
        return clean_text(work.iloc[0]["model"])

    work = work.sort_values(
        by=[
            "pr_auc",
            "roc_auc",
            "brier",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    )

    return clean_text(
        work.iloc[0]["model"]
    )


def build_summary(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    best_model: str,
) -> pd.DataFrame:
    metrics = clean_numeric_columns(
        metrics_df,
        [
            "test_rows",
            "test_positive_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "mean_probability",
            "median_probability",
            "max_probability",
        ],
    )

    best_row = (
        metrics[
            metrics["model"].astype(str) == str(best_model)
        ]
        .iloc[0]
        .to_dict()
    )

    total_dataset = len(dataset_df) if dataset_df is not None else 0

    dataset_positives = 0
    dataset_negatives = 0
    dataset_rate = np.nan

    if (
        dataset_df is not None
        and not dataset_df.empty
        and TARGET_COLUMN in dataset_df.columns
    ):
        y_all = pd.to_numeric(
            dataset_df[TARGET_COLUMN],
            errors="coerce",
        )

        y_all = y_all[
            y_all.isin([0, 1])
        ]

        dataset_positives = int(
            (y_all == 1).sum()
        )

        dataset_negatives = int(
            (y_all == 0).sum()
        )

        dataset_rate = safe_rate(
            dataset_positives,
            dataset_positives + dataset_negatives,
        )

    test_rows = int(
        safe_float(
            best_row.get("test_rows"),
            len(predictions_df),
        )
    )

    test_base_rate = safe_float(
        best_row.get("test_positive_rate"),
        np.nan,
    )

    rows = [
        {
            "metric": "best_model",
            "value": best_model,
        },
        {
            "metric": "dataset_rows",
            "value": total_dataset,
        },
        {
            "metric": "dataset_positives",
            "value": dataset_positives,
        },
        {
            "metric": "dataset_negatives",
            "value": dataset_negatives,
        },
        {
            "metric": "dataset_target_rate",
            "value": dataset_rate,
        },
        {
            "metric": "test_rows",
            "value": test_rows,
        },
        {
            "metric": "test_base_rate",
            "value": test_base_rate,
        },
        {
            "metric": "best_roc_auc",
            "value": best_row.get("roc_auc", np.nan),
        },
        {
            "metric": "best_pr_auc",
            "value": best_row.get("pr_auc", np.nan),
        },
        {
            "metric": "best_brier",
            "value": best_row.get("brier", np.nan),
        },
        {
            "metric": "best_log_loss",
            "value": best_row.get("log_loss", np.nan),
        },
        {
            "metric": "mean_probability",
            "value": best_row.get("mean_probability", np.nan),
        },
        {
            "metric": "median_probability",
            "value": best_row.get("median_probability", np.nan),
        },
        {
            "metric": "max_probability",
            "value": best_row.get("max_probability", np.nan),
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def analyze_thresholds(
    thresholds_df: pd.DataFrame,
    min_selected: int,
) -> pd.DataFrame:
    work = clean_numeric_columns(
        thresholds_df,
        [
            "threshold",
            "selected",
            "selected_rate",
            "positives",
            "negatives",
            "precision",
            "late_goal_rate",
            "base_rate",
            "lift_vs_base",
        ],
    )

    work["valid_sample"] = (
        work["selected"] >= min_selected
    ).astype(int)

    work["score"] = (
        work["lift_vs_base"].fillna(0.0)
        * np.log1p(work["selected"].fillna(0.0))
    )

    work["risk_note"] = np.where(
        work["selected"] < 20,
        "sample_too_small",
        np.where(
            work["selected"] < min_selected,
            "sample_small",
            "sample_ok",
        ),
    )

    work = work.sort_values(
        by=[
            "model",
            "valid_sample",
            "lift_vs_base",
            "precision",
            "selected",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    )

    return work


def best_thresholds_by_model(
    thresholds_df: pd.DataFrame,
    min_selected: int,
) -> pd.DataFrame:
    work = analyze_thresholds(
        thresholds_df,
        min_selected=min_selected,
    )

    rows = []

    for model, group in work.groupby("model"):
        usable = group[
            group["selected"] >= min_selected
        ].copy()

        if usable.empty:
            usable = group.copy()

        usable = usable.sort_values(
            by=[
                "lift_vs_base",
                "precision",
                "selected",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )

        row = usable.iloc[0].to_dict()
        row["selection_reason"] = (
            f"best_lift_with_selected_gte_{min_selected}"
            if int(row.get("selected", 0)) >= min_selected
            else "best_available_but_sample_small"
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=[
                "valid_sample",
                "lift_vs_base",
                "precision",
                "selected",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            na_position="last",
        )

    return result


# =============================================================================
# PROBABILITY BANDS / CALIBRATION
# =============================================================================

def build_probability_bands(
    predictions_df: pd.DataFrame,
    bands: Sequence[float],
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    if "predicted_probability" not in predictions_df.columns:
        raise ValueError(
            "Colonne `predicted_probability` manquante dans predictions."
        )

    if "y_true" not in predictions_df.columns:
        raise ValueError(
            "Colonne `y_true` manquante dans predictions."
        )

    work = predictions_df.copy()
    work["predicted_probability"] = pd.to_numeric(
        work["predicted_probability"],
        errors="coerce",
    )
    work["y_true"] = pd.to_numeric(
        work["y_true"],
        errors="coerce",
    )

    work = work[
        work["predicted_probability"].notna()
        & work["y_true"].isin([0, 1])
    ].copy()

    if work.empty:
        return pd.DataFrame()

    base_rate = float(
        work["y_true"].mean()
    )

    edges = sorted(set(float(x) for x in bands))

    if edges[0] > 0.0:
        edges = [0.0] + edges

    if edges[-1] <= 1.0:
        edges = edges + [1.01]

    rows = []

    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.01:
            mask = work["predicted_probability"] >= low
            label = f"{low:.2f}+"
        else:
            mask = (
                (work["predicted_probability"] >= low)
                & (work["predicted_probability"] < high)
            )
            label = f"{low:.2f}-{high:.2f}"

        chunk = work[mask]

        selected = len(chunk)

        if selected:
            positives = int(
                chunk["y_true"].sum()
            )
            negatives = int(
                selected - positives
            )
            rate = positives / selected
            avg_prob = float(
                chunk["predicted_probability"].mean()
            )
            min_prob = float(
                chunk["predicted_probability"].min()
            )
            max_prob = float(
                chunk["predicted_probability"].max()
            )
        else:
            positives = 0
            negatives = 0
            rate = np.nan
            avg_prob = np.nan
            min_prob = np.nan
            max_prob = np.nan

        rows.append({
            "band": label,
            "low": low,
            "high": high if high < 1.01 else np.nan,
            "selected": selected,
            "selected_rate": safe_rate(selected, len(work)),
            "positives": positives,
            "negatives": negatives,
            "late_goal_rate": rate,
            "base_rate": base_rate,
            "lift_vs_base": safe_rate(rate, base_rate) if not pd.isna(rate) else np.nan,
            "avg_predicted_probability": avg_prob,
            "min_probability": min_prob,
            "max_probability": max_prob,
        })

    return pd.DataFrame(rows)


def build_calibration_table(
    predictions_df: pd.DataFrame,
    n_bins: int,
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    work = predictions_df.copy()

    work["predicted_probability"] = pd.to_numeric(
        work["predicted_probability"],
        errors="coerce",
    )

    work["y_true"] = pd.to_numeric(
        work["y_true"],
        errors="coerce",
    )

    work = work[
        work["predicted_probability"].notna()
        & work["y_true"].isin([0, 1])
    ].copy()

    if work.empty:
        return pd.DataFrame()

    # qcut peut échouer si beaucoup de probabilités identiques.
    try:
        work["_bin"] = pd.qcut(
            work["predicted_probability"],
            q=min(n_bins, len(work)),
            duplicates="drop",
        )
    except Exception:
        work["_bin"] = pd.cut(
            work["predicted_probability"],
            bins=min(n_bins, len(work)),
            include_lowest=True,
        )

    rows = []

    for idx, (bin_label, group) in enumerate(work.groupby("_bin", observed=False), start=1):
        selected = len(group)

        if selected == 0:
            continue

        rows.append({
            "bin": idx,
            "bin_range": str(bin_label),
            "selected": selected,
            "avg_predicted_probability": float(group["predicted_probability"].mean()),
            "observed_late_goal_rate": float(group["y_true"].mean()),
            "positives": int(group["y_true"].sum()),
            "negatives": int(selected - group["y_true"].sum()),
            "calibration_error": float(
                group["predicted_probability"].mean()
                - group["y_true"].mean()
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# COMPETITION ANALYSIS
# =============================================================================

def build_competition_performance(
    predictions_df: pd.DataFrame,
    min_matches: int,
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    if "competition_id" not in predictions_df.columns:
        return pd.DataFrame()

    work = predictions_df.copy()

    work["predicted_probability"] = pd.to_numeric(
        work["predicted_probability"],
        errors="coerce",
    )

    work["y_true"] = pd.to_numeric(
        work["y_true"],
        errors="coerce",
    )

    work = work[
        work["predicted_probability"].notna()
        & work["y_true"].isin([0, 1])
    ].copy()

    if work.empty:
        return pd.DataFrame()

    rows = []

    global_base = float(
        work["y_true"].mean()
    )

    for competition_id, group in work.groupby("competition_id"):
        n = len(group)

        if n < min_matches:
            continue

        positives = int(group["y_true"].sum())
        negatives = int(n - positives)
        rate = positives / n if n else np.nan

        rows.append({
            "competition_id": competition_id,
            "matches_test": n,
            "positives": positives,
            "negatives": negatives,
            "late_goal_rate": rate,
            "global_base_rate": global_base,
            "lift_vs_global": safe_rate(rate, global_base),
            "avg_predicted_probability": float(group["predicted_probability"].mean()),
            "max_predicted_probability": float(group["predicted_probability"].max()),
            "high_prob_050_count": int((group["predicted_probability"] >= 0.50).sum()),
            "high_prob_050_positive": int(group.loc[group["predicted_probability"] >= 0.50, "y_true"].sum()),
            "high_prob_050_precision": safe_rate(
                int(group.loc[group["predicted_probability"] >= 0.50, "y_true"].sum()),
                int((group["predicted_probability"] >= 0.50).sum()),
            ),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=[
                "matches_test",
                "lift_vs_global",
                "late_goal_rate",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )

    return result


# =============================================================================
# TOP PREDICTIONS / FEATURE IMPORTANCE
# =============================================================================

def build_top_predictions(
    predictions_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    work = predictions_df.copy()

    work["predicted_probability"] = pd.to_numeric(
        work["predicted_probability"],
        errors="coerce",
    )

    work["y_true"] = pd.to_numeric(
        work["y_true"],
        errors="coerce",
    )

    work = work[
        work["predicted_probability"].notna()
    ].copy()

    keep_cols = [
        col
        for col in [
            "match_id",
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            "kickoff_utc",
            "y_true",
            "predicted_probability",
        ]
        if col in work.columns
    ]

    return (
        work[keep_cols]
        .sort_values(
            by="predicted_probability",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def build_feature_importance_top(
    importance_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if importance_df is None or importance_df.empty:
        return pd.DataFrame()

    if "importance" not in importance_df.columns:
        return pd.DataFrame()

    work = importance_df.copy()

    work["importance"] = pd.to_numeric(
        work["importance"],
        errors="coerce",
    )

    work = work.sort_values(
        by="importance",
        ascending=False,
        na_position="last",
    )

    return work.head(top_n).reset_index(drop=True)


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def build_recommendations(
    summary_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    min_selected: int,
) -> List[str]:
    summary = {
        clean_text(row["metric"]): row["value"]
        for _, row in summary_df.iterrows()
    }

    pr_auc = safe_float(
        summary.get("best_pr_auc"),
        np.nan,
    )

    test_base = safe_float(
        summary.get("test_base_rate"),
        np.nan,
    )

    recommendations = []

    if not pd.isna(pr_auc) and not pd.isna(test_base):
        if pr_auc <= test_base + 0.03:
            recommendations.append(
                "Signal faible : PR-AUC très proche du taux de base. Ne pas utiliser en production."
            )
        elif pr_auc <= test_base + 0.08:
            recommendations.append(
                "Signal léger : modèle utile comme baseline, mais shadow mode obligatoire."
            )
        else:
            recommendations.append(
                "Signal intéressant : vérifier stabilité par compétition et lancer shadow mode."
            )

    if not best_thresholds_df.empty:
        best = best_thresholds_df.iloc[0]

        selected = safe_int(
            best.get("selected"),
            0,
        )

        precision = safe_float(
            best.get("precision"),
            np.nan,
        )

        lift = safe_float(
            best.get("lift_vs_base"),
            np.nan,
        )

        threshold = safe_float(
            best.get("threshold"),
            np.nan,
        )

        model = clean_text(
            best.get("model")
        )

        if selected < min_selected:
            recommendations.append(
                f"Le meilleur seuil apparent ({model} seuil {threshold:.2f}) a seulement {selected} matchs : échantillon trop petit."
            )
        else:
            recommendations.append(
                f"Seuil candidat shadow mode : {model} seuil {threshold:.2f}, précision {precision:.2%}, lift {lift:.2f}, sélection {selected} matchs."
            )

    if not bands_df.empty:
        strong_bands = bands_df[
            (bands_df["selected"] >= min_selected)
            & (bands_df["lift_vs_base"] >= 1.20)
        ].copy()

        if strong_bands.empty:
            recommendations.append(
                "Aucune bande de probabilité avec assez de matchs ne dépasse clairement lift 1.20."
            )
        else:
            best_band = strong_bands.sort_values(
                by=[
                    "lift_vs_base",
                    "late_goal_rate",
                    "selected",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            ).iloc[0]

            recommendations.append(
                f"Bande intéressante : {best_band['band']} avec {int(best_band['selected'])} matchs, taux {best_band['late_goal_rate']:.2%}, lift {best_band['lift_vs_base']:.2f}."
            )

    recommendations.append(
        "Ne pas remplacer le scanner actuel. Utiliser le modèle V2 uniquement en shadow mode jusqu'à validation prospective."
    )

    return recommendations


# =============================================================================
# REPORT
# =============================================================================

def make_report(
    inputs: Mapping[str, Any],
    output_dir: Path,
    selected_model: str,
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    competition_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    top_predictions_df: pd.DataFrame,
    importance_top_df: pd.DataFrame,
    recommendations: Sequence[str],
) -> str:
    lines = [
        "# Analyse Model V2 — Goal After 87",
        "",
        f"Généré le : `{now_utc_iso()}`",
        "",
        "## Sources",
        "",
        f"- Metrics : `{inputs['metrics_path']}`",
        f"- Thresholds : `{inputs['thresholds_path']}`",
        f"- Predictions test : `{inputs['predictions_path']}`",
        f"- Feature importance : `{inputs['importance_path']}`",
        f"- Dataset : `{inputs['dataset_path']}`",
        f"- Schema : `{inputs['schema_path']}`",
        "",
        "## Résumé",
        "",
        dataframe_to_markdown_safe(
            summary_df,
            max_rows=100,
        ),
        "",
        "## Métriques modèles",
        "",
        dataframe_to_markdown_safe(
            metrics_df,
            max_rows=50,
        ),
        "",
        "## Meilleurs seuils par modèle",
        "",
        dataframe_to_markdown_safe(
            best_thresholds_df,
            max_rows=50,
        ),
        "",
        "## Bandes de probabilité — modèle sélectionné",
        "",
        dataframe_to_markdown_safe(
            bands_df,
            max_rows=100,
        ),
        "",
        "## Performance par compétition",
        "",
        dataframe_to_markdown_safe(
            competition_df,
            max_rows=50,
        ),
        "",
        "## Calibration approximative",
        "",
        dataframe_to_markdown_safe(
            calibration_df,
            max_rows=50,
        ),
        "",
        "## Top prédictions test",
        "",
        dataframe_to_markdown_safe(
            top_predictions_df,
            max_rows=50,
        ),
        "",
        "## Top features",
        "",
        dataframe_to_markdown_safe(
            importance_top_df,
            max_rows=50,
        ),
        "",
        "## Recommandations",
        "",
    ]

    for item in recommendations:
        lines.append(
            f"- {item}"
        )

    lines.extend([
        "",
        "## Décision",
        "",
        "Ce rapport sert à choisir un seuil candidat pour shadow mode.",
        "",
        "Il ne suffit pas à valider une mise en production.",
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse les sorties de train_model_v2.py."
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        help="Dossier outputs/model_v2.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset CORE_READY.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Schema model_schema_v2.json.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Forcer l'analyse d'un modèle précis.",
    )

    parser.add_argument(
        "--min-selected",
        type=int,
        default=30,
        help="Minimum de matchs sélectionnés pour considérer un seuil exploitable. Défaut: 30.",
    )

    parser.add_argument(
        "--competition-min-matches",
        type=int,
        default=8,
        help="Minimum de matchs test par compétition. Défaut: 8.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Nombre de top prédictions/features à exporter. Défaut: 50.",
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Nombre de bins calibration. Défaut: 10.",
    )

    args = parser.parse_args()

    model_dir = resolve_path(
        args.model_dir,
        DEFAULT_MODEL_DIR,
    )

    dataset_path = resolve_path(
        args.dataset,
        DEFAULT_DATASET_PATH,
    )

    schema_path = resolve_path(
        args.schema,
        DEFAULT_SCHEMA_PATH,
    )

    output_dir = model_dir / "analysis"

    ensure_dir(
        output_dir
    )

    print()
    print("=" * 100)
    print("ANALYZE MODEL V2 RESULTS — GOAL AFTER 87")
    print("=" * 100)
    print(f"Model dir : {model_dir}")
    print(f"Dataset : {dataset_path}")
    print(f"Schema : {schema_path}")
    print(f"Output : {output_dir}")

    inputs = load_inputs(
        model_dir=model_dir,
        dataset_path=dataset_path,
        schema_path=schema_path,
    )

    metrics_df = clean_numeric_columns(
        inputs["metrics"],
        [
            "test_rows",
            "test_positive_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "accuracy_at_050",
            "precision_at_050",
            "recall_at_050",
            "f1_at_050",
            "mean_probability",
            "median_probability",
            "max_probability",
        ],
    )

    thresholds_df = clean_numeric_columns(
        inputs["thresholds"],
        [
            "threshold",
            "selected",
            "selected_rate",
            "positives",
            "negatives",
            "precision",
            "late_goal_rate",
            "base_rate",
            "lift_vs_base",
        ],
    )

    predictions_df = inputs["predictions"].copy()

    selected_model = select_best_model(
        metrics_df,
        args.model,
    )

    print(f"Modèle analysé : {selected_model}")

    summary_df = build_summary(
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        dataset_df=inputs["dataset"],
        best_model=selected_model,
    )

    analyzed_thresholds_df = analyze_thresholds(
        thresholds_df=thresholds_df,
        min_selected=args.min_selected,
    )

    best_thresholds_df = best_thresholds_by_model(
        thresholds_df=thresholds_df,
        min_selected=args.min_selected,
    )

    selected_thresholds_df = thresholds_df[
        thresholds_df["model"].astype(str) == str(selected_model)
    ].copy()

    selected_predictions_df = predictions_df.copy()

    # train_model_v2 exporte seulement les prédictions du meilleur modèle.
    # Donc les bandes/calibration se rapportent à ce modèle exporté.
    bands_df = build_probability_bands(
        selected_predictions_df,
        DEFAULT_BANDS,
    )

    calibration_df = build_calibration_table(
        selected_predictions_df,
        n_bins=args.calibration_bins,
    )

    competition_df = build_competition_performance(
        selected_predictions_df,
        min_matches=args.competition_min_matches,
    )

    top_predictions_df = build_top_predictions(
        selected_predictions_df,
        top_n=args.top_n,
    )

    importance_top_df = build_feature_importance_top(
        inputs["importance"],
        top_n=args.top_n,
    )

    recommendations = build_recommendations(
        summary_df=summary_df,
        best_thresholds_df=best_thresholds_df,
        bands_df=bands_df,
        min_selected=args.min_selected,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    summary_path = output_dir / "analysis_summary_v2.csv"
    all_thresholds_path = output_dir / "analysis_thresholds_all_v2.csv"
    best_thresholds_path = output_dir / "analysis_best_thresholds_v2.csv"
    bands_path = output_dir / "analysis_probability_bands_v2.csv"
    competition_path = output_dir / "analysis_competition_performance_v2.csv"
    calibration_path = output_dir / "analysis_calibration_v2.csv"
    top_predictions_path = output_dir / "analysis_top_predictions_v2.csv"
    importance_top_path = output_dir / "analysis_feature_importance_top_v2.csv"
    manifest_path = output_dir / "analysis_manifest_v2.json"
    report_path = output_dir / "analysis_model_v2_report.md"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    analyzed_thresholds_df.to_csv(
        all_thresholds_path,
        index=False,
        encoding="utf-8-sig",
    )

    best_thresholds_df.to_csv(
        best_thresholds_path,
        index=False,
        encoding="utf-8-sig",
    )

    bands_df.to_csv(
        bands_path,
        index=False,
        encoding="utf-8-sig",
    )

    competition_df.to_csv(
        competition_path,
        index=False,
        encoding="utf-8-sig",
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
        encoding="utf-8-sig",
    )

    top_predictions_df.to_csv(
        top_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    importance_top_df.to_csv(
        importance_top_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "created_at_utc": now_utc_iso(),
        "model_dir": str(model_dir),
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "selected_model": selected_model,
        "min_selected": args.min_selected,
        "competition_min_matches": args.competition_min_matches,
        "top_n": args.top_n,
        "outputs": {
            "summary": str(summary_path),
            "thresholds_all": str(all_thresholds_path),
            "best_thresholds": str(best_thresholds_path),
            "probability_bands": str(bands_path),
            "competition_performance": str(competition_path),
            "calibration": str(calibration_path),
            "top_predictions": str(top_predictions_path),
            "feature_importance_top": str(importance_top_path),
            "report": str(report_path),
        },
        "recommendations": list(recommendations),
    }

    write_json(
        manifest_path,
        manifest,
    )

    report = make_report(
        inputs=inputs,
        output_dir=output_dir,
        selected_model=selected_model,
        summary_df=summary_df,
        metrics_df=metrics_df,
        best_thresholds_df=best_thresholds_df,
        bands_df=bands_df,
        competition_df=competition_df,
        calibration_df=calibration_df,
        top_predictions_df=top_predictions_df,
        importance_top_df=importance_top_df,
        recommendations=recommendations,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("-" * 100)
    print("Résumé")
    print("-" * 100)

    for _, row in summary_df.iterrows():
        print(
            f"{row['metric']}: {row['value']}"
        )

    print()
    print("-" * 100)
    print("Meilleurs seuils par modèle")
    print("-" * 100)

    display_cols = [
        col
        for col in [
            "model",
            "threshold",
            "selected",
            "positives",
            "negatives",
            "precision",
            "lift_vs_base",
            "risk_note",
            "selection_reason",
        ]
        if col in best_thresholds_df.columns
    ]

    if display_cols:
        print(
            best_thresholds_df[display_cols].to_string(index=False)
        )

    print()
    print("-" * 100)
    print("Bandes de probabilité — modèle exporté")
    print("-" * 100)

    band_cols = [
        col
        for col in [
            "band",
            "selected",
            "positives",
            "negatives",
            "late_goal_rate",
            "lift_vs_base",
            "avg_predicted_probability",
        ]
        if col in bands_df.columns
    ]

    if band_cols:
        print(
            bands_df[band_cols].to_string(index=False)
        )

    print()
    print("-" * 100)
    print("Recommandations")
    print("-" * 100)

    for item in recommendations:
        print(f"- {item}")

    print()
    print("=" * 100)
    print("ANALYSE TERMINÉE")
    print("=" * 100)
    print(f"Résumé : {summary_path}")
    print(f"Meilleurs seuils : {best_thresholds_path}")
    print(f"Bandes : {bands_path}")
    print(f"Compétitions : {competition_path}")
    print(f"Top prédictions : {top_predictions_path}")
    print(f"Calibration : {calibration_path}")
    print(f"Rapport : {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
