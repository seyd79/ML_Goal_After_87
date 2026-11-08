#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_model_v2_predictions_deep.py

AUDIT PROFOND DES PRÉDICTIONS — MODEL V2 — GOAL AFTER 87
========================================================

OBJECTIF
--------
Auditer en profondeur les prédictions du modèle V2 pour mieux l'affiner.

Ce script analyse les fichiers générés par :

    train_model_v2.py
    analyze_model_v2_results.py

Il ne réentraîne pas le modèle.
Il ne modifie pas le scanner.
Il ne modifie pas le dataset.

Il sert à comprendre :

    1) où le modèle se trompe ;
    2) quelles zones de probabilité sont réellement intéressantes ;
    3) quelles compétitions semblent bonnes ou dangereuses ;
    4) quels profils de matchs produisent des faux positifs ;
    5) quels profils de matchs produisent des faux négatifs ;
    6) si les probabilités sont mal calibrées ;
    7) quelles règles de prudence appliquer avant le shadow mode.

FICHIERS LUS PAR DÉFAUT
-----------------------
outputs/model_v2/model_v2_predictions_test.csv
outputs/model_v2/model_v2_metrics.csv
outputs/model_v2/model_v2_thresholds.csv
outputs/model_v2/model_v2_feature_importance.csv
outputs/model_v2/model_v2_manifest.json
outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
outputs/model_schema_v2/model_schema_v2.json

SORTIES
-------
outputs/model_v2/deep_audit/
├── deep_audit_summary.csv
├── deep_audit_probability_bands.csv
├── deep_audit_thresholds.csv
├── deep_audit_confusion_by_threshold.csv
├── deep_audit_top_false_positives.csv
├── deep_audit_top_false_negatives.csv
├── deep_audit_top_true_positives.csv
├── deep_audit_top_true_negatives.csv
├── deep_audit_competitions.csv
├── deep_audit_score_state.csv
├── deep_audit_goal_diff.csv
├── deep_audit_draw_state.csv
├── deep_audit_live_activity.csv
├── deep_audit_feature_direction.csv
├── deep_audit_feature_bins.csv
├── deep_audit_calibration.csv
├── deep_audit_recommendations.json
└── deep_audit_report.md

UTILISATION
-----------
Commande standard :

    python3 audit_model_v2_predictions_deep.py

Avec caffeinate :

    caffeinate -dimsu python3 audit_model_v2_predictions_deep.py

Avec un seuil minimum de volume plus strict :

    python3 audit_model_v2_predictions_deep.py --min-selected 50

Avec des seuils personnalisés :

    python3 audit_model_v2_predictions_deep.py --thresholds 0.30,0.34,0.37,0.40,0.45,0.50,0.55,0.60,0.70

Avec top 100 erreurs :

    python3 audit_model_v2_predictions_deep.py --top-n 100

INTERPRÉTATION
--------------
Ce script ne doit pas être utilisé pour confirmer une stratégie de pari en production.
Il sert à préparer un shadow mode propre.

Une prédiction élevée du modèle V2 doit être considérée comme un SCORE de risque,
pas comme une probabilité parfaitement calibrée.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = BASE_DIR / "outputs" / "model_v2"

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
    / "deep_audit"
)

TARGET_COLUMN = "goal_after_87"

PROB_COLUMN = "predicted_probability"

TRUE_COLUMN = "y_true"

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
    0.65,
    0.70,
    0.75,
    0.80,
]

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
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
    1.01,
]

LIVE_ACTIVITY_FEATURES = [
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "cards_80_87",
    "subs_80_87",
    "fouls_80_87",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "last_corner_minute",
    "minutes_since_last_corner",
]

SCORE_STATE_FEATURES = [
    "score_87_home",
    "score_87_away",
    "goal_diff_87",
    "is_draw_87",
    "leading_team_87",
]


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


def ensure_dir(path: Path) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


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

    return float(number)


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(
        value,
        np.nan,
    )

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


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def parse_csv_floats(
    text: Optional[str],
    default: Sequence[float],
) -> List[float]:
    if not clean_text(text):
        return list(default)

    values: List[float] = []

    for part in clean_text(text).replace(";", ",").split(","):
        part = part.strip()

        if not part:
            continue

        value = safe_float(
            part,
            np.nan,
        )

        if not pd.isna(value) and 0.0 <= value <= 1.0:
            values.append(value)

    if not values:
        return list(default)

    return sorted(set(values))


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
        return {}

    return {}


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


def read_csv_if_exists(
    path: Path,
    required: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        return pd.DataFrame()

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if required and df.empty:
        raise ValueError(
            f"Fichier vide : {path}"
        )

    return df


def dataframe_to_markdown_safe(
    df: pd.DataFrame,
    max_rows: int = 60,
) -> str:
    if df is None or df.empty:
        return "_Aucune donnée._"

    work = df.head(max_rows).copy()

    for col in work.columns:
        def format_value(value: Any) -> str:
            if pd.isna(value):
                return ""

            if isinstance(value, float):
                return f"{value:.6f}"

            return str(value)

        work[col] = work[col].apply(format_value)

    headers = [str(col) for col in work.columns]

    rows = [
        [str(value) for value in row]
        for row in work.values.tolist()
    ]

    widths: List[int] = []

    for idx, header in enumerate(headers):
        width = len(header)

        for row in rows:
            if idx < len(row):
                width = max(
                    width,
                    len(row[idx]),
                )

        widths.append(width)

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
        + " | ".join(
            "-" * width
            for width in widths
        )
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


# =============================================================================
# LOAD / MERGE
# =============================================================================

def load_all_inputs(
    model_dir: Path,
    dataset_path: Path,
    schema_path: Path,
) -> Dict[str, Any]:
    paths = {
        "metrics": model_dir / "model_v2_metrics.csv",
        "thresholds": model_dir / "model_v2_thresholds.csv",
        "predictions": model_dir / "model_v2_predictions_test.csv",
        "feature_importance": model_dir / "model_v2_feature_importance.csv",
        "manifest": model_dir / "model_v2_manifest.json",
        "dataset": dataset_path,
        "schema": schema_path,
    }

    inputs = {
        "paths": paths,
        "metrics": read_csv_if_exists(paths["metrics"], required=True),
        "thresholds": read_csv_if_exists(paths["thresholds"], required=False),
        "predictions": read_csv_if_exists(paths["predictions"], required=True),
        "feature_importance": read_csv_if_exists(paths["feature_importance"], required=False),
        "manifest": read_json_if_exists(paths["manifest"]),
        "dataset": read_csv_if_exists(paths["dataset"], required=False),
        "schema": read_json_if_exists(paths["schema"]),
    }

    return inputs


def prepare_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    df = predictions_df.copy()

    if PROB_COLUMN not in df.columns:
        raise ValueError(
            f"Colonne manquante dans predictions : {PROB_COLUMN}"
        )

    if TRUE_COLUMN not in df.columns:
        raise ValueError(
            f"Colonne manquante dans predictions : {TRUE_COLUMN}"
        )

    df[PROB_COLUMN] = pd.to_numeric(
        df[PROB_COLUMN],
        errors="coerce",
    )

    df[TRUE_COLUMN] = pd.to_numeric(
        df[TRUE_COLUMN],
        errors="coerce",
    )

    df = df[
        df[PROB_COLUMN].notna()
        & df[TRUE_COLUMN].isin([0, 1])
    ].copy()

    df[TRUE_COLUMN] = df[TRUE_COLUMN].astype(int)

    df["pred_050"] = (
        df[PROB_COLUMN] >= 0.50
    ).astype(int)

    df["error_type_050"] = np.select(
        [
            (df[TRUE_COLUMN] == 1) & (df["pred_050"] == 1),
            (df[TRUE_COLUMN] == 0) & (df["pred_050"] == 1),
            (df[TRUE_COLUMN] == 1) & (df["pred_050"] == 0),
            (df[TRUE_COLUMN] == 0) & (df["pred_050"] == 0),
        ],
        [
            "TP",
            "FP",
            "FN",
            "TN",
        ],
        default="UNKNOWN",
    )

    df["confidence_distance_050"] = (
        df[PROB_COLUMN] - 0.50
    ).abs()

    return df


def merge_predictions_with_dataset(
    predictions_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    if (
        dataset_df is None
        or dataset_df.empty
        or "match_id" not in predictions_df.columns
        or "match_id" not in dataset_df.columns
    ):
        return predictions_df.copy()

    meta_cols = [
        "match_id",
    ]

    for col in dataset_df.columns:
        if col == "match_id":
            continue

        if col not in predictions_df.columns:
            meta_cols.append(col)

    dataset_small = dataset_df[meta_cols].drop_duplicates(
        subset=["match_id"],
        keep="last",
    )

    merged = predictions_df.merge(
        dataset_small,
        on="match_id",
        how="left",
    )

    return merged


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    total = len(df)
    positives = int(
        (df[TRUE_COLUMN] == 1).sum()
    )
    negatives = int(
        (df[TRUE_COLUMN] == 0).sum()
    )

    base_rate = safe_rate(
        positives,
        total,
    )

    pred_050 = (
        df[PROB_COLUMN] >= 0.50
    )

    tp = int(
        ((df[TRUE_COLUMN] == 1) & pred_050).sum()
    )
    fp = int(
        ((df[TRUE_COLUMN] == 0) & pred_050).sum()
    )
    fn = int(
        ((df[TRUE_COLUMN] == 1) & (~pred_050)).sum()
    )
    tn = int(
        ((df[TRUE_COLUMN] == 0) & (~pred_050)).sum()
    )

    best_model = clean_text(
        manifest.get("best_model_name")
        or manifest.get("selected_model")
    )

    if not best_model and not metrics_df.empty and "model" in metrics_df.columns:
        metrics_work = clean_numeric_columns(
            metrics_df,
            ["pr_auc", "roc_auc", "brier"],
        )

        metrics_work = metrics_work.sort_values(
            by=["pr_auc", "roc_auc", "brier"],
            ascending=[False, False, True],
            na_position="last",
        )

        best_model = clean_text(
            metrics_work.iloc[0]["model"]
        )

    rows = [
        {
            "metric": "best_model_or_exported_predictions_model",
            "value": best_model or "unknown",
        },
        {
            "metric": "test_predictions_rows",
            "value": total,
        },
        {
            "metric": "test_positives",
            "value": positives,
        },
        {
            "metric": "test_negatives",
            "value": negatives,
        },
        {
            "metric": "test_base_rate",
            "value": base_rate,
        },
        {
            "metric": "mean_probability",
            "value": float(df[PROB_COLUMN].mean()),
        },
        {
            "metric": "median_probability",
            "value": float(df[PROB_COLUMN].median()),
        },
        {
            "metric": "min_probability",
            "value": float(df[PROB_COLUMN].min()),
        },
        {
            "metric": "max_probability",
            "value": float(df[PROB_COLUMN].max()),
        },
        {
            "metric": "selected_at_050",
            "value": int(pred_050.sum()),
        },
        {
            "metric": "precision_at_050",
            "value": safe_rate(tp, tp + fp),
        },
        {
            "metric": "recall_at_050",
            "value": safe_rate(tp, tp + fn),
        },
        {
            "metric": "tp_at_050",
            "value": tp,
        },
        {
            "metric": "fp_at_050",
            "value": fp,
        },
        {
            "metric": "fn_at_050",
            "value": fn,
        },
        {
            "metric": "tn_at_050",
            "value": tn,
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# PROBABILITY BANDS / THRESHOLDS
# =============================================================================

def build_probability_bands(
    df: pd.DataFrame,
    bands: Sequence[float],
) -> pd.DataFrame:
    work = df.copy()

    base_rate = float(
        work[TRUE_COLUMN].mean()
    )

    edges = sorted(set(float(x) for x in bands))

    if edges[0] > 0.0:
        edges = [0.0] + edges

    if edges[-1] <= 1.0:
        edges.append(1.01)

    rows = []

    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.01:
            mask = work[PROB_COLUMN] >= low
            label = f"{low:.2f}+"
        else:
            mask = (
                (work[PROB_COLUMN] >= low)
                & (work[PROB_COLUMN] < high)
            )
            label = f"{low:.2f}-{high:.2f}"

        chunk = work[mask].copy()

        selected = len(chunk)
        positives = int(
            chunk[TRUE_COLUMN].sum()
        ) if selected else 0
        negatives = int(
            selected - positives
        )

        rate = safe_rate(
            positives,
            selected,
        )

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
            "lift_vs_base": safe_div(rate, base_rate) if not pd.isna(rate) else np.nan,
            "avg_probability": float(chunk[PROB_COLUMN].mean()) if selected else np.nan,
            "min_probability": float(chunk[PROB_COLUMN].min()) if selected else np.nan,
            "max_probability": float(chunk[PROB_COLUMN].max()) if selected else np.nan,
            "volume_note": (
                "too_small"
                if selected < 20
                else (
                    "small"
                    if selected < 50
                    else "ok"
                )
            ),
        })

    return pd.DataFrame(rows)


def build_threshold_audit(
    df: pd.DataFrame,
    thresholds: Sequence[float],
    min_selected: int,
) -> pd.DataFrame:
    base_rate = float(
        df[TRUE_COLUMN].mean()
    )

    rows = []

    for threshold in thresholds:
        selected_mask = df[PROB_COLUMN] >= threshold
        rejected_mask = ~selected_mask

        selected = df[selected_mask]
        rejected = df[rejected_mask]

        tp = int(
            selected[TRUE_COLUMN].sum()
        )
        fp = int(
            len(selected) - tp
        )
        fn = int(
            rejected[TRUE_COLUMN].sum()
        )
        tn = int(
            len(rejected) - fn
        )

        precision = safe_rate(tp, tp + fp)
        recall = safe_rate(tp, tp + fn)
        specificity = safe_rate(tn, tn + fp)
        false_positive_rate = safe_rate(fp, fp + tn)
        false_negative_rate = safe_rate(fn, fn + tp)

        rows.append({
            "threshold": threshold,
            "selected": int(len(selected)),
            "selected_rate": safe_rate(len(selected), len(df)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "late_goal_rate_selected": precision,
            "base_rate": base_rate,
            "lift_vs_base": safe_div(precision, base_rate) if not pd.isna(precision) else np.nan,
            "avg_probability_selected": float(selected[PROB_COLUMN].mean()) if len(selected) else np.nan,
            "avg_probability_rejected": float(rejected[PROB_COLUMN].mean()) if len(rejected) else np.nan,
            "risk_note": (
                "too_small"
                if len(selected) < 20
                else (
                    "small"
                    if len(selected) < min_selected
                    else "ok"
                )
            ),
        })

    result = pd.DataFrame(rows)

    result["selection_score"] = (
        result["lift_vs_base"].fillna(0.0)
        * np.log1p(result["selected"].fillna(0.0))
    )

    return result


def build_confusion_by_threshold(
    threshold_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "threshold",
        "selected",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "lift_vs_base",
        "risk_note",
    ]

    return threshold_df[
        [col for col in cols if col in threshold_df.columns]
    ].copy()


# =============================================================================
# ERROR TABLES
# =============================================================================

def select_display_columns(
    df: pd.DataFrame,
    extra_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    base = [
        "match_id",
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        TRUE_COLUMN,
        PROB_COLUMN,
        "error_type_050",
    ]

    if extra_cols:
        base.extend(extra_cols)

    useful = [
        "score_87_home",
        "score_87_away",
        "goal_diff_87",
        "is_draw_87",
        "leading_team_87",
        "shots_80_87",
        "sot_80_87",
        "corners_80_87",
        "cards_80_87",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "minutes_since_last_shot",
        "minutes_since_last_shot_on_target",
        "odds_home",
        "odds_draw",
        "odds_away",
    ]

    base.extend(useful)

    return [
        col
        for col in base
        if col in df.columns
    ]


def build_error_tables(
    df: pd.DataFrame,
    top_n: int,
) -> Dict[str, pd.DataFrame]:
    work = df.copy()

    cols = select_display_columns(work)

    false_positives = (
        work[
            (work[TRUE_COLUMN] == 0)
            & (work[PROB_COLUMN] >= 0.50)
        ]
        .sort_values(
            by=PROB_COLUMN,
            ascending=False,
        )
        .head(top_n)
        .loc[:, cols]
        .reset_index(drop=True)
    )

    false_negatives = (
        work[
            (work[TRUE_COLUMN] == 1)
            & (work[PROB_COLUMN] < 0.50)
        ]
        .sort_values(
            by=PROB_COLUMN,
            ascending=True,
        )
        .head(top_n)
        .loc[:, cols]
        .reset_index(drop=True)
    )

    true_positives = (
        work[
            (work[TRUE_COLUMN] == 1)
            & (work[PROB_COLUMN] >= 0.50)
        ]
        .sort_values(
            by=PROB_COLUMN,
            ascending=False,
        )
        .head(top_n)
        .loc[:, cols]
        .reset_index(drop=True)
    )

    true_negatives = (
        work[
            (work[TRUE_COLUMN] == 0)
            & (work[PROB_COLUMN] < 0.50)
        ]
        .sort_values(
            by=PROB_COLUMN,
            ascending=True,
        )
        .head(top_n)
        .loc[:, cols]
        .reset_index(drop=True)
    )

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


# =============================================================================
# GROUP ANALYSES
# =============================================================================

def aggregate_group(
    df: pd.DataFrame,
    group_col: str,
    min_rows: int,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()

    # IMPORTANT :
    # Certains group_col viennent de pd.cut(), donc ce sont des colonnes Categorical.
    # Sur une Categorical, fillna("missing") plante si "missing" n'est pas déjà
    # une catégorie autorisée. On convertit donc d'abord en object/string.
    work[group_col] = (
        work[group_col]
        .astype("object")
        .where(
            work[group_col].notna(),
            "missing",
        )
    )

    base_rate = float(
        work[TRUE_COLUMN].mean()
    )

    rows = []

    for group_value, group in work.groupby(group_col):
        n = len(group)

        if n < min_rows:
            continue

        positives = int(
            group[TRUE_COLUMN].sum()
        )
        negatives = int(
            n - positives
        )

        rate = safe_rate(
            positives,
            n,
        )

        high = group[
            group[PROB_COLUMN] >= 0.50
        ]

        high_pos = int(
            high[TRUE_COLUMN].sum()
        )
        high_n = len(high)

        row = {
            group_col: group_value,
            "rows": n,
            "positives": positives,
            "negatives": negatives,
            "late_goal_rate": rate,
            "base_rate": base_rate,
            "lift_vs_base": safe_div(rate, base_rate),
            "avg_probability": float(group[PROB_COLUMN].mean()),
            "median_probability": float(group[PROB_COLUMN].median()),
            "max_probability": float(group[PROB_COLUMN].max()),
            "selected_050": high_n,
            "selected_050_positives": high_pos,
            "selected_050_precision": safe_rate(high_pos, high_n),
        }

        if label_col and label_col in group.columns:
            mode = group[label_col].mode(dropna=True)

            row[label_col] = mode.iloc[0] if len(mode) else ""

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=[
                "rows",
                "lift_vs_base",
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


def build_competition_audit(
    df: pd.DataFrame,
    min_rows: int,
) -> pd.DataFrame:
    return aggregate_group(
        df,
        group_col="competition_id",
        min_rows=min_rows,
    )


def build_score_state_audit(
    df: pd.DataFrame,
    min_rows: int,
) -> pd.DataFrame:
    work = df.copy()

    if "score_87_home" in work.columns and "score_87_away" in work.columns:
        work["score_state_87"] = (
            pd.to_numeric(work["score_87_home"], errors="coerce").fillna(-99).astype(int).astype(str)
            + "-"
            + pd.to_numeric(work["score_87_away"], errors="coerce").fillna(-99).astype(int).astype(str)
        )

        work["score_state_87"] = work["score_state_87"].replace(
            "-99--99",
            "missing",
        )

        return aggregate_group(
            work,
            group_col="score_state_87",
            min_rows=min_rows,
        )

    return pd.DataFrame()


def build_goal_diff_audit(
    df: pd.DataFrame,
    min_rows: int,
) -> pd.DataFrame:
    if "goal_diff_87" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["goal_diff_87_num"] = pd.to_numeric(
        work["goal_diff_87"],
        errors="coerce",
    )

    def bucket_goal_diff(value: Any) -> str:
        if pd.isna(value):
            return "missing"

        value = int(value)

        if value == 0:
            return "draw"
        if abs(value) == 1:
            return "one_goal_gap"
        if abs(value) == 2:
            return "two_goal_gap"

        return "three_plus_gap"

    work["goal_diff_bucket"] = work["goal_diff_87_num"].apply(
        bucket_goal_diff
    )

    return aggregate_group(
        work,
        group_col="goal_diff_bucket",
        min_rows=min_rows,
    )


def build_draw_state_audit(
    df: pd.DataFrame,
    min_rows: int,
) -> pd.DataFrame:
    if "is_draw_87" not in df.columns:
        return pd.DataFrame()

    work = df.copy()

    work["is_draw_87_bucket"] = pd.to_numeric(
        work["is_draw_87"],
        errors="coerce",
    ).map({
        0.0: "not_draw",
        1.0: "draw",
    }).fillna("missing")

    return aggregate_group(
        work,
        group_col="is_draw_87_bucket",
        min_rows=min_rows,
    )


def build_live_activity_audit(
    df: pd.DataFrame,
    min_rows: int,
) -> pd.DataFrame:
    work = df.copy()

    available = [
        col
        for col in LIVE_ACTIVITY_FEATURES
        if col in work.columns
    ]

    if not available:
        return pd.DataFrame()

    for col in available:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    # Buckets simples et exploitables.
    if "shots_80_87" in work.columns:
        work["shots_80_87_bucket"] = pd.cut(
            work["shots_80_87"],
            bins=[-1, 0, 1, 2, 4, 999],
            labels=["0", "1", "2", "3-4", "5+"],
        )
    else:
        work["shots_80_87_bucket"] = "missing"

    if "sot_80_87" in work.columns:
        work["sot_80_87_bucket"] = pd.cut(
            work["sot_80_87"],
            bins=[-1, 0, 1, 2, 999],
            labels=["0", "1", "2", "3+"],
        )
    else:
        work["sot_80_87_bucket"] = "missing"

    if "corners_80_87" in work.columns:
        work["corners_80_87_bucket"] = pd.cut(
            work["corners_80_87"],
            bins=[-1, 0, 1, 2, 999],
            labels=["0", "1", "2", "3+"],
        )
    else:
        work["corners_80_87_bucket"] = "missing"

    if "minutes_since_last_shot" in work.columns:
        work["recent_shot_bucket"] = pd.cut(
            work["minutes_since_last_shot"],
            bins=[-1, 1, 3, 5, 10, 999],
            labels=["0-1", "2-3", "4-5", "6-10", "11+"],
        )
    else:
        work["recent_shot_bucket"] = "missing"

    tables = []

    for group_col in [
        "shots_80_87_bucket",
        "sot_80_87_bucket",
        "corners_80_87_bucket",
        "recent_shot_bucket",
    ]:
        table = aggregate_group(
            work,
            group_col=group_col,
            min_rows=min_rows,
        )

        if not table.empty:
            table.insert(
                0,
                "dimension",
                group_col,
            )

            table = table.rename(
                columns={
                    group_col: "bucket"
                }
            )

            tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(
        tables,
        ignore_index=True,
    )


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def get_core_features(
    schema: Mapping[str, Any],
    df: pd.DataFrame,
) -> List[str]:
    features = schema.get("core_features")

    if isinstance(features, list) and features:
        return [
            clean_text(x)
            for x in features
            if clean_text(x) and clean_text(x) in df.columns
        ]

    exclude = {
        "match_id",
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        TRUE_COLUMN,
        TARGET_COLUMN,
        PROB_COLUMN,
        "pred_050",
        "error_type_050",
        "confidence_distance_050",
    }

    numeric_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if converted.notna().sum() >= 10:
            numeric_cols.append(col)

    return numeric_cols


def build_feature_direction_audit(
    df: pd.DataFrame,
    features: Sequence[str],
    top_n: int,
) -> pd.DataFrame:
    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid = values.notna() & df[TRUE_COLUMN].isin([0, 1])

        if valid.sum() < 20:
            continue

        temp = pd.DataFrame({
            "feature": values[valid],
            "target": df.loc[valid, TRUE_COLUMN].astype(int),
            "prob": df.loc[valid, PROB_COLUMN].astype(float),
        })

        pos = temp[temp["target"] == 1]["feature"]
        neg = temp[temp["target"] == 0]["feature"]

        if len(pos) < 5 or len(neg) < 5:
            continue

        mean_pos = float(pos.mean())
        mean_neg = float(neg.mean())
        diff = mean_pos - mean_neg

        std = float(temp["feature"].std(ddof=0))
        effect = safe_div(diff, std)

        corr_target = temp["feature"].corr(
            temp["target"],
            method="spearman",
        )

        corr_prob = temp["feature"].corr(
            temp["prob"],
            method="spearman",
        )

        rows.append({
            "feature": feature,
            "valid_rows": int(len(temp)),
            "mean_when_positive": mean_pos,
            "mean_when_negative": mean_neg,
            "diff_pos_minus_neg": diff,
            "std": std,
            "standardized_effect": effect,
            "spearman_corr_with_target": float(corr_target) if not pd.isna(corr_target) else np.nan,
            "spearman_corr_with_probability": float(corr_prob) if not pd.isna(corr_prob) else np.nan,
            "direction_hint": (
                "higher_when_goal_after_87"
                if diff > 0
                else "lower_when_goal_after_87"
            ),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["abs_effect"] = result["standardized_effect"].abs()
        result = result.sort_values(
            by=[
                "abs_effect",
                "valid_rows",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        ).head(top_n)

    return result


def build_feature_bins_audit(
    df: pd.DataFrame,
    features: Sequence[str],
    top_feature_names: Sequence[str],
    min_rows: int,
) -> pd.DataFrame:
    rows = []

    work = df.copy()

    base_rate = float(
        work[TRUE_COLUMN].mean()
    )

    for feature in top_feature_names:
        if feature not in work.columns:
            continue

        values = pd.to_numeric(
            work[feature],
            errors="coerce",
        )

        valid_mask = values.notna()

        if valid_mask.sum() < min_rows * 2:
            continue

        temp = work.loc[valid_mask].copy()
        temp["_feature_value"] = values[valid_mask]

        try:
            temp["_bin"] = pd.qcut(
                temp["_feature_value"],
                q=min(5, temp["_feature_value"].nunique()),
                duplicates="drop",
            )
        except Exception:
            try:
                temp["_bin"] = pd.cut(
                    temp["_feature_value"],
                    bins=min(5, temp["_feature_value"].nunique()),
                    include_lowest=True,
                )
            except Exception:
                continue

        for bin_label, group in temp.groupby("_bin", observed=False):
            n = len(group)

            if n < min_rows:
                continue

            positives = int(
                group[TRUE_COLUMN].sum()
            )

            rate = safe_rate(
                positives,
                n,
            )

            rows.append({
                "feature": feature,
                "bin": str(bin_label),
                "rows": n,
                "positives": positives,
                "negatives": int(n - positives),
                "late_goal_rate": rate,
                "base_rate": base_rate,
                "lift_vs_base": safe_div(rate, base_rate),
                "avg_probability": float(group[PROB_COLUMN].mean()),
                "min_feature_value": float(group["_feature_value"].min()),
                "max_feature_value": float(group["_feature_value"].max()),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=[
                "feature",
                "lift_vs_base",
                "late_goal_rate",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )

    return result


# =============================================================================
# CALIBRATION
# =============================================================================

def build_calibration_audit(
    df: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    work = df.copy()

    if work.empty:
        return pd.DataFrame()

    try:
        work["_calibration_bin"] = pd.qcut(
            work[PROB_COLUMN],
            q=min(bins, len(work)),
            duplicates="drop",
        )
    except Exception:
        work["_calibration_bin"] = pd.cut(
            work[PROB_COLUMN],
            bins=min(bins, len(work)),
            include_lowest=True,
        )

    rows = []

    for idx, (bin_label, group) in enumerate(work.groupby("_calibration_bin", observed=False), start=1):
        n = len(group)

        if n == 0:
            continue

        avg_prob = float(
            group[PROB_COLUMN].mean()
        )

        observed = float(
            group[TRUE_COLUMN].mean()
        )

        rows.append({
            "bin": idx,
            "bin_range": str(bin_label),
            "rows": n,
            "avg_predicted_probability": avg_prob,
            "observed_late_goal_rate": observed,
            "calibration_error_pred_minus_obs": avg_prob - observed,
            "abs_calibration_error": abs(avg_prob - observed),
            "positives": int(group[TRUE_COLUMN].sum()),
            "negatives": int(n - group[TRUE_COLUMN].sum()),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["weighted_abs_error"] = (
            result["abs_calibration_error"]
            * result["rows"]
            / result["rows"].sum()
        )

    return result


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def build_recommendations(
    summary_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    competition_df: pd.DataFrame,
    min_selected: int,
) -> Dict[str, Any]:
    summary = {
        clean_text(row["metric"]): row["value"]
        for _, row in summary_df.iterrows()
    }

    base_rate = safe_float(
        summary.get("test_base_rate"),
        np.nan,
    )

    recommendations: List[str] = []
    flags: List[str] = []

    # Best band
    best_band = None

    if not bands_df.empty:
        usable_bands = bands_df[
            bands_df["selected"] >= min_selected
        ].copy()

        if not usable_bands.empty:
            usable_bands = usable_bands.sort_values(
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
                na_position="last",
            )

            best_band = usable_bands.iloc[0].to_dict()

            recommendations.append(
                (
                    "Bande candidate shadow mode : "
                    f"{best_band['band']} | "
                    f"{int(best_band['selected'])} matchs | "
                    f"taux {safe_float(best_band['late_goal_rate']):.2%} | "
                    f"lift {safe_float(best_band['lift_vs_base']):.2f}."
                )
            )

        high_small = bands_df[
            (bands_df["low"] >= 0.70)
            & (bands_df["selected"] < min_selected)
            & (bands_df["selected"] > 0)
        ]

        if not high_small.empty:
            flags.append(
                "Les probabilités très hautes semblent prometteuses mais l'échantillon est trop petit."
            )

    # Best threshold
    best_threshold = None

    if not threshold_df.empty:
        usable_thresholds = threshold_df[
            threshold_df["selected"] >= min_selected
        ].copy()

        if not usable_thresholds.empty:
            usable_thresholds = usable_thresholds.sort_values(
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

            best_threshold = usable_thresholds.iloc[0].to_dict()

            recommendations.append(
                (
                    "Seuil candidat shadow mode : "
                    f"{safe_float(best_threshold['threshold']):.2f} | "
                    f"{int(best_threshold['selected'])} matchs | "
                    f"precision {safe_float(best_threshold['precision']):.2%} | "
                    f"lift {safe_float(best_threshold['lift_vs_base']):.2f}."
                )
            )

    # Calibration
    if not calibration_df.empty:
        weighted_error = float(
            calibration_df["weighted_abs_error"].sum()
        )

        if weighted_error >= 0.15:
            flags.append(
                "Calibration faible : les probabilités ne doivent pas être lues comme des chances réelles."
            )

            recommendations.append(
                "Ajouter une calibration plus tard : CalibratedClassifierCV, isotonic ou Platt scaling."
            )

    # Competition instability
    if not competition_df.empty:
        strong_comp = competition_df[
            (competition_df["rows"] >= 8)
            & (competition_df["lift_vs_base"] >= 1.50)
        ]

        weak_comp = competition_df[
            (competition_df["rows"] >= 8)
            & (competition_df["lift_vs_base"] <= 0.60)
        ]

        if not strong_comp.empty:
            recommendations.append(
                "Certaines compétitions semblent meilleures que la moyenne, mais il faut plus de volume avant filtre par compétition."
            )

        if not weak_comp.empty:
            recommendations.append(
                "Certaines compétitions semblent dangereuses : envisager un blocage compétition après plus de données."
            )

    if not recommendations:
        recommendations.append(
            "Aucun seuil robuste trouvé. Continuer la collecte CORE_READY avant shadow mode."
        )

    recommendations.append(
        "Ne pas remplacer le scanner de production. Utiliser ce modèle seulement en shadow mode."
    )

    return {
        "base_rate": base_rate,
        "min_selected": min_selected,
        "best_band": best_band,
        "best_threshold": best_threshold,
        "flags": flags,
        "recommendations": recommendations,
    }


# =============================================================================
# REPORT
# =============================================================================

def make_report(
    paths: Mapping[str, Path],
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    score_state_df: pd.DataFrame,
    goal_diff_df: pd.DataFrame,
    draw_state_df: pd.DataFrame,
    live_activity_df: pd.DataFrame,
    feature_direction_df: pd.DataFrame,
    feature_bins_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    false_positives_df: pd.DataFrame,
    false_negatives_df: pd.DataFrame,
    true_positives_df: pd.DataFrame,
    recommendations: Mapping[str, Any],
) -> str:
    recs = recommendations.get(
        "recommendations",
        [],
    )

    flags = recommendations.get(
        "flags",
        [],
    )

    lines = [
        "# Deep Audit Model V2 Predictions — Goal After 87",
        "",
        f"Généré le : `{now_utc_iso()}`",
        "",
        "## Fichiers audités",
        "",
        f"- Metrics : `{paths.get('metrics')}`",
        f"- Thresholds : `{paths.get('thresholds')}`",
        f"- Predictions : `{paths.get('predictions')}`",
        f"- Feature importance : `{paths.get('feature_importance')}`",
        f"- Dataset : `{paths.get('dataset')}`",
        f"- Schema : `{paths.get('schema')}`",
        "",
        "## Résumé",
        "",
        dataframe_to_markdown_safe(summary_df, max_rows=100),
        "",
        "## Métriques du modèle",
        "",
        dataframe_to_markdown_safe(metrics_df, max_rows=20),
        "",
        "## Bandes de probabilité",
        "",
        dataframe_to_markdown_safe(bands_df, max_rows=100),
        "",
        "## Audit des seuils",
        "",
        dataframe_to_markdown_safe(threshold_df, max_rows=100),
        "",
        "## Matrice de décision par seuil",
        "",
        dataframe_to_markdown_safe(confusion_df, max_rows=100),
        "",
        "## Calibration",
        "",
        dataframe_to_markdown_safe(calibration_df, max_rows=50),
        "",
        "## Compétitions",
        "",
        dataframe_to_markdown_safe(competitions_df, max_rows=80),
        "",
        "## États du score à 87'",
        "",
        dataframe_to_markdown_safe(score_state_df, max_rows=80),
        "",
        "## Différence de score à 87'",
        "",
        dataframe_to_markdown_safe(goal_diff_df, max_rows=80),
        "",
        "## Match nul ou non à 87'",
        "",
        dataframe_to_markdown_safe(draw_state_df, max_rows=20),
        "",
        "## Activité live 80'-87'",
        "",
        dataframe_to_markdown_safe(live_activity_df, max_rows=120),
        "",
        "## Direction des features",
        "",
        dataframe_to_markdown_safe(feature_direction_df, max_rows=80),
        "",
        "## Bins des features importantes",
        "",
        dataframe_to_markdown_safe(feature_bins_df, max_rows=120),
        "",
        "## Faux positifs les plus confiants",
        "",
        dataframe_to_markdown_safe(false_positives_df, max_rows=50),
        "",
        "## Faux négatifs les plus importants",
        "",
        dataframe_to_markdown_safe(false_negatives_df, max_rows=50),
        "",
        "## Vrais positifs les plus confiants",
        "",
        dataframe_to_markdown_safe(true_positives_df, max_rows=50),
        "",
        "## Flags",
        "",
    ]

    if flags:
        lines.extend(
            f"- {item}"
            for item in flags
        )
    else:
        lines.append("- Aucun flag majeur.")

    lines.extend([
        "",
        "## Recommandations",
        "",
    ])

    lines.extend(
        f"- {item}"
        for item in recs
    )

    lines.extend([
        "",
        "## Décision",
        "",
        "Ce deep audit sert à choisir des règles de shadow mode, pas à activer des alertes de production.",
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit profond des prédictions du modèle V2."
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        help="Dossier outputs/model_v2.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset CORE_READY V2.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Schema model_schema_v2.json.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie deep audit.",
    )

    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Seuils à auditer, séparés par virgule.",
    )

    parser.add_argument(
        "--bands",
        default=",".join(str(x) for x in DEFAULT_BANDS),
        help="Bornes des bandes de probabilité.",
    )

    parser.add_argument(
        "--min-selected",
        type=int,
        default=30,
        help="Minimum de matchs pour considérer un seuil/bande exploitable. Défaut: 30.",
    )

    parser.add_argument(
        "--group-min-rows",
        type=int,
        default=8,
        help="Minimum de lignes pour afficher un groupe. Défaut: 8.",
    )

    parser.add_argument(
        "--feature-bin-min-rows",
        type=int,
        default=20,
        help="Minimum de lignes par bin feature. Défaut: 20.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Nombre de lignes dans les tops erreurs/features. Défaut: 50.",
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

    output_dir = resolve_path(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    ensure_dir(output_dir)

    thresholds = parse_csv_floats(
        args.thresholds,
        DEFAULT_THRESHOLDS,
    )

    bands = parse_csv_floats(
        args.bands,
        DEFAULT_BANDS,
    )

    print()
    print("=" * 100)
    print("DEEP AUDIT MODEL V2 PREDICTIONS — GOAL AFTER 87")
    print("=" * 100)
    print(f"Model dir : {model_dir}")
    print(f"Dataset   : {dataset_path}")
    print(f"Schema    : {schema_path}")
    print(f"Output    : {output_dir}")

    inputs = load_all_inputs(
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

    predictions_df = prepare_predictions(
        inputs["predictions"]
    )

    merged_df = merge_predictions_with_dataset(
        predictions_df=predictions_df,
        dataset_df=inputs["dataset"],
    )

    print(f"Predictions test : {len(merged_df)}")

    summary_df = build_summary(
        df=merged_df,
        metrics_df=metrics_df,
        manifest=inputs["manifest"],
    )

    bands_df = build_probability_bands(
        df=merged_df,
        bands=bands,
    )

    threshold_df = build_threshold_audit(
        df=merged_df,
        thresholds=thresholds,
        min_selected=args.min_selected,
    )

    confusion_df = build_confusion_by_threshold(
        threshold_df
    )

    error_tables = build_error_tables(
        df=merged_df,
        top_n=args.top_n,
    )

    competitions_df = build_competition_audit(
        df=merged_df,
        min_rows=args.group_min_rows,
    )

    score_state_df = build_score_state_audit(
        df=merged_df,
        min_rows=args.group_min_rows,
    )

    goal_diff_df = build_goal_diff_audit(
        df=merged_df,
        min_rows=args.group_min_rows,
    )

    draw_state_df = build_draw_state_audit(
        df=merged_df,
        min_rows=args.group_min_rows,
    )

    live_activity_df = build_live_activity_audit(
        df=merged_df,
        min_rows=args.group_min_rows,
    )

    core_features = get_core_features(
        schema=inputs["schema"],
        df=merged_df,
    )

    feature_direction_df = build_feature_direction_audit(
        df=merged_df,
        features=core_features,
        top_n=args.top_n,
    )

    top_feature_names: List[str] = []

    if not inputs["feature_importance"].empty and "feature" in inputs["feature_importance"].columns:
        imp = inputs["feature_importance"].copy()

        if "importance" in imp.columns:
            imp["importance"] = pd.to_numeric(
                imp["importance"],
                errors="coerce",
            )

            top_feature_names = (
                imp.sort_values(
                    by="importance",
                    ascending=False,
                    na_position="last",
                )["feature"]
                .astype(str)
                .head(15)
                .tolist()
            )

    if not top_feature_names and not feature_direction_df.empty:
        top_feature_names = (
            feature_direction_df["feature"]
            .astype(str)
            .head(15)
            .tolist()
        )

    feature_bins_df = build_feature_bins_audit(
        df=merged_df,
        features=core_features,
        top_feature_names=top_feature_names,
        min_rows=args.feature_bin_min_rows,
    )

    calibration_df = build_calibration_audit(
        df=merged_df,
        bins=args.calibration_bins,
    )

    recommendations = build_recommendations(
        summary_df=summary_df,
        bands_df=bands_df,
        threshold_df=threshold_df,
        calibration_df=calibration_df,
        competition_df=competitions_df,
        min_selected=args.min_selected,
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    paths = inputs["paths"]

    summary_path = output_dir / "deep_audit_summary.csv"
    bands_path = output_dir / "deep_audit_probability_bands.csv"
    thresholds_path = output_dir / "deep_audit_thresholds.csv"
    confusion_path = output_dir / "deep_audit_confusion_by_threshold.csv"
    fp_path = output_dir / "deep_audit_top_false_positives.csv"
    fn_path = output_dir / "deep_audit_top_false_negatives.csv"
    tp_path = output_dir / "deep_audit_top_true_positives.csv"
    tn_path = output_dir / "deep_audit_top_true_negatives.csv"
    competitions_path = output_dir / "deep_audit_competitions.csv"
    score_state_path = output_dir / "deep_audit_score_state.csv"
    goal_diff_path = output_dir / "deep_audit_goal_diff.csv"
    draw_state_path = output_dir / "deep_audit_draw_state.csv"
    live_activity_path = output_dir / "deep_audit_live_activity.csv"
    feature_direction_path = output_dir / "deep_audit_feature_direction.csv"
    feature_bins_path = output_dir / "deep_audit_feature_bins.csv"
    calibration_path = output_dir / "deep_audit_calibration.csv"
    recommendations_path = output_dir / "deep_audit_recommendations.json"
    report_path = output_dir / "deep_audit_report.md"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    bands_df.to_csv(
        bands_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_df.to_csv(
        thresholds_path,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df.to_csv(
        confusion_path,
        index=False,
        encoding="utf-8-sig",
    )

    error_tables["false_positives"].to_csv(
        fp_path,
        index=False,
        encoding="utf-8-sig",
    )

    error_tables["false_negatives"].to_csv(
        fn_path,
        index=False,
        encoding="utf-8-sig",
    )

    error_tables["true_positives"].to_csv(
        tp_path,
        index=False,
        encoding="utf-8-sig",
    )

    error_tables["true_negatives"].to_csv(
        tn_path,
        index=False,
        encoding="utf-8-sig",
    )

    competitions_df.to_csv(
        competitions_path,
        index=False,
        encoding="utf-8-sig",
    )

    score_state_df.to_csv(
        score_state_path,
        index=False,
        encoding="utf-8-sig",
    )

    goal_diff_df.to_csv(
        goal_diff_path,
        index=False,
        encoding="utf-8-sig",
    )

    draw_state_df.to_csv(
        draw_state_path,
        index=False,
        encoding="utf-8-sig",
    )

    live_activity_df.to_csv(
        live_activity_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_direction_df.to_csv(
        feature_direction_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_bins_df.to_csv(
        feature_bins_path,
        index=False,
        encoding="utf-8-sig",
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        recommendations_path,
        recommendations,
    )

    report = make_report(
        paths=paths,
        summary_df=summary_df,
        metrics_df=metrics_df,
        bands_df=bands_df,
        threshold_df=threshold_df,
        confusion_df=confusion_df,
        competitions_df=competitions_df,
        score_state_df=score_state_df,
        goal_diff_df=goal_diff_df,
        draw_state_df=draw_state_df,
        live_activity_df=live_activity_df,
        feature_direction_df=feature_direction_df,
        feature_bins_df=feature_bins_df,
        calibration_df=calibration_df,
        false_positives_df=error_tables["false_positives"],
        false_negatives_df=error_tables["false_negatives"],
        true_positives_df=error_tables["true_positives"],
        recommendations=recommendations,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    print()
    print("-" * 100)
    print("Résumé")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("-" * 100)
    print("Meilleures bandes exploitables")
    print("-" * 100)

    bands_display = bands_df[
        bands_df["selected"] >= args.min_selected
    ].sort_values(
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
        na_position="last",
    )

    if bands_display.empty:
        print("Aucune bande avec assez de volume.")
    else:
        print(
            bands_display[
                [
                    "band",
                    "selected",
                    "positives",
                    "negatives",
                    "late_goal_rate",
                    "lift_vs_base",
                    "volume_note",
                ]
            ].head(10).to_string(index=False)
        )

    print()
    print("-" * 100)
    print("Meilleurs seuils exploitables")
    print("-" * 100)

    threshold_display = threshold_df[
        threshold_df["selected"] >= args.min_selected
    ].sort_values(
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

    if threshold_display.empty:
        print("Aucun seuil avec assez de volume.")
    else:
        print(
            threshold_display[
                [
                    "threshold",
                    "selected",
                    "tp",
                    "fp",
                    "precision",
                    "recall",
                    "lift_vs_base",
                    "risk_note",
                ]
            ].head(10).to_string(index=False)
        )

    print()
    print("-" * 100)
    print("Recommandations")
    print("-" * 100)

    for item in recommendations.get("recommendations", []):
        print(f"- {item}")

    print()
    print("=" * 100)
    print("DEEP AUDIT TERMINÉ")
    print("=" * 100)
    print(f"Rapport : {report_path}")
    print(f"Bandes : {bands_path}")
    print(f"Seuils : {thresholds_path}")
    print(f"Faux positifs : {fp_path}")
    print(f"Faux négatifs : {fn_path}")
    print(f"Compétitions : {competitions_path}")
    print(f"Features : {feature_direction_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
