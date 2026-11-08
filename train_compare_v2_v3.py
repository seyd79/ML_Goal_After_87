#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
COMPARE V2 vs V3 — GOAL AFTER 87'
===============================================================================

But
---
Réentraîner et comparer, sur le MEME split temporel :

1) V2_BASELINE_52
   - 52 features actuelles.

2) V3_A_CONSERVATIVE
   Retire :
   - home_fouls_87
   - goal_diff_87
   - cards_80_87

3) V3_B_CLEAN
   Retire :
   - home_fouls_87
   - goal_diff_87
   - cards_80_87
   - league_home_btts_freq
   - league_away_btts_freq
   - away_avg_goals_conceded

4) V3_C_AGGRESSIVE
   Retire les 14 features POTENTIAL_NOISE du dernier audit temporel :
   - home_corners_87
   - away_corners_87
   - last_shot_on_target_minute
   - away_form5
   - away_yellow_cards_87
   - away_btts_freq
   - shots_80_87
   - away_shots_on_target_87
   - goal_diff_87
   - home_fouls_87
   - away_avg_goals_conceded
   - league_away_btts_freq
   - league_home_btts_freq
   - cards_80_87

Le script :
- NE MODIFIE PAS le modèle de production.
- Clone l'architecture / les hyperparamètres du modèle V2 actuel.
- Réentraîne chaque variante uniquement sur TRAIN.
- Teste toutes les variantes sur le MEME bloc temporel récent.
- Mesure ROC-AUC, PR-AUC, Brier, LogLoss.
- Mesure aussi les prédictions >= 70 %.
- Mesure la stabilité TRAIN -> TEST.
- Génère des modèles candidats séparés.
- Génère un rapport CSV + Markdown + JSON.
- Refit ensuite chaque variante sur 100 % des données et sauvegarde ces
  modèles dans un dossier "candidates_refit_all", SANS toucher au scanner.

IMPORTANT
---------
Les modèles "refit_all" ne doivent PAS être utilisés pour comparer leurs
performances : ils sont seulement préparés comme candidats de déploiement.
La comparaison repose exclusivement sur le split temporel TRAIN/TEST.

Commande
--------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 train_compare_v2_v3.py

Sorties
-------
outputs/model_v3_experiments/
    comparison_metrics.csv
    threshold_070_metrics.csv
    predictions_test.csv
    feature_sets.json
    comparison_report.md
    experiment_metadata.json

    split_models/
        V2_BASELINE_52.joblib
        V3_A_CONSERVATIVE.joblib
        V3_B_CLEAN.joblib
        V3_C_AGGRESSIVE.joblib

    candidates_refit_all/
        V2_BASELINE_52_refit_all.joblib
        V3_A_CONSERVATIVE_refit_all.joblib
        V3_B_CLEAN_refit_all.joblib
        V3_C_AGGRESSIVE_refit_all.joblib

    schemas/
        V2_BASELINE_52_schema.json
        V3_A_CONSERVATIVE_schema.json
        V3_B_CLEAN_schema.json
        V3_C_AGGRESSIVE_schema.json
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
    )
)

MODEL_CANDIDATES = [
    "outputs/model_v2/best_model_v2_refit_all.joblib",
    "outputs/model_v2/best_model_v2.joblib",
    "outputs/model_v2/model_v2.joblib",
]

SCHEMA_CANDIDATES = [
    "outputs/model_v2/model_v2_schema_used.json",
    "outputs/model_schema_v2/model_schema_v2.json",
    "outputs/model_v2/schema.json",
]

DATASET_CANDIDATES = [
    "outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    "outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
    "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
]

TARGET_CANDIDATES = [
    "goal_after_87",
    "target",
    "label",
    "y",
]

TIME_CANDIDATES = [
    "kickoff_utc",
    "timestamp",
    "match_datetime",
    "fixture_datetime",
    "kickoff",
    "kickoff_time",
    "match_date",
    "date",
]

MATCH_ID_CANDIDATES = [
    "match_id",
    "fixture_id",
    "id",
]

ALERT_THRESHOLD = 0.70
DEFAULT_TEST_FRACTION = 0.25
DEFAULT_RANDOM_STATE = 42


# =============================================================================
# FEATURE VARIANTS
# =============================================================================

V3_A_REMOVE = [
    "home_fouls_87",
    "goal_diff_87",
    "cards_80_87",
]

V3_B_REMOVE = [
    "home_fouls_87",
    "goal_diff_87",
    "cards_80_87",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "away_avg_goals_conceded",
]

V3_C_REMOVE = [
    "home_corners_87",
    "away_corners_87",
    "last_shot_on_target_minute",
    "away_form5",
    "away_yellow_cards_87",
    "away_btts_freq",
    "shots_80_87",
    "away_shots_on_target_87",
    "goal_diff_87",
    "home_fouls_87",
    "away_avg_goals_conceded",
    "league_away_btts_freq",
    "league_home_btts_freq",
    "cards_80_87",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvalMetrics:
    model_name: str
    n_features: int
    train_n: int
    test_n: int
    train_positive_rate: float
    test_positive_rate: float

    train_roc_auc: float
    test_roc_auc: float
    auc_gap_train_minus_test: float

    train_pr_auc: float
    test_pr_auc: float
    pr_auc_gap_train_minus_test: float

    train_brier: float
    test_brier: float
    brier_gap_test_minus_train: float

    train_log_loss: float
    test_log_loss: float
    logloss_gap_test_minus_train: float

    test_mean_probability: float
    test_median_probability: float
    test_min_probability: float
    test_max_probability: float


@dataclass
class ThresholdMetrics:
    model_name: str
    threshold: float
    selected: int
    selected_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    observed_win_rate: float
    mean_probability_selected: float
    median_probability_selected: float
    lift_vs_test_base_rate: float


# =============================================================================
# HELPERS
# =============================================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def resolve_file(
    project_root: Path,
    explicit: Optional[str],
    candidates: Sequence[str],
    label: str,
) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            raise FileNotFoundError(f"{label} introuvable : {p}")
        return p

    for rel in candidates:
        p = project_root / rel
        if p.exists():
            return p

    raise FileNotFoundError(
        f"{label} introuvable. Candidats testés:\n  - "
        + "\n  - ".join(str(project_root / x) for x in candidates)
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        hit = lower_map.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _find_feature_list(obj: Any) -> Optional[List[str]]:
    preferred = (
        "features",
        "feature_names",
        "model_features",
        "used_features",
        "columns",
        "schema",
    )

    if isinstance(obj, dict):
        for key in preferred:
            if key in obj:
                value = obj[key]
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(x, str) for x in value)
                ):
                    return list(value)

        for value in obj.values():
            found = _find_feature_list(value)
            if found:
                return found

    elif isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            return list(obj)

        for value in obj:
            found = _find_feature_list(value)
            if found:
                return found

    return None


def load_feature_schema(
    schema_path: Path,
    model: Any,
) -> List[str]:
    with schema_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    features = _find_feature_list(payload)

    if not features and hasattr(model, "feature_names_in_"):
        features = [str(x) for x in model.feature_names_in_]

    if not features:
        raise ValueError(
            "Impossible d'extraire la liste exacte des features."
        )

    if len(set(features)) != len(features):
        duplicated = pd.Series(features)[
            pd.Series(features).duplicated()
        ].tolist()
        raise ValueError(
            f"Features dupliquées dans le schema : {duplicated}"
        )

    return features


# =============================================================================
# DATASET PARSING
# =============================================================================

def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()

    if not header:
        raise ValueError("Dataset vide.")

    counts = {
        "\t": header.count("\t"),
        ",": header.count(","),
        ";": header.count(";"),
        "|": header.count("|"),
    }

    delimiter = max(counts, key=counts.get)

    if counts[delimiter] <= 0:
        raise ValueError(
            f"Séparateur indétectable. Comptages : {counts}"
        )

    return delimiter


def delimiter_name(delimiter: str) -> str:
    return {
        "\t": "TABULATION",
        ",": "VIRGULE",
        ";": "POINT-VIRGULE",
        "|": "PIPE",
    }.get(delimiter, repr(delimiter))


def load_dataset(path: Path) -> pd.DataFrame:
    log(f"Lecture dataset : {path}")

    sep = detect_delimiter(path)
    log(f"Séparateur détecté : {delimiter_name(sep)}")

    try:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            encoding="utf-8-sig",
            skipinitialspace=False,
            on_bad_lines="error",
        )
    except Exception as exc:
        raise ValueError(
            "Lecture structurée impossible. "
            "Aucune ligne n'a été ignorée. "
            f"Erreur originale : {exc}"
        ) from exc

    if df.empty:
        raise ValueError("Dataset vide.")

    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 50:
        raise ValueError(
            f"Parsing suspect : seulement {len(df.columns)} colonnes."
        )

    dup = pd.Index(df.columns)[
        pd.Index(df.columns).duplicated()
    ].tolist()

    if dup:
        raise ValueError(f"Colonnes dupliquées : {dup}")

    log(
        f"Dataset chargé : {len(df):,} lignes | "
        f"{len(df.columns)} colonnes"
    )
    return df


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() >= 0.95:
        out = numeric
    else:
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "oui": 1,
            "non": 0,
            "positive": 1,
            "negative": 0,
            "won": 1,
            "lost": 0,
            "gagné": 1,
            "gagne": 1,
            "perdu": 0,
        }
        out = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    if out.isna().any():
        examples = series[out.isna()].astype(str).head(10).tolist()
        raise ValueError(
            "Target non convertible en 0/1. "
            f"Exemples : {examples}"
        )

    out = out.astype(int)

    if not set(out.unique()).issubset({0, 1}):
        raise ValueError(
            f"Target non binaire : {sorted(out.unique().tolist())}"
        )

    return out


def coerce_features(
    df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(
            "Features du modèle absentes du dataset:\n  - "
            + "\n  - ".join(missing)
        )

    X = df.loc[:, list(features)].copy()

    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(float)
            continue

        if not pd.api.types.is_numeric_dtype(X[col]):
            converted = pd.to_numeric(
                X[col],
                errors="coerce",
            )

            original_non_na = int(X[col].notna().sum())
            converted_non_na = int(converted.notna().sum())

            if original_non_na > 0 and converted_non_na == 0:
                raise ValueError(
                    f"Feature non numérique inattendue : {col}"
                )

            X[col] = converted

    return X.astype(float)


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

def parse_time_series(series: pd.Series) -> Tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())

    if numeric_ratio >= 0.80:
        vals = numeric.dropna()

        if len(vals):
            median_value = float(vals.median())

            # Excel serial days, e.g. 46278.104166...
            if 20_000 <= median_value <= 100_000:
                parsed = pd.to_datetime(
                    numeric,
                    unit="D",
                    origin="1899-12-30",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "EXCEL_SERIAL_DAYS"

            if 1e8 <= median_value < 1e11:
                return (
                    pd.to_datetime(
                        numeric,
                        unit="s",
                        errors="coerce",
                        utc=True,
                    ),
                    "UNIX_SECONDS",
                )

            if 1e11 <= median_value < 1e14:
                return (
                    pd.to_datetime(
                        numeric,
                        unit="ms",
                        errors="coerce",
                        utc=True,
                    ),
                    "UNIX_MILLISECONDS",
                )

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        ),
        "DATETIME_TEXT",
    )


def temporal_sort(
    df: pd.DataFrame,
    explicit_time_col: Optional[str],
) -> Tuple[pd.DataFrame, str, str]:
    time_col = explicit_time_col or first_existing_column(
        df,
        TIME_CANDIDATES,
    )

    if time_col is None:
        raise ValueError(
            "Aucune colonne temporelle trouvée. "
            "Ce script refuse un split non temporel."
        )

    if time_col not in df.columns:
        raise ValueError(
            f"Colonne temporelle absente : {time_col}"
        )

    parsed, parse_mode = parse_time_series(df[time_col])
    valid_ratio = float(parsed.notna().mean())

    log(
        f"Colonne temps : {time_col} | "
        f"format={parse_mode} | "
        f"parseable={valid_ratio:.2%}"
    )

    if valid_ratio < 1.0:
        bad = int(parsed.isna().sum())
        raise ValueError(
            f"{bad} lignes ont une date invalide. "
            "Split temporel annulé pour éviter un biais."
        )

    out = df.copy()
    out["__audit_time__"] = parsed
    out["__row_id_before_sort__"] = np.arange(len(out))

    out = (
        out.sort_values(
            ["__audit_time__", "__row_id_before_sort__"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    log(
        "Période complète : "
        f"{out['__audit_time__'].iloc[0].isoformat()} -> "
        f"{out['__audit_time__'].iloc[-1].isoformat()}"
    )

    return out, time_col, parse_mode


def make_split(
    n_rows: int,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.10 <= test_fraction <= 0.40:
        raise ValueError(
            "--test-fraction doit être entre 0.10 et 0.40."
        )

    test_n = int(round(n_rows * test_fraction))

    if test_n < 100:
        raise ValueError(
            f"Test trop petit : {test_n} lignes."
        )

    train_n = n_rows - test_n

    if train_n < 300:
        raise ValueError(
            f"Train trop petit : {train_n} lignes."
        )

    train_idx = np.arange(0, train_n)
    test_idx = np.arange(train_n, n_rows)

    return train_idx, test_idx


# =============================================================================
# MODEL
# =============================================================================

def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X))

        if raw.ndim == 2:
            if raw.shape[1] != 2:
                raise ValueError(
                    "predict_proba ne contient pas exactement deux classes."
                )
            p = raw[:, 1]
        else:
            p = raw.reshape(-1)

        return np.clip(p.astype(float), 1e-12, 1 - 1e-12)

    if hasattr(model, "decision_function"):
        z = np.asarray(
            model.decision_function(X),
            dtype=float,
        ).reshape(-1)

        p = 1.0 / (
            1.0 + np.exp(-np.clip(z, -40, 40))
        )
        return np.clip(p, 1e-12, 1 - 1e-12)

    raise ValueError(
        "Le modèle ne fournit ni predict_proba ni decision_function."
    )


def clone_model_template(production_model: Any) -> Any:
    """
    Reproduit exactement l'architecture et les hyperparamètres sklearn
    du modèle V2 courant, mais sans ses poids appris.
    """
    try:
        return clone(production_model)
    except Exception as exc:
        raise TypeError(
            "Impossible de cloner le modèle V2 avec sklearn.clone(). "
            "Le script refuse de remplacer silencieusement son architecture. "
            f"Erreur : {exc}"
        ) from exc


def validate_variant_features(
    base_features: Sequence[str],
    removed: Sequence[str],
    variant_name: str,
) -> List[str]:
    base_set = set(base_features)
    unknown = [f for f in removed if f not in base_set]

    if unknown:
        raise ValueError(
            f"{variant_name}: features à retirer absentes du V2 : {unknown}"
        )

    result = [
        f for f in base_features
        if f not in set(removed)
    ]

    if not result:
        raise ValueError(
            f"{variant_name}: aucune feature restante."
        )

    return result


# =============================================================================
# METRICS
# =============================================================================

def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def evaluate_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[EvalMetrics, np.ndarray, np.ndarray]:

    p_train = predict_probability(model, X_train)
    p_test = predict_probability(model, X_test)

    train_auc = safe_roc_auc(y_train.values, p_train)
    test_auc = safe_roc_auc(y_test.values, p_test)

    train_ap = safe_pr_auc(y_train.values, p_train)
    test_ap = safe_pr_auc(y_test.values, p_test)

    train_brier = float(
        brier_score_loss(y_train.values, p_train)
    )
    test_brier = float(
        brier_score_loss(y_test.values, p_test)
    )

    train_ll = float(
        log_loss(
            y_train.values,
            p_train,
            labels=[0, 1],
        )
    )
    test_ll = float(
        log_loss(
            y_test.values,
            p_test,
            labels=[0, 1],
        )
    )

    metrics = EvalMetrics(
        model_name=model_name,
        n_features=X_train.shape[1],
        train_n=len(X_train),
        test_n=len(X_test),
        train_positive_rate=float(y_train.mean()),
        test_positive_rate=float(y_test.mean()),

        train_roc_auc=train_auc,
        test_roc_auc=test_auc,
        auc_gap_train_minus_test=train_auc - test_auc,

        train_pr_auc=train_ap,
        test_pr_auc=test_ap,
        pr_auc_gap_train_minus_test=train_ap - test_ap,

        train_brier=train_brier,
        test_brier=test_brier,
        brier_gap_test_minus_train=test_brier - train_brier,

        train_log_loss=train_ll,
        test_log_loss=test_ll,
        logloss_gap_test_minus_train=test_ll - train_ll,

        test_mean_probability=float(np.mean(p_test)),
        test_median_probability=float(np.median(p_test)),
        test_min_probability=float(np.min(p_test)),
        test_max_probability=float(np.max(p_test)),
    )

    return metrics, p_train, p_test


def threshold_metrics(
    model_name: str,
    y_true: pd.Series,
    p: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    selected = tp + fp
    base_rate = float(y.mean())

    if selected:
        observed = float(tp / selected)
        selected_probs = p[p >= threshold]
        mean_selected = float(np.mean(selected_probs))
        median_selected = float(np.median(selected_probs))
    else:
        observed = float("nan")
        mean_selected = float("nan")
        median_selected = float("nan")

    lift = (
        observed / base_rate
        if selected and base_rate > 0
        else float("nan")
    )

    return ThresholdMetrics(
        model_name=model_name,
        threshold=threshold,
        selected=selected,
        selected_rate=float(selected / len(y)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        observed_win_rate=observed,
        mean_probability_selected=mean_selected,
        median_probability_selected=median_selected,
        lift_vs_test_base_rate=lift,
    )


# =============================================================================
# REPORTING
# =============================================================================

def dataframe_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # tabulate peut manquer ; fallback texte simple.
        return "```\n" + df.to_string(index=False) + "\n```"


def comparison_report(
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    variants: Mapping[str, Sequence[str]],
    removed_map: Mapping[str, Sequence[str]],
    dataset_path: Path,
    model_path: Path,
    schema_path: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> str:

    sort_auc = metrics_df.sort_values(
        "test_roc_auc",
        ascending=False,
    ).copy()

    sort_ap = metrics_df.sort_values(
        "test_pr_auc",
        ascending=False,
    ).copy()

    # Une variante n'est pas automatiquement promue.
    # On indique seulement le meilleur chiffre observé par métrique.
    best_auc_name = str(
        sort_auc.iloc[0]["model_name"]
    )
    best_ap_name = str(
        sort_ap.iloc[0]["model_name"]
    )

    cols_main = [
        "model_name",
        "n_features",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier",
        "test_log_loss",
        "auc_gap_train_minus_test",
    ]

    cols_threshold = [
        "model_name",
        "selected",
        "selected_rate",
        "observed_win_rate",
        "precision",
        "recall",
        "lift_vs_test_base_rate",
        "mean_probability_selected",
    ]

    lines = [
        "# COMPARAISON V2 vs V3 — GOAL AFTER 87'",
        "",
        "## Sources",
        f"- Modèle template V2 : `{model_path}`",
        f"- Schéma V2 : `{schema_path}`",
        f"- Dataset : `{dataset_path}`",
        "",
        "## Split temporel",
        f"- TRAIN : `{train_start}` -> `{train_end}`",
        f"- TEST : `{test_start}` -> `{test_end}`",
        "",
        "## Comparaison principale",
        "",
        dataframe_markdown(
            metrics_df[cols_main].sort_values(
                "test_roc_auc",
                ascending=False,
            )
        ),
        "",
        f"- Meilleure ROC-AUC observée : **{best_auc_name}**",
        f"- Meilleure PR-AUC observée : **{best_ap_name}**",
        "",
        "## Seuil >= 70 %",
        "",
        dataframe_markdown(
            threshold_df[cols_threshold].sort_values(
                ["observed_win_rate", "selected"],
                ascending=[False, False],
            )
        ),
        "",
        "Attention : un petit nombre de sélections >=70 % peut produire "
        "un taux observé très instable. Ce tableau ne suffit pas à lui seul "
        "pour choisir un modèle.",
        "",
        "## Features retirées",
        "",
    ]

    for name in variants:
        removed = list(removed_map.get(name, []))
        lines.append(
            f"### {name} — {len(variants[name])} features"
        )
        if removed:
            lines.extend([f"- `{x}`" for x in removed])
        else:
            lines.append("- aucune")
        lines.append("")

    lines.extend([
        "## Règle de décision recommandée",
        "",
        "Ne remplace pas automatiquement le V2 de production.",
        "Une V3 mérite une validation supplémentaire si elle améliore "
        "simultanément plusieurs critères hors-échantillon, notamment :",
        "",
        "- ROC-AUC et/ou PR-AUC;",
        "- Brier et LogLoss (plus bas = mieux);",
        "- stabilité TRAIN -> TEST;",
        "- comportement du groupe >=70 % avec un volume suffisant.",
        "",
        "Le dossier `candidates_refit_all` contient uniquement des modèles "
        "préparés pour une validation ultérieure. Aucun fichier de production "
        "n'est écrasé par ce script.",
    ])

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Réentraîne et compare V2, V3-A, V3-B et V3-C "
            "sur un split temporel strict."
        )
    )

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modèle V2 template.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Schéma exact des 52 features V2.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset historique CORE_READY V2.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Colonne cible. Défaut: auto-détection.",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Colonne temps. Défaut: kickoff_utc.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ALERT_THRESHOLD,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/model_v3_experiments",
    )
    parser.add_argument(
        "--no-refit-all",
        action="store_true",
        help=(
            "Ne sauvegarde pas les versions réentraînées "
            "sur 100 % du dataset."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()

    root = Path(args.project_root).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"Racine projet introuvable : {root}"
        )

    model_path = resolve_file(
        root,
        args.model,
        MODEL_CANDIDATES,
        "Modèle V2",
    )

    schema_path = resolve_file(
        root,
        args.schema,
        SCHEMA_CANDIDATES,
        "Schéma V2",
    )

    dataset_path = resolve_file(
        root,
        args.dataset,
        DATASET_CANDIDATES,
        "Dataset historique",
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    split_models_dir = output_dir / "split_models"
    refit_dir = output_dir / "candidates_refit_all"
    schema_dir = output_dir / "schemas"

    for d in [
        output_dir,
        split_models_dir,
        refit_dir,
        schema_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    log("=" * 84)
    log("COMPARAISON V2 vs V3 — GOAL AFTER 87'")
    log("=" * 84)
    log(f"Modèle template : {model_path}")
    log(f"Schéma          : {schema_path}")
    log(f"Dataset         : {dataset_path}")
    log(f"Sorties         : {output_dir}")
    log()

    production_model = joblib.load(model_path)

    base_features = load_feature_schema(
        schema_path,
        production_model,
    )

    log(
        f"Schéma V2 chargé : {len(base_features)} features"
    )

    if len(base_features) != 52:
        log(
            f"⚠️ Le modèle actuel contient {len(base_features)} features "
            "au lieu des 52 attendues. Le script continue avec le schéma réel."
        )

    variants: Dict[str, List[str]] = {
        "V2_BASELINE_52": list(base_features),
        "V3_A_CONSERVATIVE": validate_variant_features(
            base_features,
            V3_A_REMOVE,
            "V3_A_CONSERVATIVE",
        ),
        "V3_B_CLEAN": validate_variant_features(
            base_features,
            V3_B_REMOVE,
            "V3_B_CLEAN",
        ),
        "V3_C_AGGRESSIVE": validate_variant_features(
            base_features,
            V3_C_REMOVE,
            "V3_C_AGGRESSIVE",
        ),
    }

    removed_map: Dict[str, List[str]] = {
        "V2_BASELINE_52": [],
        "V3_A_CONSERVATIVE": V3_A_REMOVE,
        "V3_B_CLEAN": V3_B_REMOVE,
        "V3_C_AGGRESSIVE": V3_C_REMOVE,
    }

    for name, feats in variants.items():
        log(
            f"{name:<24s}: "
            f"{len(feats):>2d} features "
            f"({len(removed_map[name])} retirées)"
        )

    log()

    df = load_dataset(dataset_path)

    target_col = args.target or first_existing_column(
        df,
        TARGET_CANDIDATES,
    )

    if not target_col:
        raise ValueError(
            "Target introuvable. Utilise --target NOM."
        )

    log(f"Target : {target_col}")

    df, time_col, parse_mode = temporal_sort(
        df,
        args.time_col,
    )

    y = normalize_target(df[target_col])

    X_all = coerce_features(
        df,
        base_features,
    )

    if X_all.isna().any().any():
        missing_by_col = (
            X_all.isna()
            .sum()
            .sort_values(ascending=False)
        )
        missing_by_col = missing_by_col[
            missing_by_col > 0
        ]

        raise ValueError(
            "Le dataset CORE_READY contient des NaN dans les "
            "features du modèle. Le script refuse l'imputation "
            "silencieuse.\n"
            + missing_by_col.to_string()
        )

    train_idx, test_idx = make_split(
        len(df),
        args.test_fraction,
    )

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_start = df["__audit_time__"].iloc[
        train_idx[0]
    ].isoformat()

    train_end = df["__audit_time__"].iloc[
        train_idx[-1]
    ].isoformat()

    test_start = df["__audit_time__"].iloc[
        test_idx[0]
    ].isoformat()

    test_end = df["__audit_time__"].iloc[
        test_idx[-1]
    ].isoformat()

    log()
    log(
        f"TRAIN : {len(train_idx):,} lignes | "
        f"positifs={int(y_train.sum())} | "
        f"base={y_train.mean():.2%}"
    )
    log(
        f"  {train_start} -> {train_end}"
    )

    log(
        f"TEST  : {len(test_idx):,} lignes | "
        f"positifs={int(y_test.sum())} | "
        f"base={y_test.mean():.2%}"
    )
    log(
        f"  {test_start} -> {test_end}"
    )

    log()
    log("=" * 84)
    log("REENTRAINEMENT COMPARATIF")
    log("=" * 84)

    eval_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    fitted_split_models: Dict[str, Any] = {}

    match_id_col = first_existing_column(
        df,
        MATCH_ID_CANDIDATES,
    )

    for name, features in variants.items():
        log()
        log(
            f"[{name}] entraînement avec "
            f"{len(features)} features..."
        )

        X_train = X_all.loc[
            train_idx,
            features,
        ].copy()

        X_test = X_all.loc[
            test_idx,
            features,
        ].copy()

        candidate = clone_model_template(
            production_model
        )

        try:
            candidate.fit(
                X_train,
                y_train,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{name}: échec du fit avec le template V2. "
                "Aucune architecture alternative n'est substituée. "
                f"Erreur originale : {exc}"
            ) from exc

        fitted_split_models[name] = candidate

        metrics, p_train, p_test = evaluate_model(
            name,
            candidate,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        tm = threshold_metrics(
            name,
            y_test,
            p_test,
            args.threshold,
        )

        eval_rows.append(asdict(metrics))
        threshold_rows.append(asdict(tm))

        pred_df = pd.DataFrame({
            "model_name": name,
            "row_index_sorted": test_idx,
            "kickoff_time_utc": (
                df["__audit_time__"]
                .iloc[test_idx]
                .astype(str)
                .to_numpy()
            ),
            "target": y_test.to_numpy(),
            "probability": p_test,
            "selected_ge_threshold": (
                p_test >= args.threshold
            ).astype(int),
        })

        if match_id_col:
            pred_df.insert(
                2,
                "match_id",
                df[match_id_col]
                .iloc[test_idx]
                .astype(str)
                .to_numpy(),
            )

        prediction_frames.append(pred_df)

        model_out = (
            split_models_dir /
            f"{name}.joblib"
        )
        joblib.dump(
            candidate,
            model_out,
        )

        schema_out = (
            schema_dir /
            f"{name}_schema.json"
        )
        schema_out.write_text(
            json.dumps(
                {
                    "model_name": name,
                    "features": features,
                    "feature_count": len(features),
                    "removed_from_v2": removed_map[name],
                    "source_v2_schema": str(schema_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        log(
            f"  ROC-AUC={metrics.test_roc_auc:.6f} | "
            f"PR-AUC={metrics.test_pr_auc:.6f} | "
            f"Brier={metrics.test_brier:.6f} | "
            f"LogLoss={metrics.test_log_loss:.6f}"
        )

        log(
            f"  >= {args.threshold:.0%} : "
            f"{tm.selected} sélections | "
            f"taux observé="
            f"{tm.observed_win_rate:.2%}"
            if tm.selected
            else
            f"  >= {args.threshold:.0%} : "
            "0 sélection"
        )

    # -------------------------------------------------------------------------
    # TABLEAUX
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(eval_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    baseline_row = metrics_df[
        metrics_df["model_name"] == "V2_BASELINE_52"
    ].iloc[0]

    metrics_df["delta_test_roc_auc_vs_v2"] = (
        metrics_df["test_roc_auc"]
        - float(baseline_row["test_roc_auc"])
    )

    metrics_df["delta_test_pr_auc_vs_v2"] = (
        metrics_df["test_pr_auc"]
        - float(baseline_row["test_pr_auc"])
    )

    metrics_df["delta_test_brier_vs_v2"] = (
        metrics_df["test_brier"]
        - float(baseline_row["test_brier"])
    )

    metrics_df["delta_test_log_loss_vs_v2"] = (
        metrics_df["test_log_loss"]
        - float(baseline_row["test_log_loss"])
    )

    metrics_df = metrics_df.sort_values(
        "test_roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    threshold_df = threshold_df.sort_values(
        ["observed_win_rate", "selected"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    metrics_path = output_dir / "comparison_metrics.csv"
    threshold_path = output_dir / "threshold_070_metrics.csv"
    predictions_path = output_dir / "predictions_test.csv"
    feature_sets_path = output_dir / "feature_sets.json"
    report_path = output_dir / "comparison_report.md"
    metadata_path = output_dir / "experiment_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    feature_sets_path.write_text(
        json.dumps(
            {
                name: {
                    "feature_count": len(features),
                    "features": features,
                    "removed_from_v2": removed_map[name],
                }
                for name, features in variants.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = comparison_report(
        metrics_df=metrics_df,
        threshold_df=threshold_df,
        variants=variants,
        removed_map=removed_map,
        dataset_path=dataset_path,
        model_path=model_path,
        schema_path=schema_path,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # REFIT ALL — candidats seulement
    # -------------------------------------------------------------------------
    if not args.no_refit_all:
        log()
        log("=" * 84)
        log("REFIT ALL — MODELES CANDIDATS (PAS DE DEPLOIEMENT AUTOMATIQUE)")
        log("=" * 84)

        for name, features in variants.items():
            log(
                f"{name}: refit sur {len(df):,} lignes..."
            )

            model_all = clone_model_template(
                production_model
            )

            X_variant_all = X_all.loc[
                :,
                features,
            ].copy()

            model_all.fit(
                X_variant_all,
                y,
            )

            out = (
                refit_dir /
                f"{name}_refit_all.joblib"
            )

            joblib.dump(
                model_all,
                out,
            )

    metadata = {
        "production_model_template": str(model_path),
        "schema_path": str(schema_path),
        "dataset_path": str(dataset_path),
        "target_column": target_col,
        "time_column": time_col,
        "time_parse_mode": parse_mode,
        "test_fraction": args.test_fraction,
        "threshold": args.threshold,
        "rows_total": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "variants": {
            name: {
                "feature_count": len(features),
                "removed_from_v2": removed_map[name],
            }
            for name, features in variants.items()
        },
        "refit_all_generated": not args.no_refit_all,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # -------------------------------------------------------------------------
    log()
    log("=" * 84)
    log("COMPARAISON TERMINÉE")
    log("=" * 84)

    display_cols = [
        "model_name",
        "n_features",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier",
        "test_log_loss",
        "delta_test_roc_auc_vs_v2",
        "delta_test_pr_auc_vs_v2",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            metrics_df[display_cols]
            .to_string(index=False)
        )

    log()
    log(f"Seuil analysé : >= {args.threshold:.0%}")

    threshold_display = threshold_df[
        [
            "model_name",
            "selected",
            "observed_win_rate",
            "recall",
            "lift_vs_test_base_rate",
        ]
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            threshold_display.to_string(
                index=False
            )
        )

    log()
    log("Fichiers générés :")
    for p in [
        metrics_path,
        threshold_path,
        predictions_path,
        feature_sets_path,
        report_path,
        metadata_path,
    ]:
        log(f"  - {p}")

    log()
    log(
        "AUC plus haute ≠ déploiement automatique. "
        "Lis comparison_report.md avant toute modification du scanner."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrompu par l'utilisateur.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERREUR COMPARAISON V2/V3 : {exc}",
            file=sys.stderr,
        )
        raise
