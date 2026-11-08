#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FIRST HALF 37 — FEATURE SUBSET + MODEL COMPARISON (V1)
================================================================================

But
---
Comparer, SANS ouvrir le FINAL HOLDOUT, plusieurs tailles de schéma et
plusieurs familles de modèles sur les mêmes folds temporels de DEVELOPMENT.

Cette étape vient APRES :
    audit_select_first_half_37_features.py

Comparaisons principales
------------------------
- TOP_10
- TOP_15
- TOP_20
- TOP_30
- TOP_40
- ABLATION_STABLE  : familles dont l'ablation est positive et stable
- ABLATION_POSITIVE: familles dont l'effet moyen d'ablation est positif

Modèles sklearn uniquement
--------------------------
- logistic_l2
- logistic_l2_balanced
- hist_gradient_boosting
- hist_gradient_boosting_balanced
- extra_trees_balanced

Principes
---------
- Le FINAL HOLDOUT (15 % les plus récents, par défaut) est réservé.
- AUC / PR-AUC / Brier / LogLoss calculés uniquement sur DEVELOPMENT.
- Validation : TimeSeriesSplit expansif, aucun shuffle.
- Missing != 0 :
    * HGB garde les NaN nativement.
    * Logistic / ExtraTrees utilisent une médiane apprise sur TRAIN du fold.
- Baseline par fold = probabilité constante égale au taux positif TRAIN.
- Les probabilités du FINAL HOLDOUT ne sont JAMAIS calculées ici.
- Les métriques internes servent à choisir un pipeline candidat, pas à annoncer
  une performance finale.

Sorties
-------
outputs/first_half_37/model_experiments_v1/
    first_half_37_subset_definitions.json
    first_half_37_fold_metrics.csv
    first_half_37_model_summary.csv
    first_half_37_candidate_ranking.csv
    first_half_37_model_experiment_report.md
    first_half_37_winner_manifest.json
    candidates_dev_only/*.joblib
    schemas/*.json

Commande
--------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 compare_first_half_37_models.py

IMPORTANT
---------
Le script ne touche pas au scanner Goal After 87 et n'évalue pas le holdout
final. Si aucun pipeline ne passe les gates de stabilité, le rapport le dira
clairement au lieu de forcer un "winner".
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

DEFAULT_FEATURE_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "feature_selection"
)

DEFAULT_MODEL_READY = DEFAULT_FEATURE_DIR / "First_Half_37_MODEL_READY_v1.csv"
DEFAULT_SELECTED = DEFAULT_FEATURE_DIR / "first_half_37_selected_features.csv"
DEFAULT_ABLATION = DEFAULT_FEATURE_DIR / "first_half_37_group_ablation.csv"

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "model_experiments_v1"
)

TARGET = "goal_after_37_before_ht"
TIME_COL = "kickoff_utc"
ID_COL = "match_id"
SOURCE_COL = "source_dataset"

TOP_SIZES = (10, 15, 20, 30, 40)


@dataclass
class SplitInfo:
    total_rows: int
    development_rows: int
    holdout_rows: int
    development_start: str
    development_end: str
    holdout_start: str
    holdout_end: str
    holdout_fraction: float


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def mixed_to_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO timestamps robustly across pandas versions."""
    try:
        out = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        out = pd.to_datetime(series, utc=True, errors="coerce")
    return out


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metric_bundle(y_true: pd.Series, prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)

    if len(np.unique(y)) < 2:
        raise ValueError("Bloc d'évaluation sans les deux classes.")

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(np.mean(p)),
    }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MODEL_READY introuvable : {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {ID_COL, TIME_COL, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {sorted(missing)}")

    dt = mixed_to_datetime(df[TIME_COL])
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(
            f"{bad} kickoff_utc non parsables. Refus de continuer pour protéger le split temporel."
        )

    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Target invalide ou non résolu dans MODEL_READY.")

    df = df.copy()
    df["__kickoff__"] = dt
    df["__target__"] = target.astype(int)

    if df[ID_COL].astype(str).duplicated().any():
        dup = int(df[ID_COL].astype(str).duplicated().sum())
        raise ValueError(f"{dup} match_id dupliqués. Refus de continuer.")

    df = df.sort_values(["__kickoff__", ID_COL], kind="stable").reset_index(drop=True)
    return df


def split_development_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    if not (0.10 <= holdout_fraction <= 0.30):
        raise ValueError("holdout_fraction doit être entre 0.10 et 0.30")

    n = len(df)
    holdout_n = max(1, int(round(n * holdout_fraction)))
    dev_n = n - holdout_n

    if dev_n < 500:
        raise ValueError(f"DEVELOPMENT trop petit : {dev_n}")

    dev = df.iloc[:dev_n].copy().reset_index(drop=True)
    hold = df.iloc[dev_n:].copy().reset_index(drop=True)

    info = SplitInfo(
        total_rows=n,
        development_rows=len(dev),
        holdout_rows=len(hold),
        development_start=dev["__kickoff__"].iloc[0].isoformat(),
        development_end=dev["__kickoff__"].iloc[-1].isoformat(),
        holdout_start=hold["__kickoff__"].iloc[0].isoformat(),
        holdout_end=hold["__kickoff__"].iloc[-1].isoformat(),
        holdout_fraction=float(holdout_fraction),
    )
    return dev, hold, info


def load_ranked_features(path: Path, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"selected_features introuvable : {path}")

    tab = pd.read_csv(path)
    if "feature" not in tab.columns:
        raise ValueError("selected_features.csv doit contenir la colonne 'feature'.")

    if "selected" in tab.columns:
        tab = tab[pd.to_numeric(tab["selected"], errors="coerce").fillna(0).astype(int).eq(1)].copy()

    if "rank" in tab.columns:
        tab = tab.sort_values("rank", kind="stable")
    elif "selection_score" in tab.columns:
        tab = tab.sort_values("selection_score", ascending=False, kind="stable")

    features = []
    seen = set()
    for f in tab["feature"].astype(str):
        if f in df.columns and f not in seen:
            features.append(f)
            seen.add(f)

    if len(features) < 10:
        raise ValueError(f"Seulement {len(features)} features sélectionnées disponibles.")

    return features, tab


def load_ablation_families(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_subsets(
    ranked_features: Sequence[str],
    selected_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
) -> Dict[str, List[str]]:
    subsets: Dict[str, List[str]] = {}

    for n in TOP_SIZES:
        if len(ranked_features) >= n:
            subsets[f"TOP_{n}"] = list(ranked_features[:n])

    # Family-aware subsets derived from the prior ablation report.
    if (
        not selected_table.empty
        and "feature" in selected_table.columns
        and "family" in selected_table.columns
        and not ablation_table.empty
        and "family" in ablation_table.columns
        and "mean_delta_pr_auc" in ablation_table.columns
    ):
        ab = ablation_table.copy()
        ab["mean_delta_pr_auc"] = pd.to_numeric(ab["mean_delta_pr_auc"], errors="coerce")
        if "positive_pr_fold_fraction" in ab.columns:
            ab["positive_pr_fold_fraction"] = pd.to_numeric(
                ab["positive_pr_fold_fraction"], errors="coerce"
            )
        else:
            ab["positive_pr_fold_fraction"] = np.nan

        positive_families = set(
            ab.loc[ab["mean_delta_pr_auc"] > 0, "family"].astype(str)
        )
        stable_families = set(
            ab.loc[
                (ab["mean_delta_pr_auc"] > 0)
                & (ab["positive_pr_fold_fraction"].fillna(0) >= 0.50),
                "family",
            ].astype(str)
        )

        rank_map = {f: i for i, f in enumerate(ranked_features)}
        rows = selected_table.copy()
        rows["feature"] = rows["feature"].astype(str)
        rows["family"] = rows["family"].astype(str)
        rows = rows[rows["feature"].isin(ranked_features)]
        rows["__rank__"] = rows["feature"].map(rank_map)
        rows = rows.sort_values("__rank__", kind="stable")

        pos = rows.loc[rows["family"].isin(positive_families), "feature"].tolist()
        stable = rows.loc[rows["family"].isin(stable_families), "feature"].tolist()

        if len(pos) >= 5:
            subsets["ABLATION_POSITIVE"] = pos
        if len(stable) >= 5:
            subsets["ABLATION_STABLE"] = stable

    # Deduplicate identical feature sets while preserving first canonical name.
    unique: Dict[Tuple[str, ...], str] = {}
    final: Dict[str, List[str]] = {}
    for name, feats in subsets.items():
        key = tuple(feats)
        if key in unique:
            continue
        unique[key] = name
        final[name] = feats

    return final


# =============================================================================
# MODELS
# =============================================================================

def build_model(name: str, random_state: int):
    if name == "logistic_l2":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]), False

    if name == "logistic_l2_balanced":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]), False

    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.5,
            early_stopping=False,
            random_state=random_state,
        ), False

    if name == "hist_gradient_boosting_balanced":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.5,
            early_stopping=False,
            random_state=random_state,
        ), True

    if name == "extra_trees_balanced":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]), False

    raise KeyError(name)


MODEL_NAMES = (
    "logistic_l2",
    "logistic_l2_balanced",
    "hist_gradient_boosting",
    "hist_gradient_boosting_balanced",
    "extra_trees_balanced",
)


def fit_with_optional_balancing(model, X, y, use_sample_weight: bool):
    if use_sample_weight:
        sw = compute_sample_weight("balanced", y)
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return model


# =============================================================================
# TEMPORAL COMPARISON
# =============================================================================

def compare_configs(
    dev: pd.DataFrame,
    subsets: Dict[str, List[str]],
    n_splits: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    split_indices = list(splitter.split(dev))

    rows = []

    for subset_idx, (subset_name, features) in enumerate(subsets.items(), start=1):
        log("-")
        log(f"SUBSET {subset_name} | n_features={len(features)}")

        X_all = dev.loc[:, features].apply(pd.to_numeric, errors="coerce")
        y_all = dev["__target__"].astype(int)

        for model_idx, model_name in enumerate(MODEL_NAMES, start=1):
            fold_pr = []
            for fold, (train_idx, val_idx) in enumerate(split_indices, start=1):
                X_train = X_all.iloc[train_idx]
                X_val = X_all.iloc[val_idx]
                y_train = y_all.iloc[train_idx]
                y_val = y_all.iloc[val_idx]

                if y_train.nunique() < 2 or y_val.nunique() < 2:
                    raise ValueError(f"Fold {fold} sans les deux classes.")

                model, use_sw = build_model(
                    model_name,
                    random_state=random_state + subset_idx * 100 + model_idx * 10 + fold,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_with_optional_balancing(model, X_train, y_train, use_sw)

                prob = model.predict_proba(X_val)[:, 1]
                met = metric_bundle(y_val, prob)

                train_rate = float(y_train.mean())
                val_rate = float(y_val.mean())
                base_prob = np.full(len(y_val), train_rate, dtype=float)
                base = metric_bundle(y_val, base_prob)

                row = {
                    "subset": subset_name,
                    "n_features": len(features),
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_idx),
                    "val_rows": len(val_idx),
                    "train_start": dev["__kickoff__"].iloc[train_idx[0]].isoformat(),
                    "train_end": dev["__kickoff__"].iloc[train_idx[-1]].isoformat(),
                    "val_start": dev["__kickoff__"].iloc[val_idx[0]].isoformat(),
                    "val_end": dev["__kickoff__"].iloc[val_idx[-1]].isoformat(),
                    "train_positive_rate": train_rate,
                    "val_positive_rate": val_rate,
                    **met,
                    "baseline_roc_auc": base["roc_auc"],
                    "baseline_pr_auc": base["pr_auc"],
                    "baseline_brier": base["brier"],
                    "baseline_log_loss": base["log_loss"],
                    "pr_lift_vs_base": met["pr_auc"] - base["pr_auc"],
                    "roc_edge_vs_random": met["roc_auc"] - 0.5,
                    "brier_improvement_vs_base": base["brier"] - met["brier"],
                    "logloss_improvement_vs_base": base["log_loss"] - met["log_loss"],
                }
                rows.append(row)
                fold_pr.append(met["pr_auc"])

            log(
                f"  {model_name:<32} "
                f"mean_PR={np.mean(fold_pr):.4f}"
            )

    fold_df = pd.DataFrame(rows)

    def frac_pos(s: pd.Series) -> float:
        return float(np.mean(np.asarray(s, dtype=float) > 0))

    summary = (
        fold_df.groupby(["subset", "n_features", "model"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            mean_roc_auc=("roc_auc", "mean"),
            median_roc_auc=("roc_auc", "median"),
            min_roc_auc=("roc_auc", "min"),
            mean_pr_auc=("pr_auc", "mean"),
            median_pr_auc=("pr_auc", "median"),
            min_pr_auc=("pr_auc", "min"),
            mean_val_positive_rate=("val_positive_rate", "mean"),
            mean_pr_lift_vs_base=("pr_lift_vs_base", "mean"),
            median_pr_lift_vs_base=("pr_lift_vs_base", "median"),
            worst_pr_lift_vs_base=("pr_lift_vs_base", "min"),
            pr_above_base_fold_fraction=("pr_lift_vs_base", frac_pos),
            roc_above_050_fold_fraction=("roc_edge_vs_random", frac_pos),
            mean_brier=("brier", "mean"),
            mean_brier_improvement_vs_base=("brier_improvement_vs_base", "mean"),
            brier_better_fold_fraction=("brier_improvement_vs_base", frac_pos),
            mean_log_loss=("log_loss", "mean"),
            mean_logloss_improvement_vs_base=("logloss_improvement_vs_base", "mean"),
            logloss_better_fold_fraction=("logloss_improvement_vs_base", frac_pos),
        )
    )

    # Strict stability gate. This is intentionally conservative.
    summary["passes_stability_gate"] = (
        (summary["mean_pr_lift_vs_base"] > 0)
        & (summary["pr_above_base_fold_fraction"] >= 0.75)
        & (summary["mean_roc_auc"] > 0.50)
        & (summary["roc_above_050_fold_fraction"] >= 0.50)
        & (summary["mean_brier_improvement_vs_base"] > 0)
    ).astype(int)

    # Ranking is lexicographic in spirit, encoded for convenient sorting.
    # PR lift/stability dominate; complexity gets only a tiny tie-break penalty.
    summary["ranking_score"] = (
        4.0 * summary["mean_pr_lift_vs_base"]
        + 2.0 * summary["median_pr_lift_vs_base"]
        + 1.0 * summary["worst_pr_lift_vs_base"]
        + 0.75 * (summary["mean_roc_auc"] - 0.50)
        + 0.50 * summary["mean_brier_improvement_vs_base"]
        + 0.10 * summary["mean_logloss_improvement_vs_base"]
        + 0.02 * summary["pr_above_base_fold_fraction"]
        - 0.00005 * summary["n_features"]
    )

    summary = summary.sort_values(
        [
            "passes_stability_gate",
            "pr_above_base_fold_fraction",
            "mean_pr_lift_vs_base",
            "mean_roc_auc",
            "mean_brier_improvement_vs_base",
            "ranking_score",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return fold_df, summary


# =============================================================================
# SAVE DEVELOPMENT CANDIDATES
# =============================================================================

def fit_and_save_top_candidates(
    dev: pd.DataFrame,
    subsets: Dict[str, List[str]],
    summary: pd.DataFrame,
    output_dir: Path,
    random_state: int,
    top_k: int = 3,
) -> List[Dict]:
    model_dir = output_dir / "candidates_dev_only"
    schema_dir = output_dir / "schemas"
    model_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    top = summary.head(top_k)

    for i, row in top.iterrows():
        subset_name = str(row["subset"])
        model_name = str(row["model"])
        features = subsets[subset_name]

        X = dev.loc[:, features].apply(pd.to_numeric, errors="coerce")
        y = dev["__target__"].astype(int)

        model, use_sw = build_model(model_name, random_state + 9000 + int(i))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_with_optional_balancing(model, X, y, use_sw)

        stem = f"rank_{int(row['rank']):02d}__{subset_name}__{model_name}"
        model_path = model_dir / f"{stem}.joblib"
        schema_path = schema_dir / f"{stem}_schema.json"

        joblib.dump(model, model_path)
        json_dump(
            schema_path,
            {
                "schema_version": "first_half_37_model_experiments_v1",
                "status": "DEVELOPMENT_ONLY_NOT_FINAL_EVALUATED",
                "subset": subset_name,
                "model": model_name,
                "features": features,
                "n_features": len(features),
                "target": TARGET,
                "snapshot_minute": 37,
                "development_rows": len(dev),
                "development_start": dev["__kickoff__"].iloc[0].isoformat(),
                "development_end": dev["__kickoff__"].iloc[-1].isoformat(),
                "internal_rank": int(row["rank"]),
                "passes_stability_gate": int(row["passes_stability_gate"]),
            },
        )

        artifacts.append({
            "rank": int(row["rank"]),
            "subset": subset_name,
            "model": model_name,
            "n_features": len(features),
            "model_path": str(model_path),
            "schema_path": str(schema_path),
        })

    return artifacts


# =============================================================================
# REPORT
# =============================================================================

def dataframe_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_Aucune ligne._"
    view = df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def make_report(
    split: SplitInfo,
    subsets: Dict[str, List[str]],
    summary: pd.DataFrame,
    fold_df: pd.DataFrame,
    artifacts: List[Dict],
    output_path: Path,
) -> None:
    best = summary.iloc[0]
    eligible = summary[summary["passes_stability_gate"].eq(1)]

    top_cols = [
        "rank", "subset", "n_features", "model", "passes_stability_gate",
        "mean_roc_auc", "mean_pr_auc", "mean_val_positive_rate",
        "mean_pr_lift_vs_base", "pr_above_base_fold_fraction",
        "mean_brier", "mean_brier_improvement_vs_base",
        "mean_log_loss", "mean_logloss_improvement_vs_base",
        "worst_pr_lift_vs_base",
    ]

    lines = []
    lines.append("# First Half 37 — Feature Subset + Model Comparison V1")
    lines.append("")
    lines.append("## Protection du FINAL HOLDOUT")
    lines.append("")
    lines.append(f"- Dataset total : **{split.total_rows}**")
    lines.append(f"- DEVELOPMENT évalué ici : **{split.development_rows}**")
    lines.append(f"- FINAL HOLDOUT réservé : **{split.holdout_rows}**")
    lines.append(f"- DEVELOPMENT : `{split.development_start}` -> `{split.development_end}`")
    lines.append(f"- FINAL HOLDOUT : `{split.holdout_start}` -> `{split.holdout_end}`")
    lines.append("- **Aucune prédiction ni métrique du FINAL HOLDOUT n'est calculée dans ce script.**")
    lines.append("")

    lines.append("## Sous-ensembles comparés")
    lines.append("")
    for name, feats in subsets.items():
        lines.append(f"- `{name}` : **{len(feats)}** features")
    lines.append("")

    lines.append("## Ranking interne DEVELOPMENT")
    lines.append("")
    lines.append(dataframe_markdown(summary[top_cols], max_rows=30))
    lines.append("")

    lines.append("## Gate de stabilité")
    lines.append("")
    lines.append(
        "Un pipeline passe le gate seulement si : PR-AUC moyenne > baseline, "
        "PR-AUC > baseline sur au moins 75% des folds, ROC-AUC moyenne > 0.50, "
        "ROC > 0.50 sur au moins la moitié des folds, et Brier moyen meilleur "
        "que la baseline constante."
    )
    lines.append("")

    if eligible.empty:
        lines.append("**AUCUN pipeline ne passe le gate de stabilité.**")
        lines.append("")
        lines.append(
            "Le rang #1 reste exploratoire. Ne pas ouvrir le FINAL HOLDOUT : "
            "il faut d'abord resserrer/revoir le schéma ou les hyperparamètres."
        )
    else:
        lines.append(f"Pipelines éligibles : **{len(eligible)}**")
        lines.append("")
        lines.append(
            f"Candidat #1 : `{best['subset']} + {best['model']}` "
            f"({int(best['n_features'])} features)."
        )
        lines.append("")
        lines.append(
            "Même si un candidat passe le gate, ces performances restent des "
            "métriques DEVELOPMENT. Le FINAL HOLDOUT doit être ouvert une seule "
            "fois, dans l'étape suivante, après verrouillage du pipeline."
        )
    lines.append("")

    lines.append("## Détail folds du candidat #1")
    lines.append("")
    detail = fold_df[
        fold_df["subset"].eq(str(best["subset"]))
        & fold_df["model"].eq(str(best["model"]))
    ].copy()
    detail_cols = [
        "fold", "train_rows", "val_rows", "train_positive_rate",
        "val_positive_rate", "roc_auc", "pr_auc", "pr_lift_vs_base",
        "brier", "brier_improvement_vs_base", "log_loss",
        "logloss_improvement_vs_base",
    ]
    lines.append(dataframe_markdown(detail[detail_cols], max_rows=10))
    lines.append("")

    lines.append("## Candidats refit sur DEVELOPMENT uniquement")
    lines.append("")
    for a in artifacts:
        lines.append(
            f"- rank {a['rank']}: `{a['subset']} + {a['model']}` — "
            f"{a['n_features']} features — **DEV ONLY**"
        )
    lines.append("")
    lines.append("Ces `.joblib` ne sont pas des modèles production.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="First Half 37 subset/model comparison")
    p.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    p.add_argument("--selected-features", type=Path, default=DEFAULT_SELECTED)
    p.add_argument("--group-ablation", type=Path, default=DEFAULT_ABLATION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--random-seed", type=int, default=87)
    p.add_argument("--top-candidates-to-save", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 108)
    log("FIRST HALF 37 — FEATURE SUBSET + MODEL COMPARISON V1")
    log("=" * 108)
    log(f"MODEL_READY : {args.model_ready}")
    log(f"Selected    : {args.selected_features}")
    log(f"Ablation    : {args.group_ablation}")
    log(f"Output      : {args.output_dir}")
    log("")

    df = load_dataset(args.model_ready)
    dev, hold, split = split_development_holdout(df, args.holdout_fraction)

    ranked, selected_table = load_ranked_features(args.selected_features, df)
    ablation = load_ablation_families(args.group_ablation)
    subsets = make_subsets(ranked, selected_table, ablation)

    if not subsets:
        raise RuntimeError("Aucun sous-ensemble de features disponible.")

    log(
        f"Dataset={len(df)} | positives={int(df['__target__'].sum())} "
        f"({df['__target__'].mean()*100:.2f}%)"
    )
    log(
        f"DEVELOPMENT={len(dev)} | "
        f"{dev['__kickoff__'].iloc[0].isoformat()} -> {dev['__kickoff__'].iloc[-1].isoformat()}"
    )
    log(
        f"FINAL HOLDOUT RESERVE={len(hold)} | "
        f"{hold['__kickoff__'].iloc[0].isoformat()} -> {hold['__kickoff__'].iloc[-1].isoformat()}"
    )
    log("FINAL HOLDOUT : aucune prediction ne sera calculee.")
    log("")

    log("Sous-ensembles :")
    for name, feats in subsets.items():
        log(f"  {name:<22} {len(feats):>3} features")
    log("")

    fold_df, summary = compare_configs(
        dev=dev,
        subsets=subsets,
        n_splits=args.n_splits,
        random_state=args.random_seed,
    )

    # Save core tables first.
    fold_path = args.output_dir / "first_half_37_fold_metrics.csv"
    summary_path = args.output_dir / "first_half_37_model_summary.csv"
    rank_path = args.output_dir / "first_half_37_candidate_ranking.csv"
    subset_path = args.output_dir / "first_half_37_subset_definitions.json"

    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(rank_path, index=False)
    json_dump(subset_path, {k: v for k, v in subsets.items()})

    artifacts = fit_and_save_top_candidates(
        dev=dev,
        subsets=subsets,
        summary=summary,
        output_dir=args.output_dir,
        random_state=args.random_seed,
        top_k=max(1, args.top_candidates_to_save),
    )

    best = summary.iloc[0]
    eligible = summary[summary["passes_stability_gate"].eq(1)]
    status = "ELIGIBLE_FOR_FINAL_HOLDOUT" if not eligible.empty else "NO_STABLE_WINNER_YET"

    manifest = {
        "schema_version": "first_half_37_model_experiments_v1",
        "status": status,
        "split": asdict(split),
        "final_holdout_evaluated": False,
        "models_compared": list(MODEL_NAMES),
        "subsets": {k: v for k, v in subsets.items()},
        "best_internal_candidate": {
            "rank": int(best["rank"]),
            "subset": str(best["subset"]),
            "model": str(best["model"]),
            "n_features": int(best["n_features"]),
            "passes_stability_gate": int(best["passes_stability_gate"]),
            "mean_roc_auc": float(best["mean_roc_auc"]),
            "mean_pr_auc": float(best["mean_pr_auc"]),
            "mean_pr_lift_vs_base": float(best["mean_pr_lift_vs_base"]),
            "pr_above_base_fold_fraction": float(best["pr_above_base_fold_fraction"]),
            "mean_brier": float(best["mean_brier"]),
            "mean_brier_improvement_vs_base": float(best["mean_brier_improvement_vs_base"]),
        },
        "eligible_candidates": int(len(eligible)),
        "saved_candidates": artifacts,
    }
    manifest_path = args.output_dir / "first_half_37_winner_manifest.json"
    json_dump(manifest_path, manifest)

    report_path = args.output_dir / "first_half_37_model_experiment_report.md"
    make_report(split, subsets, summary, fold_df, artifacts, report_path)

    log("")
    log("=" * 108)
    log("COMPARAISON FIRST HALF 37 TERMINEE")
    log("=" * 108)
    log(f"Configurations testees : {len(summary)}")
    log(f"Pipelines passant le gate : {len(eligible)}")
    log(
        f"#1 interne : {best['subset']} + {best['model']} | "
        f"features={int(best['n_features'])} | "
        f"ROC={best['mean_roc_auc']:.4f} | PR={best['mean_pr_auc']:.4f} | "
        f"PR lift={best['mean_pr_lift_vs_base']:+.4f} | "
        f"folds PR>base={best['pr_above_base_fold_fraction']:.0%} | "
        f"gate={'PASS' if int(best['passes_stability_gate']) else 'FAIL'}"
    )
    log(f"Rapport  : {report_path}")
    log(f"Ranking  : {rank_path}")
    log(f"Manifest : {manifest_path}")
    log("")
    log("IMPORTANT : FINAL HOLDOUT TOUJOURS INTACT ET NON EVALUE.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
