#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FIRST HALF 37 — AUDIT + REAL FEATURE SELECTION (V1)
================================================================================

Objectif
--------
Auditer le dataset First_Half37 et sélectionner un schéma de features robuste
SANS utiliser le holdout final pour choisir les variables.

Principes
---------
- Snapshot predictor strictement à 37'.
- Target : goal_after_37_before_ht.
- Missing != 0 : aucune imputation silencieuse vers zéro.
- Parsing temporel robuste aux formats ISO mixtes.
- Déduplication par match_id.
- Rejet des colonnes de fuite directe / métadonnées de résolution du target.
- Contrôle de couverture et de dérive de disponibilité seed vs API backfill.
- Holdout final temporel réservé et NON évalué pendant la sélection.
- Validation interne par folds temporels expansifs.
- Sélection multi-signal :
    1) permutation importance hors-échantillon,
    2) stabilité L1 hors-échantillon,
    3) ablation par famille de features,
    4) suppression des redondances fortes sur DEV uniquement.
- Aucun modèle final de production n'est entraîné ici.

Sorties principales
-------------------
outputs/first_half_37/feature_selection/
    first_half_37_audit_report.md
    first_half_37_feature_audit.csv
    first_half_37_source_shift.csv
    first_half_37_temporal_fold_metrics.csv
    first_half_37_temporal_feature_scores.csv
    first_half_37_group_ablation.csv
    first_half_37_correlated_pairs.csv
    first_half_37_selected_features.csv
    first_half_37_selected_schema.json
    first_half_37_split_manifest.json
    First_Half_37_MODEL_READY_v1.csv

IMPORTANT
---------
Le fichier MODEL_READY contient toutes les lignes, y compris le holdout final,
mais le holdout n'est PAS utilisé pour sélectionner les features. Il devra
rester intact jusqu'à l'évaluation finale du modèle gagnant.

Commande standard
-----------------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 audit_select_first_half_37_features.py

Version plus rapide (moins de répétitions permutation, sans ablation) :
python3 audit_select_first_half_37_features.py \
  --permutation-repeats 2 \
  --skip-ablation
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
    )
)

DEFAULT_DATASET = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "first_half_37"
    / "historical_api"
    / "First_Half_37_COMBINED.csv"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "first_half_37"
    / "feature_selection"
)

TARGET = "goal_after_37_before_ht"
TIME_COL = "kickoff_utc"
ID_COL = "match_id"
SOURCE_COL = "source_dataset"

# Direct leakage, label-resolution metadata, IDs/names, provenance and
# availability counters are NEVER candidates for model features.
HARD_EXCLUDE = {
    "schema_version",
    "snapshot_minute",
    "match_id",
    "competition_id",
    "season_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "source_dataset",
    "collected_at_utc",

    # Target + direct post-37 information.
    TARGET,
    "target_status",
    "target_resolution_class",
    "target_confidence",
    "goal_minutes_after_37_before_ht",
    "first_half_period_end_seen",
    "timeline_full",
    "first_half_goal_count",

    # Pure data/provenance metadata — not football signal.
    "fh_odds_bookmakers_available",
    "_prematch_available",
    "_prematch_total",
    "_live37_available",
    "_fh_odds_available",
    "_fh_odds_total",
    "_live_provenance_json",
    "_live37_total_fixed_schema",
    "_live37_core_available",
    "_live37_core_total",
    "_live37_xg_available",
    "_live37_xg_total",
    "prematch_rebuild_policy",
    "prematch_cutoff_utc",
    "_prematch_home_history_before",
    "_prematch_away_history_before",
    "_prematch_league_history_before",
    "_prematch_h2h_before",
}

# *_n = nombre de bookmakers / lignes de marché. Cela mesure surtout la
# disponibilité de source et peut révéler seed vs backfill.
EXCLUDE_SUFFIXES = ("_n",)

# Features explicitement indésirables même si elles existent.
# La variable de 2e mi-temps est hors objectif First_Half37 et peut être très
# incomplète; on l'exclut du schéma principal.
EXPLICIT_SEMANTIC_EXCLUDE = {
    "league_2nd_half_over15_freq",
}


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def mixed_to_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO timestamps robustly across forms with/without milliseconds."""
    try:
        out = pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except TypeError:
        # Fallback older pandas.
        out = pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )
    return out


def json_dump(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def dataframe_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    view = df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def valid_binary_target(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    bad = ~y.isin([0, 1])
    if bad.any():
        examples = series[bad].head(10).tolist()
        raise ValueError(
            f"Target non binaire/invalide sur {int(bad.sum())} lignes. "
            f"Exemples={examples}"
        )
    return y.astype(int)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def metric_bundle(y_true: pd.Series, prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)

    out = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "log_loss": float("nan"),
    }

    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, p_clip, labels=[0, 1]))

    return out


def family_of(feature: str) -> str:
    f = feature.lower()

    if f.startswith("odds_fh_"):
        return "first_half_odds"

    if f.startswith("odds_"):
        return "prematch_odds"

    if (
        f.startswith("score_37_")
        or f in {
            "leading_team_37",
            "goal_diff_37",
            "is_draw_37",
            "total_goals_37",
            "has_goal_37",
        }
    ):
        return "score_state"

    if "xg_" in f or f.endswith("_xg") or f.startswith("xg_"):
        if any(w in f for w in ("30_37", "33_37", "35_37")):
            return "xg_recent"
        return "xg_cumulative"

    if (
        "last_shot" in f
        or "minutes_since_last_shot" in f
        or "last_corner" in f
        or "minutes_since_last_corner" in f
        or "last_goal" in f
        or "minutes_since_last_goal" in f
    ):
        return "recency"

    if "corner" in f:
        return "corners_pressure"

    if (
        "shots_" in f
        or "sot_" in f
        or "shot_" in f
        or "shots_on_target" in f
        or "shots_off_target" in f
        or "shots_blocked" in f
    ):
        if any(w in f for w in ("30_37", "33_37", "35_37")):
            return "shots_recent_pressure"
        return "shots_cumulative"

    if (
        "foul" in f
        or "card" in f
        or "yellow" in f
        or "red_card" in f
        or "offsides" in f
        or "subs_" in f
    ):
        return "discipline_and_events"

    if f.startswith("h2h") or f.startswith("league_"):
        return "prematch_h2h_league"

    if any(
        key in f
        for key in (
            "form5",
            "form10",
            "scored_freq",
            "wins_freq",
            "avg_goals_scored",
            "avg_goals_conceded",
            "btts_freq",
            "over25_freq",
        )
    ):
        return "prematch_team_strength"

    if any(
        key in f
        for key in (
            "rank",
            "position",
            "points",
            "matches_played",
            "goals_for",
            "goals_against",
            "goal_difference",
            "home_team_wins",
            "away_team_wins",
            "home_team_draws",
            "away_team_draws",
            "home_team_losses",
            "away_team_losses",
        )
    ):
        return "prematch_standings"

    return "other"


def normalize_positive(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    mx = float(x.max()) if len(x) else 0.0
    if mx <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return x / mx


def robust_source_value_shift(a: pd.Series, b: pd.Series) -> float:
    """Absolute median shift scaled by pooled IQR; report-only diagnostic."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 20 or len(b) < 20:
        return float("nan")

    pooled = pd.concat([a, b], ignore_index=True)
    q25 = float(pooled.quantile(0.25))
    q75 = float(pooled.quantile(0.75))
    iqr = q75 - q25
    if not np.isfinite(iqr) or abs(iqr) < 1e-12:
        return float("nan")

    return float(abs(float(a.median()) - float(b.median())) / abs(iqr))


# =============================================================================
# DATA AUDIT
# =============================================================================

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


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {
        ID_COL,
        TIME_COL,
        SOURCE_COL,
        TARGET,
        "target_status",
        "first_half_period_end_seen",
        "timeline_full",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing}")

    df = df.copy()
    df["__kickoff__"] = mixed_to_datetime(df[TIME_COL])

    bad_time = df["__kickoff__"].isna()
    if bad_time.any():
        examples = df.loc[bad_time, [ID_COL, TIME_COL]].head(10)
        raise ValueError(
            "Parsing kickoff_utc incomplet. "
            f"NaT={int(bad_time.sum())}\n{examples.to_string(index=False)}"
        )

    df["__target__"] = valid_binary_target(df[TARGET])

    # Strict label integrity. We do not silently remove bad rows because that
    # would hide a collector issue.
    bad_status = df["target_status"].astype(str).str.upper().ne("RESOLVED")
    bad_period = numeric_series(df["first_half_period_end_seen"]).ne(1)
    bad_timeline = numeric_series(df["timeline_full"]).ne(1)

    if bad_status.any() or bad_period.any() or bad_timeline.any():
        raise ValueError(
            "Le dataset contient des labels non strictement résolus : "
            f"status_bad={int(bad_status.sum())}, "
            f"period_end_bad={int(bad_period.sum())}, "
            f"timeline_full_bad={int(bad_timeline.sum())}."
        )

    # Duplicates are dangerous because the same match could appear on both
    # sides of a temporal split. Keep none silently: exact duplicate match_id
    # is a hard error so the user sees the issue.
    dup = df[ID_COL].astype(str).duplicated(keep=False)
    if dup.any():
        sample = df.loc[dup, [ID_COL, TIME_COL, SOURCE_COL]].head(20)
        raise ValueError(
            f"match_id dupliqué(s) : {int(dup.sum())} lignes.\n"
            + sample.to_string(index=False)
        )

    return df.sort_values("__kickoff__").reset_index(drop=True)


def candidate_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    candidates: List[str] = []
    excluded_reason: Dict[str, str] = {}

    for c in df.columns:
        if c.startswith("__"):
            excluded_reason[c] = "internal"
            continue

        if c in HARD_EXCLUDE:
            excluded_reason[c] = "hard_exclude_metadata_or_leakage"
            continue

        if c in EXPLICIT_SEMANTIC_EXCLUDE:
            excluded_reason[c] = "semantic_exclude"
            continue

        if any(c.endswith(sfx) for sfx in EXCLUDE_SUFFIXES):
            excluded_reason[c] = "availability_count_suffix"
            continue

        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().sum() == 0:
            excluded_reason[c] = "non_numeric_or_all_missing"
            continue

        candidates.append(c)

    return candidates, excluded_reason


def make_split(df: pd.DataFrame, holdout_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    if not (0.10 <= holdout_fraction <= 0.30):
        raise ValueError("holdout_fraction doit être entre 0.10 et 0.30")

    n = len(df)
    holdout_n = max(1, int(round(n * holdout_fraction)))
    dev_n = n - holdout_n
    if dev_n < 500:
        raise ValueError(
            f"Bloc development trop petit ({dev_n}). Dataset total={n}."
        )

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


def build_feature_audit(
    full: pd.DataFrame,
    dev: pd.DataFrame,
    hold: pd.DataFrame,
    candidates: Sequence[str],
    excluded_reason: Mapping[str, str],
    min_coverage: float,
    max_source_coverage_gap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    source_values = [str(x) for x in dev[SOURCE_COL].dropna().unique().tolist()]
    source_values = sorted(source_values)

    rows = []
    shift_rows = []
    prefiltered = []

    for c in full.columns:
        if c.startswith("__"):
            continue

        family = family_of(c)
        numeric_full = pd.to_numeric(full[c], errors="coerce")
        numeric_dev = pd.to_numeric(dev[c], errors="coerce")
        numeric_hold = pd.to_numeric(hold[c], errors="coerce")

        coverage_full = float(numeric_full.notna().mean())
        coverage_dev = float(numeric_dev.notna().mean())
        coverage_hold = float(numeric_hold.notna().mean())
        nunique_dev = int(numeric_dev.nunique(dropna=True))

        source_cov = {}
        for s in source_values:
            mask = dev[SOURCE_COL].astype(str).eq(s)
            source_cov[s] = (
                float(numeric_dev[mask].notna().mean())
                if int(mask.sum()) > 0
                else float("nan")
            )

        if len(source_values) >= 2:
            vals = [v for v in source_cov.values() if np.isfinite(v)]
            source_gap = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
        else:
            source_gap = 0.0

        reason = excluded_reason.get(c, "")
        status = "EXCLUDED"

        if c in candidates:
            if coverage_dev < min_coverage:
                reason = f"coverage_dev<{min_coverage:.2f}"
            elif nunique_dev <= 1:
                reason = "constant_or_single_value_dev"
            elif np.isfinite(source_gap) and source_gap > max_source_coverage_gap:
                reason = f"source_coverage_gap>{max_source_coverage_gap:.2f}"
            else:
                status = "PREFILTER_KEEP"
                reason = ""
                prefiltered.append(c)

        row = {
            "feature": c,
            "family": family,
            "status": status,
            "reason": reason,
            "coverage_full": coverage_full,
            "coverage_dev": coverage_dev,
            "coverage_holdout_report_only": coverage_hold,
            "nunique_dev": nunique_dev,
            "source_coverage_gap_dev": source_gap,
        }
        for s, v in source_cov.items():
            row[f"coverage_dev_source::{s}"] = v
        rows.append(row)

        if c in candidates:
            sr = {
                "feature": c,
                "family": family,
                "source_coverage_gap_dev": source_gap,
            }
            for s, v in source_cov.items():
                sr[f"coverage::{s}"] = v

            if len(source_values) == 2:
                s0, s1 = source_values
                a = numeric_dev[dev[SOURCE_COL].astype(str).eq(s0)]
                b = numeric_dev[dev[SOURCE_COL].astype(str).eq(s1)]
                sr["robust_median_shift_iqr_dev"] = robust_source_value_shift(a, b)
            else:
                sr["robust_median_shift_iqr_dev"] = float("nan")
            shift_rows.append(sr)

    audit = pd.DataFrame(rows)
    source_shift = pd.DataFrame(shift_rows)
    return audit, source_shift, prefiltered


# =============================================================================
# TEMPORAL FEATURE SCREENING
# =============================================================================

def build_hgb(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=15,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=random_state,
        early_stopping=False,
    )


def build_l1(random_state: int, c_value: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                random_state=random_state,
            ),
        ),
    ])


def temporal_screen(
    dev: pd.DataFrame,
    features: Sequence[str],
    n_splits: int,
    permutation_repeats: int,
    random_state: int,
    l1_c: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[np.ndarray, np.ndarray]]]:

    if len(features) < 5:
        raise ValueError(f"Trop peu de features après préfiltre : {len(features)}")

    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev["__target__"].astype(int)

    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    per_feature = {f: [] for f in features}
    fold_indices: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        fold_indices[fold] = (train_idx, val_idx)
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train.nunique() < 2 or y_val.nunique() < 2:
            raise ValueError(
                f"Fold temporel {fold} ne contient pas les deux classes."
            )

        hgb = build_hgb(random_state + fold)
        sample_weight = compute_sample_weight("balanced", y_train)
        hgb.fit(X_train, y_train, sample_weight=sample_weight)
        prob = hgb.predict_proba(X_val)[:, 1]
        met = metric_bundle(y_val, prob)

        fold_metrics.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "train_start": dev["__kickoff__"].iloc[train_idx[0]].isoformat(),
            "train_end": dev["__kickoff__"].iloc[train_idx[-1]].isoformat(),
            "val_start": dev["__kickoff__"].iloc[val_idx[0]].isoformat(),
            "val_end": dev["__kickoff__"].iloc[val_idx[-1]].isoformat(),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            **met,
        })

        # OOS permutation importance on the fold validation block.
        pi = permutation_importance(
            hgb,
            X_val,
            y_val,
            scoring="average_precision",
            n_repeats=permutation_repeats,
            random_state=random_state + 1000 + fold,
            n_jobs=1,
        )

        # L1 stability is learned only on this fold TRAIN block.
        l1 = build_l1(random_state + 2000 + fold, c_value=l1_c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l1.fit(X_train, y_train)

        coef = l1.named_steps["model"].coef_[0]

        for j, f in enumerate(features):
            per_feature[f].append({
                "fold": fold,
                "permutation_importance_pr": float(pi.importances_mean[j]),
                "permutation_std_pr": float(pi.importances_std[j]),
                "l1_coef": float(coef[j]),
                "l1_nonzero": int(abs(float(coef[j])) > 1e-12),
            })

        log(
            f"Fold {fold}/{n_splits} | "
            f"TRAIN={len(train_idx)} VAL={len(val_idx)} | "
            f"ROC={met['roc_auc']:.4f} PR={met['pr_auc']:.4f} "
            f"Brier={met['brier']:.4f} LogLoss={met['log_loss']:.4f}"
        )

    score_rows = []
    for f, records in per_feature.items():
        imp = np.array([r["permutation_importance_pr"] for r in records], dtype=float)
        coef = np.array([r["l1_coef"] for r in records], dtype=float)
        nonzero = np.array([r["l1_nonzero"] for r in records], dtype=float)

        score_rows.append({
            "feature": f,
            "family": family_of(f),
            "perm_pr_mean": float(np.nanmean(imp)),
            "perm_pr_median": float(np.nanmedian(imp)),
            "perm_pr_std": float(np.nanstd(imp)),
            "perm_positive_fold_fraction": float(np.mean(imp > 0)),
            "perm_nonnegative_fold_fraction": float(np.mean(imp >= 0)),
            "l1_nonzero_fraction": float(np.mean(nonzero)),
            "l1_abs_coef_mean": float(np.mean(np.abs(coef))),
            "l1_abs_coef_median": float(np.median(np.abs(coef))),
        })

    fold_df = pd.DataFrame(fold_metrics)
    score_df = pd.DataFrame(score_rows)

    # Multi-signal score. Permutation is weighted most because it is measured
    # directly OOS. L1 is a complementary stability signal.
    score_df["perm_positive_strength"] = normalize_positive(score_df["perm_pr_mean"])
    score_df["l1_strength"] = normalize_positive(score_df["l1_abs_coef_mean"])
    score_df["stability_score"] = (
        0.50 * score_df["perm_positive_strength"]
        + 0.25 * score_df["perm_positive_fold_fraction"]
        + 0.15 * score_df["l1_nonzero_fraction"]
        + 0.10 * score_df["l1_strength"]
    )

    score_df = score_df.sort_values(
        ["stability_score", "perm_pr_mean", "l1_abs_coef_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return fold_df, score_df, fold_indices


# =============================================================================
# GROUP ABLATION
# =============================================================================

def group_ablation(
    dev: pd.DataFrame,
    features: Sequence[str],
    fold_indices: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> pd.DataFrame:

    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev["__target__"].astype(int)

    groups: Dict[str, List[str]] = {}
    for f in features:
        groups.setdefault(family_of(f), []).append(f)

    rows = []

    for fold, (train_idx, val_idx) in fold_indices.items():
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        sw = compute_sample_weight("balanced", y_train)

        baseline = build_hgb(random_state + 3000 + fold)
        baseline.fit(X_train, y_train, sample_weight=sw)
        p_base = baseline.predict_proba(X_val)[:, 1]
        m_base = metric_bundle(y_val, p_base)

        for group, group_features in sorted(groups.items()):
            remaining = [f for f in features if f not in group_features]
            if len(remaining) < 5:
                continue

            model = build_hgb(random_state + 4000 + fold)
            model.fit(
                X_train.loc[:, remaining],
                y_train,
                sample_weight=sw,
            )
            p = model.predict_proba(X_val.loc[:, remaining])[:, 1]
            m = metric_bundle(y_val, p)

            rows.append({
                "fold": fold,
                "family": group,
                "n_features_removed": len(group_features),
                "baseline_pr_auc": m_base["pr_auc"],
                "without_group_pr_auc": m["pr_auc"],
                "delta_pr_auc_keep_group": m_base["pr_auc"] - m["pr_auc"],
                "baseline_roc_auc": m_base["roc_auc"],
                "without_group_roc_auc": m["roc_auc"],
                "delta_roc_auc_keep_group": m_base["roc_auc"] - m["roc_auc"],
                "baseline_brier": m_base["brier"],
                "without_group_brier": m["brier"],
                # positive = group helps lower Brier
                "delta_brier_improvement_keep_group": m["brier"] - m_base["brier"],
                "baseline_log_loss": m_base["log_loss"],
                "without_group_log_loss": m["log_loss"],
                # positive = group helps lower logloss
                "delta_logloss_improvement_keep_group": m["log_loss"] - m_base["log_loss"],
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

    summary = (
        detail.groupby("family", as_index=False)
        .agg(
            n_folds=("fold", "count"),
            n_features_removed=("n_features_removed", "max"),
            mean_delta_pr_auc=("delta_pr_auc_keep_group", "mean"),
            median_delta_pr_auc=("delta_pr_auc_keep_group", "median"),
            positive_pr_fold_fraction=(
                "delta_pr_auc_keep_group",
                lambda s: float(np.mean(np.asarray(s) > 0)),
            ),
            mean_delta_roc_auc=("delta_roc_auc_keep_group", "mean"),
            mean_brier_improvement=("delta_brier_improvement_keep_group", "mean"),
            mean_logloss_improvement=("delta_logloss_improvement_keep_group", "mean"),
        )
        .sort_values(
            ["mean_delta_pr_auc", "positive_pr_fold_fraction"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return summary


# =============================================================================
# REDUNDANCY + FINAL SELECTION
# =============================================================================

def correlation_pairs(
    dev: pd.DataFrame,
    features: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    if len(features) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "spearman_abs"])

    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    corr = X.corr(method="spearman", min_periods=max(20, int(len(dev) * 0.10)))

    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(float(v)) >= threshold:
                rows.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "spearman": float(v),
                    "spearman_abs": abs(float(v)),
                })

    return pd.DataFrame(rows).sort_values(
        "spearman_abs",
        ascending=False,
    ).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["feature_a", "feature_b", "spearman", "spearman_abs"]
    )


def select_features(
    dev: pd.DataFrame,
    scores: pd.DataFrame,
    group_ablation_df: pd.DataFrame,
    corr_threshold: float,
    min_features: int,
    max_features: int,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:

    scores = scores.copy()

    if not group_ablation_df.empty:
        group_map = group_ablation_df.set_index("family")[
            ["mean_delta_pr_auc", "positive_pr_fold_fraction"]
        ].to_dict(orient="index")
        scores["group_mean_delta_pr_auc"] = scores["family"].map(
            lambda g: group_map.get(g, {}).get("mean_delta_pr_auc", 0.0)
        )
        scores["group_positive_pr_fold_fraction"] = scores["family"].map(
            lambda g: group_map.get(g, {}).get("positive_pr_fold_fraction", 0.0)
        )
    else:
        scores["group_mean_delta_pr_auc"] = 0.0
        scores["group_positive_pr_fold_fraction"] = 0.0

    # Small bonus for a family that helps on group ablation. Negative group
    # effect never penalizes an individually strong feature into oblivion.
    group_bonus = normalize_positive(scores["group_mean_delta_pr_auc"])
    scores["selection_score"] = (
        scores["stability_score"]
        + 0.10 * group_bonus
        + 0.05 * scores["group_positive_pr_fold_fraction"]
    )

    # Primary stability gate.
    scores["stable_candidate"] = (
        (
            (scores["perm_pr_mean"] > 0)
            & (scores["perm_positive_fold_fraction"] >= 0.50)
        )
        | (scores["l1_nonzero_fraction"] >= 0.75)
    )

    scores = scores.sort_values(
        ["stable_candidate", "selection_score", "perm_pr_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranked = scores["feature"].tolist()
    corr_df = correlation_pairs(dev, ranked, threshold=corr_threshold)

    # Fast correlation lookup.
    correlated: Dict[str, Dict[str, float]] = {f: {} for f in ranked}
    for _, r in corr_df.iterrows():
        a, b, v = r["feature_a"], r["feature_b"], float(r["spearman_abs"])
        correlated.setdefault(a, {})[b] = v
        correlated.setdefault(b, {})[a] = v

    selected: List[str] = []
    rejection_reason: Dict[str, str] = {}

    # First pass: stable features only.
    stable_ranked = scores.loc[scores["stable_candidate"], "feature"].tolist()
    for f in stable_ranked:
        if len(selected) >= max_features:
            rejection_reason[f] = "max_features_reached"
            continue

        conflict = next(
            (
                kept
                for kept in selected
                if correlated.get(f, {}).get(kept, 0.0) >= corr_threshold
            ),
            None,
        )
        if conflict is not None:
            rejection_reason[f] = f"redundant_with::{conflict}"
            continue

        selected.append(f)

    # Second pass: if the strict gate leaves too few, fill from the strongest
    # remaining features while preserving the redundancy rule.
    if len(selected) < min_features:
        for f in ranked:
            if f in selected:
                continue
            if len(selected) >= min_features or len(selected) >= max_features:
                break

            conflict = next(
                (
                    kept
                    for kept in selected
                    if correlated.get(f, {}).get(kept, 0.0) >= corr_threshold
                ),
                None,
            )
            if conflict is not None:
                rejection_reason.setdefault(f, f"redundant_with::{conflict}")
                continue

            selected.append(f)
            rejection_reason.pop(f, None)

    scores["selected"] = scores["feature"].isin(selected).astype(int)
    scores["selection_note"] = scores["feature"].map(rejection_reason).fillna("")
    scores.loc[scores["selected"].eq(1), "selection_note"] = "SELECTED"

    return selected, scores, corr_df


# =============================================================================
# REPORTING
# =============================================================================

def make_report(
    dataset_path: Path,
    df: pd.DataFrame,
    dev: pd.DataFrame,
    hold: pd.DataFrame,
    split_info: SplitInfo,
    feature_audit: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    group_ablation_df: pd.DataFrame,
    selected: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> str:

    source_counts = df[SOURCE_COL].astype(str).value_counts(dropna=False)
    source_target = (
        df.groupby(SOURCE_COL)["__target__"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "positives", "mean": "positive_rate"})
    )

    family_selected = (
        scores.loc[scores["selected"].eq(1)]
        .groupby("family")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="selected_count")
    )

    top_scores = scores.loc[:, [
        "feature",
        "family",
        "perm_pr_mean",
        "perm_positive_fold_fraction",
        "l1_nonzero_fraction",
        "selection_score",
        "selected",
        "selection_note",
    ]].head(40)

    lines = [
        "# First Half 37 — Audit + Feature Selection V1",
        "",
        f"- Dataset : `{dataset_path}`",
        f"- Sorties : `{output_dir}`",
        f"- Lignes : **{len(df)}**",
        f"- Positifs : **{int(df['__target__'].sum())}** ({df['__target__'].mean():.2%})",
        f"- Date min : **{df['__kickoff__'].min().isoformat()}**",
        f"- Date max : **{df['__kickoff__'].max().isoformat()}**",
        f"- Match IDs uniques : **{df[ID_COL].nunique()}**",
        "",
        "## Sources",
        "",
        dataframe_markdown(source_target, max_rows=20),
        "",
        "## Split temporel réservé",
        "",
        f"- DEVELOPMENT : **{split_info.development_rows}** lignes | "
        f"{split_info.development_start} -> {split_info.development_end}",
        f"- FINAL HOLDOUT : **{split_info.holdout_rows}** lignes | "
        f"{split_info.holdout_start} -> {split_info.holdout_end}",
        "- Le target du FINAL HOLDOUT n'est pas utilisé dans les métriques de sélection.",
        "",
        "## Filtrage qualité",
        "",
        f"- Couverture minimale DEV : **{args.min_coverage:.0%}**",
        f"- Écart maximal de couverture entre sources (DEV) : **{args.max_source_coverage_gap:.0%}**",
        f"- Seuil redondance Spearman |rho| : **{args.corr_threshold:.2f}**",
        f"- Features préfiltrées : **{int((feature_audit['status'] == 'PREFILTER_KEEP').sum())}**",
        "",
        "## Validation temporelle interne",
        "",
        dataframe_markdown(
            fold_metrics.loc[:, [
                "fold",
                "train_rows",
                "val_rows",
                "train_positive_rate",
                "val_positive_rate",
                "roc_auc",
                "pr_auc",
                "brier",
                "log_loss",
            ]],
            max_rows=20,
        ),
        "",
        "## Ablation par famille",
        "",
        (
            dataframe_markdown(group_ablation_df, max_rows=30)
            if not group_ablation_df.empty
            else "Ablation désactivée pour cette exécution."
        ),
        "",
        "## Features sélectionnées",
        "",
        f"Total sélectionné : **{len(selected)}**",
        "",
        dataframe_markdown(family_selected, max_rows=30),
        "",
        "### Liste finale",
        "",
    ]

    for i, f in enumerate(selected, start=1):
        row = scores.loc[scores["feature"].eq(f)].iloc[0]
        lines.append(
            f"{i}. `{f}` — {row['family']} | "
            f"score={row['selection_score']:.4f} | "
            f"perm={row['perm_pr_mean']:.6f} | "
            f"perm+ folds={row['perm_positive_fold_fraction']:.0%} | "
            f"L1 nonzero={row['l1_nonzero_fraction']:.0%}"
        )

    lines.extend([
        "",
        "## Top ranking complet",
        "",
        dataframe_markdown(top_scores, max_rows=40),
        "",
        "## Interprétation",
        "",
        "- Une feature n'est pas gardée uniquement parce qu'elle est disponible.",
        "- Les variables de label/résolution, provenance, compteurs de disponibilité et timestamps sont exclues.",
        "- Les cotes/xG PRE très présents dans le seed mais absents du backfill sont normalement éliminés par le contrôle de source coverage gap.",
        "- Les redondances très fortes sont supprimées après ranking, sur DEVELOPMENT uniquement.",
        "- Le FINAL HOLDOUT reste réservé pour la future comparaison finale des modèles; ne pas le réutiliser pour ajuster cette sélection.",
        "",
        "## Prochaine étape",
        "",
        "Entraîner plusieurs modèles sur le schéma sélectionné et comparer Logistic / HistGradientBoosting / autres candidats par validation temporelle, puis ouvrir le FINAL HOLDOUT une seule fois lorsque le pipeline est figé.",
        "",
    ])

    return "\n".join(lines)


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit + sélection temporelle des features First_Half37."
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--final-holdout-fraction", type=float, default=0.15)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--min-coverage", type=float, default=0.60)
    p.add_argument("--max-source-coverage-gap", type=float, default=0.25)
    p.add_argument("--corr-threshold", type=float, default=0.92)
    p.add_argument("--min-features", type=int, default=20)
    p.add_argument("--max-features", type=int, default=40)
    p.add_argument("--permutation-repeats", type=int, default=3)
    p.add_argument("--l1-c", type=float, default=0.10)
    p.add_argument("--random-state", type=int, default=87)
    p.add_argument("--skip-ablation", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not (0 < args.min_coverage <= 1):
        raise ValueError("--min-coverage doit être dans ]0,1].")
    if not (0 <= args.max_source_coverage_gap <= 1):
        raise ValueError("--max-source-coverage-gap doit être dans [0,1].")
    if not (0.50 <= args.corr_threshold < 1.0):
        raise ValueError("--corr-threshold doit être entre 0.50 et 1.0.")
    if args.n_splits < 3:
        raise ValueError("--n-splits doit être >= 3.")
    if args.min_features < 5 or args.max_features < args.min_features:
        raise ValueError("Bornes min/max features invalides.")

    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("FIRST HALF 37 — AUDIT + REAL FEATURE SELECTION V1")
    log("=" * 100)
    log(f"Dataset : {dataset_path}")
    log(f"Output  : {output_dir}")
    log()

    df = load_and_validate(dataset_path)

    log(
        f"Dataset OK | rows={len(df)} | unique_match_id={df[ID_COL].nunique()} | "
        f"positives={int(df['__target__'].sum())} ({df['__target__'].mean():.2%})"
    )
    log(
        f"Dates : {df['__kickoff__'].min().isoformat()} -> "
        f"{df['__kickoff__'].max().isoformat()}"
    )

    dev, hold, split_info = make_split(
        df,
        holdout_fraction=args.final_holdout_fraction,
    )

    log()
    log(
        f"DEVELOPMENT={len(dev)} | {split_info.development_start} -> {split_info.development_end}"
    )
    log(
        f"FINAL HOLDOUT réservé={len(hold)} | {split_info.holdout_start} -> {split_info.holdout_end}"
    )
    log("Le FINAL HOLDOUT ne sera pas évalué pendant cette sélection.")

    candidates, excluded_reason = candidate_columns(df)

    feature_audit, source_shift, prefiltered = build_feature_audit(
        full=df,
        dev=dev,
        hold=hold,
        candidates=candidates,
        excluded_reason=excluded_reason,
        min_coverage=args.min_coverage,
        max_source_coverage_gap=args.max_source_coverage_gap,
    )

    log()
    log(
        f"Candidates numériques avant filtres={len(candidates)} | "
        f"après filtres qualité={len(prefiltered)}"
    )

    # Save audits early in case a later ML step is interrupted.
    feature_audit_path = output_dir / "first_half_37_feature_audit.csv"
    source_shift_path = output_dir / "first_half_37_source_shift.csv"
    feature_audit.to_csv(feature_audit_path, index=False, encoding="utf-8-sig")
    source_shift.to_csv(source_shift_path, index=False, encoding="utf-8-sig")

    log()
    log("Validation temporelle + permutation importance + L1...")
    fold_metrics, scores, fold_indices = temporal_screen(
        dev=dev,
        features=prefiltered,
        n_splits=args.n_splits,
        permutation_repeats=args.permutation_repeats,
        random_state=args.random_state,
        l1_c=args.l1_c,
    )

    if args.skip_ablation:
        ablation_df = pd.DataFrame()
        log("Ablation par famille : SKIP")
    else:
        log()
        log("Ablation par famille...")
        ablation_df = group_ablation(
            dev=dev,
            features=prefiltered,
            fold_indices=fold_indices,
            random_state=args.random_state,
        )

    selected, scores_final, corr_df = select_features(
        dev=dev,
        scores=scores,
        group_ablation_df=ablation_df,
        corr_threshold=args.corr_threshold,
        min_features=args.min_features,
        max_features=args.max_features,
    )

    # Merge quality diagnostics into score table.
    score_out = scores_final.merge(
        feature_audit.loc[:, [
            "feature",
            "coverage_dev",
            "coverage_holdout_report_only",
            "source_coverage_gap_dev",
        ]],
        on="feature",
        how="left",
    )

    selected_df = score_out.loc[score_out["selected"].eq(1)].copy()
    selected_df = selected_df.sort_values(
        ["selection_score", "perm_pr_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    selected_df.insert(0, "rank", np.arange(1, len(selected_df) + 1))

    # Create final model-ready frame. Keep minimal metadata for later temporal
    # split/audit, plus selected features and target.
    model_ready_cols = [
        ID_COL,
        "competition_id" if "competition_id" in df.columns else None,
        "season_id" if "season_id" in df.columns else None,
        "home_team" if "home_team" in df.columns else None,
        "away_team" if "away_team" in df.columns else None,
        TIME_COL,
        SOURCE_COL,
        *selected,
        TARGET,
    ]
    model_ready_cols = [c for c in model_ready_cols if c is not None]
    model_ready = df.loc[:, model_ready_cols].copy()

    # Save all artifacts.
    fold_metrics_path = output_dir / "first_half_37_temporal_fold_metrics.csv"
    score_path = output_dir / "first_half_37_temporal_feature_scores.csv"
    ablation_path = output_dir / "first_half_37_group_ablation.csv"
    corr_path = output_dir / "first_half_37_correlated_pairs.csv"
    selected_path = output_dir / "first_half_37_selected_features.csv"
    schema_path = output_dir / "first_half_37_selected_schema.json"
    split_path = output_dir / "first_half_37_split_manifest.json"
    model_ready_path = output_dir / "First_Half_37_MODEL_READY_v1.csv"
    report_path = output_dir / "first_half_37_audit_report.md"

    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    score_out.to_csv(score_path, index=False, encoding="utf-8-sig")
    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    model_ready.to_csv(model_ready_path, index=False, encoding="utf-8-sig")

    schema = {
        "schema_version": "first_half_37_feature_selection_v1",
        "target": TARGET,
        "time_column": TIME_COL,
        "match_id_column": ID_COL,
        "source_column": SOURCE_COL,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "feature_families": {f: family_of(f) for f in selected},
        "selection_rules": {
            "min_coverage_dev": args.min_coverage,
            "max_source_coverage_gap_dev": args.max_source_coverage_gap,
            "spearman_redundancy_threshold": args.corr_threshold,
            "temporal_cv_splits": args.n_splits,
            "permutation_repeats": args.permutation_repeats,
            "l1_C": args.l1_c,
            "min_features": args.min_features,
            "max_features": args.max_features,
            "group_ablation_used": not args.skip_ablation,
        },
        "anti_leakage": {
            "hard_excluded": sorted(HARD_EXCLUDE),
            "semantic_excluded": sorted(EXPLICIT_SEMANTIC_EXCLUDE),
            "excluded_suffixes": list(EXCLUDE_SUFFIXES),
            "holdout_used_for_feature_selection": False,
            "missing_equals_zero": False,
        },
        "development": {
            "rows": len(dev),
            "start": split_info.development_start,
            "end": split_info.development_end,
        },
        "final_holdout_reserved": {
            "rows": len(hold),
            "start": split_info.holdout_start,
            "end": split_info.holdout_end,
            "evaluated": False,
        },
    }
    json_dump(schema_path, schema)
    json_dump(split_path, asdict(split_info))

    report = make_report(
        dataset_path=dataset_path,
        df=df,
        dev=dev,
        hold=hold,
        split_info=split_info,
        feature_audit=feature_audit,
        fold_metrics=fold_metrics,
        scores=score_out,
        group_ablation_df=ablation_df,
        selected=selected,
        output_dir=output_dir,
        args=args,
    )
    report_path.write_text(report, encoding="utf-8")

    log()
    log("=" * 100)
    log("AUDIT + SELECTION TERMINES")
    log("=" * 100)
    log(f"Features sélectionnées : {len(selected)}")
    for i, f in enumerate(selected, start=1):
        log(f"  {i:02d}. {f}")

    log()
    log(f"Rapport      : {report_path}")
    log(f"Schema       : {schema_path}")
    log(f"Selected CSV : {selected_path}")
    log(f"MODEL_READY  : {model_ready_path}")
    log()
    log("IMPORTANT : le FINAL HOLDOUT est toujours intact et non évalué.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
