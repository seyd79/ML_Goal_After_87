#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FIRST HALF 37 — COMPACT SCHEMA + NESTED C TUNING + TEMPORAL CALIBRATION (V1)
================================================================================

But
---
Dernière phase de raffinement sur DEVELOPMENT avant toute ouverture du FINAL
HOLDOUT.

Cette étape part du TOP-20 identifié précédemment et teste des schémas plus
compacts (20 / 18 / 15 / 12 / 10 features), en privilégiant :
- la stabilité du signe des coefficients dans le temps ;
- la suppression de redondances évidentes dans les tirs ;
- un réglage de la régularisation L2 (C) strictement à l'intérieur du TRAIN ;
- une calibration temporelle apprise uniquement sur des prédictions OOF du
  TRAIN externe.

Protection anti-fuite
---------------------
Pour chaque fold externe :
1) le choix de C est fait uniquement dans le TRAIN externe via TimeSeriesSplit;
2) les calibrateurs sont appris uniquement sur des prédictions OOF temporelles
   du TRAIN externe ;
3) le modèle est ensuite évalué sur le VAL externe ;
4) aucune prédiction ni métrique n'est calculée sur le FINAL HOLDOUT.

Sous-ensembles
--------------
- TOP20_CONTROL : contrôle, les 20 features actuelles.
- STABLE18      : 18 features favorisant la stabilité du signe.
- COMPACT15     : redondances de tirs supprimées + priorité à la stabilité.
- COMPACT12     : version plus compacte.
- COMPACT10     : noyau le plus compact.

Redondance tirs
---------------
Si le total de tirs et les tirs cadrés sont présents, les composantes
"off_target" et "blocked" sont considérées comme candidates à la suppression
avant la construction des schémas compacts. Cela évite de demander à une
régression logistique de répartir artificiellement le signal entre variables
fortement liées.

Modèles
-------
- logistic_l2
- logistic_l2_balanced

Calibration
-----------
- raw
- intercept_temporal_oof : correction d'intercept uniquement, préserve le rang
- platt_temporal_oof     : Platt complet (intercept + pente)
- prior_corrected        : balanced uniquement

Sorties
-------
outputs/first_half_37/refinement_v1/
    first_half_37_compact_subsets.json
    first_half_37_c_tuning.csv
    first_half_37_refinement_fold_metrics.csv
    first_half_37_refinement_summary.csv
    first_half_37_refinement_ranking.csv
    first_half_37_refined_coefficient_stability.csv
    first_half_37_refinement_report.md
    first_half_37_refinement_manifest.json
    candidates_dev_only/

IMPORTANT
---------
Le FINAL HOLDOUT reste fermé. Même si un candidat passe le gate, ce script ne
l'ouvre pas. L'ouverture finale doit faire l'objet d'une étape séparée et unique.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

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
DEFAULT_COEF_STABILITY = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "first_half_37"
    / "calibration_v1"
    / "first_half_37_coefficient_stability.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "refinement_v1"
)

TARGET = "goal_after_37_before_ht"
TIME_COL = "kickoff_utc"
ID_COL = "match_id"
BASE_MODELS = ("logistic_l2", "logistic_l2_balanced")

# Redondances évidentes dans le TOP-20 actuel. Le total + SOT reste disponible;
# off-target / blocked sont les premières variables retirées pour les versions
# compactes. Ce choix est volontairement transparent et déterministe.
SHOT_REDUNDANCY_DROP = {
    "home_shots_off_target_37",
    "home_shots_blocked_37",
    "away_shots_off_target_37",
    "away_shots_blocked_37",
}

DEFAULT_C_GRID = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)


# =============================================================================
# HELPERS
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


def log(msg: str = "") -> None:
    print(msg, flush=True)


def mixed_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


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
        raise ValueError(
            f"{int(dt.isna().sum())} kickoff_utc non parsables. "
            "Refus de continuer pour protéger le split temporel."
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("Target non résolu dans MODEL_READY.")
    vals = set(y.astype(int).unique())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Target invalide : {sorted(vals)}")

    out = df.copy()
    out["__kickoff__"] = dt
    out["__target__"] = y.astype(int)

    ids = out[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} match_id dupliqués.")

    return out.sort_values(["__kickoff__", ID_COL], kind="stable").reset_index(drop=True)


def split_dev_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    if not (0.10 <= holdout_fraction <= 0.30):
        raise ValueError("holdout_fraction doit être entre 0.10 et 0.30")

    n = len(df)
    hold_n = max(1, int(round(n * holdout_fraction)))
    dev_n = n - hold_n
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


def load_top20(selected_path: Path, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if not selected_path.exists():
        raise FileNotFoundError(f"selected_features introuvable : {selected_path}")

    tab = pd.read_csv(selected_path)
    if "feature" not in tab.columns:
        raise ValueError("selected_features.csv doit contenir 'feature'.")

    if "selected" in tab.columns:
        flag = pd.to_numeric(tab["selected"], errors="coerce").fillna(0).astype(int)
        tab = tab.loc[flag.eq(1)].copy()

    if "rank" in tab.columns:
        tab["__rank__"] = pd.to_numeric(tab["rank"], errors="coerce")
        tab = tab.sort_values("__rank__", kind="stable")
    elif "selection_score" in tab.columns:
        tab["__score__"] = pd.to_numeric(tab["selection_score"], errors="coerce")
        tab = tab.sort_values("__score__", ascending=False, kind="stable")
    else:
        tab = tab.reset_index(drop=True)
        tab["__rank__"] = np.arange(1, len(tab) + 1)

    feats: List[str] = []
    seen = set()
    for f in tab["feature"].astype(str):
        if f in df.columns and f not in seen:
            feats.append(f)
            seen.add(f)
        if len(feats) >= 20:
            break

    if len(feats) != 20:
        raise ValueError(f"TOP_20 impossible : seulement {len(feats)} features disponibles.")
    return feats, tab


def load_stability(path: Path, top20: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"coefficient_stability introuvable : {path}\n"
            "Exécute d'abord calibrate_first_half_37_top20.py."
        )
    tab = pd.read_csv(path)
    need = {"model", "feature", "sign_consistency", "mean_abs_coefficient"}
    missing = need - set(tab.columns)
    if missing:
        raise ValueError(f"Colonnes stabilité manquantes : {sorted(missing)}")

    tab = tab.loc[tab["feature"].astype(str).isin(set(top20))].copy()
    tab["sign_consistency"] = pd.to_numeric(tab["sign_consistency"], errors="coerce")
    tab["mean_abs_coefficient"] = pd.to_numeric(tab["mean_abs_coefficient"], errors="coerce")

    rows = []
    for feat in top20:
        g = tab.loc[tab["feature"].astype(str).eq(feat)]
        if g.empty:
            min_cons = 0.0
            mean_cons = 0.0
            abs_coef = 0.0
            models = 0
        else:
            min_cons = float(g["sign_consistency"].fillna(0).min())
            mean_cons = float(g["sign_consistency"].fillna(0).mean())
            abs_coef = float(g["mean_abs_coefficient"].fillna(0).mean())
            models = int(g["model"].nunique())
        rows.append({
            "feature": feat,
            "min_sign_consistency": min_cons,
            "mean_sign_consistency": mean_cons,
            "mean_abs_coefficient_across_models": abs_coef,
            "models_seen": models,
        })
    return pd.DataFrame(rows)


def build_compact_subsets(
    top20: Sequence[str],
    selected_tab: pd.DataFrame,
    stability: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    rank_map = {f: i + 1 for i, f in enumerate(top20)}

    # Optional original selection score for tie-breaking only.
    score_map: Dict[str, float] = {}
    if "selection_score" in selected_tab.columns:
        for _, r in selected_tab.iterrows():
            f = str(r.get("feature", ""))
            if f in rank_map:
                try:
                    score_map[f] = float(r.get("selection_score", 0.0))
                except Exception:
                    score_map[f] = 0.0

    stab_map = stability.set_index("feature").to_dict("index")

    ranking_rows = []
    for f in top20:
        s = stab_map.get(f, {})
        ranking_rows.append({
            "feature": f,
            "original_rank": rank_map[f],
            "selection_score": float(score_map.get(f, 0.0)),
            "min_sign_consistency": float(s.get("min_sign_consistency", 0.0)),
            "mean_sign_consistency": float(s.get("mean_sign_consistency", 0.0)),
            "mean_abs_coefficient_across_models": float(
                s.get("mean_abs_coefficient_across_models", 0.0)
            ),
            "shot_redundancy_drop_candidate": int(f in SHOT_REDUNDANCY_DROP),
        })

    ranking = pd.DataFrame(ranking_rows)

    # STABLE18 : stabilité d'abord, ranking initial ensuite.
    stable18 = (
        ranking.sort_values(
            [
                "min_sign_consistency",
                "mean_sign_consistency",
                "original_rank",
                "mean_abs_coefficient_across_models",
            ],
            ascending=[False, False, True, False],
            kind="stable",
        )["feature"]
        .head(18)
        .tolist()
    )

    # Versions compactes : supprimer d'abord les quatre composantes les plus
    # redondantes lorsque possible, puis classer les survivantes par stabilité.
    compact_pool = ranking.loc[
        ~ranking["feature"].isin(SHOT_REDUNDANCY_DROP)
    ].copy()

    # Si la suppression rendait le pool trop petit (schéma futur différent), on
    # réinjecte les meilleures redondantes pour garantir au moins 15 features.
    if len(compact_pool) < 15:
        missing_n = 15 - len(compact_pool)
        refill = (
            ranking.loc[ranking["feature"].isin(SHOT_REDUNDANCY_DROP)]
            .sort_values(
                ["min_sign_consistency", "original_rank"],
                ascending=[False, True],
                kind="stable",
            )
            .head(missing_n)
        )
        compact_pool = pd.concat([compact_pool, refill], ignore_index=True)

    compact_ranked = compact_pool.sort_values(
        [
            "min_sign_consistency",
            "mean_sign_consistency",
            "original_rank",
            "mean_abs_coefficient_across_models",
        ],
        ascending=[False, False, True, False],
        kind="stable",
    )["feature"].tolist()

    subsets = {
        "TOP20_CONTROL": list(top20),
        "STABLE18": stable18,
        "COMPACT15": compact_ranked[:15],
        "COMPACT12": compact_ranked[:12],
        "COMPACT10": compact_ranked[:10],
    }

    # Remove any accidental duplicate subsets while preserving names only if the
    # actual feature list differs. TOP20 is always kept as control.
    seen = set()
    clean: Dict[str, List[str]] = {}
    for name, feats in subsets.items():
        key = tuple(feats)
        if name == "TOP20_CONTROL" or key not in seen:
            clean[name] = feats
            seen.add(key)

    return clean, ranking


def build_model(
    model_name: str,
    c_value: float,
    random_state: int,
) -> Pipeline:
    if model_name not in BASE_MODELS:
        raise KeyError(model_name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=float(c_value),
                solver="lbfgs",
                class_weight=("balanced" if model_name.endswith("balanced") else None),
                max_iter=5000,
                random_state=random_state,
            ),
        ),
    ])


def metric_bundle(y_true: Sequence[int], prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0 - 1e-9)
    if len(np.unique(y)) < 2:
        raise ValueError("Bloc d'évaluation sans les deux classes.")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(np.mean(p)),
    }


def ece_score(y_true: Sequence[int], prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def fit_platt(prob: np.ndarray, y: np.ndarray) -> LogisticRegression:
    x = logit(prob).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=4000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    except Exception:
        lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=4000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    return lr


def platt_predict(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit(prob).reshape(-1, 1))[:, 1]


def fit_intercept_shift(prob: np.ndarray, y: np.ndarray) -> float:
    """Fit only delta in sigmoid(logit(p)+delta) by Newton iterations."""
    x = logit(prob)
    y = np.asarray(y, dtype=float)
    delta = 0.0
    for _ in range(100):
        q = sigmoid(x + delta)
        f = float(np.sum(y - q))
        denom = float(np.sum(q * (1.0 - q)))
        if denom < 1e-10:
            break
        step = f / denom
        delta += step
        if abs(step) < 1e-10:
            break
    return float(delta)


def intercept_predict(delta: float, prob: np.ndarray) -> np.ndarray:
    return sigmoid(logit(prob) + float(delta))


def prior_correct_balanced(prob: np.ndarray, target_prevalence: float) -> np.ndarray:
    pi = float(np.clip(target_prevalence, 1e-6, 1 - 1e-6))
    shift = math.log(pi / (1.0 - pi))
    return sigmoid(logit(prob) + shift)


# =============================================================================
# NESTED TUNING / OOF
# =============================================================================


def tune_c_inner(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    c_grid: Sequence[float],
    inner_splits: int,
    random_state: int,
) -> Tuple[float, List[Dict]]:
    n = len(X)
    max_splits = min(inner_splits, max(2, n // 60))
    splitter = TimeSeriesSplit(n_splits=max_splits)

    rows: List[Dict] = []
    for c_idx, c_value in enumerate(c_grid, start=1):
        fold_pr: List[float] = []
        fold_roc: List[float] = []
        fold_lift: List[float] = []
        valid_folds = 0

        for inner_fold, (tr, va) in enumerate(splitter.split(X), start=1):
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            if y_tr.nunique() < 2 or y_va.nunique() < 2:
                continue
            model = build_model(
                model_name,
                c_value,
                random_state + c_idx * 100 + inner_fold,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[tr], y_tr)
            p = model.predict_proba(X.iloc[va])[:, 1]
            pr = float(average_precision_score(y_va, p))
            roc = float(roc_auc_score(y_va, p))
            base = float(y_va.mean())
            fold_pr.append(pr)
            fold_roc.append(roc)
            fold_lift.append(pr - base)
            valid_folds += 1

        if valid_folds == 0:
            continue

        rows.append({
            "C": float(c_value),
            "inner_valid_folds": valid_folds,
            "inner_mean_pr_auc": float(np.mean(fold_pr)),
            "inner_mean_pr_lift": float(np.mean(fold_lift)),
            "inner_worst_pr_lift": float(np.min(fold_lift)),
            "inner_pr_above_base_fraction": float(np.mean(np.asarray(fold_lift) > 0)),
            "inner_mean_roc_auc": float(np.mean(fold_roc)),
            "inner_pr_std": float(np.std(fold_pr, ddof=0)),
        })

    if not rows:
        raise ValueError("Aucun C évaluable dans le TRAIN interne.")

    tab = pd.DataFrame(rows)
    # Priorité: stabilité PR > base, lift moyen, PR moyenne, puis C plus petit
    # (régularisation plus forte) comme tie-break conservateur.
    tab = tab.sort_values(
        [
            "inner_pr_above_base_fraction",
            "inner_mean_pr_lift",
            "inner_mean_pr_auc",
            "inner_worst_pr_lift",
            "inner_mean_roc_auc",
            "C",
        ],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    best_c = float(tab.iloc[0]["C"])
    return best_c, tab.to_dict("records")


def temporal_oof_fixed_c(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    c_value: float,
    inner_splits: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    max_splits = min(inner_splits, max(2, n // 60))
    splitter = TimeSeriesSplit(n_splits=max_splits)

    probs: List[float] = []
    ys: List[int] = []
    for k, (tr, va) in enumerate(splitter.split(X), start=1):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue
        model = build_model(model_name, c_value, random_state + k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[tr], y_tr)
        p = model.predict_proba(X.iloc[va])[:, 1]
        probs.extend(np.asarray(p, dtype=float).tolist())
        ys.extend(y_va.astype(int).tolist())

    if len(probs) < 80 or len(set(ys)) < 2:
        raise ValueError(f"Pas assez d'OOF pour calibration : n={len(probs)}")
    return np.asarray(probs, dtype=float), np.asarray(ys, dtype=int)


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_refinement(
    dev: pd.DataFrame,
    subsets: Dict[str, List[str]],
    outer_splits: int,
    inner_splits: int,
    c_grid: Sequence[float],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_all = dev["__target__"].astype(int)
    outer = TimeSeriesSplit(n_splits=outer_splits)

    fold_rows: List[Dict] = []
    tuning_rows: List[Dict] = []
    coef_rows: List[Dict] = []

    for subset_idx, (subset_name, features) in enumerate(subsets.items(), start=1):
        X_all = dev.loc[:, features].apply(pd.to_numeric, errors="coerce")
        log("-")
        log(f"SUBSET {subset_name} | n_features={len(features)}")

        for fold, (tr, va) in enumerate(outer.split(dev), start=1):
            X_tr, X_va = X_all.iloc[tr], X_all.iloc[va]
            y_tr, y_va = y_all.iloc[tr], y_all.iloc[va]
            if y_tr.nunique() < 2 or y_va.nunique() < 2:
                raise ValueError(f"Fold externe {fold} sans les deux classes.")

            train_rate = float(y_tr.mean())
            val_rate = float(y_va.mean())
            base_prob = np.full(len(y_va), train_rate, dtype=float)
            baseline = metric_bundle(y_va, base_prob)

            for model_idx, model_name in enumerate(BASE_MODELS, start=1):
                seed = random_state + subset_idx * 10000 + fold * 100 + model_idx * 10

                best_c, tune_records = tune_c_inner(
                    X_tr,
                    y_tr,
                    model_name,
                    c_grid=c_grid,
                    inner_splits=inner_splits,
                    random_state=seed + 1000,
                )
                for rec in tune_records:
                    rec.update({
                        "subset": subset_name,
                        "n_features": len(features),
                        "base_model": model_name,
                        "outer_fold": fold,
                        "selected_C": int(float(rec["C"]) == float(best_c)),
                    })
                    tuning_rows.append(rec)

                oof_p, oof_y = temporal_oof_fixed_c(
                    X_tr,
                    y_tr,
                    model_name,
                    best_c,
                    inner_splits=inner_splits,
                    random_state=seed + 2000,
                )
                intercept_delta = fit_intercept_shift(oof_p, oof_y)
                platt = fit_platt(oof_p, oof_y)

                model = build_model(model_name, best_c, seed + 3000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                raw = model.predict_proba(X_va)[:, 1]

                # coefficient audit on standardized features
                coef = np.asarray(model.named_steps["model"].coef_[0], dtype=float)
                for feat, ccoef in zip(features, coef):
                    coef_rows.append({
                        "subset": subset_name,
                        "base_model": model_name,
                        "fold": fold,
                        "selected_C": best_c,
                        "feature": feat,
                        "coefficient": float(ccoef),
                        "abs_coefficient": float(abs(ccoef)),
                        "sign": int(np.sign(ccoef)),
                    })

                methods: Dict[str, np.ndarray] = {
                    "raw": raw,
                    "intercept_temporal_oof": intercept_predict(intercept_delta, raw),
                    "platt_temporal_oof": platt_predict(platt, raw),
                }
                if model_name == "logistic_l2_balanced":
                    methods["prior_corrected"] = prior_correct_balanced(raw, train_rate)

                for calibration, p in methods.items():
                    met = metric_bundle(y_va, p)
                    fold_rows.append({
                        "subset": subset_name,
                        "n_features": len(features),
                        "base_model": model_name,
                        "calibration": calibration,
                        "fold": fold,
                        "selected_C": best_c,
                        "train_rows": len(tr),
                        "val_rows": len(va),
                        "train_positive_rate": train_rate,
                        "val_positive_rate": val_rate,
                        **met,
                        "ece_10": ece_score(y_va, p, 10),
                        "baseline_pr_auc": baseline["pr_auc"],
                        "baseline_brier": baseline["brier"],
                        "baseline_log_loss": baseline["log_loss"],
                        "pr_lift_vs_base": met["pr_auc"] - baseline["pr_auc"],
                        "brier_improvement_vs_base": baseline["brier"] - met["brier"],
                        "logloss_improvement_vs_base": baseline["log_loss"] - met["log_loss"],
                    })

                # concise terminal line: best ranking/calibration signal per model/fold
                p_int = methods["intercept_temporal_oof"]
                log(
                    f"  fold={fold} {model_name:<24} C={best_c:<5g} "
                    f"PR={average_precision_score(y_va, raw):.4f} "
                    f"Brier(raw/int)={brier_score_loss(y_va, raw):.4f}/"
                    f"{brier_score_loss(y_va, p_int):.4f}"
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(tuning_rows), pd.DataFrame(coef_rows)


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["subset", "n_features", "base_model", "calibration"]
    for key, g in fold_df.groupby(keys, sort=False):
        subset, n_features, model, calibration = key
        pr_above = float((g["pr_lift_vs_base"] > 0).mean())
        roc_above = float((g["roc_auc"] > 0.50).mean())
        brier_better = float((g["brier_improvement_vs_base"] > 0).mean())
        logloss_better = float((g["logloss_improvement_vs_base"] > 0).mean())

        mean_pr_lift = float(g["pr_lift_vs_base"].mean())
        mean_brier_imp = float(g["brier_improvement_vs_base"].mean())
        mean_logloss_imp = float(g["logloss_improvement_vs_base"].mean())
        mean_roc = float(g["roc_auc"].mean())
        mean_pr = float(g["pr_auc"].mean())
        worst_pr_lift = float(g["pr_lift_vs_base"].min())

        gate = bool(
            mean_pr_lift > 0
            and pr_above >= 0.75
            and worst_pr_lift > 0
            and mean_roc > 0.50
            and roc_above >= 0.75
            and mean_brier_imp > 0
            and brier_better >= 0.75
            and mean_logloss_imp > 0
            and logloss_better >= 0.75
        )

        c_values = [float(x) for x in g["selected_C"].tolist()]
        rows.append({
            "subset": subset,
            "n_features": int(n_features),
            "base_model": model,
            "calibration": calibration,
            "n_folds": len(g),
            "passes_refinement_gate": int(gate),
            "mean_roc_auc": mean_roc,
            "mean_pr_auc": mean_pr,
            "mean_val_positive_rate": float(g["val_positive_rate"].mean()),
            "mean_pr_lift_vs_base": mean_pr_lift,
            "worst_pr_lift_vs_base": worst_pr_lift,
            "pr_above_base_fold_fraction": pr_above,
            "roc_above_050_fold_fraction": roc_above,
            "mean_brier": float(g["brier"].mean()),
            "mean_brier_improvement_vs_base": mean_brier_imp,
            "worst_brier_improvement_vs_base": float(g["brier_improvement_vs_base"].min()),
            "brier_better_fold_fraction": brier_better,
            "mean_log_loss": float(g["log_loss"].mean()),
            "mean_logloss_improvement_vs_base": mean_logloss_imp,
            "logloss_better_fold_fraction": logloss_better,
            "mean_ece_10": float(g["ece_10"].mean()),
            "selected_C_values": "|".join(f"{x:g}" for x in c_values),
            "selected_C_mode": float(Counter(c_values).most_common(1)[0][0]),
        })

    out = pd.DataFrame(rows)
    # Ranking: gate first; then robust PR, calibration, compactness.
    out = out.sort_values(
        [
            "passes_refinement_gate",
            "pr_above_base_fold_fraction",
            "worst_pr_lift_vs_base",
            "mean_pr_auc",
            "mean_roc_auc",
            "brier_better_fold_fraction",
            "mean_brier_improvement_vs_base",
            "logloss_better_fold_fraction",
            "n_features",
        ],
        ascending=[False, False, False, False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def summarize_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, model, feat), g in coef_df.groupby(
        ["subset", "base_model", "feature"], sort=False
    ):
        signs = g["sign"].to_numpy(dtype=int)
        pos = float(np.mean(signs > 0))
        neg = float(np.mean(signs < 0))
        rows.append({
            "subset": subset,
            "base_model": model,
            "feature": feat,
            "folds": len(g),
            "mean_coefficient": float(g["coefficient"].mean()),
            "median_coefficient": float(g["coefficient"].median()),
            "mean_abs_coefficient": float(g["abs_coefficient"].mean()),
            "std_coefficient": float(g["coefficient"].std(ddof=0)),
            "positive_fold_fraction": pos,
            "negative_fold_fraction": neg,
            "sign_consistency": max(pos, neg),
        })
    return pd.DataFrame(rows)


# =============================================================================
# DEV-ONLY REFIT
# =============================================================================


def choose_c_on_full_dev(
    dev: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    c_grid: Sequence[float],
    inner_splits: int,
    random_state: int,
) -> float:
    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev["__target__"].astype(int)
    best_c, _ = tune_c_inner(X, y, model_name, c_grid, inner_splits, random_state)
    return best_c


def save_dev_only_candidate(
    path: Path,
    dev: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    calibration: str,
    c_value: float,
    inner_splits: int,
    random_state: int,
    meta: Dict,
) -> None:
    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev["__target__"].astype(int)

    oof_p, oof_y = temporal_oof_fixed_c(
        X, y, model_name, c_value, inner_splits, random_state + 1000
    )
    calibrator: Dict[str, object] = {"kind": calibration}
    if calibration == "platt_temporal_oof":
        calibrator["model"] = fit_platt(oof_p, oof_y)
    elif calibration == "intercept_temporal_oof":
        calibrator["delta"] = fit_intercept_shift(oof_p, oof_y)
    elif calibration == "prior_corrected":
        calibrator["train_prevalence"] = float(y.mean())
    elif calibration == "raw":
        pass
    else:
        raise ValueError(calibration)

    model = build_model(model_name, c_value, random_state + 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    artifact = {
        "status": "DEV_ONLY_NOT_PRODUCTION",
        "features": list(features),
        "base_model": model_name,
        "C": float(c_value),
        "calibration": calibrator,
        "pipeline": model,
        "development_rows": int(len(dev)),
        "development_positive_rate": float(y.mean()),
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


# =============================================================================
# REPORT
# =============================================================================


def write_report(
    path: Path,
    split: SplitInfo,
    subsets: Dict[str, List[str]],
    ranking_features: pd.DataFrame,
    summary: pd.DataFrame,
    fold_df: pd.DataFrame,
    c_grid: Sequence[float],
) -> None:
    lines: List[str] = []
    lines += [
        "# First Half 37 — Compact Refinement V1",
        "",
        "## Protection du FINAL HOLDOUT",
        "",
        f"- Dataset total : **{split.total_rows}**",
        f"- DEVELOPMENT évalué : **{split.development_rows}**",
        f"- FINAL HOLDOUT réservé : **{split.holdout_rows}**",
        f"- DEVELOPMENT : `{split.development_start}` -> `{split.development_end}`",
        f"- FINAL HOLDOUT : `{split.holdout_start}` -> `{split.holdout_end}`",
        "- **Aucune prédiction ni métrique du FINAL HOLDOUT n'est calculée.**",
        "",
        "## Schémas testés",
        "",
    ]
    for name, feats in subsets.items():
        lines.append(f"- `{name}` : **{len(feats)}** features — " + ", ".join(f"`{x}`" for x in feats))

    lines += [
        "",
        "## Réglage régularisation",
        "",
        "- C-grid : `" + ", ".join(f"{x:g}" for x in c_grid) + "`",
        "- Le C de chaque fold externe est choisi uniquement dans son TRAIN externe.",
        "- Critère interne : stabilité PR > baseline, lift PR moyen, PR-AUC, pire lift, ROC, puis C plus petit en tie-break.",
        "",
        "## Ranking DEVELOPMENT",
        "",
        "```",
        summary.head(30).to_string(index=False),
        "```",
        "",
        "## Gate de raffinement",
        "",
        "Un candidat passe seulement si : PR lift moyen > 0, PR > baseline sur >=75% des folds, pire PR lift > 0, ROC moyen >0.50, ROC >0.50 sur >=75% des folds, Brier moyen meilleur que baseline sur >=75% des folds, et LogLoss moyen meilleur que baseline sur >=75% des folds.",
        "",
    ]

    passed = summary.loc[summary["passes_refinement_gate"].eq(1)]
    if passed.empty:
        lines += [
            "**AUCUN pipeline ne passe le gate de raffinement.**",
            "",
            "Le FINAL HOLDOUT doit rester fermé.",
            "",
        ]
    else:
        winner = passed.iloc[0]
        lines += [
            f"**{len(passed)} pipeline(s) passent le gate.**",
            "",
            f"Candidat #1 interne : `{winner['subset']} + {winner['base_model']} + {winner['calibration']}`.",
            "",
            "Même dans ce cas, le FINAL HOLDOUT reste fermé dans ce script. L'étape suivante doit figer définitivement le pipeline puis ouvrir le holdout une seule fois.",
            "",
        ]

    if not summary.empty:
        best = summary.iloc[0]
        detail = fold_df.loc[
            fold_df["subset"].eq(best["subset"])
            & fold_df["base_model"].eq(best["base_model"])
            & fold_df["calibration"].eq(best["calibration"])
        ]
        cols = [
            "fold", "selected_C", "train_rows", "val_rows",
            "train_positive_rate", "val_positive_rate", "roc_auc", "pr_auc",
            "pr_lift_vs_base", "brier", "brier_improvement_vs_base",
            "log_loss", "logloss_improvement_vs_base", "ece_10",
            "mean_probability",
        ]
        lines += [
            "## Détail folds du rang #1",
            "",
            "```",
            detail[cols].to_string(index=False),
            "```",
            "",
        ]

    lines += [
        "## Priorité / stabilité utilisée pour les sous-ensembles",
        "",
        "```",
        ranking_features.to_string(index=False),
        "```",
        "",
        "## Règle de décision",
        "",
        "Ne pas ouvrir le FINAL HOLDOUT à partir de ce rapport si aucun pipeline ne passe le gate. S'il existe un gagnant, figer son schéma, son type de modèle, sa méthode de calibration et la procédure de tuning avant l'évaluation finale unique.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_c_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v <= 0:
            raise ValueError("Tous les C doivent être > 0")
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("C-grid vide")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FH37 compact refinement before final holdout")
    p.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    p.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    p.add_argument("--coefficient-stability", type=Path, default=DEFAULT_COEF_STABILITY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    p.add_argument("--outer-splits", type=int, default=4)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument(
        "--c-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_C_GRID),
        help="ex: 0.01,0.03,0.1,0.3,1,3,10",
    )
    p.add_argument("--random-state", type=int, default=87)
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: TOP20/COMPACT12/COMPACT10, C=0.1,1,3, outer=3 inner=3",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.model_ready)
    dev, holdout, split = split_dev_holdout(df, args.holdout_fraction)
    top20, selected_tab = load_top20(args.selected, df)
    stability = load_stability(args.coefficient_stability, top20)
    subsets, feature_priority = build_compact_subsets(top20, selected_tab, stability)

    c_grid = parse_c_grid(args.c_grid)
    outer_splits = int(args.outer_splits)
    inner_splits = int(args.inner_splits)

    if args.quick:
        wanted = [x for x in ("TOP20_CONTROL", "COMPACT12", "COMPACT10") if x in subsets]
        subsets = {k: subsets[k] for k in wanted}
        c_grid = [0.1, 1.0, 3.0]
        outer_splits = 3
        inner_splits = 3

    log("=" * 108)
    log("FIRST HALF 37 — COMPACT REFINEMENT BEFORE FINAL HOLDOUT V1")
    log("=" * 108)
    log(f"MODEL_READY : {args.model_ready}")
    log(f"Selected    : {args.selected}")
    log(f"Stability   : {args.coefficient_stability}")
    log(f"Output      : {out_dir}")
    log()
    log(
        f"Dataset={len(df)} | DEVELOPMENT={len(dev)} | FINAL HOLDOUT RESERVE={len(holdout)}"
    )
    log(f"Dates DEV : {split.development_start} -> {split.development_end}")
    log(f"Dates HOLDOUT : {split.holdout_start} -> {split.holdout_end}")
    log("FINAL HOLDOUT : aucune prediction ni metrique ne sera calculee.")
    log()
    log("Sous-ensembles :")
    for name, feats in subsets.items():
        log(f"  {name:<20} {len(feats):>2} features")
    log(f"C-grid : {', '.join(f'{x:g}' for x in c_grid)}")

    fold_df, tuning_df, coef_df = run_refinement(
        dev,
        subsets,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        c_grid=c_grid,
        random_state=int(args.random_state),
    )
    summary = summarize(fold_df)
    coef_summary = summarize_coefficients(coef_df)

    # Save tables before any refit.
    json_dump(out_dir / "first_half_37_compact_subsets.json", subsets)
    feature_priority.to_csv(
        out_dir / "first_half_37_compact_feature_priority.csv", index=False
    )
    tuning_df.to_csv(out_dir / "first_half_37_c_tuning.csv", index=False)
    fold_df.to_csv(out_dir / "first_half_37_refinement_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "first_half_37_refinement_summary.csv", index=False)
    summary.to_csv(out_dir / "first_half_37_refinement_ranking.csv", index=False)
    coef_summary.to_csv(
        out_dir / "first_half_37_refined_coefficient_stability.csv", index=False
    )

    # DEV-only candidates: top 3 internal ranking. The C is reselected on full
    # DEVELOPMENT only. No HOLDOUT access.
    cand_dir = out_dir / "candidates_dev_only"
    cand_dir.mkdir(parents=True, exist_ok=True)
    saved_candidates = []
    for _, row in summary.head(3).iterrows():
        subset = str(row["subset"])
        model_name = str(row["base_model"])
        calibration = str(row["calibration"])
        features = subsets[subset]
        full_c = choose_c_on_full_dev(
            dev,
            features,
            model_name,
            c_grid,
            inner_splits=inner_splits,
            random_state=int(args.random_state) + 50000,
        )
        safe_name = f"{subset}_{model_name}_{calibration}_DEV_ONLY.joblib"
        dest = cand_dir / safe_name
        save_dev_only_candidate(
            dest,
            dev,
            features,
            model_name,
            calibration,
            full_c,
            inner_splits,
            int(args.random_state) + 60000,
            meta={
                "internal_rank": int(row["rank"]),
                "passes_refinement_gate": int(row["passes_refinement_gate"]),
                "mean_roc_auc": float(row["mean_roc_auc"]),
                "mean_pr_auc": float(row["mean_pr_auc"]),
                "mean_brier": float(row["mean_brier"]),
                "mean_log_loss": float(row["mean_log_loss"]),
                "final_holdout_used": False,
            },
        )
        saved_candidates.append({
            "rank": int(row["rank"]),
            "path": str(dest),
            "subset": subset,
            "base_model": model_name,
            "calibration": calibration,
            "C_full_dev": full_c,
        })

    report_path = out_dir / "first_half_37_refinement_report.md"
    write_report(
        report_path,
        split,
        subsets,
        feature_priority,
        summary,
        fold_df,
        c_grid,
    )

    passed = summary.loc[summary["passes_refinement_gate"].eq(1)]
    winner = summary.iloc[0].to_dict() if len(summary) else None
    manifest = {
        "schema_version": "first_half_37_compact_refinement_v1",
        "split": asdict(split),
        "final_holdout_used": False,
        "subsets": subsets,
        "c_grid": c_grid,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "winner_internal": winner,
        "pipelines_passing_gate": int(len(passed)),
        "saved_dev_only_candidates": saved_candidates,
        "decision": (
            "PIPELINE_CAN_BE_FROZEN_BEFORE_ONE_TIME_HOLDOUT"
            if len(passed)
            else "KEEP_FINAL_HOLDOUT_CLOSED"
        ),
    }
    json_dump(out_dir / "first_half_37_refinement_manifest.json", manifest)

    log()
    log("=" * 108)
    log("RAFFINEMENT FIRST HALF 37 TERMINE")
    log("=" * 108)
    log(f"Configurations evaluees : {len(summary)}")
    log(f"Pipelines passant le gate : {len(passed)}")
    if winner:
        log(
            f"#1 interne : {winner['subset']} + {winner['base_model']} + "
            f"{winner['calibration']} | features={int(winner['n_features'])} | "
            f"ROC={float(winner['mean_roc_auc']):.4f} | "
            f"PR={float(winner['mean_pr_auc']):.4f} | "
            f"Brier={float(winner['mean_brier']):.4f} | "
            f"LogLoss={float(winner['mean_log_loss']):.4f} | "
            f"gate={'PASS' if int(winner['passes_refinement_gate']) else 'FAIL'}"
        )
    log(f"Rapport : {report_path}")
    log(f"Ranking : {out_dir / 'first_half_37_refinement_ranking.csv'}")
    log(f"Manifest: {out_dir / 'first_half_37_refinement_manifest.json'}")
    log("IMPORTANT : le FINAL HOLDOUT est toujours ferme.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par utilisateur.", file=sys.stderr)
        raise SystemExit(130)
