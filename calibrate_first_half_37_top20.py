#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FIRST HALF 37 — TEMPORAL CALIBRATION AUDIT TOP-20 (V1)
================================================================================

But
---
Prendre le meilleur schéma interne actuel (TOP_20) et déterminer si le signal
observé sur DEVELOPMENT peut être transformé en probabilités mieux calibrées,
SANS ouvrir le FINAL HOLDOUT.

Base models
-----------
- logistic_l2
- logistic_l2_balanced

Calibration comparée
---------------------
- raw
- prior_corrected          (balanced uniquement)
- platt_temporal_oof
- isotonic_temporal_oof

Protection anti-fuite
---------------------
Pour chaque fold temporel externe :
1) le modèle brut est entraîné uniquement sur le TRAIN externe;
2) les calibrateurs Platt / isotonic sont appris sur des prédictions OOF
   temporelles générées à l'intérieur du TRAIN externe;
3) le calibrateur est ensuite appliqué aux probabilités du VAL externe;
4) aucune prédiction ni métrique n'est calculée sur le FINAL HOLDOUT.

Le FINAL HOLDOUT reste les 15 % les plus récents, comme dans les étapes
précédentes.

Sorties
-------
outputs/first_half_37/calibration_v1/
    first_half_37_calibration_fold_metrics.csv
    first_half_37_calibration_summary.csv
    first_half_37_reliability_bins.csv
    first_half_37_coefficient_stability.csv
    first_half_37_calibration_report.md
    first_half_37_calibration_manifest.json
    candidates_dev_only/
        TOP20_logistic_l2_<calibration>_DEV_ONLY.joblib
        TOP20_logistic_l2_balanced_<calibration>_DEV_ONLY.joblib

IMPORTANT
---------
Les objets joblib sont DEV ONLY. Ils ne doivent pas être déployés ni utilisés
pour annoncer une performance finale. Le FINAL HOLDOUT ne doit être ouvert
qu'une fois le pipeline définitivement figé.
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

from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "calibration_v1"
)

TARGET = "goal_after_37_before_ht"
TIME_COL = "kickoff_utc"
ID_COL = "match_id"

BASE_MODELS = ("logistic_l2", "logistic_l2_balanced")


# =============================================================================
# DATA / HELPERS
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

    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any():
        raise ValueError("Target non résolu dans MODEL_READY.")
    vals = set(target.astype(int).unique())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Target invalide : {sorted(vals)}")

    out = df.copy()
    out["__kickoff__"] = dt
    out["__target__"] = target.astype(int)

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


def load_top_features(path: Path, df: pd.DataFrame, n: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"selected_features introuvable : {path}")

    tab = pd.read_csv(path)
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

    feats: List[str] = []
    seen = set()
    for f in tab["feature"].astype(str):
        if f in df.columns and f not in seen:
            feats.append(f)
            seen.add(f)
        if len(feats) >= n:
            break

    if len(feats) != n:
        raise ValueError(f"TOP_{n} impossible : seulement {len(feats)} features disponibles.")
    return feats


def build_base_model(name: str, random_state: int) -> Pipeline:
    if name not in BASE_MODELS:
        raise KeyError(name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                class_weight=("balanced" if name.endswith("balanced") else None),
                max_iter=4000,
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


def calibration_line(y_true: Sequence[int], prob: np.ndarray) -> Tuple[float, float]:
    """Calibration intercept/slope fitted only for reporting on a validation block."""
    y = np.asarray(y_true, dtype=int)
    x = logit(prob).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    try:
        lr = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=3000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    except Exception:
        # Compatibility with older sklearn where penalty=None may fail.
        lr = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=3000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


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


def prior_correct_balanced(prob: np.ndarray, target_prevalence: float) -> np.ndarray:
    """
    Correct the 0.5 effective prior induced by class_weight='balanced' back to
    the observed training prevalence. This preserves ranking exactly.
    """
    pi = float(np.clip(target_prevalence, 1e-6, 1 - 1e-6))
    shift = math.log(pi / (1.0 - pi))  # effective balanced prior = 0.5 => logit 0
    return sigmoid(logit(prob) + shift)


# =============================================================================
# TEMPORAL CALIBRATORS
# =============================================================================

@dataclass
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        x = logit(p).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(np.asarray(p, dtype=float)), 1e-8, 1 - 1e-8)


def fit_platt(prob: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    x = logit(prob).reshape(-1, 1)
    try:
        lr = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=4000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    except Exception:
        lr = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=4000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(x, y)
    return PlattCalibrator(lr)


def fit_isotonic(prob: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(prob, dtype=float), np.asarray(y, dtype=int))
    return IsotonicCalibrator(iso)


def temporal_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_model_name: str,
    inner_splits: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate strictly forward inner-OOF predictions inside an outer TRAIN."""
    n = len(X_train)
    if n < 120:
        raise ValueError(f"TRAIN externe trop petit pour calibration OOF : {n}")

    max_splits = min(inner_splits, max(2, n // 60))
    splitter = TimeSeriesSplit(n_splits=max_splits)

    probs: List[float] = []
    ys: List[int] = []
    positions: List[int] = []

    for k, (tr, va) in enumerate(splitter.split(X_train), start=1):
        y_tr = y_train.iloc[tr]
        y_va = y_train.iloc[va]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue

        model = build_base_model(base_model_name, random_state + k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train.iloc[tr], y_tr)
        p = model.predict_proba(X_train.iloc[va])[:, 1]

        probs.extend(np.asarray(p, dtype=float).tolist())
        ys.extend(y_va.astype(int).tolist())
        positions.extend(va.astype(int).tolist())

    if len(probs) < 80 or len(set(ys)) < 2:
        raise ValueError(
            f"Pas assez de prédictions OOF internes pour calibrer : n={len(probs)}"
        )

    return (
        np.asarray(probs, dtype=float),
        np.asarray(ys, dtype=int),
        np.asarray(positions, dtype=int),
    )


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_temporal_calibration(
    dev: pd.DataFrame,
    features: Sequence[str],
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y_all = dev["__target__"].astype(int)

    outer = TimeSeriesSplit(n_splits=outer_splits)
    fold_rows: List[Dict] = []
    pred_rows: List[Dict] = []
    coef_rows: List[Dict] = []

    for fold, (tr, va) in enumerate(outer.split(dev), start=1):
        X_tr, X_va = X_all.iloc[tr], X_all.iloc[va]
        y_tr, y_va = y_all.iloc[tr], y_all.iloc[va]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            raise ValueError(f"Fold externe {fold} sans les deux classes.")

        train_rate = float(y_tr.mean())
        val_rate = float(y_va.mean())
        base_prob = np.full(len(y_va), train_rate, dtype=float)
        baseline = metric_bundle(y_va, base_prob)

        log(
            f"Fold {fold}/{outer_splits} | TRAIN={len(tr)} VAL={len(va)} "
            f"| train_rate={train_rate:.4f} val_rate={val_rate:.4f}"
        )

        for model_idx, model_name in enumerate(BASE_MODELS, start=1):
            seed = random_state + fold * 100 + model_idx * 10

            # Base model fit on the complete outer TRAIN.
            base_model = build_base_model(model_name, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_model.fit(X_tr, y_tr)
            raw_val = base_model.predict_proba(X_va)[:, 1]

            # Coefficients standardized because StandardScaler is in the pipeline.
            coefs = np.asarray(base_model.named_steps["model"].coef_[0], dtype=float)
            for feature, coef in zip(features, coefs):
                coef_rows.append({
                    "model": model_name,
                    "fold": fold,
                    "feature": feature,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "sign": int(np.sign(coef)),
                })

            # Inner OOF predictions are the ONLY data used to fit calibration maps.
            oof_prob, oof_y, oof_pos = temporal_oof_predictions(
                X_tr,
                y_tr,
                model_name,
                inner_splits=inner_splits,
                random_state=seed + 1000,
            )

            platt = fit_platt(oof_prob, oof_y)
            iso = fit_isotonic(oof_prob, oof_y)

            methods: Dict[str, np.ndarray] = {
                "raw": raw_val,
                "platt_temporal_oof": platt.predict(raw_val),
                "isotonic_temporal_oof": iso.predict(raw_val),
            }
            if model_name == "logistic_l2_balanced":
                methods["prior_corrected"] = prior_correct_balanced(raw_val, train_rate)

            for method, p in methods.items():
                met = metric_bundle(y_va, p)
                cal_intercept, cal_slope = calibration_line(y_va, p)
                ece = ece_score(y_va, p, n_bins=10)

                fold_rows.append({
                    "base_model": model_name,
                    "calibration": method,
                    "fold": fold,
                    "n_features": len(features),
                    "train_rows": len(tr),
                    "val_rows": len(va),
                    "train_start": dev["__kickoff__"].iloc[tr[0]].isoformat(),
                    "train_end": dev["__kickoff__"].iloc[tr[-1]].isoformat(),
                    "val_start": dev["__kickoff__"].iloc[va[0]].isoformat(),
                    "val_end": dev["__kickoff__"].iloc[va[-1]].isoformat(),
                    "train_positive_rate": train_rate,
                    "val_positive_rate": val_rate,
                    "calibration_oof_rows": int(len(oof_y)),
                    "calibration_oof_positive_rate": float(oof_y.mean()),
                    **met,
                    "ece_10": ece,
                    "calibration_intercept_report": cal_intercept,
                    "calibration_slope_report": cal_slope,
                    "baseline_pr_auc": baseline["pr_auc"],
                    "baseline_brier": baseline["brier"],
                    "baseline_log_loss": baseline["log_loss"],
                    "pr_lift_vs_base": met["pr_auc"] - baseline["pr_auc"],
                    "brier_improvement_vs_base": baseline["brier"] - met["brier"],
                    "logloss_improvement_vs_base": baseline["log_loss"] - met["log_loss"],
                })

                for local_i, row_idx in enumerate(va):
                    pred_rows.append({
                        "base_model": model_name,
                        "calibration": method,
                        "fold": fold,
                        "row_index_dev": int(row_idx),
                        "match_id": str(dev[ID_COL].iloc[row_idx]),
                        "kickoff_utc": dev["__kickoff__"].iloc[row_idx].isoformat(),
                        "target": int(y_va.iloc[local_i]),
                        "probability": float(p[local_i]),
                    })

            log(
                f"  {model_name:<24} raw_PR={average_precision_score(y_va, raw_val):.4f} "
                f"raw_Brier={brier_score_loss(y_va, raw_val):.4f} "
                f"platt_Brier={brier_score_loss(y_va, platt.predict(raw_val)):.4f} "
                f"iso_Brier={brier_score_loss(y_va, iso.predict(raw_val)):.4f}"
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows), pd.DataFrame(coef_rows)


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["base_model", "calibration"]
    for (base_model, calibration), g in fold_df.groupby(keys, sort=False):
        pr_above = (g["pr_lift_vs_base"] > 0).mean()
        roc_above = (g["roc_auc"] > 0.50).mean()
        brier_better = (g["brier_improvement_vs_base"] > 0).mean()
        logloss_better = (g["logloss_improvement_vs_base"] > 0).mean()

        mean_pr_lift = float(g["pr_lift_vs_base"].mean())
        mean_brier_imp = float(g["brier_improvement_vs_base"].mean())
        mean_logloss_imp = float(g["logloss_improvement_vs_base"].mean())
        mean_roc = float(g["roc_auc"].mean())

        gate = bool(
            mean_pr_lift > 0
            and pr_above >= 0.75
            and mean_roc > 0.50
            and roc_above >= 0.50
            and mean_brier_imp > 0
            and brier_better >= 0.75
            and mean_logloss_imp > 0
            and logloss_better >= 0.75
        )

        rows.append({
            "base_model": base_model,
            "calibration": calibration,
            "n_folds": len(g),
            "passes_calibration_gate": int(gate),
            "mean_roc_auc": float(g["roc_auc"].mean()),
            "mean_pr_auc": float(g["pr_auc"].mean()),
            "mean_val_positive_rate": float(g["val_positive_rate"].mean()),
            "mean_pr_lift_vs_base": mean_pr_lift,
            "pr_above_base_fold_fraction": float(pr_above),
            "roc_above_050_fold_fraction": float(roc_above),
            "mean_brier": float(g["brier"].mean()),
            "mean_brier_improvement_vs_base": mean_brier_imp,
            "brier_better_fold_fraction": float(brier_better),
            "mean_log_loss": float(g["log_loss"].mean()),
            "mean_logloss_improvement_vs_base": mean_logloss_imp,
            "logloss_better_fold_fraction": float(logloss_better),
            "mean_ece_10": float(g["ece_10"].mean()),
            "median_calibration_intercept_report": float(g["calibration_intercept_report"].median()),
            "median_calibration_slope_report": float(g["calibration_slope_report"].median()),
            "worst_pr_lift_vs_base": float(g["pr_lift_vs_base"].min()),
            "worst_brier_improvement_vs_base": float(g["brier_improvement_vs_base"].min()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Rank: pass gate first, then calibration quality, then discrimination.
    out = out.sort_values(
        [
            "passes_calibration_gate",
            "mean_brier_improvement_vs_base",
            "mean_logloss_improvement_vs_base",
            "mean_pr_lift_vs_base",
            "mean_roc_auc",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def reliability_bins(pred_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for (base_model, calibration), g in pred_df.groupby(
        ["base_model", "calibration"], sort=False
    ):
        p = g["probability"].to_numpy(dtype=float)
        y = g["target"].to_numpy(dtype=int)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append({
                "base_model": base_model,
                "calibration": calibration,
                "bin": i + 1,
                "prob_min": lo,
                "prob_max": hi,
                "n": n,
                "mean_probability": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
                "calibration_gap": float(y[mask].mean() - p[mask].mean()),
            })
    return pd.DataFrame(rows)


def summarize_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature), g in coef_df.groupby(["model", "feature"], sort=False):
        c = g["coefficient"].to_numpy(dtype=float)
        signs = np.sign(c)
        pos = float((signs > 0).mean())
        neg = float((signs < 0).mean())
        sign_consistency = max(pos, neg)
        rows.append({
            "model": model,
            "feature": feature,
            "folds": len(g),
            "mean_coefficient": float(np.mean(c)),
            "median_coefficient": float(np.median(c)),
            "mean_abs_coefficient": float(np.mean(np.abs(c))),
            "std_coefficient": float(np.std(c, ddof=0)),
            "positive_fold_fraction": pos,
            "negative_fold_fraction": neg,
            "sign_consistency": sign_consistency,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["model", "sign_consistency", "mean_abs_coefficient"],
            ascending=[True, False, False],
            kind="stable",
        ).reset_index(drop=True)
    return out


# =============================================================================
# DEV-ONLY REFIT FOR THE INTERNAL WINNER
# =============================================================================

def fit_dev_only_artifact(
    dev: pd.DataFrame,
    features: Sequence[str],
    base_model_name: str,
    calibration_method: str,
    inner_splits: int,
    random_state: int,
) -> Dict:
    """
    Refit DEVELOPMENT-only objects for reproducibility. The artifact is a plain
    dict containing sklearn objects + metadata, so joblib can reload it from a
    different script without requiring a custom __main__ class.
    """
    X = dev.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev["__target__"].astype(int)

    oof_prob, oof_y, _ = temporal_oof_predictions(
        X, y, base_model_name, inner_splits=inner_splits, random_state=random_state + 1000
    )

    calibrator = None
    calibrator_input = None
    if calibration_method == "platt_temporal_oof":
        # Store the underlying sklearn LogisticRegression, not the wrapper.
        calibrator = fit_platt(oof_prob, oof_y).model
        calibrator_input = "logit_probability"
    elif calibration_method == "isotonic_temporal_oof":
        calibrator = fit_isotonic(oof_prob, oof_y).model
        calibrator_input = "raw_probability"
    elif calibration_method == "prior_corrected":
        calibrator_input = "balanced_prior_to_development_prior"
    elif calibration_method == "raw":
        calibrator_input = "none"
    else:
        raise KeyError(calibration_method)

    base = build_base_model(base_model_name, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base.fit(X, y)

    return {
        "schema_version": "first_half_37_calibration_dev_only_v1",
        "dev_only": True,
        "base_model_name": base_model_name,
        "features": list(features),
        "base_model": base,
        "calibration_method": calibration_method,
        "calibrator": calibrator,
        "calibrator_input": calibrator_input,
        "development_prevalence": float(y.mean()),
        "calibration_oof_rows": int(len(oof_y)),
    }


# =============================================================================
# REPORT
# =============================================================================


def make_report(
    path: Path,
    split: SplitInfo,
    features: Sequence[str],
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    coefs: pd.DataFrame,
) -> None:
    best = summary.iloc[0] if not summary.empty else None
    gate_n = int(summary["passes_calibration_gate"].sum()) if not summary.empty else 0

    lines = []
    lines.append("# First Half 37 — Temporal Calibration Audit TOP-20 V1")
    lines.append("")
    lines.append("## Protection du FINAL HOLDOUT")
    lines.append("")
    lines.append(f"- Dataset total : **{split.total_rows}**")
    lines.append(f"- DEVELOPMENT évalué : **{split.development_rows}**")
    lines.append(f"- FINAL HOLDOUT réservé : **{split.holdout_rows}**")
    lines.append(f"- DEVELOPMENT : `{split.development_start}` -> `{split.development_end}`")
    lines.append(f"- FINAL HOLDOUT : `{split.holdout_start}` -> `{split.holdout_end}`")
    lines.append("- **Aucune prédiction ni métrique du FINAL HOLDOUT n'est calculée.**")
    lines.append("")
    lines.append("## Schéma")
    lines.append("")
    lines.append(f"- TOP-20 : **{len(features)}** features")
    for i, f in enumerate(features, start=1):
        lines.append(f"  {i}. `{f}`")
    lines.append("")
    lines.append("## Méthode")
    lines.append("")
    lines.append("- 4 folds externes TimeSeriesSplit sur DEVELOPMENT.")
    lines.append("- Platt / isotonic appris uniquement sur des prédictions OOF temporelles à l'intérieur du TRAIN externe.")
    lines.append("- Prior correction disponible uniquement pour la logistic équilibrée.")
    lines.append("- Baseline = taux positif du TRAIN externe.")
    lines.append("")
    lines.append("## Ranking calibration")
    lines.append("")
    if summary.empty:
        lines.append("Aucun résultat.")
    else:
        lines.append("```")
        lines.append(summary.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(
        "Un candidat passe seulement si : PR-AUC moyenne > baseline, PR > baseline sur >=75% des folds, "
        "ROC moyen >0.50, ROC >0.50 sur >=50% des folds, Brier moyen meilleur que baseline sur >=75% "
        "des folds, et LogLoss moyen meilleur que baseline sur >=75% des folds."
    )
    lines.append("")
    lines.append(f"- Candidats passant le gate : **{gate_n}**")
    if best is not None:
        lines.append(
            f"- Rang #1 interne : `{best['base_model']} + {best['calibration']}` | "
            f"ROC={best['mean_roc_auc']:.4f} | PR={best['mean_pr_auc']:.4f} | "
            f"Brier={best['mean_brier']:.4f} | LogLoss={best['mean_log_loss']:.4f} | "
            f"gate={'PASS' if int(best['passes_calibration_gate']) else 'FAIL'}"
        )
    lines.append("")
    lines.append("## Détail folds du rang #1")
    lines.append("")
    if best is not None:
        g = folds[
            (folds["base_model"] == best["base_model"])
            & (folds["calibration"] == best["calibration"])
        ].copy()
        cols = [
            "fold", "train_rows", "val_rows", "train_positive_rate", "val_positive_rate",
            "roc_auc", "pr_auc", "pr_lift_vs_base", "brier",
            "brier_improvement_vs_base", "log_loss", "logloss_improvement_vs_base",
            "ece_10", "mean_probability", "calibration_intercept_report",
            "calibration_slope_report",
        ]
        lines.append("```")
        lines.append(g[cols].to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Coefficients — stabilité des signes")
    lines.append("")
    if not coefs.empty:
        for model in BASE_MODELS:
            gg = coefs[coefs["model"] == model].head(20)
            lines.append(f"### {model}")
            lines.append("")
            lines.append("```")
            lines.append(gg.to_string(index=False))
            lines.append("```")
            lines.append("")
    lines.append("## Règle de décision")
    lines.append("")
    if gate_n > 0:
        lines.append(
            "Au moins un pipeline passe le gate interne. Il devient candidat à figer avant une ouverture unique du FINAL HOLDOUT. "
            "Ne pas modifier ses features/calibration après ouverture du holdout."
        )
    else:
        lines.append(
            "Aucun pipeline ne passe encore le gate. Le FINAL HOLDOUT doit rester fermé; il faut revoir la calibration ou le schéma sur DEVELOPMENT."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="First Half 37 — temporal calibration audit TOP-20",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    p.add_argument("--selected-features", type=Path, default=DEFAULT_SELECTED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    p.add_argument("--outer-splits", type=int, default=4)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--random-state", type=int, default=87)
    p.add_argument("--no-save-dev-candidate", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 108)
    log("FIRST HALF 37 — TEMPORAL CALIBRATION AUDIT TOP-20 V1")
    log("=" * 108)
    log(f"MODEL_READY : {args.model_ready}")
    log(f"Selected    : {args.selected_features}")
    log(f"Output      : {args.output_dir}")
    log("")

    df = load_dataset(args.model_ready)
    dev, holdout, split = split_dev_holdout(df, args.holdout_fraction)
    features = load_top_features(args.selected_features, df, args.top_n)

    # Do not inspect holdout target beyond dataset structural validation.
    log(
        f"Dataset={len(df)} | DEVELOPMENT={len(dev)} | FINAL HOLDOUT RESERVE={len(holdout)}"
    )
    log(f"Dates DEV : {split.development_start} -> {split.development_end}")
    log(f"Dates HOLDOUT : {split.holdout_start} -> {split.holdout_end}")
    log("FINAL HOLDOUT : aucune prediction ni metrique ne sera calculee.")
    log("")
    log(f"TOP_{args.top_n} features :")
    for i, f in enumerate(features, start=1):
        log(f"  {i:02d}. {f}")
    log("")

    folds, preds, coef_raw = run_temporal_calibration(
        dev,
        features,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
    )
    summary = summarize_metrics(folds)
    rel = reliability_bins(preds, n_bins=10)
    coef_summary = summarize_coefficients(coef_raw)

    folds_path = args.output_dir / "first_half_37_calibration_fold_metrics.csv"
    summary_path = args.output_dir / "first_half_37_calibration_summary.csv"
    rel_path = args.output_dir / "first_half_37_reliability_bins.csv"
    coef_path = args.output_dir / "first_half_37_coefficient_stability.csv"
    preds_path = args.output_dir / "first_half_37_development_oof_predictions.csv"
    report_path = args.output_dir / "first_half_37_calibration_report.md"
    manifest_path = args.output_dir / "first_half_37_calibration_manifest.json"

    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    rel.to_csv(rel_path, index=False)
    coef_summary.to_csv(coef_path, index=False)
    preds.to_csv(preds_path, index=False)

    make_report(report_path, split, features, summary, folds, coef_summary)

    best = summary.iloc[0].to_dict() if not summary.empty else None
    manifest = {
        "schema_version": "first_half_37_calibration_v1",
        "split": asdict(split),
        "top_n": args.top_n,
        "features": list(features),
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "final_holdout_opened": False,
        "best_internal": best,
        "passing_gate_count": int(summary["passes_calibration_gate"].sum()) if not summary.empty else 0,
        "outputs": {
            "fold_metrics": str(folds_path),
            "summary": str(summary_path),
            "reliability_bins": str(rel_path),
            "coefficient_stability": str(coef_path),
            "development_oof_predictions": str(preds_path),
            "report": str(report_path),
        },
    }

    if best is not None and not args.no_save_dev_candidate:
        artifact = fit_dev_only_artifact(
            dev,
            features,
            base_model_name=str(best["base_model"]),
            calibration_method=str(best["calibration"]),
            inner_splits=args.inner_splits,
            random_state=args.random_state + 5000,
        )
        cand_dir = args.output_dir / "candidates_dev_only"
        cand_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"TOP{args.top_n}_{best['base_model']}_{best['calibration']}_DEV_ONLY.joblib"
        )
        cand_path = cand_dir / fname
        joblib.dump(artifact, cand_path)
        manifest["dev_only_candidate"] = str(cand_path)

    json_dump(manifest_path, manifest)

    log("")
    log("=" * 108)
    log("CALIBRATION FIRST HALF 37 TERMINEE")
    log("=" * 108)
    log(f"Configurations calibration : {len(summary)}")
    log(
        f"Pipelines passant le gate : "
        f"{int(summary['passes_calibration_gate'].sum()) if not summary.empty else 0}"
    )
    if best is not None:
        log(
            f"#1 interne : {best['base_model']} + {best['calibration']} "
            f"| ROC={best['mean_roc_auc']:.4f} "
            f"| PR={best['mean_pr_auc']:.4f} "
            f"| Brier={best['mean_brier']:.4f} "
            f"| LogLoss={best['mean_log_loss']:.4f} "
            f"| gate={'PASS' if int(best['passes_calibration_gate']) else 'FAIL'}"
        )
    log(f"Rapport : {report_path}")
    log(f"Summary : {summary_path}")
    log(f"Manifest: {manifest_path}")
    log("IMPORTANT : le FINAL HOLDOUT est toujours ferme.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par utilisateur.", file=sys.stderr)
        raise SystemExit(130)
