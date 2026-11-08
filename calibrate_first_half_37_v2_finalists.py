#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 TEMPORAL CALIBRATION AUDIT

Finalistes:
  - TOP_8  + LogisticRegression L2
  - TOP_10 + LogisticRegression L2
  - TOP_12 + LogisticRegression L2

Décision déjà prise avant ce script:
  - C = 0.01, car le comparatif V2 l'a sélectionné sur tous les folds
    pour ces trois logistiques non équilibrées.

Comparaisons:
  - raw
  - platt_temporal_oof
  - isotonic_temporal_oof

Design:
  - DEVELOPMENT V2 = 2 105 matchs connus.
  - 5 folds temporels internes sur les 1 802 historiques.
  - 1 stress fold = les 303 matchs externes déjà consommés.
  - Les calibrateurs sont appris UNIQUEMENT sur des prédictions OOF
    temporelles produites à l'intérieur du TRAIN de chaque fold externe.
  - Les baselines Brier / LogLoss sont calculées correctement:
    on prédit le taux positif du TRAIN à chaque ligne de validation puis
    on mesure cette prédiction contre les vraies cibles de validation.
  - Aucun match postérieur au 19 septembre 2026 n'est lu.
  - Aucun futur holdout n'est utilisé.

Sorties:
  outputs/first_half_37/v2_calibration/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DEV = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "First_Half_37_V2_DEVELOPMENT.csv"
)
DEFAULT_RANKING = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "first_half_37_v2_robust_feature_ranking.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_calibration"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
STAGE_COL = "_v2_stage"

FINAL_SUBSET_SIZES = [8, 10, 12]
FIXED_C = 0.01
EPS = 1e-8


@dataclass
class FoldDef:
    name: str
    kind: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def safe_roc(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_pr(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")

    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        n = int(mask.sum())
        if n == 0:
            continue

        obs = float(y[mask].mean())
        pred = float(p[mask].mean())
        ece += (n / total) * abs(obs - pred)

    return float(ece)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Diagnostic: régression logistique y ~ logit(p).
    Retourne intercept, slope. Pas utilisée pour produire les prédictions.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)

    if len(np.unique(y)) < 2 or np.std(z) < 1e-12:
        return float("nan"), float("nan")

    try:
        m = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )
        m.fit(z, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def load_dev(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DEVELOPMENT V2 introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {TARGET, DATE_COL, ID_COL, STAGE_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    if df[DATE_COL].isna().any():
        raise ValueError("kickoff_utc non parsable.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target non résolu présent.")

    if df[ID_COL].duplicated().any():
        raise ValueError("match_id dupliqué dans DEVELOPMENT V2.")

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def load_ranked_features(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ranking robuste V2 introuvable: {path}")

    ranking = pd.read_csv(path)
    required = {"feature", "selected_v2", "robust_selection_score"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Colonnes absentes dans le ranking: {required - set(ranking.columns)}")

    selected = ranking[ranking["selected_v2"] == 1].copy()
    selected = selected.sort_values("robust_selection_score", ascending=False)

    features = selected["feature"].astype(str).tolist()

    if len(features) < max(FINAL_SUBSET_SIZES):
        raise RuntimeError(
            f"Seulement {len(features)} features sélectionnées, "
            f"mais TOP_{max(FINAL_SUBSET_SIZES)} est requis."
        )
    return features


def build_subsets(ranked_features: List[str]) -> Dict[str, List[str]]:
    return {
        f"TOP_{n}": ranked_features[:n]
        for n in FINAL_SUBSET_SIZES
    }


def make_outer_folds(df: pd.DataFrame, n_internal: int = 5) -> List[FoldDef]:
    orig_mask = df[STAGE_COL].astype(str).eq("original_consumed")
    ext_mask = df[STAGE_COL].astype(str).eq("external_consumed")

    orig_idx = np.where(orig_mask.to_numpy())[0]
    ext_idx = np.where(ext_mask.to_numpy())[0]

    if len(orig_idx) != 1802:
        print(f"WARNING: original_consumed={len(orig_idx)} (attendu 1802).")
    if len(ext_idx) != 303:
        print(f"WARNING: external_consumed={len(ext_idx)} (attendu 303).")

    if len(orig_idx) == 0 or len(ext_idx) == 0:
        raise RuntimeError("Stages original/external absents.")

    folds: List[FoldDef] = []

    tscv = TimeSeriesSplit(n_splits=n_internal)
    local = np.arange(len(orig_idx))

    for i, (tr_local, va_local) in enumerate(tscv.split(local), start=1):
        folds.append(
            FoldDef(
                name=f"INTERNAL_{i}",
                kind="internal",
                train_idx=orig_idx[tr_local],
                val_idx=orig_idx[va_local],
            )
        )

    folds.append(
        FoldDef(
            name="EXTERNAL_STRESS",
            kind="external_stress",
            train_idx=orig_idx,
            val_idx=ext_idx,
        )
    )
    return folds


def build_base_model(C: float = FIXED_C) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                class_weight=None,
                max_iter=5000,
                random_state=87,
            ),
        ),
    ])


def predict_temporal_oof(
    train_df: pd.DataFrame,
    features: List[str],
    C: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Génère des prédictions OOF temporelles exclusivement dans outer-TRAIN.
    Ces prédictions servent à apprendre Platt / Isotonic.

    Les premières lignes servant de TRAIN au premier split n'ont pas de
    prédiction OOF; elles ne sont donc pas utilisées pour calibration.
    """
    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    n = len(train_df)
    if n < 120:
        raise RuntimeError("Outer TRAIN trop petit pour calibration temporelle.")

    if n < 450:
        n_splits = 3
    elif n < 900:
        n_splits = 4
    else:
        n_splits = 5

    tscv = TimeSeriesSplit(n_splits=n_splits)

    y_all = train_df[TARGET].astype(int).to_numpy()
    X_all = train_df[features].apply(pd.to_numeric, errors="coerce")

    oof_pred = np.full(n, np.nan, dtype=float)
    fold_rows = []

    for i, (tr_idx, va_idx) in enumerate(tscv.split(X_all), start=1):
        ytr = y_all[tr_idx]
        yva = y_all[va_idx]

        if len(np.unique(ytr)) < 2:
            continue

        model = build_base_model(C=C)
        model.fit(X_all.iloc[tr_idx], ytr)
        p = model.predict_proba(X_all.iloc[va_idx])[:, 1]
        oof_pred[va_idx] = p

        fold_rows.append({
            "calibration_inner_fold": i,
            "train_rows": len(tr_idx),
            "val_rows": len(va_idx),
            "train_positive_rate": float(np.mean(ytr)),
            "val_positive_rate": float(np.mean(yva)),
            "roc_auc": safe_roc(yva, p),
            "pr_auc": safe_pr(yva, p),
            "brier": float(brier_score_loss(yva, p)),
            "log_loss": float(log_loss(yva, np.clip(p, EPS, 1-EPS))),
            "mean_probability": float(np.mean(p)),
        })

    mask = ~np.isnan(oof_pred)
    if int(mask.sum()) < 80:
        raise RuntimeError(
            f"Seulement {int(mask.sum())} prédictions OOF calibration."
        )

    return oof_pred[mask], y_all[mask], pd.DataFrame(fold_rows)


class PlattTemporal:
    def __init__(self):
        self.model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )
        self.slope_ = np.nan
        self.intercept_ = np.nan
        self.fallback_constant_ = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
        return np.log(p / (1-p)).reshape(-1, 1)

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)

        if len(np.unique(y)) < 2:
            self.fallback_constant_ = float(np.mean(y))
            return self

        self.model.fit(self._logit(p), y)
        self.slope_ = float(self.model.coef_[0, 0])
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        if self.fallback_constant_ is not None:
            return np.full(len(p), self.fallback_constant_, dtype=float)

        # Si le fit Platt produit une pente <= 0, on ne lui permet pas
        # d'inverser complètement le ranking du modèle de base.
        # Dans ce cas, fallback vers correction d'intercept uniquement.
        if not np.isfinite(self.slope_) or self.slope_ <= 0:
            raw_logit = self._logit(p).ravel()
            z = raw_logit + (self.intercept_ if np.isfinite(self.intercept_) else 0.0)
            return 1.0 / (1.0 + np.exp(-z))

        return self.model.predict_proba(self._logit(p))[:, 1]


def fit_calibrators(oof_p: np.ndarray, oof_y: np.ndarray):
    platt = PlattTemporal().fit(oof_p, oof_y)

    isotonic = IsotonicRegression(
        y_min=0.001,
        y_max=0.999,
        increasing=True,
        out_of_bounds="clip",
    )
    isotonic.fit(oof_p, oof_y)

    return platt, isotonic


def correct_constant_baseline_metrics(y_val: np.ndarray, train_rate: float) -> Tuple[float, float]:
    """
    Baseline correcte: prédire TRAIN_RATE sur chaque ligne de validation,
    puis mesurer contre y_val.
    """
    p0 = np.full(len(y_val), train_rate, dtype=float)
    brier0 = float(brier_score_loss(y_val, p0))
    ll0 = float(log_loss(y_val, np.clip(p0, EPS, 1-EPS)))
    return brier0, ll0


def evaluate_probabilities(
    y_val: np.ndarray,
    p: np.ndarray,
    train_rate: float,
) -> Dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)

    roc = safe_roc(y_val, p)
    pr = safe_pr(y_val, p)
    val_rate = float(np.mean(y_val))
    brier = float(brier_score_loss(y_val, p))
    ll = float(log_loss(y_val, p))
    ece = ece_binary(y_val, p, n_bins=10)

    brier0, ll0 = correct_constant_baseline_metrics(y_val, train_rate)

    cint, cslope = calibration_slope_intercept(y_val, p)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift_vs_val_rate": pr - val_rate,
        "brier": brier,
        "baseline_brier_train_rate_on_val": brier0,
        "brier_improvement_vs_correct_baseline": brier0 - brier,
        "log_loss": ll,
        "baseline_logloss_train_rate_on_val": ll0,
        "logloss_improvement_vs_correct_baseline": ll0 - ll,
        "ece_10": ece,
        "mean_probability": float(np.mean(p)),
        "calibration_intercept_report": cint,
        "calibration_slope_report": cslope,
    }


def run_fold(
    df: pd.DataFrame,
    fold: FoldDef,
    subset_name: str,
    features: List[str],
    C: float,
) -> Tuple[List[Dict], pd.DataFrame]:
    train_df = df.iloc[fold.train_idx].copy()
    val_df = df.iloc[fold.val_idx].copy()

    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    val_df = val_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    Xtr = train_df[features].apply(pd.to_numeric, errors="coerce")
    Xva = val_df[features].apply(pd.to_numeric, errors="coerce")

    ytr = train_df[TARGET].astype(int).to_numpy()
    yva = val_df[TARGET].astype(int).to_numpy()

    base = build_base_model(C=C)
    base.fit(Xtr, ytr)
    raw_p = base.predict_proba(Xva)[:, 1]

    oof_p, oof_y, oof_diag = predict_temporal_oof(
        train_df=train_df,
        features=features,
        C=C,
    )
    platt, isotonic = fit_calibrators(oof_p, oof_y)

    preds = {
        "raw": raw_p,
        "platt_temporal_oof": platt.predict(raw_p),
        "isotonic_temporal_oof": isotonic.predict(raw_p),
    }

    rows = []
    train_rate = float(np.mean(ytr))
    val_rate = float(np.mean(yva))

    for calibration, p in preds.items():
        metrics = evaluate_probabilities(
            y_val=yva,
            p=p,
            train_rate=train_rate,
        )

        rows.append({
            "subset": subset_name,
            "n_features": len(features),
            "base_model": "logistic_l2",
            "C": C,
            "calibration": calibration,
            "fold": fold.name,
            "kind": fold.kind,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_positive_rate": train_rate,
            "val_positive_rate": val_rate,
            "oof_calibration_rows": len(oof_y),
            "oof_calibration_rate": float(np.mean(oof_y)),
            "platt_slope": float(platt.slope_) if np.isfinite(platt.slope_) else np.nan,
            "platt_intercept": float(platt.intercept_) if np.isfinite(platt.intercept_) else np.nan,
            **metrics,
        })

    oof_diag = oof_diag.copy()
    oof_diag.insert(0, "outer_fold", fold.name)
    oof_diag.insert(0, "subset", subset_name)

    return rows, oof_diag


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (subset, n_features, calibration), block in metrics.groupby(
        ["subset", "n_features", "calibration"],
        sort=False,
    ):
        internal = block[block["kind"] == "internal"].copy()
        stress = block[block["kind"] == "external_stress"].copy()

        if len(stress) != 1:
            raise RuntimeError(f"Stress fold absent pour {subset}/{calibration}")

        s = stress.iloc[0]

        mean_internal_roc = float(internal["roc_auc"].mean())
        mean_internal_pr = float(internal["pr_auc"].mean())
        mean_internal_lift = float(internal["pr_lift_vs_val_rate"].mean())

        pr_pos_frac = float((internal["pr_lift_vs_val_rate"] > 0).mean())
        roc_pos_frac = float((internal["roc_auc"] > 0.5).mean())

        brier_mean_imp = float(
            internal["brier_improvement_vs_correct_baseline"].mean()
        )
        brier_pos_frac = float(
            (internal["brier_improvement_vs_correct_baseline"] > 0).mean()
        )

        ll_mean_imp = float(
            internal["logloss_improvement_vs_correct_baseline"].mean()
        )
        ll_pos_frac = float(
            (internal["logloss_improvement_vs_correct_baseline"] > 0).mean()
        )

        mean_ece = float(internal["ece_10"].mean())

        stress_roc = float(s["roc_auc"])
        stress_pr = float(s["pr_auc"])
        stress_lift = float(s["pr_lift_vs_val_rate"])
        stress_brier_imp = float(
            s["brier_improvement_vs_correct_baseline"]
        )
        stress_ll_imp = float(
            s["logloss_improvement_vs_correct_baseline"]
        )

        # Gate de calibration V2. On veut un pipeline réellement meilleur
        # qu'une probabilité constante aussi sur le stress fold consommé.
        passes_gate = int(
            mean_internal_roc > 0.50
            and mean_internal_lift > 0
            and pr_pos_frac >= 0.80
            and roc_pos_frac >= 0.80
            and brier_mean_imp > 0
            and brier_pos_frac >= 0.60
            and ll_mean_imp > 0
            and ll_pos_frac >= 0.60
            and stress_roc > 0.50
            and stress_lift > 0
            and stress_brier_imp > 0
            and stress_ll_imp > 0
        )

        # Score équilibré. Il ne remplace pas le gate.
        calibration_score = (
            0.18 * mean_internal_roc
            + 0.18 * mean_internal_pr
            + 0.14 * stress_roc
            + 0.14 * stress_pr
            + 0.09 * pr_pos_frac
            + 0.07 * roc_pos_frac
            + 0.06 * max(-0.05, min(0.05, brier_mean_imp)) / 0.05
            + 0.06 * max(-0.05, min(0.05, stress_brier_imp)) / 0.05
            + 0.04 * max(-0.10, min(0.10, ll_mean_imp)) / 0.10
            + 0.04 * max(-0.10, min(0.10, stress_ll_imp)) / 0.10
        )

        rows.append({
            "subset": subset,
            "n_features": int(n_features),
            "base_model": "logistic_l2",
            "C": FIXED_C,
            "calibration": calibration,
            "passes_v2_calibration_gate": passes_gate,
            "mean_internal_roc_auc": mean_internal_roc,
            "mean_internal_pr_auc": mean_internal_pr,
            "mean_internal_pr_lift": mean_internal_lift,
            "internal_pr_above_base_fraction": pr_pos_frac,
            "internal_roc_above_050_fraction": roc_pos_frac,
            "mean_internal_brier_improvement": brier_mean_imp,
            "internal_brier_better_fraction": brier_pos_frac,
            "mean_internal_logloss_improvement": ll_mean_imp,
            "internal_logloss_better_fraction": ll_pos_frac,
            "mean_internal_ece_10": mean_ece,
            "stress_roc_auc": stress_roc,
            "stress_pr_auc": stress_pr,
            "stress_pr_lift": stress_lift,
            "stress_brier": float(s["brier"]),
            "stress_baseline_brier": float(s["baseline_brier_train_rate_on_val"]),
            "stress_brier_improvement": stress_brier_imp,
            "stress_log_loss": float(s["log_loss"]),
            "stress_baseline_logloss": float(s["baseline_logloss_train_rate_on_val"]),
            "stress_logloss_improvement": stress_ll_imp,
            "stress_ece_10": float(s["ece_10"]),
            "stress_mean_probability": float(s["mean_probability"]),
            "stress_val_positive_rate": float(s["val_positive_rate"]),
            "calibration_score": calibration_score,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "passes_v2_calibration_gate",
            "stress_brier_improvement",
            "stress_logloss_improvement",
            "stress_pr_lift",
            "stress_roc_auc",
            "mean_internal_brier_improvement",
            "mean_internal_pr_lift",
            "n_features",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(
    path: Path,
    df: pd.DataFrame,
    subsets: Dict[str, List[str]],
    ranking: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# First Half 37 — V2 Temporal Calibration Audit")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- LogisticRegression L2 non équilibrée.")
    lines.append(f"- C figé à **{FIXED_C}** avant cet audit.")
    lines.append(
        "- Calibrateurs appris uniquement sur des prédictions OOF temporelles "
        "à l'intérieur du TRAIN de chaque fold."
    )
    lines.append(
        "- Les baselines Brier/LogLoss prédisent le taux positif du TRAIN sur "
        "les vraies lignes de validation."
    )
    lines.append("- Aucun match postérieur au 19 septembre 2026 n'est utilisé.")
    lines.append("")
    lines.append("## DEVELOPMENT V2")
    lines.append("")
    lines.append(
        f"- Matchs : **{len(df)}** | positifs : **{int(df[TARGET].sum())}** "
        f"({df[TARGET].mean():.2%})"
    )
    lines.append(
        f"- Dates : `{df[DATE_COL].min()}` -> `{df[DATE_COL].max()}`"
    )
    lines.append("")
    lines.append("## Finalistes")
    lines.append("")
    for name, feats in subsets.items():
        lines.append(
            f"- `{name}` : **{len(feats)}** — "
            + ", ".join(f"`{f}`" for f in feats)
        )
    lines.append("")
    lines.append("## Ranking calibration")
    lines.append("")
    show_cols = [
        "rank",
        "subset",
        "calibration",
        "passes_v2_calibration_gate",
        "mean_internal_roc_auc",
        "mean_internal_pr_auc",
        "mean_internal_pr_lift",
        "mean_internal_brier_improvement",
        "mean_internal_logloss_improvement",
        "mean_internal_ece_10",
        "stress_roc_auc",
        "stress_pr_auc",
        "stress_pr_lift",
        "stress_brier",
        "stress_baseline_brier",
        "stress_brier_improvement",
        "stress_log_loss",
        "stress_baseline_logloss",
        "stress_logloss_improvement",
        "stress_ece_10",
    ]
    lines.append("```")
    lines.append(ranking[show_cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Rang #1")
    lines.append("")
    if not ranking.empty:
        r = ranking.iloc[0]
        lines.append(
            f"- `{r['subset']} + {r['calibration']}` | "
            f"gate={'PASS' if int(r['passes_v2_calibration_gate']) else 'FAIL'}"
        )
        lines.append(
            f"- Internal ROC={r['mean_internal_roc_auc']:.4f} | "
            f"PR={r['mean_internal_pr_auc']:.4f} | "
            f"PR lift={r['mean_internal_pr_lift']:+.4f}"
        )
        lines.append(
            f"- Stress ROC={r['stress_roc_auc']:.4f} | "
            f"PR={r['stress_pr_auc']:.4f} | "
            f"PR lift={r['stress_pr_lift']:+.4f}"
        )
        lines.append(
            f"- Stress Brier={r['stress_brier']:.4f} vs "
            f"baseline={r['stress_baseline_brier']:.4f} "
            f"(imp={r['stress_brier_improvement']:+.4f})"
        )
        lines.append(
            f"- Stress LogLoss={r['stress_log_loss']:.4f} vs "
            f"baseline={r['stress_baseline_logloss']:.4f} "
            f"(imp={r['stress_logloss_improvement']:+.4f})"
        )
    lines.append("")
    lines.append("## Suite")
    lines.append("")
    lines.append(
        "Si un pipeline passe le gate, il devient candidat au gel V2. "
        "On pourra ensuite entraîner le pipeline final sur les 2 105 matchs connus, "
        "sans encore utiliser le futur holdout."
    )
    lines.append(
        "Le futur holdout V2 devra rester strictement postérieur au "
        "19 septembre 2026."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit de calibration temporelle des finalistes First Half 37 V2."
    )
    parser.add_argument("--development-csv", default=str(DEFAULT_DEV))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--internal-folds", type=int, default=5)
    parser.add_argument("--fixed-c", type=float, default=FIXED_C)
    args = parser.parse_args()

    dev_path = Path(args.development_csv)
    ranking_path = Path(args.ranking_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 116)
    print("FIRST HALF 37 — V2 TEMPORAL CALIBRATION AUDIT")
    print("=" * 116)
    print(f"DEVELOPMENT : {dev_path}")
    print(f"Ranking     : {ranking_path}")
    print(f"Output      : {out_dir}")
    print(f"Base model  : logistic_l2 | C={args.fixed_c}")
    print("Calibration : raw | platt_temporal_oof | isotonic_temporal_oof")
    print("IMPORTANT   : baselines Brier/LogLoss corrigées.")
    print("IMPORTANT   : aucun match postérieur au 19 septembre 2026 n'est lu.")
    print()

    df = load_dev(dev_path)
    ranked_features = load_ranked_features(ranking_path)
    subsets = build_subsets(ranked_features)
    folds = make_outer_folds(df, n_internal=args.internal_folds)

    print(
        f"Dataset={len(df)} | positifs={int(df[TARGET].sum())} "
        f"({df[TARGET].mean():.2%})"
    )
    print(
        f"Dates={df[DATE_COL].min()} -> {df[DATE_COL].max()}"
    )
    print()

    (out_dir / "first_half_37_v2_calibration_subset_definitions.json").write_text(
        json.dumps(subsets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_rows = []
    oof_frames = []

    for subset_name, features in subsets.items():
        print("-" * 116)
        print(f"{subset_name} | {len(features)} features")

        subset_rows = []

        for fold in folds:
            rows, oof_diag = run_fold(
                df=df,
                fold=fold,
                subset_name=subset_name,
                features=features,
                C=args.fixed_c,
            )
            all_rows.extend(rows)
            subset_rows.extend(rows)
            oof_frames.append(oof_diag)

        sb = pd.DataFrame(subset_rows)

        for calibration in ["raw", "platt_temporal_oof", "isotonic_temporal_oof"]:
            b = sb[sb["calibration"] == calibration]
            internal = b[b["kind"] == "internal"]
            stress = b[b["kind"] == "external_stress"].iloc[0]

            print(
                f"  {calibration:<22} "
                f"INT ROC={internal['roc_auc'].mean():.4f} "
                f"PR={internal['pr_auc'].mean():.4f} "
                f"BrierImp={internal['brier_improvement_vs_correct_baseline'].mean():+.4f} "
                f"LLImp={internal['logloss_improvement_vs_correct_baseline'].mean():+.4f} | "
                f"STRESS ROC={stress['roc_auc']:.4f} "
                f"PR={stress['pr_auc']:.4f} "
                f"BrierImp={stress['brier_improvement_vs_correct_baseline']:+.4f} "
                f"LLImp={stress['logloss_improvement_vs_correct_baseline']:+.4f}"
            )

    metrics = pd.DataFrame(all_rows)
    ranking = summarize(metrics)

    metrics_path = out_dir / "first_half_37_v2_calibration_fold_metrics.csv"
    ranking_out = out_dir / "first_half_37_v2_calibration_ranking.csv"
    oof_path = out_dir / "first_half_37_v2_calibration_inner_oof_diagnostics.csv"
    report_path = out_dir / "first_half_37_v2_calibration_report.md"
    manifest_path = out_dir / "first_half_37_v2_calibration_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    ranking.to_csv(ranking_out, index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(oof_path, index=False)

    write_report(
        report_path,
        df=df,
        subsets=subsets,
        ranking=ranking,
        metrics=metrics,
    )

    manifest = {
        "schema_version": "first_half_37_v2_temporal_calibration_v1",
        "development_rows": int(len(df)),
        "development_positive_rate": float(df[TARGET].mean()),
        "date_min": str(df[DATE_COL].min()),
        "date_max": str(df[DATE_COL].max()),
        "base_model": "logistic_l2",
        "fixed_C": float(args.fixed_c),
        "subsets": subsets,
        "calibrations": [
            "raw",
            "platt_temporal_oof",
            "isotonic_temporal_oof",
        ],
        "external_stress_consumed": True,
        "future_holdout_used": False,
        "baseline_rule": (
            "constant prediction = outer TRAIN positive rate, "
            "evaluated against actual outer validation targets"
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "ranking": str(ranking_out),
            "oof_diagnostics": str(oof_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    winner = ranking.iloc[0]
    pass_count = int(ranking["passes_v2_calibration_gate"].sum())

    print()
    print("=" * 116)
    print("AUDIT CALIBRATION V2 TERMINE")
    print("=" * 116)
    print(f"Configurations calibration testées : {len(ranking)}")
    print(f"Pipelines passant le gate : {pass_count}")
    print(
        f"#1 : {winner['subset']} + logistic_l2 C={args.fixed_c} + "
        f"{winner['calibration']} | "
        f"INT ROC={winner['mean_internal_roc_auc']:.4f} | "
        f"INT PR={winner['mean_internal_pr_auc']:.4f} | "
        f"STRESS ROC={winner['stress_roc_auc']:.4f} | "
        f"STRESS PR={winner['stress_pr_auc']:.4f} | "
        f"STRESS BrierImp={winner['stress_brier_improvement']:+.4f} | "
        f"STRESS LLImp={winner['stress_logloss_improvement']:+.4f} | "
        f"gate={'PASS' if int(winner['passes_v2_calibration_gate']) else 'FAIL'}"
    )
    print(f"Rapport  : {report_path}")
    print(f"Ranking  : {ranking_out}")
    print(f"Metrics  : {metrics_path}")
    print(f"OOF diag : {oof_path}")
    print(f"Manifest : {manifest_path}")
    print("IMPORTANT: aucun futur holdout n'a été utilisé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
