#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 TEMPORAL REGIME / CALIBRATION AUDIT

Objectif:
Comprendre pourquoi TOP_8 + logistic_l2 C=0.01 classe correctement les matchs
dans le temps, mais donne des probabilités brutes mieux calibrées à certaines
périodes qu'à d'autres.

Aucun tuning, aucun refit final, aucun futur holdout.

Diagnostics:
  1) mêmes 5 folds temporels internes + EXTERNAL_STRESS;
  2) taux TRAIN vs taux réel VAL;
  3) probabilité moyenne RAW vs taux réel;
  4) Brier/LogLoss RAW vs baseline constante = taux TRAIN;
  5) ECE, calibration intercept et slope;
  6) correction ORACLE d'intercept uniquement (diagnostic, jamais production):
     mesure combien de l'erreur vient simplement du changement de taux de base;
  7) drift des 8 features TOP_8 par fold (SMD + missingness);
  8) détail jour par jour dans les 303 matchs récents.

Sorties:
  outputs/first_half_37/v2_regime_audit/
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
    / "v2_regime_audit"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
STAGE_COL = "_v2_stage"
C_FIXED = 0.01
EPS = 1e-8

EXPECTED_TOP8 = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]


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
    if len(y) == 0:
        return float("nan")

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not mask.any():
            continue

        ece += (
            mask.mean()
            * abs(float(y[mask].mean()) - float(p[mask].mean()))
        )

    return float(ece)


def load_dev(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DEVELOPMENT V2 introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {TARGET, DATE_COL, ID_COL, STAGE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes: {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    if df[DATE_COL].isna().any():
        raise ValueError("kickoff_utc non parsable.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target non résolu présent.")

    if df[ID_COL].duplicated().any():
        raise ValueError("match_id dupliqué.")

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def load_top8(ranking_path: Path) -> List[str]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking introuvable: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    required = {"feature", "selected_v2", "robust_selection_score"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Ranking incomplet: {required - set(ranking.columns)}")

    selected = ranking[ranking["selected_v2"] == 1].copy()
    selected = selected.sort_values("robust_selection_score", ascending=False)
    top8 = selected["feature"].astype(str).tolist()[:8]

    if top8 != EXPECTED_TOP8:
        raise RuntimeError(
            "Le TOP_8 actuel ne correspond pas au TOP_8 figé par l'étape précédente.\n"
            f"Attendu: {EXPECTED_TOP8}\n"
            f"Trouvé : {top8}"
        )

    return top8


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=C_FIXED,
                solver="liblinear",
                class_weight=None,
                max_iter=5000,
                random_state=87,
            ),
        ),
    ])


def make_folds(df: pd.DataFrame, n_internal: int = 5) -> List[FoldDef]:
    orig_mask = df[STAGE_COL].astype(str).eq("original_consumed")
    ext_mask = df[STAGE_COL].astype(str).eq("external_consumed")

    orig_idx = np.where(orig_mask.to_numpy())[0]
    ext_idx = np.where(ext_mask.to_numpy())[0]

    if len(orig_idx) == 0 or len(ext_idx) == 0:
        raise RuntimeError("Impossible d'identifier original_consumed/external_consumed.")

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


def correct_baseline(y: np.ndarray, train_rate: float) -> Tuple[float, float]:
    p0 = np.full(len(y), train_rate, dtype=float)
    return (
        float(brier_score_loss(y, p0)),
        float(log_loss(y, np.clip(p0, EPS, 1-EPS))),
    )


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    z = np.log(p / (1-p)).reshape(-1, 1)

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


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


def oracle_intercept_shift(p: np.ndarray, target_rate: float) -> float:
    """
    Trouve alpha tel que mean(sigmoid(logit(p)+alpha)) ~= target_rate.
    Utilise le taux réel de validation: DIAGNOSTIC UNIQUEMENT.
    """
    z = logit(p)

    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        m = float(sigmoid(z + mid).mean())
        if m < target_rate:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)


def feature_shift_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    fold_name: str,
) -> pd.DataFrame:
    rows = []

    for f in features:
        tr = pd.to_numeric(train_df[f], errors="coerce")
        va = pd.to_numeric(val_df[f], errors="coerce")

        tr_mean = float(tr.mean()) if tr.notna().any() else np.nan
        va_mean = float(va.mean()) if va.notna().any() else np.nan
        tr_std = float(tr.std(ddof=0)) if tr.notna().sum() > 1 else np.nan

        if pd.notna(tr_std) and tr_std > 1e-12:
            smd = (va_mean - tr_mean) / tr_std
        else:
            smd = np.nan

        rows.append({
            "fold": fold_name,
            "feature": f,
            "train_mean": tr_mean,
            "val_mean": va_mean,
            "standardized_mean_shift": smd,
            "abs_standardized_mean_shift": abs(smd) if pd.notna(smd) else np.nan,
            "train_missing_rate": float(tr.isna().mean()),
            "val_missing_rate": float(va.isna().mean()),
            "missing_delta": float(va.isna().mean() - tr.isna().mean()),
        })

    return pd.DataFrame(rows)


def evaluate_fold(
    df: pd.DataFrame,
    fold: FoldDef,
    features: List[str],
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[fold.train_idx].copy()
    val_df = df.iloc[fold.val_idx].copy()

    Xtr = train_df[features].apply(pd.to_numeric, errors="coerce")
    Xva = val_df[features].apply(pd.to_numeric, errors="coerce")
    ytr = train_df[TARGET].astype(int).to_numpy()
    yva = val_df[TARGET].astype(int).to_numpy()

    model = build_model()
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]

    train_rate = float(np.mean(ytr))
    val_rate = float(np.mean(yva))
    mean_p = float(np.mean(p))
    gap = mean_p - val_rate

    brier = float(brier_score_loss(yva, p))
    ll = float(log_loss(yva, np.clip(p, EPS, 1-EPS)))
    brier0, ll0 = correct_baseline(yva, train_rate)

    cint, cslope = calibration_intercept_slope(yva, p)

    # Oracle intercept only: quantifie la part du problème qui vient
    # simplement d'un décalage de niveau général.
    alpha = oracle_intercept_shift(p, val_rate)
    p_oracle = sigmoid(logit(p) + alpha)
    oracle_brier = float(brier_score_loss(yva, p_oracle))
    oracle_ll = float(log_loss(yva, np.clip(p_oracle, EPS, 1-EPS)))

    if gap <= -0.03:
        direction = "UNDERPREDICT"
    elif gap >= 0.03:
        direction = "OVERPREDICT"
    else:
        direction = "LEVEL_CLOSE"

    if safe_roc(yva, p) <= 0.50:
        regime = "RANKING_WEAK"
    elif (brier0 - brier) > 0 and (ll0 - ll) > 0:
        regime = "RAW_CALIBRATION_GOOD"
    elif (oracle_brier < brier - 0.002) or (oracle_ll < ll - 0.004):
        regime = "MAINLY_INTERCEPT_BASE_RATE_SHIFT"
    else:
        regime = "MIXED_CALIBRATION_ISSUE"

    row = {
        "fold": fold.name,
        "kind": fold.kind,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_date_min": train_df[DATE_COL].min(),
        "train_date_max": train_df[DATE_COL].max(),
        "val_date_min": val_df[DATE_COL].min(),
        "val_date_max": val_df[DATE_COL].max(),
        "train_positive_rate": train_rate,
        "val_positive_rate": val_rate,
        "base_rate_shift_val_minus_train": val_rate - train_rate,
        "mean_probability": mean_p,
        "calibration_gap_meanp_minus_actual": gap,
        "calibration_direction": direction,
        "roc_auc": safe_roc(yva, p),
        "pr_auc": safe_pr(yva, p),
        "pr_lift_vs_val_rate": safe_pr(yva, p) - val_rate,
        "brier": brier,
        "baseline_brier": brier0,
        "brier_improvement": brier0 - brier,
        "log_loss": ll,
        "baseline_logloss": ll0,
        "logloss_improvement": ll0 - ll,
        "ece_10": ece_binary(yva, p, n_bins=10),
        "calibration_intercept": cint,
        "calibration_slope": cslope,
        "oracle_intercept_shift": alpha,
        "oracle_mean_probability": float(p_oracle.mean()),
        "oracle_brier": oracle_brier,
        "oracle_brier_gain_vs_raw": brier - oracle_brier,
        "oracle_logloss": oracle_ll,
        "oracle_logloss_gain_vs_raw": ll - oracle_ll,
        "diagnostic_regime": regime,
    }

    # Bands 5 points pour lire la calibration à l'intérieur de chaque période.
    band_edges = [0, .20, .25, .30, .35, .40, .50, 1.0]
    band_labels = [
        "<20%", "20-25%", "25-30%", "30-35%",
        "35-40%", "40-50%", "50%+"
    ]

    band = pd.cut(
        p,
        bins=band_edges,
        labels=band_labels,
        include_lowest=True,
        right=False,
    )

    bands = pd.DataFrame({
        "fold": fold.name,
        "probability_band": band.astype(str),
        "y": yva,
        "p": p,
    })

    band_rows = []
    for label in band_labels:
        b = bands[bands["probability_band"] == label]
        if b.empty:
            continue
        band_rows.append({
            "fold": fold.name,
            "probability_band": label,
            "n": len(b),
            "positives": int(b["y"].sum()),
            "observed_rate": float(b["y"].mean()),
            "mean_probability": float(b["p"].mean()),
            "calibration_gap_observed_minus_predicted": float(
                b["y"].mean() - b["p"].mean()
            ),
        })

    shifts = feature_shift_table(
        train_df=train_df,
        val_df=val_df,
        features=features,
        fold_name=fold.name,
    )

    return row, pd.DataFrame(band_rows), shifts


def external_daily_audit(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    orig = df[df[STAGE_COL].astype(str).eq("original_consumed")].copy()
    ext = df[df[STAGE_COL].astype(str).eq("external_consumed")].copy()

    Xtr = orig[features].apply(pd.to_numeric, errors="coerce")
    ytr = orig[TARGET].astype(int).to_numpy()

    model = build_model()
    model.fit(Xtr, ytr)

    ext = ext.sort_values([DATE_COL, ID_COL]).copy()
    ext["_date"] = ext[DATE_COL].dt.strftime("%Y-%m-%d")

    train_rate = float(np.mean(ytr))
    rows = []

    for date, g in ext.groupby("_date", sort=True):
        X = g[features].apply(pd.to_numeric, errors="coerce")
        y = g[TARGET].astype(int).to_numpy()
        p = model.predict_proba(X)[:, 1]

        brier = float(brier_score_loss(y, p))
        ll = float(log_loss(y, np.clip(p, EPS, 1-EPS)))
        brier0, ll0 = correct_baseline(y, train_rate)

        cint, cslope = calibration_intercept_slope(y, p)

        val_rate = float(np.mean(y))
        alpha = oracle_intercept_shift(p, val_rate)
        po = sigmoid(logit(p) + alpha)

        rows.append({
            "date": date,
            "n": len(g),
            "positives": int(y.sum()),
            "positive_rate": val_rate,
            "mean_probability": float(np.mean(p)),
            "calibration_gap_meanp_minus_actual": float(np.mean(p) - val_rate),
            "roc_auc": safe_roc(y, p),
            "pr_auc": safe_pr(y, p),
            "brier": brier,
            "baseline_brier": brier0,
            "brier_improvement": brier0 - brier,
            "log_loss": ll,
            "baseline_logloss": ll0,
            "logloss_improvement": ll0 - ll,
            "ece_10": ece_binary(y, p, n_bins=5),
            "calibration_intercept": cint,
            "calibration_slope": cslope,
            "oracle_intercept_shift": alpha,
            "oracle_brier_gain_vs_raw": brier - float(brier_score_loss(y, po)),
            "oracle_logloss_gain_vs_raw": ll - float(
                log_loss(y, np.clip(po, EPS, 1-EPS))
            ),
        })

    return pd.DataFrame(rows)


def build_summary(folds_df: pd.DataFrame) -> Dict:
    internal = folds_df[folds_df["kind"] == "internal"].copy()
    stress = folds_df[folds_df["kind"] == "external_stress"].iloc[0]

    good_raw = folds_df[
        (folds_df["brier_improvement"] > 0)
        & (folds_df["logloss_improvement"] > 0)
    ]["fold"].tolist()

    bad_raw = folds_df[
        (folds_df["brier_improvement"] <= 0)
        | (folds_df["logloss_improvement"] <= 0)
    ]["fold"].tolist()

    intercept_like = folds_df[
        folds_df["diagnostic_regime"].eq("MAINLY_INTERCEPT_BASE_RATE_SHIFT")
    ]["fold"].tolist()

    return {
        "internal_mean_roc": float(internal["roc_auc"].mean()),
        "internal_mean_pr": float(internal["pr_auc"].mean()),
        "raw_calibration_good_folds": good_raw,
        "raw_calibration_bad_folds": bad_raw,
        "mainly_intercept_shift_folds": intercept_like,
        "stress_roc": float(stress["roc_auc"]),
        "stress_brier_improvement": float(stress["brier_improvement"]),
        "stress_logloss_improvement": float(stress["logloss_improvement"]),
        "stress_calibration_gap": float(
            stress["calibration_gap_meanp_minus_actual"]
        ),
        "stress_calibration_slope": float(stress["calibration_slope"]),
        "stress_oracle_brier_gain_vs_raw": float(
            stress["oracle_brier_gain_vs_raw"]
        ),
        "stress_oracle_logloss_gain_vs_raw": float(
            stress["oracle_logloss_gain_vs_raw"]
        ),
    }


def write_report(
    path: Path,
    features: List[str],
    folds_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    summary: Dict,
) -> None:
    lines = []
    lines.append("# First Half 37 — V2 Temporal Regime / Calibration Audit")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- Pipeline diagnostiqué : `TOP_8 + logistic_l2 + C=0.01 + raw`.")
    lines.append("- Aucun tuning, aucune calibration appliquée, aucun futur holdout.")
    lines.append(
        "- La correction ORACLE d'intercept utilise le target de validation "
        "uniquement pour comprendre l'origine de l'erreur; elle est interdite en production."
    )
    lines.append("")
    lines.append("## TOP_8")
    lines.append("")
    for i, f in enumerate(features, start=1):
        lines.append(f"{i}. `{f}`")
    lines.append("")
    lines.append("## Fold par fold")
    lines.append("")
    show = [
        "fold",
        "val_date_min",
        "val_date_max",
        "train_positive_rate",
        "val_positive_rate",
        "mean_probability",
        "calibration_gap_meanp_minus_actual",
        "roc_auc",
        "pr_auc",
        "brier_improvement",
        "logloss_improvement",
        "ece_10",
        "calibration_intercept",
        "calibration_slope",
        "oracle_brier_gain_vs_raw",
        "oracle_logloss_gain_vs_raw",
        "diagnostic_regime",
    ]
    lines.append("```")
    lines.append(folds_df[show].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Stress externe jour par jour")
    lines.append("")
    lines.append("```")
    lines.append(daily_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Drift TOP_8 dans les folds")
    lines.append("")
    top_shift = (
        shifts_df.sort_values(
            ["fold", "abs_standardized_mean_shift"],
            ascending=[True, False],
        )
        .groupby("fold", as_index=False)
        .head(3)
    )
    lines.append(
        "Ci-dessous, les 3 plus gros déplacements moyens standardisés par fold."
    )
    lines.append("")
    lines.append("```")
    lines.append(
        top_shift[
            [
                "fold",
                "feature",
                "standardized_mean_shift",
                "missing_delta",
            ]
        ].to_string(index=False)
    )
    lines.append("```")
    lines.append("")
    lines.append("## Résumé")
    lines.append("")
    lines.append(
        f"- Folds RAW bien calibrés vs baseline : "
        f"`{', '.join(summary['raw_calibration_good_folds']) or 'aucun'}`"
    )
    lines.append(
        f"- Folds RAW moins bons que baseline : "
        f"`{', '.join(summary['raw_calibration_bad_folds']) or 'aucun'}`"
    )
    lines.append(
        f"- Folds ressemblant surtout à un shift d'intercept/base-rate : "
        f"`{', '.join(summary['mainly_intercept_shift_folds']) or 'aucun'}`"
    )
    lines.append(
        f"- Stress ROC : **{summary['stress_roc']:.4f}**"
    )
    lines.append(
        f"- Stress Brier improvement : **{summary['stress_brier_improvement']:+.4f}**"
    )
    lines.append(
        f"- Stress LogLoss improvement : **{summary['stress_logloss_improvement']:+.4f}**"
    )
    lines.append(
        f"- Stress gap moyen p-réel : **{summary['stress_calibration_gap']:+.4f}**"
    )
    lines.append(
        f"- Stress slope calibration : **{summary['stress_calibration_slope']:.4f}**"
    )
    lines.append("")
    lines.append("## Comment lire l'ORACLE intercept")
    lines.append("")
    lines.append(
        "Si l'ORACLE intercept améliore fortement Brier/LogLoss sans changer le ROC, "
        "cela signifie qu'une grande partie du problème vient du niveau général des "
        "probabilités (taux de base), pas du classement des matchs."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit temporel de régime/calibration pour First Half 37 V2 TOP_8 raw."
    )
    parser.add_argument("--development-csv", default=str(DEFAULT_DEV))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--internal-folds", type=int, default=5)
    args = parser.parse_args()

    dev_path = Path(args.development_csv)
    ranking_path = Path(args.ranking_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("FIRST HALF 37 — V2 TEMPORAL REGIME / CALIBRATION AUDIT")
    print("=" * 118)
    print(f"DEVELOPMENT : {dev_path}")
    print(f"Ranking     : {ranking_path}")
    print(f"Output      : {out_dir}")
    print("Pipeline    : TOP_8 + logistic_l2 + C=0.01 + RAW")
    print("IMPORTANT   : diagnostic uniquement; aucun futur holdout.")
    print()

    df = load_dev(dev_path)
    features = load_top8(ranking_path)
    folds = make_folds(df, n_internal=args.internal_folds)

    fold_rows = []
    band_frames = []
    shift_frames = []

    for fold in folds:
        row, bands, shifts = evaluate_fold(
            df=df,
            fold=fold,
            features=features,
        )
        fold_rows.append(row)
        band_frames.append(bands)
        shift_frames.append(shifts)

        print(
            f"{fold.name:<16} | "
            f"VAL={row['val_rows']:3d} | "
            f"rate={row['val_positive_rate']:.3f} | "
            f"mean_p={row['mean_probability']:.3f} | "
            f"gap={row['calibration_gap_meanp_minus_actual']:+.3f} | "
            f"ROC={row['roc_auc']:.3f} | "
            f"BrierImp={row['brier_improvement']:+.4f} | "
            f"LLImp={row['logloss_improvement']:+.4f} | "
            f"slope={row['calibration_slope']:.3f} | "
            f"{row['diagnostic_regime']}"
        )

    folds_df = pd.DataFrame(fold_rows)
    bands_df = pd.concat(band_frames, ignore_index=True)
    shifts_df = pd.concat(shift_frames, ignore_index=True)
    daily_df = external_daily_audit(df=df, features=features)

    summary = build_summary(folds_df)

    folds_path = out_dir / "first_half_37_v2_regime_fold_metrics.csv"
    bands_path = out_dir / "first_half_37_v2_regime_probability_bands.csv"
    shifts_path = out_dir / "first_half_37_v2_regime_feature_shift.csv"
    daily_path = out_dir / "first_half_37_v2_regime_external_daily.csv"
    report_path = out_dir / "first_half_37_v2_regime_report.md"
    manifest_path = out_dir / "first_half_37_v2_regime_manifest.json"

    folds_df.to_csv(folds_path, index=False)
    bands_df.to_csv(bands_path, index=False)
    shifts_df.to_csv(shifts_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    write_report(
        report_path,
        features=features,
        folds_df=folds_df,
        daily_df=daily_df,
        shifts_df=shifts_df,
        summary=summary,
    )

    manifest = {
        "schema_version": "first_half_37_v2_regime_audit_v1",
        "pipeline": {
            "subset": "TOP_8",
            "features": features,
            "model": "logistic_l2",
            "C": C_FIXED,
            "calibration": "raw",
        },
        "development_rows": int(len(df)),
        "date_min": str(df[DATE_COL].min()),
        "date_max": str(df[DATE_COL].max()),
        "future_holdout_used": False,
        "oracle_intercept_is_diagnostic_only": True,
        "summary": summary,
        "outputs": {
            "fold_metrics": str(folds_path),
            "probability_bands": str(bands_path),
            "feature_shift": str(shifts_path),
            "external_daily": str(daily_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("AUDIT REGIME / CALIBRATION V2 TERMINE")
    print("=" * 118)
    print(
        "RAW calibration good folds : "
        + (", ".join(summary["raw_calibration_good_folds"]) or "aucun")
    )
    print(
        "RAW calibration bad folds  : "
        + (", ".join(summary["raw_calibration_bad_folds"]) or "aucun")
    )
    print(
        "Mostly intercept/base-rate : "
        + (", ".join(summary["mainly_intercept_shift_folds"]) or "aucun")
    )
    print(
        f"STRESS : ROC={summary['stress_roc']:.4f} | "
        f"BrierImp={summary['stress_brier_improvement']:+.4f} | "
        f"LLImp={summary['stress_logloss_improvement']:+.4f} | "
        f"gap={summary['stress_calibration_gap']:+.4f} | "
        f"slope={summary['stress_calibration_slope']:.4f}"
    )
    print(f"Rapport : {report_path}")
    print(f"Folds   : {folds_path}")
    print(f"Daily   : {daily_path}")
    print(f"Shift   : {shifts_path}")
    print("IMPORTANT: aucune décision de production automatique n'est prise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
