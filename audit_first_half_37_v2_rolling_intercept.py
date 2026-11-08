#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 ROLLING INTERCEPT CALIBRATION AUDIT

But:
Tester une correction d'intercept dynamique, sans toucher au ranking du modèle,
pour corriger les variations de taux de base observées dans le temps.

Pipeline de base figé:
  TOP_8 + LogisticRegression L2 + C=0.01 + RAW

Principe anti-fuite:
  - Dans chaque outer TRAIN, on génère des prédictions OOF temporelles.
  - On estime ensuite un shift d'intercept uniquement sur les DERNIERES
    prédictions OOF disponibles dans le TRAIN.
  - On entraîne enfin le modèle de base sur tout l'outer TRAIN.
  - On applique le shift d'intercept à l'outer VAL.
  - Jamais de target de l'outer VAL pour calibrer.
  - Aucun match postérieur au 19 septembre 2026 n'est lu.

Stratégies:
  RAW
  OOF_INTERCEPT_75
  OOF_INTERCEPT_150
  OOF_INTERCEPT_300
  OOF_INTERCEPT_EXPANDING

Sorties:
  outputs/first_half_37/v2_rolling_intercept/
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
    / "v2_rolling_intercept"
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

WINDOWS = {
    "RAW": None,
    "OOF_INTERCEPT_75": 75,
    "OOF_INTERCEPT_150": 150,
    "OOF_INTERCEPT_300": 300,
    "OOF_INTERCEPT_EXPANDING": -1,
}


@dataclass
class FoldDef:
    name: str
    kind: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def safe_roc(y, p):
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_pr(y, p):
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


def ece_binary(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def load_dev(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    needed = {TARGET, DATE_COL, ID_COL, STAGE_COL}
    if not needed.issubset(df.columns):
        raise ValueError(f"Colonnes absentes: {sorted(needed - set(df.columns))}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
    if df[DATE_COL].isna().any():
        raise ValueError("kickoff_utc non parsable")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target non résolu")
    if df[ID_COL].duplicated().any():
        raise ValueError("match_id dupliqué")
    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def load_top8(path: Path) -> List[str]:
    r = pd.read_csv(path)
    selected = r[r["selected_v2"] == 1].copy()
    selected = selected.sort_values("robust_selection_score", ascending=False)
    top8 = selected["feature"].astype(str).tolist()[:8]
    if top8 != EXPECTED_TOP8:
        raise RuntimeError(f"TOP8 inattendu.\nAttendu={EXPECTED_TOP8}\nTrouvé={top8}")
    return top8


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=C_FIXED,
            solver="liblinear",
            class_weight=None,
            max_iter=5000,
            random_state=87,
        ))
    ])


def make_folds(df, n_internal=5):
    orig_idx = np.where(df[STAGE_COL].astype(str).eq("original_consumed").to_numpy())[0]
    ext_idx = np.where(df[STAGE_COL].astype(str).eq("external_consumed").to_numpy())[0]

    if len(orig_idx) == 0 or len(ext_idx) == 0:
        raise RuntimeError("Stages absents")

    folds = []
    tscv = TimeSeriesSplit(n_splits=n_internal)
    local = np.arange(len(orig_idx))

    for i, (tr, va) in enumerate(tscv.split(local), 1):
        folds.append(FoldDef(
            f"INTERNAL_{i}", "internal", orig_idx[tr], orig_idx[va]
        ))

    folds.append(FoldDef(
        "EXTERNAL_STRESS", "external_stress", orig_idx, ext_idx
    ))
    return folds


def temporal_oof(train_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    n = len(train_df)
    if n < 120:
        raise RuntimeError("TRAIN trop petit")

    if n < 450:
        n_splits = 3
    elif n < 900:
        n_splits = 4
    else:
        n_splits = 5

    X = train_df[features].apply(pd.to_numeric, errors="coerce")
    y = train_df[TARGET].astype(int).to_numpy()

    rows = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for k, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        m = build_model()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        p = m.predict_proba(X.iloc[va_idx])[:, 1]

        for local_i, pred in zip(va_idx, p):
            rows.append({
                "row_order": int(local_i),
                "kickoff_utc": train_df.iloc[local_i][DATE_COL],
                "y": int(y[local_i]),
                "p": float(pred),
                "inner_fold": k,
            })

    out = pd.DataFrame(rows).sort_values(["kickoff_utc", "row_order"]).reset_index(drop=True)
    if len(out) < 50:
        raise RuntimeError("Pas assez de prédictions OOF")
    return out


def fit_intercept_shift(p: np.ndarray, y: np.ndarray) -> float:
    """
    Fit alpha dans sigmoid(logit(p)+alpha), slope fixé à 1.
    Strictement calibration d'intercept.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    y = np.asarray(y, dtype=float)
    z = logit(p)

    lo, hi = -8.0, 8.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        mean_pred = float(sigmoid(z + mid).mean())
        target_rate = float(y.mean())
        if mean_pred < target_rate:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def correct_baseline(y, train_rate):
    p0 = np.full(len(y), train_rate, dtype=float)
    return (
        float(brier_score_loss(y, p0)),
        float(log_loss(y, np.clip(p0, EPS, 1-EPS))),
    )


def evaluate_fold(df, fold, features):
    train = df.iloc[fold.train_idx].copy().sort_values([DATE_COL, ID_COL])
    val = df.iloc[fold.val_idx].copy().sort_values([DATE_COL, ID_COL])

    Xtr = train[features].apply(pd.to_numeric, errors="coerce")
    Xva = val[features].apply(pd.to_numeric, errors="coerce")
    ytr = train[TARGET].astype(int).to_numpy()
    yva = val[TARGET].astype(int).to_numpy()

    base = build_model()
    base.fit(Xtr, ytr)
    p_raw = base.predict_proba(Xva)[:, 1]

    oof = temporal_oof(train, features)

    train_rate = float(ytr.mean())
    val_rate = float(yva.mean())
    brier0, ll0 = correct_baseline(yva, train_rate)

    rows = []

    for strategy, window in WINDOWS.items():
        if strategy == "RAW":
            alpha = 0.0
            calib = pd.DataFrame()
        else:
            calib = oof.copy()
            if window != -1:
                calib = calib.tail(min(window, len(calib))).copy()

            alpha = fit_intercept_shift(
                calib["p"].to_numpy(),
                calib["y"].to_numpy(),
            )

        p = sigmoid(logit(p_raw) + alpha)

        rows.append({
            "fold": fold.name,
            "kind": fold.kind,
            "strategy": strategy,
            "window_requested": window,
            "calibration_rows": 0 if strategy == "RAW" else len(calib),
            "calibration_date_min": None if strategy == "RAW" else calib["kickoff_utc"].min(),
            "calibration_date_max": None if strategy == "RAW" else calib["kickoff_utc"].max(),
            "alpha_intercept_shift": alpha,
            "train_rows": len(train),
            "val_rows": len(val),
            "train_positive_rate": train_rate,
            "val_positive_rate": val_rate,
            "mean_probability": float(p.mean()),
            "gap_meanp_minus_actual": float(p.mean() - val_rate),
            "roc_auc": safe_roc(yva, p),
            "pr_auc": safe_pr(yva, p),
            "pr_lift_vs_val_rate": safe_pr(yva, p) - val_rate,
            "brier": float(brier_score_loss(yva, p)),
            "baseline_brier": brier0,
            "brier_improvement": brier0 - float(brier_score_loss(yva, p)),
            "log_loss": float(log_loss(yva, np.clip(p, EPS, 1-EPS))),
            "baseline_logloss": ll0,
            "logloss_improvement": ll0 - float(log_loss(yva, np.clip(p, EPS, 1-EPS))),
            "ece_10": ece_binary(yva, p),
        })

    return pd.DataFrame(rows), oof


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for strategy, b in metrics.groupby("strategy", sort=False):
        internal = b[b["kind"] == "internal"]
        stress = b[b["kind"] == "external_stress"].iloc[0]

        rows.append({
            "strategy": strategy,
            "mean_internal_roc": float(internal["roc_auc"].mean()),
            "mean_internal_pr": float(internal["pr_auc"].mean()),
            "mean_internal_pr_lift": float(internal["pr_lift_vs_val_rate"].mean()),
            "mean_internal_brier_improvement": float(internal["brier_improvement"].mean()),
            "internal_brier_better_fraction": float((internal["brier_improvement"] > 0).mean()),
            "mean_internal_logloss_improvement": float(internal["logloss_improvement"].mean()),
            "internal_logloss_better_fraction": float((internal["logloss_improvement"] > 0).mean()),
            "mean_internal_abs_gap": float(internal["gap_meanp_minus_actual"].abs().mean()),
            "mean_internal_ece": float(internal["ece_10"].mean()),
            "stress_roc": float(stress["roc_auc"]),
            "stress_pr": float(stress["pr_auc"]),
            "stress_pr_lift": float(stress["pr_lift_vs_val_rate"]),
            "stress_brier_improvement": float(stress["brier_improvement"]),
            "stress_logloss_improvement": float(stress["logloss_improvement"]),
            "stress_abs_gap": abs(float(stress["gap_meanp_minus_actual"])),
            "stress_ece": float(stress["ece_10"]),
            "stress_alpha": float(stress["alpha_intercept_shift"]),
        })

    out = pd.DataFrame(rows)

    out["passes_dynamic_calibration_gate"] = (
        (out["mean_internal_brier_improvement"] > 0)
        & (out["internal_brier_better_fraction"] >= 0.60)
        & (out["mean_internal_logloss_improvement"] > 0)
        & (out["internal_logloss_better_fraction"] >= 0.60)
        & (out["stress_brier_improvement"] > 0)
        & (out["stress_logloss_improvement"] > 0)
        & (out["stress_roc"] > 0.50)
        & (out["stress_pr_lift"] > 0)
    ).astype(int)

    out = out.sort_values(
        [
            "passes_dynamic_calibration_gate",
            "stress_brier_improvement",
            "stress_logloss_improvement",
            "mean_internal_brier_improvement",
            "mean_internal_logloss_improvement",
            "mean_internal_abs_gap",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(path, metrics, ranking):
    lines = []
    lines.append("# First Half 37 — V2 Rolling Intercept Calibration Audit")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- Base figée : `TOP_8 + logistic_l2 + C=0.01`.")
    lines.append("- Le shift d'intercept est appris uniquement sur des prédictions OOF temporelles du TRAIN.")
    lines.append("- Aucun target de validation n'est utilisé pour calibrer.")
    lines.append("- Aucun match postérieur au 19 septembre 2026 n'est lu.")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("```")
    lines.append(ranking.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Fold par fold")
    lines.append("")
    show = [
        "fold","strategy","calibration_rows","alpha_intercept_shift",
        "val_positive_rate","mean_probability","gap_meanp_minus_actual",
        "roc_auc","pr_auc","brier_improvement","logloss_improvement","ece_10"
    ]
    lines.append("```")
    lines.append(metrics[show].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Interprétation")
    lines.append("")
    lines.append(
        "Une stratégie dynamique est intéressante seulement si elle améliore la calibration "
        "sur la majorité des folds internes ET conserve une amélioration sur le stress externe."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
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
    print("FIRST HALF 37 — V2 ROLLING INTERCEPT CALIBRATION AUDIT")
    print("=" * 118)
    print(f"DEVELOPMENT : {dev_path}")
    print(f"Ranking     : {ranking_path}")
    print(f"Output      : {out_dir}")
    print("Pipeline    : TOP_8 + logistic_l2 + C=0.01")
    print("IMPORTANT   : calibration uniquement sur OOF TRAIN; aucun futur holdout.")
    print()

    df = load_dev(dev_path)
    features = load_top8(ranking_path)
    folds = make_folds(df, args.internal_folds)

    metric_frames = []
    oof_frames = []

    for fold in folds:
        m, oof = evaluate_fold(df, fold, features)
        metric_frames.append(m)
        oof = oof.copy()
        oof.insert(0, "outer_fold", fold.name)
        oof_frames.append(oof)

        print("-" * 118)
        print(f"{fold.name}")
        for _, r in m.iterrows():
            print(
                f"  {r['strategy']:<24} "
                f"alpha={r['alpha_intercept_shift']:+.3f} | "
                f"mean_p={r['mean_probability']:.3f} vs rate={r['val_positive_rate']:.3f} | "
                f"BrierImp={r['brier_improvement']:+.4f} | "
                f"LLImp={r['logloss_improvement']:+.4f} | "
                f"ROC={r['roc_auc']:.3f}"
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    oof_all = pd.concat(oof_frames, ignore_index=True)
    ranking = summarize(metrics)

    metrics_path = out_dir / "first_half_37_v2_rolling_intercept_fold_metrics.csv"
    ranking_path_out = out_dir / "first_half_37_v2_rolling_intercept_ranking.csv"
    oof_path = out_dir / "first_half_37_v2_rolling_intercept_oof.csv"
    report_path = out_dir / "first_half_37_v2_rolling_intercept_report.md"
    manifest_path = out_dir / "first_half_37_v2_rolling_intercept_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    ranking.to_csv(ranking_path_out, index=False)
    oof_all.to_csv(oof_path, index=False)
    write_report(report_path, metrics, ranking)

    manifest = {
        "schema_version": "first_half_37_v2_rolling_intercept_v1",
        "pipeline": {
            "subset": "TOP_8",
            "features": features,
            "model": "logistic_l2",
            "C": C_FIXED,
        },
        "strategies": WINDOWS,
        "future_holdout_used": False,
        "outputs": {
            "metrics": str(metrics_path),
            "ranking": str(ranking_path_out),
            "oof": str(oof_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    winner = ranking.iloc[0]

    print()
    print("=" * 118)
    print("ROLLING INTERCEPT AUDIT TERMINE")
    print("=" * 118)
    print(f"Strategies testées : {len(ranking)}")
    print(f"Strategies PASS : {int(ranking['passes_dynamic_calibration_gate'].sum())}")
    print(
        f"#1 : {winner['strategy']} | "
        f"INT BrierImp={winner['mean_internal_brier_improvement']:+.4f} | "
        f"INT LLImp={winner['mean_internal_logloss_improvement']:+.4f} | "
        f"STRESS BrierImp={winner['stress_brier_improvement']:+.4f} | "
        f"STRESS LLImp={winner['stress_logloss_improvement']:+.4f} | "
        f"STRESS ROC={winner['stress_roc']:.4f} | "
        f"PASS={int(winner['passes_dynamic_calibration_gate'])}"
    )
    print(f"Rapport : {report_path}")
    print(f"Ranking : {ranking_path_out}")
    print(f"Metrics : {metrics_path}")
    print("IMPORTANT: aucune stratégie n'est figée automatiquement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
