#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3-I BATCH 1 — VALIDATION EXACTE V3-C vs V3-H
=============================================

Cette version N'ESSAIE PLUS de reconstruire V3-H à la main.
Elle importe directement les helpers du script historique original :

    train_compare_v3c_v3h_calibration.py

Ainsi, elle réutilise exactement :
- le loader historique robuste ;
- le parsing temporel ;
- rolling_splits ;
- ExtraTrees LEAF5 ;
- temporal_oof_probabilities ;
- PlattCalibrator sur logit(p) ;
- les valeurs par défaut du protocole original via parse_args().

Protection de fidélité AVANT HOLDOUT
------------------------------------
Avant de scorer les 279 matchs, le script doit reproduire :
- les 4 ROC historiques ExtraTrees LEAF5 ;
- les 684 prédictions rolling OOS ;
- les bandes historiques V3-H déjà observées :
    31.0-34.0 : 126 matchs / 50 buts
    32.0-34.0 :  55 matchs / 23 buts
    32.5-34.0 :  39 matchs / 16 buts
    32.0-36.0 :  89 matchs / 34 buts

Si un contrôle échoue, le script S'ARRÊTE AVANT de scorer V3-I.

Ensuite il crée un candidat de déploiement V3-H gelé :
- ExtraTrees exact entraîné sur tout DEV=1368 ;
- Platt exact appris sur OOF temporel interne au DEV=1368 ;
- aucun usage des labels V3-I pour fit/calibration.

Bandes gelées testées sur V3-I
------------------------------
V3-C :
    55 <= p < 65 %
    58 <= p < 61 %

V3-H :
    31 <= p < 34 %
    32 <= p < 34 %
    32.5 <= p < 34 %

Usage
-----
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 -u evaluate_v3i_batch1_279_v3c_v3h_EXACT.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

ORIGINAL_V3H_SCRIPT = ROOT / "train_compare_v3c_v3h_calibration.py"

V3C_MODEL_PATH = (
    ROOT
    / "outputs/model_v3_experiments/candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)

HOLDOUT_PATH = (
    ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv"
)

OUT_DIR = (
    ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "evaluation_batch1_279_EXACT"
)

EXPECTED_HOLDOUT_N = 279
EXPECTED_DEV_N = 1368
EXPECTED_ROLLING_OOS_N = 684

TARGET = "goal_after_87"
MATCH_ID = "match_id"

# Résultats historiques de référence déjà obtenus avec le vrai V3-H.
EXPECTED_ET_RAW_ROC = [
    0.559719,
    0.506460,
    0.598338,
    0.591112,
]

# Ces bandes sont calculées sur les 684 rolling OOS du vrai
# V3H_ET_LEAF5_PLATT.
EXPECTED_HISTORICAL_BANDS = {
    "31.0-34.0": (0.310, 0.340, 126, 50),
    "32.0-34.0": (0.320, 0.340, 55, 23),
    "32.5-34.0": (0.325, 0.340, 39, 16),
    "32.0-36.0": (0.320, 0.360, 89, 34),
}

ROC_TOL = 5e-6
EPS = 1e-12


def hr(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def load_original_module():
    if not ORIGINAL_V3H_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script V3-H original introuvable : {ORIGINAL_V3H_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "goal87_v3h_original",
        ORIGINAL_V3H_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de créer le spec du script V3-H original.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_original_defaults(mod):
    """
    Récupère les vraies valeurs par défaut du parse_args() original
    sans laisser les arguments de CE script interférer.
    """
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(ORIGINAL_V3H_SCRIPT)]
        args = mod.parse_args()
    finally:
        sys.argv = old_argv
    return args


def resolve_original_inputs(mod, args):
    root = ROOT

    template_path = mod.resolve_explicit_or_default(
        root,
        args.template,
        mod.DEFAULT_TEMPLATE_REL,
        "Template V3-C",
    )
    schema_path = mod.resolve_explicit_or_default(
        root,
        args.schema,
        mod.DEFAULT_SCHEMA_REL,
        "Schéma V3-C",
    )
    dataset_path = mod.resolve_dataset(root, args.dataset)

    return template_path, schema_path, dataset_path


def prepare_historical_exact(mod, args, schema_path: Path, dataset_path: Path):
    features = mod.load_schema(schema_path)

    df, target_col, time_col, match_id_col, csv_audit = mod.load_dataset_robust(
        dataset_path,
        features,
        args.target,
        args.time_col,
    )

    y = mod.normalize_target(df[target_col])
    times = mod.parse_time_series(df[time_col])

    if times.isna().any():
        raise ValueError(
            f"Temps historique non parsé : {int(times.isna().sum())} lignes."
        )

    X = pd.DataFrame(index=df.index)
    for feat in features:
        X[feat] = pd.to_numeric(df[feat], errors="coerce")

    finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    df2 = df.loc[finite_mask].copy()
    X = X.loc[finite_mask].copy()
    y = y.loc[finite_mask].copy()
    times = times.loc[finite_mask].copy()

    order = np.argsort(times.to_numpy())

    df2 = df2.iloc[order].reset_index(drop=True)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    times = times.iloc[order].reset_index(drop=True)

    n = len(y)

    seen_tail_fraction = float(args.seen_tail_fraction)
    seen_tail_n = max(1, int(math.ceil(n * seen_tail_fraction)))
    dev_n = n - seen_tail_n

    X_dev = X.iloc[:dev_n].reset_index(drop=True)
    y_dev = y.iloc[:dev_n].reset_index(drop=True)
    t_dev = times.iloc[:dev_n].reset_index(drop=True)
    df_dev = df2.iloc[:dev_n].reset_index(drop=True)

    if dev_n != EXPECTED_DEV_N:
        raise RuntimeError(
            f"DEV historique inattendu : {dev_n}, attendu={EXPECTED_DEV_N}. "
            "STOP avant holdout."
        )

    outer_splits = mod.rolling_splits(
        dev_n,
        int(args.rolling_folds),
        float(args.rolling_initial_train_fraction),
    )

    return {
        "features": features,
        "df_all": df2,
        "X_all": X,
        "y_all": y,
        "times_all": times,
        "df_dev": df_dev,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "t_dev": t_dev,
        "outer_splits": outer_splits,
        "match_id_col": match_id_col,
        "target_col": target_col,
        "time_col": time_col,
        "csv_audit": csv_audit,
        "n_complete": n,
        "seen_tail_n": seen_tail_n,
    }


def safe_roc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_pr(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def ece10(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, 11)
    total = len(y)
    out = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i == 9 else (p < hi))
        n = int(mask.sum())
        if n:
            out += (n / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def wilson(pos: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = pos / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c-m, c+m


def check_historical_fidelity(mod, args, hist):
    """
    Rejoue EXACTEMENT la partie ET LEAF5 + Platt du protocole original.
    Aucun holdout V3-I n'est touché ici.
    """
    X_dev = hist["X_dev"]
    y_dev = hist["y_dev"]
    t_dev = hist["t_dev"]
    outer_splits = hist["outer_splits"]

    et_factory = mod.make_et_leaf5_factory(
        int(args.random_state),
        int(args.n_jobs),
        int(args.et_n_estimators),
    )

    pooled_frames = []
    fold_rows = []

    hr("FIDÉLITÉ V3-H — REPRODUCTION HISTORIQUE AVANT HOLDOUT")

    for fold_no, (train_idx, val_idx) in enumerate(outer_splits, start=1):
        Xtr = X_dev.iloc[train_idx].reset_index(drop=True)
        ytr = y_dev.iloc[train_idx].reset_index(drop=True)
        Xv = X_dev.iloc[val_idx]
        yv = y_dev.iloc[val_idx].to_numpy(dtype=int)

        et = et_factory()
        et.fit(Xtr, ytr)
        p_raw = mod.predict_probability(et, Xv)

        roc = safe_roc(yv, p_raw)
        expected_roc = EXPECTED_ET_RAW_ROC[fold_no - 1]
        delta = roc - expected_roc

        roc_ok = abs(delta) <= ROC_TOL

        print(
            f"CHECK F{fold_no} RAW ROC | "
            f"attendu={expected_roc:.6f} | obtenu={roc:.6f} | "
            f"delta={delta:+.8f} | {'OK' if roc_ok else 'FAIL'}"
        )

        if not roc_ok:
            raise RuntimeError(
                f"FIDÉLITÉ ÉCHOUÉE F{fold_no}: ROC brut V3-H différent. "
                "STOP avant scoring V3-I."
            )

        # Calibration EXACTE du script original : OOF temporel INTERNE
        y_oof, p_oof, oof_info = mod.temporal_oof_probabilities(
            et_factory,
            Xtr,
            ytr,
            int(args.inner_calibration_folds),
            float(args.inner_calibration_initial_train_fraction),
        )

        platt = mod.fit_calibrator("platt", p_oof, y_oof)
        p_platt = platt.transform(p_raw)

        fold_rows.append({
            "fold": fold_no,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "raw_roc": roc,
            "platt_roc": safe_roc(yv, p_platt),
            "platt_pr": safe_pr(yv, p_platt),
            "platt_brier": float(brier_score_loss(yv, p_platt)),
            "platt_logloss": float(
                log_loss(
                    yv,
                    np.clip(p_platt, EPS, 1-EPS),
                    labels=[0, 1],
                )
            ),
            "oof_calibration_n": int(oof_info["oof_calibration_n"]),
            "oof_positive_rate": float(oof_info["oof_positive_rate"]),
            "val_start": str(t_dev.iloc[val_idx[0]]),
            "val_end": str(t_dev.iloc[val_idx[-1]]),
        })

        pooled_frames.append(
            pd.DataFrame({
                "fold": fold_no,
                "target": yv,
                "p_v3h": p_platt,
            })
        )

    pooled = pd.concat(pooled_frames, ignore_index=True)

    if len(pooled) != EXPECTED_ROLLING_OOS_N:
        raise RuntimeError(
            f"Rolling OOS={len(pooled)}, attendu={EXPECTED_ROLLING_OOS_N}. "
            "STOP avant holdout."
        )

    print()
    print("CHECK BANDES HISTORIQUES V3-H")

    for name, (lo, hi, expected_n, expected_pos) in EXPECTED_HISTORICAL_BANDS.items():
        z = pooled[
            (pooled["p_v3h"] >= lo)
            & (pooled["p_v3h"] < hi)
        ]
        n = len(z)
        pos = int(z["target"].sum())
        ok = (n == expected_n and pos == expected_pos)

        print(
            f"{name} | attendu={expected_pos}/{expected_n} | "
            f"obtenu={pos}/{n} | {'OK' if ok else 'FAIL'}"
        )

        if not ok:
            raise RuntimeError(
                f"FIDÉLITÉ ÉCHOUÉE bande {name}: "
                f"attendu {expected_pos}/{expected_n}, obtenu {pos}/{n}. "
                "STOP avant scoring V3-I."
            )

    print()
    print(">>> FIDÉLITÉ HISTORIQUE V3-H CONFIRMÉE. Le holdout peut maintenant être scoré.")

    return et_factory, pooled, pd.DataFrame(fold_rows)


def fit_final_v3h_exact(mod, args, hist, et_factory):
    """
    Freeze déployable :
      - ET exact fit sur tout DEV=1368.
      - Platt exact fit sur OOF temporel interne au même DEV.
      - holdout jamais utilisé pour fit/calibration.
    """
    X_dev = hist["X_dev"]
    y_dev = hist["y_dev"]

    y_oof, p_oof, oof_info = mod.temporal_oof_probabilities(
        et_factory,
        X_dev,
        y_dev,
        int(args.inner_calibration_folds),
        float(args.inner_calibration_initial_train_fraction),
    )

    platt = mod.fit_calibrator("platt", p_oof, y_oof)

    et_final = et_factory()
    et_final.fit(X_dev, y_dev)

    # Pour persistance robuste, sauvegarde le sklearn LR interne du calibrateur,
    # pas l'instance d'une classe importée dynamiquement.
    platt_model = platt.model

    package = {
        "name": "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368",
        "features": list(hist["features"]),
        "et_model": et_final,
        "platt_logistic_model": platt_model,
        "calibration_input": "logit(raw_probability)",
        "dev_rows": int(len(X_dev)),
        "oof_calibration_n": int(oof_info["oof_calibration_n"]),
        "oof_positive_rate": float(oof_info["oof_positive_rate"]),
        "original_protocol_defaults": {
            "rolling_folds": int(args.rolling_folds),
            "rolling_initial_train_fraction": float(args.rolling_initial_train_fraction),
            "seen_tail_fraction": float(args.seen_tail_fraction),
            "inner_calibration_folds": int(args.inner_calibration_folds),
            "inner_calibration_initial_train_fraction": float(
                args.inner_calibration_initial_train_fraction
            ),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "et_n_estimators": int(args.et_n_estimators),
        },
    }

    return package


def apply_saved_platt(package, raw_p):
    raw_p = np.clip(np.asarray(raw_p, dtype=float), EPS, 1-EPS)
    x = np.log(raw_p / (1-raw_p)).reshape(-1, 1)
    return package["platt_logistic_model"].predict_proba(x)[:, 1]


def load_holdout(features: Sequence[str], historical_ids: set):
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(HOLDOUT_PATH)

    h = pd.read_csv(HOLDOUT_PATH, low_memory=False)

    if len(h) != EXPECTED_HOLDOUT_N:
        raise RuntimeError(
            f"Holdout={len(h)} lignes, attendu={EXPECTED_HOLDOUT_N}."
        )

    if TARGET not in h.columns:
        raise ValueError(f"Colonne target absente : {TARGET}")

    missing_features = [f for f in features if f not in h.columns]
    if missing_features:
        raise ValueError(
            f"Features V3-C absentes du holdout : {missing_features}"
        )

    if MATCH_ID not in h.columns:
        raise ValueError("match_id absent du holdout.")

    if h[MATCH_ID].duplicated().any():
        raise ValueError("match_id dupliqué dans le holdout.")

    overlap = sorted(
        set(h[MATCH_ID].astype(str))
        .intersection(historical_ids)
    )
    if overlap:
        raise RuntimeError(
            f"CONTAMINATION : {len(overlap)} IDs du holdout sont dans le master. "
            f"Exemples={overlap[:10]}"
        )

    h[TARGET] = pd.to_numeric(h[TARGET], errors="coerce")
    if not h[TARGET].isin([0,1]).all():
        raise ValueError("Targets V3-I non binaires ou manquants.")

    X = h[list(features)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        bad = int((~finite).sum())
        raise RuntimeError(
            f"Holdout READY contient {bad} lignes non 38/38."
        )

    return h.copy(), X


def predict_v3c(X):
    if not V3C_MODEL_PATH.exists():
        raise FileNotFoundError(V3C_MODEL_PATH)
    model = joblib.load(V3C_MODEL_PATH)
    p = np.asarray(model.predict_proba(X), dtype=float)
    return p[:, 1]


def model_metrics(y, probs: Mapping[str, np.ndarray]):
    rows = []
    y = np.asarray(y, dtype=int)

    for name, p in probs.items():
        p = np.asarray(p, dtype=float)
        rows.append({
            "model": name,
            "n": len(y),
            "positive_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "roc_auc": safe_roc(y, p),
            "pr_auc": safe_pr(y, p),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(
                log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0,1])
            ),
            "ece_10": ece10(y, p),
        })

    return pd.DataFrame(rows)


def band_stats(scored, name, col, lo, hi):
    z = scored[
        (scored[col] >= lo)
        & (scored[col] < hi)
    ]
    n = len(z)
    pos = int(z[TARGET].sum()) if n else 0
    rate = pos/n if n else float("nan")
    mean_p = float(z[col].mean()) if n else float("nan")
    ci_lo, ci_hi = wilson(pos, n)

    return {
        "rule": name,
        "n": n,
        "positives": pos,
        "observed_rate": rate,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "mean_probability": mean_p,
        "fair_odds_empirical": 1/rate if n and rate > 0 else float("nan"),
        "roi_at_3.20": rate*3.20 - 1 if n else float("nan"),
    }


def group_stats(name, z):
    n = len(z)
    pos = int(z[TARGET].sum()) if n else 0
    rate = pos/n if n else float("nan")
    lo, hi = wilson(pos, n)
    return {
        "group": name,
        "n": n,
        "positives": pos,
        "observed_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "fair_odds_empirical": 1/rate if n and rate > 0 else float("nan"),
        "roi_at_3.20": rate*3.20 - 1 if n else float("nan"),
    }


def fmt_pct(x):
    try:
        x = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(x) else f"{x:.2%}"


def fmt_num(x, d=6):
    try:
        x = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(x) else f"{x:.{d}f}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hr("V3-I BATCH 1 — VALIDATION EXACTE V3-C vs V3-H")

    mod = load_original_module()
    args = get_original_defaults(mod)

    template_path, schema_path, dataset_path = resolve_original_inputs(
        mod, args
    )

    print(f"Script V3-H original : {ORIGINAL_V3H_SCRIPT}")
    print(f"Dataset historique   : {dataset_path}")
    print(f"Schema V3-C          : {schema_path}")
    print(f"Template V3-C        : {template_path}")
    print(f"Holdout V3-I         : {HOLDOUT_PATH}")

    print()
    print("PARAMÈTRES ORIGINAUX DÉTECTÉS")
    for name in [
        "seen_tail_fraction",
        "rolling_folds",
        "rolling_initial_train_fraction",
        "inner_calibration_folds",
        "inner_calibration_initial_train_fraction",
        "random_state",
        "n_jobs",
        "et_n_estimators",
    ]:
        print(f"  {name} = {getattr(args, name)}")

    hist = prepare_historical_exact(
        mod,
        args,
        schema_path,
        dataset_path,
    )

    print()
    print(
        f"HIST COMPLETE={hist['n_complete']} | "
        f"DEV={len(hist['X_dev'])} | "
        f"TAIL={hist['seen_tail_n']}"
    )
    print(
        f"DEV TIME : {hist['t_dev'].iloc[0]} -> {hist['t_dev'].iloc[-1]}"
    )

    # Crucial : aucun holdout n'est chargé avant la réussite de ce contrôle.
    et_factory, pooled_hist, fidelity_folds = check_historical_fidelity(
        mod,
        args,
        hist,
    )

    # Freeze final exact.
    package = fit_final_v3h_exact(
        mod,
        args,
        hist,
        et_factory,
    )

    model_path = OUT_DIR / "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368.joblib"
    joblib.dump(package, model_path)

    # Seulement maintenant, ouverture du holdout.
    historical_ids = set(
        hist["df_all"][hist["match_id_col"]].astype(str)
    )

    holdout, Xh = load_holdout(
        hist["features"],
        historical_ids,
    )

    y = holdout[TARGET].astype(int).to_numpy()

    p_v3c = predict_v3c(Xh)

    p_v3h_raw = mod.predict_probability(
        package["et_model"],
        Xh,
    )
    p_v3h = apply_saved_platt(
        package,
        p_v3h_raw,
    )

    keep = [
        c for c in [
            MATCH_ID,
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            "kickoff_utc",
            "score_87_home",
            "score_87_away",
            TARGET,
            "target_status",
            "target_resolution_class",
            "target_confidence",
            "goal_minutes_after_87",
            "_v3i_selection_order",
        ]
        if c in holdout.columns
    ]

    scored = holdout[keep].copy()
    scored["p_v3c"] = p_v3c
    scored["p_v3h_raw"] = p_v3h_raw
    scored["p_v3h"] = p_v3h

    scored["sig_v3c_55_65"] = (
        (scored["p_v3c"] >= 0.55)
        & (scored["p_v3c"] < 0.65)
    )
    scored["sig_v3c_58_61"] = (
        (scored["p_v3c"] >= 0.58)
        & (scored["p_v3c"] < 0.61)
    )
    scored["sig_v3h_31_34"] = (
        (scored["p_v3h"] >= 0.31)
        & (scored["p_v3h"] < 0.34)
    )
    scored["sig_v3h_32_34"] = (
        (scored["p_v3h"] >= 0.32)
        & (scored["p_v3h"] < 0.34)
    )
    scored["sig_v3h_32p5_34"] = (
        (scored["p_v3h"] >= 0.325)
        & (scored["p_v3h"] < 0.34)
    )

    metrics = model_metrics(
        y,
        {
            "V3C_PRODUCTION": p_v3c,
            "V3H_ET_LEAF5_RAW_EXACT_FINAL": p_v3h_raw,
            "V3H_ET_LEAF5_PLATT_EXACT_FINAL": p_v3h,
        },
    )

    bands = pd.DataFrame([
        band_stats(scored, "V3C_55_65", "p_v3c", 0.55, 0.65),
        band_stats(scored, "V3C_58_61", "p_v3c", 0.58, 0.61),
        band_stats(scored, "V3H_31_34", "p_v3h", 0.31, 0.34),
        band_stats(scored, "V3H_32_34", "p_v3h", 0.32, 0.34),
        band_stats(scored, "V3H_32p5_34", "p_v3h", 0.325, 0.34),
    ])

    c = scored["sig_v3c_55_65"]
    h = scored["sig_v3h_31_34"]

    consensus = pd.DataFrame([
        group_stats("BOTH_MAIN", scored[c & h]),
        group_stats("V3C_ONLY_MAIN", scored[c & ~h]),
        group_stats("V3H_ONLY_MAIN", scored[~c & h]),
        group_stats("NEITHER_MAIN", scored[~c & ~h]),
        group_stats("V3C_55_65_ALL", scored[c]),
        group_stats("V3H_31_34_ALL", scored[h]),
        group_stats(
            "CONSENSUS_TARGETED_V3C58_61_AND_V3H31_34",
            scored[
                scored["sig_v3c_58_61"]
                & scored["sig_v3h_31_34"]
            ],
        ),
    ])

    # Save.
    fidelity_folds.to_csv(
        OUT_DIR / "historical_fidelity_fold_checks.csv",
        index=False,
    )
    pooled_hist.to_csv(
        OUT_DIR / "historical_v3h_rolling_oos_reproduced.csv",
        index=False,
    )
    scored.to_csv(
        OUT_DIR / "v3i_batch1_279_predictions_EXACT.csv",
        index=False,
    )
    metrics.to_csv(
        OUT_DIR / "v3i_batch1_279_model_metrics_EXACT.csv",
        index=False,
    )
    bands.to_csv(
        OUT_DIR / "v3i_batch1_279_frozen_band_metrics_EXACT.csv",
        index=False,
    )
    consensus.to_csv(
        OUT_DIR / "v3i_batch1_279_consensus_metrics_EXACT.csv",
        index=False,
    )

    manifest = {
        "status": "HISTORICAL_FIDELITY_CONFIRMED_BEFORE_HOLDOUT",
        "holdout_n": len(scored),
        "holdout_positives": int(scored[TARGET].sum()),
        "holdout_rate": float(scored[TARGET].mean()),
        "v3h_package": {
            k: v
            for k, v in package.items()
            if k not in {
                "et_model",
                "platt_logistic_model",
            }
        },
        "historical_expected_bands": EXPECTED_HISTORICAL_BANDS,
        "historical_expected_raw_roc": EXPECTED_ET_RAW_ROC,
    }

    (OUT_DIR / "validation_manifest_EXACT.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    hr("V3-I BATCH 1 — RÉSUMÉ EXACT")
    print(
        f"N={len(scored)} | buts après 87={int(scored[TARGET].sum())} | "
        f"taux={scored[TARGET].mean():.2%}"
    )

    hr("MÉTRIQUES GLOBALES")
    m = metrics.copy()
    for ccol in ["positive_rate", "mean_probability", "ece_10"]:
        m[ccol] = m[ccol].map(fmt_pct)
    for ccol in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        m[ccol] = m[ccol].map(lambda x: fmt_num(x, 6))
    print(m.to_string(index=False))

    hr("BANDES GELÉES")
    b = bands.copy()
    for ccol in [
        "observed_rate",
        "ci95_low",
        "ci95_high",
        "mean_probability",
        "roi_at_3.20",
    ]:
        b[ccol] = b[ccol].map(fmt_pct)
    b["fair_odds_empirical"] = b["fair_odds_empirical"].map(
        lambda x: fmt_num(x, 2)
    )
    print(b.to_string(index=False))

    hr("CONSENSUS")
    g = consensus.copy()
    for ccol in [
        "observed_rate",
        "ci95_low",
        "ci95_high",
        "roi_at_3.20",
    ]:
        g[ccol] = g[ccol].map(fmt_pct)
    g["fair_odds_empirical"] = g["fair_odds_empirical"].map(
        lambda x: fmt_num(x, 2)
    )
    print(g.to_string(index=False))

    hr("SORTIES")
    print(OUT_DIR)
    print()
    print(
        "IMPORTANT : ce Batch V3-I-1 est consommé. "
        "Ne pas déplacer les bandes après lecture de ces résultats."
    )


if __name__ == "__main__":
    main()
