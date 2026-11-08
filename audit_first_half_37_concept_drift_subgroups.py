#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — CONCEPT DRIFT / SUBGROUP PERFORMANCE AUDIT V1
=============================================================

Objectif
--------
Diagnostiquer POURQUOI le pipeline First_Half37 V1 a réussi le FINAL HOLDOUT
mais a échoué sur le nouvel échantillon externe, sans modifier le modèle.

Le script compare le FINAL HOLDOUT consommé aux 303 matchs externes et audite :
- performance globale et bootstrap de la chute de performance ;
- performance par jour ;
- performance par compétition ;
- performance par état du score à 37' ;
- performance selon la provenance shotmap / timeline fallback ;
- performance selon disponibilité xG ;
- performance selon complétude COMPACT10 ;
- bins des 10 features (bornes définies sur le dataset original uniquement) ;
- contribution logit de chaque feature du modèle logistique figé ;
- sensibilité par neutralisation d'une feature à la fois (PAS de refit).

IMPORTANT
---------
- Aucun réentraînement.
- Aucun tuning.
- Aucune nouvelle sélection de features.
- Aucune modification du C ou de la calibration.
- Les résultats servent uniquement à diagnostiquer la rupture de généralisation.
- Toute modification future du modèle crée une NOUVELLE version qui devra être
  validée sur un futur échantillon indépendant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.getenv("GOAL87_PROJECT_ROOT", str(BASE_DIR))
).expanduser()

DEFAULT_ORIGINAL = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "historical_api"
    / "First_Half_37_COMBINED.csv"
)
DEFAULT_EXTERNAL = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "post_model_external_v1"
    / "First_Half_37_POST_MODEL_EXTERNAL.csv"
)
DEFAULT_FINAL_PRED = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "final_holdout_v1"
    / "first_half_37_final_holdout_predictions.csv"
)
DEFAULT_EXTERNAL_PRED = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "post_model_external_v1"
    / "evaluation_v1" / "first_half_37_post_model_predictions.csv"
)
DEFAULT_FROZEN = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "final_holdout_v1"
    / "first_half_37_final_frozen_before_scoring.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "first_half_37" / "concept_drift_audit_v1"
)

TARGET = "goal_after_37_before_ht"
ID_COL = "match_id"
TIME_COL = "kickoff_utc"

EXPECTED_COMPACT10 = [
    "home_over25_freq",
    "away_wins_freq",
    "home_shots_37",
    "away_xg_37",
    "away_shots_on_target_37",
    "last_corner_minute",
    "away_form10",
    "league_home_over25_freq",
    "home_form5",
    "away_shots_37",
]

MIN_GROUP_N_DISPLAY = 5
MIN_GROUP_N_METRIC = 10
MIN_GROUP_CLASS = 3
EPS = 1e-9


# =============================================================================
# BASIC HELPERS
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def mixed_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, utc=True, errors="coerce")


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def apply_calibration(artifact: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    cal = artifact.get("calibration", {})
    if isinstance(cal, str):
        kind = cal
        cal = {"kind": kind}
    else:
        kind = str(cal.get("kind", "raw"))

    raw = np.asarray(raw, dtype=float)
    if kind == "raw":
        return raw
    if kind == "platt_temporal_oof":
        model = cal.get("model")
        if model is None:
            raise ValueError("Calibrateur Platt absent de l'artifact.")
        return model.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    if kind == "intercept_temporal_oof":
        return sigmoid(logit(raw) + float(cal["delta"]))
    if kind == "prior_corrected":
        pi = float(np.clip(float(cal["train_prevalence"]), 1e-6, 1 - 1e-6))
        return sigmoid(logit(raw) + math.log(pi / (1 - pi)))
    raise ValueError(f"Calibration inconnue : {kind}")


def fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:.2%}"


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_auc(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(pp)
    yy = yy[mask]
    pp = pp[mask]
    if len(yy) < 2 or len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, pp))
    except Exception:
        return float("nan")


def safe_ap(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(pp)
    yy = yy[mask]
    pp = pp[mask]
    if len(yy) < 2 or len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(yy, pp))
    except Exception:
        return float("nan")


def safe_brier(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(pp)
    if mask.sum() == 0:
        return float("nan")
    return float(brier_score_loss(yy[mask], pp[mask]))


def safe_logloss(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(pp)
    if mask.sum() == 0:
        return float("nan")
    pp = np.clip(pp[mask], EPS, 1 - EPS)
    return float(log_loss(yy[mask], pp, labels=[0, 1]))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - h), min(1, c + h)


# =============================================================================
# DATA LOADING / MERGING
# =============================================================================

def load_dataset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} introuvable : {path}")
    df = pd.read_csv(path, low_memory=False)
    if ID_COL not in df.columns or TARGET not in df.columns:
        raise ValueError(f"{label}: match_id/target absent")
    df = df.drop_duplicates(ID_COL, keep="last").copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if TIME_COL in df.columns:
        df["__kickoff__"] = mixed_datetime(df[TIME_COL])
    else:
        df["__kickoff__"] = pd.NaT
    return df


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} introuvable : {path}\n"
            "Ce diagnostic utilise les prédictions déjà produites ; il ne rescrore pas le holdout."
        )
    df = pd.read_csv(path, low_memory=False)
    if ID_COL not in df.columns:
        raise ValueError(f"{label}: match_id absent")
    pcol = None
    for c in ("final_probability", "probability", "calibrated_probability"):
        if c in df.columns:
            pcol = c
            break
    if pcol is None:
        raise ValueError(f"{label}: aucune colonne de probabilité finale reconnue")
    out = df[[ID_COL, pcol] + (["raw_probability"] if "raw_probability" in df.columns else [])].copy()
    out = out.rename(columns={pcol: "__prob__"})
    out["__prob__"] = pd.to_numeric(out["__prob__"], errors="coerce")
    if "raw_probability" in out.columns:
        out["raw_probability"] = pd.to_numeric(out["raw_probability"], errors="coerce")
    return out.drop_duplicates(ID_COL, keep="last")


def merge_scored(dataset: pd.DataFrame, pred: pd.DataFrame, label: str) -> pd.DataFrame:
    m = dataset.merge(pred, on=ID_COL, how="inner", validate="one_to_one")
    if len(m) != len(pred):
        missing = len(pred) - len(m)
        raise ValueError(f"{label}: {missing} prédictions sans ligne dataset associée")
    m[TARGET] = pd.to_numeric(m[TARGET], errors="coerce")
    if m[TARGET].isna().any() or m["__prob__"].isna().any():
        raise ValueError(f"{label}: target/probabilité manquant après merge")
    m[TARGET] = m[TARGET].astype(int)
    return m


# =============================================================================
# GROUP DERIVATIONS
# =============================================================================

def score_state(df: pd.DataFrame) -> pd.Series:
    h = numeric(df.get("score_37_home", pd.Series(np.nan, index=df.index)))
    a = numeric(df.get("score_37_away", pd.Series(np.nan, index=df.index)))
    out = pd.Series("<missing>", index=df.index, dtype="object")
    valid = h.notna() & a.notna()
    out.loc[valid & (h == 0) & (a == 0)] = "0-0"
    out.loc[valid & (h > a)] = "home_lead"
    out.loc[valid & (h < a)] = "away_lead"
    out.loc[valid & (h == a) & ((h + a) > 0)] = "draw_with_goals"
    return out


def parse_provenance(value: Any) -> Dict[str, str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    try:
        obj = json.loads(str(value))
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def derive_groups(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["__date__"] = out["__kickoff__"].dt.strftime("%Y-%m-%d")
    out["__competition__"] = out.get("competition_id", pd.Series("<missing>", index=out.index)).astype(str)
    out["__score_state__"] = score_state(out)
    out["__xg_available__"] = np.where(numeric(out.get("away_xg_37", pd.Series(np.nan, index=out.index))).notna(), "xg_available", "xg_missing")

    fmat = out.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    out["__compact10_complete__"] = np.where(fmat.notna().all(axis=1), "complete_10_10", "has_missing")

    prov_col = out.get("_live_provenance_json", pd.Series("", index=out.index))
    shot_source = []
    for v in prov_col:
        d = parse_provenance(v)
        p = d.get("away_shots_37", "<missing_provenance>")
        if "timeline_shot_fallback" in p:
            shot_source.append("timeline_fallback")
        elif "shotmap" in p:
            shot_source.append("shotmap")
        else:
            shot_source.append(p or "<missing_provenance>")
    out["__shot_source__"] = shot_source

    return out


# =============================================================================
# METRICS BY GROUP
# =============================================================================

def metric_row(g: pd.DataFrame, group_name: str, group_value: str, dataset_label: str) -> Dict[str, Any]:
    y = g[TARGET].astype(int).to_numpy()
    p = g["__prob__"].astype(float).to_numpy()
    n = len(g)
    pos = int(y.sum())
    neg = int(n - pos)
    rate = float(y.mean()) if n else float("nan")
    lo, hi = wilson(pos, n)

    enough = n >= MIN_GROUP_N_METRIC and pos >= MIN_GROUP_CLASS and neg >= MIN_GROUP_CLASS
    auc = safe_auc(y, p) if enough else float("nan")
    ap = safe_ap(y, p) if enough else float("nan")
    return {
        "dataset": dataset_label,
        "group_type": group_name,
        "group_value": str(group_value),
        "n": n,
        "positives": pos,
        "negatives": neg,
        "positive_rate": rate,
        "positive_rate_ci95_low": lo,
        "positive_rate_ci95_high": hi,
        "mean_probability": float(np.mean(p)) if n else float("nan"),
        "roc_auc": auc,
        "pr_auc": ap,
        "pr_lift_vs_group_rate": (ap - rate) if np.isfinite(ap) else float("nan"),
        "brier": safe_brier(y, p),
        "log_loss": safe_logloss(y, p),
        "metric_reliable_flag": int(enough),
    }


def group_metrics(df: pd.DataFrame, column: str, group_name: str, dataset_label: str) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(column, dropna=False, sort=True):
        rows.append(metric_row(g, group_name, str(val), dataset_label))
    return pd.DataFrame(rows)


def overall_metric(df: pd.DataFrame, dataset_label: str) -> Dict[str, Any]:
    r = metric_row(df, "overall", "ALL", dataset_label)
    return r


# =============================================================================
# FEATURE BIN ANALYSIS
# =============================================================================

def quantile_edges(reference: pd.Series, q: int = 5) -> np.ndarray:
    x = numeric(reference).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, q + 1)
    vals = np.unique(np.quantile(x, qs))
    if len(vals) < 2:
        return np.array([-np.inf, np.inf])
    internal = vals[1:-1]
    return np.concatenate(([-np.inf], internal, [np.inf]))


def edge_label(lo: float, hi: float) -> str:
    if np.isneginf(lo):
        return f"< {hi:.4g}"
    if np.isposinf(hi):
        return f">= {lo:.4g}"
    return f"{lo:.4g} – {hi:.4g}"


def feature_bin_table(
    original: pd.DataFrame,
    final_df: pd.DataFrame,
    external_df: pd.DataFrame,
    features: Sequence[str],
    n_bins: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feat in features:
        edges = quantile_edges(original[feat], n_bins)
        labels = [edge_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        for dataset_label, df in (("final_holdout", final_df), ("external", external_df)):
            values = numeric(df[feat])
            bins = pd.cut(values, bins=edges, labels=labels, right=False, include_lowest=True)
            temp = df.copy()
            temp["__feat_bin__"] = bins.astype("object").fillna("<missing>")
            for bin_name, g in temp.groupby("__feat_bin__", dropna=False, sort=False):
                n = len(g)
                y = g[TARGET].astype(int).to_numpy()
                p = g["__prob__"].astype(float).to_numpy()
                rows.append({
                    "feature": feat,
                    "dataset": dataset_label,
                    "bin": str(bin_name),
                    "n": n,
                    "positives": int(y.sum()),
                    "observed_rate": float(y.mean()) if n else np.nan,
                    "mean_probability": float(p.mean()) if n else np.nan,
                    "calibration_gap": float(y.mean() - p.mean()) if n else np.nan,
                })
    return pd.DataFrame(rows)


# =============================================================================
# MODEL CONTRIBUTIONS / NEUTRALIZATION
# =============================================================================

def get_model_components(artifact: Mapping[str, Any]):
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        raise ValueError("Pipeline sklearn figé introuvable dans artifact")
    steps = pipeline.named_steps
    for required in ("imputer", "scale", "model"):
        if required not in steps:
            raise ValueError(f"Étape {required!r} absente du pipeline")
    return pipeline, steps["imputer"], steps["scale"], steps["model"]


def transformed_matrix(artifact: Mapping[str, Any], df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipeline, imputer, scaler, model = get_model_components(artifact)
    X = df.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    Xi = imputer.transform(X)
    Xs = scaler.transform(Xi)
    coef = np.asarray(model.coef_[0], dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if Xs.shape[1] != len(features) or len(coef) != len(features):
        raise ValueError(
            f"Dimensions contribution incompatibles: Xs={Xs.shape}, coef={len(coef)}, features={len(features)}"
        )
    return np.asarray(Xs, dtype=float), coef, intercept


def contribution_table(artifact: Mapping[str, Any], df: pd.DataFrame, features: Sequence[str], dataset_label: str) -> pd.DataFrame:
    Xs, coef, intercept = transformed_matrix(artifact, df, features)
    y = df[TARGET].astype(int).to_numpy()
    contrib = Xs * coef.reshape(1, -1)
    rows = []
    for j, feat in enumerate(features):
        c = contrib[:, j]
        pos = c[y == 1]
        neg = c[y == 0]
        auc = safe_auc(y, c)
        rows.append({
            "dataset": dataset_label,
            "feature": feat,
            "coefficient": float(coef[j]),
            "mean_contribution": float(np.mean(c)),
            "mean_abs_contribution": float(np.mean(np.abs(c))),
            "mean_contribution_positive": float(np.mean(pos)) if len(pos) else np.nan,
            "mean_contribution_negative": float(np.mean(neg)) if len(neg) else np.nan,
            "aligned_contribution_gap_pos_minus_neg": float(np.mean(pos) - np.mean(neg)) if len(pos) and len(neg) else np.nan,
            "contribution_auc": auc,
            "contribution_corr_target": safe_corr(c, y),
        })
    return pd.DataFrame(rows)


def contribution_comparison(final_c: pd.DataFrame, external_c: pd.DataFrame) -> pd.DataFrame:
    f = final_c.set_index("feature")
    e = external_c.set_index("feature")
    rows = []
    for feat in f.index:
        rf = f.loc[feat]
        re = e.loc[feat]
        final_auc = float(rf["contribution_auc"])
        ext_auc = float(re["contribution_auc"])
        final_gap = float(rf["aligned_contribution_gap_pos_minus_neg"])
        ext_gap = float(re["aligned_contribution_gap_pos_minus_neg"])
        # Stronger diagnostic than raw correlation: did the feature's actual model
        # contribution rank positives above negatives in one sample but not the next?
        inversion = int(
            np.isfinite(final_auc) and np.isfinite(ext_auc)
            and ((final_auc > 0.52 and ext_auc < 0.48) or (final_auc < 0.48 and ext_auc > 0.52))
        )
        breakdown = int(
            np.isfinite(final_auc) and np.isfinite(ext_auc)
            and final_auc >= 0.52 and ext_auc <= 0.50
        )
        rows.append({
            "feature": feat,
            "coefficient": float(rf["coefficient"]),
            "final_contribution_auc": final_auc,
            "external_contribution_auc": ext_auc,
            "auc_delta_external_minus_final": ext_auc - final_auc,
            "final_aligned_gap": final_gap,
            "external_aligned_gap": ext_gap,
            "gap_delta_external_minus_final": ext_gap - final_gap,
            "final_mean_abs_contribution": float(rf["mean_abs_contribution"]),
            "external_mean_abs_contribution": float(re["mean_abs_contribution"]),
            "strong_direction_inversion": inversion,
            "signal_breakdown_flag": breakdown,
        })
    return pd.DataFrame(rows).sort_values(
        ["signal_breakdown_flag", "auc_delta_external_minus_final"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def score_with_artifact(artifact: Mapping[str, Any], X: pd.DataFrame) -> np.ndarray:
    pipeline = artifact["pipeline"]
    raw = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    p = np.asarray(apply_calibration(artifact, raw), dtype=float)
    return np.clip(p, EPS, 1 - EPS)


def neutralization_sensitivity(
    artifact: Mapping[str, Any],
    df: pd.DataFrame,
    features: Sequence[str],
    dataset_label: str,
) -> pd.DataFrame:
    pipeline, imputer, scaler, model = get_model_components(artifact)
    X = df.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").copy()
    y = df[TARGET].astype(int).to_numpy()
    base_p = df["__prob__"].to_numpy(dtype=float)
    base_auc = safe_auc(y, base_p)
    base_ap = safe_ap(y, base_p)
    base_brier = safe_brier(y, base_p)
    stats = np.asarray(imputer.statistics_, dtype=float)
    if len(stats) != len(features):
        raise ValueError("Nombre de statistiques imputer != features")

    rows = []
    for j, feat in enumerate(features):
        Xn = X.copy()
        Xn[feat] = stats[j]
        p = score_with_artifact(artifact, Xn)
        auc = safe_auc(y, p)
        ap = safe_ap(y, p)
        brier = safe_brier(y, p)
        rows.append({
            "dataset": dataset_label,
            "feature": feat,
            "neutral_value_train_median": float(stats[j]),
            "baseline_auc": base_auc,
            "neutralized_auc": auc,
            "delta_auc_neutralized_minus_base": auc - base_auc if np.isfinite(auc) and np.isfinite(base_auc) else np.nan,
            "baseline_pr_auc": base_ap,
            "neutralized_pr_auc": ap,
            "delta_pr_neutralized_minus_base": ap - base_ap if np.isfinite(ap) and np.isfinite(base_ap) else np.nan,
            "baseline_brier": base_brier,
            "neutralized_brier": brier,
            "brier_improvement_when_neutralized": base_brier - brier if np.isfinite(brier) and np.isfinite(base_brier) else np.nan,
        })
    return pd.DataFrame(rows)


def neutralization_comparison(final_n: pd.DataFrame, external_n: pd.DataFrame) -> pd.DataFrame:
    f = final_n.set_index("feature")
    e = external_n.set_index("feature")
    rows = []
    for feat in f.index:
        rf, re = f.loc[feat], e.loc[feat]
        fd = float(rf["delta_auc_neutralized_minus_base"])
        ed = float(re["delta_auc_neutralized_minus_base"])
        # If neutralizing hurt the final holdout (fd<0) but helps external (ed>0),
        # this is a useful concept-drift suspect. Diagnostic only.
        suspect = int(np.isfinite(fd) and np.isfinite(ed) and fd <= -0.005 and ed >= 0.005)
        rows.append({
            "feature": feat,
            "final_delta_auc_if_neutralized": fd,
            "external_delta_auc_if_neutralized": ed,
            "external_delta_pr_if_neutralized": float(re["delta_pr_neutralized_minus_base"]),
            "external_brier_improvement_if_neutralized": float(re["brier_improvement_when_neutralized"]),
            "concept_drift_suspect_flag": suspect,
        })
    return pd.DataFrame(rows).sort_values(
        ["concept_drift_suspect_flag", "external_delta_auc_if_neutralized"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# BOOTSTRAP FINAL vs EXTERNAL
# =============================================================================

def bootstrap_difference(final_df: pd.DataFrame, external_df: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    yf = final_df[TARGET].astype(int).to_numpy()
    pf = final_df["__prob__"].astype(float).to_numpy()
    ye = external_df[TARGET].astype(int).to_numpy()
    pe = external_df["__prob__"].astype(float).to_numpy()

    auc_delta: List[float] = []
    prlift_delta: List[float] = []
    for _ in range(int(n_boot)):
        fi = rng.integers(0, len(yf), size=len(yf))
        ei = rng.integers(0, len(ye), size=len(ye))
        yfb, pfb = yf[fi], pf[fi]
        yeb, peb = ye[ei], pe[ei]
        if len(np.unique(yfb)) < 2 or len(np.unique(yeb)) < 2:
            continue
        af = roc_auc_score(yfb, pfb)
        ae = roc_auc_score(yeb, peb)
        prf = average_precision_score(yfb, pfb) - yfb.mean()
        pre = average_precision_score(yeb, peb) - yeb.mean()
        auc_delta.append(float(ae - af))
        prlift_delta.append(float(pre - prf))

    def summary(vals: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(vals, dtype=float)
        return {
            "n_valid": int(len(a)),
            "p2_5": float(np.quantile(a, 0.025)),
            "median": float(np.quantile(a, 0.50)),
            "p97_5": float(np.quantile(a, 0.975)),
            "probability_delta_below_zero": float(np.mean(a < 0)),
        }

    return {
        "roc_auc_external_minus_final": summary(auc_delta),
        "pr_lift_external_minus_final": summary(prlift_delta),
    }


# =============================================================================
# REPORT / DIAGNOSTIC
# =============================================================================

def subgroup_summary_text(group_df: pd.DataFrame, dataset: str, group_type: str, top_n: int = 12) -> str:
    g = group_df[(group_df["dataset"] == dataset) & (group_df["group_type"] == group_type)].copy()
    if g.empty:
        return "(aucune ligne)"
    g = g[g["n"] >= MIN_GROUP_N_DISPLAY]
    cols = ["group_value", "n", "positives", "positive_rate", "mean_probability", "roc_auc", "pr_auc", "pr_lift_vs_group_rate"]
    g = g.sort_values("n", ascending=False).head(top_n)
    return g[cols].to_string(index=False)


def automatic_diagnostic(
    overall: pd.DataFrame,
    subgroup: pd.DataFrame,
    contrib_cmp: pd.DataFrame,
    neutral_cmp: pd.DataFrame,
) -> Dict[str, Any]:
    final = overall[overall["dataset"] == "final_holdout"].iloc[0]
    ext = overall[overall["dataset"] == "external"].iloc[0]

    flags: List[str] = []
    if float(ext["roc_auc"]) < 0.50:
        flags.append("external_roc_below_0_50")
    if float(ext["pr_lift_vs_group_rate"]) < 0:
        flags.append("external_pr_below_prevalence")
    if float(ext["positive_rate"]) - float(final["positive_rate"]) >= 0.04:
        flags.append("base_rate_shift_ge_4pp")

    # Is failure obvious in a specific provenance group?
    shot = subgroup[(subgroup.dataset == "external") & (subgroup.group_type == "shot_source") & (subgroup.metric_reliable_flag == 1)]
    provenance_concentrated = False
    if len(shot) >= 2:
        good = shot[shot.roc_auc >= 0.52]
        bad = shot[shot.roc_auc <= 0.48]
        if len(good) and len(bad):
            provenance_concentrated = True
            flags.append("possible_provenance_specific_failure")

    scoreg = subgroup[(subgroup.dataset == "external") & (subgroup.group_type == "score_state") & (subgroup.metric_reliable_flag == 1)]
    broad_score_failure = bool(len(scoreg) >= 3 and (scoreg.roc_auc < 0.50).mean() >= 0.67)
    if broad_score_failure:
        flags.append("failure_across_multiple_score_states")

    days = subgroup[(subgroup.dataset == "external") & (subgroup.group_type == "day") & (subgroup.metric_reliable_flag == 1)]
    broad_day_failure = bool(len(days) >= 3 and (days.roc_auc < 0.50).mean() >= 0.60)
    if broad_day_failure:
        flags.append("failure_across_multiple_days")

    suspect_features = neutral_cmp.loc[neutral_cmp["concept_drift_suspect_flag"] == 1, "feature"].tolist()
    breakdown_features = contrib_cmp.loc[contrib_cmp["signal_breakdown_flag"] == 1, "feature"].tolist()

    if broad_score_failure and broad_day_failure and not provenance_concentrated:
        verdict = "BROAD_SIGNAL_BREAKDOWN"
    elif provenance_concentrated or suspect_features:
        verdict = "POSSIBLE_LOCALIZED_CONCEPT_DRIFT"
    else:
        verdict = "MIXED_OR_INCONCLUSIVE_CONCEPT_DRIFT"

    return {
        "verdict": verdict,
        "flags": flags,
        "neutralization_suspect_features": suspect_features,
        "contribution_breakdown_features": breakdown_features,
        "final_roc_auc": float(final["roc_auc"]),
        "external_roc_auc": float(ext["roc_auc"]),
        "final_pr_auc": float(final["pr_auc"]),
        "external_pr_auc": float(ext["pr_auc"]),
        "final_positive_rate": float(final["positive_rate"]),
        "external_positive_rate": float(ext["positive_rate"]),
    }


def write_report(
    path: Path,
    overall: pd.DataFrame,
    subgroup: pd.DataFrame,
    contrib_cmp: pd.DataFrame,
    neutral_cmp: pd.DataFrame,
    feature_bins: pd.DataFrame,
    boot: Dict[str, Any],
    diagnostic: Dict[str, Any],
    features: Sequence[str],
) -> None:
    final = overall[overall.dataset == "final_holdout"].iloc[0]
    ext = overall[overall.dataset == "external"].iloc[0]

    lines: List[str] = [
        "# First Half 37 — Concept Drift / Subgroup Audit V1",
        "",
        "## Discipline",
        "",
        "- Aucun refit, tuning, changement de feature, C ou calibration.",
        "- Comparaison principale : FINAL HOLDOUT consommé vs échantillon externe post-modèle.",
        "- Les analyses de neutralisation sont des diagnostics contrefactuels du modèle figé, pas une nouvelle sélection de features.",
        "",
        "## Pipeline figé",
        "",
        f"- Features : **{len(features)}** — " + ", ".join(f"`{x}`" for x in features),
        "",
        "## Performance globale",
        "",
        "| sample | n | positives | rate | mean p | ROC-AUC | PR-AUC | PR lift | Brier | LogLoss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in (final, ext):
        lines.append(
            f"| {r['dataset']} | {int(r['n'])} | {int(r['positives'])} | {float(r['positive_rate']):.2%} | "
            f"{float(r['mean_probability']):.2%} | {float(r['roc_auc']):.4f} | {float(r['pr_auc']):.4f} | "
            f"{float(r['pr_lift_vs_group_rate']):+.4f} | {float(r['brier']):.4f} | {float(r['log_loss']):.4f} |"
        )

    lines += [
        "",
        "## Bootstrap de la chute externe vs final",
        "",
        "```json",
        json.dumps(boot, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Diagnostic automatique",
        "",
        f"- Verdict : **`{diagnostic['verdict']}`**",
        f"- Flags : `{', '.join(diagnostic['flags']) if diagnostic['flags'] else 'none'}`",
        f"- Features suspectes par neutralisation : `{', '.join(diagnostic['neutralization_suspect_features']) if diagnostic['neutralization_suspect_features'] else 'none'}`",
        f"- Features avec rupture contribution : `{', '.join(diagnostic['contribution_breakdown_features']) if diagnostic['contribution_breakdown_features'] else 'none'}`",
        "",
        "## Performance EXTERNE par jour",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "day", 20),
        "```",
        "",
        "## Performance EXTERNE par état du score à 37'",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "score_state", 20),
        "```",
        "",
        "## Performance EXTERNE par provenance des tirs",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "shot_source", 20),
        "```",
        "",
        "## Performance EXTERNE selon disponibilité xG",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "xg_availability", 20),
        "```",
        "",
        "## Performance EXTERNE selon complétude COMPACT10",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "compact10_completeness", 20),
        "```",
        "",
        "## Compétitions externes — plus gros volumes",
        "",
        "```",
        subgroup_summary_text(subgroup, "external", "competition", 30),
        "```",
        "",
        "## Contribution des features : FINAL vs EXTERNE",
        "",
        "Une `contribution_auc` > 0.50 signifie que la contribution réelle de la feature dans le logit du modèle classe, seule, les positifs au-dessus des négatifs. < 0.50 signifie l'inverse.",
        "",
        "```",
        contrib_cmp.to_string(index=False),
        "```",
        "",
        "## Sensibilité par neutralisation",
        "",
        "On remplace une feature par sa médiane de TRAIN, sans réentraîner. Un delta AUC externe positif signifie que le modèle externe se classe mieux quand l'information de cette feature est neutralisée. Diagnostic uniquement.",
        "",
        "```",
        neutral_cmp.to_string(index=False),
        "```",
        "",
        "## Bins des features",
        "",
        "Les bornes sont définies à partir du dataset original, puis réutilisées telles quelles pour FINAL et EXTERNE. Le détail complet est dans le CSV `first_half_37_feature_bins.csv`.",
        "",
        "## Discipline pour la suite",
        "",
        "Si ce diagnostic conduit à modifier le pipeline, le résultat devient une nouvelle version expérimentale. Les 303 matchs externes sont alors consommés pour diagnostic et ne peuvent plus servir de validation externe indépendante de la nouvelle version.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit concept drift / sous-groupes First_Half37 V1")
    p.add_argument("--original-csv", type=Path, default=DEFAULT_ORIGINAL)
    p.add_argument("--external-csv", type=Path, default=DEFAULT_EXTERNAL)
    p.add_argument("--final-predictions", type=Path, default=DEFAULT_FINAL_PRED)
    p.add_argument("--external-predictions", type=Path, default=DEFAULT_EXTERNAL_PRED)
    p.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--random-seed", type=int, default=87)
    p.add_argument("--feature-bins", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pth, label in (
        (args.original_csv, "dataset original"),
        (args.external_csv, "dataset externe"),
        (args.final_predictions, "prédictions FINAL"),
        (args.external_predictions, "prédictions externes"),
        (args.frozen_manifest, "manifest figé"),
    ):
        if not pth.exists():
            raise FileNotFoundError(f"{label} introuvable : {pth}")

    frozen = load_json(args.frozen_manifest)
    features = list(frozen.get("features", []))
    if not features:
        raise ValueError("Features absentes du manifest figé")
    if features != EXPECTED_COMPACT10:
        raise ValueError(
            "Le schéma figé n'est pas exactement COMPACT10 attendu.\n"
            f"Manifest: {features}\nAttendu: {EXPECTED_COMPACT10}"
        )

    candidate_path = Path(str(frozen.get("candidate_path", "")))
    if not candidate_path.exists():
        raise FileNotFoundError(f"Artifact candidat figé introuvable : {candidate_path}")
    expected_hash = str(frozen.get("candidate_sha256", ""))
    actual_hash = sha256_file(candidate_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("SHA-256 du candidat différent de celui figé avant scoring.")
    artifact = joblib.load(candidate_path)
    if list(artifact.get("features", [])) != features:
        raise ValueError("Features artifact != manifest")

    original = load_dataset(args.original_csv, "original")
    external = load_dataset(args.external_csv, "external")
    final_pred = load_predictions(args.final_predictions, "FINAL predictions")
    ext_pred = load_predictions(args.external_predictions, "external predictions")

    final_df = merge_scored(original, final_pred, "FINAL")
    ext_df = merge_scored(external, ext_pred, "EXTERNAL")

    # Strict disjointness.
    overlap = len(set(final_df[ID_COL].astype(str)) & set(ext_df[ID_COL].astype(str)))
    if overlap:
        raise ValueError(f"Overlap FINAL/externe interdit : {overlap}")

    for feat in features:
        if feat not in original.columns or feat not in external.columns:
            raise ValueError(f"Feature COMPACT10 absente : {feat}")

    final_df = derive_groups(final_df, features)
    ext_df = derive_groups(ext_df, features)

    # Global metrics.
    overall = pd.DataFrame([
        overall_metric(final_df, "final_holdout"),
        overall_metric(ext_df, "external"),
    ])

    # Subgroups for both samples.
    group_specs = [
        ("__date__", "day"),
        ("__competition__", "competition"),
        ("__score_state__", "score_state"),
        ("__shot_source__", "shot_source"),
        ("__xg_available__", "xg_availability"),
        ("__compact10_complete__", "compact10_completeness"),
    ]
    group_frames = []
    for col, name in group_specs:
        group_frames.append(group_metrics(final_df, col, name, "final_holdout"))
        group_frames.append(group_metrics(ext_df, col, name, "external"))
    subgroup = pd.concat(group_frames, ignore_index=True)

    # Feature bins.
    feature_bins = feature_bin_table(
        original=original,
        final_df=final_df,
        external_df=ext_df,
        features=features,
        n_bins=max(3, int(args.feature_bins)),
    )

    # Actual model contributions.
    final_contrib = contribution_table(artifact, final_df, features, "final_holdout")
    ext_contrib = contribution_table(artifact, ext_df, features, "external")
    contrib_cmp = contribution_comparison(final_contrib, ext_contrib)

    # One-feature neutralization diagnostics.
    final_neutral = neutralization_sensitivity(artifact, final_df, features, "final_holdout")
    ext_neutral = neutralization_sensitivity(artifact, ext_df, features, "external")
    neutral_cmp = neutralization_comparison(final_neutral, ext_neutral)

    # Bootstrap performance deterioration.
    boot = bootstrap_difference(
        final_df, ext_df, n_boot=max(200, int(args.bootstrap)), seed=int(args.random_seed)
    )

    diagnostic = automatic_diagnostic(overall, subgroup, contrib_cmp, neutral_cmp)

    # Outputs.
    overall_path = out_dir / "first_half_37_overall_final_vs_external.csv"
    subgroup_path = out_dir / "first_half_37_subgroup_performance.csv"
    bins_path = out_dir / "first_half_37_feature_bins.csv"
    contrib_path = out_dir / "first_half_37_feature_contribution_comparison.csv"
    neutral_path = out_dir / "first_half_37_feature_neutralization_sensitivity.csv"
    report_path = out_dir / "first_half_37_concept_drift_report.md"
    manifest_path = out_dir / "first_half_37_concept_drift_manifest.json"

    overall.to_csv(overall_path, index=False)
    subgroup.to_csv(subgroup_path, index=False)
    feature_bins.to_csv(bins_path, index=False)
    contrib_cmp.to_csv(contrib_path, index=False)
    neutral_cmp.to_csv(neutral_path, index=False)

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "diagnostic_only": True,
        "no_refit_no_tuning": True,
        "verdict": diagnostic["verdict"],
        "diagnostic": diagnostic,
        "bootstrap_difference": boot,
        "features": features,
        "candidate_path": str(candidate_path),
        "candidate_sha256": actual_hash,
        "original_csv": str(args.original_csv),
        "external_csv": str(args.external_csv),
        "final_predictions": str(args.final_predictions),
        "external_predictions": str(args.external_predictions),
        "rows_final": int(len(final_df)),
        "rows_external": int(len(ext_df)),
        "outputs": {
            "overall": str(overall_path),
            "subgroups": str(subgroup_path),
            "feature_bins": str(bins_path),
            "contributions": str(contrib_path),
            "neutralization": str(neutral_path),
            "report": str(report_path),
        },
    }
    dump_json(manifest_path, manifest)

    write_report(
        report_path,
        overall,
        subgroup,
        contrib_cmp,
        neutral_cmp,
        feature_bins,
        boot,
        diagnostic,
        features,
    )

    # Terminal summary.
    f = overall[overall.dataset == "final_holdout"].iloc[0]
    e = overall[overall.dataset == "external"].iloc[0]
    log("=" * 108)
    log("FIRST HALF 37 — CONCEPT DRIFT / SUBGROUP AUDIT V1")
    log("=" * 108)
    log(f"FINAL    : n={int(f['n'])} | rate={f['positive_rate']:.2%} | ROC={f['roc_auc']:.4f} | PR={f['pr_auc']:.4f} | lift={f['pr_lift_vs_group_rate']:+.4f}")
    log(f"EXTERNAL : n={int(e['n'])} | rate={e['positive_rate']:.2%} | ROC={e['roc_auc']:.4f} | PR={e['pr_auc']:.4f} | lift={e['pr_lift_vs_group_rate']:+.4f}")
    log()
    log("BOOTSTRAP DELTA EXTERNAL - FINAL")
    log(json.dumps(boot, ensure_ascii=False, indent=2))
    log()
    log("FEATURE CONTRIBUTION BREAKDOWN")
    cshow = contrib_cmp[[
        "feature", "final_contribution_auc", "external_contribution_auc",
        "auc_delta_external_minus_final", "signal_breakdown_flag"
    ]].copy()
    log(cshow.to_string(index=False))
    log()
    log("NEUTRALISATION — SUSPECTS")
    nshow = neutral_cmp[[
        "feature", "final_delta_auc_if_neutralized",
        "external_delta_auc_if_neutralized", "concept_drift_suspect_flag"
    ]].copy()
    log(nshow.to_string(index=False))
    log()
    log("=" * 108)
    log("AUDIT CONCEPT DRIFT TERMINE")
    log("=" * 108)
    log(f"Verdict : {diagnostic['verdict']}")
    log(f"Flags   : {', '.join(diagnostic['flags']) if diagnostic['flags'] else 'none'}")
    log(f"Suspects neutralisation : {', '.join(diagnostic['neutralization_suspect_features']) if diagnostic['neutralization_suspect_features'] else 'none'}")
    log(f"Ruptures contribution   : {', '.join(diagnostic['contribution_breakdown_features']) if diagnostic['contribution_breakdown_features'] else 'none'}")
    log(f"Rapport : {report_path}")
    log(f"Sous-groupes : {subgroup_path}")
    log(f"Contributions: {contrib_path}")
    log(f"Neutralisation: {neutral_path}")
    log("IMPORTANT : diagnostic uniquement — modèle inchangé.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
