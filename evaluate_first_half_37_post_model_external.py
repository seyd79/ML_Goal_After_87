#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — POST-MODEL EXTERNAL VALIDATION V1
==================================================

Scores a NEW post-model dataset with the exact frozen model/candidate that was
used for the one-time FINAL HOLDOUT evaluation.

This script does NOT retrain, tune, select features, alter C, or recalibrate.
It is intended only to validate generalization on matches strictly after the
original 1,802-match dataset.

Default external dataset:
    outputs/first_half_37/post_model_external_v1/
    First_Half_37_POST_MODEL_EXTERNAL.csv

Frozen model source:
    outputs/first_half_37/final_holdout_v1/
    first_half_37_final_frozen_before_scoring.json

Pre-registered descriptive probability bands (fixed before this external score):
    <20%, 20-23%, 23-25%, 25-27%, 27-29%, 29-31%, 31-33%,
    33-35%, 35-40%, 40-50%, 50%+

Special zones reported without tuning:
    25-31%  (zone that looked strongest overall in the consumed holdout)
    27-29%  (sub-band of particular interest in the consumed holdout)

IMPORTANT
---------
The original FINAL HOLDOUT has already been consumed. This script never uses it
for tuning. The post-model dataset must remain external-only and must not be
merged into training before this evaluation is reviewed.
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
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

EXTERNAL_DIR = PROJECT_ROOT / "outputs" / "first_half_37" / "post_model_external_v1"
DEFAULT_EXTERNAL = EXTERNAL_DIR / "First_Half_37_POST_MODEL_EXTERNAL.csv"
DEFAULT_FROZEN = (
    PROJECT_ROOT
    / "outputs"
    / "first_half_37"
    / "final_holdout_v1"
    / "first_half_37_final_frozen_before_scoring.json"
)
DEFAULT_MODEL_READY = (
    PROJECT_ROOT
    / "outputs"
    / "first_half_37"
    / "feature_selection"
    / "First_Half_37_MODEL_READY_v1.csv"
)
DEFAULT_OUTPUT_DIR = EXTERNAL_DIR / "evaluation_v1"

TARGET = "goal_after_37_before_ht"
ID_COL = "match_id"
TIME_COL = "kickoff_utc"

FINE_BANDS = [
    -1e-12,
    0.20,
    0.23,
    0.25,
    0.27,
    0.29,
    0.31,
    0.33,
    0.35,
    0.40,
    0.50,
    1.0000001,
]
FINE_LABELS = [
    "<20%",
    "20-23%",
    "23-25%",
    "25-27%",
    "27-29%",
    "29-31%",
    "31-33%",
    "33-35%",
    "35-40%",
    "40-50%",
    "50%+",
]


# =============================================================================
# HELPERS
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


def json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def mixed_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def apply_calibration(artifact: Mapping, raw_probability: np.ndarray) -> np.ndarray:
    cal = artifact.get("calibration", {})
    if isinstance(cal, str):
        kind = cal
        cal = {"kind": kind}
    else:
        kind = str(cal.get("kind", "raw"))

    raw = np.asarray(raw_probability, dtype=float)

    if kind == "raw":
        return raw
    if kind == "platt_temporal_oof":
        model = cal.get("model")
        if model is None:
            raise ValueError("Calibrateur Platt absent de l'artifact figé.")
        return model.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    if kind == "intercept_temporal_oof":
        return sigmoid(logit(raw) + float(cal["delta"]))
    if kind == "prior_corrected":
        pi = float(np.clip(float(cal["train_prevalence"]), 1e-6, 1.0 - 1e-6))
        return sigmoid(logit(raw) + math.log(pi / (1.0 - pi)))

    raise ValueError(f"Calibration inconnue : {kind}")


def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i == n_bins - 1 else (p < hi))
        n = int(mask.sum())
        if n:
            ece += (n / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def metric_bundle(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(np.mean(p)),
    }


def fine_probability_bands(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    cats = pd.cut(
        p,
        bins=FINE_BANDS,
        labels=FINE_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: List[Dict] = []
    for label in FINE_LABELS:
        mask = np.asarray(cats == label)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "probability_band": label,
                "n": 0,
                "positives": 0,
                "observed_rate": np.nan,
                "mean_probability": np.nan,
                "calibration_gap": np.nan,
                "lift_vs_external_rate": np.nan,
            })
            continue
        positives = int(y[mask].sum())
        rate = float(y[mask].mean())
        mean_p = float(p[mask].mean())
        rows.append({
            "probability_band": label,
            "n": n,
            "positives": positives,
            "observed_rate": rate,
            "mean_probability": mean_p,
            "calibration_gap": rate - mean_p,
            "lift_vs_external_rate": rate - float(y.mean()),
        })
    return pd.DataFrame(rows)


def zone_row(name: str, y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> Dict:
    mask = (p >= lo) & (p < hi)
    n = int(mask.sum())
    if n == 0:
        return {
            "zone": name,
            "lower": lo,
            "upper": hi,
            "n": 0,
            "positives": 0,
            "observed_rate": np.nan,
            "mean_probability": np.nan,
            "lift_vs_external_rate": np.nan,
            "calibration_gap": np.nan,
        }
    positives = int(y[mask].sum())
    rate = float(y[mask].mean())
    mean_p = float(p[mask].mean())
    return {
        "zone": name,
        "lower": lo,
        "upper": hi,
        "n": n,
        "positives": positives,
        "observed_rate": rate,
        "mean_probability": mean_p,
        "lift_vs_external_rate": rate - float(y.mean()),
        "calibration_gap": rate - mean_p,
    }


def wilson_interval(positives: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = positives / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_metrics(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(random_state)
    n = len(y)
    values = {"roc_auc": [], "pr_lift": []}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = p[idx]
        values["roc_auc"].append(float(roc_auc_score(yy, pp)))
        values["pr_lift"].append(float(average_precision_score(yy, pp) - yy.mean()))

    out = {}
    for key, vals in values.items():
        a = np.asarray(vals, dtype=float)
        out[key] = {
            "n_boot_valid": int(len(a)),
            "p2_5": float(np.quantile(a, 0.025)),
            "median": float(np.quantile(a, 0.50)),
            "p97_5": float(np.quantile(a, 0.975)),
        }
    return out


def format_pct(v) -> str:
    if v is None or not np.isfinite(float(v)):
        return "-"
    return f"{float(v):.2%}"


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate frozen First_Half37 on post-model external matches.")
    p.add_argument("--external-csv", type=Path, default=DEFAULT_EXTERNAL)
    p.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN)
    p.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--random-seed", type=int, default=87)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.external_csv.exists():
        raise FileNotFoundError(f"Dataset externe introuvable : {args.external_csv}")
    if not args.frozen_manifest.exists():
        raise FileNotFoundError(f"Manifest figé introuvable : {args.frozen_manifest}")

    frozen = json_load(args.frozen_manifest)
    features = list(frozen.get("features", []))
    if not features:
        raise ValueError("Aucune feature figée trouvée dans le manifest final.")

    candidate_path = Path(str(frozen.get("candidate_path", "")))
    if not candidate_path.exists():
        raise FileNotFoundError(f"Artifact candidat figé introuvable : {candidate_path}")

    expected_candidate_hash = str(frozen.get("candidate_sha256", ""))
    actual_candidate_hash = sha256_file(candidate_path)
    if expected_candidate_hash and actual_candidate_hash != expected_candidate_hash:
        raise ValueError(
            "SHA-256 du candidat différent de celui figé avant FINAL HOLDOUT. "
            "Refus de scorer avec un modèle modifié."
        )

    artifact = joblib.load(candidate_path)
    if list(artifact.get("features", [])) != features:
        raise ValueError("Features de l'artifact != features du manifest figé.")

    external = pd.read_csv(args.external_csv, low_memory=False)
    required = {ID_COL, TIME_COL, TARGET, *features}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset externe : {missing}")

    ext = external.copy()
    ext["__kickoff__"] = mixed_to_datetime(ext[TIME_COL])
    if ext["__kickoff__"].isna().any():
        raise ValueError(f"kickoff_utc non parsables : {int(ext['__kickoff__'].isna().sum())}")

    y_series = pd.to_numeric(ext[TARGET], errors="coerce")
    if y_series.isna().any():
        raise ValueError(f"Targets non résolus : {int(y_series.isna().sum())}")
    y = y_series.astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Target invalide : {sorted(set(np.unique(y)))}")

    ids = ext[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"match_id dupliqués : {int(ids.duplicated().sum())}")

    if "external_validation_only" in ext.columns:
        external_flag = pd.to_numeric(ext["external_validation_only"], errors="coerce")
        if not bool((external_flag == 1).all()):
            raise ValueError("external_validation_only doit valoir 1 sur toutes les lignes.")

    # Strict chronology against the original consumed dataset endpoint.
    original_end_text = frozen.get("split", {}).get("holdout_end")
    if original_end_text:
        original_end = pd.to_datetime(original_end_text, utc=True)
        bad = ext["__kickoff__"] <= original_end
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} matchs externes ne sont pas strictement postérieurs au dataset original "
                f"(fin={original_end})."
            )

    overlap_count = None
    if args.model_ready.exists():
        old = pd.read_csv(args.model_ready, usecols=[ID_COL], low_memory=False)
        old_ids = set(old[ID_COL].astype(str))
        overlap_count = int(ids.isin(old_ids).sum())
        if overlap_count:
            raise ValueError(f"Overlap interdit avec MODEL_READY : {overlap_count} match_id")

    X = ext.loc[:, features].apply(pd.to_numeric, errors="coerce")

    pipeline = artifact.get("pipeline")
    if pipeline is None:
        raise ValueError("Pipeline sklearn absent de l'artifact figé.")

    raw_p = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    final_p = np.asarray(apply_calibration(artifact, raw_p), dtype=float)
    final_p = np.clip(final_p, 1e-9, 1.0 - 1e-9)

    external_rate = float(np.mean(y))
    dev_rate = float(artifact.get("development_positive_rate", np.nan))
    if not np.isfinite(dev_rate):
        dev_rate = float(frozen.get("candidate", {}).get("development_positive_rate", np.nan))

    metrics = metric_bundle(y, final_p)
    metrics.update({
        "evaluated_at_utc": utc_now_iso(),
        "rows": int(len(ext)),
        "positives": int(y.sum()),
        "external_positive_rate": external_rate,
        "development_positive_rate": dev_rate,
        "pr_baseline_external_prevalence": external_rate,
        "pr_lift_vs_external_prevalence": float(metrics["pr_auc"] - external_rate),
        "ece_10": ece_score(y, final_p, 10),
        "external_start": ext["__kickoff__"].min().isoformat(),
        "external_end": ext["__kickoff__"].max().isoformat(),
        "candidate_sha256": actual_candidate_hash,
        "external_dataset_sha256": sha256_file(args.external_csv),
        "overlap_with_original": overlap_count,
    })

    # Baseline Brier/LogLoss at frozen DEVELOPMENT prevalence, for comparability.
    if np.isfinite(dev_rate):
        base_p = np.full(len(y), dev_rate, dtype=float)
        metrics["baseline_dev_rate_brier"] = float(brier_score_loss(y, base_p))
        metrics["baseline_dev_rate_log_loss"] = float(log_loss(y, base_p, labels=[0, 1]))
        metrics["brier_improvement_vs_dev_rate_baseline"] = float(
            metrics["baseline_dev_rate_brier"] - metrics["brier"]
        )
        metrics["logloss_improvement_vs_dev_rate_baseline"] = float(
            metrics["baseline_dev_rate_log_loss"] - metrics["log_loss"]
        )

    bands = fine_probability_bands(y, final_p)
    zones = pd.DataFrame([
        zone_row("23-25%", y, final_p, 0.23, 0.25),
        zone_row("25-31%", y, final_p, 0.25, 0.31),
        zone_row("27-29%", y, final_p, 0.27, 0.29),
        zone_row("29-31%", y, final_p, 0.29, 0.31),
    ])

    # Add Wilson 95% CIs for each band / zone with non-zero n.
    for frame in (bands, zones):
        lows, highs = [], []
        for _, row in frame.iterrows():
            lo, hi = wilson_interval(int(row["positives"]), int(row["n"])) if int(row["n"]) else (np.nan, np.nan)
            lows.append(lo)
            highs.append(hi)
        frame["observed_rate_ci95_low"] = lows
        frame["observed_rate_ci95_high"] = highs

    boot = bootstrap_metrics(y, final_p, args.bootstrap, args.random_seed)
    metrics["bootstrap_95"] = boot

    pred_df = pd.DataFrame({
        ID_COL: ids.to_numpy(),
        TIME_COL: ext["__kickoff__"].astype(str).to_numpy(),
        "home_team": ext.get("home_team", pd.Series([""] * len(ext))).astype(str).to_numpy(),
        "away_team": ext.get("away_team", pd.Series([""] * len(ext))).astype(str).to_numpy(),
        TARGET: y,
        "raw_probability": raw_p,
        "final_probability": final_p,
    })

    pred_path = out_dir / "first_half_37_post_model_predictions.csv"
    band_path = out_dir / "first_half_37_post_model_fine_bands.csv"
    zone_path = out_dir / "first_half_37_post_model_zones.csv"
    metric_path = out_dir / "first_half_37_post_model_metrics.json"
    report_path = out_dir / "first_half_37_post_model_report.md"

    pred_df.to_csv(pred_path, index=False)
    bands.to_csv(band_path, index=False)
    zones.to_csv(zone_path, index=False)
    json_dump(metric_path, metrics)

    report_lines = [
        "# First Half 37 — POST-MODEL EXTERNAL VALIDATION V1",
        "",
        "## Frozen pipeline",
        "",
        f"- Features: **{len(features)}**",
        f"- Base model: `{artifact.get('base_model')}`",
        f"- C: **{artifact.get('C')}**",
        f"- Calibration: `{artifact.get('calibration', {}).get('kind') if isinstance(artifact.get('calibration'), dict) else artifact.get('calibration')}`",
        f"- Candidate SHA-256: `{actual_candidate_hash}`",
        "- No retraining / no feature selection / no recalibration.",
        "",
        "## External dataset",
        "",
        f"- Matches: **{len(ext)}**",
        f"- Positives: **{int(y.sum())}/{len(y)} ({external_rate:.2%})**",
        f"- Dates: `{metrics['external_start']}` -> `{metrics['external_end']}`",
        f"- Overlap with original dataset: **{overlap_count if overlap_count is not None else 'not checked'}**",
        "",
        "## Overall metrics",
        "",
        f"- ROC-AUC: **{metrics['roc_auc']:.4f}**",
        f"- PR-AUC: **{metrics['pr_auc']:.4f}** | baseline prevalence={external_rate:.4f}",
        f"- PR lift: **{metrics['pr_lift_vs_external_prevalence']:+.4f}**",
        f"- Brier: **{metrics['brier']:.4f}**",
        f"- LogLoss: **{metrics['log_loss']:.4f}**",
        f"- ECE-10: **{metrics['ece_10']:.4f}**",
        f"- Mean predicted probability: **{metrics['mean_probability']:.2%}**",
        "",
        "## Fine probability bands — pre-registered",
        "",
        "```",
        bands.to_string(index=False),
        "```",
        "",
        "## Zones of interest — pre-registered before this external score",
        "",
        "```",
        zones.to_string(index=False),
        "```",
        "",
        "## Bootstrap 95% (descriptive)",
        "",
        "```json",
        json.dumps(boot, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation rule",
        "",
        "This dataset is an external post-model validation set. Do not alter features, C or calibration based on these results and then continue calling the same sample external validation. Any new tuning starts a new model version and requires another future dataset.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    log("=" * 108)
    log("FIRST HALF 37 — POST-MODEL EXTERNAL VALIDATION V1")
    log("=" * 108)
    log(f"Dataset externe : {args.external_csv}")
    log(f"Matchs uniques  : {len(ext)}")
    log(f"Positifs        : {int(y.sum())}/{len(y)} ({external_rate:.2%})")
    log(f"Dates           : {metrics['external_start']} -> {metrics['external_end']}")
    log(f"Overlap original: {overlap_count if overlap_count is not None else 'non vérifié'}")
    log()
    log(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    log(f"PR-AUC  : {metrics['pr_auc']:.4f} | baseline={external_rate:.4f} | lift={metrics['pr_lift_vs_external_prevalence']:+.4f}")
    log(f"Brier   : {metrics['brier']:.4f}")
    log(f"LogLoss : {metrics['log_loss']:.4f}")
    log(f"ECE-10  : {metrics['ece_10']:.4f}")
    log()
    log("TRANCHES FINES")
    show = bands.copy()
    for col in ["observed_rate", "mean_probability", "calibration_gap", "lift_vs_external_rate"]:
        show[col] = show[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    log(show[["probability_band", "n", "positives", "observed_rate", "mean_probability", "lift_vs_external_rate"]].to_string(index=False))
    log()
    log("ZONES CIBLES")
    zshow = zones.copy()
    for col in ["observed_rate", "mean_probability", "lift_vs_external_rate", "calibration_gap"]:
        zshow[col] = zshow[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    log(zshow[["zone", "n", "positives", "observed_rate", "mean_probability", "lift_vs_external_rate"]].to_string(index=False))
    log()
    log(f"Rapport     : {report_path}")
    log(f"Predictions : {pred_path}")
    log(f"Bands       : {band_path}")
    log(f"Zones       : {zone_path}")
    log("IMPORTANT : aucune modification du modèle n'a été faite.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
