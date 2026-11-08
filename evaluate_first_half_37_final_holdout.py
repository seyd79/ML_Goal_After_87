#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FIRST HALF 37 — ONE-TIME FINAL HOLDOUT EVALUATION (V1)
================================================================================

Purpose
-------
Freeze the winning DEVELOPMENT pipeline from refinement_v1 and open the FINAL
HOLDOUT exactly once.

This script DOES NOT tune anything on the FINAL HOLDOUT. It:
1) reads the frozen refinement manifest;
2) verifies that the internal winner passed the refinement gate;
3) loads the rank-1 DEV-only candidate already refit using DEVELOPMENT only;
4) checks dataset/split/candidate fingerprints and schema consistency;
5) writes a pre-registered "frozen before scoring" manifest;
6) computes one and only one set of predictions on FINAL HOLDOUT;
7) reports ROC-AUC, PR-AUC, Brier, LogLoss, ECE and probability bands;
8) compares Brier/LogLoss to a constant baseline equal to DEVELOPMENT prevalence;
9) writes a permanent lock so accidental reruns do not re-open the holdout.

IMPORTANT
---------
Run only after refinement_v1 reports >=1 pipeline passing the gate.
The command requires --confirm-open-final-holdout. This is intentionally
explicit because the holdout loses its pristine status at first evaluation.

No model is refit on the holdout. A production refit on all data, if desired,
must happen in a separate later step after reviewing this report.
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
DEFAULT_REFINEMENT_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "refinement_v1"
)
DEFAULT_REFINEMENT_MANIFEST = (
    DEFAULT_REFINEMENT_DIR / "first_half_37_refinement_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT / "outputs" / "first_half_37" / "final_holdout_v1"
)

TARGET = "goal_after_37_before_ht"
TIME_COL = "kickoff_utc"
ID_COL = "match_id"

FINAL_LOCK = "FINAL_HOLDOUT_OPENED.lock"
PRE_SCORE_MANIFEST = "first_half_37_final_frozen_before_scoring.json"
FINAL_METRICS = "first_half_37_final_holdout_metrics.json"
FINAL_PREDICTIONS = "first_half_37_final_holdout_predictions.csv"
FINAL_BANDS = "first_half_37_final_probability_bands.csv"
FINAL_RELIABILITY = "first_half_37_final_reliability_bins.csv"
FINAL_REPORT = "first_half_37_final_holdout_report.md"
FINAL_MANIFEST = "first_half_37_final_holdout_manifest.json"

# These criteria are frozen BEFORE looking at holdout predictions.
# The final evaluation is considered a clean PASS only if all four predictive
# metrics beat their appropriate no-skill / constant-prevalence references.
FINAL_GATE = {
    "roc_auc_gt": 0.50,
    "pr_auc_gt_holdout_prevalence": True,
    "brier_better_than_dev_prevalence_baseline": True,
    "logloss_better_than_dev_prevalence_baseline": True,
}

FIXED_BANDS = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0000001]
FIXED_BAND_LABELS = [
    "<20%",
    "20-25%",
    "25-30%",
    "30-35%",
    "35-40%",
    "40-50%",
    "50%+",
]


# =============================================================================
# DATA CLASSES / HELPERS
# =============================================================================

@dataclass
class SplitAudit:
    total_rows: int
    development_rows: int
    holdout_rows: int
    development_start: str
    development_end: str
    holdout_start: str
    holdout_end: str
    development_positive_rate: float
    holdout_positive_rate: Optional[float]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mixed_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


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
            "Refus d'ouvrir le FINAL HOLDOUT."
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


def split_from_frozen_manifest(
    df: pd.DataFrame,
    split_manifest: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitAudit]:
    total_expected = int(split_manifest["total_rows"])
    dev_n = int(split_manifest["development_rows"])
    hold_n = int(split_manifest["holdout_rows"])

    if len(df) != total_expected:
        raise ValueError(
            f"Dataset changé depuis le raffinement : rows={len(df)} "
            f"vs attendu={total_expected}. Refus d'ouvrir le holdout."
        )
    if dev_n + hold_n != len(df):
        raise ValueError("Split manifest incohérent avec le dataset.")

    dev = df.iloc[:dev_n].copy().reset_index(drop=True)
    hold = df.iloc[dev_n:].copy().reset_index(drop=True)

    actual = {
        "development_start": dev["__kickoff__"].iloc[0].isoformat(),
        "development_end": dev["__kickoff__"].iloc[-1].isoformat(),
        "holdout_start": hold["__kickoff__"].iloc[0].isoformat(),
        "holdout_end": hold["__kickoff__"].iloc[-1].isoformat(),
    }
    for key, value in actual.items():
        expected = str(split_manifest[key])
        if value != expected:
            raise ValueError(
                f"Boundary mismatch {key}: actuel={value} attendu={expected}. "
                "Refus d'ouvrir le holdout."
            )

    audit = SplitAudit(
        total_rows=len(df),
        development_rows=len(dev),
        holdout_rows=len(hold),
        development_start=actual["development_start"],
        development_end=actual["development_end"],
        holdout_start=actual["holdout_start"],
        holdout_end=actual["holdout_end"],
        development_positive_rate=float(dev["__target__"].mean()),
        holdout_positive_rate=None,  # intentionally not filled before scoring
    )
    return dev, hold, audit


def metric_bundle(y_true: Sequence[int], prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0 - 1e-9)
    if len(np.unique(y)) < 2:
        raise ValueError("FINAL HOLDOUT sans les deux classes.")
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


def apply_calibration(artifact: Dict, raw_prob: np.ndarray) -> np.ndarray:
    cal = artifact.get("calibration", {})
    if isinstance(cal, str):
        kind = cal
        cal = {"kind": kind}
    else:
        kind = str(cal.get("kind", "raw"))

    raw = np.asarray(raw_prob, dtype=float)
    if kind == "raw":
        return raw
    if kind == "platt_temporal_oof":
        model = cal.get("model")
        if model is None:
            raise ValueError("Calibrateur Platt absent de l'artifact.")
        return model.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    if kind == "intercept_temporal_oof":
        delta = float(cal["delta"])
        return sigmoid(logit(raw) + delta)
    if kind == "prior_corrected":
        pi = float(cal["train_prevalence"])
        pi = float(np.clip(pi, 1e-6, 1 - 1e-6))
        shift = math.log(pi / (1.0 - pi))
        return sigmoid(logit(raw) + shift)
    raise ValueError(f"Calibration inconnue : {kind}")


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": 0,
                "mean_probability": np.nan,
                "observed_rate": np.nan,
                "calibration_gap": np.nan,
            })
            continue
        mp = float(np.mean(p[mask]))
        rate = float(np.mean(y[mask]))
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "mean_probability": mp,
            "observed_rate": rate,
            "calibration_gap": rate - mp,
        })
    return pd.DataFrame(rows)


def probability_bands(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    cat = pd.cut(
        p,
        bins=FIXED_BANDS,
        labels=FIXED_BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: List[Dict] = []
    for label in FIXED_BAND_LABELS:
        mask = np.asarray(cat == label)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "probability_band": label,
                "n": 0,
                "positives": 0,
                "observed_rate": np.nan,
                "mean_probability": np.nan,
                "calibration_gap": np.nan,
            })
            continue
        positives = int(y[mask].sum())
        observed = float(y[mask].mean())
        mean_p = float(p[mask].mean())
        rows.append({
            "probability_band": label,
            "n": n,
            "positives": positives,
            "observed_rate": observed,
            "mean_probability": mean_p,
            "calibration_gap": observed - mean_p,
        })
    return pd.DataFrame(rows)


def bootstrap_deltas(
    y: np.ndarray,
    p: np.ndarray,
    baseline_p: np.ndarray,
    n_boot: int,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    """Descriptive 95% bootstrap CIs. Not used to tune or alter the model."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    vals = {"roc_auc": [], "pr_lift": [], "brier_improvement": [], "logloss_improvement": []}

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = p[idx]
        bb = baseline_p[idx]
        vals["roc_auc"].append(float(roc_auc_score(yy, pp)))
        vals["pr_lift"].append(float(average_precision_score(yy, pp) - yy.mean()))
        vals["brier_improvement"].append(
            float(brier_score_loss(yy, bb) - brier_score_loss(yy, pp))
        )
        vals["logloss_improvement"].append(
            float(log_loss(yy, bb, labels=[0, 1]) - log_loss(yy, pp, labels=[0, 1]))
        )

    out: Dict[str, Dict[str, float]] = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[key] = {
            "n_boot_valid": int(len(a)),
            "p2_5": float(np.quantile(a, 0.025)),
            "median": float(np.quantile(a, 0.50)),
            "p97_5": float(np.quantile(a, 0.975)),
        }
    return out


def find_rank1_candidate(manifest: Dict, winner: Dict) -> Path:
    candidates = manifest.get("saved_dev_only_candidates", [])
    if not candidates:
        raise ValueError("Aucun candidat DEV-only enregistré dans le manifest.")

    exact = []
    for item in candidates:
        if (
            int(item.get("rank", -1)) == 1
            and str(item.get("subset")) == str(winner.get("subset"))
            and str(item.get("base_model")) == str(winner.get("base_model"))
            and str(item.get("calibration")) == str(winner.get("calibration"))
        ):
            exact.append(item)
    if len(exact) != 1:
        raise ValueError(f"Candidat rank-1 exact introuvable/ambigu : {len(exact)}")
    return Path(str(exact[0]["path"]))


def validate_candidate(
    artifact: Dict,
    winner: Dict,
    expected_features: Sequence[str],
    development_rows: int,
) -> None:
    if artifact.get("status") != "DEV_ONLY_NOT_PRODUCTION":
        raise ValueError(f"Status artifact inattendu : {artifact.get('status')}")
    if list(artifact.get("features", [])) != list(expected_features):
        raise ValueError("Features artifact != schéma winner figé.")
    if str(artifact.get("base_model")) != str(winner.get("base_model")):
        raise ValueError("Base model artifact != winner.")

    cal = artifact.get("calibration", {})
    kind = cal.get("kind") if isinstance(cal, dict) else cal
    if str(kind) != str(winner.get("calibration")):
        raise ValueError("Calibration artifact != winner.")
    if int(artifact.get("development_rows", -1)) != int(development_rows):
        raise ValueError("Artifact refit sur un nombre de lignes DEVELOPMENT différent.")
    if artifact.get("pipeline") is None:
        raise ValueError("Pipeline sklearn absent de l'artifact.")


def final_gate_decision(metrics: Dict) -> Tuple[str, Dict[str, bool]]:
    checks = {
        "roc_auc_gt_0_50": bool(metrics["roc_auc"] > 0.50),
        "pr_auc_gt_holdout_prevalence": bool(
            metrics["pr_auc"] > metrics["holdout_positive_rate"]
        ),
        "brier_better_than_dev_prevalence_baseline": bool(
            metrics["brier_improvement_vs_baseline"] > 0
        ),
        "logloss_better_than_dev_prevalence_baseline": bool(
            metrics["logloss_improvement_vs_baseline"] > 0
        ),
    }
    if all(checks.values()):
        return "FINAL_PASS", checks
    if checks["roc_auc_gt_0_50"] and checks["pr_auc_gt_holdout_prevalence"]:
        return "FINAL_MIXED_DISCRIMINATION_ONLY", checks
    return "FINAL_FAIL", checks


def write_report(
    path: Path,
    frozen: Dict,
    metrics: Dict,
    checks: Dict[str, bool],
    decision: str,
    bands: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstrap: Dict,
) -> None:
    w = frozen["winner"]
    lines: List[str] = [
        "# First Half 37 — FINAL HOLDOUT V1",
        "",
        "## Pipeline figé avant ouverture",
        "",
        f"- Subset : `{w['subset']}`",
        f"- Features : **{int(w['n_features'])}**",
        f"- Base model : `{w['base_model']}`",
        f"- Calibration : `{w['calibration']}`",
        f"- C final DEVELOPMENT : **{frozen['candidate']['C']}**",
        f"- DEVELOPMENT : **{frozen['split']['development_rows']}** matchs",
        f"- FINAL HOLDOUT : **{frozen['split']['holdout_rows']}** matchs",
        f"- Holdout dates : `{frozen['split']['holdout_start']}` -> `{frozen['split']['holdout_end']}`",
        "",
        "## Résultat FINAL HOLDOUT",
        "",
        f"- Positifs : **{metrics['holdout_positives']}/{metrics['holdout_rows']}** ({metrics['holdout_positive_rate']:.2%})",
        f"- ROC-AUC : **{metrics['roc_auc']:.4f}**",
        f"- PR-AUC : **{metrics['pr_auc']:.4f}** (baseline/no-skill PR={metrics['holdout_positive_rate']:.4f})",
        f"- PR lift : **{metrics['pr_lift_vs_holdout_prevalence']:+.4f}**",
        f"- Brier : **{metrics['brier']:.4f}** vs baseline **{metrics['baseline_brier']:.4f}** (improvement {metrics['brier_improvement_vs_baseline']:+.4f})",
        f"- LogLoss : **{metrics['log_loss']:.4f}** vs baseline **{metrics['baseline_log_loss']:.4f}** (improvement {metrics['logloss_improvement_vs_baseline']:+.4f})",
        f"- ECE-10 : **{metrics['ece_10']:.4f}**",
        f"- Probabilité moyenne modèle : **{metrics['mean_probability']:.2%}**",
        f"- Taux positif DEVELOPMENT utilisé par la baseline : **{metrics['development_positive_rate']:.2%}**",
        "",
        "## Gate final pré-enregistré",
        "",
        f"**Décision : `{decision}`**",
        "",
    ]
    for key, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — `{key}`")

    lines += [
        "",
        "## Bandes de probabilité (descriptif, aucun tuning)",
        "",
        "```",
        bands.to_string(index=False),
        "```",
        "",
        "## Reliability bins",
        "",
        "```",
        reliability.to_string(index=False),
        "```",
        "",
        "## Bootstrap descriptif 95%",
        "",
        "```json",
        json.dumps(bootstrap, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Règle après ouverture",
        "",
        "Le FINAL HOLDOUT est maintenant consommé. Il ne doit plus être utilisé pour modifier features, C, calibration ou choisir un autre pipeline. Si le pipeline est accepté, la prochaine étape peut être un refit production séparé. S'il échoue, toute nouvelle itération doit être considérée comme une nouvelle expérience sans prétendre que ce holdout est encore pristine.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FH37 one-time final holdout evaluation")
    p.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY)
    p.add_argument("--refinement-manifest", type=Path, default=DEFAULT_REFINEMENT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--random-state", type=int, default=87)
    p.add_argument(
        "--confirm-open-final-holdout",
        action="store_true",
        help="Required explicit acknowledgement that this permanently opens the final holdout.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / FINAL_LOCK

    if lock_path.exists():
        log("=" * 108)
        log("FINAL HOLDOUT DEJA OUVERT — AUCUNE NOUVELLE PREDICTION CALCULEE")
        log("=" * 108)
        log(f"Lock : {lock_path}")
        metrics_path = out_dir / FINAL_METRICS
        report_path = out_dir / FINAL_REPORT
        if metrics_path.exists():
            try:
                m = json_load(metrics_path)
                log(
                    f"Résultat existant : decision={m.get('decision')} | "
                    f"ROC={m.get('roc_auc', float('nan')):.4f} | "
                    f"PR={m.get('pr_auc', float('nan')):.4f} | "
                    f"Brier={m.get('brier', float('nan')):.4f}"
                )
            except Exception:
                pass
        log(f"Rapport existant : {report_path}")
        return 0

    if not args.confirm_open_final_holdout:
        log("REFUS : le FINAL HOLDOUT est encore pristine.")
        log("Relance avec --confirm-open-final-holdout pour l'ouvrir UNE SEULE FOIS.")
        return 2

    if not args.refinement_manifest.exists():
        raise FileNotFoundError(f"Manifest refinement introuvable : {args.refinement_manifest}")

    manifest = json_load(args.refinement_manifest)
    if bool(manifest.get("final_holdout_used", False)):
        raise ValueError("Le refinement manifest indique déjà final_holdout_used=true.")
    if int(manifest.get("pipelines_passing_gate", 0)) < 1:
        raise ValueError("Aucun pipeline n'a passé le gate de raffinement. Refus d'ouvrir le holdout.")

    winner = manifest.get("winner_internal")
    if not winner:
        raise ValueError("winner_internal absent du manifest.")
    if int(winner.get("rank", -1)) != 1 or int(winner.get("passes_refinement_gate", 0)) != 1:
        raise ValueError("Le rank-1 n'est pas un winner gate=PASS. Refus d'ouvrir le holdout.")

    subsets = manifest.get("subsets", {})
    subset_name = str(winner["subset"])
    if subset_name not in subsets:
        raise ValueError(f"Subset winner absent du manifest : {subset_name}")
    features = list(subsets[subset_name])
    if len(features) != int(winner["n_features"]):
        raise ValueError("Nombre de features winner incohérent.")

    candidate_path = find_rank1_candidate(manifest, winner)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidat rank-1 introuvable : {candidate_path}\n"
            "Ne reconstruis pas un nouveau modèle avant l'ouverture; utilise l'artifact DEV-only figé."
        )

    df = load_dataset(args.model_ready)
    dev, holdout, split_audit = split_from_frozen_manifest(df, manifest["split"])
    artifact = joblib.load(candidate_path)
    validate_candidate(artifact, winner, features, len(dev))

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features manquantes dans MODEL_READY : {missing_features}")

    # Freeze everything BEFORE any holdout target is used in a metric.
    dataset_hash = sha256_file(args.model_ready)
    candidate_hash = sha256_file(candidate_path)
    feature_hash = sha256_text("\n".join(features))
    frozen = {
        "schema_version": "first_half_37_final_holdout_v1",
        "frozen_at_utc": utc_now_iso(),
        "dataset_path": str(args.model_ready),
        "dataset_sha256": dataset_hash,
        "refinement_manifest_path": str(args.refinement_manifest),
        "refinement_manifest_sha256": sha256_file(args.refinement_manifest),
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "feature_list_sha256": feature_hash,
        "winner": winner,
        "features": features,
        "candidate": {
            "base_model": artifact.get("base_model"),
            "C": float(artifact.get("C")),
            "calibration": (
                artifact.get("calibration", {}).get("kind")
                if isinstance(artifact.get("calibration"), dict)
                else artifact.get("calibration")
            ),
            "development_rows": int(artifact.get("development_rows")),
            "development_positive_rate": float(artifact.get("development_positive_rate")),
        },
        "split": asdict(split_audit),
        "final_gate_pre_registered": FINAL_GATE,
        "holdout_predictions_computed": False,
    }
    json_dump(out_dir / PRE_SCORE_MANIFEST, frozen)

    log("=" * 108)
    log("FIRST HALF 37 — ONE-TIME FINAL HOLDOUT EVALUATION V1")
    log("=" * 108)
    log(f"Winner figé : {subset_name} + {winner['base_model']} + {winner['calibration']}")
    log(f"Features     : {len(features)}")
    log(f"C final DEV  : {float(artifact['C']):g}")
    log(f"DEVELOPMENT  : {len(dev)} | rate={dev['__target__'].mean():.2%}")
    log(f"FINAL HOLDOUT: {len(holdout)} | {split_audit.holdout_start} -> {split_audit.holdout_end}")
    log("Configuration + fingerprints figés AVANT scoring.")
    log()

    # -------------------------------------------------------------------------
    # FIRST AND ONLY HOLDOUT SCORING
    # -------------------------------------------------------------------------
    X_hold = holdout.loc[:, features].apply(pd.to_numeric, errors="coerce")
    y_hold = holdout["__target__"].to_numpy(dtype=int)

    pipeline = artifact["pipeline"]
    raw_p = np.asarray(pipeline.predict_proba(X_hold)[:, 1], dtype=float)
    p = np.asarray(apply_calibration(artifact, raw_p), dtype=float)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)

    dev_rate = float(dev["__target__"].mean())
    baseline_p = np.full(len(y_hold), dev_rate, dtype=float)

    met = metric_bundle(y_hold, p)
    base = metric_bundle(y_hold, baseline_p)
    hold_rate = float(np.mean(y_hold))

    metrics = {
        "evaluated_at_utc": utc_now_iso(),
        "holdout_rows": int(len(y_hold)),
        "holdout_positives": int(y_hold.sum()),
        "holdout_positive_rate": hold_rate,
        "development_positive_rate": dev_rate,
        **met,
        "ece_10": ece_score(y_hold, p, 10),
        "baseline_probability": dev_rate,
        "baseline_pr_auc": hold_rate,
        "baseline_brier": base["brier"],
        "baseline_log_loss": base["log_loss"],
        "pr_lift_vs_holdout_prevalence": float(met["pr_auc"] - hold_rate),
        "brier_improvement_vs_baseline": float(base["brier"] - met["brier"]),
        "logloss_improvement_vs_baseline": float(base["log_loss"] - met["log_loss"]),
    }
    decision, checks = final_gate_decision(metrics)
    metrics["decision"] = decision
    metrics["gate_checks"] = checks

    pred_df = pd.DataFrame({
        ID_COL: holdout[ID_COL].astype(str).to_numpy(),
        TIME_COL: holdout["__kickoff__"].astype(str).to_numpy(),
        TARGET: y_hold,
        "raw_probability": raw_p,
        "final_probability": p,
        "baseline_probability_dev_rate": baseline_p,
    })
    pred_df.to_csv(out_dir / FINAL_PREDICTIONS, index=False)

    bands = probability_bands(y_hold, p)
    bands.to_csv(out_dir / FINAL_BANDS, index=False)
    reliability = reliability_bins(y_hold, p, 10)
    reliability.to_csv(out_dir / FINAL_RELIABILITY, index=False)

    boot = bootstrap_deltas(
        y_hold,
        p,
        baseline_p,
        n_boot=max(200, int(args.bootstrap)),
        random_state=int(args.random_state),
    )
    metrics["bootstrap_95"] = boot
    json_dump(out_dir / FINAL_METRICS, metrics)

    frozen_after = dict(frozen)
    frozen_after["holdout_predictions_computed"] = True
    frozen_after["holdout_evaluated_at_utc"] = metrics["evaluated_at_utc"]
    frozen_after["decision"] = decision
    frozen_after["metrics_path"] = str(out_dir / FINAL_METRICS)
    frozen_after["predictions_path"] = str(out_dir / FINAL_PREDICTIONS)
    json_dump(out_dir / FINAL_MANIFEST, frozen_after)

    report_path = out_dir / FINAL_REPORT
    write_report(report_path, frozen, metrics, checks, decision, bands, reliability, boot)

    # Lock is written LAST, after all final outputs have been persisted.
    lock_payload = {
        "opened_at_utc": metrics["evaluated_at_utc"],
        "decision": decision,
        "dataset_sha256": dataset_hash,
        "candidate_sha256": candidate_hash,
        "report": str(report_path),
        "warning": "FINAL HOLDOUT consumed. Do not use it for further tuning/selection.",
    }
    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    log("=" * 108)
    log("FINAL HOLDOUT OUVERT — EVALUATION UNIQUE TERMINEE")
    log("=" * 108)
    log(f"Holdout positives : {int(y_hold.sum())}/{len(y_hold)} ({hold_rate:.2%})")
    log(f"ROC-AUC           : {met['roc_auc']:.4f}")
    log(f"PR-AUC            : {met['pr_auc']:.4f} | baseline={hold_rate:.4f} | lift={metrics['pr_lift_vs_holdout_prevalence']:+.4f}")
    log(f"Brier             : {met['brier']:.4f} | baseline={base['brier']:.4f} | imp={metrics['brier_improvement_vs_baseline']:+.4f}")
    log(f"LogLoss           : {met['log_loss']:.4f} | baseline={base['log_loss']:.4f} | imp={metrics['logloss_improvement_vs_baseline']:+.4f}")
    log(f"ECE-10            : {metrics['ece_10']:.4f}")
    log(f"DECISION          : {decision}")
    log(f"Rapport           : {report_path}")
    log(f"Predictions       : {out_dir / FINAL_PREDICTIONS}")
    log(f"Lock              : {lock_path}")
    log("IMPORTANT : ce FINAL HOLDOUT est maintenant consommé et ne doit plus servir au tuning.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par utilisateur.", file=sys.stderr)
        raise SystemExit(130)
