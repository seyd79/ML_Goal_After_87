#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 FINAL HOLDOUT BLIND EVALUATOR

Evaluation ONE-SHOT du pipeline V2 figé.

AVANT tout scoring, le script vérifie:
- FREEZE_LOCK présent et conforme;
- SHA256 du modèle = SHA pré-enregistré;
- SHA256 schéma/manifest = valeurs du FREEZE_LOCK;
- bundle joblib = TOP_8 + Logistic L2 C=0.01 + alpha gelé + OOF150;
- manifest de collecte = fenêtre 2026-09-20 -> 2026-10-03, collecte complète;
- SHA256 du CSV holdout = SHA enregistré par le collecteur;
- aucune collision match_id avec les 2 105 matchs DEVELOPMENT;
- aucun match hors fenêtre;
- target résolu 0/1;
- aucune modification du pipeline après gel.

Verdict pré-enregistré (pipeline CALIBRE uniquement):
PASS si les 4 conditions sont vraies:
  1. ROC-AUC > 0.50
  2. PR-AUC > prévalence du holdout
  3. Brier < baseline constante gelée (610/2105)
  4. LogLoss < baseline constante gelée (610/2105)

Les bootstrap/subgroups/bands sont descriptifs et ne modifient jamais le verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_frozen"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FREEZE_LOCK.json"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FROZEN.joblib"
SCHEMA_PATH = FROZEN_DIR / "first_half_37_v2_frozen_schema.json"
FREEZE_MANIFEST_PATH = FROZEN_DIR / "first_half_37_v2_freeze_manifest.json"

HOLDOUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_final_holdout"
HOLDOUT_PATH = HOLDOUT_DIR / "First_Half_37_V2_FINAL_HOLDOUT.csv"
COLLECTION_MANIFEST_PATH = (
    HOLDOUT_DIR / "first_half_37_v2_final_holdout_collection_manifest.json"
)

DEVELOPMENT_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "First_Half_37_V2_DEVELOPMENT.csv"
)

DEFAULT_OUTPUT_DIR = HOLDOUT_DIR / "evaluation"

START_UTC = pd.Timestamp("2026-09-20 00:00:00", tz="UTC")
END_UTC = pd.Timestamp("2026-10-03 23:59:59", tz="UTC")

EXPECTED_MODEL_SHA256 = (
    "defcbfadde1a7d15528b74ec3ddd38ca44a415fbc64ade7b73cc09237e09dcba"
)
EXPECTED_MODEL_NAME = "FIRST_HALF37_V2_TOP8_LOGISTIC_L2_OOF150"
EXPECTED_C = 0.01
EXPECTED_ALPHA = -0.173842194905
EXPECTED_WINDOW = 150
EXPECTED_FEATURES = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]
TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"

FROZEN_DEV_ROWS = 2105
FROZEN_DEV_POSITIVES = 610
FROZEN_BASE_RATE = FROZEN_DEV_POSITIVES / FROZEN_DEV_ROWS
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 87
EPS = 1e-10


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON invalide: {path}")
    return value


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


def calibrated_probability(raw_p, alpha: float):
    return sigmoid(logit(raw_p) + alpha)


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


def ece_binary(y, p, n_bins=10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        result += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def baseline_metrics(y: np.ndarray, p0: float) -> Tuple[float, float]:
    b = np.full(len(y), p0, dtype=float)
    return (
        float(brier_score_loss(y, b)),
        float(log_loss(y, np.clip(b, EPS, 1-EPS))),
    )


def verify_freeze_before_scoring() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (FREEZE_LOCK_PATH, MODEL_PATH, SCHEMA_PATH, FREEZE_MANIFEST_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Artefact gelé manquant: {p}")

    lock = load_json(FREEZE_LOCK_PATH)

    if lock.get("status") != "LOCKED_BEFORE_NEW_HOLDOUT":
        raise RuntimeError("FREEZE_LOCK status invalide.")
    if lock.get("model_name") != EXPECTED_MODEL_NAME:
        raise RuntimeError("FREEZE_LOCK model_name invalide.")
    if lock.get("features") != EXPECTED_FEATURES:
        raise RuntimeError("FREEZE_LOCK TOP_8 invalide.")
    if abs(float(lock.get("C")) - EXPECTED_C) > 1e-15:
        raise RuntimeError("FREEZE_LOCK C invalide.")
    if int(lock.get("calibration_window")) != EXPECTED_WINDOW:
        raise RuntimeError("FREEZE_LOCK fenêtre calibration invalide.")
    if abs(float(lock.get("calibration_alpha")) - EXPECTED_ALPHA) > 1e-12:
        raise RuntimeError("FREEZE_LOCK alpha invalide.")

    actual_model_sha = sha256_file(MODEL_PATH)
    if actual_model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "MODELE MODIFIE DEPUIS LE GEL.\n"
            f"Attendu: {EXPECTED_MODEL_SHA256}\n"
            f"Trouvé : {actual_model_sha}"
        )
    if lock.get("model_sha256") != actual_model_sha:
        raise RuntimeError("FREEZE_LOCK model_sha256 incohérent.")

    schema_sha = sha256_file(SCHEMA_PATH)
    if lock.get("schema_sha256") != schema_sha:
        raise RuntimeError("Schéma gelé modifié.")

    freeze_manifest_sha = sha256_file(FREEZE_MANIFEST_PATH)
    if lock.get("manifest_sha256") != freeze_manifest_sha:
        raise RuntimeError("Manifest gelé modifié.")

    freeze_manifest = load_json(FREEZE_MANIFEST_PATH)

    dev = freeze_manifest.get("development", {})
    if int(dev.get("rows", -1)) != FROZEN_DEV_ROWS:
        raise RuntimeError("Manifest gelé: nombre de lignes DEVELOPMENT invalide.")
    if int(dev.get("positives", -1)) != FROZEN_DEV_POSITIVES:
        raise RuntimeError("Manifest gelé: nombre de positifs DEVELOPMENT invalide.")
    if abs(float(dev.get("positive_rate")) - FROZEN_BASE_RATE) > 1e-12:
        raise RuntimeError("Manifest gelé: base rate DEVELOPMENT invalide.")

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, Mapping):
        raise RuntimeError("Bundle joblib invalide.")
    if bundle.get("model_name") != EXPECTED_MODEL_NAME:
        raise RuntimeError("Bundle model_name invalide.")
    if list(bundle.get("features", [])) != EXPECTED_FEATURES:
        raise RuntimeError("Bundle features invalides.")
    if abs(float(bundle.get("C")) - EXPECTED_C) > 1e-15:
        raise RuntimeError("Bundle C invalide.")

    cal = bundle.get("calibration", {})
    if cal.get("type") != "intercept_only":
        raise RuntimeError("Bundle calibration type invalide.")
    if int(cal.get("window")) != EXPECTED_WINDOW:
        raise RuntimeError("Bundle calibration window invalide.")
    if abs(float(cal.get("alpha")) - EXPECTED_ALPHA) > 1e-12:
        raise RuntimeError("Bundle alpha invalide.")

    return lock, freeze_manifest, dict(bundle)


def verify_collection_before_scoring() -> Tuple[Dict[str, Any], str]:
    if not COLLECTION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Manifest de collecte absent. Exécuter d'abord "
            "collect_first_half_37_v2_final_holdout.py"
        )
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(HOLDOUT_PATH)

    m = load_json(COLLECTION_MANIFEST_PATH)

    if m.get("blind_collection") is not True:
        raise RuntimeError("Manifest collecte: blind_collection != true.")
    if m.get("scoring_performed") is not False:
        raise RuntimeError("Manifest collecte indique qu'un scoring a déjà été effectué.")

    window = m.get("window", {})
    if pd.Timestamp(window.get("start_utc")) != START_UTC:
        raise RuntimeError("Fenêtre START différente du protocole pré-enregistré.")
    if pd.Timestamp(window.get("end_utc")) != END_UTC:
        raise RuntimeError("Fenêtre END différente du protocole pré-enregistré.")

    collection = m.get("collection", {})
    if collection.get("collection_complete") is not True:
        raise RuntimeError(
            "COLLECTION INCOMPLETE: l'évaluation aveugle est refusée."
        )
    if collection.get("limited_test_run") is True:
        raise RuntimeError("Dataset issu d'un --limit: évaluation finale refusée.")

    listing = m.get("listing", {})
    if listing.get("listing_complete") is not True:
        raise RuntimeError("Listing API incomplet: évaluation finale refusée.")

    freeze = m.get("freeze", {})
    if freeze.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Collecte liée à un autre SHA modèle.")
    if freeze.get("features") != EXPECTED_FEATURES:
        raise RuntimeError("Collecte liée à un autre TOP_8.")
    if abs(float(freeze.get("alpha")) - EXPECTED_ALPHA) > 1e-12:
        raise RuntimeError("Collecte liée à un autre alpha.")

    listing_file = Path(m.get("listing", {}).get("listed_matches_file", ""))
    expected_listing_sha = m.get("listing", {}).get("listed_matches_sha256")
    if not listing_file.exists():
        raise RuntimeError("Fichier de listing pré-collecte absent.")
    if expected_listing_sha and sha256_file(listing_file) != expected_listing_sha:
        raise RuntimeError("Fichier de listing modifié après collecte.")

    actual_dataset_sha = sha256_file(HOLDOUT_PATH)
    expected_dataset_sha = m.get("files", {}).get("dataset_sha256")
    if actual_dataset_sha != expected_dataset_sha:
        raise RuntimeError(
            "CSV HOLDOUT MODIFIE APRES COLLECTE.\n"
            f"Manifest: {expected_dataset_sha}\n"
            f"Actuel  : {actual_dataset_sha}"
        )

    return m, actual_dataset_sha


def load_and_validate_holdout() -> pd.DataFrame:
    df = pd.read_csv(HOLDOUT_PATH, low_memory=False)

    required = set(EXPECTED_FEATURES + [TARGET, DATE_COL, ID_COL])
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Holdout: colonnes requises absentes: {missing}")

    if len(df) == 0:
        raise RuntimeError("Holdout vide.")

    if df[ID_COL].isna().any():
        raise RuntimeError("Holdout contient match_id manquant.")
    if df[ID_COL].astype(str).duplicated().any():
        raise RuntimeError("Holdout contient des match_id dupliqués.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
    if df[DATE_COL].isna().any():
        raise RuntimeError("Holdout contient kickoff_utc non parsable.")

    if (df[DATE_COL] < START_UTC).any() or (df[DATE_COL] > END_UTC).any():
        bad = df.loc[
            (df[DATE_COL] < START_UTC) | (df[DATE_COL] > END_UTC),
            [ID_COL, DATE_COL],
        ].head(10)
        raise RuntimeError(f"Holdout contient des matchs hors fenêtre:\n{bad}")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise RuntimeError("Holdout contient target non résolu.")
    if not set(y.unique()).issubset({0, 1, 0.0, 1.0}):
        raise RuntimeError("Holdout target contient des valeurs autres que 0/1.")
    df[TARGET] = y.astype(int)

    if "target_status" in df.columns:
        bad = ~df["target_status"].astype(str).eq("RESOLVED")
        if bad.any():
            raise RuntimeError("Holdout contient target_status != RESOLVED.")

    if "freeze_model_sha256" in df.columns:
        vals = set(df["freeze_model_sha256"].dropna().astype(str))
        if vals != {EXPECTED_MODEL_SHA256}:
            raise RuntimeError("Certaines lignes holdout sont liées à un autre modèle gelé.")

    # Vérification overlap après toutes les vérifications de SHA/gel.
    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"DEVELOPMENT V2 requis pour overlap check: {DEVELOPMENT_PATH}"
        )

    # Le fichier DEVELOPMENT utilisé pour l'overlap doit être exactement celui
    # qui existait au moment du gel.
    freeze_lock = load_json(FREEZE_LOCK_PATH)
    expected_dev_sha = freeze_lock.get("development_sha256")
    actual_dev_sha = sha256_file(DEVELOPMENT_PATH)
    if expected_dev_sha and actual_dev_sha != expected_dev_sha:
        raise RuntimeError(
            "DEVELOPMENT V2 modifié depuis le gel; overlap check refusé.\n"
            f"FREEZE_LOCK: {expected_dev_sha}\n"
            f"Actuel     : {actual_dev_sha}"
        )

    dev = pd.read_csv(DEVELOPMENT_PATH, usecols=[ID_COL], low_memory=False)
    dev_ids = set(dev[ID_COL].dropna().astype(str))
    holdout_ids = set(df[ID_COL].astype(str))
    overlap = sorted(dev_ids & holdout_ids)
    if overlap:
        raise RuntimeError(
            f"LEAKAGE: {len(overlap)} match_id du holdout existent dans DEVELOPMENT. "
            f"Exemples: {overlap[:10]}"
        )

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def metric_row(y, p, label: str) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    prevalence = float(y.mean())
    b0, ll0 = baseline_metrics(y, FROZEN_BASE_RATE)

    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, np.clip(p, EPS, 1-EPS)))

    return {
        "pipeline": label,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prevalence,
        "mean_probability": float(p.mean()),
        "mean_probability_minus_actual": float(p.mean() - prevalence),
        "roc_auc": safe_roc(y, p),
        "pr_auc": safe_pr(y, p),
        "pr_lift_vs_holdout_prevalence": safe_pr(y, p) - prevalence,
        "brier": brier,
        "baseline_probability_frozen_dev": FROZEN_BASE_RATE,
        "baseline_brier": b0,
        "brier_improvement": b0 - brier,
        "log_loss": ll,
        "baseline_logloss": ll0,
        "logloss_improvement": ll0 - ll,
        "ece_10": ece_binary(y, p, 10),
    }


def bootstrap_metrics(y, p, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    vals = {
        "roc_auc": [],
        "pr_auc": [],
        "pr_lift": [],
        "brier_improvement": [],
        "logloss_improvement": [],
        "mean_probability_minus_actual": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]
        if len(np.unique(yy)) < 2:
            continue

        prevalence = float(yy.mean())
        b0, ll0 = baseline_metrics(yy, FROZEN_BASE_RATE)
        b = float(brier_score_loss(yy, pp))
        ll = float(log_loss(yy, np.clip(pp, EPS, 1-EPS)))

        vals["roc_auc"].append(safe_roc(yy, pp))
        vals["pr_auc"].append(safe_pr(yy, pp))
        vals["pr_lift"].append(safe_pr(yy, pp) - prevalence)
        vals["brier_improvement"].append(b0 - b)
        vals["logloss_improvement"].append(ll0 - ll)
        vals["mean_probability_minus_actual"].append(float(pp.mean() - prevalence))

    out = {}
    for name, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        out[name] = {
            "n_valid": int(len(a)),
            "p2_5": float(np.percentile(a, 2.5)) if len(a) else None,
            "median": float(np.percentile(a, 50)) if len(a) else None,
            "p97_5": float(np.percentile(a, 97.5)) if len(a) else None,
            "probability_above_zero": float((a > 0).mean()) if len(a) else None,
        }
    return out


def subgroup_table(df: pd.DataFrame, p: np.ndarray, group_col: str) -> pd.DataFrame:
    temp = df.copy()
    temp["_p"] = p
    rows = []
    for group, g in temp.groupby(group_col, dropna=False):
        y = g[TARGET].astype(int).to_numpy()
        pp = g["_p"].to_numpy()
        r = metric_row(y, pp, "frozen_calibrated_v2")
        r[group_col] = group
        rows.append(r)
    return pd.DataFrame(rows)


def daily_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    temp = df.copy()
    temp["_p"] = p
    temp["_day"] = temp[DATE_COL].dt.strftime("%Y-%m-%d")
    rows = []
    for day, g in temp.groupby("_day", sort=True):
        r = metric_row(
            g[TARGET].astype(int).to_numpy(),
            g["_p"].to_numpy(),
            "frozen_calibrated_v2",
        )
        r["date"] = day
        rows.append(r)
    return pd.DataFrame(rows)


def probability_bands(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    edges = [0, .20, .25, .30, .35, .40, 1.000001]
    labels = ["<20%", "20-25%", "25-30%", "30-35%", "35-40%", "40%+"]
    temp = pd.DataFrame({
        "y": df[TARGET].astype(int).to_numpy(),
        "p": p,
        "band": pd.cut(
            p,
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype(str),
    })
    rows = []
    for label in labels:
        g = temp[temp["band"] == label]
        if g.empty:
            continue
        rows.append({
            "probability_band": label,
            "n": int(len(g)),
            "positives": int(g["y"].sum()),
            "observed_rate": float(g["y"].mean()),
            "mean_probability": float(g["p"].mean()),
            "observed_minus_predicted": float(g["y"].mean() - g["p"].mean()),
        })
    return pd.DataFrame(rows)


def score_state(df: pd.DataFrame) -> Optional[pd.Series]:
    if "score_37_home" not in df.columns or "score_37_away" not in df.columns:
        return None
    h = pd.to_numeric(df["score_37_home"], errors="coerce")
    a = pd.to_numeric(df["score_37_away"], errors="coerce")

    values = []
    for hh, aa in zip(h, a):
        if pd.isna(hh) or pd.isna(aa):
            values.append("missing")
        elif hh == 0 and aa == 0:
            values.append("0-0")
        elif hh == aa:
            values.append("draw_with_goals")
        elif hh > aa:
            values.append("home_lead")
        else:
            values.append("away_lead")
    return pd.Series(values, index=df.index, name="score_state_37")


def write_report(
    path: Path,
    calibrated: Dict[str, Any],
    raw: Dict[str, Any],
    verdict: str,
    conditions: Dict[str, bool],
    bootstrap: Dict[str, Any],
    model_sha: str,
    dataset_sha: str,
):
    lines = []
    lines.append("# First Half 37 — V2 FINAL EXTERNAL HOLDOUT")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"- Model SHA256: `{model_sha}`")
    lines.append(f"- Holdout SHA256: `{dataset_sha}`")
    lines.append("- FREEZE_LOCK: **PASS**")
    lines.append("- Holdout overlap with DEVELOPMENT: **0**")
    lines.append("- Pipeline: `TOP_8 + Logistic L2 C=0.01 + intercept OOF150`")
    lines.append("")
    lines.append("## Verdict pré-enregistré")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    for name, ok in conditions.items():
        lines.append(f"- `{name}`: **{'PASS' if ok else 'FAIL'}**")
    lines.append("")
    lines.append("## Pipeline gelé calibré")
    lines.append("")
    for key, value in calibrated.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## RAW (diagnostic seulement)")
    lines.append("")
    for key, value in raw.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Bootstrap 95% (descriptif)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(bootstrap, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append(
        "Le holdout est désormais consommé. Aucun changement de feature, C, "
        "calibration, alpha, fenêtre ou prétraitement ne peut transformer ce "
        "même holdout en nouvelle validation pristine."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation aveugle ONE-SHOT du FINAL HOLDOUT First Half 37 V2."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--resume-opened",
        action="store_true",
        help=(
            "Uniquement après crash technique: autorise la reprise si le holdout/model "
            "ont exactement les mêmes SHA que le fichier OPENED."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opened_path = output_dir / "FIRST_HALF37_V2_HOLDOUT_OPENED.json"
    final_lock_path = output_dir / "FIRST_HALF37_V2_HOLDOUT_EVALUATION_LOCK.json"

    if final_lock_path.exists():
        lock = load_json(final_lock_path)
        raise RuntimeError(
            "CE HOLDOUT A DEJA ETE EVALUE.\n"
            f"Verdict existant: {lock.get('verdict')}\n"
            f"Lock: {final_lock_path}"
        )

    print("=" * 120)
    print("FIRST HALF 37 — V2 FINAL HOLDOUT BLIND EVALUATOR")
    print("=" * 120)
    print("Phase 1 : intégrité du gel et SHA256 — AVANT SCORING")
    print()

    freeze_lock, freeze_manifest, bundle = verify_freeze_before_scoring()
    collection_manifest, dataset_sha = verify_collection_before_scoring()

    print("FREEZE_LOCK                : PASS")
    print(f"MODEL SHA256               : {EXPECTED_MODEL_SHA256}")
    print("SCHEMA / MANIFEST SHA      : PASS")
    print("COLLECTION MANIFEST        : PASS")
    print(f"HOLDOUT DATASET SHA256     : {dataset_sha}")
    print()

    if opened_path.exists():
        opened = load_json(opened_path)
        if not args.resume_opened:
            raise RuntimeError(
                "Le holdout a déjà été OUVERT lors d'une exécution précédente.\n"
                "Si cette exécution a seulement crashé techniquement et qu'aucun choix "
                "n'a été modifié, relancer avec --resume-opened."
            )
        if opened.get("model_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("OPENED lock lié à un autre modèle.")
        if opened.get("holdout_sha256") != dataset_sha:
            raise RuntimeError("OPENED lock lié à un autre CSV holdout.")
        print("REPRISE APRES CRASH : SHA identiques, autorisée.")
    else:
        atomic_json(
            {
                "schema_version": "first_half_37_v2_holdout_opened_v1",
                "opened_at_utc": now_utc_iso(),
                "model_sha256": EXPECTED_MODEL_SHA256,
                "holdout_sha256": dataset_sha,
                "freeze_lock_sha256": sha256_file(FREEZE_LOCK_PATH),
                "collection_manifest_sha256": sha256_file(COLLECTION_MANIFEST_PATH),
                "status": "HOLDOUT_OPENED_FOR_ONE_SHOT_EVALUATION",
            },
            opened_path,
        )

    print()
    print("Phase 2 : ouverture du holdout et contrôles anti-leakage")
    df = load_and_validate_holdout()

    print(f"Rows holdout               : {len(df)}")
    print(f"Dates                      : {df[DATE_COL].min()} -> {df[DATE_COL].max()}")
    print("Overlap DEVELOPMENT        : 0")
    print("Target                     : RESOLVED 0/1")
    print()

    # A partir d'ici, le holdout est officiellement consommé.
    X = df[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int).to_numpy()

    base_model = bundle["base_model"]
    raw_p = base_model.predict_proba(X)[:, 1]
    alpha = float(bundle["calibration"]["alpha"])
    cal_p = calibrated_probability(raw_p, alpha)

    raw_metrics = metric_row(y, raw_p, "raw_frozen_base_model")
    calibrated_metrics = metric_row(y, cal_p, "frozen_calibrated_v2")

    conditions = {
        "roc_auc_gt_0_50": bool(calibrated_metrics["roc_auc"] > 0.50),
        "pr_auc_gt_holdout_prevalence": bool(
            calibrated_metrics["pr_auc"] > calibrated_metrics["positive_rate"]
        ),
        "brier_better_than_frozen_base_rate": bool(
            calibrated_metrics["brier_improvement"] > 0
        ),
        "logloss_better_than_frozen_base_rate": bool(
            calibrated_metrics["logloss_improvement"] > 0
        ),
    }
    verdict = "FINAL_PASS" if all(conditions.values()) else "FINAL_FAIL"

    bootstrap = bootstrap_metrics(y, cal_p)

    predictions = df[
        [c for c in [
            ID_COL, DATE_COL, "home_team", "away_team",
            "score_37_home", "score_37_away", TARGET
        ] if c in df.columns]
    ].copy()
    predictions["raw_probability"] = raw_p
    predictions["calibrated_probability"] = cal_p
    predictions["frozen_alpha"] = alpha
    predictions["model_sha256"] = EXPECTED_MODEL_SHA256

    daily = daily_table(df, cal_p)
    bands = probability_bands(df, cal_p)

    state = score_state(df)
    if state is not None:
        temp = df.copy()
        temp["score_state_37"] = state
        score_groups = subgroup_table(temp, cal_p, "score_state_37")
    else:
        score_groups = pd.DataFrame()

    metrics_path = output_dir / "first_half_37_v2_final_holdout_metrics.json"
    predictions_path = output_dir / "first_half_37_v2_final_holdout_predictions.csv"
    daily_path = output_dir / "first_half_37_v2_final_holdout_daily.csv"
    bands_path = output_dir / "first_half_37_v2_final_holdout_probability_bands.csv"
    score_path = output_dir / "first_half_37_v2_final_holdout_score_states.csv"
    bootstrap_path = output_dir / "first_half_37_v2_final_holdout_bootstrap.json"
    report_path = output_dir / "first_half_37_v2_final_holdout_report.md"

    metrics_payload = {
        "schema_version": "first_half_37_v2_final_holdout_metrics_v1",
        "evaluated_at_utc": now_utc_iso(),
        "verdict": verdict,
        "conditions": conditions,
        "frozen_pipeline": {
            "model_name": EXPECTED_MODEL_NAME,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "features": EXPECTED_FEATURES,
            "C": EXPECTED_C,
            "calibration_type": "intercept_only",
            "calibration_window": EXPECTED_WINDOW,
            "alpha": EXPECTED_ALPHA,
            "baseline_probability": FROZEN_BASE_RATE,
        },
        "holdout": {
            "sha256": dataset_sha,
            "n": int(len(df)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "date_min": str(df[DATE_COL].min()),
            "date_max": str(df[DATE_COL].max()),
        },
        "calibrated": calibrated_metrics,
        "raw_diagnostic_only": raw_metrics,
        "bootstrap_95": bootstrap,
    }

    atomic_json(metrics_payload, metrics_path)
    atomic_json(bootstrap, bootstrap_path)
    atomic_csv(predictions, predictions_path)
    atomic_csv(daily, daily_path)
    atomic_csv(bands, bands_path)
    atomic_csv(score_groups, score_path)

    write_report(
        report_path,
        calibrated=calibrated_metrics,
        raw=raw_metrics,
        verdict=verdict,
        conditions=conditions,
        bootstrap=bootstrap,
        model_sha=EXPECTED_MODEL_SHA256,
        dataset_sha=dataset_sha,
    )

    results_sha = sha256_file(metrics_path)
    report_sha = sha256_file(report_path)

    evaluation_lock = {
        "schema_version": "first_half_37_v2_holdout_evaluation_lock_v1",
        "evaluated_at_utc": now_utc_iso(),
        "status": "FINAL_HOLDOUT_CONSUMED",
        "verdict": verdict,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "holdout_sha256": dataset_sha,
        "freeze_lock_sha256": sha256_file(FREEZE_LOCK_PATH),
        "collection_manifest_sha256": sha256_file(COLLECTION_MANIFEST_PATH),
        "metrics_sha256": results_sha,
        "report_sha256": report_sha,
        "conditions": conditions,
        "holdout_rows": int(len(df)),
        "holdout_positives": int(y.sum()),
        "holdout_positive_rate": float(y.mean()),
        "do_not_retune_on_this_holdout": True,
    }
    atomic_json(evaluation_lock, final_lock_path)

    print("=" * 120)
    print("FINAL HOLDOUT V2 — EVALUATION ONE-SHOT TERMINEE")
    print("=" * 120)
    print(f"VERDICT                  : {verdict}")
    print(f"N                        : {len(df)}")
    print(f"Positifs                 : {int(y.sum())} ({y.mean():.2%})")
    print(f"Mean probability CAL     : {cal_p.mean():.2%}")
    print(f"ROC-AUC                  : {calibrated_metrics['roc_auc']:.4f}")
    print(
        f"PR-AUC                   : {calibrated_metrics['pr_auc']:.4f} "
        f"| base={calibrated_metrics['positive_rate']:.4f} "
        f"| lift={calibrated_metrics['pr_lift_vs_holdout_prevalence']:+.4f}"
    )
    print(
        f"Brier                    : {calibrated_metrics['brier']:.4f} "
        f"| baseline={calibrated_metrics['baseline_brier']:.4f} "
        f"| imp={calibrated_metrics['brier_improvement']:+.4f}"
    )
    print(
        f"LogLoss                  : {calibrated_metrics['log_loss']:.4f} "
        f"| baseline={calibrated_metrics['baseline_logloss']:.4f} "
        f"| imp={calibrated_metrics['logloss_improvement']:+.4f}"
    )
    print(f"ECE-10                   : {calibrated_metrics['ece_10']:.4f}")
    print()
    print("CONDITIONS PRE-ENREGISTREES")
    for name, ok in conditions.items():
        print(f"  {name:<38} {'PASS' if ok else 'FAIL'}")
    print()
    print(f"Rapport                   : {report_path}")
    print(f"Metrics                   : {metrics_path}")
    print(f"Predictions               : {predictions_path}")
    print(f"Evaluation lock           : {final_lock_path}")
    print()
    print("HOLDOUT STATUS            : CONSOMME")
    print("IMPORTANT                 : ne plus retuner V2 sur ce holdout.")
    return 0 if verdict == "FINAL_PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
