#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3-I BATCH 1 (279) — VALIDATION EXTERNE V3-C vs V3-H
====================================================

Objectif
--------
Évaluer les 279 matchs V3-I déjà collectés, SANS réoptimiser les bandes.

Modèles
-------
1) V3-C production :
   outputs/model_v3_experiments/candidates_refit_all/
   V3_C_AGGRESSIVE_refit_all.joblib

2) V3-H frozen deploy candidate :
   - mêmes 38 features V3-C
   - ExtraTrees "LEAF5"
   - entraîné uniquement sur les 1368 lignes DEVELOPMENT historiques
   - calibration Platt apprise uniquement sur des prédictions OOF temporelles
     produites à l'intérieur de ce DEVELOPMENT
   - AUC/labels du holdout V3-I ne servent JAMAIS à l'entraînement/calibration

Bandes GELÉES avant ouverture du holdout
----------------------------------------
V3-C :
    55.0 <= p < 65.0 %
    58.0 <= p < 61.0 %

V3-H :
    31.0 <= p < 34.0 %
    32.0 <= p < 34.0 %
    32.5 <= p < 34.0 %

Consensus principal :
    V3-C 55-65 ET V3-H 31-34

IMPORTANT
---------
- Ce script consomme le Batch V3-I-1 comme validation.
- Il ne modifie PAS le scanner.
- Il ne modifie PAS le modèle V3-C de production.
- Il ne choisit PAS de nouvelle bande à partir des résultats.
- Toute nouvelle optimisation après lecture de ces résultats serait du tuning
  sur holdout et doit être évitée.

Commande
--------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 -u evaluate_v3i_batch1_279_v3c_v3h.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

V3C_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/model_v3_experiments/candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)

V3C_SCHEMA_PATH = (
    PROJECT_ROOT
    / "outputs/model_v3_experiments/schemas"
    / "V3_C_AGGRESSIVE_schema.json"
)

HISTORICAL_DATASET_CANDIDATES = [
    PROJECT_ROOT
    / "outputs/historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    PROJECT_ROOT
    / "outputs/historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
]

HOLDOUT_CANDIDATES = [
    PROJECT_ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv",
    PROJECT_ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "Historical_Dataset_Goal_After_87_RAW_v2.csv",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "evaluation_batch1_279"
)

EXPECTED_HOLDOUT_ROWS = 279

TARGET_COL = "goal_after_87"
TIME_COL = "kickoff_utc"
MATCH_ID_COL = "match_id"

DEV_ROWS = 1368

# Folds historiques V3-H déjà figés :
# F1 train 684 -> val 171
# F2 train 855 -> val 171
# F3 train 1026 -> val 171
# F4 train 1197 -> val 171
OOF_INITIAL_TRAIN = 684
OOF_VAL_SIZE = 171
OOF_FOLDS = 4

RANDOM_STATE = 42

# V3G_ET_LEAF5 : référence ExtraTrees 700 / depth 14 / sqrt,
# seule différence principale vs la référence V3-F : min_samples_leaf=5.
ET_PARAMS = dict(
    n_estimators=700,
    max_depth=14,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Bandes déclarées AVANT lecture de ce holdout.
FROZEN_BANDS = [
    ("V3C_55_65", "p_v3c", 0.55, 0.65),
    ("V3C_58_61", "p_v3c", 0.58, 0.61),
    ("V3H_31_34", "p_v3h", 0.31, 0.34),
    ("V3H_32_34", "p_v3h", 0.32, 0.34),
    ("V3H_32p5_34", "p_v3h", 0.325, 0.34),
]

EPS = 1e-12


# =============================================================================
# UTILS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def smart_read_csv(path: Path) -> pd.DataFrame:
    """
    Le dataset historique V2 a déjà existé en TAB malgré extension .csv.
    On essaie plusieurs séparateurs et on conserve la lecture qui produit
    le plus de colonnes cohérentes.
    """
    attempts: List[Tuple[int, str, pd.DataFrame]] = []

    for sep in ("\t", ",", ";"):
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
            )
            attempts.append((df.shape[1], sep, df))
        except Exception:
            pass

    if not attempts:
        raise RuntimeError(f"Impossible de lire : {path}")

    attempts.sort(key=lambda x: x[0], reverse=True)
    ncols, sep, df = attempts[0]

    if ncols < 10:
        raise RuntimeError(
            f"Lecture suspecte de {path}: seulement {ncols} colonnes."
        )

    print(f"READ | {path}")
    print(f"     | séparateur={repr(sep)} | rows={len(df)} | cols={df.shape[1]}")
    return df


def find_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    formatted = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"{label} introuvable. Candidats:\n{formatted}")


def parse_time_series(s: pd.Series) -> pd.Series:
    """
    ISO timestamps d'abord; fallback Excel serial si une série numérique
    ressemble à des dates Excel.
    """
    out = pd.to_datetime(s, utc=True, errors="coerce")

    # Fallback ciblé pour valeurs numériques type Excel serial.
    missing = out.isna()
    if missing.any():
        numeric = pd.to_numeric(s[missing], errors="coerce")
        plausible = numeric.between(20000, 80000)
        if plausible.any():
            converted = pd.to_datetime(
                numeric[plausible],
                unit="D",
                origin="1899-12-30",
                utc=True,
                errors="coerce",
            )
            out.loc[converted.index] = converted

    return out


def extract_schema_features(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    priority_keys = {
        "features",
        "feature_names",
        "model_features",
        "feature_order",
        "columns",
        "schema",
    }

    priority_candidates: List[List[str]] = []
    any_candidates: List[List[str]] = []

    def walk(obj: Any, parent_key: str = "") -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if (
                    isinstance(v, list)
                    and len(v) == 38
                    and all(isinstance(x, str) for x in v)
                ):
                    if str(k).lower() in priority_keys:
                        priority_candidates.append(list(v))
                    else:
                        any_candidates.append(list(v))
                walk(v, str(k))
        elif isinstance(obj, list):
            if len(obj) == 38 and all(isinstance(x, str) for x in obj):
                any_candidates.append(list(obj))
            else:
                for v in obj:
                    walk(v, parent_key)

    walk(payload)

    candidates = priority_candidates + any_candidates
    if not candidates:
        raise ValueError(
            f"Impossible d'extraire une liste de 38 features depuis {path}"
        )

    # Déduplique en gardant l'ordre.
    unique: List[List[str]] = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if len(unique) > 1:
        print(
            f"WARNING | plusieurs listes de 38 features trouvées; "
            f"utilisation de la première dans {path.name}"
        )

    features = unique[0]

    if len(features) != 38 or len(set(features)) != 38:
        raise ValueError("Schéma V3-C invalide: 38 features uniques requises.")

    return features


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(X), dtype=float)

    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("predict_proba ne renvoie pas 2 classes.")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = list(classes)
        if 1 in classes:
            idx = classes.index(1)
            return p[:, idx]

    return p[:, 1]


def build_v3h_et() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(**ET_PARAMS)),
        ]
    )


def fit_platt(raw_prob: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Platt scaling standard sur le score brut ExtraTrees :
    régression logistique 1D, sans utiliser le holdout V3-I.
    """
    x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(x, y)
    return calibrator


def apply_platt(calibrator: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(x)[:, 1]


def safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(log_loss(y, p, labels=[0, 1]))


def ece_10(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0.0, 1.0, 11)
    total = len(y)
    if total == 0:
        return float("nan")

    acc = 0.0
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        n = int(mask.sum())
        if n == 0:
            continue

        acc += (n / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))

    return float(acc)


def wilson_interval(pos: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")

    phat = pos / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(
            (phat * (1.0 - phat) / n)
            + ((z * z) / (4.0 * n * n))
        )
        / denom
    )
    return center - margin, center + margin


def band_row(
    df: pd.DataFrame,
    name: str,
    probability_col: str,
    low: float,
    high: float,
) -> Dict[str, Any]:
    z = df[
        (df[probability_col] >= low)
        & (df[probability_col] < high)
    ].copy()

    n = len(z)
    pos = int(z[TARGET_COL].sum()) if n else 0
    rate = pos / n if n else float("nan")
    mean_p = float(z[probability_col].mean()) if n else float("nan")
    ci_low, ci_high = wilson_interval(pos, n)

    return {
        "rule": name,
        "probability_col": probability_col,
        "low": low,
        "high": high,
        "n": n,
        "positives": pos,
        "observed_rate": rate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_probability": mean_p,
        "calibration_gap": (
            rate - mean_p
            if n and math.isfinite(mean_p)
            else float("nan")
        ),
        "fair_odds_empirical": (
            1.0 / rate
            if n and rate > 0
            else float("nan")
        ),
        "fair_odds_model": (
            1.0 / mean_p
            if n and mean_p > 0
            else float("nan")
        ),
        "roi_at_3.20": (
            rate * 3.20 - 1.0
            if n
            else float("nan")
        ),
        "roi_at_3.50": (
            rate * 3.50 - 1.0
            if n
            else float("nan")
        ),
    }


def group_row(
    name: str,
    z: pd.DataFrame,
) -> Dict[str, Any]:
    n = len(z)
    pos = int(z[TARGET_COL].sum()) if n else 0
    rate = pos / n if n else float("nan")
    lo, hi = wilson_interval(pos, n)

    return {
        "group": name,
        "n": n,
        "positives": pos,
        "observed_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "fair_odds_empirical": (
            1.0 / rate
            if n and rate > 0
            else float("nan")
        ),
        "roi_at_3.20": (
            rate * 3.20 - 1.0
            if n
            else float("nan")
        ),
    }


def format_pct(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.2%}"


def format_num(x: Any, decimals: int = 3) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.{decimals}f}"


# =============================================================================
# PREP DATA
# =============================================================================

@dataclass
class PreparedData:
    historical_path: Path
    holdout_path: Path
    features: List[str]
    dev: pd.DataFrame
    holdout: pd.DataFrame


def prepare_data(expected_holdout_rows: int) -> PreparedData:
    if not V3C_SCHEMA_PATH.exists():
        raise FileNotFoundError(V3C_SCHEMA_PATH)

    features = extract_schema_features(V3C_SCHEMA_PATH)
    print(f"SCHEMA V3-C | {len(features)}/38 features")

    historical_path = find_existing(
        HISTORICAL_DATASET_CANDIDATES,
        "Dataset historique",
    )
    holdout_path = find_existing(
        HOLDOUT_CANDIDATES,
        "Holdout V3-I",
    )

    historical = smart_read_csv(historical_path)
    holdout = smart_read_csv(holdout_path)

    required_base = {MATCH_ID_COL, TIME_COL, TARGET_COL}
    for label, df in (("historical", historical), ("holdout", holdout)):
        missing = required_base.difference(df.columns)
        if missing:
            raise ValueError(
                f"{label}: colonnes indispensables absentes: {sorted(missing)}"
            )

        feature_missing = [f for f in features if f not in df.columns]
        if feature_missing:
            raise ValueError(
                f"{label}: features V3-C absentes: {feature_missing}"
            )

    # Targets propres.
    historical[TARGET_COL] = pd.to_numeric(
        historical[TARGET_COL],
        errors="coerce",
    )
    holdout[TARGET_COL] = pd.to_numeric(
        holdout[TARGET_COL],
        errors="coerce",
    )

    historical = historical[historical[TARGET_COL].isin([0, 1])].copy()
    holdout = holdout[holdout[TARGET_COL].isin([0, 1])].copy()

    # Temps.
    historical["_time"] = parse_time_series(historical[TIME_COL])
    holdout["_time"] = parse_time_series(holdout[TIME_COL])

    if historical["_time"].isna().any():
        raise ValueError(
            f"Historique: {int(historical['_time'].isna().sum())} dates invalides."
        )
    if holdout["_time"].isna().any():
        raise ValueError(
            f"Holdout: {int(holdout['_time'].isna().sum())} dates invalides."
        )

    historical = historical.sort_values(
        ["_time", MATCH_ID_COL],
        kind="mergesort",
    ).reset_index(drop=True)

    sort_cols = ["_time"]
    if "_v3i_selection_order" in holdout.columns:
        sort_cols.append("_v3i_selection_order")
    sort_cols.append(MATCH_ID_COL)

    holdout = holdout.sort_values(
        sort_cols,
        kind="mergesort",
    ).reset_index(drop=True)

    # Holdout READY strict = 38/38.
    holdout_numeric = holdout[features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete_mask = holdout_numeric.notna().all(axis=1)

    if not complete_mask.all():
        n_bad = int((~complete_mask).sum())
        # Si on est tombé sur RAW en fallback, on garde uniquement les READY.
        if holdout_path.name == "Historical_Dataset_Goal_After_87_RAW_v2.csv":
            print(
                f"RAW fallback | exclusion de {n_bad} lignes non 38/38."
            )
            holdout = holdout.loc[complete_mask].copy().reset_index(drop=True)
        else:
            bad_ids = holdout.loc[
                ~complete_mask,
                MATCH_ID_COL,
            ].astype(str).head(10).tolist()
            raise ValueError(
                f"READY contient {n_bad} lignes non 38/38. Exemples={bad_ids}"
            )

    # Dédoublonnage strict.
    if holdout[MATCH_ID_COL].duplicated().any():
        dups = holdout.loc[
            holdout[MATCH_ID_COL].duplicated(keep=False),
            MATCH_ID_COL,
        ].astype(str).unique().tolist()
        raise ValueError(
            f"Holdout contient des match_id dupliqués. Exemples={dups[:10]}"
        )

    # Snapshot exact 279.
    if expected_holdout_rows > 0 and len(holdout) != expected_holdout_rows:
        raise ValueError(
            f"Holdout prêt={len(holdout)} lignes, attendu={expected_holdout_rows}. "
            "Pour analyser un autre snapshot, relancer avec "
            "--expected-holdout-rows N."
        )

    if len(historical) < DEV_ROWS:
        raise ValueError(
            f"Historique insuffisant: {len(historical)} < DEV_ROWS={DEV_ROWS}"
        )

    dev = historical.iloc[:DEV_ROWS].copy().reset_index(drop=True)

    # Aucun overlap entre le holdout et l'historique complet.
    master_ids = set(historical[MATCH_ID_COL].astype(str))
    holdout_ids = set(holdout[MATCH_ID_COL].astype(str))
    overlap = sorted(master_ids.intersection(holdout_ids))
    if overlap:
        raise ValueError(
            f"CONTAMINATION: {len(overlap)} match_id du holdout sont dans "
            f"l'historique. Exemples={overlap[:10]}"
        )

    # Chronologie informative.
    print(
        f"DEV V3-H | rows={len(dev)} | "
        f"{dev['_time'].min()} -> {dev['_time'].max()}"
    )
    print(
        f"HOLDOUT   | rows={len(holdout)} | "
        f"{holdout['_time'].min()} -> {holdout['_time'].max()}"
    )

    if holdout["_time"].min() <= dev["_time"].max():
        print(
            "WARNING | le holdout ne commence pas strictement après le DEV V3-H."
        )

    return PreparedData(
        historical_path=historical_path,
        holdout_path=holdout_path,
        features=features,
        dev=dev,
        holdout=holdout,
    )


# =============================================================================
# V3-H FREEZE
# =============================================================================

def freeze_v3h_candidate(
    dev: pd.DataFrame,
    features: Sequence[str],
) -> Dict[str, Any]:
    """
    Génère un V3-H final SANS toucher au holdout :
      1. 4 folds temporels historiques
      2. raw OOF ExtraTrees LEAF5
      3. fit Platt sur OOF uniquement
      4. refit ExtraTrees sur les 1368 DEV
    """
    X = dev[list(features)].apply(pd.to_numeric, errors="coerce")
    y = dev[TARGET_COL].astype(int).to_numpy()

    if X.isna().any().any():
        n_missing = int(X.isna().sum().sum())
        raise ValueError(
            f"DEV V3-H contient {n_missing} valeurs manquantes sur les 38 features."
        )

    oof_raw = np.full(len(dev), np.nan, dtype=float)
    fold_manifest: List[Dict[str, Any]] = []

    print_header("FREEZE V3-H — OOF temporel historique uniquement")

    for fold_idx in range(OOF_FOLDS):
        train_end = OOF_INITIAL_TRAIN + fold_idx * OOF_VAL_SIZE
        val_start = train_end
        val_end = val_start + OOF_VAL_SIZE

        if val_end > len(dev):
            raise ValueError(
                f"Fold {fold_idx+1}: val_end={val_end} > DEV={len(dev)}"
            )

        model = build_v3h_et()
        model.fit(
            X.iloc[:train_end],
            y[:train_end],
        )

        raw = positive_probability(
            model,
            X.iloc[val_start:val_end],
        )
        oof_raw[val_start:val_end] = raw

        fold_manifest.append(
            {
                "fold": fold_idx + 1,
                "train_rows": train_end,
                "val_rows": val_end - val_start,
                "val_start": val_start,
                "val_end_exclusive": val_end,
                "train_positive_rate": float(y[:train_end].mean()),
                "val_positive_rate": float(y[val_start:val_end].mean()),
                "raw_roc_auc": safe_roc(y[val_start:val_end], raw),
                "raw_pr_auc": safe_pr(y[val_start:val_end], raw),
            }
        )

        print(
            f"F{fold_idx+1} | train={train_end} | val={val_end-val_start} | "
            f"ROC={fold_manifest[-1]['raw_roc_auc']:.6f} | "
            f"PR={fold_manifest[-1]['raw_pr_auc']:.6f}"
        )

    mask = np.isfinite(oof_raw)
    if int(mask.sum()) != OOF_VAL_SIZE * OOF_FOLDS:
        raise RuntimeError(
            f"OOF attendu={OOF_VAL_SIZE * OOF_FOLDS}, obtenu={int(mask.sum())}"
        )

    calibrator = fit_platt(
        oof_raw[mask],
        y[mask],
    )

    oof_platt = apply_platt(
        calibrator,
        oof_raw[mask],
    )

    final_et = build_v3h_et()
    final_et.fit(X, y)

    package = {
        "name": "V3H_ET_LEAF5_PLATT_FROZEN_DEV1368",
        "features": list(features),
        "et_model": final_et,
        "platt_calibrator": calibrator,
        "et_params": ET_PARAMS,
        "dev_rows": int(len(dev)),
        "oof_rows": int(mask.sum()),
        "fold_manifest": fold_manifest,
        "oof_raw_metrics": {
            "roc_auc": safe_roc(y[mask], oof_raw[mask]),
            "pr_auc": safe_pr(y[mask], oof_raw[mask]),
            "brier": safe_brier(y[mask], oof_raw[mask]),
            "log_loss": safe_logloss(y[mask], oof_raw[mask]),
            "ece_10": ece_10(y[mask], oof_raw[mask]),
        },
        "oof_platt_in_sample_calibration_metrics": {
            # Attention: calibrateur appris sur ces OOF; ceci est un audit
            # de mapping, pas une nouvelle estimation externe.
            "roc_auc": safe_roc(y[mask], oof_platt),
            "pr_auc": safe_pr(y[mask], oof_platt),
            "brier": safe_brier(y[mask], oof_platt),
            "log_loss": safe_logloss(y[mask], oof_platt),
            "ece_10": ece_10(y[mask], oof_platt),
        },
        "platt_coef": float(calibrator.coef_[0][0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "note": (
            "Calibrateur fit uniquement sur 684 prédictions OOF temporelles "
            "du DEV historique; holdout V3-I jamais utilisé."
        ),
    }

    return package


# =============================================================================
# SCORE + EVALUATE
# =============================================================================

def score_holdout(
    prepared: PreparedData,
    v3h_package: Mapping[str, Any],
) -> pd.DataFrame:
    if not V3C_MODEL_PATH.exists():
        raise FileNotFoundError(V3C_MODEL_PATH)

    v3c = joblib.load(V3C_MODEL_PATH)

    features = prepared.features
    Xh = prepared.holdout[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if Xh.isna().any().any():
        raise ValueError("Holdout: NaN détecté dans les 38 features.")

    p_v3c = positive_probability(v3c, Xh)

    et_model = v3h_package["et_model"]
    calibrator = v3h_package["platt_calibrator"]

    p_v3h_raw = positive_probability(et_model, Xh)
    p_v3h = apply_platt(calibrator, p_v3h_raw)

    keep_cols = [
        c
        for c in [
            MATCH_ID_COL,
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            TIME_COL,
            "score_87_home",
            "score_87_away",
            TARGET_COL,
            "target_status",
            "target_resolution_class",
            "target_confidence",
            "goal_minutes_after_87",
            "_v3i_selection_order",
        ]
        if c in prepared.holdout.columns
    ]

    scored = prepared.holdout[keep_cols].copy()
    scored["p_v3c"] = p_v3c
    scored["p_v3h_raw"] = p_v3h_raw
    scored["p_v3h"] = p_v3h

    # Frozen signals
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

    scored["sig_consensus_main"] = (
        scored["sig_v3c_55_65"]
        & scored["sig_v3h_31_34"]
    )
    scored["sig_consensus_targeted"] = (
        scored["sig_v3c_58_61"]
        & scored["sig_v3h_31_34"]
    )

    return scored


def model_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    y = scored[TARGET_COL].astype(int).to_numpy()

    rows = []
    for name, col in [
        ("V3C_PRODUCTION", "p_v3c"),
        ("V3H_ET_LEAF5_RAW_FROZEN", "p_v3h_raw"),
        ("V3H_ET_LEAF5_PLATT_FROZEN", "p_v3h"),
    ]:
        p = scored[col].astype(float).to_numpy()

        rows.append(
            {
                "model": name,
                "n": len(y),
                "positive_rate": float(y.mean()),
                "mean_probability": float(p.mean()),
                "roc_auc": safe_roc(y, p),
                "pr_auc": safe_pr(y, p),
                "brier": safe_brier(y, p),
                "log_loss": safe_logloss(y, p),
                "ece_10": ece_10(y, p),
            }
        )

    return pd.DataFrame(rows)


def frozen_band_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = [
        band_row(scored, name, col, low, high)
        for name, col, low, high in FROZEN_BANDS
    ]
    return pd.DataFrame(rows)


def consensus_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    c = scored["sig_v3c_55_65"].astype(bool)
    h = scored["sig_v3h_31_34"].astype(bool)

    groups = [
        ("BOTH_MAIN", scored[c & h]),
        ("V3C_ONLY_MAIN", scored[c & ~h]),
        ("V3H_ONLY_MAIN", scored[~c & h]),
        ("NEITHER_MAIN", scored[~c & ~h]),
        ("V3C_55_65_ALL", scored[c]),
        ("V3H_31_34_ALL", scored[h]),
        ("CONSENSUS_TARGETED_V3C58_61_AND_V3H31_34",
         scored[
             scored["sig_v3c_58_61"]
             & scored["sig_v3h_31_34"]
         ]),
    ]

    return pd.DataFrame(
        [group_row(name, z) for name, z in groups]
    )


# =============================================================================
# REPORT
# =============================================================================

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def make_report(
    prepared: PreparedData,
    v3h_package: Mapping[str, Any],
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    bands: pd.DataFrame,
    consensus: pd.DataFrame,
) -> str:
    lines: List[str] = []

    lines += [
        "# V3-I Batch 1 — Validation externe V3-C vs V3-H",
        "",
        "## Protection méthodologique",
        "",
        f"- Holdout analysé : **{len(scored)} matchs**.",
        f"- Positifs globaux : **{int(scored[TARGET_COL].sum())}** "
        f"({scored[TARGET_COL].mean():.2%}).",
        "- Les bandes ont été déclarées avant lecture de ce holdout.",
        "- Aucun seuil n'est réoptimisé dans ce rapport.",
        "- V3-H est entraîné/calibré uniquement avec le DEV historique de 1368 lignes.",
        "- V3-C est l'artefact de production existant.",
        "",
        "## Fichiers",
        "",
        f"- Historique : `{prepared.historical_path}`",
        f"- Holdout : `{prepared.holdout_path}`",
        f"- V3-C : `{V3C_MODEL_PATH}`",
        f"- Schéma : `{V3C_SCHEMA_PATH}`",
        "",
        "## Métriques globales",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Bandes GELÉES",
        "",
        dataframe_to_markdown(bands),
        "",
        "## Consensus principal",
        "",
        dataframe_to_markdown(consensus),
        "",
        "## V3-H freeze",
        "",
        f"- DEV rows : {v3h_package['dev_rows']}",
        f"- OOF calibration rows : {v3h_package['oof_rows']}",
        f"- Platt coef : {v3h_package['platt_coef']:.8f}",
        f"- Platt intercept : {v3h_package['platt_intercept']:.8f}",
        f"- ExtraTrees params : `{json.dumps(v3h_package['et_params'], sort_keys=True)}`",
        "",
        "## Règle d'interprétation",
        "",
        "Ce Batch V3-I-1 est maintenant consommé comme validation. "
        "On peut comparer les règles pré-déclarées, mais on ne doit pas "
        "déplacer leurs bornes après avoir vu ces résultats. "
        "Les matchs supplémentaires devront être considérés comme un nouveau "
        "batch de confirmation.",
        "",
    ]

    return "\n".join(lines)


def print_results(
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    bands: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    print_header("V3-I BATCH 1 — RÉSUMÉ")
    print(
        f"N={len(scored)} | "
        f"buts après 87={int(scored[TARGET_COL].sum())} | "
        f"taux={scored[TARGET_COL].mean():.2%}"
    )

    print_header("MÉTRIQUES GLOBALES")
    display = metrics.copy()
    for c in ["positive_rate", "mean_probability", "ece_10"]:
        display[c] = display[c].map(format_pct)
    for c in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        display[c] = display[c].map(lambda x: format_num(x, 6))
    print(display.to_string(index=False))

    print_header("BANDES GELÉES")
    display = bands.copy()
    for c in [
        "observed_rate",
        "ci95_low",
        "ci95_high",
        "mean_probability",
        "calibration_gap",
        "roi_at_3.20",
        "roi_at_3.50",
    ]:
        display[c] = display[c].map(format_pct)
    for c in ["fair_odds_empirical", "fair_odds_model"]:
        display[c] = display[c].map(lambda x: format_num(x, 2))
    print(
        display[
            [
                "rule",
                "n",
                "positives",
                "observed_rate",
                "ci95_low",
                "ci95_high",
                "mean_probability",
                "fair_odds_empirical",
                "roi_at_3.20",
            ]
        ].to_string(index=False)
    )

    print_header("CONSENSUS V3-C 55-65 vs V3-H 31-34")
    display = consensus.copy()
    for c in [
        "observed_rate",
        "ci95_low",
        "ci95_high",
        "roi_at_3.20",
    ]:
        display[c] = display[c].map(format_pct)
    display["fair_odds_empirical"] = display[
        "fair_odds_empirical"
    ].map(lambda x: format_num(x, 2))
    print(display.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-holdout-rows",
        type=int,
        default=EXPECTED_HOLDOUT_ROWS,
        help=(
            "Protection du snapshot. Par défaut 279. "
            "Mettre 0 pour désactiver le contrôle."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("V3-I BATCH 1 — VALIDATION EXTERNE")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
    print(f"Expected N   : {args.expected_holdout_rows}")
    print("Aucune réoptimisation de seuil n'est effectuée.")

    prepared = prepare_data(args.expected_holdout_rows)

    v3h_package = freeze_v3h_candidate(
        prepared.dev,
        prepared.features,
    )

    # Sauvegarde du candidat V3-H frozen AVANT scoring du holdout dans le flux.
    v3h_model_path = OUTPUT_DIR / "V3H_ET_LEAF5_PLATT_FROZEN_DEV1368.joblib"
    joblib.dump(v3h_package, v3h_model_path)

    v3h_manifest_path = OUTPUT_DIR / "V3H_ET_LEAF5_PLATT_FROZEN_DEV1368_manifest.json"
    manifest_serializable = {
        k: v
        for k, v in v3h_package.items()
        if k not in {"et_model", "platt_calibrator"}
    }
    with open(v3h_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest_serializable,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(f"V3-H frozen : {v3h_model_path}")

    scored = score_holdout(
        prepared,
        v3h_package,
    )

    metrics = model_metrics(scored)
    bands = frozen_band_metrics(scored)
    consensus = consensus_metrics(scored)

    # Sorties
    predictions_path = OUTPUT_DIR / "v3i_batch1_279_predictions_v3c_v3h.csv"
    metrics_path = OUTPUT_DIR / "v3i_batch1_279_model_metrics.csv"
    bands_path = OUTPUT_DIR / "v3i_batch1_279_frozen_band_metrics.csv"
    consensus_path = OUTPUT_DIR / "v3i_batch1_279_consensus_metrics.csv"
    report_path = OUTPUT_DIR / "v3i_batch1_279_report.md"

    scored.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bands.to_csv(bands_path, index=False)
    consensus.to_csv(consensus_path, index=False)

    report = make_report(
        prepared,
        v3h_package,
        scored,
        metrics,
        bands,
        consensus,
    )
    report_path.write_text(report, encoding="utf-8")

    print_results(
        scored,
        metrics,
        bands,
        consensus,
    )

    print_header("SORTIES")
    for p in [
        predictions_path,
        metrics_path,
        bands_path,
        consensus_path,
        report_path,
        v3h_model_path,
        v3h_manifest_path,
    ]:
        print(p)

    print()
    print(
        "IMPORTANT : ne pas modifier les bandes après lecture de ce résultat. "
        "Le prochain lot de nouveaux matchs doit servir de confirmation séparée."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
