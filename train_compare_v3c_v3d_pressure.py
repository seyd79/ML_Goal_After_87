#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3-C vs V3-E ABLATION — GOAL AFTER 87'
=========================================

OBJECTIF
--------
Identifier quelles variables de pression / récence ajoutent réellement du
signal à V3-C, UNE PAR UNE puis dans quelques petits groupes prédéfinis,
sans modifier le scanner ni le modèle de production.

POINT MÉTHODOLOGIQUE IMPORTANT
------------------------------
Le dernier holdout historique 20 % (utilisé dans l'expérience V3-D) a déjà été
consulté. Il n'est donc PLUS un juge final vierge. Ce script l'EXCLUT totalement
:
- aucun entraînement dessus,
- aucune prédiction dessus,
- aucune sélection de feature dessus,
- aucune métrique dessus.

La sélection V3-E ci-dessous est un SCREENING DE DÉVELOPPEMENT basé uniquement
sur le bloc historique antérieur (les premiers 80 % du dataset, triés dans le
temps) avec 4 folds rolling/expanding. Toute feature retenue devra ensuite être
confirmée sur de nouveaux matchs réellement futurs/postérieurs au dataset.

BASELINE
--------
V3-C = 38 features exactes du schéma :
outputs/model_v3_experiments/schemas/V3_C_AGGRESSIVE_schema.json

CANDIDATS SINGLE-FEATURE
------------------------
1.  + shots_80_87
2.  + total_shots_87
3.  + total_sot_87
4.  + recent_sot_flag
5.  + shot_pressure_ratio_80_87
6.  + sot_share_80_87
7.  + pressure_index_80_87
8.  + score_margin_abs
9.  + favorite_goal_margin_87
10. + pressure_x_close_score

PETITS GROUPES PRÉDÉFINIS
-------------------------
- SHOT_ACTIVITY       = shots_80_87 + total_sot_87
- PRESSURE_RECENCY    = pressure_index_80_87 + recent_sot_flag
- CLOSE_SCORE_PRESSURE= pressure_index_80_87 + score_margin_abs
                         + pressure_x_close_score
- SHOT_QUALITY        = total_sot_87 + sot_share_80_87

Aucun groupe n'est construit après lecture des résultats : ils sont gelés dans
ce fichier avant exécution afin de limiter le data-snooping.

GATE DE SCREENING
-----------------
Un challenger est marqué SCREEN_PASS seulement s'il respecte simultanément :
- delta ROC poolé >= +0.010
- delta PR poolé >= 0
- ROC gagné sur >= 3/4 folds
- pire delta ROC de fold > -0.030
- Brier poolé ne se dégrade pas de plus de 0.005
- bootstrap P(delta ROC > 0) >= 0.90

Ce gate est volontairement conservateur mais reste un screening de développement,
pas une autorisation de déploiement.

SORTIES
-------
outputs/v3e_ablation_experiment/
    candidate_definitions.json
    feature_schema_v3e_ablation.json
    data_audit.csv
    csv_loader_audit.json
    excluded_seen_holdout_info.json
    rolling_fold_metrics.csv
    rolling_fold_deltas.csv
    rolling_predictions.csv
    rolling_pooled_metrics.csv
    paired_bootstrap.csv
    candidate_summary.csv
    probability_band_55_65.csv
    comparison_report.md
    experiment_manifest.json

USAGE
-----
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 -u train_compare_v3c_v3e_ablation.py

IMPORTANT
---------
- Aucun refit_all.
- Aucun fichier de production n'est modifié.
- Aucune imputation silencieuse.
- Toutes les variantes sont comparées sur EXACTEMENT les mêmes matchs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    os.getenv("GOAL87_PROJECT_ROOT", "/Users/diakiteseyd/Desktop/ML_Goal_After_87:")
)

DEFAULT_TEMPLATE_REL = Path(
    "outputs/model_v3_experiments/candidates_refit_all/V3_C_AGGRESSIVE_refit_all.joblib"
)
DEFAULT_SCHEMA_REL = Path(
    "outputs/model_v3_experiments/schemas/V3_C_AGGRESSIVE_schema.json"
)
DATASET_CANDIDATES = [
    Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv"),
    Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"),
    Path("Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv"),
    Path("Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"),
]
DEFAULT_OUTPUT_REL = Path("outputs/v3e_ablation_experiment")

TARGET_CANDIDATES = ["goal_after_87", "target", "label", "y"]
TIME_CANDIDATES = [
    "kickoff_utc", "kickoff_time_utc", "timestamp", "match_datetime",
    "fixture_datetime", "kickoff", "kickoff_time", "match_date", "date",
]
MATCH_ID_CANDIDATES = ["match_id", "fixture_id", "id"]

# The historical tail was already inspected in the V3-D experiment.
SEEN_TAIL_FRACTION = 0.20
ROLLING_FOLDS = 4
ROLLING_INITIAL_TRAIN_FRACTION = 0.50
BOOTSTRAP_REPS = 3000
RANDOM_STATE = 42
EPS = 1e-12

# Screening thresholds. These are not deployment thresholds.
SCREEN_MIN_DELTA_ROC = 0.010
SCREEN_MIN_DELTA_PR = 0.0
SCREEN_MIN_FOLD_WINS = 3
SCREEN_MIN_WORST_FOLD_DELTA_ROC = -0.030
SCREEN_MIN_BRIER_IMPROVEMENT = -0.005
SCREEN_MIN_BOOTSTRAP_P_GT_0 = 0.90

ENGINEERED_FEATURES = [
    "shots_80_87",
    "total_shots_87",
    "total_sot_87",
    "recent_sot_flag",
    "shot_pressure_ratio_80_87",
    "sot_share_80_87",
    "pressure_index_80_87",
    "score_margin_abs",
    "favorite_goal_margin_87",
    "pressure_x_close_score",
]

RAW_REQUIRED = [
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "minutes_since_last_shot_on_target",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "score_87_home",
    "score_87_away",
    "odds_home",
    "odds_away",
]

# Frozen before reading experiment results.
CANDIDATE_ADDITIONS: Dict[str, List[str]] = {
    "V3C_BASELINE": [],
    "V3E_01_SHOTS_80_87": ["shots_80_87"],
    "V3E_02_TOTAL_SHOTS_87": ["total_shots_87"],
    "V3E_03_TOTAL_SOT_87": ["total_sot_87"],
    "V3E_04_RECENT_SOT_FLAG": ["recent_sot_flag"],
    "V3E_05_SHOT_PRESSURE_RATIO": ["shot_pressure_ratio_80_87"],
    "V3E_06_SOT_SHARE_80_87": ["sot_share_80_87"],
    "V3E_07_PRESSURE_INDEX": ["pressure_index_80_87"],
    "V3E_08_SCORE_MARGIN_ABS": ["score_margin_abs"],
    "V3E_09_FAVORITE_GOAL_MARGIN": ["favorite_goal_margin_87"],
    "V3E_10_PRESSURE_X_CLOSE": ["pressure_x_close_score"],
    "V3E_G1_SHOT_ACTIVITY": ["shots_80_87", "total_sot_87"],
    "V3E_G2_PRESSURE_RECENCY": ["pressure_index_80_87", "recent_sot_flag"],
    "V3E_G3_CLOSE_SCORE_PRESSURE": [
        "pressure_index_80_87", "score_margin_abs", "pressure_x_close_score"
    ],
    "V3E_G4_SHOT_QUALITY": ["total_sot_87", "sot_share_80_87"],
}


# =============================================================================
# HELPERS / IO
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_explicit_or_default(root: Path, raw: Optional[str], default_rel: Path, label: str) -> Path:
    p = Path(raw).expanduser() if raw else (root / default_rel)
    if raw and not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"{label} introuvable : {p}")
    return p


def resolve_dataset(root: Path, raw: Optional[str]) -> Path:
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset introuvable : {p}")
        return p
    for rel in DATASET_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset CORE_READY introuvable. Cherché:\n" +
        "\n".join(str(root / x) for x in DATASET_CANDIDATES)
    )


def _detect_header_and_delimiter(path: Path) -> Tuple[str, int, List[str]]:
    delimiters = [",", "\t", ";", "|"]
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        physical: List[str] = []
        for _ in range(120):
            line = fh.readline()
            if line == "":
                break
            physical.append(line.rstrip("\r\n"))

    required = {"match_id", "score_87_home", "score_87_away"}
    for line_no, line in enumerate(physical, start=1):
        if "match_id" not in line or "score_87_home" not in line:
            continue
        for delim in delimiters:
            try:
                row = next(csv.reader([line], delimiter=delim))
            except Exception:
                continue
            header = [str(x).strip() for x in row]
            if required.issubset({h.lower() for h in header}):
                return delim, line_no, header

    raise ValueError(
        "Impossible de trouver le vrai header historique dans les 120 premières lignes "
        "(match_id + score_87_home + score_87_away requis)."
    )


def _pick_header_column(
    header: Sequence[str],
    candidates: Sequence[str],
    explicit: Optional[str] = None,
    label: str = "colonne",
) -> Optional[str]:
    lower_to_real = {str(h).strip().lower(): str(h).strip() for h in header}
    if explicit:
        real = lower_to_real.get(str(explicit).strip().lower())
        if real is None:
            raise ValueError(f"{label} explicite absente du header: {explicit}")
        return real
    for c in candidates:
        real = lower_to_real.get(str(c).strip().lower())
        if real is not None:
            return real
    return None


def load_schema(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        feats = payload
    elif isinstance(payload, Mapping):
        feats = (
            payload.get("features")
            or payload.get("features_used")
            or payload.get("core_features")
            or payload.get("feature_names")
            or []
        )
    else:
        feats = []
    feats = [str(x).strip() for x in feats if str(x).strip()]
    if len(feats) != 38:
        raise ValueError(f"Le schéma V3-C doit contenir 38 features; trouvé={len(feats)}")
    if len(set(feats)) != len(feats):
        raise ValueError("Doublons dans le schéma V3-C.")
    return feats


def load_dataset_robust(
    path: Path,
    v3c_features: Sequence[str],
    explicit_target: Optional[str] = None,
    explicit_time: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, Optional[str], Dict[str, Any]]:
    """Stream only required columns. Never silently skip malformed required rows."""
    delimiter, header_line_no, header = _detect_header_and_delimiter(path)
    header = [str(x).strip() for x in header]
    if len(set(h.lower() for h in header)) != len(header):
        raise ValueError("Header historique contient des noms de colonnes dupliqués.")

    target_col = _pick_header_column(header, TARGET_CANDIDATES, explicit_target, "Target")
    time_col = _pick_header_column(header, TIME_CANDIDATES, explicit_time, "Colonne temps")
    match_id_col = _pick_header_column(header, MATCH_ID_CANDIDATES, None, "match_id")
    if target_col is None:
        raise ValueError("Target introuvable. Utilise --target NOM.")
    if time_col is None:
        raise ValueError("Colonne temps introuvable. Utilise --time-col NOM.")

    lower_to_real = {h.lower(): h for h in header}
    required_names: List[str] = []
    for name in list(v3c_features) + list(RAW_REQUIRED):
        real = lower_to_real.get(str(name).strip().lower())
        if real is None:
            raise ValueError(f"Colonne requise absente du dataset: {name}")
        if real not in required_names:
            required_names.append(real)
    for name in [target_col, time_col, match_id_col]:
        if name and name not in required_names:
            required_names.append(name)

    index = {name: i for i, name in enumerate(header)}
    max_needed_idx = max(index[name] for name in required_names)
    records: List[Dict[str, Any]] = []
    malformed_required: List[Tuple[int, str, int]] = []
    extra_field_rows = 0
    repeated_headers = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for _ in range(header_line_no - 1):
            fh.readline()
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            parsed_header = [str(x).strip() for x in next(reader)]
        except StopIteration:
            raise ValueError("Dataset vide après le header détecté.")

        if [x.lower() for x in parsed_header] != [x.lower() for x in header]:
            raise RuntimeError("Le header détecté diffère du header réellement parsé.")

        id_idx = index[match_id_col] if match_id_col else None
        for logical_line, row in enumerate(reader, start=header_line_no + 1):
            if not row or not any(str(x).strip() for x in row):
                continue
            if id_idx is not None and id_idx < len(row):
                mid = str(row[id_idx]).strip()
                if mid.lower() == str(match_id_col).lower():
                    repeated_headers += 1
                    continue
            else:
                mid = ""

            if len(row) <= max_needed_idx:
                malformed_required.append((logical_line, mid or "?", len(row)))
                continue
            if len(row) > len(header):
                extra_field_rows += 1
            records.append({name: row[index[name]] for name in required_names})

    if malformed_required:
        sample = ", ".join(
            f"ligne~{ln}:{mid}({count} champs)"
            for ln, mid, count in malformed_required[:8]
        )
        raise RuntimeError(
            f"{len(malformed_required)} ligne(s) trop courtes pour les colonnes requises. "
            "Aucune ligne n'a été ignorée silencieusement. Exemples: " + sample
        )
    if not records:
        raise ValueError("Aucune ligne exploitable après le vrai header.")

    df = pd.DataFrame.from_records(records, columns=required_names)
    delim_name = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(
        delimiter, repr(delimiter)
    )
    audit = {
        "header_line": int(header_line_no),
        "delimiter": delim_name,
        "header_columns": int(len(header)),
        "loaded_rows": int(len(df)),
        "extra_trailing_field_rows": int(extra_field_rows),
        "repeated_headers_skipped": int(repeated_headers),
        "loaded_columns": int(len(required_names)),
    }
    return df, target_col, time_col, match_id_col, audit


def normalize_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        out = s.astype(int)
    else:
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            out = numeric
        else:
            mapping = {
                "1": 1, "1.0": 1, "true": 1, "yes": 1, "oui": 1,
                "positive": 1, "won": 1, "win": 1, "goal": 1,
                "0": 0, "0.0": 0, "false": 0, "no": 0, "non": 0,
                "negative": 0, "lost": 0, "loss": 0, "no_goal": 0,
            }
            out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f"Target non résolue sur {int(out.isna().sum())} lignes.")
    vals = set(pd.Series(out).astype(int).unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Target non binaire : {sorted(vals)}")
    return pd.Series(out, index=s.index).astype(int)


def parse_time_series(s: pd.Series) -> pd.Series:
    # Try ISO8601 explicitly first to avoid pandas' mixed-format warning.
    try:
        parsed = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(s, utc=True, errors="coerce")
    if parsed.notna().mean() < 0.90:
        num = pd.to_numeric(s, errors="coerce")
        excel_mask = num.between(20000, 80000)
        excel = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
        if excel_mask.any():
            excel.loc[excel_mask] = (
                pd.Timestamp("1899-12-30", tz="UTC")
                + pd.to_timedelta(num.loc[excel_mask], unit="D")
            )
        parsed = parsed.where(parsed.notna(), excel)
    return parsed


def require_numeric_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes pour {label}: {', '.join(missing)}")
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = require_numeric_columns(df, RAW_REQUIRED, "V3-E ABLATION")

    total_shots = raw["home_shots_87"] + raw["away_shots_87"]
    total_sot = raw["home_shots_on_target_87"] + raw["away_shots_on_target_87"]
    shots_recent = raw["shots_80_87"]
    sot_recent = raw["sot_80_87"]
    corners_recent = raw["corners_80_87"]
    recency_sot = raw["minutes_since_last_shot_on_target"]

    pressure = shots_recent + 2.0 * sot_recent + corners_recent
    score_margin_abs = (raw["score_87_home"] - raw["score_87_away"]).abs()

    recent_sot_flag = pd.Series(np.nan, index=df.index, dtype=float)
    ok = recency_sot.notna()
    recent_sot_flag.loc[ok] = (recency_sot.loc[ok] <= 1.0).astype(float)

    favorite_goal_margin = pd.Series(np.nan, index=df.index, dtype=float)
    fav_ok = raw[["odds_home", "odds_away", "score_87_home", "score_87_away"]].notna().all(axis=1)
    home_fav = fav_ok & (raw["odds_home"] < raw["odds_away"])
    away_fav = fav_ok & (raw["odds_away"] < raw["odds_home"])
    equal_odds = fav_ok & (raw["odds_home"] == raw["odds_away"])
    favorite_goal_margin.loc[home_fav] = (
        raw.loc[home_fav, "score_87_home"] - raw.loc[home_fav, "score_87_away"]
    )
    favorite_goal_margin.loc[away_fav] = (
        raw.loc[away_fav, "score_87_away"] - raw.loc[away_fav, "score_87_home"]
    )
    favorite_goal_margin.loc[equal_odds] = 0.0

    pressure_x_close = pd.Series(np.nan, index=df.index, dtype=float)
    close_ok = pressure.notna() & score_margin_abs.notna()
    pressure_x_close.loc[close_ok] = (
        pressure.loc[close_ok] * (score_margin_abs.loc[close_ok] <= 1.0).astype(float)
    )

    out = pd.DataFrame(index=df.index)
    out["shots_80_87"] = shots_recent
    out["total_shots_87"] = total_shots
    out["total_sot_87"] = total_sot
    out["recent_sot_flag"] = recent_sot_flag
    out["shot_pressure_ratio_80_87"] = shots_recent / np.maximum(total_shots, 1.0)
    out["sot_share_80_87"] = sot_recent / np.maximum(shots_recent, 1.0)
    out["pressure_index_80_87"] = pressure
    out["score_margin_abs"] = score_margin_abs
    out["favorite_goal_margin_87"] = favorite_goal_margin
    out["pressure_x_close_score"] = pressure_x_close

    if list(out.columns) != ENGINEERED_FEATURES:
        raise RuntimeError("Ordre inattendu des features V3-E.")
    return out


# =============================================================================
# MODEL / METRICS
# =============================================================================

def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim != 2 or p.shape[1] < 2:
            raise ValueError("predict_proba ne retourne pas 2 classes.")
        return np.clip(p[:, 1], EPS, 1.0 - EPS)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        return np.clip(p, EPS, 1.0 - EPS)
    raise ValueError("Le template ne fournit ni predict_proba ni decision_function.")


def safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")


def safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i == bins - 1 else (p < hi))
        n = int(mask.sum())
        if n:
            ece += (n / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metrics_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": safe_roc(y, p),
        "pr_auc": safe_pr(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece10": ece_score(y, p, bins=10),
        "mean_probability": float(np.mean(p)),
    }


def set_n_jobs_if_possible(model: Any, n_jobs: int) -> Any:
    try:
        params = model.get_params(deep=True)
    except Exception:
        return model
    updates: Dict[str, Any] = {}
    if "n_jobs" in params:
        updates["n_jobs"] = n_jobs
    else:
        keys = [k for k in params if k.endswith("__n_jobs")]
        if keys:
            updates[keys[-1]] = n_jobs
    if updates:
        try:
            model.set_params(**updates)
        except Exception:
            pass
    return model


def rolling_splits(n_dev: int, folds: int, initial_fraction: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_dev < 100:
        raise ValueError(f"Bloc développement trop petit: {n_dev}")
    initial_n = max(1, int(math.floor(n_dev * initial_fraction)))
    remaining = n_dev - initial_n
    if remaining < folds:
        raise ValueError("Pas assez de lignes après l'initial train.")
    base = remaining // folds
    rem = remaining % folds
    sizes = [base + (1 if i < rem else 0) for i in range(folds)]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = initial_n
    for size in sizes:
        end = start + size
        splits.append((np.arange(0, start, dtype=int), np.arange(start, end, dtype=int)))
        start = end
    if start != n_dev:
        raise RuntimeError("Erreur interne de construction des folds.")
    return splits


def paired_bootstrap(
    y: np.ndarray,
    p_base: np.ndarray,
    p_challenger: np.ndarray,
    reps: int,
    seed: int,
    challenger_name: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    roc_delta: List[float] = []
    pr_delta: List[float] = []
    brier_imp: List[float] = []
    ll_imp: List[float] = []

    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pb = p_base[idx]
        pc = p_challenger[idx]
        roc_delta.append(roc_auc_score(yy, pc) - roc_auc_score(yy, pb))
        pr_delta.append(average_precision_score(yy, pc) - average_precision_score(yy, pb))
        brier_imp.append(brier_score_loss(yy, pb) - brier_score_loss(yy, pc))
        ll_imp.append(log_loss(yy, pb, labels=[0, 1]) - log_loss(yy, pc, labels=[0, 1]))

    def s(values: Sequence[float], prefix: str) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_ci025": float("nan"),
                f"{prefix}_ci975": float("nan"),
                f"{prefix}_p_gt_0": float("nan"),
            }
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_ci025": float(np.quantile(arr, 0.025)),
            f"{prefix}_ci975": float(np.quantile(arr, 0.975)),
            f"{prefix}_p_gt_0": float(np.mean(arr > 0)),
        }

    out: Dict[str, Any] = {
        "challenger": challenger_name,
        "n": int(n),
        "bootstrap_requested": int(reps),
        "bootstrap_valid": int(len(roc_delta)),
    }
    out.update(s(roc_delta, "delta_roc_challenger_minus_v3c"))
    out.update(s(pr_delta, "delta_pr_challenger_minus_v3c"))
    out.update(s(brier_imp, "brier_improvement_v3c_minus_challenger"))
    out.update(s(ll_imp, "logloss_improvement_v3c_minus_challenger"))
    return out


# =============================================================================
# REPORTING
# =============================================================================

def fmt_float(x: Any, digits: int = 6) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return ""
        return f"{v:.{digits}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "_Aucune ligne._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    cols = [c for c in cols if c in view.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in view[cols].itertuples(index=False, name=None):
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(fmt_float(v))
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3-E ABLATION sur V3-C avec rolling temporal validation.")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--template", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_REL))
    p.add_argument("--seen-tail-fraction", type=float, default=SEEN_TAIL_FRACTION)
    p.add_argument("--rolling-folds", type=int, default=ROLLING_FOLDS)
    p.add_argument("--rolling-initial-train-fraction", type=float, default=ROLLING_INITIAL_TRAIN_FRACTION)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Racine projet introuvable : {root}")

    template_path = resolve_explicit_or_default(root, args.template, DEFAULT_TEMPLATE_REL, "Template V3-C")
    schema_path = resolve_explicit_or_default(root, args.schema, DEFAULT_SCHEMA_REL, "Schéma V3-C")
    dataset_path = resolve_dataset(root, args.dataset)

    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 104)
    log("V3-E ABLATION — ROLLING TEMPORAL SCREENING")
    log("=" * 104)
    log(f"Template : {template_path}")
    log(f"Schema   : {schema_path}")
    log(f"Dataset  : {dataset_path}")
    log(f"Sorties  : {out_dir}")

    template = joblib.load(template_path)
    v3c_features = load_schema(schema_path)
    log(f"V3-C baseline : {len(v3c_features)} features")
    log(f"Challengers   : {len(CANDIDATE_ADDITIONS) - 1}")

    # Ensure every ablation feature is genuinely an addition, not a duplicate of V3-C.
    duplicated = sorted(
        {f for fs in CANDIDATE_ADDITIONS.values() for f in fs if f in v3c_features}
    )
    if duplicated:
        raise ValueError(
            "Certaines features d'ablation sont déjà dans V3-C; protocole ambigu: "
            + ", ".join(duplicated)
        )

    df, target_col, time_col, match_id_col, csv_audit = load_dataset_robust(
        dataset_path,
        v3c_features=v3c_features,
        explicit_target=args.target,
        explicit_time=args.time_col,
    )
    raw_n = len(df)
    log(
        "CSV robuste : vrai header ligne={header_line} | delimiter={delimiter} | "
        "header_cols={header_columns} | lignes={loaded_rows} | "
        "lignes avec champs finaux supplémentaires={extra_trailing_field_rows}".format(**csv_audit)
    )

    if match_id_col:
        mids = df[match_id_col].astype(str).str.strip()
        dup = mids[mids.ne("") & mids.duplicated(keep=False)]
        if not dup.empty:
            raise ValueError(
                f"{dup.nunique()} match_id dupliqués; refus anti-leakage. "
                f"Exemples: {', '.join(dup.head(10).tolist())}"
            )

    y = normalize_target(df[target_col])
    times = parse_time_series(df[time_col])
    if times.isna().any():
        raise ValueError(f"Temps non parsable sur {int(times.isna().sum())} lignes.")

    Xc = require_numeric_columns(df, v3c_features, "V3-C")
    Xe = build_engineered_features(df)

    # Fair experiment: all variants use exactly the same complete rows.
    finite_c = np.isfinite(Xc.to_numpy(dtype=float)).all(axis=1)
    finite_e = np.isfinite(Xe.to_numpy(dtype=float)).all(axis=1)
    common_mask = finite_c & finite_e

    audit_df = pd.DataFrame({
        "metric": [
            "raw_rows", "v3c_complete_rows", "engineered_complete_rows",
            "common_complete_rows", "excluded_common_incomplete", "target_positive_rate_raw",
        ],
        "value": [
            raw_n, int(finite_c.sum()), int(finite_e.sum()), int(common_mask.sum()),
            int((~common_mask).sum()), float(y.mean()),
        ],
    })
    audit_df.to_csv(out_dir / "data_audit.csv", index=False)
    (out_dir / "csv_loader_audit.json").write_text(
        json.dumps(csv_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if int(common_mask.sum()) < 300:
        raise ValueError(f"Seulement {int(common_mask.sum())} lignes communes complètes.")

    df2 = df.loc[common_mask].copy()
    Xc = Xc.loc[common_mask].copy()
    Xe = Xe.loc[common_mask].copy()
    y = y.loc[common_mask].copy()
    times = times.loc[common_mask].copy()

    order = np.argsort(times.to_numpy())
    df2 = df2.iloc[order].reset_index(drop=True)
    Xc = Xc.iloc[order].reset_index(drop=True)
    Xe = Xe.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    times = times.iloc[order].reset_index(drop=True)

    n = len(y)
    seen_tail_fraction = float(args.seen_tail_fraction)
    if not (0.10 <= seen_tail_fraction <= 0.40):
        raise ValueError("--seen-tail-fraction doit être entre 0.10 et 0.40")
    seen_tail_n = max(1, int(math.ceil(n * seen_tail_fraction)))
    dev_n = n - seen_tail_n
    if dev_n < 200 or seen_tail_n < 50:
        raise ValueError(f"Découpage trop petit: dev={dev_n}, seen_tail={seen_tail_n}")

    # Crucial: the seen tail is never passed to fit/predict below.
    df_dev = df2.iloc[:dev_n].reset_index(drop=True)
    Xc_dev = Xc.iloc[:dev_n].reset_index(drop=True)
    Xe_dev = Xe.iloc[:dev_n].reset_index(drop=True)
    y_dev = y.iloc[:dev_n].reset_index(drop=True)
    t_dev = times.iloc[:dev_n].reset_index(drop=True)

    seen_info = {
        "status": "EXCLUDED_FROM_ALL_MODEL_SELECTION_AND_SCORING",
        "reason": "This historical tail was already inspected in the prior V3-D experiment.",
        "total_common_rows": int(n),
        "development_rows_used": int(dev_n),
        "excluded_seen_tail_rows": int(seen_tail_n),
        "development_start": str(times.iloc[0]),
        "development_end": str(times.iloc[dev_n - 1]),
        "excluded_tail_start": str(times.iloc[dev_n]),
        "excluded_tail_end": str(times.iloc[-1]),
        "excluded_fraction": seen_tail_fraction,
    }
    (out_dir / "excluded_seen_holdout_info.json").write_text(
        json.dumps(seen_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    splits = rolling_splits(
        dev_n,
        int(args.rolling_folds),
        float(args.rolling_initial_train_fraction),
    )

    log()
    log(f"Lignes communes complètes : {n:,}/{raw_n:,}")
    log(f"DEV utilisé                 : {dev_n:,} ({t_dev.iloc[0]} -> {t_dev.iloc[-1]})")
    log(
        f"TAIL HISTORIQUE EXCLU       : {seen_tail_n:,} "
        f"({times.iloc[dev_n]} -> {times.iloc[-1]})"
    )
    log("IMPORTANT : cette tail déjà vue ne sera NI fit, NI scorée, NI utilisée pour choisir V3-E.")
    log()

    # Build candidate matrices on the DEV block only.
    matrices: Dict[str, pd.DataFrame] = {}
    feature_lists: Dict[str, List[str]] = {}
    for name, additions in CANDIDATE_ADDITIONS.items():
        feats = list(v3c_features) + list(additions)
        feature_lists[name] = feats
        if additions:
            matrices[name] = pd.concat([Xc_dev, Xe_dev[additions]], axis=1).loc[:, feats]
        else:
            matrices[name] = Xc_dev.loc[:, feats].copy()
        if not np.isfinite(matrices[name].to_numpy(dtype=float)).all():
            raise RuntimeError(f"NaN/inf inattendu dans la matrice {name}.")

    # Save frozen candidate definitions before training.
    candidate_payload = {
        "created_at_utc": now_utc(),
        "base_model": "V3_C_AGGRESSIVE",
        "baseline_feature_count": len(v3c_features),
        "candidate_additions": CANDIDATE_ADDITIONS,
        "screening_gate": {
            "min_delta_roc_pooled": SCREEN_MIN_DELTA_ROC,
            "min_delta_pr_pooled": SCREEN_MIN_DELTA_PR,
            "min_fold_roc_wins": SCREEN_MIN_FOLD_WINS,
            "min_worst_fold_delta_roc": SCREEN_MIN_WORST_FOLD_DELTA_ROC,
            "min_brier_improvement": SCREEN_MIN_BRIER_IMPROVEMENT,
            "min_bootstrap_p_delta_roc_gt_0": SCREEN_MIN_BOOTSTRAP_P_GT_0,
        },
        "note": "Pre-specified screening candidates; no candidate is a production model.",
    }
    (out_dir / "candidate_definitions.json").write_text(
        json.dumps(candidate_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "feature_schema_v3e_ablation.json").write_text(
        json.dumps(feature_lists, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fold_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        log("-" * 104)
        log(
            f"FOLD {fold_no}/{len(splits)} | train={len(train_idx):,} | val={len(val_idx):,} | "
            f"val {t_dev.iloc[val_idx[0]]} -> {t_dev.iloc[val_idx[-1]]}"
        )
        yt = y_dev.iloc[train_idx].to_numpy(dtype=int)
        yv = y_dev.iloc[val_idx].to_numpy(dtype=int)
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            raise ValueError(f"Fold {fold_no} sans les deux classes.")

        for candidate_no, (name, X) in enumerate(matrices.items()):
            m = clone(template)
            m = set_n_jobs_if_possible(m, int(args.n_jobs))
            m.fit(X.iloc[train_idx], y_dev.iloc[train_idx])
            pv = predict_probability(m, X.iloc[val_idx])
            met = metrics_dict(yv, pv)
            fold_rows.append({
                "fold": fold_no,
                "model": name,
                "n_features": int(X.shape[1]),
                "added_features": "|".join(CANDIDATE_ADDITIONS[name]),
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "train_start": str(t_dev.iloc[train_idx[0]]),
                "train_end": str(t_dev.iloc[train_idx[-1]]),
                "val_start": str(t_dev.iloc[val_idx[0]]),
                "val_end": str(t_dev.iloc[val_idx[-1]]),
                **met,
            })

            if name == "V3C_BASELINE":
                prefix = "BASE"
            else:
                prefix = "    "
            log(
                f"  {prefix} {name:<31s} ROC={met['roc_auc']:.6f} | "
                f"PR={met['pr_auc']:.6f} | Brier={met['brier']:.6f} | LL={met['log_loss']:.6f}"
            )

            pred = pd.DataFrame({
                "fold": fold_no,
                "model": name,
                "row_index_dev": val_idx,
                "kickoff_time_utc": t_dev.iloc[val_idx].astype(str).to_numpy(),
                "target": yv,
                "probability": pv,
            })
            if match_id_col:
                pred.insert(
                    3, "match_id",
                    df_dev[match_id_col].iloc[val_idx].astype(str).to_numpy()
                )
            pred_frames.append(pred)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "rolling_fold_metrics.csv", index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(out_dir / "rolling_predictions.csv", index=False)

    # Fold deltas against baseline.
    delta_rows: List[Dict[str, Any]] = []
    for fold_no in sorted(fold_df["fold"].unique()):
        base = fold_df[(fold_df.fold == fold_no) & (fold_df.model == "V3C_BASELINE")].iloc[0]
        for name in CANDIDATE_ADDITIONS:
            if name == "V3C_BASELINE":
                continue
            ch = fold_df[(fold_df.fold == fold_no) & (fold_df.model == name)].iloc[0]
            delta_rows.append({
                "fold": int(fold_no),
                "challenger": name,
                "delta_roc": float(ch.roc_auc - base.roc_auc),
                "delta_pr": float(ch.pr_auc - base.pr_auc),
                "brier_improvement": float(base.brier - ch.brier),
                "logloss_improvement": float(base.log_loss - ch.log_loss),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "rolling_fold_deltas.csv", index=False)

    # Pooled rolling OOS metrics.
    pooled_rows: List[Dict[str, Any]] = []
    pooled_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in CANDIDATE_ADDITIONS:
        sub = preds[preds.model == name].sort_values("row_index_dev")
        yy = sub["target"].to_numpy(dtype=int)
        pp = sub["probability"].to_numpy(dtype=float)
        pooled_probs[name] = (yy, pp)
        pooled_rows.append({
            "model": name,
            "n_features": len(feature_lists[name]),
            "added_features": "|".join(CANDIDATE_ADDITIONS[name]),
            **metrics_dict(yy, pp),
        })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "rolling_pooled_metrics.csv", index=False)

    # Paired bootstrap against V3-C on identical pooled OOS observations.
    y_base, p_base = pooled_probs["V3C_BASELINE"]
    boot_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(CANDIDATE_ADDITIONS):
        if name == "V3C_BASELINE":
            continue
        yy, pp = pooled_probs[name]
        if not np.array_equal(y_base, yy):
            raise RuntimeError(f"Targets non alignées pour {name}.")
        boot_rows.append(
            paired_bootstrap(
                y_base, p_base, pp,
                int(args.bootstrap_reps), int(args.random_state) + i,
                name,
            )
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "paired_bootstrap.csv", index=False)

    # 55-65% diagnostics, secondary only.
    band_rows: List[Dict[str, Any]] = []
    for name in CANDIDATE_ADDITIONS:
        sub = preds[preds.model == name].sort_values("row_index_dev")
        z = sub[(sub.probability >= 0.55) & (sub.probability < 0.65)]
        yy = z.target.to_numpy(dtype=int)
        pp = z.probability.to_numpy(dtype=float)
        band_rows.append({
            "model": name,
            "n_55_65": int(len(z)),
            "positives_55_65": int(yy.sum()) if len(z) else 0,
            "positive_rate_55_65": float(yy.mean()) if len(z) else float("nan"),
            "mean_probability_55_65": float(pp.mean()) if len(z) else float("nan"),
            "roc_auc_within_55_65": safe_roc(yy, pp) if len(z) else float("nan"),
            "pr_auc_within_55_65": safe_pr(yy, pp) if len(z) else float("nan"),
        })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(out_dir / "probability_band_55_65.csv", index=False)

    # Candidate summary and screening gate.
    base_pool = pooled_df[pooled_df.model == "V3C_BASELINE"].iloc[0]
    summary_rows: List[Dict[str, Any]] = []
    for name in CANDIDATE_ADDITIONS:
        if name == "V3C_BASELINE":
            continue
        ch_pool = pooled_df[pooled_df.model == name].iloc[0]
        fd = delta_df[delta_df.challenger == name].copy()
        bt = boot_df[boot_df.challenger == name].iloc[0]

        delta_roc = float(ch_pool.roc_auc - base_pool.roc_auc)
        delta_pr = float(ch_pool.pr_auc - base_pool.pr_auc)
        brier_imp = float(base_pool.brier - ch_pool.brier)
        ll_imp = float(base_pool.log_loss - ch_pool.log_loss)
        fold_wins = int((fd.delta_roc > 0).sum())
        fold_pr_wins = int((fd.delta_pr > 0).sum())
        worst_fold = float(fd.delta_roc.min())
        mean_fold = float(fd.delta_roc.mean())
        p_gt0 = float(bt["delta_roc_challenger_minus_v3c_p_gt_0"])

        screen_pass = (
            delta_roc >= SCREEN_MIN_DELTA_ROC
            and delta_pr >= SCREEN_MIN_DELTA_PR
            and fold_wins >= min(SCREEN_MIN_FOLD_WINS, int(args.rolling_folds))
            and worst_fold > SCREEN_MIN_WORST_FOLD_DELTA_ROC
            and brier_imp >= SCREEN_MIN_BRIER_IMPROVEMENT
            and p_gt0 >= SCREEN_MIN_BOOTSTRAP_P_GT_0
        )

        summary_rows.append({
            "challenger": name,
            "n_features": len(feature_lists[name]),
            "added_features": "|".join(CANDIDATE_ADDITIONS[name]),
            "pooled_roc": float(ch_pool.roc_auc),
            "pooled_pr": float(ch_pool.pr_auc),
            "delta_roc_vs_v3c": delta_roc,
            "delta_pr_vs_v3c": delta_pr,
            "brier_improvement_vs_v3c": brier_imp,
            "logloss_improvement_vs_v3c": ll_imp,
            "fold_roc_wins": fold_wins,
            "fold_pr_wins": fold_pr_wins,
            "mean_fold_delta_roc": mean_fold,
            "worst_fold_delta_roc": worst_fold,
            "bootstrap_p_delta_roc_gt_0": p_gt0,
            "bootstrap_delta_roc_ci025": float(bt["delta_roc_challenger_minus_v3c_ci025"]),
            "bootstrap_delta_roc_ci975": float(bt["delta_roc_challenger_minus_v3c_ci975"]),
            "screen_pass": bool(screen_pass),
        })

    summary_df = pd.DataFrame(summary_rows)
    # Lexicographic ranking: passes first, then ROC improvement, PR, fold wins.
    summary_df = summary_df.sort_values(
        ["screen_pass", "delta_roc_vs_v3c", "delta_pr_vs_v3c", "fold_roc_wins"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    summary_df.to_csv(out_dir / "candidate_summary.csv", index=False)

    passes = summary_df[summary_df.screen_pass == True].copy()  # noqa: E712
    best = summary_df.iloc[0]
    best_pass = passes.iloc[0] if not passes.empty else None

    log()
    log("=" * 104)
    log("RÉSUMÉ V3-E ABLATION")
    log("=" * 104)
    log(
        f"V3-C rolling pooled : ROC={float(base_pool.roc_auc):.6f} | "
        f"PR={float(base_pool.pr_auc):.6f} | Brier={float(base_pool.brier):.6f}"
    )
    log(f"Candidats testés      : {len(summary_df)}")
    log(f"SCREEN_PASS           : {len(passes)}")
    log(
        f"Meilleur ranking      : {best.challenger} | ΔROC={float(best.delta_roc_vs_v3c):+.6f} | "
        f"ΔPR={float(best.delta_pr_vs_v3c):+.6f} | folds ROC={int(best.fold_roc_wins)}/{int(args.rolling_folds)} | "
        f"Pboot={float(best.bootstrap_p_delta_roc_gt_0):.3f} | pass={bool(best.screen_pass)}"
    )
    if best_pass is not None:
        log(
            f"Meilleur SCREEN_PASS  : {best_pass.challenger} | "
            f"ΔROC={float(best_pass.delta_roc_vs_v3c):+.6f} | "
            f"features={best_pass.added_features}"
        )
    else:
        log("Meilleur SCREEN_PASS  : AUCUN")
    log("TAIL déjà vue         : EXCLUE intégralement des métriques et de la sélection")
    log("Décision production   : AUCUNE — confirmation future requise")
    log(f"Rapport               : {out_dir / 'comparison_report.md'}")

    manifest = {
        "created_at_utc": now_utc(),
        "status": "DEVELOPMENT_SCREENING_ONLY_NO_PRODUCTION_CHANGE",
        "project_root": str(root),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "target_col": target_col,
        "time_col": time_col,
        "match_id_col": match_id_col,
        "raw_rows": raw_n,
        "common_complete_rows": int(n),
        "development_rows_used": int(dev_n),
        "excluded_seen_tail_rows": int(seen_tail_n),
        "excluded_seen_tail_fraction": seen_tail_fraction,
        "rolling_folds": int(args.rolling_folds),
        "rolling_initial_train_fraction": float(args.rolling_initial_train_fraction),
        "bootstrap_reps": int(args.bootstrap_reps),
        "baseline_rolling_pooled": {
            "roc_auc": float(base_pool.roc_auc),
            "pr_auc": float(base_pool.pr_auc),
            "brier": float(base_pool.brier),
            "log_loss": float(base_pool.log_loss),
        },
        "screen_pass_count": int(len(passes)),
        "best_ranked_candidate": str(best.challenger),
        "best_ranked_delta_roc": float(best.delta_roc_vs_v3c),
        "best_ranked_screen_pass": bool(best.screen_pass),
        "best_screen_pass_candidate": None if best_pass is None else str(best_pass.challenger),
        "methodology_note": (
            "The previously inspected final historical 20% tail was excluded from all fitting, scoring, "
            "ranking, and feature selection. Results are development screening only and require a genuinely "
            "future labeled holdout before any production change."
        ),
    }
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_cols = [
        "rank", "challenger", "added_features", "pooled_roc", "delta_roc_vs_v3c",
        "pooled_pr", "delta_pr_vs_v3c", "fold_roc_wins", "worst_fold_delta_roc",
        "bootstrap_p_delta_roc_gt_0", "screen_pass",
    ]
    report = "\n".join([
        "# V3-E ABLATION — RAPPORT",
        "",
        "## Statut",
        "**DEVELOPMENT SCREENING ONLY — AUCUN CHANGEMENT PRODUCTION**",
        "",
        "## Anti-contamination",
        f"- Lignes communes complètes : **{n:,}** / {raw_n:,}",
        f"- Bloc DEV utilisé : **{dev_n:,}** ({times.iloc[0]} -> {times.iloc[dev_n - 1]})",
        f"- Tail historique déjà vue et EXCLUE : **{seen_tail_n:,}** ({times.iloc[dev_n]} -> {times.iloc[-1]})",
        "- La tail exclue n'a été ni entraînée, ni scorée, ni utilisée pour classer les features.",
        "",
        "## Baseline V3-C rolling pooled",
        f"- ROC-AUC : **{float(base_pool.roc_auc):.6f}**",
        f"- PR-AUC : **{float(base_pool.pr_auc):.6f}**",
        f"- Brier : **{float(base_pool.brier):.6f}**",
        f"- LogLoss : **{float(base_pool.log_loss):.6f}**",
        "",
        "## Ranking challengers",
        markdown_table(summary_df, top_cols),
        "",
        "## SCREEN_PASS",
        markdown_table(passes, top_cols) if not passes.empty else "_Aucun candidat ne passe le gate._",
        "",
        "## Gate de screening",
        f"- ΔROC poolé >= {SCREEN_MIN_DELTA_ROC:+.3f}",
        f"- ΔPR poolé >= {SCREEN_MIN_DELTA_PR:+.3f}",
        f"- ROC gagné sur >= {min(SCREEN_MIN_FOLD_WINS, int(args.rolling_folds))}/{int(args.rolling_folds)} folds",
        f"- pire ΔROC fold > {SCREEN_MIN_WORST_FOLD_DELTA_ROC:+.3f}",
        f"- amélioration Brier >= {SCREEN_MIN_BRIER_IMPROVEMENT:+.3f}",
        f"- bootstrap P(ΔROC > 0) >= {SCREEN_MIN_BOOTSTRAP_P_GT_0:.2f}",
        "",
        "## Interprétation",
        "Même un SCREEN_PASS n'est pas un modèle validé. Comme plusieurs challengers sont testés sur le même "
        "bloc historique, le meilleur résultat reste sujet au multiple testing. La prochaine étape correcte est "
        "de geler 1 à 3 candidats maximum puis de les tester sur de nouveaux matchs postérieurs au dataset.",
        "",
        "## 55–65 %",
        markdown_table(
            band_df.sort_values(["positive_rate_55_65", "n_55_65"], ascending=[False, False]),
            ["model", "n_55_65", "positives_55_65", "positive_rate_55_65", "mean_probability_55_65", "roc_auc_within_55_65"],
        ),
    ])
    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")

    log()
    log("Aucun fichier de production n'a été modifié.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
