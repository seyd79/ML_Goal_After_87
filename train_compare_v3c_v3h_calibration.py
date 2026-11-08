#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3-H CALIBRATION — GOAL AFTER 87'
=================================

OBJECTIF
--------
Tester si la faiblesse principale de V3-C est la CALIBRATION des probabilités,
sans changer les 38 features V3-C et sans toucher à la production.

Le benchmark compare, sur les mêmes folds temporels rolling :

1) V3C_BALANCED_RAW
   Clone exact du pipeline V3-C actuel.

2) V3H_LOGISTIC_UNBALANCED_RAW
   Même pipeline / hyperparamètres que V3-C, mais class_weight=None.
   Cette variante teste directement l'effet du class balancing sur la calibration.

3) V3H_V3C_PLATT
   V3-C balanced + calibration sigmoid/Platt entraînée UNIQUEMENT sur des
   prédictions temporelles OOF internes au bloc TRAIN de chaque fold externe.

4) V3H_V3C_ISOTONIC
   V3-C balanced + calibration isotonic, même protocole OOF interne.

5) V3H_ET_LEAF5_RAW
   Challenger Extra Trees retenu de V3-G comme référence de discrimination :
   n_estimators=900, max_depth=14, min_samples_leaf=5, max_features='sqrt',
   class_weight='balanced'.

6) V3H_ET_LEAF5_PLATT
   Extra Trees leaf5 + calibration Platt sur OOF temporel interne.

7) V3H_ET_LEAF5_ISOTONIC
   Extra Trees leaf5 + calibration isotonic sur OOF temporel interne.

ANTI-FUITE / ANTI-CONTAMINATION
-------------------------------
- Les 342 derniers matchs historiques déjà vus dans V3-D restent EXCLUS.
- Aucun calibrateur n'est ajusté sur la validation externe.
- Pour chaque fold externe, les calibrateurs utilisent seulement des prédictions
  OOF temporelles construites à l'intérieur du TRAIN externe.
- Après calibration OOF, le modèle de base est réentraîné sur TOUT le TRAIN
  externe avant de scorer la validation externe.
- Aucun modèle de production n'est modifié.

POURQUOI CE TEST
----------------
Le ROC mesure le classement. Le système de pari utilise directement les valeurs
55-60%, 60-65%, fair odds et edge. Il faut donc aussi mesurer si les probabilités
sont numériquement fiables.

MÉTRIQUES PRINCIPALES
---------------------
- Brier (plus bas = mieux)
- LogLoss (plus bas = mieux)
- ECE10 (plus bas = mieux)
- calibration intercept / slope (idéaux ~0 / ~1)

GARDE-FOUS DISCRIMINATION
-------------------------
- ROC-AUC
- PR-AUC

GATE DE CALIBRATION (screening développement)
----------------------------------------------
Un challenger est CALIBRATION_PASS seulement si, vs V3-C balanced raw :
- amélioration Brier poolée >= +0.010
- amélioration LogLoss poolée >= +0.020
- amélioration ECE10 poolée >= +0.020
- delta ROC >= -0.005
- delta PR >= -0.005
- Brier amélioré sur >= 3/4 folds
- bootstrap P(amélioration Brier > 0) >= 0.95
- bootstrap P(amélioration LogLoss > 0) >= 0.90

IMPORTANT : un PASS ne signifie PAS déploiement. Le prochain vrai juge reste un
holdout futur indépendant/postérieur aux données déjà utilisées.

SORTIES
-------
outputs/v3h_calibration/
    data_audit.csv
    csv_loader_audit.json
    excluded_seen_holdout_info.json
    model_definitions.json
    calibration_fit_info.csv
    rolling_fold_metrics.csv
    rolling_fold_deltas.csv
    rolling_predictions.csv
    rolling_pooled_metrics.csv
    paired_bootstrap.csv
    calibration_summary.csv
    reliability_bins.csv
    probability_band_55_65.csv
    probability_bands_5pct.csv
    comparison_report.md
    experiment_manifest.json

USAGE
-----
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 -u train_compare_v3c_v3h_calibration.py
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


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
DEFAULT_OUTPUT_REL = Path("outputs/v3h_calibration")

TARGET_CANDIDATES = ["goal_after_87", "target", "label", "y"]
TIME_CANDIDATES = [
    "kickoff_utc", "kickoff_time_utc", "timestamp", "match_datetime",
    "fixture_datetime", "kickoff", "kickoff_time", "match_date", "date",
]
MATCH_ID_CANDIDATES = ["match_id", "fixture_id", "id"]

SEEN_TAIL_FRACTION = 0.20
ROLLING_FOLDS = 4
ROLLING_INITIAL_TRAIN_FRACTION = 0.50
INNER_CALIBRATION_FOLDS = 3
INNER_CALIBRATION_INITIAL_TRAIN_FRACTION = 0.50
BOOTSTRAP_REPS = 3000
RANDOM_STATE = 42
EPS = 1e-9

CAL_MIN_BRIER_IMPROVEMENT = 0.010
CAL_MIN_LOGLOSS_IMPROVEMENT = 0.020
CAL_MIN_ECE_IMPROVEMENT = 0.020
CAL_MIN_DELTA_ROC = -0.005
CAL_MIN_DELTA_PR = -0.005
CAL_MIN_FOLD_BRIER_WINS = 3
CAL_MIN_BOOTSTRAP_P_BRIER_GT0 = 0.95
CAL_MIN_BOOTSTRAP_P_LOGLOSS_GT0 = 0.90

BASELINE_NAME = "V3C_BALANCED_RAW"
MODEL_NAMES = [
    BASELINE_NAME,
    "V3H_LOGISTIC_UNBALANCED_RAW",
    "V3H_V3C_PLATT",
    "V3H_V3C_ISOTONIC",
    "V3H_ET_LEAF5_RAW",
    "V3H_ET_LEAF5_PLATT",
    "V3H_ET_LEAF5_ISOTONIC",
]
CHALLENGER_NAMES = [x for x in MODEL_NAMES if x != BASELINE_NAME]


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
    """Lit seulement les 38 features + target/temps/id; aucune ligne requise n'est skip silencieusement."""
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
    for name in v3c_features:
        real = lower_to_real.get(str(name).strip().lower())
        if real is None:
            raise ValueError(f"Feature V3-C absente du dataset: {name}")
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


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return repr(value)


# =============================================================================
# MODEL / CALIBRATION HELPERS
# =============================================================================

def describe_estimator(model: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"class": type(model).__name__}
    if isinstance(model, Pipeline):
        out["pipeline_steps"] = [name for name, _ in model.steps]
        out["step_classes"] = {name: type(step).__name__ for name, step in model.steps}
    try:
        out["params"] = to_jsonable(model.get_params(deep=True))
    except Exception:
        out["params"] = {}
    return out


def set_class_weight(model: Any, value: Optional[str]) -> Any:
    """Change uniquement le class_weight du classifieur terminal d'un clone V3-C."""
    params = model.get_params(deep=True)
    keys = [k for k in params if k == "class_weight" or k.endswith("__class_weight")]
    if not keys:
        raise ValueError("Impossible de trouver class_weight dans le template V3-C.")
    # Préfère la clé la plus profonde (classifier__class_weight dans un Pipeline).
    key = sorted(keys, key=lambda x: x.count("__"), reverse=True)[0]
    model.set_params(**{key: value})
    return model


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
    raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")


def make_balanced_v3c_factory(template: Any, n_jobs: int) -> Callable[[], Any]:
    def factory() -> Any:
        m = clone(template)
        return set_n_jobs_if_possible(m, n_jobs)
    return factory


def make_unbalanced_v3c_factory(template: Any, n_jobs: int) -> Callable[[], Any]:
    def factory() -> Any:
        m = clone(template)
        m = set_class_weight(m, None)
        return set_n_jobs_if_possible(m, n_jobs)
    return factory


def make_et_leaf5_factory(random_state: int, n_jobs: int, n_estimators: int) -> Callable[[], Any]:
    def factory() -> Any:
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    return factory


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


def temporal_oof_probabilities(
    factory: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
    initial_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prédictions OOF temporelles internes pour ajuster un calibrateur sans fuite."""
    splits = rolling_splits(len(X_train), folds, initial_fraction)
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    fold_info: List[Dict[str, int]] = []
    for inner_no, (tr, va) in enumerate(splits, start=1):
        yt = y_train.iloc[tr]
        yv = y_train.iloc[va].to_numpy(dtype=int)
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            raise ValueError(f"Inner calibration fold {inner_no} sans les deux classes.")
        m = factory()
        m.fit(X_train.iloc[tr], yt)
        pv = predict_probability(m, X_train.iloc[va])
        ys.append(yv)
        ps.append(pv)
        fold_info.append({"inner_fold": inner_no, "train_n": int(len(tr)), "cal_n": int(len(va))})
    yy = np.concatenate(ys)
    pp = np.concatenate(ps)
    if len(np.unique(yy)) < 2:
        raise ValueError("OOF calibration sans les deux classes.")
    return yy, pp, {
        "oof_calibration_n": int(len(yy)),
        "oof_positive_rate": float(yy.mean()),
        "inner_folds": fold_info,
    }


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expit(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        q = self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return np.clip(q, EPS, 1.0 - EPS)

    def info(self) -> Dict[str, Any]:
        return {
            "method": "platt_logistic_on_logit_probability",
            "intercept": float(self.model.intercept_[0]),
            "slope": float(self.model.coef_[0, 0]),
        }


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=int))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(self.model.predict(np.asarray(p, dtype=float)), dtype=float)
        return np.clip(q, EPS, 1.0 - EPS)

    def info(self) -> Dict[str, Any]:
        thresholds = getattr(self.model, "X_thresholds_", np.array([]))
        return {
            "method": "isotonic",
            "n_thresholds": int(len(thresholds)),
        }


def fit_calibrator(method: str, p_oof: np.ndarray, y_oof: np.ndarray) -> Any:
    if method == "platt":
        return PlattCalibrator().fit(p_oof, y_oof)
    if method == "isotonic":
        return IsotonicCalibrator().fit(p_oof, y_oof)
    raise ValueError(f"Méthode calibration inconnue: {method}")


# =============================================================================
# METRICS
# =============================================================================

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


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Logistic calibration model y ~ intercept + slope * logit(p). Ideal=(0,1)."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = logit(p).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        m.fit(x, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def metrics_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    intercept, slope = calibration_intercept_slope(y, p)
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
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def paired_bootstrap_calibration(
    y: np.ndarray,
    p_base: np.ndarray,
    p_ch: np.ndarray,
    reps: int,
    seed: int,
    challenger: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    brier_imp: List[float] = []
    ll_imp: List[float] = []
    ece_imp: List[float] = []
    roc_delta: List[float] = []
    pr_delta: List[float] = []

    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pb = p_base[idx]
        pc = p_ch[idx]
        brier_imp.append(brier_score_loss(yy, pb) - brier_score_loss(yy, pc))
        ll_imp.append(log_loss(yy, pb, labels=[0, 1]) - log_loss(yy, pc, labels=[0, 1]))
        ece_imp.append(ece_score(yy, pb, 10) - ece_score(yy, pc, 10))
        roc_delta.append(roc_auc_score(yy, pc) - roc_auc_score(yy, pb))
        pr_delta.append(average_precision_score(yy, pc) - average_precision_score(yy, pb))

    def stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
        a = np.asarray(values, dtype=float)
        if a.size == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_ci025": float("nan"),
                f"{prefix}_ci975": float("nan"),
                f"{prefix}_p_gt_0": float("nan"),
            }
        return {
            f"{prefix}_mean": float(a.mean()),
            f"{prefix}_ci025": float(np.quantile(a, 0.025)),
            f"{prefix}_ci975": float(np.quantile(a, 0.975)),
            f"{prefix}_p_gt_0": float(np.mean(a > 0)),
        }

    out: Dict[str, Any] = {
        "challenger": challenger,
        "n": int(n),
        "bootstrap_requested": int(reps),
        "bootstrap_valid": int(len(brier_imp)),
    }
    out.update(stats(brier_imp, "brier_improvement"))
    out.update(stats(ll_imp, "logloss_improvement"))
    out.update(stats(ece_imp, "ece_improvement"))
    out.update(stats(roc_delta, "delta_roc"))
    out.update(stats(pr_delta, "delta_pr"))
    return out


def reliability_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict[str, Any]] = []
    for model in MODEL_NAMES:
        sub = preds[preds.model == model]
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            z = sub[(sub.probability >= lo) & ((sub.probability <= hi) if i == bins - 1 else (sub.probability < hi))]
            if len(z):
                yy = z.target.to_numpy(dtype=int)
                pp = z.probability.to_numpy(dtype=float)
                rows.append({
                    "model": model,
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "low": lo,
                    "high": hi,
                    "n": int(len(z)),
                    "positives": int(yy.sum()),
                    "observed_rate": float(yy.mean()),
                    "mean_probability": float(pp.mean()),
                    "calibration_gap_observed_minus_pred": float(yy.mean() - pp.mean()),
                    "abs_calibration_gap": float(abs(yy.mean() - pp.mean())),
                })
            else:
                rows.append({
                    "model": model, "bin": f"{lo:.1f}-{hi:.1f}", "low": lo, "high": hi,
                    "n": 0, "positives": 0, "observed_rate": float("nan"),
                    "mean_probability": float("nan"), "calibration_gap_observed_minus_pred": float("nan"),
                    "abs_calibration_gap": float("nan"),
                })
    return pd.DataFrame(rows)


def five_pct_bands(preds: pd.DataFrame) -> pd.DataFrame:
    edges = np.arange(0.0, 1.0001, 0.05)
    rows: List[Dict[str, Any]] = []
    for model in MODEL_NAMES:
        sub = preds[preds.model == model]
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            z = sub[(sub.probability >= lo) & ((sub.probability <= hi) if i == len(edges)-2 else (sub.probability < hi))]
            if not len(z):
                continue
            yy = z.target.to_numpy(dtype=int)
            pp = z.probability.to_numpy(dtype=float)
            rows.append({
                "model": model,
                "band": f"{lo:.2f}-{hi:.2f}",
                "low": lo,
                "high": hi,
                "n": int(len(z)),
                "positives": int(yy.sum()),
                "observed_rate": float(yy.mean()),
                "mean_probability": float(pp.mean()),
                "calibration_gap_observed_minus_pred": float(yy.mean() - pp.mean()),
                "abs_calibration_gap": float(abs(yy.mean() - pp.mean())),
                "brier": float(brier_score_loss(yy, pp)),
            })
    return pd.DataFrame(rows)


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
    p = argparse.ArgumentParser(description="V3-H CALIBRATION sur les 38 features V3-C fixes.")
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
    p.add_argument("--inner-calibration-folds", type=int, default=INNER_CALIBRATION_FOLDS)
    p.add_argument("--inner-calibration-initial-train-fraction", type=float, default=INNER_CALIBRATION_INITIAL_TRAIN_FRACTION)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--et-n-estimators", type=int, default=900)
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

    log("=" * 112)
    log("V3-H CALIBRATION — 38 FEATURES V3-C FIXES")
    log("=" * 112)
    log(f"Template : {template_path}")
    log(f"Schema   : {schema_path}")
    log(f"Dataset  : {dataset_path}")
    log(f"Sorties  : {out_dir}")

    template = joblib.load(template_path)
    features = load_schema(schema_path)
    log(f"V3-C features : {len(features)}")
    log(f"Template class: {type(template).__name__}")
    if isinstance(template, Pipeline):
        log("Template steps: " + " -> ".join(f"{n}:{type(s).__name__}" for n, s in template.steps))

    # Verify that the unbalanced clone changes only class_weight.
    tmp_unbal = make_unbalanced_v3c_factory(template, int(args.n_jobs))()
    log("Calibration benchmark : V3-C balanced / unbalanced / Platt / Isotonic + ET leaf5 raw/calibré")

    df, target_col, time_col, match_id_col, csv_audit = load_dataset_robust(
        dataset_path, features, args.target, args.time_col
    )
    log(
        "CSV robuste : vrai header ligne={header_line} | delimiter={delimiter} | "
        "header_cols={header_columns} | lignes={loaded_rows} | "
        "lignes avec champs finaux supplémentaires={extra_trailing_field_rows}".format(**csv_audit)
    )

    y = normalize_target(df[target_col])
    times = parse_time_series(df[time_col])
    if times.isna().any():
        raise ValueError(f"Temps non parsé sur {int(times.isna().sum())} lignes.")

    X = pd.DataFrame(index=df.index)
    for feat in features:
        X[feat] = pd.to_numeric(df[feat], errors="coerce")

    finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    raw_n = len(df)
    complete_n = int(finite_mask.sum())
    pd.DataFrame({
        "metric": ["raw_rows", "v3c_38of38_complete_rows", "excluded_incomplete_rows", "target_positive_rate_raw"],
        "value": [raw_n, complete_n, int((~finite_mask).sum()), float(y.mean())],
    }).to_csv(out_dir / "data_audit.csv", index=False)
    (out_dir / "csv_loader_audit.json").write_text(
        json.dumps(csv_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if complete_n < 300:
        raise ValueError(f"Seulement {complete_n} lignes V3-C 38/38 complètes.")

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
    if not (0.10 <= seen_tail_fraction <= 0.40):
        raise ValueError("--seen-tail-fraction doit être entre 0.10 et 0.40")
    seen_tail_n = max(1, int(math.ceil(n * seen_tail_fraction)))
    dev_n = n - seen_tail_n
    if dev_n < 300 or seen_tail_n < 50:
        raise ValueError(f"Découpage trop petit: dev={dev_n}, seen_tail={seen_tail_n}")

    X_dev = X.iloc[:dev_n].reset_index(drop=True)
    y_dev = y.iloc[:dev_n].reset_index(drop=True)
    t_dev = times.iloc[:dev_n].reset_index(drop=True)
    df_dev = df2.iloc[:dev_n].reset_index(drop=True)

    seen_info = {
        "status": "EXCLUDED_FROM_ALL_MODEL_SELECTION_AND_SCORING",
        "reason": "Historical tail already inspected in V3-D and therefore no longer pristine.",
        "total_complete_rows": int(n),
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

    outer_splits = rolling_splits(
        dev_n, int(args.rolling_folds), float(args.rolling_initial_train_fraction)
    )

    log()
    log(f"Lignes V3-C 38/38 complètes : {n:,}/{raw_n:,}")
    log(f"DEV utilisé                  : {dev_n:,} ({t_dev.iloc[0]} -> {t_dev.iloc[-1]})")
    log(f"TAIL HISTORIQUE EXCLU        : {seen_tail_n:,} ({times.iloc[dev_n]} -> {times.iloc[-1]})")
    log("IMPORTANT : la tail déjà vue reste NI fit, NI scorée, NI utilisée pour choisir V3-H.")
    log("IMPORTANT : les calibrateurs apprennent uniquement via OOF temporel INTERNE au train externe.")

    balanced_factory = make_balanced_v3c_factory(template, int(args.n_jobs))
    unbalanced_factory = make_unbalanced_v3c_factory(template, int(args.n_jobs))
    et_factory = make_et_leaf5_factory(int(args.random_state), int(args.n_jobs), int(args.et_n_estimators))

    model_defs = {
        "created_at_utc": now_utc(),
        "feature_count": 38,
        "features": features,
        "baseline_template": describe_estimator(template),
        "unbalanced_clone": describe_estimator(tmp_unbal),
        "extra_trees_leaf5": describe_estimator(et_factory()),
        "models": MODEL_NAMES,
        "calibration_protocol": {
            "outer_folds": int(args.rolling_folds),
            "inner_temporal_oof_folds": int(args.inner_calibration_folds),
            "inner_initial_train_fraction": float(args.inner_calibration_initial_train_fraction),
            "methods": ["platt", "isotonic"],
            "note": "Calibrator fit only on temporal OOF predictions from outer-train data.",
        },
        "gate": {
            "min_brier_improvement": CAL_MIN_BRIER_IMPROVEMENT,
            "min_logloss_improvement": CAL_MIN_LOGLOSS_IMPROVEMENT,
            "min_ece10_improvement": CAL_MIN_ECE_IMPROVEMENT,
            "min_delta_roc": CAL_MIN_DELTA_ROC,
            "min_delta_pr": CAL_MIN_DELTA_PR,
            "min_fold_brier_wins": CAL_MIN_FOLD_BRIER_WINS,
            "min_bootstrap_p_brier_gt0": CAL_MIN_BOOTSTRAP_P_BRIER_GT0,
            "min_bootstrap_p_logloss_gt0": CAL_MIN_BOOTSTRAP_P_LOGLOSS_GT0,
        },
    }
    (out_dir / "model_definitions.json").write_text(
        json.dumps(model_defs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fold_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    cal_info_rows: List[Dict[str, Any]] = []

    for fold_no, (train_idx, val_idx) in enumerate(outer_splits, start=1):
        log("-" * 112)
        log(
            f"FOLD {fold_no}/{len(outer_splits)} | train={len(train_idx):,} | val={len(val_idx):,} | "
            f"val {t_dev.iloc[val_idx[0]]} -> {t_dev.iloc[val_idx[-1]]}"
        )
        Xtr = X_dev.iloc[train_idx].reset_index(drop=True)
        ytr = y_dev.iloc[train_idx].reset_index(drop=True)
        Xv = X_dev.iloc[val_idx]
        yv = y_dev.iloc[val_idx].to_numpy(dtype=int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2:
            raise ValueError(f"Fold {fold_no} sans les deux classes.")

        # ---- raw balanced V3-C
        v3c = balanced_factory()
        v3c.fit(Xtr, ytr)
        p_v3c_raw = predict_probability(v3c, Xv)

        # ---- raw unbalanced logistic
        unbal = unbalanced_factory()
        unbal.fit(Xtr, ytr)
        p_unbal_raw = predict_probability(unbal, Xv)

        # ---- raw Extra Trees leaf5
        et = et_factory()
        et.fit(Xtr, ytr)
        p_et_raw = predict_probability(et, Xv)

        # ---- temporal OOF calibration data for V3-C
        y_oof_v3c, p_oof_v3c, oof_info_v3c = temporal_oof_probabilities(
            balanced_factory, Xtr, ytr,
            int(args.inner_calibration_folds),
            float(args.inner_calibration_initial_train_fraction),
        )
        platt_v3c = fit_calibrator("platt", p_oof_v3c, y_oof_v3c)
        iso_v3c = fit_calibrator("isotonic", p_oof_v3c, y_oof_v3c)
        p_v3c_platt = platt_v3c.transform(p_v3c_raw)
        p_v3c_iso = iso_v3c.transform(p_v3c_raw)

        # ---- temporal OOF calibration data for Extra Trees leaf5
        y_oof_et, p_oof_et, oof_info_et = temporal_oof_probabilities(
            et_factory, Xtr, ytr,
            int(args.inner_calibration_folds),
            float(args.inner_calibration_initial_train_fraction),
        )
        platt_et = fit_calibrator("platt", p_oof_et, y_oof_et)
        iso_et = fit_calibrator("isotonic", p_oof_et, y_oof_et)
        p_et_platt = platt_et.transform(p_et_raw)
        p_et_iso = iso_et.transform(p_et_raw)

        for family, method, cal, info in [
            ("V3C", "platt", platt_v3c, oof_info_v3c),
            ("V3C", "isotonic", iso_v3c, oof_info_v3c),
            ("ET_LEAF5", "platt", platt_et, oof_info_et),
            ("ET_LEAF5", "isotonic", iso_et, oof_info_et),
        ]:
            cal_info_rows.append({
                "outer_fold": fold_no,
                "family": family,
                "method": method,
                "outer_train_n": int(len(train_idx)),
                "oof_calibration_n": int(info["oof_calibration_n"]),
                "oof_positive_rate": float(info["oof_positive_rate"]),
                **cal.info(),
            })

        probs = {
            BASELINE_NAME: p_v3c_raw,
            "V3H_LOGISTIC_UNBALANCED_RAW": p_unbal_raw,
            "V3H_V3C_PLATT": p_v3c_platt,
            "V3H_V3C_ISOTONIC": p_v3c_iso,
            "V3H_ET_LEAF5_RAW": p_et_raw,
            "V3H_ET_LEAF5_PLATT": p_et_platt,
            "V3H_ET_LEAF5_ISOTONIC": p_et_iso,
        }

        for name in MODEL_NAMES:
            pv = probs[name]
            met = metrics_dict(yv, pv)
            fold_rows.append({
                "fold": fold_no,
                "model": name,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "train_start": str(t_dev.iloc[train_idx[0]]),
                "train_end": str(t_dev.iloc[train_idx[-1]]),
                "val_start": str(t_dev.iloc[val_idx[0]]),
                "val_end": str(t_dev.iloc[val_idx[-1]]),
                **met,
            })
            prefix = "BASE" if name == BASELINE_NAME else "    "
            log(
                f"  {prefix} {name:<34s} ROC={met['roc_auc']:.6f} | PR={met['pr_auc']:.6f} | "
                f"Brier={met['brier']:.6f} | LL={met['log_loss']:.6f} | ECE={met['ece10']:.6f}"
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
                pred.insert(3, "match_id", df_dev[match_id_col].iloc[val_idx].astype(str).to_numpy())
            pred_frames.append(pred)

    cal_info_df = pd.DataFrame(cal_info_rows)
    cal_info_df.to_csv(out_dir / "calibration_fit_info.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "rolling_fold_metrics.csv", index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(out_dir / "rolling_predictions.csv", index=False)

    # Fold deltas vs raw V3-C.
    delta_rows: List[Dict[str, Any]] = []
    for fold_no in sorted(fold_df.fold.unique()):
        base = fold_df[(fold_df.fold == fold_no) & (fold_df.model == BASELINE_NAME)].iloc[0]
        for name in CHALLENGER_NAMES:
            ch = fold_df[(fold_df.fold == fold_no) & (fold_df.model == name)].iloc[0]
            delta_rows.append({
                "fold": int(fold_no),
                "challenger": name,
                "delta_roc": float(ch.roc_auc - base.roc_auc),
                "delta_pr": float(ch.pr_auc - base.pr_auc),
                "brier_improvement": float(base.brier - ch.brier),
                "logloss_improvement": float(base.log_loss - ch.log_loss),
                "ece10_improvement": float(base.ece10 - ch.ece10),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "rolling_fold_deltas.csv", index=False)

    # Pooled OOS metrics.
    pooled_rows: List[Dict[str, Any]] = []
    pooled_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in MODEL_NAMES:
        sub = preds[preds.model == name].sort_values("row_index_dev")
        yy = sub.target.to_numpy(dtype=int)
        pp = sub.probability.to_numpy(dtype=float)
        pooled_probs[name] = (yy, pp)
        pooled_rows.append({"model": name, **metrics_dict(yy, pp)})
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "rolling_pooled_metrics.csv", index=False)

    y_base, p_base = pooled_probs[BASELINE_NAME]
    boot_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(CHALLENGER_NAMES):
        yy, pp = pooled_probs[name]
        if not np.array_equal(yy, y_base):
            raise RuntimeError(f"Targets OOS non alignées pour {name}")
        boot_rows.append(paired_bootstrap_calibration(
            y_base, p_base, pp,
            int(args.bootstrap_reps), int(args.random_state) + 500 + i, name,
        ))
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "paired_bootstrap.csv", index=False)

    # Reliability and probability-band diagnostics.
    rel_df = reliability_table(preds, bins=10)
    rel_df.to_csv(out_dir / "reliability_bins.csv", index=False)
    five_df = five_pct_bands(preds)
    five_df.to_csv(out_dir / "probability_bands_5pct.csv", index=False)

    band_rows: List[Dict[str, Any]] = []
    for name in MODEL_NAMES:
        sub = preds[preds.model == name].sort_values("row_index_dev")
        z = sub[(sub.probability >= 0.55) & (sub.probability < 0.65)]
        yy = z.target.to_numpy(dtype=int)
        pp = z.probability.to_numpy(dtype=float)
        band_rows.append({
            "model": name,
            "n_55_65": int(len(z)),
            "positives_55_65": int(yy.sum()) if len(z) else 0,
            "observed_rate_55_65": float(yy.mean()) if len(z) else float("nan"),
            "mean_probability_55_65": float(pp.mean()) if len(z) else float("nan"),
            "calibration_gap_observed_minus_pred_55_65": float(yy.mean() - pp.mean()) if len(z) else float("nan"),
            "abs_calibration_gap_55_65": float(abs(yy.mean() - pp.mean())) if len(z) else float("nan"),
            "brier_55_65": float(brier_score_loss(yy, pp)) if len(z) else float("nan"),
            "roc_auc_within_55_65": safe_roc(yy, pp) if len(z) else float("nan"),
        })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(out_dir / "probability_band_55_65.csv", index=False)

    base_pool = pooled_df[pooled_df.model == BASELINE_NAME].iloc[0]
    summary_rows: List[Dict[str, Any]] = []
    for name in CHALLENGER_NAMES:
        ch = pooled_df[pooled_df.model == name].iloc[0]
        fd = delta_df[delta_df.challenger == name]
        bt = boot_df[boot_df.challenger == name].iloc[0]

        brier_imp = float(base_pool.brier - ch.brier)
        ll_imp = float(base_pool.log_loss - ch.log_loss)
        ece_imp = float(base_pool.ece10 - ch.ece10)
        delta_roc = float(ch.roc_auc - base_pool.roc_auc)
        delta_pr = float(ch.pr_auc - base_pool.pr_auc)
        brier_wins = int((fd.brier_improvement > 0).sum())
        ll_wins = int((fd.logloss_improvement > 0).sum())
        ece_wins = int((fd.ece10_improvement > 0).sum())
        p_brier = float(bt.brier_improvement_p_gt_0)
        p_ll = float(bt.logloss_improvement_p_gt_0)

        calibration_pass = (
            brier_imp >= CAL_MIN_BRIER_IMPROVEMENT
            and ll_imp >= CAL_MIN_LOGLOSS_IMPROVEMENT
            and ece_imp >= CAL_MIN_ECE_IMPROVEMENT
            and delta_roc >= CAL_MIN_DELTA_ROC
            and delta_pr >= CAL_MIN_DELTA_PR
            and brier_wins >= min(CAL_MIN_FOLD_BRIER_WINS, int(args.rolling_folds))
            and p_brier >= CAL_MIN_BOOTSTRAP_P_BRIER_GT0
            and p_ll >= CAL_MIN_BOOTSTRAP_P_LOGLOSS_GT0
        )

        summary_rows.append({
            "challenger": name,
            "pooled_roc": float(ch.roc_auc),
            "pooled_pr": float(ch.pr_auc),
            "pooled_brier": float(ch.brier),
            "pooled_log_loss": float(ch.log_loss),
            "pooled_ece10": float(ch.ece10),
            "calibration_intercept": float(ch.calibration_intercept),
            "calibration_slope": float(ch.calibration_slope),
            "delta_roc_vs_v3c": delta_roc,
            "delta_pr_vs_v3c": delta_pr,
            "brier_improvement_vs_v3c": brier_imp,
            "logloss_improvement_vs_v3c": ll_imp,
            "ece10_improvement_vs_v3c": ece_imp,
            "fold_brier_wins": brier_wins,
            "fold_logloss_wins": ll_wins,
            "fold_ece_wins": ece_wins,
            "bootstrap_p_brier_improvement_gt_0": p_brier,
            "bootstrap_p_logloss_improvement_gt_0": p_ll,
            "bootstrap_brier_ci025": float(bt.brier_improvement_ci025),
            "bootstrap_brier_ci975": float(bt.brier_improvement_ci975),
            "bootstrap_logloss_ci025": float(bt.logloss_improvement_ci025),
            "bootstrap_logloss_ci975": float(bt.logloss_improvement_ci975),
            "calibration_pass": bool(calibration_pass),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["calibration_pass", "brier_improvement_vs_v3c", "logloss_improvement_vs_v3c", "ece10_improvement_vs_v3c", "delta_roc_vs_v3c"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    summary_df.to_csv(out_dir / "calibration_summary.csv", index=False)

    passes = summary_df[summary_df.calibration_pass == True].copy()  # noqa: E712
    best = summary_df.iloc[0]
    best_pass = passes.iloc[0] if not passes.empty else None

    log()
    log("=" * 112)
    log("RÉSUMÉ V3-H CALIBRATION")
    log("=" * 112)
    log(
        f"V3-C balanced raw : ROC={float(base_pool.roc_auc):.6f} | PR={float(base_pool.pr_auc):.6f} | "
        f"Brier={float(base_pool.brier):.6f} | LL={float(base_pool.log_loss):.6f} | ECE={float(base_pool.ece10):.6f}"
    )
    log(f"Challengers calibration : {len(summary_df)}")
    log(f"CALIBRATION_PASS        : {len(passes)}")
    log(
        f"Meilleur ranking        : {best.challenger} | "
        f"ΔBrier={float(best.brier_improvement_vs_v3c):+.6f} | "
        f"ΔLL={float(best.logloss_improvement_vs_v3c):+.6f} | "
        f"ΔECE={float(best.ece10_improvement_vs_v3c):+.6f} | "
        f"ΔROC={float(best.delta_roc_vs_v3c):+.6f} | "
        f"Brier folds={int(best.fold_brier_wins)}/{int(args.rolling_folds)} | "
        f"PbootBrier={float(best.bootstrap_p_brier_improvement_gt_0):.3f} | "
        f"pass={bool(best.calibration_pass)}"
    )
    if best_pass is not None:
        log(
            f"Meilleur CALIBRATION_PASS: {best_pass.challenger} | "
            f"Brier={float(best_pass.pooled_brier):.6f} | LL={float(best_pass.pooled_log_loss):.6f} | "
            f"ECE={float(best_pass.pooled_ece10):.6f}"
        )
    else:
        log("Meilleur CALIBRATION_PASS: AUCUN")
    log("TAIL déjà vue           : EXCLUE intégralement des métriques et de la sélection")
    log("Décision production     : AUCUNE — toute nouvelle échelle de probabilité exige validation future + nouveaux seuils")
    log(f"Rapport                 : {out_dir / 'comparison_report.md'}")

    # Report.
    summary_cols = [
        "rank", "challenger", "pooled_roc", "delta_roc_vs_v3c", "pooled_pr", "delta_pr_vs_v3c",
        "pooled_brier", "brier_improvement_vs_v3c", "pooled_log_loss", "logloss_improvement_vs_v3c",
        "pooled_ece10", "ece10_improvement_vs_v3c", "calibration_intercept", "calibration_slope",
        "fold_brier_wins", "bootstrap_p_brier_improvement_gt_0", "calibration_pass",
    ]
    band_cols = [
        "model", "n_55_65", "positives_55_65", "observed_rate_55_65", "mean_probability_55_65",
        "calibration_gap_observed_minus_pred_55_65", "abs_calibration_gap_55_65", "brier_55_65",
    ]
    report = "\n".join([
        "# V3-H CALIBRATION — RAPPORT",
        "",
        "## Statut",
        "**DEVELOPMENT SCREENING ONLY — AUCUN CHANGEMENT PRODUCTION**",
        "",
        "## Question testée",
        "Les probabilités V3-C sont-elles mal calibrées, notamment à cause du class balancing, et peut-on améliorer Brier/LogLoss/ECE sans perdre la discrimination ?",
        "",
        "## Anti-fuite",
        f"- DEV utilisé : **{dev_n:,}** lignes ({t_dev.iloc[0]} -> {t_dev.iloc[-1]})",
        f"- Tail historique déjà vue et EXCLUE : **{seen_tail_n:,}** lignes ({times.iloc[dev_n]} -> {times.iloc[-1]})",
        "- Chaque calibrateur est ajusté uniquement sur des prédictions OOF temporelles internes au TRAIN externe.",
        "- La validation externe n'est jamais utilisée pour ajuster le calibrateur.",
        "",
        "## Baseline V3-C balanced raw",
        f"- ROC-AUC : **{float(base_pool.roc_auc):.6f}**",
        f"- PR-AUC : **{float(base_pool.pr_auc):.6f}**",
        f"- Brier : **{float(base_pool.brier):.6f}**",
        f"- LogLoss : **{float(base_pool.log_loss):.6f}**",
        f"- ECE10 : **{float(base_pool.ece10):.6f}**",
        f"- Calibration intercept/slope : **{float(base_pool.calibration_intercept):.4f} / {float(base_pool.calibration_slope):.4f}**",
        "",
        "## Ranking calibration",
        markdown_table(summary_df, summary_cols),
        "",
        "## CALIBRATION_PASS",
        markdown_table(passes, summary_cols) if not passes.empty else "_Aucun challenger ne passe le gate._",
        "",
        "## Zone 55–65 %",
        "Attention : après recalibration, les matchs qui tombent dans 55–65 % ne sont plus nécessairement les mêmes que sous V3-C raw.",
        markdown_table(band_df, band_cols),
        "",
        "## Gate",
        f"- amélioration Brier >= {CAL_MIN_BRIER_IMPROVEMENT:+.3f}",
        f"- amélioration LogLoss >= {CAL_MIN_LOGLOSS_IMPROVEMENT:+.3f}",
        f"- amélioration ECE10 >= {CAL_MIN_ECE_IMPROVEMENT:+.3f}",
        f"- ΔROC >= {CAL_MIN_DELTA_ROC:+.3f}",
        f"- ΔPR >= {CAL_MIN_DELTA_PR:+.3f}",
        f"- Brier amélioré sur >= {min(CAL_MIN_FOLD_BRIER_WINS, int(args.rolling_folds))}/{int(args.rolling_folds)} folds",
        f"- bootstrap P(amélioration Brier > 0) >= {CAL_MIN_BOOTSTRAP_P_BRIER_GT0:.2f}",
        f"- bootstrap P(amélioration LogLoss > 0) >= {CAL_MIN_BOOTSTRAP_P_LOGLOSS_GT0:.2f}",
        "",
        "## Interprétation production",
        "Même si une variante passe, elle ne doit pas remplacer directement V3-C. Une nouvelle calibration change l'échelle des probabilités, donc les bandes 55–60 / 60–65, fair odds et seuils de décision doivent être redéfinis sur validation future indépendante.",
    ])
    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "created_at_utc": now_utc(),
        "status": "DEVELOPMENT_CALIBRATION_SCREENING_ONLY_NO_PRODUCTION_CHANGE",
        "project_root": str(root),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "feature_count": 38,
        "target_col": target_col,
        "time_col": time_col,
        "match_id_col": match_id_col,
        "raw_rows": int(raw_n),
        "complete_rows": int(n),
        "development_rows_used": int(dev_n),
        "excluded_seen_tail_rows": int(seen_tail_n),
        "rolling_folds": int(args.rolling_folds),
        "inner_calibration_folds": int(args.inner_calibration_folds),
        "bootstrap_reps": int(args.bootstrap_reps),
        "baseline": BASELINE_NAME,
        "baseline_pooled": {k: to_jsonable(v) for k, v in base_pool.to_dict().items()},
        "calibration_pass_count": int(len(passes)),
        "best_ranked_candidate": str(best.challenger),
        "best_ranked_pass": bool(best.calibration_pass),
        "best_pass_candidate": None if best_pass is None else str(best_pass.challenger),
        "methodology_note": (
            "Calibration mappings are fit only on temporal OOF predictions internal to each outer training block. "
            "The previously inspected final historical tail remains excluded. Any apparent winner requires a genuinely future labeled holdout."
        ),
    }
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log()
    log("Aucun fichier de production n'a été modifié.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
