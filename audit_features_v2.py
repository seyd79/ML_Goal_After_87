#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit feature par feature du modèle V2 Goal After 87.

Ce script NE modifie PAS le modèle de production. Il charge le modèle V2,
le schéma exact des features et le dataset historique, puis effectue un split
chronologique pour mesurer quelles variables aident réellement sur la période
la plus récente et lesquelles semblent neutres, instables ou bruitées.

Sorties par défaut:
  outputs/feature_audit_v2/feature_audit_v2.csv
  outputs/feature_audit_v2/feature_helpful.csv
  outputs/feature_audit_v2/feature_noise_candidates.csv
  outputs/feature_audit_v2/feature_redundancy_pairs.csv
  outputs/feature_audit_v2/audit_summary.md
  outputs/feature_audit_v2/audit_metadata.json

Exécution:
  cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
  python3 audit_features_v2.py

Dataset explicite si nécessaire:
  python3 audit_features_v2.py --dataset outputs/Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = 1e-12

DEFAULT_PROJECT_ROOT = Path(os.getenv(
    "GOAL87_PROJECT_ROOT",
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
))

MODEL_CANDIDATES = [
    "outputs/model_v2/best_model_v2_refit_all.joblib",
    "outputs/model_v2/best_model_v2.joblib",
    "outputs/model_v2/model_v2.joblib",
]
SCHEMA_CANDIDATES = [
    # Schéma exact exporté avec le modèle V2 courant
    "outputs/model_v2/model_v2_schema_used.json",

    # Schéma utilisé par le pipeline historique / Deep Audit V2
    "outputs/model_schema_v2/model_schema_v2.json",

    # Ancien fallback
    "outputs/model_v2/schema.json",
]
DATASET_CANDIDATES = [
    # Emplacement réel utilisé par le pipeline V2 / Deep Audit
    "outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    "outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",

    # Emplacements alternatifs historiques
    "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    "outputs/Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
    "outputs/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
    "Historical_Dataset_Goal_After_87.csv",
    "outputs/Historical_Dataset_Goal_After_87.csv",
]
TARGET_CANDIDATES = ["goal_after_87", "actual", "target", "label", "y"]
TIME_CANDIDATES = [
    "timestamp",
    "kickoff_utc",
    "match_datetime",
    "fixture_datetime",
    "kickoff",
    "kickoff_time",
    "match_date",
    "date",
]
MATCH_ID_CANDIDATES = ["match_id", "fixture_id", "id"]


@dataclass
class Metrics:
    n: int
    positives: int
    negatives: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    logloss: float


def log(msg: str = "") -> None:
    print(msg, flush=True)


def pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{100*x:.2f}%"


def f6(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.6f}"


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def resolve_file(root: Path, explicit: Optional[str], candidates: Sequence[str], label: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"{label} introuvable: {p}")
        return p
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} introuvable. Candidats testés:\n  - " +
        "\n  - ".join(str(root / c) for c in candidates)
    )


def _find_feature_list(obj: Any) -> Optional[List[str]]:
    preferred = ("features", "feature_names", "model_features", "used_features", "columns", "schema")
    if isinstance(obj, dict):
        for k in preferred:
            v = obj.get(k)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return list(v)
        for v in obj.values():
            found = _find_feature_list(v)
            if found:
                return found
    elif isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            return list(obj)
        for v in obj:
            found = _find_feature_list(v)
            if found:
                return found
    return None


def load_schema(path: Path, model: Any) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = _find_feature_list(payload)
    if not features and hasattr(model, "feature_names_in_"):
        features = [str(x) for x in model.feature_names_in_]
    if not features:
        raise ValueError("Impossible d'extraire les features du schema ou du modèle.")
    if len(set(features)) != len(features):
        raise ValueError("Features dupliquées dans le schema.")
    return features


def _detect_delimiter(path: Path) -> str:
    """
    Détecte le séparateur réel à partir de l'en-tête.

    Le dataset V2 actuel peut porter l'extension .csv tout en étant
    réellement séparé par des tabulations. Dans ce cas, une lecture
    pandas avec le séparateur virgule casse les champs JSON de provenance,
    car leurs virgules internes sont prises pour des séparateurs.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()

    if not header:
        raise ValueError("Dataset vide.")

    counts = {
        "\t": header.count("\t"),
        ",": header.count(","),
        ";": header.count(";"),
        "|": header.count("|"),
    }

    delimiter = max(counts, key=counts.get)

    if counts[delimiter] <= 0:
        raise ValueError(
            "Impossible de détecter le séparateur. "
            f"Comptages en-tête: {counts}"
        )

    return delimiter


def _delimiter_name(delimiter: str) -> str:
    return {
        "\t": "TABULATION",
        ",": "VIRGULE",
        ";": "POINT-VIRGULE",
        "|": "PIPE",
    }.get(delimiter, repr(delimiter))


def load_dataset(path: Path) -> pd.DataFrame:
    log(f"Lecture dataset: {path}")

    delimiter = _detect_delimiter(path)
    log(f"Séparateur détecté: {_delimiter_name(delimiter)}")

    # Data integrity:
    # - aucune ligne n'est sautée silencieusement ;
    # - les erreurs de structure restent bloquantes ;
    # - engine='python' gère mieux les gros champs texte/JSON.
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            engine="python",
            encoding="utf-8-sig",
            skipinitialspace=False,
            on_bad_lines="error",
        )
    except Exception as exc:
        raise ValueError(
            "Échec de lecture structurée du dataset avec le séparateur "
            f"{_delimiter_name(delimiter)}. Aucune ligne n'a été ignorée. "
            f"Erreur originale: {exc}"
        ) from exc

    if df.empty:
        raise ValueError("Dataset vide.")

    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 50:
        raise ValueError(
            f"Parsing suspect: seulement {len(df.columns)} colonnes détectées. "
            "Le séparateur est probablement incorrect."
        )

    duplicated = pd.Index(df.columns)[
        pd.Index(df.columns).duplicated()
    ].tolist()

    if duplicated:
        raise ValueError(
            f"Colonnes dupliquées après parsing: {duplicated}"
        )

    log(
        f"Dataset chargé: {len(df):,} lignes | "
        f"{len(df.columns)} colonnes"
    )

    return df


def normalize_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        x = s.astype(int)
    elif pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
    else:
        mapping = {
            "true": 1, "false": 0, "yes": 1, "no": 0,
            "oui": 1, "non": 0, "won": 1, "lost": 0,
            "gagne": 1, "gagné": 1, "perdu": 0, "1": 1, "0": 0,
        }
        x = s.astype(str).str.strip().str.lower().map(mapping)
    if x.isna().any():
        bad = s[x.isna()].astype(str).head(10).tolist()
        raise ValueError(f"Target non convertible en 0/1. Exemples: {bad}")
    vals = set(x.astype(int).unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Target non binaire: {sorted(vals)}")
    return x.astype(int)


def coerce_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} features absentes du dataset:\n" +
            "\n".join(f"  - {f}" for f in missing)
        )
    X = df.loc[:, list(features)].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            converted = pd.to_numeric(X[c], errors="coerce")
            if X[c].notna().sum() > 0 and converted.notna().sum() == 0:
                raise ValueError(f"Feature non numérique inattendue: {c}")
            X[c] = converted
    return X.astype(float)


def _parse_time_series(series: pd.Series, column_name: str) -> Tuple[pd.Series, str]:
    """
    Parse une colonne temporelle en UTC.

    Cas pris en charge :
    1) dates ISO / texte classiques ;
    2) timestamps numériques Unix plausibles ;
    3) dates sérielles Excel, ex. 46278.104166666664.

    Le dataset V2 actuel stocke kickoff_utc sous forme de date Excel.
    L'origine Excel utilisée par pandas est 1899-12-30.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())

    # ------------------------------------------------------------------
    # Cas numérique
    # ------------------------------------------------------------------
    if numeric_ratio >= 0.80:
        vals = numeric.dropna()

        if len(vals):
            median_value = float(vals.median())

            # Excel serial dates actuelles : typiquement ~ 40 000 - 60 000.
            if 20_000 <= median_value <= 100_000:
                parsed = pd.to_datetime(
                    numeric,
                    unit="D",
                    origin="1899-12-30",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "EXCEL_SERIAL_DAYS"

            # Unix seconds, milliseconds, microseconds, nanoseconds.
            if 1e8 <= median_value < 1e11:
                parsed = pd.to_datetime(
                    numeric,
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "UNIX_SECONDS"

            if 1e11 <= median_value < 1e14:
                parsed = pd.to_datetime(
                    numeric,
                    unit="ms",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "UNIX_MILLISECONDS"

            if 1e14 <= median_value < 1e17:
                parsed = pd.to_datetime(
                    numeric,
                    unit="us",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "UNIX_MICROSECONDS"

            if 1e17 <= median_value:
                parsed = pd.to_datetime(
                    numeric,
                    unit="ns",
                    errors="coerce",
                    utc=True,
                )
                return parsed, "UNIX_NANOSECONDS"

    # ------------------------------------------------------------------
    # Cas texte / datetime classique
    # ------------------------------------------------------------------
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )
    return parsed, "DATETIME_TEXT"


def temporal_order(
    df: pd.DataFrame,
    explicit: Optional[str],
) -> Tuple[pd.DataFrame, str, bool]:
    col = explicit or first_existing_column(
        df,
        TIME_CANDIDATES,
    )

    if col is None:
        log(
            "⚠️ Pas de colonne temps détectée: "
            "ordre actuel des lignes utilisé."
        )
        return (
            df.reset_index(drop=True),
            "__row_order__",
            False,
        )

    if col not in df.columns:
        raise ValueError(
            f"Colonne temporelle absente: {col}"
        )

    parsed, parse_mode = _parse_time_series(
        df[col],
        col,
    )

    valid_ratio = float(parsed.notna().mean())

    log(
        f"Colonne temps détectée: {col} | "
        f"format={parse_mode} | "
        f"parseable={valid_ratio:.2%}"
    )

    if valid_ratio < 0.80:
        log(
            f"⚠️ Colonne '{col}' parseable à seulement "
            f"{valid_ratio:.2%}: ordre des lignes utilisé."
        )
        return (
            df.reset_index(drop=True),
            "__row_order__",
            False,
        )

    out = df.copy()
    out["__audit_time__"] = parsed
    out["__audit_row__"] = np.arange(
        len(out)
    )

    # Les lignes non parseables, si elles existent (<20 %), sont rejetées
    # explicitement de l'audit temporel plutôt que triées arbitrairement.
    invalid_mask = out["__audit_time__"].isna()

    if invalid_mask.any():
        bad_n = int(invalid_mask.sum())
        raise ValueError(
            f"{bad_n} lignes ont une date '{col}' invalide. "
            "Audit temporel interrompu pour éviter un split incohérent."
        )

    out = (
        out.sort_values(
            ["__audit_time__", "__audit_row__"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    first_dt = out["__audit_time__"].iloc[0]
    last_dt = out["__audit_time__"].iloc[-1]

    log(
        "Ordre temporel validé: "
        f"{first_dt.isoformat()} -> {last_dt.isoformat()}"
    )

    return out, col, True


def split_indices(n: int, test_fraction: float, min_test_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 50:
        raise ValueError(f"Trop peu de lignes pour l'audit: {n}")
    test_n = max(min_test_rows, int(round(n * test_fraction)))
    test_n = min(test_n, max(1, int(n * 0.40)))
    if n - test_n < 30:
        test_n = n - 30
    if test_n <= 0:
        raise ValueError("Split temporel impossible.")
    return np.arange(0, n-test_n), np.arange(n-test_n, n)


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        p = p[:, 1] if p.ndim == 2 else p.reshape(-1)
    elif hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    else:
        p = np.asarray(model.predict(X), dtype=float).reshape(-1)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Le modèle ne fournit pas de probabilités exploitables.")
    return np.clip(p.astype(float), EPS, 1-EPS)


def compute_metrics(y: np.ndarray, p: np.ndarray) -> Metrics:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    roc = float(roc_auc_score(y, p)) if pos and neg else np.nan
    ap = float(average_precision_score(y, p)) if pos and neg else np.nan
    return Metrics(
        n=n,
        positives=pos,
        negatives=neg,
        base_rate=pos/n if n else np.nan,
        roc_auc=roc,
        pr_auc=ap,
        brier=float(brier_score_loss(y, p)),
        logloss=float(log_loss(y, p, labels=[0, 1])),
    )


def percentile_ci(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def point_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    xx, yy = x[mask].astype(float), y[mask].astype(float)
    if len(xx) < 5 or xx.nunique() < 2 or yy.nunique() < 2:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def single_auc(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    xx, yy = x[mask].astype(float), y[mask].astype(int)
    if len(xx) < 10 or xx.nunique() < 2 or yy.nunique() < 2:
        return np.nan, np.nan
    raw = float(roc_auc_score(yy, xx))
    return raw, max(raw, 1-raw)


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    a = x[mask & (y == 1)].astype(float)
    b = x[mask & (y == 0)].astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    den = len(a) + len(b) - 2
    pooled = math.sqrt(max(0.0, (((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / den)))
    if pooled <= EPS:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    tr = pd.to_numeric(train, errors="coerce").dropna().astype(float)
    te = pd.to_numeric(test, errors="coerce").dropna().astype(float)
    if len(tr) < 20 or len(te) < 20 or tr.nunique() < 2:
        return np.nan
    edges = np.unique(np.quantile(tr, np.linspace(0, 1, bins+1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(tr, bins=edges)
    b, _ = np.histogram(te, bins=edges)
    pa = np.maximum(a/max(1, a.sum()), 1e-6)
    pb = np.maximum(b/max(1, b.sum()), 1e-6)
    return float(np.sum((pb-pa) * np.log(pb/pa)))


def final_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model


def extract_coefficients(model: Any, features: Sequence[str]) -> Dict[str, float]:
    est = final_estimator(model)
    if not hasattr(est, "coef_"):
        return {}
    c = np.asarray(est.coef_)
    if c.ndim == 2:
        if c.shape[0] != 1:
            return {}
        c = c[0]
    c = c.reshape(-1)
    if len(c) != len(features):
        return {}
    return {f: float(v) for f, v in zip(features, c)}


def audit_feature(
    feature: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Any,
    baseline_p: np.ndarray,
    baseline: Metrics,
    repeats: int,
    rng: np.random.Generator,
    coefficient: float,
) -> Dict[str, Any]:
    tr, te = X_train[feature], X_test[feature]
    med = float(tr.dropna().median()) if tr.notna().any() else 0.0
    raw_auc, oriented_auc = single_auc(te, y_test)

    # Neutralisation: remplace la feature par la médiane du TRAIN.
    Xo = X_test.copy()
    Xo[feature] = med
    po = predict_probability(model, Xo)
    mo = compute_metrics(y_test.values, po)

    occ_auc_drop = baseline.roc_auc - mo.roc_auc if np.isfinite(baseline.roc_auc) and np.isfinite(mo.roc_auc) else np.nan
    occ_ap_drop = baseline.pr_auc - mo.pr_auc if np.isfinite(baseline.pr_auc) and np.isfinite(mo.pr_auc) else np.nan
    occ_brier_inc = mo.brier - baseline.brier
    prob_change = float(np.mean(np.abs(baseline_p - po)))

    # Permutation répétée sur le TEST récent.
    original = te.to_numpy(copy=True)
    auc_drops, ap_drops, brier_incs = [], [], []
    for _ in range(repeats):
        perm = original.copy()
        rng.shuffle(perm)
        Xp = X_test.copy()
        Xp[feature] = perm
        pp = predict_probability(model, Xp)
        mp = compute_metrics(y_test.values, pp)
        if np.isfinite(baseline.roc_auc) and np.isfinite(mp.roc_auc):
            auc_drops.append(baseline.roc_auc - mp.roc_auc)
        if np.isfinite(baseline.pr_auc) and np.isfinite(mp.pr_auc):
            ap_drops.append(baseline.pr_auc - mp.pr_auc)
        brier_incs.append(mp.brier - baseline.brier)

    auc_mean = float(np.mean(auc_drops)) if auc_drops else np.nan
    auc_std = float(np.std(auc_drops, ddof=1)) if len(auc_drops) > 1 else np.nan
    auc_lo, auc_hi = percentile_ci(auc_drops)
    ap_mean = float(np.mean(ap_drops)) if ap_drops else np.nan
    ap_lo, ap_hi = percentile_ci(ap_drops)
    br_mean = float(np.mean(brier_incs)) if brier_incs else np.nan
    br_lo, br_hi = percentile_ci(brier_incs)
    positive_share = float(np.mean(np.asarray(auc_drops) > 0)) if auc_drops else np.nan

    miss_train = float(tr.isna().mean())
    miss_test = float(te.isna().mean())
    nuniq_train = int(tr.nunique(dropna=True))
    train_std = float(tr.dropna().std(ddof=0)) if tr.notna().any() else np.nan
    drift = psi(tr, te)

    near_constant = nuniq_train <= 1 or (np.isfinite(train_std) and train_std <= 1e-12)
    data_risk = miss_train >= 0.60 or miss_test >= 0.60 or near_constant

    if data_risk:
        status = "DATA_QUALITY_RISK"
    elif np.isfinite(auc_lo) and auc_lo > 0 and auc_mean > 0.001:
        status = "ROBUST_HELPFUL"
    elif np.isfinite(auc_mean) and auc_mean > 0.001 and (not np.isfinite(positive_share) or positive_share >= 0.60):
        status = "HELPFUL"
    elif np.isfinite(auc_hi) and auc_hi < 0:
        status = "POTENTIAL_NOISE"
    elif np.isfinite(auc_mean) and auc_mean < -0.001 and np.isfinite(occ_auc_drop) and occ_auc_drop < 0:
        status = "POTENTIAL_NOISE"
    else:
        status = "NEUTRAL"

    drift_status = (
        "HIGH" if np.isfinite(drift) and drift >= 0.25 else
        "MODERATE" if np.isfinite(drift) and drift >= 0.10 else
        "LOW" if np.isfinite(drift) else "NA"
    )

    pos = te[y_test.values == 1].dropna()
    neg = te[y_test.values == 0].dropna()

    return {
        "feature": feature,
        "status": status,
        "coefficient": coefficient,
        "missing_train": miss_train,
        "missing_test": miss_test,
        "missing_pos_test": float(te[y_test.values == 1].isna().mean()),
        "missing_neg_test": float(te[y_test.values == 0].isna().mean()),
        "nunique_train": nuniq_train,
        "nunique_test": int(te.nunique(dropna=True)),
        "train_mean": float(tr.dropna().mean()) if tr.notna().any() else np.nan,
        "test_mean": float(te.dropna().mean()) if te.notna().any() else np.nan,
        "positive_mean_test": float(pos.mean()) if len(pos) else np.nan,
        "negative_mean_test": float(neg.mean()) if len(neg) else np.nan,
        "cohen_d_test": cohen_d(te, y_test),
        "target_corr_test": point_corr(te, y_test),
        "univariate_auc_raw_test": raw_auc,
        "univariate_auc_oriented_test": oriented_auc,
        "psi_train_vs_test": drift,
        "drift_status": drift_status,
        "occlusion_auc_drop": occ_auc_drop,
        "occlusion_pr_auc_drop": occ_ap_drop,
        "occlusion_brier_increase": occ_brier_inc,
        "mean_abs_probability_change_when_neutralized": prob_change,
        "permutation_auc_drop_mean": auc_mean,
        "permutation_auc_drop_std": auc_std,
        "permutation_auc_drop_ci95_low": auc_lo,
        "permutation_auc_drop_ci95_high": auc_hi,
        "permutation_positive_auc_share": positive_share,
        "permutation_pr_auc_drop_mean": ap_mean,
        "permutation_pr_auc_drop_ci95_low": ap_lo,
        "permutation_pr_auc_drop_ci95_high": ap_hi,
        "permutation_brier_increase_mean": br_mean,
        "permutation_brier_increase_ci95_low": br_lo,
        "permutation_brier_increase_ci95_high": br_hi,
    }


def redundancy_pairs(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = X.corr(method="spearman", min_periods=max(20, int(len(X)*0.10))).abs()
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v) and v >= threshold:
                rows.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "abs_spearman_corr_train": float(v),
                })
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_spearman_corr_train"])
    return pd.DataFrame(rows).sort_values("abs_spearman_corr_train", ascending=False).reset_index(drop=True)


def metric_md(title: str, m: Metrics) -> str:
    return (
        f"### {title}\n\n"
        f"- N: **{m.n}**\n"
        f"- Positifs: **{m.positives}**\n"
        f"- Négatifs: **{m.negatives}**\n"
        f"- Taux cible: **{pct(m.base_rate)}**\n"
        f"- ROC-AUC: **{f6(m.roc_auc)}**\n"
        f"- PR-AUC: **{f6(m.pr_auc)}**\n"
        f"- Brier: **{f6(m.brier)}**\n"
        f"- Log loss: **{f6(m.logloss)}**\n"
    )


def df_md(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_Aucune feature dans cette catégorie._\n"
    v = df.loc[:, list(cols)].copy()
    for c in v.columns:
        if pd.api.types.is_numeric_dtype(v[c]):
            v[c] = v[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    try:
        return v.to_markdown(index=False) + "\n"
    except Exception:
        return v.to_csv(index=False) + "\n"


def build_report(
    audit: pd.DataFrame,
    redundant: pd.DataFrame,
    train_m: Metrics,
    test_m: Metrics,
    model_path: Path,
    schema_path: Path,
    dataset_path: Path,
    time_col: str,
    temporal: bool,
    repeats: int,
    top_k: int,
) -> str:
    counts = audit["status"].value_counts().to_dict()
    helpful = audit[audit["status"].isin(["ROBUST_HELPFUL", "HELPFUL"])].sort_values(
        ["permutation_auc_drop_mean", "permutation_pr_auc_drop_mean"], ascending=False
    ).head(top_k)
    noise = audit[audit["status"] == "POTENTIAL_NOISE"].sort_values(
        "permutation_auc_drop_mean", ascending=True
    ).head(top_k)
    drift = audit[audit["drift_status"].isin(["HIGH", "MODERATE"])].sort_values(
        "psi_train_vs_test", ascending=False
    ).head(top_k)

    return "\n".join([
        "# AUDIT FEATURE PAR FEATURE — MODELE V2\n",
        "## Sources",
        f"- Modèle: `{model_path}`",
        f"- Schéma: `{schema_path}`",
        f"- Dataset: `{dataset_path}`",
        f"- Split temporel: **{'OUI' if temporal else 'NON'}**",
        f"- Colonne temps: `{time_col}`",
        f"- Répétitions permutation: **{repeats}**\n",
        metric_md("Performance TRAIN", train_m),
        metric_md("Performance TEST récent", test_m),
        "## Classification",
        f"- ROBUST_HELPFUL: **{counts.get('ROBUST_HELPFUL', 0)}**",
        f"- HELPFUL: **{counts.get('HELPFUL', 0)}**",
        f"- NEUTRAL: **{counts.get('NEUTRAL', 0)}**",
        f"- POTENTIAL_NOISE: **{counts.get('POTENTIAL_NOISE', 0)}**",
        f"- DATA_QUALITY_RISK: **{counts.get('DATA_QUALITY_RISK', 0)}**\n",
        "## Features les plus utiles",
        df_md(helpful, [
            "feature", "status", "permutation_auc_drop_mean",
            "permutation_auc_drop_ci95_low", "permutation_auc_drop_ci95_high",
            "permutation_pr_auc_drop_mean", "occlusion_auc_drop",
            "univariate_auc_oriented_test", "psi_train_vs_test"
        ]),
        "## Candidats bruit potentiel",
        df_md(noise, [
            "feature", "status", "permutation_auc_drop_mean",
            "permutation_auc_drop_ci95_low", "permutation_auc_drop_ci95_high",
            "occlusion_auc_drop", "occlusion_pr_auc_drop",
            "mean_abs_probability_change_when_neutralized", "psi_train_vs_test"
        ]),
        "## Dérive temporelle",
        df_md(drift, [
            "feature", "drift_status", "psi_train_vs_test",
            "missing_train", "missing_test", "permutation_auc_drop_mean"
        ]),
        "## Redondances fortes",
        df_md(redundant.head(top_k), ["feature_a", "feature_b", "abs_spearman_corr_train"]),
        """
## Interprétation

- `ROBUST_HELPFUL`: la permutation de la feature dégrade l'AUC avec un IC 95 % entièrement > 0.
- `HELPFUL`: effet positif moyen, mais encore incertain.
- `NEUTRAL`: contribution faible ou difficile à distinguer du bruit.
- `POTENTIAL_NOISE`: mélanger / neutraliser la feature tend à améliorer le modèle.
- `DATA_QUALITY_RISK`: trop de données manquantes ou feature quasi constante.

Ne supprime pas automatiquement une feature sur ce seul rapport. Le test décisif
est ensuite de réentraîner le modèle sans les candidats bruit et de comparer sur
une période strictement future.
"""
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit feature par feature du modèle V2 Goal After 87")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--model", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--output-dir", default="outputs/feature_audit_v2")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--min-test-rows", type=int, default=100)
    p.add_argument("--permutation-repeats", type=int, default=30)
    p.add_argument("--redundancy-threshold", type=float, default=0.90)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Racine projet introuvable: {root}")

    model_path = resolve_file(root, args.model, MODEL_CANDIDATES, "Modèle V2")
    schema_path = resolve_file(root, args.schema, SCHEMA_CANDIDATES, "Schéma V2")
    dataset_path = resolve_file(root, args.dataset, DATASET_CANDIDATES, "Dataset historique")
    outdir = Path(args.output_dir).expanduser()
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    log("="*80)
    log("AUDIT FEATURE PAR FEATURE — MODELE V2")
    log("="*80)
    log(f"Modèle : {model_path}")
    log(f"Schéma : {schema_path}")
    log(f"Dataset: {dataset_path}")
    log(f"Sorties: {outdir}\n")

    model = joblib.load(model_path)
    features = load_schema(schema_path, model)
    log(f"Features modèle: {len(features)}")

    df = load_dataset(dataset_path)
    target_col = args.target or first_existing_column(df, TARGET_CANDIDATES)
    if not target_col:
        raise ValueError("Target introuvable. Utilise --target NOM_COLONNE")
    log(f"Target: {target_col}")

    match_col = first_existing_column(df, MATCH_ID_CANDIDATES)
    duplicate_match_rows = 0
    if match_col:
        duplicate_match_rows = int(df[match_col].duplicated(keep=False).sum())
        if duplicate_match_rows:
            log(f"⚠️ {duplicate_match_rows} lignes appartiennent à des match_id dupliqués. Aucune suppression automatique.")

    df, time_col, temporal = temporal_order(df, args.time_col)
    y = normalize_target(df[target_col])
    X = coerce_features(df, features)
    train_idx, test_idx = split_indices(len(df), args.test_fraction, args.min_test_rows)
    Xtr, Xte = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    ytr, yte = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError("TRAIN ou TEST ne contient pas les deux classes.")

    log(f"TRAIN: {len(Xtr):,} | TEST récent: {len(Xte):,}")
    ptr = predict_probability(model, Xtr)
    pte = predict_probability(model, Xte)
    mtr = compute_metrics(ytr.values, ptr)
    mte = compute_metrics(yte.values, pte)
    log(f"TEST | ROC-AUC={f6(mte.roc_auc)} | PR-AUC={f6(mte.pr_auc)} | Brier={f6(mte.brier)} | base={pct(mte.base_rate)}\n")

    coefs = extract_coefficients(model, features)
    rng = np.random.default_rng(args.random_state)
    rows = []
    for i, feature in enumerate(features, 1):
        log(f"[{i:02d}/{len(features):02d}] {feature}")
        rows.append(audit_feature(
            feature, Xtr, Xte, yte, model, pte, mte,
            args.permutation_repeats, rng, coefs.get(feature, np.nan)
        ))

    audit = pd.DataFrame(rows)
    order = {"ROBUST_HELPFUL": 0, "HELPFUL": 1, "NEUTRAL": 2, "POTENTIAL_NOISE": 3, "DATA_QUALITY_RISK": 4}
    audit["__order__"] = audit["status"].map(order).fillna(99)
    audit = audit.sort_values(
        ["__order__", "permutation_auc_drop_mean", "permutation_pr_auc_drop_mean"],
        ascending=[True, False, False]
    ).drop(columns="__order__").reset_index(drop=True)

    redundant = redundancy_pairs(Xtr, args.redundancy_threshold)
    helpful = audit[audit["status"].isin(["ROBUST_HELPFUL", "HELPFUL"])].copy()
    noise = audit[audit["status"].isin(["POTENTIAL_NOISE", "DATA_QUALITY_RISK"])].copy()

    audit_path = outdir / "feature_audit_v2.csv"
    helpful_path = outdir / "feature_helpful.csv"
    noise_path = outdir / "feature_noise_candidates.csv"
    red_path = outdir / "feature_redundancy_pairs.csv"
    md_path = outdir / "audit_summary.md"
    meta_path = outdir / "audit_metadata.json"

    audit.to_csv(audit_path, index=False)
    helpful.to_csv(helpful_path, index=False)
    noise.to_csv(noise_path, index=False)
    redundant.to_csv(red_path, index=False)
    md_path.write_text(build_report(
        audit, redundant, mtr, mte, model_path, schema_path, dataset_path,
        time_col, temporal, args.permutation_repeats, args.top_k
    ), encoding="utf-8")

    metadata = {
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "dataset_path": str(dataset_path),
        "target_column": target_col,
        "time_column": time_col,
        "temporal_split": temporal,
        "feature_count": len(features),
        "row_count": len(df),
        "train_rows": len(Xtr),
        "test_rows": len(Xte),
        "duplicate_match_id_rows": duplicate_match_rows,
        "permutation_repeats": args.permutation_repeats,
        "random_state": args.random_state,
        "train_metrics": asdict(mtr),
        "test_metrics": asdict(mte),
        "status_counts": {str(k): int(v) for k, v in audit["status"].value_counts().items()},
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n" + "="*80)
    log("AUDIT TERMINÉ")
    log("="*80)
    for status in ["ROBUST_HELPFUL", "HELPFUL", "NEUTRAL", "POTENTIAL_NOISE", "DATA_QUALITY_RISK"]:
        log(f"{status:20s}: {int((audit['status'] == status).sum())}")
    log("\nFichiers générés:")
    for p in [audit_path, helpful_path, noise_path, red_path, md_path, meta_path]:
        log(f"  - {p}")
    log("\nNe supprime aucune feature automatiquement: le prochain test doit être un réentraînement contrôlé sans les candidats bruit, puis validation hors-échantillon future.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrompu.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERREUR AUDIT FEATURE V2: {exc}", file=sys.stderr)
        raise
