#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v3c_specialized_contexts.py

V3-C — ÉTUDE DE MODÈLES SPÉCIALISÉS PAR CONTEXTE @87'
=======================================================

OBJECTIF
--------
Tester, SANS modifier le scanner ni le modèle de production, si des modèles
entraînés uniquement dans des contextes football pré-déclarés font mieux que
un V3-C général réentraîné sur la même période historique.

CONTEXTES GELÉS AVANT L'EXÉCUTION
---------------------------------
1) SPEC_SCORE_2_0
   score à 87' = 2-0

2) SPEC_SOT2
   sot_80_87 >= 2

3) SPEC_SHOTS35
   home_shots_87 + away_shots_87 >= 35

4) SPEC_RECENT_SOT
   minutes_since_last_shot_on_target <= 1

5) SPEC_HIGH_PRESSURE
   sot_80_87 >= 2 ET minutes_since_last_shot_on_target <= 1

MÉTHODE
-------
- Unité statistique : 1 match = 1 snapshot exact 87'.
- Targets résolus uniquement.
- Schéma V3-C exact : 38 features lues depuis
  outputs/model_v3_experiments/schemas/V3_C_AGGRESSIVE_schema.json.
- Par défaut, seules les lignes 38/38 sont utilisées pour l'entraînement.
- Split temporel global 70/30, identique pour tous les contextes.
- BASELINE_GLOBAL : clone de l'architecture V3-C, entraîné sur TOUS les matchs
  du TRAIN puis évalué uniquement sur le contexte du TEST.
- SPECIALIZED : même architecture / mêmes 38 features, mais entraînée seulement
  sur les matchs du contexte dans TRAIN, puis évaluée sur exactement les mêmes
  matchs de contexte dans TEST.
- Aucun refit-all n'est généré : le bloc récent a déjà été observé et est donc
  consommé pour cette étude.

MÉTRIQUES
---------
ROC-AUC, PR-AUC, Brier, LogLoss, ECE(10), moyenne de probabilité, calibration.
Un bootstrap apparié mesure les deltas SPECIALIZED - GLOBAL sur les mêmes matchs.

IMPORTANT
---------
- Ce script ne modifie AUCUN fichier de production.
- Il ne remplace pas live_scanner_engine.py.
- Il ne crée pas de candidat refit-all déployable.
- Un résultat favorable doit encore être validé sur un futur holdout pristine /
  shadow avant toute règle opérationnelle.

UTILISATION
-----------
python3 train_v3c_specialized_contexts.py

Sorties :
outputs/v3c_specialized_contexts/
  specialized_context_sizes.csv
  specialized_metrics.csv
  specialized_comparison.csv
  specialized_predictions_test.csv
  specialized_bootstrap.csv
  specialized_data_audit.csv
  specialized_exclusions.csv
  specialized_report.md
  experiment_manifest.json
  split_models/
    GLOBAL_V3C_RETRAIN.joblib
    SPEC_*.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "outputs" / "model_v3_experiments"
DEFAULT_TEMPLATE = MODEL_DIR / "candidates_refit_all" / "V3_C_AGGRESSIVE_refit_all.joblib"
DEFAULT_SCHEMA = MODEL_DIR / "schemas" / "V3_C_AGGRESSIVE_schema.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "v3c_specialized_contexts"

DEFAULT_VALIDATION_FRACTION = 0.30
DEFAULT_MIN_FEATURE_COVERAGE = 38
DEFAULT_BOOTSTRAP_REPS = 1000
RANDOM_STATE = 42
EPS = 1e-12

CONTEXTS = [
    "SPEC_SCORE_2_0",
    "SPEC_SOT2",
    "SPEC_SHOTS35",
    "SPEC_RECENT_SOT",
    "SPEC_HIGH_PRESSURE",
]

# Metadata / context fields read in addition to the 38 model features.
FIELD_ALIASES: Dict[str, Sequence[str]] = {
    "match_id": ["match_id", "fixture_id", "id"],
    "target": ["goal_after_87", "target", "label", "y"],
    "target_status": ["target_status", "label_status_for_backtest", "status"],
    "target_class": ["target_resolution_class", "resolution_class"],
    "target_confidence": ["target_confidence", "confidence"],
    "label_conflict": ["label_conflict", "target_conflict"],
    "minute": ["minute", "snapshot_minute"],
    "score_home": ["score_87_home", "score_home_snapshot", "score_home"],
    "score_away": ["score_87_away", "score_away_snapshot", "score_away"],
    "time": [
        "kickoff_utc", "kickoff_time_utc", "match_datetime", "fixture_datetime",
        "kickoff", "kickoff_time", "match_date", "date",
        "snapshot_timestamp_utc", "timestamp", "collected_at_utc",
    ],
    "competition_id": ["competition_id", "league_id", "competition"],
    "season_id": ["season_id"],
    "home_team": ["home_team", "home_name"],
    "away_team": ["away_team", "away_name"],
    "sot_80_87": ["sot_80_87"],
    "home_shots_87": ["home_shots_87"],
    "away_shots_87": ["away_shots_87"],
    "mins_since_last_sot": ["minutes_since_last_shot_on_target"],
    "last_sot_minute": ["last_shot_on_target_minute"],
}


# =============================================================================
# HELPERS
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def norm_name(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean(v).lower()).strip("_")


def safe_num(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            return float(v)
        except Exception:
            return float("nan")
    s = clean(v)
    if not s:
        return float("nan")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    # decimal comma only if no dot is already present
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        x = float(s)
        return x / 100.0 if pct else x
    except Exception:
        return float("nan")


def safe_int(v: Any) -> Optional[int]:
    x = safe_num(v)
    if not np.isfinite(x):
        return None
    return int(round(x))


def boolish(v: Any) -> Optional[bool]:
    s = clean(v).lower()
    if s in {"1", "true", "yes", "oui", "y"}:
        return True
    if s in {"0", "false", "no", "non", "n"}:
        return False
    return None


def normalize_target(v: Any) -> Optional[int]:
    s = clean(v).lower()
    if s in {"1", "1.0", "true", "yes", "oui", "positive", "pos", "goal"}:
        return 1
    if s in {"0", "0.0", "false", "no", "non", "negative", "neg", "no_goal"}:
        return 0
    x = safe_num(v)
    if np.isfinite(x) and x in {0.0, 1.0}:
        return int(x)
    return None


def parse_time(v: Any) -> pd.Timestamp:
    s = clean(v)
    if not s:
        return pd.NaT
    x = safe_num(s)
    if np.isfinite(x) and 20000 <= x <= 80000 and re.fullmatch(r"[0-9.,]+", s):
        try:
            return pd.Timestamp("1899-12-30", tz="UTC") + pd.to_timedelta(x, unit="D")
        except Exception:
            pass
    return pd.to_datetime(s, utc=True, errors="coerce")


def label_resolved(rec: Mapping[str, Any]) -> bool:
    if boolish(rec.get("label_conflict")) is True:
        return False
    status = norm_name(rec.get("target_status", ""))
    klass = norm_name(rec.get("target_class", ""))
    conf = norm_name(rec.get("target_confidence", ""))
    bad = ("pending", "ambiguous", "unresolved", "to_label", "unknown", "conflict")
    if any(x in status for x in bad):
        return False
    if any(x in klass for x in bad):
        return False
    if conf in {"low", "very_low"}:
        return False
    return True


def source_priority(path: Path) -> int:
    """Prefer historical feature rows for semantic consistency."""
    n = path.name.lower()
    if "fixed" in n:
        return 120
    if "core_ready" in n:
        return 110
    if "raw_v2" in n:
        return 100
    if "labeled_v3" in n:
        return 90
    if "labeled_v2" in n:
        return 85
    if "labeled" in n:
        return 80
    if "live_feature_snapshots" in n:
        return 70
    return 50


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_Aucune ligne._"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows)

    def fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows = [[fmt(v) for v in row] for row in view.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


# =============================================================================
# CLI / FILE DISCOVERY
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Étude V3-C spécialisée par contexte, split temporel strict.")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Artefact V3-C servant uniquement de template sklearn.clone().")
    p.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Schéma exact des 38 features V3-C.")
    p.add_argument("--source", action="append", default=[], help="Source supplémentaire CSV/TSV/XLSX; répétable.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    p.add_argument("--min-feature-coverage", type=int, default=DEFAULT_MIN_FEATURE_COVERAGE, help="Minimum de features V3-C observées sur 38. Défaut=38.")
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--n-jobs", type=int, default=-1, help="Appliqué au classifier si le paramètre n_jobs existe.")
    return p.parse_args()


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else BASE_DIR / p


def discover_sources(extra: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    hist = BASE_DIR / "outputs" / "historical_dataset_v2"
    live = BASE_DIR / "outputs" / "history"
    if hist.exists():
        candidates += sorted(hist.glob("Historical_Dataset_Goal_After_87_*.csv"))
    if live.exists():
        candidates += sorted(live.glob("live_feature_snapshots*.csv"))
    for raw in extra:
        p = resolve_path(raw)
        candidates.append(p)

    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.absolute())
        if key in seen or not p.exists() or not p.is_file():
            continue
        seen.add(key)
        out.append(p)
    return out


def load_schema(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Schéma V3-C introuvable : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        features = [clean(x) for x in payload if clean(x)]
    elif isinstance(payload, Mapping):
        candidate = (
            payload.get("features")
            or payload.get("features_used")
            or payload.get("core_features")
            or payload.get("feature_names")
        )
        features = [clean(x) for x in candidate] if isinstance(candidate, list) else []
    else:
        features = []
    if len(features) != 38:
        raise ValueError(f"V3-C exige exactement 38 features; schéma lu={len(features)}")
    if len(set(features)) != 38:
        raise ValueError("Le schéma V3-C contient des features dupliquées.")
    return features


# =============================================================================
# ROBUST SOURCE READER
# =============================================================================

def detect_header(path: Path) -> Tuple[str, int, List[str]]:
    delimiters = [",", "\t", ";", "|"]
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        lines = []
        for _ in range(120):
            line = fh.readline()
            if line == "":
                break
            lines.append(line.rstrip("\r\n"))

    for line_no, line in enumerate(lines, start=1):
        if "match_id" not in line:
            continue
        for delim in delimiters:
            try:
                header = [clean(x) for x in next(csv.reader([line], delimiter=delim))]
            except Exception:
                continue
            hn = {norm_name(x) for x in header}
            has_id = "match_id" in hn or "fixture_id" in hn
            has_target = any(norm_name(x) in hn for x in FIELD_ALIASES["target"])
            has_sh = any(norm_name(x) in hn for x in FIELD_ALIASES["score_home"])
            has_sa = any(norm_name(x) in hn for x in FIELD_ALIASES["score_away"])
            if has_id and has_target and has_sh and has_sa:
                return delim, line_no, header
    raise ValueError("header match_id + target + score@87 non détecté")


def map_columns(header: Sequence[str], features: Sequence[str]) -> Dict[str, Optional[str]]:
    by_norm = {norm_name(h): h for h in header}
    out: Dict[str, Optional[str]] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        out[canonical] = next((by_norm[norm_name(a)] for a in aliases if norm_name(a) in by_norm), None)
    for f in features:
        out[f"feature::{f}"] = by_norm.get(norm_name(f))
    return out


@dataclass
class SourceAudit:
    source: str
    delimiter: str = ""
    header_line: int = 0
    physical_rows: int = 0
    parsed_rows: int = 0
    exact87_rows: int = 0
    resolved_rows: int = 0
    valid_score_rows: int = 0
    accepted_rows: int = 0
    malformed_rows: int = 0
    non87_rows: int = 0
    unresolved_rows: int = 0
    bad_score_rows: int = 0
    feature_columns_present: int = 0
    note: str = ""


def row_from_values(
    values: Mapping[str, Any],
    path: Path,
    features: Sequence[str],
    line_no: int,
) -> Optional[Dict[str, Any]]:
    rec: Dict[str, Any] = {
        "__source": str(path),
        "__priority": source_priority(path),
        "__line": int(line_no),
    }
    rec.update(values)

    mid = clean(rec.get("match_id"))
    if not mid or mid.lower() == "match_id":
        return None

    y = normalize_target(rec.get("target"))
    sh = safe_int(rec.get("score_home"))
    sa = safe_int(rec.get("score_away"))
    if y is None or not label_resolved(rec):
        return None
    if sh is None or sa is None or sh < 0 or sa < 0:
        return None

    rec["match_id"] = mid
    rec["__y"] = int(y)
    rec["__score_home"] = int(sh)
    rec["__score_away"] = int(sa)
    rec["__time"] = parse_time(rec.get("time"))

    # Numeric context values.
    for c in ["sot_80_87", "home_shots_87", "away_shots_87", "mins_since_last_sot", "last_sot_minute"]:
        rec[c] = safe_num(rec.get(c))

    # Exact 38 model inputs.
    coverage = 0
    for f in features:
        x = safe_num(rec.get(f))
        rec[f] = x
        if np.isfinite(x):
            coverage += 1
    rec["__feature_coverage"] = int(coverage)
    return rec


def read_csv_source(path: Path, features: Sequence[str]) -> Tuple[List[Dict[str, Any]], SourceAudit, List[Dict[str, Any]]]:
    delim, header_line, header = detect_header(path)
    cols = map_columns(header, features)
    audit = SourceAudit(
        source=str(path),
        delimiter={",":"comma", "\t":"tab", ";":"semicolon", "|":"pipe"}.get(delim, delim),
        header_line=header_line,
        feature_columns_present=sum(cols.get(f"feature::{f}") is not None for f in features),
    )
    required = ["match_id", "target", "score_home", "score_away"]
    missing = [x for x in required if cols.get(x) is None]
    if missing:
        raise ValueError("colonnes requises absentes: " + ", ".join(missing))

    idx = {h: i for i, h in enumerate(header)}
    real_needed = [v for v in cols.values() if v is not None]
    max_idx = max(idx[v] for v in real_needed)
    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for _ in range(header_line - 1):
            fh.readline()
        reader = csv.reader(fh, delimiter=delim)
        try:
            parsed_header = [clean(x) for x in next(reader)]
        except StopIteration:
            return rows, audit, exclusions
        if [norm_name(x) for x in parsed_header] != [norm_name(x) for x in header]:
            raise RuntimeError("header parsé différent du header détecté")

        for line_no, raw in enumerate(reader, start=header_line + 1):
            audit.physical_rows += 1
            if not raw:
                continue
            if len(raw) <= max_idx:
                audit.malformed_rows += 1
                exclusions.append({"source":str(path), "line":line_no, "match_id":"", "reason":"row_too_short"})
                continue
            audit.parsed_rows += 1

            values: Dict[str, Any] = {}
            for canonical, real in cols.items():
                if real is None:
                    continue
                val = raw[idx[real]]
                if canonical.startswith("feature::"):
                    values[canonical.split("::", 1)[1]] = val
                else:
                    values[canonical] = val

            # Exact 87 only. If no minute column exists, score_87 fields imply 87.
            minute = safe_int(values.get("minute"))
            if cols.get("minute") is not None and minute != 87:
                audit.non87_rows += 1
                continue
            audit.exact87_rows += 1

            y = normalize_target(values.get("target"))
            if y is None or not label_resolved(values):
                audit.unresolved_rows += 1
                continue
            audit.resolved_rows += 1

            sh = safe_int(values.get("score_home"))
            sa = safe_int(values.get("score_away"))
            if sh is None or sa is None or sh < 0 or sa < 0:
                audit.bad_score_rows += 1
                continue
            audit.valid_score_rows += 1

            rec = row_from_values(values, path, features, line_no)
            if rec is not None:
                rows.append(rec)
                audit.accepted_rows += 1

    return rows, audit, exclusions


def read_excel_source(path: Path, features: Sequence[str]) -> Tuple[List[Dict[str, Any]], SourceAudit, List[Dict[str, Any]]]:
    df = pd.read_excel(path)
    header = [clean(x) for x in df.columns]
    cols = map_columns(header, features)
    audit = SourceAudit(
        source=str(path), delimiter="excel", header_line=1,
        physical_rows=len(df), parsed_rows=len(df),
        feature_columns_present=sum(cols.get(f"feature::{f}") is not None for f in features),
    )
    required = ["match_id", "target", "score_home", "score_away"]
    if any(cols.get(x) is None for x in required):
        raise ValueError("colonnes requises absentes")

    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for i, s in df.iterrows():
        values: Dict[str, Any] = {}
        for canonical, real in cols.items():
            if real is None:
                continue
            val = s.get(real, "")
            if canonical.startswith("feature::"):
                values[canonical.split("::",1)[1]] = val
            else:
                values[canonical] = val

        minute = safe_int(values.get("minute"))
        if cols.get("minute") is not None and minute != 87:
            audit.non87_rows += 1
            continue
        audit.exact87_rows += 1
        y = normalize_target(values.get("target"))
        if y is None or not label_resolved(values):
            audit.unresolved_rows += 1
            continue
        audit.resolved_rows += 1
        sh = safe_int(values.get("score_home")); sa = safe_int(values.get("score_away"))
        if sh is None or sa is None or sh < 0 or sa < 0:
            audit.bad_score_rows += 1
            continue
        audit.valid_score_rows += 1
        rec = row_from_values(values, path, features, int(i)+2)
        if rec is not None:
            rows.append(rec); audit.accepted_rows += 1
    return rows, audit, exclusions


def read_source(path: Path, features: Sequence[str]) -> Tuple[List[Dict[str, Any]], SourceAudit, List[Dict[str, Any]]]:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        return read_excel_source(path, features)
    return read_csv_source(path, features)


def merge_matches(rows: Sequence[Dict[str, Any]], features: Sequence[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[clean(r.get("match_id"))].append(r)

    merged: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for mid, group in grouped.items():
        ys = {int(r["__y"]) for r in group}
        scores = {(int(r["__score_home"]), int(r["__score_away"])) for r in group}
        if len(ys) != 1:
            exclusions.append({"match_id":mid, "reason":"target_conflict_across_sources", "details":str(sorted(ys))})
            continue
        if len(scores) != 1:
            exclusions.append({"match_id":mid, "reason":"score87_conflict_across_sources", "details":str(sorted(scores))})
            continue

        # IMPORTANT: do not mix model features across sources. Pick one physical
        # row with maximum V3-C coverage, then historical-source priority.
        ranked = sorted(
            group,
            key=lambda r: (int(r.get("__feature_coverage", 0)), int(r.get("__priority", 0))),
            reverse=True,
        )
        best = dict(ranked[0])

        # Only metadata may be filled from another concordant source.
        if pd.isna(best.get("__time")):
            for r in ranked[1:]:
                if not pd.isna(r.get("__time")):
                    best["__time"] = r.get("__time")
                    break
        for field in ["competition_id", "season_id", "home_team", "away_team"]:
            if not clean(best.get(field)):
                for r in ranked[1:]:
                    if clean(r.get(field)):
                        best[field] = r.get(field)
                        break

        best["__sources_count"] = len(group)
        best["__sources"] = "|".join(sorted({Path(clean(r.get("__source"))).name for r in group}))
        merged.append(best)

    df = pd.DataFrame(merged)
    if df.empty:
        raise RuntimeError("Aucun match exact87 résolu après dédoublonnage.")
    return df, exclusions


# =============================================================================
# CONTEXT MASKS
# =============================================================================

def add_derived_context_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sh = pd.to_numeric(out["__score_home"], errors="coerce")
    sa = pd.to_numeric(out["__score_away"], errors="coerce")
    sot = pd.to_numeric(out.get("sot_80_87", np.nan), errors="coerce")
    hshots = pd.to_numeric(out.get("home_shots_87", np.nan), errors="coerce")
    ashots = pd.to_numeric(out.get("away_shots_87", np.nan), errors="coerce")
    mins = pd.to_numeric(out.get("mins_since_last_sot", np.nan), errors="coerce")
    last = pd.to_numeric(out.get("last_sot_minute", np.nan), errors="coerce")
    mins = mins.where(mins.notna(), 87.0 - last)

    out["__total_shots_87"] = hshots + ashots
    out["__mins_since_last_sot_context"] = mins
    out["__ctx_SPEC_SCORE_2_0"] = (sh == 2) & (sa == 0)
    out["__ctx_SPEC_SOT2"] = sot >= 2
    out["__ctx_SPEC_SHOTS35"] = out["__total_shots_87"] >= 35
    out["__ctx_SPEC_RECENT_SOT"] = mins <= 1
    out["__ctx_SPEC_HIGH_PRESSURE"] = (sot >= 2) & (mins <= 1)
    return out


def context_mask(df: pd.DataFrame, name: str) -> pd.Series:
    col = f"__ctx_{name}"
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


# =============================================================================
# MODEL / METRICS
# =============================================================================

def tune_n_jobs(model: Any, n_jobs: int) -> Any:
    """Preserve architecture; only alter parallel execution where supported."""
    try:
        params = model.get_params(deep=True)
    except Exception:
        return model
    updates = {}
    # Pipeline convention and bare estimators.
    for key in params:
        if key == "n_jobs" or key.endswith("__n_jobs"):
            updates[key] = n_jobs
    if updates:
        try:
            model.set_params(**updates)
        except Exception:
            pass
    return model


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.clip(p[:, 1].astype(float), EPS, 1-EPS)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        return np.clip(p, EPS, 1-EPS)
    raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")


def ece_10(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (p < hi if i < bins-1 else p <= hi)
        if not np.any(mask):
            continue
        out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def metric_block(y: Sequence[int], p: Sequence[float]) -> Dict[str, float]:
    yv = np.asarray(y, dtype=int)
    pv = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    prev = float(yv.mean()) if len(yv) else float("nan")
    roc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
    pr = float(average_precision_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
    return {
        "n": int(len(yv)),
        "positives": int(yv.sum()),
        "positive_rate": prev,
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift_vs_prevalence": pr - prev if np.isfinite(pr) else float("nan"),
        "brier": float(brier_score_loss(yv, pv)) if len(yv) else float("nan"),
        "logloss": float(log_loss(yv, pv, labels=[0,1])) if len(yv) else float("nan"),
        "ece_10": ece_10(yv, pv),
        "mean_probability": float(pv.mean()) if len(pv) else float("nan"),
        "calibration_gap_pp": (float(pv.mean()) - prev) * 100 if len(pv) else float("nan"),
    }


def paired_bootstrap(
    y: np.ndarray,
    p_global: np.ndarray,
    p_spec: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    if reps <= 0 or len(y) < 5:
        return {}
    rng = np.random.default_rng(seed)
    deltas: Dict[str, List[float]] = {"roc":[], "pr":[], "brier":[], "logloss":[]}
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pg = p_global[idx]
        ps = p_spec[idx]
        if len(np.unique(yy)) > 1:
            try:
                deltas["roc"].append(float(roc_auc_score(yy, ps) - roc_auc_score(yy, pg)))
                deltas["pr"].append(float(average_precision_score(yy, ps) - average_precision_score(yy, pg)))
            except Exception:
                pass
        # Positive = specialized better for Brier/LogLoss.
        deltas["brier"].append(float(brier_score_loss(yy, pg) - brier_score_loss(yy, ps)))
        deltas["logloss"].append(float(log_loss(yy, pg, labels=[0,1]) - log_loss(yy, ps, labels=[0,1])))

    out: Dict[str, Any] = {"bootstrap_reps": int(reps)}
    for key, vals in deltas.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        out[f"delta_{key}_ci95_low"] = float(np.quantile(arr, 0.025))
        out[f"delta_{key}_ci95_high"] = float(np.quantile(arr, 0.975))
        out[f"p_{key}_improves"] = float(np.mean(arr > 0))
    return out


def candidate_label(comp: Mapping[str, Any]) -> str:
    n_train = safe_num(comp.get("context_train_n"))
    n_test = safe_num(comp.get("context_test_n"))
    d_roc = safe_num(comp.get("delta_roc_auc_spec_minus_global"))
    d_pr = safe_num(comp.get("delta_pr_auc_spec_minus_global"))
    d_brier = safe_num(comp.get("brier_improvement_global_minus_spec"))
    d_ll = safe_num(comp.get("logloss_improvement_global_minus_spec"))
    spec_roc = safe_num(comp.get("spec_roc_auc"))
    p_roc = safe_num(comp.get("p_roc_improves"))
    p_brier = safe_num(comp.get("p_brier_improves"))

    enough = np.isfinite(n_train) and np.isfinite(n_test) and n_train >= 60 and n_test >= 25
    directions = [d_roc > 0 if np.isfinite(d_roc) else False,
                  d_pr > 0 if np.isfinite(d_pr) else False,
                  d_brier > 0 if np.isfinite(d_brier) else False,
                  d_ll > 0 if np.isfinite(d_ll) else False]
    improved = sum(directions)

    strong = (
        enough and improved == 4 and np.isfinite(spec_roc) and spec_roc >= 0.62
        and np.isfinite(d_roc) and d_roc >= 0.03
        and np.isfinite(p_roc) and p_roc >= 0.80
        and np.isfinite(p_brier) and p_brier >= 0.80
    )
    promising = (
        enough and improved >= 3 and np.isfinite(spec_roc) and spec_roc >= 0.58
        and ((np.isfinite(p_roc) and p_roc >= 0.65) or (np.isfinite(p_brier) and p_brier >= 0.65))
    )
    return "STRONG" if strong else ("PROMISING" if promising else "FAIL")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    if not 0.15 <= args.validation_fraction <= 0.45:
        raise ValueError("--validation-fraction doit être entre 0.15 et 0.45")
    if not 1 <= args.min_feature_coverage <= 38:
        raise ValueError("--min-feature-coverage doit être entre 1 et 38")

    template_path = resolve_path(args.template)
    schema_path = resolve_path(args.schema)
    out_dir = resolve_path(args.output_dir)
    split_models_dir = out_dir / "split_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_models_dir.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        raise FileNotFoundError(f"Template V3-C introuvable : {template_path}")
    features = load_schema(schema_path)
    template = joblib.load(template_path)

    print("="*104)
    print("V3-C — SPECIALIZED CONTEXT MODELS @87'")
    print("="*104)
    print(f"Template : {template_path}")
    print(f"Schéma   : {schema_path} | {len(features)} features")
    print(f"Coverage : >= {args.min_feature_coverage}/38")
    print("Mode     : RESEARCH ONLY — SPLIT TEMPOREL — AUCUN REFIT-ALL / AUCUN DEPLOIEMENT")

    sources = discover_sources(args.source)
    if not sources:
        raise FileNotFoundError("Aucune source historique/snapshot trouvée.")

    all_rows: List[Dict[str, Any]] = []
    audits: List[SourceAudit] = []
    exclusions: List[Dict[str, Any]] = []
    print(f"Sources détectées : {len(sources)}")
    for p in sources:
        try:
            rows, audit, exc = read_source(p, features)
            all_rows.extend(rows)
            audits.append(audit)
            exclusions.extend(exc)
            print(
                f"[OK] {p.name:<58s} | accepted87={audit.accepted_rows:>5d} | "
                f"feature_cols={audit.feature_columns_present:>2d}/38 | unresolved={audit.unresolved_rows:>5d} | "
                f"non87={audit.non87_rows:>5d}"
            )
        except Exception as exc:
            audits.append(SourceAudit(source=str(p), note=f"SKIPPED: {type(exc).__name__}: {exc}"))
            print(f"[SKIP] {p.name}: {type(exc).__name__}: {exc}")

    if not all_rows:
        raise RuntimeError("Aucune ligne exact87 résolue exploitable.")

    universe, dup_exc = merge_matches(all_rows, features)
    exclusions.extend(dup_exc)
    universe = add_derived_context_values(universe)

    # Require usable time and requested model-feature coverage.
    universe["__feature_coverage"] = pd.to_numeric(universe["__feature_coverage"], errors="coerce").fillna(0).astype(int)
    time_ok = universe["__time"].notna()
    coverage_ok = universe["__feature_coverage"] >= args.min_feature_coverage
    model_ready = universe.loc[time_ok & coverage_ok].copy()
    model_ready = model_ready.sort_values(["__time", "match_id"], kind="stable").reset_index(drop=True)

    dropped_time = int((~time_ok).sum())
    dropped_cov = int((time_ok & ~coverage_ok).sum())
    if len(model_ready) < 200:
        raise RuntimeError(f"Seulement {len(model_ready)} matchs model-ready; étude spécialisée refusée.")

    n = len(model_ready)
    split = int(round(n * (1 - args.validation_fraction)))
    split = min(max(split, 1), n-1)
    train = model_ready.iloc[:split].copy()
    test = model_ready.iloc[split:].copy()

    # Ensure both classes globally.
    if train["__y"].nunique() < 2 or test["__y"].nunique() < 2:
        raise RuntimeError("Split global mono-classe; comparaison impossible.")

    X_train_global = train.loc[:, features].astype(float)
    y_train_global = train["__y"].astype(int)

    global_model = tune_n_jobs(clone(template), args.n_jobs)
    global_model.fit(X_train_global, y_train_global)
    joblib.dump(global_model, split_models_dir / "GLOBAL_V3C_RETRAIN.joblib")

    context_sizes: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    bootstrap_rows: List[Dict[str, Any]] = []

    for i, ctx in enumerate(CONTEXTS, start=1):
        tr = train.loc[context_mask(train, ctx)].copy()
        te = test.loc[context_mask(test, ctx)].copy()
        context_sizes.append({
            "context":ctx,
            "train_n":len(tr), "train_positives":int(tr["__y"].sum()) if len(tr) else 0,
            "train_positive_rate":float(tr["__y"].mean()) if len(tr) else float("nan"),
            "test_n":len(te), "test_positives":int(te["__y"].sum()) if len(te) else 0,
            "test_positive_rate":float(te["__y"].mean()) if len(te) else float("nan"),
        })

        print()
        print(f"[{i}/{len(CONTEXTS)}] {ctx} | train={len(tr)} | test={len(te)}")

        if len(tr) < 20 or len(te) < 10 or tr["__y"].nunique() < 2 or te["__y"].nunique() < 2:
            comparison_rows.append({
                "context":ctx,
                "context_train_n":len(tr),
                "context_test_n":len(te),
                "candidate_gate":"INSUFFICIENT_DATA",
                "note":"minimum technique 20 train / 10 test et 2 classes non atteint",
            })
            print("  -> INSUFFICIENT_DATA")
            continue

        Xtr = tr.loc[:, features].astype(float)
        ytr = tr["__y"].astype(int)
        Xte = te.loc[:, features].astype(float)
        yte = te["__y"].astype(int)

        # Fair baseline: same global model, same exact context test rows.
        p_global = positive_probability(global_model, Xte)

        spec_model = tune_n_jobs(clone(template), args.n_jobs)
        spec_model.fit(Xtr, ytr)
        p_spec = positive_probability(spec_model, Xte)
        joblib.dump(spec_model, split_models_dir / f"{ctx}.joblib")

        m_global = metric_block(yte.to_numpy(), p_global)
        m_spec = metric_block(yte.to_numpy(), p_spec)

        # Constant baseline is train-context prevalence: a calibration sanity check.
        train_prev = float(ytr.mean())
        p_const = np.full(len(yte), np.clip(train_prev, EPS, 1-EPS), dtype=float)
        m_const = metric_block(yte.to_numpy(), p_const)

        for model_name, met in [
            ("GLOBAL_V3C_RETRAIN", m_global),
            ("SPECIALIZED", m_spec),
            ("CONTEXT_TRAIN_PREVALENCE_CONSTANT", m_const),
        ]:
            rec = {"context":ctx, "model":model_name}
            rec.update(met)
            metric_rows.append(rec)

        boot = paired_bootstrap(
            yte.to_numpy(dtype=int), p_global, p_spec,
            reps=args.bootstrap_reps,
            seed=args.random_state + i * 997,
        )
        boot_rec = {"context":ctx, **boot}
        bootstrap_rows.append(boot_rec)

        comp: Dict[str, Any] = {
            "context":ctx,
            "context_train_n":len(tr),
            "context_train_positive_rate":train_prev,
            "context_test_n":len(te),
            "context_test_positive_rate":float(yte.mean()),
            "global_roc_auc":m_global["roc_auc"],
            "spec_roc_auc":m_spec["roc_auc"],
            "delta_roc_auc_spec_minus_global":m_spec["roc_auc"] - m_global["roc_auc"] if np.isfinite(m_spec["roc_auc"]) and np.isfinite(m_global["roc_auc"]) else float("nan"),
            "global_pr_auc":m_global["pr_auc"],
            "spec_pr_auc":m_spec["pr_auc"],
            "delta_pr_auc_spec_minus_global":m_spec["pr_auc"] - m_global["pr_auc"] if np.isfinite(m_spec["pr_auc"]) and np.isfinite(m_global["pr_auc"]) else float("nan"),
            "global_brier":m_global["brier"],
            "spec_brier":m_spec["brier"],
            "brier_improvement_global_minus_spec":m_global["brier"] - m_spec["brier"],
            "constant_brier":m_const["brier"],
            "brier_improvement_constant_minus_spec":m_const["brier"] - m_spec["brier"],
            "global_logloss":m_global["logloss"],
            "spec_logloss":m_spec["logloss"],
            "logloss_improvement_global_minus_spec":m_global["logloss"] - m_spec["logloss"],
            "constant_logloss":m_const["logloss"],
            "logloss_improvement_constant_minus_spec":m_const["logloss"] - m_spec["logloss"],
            "global_ece_10":m_global["ece_10"],
            "spec_ece_10":m_spec["ece_10"],
            "spec_mean_probability":m_spec["mean_probability"],
        }
        comp.update(boot)
        comp["candidate_gate"] = candidate_label(comp)
        comparison_rows.append(comp)

        print(
            f"  GLOBAL      ROC={m_global['roc_auc']:.4f} | PR={m_global['pr_auc']:.4f} | "
            f"Brier={m_global['brier']:.4f} | LL={m_global['logloss']:.4f}"
        )
        print(
            f"  SPECIALIZED ROC={m_spec['roc_auc']:.4f} | PR={m_spec['pr_auc']:.4f} | "
            f"Brier={m_spec['brier']:.4f} | LL={m_spec['logloss']:.4f} | gate={comp['candidate_gate']}"
        )

        pred = pd.DataFrame({
            "context":ctx,
            "match_id":te["match_id"].astype(str).to_numpy(),
            "time":te["__time"].astype(str).to_numpy(),
            "home_team":te.get("home_team", pd.Series("", index=te.index)).astype(str).to_numpy(),
            "away_team":te.get("away_team", pd.Series("", index=te.index)).astype(str).to_numpy(),
            "score_home_87":te["__score_home"].astype(int).to_numpy(),
            "score_away_87":te["__score_away"].astype(int).to_numpy(),
            "target":yte.to_numpy(dtype=int),
            "p_global":p_global,
            "p_specialized":p_spec,
            "p_delta_spec_minus_global":p_spec - p_global,
        })
        prediction_frames.append(pred)

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------
    sizes_df = pd.DataFrame(context_sizes)
    metrics_df = pd.DataFrame(metric_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    audits_df = pd.DataFrame([asdict(x) for x in audits])
    exclusions_df = pd.DataFrame(exclusions)

    # Rank explicit gates, then conservative composite of improvements.
    if not comparison_df.empty:
        order = {"STRONG":0, "PROMISING":1, "FAIL":2, "INSUFFICIENT_DATA":3}
        comparison_df["__gate_order"] = comparison_df["candidate_gate"].map(order).fillna(9)
        comparison_df["__score"] = (
            pd.to_numeric(comparison_df.get("delta_roc_auc_spec_minus_global"), errors="coerce").fillna(-1) * 4
            + pd.to_numeric(comparison_df.get("delta_pr_auc_spec_minus_global"), errors="coerce").fillna(-1) * 2
            + pd.to_numeric(comparison_df.get("brier_improvement_global_minus_spec"), errors="coerce").fillna(-1) * 3
            + pd.to_numeric(comparison_df.get("logloss_improvement_global_minus_spec"), errors="coerce").fillna(-1)
        )
        comparison_df = comparison_df.sort_values(["__gate_order", "__score"], ascending=[True, False]).drop(columns=["__gate_order", "__score"]).reset_index(drop=True)

    sizes_df.to_csv(out_dir / "specialized_context_sizes.csv", index=False)
    metrics_df.to_csv(out_dir / "specialized_metrics.csv", index=False)
    comparison_df.to_csv(out_dir / "specialized_comparison.csv", index=False)
    predictions_df.to_csv(out_dir / "specialized_predictions_test.csv", index=False)
    bootstrap_df.to_csv(out_dir / "specialized_bootstrap.csv", index=False)
    audits_df.to_csv(out_dir / "specialized_data_audit.csv", index=False)
    exclusions_df.to_csv(out_dir / "specialized_exclusions.csv", index=False)

    source_counts = model_ready["__source"].map(lambda x: Path(str(x)).name).value_counts().rename_axis("source").reset_index(name="model_ready_rows")
    source_counts.to_csv(out_dir / "specialized_model_ready_source_mix.csv", index=False)

    manifest = {
        "generated_at_utc": now_utc(),
        "mode": "V3C_SPECIALIZED_CONTEXT_RETRAIN_RESEARCH_ONLY",
        "template": str(template_path),
        "schema": str(schema_path),
        "features": features,
        "feature_count": len(features),
        "contexts_frozen_before_fit": CONTEXTS,
        "validation_fraction": args.validation_fraction,
        "minimum_feature_coverage": args.min_feature_coverage,
        "bootstrap_reps": args.bootstrap_reps,
        "rows_exact87_before_dedup": len(all_rows),
        "unique_matches_after_dedup": len(universe),
        "model_ready_n": len(model_ready),
        "dropped_missing_time": dropped_time,
        "dropped_below_feature_coverage": dropped_cov,
        "train_n": len(train),
        "test_n": len(test),
        "train_start": str(train["__time"].min()),
        "train_end": str(train["__time"].max()),
        "test_start": str(test["__time"].min()),
        "test_end": str(test["__time"].max()),
        "train_positive_rate": float(train["__y"].mean()),
        "test_positive_rate": float(test["__y"].mean()),
        "notes": [
            "Global baseline and specialized models share the exact same sklearn architecture and 38-feature schema.",
            "Global baseline is fit only on the old TRAIN block.",
            "Specialized models are fit only on context rows inside the old TRAIN block.",
            "All comparisons use the exact same recent context TEST rows.",
            "No refit-all is generated because the recent validation block is consumed by this analysis.",
            "Any winner still requires a future pristine/shadow holdout before deployment.",
        ],
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# V3-C — Étude de modèles spécialisés par contexte @87'",
        "",
        f"Généré : `{now_utc()}`",
        "",
        "## Méthode",
        "",
        "- Même architecture sklearn que V3-C actuel, clonée sans poids appris.",
        "- Même schéma exact de 38 features pour le baseline global et les modèles spécialisés.",
        "- Split temporel global ancien/récent, puis évaluation appariée sur les mêmes matchs de contexte.",
        f"- Coverage minimale d'entraînement : **{args.min_feature_coverage}/38**.",
        "- Aucun refit-all et aucune modification du scanner.",
        "",
        "## Volume global",
        "",
        f"- Exact87 avant déduplication : **{len(all_rows)}**",
        f"- Matchs uniques après déduplication : **{len(universe)}**",
        f"- Model-ready : **{len(model_ready)}**",
        f"- TRAIN : **{len(train)}** — base={train['__y'].mean():.2%}",
        f"- TEST récent : **{len(test)}** — base={test['__y'].mean():.2%}",
        "",
        "## Taille des contextes",
        "",
        markdown_table(sizes_df, ["context","train_n","train_positives","train_positive_rate","test_n","test_positives","test_positive_rate"], 20),
        "",
        "## Comparaison SPECIALIZED vs GLOBAL V3-C réentraîné",
        "",
        markdown_table(comparison_df, [
            "context","candidate_gate","context_train_n","context_test_n","context_test_positive_rate",
            "global_roc_auc","spec_roc_auc","delta_roc_auc_spec_minus_global",
            "global_pr_auc","spec_pr_auc","delta_pr_auc_spec_minus_global",
            "global_brier","spec_brier","brier_improvement_global_minus_spec",
            "global_logloss","spec_logloss","logloss_improvement_global_minus_spec",
            "p_roc_improves","p_pr_improves","p_brier_improves","p_logloss_improves",
        ], 20),
        "",
        "## Lecture des gates",
        "",
        "- **STRONG** : le spécialisé améliore les 4 familles de métriques, volume suffisant, ROC spécialisé >=0.62, delta ROC >=0.03 et bootstrap favorable.",
        "- **PROMISING** : au moins 3 familles de métriques s'améliorent, volume suffisant, ROC spécialisé >=0.58 et bootstrap au moins modérément favorable.",
        "- **FAIL** : ne signifie pas que le contexte football est mauvais; cela signifie que le réentraînement spécialisé n'améliore pas suffisamment le V3-C global sur ce test.",
        "",
        "## Statut méthodologique",
        "",
        "Ce TEST récent est maintenant consommé pour le développement. Même un résultat STRONG ne doit pas être déployé directement. Le contexte gagnant doit ensuite être figé puis validé sur un futur bloc pristine/shadow.",
    ]
    (out_dir / "specialized_report.md").write_text("\n".join(report), encoding="utf-8")

    print()
    print("="*104)
    print("RÉSUMÉ SPÉCIALISATION")
    print("="*104)
    print(f"Exact87 avant dédup : {len(all_rows)}")
    print(f"Matchs uniques       : {len(universe)}")
    print(f"Model-ready          : {len(model_ready)} | coverage >= {args.min_feature_coverage}/38")
    print(f"TRAIN                : {len(train)} | base={train['__y'].mean():.2%}")
    print(f"TEST récent          : {len(test)} | base={test['__y'].mean():.2%}")
    print(f"Sorties              : {out_dir}")
    print()
    print("COMPARAISON FINALE:")
    if comparison_df.empty:
        print("Aucun contexte évalué.")
    else:
        cols = [c for c in [
            "context","candidate_gate","context_train_n","context_test_n","context_test_positive_rate",
            "global_roc_auc","spec_roc_auc","delta_roc_auc_spec_minus_global",
            "global_pr_auc","spec_pr_auc","delta_pr_auc_spec_minus_global",
            "global_brier","spec_brier","brier_improvement_global_minus_spec",
            "p_roc_improves","p_brier_improves",
        ] if c in comparison_df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(comparison_df[cols].to_string(index=False))
    print()
    print("IMPORTANT: aucun modèle spécialisé n'est déployé ni refit-all automatiquement.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERREUR V3-C SPECIALIZED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
