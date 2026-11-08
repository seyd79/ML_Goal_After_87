#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_v3c_contexts_massive.py

V3-C — MASSIVE CONTEXT AUDIT @87'
=================================

OBJECTIF
--------
Utiliser TOUS les snapshots / datasets historiques disponibles pour chercher des
contextes football favorables au but strictement après 87', avec une unité
statistique correcte : 1 match = 1 snapshot exact 87'.

L'étude sépare volontairement deux questions :

A) CONTEXTE FOOTBALL (grand univers historique)
   P(goal_after_87 | contexte), stabilité discovery -> validation temporelle.

B) PERFORMANCE V3-C DANS LE CONTEXTE (bloc OOS déjà existant)
   Les probabilités de outputs/model_v3_experiments/predictions_test.csv sont
   jointes aux mêmes contextes pour mesurer ROC-AUC / PR-AUC / Brier / LogLoss.

IMPORTANT
---------
- Aucun réentraînement.
- Aucun changement du scanner.
- Aucun target n'est reconstruit ici : seuls les labels déjà résolus sont lus.
- Les snapshots 85'/86' ne sont PAS mélangés avec 87'.
- Les doublons de match_id sont fusionnés seulement s'ils ont le même target et
  le même score à 87'; les conflits sont exclus et reportés.
- Le bloc OOS V3-C est déjà consommé pour la sélection du modèle : il sert de
  contrôle secondaire, pas de holdout pristine.

SOURCES AUTO-DÉTECTÉES
----------------------
- outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_*.csv
- outputs/history/live_feature_snapshots*.csv

On peut ajouter des sources manuellement avec --source CHEMIN (répétable).

SORTIES
-------
outputs/v3c_context_massive_audit/
  v3c_massive_universe.csv
  v3c_massive_source_summary.csv
  v3c_massive_exclusions.csv
  v3c_massive_overall.csv
  v3c_massive_context_all.csv
  v3c_massive_context_discovery.csv
  v3c_massive_context_validation.csv
  v3c_massive_candidates.csv
  v3c_massive_v3c_oos_check.csv
  v3c_massive_report.md
  v3c_massive_manifest.json

UTILISATION
-----------
python3 audit_v3c_contexts_massive.py

Avec une autre fraction de validation récente :
python3 audit_v3c_contexts_massive.py --validation-fraction 0.30

Avec une source supplémentaire :
python3 audit_v3c_contexts_massive.py --source /chemin/autre_snapshot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "v3c_context_massive_audit"
DEFAULT_V3C_OOS = BASE_DIR / "outputs" / "model_v3_experiments" / "predictions_test.csv"

EPS = 1e-12
MODEL_NAME = "V3_C_AGGRESSIVE"

# Fixed before looking at the results.
DEFAULT_VALIDATION_FRACTION = 0.30

# Minimums used only for candidate labels. Every context is still exported.
MIN_DISC_PRIMARY = 40
MIN_VAL_PRIMARY = 20
MIN_DISC_EXACT = 20
MIN_VAL_EXACT = 10
MIN_DISC_INTERACTION = 25
MIN_VAL_INTERACTION = 12


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Massive context audit V3-C @87' sans retrain.")
    p.add_argument("--source", action="append", default=[], help="Source CSV/TSV/XLSX supplémentaire; répétable.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--predictions-oos", default=str(DEFAULT_V3C_OOS))
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    p.add_argument("--min-time-parse-ratio", type=float, default=0.80)
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


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
    # Percent is not expected for historical features, but tolerate it.
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    # Decimal comma only if there is no dot.
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
    if s in {"0", "false", "no", "non", "n", ""}:
        return False if s else None
    return None


def normalize_target_value(v: Any) -> Optional[int]:
    s = clean(v).lower()
    if s in {"1", "1.0", "true", "yes", "oui", "positive", "pos", "goal"}:
        return 1
    if s in {"0", "0.0", "false", "no", "non", "negative", "neg", "no_goal"}:
        return 0
    x = safe_num(v)
    if np.isfinite(x) and x in {0.0, 1.0}:
        return int(x)
    return None


def parse_time_value(v: Any) -> pd.Timestamp:
    s = clean(v)
    if not s:
        return pd.NaT
    # Excel serial dates are common in manual exports.
    x = safe_num(s)
    if np.isfinite(x) and 20000 <= x <= 80000 and re.fullmatch(r"[0-9.,]+", s):
        try:
            return pd.Timestamp("1899-12-30", tz="UTC") + pd.to_timedelta(x, unit="D")
        except Exception:
            pass
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided pooled z test; no scipy dependency."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p1 = k1 / n1
    p2 = k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    var = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if var <= 0:
        return 1.0
    z = abs(p1 - p2) / math.sqrt(var)
    # 2 * (1-Phi(z)) = erfc(z/sqrt(2))
    return math.erfc(z / math.sqrt(2))


def bh_qvalues(pvalues: Sequence[float]) -> List[float]:
    vals = np.asarray([float(x) if np.isfinite(x) else np.nan for x in pvalues], dtype=float)
    out = np.full(len(vals), np.nan)
    valid_idx = np.where(np.isfinite(vals))[0]
    if len(valid_idx) == 0:
        return out.tolist()
    order = valid_idx[np.argsort(vals[valid_idx])]
    m = len(order)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        q = vals[idx] * m / rank
        running = min(running, q)
        out[idx] = min(1.0, running)
    return out.tolist()


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_Aucune ligne._"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows).copy()
    def fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    rows = [[fmt(v) for v in row] for row in view.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep] + body)


# =============================================================================
# Source discovery and robust row extraction
# =============================================================================

ALIASES: Dict[str, Sequence[str]] = {
    "match_id": ["match_id", "fixture_id", "id"],
    "target": ["goal_after_87", "target", "label", "y"],
    "target_status": ["target_status", "label_status_for_backtest", "status"],
    "target_class": ["target_resolution_class", "resolution_class"],
    "target_confidence": ["target_confidence", "confidence"],
    "label_conflict": ["label_conflict", "target_conflict"],
    "minute": ["minute", "snapshot_minute"],
    "score_home": ["score_87_home", "score_home_snapshot", "score_home"],
    "score_away": ["score_87_away", "score_away_snapshot", "score_away"],
    "time": ["kickoff_utc", "kickoff_time_utc", "match_datetime", "fixture_datetime", "snapshot_timestamp_utc", "timestamp", "collected_at_utc"],
    "competition_id": ["competition_id", "league_id", "competition"],
    "home_team": ["home_team", "home_name"],
    "away_team": ["away_team", "away_name"],
    "odds_home": ["odds_home"],
    "odds_away": ["odds_away"],
    "shots_80_87": ["shots_80_87"],
    "sot_80_87": ["sot_80_87"],
    "corners_80_87": ["corners_80_87"],
    "home_shots_87": ["home_shots_87"],
    "away_shots_87": ["away_shots_87"],
    "home_sot_87": ["home_shots_on_target_87"],
    "away_sot_87": ["away_shots_on_target_87"],
    "last_shot_minute": ["last_shot_minute"],
    "last_sot_minute": ["last_shot_on_target_minute"],
    "mins_since_last_shot": ["minutes_since_last_shot"],
    "mins_since_last_sot": ["minutes_since_last_shot_on_target"],
}


@dataclass
class SourceSummary:
    source: str
    delimiter: str = ""
    header_line: int = 0
    physical_rows: int = 0
    parsed_rows: int = 0
    exact87_rows: int = 0
    valid_target_rows: int = 0
    valid_score_rows: int = 0
    accepted_rows: int = 0
    malformed_rows: int = 0
    skipped_non87: int = 0
    skipped_unresolved: int = 0
    skipped_bad_score: int = 0
    note: str = ""


def discover_sources(extra: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    hist = BASE_DIR / "outputs" / "historical_dataset_v2"
    live = BASE_DIR / "outputs" / "history"
    if hist.exists():
        candidates += sorted(hist.glob("Historical_Dataset_Goal_After_87_*.csv"))
    if live.exists():
        candidates += sorted(live.glob("live_feature_snapshots*.csv"))
    for x in extra:
        p = Path(x).expanduser()
        if not p.is_absolute():
            p = BASE_DIR / p
        candidates.append(p)

    # Deduplicate resolved-ish paths while preserving order.
    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.absolute())
        if key in seen or not p.exists() or not p.is_file():
            continue
        seen.add(key)
        out.append(p)
    return out


def detect_header(path: Path) -> Tuple[str, int, List[str]]:
    delimiters = [",", "\t", ";", "|"]
    required_any_score = {"score_87_home", "score_home_snapshot", "score_home"}
    required_any_target = {"goal_after_87", "target", "label", "y"}
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
                row = next(csv.reader([line], delimiter=delim))
            except Exception:
                continue
            header = [clean(x) for x in row]
            hn = {norm_name(x) for x in header}
            if "match_id" in hn and any(norm_name(x) in hn for x in required_any_score) and any(norm_name(x) in hn for x in required_any_target):
                return delim, line_no, header
    raise ValueError("header match_id + score@87 + target non détecté")


def source_priority(path: Path) -> int:
    n = path.name.lower()
    if "labeled_v3" in n:
        return 100
    if "labeled_v2" in n:
        return 95
    if "fixed" in n:
        return 90
    if "core_ready" in n:
        return 80
    if "raw_v2" in n:
        return 70
    if "live_feature_snapshots" in n:
        return 60
    return 50


def row_label_resolved(row: Mapping[str, Any]) -> bool:
    status = norm_name(row.get("target_status", ""))
    klass = norm_name(row.get("target_class", ""))
    conf = norm_name(row.get("target_confidence", ""))
    conflict = boolish(row.get("label_conflict", ""))
    if conflict is True:
        return False
    bad_tokens = ("pending", "ambiguous", "unresolved", "to_label", "unknown", "conflict")
    if any(t in status for t in bad_tokens):
        return False
    if any(t in klass for t in bad_tokens):
        return False
    if conf in {"low", "very_low"}:
        return False
    return True


def select_header_columns(header: Sequence[str]) -> Dict[str, Optional[str]]:
    by_norm = {norm_name(h): h for h in header}
    out: Dict[str, Optional[str]] = {}
    for canonical, aliases in ALIASES.items():
        real = None
        for a in aliases:
            if norm_name(a) in by_norm:
                real = by_norm[norm_name(a)]
                break
        out[canonical] = real
    return out


def read_csv_source(path: Path) -> Tuple[List[Dict[str, Any]], SourceSummary, List[Dict[str, Any]]]:
    delim, header_line, header = detect_header(path)
    summary = SourceSummary(source=str(path), delimiter={",":"comma", "\t":"tab", ";":"semicolon", "|":"pipe"}.get(delim, delim), header_line=header_line)
    cols = select_header_columns(header)
    required = ["match_id", "target", "score_home", "score_away"]
    missing = [c for c in required if cols.get(c) is None]
    if missing:
        raise ValueError("colonnes requises absentes: " + ", ".join(missing))
    idx = {h: i for i, h in enumerate(header)}
    needed_real = [v for v in cols.values() if v is not None]
    max_idx = max(idx[v] for v in needed_real)

    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for _ in range(header_line - 1):
            fh.readline()
        reader = csv.reader(fh, delimiter=delim)
        try:
            parsed_header = [clean(x) for x in next(reader)]
        except StopIteration:
            return rows, summary, exclusions
        if [norm_name(x) for x in parsed_header] != [norm_name(x) for x in header]:
            raise RuntimeError("header parsé différent du header détecté")

        for line_no, raw in enumerate(reader, start=header_line + 1):
            summary.physical_rows += 1
            if not raw:
                continue
            if len(raw) <= max_idx:
                summary.malformed_rows += 1
                exclusions.append({"source": str(path), "line": line_no, "match_id": raw[idx[cols['match_id']]] if idx[cols['match_id']] < len(raw) else "", "reason": "row_too_short"})
                continue
            summary.parsed_rows += 1
            rec: Dict[str, Any] = {"__source": str(path), "__priority": source_priority(path), "__line": line_no}
            for canonical, real in cols.items():
                rec[canonical] = raw[idx[real]] if real is not None else ""

            mid = clean(rec["match_id"])
            if not mid or mid.lower() == "match_id":
                continue

            # Exact 87 only. Historical score_87 rows without a minute column are
            # inherently 87 snapshots and are accepted.
            minute = safe_int(rec.get("minute"))
            if cols.get("minute") is not None:
                if minute != 87:
                    summary.skipped_non87 += 1
                    continue
            summary.exact87_rows += 1

            y = normalize_target_value(rec.get("target"))
            if y is None or not row_label_resolved(rec):
                summary.skipped_unresolved += 1
                continue
            summary.valid_target_rows += 1

            sh = safe_int(rec.get("score_home"))
            sa = safe_int(rec.get("score_away"))
            if sh is None or sa is None or sh < 0 or sa < 0:
                summary.skipped_bad_score += 1
                continue
            summary.valid_score_rows += 1

            rec["__y"] = y
            rec["__score_home"] = sh
            rec["__score_away"] = sa
            rec["__time"] = parse_time_value(rec.get("time"))
            # Normalize numerics used by contexts.
            for c in ["odds_home", "odds_away", "shots_80_87", "sot_80_87", "corners_80_87", "home_shots_87", "away_shots_87", "home_sot_87", "away_sot_87", "last_shot_minute", "last_sot_minute", "mins_since_last_shot", "mins_since_last_sot"]:
                rec[c] = safe_num(rec.get(c))
            rows.append(rec)
            summary.accepted_rows += 1

    return rows, summary, exclusions


def read_excel_source(path: Path) -> Tuple[List[Dict[str, Any]], SourceSummary, List[Dict[str, Any]]]:
    df = pd.read_excel(path)
    summary = SourceSummary(source=str(path), delimiter="excel", header_line=1, physical_rows=len(df), parsed_rows=len(df))
    by_norm = {norm_name(c): c for c in df.columns}
    resolved: Dict[str, Optional[str]] = {}
    for canonical, aliases in ALIASES.items():
        resolved[canonical] = next((by_norm[norm_name(a)] for a in aliases if norm_name(a) in by_norm), None)
    if any(resolved.get(c) is None for c in ["match_id", "target", "score_home", "score_away"]):
        raise ValueError("colonnes requises absentes")
    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for i, s in df.iterrows():
        rec: Dict[str, Any] = {"__source": str(path), "__priority": source_priority(path), "__line": int(i) + 2}
        for canonical, real in resolved.items():
            rec[canonical] = s.get(real, "") if real is not None else ""
        minute = safe_int(rec.get("minute"))
        if resolved.get("minute") is not None and minute != 87:
            summary.skipped_non87 += 1
            continue
        summary.exact87_rows += 1
        y = normalize_target_value(rec.get("target"))
        if y is None or not row_label_resolved(rec):
            summary.skipped_unresolved += 1
            continue
        summary.valid_target_rows += 1
        sh, sa = safe_int(rec.get("score_home")), safe_int(rec.get("score_away"))
        if sh is None or sa is None or sh < 0 or sa < 0:
            summary.skipped_bad_score += 1
            continue
        summary.valid_score_rows += 1
        rec["__y"], rec["__score_home"], rec["__score_away"] = y, sh, sa
        rec["__time"] = parse_time_value(rec.get("time"))
        for c in ["odds_home", "odds_away", "shots_80_87", "sot_80_87", "corners_80_87", "home_shots_87", "away_shots_87", "home_sot_87", "away_sot_87", "last_shot_minute", "last_sot_minute", "mins_since_last_shot", "mins_since_last_sot"]:
            rec[c] = safe_num(rec.get(c))
        rows.append(rec)
        summary.accepted_rows += 1
    return rows, summary, exclusions


def read_source(path: Path) -> Tuple[List[Dict[str, Any]], SourceSummary, List[Dict[str, Any]]]:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        return read_excel_source(path)
    return read_csv_source(path)


def completeness_score(r: Mapping[str, Any]) -> int:
    cols = ["odds_home", "odds_away", "shots_80_87", "sot_80_87", "corners_80_87", "home_shots_87", "away_shots_87", "home_sot_87", "away_sot_87", "last_shot_minute", "last_sot_minute", "mins_since_last_shot", "mins_since_last_sot", "competition_id", "home_team", "away_team"]
    score = 0
    for c in cols:
        v = r.get(c)
        if isinstance(v, (float, np.floating)):
            score += int(np.isfinite(v))
        else:
            score += int(bool(clean(v)))
    score += 2 * int(not pd.isna(r.get("__time", pd.NaT)))
    return score


def merge_matches(rows: Sequence[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[clean(r.get("match_id"))].append(r)

    merged: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for mid, group in grouped.items():
        ys = {int(r["__y"]) for r in group}
        scores = {(int(r["__score_home"]), int(r["__score_away"])) for r in group}
        if len(ys) != 1:
            exclusions.append({"match_id": mid, "reason": "target_conflict_across_sources", "details": str(sorted(ys))})
            continue
        if len(scores) != 1:
            exclusions.append({"match_id": mid, "reason": "score87_conflict_across_sources", "details": str(sorted(scores))})
            continue

        ranked = sorted(group, key=lambda r: (completeness_score(r), int(r.get("__priority", 0))), reverse=True)
        base = dict(ranked[0])
        # Fill missing fields from duplicate sources with the same target/score.
        for other in ranked[1:]:
            for k, v in other.items():
                if k.startswith("__") and k not in {"__time"}:
                    continue
                current = base.get(k)
                try:
                    current_missing = bool(pd.isna(current))
                except Exception:
                    current_missing = False
                if not current_missing and not isinstance(current, (int, float, np.integer, np.floating, pd.Timestamp)):
                    current_missing = not clean(current)
                if current_missing:
                    if isinstance(v, (float, np.floating)):
                        if np.isfinite(v):
                            base[k] = v
                    elif not pd.isna(v) and clean(v):
                        base[k] = v
        base["__sources_count"] = len(group)
        base["__sources"] = "|".join(sorted({Path(clean(r['__source'])).name for r in group}))
        merged.append(base)

    df = pd.DataFrame(merged)
    if df.empty:
        raise RuntimeError("Aucun match exact 87' avec target résolu et score valide.")
    return df, exclusions


# =============================================================================
# Context construction
# =============================================================================

def bucket_numeric(s: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels, right=False, include_lowest=True).astype("object").fillna("unknown")


def add_contexts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = pd.to_numeric(out["__score_home"], errors="coerce")
    a = pd.to_numeric(out["__score_away"], errors="coerce")
    diff = h - a
    total = h + a

    out["score_state"] = np.select(
        [(h == 0) & (a == 0), (h == a) & (h > 0), h > a, a > h],
        ["0-0", "draw_with_goals", "home_lead", "away_lead"], default="unknown")
    out["goal_margin"] = np.select(
        [diff == 0, diff == 1, diff >= 2, diff == -1, diff <= -2],
        ["draw", "home_+1", "home_+2plus", "away_+1", "away_+2plus"], default="unknown")
    out["total_goals_bucket"] = bucket_numeric(total, [-0.1,0.5,1.5,2.5,3.5,4.5,np.inf], ["0","1","2","3","4","5plus"])
    out["exact_score"] = h.astype("Int64").astype(str) + "-" + a.astype("Int64").astype(str)

    oh = pd.to_numeric(out.get("odds_home", np.nan), errors="coerce")
    oa = pd.to_numeric(out.get("odds_away", np.nan), errors="coerce")
    valid_odds = (oh > 1) & (oa > 1)
    side = pd.Series("unknown", index=out.index, dtype=object)
    # Balanced first, then clear favourite.
    balanced = valid_odds & ((oh - oa).abs() <= 0.03)
    side.loc[balanced] = "balanced"
    side.loc[valid_odds & ~balanced & (oh < oa)] = "home_favorite"
    side.loc[valid_odds & ~balanced & (oa < oh)] = "away_favorite"
    out["favorite_side"] = side

    fav_odds = pd.Series(np.nan, index=out.index, dtype=float)
    fav_odds.loc[side == "home_favorite"] = oh.loc[side == "home_favorite"]
    fav_odds.loc[side == "away_favorite"] = oa.loc[side == "away_favorite"]
    out["favorite_odds"] = fav_odds
    out["favorite_strength"] = bucket_numeric(fav_odds, [1.0,1.40,1.65,1.90,2.20,np.inf], ["very_strong_<1.40","strong_1.40-1.65","medium_1.65-1.90","mild_1.90-2.20","weak_2.20+"])

    homefav = side == "home_favorite"
    awayfav = side == "away_favorite"
    fav_state = pd.Series("unknown", index=out.index, dtype=object)
    fav_state.loc[homefav & (h > a)] = "favorite_leading"
    fav_state.loc[homefav & (h == a)] = "favorite_drawing"
    fav_state.loc[homefav & (h < a)] = "favorite_trailing"
    fav_state.loc[awayfav & (a > h)] = "favorite_leading"
    fav_state.loc[awayfav & (h == a)] = "favorite_drawing"
    fav_state.loc[awayfav & (a < h)] = "favorite_trailing"
    out["favorite_live_state"] = fav_state

    # Recent activity buckets.
    out["shots_80_87_bucket"] = bucket_numeric(out.get("shots_80_87", np.nan), [-0.1,1,3,5,np.inf], ["0","1-2","3-4","5plus"])
    out["sot_80_87_bucket"] = bucket_numeric(out.get("sot_80_87", np.nan), [-0.1,1,2,np.inf], ["0","1","2plus"])
    out["corners_80_87_bucket"] = bucket_numeric(out.get("corners_80_87", np.nan), [-0.1,1,2,np.inf], ["0","1","2plus"])

    mins_shot = pd.to_numeric(out.get("mins_since_last_shot", np.nan), errors="coerce")
    last_shot = pd.to_numeric(out.get("last_shot_minute", np.nan), errors="coerce")
    mins_shot = mins_shot.where(mins_shot.notna(), 87 - last_shot)
    mins_sot = pd.to_numeric(out.get("mins_since_last_sot", np.nan), errors="coerce")
    last_sot = pd.to_numeric(out.get("last_sot_minute", np.nan), errors="coerce")
    mins_sot = mins_sot.where(mins_sot.notna(), 87 - last_sot)
    out["last_shot_recency"] = bucket_numeric(mins_shot, [-0.1,2,4,7,np.inf], ["0-1m","2-3m","4-6m","7plus_m"])
    out["last_sot_recency"] = bucket_numeric(mins_sot, [-0.1,2,4,7,np.inf], ["0-1m","2-3m","4-6m","7plus_m"])

    total_shots = pd.to_numeric(out.get("home_shots_87", np.nan), errors="coerce") + pd.to_numeric(out.get("away_shots_87", np.nan), errors="coerce")
    total_sot = pd.to_numeric(out.get("home_sot_87", np.nan), errors="coerce") + pd.to_numeric(out.get("away_sot_87", np.nan), errors="coerce")
    out["total_shots_87_bucket"] = bucket_numeric(total_shots, [-0.1,15,25,35,np.inf], ["<15","15-24","25-34","35plus"])
    out["total_sot_87_bucket"] = bucket_numeric(total_sot, [-0.1,5,10,np.inf], ["<5","5-9","10plus"])

    # Fixed binary contexts. A row may belong to several; stored later via masks.
    return out


CATEGORICAL_FAMILIES = {
    "score_state": "score_state",
    "goal_margin": "goal_margin",
    "total_goals": "total_goals_bucket",
    "exact_score": "exact_score",
    "favorite_side": "favorite_side",
    "favorite_strength": "favorite_strength",
    "favorite_state": "favorite_live_state",
    "shots_80_87": "shots_80_87_bucket",
    "sot_80_87": "sot_80_87_bucket",
    "corners_80_87": "corners_80_87_bucket",
    "last_shot_recency": "last_shot_recency",
    "last_sot_recency": "last_sot_recency",
    "total_shots_87": "total_shots_87_bucket",
    "total_sot_87": "total_sot_87_bucket",
}


def binary_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    h = pd.to_numeric(df["__score_home"], errors="coerce")
    a = pd.to_numeric(df["__score_away"], errors="coerce")
    diff = h - a
    side = df["favorite_side"].astype(str)
    fav_state = df["favorite_live_state"].astype(str)
    sot_recent = pd.to_numeric(df.get("sot_80_87", np.nan), errors="coerce")
    shots_recent = pd.to_numeric(df.get("shots_80_87", np.nan), errors="coerce")
    corners_recent = pd.to_numeric(df.get("corners_80_87", np.nan), errors="coerce")

    favorite_trailing_by1 = ((side == "home_favorite") & (diff == -1)) | ((side == "away_favorite") & (diff == 1))
    favorite_trailing_by2p = ((side == "home_favorite") & (diff <= -2)) | ((side == "away_favorite") & (diff >= 2))
    favorite_leading_by1 = ((side == "home_favorite") & (diff == 1)) | ((side == "away_favorite") & (diff == -1))

    return {
        "HOME_TRAILING_BY_1": diff == -1,
        "AWAY_TRAILING_BY_1": diff == 1,
        "ONE_GOAL_MARGIN": diff.abs() == 1,
        "FAVORITE_TRAILING": fav_state == "favorite_trailing",
        "FAVORITE_TRAILING_BY_1": favorite_trailing_by1,
        "FAVORITE_TRAILING_BY_2PLUS": favorite_trailing_by2p,
        "FAVORITE_DRAWING": fav_state == "favorite_drawing",
        "FAVORITE_LEADING_BY_1": favorite_leading_by1,
        "HOME_FAVORITE_TRAILING_BY_1": (side == "home_favorite") & (diff == -1),
        "AWAY_FAVORITE_TRAILING_BY_1": (side == "away_favorite") & (diff == 1),
        "HOME_FAVORITE_DRAWING": (side == "home_favorite") & (diff == 0),
        "AWAY_FAVORITE_DRAWING": (side == "away_favorite") & (diff == 0),
        "DRAW_WITH_GOALS": (h == a) & (h > 0),
        "SCORE_1_0_OR_2_0": ((h == 1) | (h == 2)) & (a == 0),
        "SCORE_0_1_OR_0_2": (h == 0) & ((a == 1) | (a == 2)),
        "ONE_GOAL_MARGIN_AND_SOT_2PLUS": (diff.abs() == 1) & (sot_recent >= 2),
        "ONE_GOAL_MARGIN_AND_SHOTS_3PLUS": (diff.abs() == 1) & (shots_recent >= 3),
        "FAVORITE_TRAILING_BY_1_AND_SOT_1PLUS": favorite_trailing_by1 & (sot_recent >= 1),
        "FAVORITE_TRAILING_BY_1_AND_SHOTS_3PLUS": favorite_trailing_by1 & (shots_recent >= 3),
        "HOME_FAVORITE_TRAILING_BY_1_AND_SOT_1PLUS": (side == "home_favorite") & (diff == -1) & (sot_recent >= 1),
        "HOME_FAVORITE_TRAILING_BY_1_AND_CORNERS_1PLUS": (side == "home_favorite") & (diff == -1) & (corners_recent >= 1),
    }


def context_mask(df: pd.DataFrame, family: str, context: str) -> pd.Series:
    if family == "interactions":
        return binary_masks(df).get(context, pd.Series(False, index=df.index))
    col = CATEGORICAL_FAMILIES.get(family)
    if col is None or col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str) == str(context)


# =============================================================================
# Context metrics / temporal split
# =============================================================================

def one_context_metrics(df: pd.DataFrame, mask: pd.Series, family: str, context: str) -> Dict[str, Any]:
    mask = mask.fillna(False).astype(bool)
    sub = df.loc[mask]
    comp = df.loc[~mask]
    n = len(sub)
    k = int(pd.to_numeric(sub["__y"], errors="coerce").sum()) if n else 0
    rate = k / n if n else float("nan")
    global_rate = float(pd.to_numeric(df["__y"], errors="coerce").mean()) if len(df) else float("nan")
    lo, hi = wilson(k, n)
    k2 = int(pd.to_numeric(comp["__y"], errors="coerce").sum()) if len(comp) else 0
    pval = two_prop_pvalue(k, n, k2, len(comp)) if n and len(comp) else float("nan")
    return {
        "family": family,
        "context": context,
        "n": n,
        "positives": k,
        "positive_rate": rate,
        "global_rate": global_rate,
        "uplift_vs_global_pp": (rate - global_rate) * 100 if np.isfinite(rate) and np.isfinite(global_rate) else float("nan"),
        "relative_lift": rate / global_rate if np.isfinite(rate) and np.isfinite(global_rate) and global_rate > 0 else float("nan"),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_vs_complement": pval,
    }


def context_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family, col in CATEGORICAL_FAMILIES.items():
        if col not in df.columns:
            continue
        vals = [x for x in df[col].dropna().astype(str).unique().tolist() if x not in {"unknown", "nan", "<NA>"}]
        for value in sorted(vals):
            rows.append(one_context_metrics(df, df[col].astype(str) == value, family, value))
    for name, mask in binary_masks(df).items():
        rows.append(one_context_metrics(df, mask, "interactions", name))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_qvalues(out["p_vs_complement"].tolist())
        out = out.sort_values(["uplift_vs_global_pp", "n"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out


def choose_minimums(family: str) -> Tuple[int, int]:
    if family == "exact_score":
        return MIN_DISC_EXACT, MIN_VAL_EXACT
    if family == "interactions":
        return MIN_DISC_INTERACTION, MIN_VAL_INTERACTION
    return MIN_DISC_PRIMARY, MIN_VAL_PRIMARY


def merge_candidate_tables(disc: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    d = disc.add_suffix("_disc").rename(columns={"family_disc":"family", "context_disc":"context"})
    v = val.add_suffix("_val").rename(columns={"family_val":"family", "context_val":"context"})
    m = d.merge(v, on=["family", "context"], how="outer")
    labels = []
    scores = []
    for _, r in m.iterrows():
        fam = clean(r.get("family"))
        min_d, min_v = choose_minimums(fam)
        nd = safe_num(r.get("n_disc")); nv = safe_num(r.get("n_val"))
        ud = safe_num(r.get("uplift_vs_global_pp_disc")); uv = safe_num(r.get("uplift_vs_global_pp_val"))
        rd = safe_num(r.get("relative_lift_disc")); rv = safe_num(r.get("relative_lift_val"))
        qv = safe_num(r.get("q_bh_val"))
        rate_d = safe_num(r.get("positive_rate_disc")); rate_v = safe_num(r.get("positive_rate_val"))
        enough = np.isfinite(nd) and np.isfinite(nv) and nd >= min_d and nv >= min_v
        direction = np.isfinite(ud) and np.isfinite(uv) and ud > 0 and uv > 0
        stable = np.isfinite(rate_d) and np.isfinite(rate_v) and abs(rate_d - rate_v) <= 0.15
        strong = enough and direction and stable and ud >= 5 and uv >= 5 and rd >= 1.15 and rv >= 1.15 and np.isfinite(qv) and qv <= 0.20
        promising = enough and direction and stable and ud >= 3 and uv >= 3 and rd >= 1.10 and rv >= 1.10
        label = "STRONG" if strong else ("PROMISING" if promising else "FAIL")
        labels.append(label)
        # Conservative score rewards the weaker split, volume and stability.
        if enough and np.isfinite(ud) and np.isfinite(uv):
            min_u = min(ud, uv)
            vol = math.log1p(min(nd, nv))
            stability_pen = abs(rate_d - rate_v) * 100 if np.isfinite(rate_d) and np.isfinite(rate_v) else 20
            q_bonus = max(0.0, 1.0 - qv) if np.isfinite(qv) else 0.0
            score = min_u * vol + 5 * q_bonus - 0.5 * stability_pen
        else:
            score = -999.0
        scores.append(score)
    m["candidate_gate"] = labels
    m["candidate_score"] = scores
    return m.sort_values(["candidate_gate", "candidate_score"], ascending=[True, False]).reset_index(drop=True)


# =============================================================================
# V3-C OOS overlay
# =============================================================================

def probability_series(s: pd.Series) -> pd.Series:
    def conv(v: Any) -> float:
        x = safe_num(v)
        if not np.isfinite(x):
            return float("nan")
        # Handles both 0-1 and percentages.
        return x / 100.0 if x > 1.000001 else x
    return s.map(conv).astype(float)


def load_oos(path: Path, model_name: str, universe: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    if "model_name" in pred.columns:
        pred = pred[pred["model_name"].astype(str) == model_name].copy()
    if pred.empty or "match_id" not in pred.columns:
        return pd.DataFrame()
    target_col = next((c for c in ["target", "goal_after_87", "label", "y"] if c in pred.columns), None)
    prob_col = next((c for c in ["probability", "v3c_probability", "model_probability"] if c in pred.columns), None)
    if target_col is None or prob_col is None:
        return pd.DataFrame()
    pred["__oos_y"] = pd.to_numeric(pred[target_col], errors="coerce")
    pred["__p"] = probability_series(pred[prob_col])
    pred = pred.dropna(subset=["__oos_y", "__p"]).copy()
    pred = pred[pred["__oos_y"].isin([0,1])]
    # Universe is one row per match. Keep context columns and target only for a consistency check.
    keep = [c for c in universe.columns if c not in {"__y"}]
    joined = pred.merge(universe[keep], on="match_id", how="inner")
    return joined


def oos_metric_block(df: pd.DataFrame, global_oos_rate: float) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"oos_n":0}
    y = pd.to_numeric(df["__oos_y"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["__p"], errors="coerce").clip(EPS, 1-EPS).to_numpy(dtype=float)
    k = int(y.sum())
    prev = float(y.mean())
    roc = float(roc_auc_score(y,p)) if len(np.unique(y)) > 1 else float("nan")
    pr = float(average_precision_score(y,p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y,p))
    ll = float(log_loss(y,p,labels=[0,1]))
    const_context = np.full(len(y), np.clip(prev, EPS, 1-EPS))
    const_global = np.full(len(y), np.clip(global_oos_rate, EPS, 1-EPS))
    brier_context = float(brier_score_loss(y,const_context))
    brier_global = float(brier_score_loss(y,const_global))
    ll_context = float(log_loss(y,const_context,labels=[0,1]))
    ll_global = float(log_loss(y,const_global,labels=[0,1]))
    return {
        "oos_n": n,
        "oos_positives": k,
        "oos_positive_rate": prev,
        "oos_uplift_vs_global_pp": (prev-global_oos_rate)*100,
        "oos_mean_probability": float(np.mean(p)),
        "oos_roc_auc": roc,
        "oos_pr_auc": pr,
        "oos_pr_lift_vs_prevalence": (pr-prev) if np.isfinite(pr) else float("nan"),
        "oos_brier": brier,
        "oos_brier_improvement_vs_context_constant": brier_context-brier,
        "oos_brier_improvement_vs_global_constant": brier_global-brier,
        "oos_logloss": ll,
        "oos_logloss_improvement_vs_context_constant": ll_context-ll,
        "oos_logloss_improvement_vs_global_constant": ll_global-ll,
    }


def build_oos_context_check(oos: pd.DataFrame, candidate_keys: pd.DataFrame) -> pd.DataFrame:
    if oos.empty or candidate_keys.empty:
        return pd.DataFrame()
    global_rate = float(oos["__oos_y"].mean())
    rows = []
    for _, r in candidate_keys[["family","context"]].drop_duplicates().iterrows():
        family, context = clean(r["family"]), clean(r["context"])
        mask = context_mask(oos, family, context)
        rec = {"family":family, "context":context}
        rec.update(oos_metric_block(oos.loc[mask], global_rate))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["oos_n","oos_roc_auc"], ascending=[False,False], na_position="last").reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    if not 0.15 <= args.validation_fraction <= 0.45:
        raise ValueError("--validation-fraction doit être entre 0.15 et 0.45")

    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(args.source)
    if not sources:
        raise FileNotFoundError("Aucune source snapshot/historique trouvée.")

    print("="*100)
    print("V3-C — MASSIVE CONTEXT AUDIT @87'")
    print("="*100)
    print(f"Sources détectées : {len(sources)}")

    all_rows: List[Dict[str, Any]] = []
    summaries: List[SourceSummary] = []
    exclusions: List[Dict[str, Any]] = []
    for p in sources:
        try:
            rows, summary, exc = read_source(p)
            all_rows.extend(rows)
            summaries.append(summary)
            exclusions.extend(exc)
            print(f"[OK] {p.name:<60s} | accepted87={summary.accepted_rows:>5d} | unresolved={summary.skipped_unresolved:>5d} | non87={summary.skipped_non87:>5d} | malformed={summary.malformed_rows:>4d}")
        except Exception as e:
            s = SourceSummary(source=str(p), note=f"SKIPPED: {type(e).__name__}: {e}")
            summaries.append(s)
            print(f"[SKIP] {p.name}: {type(e).__name__}: {e}")

    if not all_rows:
        raise RuntimeError("Aucune ligne exacte 87' exploitable trouvée dans les sources.")

    universe, dup_exc = merge_matches(all_rows)
    exclusions.extend(dup_exc)
    universe = add_contexts(universe)

    # Temporal validity.
    time_ok = universe["__time"].notna()
    time_ratio = float(time_ok.mean())
    if time_ratio < args.min_time_parse_ratio:
        raise RuntimeError(f"Seulement {time_ratio:.1%} des matchs ont une date exploitable; split temporel refusé.")
    temporal = universe.loc[time_ok].sort_values(["__time","match_id"], kind="stable").reset_index(drop=True)
    n = len(temporal)
    split = int(round(n * (1 - args.validation_fraction)))
    split = min(max(split, 1), n-1)
    discovery = temporal.iloc[:split].copy()
    validation = temporal.iloc[split:].copy()

    all_table = context_table(temporal)
    disc_table = context_table(discovery)
    val_table = context_table(validation)
    candidates = merge_candidate_tables(disc_table, val_table)

    # OOS overlay for all contexts that have enough validation volume OR rank near the top.
    candidate_keys = candidates[
        (pd.to_numeric(candidates.get("n_val"), errors="coerce").fillna(0) >= 8)
        & (pd.to_numeric(candidates.get("n_disc"), errors="coerce").fillna(0) >= 15)
    ][["family","context"]].copy()
    oos_path = Path(args.predictions_oos).expanduser()
    if not oos_path.is_absolute():
        oos_path = BASE_DIR / oos_path
    oos = load_oos(oos_path, args.model_name, universe)
    oos_check = build_oos_context_check(oos, candidate_keys)
    if not oos_check.empty:
        candidates = candidates.merge(oos_check, on=["family","context"], how="left")

    overall = pd.DataFrame([
        {"split":"ALL_TIME_VALID", "n":len(temporal), "positives":int(temporal['__y'].sum()), "positive_rate":float(temporal['__y'].mean()), "start":temporal['__time'].min(), "end":temporal['__time'].max()},
        {"split":"DISCOVERY_OLD", "n":len(discovery), "positives":int(discovery['__y'].sum()), "positive_rate":float(discovery['__y'].mean()), "start":discovery['__time'].min(), "end":discovery['__time'].max()},
        {"split":"VALIDATION_RECENT", "n":len(validation), "positives":int(validation['__y'].sum()), "positive_rate":float(validation['__y'].mean()), "start":validation['__time'].min(), "end":validation['__time'].max()},
        {"split":"V3C_OOS_OVERLAY", "n":len(oos), "positives":int(oos['__oos_y'].sum()) if not oos.empty else 0, "positive_rate":float(oos['__oos_y'].mean()) if not oos.empty else float('nan'), "start":"", "end":""},
    ])

    # Exports.
    universe_export = universe.copy()
    if "__time" in universe_export.columns:
        universe_export["__time"] = universe_export["__time"].astype(str)
    universe_export.to_csv(out_dir / "v3c_massive_universe.csv", index=False)
    pd.DataFrame([asdict(x) for x in summaries]).to_csv(out_dir / "v3c_massive_source_summary.csv", index=False)
    pd.DataFrame(exclusions).to_csv(out_dir / "v3c_massive_exclusions.csv", index=False)
    overall.to_csv(out_dir / "v3c_massive_overall.csv", index=False)
    all_table.to_csv(out_dir / "v3c_massive_context_all.csv", index=False)
    disc_table.to_csv(out_dir / "v3c_massive_context_discovery.csv", index=False)
    val_table.to_csv(out_dir / "v3c_massive_context_validation.csv", index=False)
    candidates.to_csv(out_dir / "v3c_massive_candidates.csv", index=False)
    oos_check.to_csv(out_dir / "v3c_massive_v3c_oos_check.csv", index=False)

    manifest = {
        "generated_at_utc": now_utc(),
        "mode": "MASSIVE_CONTEXT_DISCOVERY_PLUS_TEMPORAL_VALIDATION_NO_RETRAIN",
        "sources": [str(p) for p in sources],
        "rows_before_dedup": len(all_rows),
        "unique_matches_exact87": len(universe),
        "time_parse_ratio": time_ratio,
        "temporal_matches": len(temporal),
        "validation_fraction": args.validation_fraction,
        "discovery_n": len(discovery),
        "validation_n": len(validation),
        "v3c_oos_joined_n": len(oos),
        "notes": [
            "One match = one exact-87 snapshot.",
            "85/86 snapshots excluded from the main universe.",
            "Context discovery uses targets, not in-sample V3-C probabilities.",
            "V3-C probability metrics come only from the existing consumed OOS predictions_test block.",
        ],
    }
    (out_dir / "v3c_massive_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Report.
    top = candidates.sort_values(["candidate_gate","candidate_score"], ascending=[True,False]).copy()
    # Put STRONG/PROMISING first explicitly.
    order = {"STRONG":0,"PROMISING":1,"FAIL":2}
    top["__order"] = top["candidate_gate"].map(order).fillna(9)
    top = top.sort_values(["__order","candidate_score"], ascending=[True,False]).drop(columns="__order")

    report = [
        "# V3-C Massive Context Audit @87'",
        "",
        f"Généré : `{now_utc()}`",
        "",
        "## Méthode",
        "",
        "- Unité statistique : **1 match = 1 snapshot exact 87'**.",
        "- Les snapshots 85'/86' sont exclus de l'univers principal.",
        "- Discovery = partie chronologiquement ancienne; validation = partie récente.",
        "- Les contextes sont découverts sur le target réel, sans utiliser les probabilités refit_all in-sample.",
        "- Le bloc `predictions_test.csv` V3-C est joint séparément comme contrôle OOS déjà consommé.",
        "",
        "## Volume",
        "",
        markdown_table(overall, ["split","n","positives","positive_rate","start","end"], 20),
        "",
        "## Top contextes stables discovery -> validation",
        "",
        markdown_table(top, [
            "family","context","candidate_gate","candidate_score",
            "n_disc","positive_rate_disc","uplift_vs_global_pp_disc","relative_lift_disc",
            "n_val","positive_rate_val","uplift_vs_global_pp_val","relative_lift_val","q_bh_val",
            "oos_n","oos_positive_rate","oos_roc_auc","oos_pr_lift_vs_prevalence",
            "oos_brier_improvement_vs_context_constant","oos_logloss_improvement_vs_context_constant"
        ], 40),
        "",
        "## Interprétation",
        "",
        "- `STRONG` signifie : volume minimum, uplift positif dans les deux périodes, stabilité, lift >= 1.15 et q-value validation <= 0.20.",
        "- `PROMISING` signifie : volume minimum, uplift >= 3 points dans les deux périodes, lift >= 1.10 et direction stable.",
        "- Un contexte prometteur n'est pas encore un filtre de production : il doit ensuite être testé sur un futur bloc pristine / shadow.",
        "- Les métriques V3-C OOS servent à savoir si V3-C sait encore classer les matchs à l'intérieur du contexte; elles ne transforment pas le bloc déjà consommé en validation externe.",
    ]
    (out_dir / "v3c_massive_report.md").write_text("\n".join(report), encoding="utf-8")

    print()
    print("="*100)
    print("RÉSUMÉ")
    print("="*100)
    print(f"Lignes exact87 avant dédup : {len(all_rows)}")
    print(f"Matchs uniques exact87      : {len(universe)}")
    print(f"Dates exploitables          : {time_ratio:.2%} ({len(temporal)}/{len(universe)})")
    print(f"Discovery                   : {len(discovery)} | base={discovery['__y'].mean():.2%}")
    print(f"Validation récente          : {len(validation)} | base={validation['__y'].mean():.2%}")
    print(f"V3-C OOS joint              : {len(oos)}")
    print(f"Sorties                     : {out_dir}")
    print()
    print("TOP CONTEXTS STABLES:")
    show = top.head(20)
    cols = [c for c in ["family","context","candidate_gate","n_disc","positive_rate_disc","uplift_vs_global_pp_disc","n_val","positive_rate_val","uplift_vs_global_pp_val","q_bh_val","oos_n","oos_roc_auc"] if c in show.columns]
    print(show[cols].to_string(index=False))
    print()
    print("IMPORTANT: aucun contexte n'est automatiquement ajouté au scanner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
