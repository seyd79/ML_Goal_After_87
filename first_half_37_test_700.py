#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIRST HALF 37 — RECENT 2026 EXTERNAL 5/5
========================================

Objectif
--------
Collecter immédiatement les matchs les plus récents disponibles en 2026 pour tester
le modèle gelé V3 DRAW_WITH_GOALS, sans attendre de nouveaux matchs.

La sélection est HISTORIQUE RECENTE / EXTERNE PAR ID, pas un pristine temporal holdout :
- le script part du jour UTC verrouillé au premier lancement ;
- il remonte vers le 1er janvier 2026 ;
- il exclut tous les match_id déjà consommés par le développement V3 ;
- il garde uniquement les matchs draw_with_goals avec EXACTEMENT 5/5 features.

Collecte blind (défaut)
-----------------------
    python3 -u first_half_37_recent_2026_5of5.py

Le script collecte les 200 matchs admissibles LES PLUS RECENTS de 2026, en remontant
jour par jour depuis la date d'ancrage verrouillée au premier lancement.

Admissibilité :
    - match non consommé par les datasets V3 connus ;
    - timeline 1re mi-temps complète ;
    - fin de 1re mi-temps observée ;
    - score à 37' reconstructible ;
    - score_37_home == score_37_away > 0 ;
    - EXACTEMENT 5/5 features gelées présentes, numériques et finies.

Un match à 4/5, 2/5, 1/5 ou 0/5 est QUALITY_EXCLUDED.
Aucune imputation artificielle. Missing != 0.
Aucun scoring modèle pendant la collecte. Targets masqués.

Si 200 matchs ne sont pas trouvés avant le 1er janvier 2026, le script termine avec
INSUFFICIENT_2026_MATCHES. Il n'attend pas de nouveaux matchs.

Evaluation one-shot
-------------------
Préflight sans ouverture des targets :
    python3 -u first_half_37_recent_2026_5of5.py --evaluate

Ouverture définitive :
    python3 -u first_half_37_recent_2026_5of5.py --evaluate --open

PASS FINAL si les 4 conditions passent :
    - ROC-AUC > 0.50
    - PR-AUC > prévalence du test
    - Brier calibré < baseline constante 86/240
    - LogLoss calibrée < même baseline

Le modèle gelé, ses 5 features et sa calibration ne sont jamais modifiés.
Le futur pristine holdout reste séparé et intact.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
# =============================================================================
# FIXED PROTOCOL
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
FEATURE_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"
FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_frozen"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json"
SCHEMA_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_SCHEMA.json"
TRAINING_MANIFEST_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_TRAINING_MANIFEST.json"
TRAINING_DATASET_PATH = (
    BASE_DIR / "outputs" / "first_half_37"
    / "v3_draw_with_goals_feature_selection"
    / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
)
V3_CONSUMED_PATH = (
    BASE_DIR / "outputs" / "first_half_37"
    / "v3_context_benchmark"
    / "First_Half_37_V3_DEVELOPMENT_CONSUMED.csv"
)
OUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "recent_2026_external_5of5"
PROTOCOL_PATH = OUT_DIR / "RECENT_2026_5OF5_PROTOCOL_LOCK.json"
BLIND_PATH = OUT_DIR / "FIRST_HALF37_RECENT_2026_5OF5_BLIND.csv"
AUDIT_PATH = OUT_DIR / "RECENT_2026_5OF5_MATCH_AUDIT.csv"
STATE_PATH = OUT_DIR / "RECENT_2026_5OF5_COLLECTION_STATE.json"
SEALED_DIR = OUT_DIR / "sealed_timelines"
CACHE_DIR = OUT_DIR / "cache"
EVAL_DIR = OUT_DIR / "evaluation"
OPEN_LOCK_PATH = EVAL_DIR / "RECENT_2026_5OF5_OPENED_LOCK.json"
EVAL_LOCK_PATH = EVAL_DIR / "RECENT_2026_5OF5_EVALUATION_LOCK.json"
PREDICTIONS_PATH = EVAL_DIR / "RECENT_2026_5OF5_PREDICTIONS.csv"
METRICS_PATH = EVAL_DIR / "RECENT_2026_5OF5_METRICS.json"
RECENT_2026_FLOOR = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# Taille fixée avant ouverture des targets. Le collecteur remonte dans 2026 jusqu'à
# trouver 200 matchs 5/5 non consommés, sans attendre de nouveaux matchs.
TARGET_N = 200
EXPECTED_MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
EXPECTED_MODEL_STATUS = "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT"
EXPECTED_TRAINING_ROWS = 240
EXPECTED_TRAINING_POSITIVES = 86
BASELINE_RATE = EXPECTED_TRAINING_POSITIVES / EXPECTED_TRAINING_ROWS
FEATURES = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
]
TARGET = "goal_after_37_before_ht"
PER_PAGE = 100
MAX_PAGES_DAY = 50
DEFAULT_MIN_API_INTERVAL = 0.35
DEFAULT_MAX_API_REQUESTS = 50000
EPS = 1e-10
TERMINAL = {
    "accepted_blind",
    "context_excluded",
    "quality_excluded",
    "overlap_excluded",
}
# =============================================================================
# HELPERS
# =============================================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
def clean(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()
def safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(float(v))
    except Exception:
        return None
def safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON invalide: {path}")
    return obj
def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
def upsert(df: pd.DataFrame, row: Mapping[str, Any], key="match_id") -> pd.DataFrame:
    new = pd.DataFrame([dict(row)])
    if df.empty or key not in df.columns:
        return new
    kval = clean(row.get(key))
    df = df.loc[~df[key].astype(str).eq(kval)].copy()
    return pd.concat([df, new], ignore_index=True, sort=False)
def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def parse_dt(v: Any) -> Optional[datetime]:
    text = clean(v)
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()
def payload_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    data = payload.get("data", payload)
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, Mapping)]
    if isinstance(data, Mapping):
        for key in ("matches", "events", "shots", "items", "results"):
            x = data.get(key)
            if isinstance(x, list):
                return [dict(v) for v in x if isinstance(v, Mapping)]
    return []
def payload_meta(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        return meta
    data = payload.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("meta"), Mapping):
        return data["meta"]
    return {}
def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def iter_dates_desc(end: date, start: date):
    """Itère du jour le plus récent vers le plus ancien, bornes incluses."""
    d = end
    while d >= start:
        yield d
        d -= timedelta(days=1)
def sanitize(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(v))
# =============================================================================
# FREEZE + PROTOCOL
# =============================================================================
def verify_freeze() -> Dict[str, Any]:
    required = [
        MODEL_PATH,
        FREEZE_LOCK_PATH,
        SCHEMA_PATH,
        TRAINING_MANIFEST_PATH,
        TRAINING_DATASET_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    lock = load_json(FREEZE_LOCK_PATH)
    if lock.get("status") != EXPECTED_MODEL_STATUS:
        raise RuntimeError("Freeze status inattendu.")
    if lock.get("model_name") != EXPECTED_MODEL_NAME:
        raise RuntimeError("Frozen model_name inattendu.")
    if list(lock.get("features", [])) != FEATURES:
        raise RuntimeError("Frozen features inattendues.")
    if int(lock.get("training_rows", -1)) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError("Frozen training rows inattendu.")
    if int(lock.get("training_positives", -1)) != EXPECTED_TRAINING_POSITIVES:
        raise RuntimeError("Frozen training positives inattendu.")
    if sha256_file(MODEL_PATH) != lock.get("model_sha256"):
        raise RuntimeError("MODEL SHA mismatch.")
    if sha256_file(TRAINING_DATASET_PATH) != lock.get("training_dataset_sha256"):
        raise RuntimeError("TRAINING DATASET SHA mismatch.")
    train = pd.read_csv(TRAINING_DATASET_PATH, usecols=["kickoff_utc"])
    train_dates = pd.to_datetime(
        train["kickoff_utc"], utc=True, errors="coerce", format="mixed"
    )
    if train_dates.isna().any() or train_dates.empty:
        raise RuntimeError("Training dates invalides.")
    return lock
def ensure_protocol(lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Verrouille la date d'ancrage au premier lancement.

    Cela garantit que le jeu évalué ne change pas demain simplement parce qu'une
    nouvelle journée est devenue disponible. La sélection reste donc reproductible.
    """
    static_expected = {
        "schema_version": "first_half_37_recent_2026_external_5of5_v1",
        "sample_rule": (
            "newest 200 non-consumed draw_with_goals matches from locked anchor date "
            "backward to 2026-01-01, with all 5 frozen features available"
        ),
        "target_n": TARGET_N,
        "recent_2026_floor_utc": RECENT_2026_FLOOR.isoformat(),
        "scan_direction": "newest_to_oldest",
        "context_rule": "score_37_home == score_37_away and score_37_home > 0",
        "feature_completeness_rule": "all frozen model features present: 5/5; no imputation",
        "anti_overlap_rule": "exclude every match_id found in frozen training or V3 consumed datasets",
        "test_kind": "RECENT_2026_HISTORICAL_EXTERNAL_BY_ID_NOT_PRISTINE_TEMPORAL",
        "features": FEATURES,
        "model_name": lock["model_name"],
        "model_sha256": lock["model_sha256"],
        "calibration_alpha": float(lock["calibration_alpha"]),
        "baseline_rate": BASELINE_RATE,
        "pass_conditions": [
            "ROC_AUC > 0.50",
            "PR_AUC > holdout prevalence",
            "Brier calibrated < frozen constant baseline",
            "LogLoss calibrated < frozen constant baseline",
        ],
        "all_four_required": True,
        "targets_masked_during_collection": True,
        "scoring_disabled_during_collection": True,
        "future_pristine_holdout_untouched": True,
    }
    if not PROTOCOL_PATH.exists():
        anchor_day = datetime.now(timezone.utc).date()
        if anchor_day.year != 2026:
            anchor_day = date(2026, 12, 31) if anchor_day.year > 2026 else date(2026, 1, 1)
        created = {
            **static_expected,
            "scan_anchor_date_utc": anchor_day.isoformat(),
            "created_at_utc": now_iso(),
        }
        atomic_json(created, PROTOCOL_PATH)
        return load_json(PROTOCOL_PATH)

    current = load_json(PROTOCOL_PATH)
    for key, value in static_expected.items():
        if current.get(key) != value:
            raise RuntimeError(
                f"Protocol lock mismatch sur {key}. Ne pas modifier le test après démarrage."
            )
    anchor_text = clean(current.get("scan_anchor_date_utc"))
    try:
        anchor_day = date.fromisoformat(anchor_text)
    except Exception as exc:
        raise RuntimeError("scan_anchor_date_utc invalide dans le protocol lock.") from exc
    if anchor_day.year != 2026 or anchor_day < RECENT_2026_FLOOR.date():
        raise RuntimeError("Date d'ancrage hors de 2026.")
    return current
def consumed_ids() -> Set[str]:
    ids: Set[str] = set()
    for path in (TRAINING_DATASET_PATH, V3_CONSUMED_PATH):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["match_id"])
        except Exception:
            continue
        ids.update(df["match_id"].dropna().astype(str).str.strip())
    ids.discard("")
    return ids
# =============================================================================
# API
# =============================================================================
class BudgetReached(RuntimeError):
    pass
class API:
    def __init__(
        self,
        key: str,
        base_url: str,
        timeout: int,
        min_interval: float,
        max_requests: int,
    ):
        key = clean(key)
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = max(0.1, float(min_interval))
        self.max_requests = max(0, int(max_requests))
        self.requests = 0
        self.retries_429 = 0
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    def cache_path(self, endpoint: str, params: Optional[Mapping[str, Any]]) -> Path:
        raw = json.dumps(
            {"endpoint": endpoint, "params": dict(params or {})},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return CACHE_DIR / digest[:2] / f"{digest}.json"
    def request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        use_cache=True,
        retries=6,
    ):
        endpoint = "/" + endpoint.lstrip("/")
        cp = self.cache_path(endpoint, params)
        if use_cache and cp.exists():
            rec = load_json(cp)
            if "payload" in rec:
                return rec["payload"], 200, ""
        for attempt in range(retries):
            if self.max_requests > 0 and self.requests >= self.max_requests:
                raise BudgetReached(
                    f"budget API atteint: {self.requests}/{self.max_requests}"
                )
            elapsed = time.monotonic() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                response = self.session.get(
                    self.base_url + endpoint,
                    params=dict(params or {}),
                    timeout=self.timeout,
                )
            except Exception as exc:
                if attempt == retries - 1:
                    return None, None, f"network:{exc}"
                time.sleep(min(2 ** attempt, 20))
                continue
            self.requests += 1
            self.last_request = time.monotonic()
            if response.status_code == 429:
                self.retries_429 += 1
                if attempt == retries - 1:
                    return None, 429, "http_429"
                try:
                    wait = float(response.headers.get("Retry-After", ""))
                except Exception:
                    wait = min(2 ** attempt, 20)
                time.sleep(max(0.5, wait))
                continue
            if 500 <= response.status_code < 600:
                if attempt == retries - 1:
                    return None, response.status_code, f"http_{response.status_code}"
                time.sleep(min(2 ** attempt, 20))
                continue
            if response.status_code >= 400:
                return None, response.status_code, f"http_{response.status_code}"
            try:
                payload = response.json()
            except Exception as exc:
                return None, response.status_code, f"json:{exc}"
            if use_cache:
                cp.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    {
                        "cached_at": now_iso(),
                        "payload": payload,
                    },
                    cp,
                )
            return payload, response.status_code, ""
        return None, None, "failed"
# =============================================================================
# MATCH LISTING
# =============================================================================
KICKOFF_KEYS = (
    "kickoff_utc",
    "utc_date",
    "start_time",
    "start_at",
    "scheduled_at",
    "kickoff",
    "date",
)
def kickoff(match: Mapping[str, Any]) -> Optional[datetime]:
    for key in KICKOFF_KEYS:
        dt = parse_dt(match.get(key))
        if dt:
            return dt
    return None
def match_row(match: Mapping[str, Any], audit_mod: Any) -> Dict[str, Any]:
    mid = clean(audit_mod.extract_match_id(match))
    try:
        home = clean(audit_mod.extract_team_name(match, "home"))
        away = clean(audit_mod.extract_team_name(match, "away"))
    except Exception:
        home = away = ""
    try:
        home_id = clean(audit_mod.extract_team_id(match, "home"))
        away_id = clean(audit_mod.extract_team_id(match, "away"))
    except Exception:
        home_id = away_id = ""
    ko = kickoff(match)
    return {
        "match_id": mid,
        "kickoff_utc": ko.isoformat() if ko else "",
        "home_team": home,
        "away_team": away,
        "home_team_id": home_id,
        "away_team_id": away_id,
    }
def list_day(api: API, audit_mod: Any, d: date) -> List[Dict[str, Any]]:
    day = d.isoformat()
    matches: Dict[str, Dict[str, Any]] = {}
    for page in range(1, MAX_PAGES_DAY + 1):
        payload, status, error = api.request(
            "/football/matches",
            params={
                "date_from": day,
                "date_to": day,
                "status": "finished",
                "page": page,
                "per_page": PER_PAGE,
            },
            use_cache=True,
        )
        if payload is None:
            raise RuntimeError(
                f"listing {day} page={page}: {error or status}"
            )
        items = payload_list(payload)
        for item in items:
            row = match_row(item, audit_mod)
            if not row["match_id"]:
                continue
            ko = parse_dt(row["kickoff_utc"])
            if ko is not None and ko.date() != d:
                continue
            matches[row["match_id"]] = row
        meta = payload_meta(payload)
        current = (
            safe_int(meta.get("current_page"))
            or safe_int(meta.get("page"))
            or page
        )
        total = (
            safe_int(meta.get("total_pages"))
            or safe_int(meta.get("last_page"))
        )
        if total is not None and current >= total:
            break
        if total is None and len(items) < PER_PAGE:
            break
        if not items:
            break
    rows = list(matches.values())
    rows.sort(key=lambda x: (x["kickoff_utc"], x["match_id"]))
    return rows
# =============================================================================
# RECONSTRUCTION
# =============================================================================
def make_context(listing: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "match": dict(listing),
        "home_team_id": clean(listing.get("home_team_id")),
        "away_team_id": clean(listing.get("away_team_id")),
        "home_team": clean(listing.get("home_team")),
        "away_team": clean(listing.get("away_team")),
    }
def process(
    listing: Mapping[str, Any],
    api: API,
    feat_mod: Any,
    audit_mod: Any,
    used_ids: Set[str],
):
    mid = clean(listing.get("match_id"))
    audit = {
        "match_id": mid,
        "kickoff_utc": clean(listing.get("kickoff_utc")),
        "status": "",
        "reason": "",
        "audited_at_utc": now_iso(),
    }
    if mid in used_ids:
        audit.update(status="overlap_excluded", reason="already_consumed")
        return None, audit, False
    timeline_payload, status, error = api.request(
        f"/football/matches/{mid}/timeline",
        use_cache=True,
    )
    if timeline_payload is None:
        retryable = status is None or status == 429 or (status and 500 <= status < 600)
        audit.update(
            status="retryable_failed" if retryable else "quality_excluded",
            reason="timeline_unavailable",
        )
        return None, audit, retryable
    events = feat_mod.payload_list(timeline_payload)
    if not events:
        audit.update(status="quality_excluded", reason="empty_timeline")
        return None, audit, False
    full = bool(feat_mod.timeline_is_full(timeline_payload))
    period_end = bool(
        feat_mod.first_half_period_end_seen(events, audit_mod)
    )
    if not (full and period_end):
        audit.update(status="quality_excluded", reason="first_half_not_complete")
        return None, audit, False
    context = make_context(listing)
    base, base_prov = feat_mod.reconstruct_base_live37(
        timeline_events=events,
        shot_events=[],
        team_context=context,
        timeline_payload=timeline_payload,
        audit=audit_mod,
    )
    sh = safe_float(base.get("score_37_home"))
    sa = safe_float(base.get("score_37_away"))
    # If listing metadata insufficient, fetch exact detail and retry score.
    if sh is None or sa is None:
        detail, dstatus, derror = api.request(
            f"/football/matches/{mid}",
            use_cache=True,
        )
        if detail is not None:
            context = feat_mod.get_detail_team_context(
                detail,
                listing,
                audit_mod,
            )
            base, base_prov = feat_mod.reconstruct_base_live37(
                timeline_events=events,
                shot_events=[],
                team_context=context,
                timeline_payload=timeline_payload,
                audit=audit_mod,
            )
            sh = safe_float(base.get("score_37_home"))
            sa = safe_float(base.get("score_37_away"))
    if sh is None or sa is None:
        audit.update(status="quality_excluded", reason="score37_unresolved")
        return None, audit, False
    if not (sh == sa and sh > 0):
        audit.update(status="context_excluded", reason="not_draw_with_goals")
        return None, audit, False
    # Only qualifying-context matches pay for shotmap.
    shot_payload, shot_status, shot_error = api.request(
        f"/football/matches/{mid}/shotmap",
        use_cache=True,
    )
    shots = (
        feat_mod.payload_list(shot_payload)
        if shot_payload is not None
        else []
    )
    base, p1 = feat_mod.reconstruct_base_live37(
        timeline_events=events,
        shot_events=shots,
        team_context=context,
        timeline_payload=timeline_payload,
        audit=audit_mod,
    )
    recent, p2 = feat_mod.reconstruct_recent_windows(
        timeline_events=events,
        shot_events=shots,
        team_context=context,
        audit=audit_mod,
        timeline_full_flag=True,
        shotmap_feed_proven=(
            shot_status == 200
            and (
                bool(shots)
                or feat_mod.payload_explicitly_contains_event_list(shot_payload)
            )
        ),
    )
    xg, p3 = feat_mod.reconstruct_xg_cumulative(
        shot_events=shots,
        team_context=context,
        audit=audit_mod,
    )
    values = {}
    values.update(base)
    values.update(recent)
    values.update(xg)
    sh2 = safe_float(values.get("score_37_home"))
    sa2 = safe_float(values.get("score_37_away"))
    if sh2 is None or sa2 is None or not (sh2 == sa2 and sh2 > 0):
        audit.update(status="quality_excluded", reason="context_recheck_failed")
        return None, audit, False
    # STRICT 5/5 : aucune prédiction possible si une feature gelée est absente.
    missing_features = [f for f in FEATURES if safe_float(values.get(f)) is None]
    if missing_features:
        audit.update(
            status="quality_excluded",
            reason="frozen_features_incomplete",
            model_features_available=len(FEATURES) - len(missing_features),
            missing_model_features="|".join(missing_features),
        )
        return None, audit, False
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    timeline_file = SEALED_DIR / f"{sanitize(mid)}.json"
    atomic_json(timeline_payload, timeline_file)
    provenance = {}
    provenance.update(p1)
    provenance.update(p2)
    provenance.update(p3)
    row = {
        "match_id": mid,
        "kickoff_utc": clean(listing.get("kickoff_utc")),
        "home_team": clean(context.get("home_team")) or clean(listing.get("home_team")),
        "away_team": clean(context.get("away_team")) or clean(listing.get("away_team")),
        "score_37_home": sh2,
        "score_37_away": sa2,
        "score_37_exact": f"{int(sh2)}-{int(sa2)}",
        "score_context_37": "draw_with_goals",
        "target_status": "MASKED_BLIND",
        TARGET: np.nan,
        "sealed_timeline_file": str(timeline_file.relative_to(OUT_DIR)),
        "sealed_timeline_sha256": sha256_file(timeline_file),
        "collected_at_utc": now_iso(),
    }
    for f in FEATURES:
        row[f] = safe_float(values.get(f))
    row["model_features_available"] = int(
        sum(pd.notna(row[f]) for f in FEATURES)
    )
    row["feature_provenance_json"] = json.dumps(
        {f: provenance.get(f, "") for f in FEATURES},
        ensure_ascii=False,
        sort_keys=True,
    )
    audit.update(
        status="accepted_blind",
        reason="",
        score_37_exact=row["score_37_exact"],
        model_features_available=row["model_features_available"],
    )
    return row, audit, False
# =============================================================================
# COLLECTION
# =============================================================================
def collect(args) -> int:
    print("=" * 110, flush=True)
    print("FIRST HALF 37 — RECENT 2026 EXTERNAL 5/5", flush=True)
    print("=" * 110, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = verify_freeze()
    protocol = ensure_protocol(lock)
    used_ids = consumed_ids()
    scan_start_day = RECENT_2026_FLOOR.date()
    scan_end_day = date.fromisoformat(protocol["scan_anchor_date_utc"])
    print(f"Model SHA : {lock['model_sha256']}", flush=True)
    print(f"Ancrage récent : {scan_end_day} UTC", flush=True)
    print(f"Borne basse    : {scan_start_day} UTC", flush=True)
    print(f"Cible          : {TARGET_N} matchs LES PLUS RECENTS draw_with_goals avec 5/5", flush=True)
    print(f"Anti-overlap IDs chargés : {len(used_ids)}", flush=True)
    print("Scoring   : DESACTIVE", flush=True)
    print("Targets   : MASQUES", flush=True)
    print(flush=True)
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()
    key = clean(os.getenv("STATS_API_KEY"))
    if not key:
        print("ERREUR: STATS_API_KEY absent du .env", flush=True)
        return 2
    api = API(
        key=key,
        base_url=clean(
            os.getenv(
                "STATS_API_BASE_URL",
                "https://api.thestatsapi.com/api",
            )
        ),
        timeout=safe_int(os.getenv("STATS_API_TIMEOUT")) or 20,
        min_interval=args.min_api_interval,
        max_requests=args.max_api_requests,
    )
    print("Chargement des modules reconstruction...", flush=True)
    feat_mod = load_module("fh37_recent5_features", FEATURE_MODULE_PATH)
    audit_mod = load_module("fh37_recent5_audit", AUDIT_MODULE_PATH)
    print("Modules: OK", flush=True)
    print(flush=True)
    blind = read_csv(BLIND_PATH)
    audit_df = read_csv(AUDIT_PATH)
    accepted_ids = set()
    if not blind.empty:
        accepted_ids = set(blind["match_id"].astype(str))
    terminal_ids = set()
    if not audit_df.empty:
        terminal_ids = set(
            audit_df.loc[
                audit_df["status"].astype(str).isin(TERMINAL),
                "match_id",
            ].astype(str)
        )
    if len(accepted_ids) >= TARGET_N:
        print(f"Collection déjà complète: {len(accepted_ids)}/{TARGET_N}", flush=True)
        write_state(
            lock, blind, audit_df, api,
            status="READY_FOR_ONE_SHOT",
            last_day=None,
        )
        return 0
    last_day = None
    try:
        for d in iter_dates_desc(scan_end_day, scan_start_day):
            if len(accepted_ids) >= TARGET_N:
                break
            last_day = d.isoformat()
            print(
                f"\n=== {last_day} | accepted={len(accepted_ids)}/{TARGET_N} ===",
                flush=True,
            )
            day_matches = list_day(api, audit_mod, d)
            print(
                f"Matchs terminés listés ce jour: {len(day_matches)}",
                flush=True,
            )
            for i, listing in enumerate(reversed(day_matches), 1):
                if len(accepted_ids) >= TARGET_N:
                    break
                mid = clean(listing.get("match_id"))
                if not mid or mid in terminal_ids:
                    continue
                row, audit_row, retryable = process(
                    listing,
                    api,
                    feat_mod,
                    audit_mod,
                    used_ids,
                )
                audit_df = upsert(audit_df, audit_row)
                atomic_csv(audit_df, AUDIT_PATH)
                if retryable:
                    write_state(
                        lock,
                        blind,
                        audit_df,
                        api,
                        status="PAUSED_RETRYABLE",
                        last_day=last_day,
                    )
                    print(
                        f"RETRYABLE sur {mid}. Arrêt pour préserver la sélection newest-first.",
                        flush=True,
                    )
                    return 3
                terminal_ids.add(mid)
                if row is not None:
                    blind = upsert(blind, row)
                    blind[TARGET] = np.nan
                    blind["target_status"] = "MASKED_BLIND"
                    blind = blind.sort_values(
                        ["kickoff_utc", "match_id"]
                    ).reset_index(drop=True)
                    atomic_csv(blind, BLIND_PATH)
                    accepted_ids.add(mid)
                    print(
                        f"[{len(accepted_ids):03d}/{TARGET_N}] "
                        f"ACCEPTE {mid} | {row['score_37_exact']} | "
                        f"features={row['model_features_available']}/5",
                        flush=True,
                    )
            write_state(
                lock,
                blind,
                audit_df,
                api,
                status=(
                    "READY_FOR_ONE_SHOT"
                    if len(accepted_ids) >= TARGET_N
                    else "COLLECTING"
                ),
                last_day=last_day,
            )
    except BudgetReached as exc:
        print(f"\n{exc}", flush=True)
        print("Etat sauvegardé. Relance exactement la même commande.", flush=True)
        write_state(
            lock,
            blind,
            audit_df,
            api,
            status="PAUSED_API_BUDGET",
            last_day=last_day,
        )
        return 3
    ready = len(accepted_ids) >= TARGET_N
    write_state(
        lock,
        blind,
        audit_df,
        api,
        status="READY_FOR_ONE_SHOT" if ready else "INSUFFICIENT_2026_MATCHES",
        last_day=last_day,
    )
    print("\n" + "=" * 110, flush=True)
    print("RECENT 2026 5/5 — COLLECTION STATUS", flush=True)
    print("=" * 110, flush=True)
    print(f"Accepted : {len(accepted_ids)}/{TARGET_N}", flush=True)
    print(f"API calls this run : {api.requests}", flush=True)
    print(f"429 retries        : {api.retries_429}", flush=True)
    print(f"Blind dataset      : {BLIND_PATH}", flush=True)
    print(
        f"Status             : {'READY_FOR_ONE_SHOT' if ready else 'INSUFFICIENT_2026_MATCHES'}",
        flush=True,
    )
    print("AUCUN TARGET OUVERT. AUCUN SCORING EFFECTUE. 5/5 STRICT. PAS D'ATTENTE.", flush=True)
    return 0 if ready else 3
def write_state(lock, blind, audit_df, api, status, last_day):
    accepted = 0 if blind.empty else len(blind)
    state = {
        "schema_version": "first_half_37_recent_2026_5of5_collection_state_v1",
        "updated_at_utc": now_iso(),
        "status": status,
        "target_n": TARGET_N,
        "accepted_n": int(accepted),
        "last_day_processed": last_day,
        "scan_floor_date_utc": RECENT_2026_FLOOR.date().isoformat(),
        "scan_anchor_date_utc": load_json(PROTOCOL_PATH)["scan_anchor_date_utc"],
        "model_sha256": lock["model_sha256"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "blind_dataset_sha256": (
            sha256_file(BLIND_PATH)
            if BLIND_PATH.exists()
            else None
        ),
        "audit_sha256": (
            sha256_file(AUDIT_PATH)
            if AUDIT_PATH.exists()
            else None
        ),
        "api_requests_last_run": api.requests,
        "http_429_retries_last_run": api.retries_429,
        "strict_5of5": True,
        "future_pristine_holdout_untouched": True,
    }
    atomic_json(state, STATE_PATH)
# =============================================================================
# EVALUATION
# =============================================================================
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
def metric_block(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    prevalence = float(y.mean())
    pr = float(average_precision_score(y, p))
    roc = float(roc_auc_score(y, p))
    base = np.full(len(y), BASELINE_RATE, dtype=float)
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, base))
    ll = float(
        log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0,1])
    )
    base_ll = float(
        log_loss(y, np.clip(base, EPS, 1-EPS), labels=[0,1])
    )
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prevalence,
        "mean_probability": float(p.mean()),
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift": pr - prevalence,
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_improvement": base_brier - brier,
        "log_loss": ll,
        "baseline_logloss": base_ll,
        "logloss_improvement": base_ll - ll,
    }
def evaluate(open_targets: bool) -> int:
    print("=" * 110, flush=True)
    print("FIRST HALF 37 — RECENT 2026 EXTERNAL 5/5 ONE-SHOT", flush=True)
    print("=" * 110, flush=True)
    if EVAL_LOCK_PATH.exists():
        old = load_json(EVAL_LOCK_PATH)
        raise RuntimeError(
            f"RECENT 2026 5/5 DEJA CONSOMME: verdict={old.get('verdict')}"
        )
    if OPEN_LOCK_PATH.exists():
        old = load_json(OPEN_LOCK_PATH)
        raise RuntimeError(
            f"RECENT 2026 5/5 DEJA OUVERT: {old.get('opened_at_utc')}"
        )
    lock = verify_freeze()
    ensure_protocol(lock)
    if not STATE_PATH.exists():
        raise RuntimeError("Aucune collection RECENT 2026 5/5.")
    state = load_json(STATE_PATH)
    if state.get("status") != "READY_FOR_ONE_SHOT":
        raise RuntimeError(
            f"Collection pas prête: status={state.get('status')} "
            f"N={state.get('accepted_n')}/{TARGET_N}"
        )
    blind = pd.read_csv(BLIND_PATH, low_memory=False)
    if len(blind) != TARGET_N:
        raise RuntimeError(
            f"Le blind dataset RECENT 2026 5/5 doit contenir exactement {TARGET_N} lignes, "
            f"trouvé={len(blind)}."
        )
    if state.get("blind_dataset_sha256") != sha256_file(BLIND_PATH):
        raise RuntimeError("Blind dataset SHA mismatch.")
    if pd.to_numeric(blind[TARGET], errors="coerce").notna().any():
        raise RuntimeError("TARGET LEAK détecté avant ouverture.")
    if blind["match_id"].astype(str).duplicated().any():
        raise RuntimeError("match_id dupliqué.")
    sh = pd.to_numeric(blind["score_37_home"], errors="coerce")
    sa = pd.to_numeric(blind["score_37_away"], errors="coerce")
    if (sh.isna() | sa.isna() | (sh != sa) | (sh <= 0)).any():
        raise RuntimeError("Contexte draw_with_goals invalide.")
    X_check = blind[FEATURES].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(X_check.to_numpy(dtype=float))
    if X_check.isna().any().any() or not finite_mask.all():
        bad_rows = int((~finite_mask.all(axis=1)).sum())
        raise RuntimeError(
            f"Couverture modèle invalide: {bad_rows} ligne(s) ne sont pas 5/5 numériques."
        )
    if "model_features_available" in blind.columns:
        coverage = pd.to_numeric(blind["model_features_available"], errors="coerce")
        if coverage.isna().any() or not coverage.eq(len(FEATURES)).all():
            raise RuntimeError(
                "Préflight refusé: model_features_available doit être 5/5 partout."
            )
    kickoff_dates = pd.to_datetime(
        blind["kickoff_utc"], utc=True, errors="coerce", format="mixed"
    )
    protocol_now = load_json(PROTOCOL_PATH)
    floor_ts = pd.Timestamp(RECENT_2026_FLOOR)
    anchor_ts = pd.Timestamp(protocol_now["scan_anchor_date_utc"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if kickoff_dates.isna().any() or (kickoff_dates < floor_ts).any() or (kickoff_dates > anchor_ts).any():
        raise RuntimeError(
            "Préflight refusé: au moins une date est hors de la fenêtre 2026 verrouillée."
        )
    used = consumed_ids()
    overlaps = sorted(set(blind["match_id"].astype(str)) & used)
    if overlaps:
        raise RuntimeError(
            f"Préflight refusé: {len(overlaps)} match(s) déjà consommé(s) détecté(s)."
        )
    bundle = joblib.load(MODEL_PATH)
    print("PREFLIGHT: PASS", flush=True)
    print(f"N blind    : {len(blind)}", flush=True)
    print(f"Model SHA  : {lock['model_sha256']}", flush=True)
    print("Test récent 2026 : EXTERNAL BY ID | 5/5 STRICT", flush=True)
    if not open_targets:
        print("TARGETS: FERMES", flush=True)
        print(
            "\nPour ouvrir UNE SEULE FOIS:\n"
            "python3 -u first_half_37_recent_2026_5of5.py --evaluate --open",
            flush=True,
        )
        return 0
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(
        {
            "status": "OPENED_IN_PROGRESS",
            "opened_at_utc": now_iso(),
            "n": TARGET_N,
            "model_sha256": lock["model_sha256"],
            "blind_sha256": sha256_file(BLIND_PATH),
            "warning": "RECENT 2026 external 5/5 test consumed from this moment.",
        },
        OPEN_LOCK_PATH,
    )
    feat_mod = load_module("fh37_recent5_eval_feat", FEATURE_MODULE_PATH)
    audit_mod = load_module("fh37_recent5_eval_audit", AUDIT_MODULE_PATH)
    targets = []
    for _, row in blind.iterrows():
        timeline_path = OUT_DIR / str(row["sealed_timeline_file"])
        if sha256_file(timeline_path) != str(row["sealed_timeline_sha256"]):
            raise RuntimeError(
                f"Timeline SHA mismatch: {row['match_id']}"
            )
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
        events = feat_mod.payload_list(payload)
        result = feat_mod.resolve_target(
            timeline_events=events,
            timeline_payload=payload,
            audit=audit_mod,
        )
        if result.get("target_status") != "RESOLVED":
            raise RuntimeError(
                f"Target unresolved: {row['match_id']}"
            )
        target = result.get(TARGET)
        if target not in (0, 1, 0.0, 1.0):
            raise RuntimeError(
                f"Target invalide: {row['match_id']} -> {target}"
            )
        targets.append(int(target))
    opened = blind.copy()
    opened[TARGET] = targets
    opened["target_status"] = "RESOLVED_ONE_SHOT"
    X = opened[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    y = opened[TARGET].to_numpy(dtype=int)
    p_raw = bundle["base_model"].predict_proba(X)[:, 1]
    alpha = float(bundle["calibration"]["alpha"])
    p_cal = sigmoid(logit(p_raw) + alpha)
    opened["probability_raw"] = p_raw
    opened["probability_calibrated"] = p_cal
    m = metric_block(y, p_cal)
    raw_m = metric_block(y, p_raw)
    conditions = {
        "roc_auc_gt_0_50": m["roc_auc"] > 0.50,
        "pr_auc_gt_prevalence": m["pr_auc"] > m["positive_rate"],
        "brier_better_than_baseline": m["brier"] < m["baseline_brier"],
        "logloss_better_than_baseline": m["log_loss"] < m["baseline_logloss"],
    }
    passed = all(conditions.values())
    verdict = "RECENT_5OF5_PASS" if passed else "RECENT_5OF5_FAIL"
    opened.to_csv(PREDICTIONS_PATH, index=False)
    atomic_json(
        {
            "verdict": verdict,
            "calibrated": m,
            "raw_diagnostic": raw_m,
            "conditions": conditions,
        },
        METRICS_PATH,
    )
    atomic_json(
        {
            "status": "RECENT_5OF5_CONSUMED",
            "evaluated_at_utc": now_iso(),
            "verdict": verdict,
            "n": TARGET_N,
            "model_sha256": lock["model_sha256"],
            "blind_sha256": sha256_file(BLIND_PATH),
            "metrics": m,
            "conditions": conditions,
            "strict_5of5": True,
        "future_pristine_holdout_untouched": True,
        },
        EVAL_LOCK_PATH,
    )
    open_lock = load_json(OPEN_LOCK_PATH)
    open_lock["status"] = "COMPLETED"
    open_lock["verdict"] = verdict
    open_lock["completed_at_utc"] = now_iso()
    atomic_json(open_lock, OPEN_LOCK_PATH)
    print("\n" + "=" * 110, flush=True)
    print("RECENT 2026 5/5 — RESULTAT FINAL", flush=True)
    print("=" * 110, flush=True)
    print(f"VERDICT : {verdict}", flush=True)
    print(f"N        : {TARGET_N}", flush=True)
    print(
        f"Positifs : {m['positives']} ({m['positive_rate']:.2%})",
        flush=True,
    )
    print(f"ROC-AUC  : {m['roc_auc']:.4f}", flush=True)
    print(
        f"PR-AUC   : {m['pr_auc']:.4f} | "
        f"base={m['positive_rate']:.4f} | "
        f"lift={m['pr_lift']:+.4f}",
        flush=True,
    )
    print(
        f"Brier    : {m['brier']:.4f} | "
        f"baseline={m['baseline_brier']:.4f} | "
        f"imp={m['brier_improvement']:+.4f}",
        flush=True,
    )
    print(
        f"LogLoss  : {m['log_loss']:.4f} | "
        f"baseline={m['baseline_logloss']:.4f} | "
        f"imp={m['logloss_improvement']:+.4f}",
        flush=True,
    )
    print("Conditions:", flush=True)
    for key, value in conditions.items():
        print(
            f"  {key:<38} {'PASS' if value else 'FAIL'}",
            flush=True,
        )
    print("Test récent 2026 : EXTERNAL BY ID | 5/5 STRICT", flush=True)
    return 0 if passed else 4
# =============================================================================
# ENTRY
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Préflight/évaluation du test externe récent 2026 5/5 au lieu de collecter.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Avec --evaluate: ouvre les targets UNE SEULE FOIS.",
    )
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=DEFAULT_MAX_API_REQUESTS,
    )
    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=DEFAULT_MIN_API_INTERVAL,
    )
    args = parser.parse_args()
    if args.open and not args.evaluate:
        parser.error("--open nécessite --evaluate")
    if args.evaluate:
        return evaluate(args.open)
    return collect(args)
if __name__ == "__main__":
    raise SystemExit(main())
