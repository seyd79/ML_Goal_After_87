#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS HISTORICAL EXTERNAL HOLDOUT COLLECTOR (BLIND)
================================================================================

OBJECTIF
--------
Construire un HOLDOUT HISTORIQUE EXTERNE, séparé temporellement du développement,
sans toucher au Future Pristine Holdout.

Fenêtre verrouillée:
    2024-01-01 00:00:00 UTC
    ->
    2025-04-12 23:59:59 UTC

Pourquoi cette fenêtre:
    Le développement First Half 37 utilisé pour construire la V3 commence
    ensuite. Cette fenêtre est donc entièrement antérieure au bloc de
    développement connu.

Population:
    TOUS les matchs football terminés listés par TheStatsAPI dans la fenêtre.

Inclusion du holdout:
    - timeline 1re mi-temps prouvée complète;
    - fin de première période / halftime observée;
    - score à 37' reconstructible;
    - score_37_home == score_37_away > 0  (draw_with_goals).

Blind protocol:
    - aucun scoring modèle ici;
    - aucune probabilité calculée;
    - target non affiché;
    - target écrit NaN dans le dataset blind;
    - timeline brute sauvegardée dans sealed_target_source;
    - l'évaluateur one-shot ouvrira les targets plus tard;
    - missing != 0;
    - stats finales interdites pour reconstruire les 5 prédicteurs.

Optimisation API:
    1) listing historique par jour;
    2) timeline pour chaque match listé;
    3) shotmap seulement si le match est draw_with_goals à 37';
    4) détail match seulement si les noms ne suffisent pas à reconstruire
       le score ou à identifier les côtés;
    5) cache local + reprise.

Le collecteur est resumable. Un garde-fou d'appels API par exécution est
disponible via --max-api-requests. Le protocole est fixe et ne permet pas
de modifier la fenêtre via CLI.

Dépendances projet à la racine:
    collect_first_half_37_dataset.py
    audit_historical_live87_reconstructability.py
    .env
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
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

FEATURE_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_frozen"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json"
SCHEMA_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_SCHEMA.json"
TRAINING_MANIFEST_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_TRAINING_MANIFEST.json"

PRIMARY_CONSUMED_DATASET = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_context_benchmark"
    / "First_Half_37_V3_DEVELOPMENT_CONSUMED.csv"
)

TRAINING_DATASET_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_feature_selection"
    / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_historical_external_holdout"
)

PROTOCOL_LOCK_PATH = OUTPUT_DIR / "FIRST_HALF37_V3_HISTORICAL_EXTERNAL_PROTOCOL_LOCK.json"
BLIND_DATASET_PATH = OUTPUT_DIR / "First_Half_37_V3_HISTORICAL_EXTERNAL_HOLDOUT_BLIND.csv"
LISTING_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_listing.csv"
DAY_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_listing_by_day.csv"
MATCH_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_match_audit.csv"
COLLECTION_MANIFEST_PATH = OUTPUT_DIR / "FIRST_HALF37_V3_HISTORICAL_EXTERNAL_COLLECTION_MANIFEST.json"
RAW_TIMELINE_DIR = OUTPUT_DIR / "sealed_target_source" / "timeline"
API_CACHE_DIR = OUTPUT_DIR / "cache" / "api"

WINDOW_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 4, 12, 23, 59, 59, tzinfo=timezone.utc)

EXPECTED_MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
EXPECTED_MODEL_STATUS = "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT"
EXPECTED_TRAINING_ROWS = 240
EXPECTED_TRAINING_POSITIVES = 86
FROZEN_BASELINE_RATE = EXPECTED_TRAINING_POSITIVES / EXPECTED_TRAINING_ROWS

FEATURES = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
]

TARGET = "goal_after_37_before_ht"

# One-shot evaluation will refuse if the historical external sample is tiny.
MIN_ONE_SHOT_N = 100

PER_PAGE = 100
MAX_LIST_PAGES = 50
DEFAULT_MIN_API_INTERVAL = 0.35
DEFAULT_MAX_API_REQUESTS = 50000

TERMINAL_STATUSES = {
    "accepted_blind",
    "context_excluded",
    "quality_excluded",
    "overlap_excluded",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(float(value))
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def upsert(df: pd.DataFrame, row: Mapping[str, Any], key: str = "match_id") -> pd.DataFrame:
    new = pd.DataFrame([dict(row)])
    if df.empty or key not in df.columns:
        return new.reset_index(drop=True)

    key_value = clean(row.get(key))
    mask = df[key].astype(str).eq(key_value)
    if mask.any():
        df = df.loc[~mask].copy()

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


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_dt(value: Any) -> Optional[datetime]:
    text = clean(value)
    if not text:
        return None

    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


# =============================================================================
# GENERIC PAYLOAD HELPERS
# =============================================================================

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
            value = data.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]

    for key in ("matches", "events", "shots", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, Mapping)]

    return []


def payload_meta(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        return meta

    data = payload.get("data")
    if isinstance(data, Mapping):
        meta = data.get("meta")
        if isinstance(meta, Mapping):
            return meta

    return {}


def deep_get(obj: Any, path: Sequence[str]) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def first_value(obj: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = deep_get(obj, path)
        if value is not None and clean(value):
            return value
    return None


KICKOFF_PATHS = [
    ("kickoff_utc",),
    ("utc_date",),
    ("start_time",),
    ("start_at",),
    ("scheduled_at",),
    ("kickoff",),
    ("date",),
    ("start", "utc"),
    ("start", "date"),
    ("time", "utc"),
    ("time", "date"),
    ("kickoff", "utc"),
    ("kickoff", "date"),
]


def extract_kickoff(match: Mapping[str, Any]) -> Optional[datetime]:
    for path in KICKOFF_PATHS:
        value = deep_get(match, path)
        parsed = parse_dt(value)
        if parsed is not None:
            return parsed
    return None


def extract_competition_id_generic(match: Mapping[str, Any]) -> str:
    value = first_value(
        match,
        [
            ("competition_id",),
            ("competition", "id"),
            ("competition", "competition_id"),
            ("tournament", "id"),
            ("league", "id"),
        ],
    )
    return clean(value)


def extract_season_id_generic(match: Mapping[str, Any]) -> str:
    value = first_value(
        match,
        [
            ("season_id",),
            ("season", "id"),
            ("season", "season_id"),
            ("competition", "season_id"),
        ],
    )
    return clean(value)


# =============================================================================
# PROTOCOL LOCK
# =============================================================================

def expected_protocol(lock: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "first_half_37_v3_historical_external_protocol_v1",
        "locked_before_target_open": True,
        "window_start_utc": WINDOW_START.isoformat(),
        "window_end_utc": WINDOW_END.isoformat(),
        "population": "ALL finished football matches returned by TheStatsAPI in fixed window",
        "inclusion_rule": (
            "full first-half timeline AND first-half period end observed AND "
            "score_37_home == score_37_away > 0"
        ),
        "target": TARGET,
        "target_rule": (
            "1 iff a first-half goal has base minute > 37 before first-half period end; "
            "goal stamped 37 belongs to predictors"
        ),
        "features": FEATURES,
        "model_name": lock.get("model_name"),
        "model_sha256": lock.get("model_sha256"),
        "calibration_alpha": float(lock.get("calibration_alpha")),
        "frozen_baseline_rate": FROZEN_BASELINE_RATE,
        "minimum_one_shot_n": MIN_ONE_SHOT_N,
        "final_pass_conditions": {
            "roc_auc_gt_0_50": True,
            "pr_auc_gt_holdout_prevalence": True,
            "brier_lt_constant_frozen_train_rate_baseline": True,
            "logloss_lt_constant_frozen_train_rate_baseline": True,
            "all_four_required": True,
        },
        "no_tuning_after_collection": True,
        "no_feature_changes": True,
        "no_model_changes": True,
        "no_calibration_changes": True,
        "historical_result_is_external_validation_not_future_prospective_validation": True,
    }


def ensure_protocol_lock(lock: Mapping[str, Any]) -> Dict[str, Any]:
    expected = expected_protocol(lock)

    if not PROTOCOL_LOCK_PATH.exists():
        payload = {
            **expected,
            "created_at_utc": now_utc_iso(),
        }
        atomic_json(payload, PROTOCOL_LOCK_PATH)
        return payload

    current = load_json(PROTOCOL_LOCK_PATH)

    # Ignore creation timestamp when checking immutability.
    comparable = dict(current)
    comparable.pop("created_at_utc", None)

    if comparable != expected:
        raise RuntimeError(
            "PROTOCOL LOCK mismatch. Ne pas modifier la fenêtre/gates après création."
        )

    return current


# =============================================================================
# FREEZE / OVERLAP VERIFICATION
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
        raise RuntimeError(
            f"Freeze status={lock.get('status')}, attendu={EXPECTED_MODEL_STATUS}"
        )

    if lock.get("model_name") != EXPECTED_MODEL_NAME:
        raise RuntimeError("Freeze model_name inattendu.")

    if list(lock.get("features", [])) != FEATURES:
        raise RuntimeError("Freeze feature list inattendue.")

    if int(lock.get("training_rows", -1)) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError("Freeze training_rows inattendu.")

    if int(lock.get("training_positives", -1)) != EXPECTED_TRAINING_POSITIVES:
        raise RuntimeError("Freeze training_positives inattendu.")

    if sha256_file(MODEL_PATH) != lock.get("model_sha256"):
        raise RuntimeError("MODEL SHA mismatch.")

    if sha256_file(SCHEMA_PATH) != lock.get("schema_sha256"):
        raise RuntimeError("SCHEMA SHA mismatch.")

    if sha256_file(TRAINING_MANIFEST_PATH) != lock.get("training_manifest_sha256"):
        raise RuntimeError("TRAINING MANIFEST SHA mismatch.")

    if sha256_file(TRAINING_DATASET_PATH) != lock.get("training_dataset_sha256"):
        raise RuntimeError("TRAINING DATASET SHA mismatch.")

    # Hard temporal separation from the specialized training dataset.
    train = pd.read_csv(
        TRAINING_DATASET_PATH,
        usecols=["match_id", "kickoff_utc"],
        low_memory=False,
    )
    train_dates = pd.to_datetime(
        train["kickoff_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    if train_dates.isna().any():
        raise RuntimeError("Training kickoff_utc non parsable.")

    if WINDOW_END >= train_dates.min().to_pydatetime():
        raise RuntimeError(
            "Historical external window overlaps temporal training range."
        )

    return lock


def load_consumed_match_ids() -> Set[str]:
    """
    Défense supplémentaire. Le bloc V3 2699 est la source principale si présent.
    On ajoute plusieurs artefacts connus, sans dépendre d'eux pour la séparation
    temporelle.
    """
    candidates = [
        PRIMARY_CONSUMED_DATASET,
        TRAINING_DATASET_PATH,
        BASE_DIR / "outputs" / "first_half_37" / "historical_api" / "First_Half_37_COMBINED.csv",
    ]

    ids: Set[str] = set()

    for path in candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path, usecols=["match_id"], low_memory=False)
        except Exception:
            continue

        ids.update(df["match_id"].dropna().astype(str).str.strip())
        ids.discard("")

    return ids


# =============================================================================
# API CLIENT
# =============================================================================

class APIBudgetReached(RuntimeError):
    pass


class APIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int,
        min_interval: float,
        max_api_requests: int,
    ):
        key = clean(api_key)
        if key.lower().startswith("bearer "):
            key = key[7:].strip()

        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.min_interval = max(0.1, float(min_interval))
        self.max_api_requests = max(0, int(max_api_requests))
        self.request_count = 0
        self.retry_429_count = 0
        self.last_request_monotonic = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
        )

        API_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> Path:
        raw = json.dumps(
            {
                "endpoint": endpoint,
                "params": dict(params or {}),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return API_CACHE_DIR / digest[:2] / f"{digest}.json"

    def _check_budget(self) -> None:
        if self.max_api_requests > 0 and self.request_count >= self.max_api_requests:
            raise APIBudgetReached(
                f"Budget d'appels atteint: {self.request_count}/{self.max_api_requests}"
            )

    def request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        use_cache: bool = True,
        max_retries: int = 6,
    ) -> Tuple[Optional[Any], Optional[int], str]:
        endpoint = "/" + endpoint.lstrip("/")

        cache_path = self._cache_path(endpoint, params)

        if use_cache and cache_path.exists():
            try:
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(record, Mapping) and "payload" in record:
                    return record["payload"], 200, ""
            except Exception:
                pass

        for attempt in range(1, max_retries + 1):
            self._check_budget()

            elapsed = time.monotonic() - self.last_request_monotonic
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                response = self.session.get(
                    self.base_url + endpoint,
                    params=dict(params or {}),
                    timeout=self.timeout,
                )
            except Exception as exc:
                if attempt >= max_retries:
                    return None, None, f"network:{exc}"
                time.sleep(min(2 ** (attempt - 1), 30))
                continue

            self.request_count += 1
            self.last_request_monotonic = time.monotonic()

            if response.status_code == 429:
                self.retry_429_count += 1

                if attempt >= max_retries:
                    return None, 429, "http_429_exhausted"

                try:
                    retry_after = float(response.headers.get("Retry-After", ""))
                except Exception:
                    retry_after = min(2 ** (attempt - 1), 30)

                time.sleep(max(0.5, retry_after))
                continue

            if 500 <= response.status_code < 600:
                if attempt >= max_retries:
                    return None, response.status_code, f"http_{response.status_code}"
                time.sleep(min(2 ** (attempt - 1), 30))
                continue

            if response.status_code >= 400:
                return None, response.status_code, f"http_{response.status_code}"

            try:
                payload = response.json()
            except Exception as exc:
                return None, response.status_code, f"json:{exc}"

            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    {
                        "cached_at_utc": now_utc_iso(),
                        "endpoint": endpoint,
                        "params": dict(params or {}),
                        "payload": payload,
                    },
                    cache_path,
                )

            return payload, response.status_code, ""

        return None, None, "retry_exhausted"


# =============================================================================
# LISTING
# =============================================================================

def listing_record(
    match: Mapping[str, Any],
    audit_mod: Any,
    listing_day: str,
) -> Dict[str, Any]:
    match_id = clean(audit_mod.extract_match_id(match))
    kickoff = extract_kickoff(match)

    try:
        home_team = clean(audit_mod.extract_team_name(match, "home"))
        away_team = clean(audit_mod.extract_team_name(match, "away"))
    except Exception:
        home_team = ""
        away_team = ""

    return {
        "match_id": match_id,
        "listing_day_utc": listing_day,
        "kickoff_utc": kickoff.isoformat() if kickoff else "",
        "home_team": home_team,
        "away_team": away_team,
        "competition_id": extract_competition_id_generic(match),
        "season_id": extract_season_id_generic(match),
    }


def list_finished_day(
    client: APIClient,
    audit_mod: Any,
    day: date,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    day_string = day.isoformat()
    seen: Dict[str, Dict[str, Any]] = {}
    pages_used = 0
    complete = False
    error = ""

    for page in range(1, MAX_LIST_PAGES + 1):
        payload, status, request_error = client.request(
            "/football/matches",
            params={
                "date_from": day_string,
                "date_to": day_string,
                "status": "finished",
                "page": page,
                "per_page": PER_PAGE,
            },
            use_cache=True,
        )

        if payload is None:
            error = request_error or f"http_{status}"
            break

        items = payload_list(payload)
        pages_used = page

        for item in items:
            rec = listing_record(item, audit_mod, day_string)
            if not rec["match_id"]:
                continue

            kickoff = parse_dt(rec["kickoff_utc"])
            if kickoff is not None and kickoff.date() != day:
                continue

            seen[rec["match_id"]] = rec

        meta = payload_meta(payload)
        current_page = (
            safe_int(meta.get("current_page"))
            or safe_int(meta.get("page"))
            or page
        )
        total_pages = (
            safe_int(meta.get("total_pages"))
            or safe_int(meta.get("last_page"))
            or safe_int(meta.get("pages"))
        )

        if total_pages is not None:
            if current_page >= total_pages:
                complete = True
                break
        else:
            if len(items) < PER_PAGE:
                complete = True
                break

        if len(items) == 0:
            complete = True
            break

    if pages_used >= MAX_LIST_PAGES and not complete and not error:
        error = "max_pages_reached"

    df = pd.DataFrame(list(seen.values()))
    if not df.empty:
        df = df.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)

    audit = {
        "listing_day_utc": day_string,
        "match_count": int(len(df)),
        "pages": int(pages_used),
        "complete": int(complete),
        "error": error,
        "listed_at_utc": now_utc_iso(),
    }

    return df, audit


# =============================================================================
# MATCH COLLECTION
# =============================================================================

def endpoint_retryable(status: Optional[int], error: str) -> bool:
    if status is None:
        return True
    if status == 429:
        return True
    if 500 <= status < 600:
        return True
    return error.startswith("network:") or error.startswith("json:")


def sealed_timeline_path(match_id: str) -> Path:
    return RAW_TIMELINE_DIR / f"{sanitize_filename(match_id)}.json"


def context_from_listing(
    listing_row: Mapping[str, Any],
    audit_mod: Any,
) -> Dict[str, Any]:
    return {
        "match": dict(listing_row),
        "home_team_id": "",
        "away_team_id": "",
        "home_team": clean(listing_row.get("home_team")),
        "away_team": clean(listing_row.get("away_team")),
    }


def detail_team_context(
    client: APIClient,
    match_id: str,
    listing_row: Mapping[str, Any],
    feat_mod: Any,
    audit_mod: Any,
) -> Tuple[Dict[str, Any], Optional[int], str]:
    payload, status, error = client.request(
        f"/football/matches/{match_id}",
        use_cache=True,
    )

    if payload is None:
        return context_from_listing(listing_row, audit_mod), status, error

    return (
        feat_mod.get_detail_team_context(
            payload,
            listing_row,
            audit_mod,
        ),
        status,
        error,
    )


def reconstruct_score37(
    timeline_events: Sequence[Mapping[str, Any]],
    timeline_payload: Any,
    team_context: Mapping[str, Any],
    feat_mod: Any,
    audit_mod: Any,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, str]]:
    features, provenance = feat_mod.reconstruct_base_live37(
        timeline_events=timeline_events,
        shot_events=[],
        team_context=team_context,
        timeline_payload=timeline_payload,
        audit=audit_mod,
    )

    return (
        safe_float(features.get("score_37_home")),
        safe_float(features.get("score_37_away")),
        features,
        provenance,
    )


def process_match(
    listing_row: Mapping[str, Any],
    client: APIClient,
    feat_mod: Any,
    audit_mod: Any,
    consumed_ids: Set[str],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    match_id = clean(listing_row.get("match_id"))

    audit: Dict[str, Any] = {
        "match_id": match_id,
        "audited_at_utc": now_utc_iso(),
        "status": "",
        "reason": "",
        "kickoff_utc": clean(listing_row.get("kickoff_utc")),
        "home_team": clean(listing_row.get("home_team")),
        "away_team": clean(listing_row.get("away_team")),
    }

    if not match_id:
        audit.update(status="quality_excluded", reason="missing_match_id")
        return None, audit

    if match_id in consumed_ids:
        audit.update(status="overlap_excluded", reason="match_id_seen_in_consumed_development")
        return None, audit

    kickoff = parse_dt(listing_row.get("kickoff_utc"))
    if kickoff is not None:
        if kickoff < WINDOW_START or kickoff > WINDOW_END:
            audit.update(status="quality_excluded", reason="outside_locked_window")
            return None, audit

    timeline_payload, timeline_status, timeline_error = client.request(
        f"/football/matches/{match_id}/timeline",
        use_cache=True,
    )

    audit["timeline_http_status"] = timeline_status
    audit["timeline_error"] = timeline_error

    if timeline_payload is None:
        if endpoint_retryable(timeline_status, timeline_error):
            audit.update(status="retryable_failed", reason="timeline_unavailable_retryable")
        else:
            audit.update(status="quality_excluded", reason="timeline_unavailable_permanent")
        return None, audit

    timeline_events = feat_mod.payload_list(timeline_payload)

    if not timeline_events:
        audit.update(status="quality_excluded", reason="empty_timeline")
        return None, audit

    timeline_full = bool(feat_mod.timeline_is_full(timeline_payload))
    period_end = bool(
        feat_mod.first_half_period_end_seen(
            timeline_events,
            audit_mod,
        )
    )

    audit["timeline_full"] = int(timeline_full)
    audit["first_half_period_end_seen"] = int(period_end)

    if not (timeline_full and period_end):
        audit.update(status="quality_excluded", reason="first_half_not_proven_complete")
        return None, audit

    # First attempt without match-detail call.
    team_context = context_from_listing(listing_row, audit_mod)

    score_home, score_away, base_features, base_prov = reconstruct_score37(
        timeline_events,
        timeline_payload,
        team_context,
        feat_mod,
        audit_mod,
    )

    # If score unresolved, use exact match detail and retry.
    if score_home is None or score_away is None:
        team_context, detail_status, detail_error = detail_team_context(
            client,
            match_id,
            listing_row,
            feat_mod,
            audit_mod,
        )
        audit["detail_http_status"] = detail_status
        audit["detail_error"] = detail_error

        score_home, score_away, base_features, base_prov = reconstruct_score37(
            timeline_events,
            timeline_payload,
            team_context,
            feat_mod,
            audit_mod,
        )

    if score_home is None or score_away is None:
        audit.update(status="quality_excluded", reason="score_37_unresolved")
        return None, audit

    audit["score_37_home"] = score_home
    audit["score_37_away"] = score_away

    if not (score_home == score_away and score_home > 0):
        audit.update(status="context_excluded", reason="not_draw_with_goals_at_37")
        return None, audit

    # For an accepted-context candidate, ensure team ids/names are as strong as possible.
    if not clean(team_context.get("home_team_id")) or not clean(team_context.get("away_team_id")):
        stronger_context, detail_status, detail_error = detail_team_context(
            client,
            match_id,
            listing_row,
            feat_mod,
            audit_mod,
        )
        audit["detail_http_status"] = detail_status
        audit["detail_error"] = detail_error

        if (
            clean(stronger_context.get("home_team"))
            or clean(stronger_context.get("away_team"))
            or clean(stronger_context.get("home_team_id"))
            or clean(stronger_context.get("away_team_id"))
        ):
            team_context = stronger_context

    # Shotmap only after context qualifies.
    shot_payload, shot_status, shot_error = client.request(
        f"/football/matches/{match_id}/shotmap",
        use_cache=True,
    )

    audit["shotmap_http_status"] = shot_status
    audit["shotmap_error"] = shot_error

    shot_events = (
        feat_mod.payload_list(shot_payload)
        if shot_payload is not None
        else []
    )

    base_features, base_prov = feat_mod.reconstruct_base_live37(
        timeline_events=timeline_events,
        shot_events=shot_events,
        team_context=team_context,
        timeline_payload=timeline_payload,
        audit=audit_mod,
    )

    recent_features, recent_prov = feat_mod.reconstruct_recent_windows(
        timeline_events=timeline_events,
        shot_events=shot_events,
        team_context=team_context,
        audit=audit_mod,
        timeline_full_flag=True,
        shotmap_feed_proven=(
            shot_status == 200
            and (
                bool(shot_events)
                or feat_mod.payload_explicitly_contains_event_list(shot_payload)
            )
        ),
    )

    xg_features, xg_prov = feat_mod.reconstruct_xg_cumulative(
        shot_events=shot_events,
        team_context=team_context,
        audit=audit_mod,
    )

    all_features: Dict[str, Any] = {}
    all_features.update(base_features)
    all_features.update(recent_features)
    all_features.update(xg_features)

    score_home2 = safe_float(all_features.get("score_37_home"))
    score_away2 = safe_float(all_features.get("score_37_away"))

    if (
        score_home2 is None
        or score_away2 is None
        or not (score_home2 == score_away2 and score_home2 > 0)
    ):
        audit.update(status="quality_excluded", reason="context_recheck_failed")
        return None, audit

    sealed = sealed_timeline_path(match_id)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(timeline_payload, sealed)
    timeline_sha = sha256_file(sealed)

    provenance: Dict[str, str] = {}
    provenance.update(base_prov)
    provenance.update(recent_prov)
    provenance.update(xg_prov)

    row: Dict[str, Any] = {
        "schema_version": "first_half_37_v3_historical_external_blind_v1",
        "match_id": match_id,
        "kickoff_utc": clean(listing_row.get("kickoff_utc")),
        "home_team": clean(team_context.get("home_team")) or clean(listing_row.get("home_team")),
        "away_team": clean(team_context.get("away_team")) or clean(listing_row.get("away_team")),
        "competition_id": clean(listing_row.get("competition_id")),
        "season_id": clean(listing_row.get("season_id")),
        "score_37_home": score_home2,
        "score_37_away": score_away2,
        "score_37_exact": f"{int(score_home2)}-{int(score_away2)}",
        "score_context_37": "draw_with_goals",
        "timeline_full": 1,
        "first_half_period_end_seen": 1,
        "target_status": "MASKED_BLIND",
        TARGET: np.nan,
        "sealed_timeline_file": str(sealed.relative_to(OUTPUT_DIR)),
        "sealed_timeline_sha256": timeline_sha,
        "collected_at_utc": now_utc_iso(),
    }

    for feature in FEATURES:
        value = all_features.get(feature, np.nan)
        row[feature] = np.nan if value is None else value

    available = sum(pd.notna(row[feature]) for feature in FEATURES)

    row["model_features_available"] = int(available)
    row["model_features_total"] = len(FEATURES)
    row["feature_provenance_json"] = json.dumps(
        {feature: provenance.get(feature, "") for feature in FEATURES},
        ensure_ascii=False,
        sort_keys=True,
    )

    audit.update(
        {
            "status": "accepted_blind",
            "reason": "",
            "score_37_home": score_home2,
            "score_37_away": score_away2,
            "model_features_available": int(available),
            "model_features_total": len(FEATURES),
            "sealed_timeline_sha256": timeline_sha,
        }
    )

    return row, audit


# =============================================================================
# MANIFEST
# =============================================================================

def build_manifest(
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    listing_df: pd.DataFrame,
    day_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    consumed_ids_count: int,
) -> Dict[str, Any]:
    expected_days = {
        day.isoformat()
        for day in iter_dates(WINDOW_START.date(), WINDOW_END.date())
    }

    complete_days = set()
    if not day_df.empty and "listing_day_utc" in day_df.columns:
        for _, row in day_df.iterrows():
            if safe_int(row.get("complete")) == 1:
                complete_days.add(clean(row.get("listing_day_utc")))

    listing_complete = expected_days.issubset(complete_days)

    listed_ids = (
        set(listing_df["match_id"].dropna().astype(str).str.strip())
        if not listing_df.empty and "match_id" in listing_df.columns
        else set()
    )

    terminal_ids = set()
    retryable_ids = set()

    if not audit_df.empty and "match_id" in audit_df.columns:
        for _, row in audit_df.iterrows():
            match_id = clean(row.get("match_id"))
            status = clean(row.get("status"))

            if status in TERMINAL_STATUSES:
                terminal_ids.add(match_id)
            elif status == "retryable_failed":
                retryable_ids.add(match_id)

    pending_ids = listed_ids - terminal_ids

    collection_complete = (
        listing_complete
        and len(pending_ids) == 0
    )

    accepted = len(blind_df)

    feature_coverage: Dict[str, Any] = {}
    if accepted > 0:
        for feature in FEATURES:
            values = pd.to_numeric(blind_df[feature], errors="coerce")
            feature_coverage[feature] = {
                "available": int(values.notna().sum()),
                "total": int(accepted),
                "coverage": float(values.notna().mean()),
            }

    status_counts: Dict[str, int] = {}
    if not audit_df.empty and "status" in audit_df.columns:
        status_counts = {
            str(key): int(value)
            for key, value in audit_df["status"].astype(str).value_counts().to_dict().items()
        }

    return {
        "schema_version": "first_half_37_v3_historical_external_collection_manifest_v1",
        "created_at_utc": now_utc_iso(),
        "historical_external_not_future_pristine": True,
        "window_start_utc": WINDOW_START.isoformat(),
        "window_end_utc": WINDOW_END.isoformat(),
        "protocol_lock_path": str(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "selection_rule": protocol["inclusion_rule"],
        "blind_protocol": {
            "model_scoring_performed": False,
            "target_written_to_blind_dataset": False,
            "target_console_output": False,
            "target_source": "sealed raw timeline opened only by one-shot evaluator",
        },
        "frozen_model_name": lock.get("model_name"),
        "model_sha256": lock.get("model_sha256"),
        "freeze_lock_sha256": sha256_file(FREEZE_LOCK_PATH),
        "training_dataset_sha256": lock.get("training_dataset_sha256"),
        "consumed_match_ids_loaded": int(consumed_ids_count),
        "days_expected": int(len(expected_days)),
        "days_complete": int(len(expected_days & complete_days)),
        "listing_complete": bool(listing_complete),
        "matches_listed": int(len(listed_ids)),
        "matches_terminally_audited": int(len(listed_ids & terminal_ids)),
        "matches_pending_or_retryable": int(len(pending_ids)),
        "accepted_draw_with_goals": int(accepted),
        "status_counts": status_counts,
        "minimum_one_shot_n": MIN_ONE_SHOT_N,
        "minimum_one_shot_n_reached": bool(accepted >= MIN_ONE_SHOT_N),
        "collection_complete": bool(collection_complete),
        "feature_coverage": feature_coverage,
        "blind_dataset_path": str(BLIND_DATASET_PATH),
        "blind_dataset_sha256": (
            sha256_file(BLIND_DATASET_PATH)
            if BLIND_DATASET_PATH.exists()
            else None
        ),
        "listing_sha256": (
            sha256_file(LISTING_PATH)
            if LISTING_PATH.exists()
            else None
        ),
        "day_audit_sha256": (
            sha256_file(DAY_AUDIT_PATH)
            if DAY_AUDIT_PATH.exists()
            else None
        ),
        "match_audit_sha256": (
            sha256_file(MATCH_AUDIT_PATH)
            if MATCH_AUDIT_PATH.exists()
            else None
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collecteur BLIND du Historical External Holdout V3 draw_with_goals."
    )
    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=DEFAULT_MIN_API_INTERVAL,
    )
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=DEFAULT_MAX_API_REQUESTS,
        help=(
            "Garde-fou par exécution. Si atteint, le script sauvegarde l'état "
            "et peut être relancé. 0 = pas de limite interne."
        ),
    )
    parser.add_argument(
        "--max-new-audits",
        type=int,
        default=0,
        help="Debug/reprise: limite le nombre de nouveaux matchs audités cette exécution. 0=tous.",
    )
    args = parser.parse_args()

    print("[BOOT 1/8] main() démarré", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    print("[BOOT 2/8] dossiers output OK", flush=True)

    lock = verify_freeze()
    print("[BOOT 3/8] freeze vérifié OK", flush=True)

    protocol = ensure_protocol_lock(lock)
    print("[BOOT 4/8] protocol lock OK", flush=True)

    consumed_ids = load_consumed_match_ids()
    print(f"[BOOT 5/8] consumed IDs chargés: {len(consumed_ids)}", flush=True)

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()
    print("[BOOT 6/8] .env chargé", flush=True)

    api_key = clean(os.getenv("STATS_API_KEY"))
    base_url = clean(
        os.getenv(
            "STATS_API_BASE_URL",
            "https://api.thestatsapi.com/api",
        )
    )
    timeout = safe_int(os.getenv("STATS_API_TIMEOUT")) or 20

    if not api_key:
        print("ERREUR: STATS_API_KEY absent du .env")
        return 2

    print("[BOOT 7/8] chargement modules reconstruction...", flush=True)

    try:
        feat_mod = load_module(
            "fh37_historical_external_features",
            FEATURE_MODULE_PATH,
        )
        print("[BOOT 7a/8] collect_first_half_37_dataset.py chargé", flush=True)

        audit_mod = load_module(
            "fh37_historical_external_audit",
            AUDIT_MODULE_PATH,
        )
        print("[BOOT 7b/8] audit_historical_live87_reconstructability.py chargé", flush=True)
    except BaseException:
        print("[BOOT ERROR] échec pendant le chargement des modules:", flush=True)
        traceback.print_exc()
        raise

    client = APIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        min_interval=args.min_api_interval,
        max_api_requests=args.max_api_requests,
    )

    print("[BOOT 8/8] initialisation terminée", flush=True)
    print("=" * 120, flush=True)
    print("FIRST HALF 37 — V3 HISTORICAL EXTERNAL HOLDOUT COLLECTOR (BLIND)", flush=True)
    print("=" * 120)
    print(f"Fenêtre UTC : {WINDOW_START} -> {WINDOW_END}")
    print(f"Output      : {OUTPUT_DIR}")
    print("Scoring     : DESACTIVE")
    print("Target      : MASQUE")
    print("Population  : TOUS les matchs terminés de la fenêtre")
    print()
    print("FREEZE_LOCK : PASS")
    print(f"MODEL SHA256: {lock.get('model_sha256')}")
    print(f"Pipeline    : {lock.get('model_name')}")
    print("Features    : " + ", ".join(FEATURES))
    print(f"Consumed IDs chargés pour anti-overlap : {len(consumed_ids)}")
    print(f"Protocol SHA256 : {sha256_file(PROTOCOL_LOCK_PATH)}")
    print()

    listing_df = read_csv_or_empty(LISTING_PATH)
    day_df = read_csv_or_empty(DAY_AUDIT_PATH)
    audit_df = read_csv_or_empty(MATCH_AUDIT_PATH)
    blind_df = read_csv_or_empty(BLIND_DATASET_PATH)

    # ---------------------------------------------------------------------
    # Phase 1: complete historical listing.
    # Historical days are immutable enough for this use; complete days are
    # not relisted on resume.
    # ---------------------------------------------------------------------
    existing_complete_days = set()
    if not day_df.empty and "listing_day_utc" in day_df.columns:
        complete_mask = pd.to_numeric(
            day_df.get("complete", 0),
            errors="coerce",
        ).fillna(0).astype(int).eq(1)

        existing_complete_days = set(
            day_df.loc[complete_mask, "listing_day_utc"].astype(str)
        )

    budget_hit = False

    try:
        for day in iter_dates(WINDOW_START.date(), WINDOW_END.date()):
            day_string = day.isoformat()

            if day_string in existing_complete_days:
                continue

            df_day, day_audit = list_finished_day(
                client,
                audit_mod,
                day,
            )

            print(
                f"Listing {day_string} | matchs uniques={len(df_day)} | "
                f"pages={day_audit['pages']} | "
                f"complete={'YES' if day_audit['complete'] else 'NO'}"
            )

            if not df_day.empty:
                # Replace only this day's listing.
                if not listing_df.empty and "listing_day_utc" in listing_df.columns:
                    listing_df = listing_df[
                        ~listing_df["listing_day_utc"].astype(str).eq(day_string)
                    ].copy()

                listing_df = pd.concat(
                    [listing_df, df_day],
                    ignore_index=True,
                    sort=False,
                )

                listing_df = (
                    listing_df
                    .drop_duplicates(subset=["match_id"], keep="last")
                    .sort_values(["kickoff_utc", "match_id"])
                    .reset_index(drop=True)
                )

                atomic_csv(listing_df, LISTING_PATH)

            # Upsert day audit.
            if day_df.empty or "listing_day_utc" not in day_df.columns:
                day_df = pd.DataFrame([day_audit])
            else:
                day_df = day_df[
                    ~day_df["listing_day_utc"].astype(str).eq(day_string)
                ].copy()
                day_df = pd.concat(
                    [day_df, pd.DataFrame([day_audit])],
                    ignore_index=True,
                    sort=False,
                )

            day_df = day_df.sort_values("listing_day_utc").reset_index(drop=True)
            atomic_csv(day_df, DAY_AUDIT_PATH)

    except APIBudgetReached as exc:
        budget_hit = True
        print()
        print(f"API BUDGET GUARD: {exc}")
        print("Etat listing sauvegardé. Relance le même script pour reprendre.")

    print()
    print(f"TOTAL MATCHS LISTES ACTUELLEMENT : {len(listing_df)}")

    # ---------------------------------------------------------------------
    # Phase 2: blind audit of listed matches.
    # ---------------------------------------------------------------------
    terminal_ids = set()

    if not audit_df.empty and "match_id" in audit_df.columns:
        for _, row in audit_df.iterrows():
            if clean(row.get("status")) in TERMINAL_STATUSES:
                terminal_ids.add(clean(row.get("match_id")))

    new_audits = 0

    if not budget_hit and not listing_df.empty:
        try:
            for idx, row in listing_df.iterrows():
                match_id = clean(row.get("match_id"))

                if not match_id or match_id in terminal_ids:
                    continue

                if args.max_new_audits > 0 and new_audits >= args.max_new_audits:
                    break

                new_audits += 1

                print("-" * 120)
                print(
                    f"[{idx+1}/{len(listing_df)}] {match_id} | "
                    f"{clean(row.get('home_team'))} vs {clean(row.get('away_team'))} | "
                    f"{clean(row.get('kickoff_utc'))}"
                )

                blind_row, audit_row = process_match(
                    listing_row=row.to_dict(),
                    client=client,
                    feat_mod=feat_mod,
                    audit_mod=audit_mod,
                    consumed_ids=consumed_ids,
                )

                audit_df = upsert(audit_df, audit_row)
                atomic_csv(audit_df, MATCH_AUDIT_PATH)

                if blind_row is not None:
                    blind_df = upsert(blind_df, blind_row)

                    # Hard anti-leak: target remains NaN.
                    blind_df[TARGET] = np.nan
                    blind_df["target_status"] = "MASKED_BLIND"

                    blind_df = (
                        blind_df
                        .sort_values(["kickoff_utc", "match_id"])
                        .reset_index(drop=True)
                    )

                    atomic_csv(blind_df, BLIND_DATASET_PATH)

                    print(
                        f"ACCEPTE BLIND | score37={blind_row['score_37_exact']} | "
                        f"features={blind_row['model_features_available']}/5 | "
                        "TARGET MASQUE"
                    )
                else:
                    print(
                        f"{clean(audit_row.get('status')).upper()} | "
                        f"{clean(audit_row.get('reason'))}"
                    )

        except APIBudgetReached as exc:
            budget_hit = True
            print()
            print(f"API BUDGET GUARD: {exc}")
            print("Etat audit sauvegardé. Relance le même script pour reprendre.")

    if not blind_df.empty:
        blind_df[TARGET] = np.nan
        blind_df["target_status"] = "MASKED_BLIND"
        atomic_csv(blind_df, BLIND_DATASET_PATH)

    manifest = build_manifest(
        lock=lock,
        protocol=protocol,
        listing_df=listing_df,
        day_df=day_df,
        audit_df=audit_df,
        blind_df=blind_df,
        consumed_ids_count=len(consumed_ids),
    )

    atomic_json(manifest, COLLECTION_MANIFEST_PATH)

    print()
    print("=" * 120)
    print("HISTORICAL EXTERNAL HOLDOUT — COLLECTION BLIND STATUS")
    print("=" * 120)
    print(f"Days complete             : {manifest['days_complete']}/{manifest['days_expected']}")
    print(f"Listing complete          : {'YES' if manifest['listing_complete'] else 'NO'}")
    print(f"Matches listed            : {manifest['matches_listed']}")
    print(f"Terminally audited        : {manifest['matches_terminally_audited']}")
    print(f"Pending/retryable         : {manifest['matches_pending_or_retryable']}")
    print(f"Accepted draw_with_goals  : {manifest['accepted_draw_with_goals']}")
    print(
        f"Minimum N one-shot        : "
        f"{manifest['accepted_draw_with_goals']}/{manifest['minimum_one_shot_n']} | "
        f"{'READY' if manifest['minimum_one_shot_n_reached'] else 'NOT READY'}"
    )
    print(f"Collection complete       : {'YES' if manifest['collection_complete'] else 'NO'}")
    print(f"API requests this run     : {client.request_count}")
    print(f"HTTP 429 retries          : {client.retry_429_count}")
    print(f"Manifest                  : {COLLECTION_MANIFEST_PATH}")
    print()
    print("AUCUN SCORING MODELE N'A ETE EFFECTUE.")
    print("LE FUTURE PRISTINE HOLDOUT N'A PAS ETE TOUCHE.")

    if manifest["collection_complete"] and manifest["minimum_one_shot_n_reached"]:
        return 0

    return 3


if __name__ == "__main__":
    print("[START] collecteur historique externe lancé", flush=True)
    try:
        exit_code = main()
        print(f"[EXIT] code={exit_code}", flush=True)
        raise SystemExit(exit_code)
    except SystemExit as exc:
        # A SystemExit venant d'un module importé est normalement silencieux.
        # On l'affiche explicitement pour diagnostic.
        code = exc.code
        if code not in (0, 3):
            print(f"[SYSTEM EXIT] code={code}", flush=True)
            traceback.print_exc()
        raise
    except BaseException:
        print("[FATAL] exception non gérée:", flush=True)
        traceback.print_exc()
        raise
