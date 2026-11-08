#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS PRISTINE HOLDOUT COLLECTOR (BLIND)
=====================================================================

Fenêtre pré-enregistrée UTC:
    2026-10-06 00:00:00 -> 2026-10-31 23:59:59

Population:
    TOUS les matchs football terminés listés par TheStatsAPI dans la fenêtre.

Inclusion HOLDOUT:
    - timeline 1re mi-temps prouvée complète;
    - period_end / halftime observé;
    - score à 37' reconstructible;
    - score_37_home == score_37_away > 0  (draw_with_goals).

IMPORTANT:
    - AUCUN scoring modèle dans ce script.
    - AUCUNE probabilité calculée.
    - Le target n'est ni affiché ni écrit dans le dataset blind.
    - Le target sera reconstruit UNE SEULE FOIS par l'évaluateur final à partir
      des timelines brutes sauvegardées.
    - Les features manquantes restent NaN. Missing != 0.
    - Les 5 features gelées seulement sont nécessaires au modèle final.

Le script est resumable. Pendant la fenêtre, il peut être relancé; le manifest
ne passe collection_complete=true qu'après fermeture de la fenêtre ET audit
terminal de tous les matchs listés.

Dépendances projet à la racine:
    collect_first_half_37_dataset.py
    audit_historical_live87_reconstructability.py
    .env

Sorties:
    outputs/first_half_37/v3_draw_with_goals_final_holdout/
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

TRAINING_DATASET_PATH = (
    BASE_DIR / "outputs" / "first_half_37"
    / "v3_draw_with_goals_feature_selection"
    / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "first_half_37"
    / "v3_draw_with_goals_final_holdout"
)

BLIND_DATASET_PATH = OUTPUT_DIR / "First_Half_37_V3_DRAW_WITH_GOALS_FINAL_HOLDOUT_BLIND.csv"
LISTING_PATH = OUTPUT_DIR / "first_half_37_v3_holdout_listing.csv"
DAY_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_holdout_listing_by_day.csv"
MATCH_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_holdout_match_audit.csv"
COLLECTION_MANIFEST_PATH = OUTPUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_COLLECTION_MANIFEST.json"
RAW_TIMELINE_DIR = OUTPUT_DIR / "sealed_target_source" / "timeline"
API_CACHE_DIR = OUTPUT_DIR / "cache" / "api"

WINDOW_START = datetime(2026, 10, 6, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc)

EXPECTED_MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
EXPECTED_STATUS = "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT"
EXPECTED_TRAINING_ROWS = 240
EXPECTED_TRAINING_POSITIVES = 86

FEATURES = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
]

TARGET = "goal_after_37_before_ht"
PER_PAGE = 100
MAX_LIST_PAGES = 50
DEFAULT_MIN_API_INTERVAL = 0.35

TERMINAL_STATUSES = {
    "accepted_blind",
    "context_excluded",
    "quality_excluded",
}


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


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON invalide: {path}")
    return obj


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
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
    k = str(row.get(key, ""))
    mask = df[key].astype(str).eq(k)
    if mask.any():
        df = df.loc[~mask].copy()
    return pd.concat([df, new], ignore_index=True, sort=False)


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
            v = data.get(key)
            if isinstance(v, list):
                return [dict(x) for x in v if isinstance(x, Mapping)]

    for key in ("matches", "events", "shots", "items", "results"):
        v = payload.get(key)
        if isinstance(v, list):
            return [dict(x) for x in v if isinstance(x, Mapping)]

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
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


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
    for p in KICKOFF_PATHS:
        value = deep_get(match, p)
        dt = parse_dt(value)
        if dt is not None:
            return dt
    return None


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


class APIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 20,
        min_interval: float = DEFAULT_MIN_API_INTERVAL,
        max_retries: int = 6,
    ):
        key = clean(api_key)
        if key.lower().startswith("bearer "):
            key = key[7:].strip()

        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.min_interval = max(0.1, float(min_interval))
        self.max_retries = int(max_retries)
        self.last_request_at = 0.0
        self.request_count = 0
        self.retry_429_count = 0

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        API_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, endpoint: str, params: Optional[Mapping[str, Any]]) -> Path:
        raw = json.dumps(
            {"endpoint": endpoint, "params": dict(params or {})},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return API_CACHE_DIR / key[:2] / f"{key}.json"

    def request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        use_cache: bool = True,
    ) -> Tuple[Optional[Any], Optional[int], str]:
        endpoint = "/" + endpoint.lstrip("/")

        if use_cache:
            cp = self._cache_path(endpoint, params)
            if cp.exists():
                try:
                    rec = json.loads(cp.read_text(encoding="utf-8"))
                    if isinstance(rec, Mapping) and "payload" in rec:
                        return rec["payload"], 200, ""
                except Exception:
                    pass

        for attempt in range(1, self.max_retries + 1):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                response = self.session.get(
                    self.base_url + endpoint,
                    params=dict(params or {}),
                    timeout=self.timeout,
                )
            except Exception as exc:
                if attempt >= self.max_retries:
                    return None, None, f"network:{exc}"
                time.sleep(min(2 ** (attempt - 1), 30))
                continue

            self.last_request_at = time.monotonic()
            self.request_count += 1

            if response.status_code == 429:
                self.retry_429_count += 1
                if attempt >= self.max_retries:
                    return None, 429, "http_429_exhausted"
                try:
                    wait = float(response.headers.get("Retry-After", ""))
                except Exception:
                    wait = min(2 ** (attempt - 1), 30)
                time.sleep(max(wait, 0.5))
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
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
                cp = self._cache_path(endpoint, params)
                cp.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    {
                        "cached_at_utc": now_utc_iso(),
                        "endpoint": endpoint,
                        "params": dict(params or {}),
                        "payload": payload,
                    },
                    cp,
                )

            return payload, response.status_code, ""

        return None, None, "retry_exhausted"


def verify_freeze() -> Dict[str, Any]:
    for p in (MODEL_PATH, FREEZE_LOCK_PATH, SCHEMA_PATH, TRAINING_MANIFEST_PATH, TRAINING_DATASET_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    lock = load_json(FREEZE_LOCK_PATH)

    if lock.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"Freeze status={lock.get('status')}, attendu={EXPECTED_STATUS}")

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

    return lock


def listing_record(match: Mapping[str, Any], audit_mod: Any, listing_day: str) -> Dict[str, Any]:
    mid = clean(audit_mod.extract_match_id(match))
    kickoff = extract_kickoff(match)

    try:
        home = clean(audit_mod.extract_team_name(match, "home"))
        away = clean(audit_mod.extract_team_name(match, "away"))
    except Exception:
        home = away = ""

    try:
        competition_id = clean(audit_mod.extract_competition_id(match))
    except Exception:
        competition_id = ""

    try:
        season_id = clean(audit_mod.extract_season_id(match))
    except Exception:
        season_id = ""

    return {
        "match_id": mid,
        "listing_day_utc": listing_day,
        "kickoff_utc": kickoff.isoformat() if kickoff else "",
        "home_team": home,
        "away_team": away,
        "competition_id": competition_id,
        "season_id": season_id,
    }


def list_finished_day(
    client: APIClient,
    audit_mod: Any,
    day: date,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    day_s = day.isoformat()
    seen: Dict[str, Dict[str, Any]] = {}
    complete = False
    pages_used = 0
    error = ""

    for page in range(1, MAX_LIST_PAGES + 1):
        payload, status, err = client.request(
            "/football/matches",
            params={
                "date_from": day_s,
                "date_to": day_s,
                "status": "finished",
                "page": page,
                "per_page": PER_PAGE,
            },
            use_cache=False,  # listing is always refreshed; avoids stale in-window pages
        )

        if payload is None:
            error = err or f"http_{status}"
            complete = False
            break

        rows = payload_list(payload)
        pages_used = page

        for item in rows:
            rec = listing_record(item, audit_mod, day_s)
            if not rec["match_id"]:
                continue

            # Filter again by parsed UTC kickoff when available.
            ko = parse_dt(rec["kickoff_utc"])
            if ko is not None and ko.date() != day:
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
            if len(rows) < PER_PAGE:
                complete = True
                break

        if not rows:
            complete = True
            break

    if pages_used >= MAX_LIST_PAGES and not complete and not error:
        error = "max_pages_reached"

    df = pd.DataFrame(list(seen.values()))
    if not df.empty:
        df = df.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)

    audit = {
        "listing_day_utc": day_s,
        "match_count": int(len(df)),
        "pages": int(pages_used),
        "complete": int(complete),
        "error": error,
        "listed_at_utc": now_utc_iso(),
    }

    return df, audit


def endpoint_is_retryable(status: Optional[int], error: str) -> bool:
    if status is None:
        return True
    if status == 429 or 500 <= status < 600:
        return True
    if error.startswith("network:") or error.startswith("json:"):
        return True
    return False


def raw_timeline_path(match_id: str) -> Path:
    return RAW_TIMELINE_DIR / f"{sanitize_filename(match_id)}.json"


def process_match(
    listing_row: Mapping[str, Any],
    client: APIClient,
    feat_mod: Any,
    audit_mod: Any,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    match_id = clean(listing_row.get("match_id"))

    audit = {
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

    # Timeline first: it determines quality + context and is the sealed target source.
    timeline_payload, timeline_status, timeline_error = client.request(
        f"/football/matches/{match_id}/timeline",
        use_cache=True,
    )
    audit["timeline_http_status"] = timeline_status
    audit["timeline_error"] = timeline_error

    if timeline_payload is None:
        if endpoint_is_retryable(timeline_status, timeline_error):
            audit.update(status="retryable_failed", reason="timeline_unavailable_retryable")
        else:
            audit.update(status="quality_excluded", reason="timeline_unavailable_permanent")
        return None, audit

    timeline_events = feat_mod.payload_list(timeline_payload)
    if not timeline_events:
        audit.update(status="quality_excluded", reason="empty_timeline")
        return None, audit

    timeline_full = bool(feat_mod.timeline_is_full(timeline_payload))
    period_end = bool(feat_mod.first_half_period_end_seen(timeline_events, audit_mod))
    audit["timeline_full"] = int(timeline_full)
    audit["first_half_period_end_seen"] = int(period_end)

    # Symmetric target admissibility: BOTH classes require full first-half proof.
    if not (timeline_full and period_end):
        audit.update(status="quality_excluded", reason="first_half_not_proven_complete")
        return None, audit

    # Use listing object as team/detail context first. Fetch detail only if necessary.
    listing_obj = dict(listing_row)
    team_context = feat_mod.get_detail_team_context(
        listing_obj,
        listing_obj,
        audit_mod,
    )

    need_detail = (
        not clean(team_context.get("home_team"))
        or not clean(team_context.get("away_team"))
        or not clean(team_context.get("home_team_id"))
        or not clean(team_context.get("away_team_id"))
    )

    detail_payload = listing_obj
    detail_status = 200
    detail_error = ""

    if need_detail:
        detail_payload, detail_status, detail_error = client.request(
            f"/football/matches/{match_id}",
            use_cache=True,
        )
        audit["detail_http_status"] = detail_status
        audit["detail_error"] = detail_error

        if detail_payload is not None:
            team_context = feat_mod.get_detail_team_context(
                detail_payload,
                listing_obj,
                audit_mod,
            )

    # Reconstruct score with timeline only before paying for shotmap.
    base_score_features, _ = feat_mod.reconstruct_base_live37(
        timeline_events=timeline_events,
        shot_events=[],
        team_context=team_context,
        timeline_payload=timeline_payload,
        audit=audit_mod,
    )

    sh = safe_float(base_score_features.get("score_37_home"))
    sa = safe_float(base_score_features.get("score_37_away"))

    if sh is None or sa is None:
        audit.update(status="quality_excluded", reason="score_37_unresolved")
        return None, audit

    audit["score_37_home"] = sh
    audit["score_37_away"] = sa

    if not (sh == sa and sh > 0):
        audit.update(status="context_excluded", reason="not_draw_with_goals_at_37")
        return None, audit

    # Candidate belongs to the pre-registered context. Fetch shotmap for model features.
    shot_payload, shot_status, shot_error = client.request(
        f"/football/matches/{match_id}/shotmap",
        use_cache=True,
    )
    audit["shotmap_http_status"] = shot_status
    audit["shotmap_error"] = shot_error

    shot_events = feat_mod.payload_list(shot_payload) if shot_payload is not None else []

    # Rebuild full <=37 predictor set. Missing source stays NaN and is handled by frozen imputer.
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

    # Recheck context from final reconstructed predictor set.
    sh2 = safe_float(all_features.get("score_37_home"))
    sa2 = safe_float(all_features.get("score_37_away"))
    if sh2 is None or sa2 is None or not (sh2 == sa2 and sh2 > 0):
        audit.update(status="quality_excluded", reason="context_recheck_failed")
        return None, audit

    # Store raw timeline payload for one-shot target opening later.
    sealed_path = raw_timeline_path(match_id)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(timeline_payload, sealed_path)
    timeline_sha = sha256_file(sealed_path)

    kickoff = clean(listing_row.get("kickoff_utc"))
    if not kickoff and isinstance(detail_payload, Mapping):
        ko = extract_kickoff(detail_payload)
        kickoff = ko.isoformat() if ko else ""

    row: Dict[str, Any] = {
        "schema_version": "first_half_37_v3_draw_with_goals_pristine_holdout_blind_v1",
        "match_id": match_id,
        "kickoff_utc": kickoff,
        "home_team": clean(team_context.get("home_team")) or clean(listing_row.get("home_team")),
        "away_team": clean(team_context.get("away_team")) or clean(listing_row.get("away_team")),
        "competition_id": clean(listing_row.get("competition_id")),
        "season_id": clean(listing_row.get("season_id")),
        "score_37_home": sh2,
        "score_37_away": sa2,
        "score_37_exact": f"{int(sh2)}-{int(sa2)}",
        "score_context_37": "draw_with_goals",
        "timeline_full": 1,
        "first_half_period_end_seen": 1,
        "target_status": "MASKED_BLIND",
        TARGET: np.nan,
        "sealed_timeline_file": str(sealed_path.relative_to(OUTPUT_DIR)),
        "sealed_timeline_sha256": timeline_sha,
        "collected_at_utc": now_utc_iso(),
    }

    for feature in FEATURES:
        value = all_features.get(feature, np.nan)
        row[feature] = np.nan if value is None else value

    available = sum(pd.notna(row[f]) for f in FEATURES)
    row["model_features_available"] = int(available)
    row["model_features_total"] = len(FEATURES)

    provenance = {}
    provenance.update(base_prov)
    provenance.update(recent_prov)
    provenance.update(xg_prov)
    row["feature_provenance_json"] = json.dumps(
        {f: provenance.get(f, "") for f in FEATURES},
        ensure_ascii=False,
        sort_keys=True,
    )

    audit.update({
        "status": "accepted_blind",
        "reason": "",
        "score_37_home": sh2,
        "score_37_away": sa2,
        "model_features_available": int(available),
        "model_features_total": len(FEATURES),
        "sealed_timeline_sha256": timeline_sha,
    })

    return row, audit


def manifest_payload(
    lock: Mapping[str, Any],
    listing_df: pd.DataFrame,
    day_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    blind_df: pd.DataFrame,
) -> Dict[str, Any]:
    expected_days = list(iter_dates(WINDOW_START.date(), WINDOW_END.date()))
    expected_day_strings = {d.isoformat() for d in expected_days}

    complete_days = set()
    if not day_df.empty:
        for _, r in day_df.iterrows():
            if int(r.get("complete", 0) or 0) == 1:
                complete_days.add(clean(r.get("listing_day_utc")))

    listing_complete = expected_day_strings.issubset(complete_days)

    listed_ids = set(listing_df["match_id"].astype(str)) if not listing_df.empty else set()

    terminal_ids = set()
    retryable_ids = set()

    if not audit_df.empty and "match_id" in audit_df.columns:
        for _, r in audit_df.iterrows():
            mid = clean(r.get("match_id"))
            status = clean(r.get("status"))
            if status in TERMINAL_STATUSES:
                terminal_ids.add(mid)
            elif status == "retryable_failed":
                retryable_ids.add(mid)

    unaudited_ids = listed_ids - terminal_ids
    window_closed = now_utc() > WINDOW_END

    collection_complete = (
        window_closed
        and listing_complete
        and len(unaudited_ids) == 0
    )

    accepted = 0 if blind_df.empty else len(blind_df)
    context_excluded = 0
    quality_excluded = 0
    if not audit_df.empty and "status" in audit_df.columns:
        context_excluded = int(audit_df["status"].astype(str).eq("context_excluded").sum())
        quality_excluded = int(audit_df["status"].astype(str).eq("quality_excluded").sum())

    feature_coverage = {}
    if not blind_df.empty:
        for f in FEATURES:
            feature_coverage[f] = {
                "available": int(pd.to_numeric(blind_df[f], errors="coerce").notna().sum()),
                "total": int(len(blind_df)),
                "coverage": float(pd.to_numeric(blind_df[f], errors="coerce").notna().mean()),
            }

    payload = {
        "schema_version": "first_half_37_v3_draw_with_goals_collection_manifest_v1",
        "created_at_utc": now_utc_iso(),
        "window_start_utc": WINDOW_START.isoformat(),
        "window_end_utc": WINDOW_END.isoformat(),
        "window_closed": bool(window_closed),
        "selection_rule": (
            "timeline first half full AND first_half_period_end_seen "
            "AND score_37_home == score_37_away > 0"
        ),
        "blind_protocol": {
            "model_scoring_performed": False,
            "target_written_to_dataset": False,
            "target_console_output": False,
            "target_source": "sealed raw timeline opened only by one-shot evaluator",
        },
        "frozen_model_name": lock.get("model_name"),
        "model_sha256": lock.get("model_sha256"),
        "freeze_lock_sha256": sha256_file(FREEZE_LOCK_PATH),
        "training_dataset_sha256": lock.get("training_dataset_sha256"),
        "days_expected": len(expected_days),
        "days_complete": len(expected_day_strings & complete_days),
        "listing_complete": bool(listing_complete),
        "matches_listed": int(len(listed_ids)),
        "matches_terminally_audited": int(len(listed_ids & terminal_ids)),
        "matches_pending_or_retryable": int(len(unaudited_ids)),
        "accepted_draw_with_goals": int(accepted),
        "context_excluded": int(context_excluded),
        "quality_excluded": int(quality_excluded),
        "collection_complete": bool(collection_complete),
        "feature_coverage": feature_coverage,
        "blind_dataset_path": str(BLIND_DATASET_PATH),
        "blind_dataset_sha256": sha256_file(BLIND_DATASET_PATH) if BLIND_DATASET_PATH.exists() else None,
        "listing_sha256": sha256_file(LISTING_PATH) if LISTING_PATH.exists() else None,
        "day_audit_sha256": sha256_file(DAY_AUDIT_PATH) if DAY_AUDIT_PATH.exists() else None,
        "match_audit_sha256": sha256_file(MATCH_AUDIT_PATH) if MATCH_AUDIT_PATH.exists() else None,
    }

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collecteur blind du pristine holdout V3 draw_with_goals."
    )
    parser.add_argument("--min-api-interval", type=float, default=DEFAULT_MIN_API_INTERVAL)
    parser.add_argument("--force", action="store_true", help="Reprocess terminally audited matches.")
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Debug seulement: limite le nombre de matchs audités cette exécution. 0=tous.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    lock = verify_freeze()

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()

    api_key = clean(os.getenv("STATS_API_KEY"))
    base_url = clean(
        os.getenv("STATS_API_BASE_URL", "https://api.thestatsapi.com/api")
    )
    timeout = safe_int(os.getenv("STATS_API_TIMEOUT")) or 20

    if not api_key:
        print("ERREUR: STATS_API_KEY absent du .env")
        return 2

    feat_mod = load_module("fh37_feature_collector", FEATURE_MODULE_PATH)
    audit_mod = load_module("fh37_historical_audit", AUDIT_MODULE_PATH)

    client = APIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        min_interval=max(0.1, args.min_api_interval),
        max_retries=6,
    )

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS PRISTINE HOLDOUT COLLECTOR (BLIND)")
    print("=" * 120)
    print(f"Fenêtre UTC : {WINDOW_START} -> {WINDOW_END}")
    print(f"Output      : {OUTPUT_DIR}")
    print("Scoring     : DESACTIVE")
    print("Target      : MASQUE")
    print()
    print("FREEZE_LOCK : PASS")
    print(f"MODEL SHA256: {lock.get('model_sha256')}")
    print(f"Pipeline    : {lock.get('model_name')}")
    print("Features    : " + ", ".join(FEATURES))
    print()

    listing_df = read_csv_or_empty(LISTING_PATH)
    day_df = read_csv_or_empty(DAY_AUDIT_PATH)
    audit_df = read_csv_or_empty(MATCH_AUDIT_PATH)
    blind_df = read_csv_or_empty(BLIND_DATASET_PATH)

    # Decide closed days to list. Never finalize today's partial UTC schedule.
    current = now_utc()
    if current > WINDOW_END:
        through = WINDOW_END.date()
    else:
        through = min((current.date() - timedelta(days=1)), WINDOW_END.date())

    if through >= WINDOW_START.date():
        day_frames = []
        day_audits = []

        for d in iter_dates(WINDOW_START.date(), through):
            df_day, day_audit = list_finished_day(client, audit_mod, d)
            day_frames.append(df_day)
            day_audits.append(day_audit)

            print(
                f"Listing {d.isoformat()} | matchs uniques={len(df_day)} | "
                f"pages={day_audit['pages']} | "
                f"complete={'YES' if day_audit['complete'] else 'NO'}"
            )

        if day_frames:
            fresh_listing = pd.concat(day_frames, ignore_index=True, sort=False)
            if not fresh_listing.empty:
                # Refresh listed days while preserving any future already-stored days.
                listed_days = set(fresh_listing["listing_day_utc"].astype(str))
                if not listing_df.empty and "listing_day_utc" in listing_df.columns:
                    listing_df = listing_df[
                        ~listing_df["listing_day_utc"].astype(str).isin(listed_days)
                    ].copy()
                listing_df = pd.concat([listing_df, fresh_listing], ignore_index=True, sort=False)
                listing_df = (
                    listing_df.drop_duplicates(subset=["match_id"], keep="last")
                    .sort_values(["kickoff_utc", "match_id"])
                    .reset_index(drop=True)
                )
                atomic_csv(listing_df, LISTING_PATH)

        for rec in day_audits:
            if day_df.empty or "listing_day_utc" not in day_df.columns:
                day_df = pd.DataFrame([rec])
            else:
                day_df = day_df[
                    ~day_df["listing_day_utc"].astype(str).eq(rec["listing_day_utc"])
                ].copy()
                day_df = pd.concat([day_df, pd.DataFrame([rec])], ignore_index=True, sort=False)

        if not day_df.empty:
            day_df = day_df.sort_values("listing_day_utc").reset_index(drop=True)
            atomic_csv(day_df, DAY_AUDIT_PATH)

    print()
    print(f"TOTAL MATCHS LISTES ACTUELLEMENT : {len(listing_df)}")

    training_ids = set(
        pd.read_csv(TRAINING_DATASET_PATH, usecols=["match_id"])["match_id"].astype(str)
    )

    terminal = {}
    if not audit_df.empty and "match_id" in audit_df.columns:
        for _, r in audit_df.iterrows():
            if clean(r.get("status")) in TERMINAL_STATUSES:
                terminal[clean(r.get("match_id"))] = clean(r.get("status"))

    processed = 0

    if not listing_df.empty:
        for idx, r in listing_df.iterrows():
            mid = clean(r.get("match_id"))

            if not mid:
                continue

            if mid in training_ids:
                raise RuntimeError(f"LEAKAGE: match_id holdout déjà dans training: {mid}")

            if not args.force and mid in terminal:
                continue

            if args.max_matches > 0 and processed >= args.max_matches:
                break

            processed += 1

            print("-" * 120)
            print(
                f"[{idx+1}/{len(listing_df)}] {mid} | "
                f"{clean(r.get('home_team'))} vs {clean(r.get('away_team'))} | "
                f"{clean(r.get('kickoff_utc'))}"
            )

            row, audit_row = process_match(
                listing_row=r.to_dict(),
                client=client,
                feat_mod=feat_mod,
                audit_mod=audit_mod,
            )

            audit_df = upsert(audit_df, audit_row)
            atomic_csv(audit_df, MATCH_AUDIT_PATH)

            if row is not None:
                blind_df = upsert(blind_df, row)
                # Target must remain fully masked.
                blind_df[TARGET] = np.nan
                blind_df["target_status"] = "MASKED_BLIND"
                blind_df = blind_df.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
                atomic_csv(blind_df, BLIND_DATASET_PATH)

                print(
                    f"ACCEPTE BLIND | score37={row['score_37_exact']} | "
                    f"features={row['model_features_available']}/{row['model_features_total']} | "
                    "TARGET MASQUE"
                )
            else:
                print(
                    f"{audit_row.get('status','').upper()} | "
                    f"{audit_row.get('reason','')}"
                )

    # Remove any accidental target leakage before manifest.
    if not blind_df.empty:
        blind_df[TARGET] = np.nan
        blind_df["target_status"] = "MASKED_BLIND"
        atomic_csv(blind_df, BLIND_DATASET_PATH)

    manifest = manifest_payload(lock, listing_df, day_df, audit_df, blind_df)
    atomic_json(manifest, COLLECTION_MANIFEST_PATH)

    print()
    print("=" * 120)
    print("V3 DRAW_WITH_GOALS — COLLECTION BLIND STATUS")
    print("=" * 120)
    print(f"Window closed             : {'YES' if manifest['window_closed'] else 'NO'}")
    print(f"Days complete             : {manifest['days_complete']}/{manifest['days_expected']}")
    print(f"Listing complete          : {'YES' if manifest['listing_complete'] else 'NO'}")
    print(f"Matches listed            : {manifest['matches_listed']}")
    print(f"Terminally audited        : {manifest['matches_terminally_audited']}")
    print(f"Pending/retryable         : {manifest['matches_pending_or_retryable']}")
    print(f"Accepted draw_with_goals  : {manifest['accepted_draw_with_goals']}")
    print(f"Collection complete       : {'YES' if manifest['collection_complete'] else 'NO'}")
    print(f"API requests this run     : {client.request_count}")
    print(f"HTTP 429 retries          : {client.retry_429_count}")
    print(f"Manifest                  : {COLLECTION_MANIFEST_PATH}")
    print()
    print("AUCUN SCORING MODELE N'A ETE EFFECTUE.")

    return 0 if manifest["collection_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
