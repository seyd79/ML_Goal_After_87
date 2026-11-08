#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — LIVE SHADOW SCANNER V6 — FROZEN CONTEXT 37
======================================

Purpose
-------
Score the frozen FIRST_HALF37_V3_DRAW_WITH_GOALS model in real time, in SHADOW
mode only.

V6 fixes one critical retry invariant: once a match is first observed at exactly
37' with an eligible draw-with-goals score, that 37' score/context is frozen and
can never be replaced by the live score seen at 38'+ (or by a later feed refresh).

The scanner:
- watches live matches around the 37th minute;
- reconstructs the exact snapshot at <=37' with /live-stats as the operational backbone and event feeds as fidelity-first sources;
- treats HTTP 409 from live auxiliary endpoints as temporary source unavailability;
- snapshots cumulative /live-stats at 29' (xG, shots, corners) so 30-37 deltas can be derived exactly;
- can recover a missed 37' snapshot at 38'-44' in SHADOW only, still hard-cutting features at <=37';
- keeps ONLY the draw_with_goals context: score_37_home == score_37_away > 0;
- requires the frozen 5/5 model features with no silent imputation;
- scores EVERY valid 5/5 match (no probability selection filter);
- saves all shadow predictions for later evaluation;
- uses /live-timeline and /live-stats while a match is in progress;
- optionally resolves targets from live/settled timelines after the snapshot;
- never places a bet and never changes the frozen model.

Required project files (same directory as this script)
------------------------------------------------------
.env
collect_first_half_37_dataset.py
audit_historical_live87_reconstructability.py
outputs/first_half_37/v3_draw_with_goals_frozen/
    FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib
    FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json

Run
---
    caffeinate -dimsu python3 -u first_half_37_shadow_scanner_V6_FROZEN_CONTEXT37.py

One scan only (diagnostic)
--------------------------
    python3 -u first_half_37_shadow_scanner_V6_FROZEN_CONTEXT37.py --once

Outputs
-------
outputs/first_half_37/shadow_live/
    FIRST_HALF37_SHADOW_PREDICTIONS.csv
    FIRST_HALF37_SHADOW_AUDIT.csv

IMPORTANT
---------
Shadow results are observed live outcomes. Once you inspect their targets or
performance, those matches are consumed validation data and must not later be
called a pristine holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
except Exception:
    average_precision_score = None
    brier_score_loss = None
    log_loss = None
    roc_auc_score = None


# =============================================================================
# PATHS / ENV
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

FEATURE_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

FROZEN_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_frozen"
)

MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json"

OUTPUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "shadow_live"
PREDICTIONS_PATH = OUTPUT_DIR / "FIRST_HALF37_SHADOW_PREDICTIONS.csv"
AUDIT_PATH = OUTPUT_DIR / "FIRST_HALF37_SHADOW_AUDIT.csv"
CORNER_BASELINES_PATH = OUTPUT_DIR / "FIRST_HALF37_CORNER_BASELINES.json"
LIVE_STATS_SNAPSHOTS_PATH = OUTPUT_DIR / "FIRST_HALF37_LIVE_STATS_SNAPSHOTS.json"
# V6: persistent as-of-37 context. Once first observed at exact 37', the score
# is immutable for every retry/recovery and survives scanner restarts.
SCORE37_CONTEXT_PATH = OUTPUT_DIR / "FIRST_HALF37_SCORE37_CONTEXT.json"
LIVE_STATS_RAW_DIR = OUTPUT_DIR / "live_stats_raw"


# =============================================================================
# FROZEN MODEL CONTRACT
# =============================================================================

MODEL_FEATURES: List[str] = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
]

EXPECTED_MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
SNAPSHOT_MINUTE = 37


# =============================================================================
# CONFIGURATION
# =============================================================================

STATS_API_KEY = os.getenv("STATS_API_KEY", "").strip()
STATS_API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api",
).rstrip("/")
STATS_API_TIMEOUT = int(os.getenv("STATS_API_TIMEOUT", "20"))

SCAN_INTERVAL_SECONDS = int(os.getenv("FH37_SCAN_INTERVAL_SECONDS", "10"))
SCANNER_WORKERS = int(os.getenv("FH37_SCANNER_WORKERS", "4"))

# Primary prospective decision time is exactly 37'. A match first seen at 37'
# but blocked by a temporary source conflict (notably HTTP 409 on /timeline) may
# be reconstructed later for SHADOW EVALUATION only. Every feature is still
# hard-cut at <=37'. Late recovery is never labelled operationally usable.
MIN_OBSERVED_MINUTE = 37
MAX_OBSERVED_MINUTE = 37
RECOVERY_MAX_MINUTE = int(os.getenv("FH37_RECOVERY_MAX_MINUTE", "44"))
CORNER_BASELINE_MINUTE = int(os.getenv("FH37_CORNER_BASELINE_MINUTE", "29"))
LATE_RECOVERY_ENABLED = os.getenv(
    "FH37_LATE_RECOVERY_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

API_MAX_REQUESTS_PER_SECOND = float(
    os.getenv("FH37_API_MAX_REQUESTS_PER_SECOND", "3.0")
)
API_429_MAX_RETRIES = int(os.getenv("FH37_API_429_MAX_RETRIES", "4"))
API_429_DEFAULT_BACKOFF_SECONDS = float(
    os.getenv("FH37_API_429_DEFAULT_BACKOFF_SECONDS", "5.0")
)

LIVE_MATCHES_PER_PAGE = int(os.getenv("FH37_LIVE_MATCHES_PER_PAGE", "100"))
LIVE_MATCHES_MAX_PAGES = int(os.getenv("FH37_LIVE_MATCHES_MAX_PAGES", "20"))

# If one of the five features is temporarily lagging, retry quickly. The match
# is NOT permanently consumed until a valid 5/5 prediction is saved or the
# exact 37' context is proven ineligible.
MAX_IMMEDIATE_FEATURE_RETRIES = int(
    os.getenv("FH37_MAX_IMMEDIATE_FEATURE_RETRIES", "1")
)
RETRY_DELAY_SECONDS = float(os.getenv("FH37_RETRY_DELAY_SECONDS", "1.0"))
# Backward-compatible diagnostic only. V5 does not permanently reject a match
# merely because a live auxiliary source returned 409 several times during 37'.
MAX_TOTAL_ATTEMPTS_PER_MATCH = int(
    os.getenv("FH37_MAX_TOTAL_ATTEMPTS_PER_MATCH", "999")
)

# Pending targets are checked after the snapshot. Positive targets may resolve
# early; negative targets follow the original historical resolve_target rules.
TARGET_RESOLUTION_ENABLED = os.getenv(
    "FH37_TARGET_RESOLUTION_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
TARGET_FIRST_CHECK_SECONDS = int(
    os.getenv("FH37_TARGET_FIRST_CHECK_SECONDS", "120")
)
TARGET_RECHECK_SECONDS = int(
    os.getenv("FH37_TARGET_RECHECK_SECONDS", "90")
)

MACOS_NOTIFICATIONS_ENABLED = os.getenv(
    "FH37_MACOS_NOTIFICATIONS_ENABLED",
    os.getenv("MACOS_NOTIFICATIONS_ENABLED", "1"),
).strip().lower() not in {"0", "false", "no", "off"}

TELEGRAM_NOTIFICATIONS_ENABLED = os.getenv(
    "FH37_TELEGRAM_NOTIFICATIONS_ENABLED",
    os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "0"),
).strip().lower() not in {"0", "false", "no", "off"}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_API_TIMEOUT = int(os.getenv("TELEGRAM_API_TIMEOUT", "10"))


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("first_half_37_shadow_scanner")


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    return False


def clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_match_minute(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return safe_int(value)
    match = re.match(r"^\s*(\d+)", str(value).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_elapsed_minute(raw: Mapping[str, Any]) -> Optional[int]:
    candidates: List[Any] = [
        raw.get("elapsed_minutes"),
        raw.get("elapsed_minute"),
        raw.get("minute"),
        raw.get("minutes"),
        raw.get("elapsed"),
        raw.get("match_minute"),
        raw.get("current_minute"),
    ]

    for key in ("meta", "status", "time", "clock", "live", "match_status"):
        container = raw.get(key)
        if isinstance(container, Mapping):
            candidates.extend([
                container.get("elapsed_minutes"),
                container.get("elapsed_minute"),
                container.get("minute"),
                container.get("minutes"),
                container.get("elapsed"),
                container.get("current_minute"),
            ])

    data = raw.get("data")
    if isinstance(data, Mapping):
        candidates.extend([
            data.get("elapsed_minutes"),
            data.get("elapsed_minute"),
            data.get("minute"),
            data.get("minutes"),
            data.get("elapsed"),
        ])

    for candidate in candidates:
        minute = parse_match_minute(candidate)
        if minute is not None:
            return minute
    return None


def extract_score(raw: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Robust live-score parser.

    TheStatsAPI can expose the current score in several shapes, including:
      - home_score / away_score
      - home_team.score / away_team.score
      - score.home / score.away
      - scores.home / scores.away
      - home.score / away.score
      - nested data.<same shapes>

    Missing score is never converted to zero.
    """

    def score_value(value: Any) -> Optional[int]:
        if isinstance(value, Mapping):
            value = first_not_none(
                value.get("score"),
                value.get("goals"),
                value.get("value"),
                value.get("current"),
                value.get("total"),
            )
        return safe_int(value)

    def from_container(container: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        home = score_value(first_not_none(
            container.get("home_goals"),
            container.get("home_score"),
            container.get("score_home"),
            container.get("home_score_current"),
        ))
        away = score_value(first_not_none(
            container.get("away_goals"),
            container.get("away_score"),
            container.get("score_away"),
            container.get("away_score_current"),
        ))

        # Common current API shape: score: {home: N, away: N, ...}
        for score_key in ("score", "scores", "current_score", "live_score"):
            score_obj = container.get(score_key)
            if not isinstance(score_obj, Mapping):
                continue
            if home is None:
                home = score_value(first_not_none(
                    score_obj.get("home"),
                    score_obj.get("home_score"),
                    score_obj.get("score_home"),
                ))
            if away is None:
                away = score_value(first_not_none(
                    score_obj.get("away"),
                    score_obj.get("away_score"),
                    score_obj.get("score_away"),
                ))

        # Team objects used by /football/matches.
        home_team = container.get("home_team")
        away_team = container.get("away_team")
        if home is None and isinstance(home_team, Mapping):
            home = score_value(first_not_none(
                home_team.get("score"),
                home_team.get("goals"),
                home_team.get("current_score"),
            ))
        if away is None and isinstance(away_team, Mapping):
            away = score_value(first_not_none(
                away_team.get("score"),
                away_team.get("goals"),
                away_team.get("current_score"),
            ))

        # Alternate API shape: home: {name, score}, away: {name, score}.
        home_obj = container.get("home")
        away_obj = container.get("away")
        if home is None and isinstance(home_obj, Mapping):
            home = score_value(first_not_none(
                home_obj.get("score"), home_obj.get("goals"), home_obj.get("current_score")
            ))
        if away is None and isinstance(away_obj, Mapping):
            away = score_value(first_not_none(
                away_obj.get("score"), away_obj.get("goals"), away_obj.get("current_score")
            ))

        return home, away

    home, away = from_container(raw)

    data = raw.get("data")
    if (home is None or away is None) and isinstance(data, Mapping):
        nested_home, nested_away = from_container(data)
        if home is None:
            home = nested_home
        if away is None:
            away = nested_away

    return home, away


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-10) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def fair_odds(probability: float) -> float:
    return float("inf") if probability <= 0 else 1.0 / probability


def probability_band(p: float) -> str:
    if p >= 0.70:
        return "70_PLUS"
    if p >= 0.60:
        return "60_70"
    if p >= 0.50:
        return "50_60"
    if p >= 0.40:
        return "40_50"
    if p >= 0.30:
        return "30_40"
    return "BELOW_30"


# =============================================================================
# LIVE MATCH
# =============================================================================

@dataclass
class LiveMatch:
    match_id: str
    home_team: str
    away_team: str
    minute: int
    home_score: Optional[int]
    away_score: Optional[int]
    competition_id: Optional[str] = None
    season_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def parse_live_match(raw: Any) -> Optional[LiveMatch]:
    if not isinstance(raw, Mapping):
        return None

    match_id = first_not_none(raw.get("match_id"), raw.get("id"), raw.get("event_id"))
    if match_id is None:
        data = raw.get("data")
        if isinstance(data, Mapping):
            match_id = first_not_none(data.get("match_id"), data.get("id"))
    if match_id is None:
        return None

    minute = extract_elapsed_minute(raw)
    if minute is None:
        return None

    def team_name(side: str) -> str:
        team = raw.get(f"{side}_team")
        if isinstance(team, Mapping):
            name = first_not_none(team.get("name"), team.get("team_name"))
            if name:
                return str(name)
        return str(first_not_none(raw.get(f"{side}_name"), side.title()))

    home_score, away_score = extract_score(raw)

    competition_id = first_not_none(raw.get("competition_id"), raw.get("league_id"))
    competition = raw.get("competition")
    if competition_id is None and isinstance(competition, Mapping):
        competition_id = first_not_none(competition.get("id"), competition.get("competition_id"))

    season_id = raw.get("season_id")
    season = raw.get("season")
    if season_id is None and isinstance(season, Mapping):
        season_id = first_not_none(season.get("id"), season.get("season_id"))

    return LiveMatch(
        match_id=str(match_id),
        home_team=team_name("home"),
        away_team=team_name("away"),
        minute=minute,
        home_score=home_score,
        away_score=away_score,
        competition_id=str(competition_id) if competition_id is not None else None,
        season_id=str(season_id) if season_id is not None else None,
        raw=dict(raw),
    )


# =============================================================================
# GLOBAL API RATE LIMITER
# =============================================================================

class GlobalRateLimiter:
    def __init__(self, max_requests_per_second: float):
        self.min_interval = (
            1.0 / float(max_requests_per_second)
            if max_requests_per_second > 0
            else 0.0
        )
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._next_allowed = time.monotonic() + self.min_interval


class StatsAPIClient:
    def __init__(self, api_key: str, base_url: str, timeout: int):
        key = str(api_key or "").strip()
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._thread_local = threading.local()
        self._rate_limiter = GlobalRateLimiter(API_MAX_REQUESTS_PER_SECOND)

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            })
            self._thread_local.session = session
        return session

    def _retry_after(self, response: requests.Response, attempt: int) -> float:
        raw = str(response.headers.get("Retry-After", "") or "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(raw)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except Exception:
                    pass
        return min(
            60.0,
            API_429_DEFAULT_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)),
        )

    def _request(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        soft_statuses: Optional[Set[int]] = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        max_attempts = max(1, API_429_MAX_RETRIES + 1)
        last_response: Optional[requests.Response] = None
        soft_statuses = set(soft_statuses or set())

        for attempt in range(1, max_attempts + 1):
            self._rate_limiter.wait()
            response = self._session().get(
                url,
                params=dict(params or {}),
                timeout=self.timeout,
            )
            last_response = response

            if response.status_code == 429:
                if attempt >= max_attempts:
                    response.raise_for_status()
                wait_seconds = self._retry_after(response, attempt)
                logger.warning(
                    "RATE LIMIT 429 | attempt=%d/%d | path=%s | wait=%.1fs",
                    attempt,
                    max_attempts,
                    path,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code in soft_statuses:
                body = (response.text or "").replace("\n", " ").strip()[:300]
                logger.warning(
                    "TEMP SOURCE HTTP %d | path=%s | body=%s",
                    response.status_code,
                    path,
                    body or "<empty>",
                )
                return None

            response.raise_for_status()
            if not response.content:
                return None
            return response.json()

        if last_response is not None:
            last_response.raise_for_status()
        raise requests.HTTPError(f"Request failed: {path}")

    def test_auth(self) -> bool:
        try:
            self._request("/football/matches", {"status": "live", "page": 1, "per_page": 1})
            logger.info("TheStatsAPI authentication : OK")
            return True
        except Exception as exc:
            logger.error("TheStatsAPI authentication FAILED : %s", exc)
            return False

    def get_live_matches(self) -> List[Dict[str, Any]]:
        all_matches: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for page in range(1, LIVE_MATCHES_MAX_PAGES + 1):
            payload = self._request(
                "/football/matches",
                {"status": "live", "page": page, "per_page": LIVE_MATCHES_PER_PAGE},
            )
            data = payload.get("data", []) if isinstance(payload, Mapping) else []
            meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
            if not isinstance(data, list):
                data = []

            for item in data:
                if not isinstance(item, Mapping):
                    continue
                mid = first_not_none(item.get("match_id"), item.get("id"))
                if mid is None:
                    continue
                mid = str(mid)
                if mid in seen:
                    continue
                seen.add(mid)
                all_matches.append(dict(item))

            current_page = safe_int(meta.get("page")) if isinstance(meta, Mapping) else page
            total_pages = safe_int(meta.get("total_pages")) if isinstance(meta, Mapping) else 1
            current_page = current_page or page
            total_pages = total_pages or 1

            logger.info(
                "LIVE page=%d/%d | received=%d | unique_total=%d",
                current_page,
                total_pages,
                len(data),
                len(all_matches),
            )

            if current_page >= total_pages or not data:
                break

        return all_matches

    def get_match(self, match_id: str, soft: bool = False) -> Any:
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(f"/football/matches/{match_id}", soft_statuses=statuses)

    def get_live_timeline(self, match_id: str, soft: bool = False) -> Any:
        """Timeline dédiée aux matchs en cours.

        TheStatsAPI renvoie HTTP 409 sur /timeline lorsqu'un match est LIVE et
        demande explicitement d'utiliser /live-timeline.
        """
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(
            f"/football/matches/{match_id}/live-timeline",
            soft_statuses=statuses,
        )

    def get_timeline(self, match_id: str, soft: bool = False) -> Any:
        """Timeline settled/historique, à utiliser après le live."""
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(
            f"/football/matches/{match_id}/timeline",
            soft_statuses=statuses,
        )

    def get_shotmap(self, match_id: str, soft: bool = False) -> Any:
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(f"/football/matches/{match_id}/shotmap", soft_statuses=statuses)

    def get_live_stats(self, match_id: str, soft: bool = False) -> Any:
        """Statistiques cumulées dédiées aux matchs en cours."""
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(
            f"/football/matches/{match_id}/live-stats",
            soft_statuses=statuses,
        )

    def get_stats(self, match_id: str, soft: bool = False) -> Any:
        """Statistiques settled/historiques, à utiliser après le live."""
        statuses = {404, 409, 425, 503} if soft else None
        return self._request(f"/football/matches/{match_id}/stats", soft_statuses=statuses)


def _stats_side_value(side: Any, keys: Sequence[str]) -> Optional[float]:
    if not isinstance(side, Mapping):
        return None
    for key in keys:
        value = side.get(key)
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def extract_home_corners_from_stats(payload: Any) -> Optional[float]:
    """Extract cumulative HOME corners from /stats without inventing zeros."""
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        return None

    home = data.get("home")
    value = _stats_side_value(home, ("corners", "corner_kicks"))
    if value is not None:
        return value

    stats = data.get("stats", data)
    if isinstance(stats, Mapping):
        for key in ("corners", "corner_kicks"):
            node = stats.get(key)
            if not isinstance(node, Mapping):
                continue
            all_node = node.get("all")
            if isinstance(all_node, Mapping):
                value = safe_float(first_not_none(
                    all_node.get("home"), all_node.get("Home"), all_node.get("1")
                ))
                if value is not None:
                    return value
            value = safe_float(first_not_none(
                node.get("home"), node.get("Home"), node.get("1")
            ))
            if value is not None:
                return value
    return None


LIVE_STAT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "xg": (
        "expected_goals",
        "xg",
        "expected_goal",
    ),
    "shots": (
        "total_shots",
        "shots",
        "total_attempts",
    ),
    "corners": (
        "corners",
        "corner_kicks",
        "corner",
    ),
}


def _normalize_stat_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _extract_home_away_from_node(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(value, Mapping):
        return None, None

    # Typical TheStatsAPI shapes:
    # {"home": 10, "away": 6}
    # {"all": {"home": 10, "away": 6}}
    # {"total": {"home": 10, "away": 6}}
    candidates: List[Mapping[str, Any]] = [value]
    for key in ("all", "total", "overall", "match"):
        child = value.get(key)
        if isinstance(child, Mapping):
            candidates.append(child)

    for node in candidates:
        home = safe_float(first_not_none(
            node.get("home"), node.get("Home"), node.get("1")
        ))
        away = safe_float(first_not_none(
            node.get("away"), node.get("Away"), node.get("2")
        ))
        if home is not None or away is not None:
            return home, away

    return None, None


def _side_stat_value(side: Any, aliases: Sequence[str]) -> Optional[float]:
    if not isinstance(side, Mapping):
        return None
    wanted = {_normalize_stat_key(x) for x in aliases}

    def direct(container: Mapping[str, Any]) -> Optional[float]:
        for key, value in container.items():
            if _normalize_stat_key(key) not in wanted:
                continue
            parsed = safe_float(value)
            if parsed is not None:
                return parsed
            if isinstance(value, Mapping):
                for inner_key in ("all", "total", "value", "match"):
                    parsed = safe_float(value.get(inner_key))
                    if parsed is not None:
                        return parsed
        return None

    # Prefer shallow/current-match branches before recursive fallback.
    parsed = direct(side)
    if parsed is not None:
        return parsed
    for branch in ("stats", "statistics", "all", "total", "match"):
        child = side.get(branch)
        if isinstance(child, Mapping):
            parsed = direct(child)
            if parsed is not None:
                return parsed

    stack: List[Any] = [side]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            parsed = direct(node)
            if parsed is not None:
                return parsed
            stack.extend(v for v in node.values() if isinstance(v, (Mapping, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (Mapping, list)))
    return None


def extract_live_stat_pair(
    payload: Any,
    aliases: Sequence[str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Robustly extract a cumulative HOME/AWAY pair from /live-stats.

    Supported shapes include the formats already used by the Goal After 87
    scanner (`stats.<metric>.all.home/away`) and `data.home` / `data.away`.
    No missing value is converted to zero.
    """
    if not isinstance(payload, Mapping):
        return None, None

    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        return None, None

    # 1) data.stats.<metric> or data.<metric>, exact alias lookup.
    for container in (data.get("stats"), data):
        if not isinstance(container, Mapping):
            continue
        normalized = {_normalize_stat_key(k): v for k, v in container.items()}
        for alias in aliases:
            value = normalized.get(_normalize_stat_key(alias))
            if value is None:
                continue
            home, away = _extract_home_away_from_node(value)
            if home is not None or away is not None:
                return home, away

    # 2) data.home / data.away scalar stats.
    home_side = data.get("home")
    away_side = data.get("away")
    home = _side_stat_value(home_side, aliases)
    away = _side_stat_value(away_side, aliases)
    if home is not None or away is not None:
        return home, away

    # 3) Recursive fallback: find an alias key containing a pair.
    wanted = {_normalize_stat_key(x) for x in aliases}
    stack: List[Any] = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            for key, child in node.items():
                if _normalize_stat_key(key) in wanted:
                    h, a = _extract_home_away_from_node(child)
                    if h is not None or a is not None:
                        return h, a
                if isinstance(child, (Mapping, list)):
                    stack.append(child)
        elif isinstance(node, list):
            stack.extend(x for x in node if isinstance(x, (Mapping, list)))

    return None, None


def extract_cumulative_live_stats(payload: Any) -> Dict[str, Any]:
    """Return only cumulative quantities needed for the frozen FH37 features."""
    result: Dict[str, Any] = {
        "home_xg": None,
        "away_xg": None,
        "total_xg": None,
        "home_shots": None,
        "away_shots": None,
        "home_corners": None,
        "away_corners": None,
    }

    hxg, axg = extract_live_stat_pair(payload, LIVE_STAT_ALIASES["xg"])
    hshots, ashots = extract_live_stat_pair(payload, LIVE_STAT_ALIASES["shots"])
    hcorners, acorners = extract_live_stat_pair(payload, LIVE_STAT_ALIASES["corners"])

    result.update({
        "home_xg": hxg,
        "away_xg": axg,
        "home_shots": hshots,
        "away_shots": ashots,
        "home_corners": hcorners,
        "away_corners": acorners,
    })

    # total_xg_37 from live-stats is accepted only when BOTH team values exist.
    # This avoids silently treating a missing side as zero.
    if hxg is not None and axg is not None:
        result["total_xg"] = float(hxg + axg)

    return result


def live_stats_availability(stats: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "xg": safe_float(stats.get("home_xg")) is not None and safe_float(stats.get("away_xg")) is not None,
        "shots": safe_float(stats.get("home_shots")) is not None and safe_float(stats.get("away_shots")) is not None,
        "corners": safe_float(stats.get("home_corners")) is not None and safe_float(stats.get("away_corners")) is not None,
    }


def _fmt_stat(value: Any, decimals: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "?"
    if decimals == 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


# =============================================================================
# FROZEN MODEL
# =============================================================================

class FrozenFirstHalf37Model:
    def __init__(self):
        self.bundle: Any = None
        self.base_model: Any = None
        self.lock: Dict[str, Any] = {}
        self.alpha: float = 0.0
        self.model_sha256: str = ""
        self.model_name: str = ""

    def load(self) -> None:
        for path in (MODEL_PATH, FREEZE_LOCK_PATH):
            if not path.exists():
                raise FileNotFoundError(path)

        self.lock = json.loads(FREEZE_LOCK_PATH.read_text(encoding="utf-8"))
        self.model_sha256 = sha256_file(MODEL_PATH)
        self.model_name = str(self.lock.get("model_name", ""))

        if self.model_name != EXPECTED_MODEL_NAME:
            raise RuntimeError(
                f"Frozen model_name inattendu: {self.model_name!r}"
            )

        lock_features = list(self.lock.get("features", []))
        if lock_features != MODEL_FEATURES:
            raise RuntimeError(
                f"Frozen features inattendues: {lock_features}"
            )

        expected_sha = str(self.lock.get("model_sha256", ""))
        if expected_sha and expected_sha != self.model_sha256:
            raise RuntimeError("MODEL SHA mismatch avec le freeze lock.")

        self.alpha = float(self.lock.get("calibration_alpha"))
        self.bundle = joblib.load(MODEL_PATH)

        if not isinstance(self.bundle, Mapping) or "base_model" not in self.bundle:
            raise RuntimeError(
                "Bundle gelé invalide: clé 'base_model' absente."
            )

        self.base_model = self.bundle["base_model"]

        logger.info("Frozen model loaded : %s", self.model_name)
        logger.info("Model SHA : %s", self.model_sha256)
        logger.info("Calibration alpha : %.12f", self.alpha)
        logger.info("Features : %s", ", ".join(MODEL_FEATURES))
        logger.info("Freeze status : %s", self.lock.get("status", "?"))

    def predict(self, values: Mapping[str, Any]) -> Tuple[float, float]:
        missing = [f for f in MODEL_FEATURES if is_missing(values.get(f))]
        if missing:
            raise ValueError(f"5/5 requis; missing={missing}")

        X = pd.DataFrame(
            [[values[f] for f in MODEL_FEATURES]],
            columns=MODEL_FEATURES,
        ).apply(pd.to_numeric, errors="coerce")

        if X.isna().any(axis=None):
            raise ValueError("5/5 requis; conversion numérique a produit NaN.")

        arr = X.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Feature non finie.")

        probabilities = self.base_model.predict_proba(X)
        classes = getattr(self.base_model, "classes_", None)
        if classes is None:
            named_steps = getattr(self.base_model, "named_steps", {})
            classifier = named_steps.get("classifier") if isinstance(named_steps, Mapping) else None
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            raise RuntimeError("Impossible de déterminer les classes du modèle gelé.")

        classes = list(classes)
        if 1 not in classes:
            raise RuntimeError("Classe positive 1 absente du modèle gelé.")

        p_raw = float(probabilities[0][classes.index(1)])
        p_cal = sigmoid(logit(p_raw) + self.alpha)

        if not (0.0 <= p_raw <= 1.0 and 0.0 <= p_cal <= 1.0):
            raise RuntimeError("Probabilité invalide.")

        return p_raw, p_cal


# =============================================================================
# NOTIFICATIONS
# =============================================================================

def _escape_applescript(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


def send_macos_notification(title: str, message: str, sound: str = "Glass") -> bool:
    if not MACOS_NOTIFICATIONS_ENABLED:
        return False
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}" '
        f'sound name "{_escape_applescript(sound)}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        logger.warning("Notification macOS indisponible : %s", exc)
        return False


def send_telegram_notification(message: str) -> bool:
    if not TELEGRAM_NOTIFICATIONS_ENABLED:
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram activé mais token/chat_id manquant.")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": str(message),
                "disable_web_page_preview": True,
            },
            timeout=TELEGRAM_API_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        return bool(body.get("ok"))
    except Exception as exc:
        logger.warning("Notification Telegram indisponible : %s", exc)
        return False


# =============================================================================
# SHADOW SCANNER
# =============================================================================

AUDIT_COLUMNS = [
    "timestamp_utc",
    "match_id",
    "home_team",
    "away_team",
    "observed_live_minute",
    "status",
    "score_37_home",
    "score_37_away",
    "model_features_available",
    "missing_features",
    "total_attempt",
    "probability",
    "reason",
]

@dataclass
class SnapshotResult:
    match: LiveMatch
    score_37_home: Optional[int]
    score_37_away: Optional[int]
    values: Dict[str, Any]
    provenance: Dict[str, str]
    coverage: int
    missing: List[str]
    timeline_payload: Any
    timeline_events: List[Dict[str, Any]]


class FirstHalf37ShadowScanner:
    def __init__(
        self,
        client: StatsAPIClient,
        model: FrozenFirstHalf37Model,
        feature_module: Any,
        audit_module: Any,
    ):
        self.client = client
        self.model = model
        self.feat = feature_module
        self.audit = audit_module

        self.run_count = 0
        self._file_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self.scored_ids: Set[str] = set()
        self.permanent_rejected_ids: Set[str] = set()
        self.attempt_counts: Dict[str, int] = {}
        self.last_target_check: Dict[str, float] = {}
        self.corner_baselines: Dict[str, Dict[str, Any]] = {}
        self.live_stats_snapshots: Dict[str, Dict[str, Any]] = {}
        self.recovery_ids: Set[str] = set()
        # Preserve the exact score observed at 37' for late shadow recovery.
        # This is an as-of-37 snapshot; it is never replaced with a later score.
        self.score37_cache: Dict[str, Tuple[int, int]] = {}

        # Load the frozen 37' context first. _load_existing_state() then
        # reconciles legacy audit rows and can recover contexts written before V6.
        self._load_score37_contexts()
        self._load_existing_state()
        self._load_corner_baselines()
        self._load_live_stats_snapshots()

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def _load_score37_contexts(self) -> None:
        """Load immutable exact-37 score contexts persisted by V6."""
        if not SCORE37_CONTEXT_PATH.exists():
            return
        try:
            raw = json.loads(SCORE37_CONTEXT_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                return

            loaded = 0
            for match_id, payload in raw.items():
                if not isinstance(payload, Mapping):
                    continue
                sh = safe_int(payload.get("score_37_home"))
                sa = safe_int(payload.get("score_37_away"))
                if sh is None or sa is None:
                    continue
                self.score37_cache[str(match_id)] = (int(sh), int(sa))
                loaded += 1

            logger.info("FH37 frozen contexts loaded : %d", loaded)
        except Exception as exc:
            logger.warning("Frozen score37 context cache illisible : %s", exc)

    def _save_score37_contexts(self) -> None:
        """Persist immutable exact-37 score contexts atomically."""
        SCORE37_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(match_id): {
                "score_37_home": int(score[0]),
                "score_37_away": int(score[1]),
            }
            for match_id, score in self.score37_cache.items()
        }
        tmp = SCORE37_CONTEXT_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, SCORE37_CONTEXT_PATH)

    def _freeze_exact37_context(self, match: LiveMatch) -> Optional[Tuple[int, int]]:
        """
        Freeze the FIRST score observed while the live listing says minute 37.

        setdefault semantics are intentional: a later poll, retry or feed refresh
        must never overwrite the original as-of-37 context.
        """
        if match.minute != SNAPSHOT_MINUTE:
            return self.score37_cache.get(match.match_id)

        if match.home_score is None or match.away_score is None:
            return self.score37_cache.get(match.match_id)

        candidate = (int(match.home_score), int(match.away_score))

        with self._state_lock:
            existing = self.score37_cache.get(match.match_id)
            if existing is None:
                self.score37_cache[match.match_id] = candidate
                existing = candidate
                should_save = True
            else:
                should_save = False

        if should_save:
            self._save_score37_contexts()
            logger.info(
                "FH37 CONTEXT FROZEN | %s %d-%d %s | first_observed_exact37",
                match.home_team,
                existing[0],
                existing[1],
                match.away_team,
            )
        elif existing != candidate:
            logger.warning(
                "FH37 CONTEXT IMMUTABLE | %s vs %s | frozen37=%d-%d | "
                "current_listing_at_37=%d-%d | frozen value kept",
                match.home_team,
                match.away_team,
                existing[0],
                existing[1],
                candidate[0],
                candidate[1],
            )

        return existing

    def _frozen_score37(self, match_id: str) -> Optional[Tuple[int, int]]:
        with self._state_lock:
            return self.score37_cache.get(str(match_id))

    def _drop_score37_context(self, match_id: str) -> None:
        """Remove only after a match is definitively closed."""
        with self._state_lock:
            existed = self.score37_cache.pop(str(match_id), None)
        if existed is not None:
            self._save_score37_contexts()

    def _load_existing_state(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if PREDICTIONS_PATH.exists():
            try:
                df = pd.read_csv(PREDICTIONS_PATH, usecols=["match_id"])
                self.scored_ids = set(df["match_id"].dropna().astype(str))
            except Exception as exc:
                logger.warning("Predictions existantes illisibles : %s", exc)

        if AUDIT_PATH.exists():
            try:
                audit_df = pd.read_csv(
                    AUDIT_PATH,
                    usecols=lambda c: c in {
                        "match_id",
                        "status",
                        "observed_live_minute",
                        "score_37_home",
                        "score_37_away",
                    },
                )
                audit_df["match_id"] = audit_df["match_id"].astype(str)
                status_text = audit_df["status"].astype(str)
                mins = pd.to_numeric(
                    audit_df.get("observed_live_minute"),
                    errors="coerce",
                )
                sh = pd.to_numeric(
                    audit_df.get("score_37_home"),
                    errors="coerce",
                )
                sa = pd.to_numeric(
                    audit_df.get("score_37_away"),
                    errors="coerce",
                )

                # Recover immutable eligible contexts from legacy exact-37 audit
                # rows. This also repairs sessions where an old V5 later wrote a
                # false CONTEXT_REJECTED at 38'+ after the match had been 1-1 at 37'.
                exact37_eligible = (
                    mins.eq(SNAPSHOT_MINUTE)
                    & sh.notna()
                    & sa.notna()
                    & sh.eq(sa)
                    & sh.gt(0)
                )

                recovered_context = False
                for row in audit_df.loc[
                    exact37_eligible,
                    ["match_id", "score_37_home", "score_37_away"],
                ].itertuples(index=False):
                    mid = str(row.match_id)
                    candidate = (
                        int(float(row.score_37_home)),
                        int(float(row.score_37_away)),
                    )
                    if mid not in self.score37_cache:
                        self.score37_cache[mid] = candidate
                        recovered_context = True

                if recovered_context:
                    self._save_score37_contexts()

                # SCORING_COMPLETE is always final.
                scored_from_audit = set(
                    audit_df.loc[
                        status_text.eq("SCORING_COMPLETE"),
                        "match_id",
                    ].dropna().astype(str)
                )

                # CONTEXT_REJECTED is final EXCEPT for the known V5 failure mode:
                # exact 37' eligible context existed, then a later 38'+ row was
                # rejected because the current score had changed.
                rejected_rows = audit_df.loc[
                    status_text.eq("CONTEXT_REJECTED")
                ].copy()

                false_late_reject_ids: Set[str] = set()
                eligible37_ids = set(
                    audit_df.loc[
                        exact37_eligible,
                        "match_id",
                    ].dropna().astype(str)
                )

                if len(rejected_rows):
                    rejected_mins = pd.to_numeric(
                        rejected_rows.get("observed_live_minute"),
                        errors="coerce",
                    )
                    late_rejected_ids = set(
                        rejected_rows.loc[
                            rejected_mins.gt(SNAPSHOT_MINUTE),
                            "match_id",
                        ].dropna().astype(str)
                    )
                    false_late_reject_ids = eligible37_ids & late_rejected_ids

                truly_rejected = set(
                    rejected_rows["match_id"].dropna().astype(str)
                ) - false_late_reject_ids

                self.permanent_rejected_ids = scored_from_audit | truly_rejected

                retry_mask = status_text.isin({
                    "FEATURES_INCOMPLETE_RETRYABLE",
                    "SOURCE_TEMPORARY_UNAVAILABLE",
                    "EXPIRED_INCOMPLETE",
                })

                # Every exact-37 eligible match remains recoverable unless it
                # already scored or was truly rejected at exact 37.
                recoverable_ids = set(
                    audit_df.loc[
                        retry_mask & mins.eq(SNAPSHOT_MINUTE),
                        "match_id",
                    ].dropna().astype(str)
                )
                recoverable_ids.update(false_late_reject_ids)
                recoverable_ids.update(eligible37_ids)

                recoverable_ids -= self.permanent_rejected_ids
                recoverable_ids -= self.scored_ids
                self.recovery_ids.update(recoverable_ids)

                if false_late_reject_ids:
                    logger.warning(
                        "V6 repaired %d legacy late CONTEXT_REJECTED row(s) "
                        "using frozen eligible exact-37 context.",
                        len(false_late_reject_ids),
                    )
            except Exception as exc:
                logger.warning("Audit existant illisible : %s", exc)

        logger.info(
            "State loaded | scored=%d | permanently_closed=%d",
            len(self.scored_ids),
            len(self.permanent_rejected_ids),
        )

    def _load_corner_baselines(self) -> None:
        if not CORNER_BASELINES_PATH.exists():
            return
        try:
            raw = json.loads(CORNER_BASELINES_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping):
                self.corner_baselines = {
                    str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)
                }
            logger.info("FH37 corner baselines loaded : %d", len(self.corner_baselines))
        except Exception as exc:
            logger.warning("Corner baseline cache illisible : %s", exc)

    def _save_corner_baselines(self) -> None:
        CORNER_BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CORNER_BASELINES_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.corner_baselines, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, CORNER_BASELINES_PATH)

    def _load_live_stats_snapshots(self) -> None:
        if not LIVE_STATS_SNAPSHOTS_PATH.exists():
            return
        try:
            raw = json.loads(LIVE_STATS_SNAPSHOTS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping):
                self.live_stats_snapshots = {
                    str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)
                }
            logger.info(
                "FH37 live-stats snapshot cache loaded : %d",
                len(self.live_stats_snapshots),
            )
        except Exception as exc:
            logger.warning("Live-stats snapshot cache illisible : %s", exc)

    def _save_live_stats_snapshots(self) -> None:
        LIVE_STATS_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LIVE_STATS_SNAPSHOTS_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                self.live_stats_snapshots,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, LIVE_STATS_SNAPSHOTS_PATH)

    def _save_raw_live_stats(
        self,
        match_id: str,
        minute: int,
        payload: Any,
    ) -> Optional[str]:
        if payload is None:
            return None
        try:
            LIVE_STATS_RAW_DIR.mkdir(parents=True, exist_ok=True)
            path = LIVE_STATS_RAW_DIR / f"{match_id}_m{minute}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            return str(path.relative_to(OUTPUT_DIR))
        except Exception as exc:
            logger.warning("Impossible de sauvegarder live-stats raw %s @%d : %s", match_id, minute, exc)
            return None

    def _capture_live_stats_snapshot(
        self,
        match: LiveMatch,
        minute: int,
        payload: Any,
    ) -> Dict[str, Any]:
        parsed = extract_cumulative_live_stats(payload)
        availability = live_stats_availability(parsed)
        raw_file = self._save_raw_live_stats(match.match_id, minute, payload)

        record: Dict[str, Any] = {
            "minute": int(minute),
            "captured_at_utc": now_utc_iso(),
            "home_team": match.home_team,
            "away_team": match.away_team,
            "raw_file": raw_file or "",
            **parsed,
        }

        with self._state_lock:
            entry = self.live_stats_snapshots.setdefault(match.match_id, {})
            entry[str(minute)] = record
            self._save_live_stats_snapshots()

            # Keep the old corner cache in sync for backward compatibility.
            home_corners = safe_float(parsed.get("home_corners"))
            if minute == CORNER_BASELINE_MINUTE and home_corners is not None:
                self.corner_baselines[match.match_id] = {
                    "minute": minute,
                    "home_corners": home_corners,
                    "captured_at_utc": record["captured_at_utc"],
                }
                self._save_corner_baselines()

        logger.info(
            "FH37 LIVE-STATS @%d | %s vs %s | xg=%s shots=%s corners=%s | "
            "xg=%s-%s total=%s | shots=%s-%s | corners=%s-%s",
            minute,
            match.home_team,
            match.away_team,
            "YES" if availability["xg"] else "NO",
            "YES" if availability["shots"] else "NO",
            "YES" if availability["corners"] else "NO",
            _fmt_stat(parsed.get("home_xg")),
            _fmt_stat(parsed.get("away_xg")),
            _fmt_stat(parsed.get("total_xg")),
            _fmt_stat(parsed.get("home_shots"), 0),
            _fmt_stat(parsed.get("away_shots"), 0),
            _fmt_stat(parsed.get("home_corners"), 0),
            _fmt_stat(parsed.get("away_corners"), 0),
        )
        return record

    def _get_live_stats_snapshot(self, match_id: str, minute: int) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            entry = self.live_stats_snapshots.get(match_id)
            if not isinstance(entry, Mapping):
                return None
            record = entry.get(str(minute))
            return dict(record) if isinstance(record, Mapping) else None

    def preload_corner_baseline(self, match: LiveMatch) -> None:
        """Capture the full cumulative live-stats snapshot at 29'."""
        if match.minute != CORNER_BASELINE_MINUTE:
            return
        with self._state_lock:
            existing = self.live_stats_snapshots.get(match.match_id, {})
            if isinstance(existing, Mapping) and str(CORNER_BASELINE_MINUTE) in existing:
                return

        payload = self.client.get_live_stats(match.match_id, soft=True)
        if payload is None:
            logger.info(
                "FH37 LIVE-STATS baseline @%d unavailable | %s vs %s",
                CORNER_BASELINE_MINUTE,
                match.home_team,
                match.away_team,
            )
            return

        record = self._capture_live_stats_snapshot(
            match,
            CORNER_BASELINE_MINUTE,
            payload,
        )

        useful = [
            record.get("total_xg"),
            record.get("home_shots"),
            record.get("home_corners"),
        ]
        if not any(safe_float(v) is not None for v in useful):
            logger.info(
                "FH37 baseline @%d saved but model stats absent | %s vs %s",
                CORNER_BASELINE_MINUTE,
                match.home_team,
                match.away_team,
            )

    # ------------------------------------------------------------------
    # AUDIT / SAVE
    # ------------------------------------------------------------------

    def _append_csv(self, path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            exists = path.exists()
            pd.DataFrame([dict(row)]).to_csv(
                path,
                mode="a",
                header=not exists,
                index=False,
            )

    def audit_row(
        self,
        match: LiveMatch,
        status: str,
        **extra: Any,
    ) -> None:
        row: Dict[str, Any] = {column: "" for column in AUDIT_COLUMNS}
        row.update({
            "timestamp_utc": now_utc_iso(),
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "observed_live_minute": match.minute,
            "status": status,
        })
        for key, value in extra.items():
            if key in row:
                row[key] = value

        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            exists = AUDIT_PATH.exists()
            pd.DataFrame([row], columns=AUDIT_COLUMNS).to_csv(
                AUDIT_PATH,
                mode="a",
                header=not exists,
                index=False,
            )

    # ------------------------------------------------------------------
    # RECONSTRUCTION
    # ------------------------------------------------------------------

    def _derive_live_zero_home_corners_30_37(
        self,
        recent: Dict[str, Any],
        provenance: Dict[str, str],
        timeline_events: Sequence[Mapping[str, Any]],
        team_context: Mapping[str, Any],
        observed_minute: int,
    ) -> None:
        """
        The historical builder only writes a zero recent-corner count when a
        historical timeline is explicitly complete. In live mode, a successful
        current timeline at >=37' is an as-of feed, not a final-match timeline.

        We only synthesize home_corners_30_37=0 when there is literally NO
        corner event at all in 30..37. If a corner exists but ownership is
        unresolved, the feature stays missing and the 5/5 gate blocks scoring.
        """
        if recent.get("home_corners_30_37") is not None:
            return
        if observed_minute < SNAPSHOT_MINUTE or not timeline_events:
            return

        any_corner = False
        for event in timeline_events:
            try:
                if not self.feat.is_first_half_event(event, self.audit):
                    continue
                minute = self.feat.event_base_minute(event, self.audit)
                if minute is None or not (30 <= minute <= SNAPSHOT_MINUTE):
                    continue
                flags = self.audit.event_flags(event)
                if flags.get("corner"):
                    any_corner = True
                    break
            except Exception:
                continue

        if not any_corner:
            recent["home_corners_30_37"] = 0
            provenance["home_corners_30_37"] = "live_timeline_asof37_proven_zero"

    def _apply_live_stats_fallback(
        self,
        match: LiveMatch,
        values: Dict[str, Any],
        provenance: Dict[str, str],
        stats37: Optional[Mapping[str, Any]],
        stats29: Optional[Mapping[str, Any]],
    ) -> None:
        """
        Fill only MISSING frozen features from exact cumulative /live-stats.

        Event reconstruction (shotmap/timeline) remains canonical whenever it
        exists because that is closest to the historical training definition.
        /live-stats is the operational backbone/fallback and never overwrites a
        non-missing event-derived feature.
        """
        if not isinstance(stats37, Mapping):
            return

        total37 = safe_float(stats37.get("total_xg"))
        away37 = safe_float(stats37.get("away_xg"))
        home_shots37 = safe_float(stats37.get("home_shots"))
        home_corners37 = safe_float(stats37.get("home_corners"))

        direct = {
            "total_xg_37": total37,
            "away_xg_37": away37,
            "home_shots_37": home_shots37,
        }
        for feature, candidate in direct.items():
            if is_missing(values.get(feature)) and candidate is not None:
                values[feature] = float(candidate)
                provenance[feature] = "live_stats_cumulative_exact37"

        if isinstance(stats29, Mapping):
            total29 = safe_float(stats29.get("total_xg"))
            home_corners29 = safe_float(stats29.get("home_corners"))

            # Historical xg_30_37 = sum xG for shots with 30 <= minute <= 37.
            # Cumulative@37 - cumulative@29 reproduces that window exactly.
            if is_missing(values.get("xg_30_37")) and total37 is not None and total29 is not None:
                delta_xg = total37 - total29
                if delta_xg >= -1e-6:
                    values["xg_30_37"] = float(max(0.0, delta_xg))
                    provenance["xg_30_37"] = "live_stats_delta_total_xg_37_minus_29"
                else:
                    logger.warning(
                        "FH37 LIVE-STATS NONMONOTONIC xG | %s vs %s | @29=%.3f @37=%.3f",
                        match.home_team,
                        match.away_team,
                        total29,
                        total37,
                    )

            if (
                is_missing(values.get("home_corners_30_37"))
                and home_corners37 is not None
                and home_corners29 is not None
            ):
                delta_corners = home_corners37 - home_corners29
                if delta_corners >= -1e-6:
                    values["home_corners_30_37"] = float(max(0.0, delta_corners))
                    provenance["home_corners_30_37"] = "live_stats_delta_home_corners_37_minus_29"
                else:
                    logger.warning(
                        "FH37 LIVE-STATS NONMONOTONIC corners | %s vs %s | @29=%.0f @37=%.0f",
                        match.home_team,
                        match.away_team,
                        home_corners29,
                        home_corners37,
                    )

    def reconstruct_snapshot(self, match: LiveMatch) -> SnapshotResult:
        exact37 = match.minute == SNAPSHOT_MINUTE

        # ------------------------------------------------------------------
        # 1) LIVE-STATS — operational backbone.
        # ------------------------------------------------------------------
        # We may use cumulative stats ONLY if they were captured at the exact
        # snapshot minute 37. During late recovery (38'-44'), current cumulative
        # stats would contain future information and are therefore forbidden.
        stats37: Optional[Dict[str, Any]] = self._get_live_stats_snapshot(
            match.match_id,
            SNAPSHOT_MINUTE,
        )

        if exact37:
            live_stats_payload = self.client.get_live_stats(match.match_id, soft=True)
            if live_stats_payload is not None:
                stats37 = self._capture_live_stats_snapshot(
                    match,
                    SNAPSHOT_MINUTE,
                    live_stats_payload,
                )
            elif stats37 is None:
                logger.info(
                    "FH37 LIVE-STATS @37 unavailable | %s vs %s",
                    match.home_team,
                    match.away_team,
                )
        elif stats37 is not None:
            availability = live_stats_availability(stats37)
            logger.info(
                "FH37 LIVE-STATS @37 CACHE | %s vs %s | recovery_at=%d' | "
                "xg=%s shots=%s corners=%s",
                match.home_team,
                match.away_team,
                match.minute,
                "YES" if availability["xg"] else "NO",
                "YES" if availability["shots"] else "NO",
                "YES" if availability["corners"] else "NO",
            )

        stats29 = self._get_live_stats_snapshot(
            match.match_id,
            CORNER_BASELINE_MINUTE,
        )

        # Backward compatibility with the V4 corner-only baseline cache.
        if stats29 is None:
            old_baseline = self.corner_baselines.get(match.match_id)
            if isinstance(old_baseline, Mapping):
                stats29 = {
                    "minute": CORNER_BASELINE_MINUTE,
                    "home_corners": safe_float(old_baseline.get("home_corners")),
                }

        # ------------------------------------------------------------------
        # 2) EVENT FEEDS — fidelity-first reconstruction where available.
        # ------------------------------------------------------------------
        # Live auxiliary endpoints can lag. 409/404/425/503 are soft so one
        # source does not abort reconstruction from the others.
        detail_payload = self.client.get_match(match.match_id, soft=True)
        timeline_payload = self.client.get_live_timeline(match.match_id, soft=True)
        shotmap_payload = self.client.get_shotmap(match.match_id, soft=True)

        timeline_events = self.feat.payload_list(timeline_payload)
        shot_events = self.feat.payload_list(shotmap_payload)

        logger.info(
            "FH37 SOURCES | %s vs %s | observed=%d' | listing_score=%s-%s | "
            "live_stats37=%s | baseline29=%s | timeline_events=%d | shot_events=%d",
            match.home_team,
            match.away_team,
            match.minute,
            match.home_score if match.home_score is not None else "?",
            match.away_score if match.away_score is not None else "?",
            "YES" if stats37 is not None else "NO",
            "YES" if stats29 is not None else "NO",
            len(timeline_events),
            len(shot_events),
        )

        source_row = {
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "competition_id": match.competition_id,
            "season_id": match.season_id,
        }

        team_context = self.feat.get_detail_team_context(
            detail_payload if detail_payload is not None else match.raw,
            source_row,
            self.audit,
        )

        base, base_prov = self.feat.reconstruct_base_live37(
            timeline_events=timeline_events,
            shot_events=shot_events,
            team_context=team_context,
            timeline_payload=timeline_payload,
            audit=self.audit,
        )

        # V6 CONTEXT INVARIANT
        # --------------------
        # The FIRST live score observed at exact 37' is authoritative for the
        # strategy context. Once frozen, it ALWAYS overrides any score rebuilt
        # from later/current feeds. This prevents a 1-1 @37 match from becoming
        # a false 1-2 @38 context during feature retries.
        frozen37 = self._frozen_score37(match.match_id)

        if frozen37 is not None:
            sh, sa = frozen37
            base.update({
                "score_37_home": sh,
                "score_37_away": sa,
                "goal_diff_37": abs(sh - sa),
                "is_draw_37": int(sh == sa),
                "leading_team_37": 0 if sh > sa else 1 if sa > sh else 2,
                "total_goals_37": sh + sa,
                "has_goal_37": int((sh + sa) > 0),
            })
            for name in (
                "score_37_home", "score_37_away", "goal_diff_37",
                "is_draw_37", "leading_team_37", "total_goals_37", "has_goal_37",
            ):
                base_prov[name] = "frozen_first_live_listing_exact37"

        elif exact37 and match.home_score is not None and match.away_score is not None:
            # Defensive fallback. analyze_match normally freezes this before
            # reconstruction; this branch avoids losing the exact-37 score if
            # reconstruct_snapshot is called independently.
            sh = int(match.home_score)
            sa = int(match.away_score)
            base.update({
                "score_37_home": sh,
                "score_37_away": sa,
                "goal_diff_37": abs(sh - sa),
                "is_draw_37": int(sh == sa),
                "leading_team_37": 0 if sh > sa else 1 if sa > sh else 2,
                "total_goals_37": sh + sa,
                "has_goal_37": int((sh + sa) > 0),
            })
            for name in (
                "score_37_home", "score_37_away", "goal_diff_37",
                "is_draw_37", "leading_team_37", "total_goals_37", "has_goal_37",
            ):
                base_prov[name] = "live_match_listing_exact37_unfrozen_fallback"

        recent, recent_prov = self.feat.reconstruct_recent_windows(
            timeline_events=timeline_events,
            shot_events=shot_events,
            team_context=team_context,
            audit=self.audit,
            timeline_full_flag=False,
            shotmap_feed_proven=bool(shot_events),
        )

        self._derive_live_zero_home_corners_30_37(
            recent,
            recent_prov,
            timeline_events,
            team_context,
            match.minute,
        )

        xg, xg_prov = self.feat.reconstruct_xg_cumulative(
            shot_events=shot_events,
            team_context=team_context,
            audit=self.audit,
        )

        values: Dict[str, Any] = {}
        values.update(base)
        values.update(recent)
        values.update(xg)

        provenance: Dict[str, str] = {}
        provenance.update(base_prov)
        provenance.update(recent_prov)
        provenance.update(xg_prov)

        # ------------------------------------------------------------------
        # 3) Fill only missing frozen features from exact 29'/37' live-stats.
        # ------------------------------------------------------------------
        self._apply_live_stats_fallback(
            match=match,
            values=values,
            provenance=provenance,
            stats37=stats37,
            stats29=stats29,
        )

        sh = safe_int(values.get("score_37_home"))
        sa = safe_int(values.get("score_37_away"))

        missing = [
            feature for feature in MODEL_FEATURES if is_missing(values.get(feature))
        ]

        logger.info(
            "FH37 FEATURE SOURCES | %s vs %s | %s",
            match.home_team,
            match.away_team,
            " | ".join(
                f"{f}={_fmt_stat(values.get(f)) if 'xg' in f else _fmt_stat(values.get(f), 0)}"
                f"[{provenance.get(f, 'MISSING')}]"
                for f in MODEL_FEATURES
            ),
        )

        return SnapshotResult(
            match=match,
            score_37_home=sh,
            score_37_away=sa,
            values=values,
            provenance=provenance,
            coverage=len(MODEL_FEATURES) - len(missing),
            missing=missing,
            timeline_payload=timeline_payload,
            timeline_events=timeline_events,
        )

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    def _prediction_row(
        self,
        result: SnapshotResult,
        p_raw: float,
        p_cal: float,
    ) -> Dict[str, Any]:
        match = result.match
        row: Dict[str, Any] = {
            "scored_at_utc": now_utc_iso(),
            "mode": "SHADOW_ONLY",
            "model_name": self.model.model_name,
            "model_sha256": self.model.model_sha256,
            "calibration_alpha": self.model.alpha,
            "snapshot_minute": SNAPSHOT_MINUTE,
            "observed_live_minute": match.minute,
            "timing_quality": "EXACT_37" if match.minute == 37 else f"LATE_RECOVERY_AT_{match.minute}",
            "match_id": match.match_id,
            "competition_id": match.competition_id,
            "season_id": match.season_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "score_37_home": result.score_37_home,
            "score_37_away": result.score_37_away,
            "draw_score_37": f"{result.score_37_home}-{result.score_37_away}",
            "context": "draw_with_goals",
        }

        for feature in MODEL_FEATURES:
            row[feature] = result.values.get(feature, np.nan)

        row.update({
            "model_features_available": result.coverage,
            "prediction_quality": "FULL_5OF5",
            "raw_probability": p_raw,
            "probability": p_cal,
            "fair_odds": fair_odds(p_cal),
            "probability_band": probability_band(p_cal),
            "decision": "SHADOW_ONLY_NO_BET",
            "feature_provenance_json": json.dumps(
                {f: result.provenance.get(f, "") for f in MODEL_FEATURES},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "goal_after_37_before_ht": np.nan,
            "target_status": "PENDING",
            "target_resolution_class": "",
            "target_confidence": "",
            "goal_minutes_after_37_before_ht": "",
            "first_half_period_end_seen": np.nan,
            "timeline_full": np.nan,
            "first_half_goal_count": np.nan,
        })
        return row

    def _notify_prediction(self, row: Mapping[str, Any]) -> None:
        p = float(row["probability"])
        fair = float(row["fair_odds"])
        title = "👁 FIRST HALF 37 — SHADOW"
        message = (
            f"{row['home_team']} {row['score_37_home']}-{row['score_37_away']} "
            f"{row['away_team']} | P={p * 100:.1f}% | Fair={fair:.2f} | 5/5"
        )
        send_macos_notification(title, message, sound="Glass")
        send_telegram_notification(
            "👁 FIRST HALF 37 — SHADOW\n\n"
            f"{row['home_team']} {row['score_37_home']}-{row['score_37_away']} {row['away_team']}\n"
            f"Snapshot : 37' (observé {row['observed_live_minute']}')\n"
            f"Probabilité calibrée : {p * 100:.2f}%\n"
            f"Fair odds : {fair:.2f}\n"
            f"Qualité : FULL_5OF5\n"
            f"Mode : SHADOW — AUCUN PARI"
        )

    def analyze_match(self, match: LiveMatch) -> Optional[Dict[str, Any]]:
        exact37 = match.minute == SNAPSHOT_MINUTE

        with self._state_lock:
            if match.match_id in self.scored_ids or match.match_id in self.permanent_rejected_ids:
                return None
            self.attempt_counts[match.match_id] = self.attempt_counts.get(match.match_id, 0) + 1
            total_attempt = self.attempt_counts[match.match_id]
            if exact37:
                self.recovery_ids.add(match.match_id)

        # V6: freeze FIRST, then evaluate the strategy context from the frozen
        # as-of-37 score. Later retries must never evaluate the current score.
        frozen37 = self._freeze_exact37_context(match) if exact37 else self._frozen_score37(match.match_id)

        if exact37 and frozen37 is not None:
            frozen_home, frozen_away = frozen37
            if not (frozen_home == frozen_away and frozen_home > 0):
                self.audit_row(
                    match,
                    "CONTEXT_REJECTED",
                    score_37_home=frozen_home,
                    score_37_away=frozen_away,
                    reason="frozen_exact37_not_draw_with_goals",
                    total_attempt=total_attempt,
                )
                with self._state_lock:
                    self.permanent_rejected_ids.add(match.match_id)
                    self.recovery_ids.discard(match.match_id)
                self._drop_score37_context(match.match_id)
                logger.info(
                    "FH37 context rejected | %s %d-%d %s | frozen exact37",
                    match.home_team,
                    frozen_home,
                    frozen_away,
                    match.away_team,
                )
                return None

        # A late-recovery candidate MUST have a frozen exact-37 context.
        # If it does not, never infer the 37' strategy context from the current
        # 38'+ score.
        if not exact37 and frozen37 is None:
            self.audit_row(
                match,
                "SOURCE_TEMPORARY_UNAVAILABLE",
                score_37_home=None,
                score_37_away=None,
                reason="late_recovery_missing_frozen_exact37_context",
                total_attempt=total_attempt,
            )
            logger.warning(
                "FH37 RECOVERY BLOCKED | %s vs %s | observed=%d' | "
                "no frozen exact-37 context; current score ignored",
                match.home_team,
                match.away_team,
                match.minute,
            )
            return None

        last_result: Optional[SnapshotResult] = None
        last_error: Optional[str] = None

        for immediate_try in range(MAX_IMMEDIATE_FEATURE_RETRIES + 1):
            try:
                result = self.reconstruct_snapshot(match)
                last_result = result
                sh = result.score_37_home
                sa = result.score_37_away

                if sh is None or sa is None:
                    last_error = "score_37_unavailable"
                    logger.warning(
                        "FH37 BLOCKED | %s vs %s | observed=%d' | score37=UNAVAILABLE | "
                        "listing_score=%s-%s | features=%d/5 | missing=%s | timeline_events=%d",
                        match.home_team,
                        match.away_team,
                        match.minute,
                        match.home_score if match.home_score is not None else "?",
                        match.away_score if match.away_score is not None else "?",
                        result.coverage,
                        ", ".join(result.missing),
                        len(result.timeline_events),
                    )
                elif not (sh == sa and sh > 0):
                    # With a frozen eligible context this branch must be
                    # impossible. Never allow a later/current feed to overturn
                    # the immutable exact-37 context.
                    frozen_guard = self._frozen_score37(match.match_id)
                    if (
                        frozen_guard is not None
                        and frozen_guard[0] == frozen_guard[1]
                        and frozen_guard[0] > 0
                    ):
                        logger.error(
                            "FH37 CONTEXT INVARIANT VIOLATION prevented | %s vs %s | "
                            "reconstructed=%s-%s | frozen37=%d-%d",
                            match.home_team,
                            match.away_team,
                            sh,
                            sa,
                            frozen_guard[0],
                            frozen_guard[1],
                        )
                        last_error = "context_invariant_violation_retryable"
                        continue

                    self.audit_row(
                        match,
                        "CONTEXT_REJECTED",
                        score_37_home=sh,
                        score_37_away=sa,
                        reason="not_draw_with_goals_at_frozen_37",
                        total_attempt=total_attempt,
                    )
                    with self._state_lock:
                        self.permanent_rejected_ids.add(match.match_id)
                        self.recovery_ids.discard(match.match_id)
                    self._drop_score37_context(match.match_id)
                    return None
                elif result.coverage == len(MODEL_FEATURES):
                    p_raw, p_cal = self.model.predict(result.values)
                    row = self._prediction_row(result, p_raw, p_cal)
                    self._append_csv(PREDICTIONS_PATH, row)
                    self.audit_row(
                        match,
                        "SCORING_COMPLETE",
                        score_37_home=sh,
                        score_37_away=sa,
                        model_features_available=result.coverage,
                        missing_features="",
                        probability=p_cal,
                        total_attempt=total_attempt,
                        reason=(
                            "exact37" if exact37
                            else "late_shadow_recovery_hardcut_le37"
                        ),
                    )
                    with self._state_lock:
                        self.scored_ids.add(match.match_id)
                        self.permanent_rejected_ids.add(match.match_id)
                        self.recovery_ids.discard(match.match_id)
                    self._drop_score37_context(match.match_id)

                    logger.warning(
                        "👁 SHADOW SCORED | %s %d-%d %s | snapshot=37' observed=%d' | "
                        "P=%.2f%% | fair=%.2f | 5/5 | timing=%s",
                        match.home_team,
                        sh,
                        sa,
                        match.away_team,
                        match.minute,
                        p_cal * 100.0,
                        fair_odds(p_cal),
                        row["timing_quality"],
                    )

                    if exact37:
                        self._notify_prediction(row)
                        print("\a", end="", flush=True)
                    return row
                else:
                    last_error = "features_incomplete"
                    logger.warning(
                        "FH37 BLOCKED | %s vs %s | observed=%d' | features=%d/5 | missing=%s",
                        match.home_team,
                        match.away_team,
                        match.minute,
                        result.coverage,
                        ", ".join(result.missing),
                    )

            except Exception as exc:
                last_error = f"reconstruction_error:{type(exc).__name__}:{exc}"
                logger.warning(
                    "FH37 reconstruction failed | %s vs %s | %s",
                    match.home_team,
                    match.away_team,
                    exc,
                )

            if immediate_try < MAX_IMMEDIATE_FEATURE_RETRIES:
                time.sleep(max(0.0, RETRY_DELAY_SECONDS))

        coverage = last_result.coverage if last_result is not None else 0
        missing = last_result.missing if last_result is not None else MODEL_FEATURES
        frozen_final = self._frozen_score37(match.match_id)
        sh = (
            frozen_final[0]
            if frozen_final is not None
            else (
                last_result.score_37_home
                if last_result is not None
                else None
            )
        )
        sa = (
            frozen_final[1]
            if frozen_final is not None
            else (
                last_result.score_37_away
                if last_result is not None
                else None
            )
        )

        status = (
            "SOURCE_TEMPORARY_UNAVAILABLE"
            if "reconstruction_error" in str(last_error)
            else "FEATURES_INCOMPLETE_RETRYABLE"
        )
        self.audit_row(
            match,
            status,
            score_37_home=sh,
            score_37_away=sa,
            model_features_available=coverage,
            missing_features="|".join(missing),
            total_attempt=total_attempt,
            reason=(
                f"{last_error or 'unknown'}|context37_frozen"
                if self._frozen_score37(match.match_id) is not None
                else (last_error or "unknown")
            ),
        )

        # No permanent close for transient incompleteness. A match first seen at
        # 37' remains eligible for late shadow-only recovery through 44'.
        return None

    # ------------------------------------------------------------------
    # TARGET RESOLUTION
    # ------------------------------------------------------------------

    def _update_target_rows(self) -> int:
        if not TARGET_RESOLUTION_ENABLED or not PREDICTIONS_PATH.exists():
            return 0

        with self._file_lock:
            try:
                df = pd.read_csv(PREDICTIONS_PATH)
            except Exception as exc:
                logger.warning("Impossible de lire le shadow CSV pour targets : %s", exc)
                return 0

        if df.empty or "target_status" not in df.columns:
            return 0

        pending_indices = df.index[df["target_status"].fillna("PENDING") != "RESOLVED"].tolist()
        if not pending_indices:
            return 0

        changed = 0
        now = datetime.now(timezone.utc)

        for idx in pending_indices:
            match_id = str(df.at[idx, "match_id"])
            scored_at = pd.to_datetime(df.at[idx, "scored_at_utc"], utc=True, errors="coerce")
            if pd.isna(scored_at):
                continue

            age = (now - scored_at.to_pydatetime()).total_seconds()
            if age < TARGET_FIRST_CHECK_SECONDS:
                continue

            monotonic_now = time.monotonic()
            last_check = self.last_target_check.get(match_id, 0.0)
            if monotonic_now - last_check < TARGET_RECHECK_SECONDS:
                continue
            self.last_target_check[match_id] = monotonic_now

            try:
                # Tant que le match est en cours, /timeline renvoie 409.
                # On essaie donc d'abord la ressource LIVE; la timeline settled
                # reste un fallback pour les contrôles plus tardifs.
                timeline_payload = self.client.get_live_timeline(match_id, soft=True)
                events = self.feat.payload_list(timeline_payload)

                if not events:
                    timeline_payload = self.client.get_timeline(match_id, soft=True)
                    events = self.feat.payload_list(timeline_payload)

                if not events:
                    continue

                target = self.feat.resolve_target(
                    timeline_events=events,
                    timeline_payload=timeline_payload,
                    audit=self.audit,
                )
            except Exception as exc:
                logger.warning("Target check failed | %s | %s", match_id, exc)
                continue

            for column in (
                "goal_after_37_before_ht",
                "target_status",
                "target_resolution_class",
                "target_confidence",
                "goal_minutes_after_37_before_ht",
                "first_half_period_end_seen",
                "timeline_full",
                "first_half_goal_count",
            ):
                if column in target:
                    df.at[idx, column] = target[column]

            if target.get("target_status") == "RESOLVED":
                changed += 1
                y = safe_int(target.get("goal_after_37_before_ht"))
                home = str(df.at[idx, "home_team"])
                away = str(df.at[idx, "away_team"])
                score = f"{safe_int(df.at[idx, 'score_37_home'])}-{safe_int(df.at[idx, 'score_37_away'])}"
                p = safe_float(df.at[idx, "probability"]) or 0.0

                logger.warning(
                    "SHADOW TARGET RESOLVED | %s %s %s | y=%s | P=%.2f%% | class=%s",
                    home,
                    score,
                    away,
                    y,
                    p * 100.0,
                    target.get("target_resolution_class", ""),
                )

                outcome = "✅ HIT" if y == 1 else "❌ MISS"
                send_macos_notification(
                    "FIRST HALF 37 — SHADOW RESULT",
                    f"{outcome} | {home} {score} {away} | P={p * 100:.1f}%",
                    sound="Glass",
                )

        if changed:
            tmp = PREDICTIONS_PATH.with_suffix(".tmp")
            with self._file_lock:
                df.to_csv(tmp, index=False)
                os.replace(tmp, PREDICTIONS_PATH)
            self.print_shadow_performance(df)

        return changed

    def print_shadow_performance(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            if not PREDICTIONS_PATH.exists():
                return
            try:
                df = pd.read_csv(PREDICTIONS_PATH)
            except Exception:
                return

        y = pd.to_numeric(df.get("goal_after_37_before_ht"), errors="coerce")
        p = pd.to_numeric(df.get("probability"), errors="coerce")
        mask = y.notna() & p.notna()
        if not mask.any():
            return

        yy = y.loc[mask].astype(int).to_numpy()
        pp = p.loc[mask].astype(float).to_numpy()
        n = len(yy)
        positives = int(yy.sum())

        parts = [
            f"resolved={n}",
            f"goals={positives}",
            f"rate={positives / n:.2%}",
            f"mean_p={float(np.mean(pp)):.2%}",
        ]

        if brier_score_loss is not None:
            try:
                parts.append(f"brier={brier_score_loss(yy, pp):.4f}")
            except Exception:
                pass

        if len(set(yy.tolist())) >= 2:
            if roc_auc_score is not None:
                try:
                    parts.append(f"roc={roc_auc_score(yy, pp):.4f}")
                except Exception:
                    pass
            if average_precision_score is not None:
                try:
                    parts.append(f"pr={average_precision_score(yy, pp):.4f}")
                except Exception:
                    pass

        logger.info("SHADOW PERFORMANCE | %s", " | ".join(parts))

    # ------------------------------------------------------------------
    # SCANNING
    # ------------------------------------------------------------------

    def find_candidates(
        self,
    ) -> Tuple[List[LiveMatch], List[LiveMatch], Dict[int, int], int]:
        raw_matches = self.client.get_live_matches()
        candidates: List[LiveMatch] = []
        baseline_candidates: List[LiveMatch] = []
        minute_distribution: Dict[int, int] = {}
        unparsed = 0

        for raw in raw_matches:
            match = parse_live_match(raw)
            if match is None:
                unparsed += 1
                continue

            minute_distribution[match.minute] = minute_distribution.get(match.minute, 0) + 1

            if match.minute == CORNER_BASELINE_MINUTE:
                baseline = self._get_live_stats_snapshot(
                    match.match_id, CORNER_BASELINE_MINUTE
                )
                baseline_complete = bool(
                    isinstance(baseline, Mapping)
                    and safe_float(baseline.get("total_xg")) is not None
                    and safe_float(baseline.get("home_corners")) is not None
                )
                if not baseline_complete:
                    baseline_candidates.append(match)

            with self._state_lock:
                closed = (
                    match.match_id in self.scored_ids
                    or match.match_id in self.permanent_rejected_ids
                )
                recovery = match.match_id in self.recovery_ids

            if closed:
                continue

            if match.minute == SNAPSHOT_MINUTE:
                candidates.append(match)
                continue

            if (
                LATE_RECOVERY_ENABLED
                and recovery
                and SNAPSHOT_MINUTE < match.minute <= RECOVERY_MAX_MINUTE
            ):
                candidates.append(match)
                continue

            if recovery and match.minute > RECOVERY_MAX_MINUTE:
                self.audit_row(
                    match,
                    "RECOVERY_WINDOW_EXPIRED",
                    total_attempt=self.attempt_counts.get(match.match_id, 0),
                    reason=f"not_scored_by_{RECOVERY_MAX_MINUTE}",
                )
                with self._state_lock:
                    self.recovery_ids.discard(match.match_id)
                self._drop_score37_context(match.match_id)

        candidates.sort(
            key=lambda m: (0 if m.minute == 37 else 1, m.minute, m.match_id)
        )
        baseline_candidates.sort(key=lambda m: m.match_id)
        return candidates, baseline_candidates, minute_distribution, unparsed

    def scan_once(self) -> List[Dict[str, Any]]:
        self.run_count += 1
        logger.info("")
        logger.info("=" * 92)
        logger.info("FIRST HALF 37 SHADOW | SCAN #%d", self.run_count)
        logger.info("=" * 92)

        candidates, baseline_candidates, minute_distribution, unparsed = self.find_candidates()
        logger.info("Live minute distribution : %s", sorted(minute_distribution.items()))
        logger.info("Unparsed live matches : %d", unparsed)
        logger.info(
            "LIVE-STATS baseline candidates @%d' : %d",
            CORNER_BASELINE_MINUTE,
            len(baseline_candidates),
        )
        logger.info(
            "FH37 candidates : %d (exact37 + tracked late recovery through %d')",
            len(candidates),
            RECOVERY_MAX_MINUTE,
        )

        if baseline_candidates:
            workers = max(1, min(SCANNER_WORKERS, len(baseline_candidates)))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="fh37_baseline",
            ) as executor:
                futures = [
                    executor.submit(self.preload_corner_baseline, match)
                    for match in baseline_candidates
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.warning("FH37 baseline worker failed : %s", exc)

        scored_rows: List[Dict[str, Any]] = []

        if candidates:
            workers = max(1, min(SCANNER_WORKERS, len(candidates)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fh37_shadow") as executor:
                future_map = {
                    executor.submit(self.analyze_match, match): match
                    for match in candidates
                }
                for future in as_completed(future_map):
                    match = future_map[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        logger.exception(
                            "Worker failed | %s vs %s | %s",
                            match.home_team,
                            match.away_team,
                            exc,
                        )
                        row = None
                    if row is not None:
                        scored_rows.append(row)

        resolved = self._update_target_rows()

        logger.info(
            "SCAN SUMMARY | candidates=%d | newly_scored=%d | targets_resolved=%d | total_scored=%d",
            len(candidates),
            len(scored_rows),
            resolved,
            len(self.scored_ids),
        )
        logger.info("Predictions : %s", PREDICTIONS_PATH)
        logger.info("Audit       : %s", AUDIT_PATH)

        return scored_rows

    def run(self) -> None:
        logger.info("FIRST HALF 37 SHADOW SCANNER START")
        logger.info("Primary decision minute : 37' exact | snapshot hard-cut = <=37'")
        logger.info("LIVE-STATS baseline preload : %d'", CORNER_BASELINE_MINUTE)
        logger.info(
            "Late shadow recovery    : %s through %d'",
            LATE_RECOVERY_ENABLED,
            RECOVERY_MAX_MINUTE,
        )
        logger.info("Strict model gate : 5/5")
        logger.info("Mode : SHADOW ONLY — NO AUTOMATIC BETTING")

        while True:
            try:
                self.scan_once()
            except KeyboardInterrupt:
                logger.info("Scanner arrêté par utilisateur.")
                break
            except Exception as exc:
                logger.exception("Erreur globale scanner : %s", exc)

            logger.info("Prochain scan dans %ds...", SCAN_INTERVAL_SECONDS)
            time.sleep(SCAN_INTERVAL_SECONDS)


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_configuration() -> bool:
    valid = True

    if not STATS_API_KEY:
        logger.error("STATS_API_KEY manquante dans .env")
        valid = False

    for path in (FEATURE_MODULE_PATH, AUDIT_MODULE_PATH, MODEL_PATH, FREEZE_LOCK_PATH):
        if not path.exists():
            logger.error("Fichier requis absent : %s", path)
            valid = False

    if SCAN_INTERVAL_SECONDS < 1:
        logger.error("FH37_SCAN_INTERVAL_SECONDS doit être >= 1")
        valid = False

    if SCANNER_WORKERS < 1:
        logger.error("FH37_SCANNER_WORKERS doit être >= 1")
        valid = False

    if API_MAX_REQUESTS_PER_SECOND <= 0:
        logger.error("FH37_API_MAX_REQUESTS_PER_SECOND doit être > 0")
        valid = False

    if CORNER_BASELINE_MINUTE >= SNAPSHOT_MINUTE:
        logger.error("FH37_CORNER_BASELINE_MINUTE doit être < 37")
        valid = False

    if RECOVERY_MAX_MINUTE < SNAPSHOT_MINUTE:
        logger.error("FH37_RECOVERY_MAX_MINUTE doit être >= 37")
        valid = False

    logger.info("Configuration validation : %s", "OK" if valid else "FAIL")
    return valid


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="First Half 37 live shadow scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Effectue un seul scan puis quitte.",
    )
    parser.add_argument(
        "--no-target-resolve",
        action="store_true",
        help="N'essaie pas de résoudre automatiquement les targets.",
    )
    args = parser.parse_args()

    global TARGET_RESOLUTION_ENABLED
    if args.no_target_resolve:
        TARGET_RESOLUTION_ENABLED = False

    print()
    print("============================================================")
    print(" FIRST HALF 37 — DRAW_WITH_GOALS — LIVE SHADOW SCANNER")
    print("============================================================")
    print(" Snapshot : features hard-cut <=37'")
    print(" Primary  : 37' exact")
    print(f" Recovery : 38'-{RECOVERY_MAX_MINUTE}' shadow-only if source lagged")
    print(" LIVE API : /live-stats PRIMARY + /live-timeline + /shotmap fidelity")
    print(f" Baseline : cumulative xG/shots/corners captured at {CORNER_BASELINE_MINUTE}'")
    print(" Context  : draw_with_goals")
    print(" Gate     : 5/5 strict")
    print(" Mode     : SHADOW ONLY — NO BET")
    print("============================================================")
    print()

    if not validate_configuration():
        sys.exit(1)

    feature_module = load_module("fh37_shadow_features", FEATURE_MODULE_PATH)
    audit_module = load_module("fh37_shadow_audit", AUDIT_MODULE_PATH)

    model = FrozenFirstHalf37Model()
    model.load()

    client = StatsAPIClient(
        api_key=STATS_API_KEY,
        base_url=STATS_API_BASE_URL,
        timeout=STATS_API_TIMEOUT,
    )

    if not client.test_auth():
        sys.exit(1)

    scanner = FirstHalf37ShadowScanner(
        client=client,
        model=model,
        feature_module=feature_module,
        audit_module=audit_module,
    )

    if args.once:
        scanner.scan_once()
        return

    scanner.run()


if __name__ == "__main__":
    main()
