# ============================================================================
# LIVE SCANNER ENGINE - GOAL AFTER 87'
# ============================================================================
#
# Production live scanner:
#
#   LIVE MATCH
#       ↓
#   85' - 87'
#       ↓
#   PRE-MATCH FEATURES
#       ↓
#   LIVE-STATS
#       ↓
#   TIMELINE FALLBACK
#       ↓
#   EXACT 96-FEATURE VECTOR
#       ↓
#   INTERNAL STRICT VALIDATOR
#       ↓
#   EXTERNAL feature_validator.py
#       ↓
#   96/96 + COHERENT
#       ↓
#   PRODUCTION MODEL
#       ↓
#   P(goal_after_87)
#
# IMPORTANT:
# - No silent zero filling.
# - Missing data != zero.
# - Odds must come from OPENING prices.
# - Model prediction is blocked unless 96/96 and coherent.
# - No automatic betting.
# ============================================================================

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import subprocess
import re
import sys
import time
import threading
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================================
# PATHS / ENV
# ============================================================================

# Resolve the project directory first, then load THIS project's .env file.
# override=True is intentional: a stale STATS_API_BASE_URL / STATS_API_KEY
# exported in the shell must never override the values saved in .env.
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

OUTPUT_DIR = BASE_DIR / "outputs"
PRODUCTION_DIR = OUTPUT_DIR / "production"

MODEL_V3C_DIR = (
    OUTPUT_DIR /
    "model_v3_experiments"
)

DEFAULT_MODEL_PATH = (
    MODEL_V3C_DIR /
    "candidates_refit_all" /
    "V3_C_AGGRESSIVE_refit_all.joblib"
)

DEFAULT_MODEL_FEATURES_PATH = (
    MODEL_V3C_DIR /
    "schemas" /
    "V3_C_AGGRESSIVE_schema.json"
)

DEFAULT_MODEL_METADATA_PATH = (
    MODEL_V3C_DIR /
    "experiment_metadata.json"
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


# V3-C ONLY.
# IMPORTANT: old MODEL_PATH / MODEL_FEATURES_PATH values from .env are ignored.
# This prevents a stale V2 configuration from being loaded by accident.
MODEL_PATH = DEFAULT_MODEL_PATH
MODEL_FEATURES_PATH = DEFAULT_MODEL_FEATURES_PATH
MODEL_METADATA_PATH = DEFAULT_MODEL_METADATA_PATH

PREDICTIONS_PATH = (
    OUTPUT_DIR /
    "live_predictions_v3c.csv"
)

HISTORY_DIR = (
    OUTPUT_DIR /
    "history"
)

HISTORY_SNAPSHOTS_PATH = (
    HISTORY_DIR /
    "live_feature_snapshots_v3c.csv"
)

HISTORY_SCHEMA_VERSION = (
    "goal_after_87_history_v3c"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

STATS_API_KEY = os.getenv(
    "STATS_API_KEY",
    "",
).strip()

STATS_API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api",
).rstrip("/")

STATS_API_TIMEOUT = int(
    os.getenv(
        "STATS_API_TIMEOUT",
        "20",
    )
)

SCAN_INTERVAL_SECONDS = int(
    os.getenv(
        "SCAN_INTERVAL_SECONDS",
        "30",
    )
)

# Nombre maximum de matchs analysés simultanément.
# 4 est volontairement conservateur pour ne pas saturer TheStatsAPI.
SCANNER_WORKERS = int(
    os.getenv(
        "SCANNER_WORKERS",
        "4",
    )
)

# À 87', le scanner peut refaire immédiatement les sources LIVE si les
# données critiques restent incomplètes. Aucune attente entre les tentatives.
MAX_87_LIVE_RETRIES = int(
    os.getenv(
        "MAX_87_LIVE_RETRIES",
        "2",
    )
)

# Si le cache PRE-MATCH est très incomplet à 87', on le reconstruit une fois.
PRE_MATCH_REBUILD_MIN_COVERAGE = int(
    os.getenv(
        "PRE_MATCH_REBUILD_MIN_COVERAGE",
        "59",
    )
)

MIN_SCAN_MINUTE = int(
    os.getenv(
        "MIN_SCAN_MINUTE",
        "85",
    )
)

MAX_SCAN_MINUTE = int(
    os.getenv(
        "MAX_SCAN_MINUTE",
        "87",
    )
)

# Préchargement PRE-MATCH avant la fenêtre d'analyse.
# Les matchs à 83'-84' peuvent être préparés sans lancer de prédiction.
PRELOAD_MINUTE_START = int(
    os.getenv(
        "PRELOAD_MINUTE_START",
        "83",
    )
)

PRELOAD_MINUTE_END = int(
    os.getenv(
        "PRELOAD_MINUTE_END",
        "84",
    )
)

# Limiteur GLOBAL des appels TheStatsAPI.
# Toutes les threads passent par le même limiteur afin d'éviter les pics
# de requêtes qui provoquent des HTTP 429.
API_MAX_REQUESTS_PER_SECOND = float(
    os.getenv(
        "API_MAX_REQUESTS_PER_SECOND",
        "3.0",
    )
)

API_429_MAX_RETRIES = int(
    os.getenv(
        "API_429_MAX_RETRIES",
        "4",
    )
)

API_429_DEFAULT_BACKOFF_SECONDS = float(
    os.getenv(
        "API_429_DEFAULT_BACKOFF_SECONDS",
        "5.0",
    )
)

SIMULATED_ODDS = float(
    os.getenv(
        "SIMULATED_ODDS",
        "3.20",
    )
)

MIN_DISPLAY_PROBABILITY = float(
    os.getenv(
        "MIN_DISPLAY_PROBABILITY",
        "0.30",
    )
)


# V3-C PRIORITY ZONE.
#
# The temporal holdout showed the strongest volume/performance compromise in
# the 55%-65% region. Alerts therefore target ONLY this interval:
#
#       0.55 <= P(goal_after_87) < 0.65
#
# Predictions outside this interval are still saved for future auditing.
# Values are fixed here so stale V2 values in .env cannot change the rule.
PRIORITY_PROBABILITY_MIN = 0.55
PRIORITY_PROBABILITY_MAX = 0.65

# Backward-compatible field written to CSV.
ALERT_PROBABILITY = PRIORITY_PROBABILITY_MIN


MACOS_NOTIFICATIONS_ENABLED = (
    os.getenv(
        "MACOS_NOTIFICATIONS_ENABLED",
        "1",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)


TELEGRAM_NOTIFICATIONS_ENABLED = (
    os.getenv(
        "TELEGRAM_NOTIFICATIONS_ENABLED",
        "0",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    "",
).strip()

TELEGRAM_API_TIMEOUT = int(
    os.getenv(
        "TELEGRAM_API_TIMEOUT",
        "10",
    )
)


HISTORY_CAPTURE_ENABLED = (
    os.getenv(
        "HISTORY_CAPTURE_ENABLED",
        "1",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)


# V3-C production gate: EXACT 38/38.
# This deliberately ignores any stale MIN_MODEL_COVERAGE value from .env.
# Missing model feature => no prediction; no degraded imputation in production.
MIN_MODEL_COVERAGE = 38

LIVE_MATCHES_PER_PAGE = int(
    os.getenv(
        "LIVE_MATCHES_PER_PAGE",
        "100",
    )
)

LIVE_MATCHES_MAX_PAGES = int(
    os.getenv(
        "LIVE_MATCHES_MAX_PAGES",
        "20",
    )
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "live_scanner_engine"
)


# ============================================================================
# PRE-MATCH FEATURE ENGINEERING IMPORT
# ============================================================================

pre_match = None

_PRE_MATCH_MODULE_NAMES = [
    "features.pre_match_feature_engineering",
    "pre_match_feature_engineering",
    "features.pre_match",
    "pre_match",
]

for _module_name in _PRE_MATCH_MODULE_NAMES:

    try:

        pre_match = importlib.import_module(
            _module_name
        )

        logger.info(
            "Module PRE-MATCH chargé : %s",
            _module_name,
        )

        break

    except ImportError:
        continue

if pre_match is None:

    logger.warning(
        "Module PRE-MATCH non trouvé. "
        "Les features PRE-MATCH seront indisponibles."
    )


# ============================================================================
# EXACT PRODUCTION SCHEMA - 96 FEATURES
# ============================================================================

EXPECTED_FEATURES: List[str] = [

    # ------------------------------------------------------------------------
    # PRE-MATCH - STANDINGS / FORM / TEAM
    # ------------------------------------------------------------------------

    "home_team_rank",
    "away_team_rank",

    "home_team_points_standing",
    "away_team_points_standing",

    "home_team_matches_played",
    "away_team_matches_played",

    "home_team_wins",
    "away_team_wins",

    "home_team_draws",
    "away_team_draws",

    "home_team_losses",
    "away_team_losses",

    "home_team_goals_for",
    "away_team_goals_for",

    "home_team_goals_against",
    "away_team_goals_against",

    "home_team_goal_difference",
    "away_team_goal_difference",

    "home_position",
    "away_position",

    "home_points",
    "away_points",

    "home_form5",
    "away_form5",

    "home_form10",
    "away_form10",

    "home_scored_freq",
    "away_scored_freq",

    "home_wins_freq",
    "away_wins_freq",

    "home_avg_goals_scored",
    "away_avg_goals_scored",

    "home_avg_goals_conceded",
    "away_avg_goals_conceded",

    "home_avg_xg",
    "away_avg_xg",

    "home_btts_freq",
    "away_btts_freq",

    "home_over25_freq",
    "away_over25_freq",

    "h2h5_over25_freq",
    "h2h5_btts_freq",

    "league_home_over25_freq",
    "league_away_over25_freq",

    "league_home_over05_freq",
    "league_away_over05_freq",

    "league_home_btts_freq",
    "league_away_btts_freq",

    "league_2nd_half_over15_freq",

    # ------------------------------------------------------------------------
    # ODDS
    # ------------------------------------------------------------------------

    "odds_home",
    "odds_draw",
    "odds_away",

    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",

    "odds_under25",

    "odds_btts_yes",
    "odds_btts_no",

    # ------------------------------------------------------------------------
    # LIVE @ 87
    # ------------------------------------------------------------------------

    "score_87_home",
    "score_87_away",

    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",

    "home_shots_87",
    "away_shots_87",

    "home_shots_on_target_87",
    "away_shots_on_target_87",

    "home_shots_off_target_87",
    "away_shots_off_target_87",

    "home_shots_blocked_87",
    "away_shots_blocked_87",

    "home_corners_87",
    "away_corners_87",

    "home_fouls_87",
    "away_fouls_87",

    "home_yellow_cards_87",
    "away_yellow_cards_87",

    "home_red_cards_87",
    "away_red_cards_87",

    "home_offsides_87",
    "away_offsides_87",

    # ------------------------------------------------------------------------
    # LIVE 80 -> 87
    # ------------------------------------------------------------------------

    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",

    # ------------------------------------------------------------------------
    # TEMPORAL
    # ------------------------------------------------------------------------

    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",

    "minutes_since_last_goal",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "minutes_since_last_corner",
]


PRE_MATCH_FEATURES: List[str] = EXPECTED_FEATURES[:59]

LIVE_FEATURES: List[str] = EXPECTED_FEATURES[59:]


assert len(EXPECTED_FEATURES) == 96
assert len(PRE_MATCH_FEATURES) == 59
assert len(LIVE_FEATURES) == 37


# ============================================================================
# FEATURE GROUPS
# ============================================================================

FREQUENCY_FEATURES: Set[str] = {
    "home_form5",
    "away_form5",
    "home_form10",
    "away_form10",
    "home_scored_freq",
    "away_scored_freq",
    "home_wins_freq",
    "away_wins_freq",
    "home_btts_freq",
    "away_btts_freq",
    "home_over25_freq",
    "away_over25_freq",
    "h2h5_over25_freq",
    "h2h5_btts_freq",
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "league_2nd_half_over15_freq",
}


# Features obligatoires même en mode DEGRADED.
# Si l'une manque, la prédiction reste bloquée quelle que soit la couverture.
CRITICAL_MODEL_FEATURES: Set[str] = {
    "score_87_home",
    "score_87_away",
    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",

    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",

    "shots_80_87",
    "sot_80_87",

    "last_shot_minute",
    "last_shot_on_target_minute",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
}


ODDS_FEATURES: Set[str] = {
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
}


# Features qui peuvent légitimement être négatives.
# Les goal differences de classement sont signées par définition.
SIGNED_FEATURES: Set[str] = {
    "home_team_goal_difference",
    "away_team_goal_difference",
}

NON_NEGATIVE_FEATURES: Set[str] = {
    feature
    for feature in EXPECTED_FEATURES
    if (
        feature not in FREQUENCY_FEATURES
        and feature not in SIGNED_FEATURES
    )
}


# ============================================================================
# FEATURE SOURCE MAP
# ============================================================================

FEATURE_SOURCE_MAP: Dict[str, str] = {

    "home_team_rank":
        "PRE-MATCH → standings",

    "away_team_rank":
        "PRE-MATCH → standings",

    "home_team_points_standing":
        "PRE-MATCH → standings",

    "away_team_points_standing":
        "PRE-MATCH → standings",

    "home_team_matches_played":
        "PRE-MATCH → standings",

    "away_team_matches_played":
        "PRE-MATCH → standings",

    "home_team_wins":
        "PRE-MATCH → standings/team history",

    "away_team_wins":
        "PRE-MATCH → standings/team history",

    "home_team_draws":
        "PRE-MATCH → standings/team history",

    "away_team_draws":
        "PRE-MATCH → standings/team history",

    "home_team_losses":
        "PRE-MATCH → standings/team history",

    "away_team_losses":
        "PRE-MATCH → standings/team history",

    "home_team_goals_for":
        "PRE-MATCH → standings/team history",

    "away_team_goals_for":
        "PRE-MATCH → standings/team history",

    "home_team_goals_against":
        "PRE-MATCH → standings/team history",

    "away_team_goals_against":
        "PRE-MATCH → standings/team history",

    "home_team_goal_difference":
        "PRE-MATCH → standings/team history",

    "away_team_goal_difference":
        "PRE-MATCH → standings/team history",

    "home_position":
        "PRE-MATCH → standings",

    "away_position":
        "PRE-MATCH → standings",

    "home_points":
        "PRE-MATCH → standings",

    "away_points":
        "PRE-MATCH → standings",

    "home_form5":
        "PRE-MATCH → team history",

    "away_form5":
        "PRE-MATCH → team history",

    "home_form10":
        "PRE-MATCH → team history",

    "away_form10":
        "PRE-MATCH → team history",

    "home_scored_freq":
        "PRE-MATCH → team history",

    "away_scored_freq":
        "PRE-MATCH → team history",

    "home_wins_freq":
        "PRE-MATCH → team history",

    "away_wins_freq":
        "PRE-MATCH → team history",

    "home_avg_goals_scored":
        "PRE-MATCH → team history",

    "away_avg_goals_scored":
        "PRE-MATCH → team history",

    "home_avg_goals_conceded":
        "PRE-MATCH → team history",

    "away_avg_goals_conceded":
        "PRE-MATCH → team history",

    "home_avg_xg":
        "PRE-MATCH → team history → xG",

    "away_avg_xg":
        "PRE-MATCH → team history → xG",

    "home_btts_freq":
        "PRE-MATCH → team history",

    "away_btts_freq":
        "PRE-MATCH → team history",

    "home_over25_freq":
        "PRE-MATCH → team history",

    "away_over25_freq":
        "PRE-MATCH → team history",

    "h2h5_over25_freq":
        "PRE-MATCH → H2H",

    "h2h5_btts_freq":
        "PRE-MATCH → H2H",

    "league_home_over25_freq":
        "PRE-MATCH → league history",

    "league_away_over25_freq":
        "PRE-MATCH → league history",

    "league_home_over05_freq":
        "PRE-MATCH → league history",

    "league_away_over05_freq":
        "PRE-MATCH → league history",

    "league_home_btts_freq":
        "PRE-MATCH → league history",

    "league_away_btts_freq":
        "PRE-MATCH → league history",

    "league_2nd_half_over15_freq":
        "PRE-MATCH → league history",

    "odds_home":
        "PRE-MATCH → /football/matches/{id}/odds → match_odds.home.opening",

    "odds_draw":
        "PRE-MATCH → /football/matches/{id}/odds → match_odds.draw.opening",

    "odds_away":
        "PRE-MATCH → /football/matches/{id}/odds → match_odds.away.opening",

    "odds_over05":
        "PRE-MATCH → /football/matches/{id}/odds → total_goals.0.5.over.opening",

    "odds_over15":
        "PRE-MATCH → /football/matches/{id}/odds → total_goals.1.5.over.opening",

    "odds_over25":
        "PRE-MATCH → /football/matches/{id}/odds → total_goals.2.5.over.opening",

    "odds_over35":
        "PRE-MATCH → /football/matches/{id}/odds → total_goals.3.5.over.opening",

    "odds_under25":
        "PRE-MATCH → /football/matches/{id}/odds → total_goals.2.5.under.opening",

    "odds_btts_yes":
        "PRE-MATCH → /football/matches/{id}/odds → btts.yes.opening",

    "odds_btts_no":
        "PRE-MATCH → /football/matches/{id}/odds → btts.no.opening",

    "score_87_home":
        "LIVE → match score / live-stats",

    "score_87_away":
        "LIVE → match score / live-stats",

    "leading_team_87":
        "DERIVED → score_87_home / score_87_away",

    "goal_diff_87":
        "DERIVED → abs(score_87_home - score_87_away)",

    "is_draw_87":
        "DERIVED → score_87_home == score_87_away",

    "home_shots_87":
        "LIVE → live-stats total_shots / timeline",

    "away_shots_87":
        "LIVE → live-stats total_shots / timeline",

    "home_shots_on_target_87":
        "LIVE → live-stats shots_on_target / timeline",

    "away_shots_on_target_87":
        "LIVE → live-stats shots_on_target / timeline",

    "home_shots_off_target_87":
        "LIVE → live-stats shots_off_target / timeline",

    "away_shots_off_target_87":
        "LIVE → live-stats shots_off_target / timeline",

    "home_shots_blocked_87":
        "LIVE → live-stats shots_blocked / timeline",

    "away_shots_blocked_87":
        "LIVE → live-stats shots_blocked / timeline",

    "home_corners_87":
        "LIVE → live-stats corners / timeline",

    "away_corners_87":
        "LIVE → live-stats corners / timeline",

    "home_fouls_87":
        "LIVE → live-stats fouls / timeline",

    "away_fouls_87":
        "LIVE → live-stats fouls / timeline",

    "home_yellow_cards_87":
        "LIVE → live-stats yellow_cards / timeline",

    "away_yellow_cards_87":
        "LIVE → live-stats yellow_cards / timeline",

    "home_red_cards_87":
        "LIVE → live-stats red_cards / timeline",

    "away_red_cards_87":
        "LIVE → live-stats red_cards / timeline",

    "home_offsides_87":
        "LIVE → live-stats offsides / timeline",

    "away_offsides_87":
        "LIVE → live-stats offsides / timeline",

    "shots_80_87":
        "LIVE → timeline 80-87",

    "sot_80_87":
        "LIVE → timeline 80-87",

    "corners_80_87":
        "LIVE → timeline 80-87",

    "fouls_80_87":
        "LIVE → timeline 80-87",

    "cards_80_87":
        "LIVE → timeline 80-87",

    "subs_80_87":
        "LIVE → timeline 80-87",

    "last_goal_minute":
        "LIVE → timeline",

    "last_shot_minute":
        "LIVE → timeline",

    "last_shot_on_target_minute":
        "LIVE → timeline",

    "last_corner_minute":
        "LIVE → timeline",

    "minutes_since_last_goal":
        "DERIVED → current minute - last_goal_minute",

    "minutes_since_last_shot":
        "DERIVED → current minute - last_shot_minute",

    "minutes_since_last_shot_on_target":
        "DERIVED → current minute - last_shot_on_target_minute",

    "minutes_since_last_corner":
        "DERIVED → current minute - last_corner_minute",
}


# ============================================================================
# DATACLASSES
# ============================================================================

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

    raw: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Prediction:

    match_id: str

    home_team: str
    away_team: str

    minute: int

    score_home: int
    score_away: int

    probability: float

    fair_odds: float

    simulated_odds: float

    edge: float

    zone: str

    decision: str

    validation_status: str

    coverage: int

    feature_count: int = 38

    prediction_quality: str = "FULL_V3C"

    missing_features: List[str] = field(
        default_factory=list
    )


    alert: bool = False

    alert_threshold: float = 0.55


@dataclass
class ValidationResult:

    status: str

    ready_for_model: bool

    errors: List[str]

    warnings: List[str]

    missing: List[str]

    coverage: int


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def first_not_none(
    *values: Any,
) -> Any:

    for value in values:

        if value is not None:
            return value

    return None


def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return float(value)

    try:

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (
        TypeError,
        ValueError,
    ):

        return None


def safe_int(
    value: Any,
) -> Optional[int]:

    if value is None:
        return None

    try:

        if isinstance(
            value,
            bool,
        ):
            return int(value)

        number = float(value)

        if not math.isfinite(number):
            return None

        return int(number)

    except (
        TypeError,
        ValueError,
    ):

        return None


def is_missing(
    value: Any,
) -> bool:

    if value is None:
        return True

    if isinstance(
        value,
        float,
    ):
        return math.isnan(value)

    try:

        result = pd.isna(value)

        if isinstance(
            result,
            (bool, np.bool_),
        ):
            return bool(result)

    except Exception:
        pass

    return False


def clean_string(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def mapping_get_nested(
    data: Any,
    *keys: str,
) -> Any:

    current = data

    for key in keys:

        if not isinstance(
            current,
            Mapping,
        ):
            return None

        current = current.get(key)

    return current


def as_mapping(
    value: Any,
) -> Mapping[str, Any]:

    if isinstance(
        value,
        Mapping,
    ):
        return value

    return {}


# ============================================================================
# MATCH CLOCK / SCORE PARSING
# ============================================================================

def parse_match_minute(
    value: Any,
) -> Optional[int]:

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):

        number = safe_int(value)

        if number is not None:
            return max(0, number)

        return None

    text = str(value).strip()

    if not text:
        return None

    # Examples:
    #   85
    #   85'
    #   85+
    #   85+2
    #   85:32
    match = re.match(
        r"^\s*(\d+)",
        text,
    )

    if match:

        try:
            return int(
                match.group(1)
            )
        except ValueError:
            return None

    return None


def extract_elapsed_minute(
    raw: Mapping[str, Any],
) -> Optional[int]:

    candidates: List[Any] = []

    candidates.extend([
        raw.get("elapsed_minutes"),
        raw.get("elapsed_minute"),
        raw.get("minute"),
        raw.get("minutes"),
        raw.get("elapsed"),
        raw.get("match_minute"),
        raw.get("current_minute"),
    ])

    for container_key in (
        "meta",
        "status",
        "time",
        "clock",
        "live",
        "match_status",
    ):

        container = raw.get(
            container_key
        )

        if isinstance(
            container,
            Mapping,
        ):

            candidates.extend([
                container.get(
                    "elapsed_minutes"
                ),
                container.get(
                    "elapsed_minute"
                ),
                container.get(
                    "minute"
                ),
                container.get(
                    "minutes"
                ),
                container.get(
                    "elapsed"
                ),
                container.get(
                    "current_minute"
                ),
            ])

    data = raw.get("data")

    if isinstance(
        data,
        Mapping,
    ):

        candidates.extend([
            data.get(
                "elapsed_minutes"
            ),
            data.get(
                "elapsed_minute"
            ),
            data.get(
                "minute"
            ),
            data.get(
                "minutes"
            ),
            data.get(
                "elapsed"
            ),
        ])

        data_meta = data.get(
            "meta"
        )

        if isinstance(
            data_meta,
            Mapping,
        ):

            candidates.extend([
                data_meta.get(
                    "elapsed_minutes"
                ),
                data_meta.get(
                    "elapsed_minute"
                ),
                data_meta.get(
                    "minute"
                ),
                data_meta.get(
                    "minutes"
                ),
                data_meta.get(
                    "elapsed"
                ),
            ])

    for candidate in candidates:

        minute = parse_match_minute(
            candidate
        )

        if minute is not None:
            return minute

    return None


def parse_score_value(
    value: Any,
) -> Optional[int]:

    if value is None:
        return None

    if isinstance(
        value,
        Mapping,
    ):

        value = first_not_none(
            value.get("score"),
            value.get("goals"),
            value.get("value"),
            value.get("current"),
            value.get("total"),
        )

    return safe_int(value)


def extract_match_score(
    raw: Mapping[str, Any],
) -> Tuple[
    Optional[int],
    Optional[int],
]:

    home_score = None
    away_score = None

    # Direct
    home_score = parse_score_value(
        first_not_none(
            raw.get("home_goals"),
            raw.get("home_score"),
            raw.get("score_home"),
            raw.get("home_score_current"),
        )
    )

    away_score = parse_score_value(
        first_not_none(
            raw.get("away_goals"),
            raw.get("away_score"),
            raw.get("score_away"),
            raw.get("away_score_current"),
        )
    )

    # TheStatsAPI commonly exposes the current score inside home_team.score
    # and away_team.score on match objects.
    home_team_obj = raw.get("home_team")
    away_team_obj = raw.get("away_team")

    if home_score is None and isinstance(home_team_obj, Mapping):
        home_score = parse_score_value(
            first_not_none(
                home_team_obj.get("score"),
                home_team_obj.get("goals"),
                home_team_obj.get("current_score"),
            )
        )

    if away_score is None and isinstance(away_team_obj, Mapping):
        away_score = parse_score_value(
            first_not_none(
                away_team_obj.get("score"),
                away_team_obj.get("goals"),
                away_team_obj.get("current_score"),
            )
        )

    # Nested meta
    for container_key in (
        "meta",
        "score",
        "scores",
        "status",
    ):

        container = raw.get(
            container_key
        )

        if not isinstance(
            container,
            Mapping,
        ):
            continue

        if home_score is None:

            home_score = parse_score_value(
                first_not_none(
                    container.get(
                        "home_goals"
                    ),
                    container.get(
                        "home_score"
                    ),
                    container.get(
                        "home"
                    ),
                )
            )

        if away_score is None:

            away_score = parse_score_value(
                first_not_none(
                    container.get(
                        "away_goals"
                    ),
                    container.get(
                        "away_score"
                    ),
                    container.get(
                        "away"
                    ),
                )
            )

    # data.meta
    data = raw.get("data")

    if isinstance(
        data,
        Mapping,
    ):

        nested_home, nested_away = (
            extract_match_score(
                data
            )
        )

        if home_score is None:
            home_score = nested_home

        if away_score is None:
            away_score = nested_away

    # Score strings
    score_strings = []

    for key in (
        "score",
        "ft_score",
        "current_score",
        "live_score",
        "ht_score",
    ):

        value = raw.get(key)

        if value is not None:
            score_strings.append(
                str(value)
            )

    for score_text in score_strings:

        match = re.search(
            r"(\d+)\s*[-:]\s*(\d+)",
            score_text,
        )

        if match:

            if home_score is None:
                home_score = int(
                    match.group(1)
                )

            if away_score is None:
                away_score = int(
                    match.group(2)
                )

            break

    return (
        home_score,
        away_score,
    )


def extract_team_id(
    raw: Mapping[str, Any],
    side: str,
) -> Optional[str]:

    team = raw.get(
        f"{side}_team"
    )

    if isinstance(
        team,
        Mapping,
    ):

        value = first_not_none(
            team.get("id"),
            team.get("team_id"),
        )

        if value is not None:
            return str(value)

    value = first_not_none(
        raw.get(
            f"{side}_team_id"
        ),
        raw.get(
            f"{side}_id"
        ),
    )

    if value is not None:
        return str(value)

    return None


# ============================================================================
# PARSE LIVE MATCH
# ============================================================================

def parse_live_match(
    raw: Any,
) -> Optional[LiveMatch]:

    if not isinstance(
        raw,
        Mapping,
    ):
        return None

    match_id = first_not_none(
        raw.get("match_id"),
        raw.get("id"),
        raw.get("event_id"),
    )

    if match_id is None:

        data = raw.get("data")

        if isinstance(
            data,
            Mapping,
        ):

            match_id = first_not_none(
                data.get("match_id"),
                data.get("id"),
            )

    if match_id is None:
        return None

    match_id = str(
        match_id
    )

    minute = extract_elapsed_minute(
        raw
    )

    if minute is None:

        logger.debug(
            "LIVE match sans minute exploitable | id=%s",
            match_id,
        )

        return None

    home_team = None
    away_team = None

    for container in (
        raw,
        raw.get("data")
        if isinstance(
            raw.get("data"),
            Mapping,
        )
        else {},
    ):

        if not isinstance(
            container,
            Mapping,
        ):
            continue

        home = container.get(
            "home_team"
        )

        away = container.get(
            "away_team"
        )

        if isinstance(
            home,
            Mapping,
        ):

            home_team = first_not_none(
                home_team,
                home.get("name"),
                home.get("team_name"),
            )

        elif home is not None:

            home_team = first_not_none(
                home_team,
                clean_string(home),
            )

        if isinstance(
            away,
            Mapping,
        ):

            away_team = first_not_none(
                away_team,
                away.get("name"),
                away.get("team_name"),
            )

        elif away is not None:

            away_team = first_not_none(
                away_team,
                clean_string(away),
            )

    home_team = (
        home_team
        or clean_string(
            raw.get("home_name")
        )
        or "Home"
    )

    away_team = (
        away_team
        or clean_string(
            raw.get("away_name")
        )
        or "Away"
    )

    home_score, away_score = (
        extract_match_score(
            raw
        )
    )

    competition_id = first_not_none(
        raw.get("competition_id"),
        raw.get("competition"),
        raw.get("league_id"),
    )

    season_id = first_not_none(
        raw.get("season_id"),
        raw.get("season"),
    )

    if isinstance(
        competition_id,
        Mapping,
    ):

        competition_id = first_not_none(
            competition_id.get("id"),
            competition_id.get("competition_id"),
        )

    if isinstance(
        season_id,
        Mapping,
    ):

        season_id = first_not_none(
            season_id.get("id"),
            season_id.get("season_id"),
        )

    return LiveMatch(
        match_id=match_id,
        home_team=str(home_team),
        away_team=str(away_team),
        minute=minute,
        home_score=home_score,
        away_score=away_score,
        competition_id=(
            str(competition_id)
            if competition_id is not None
            else None
        ),
        season_id=(
            str(season_id)
            if season_id is not None
            else None
        ),
        raw=dict(raw),
    )


# ============================================================================
# GLOBAL API RATE LIMITER
# ============================================================================

class GlobalRateLimiter:
    """
    Thread-safe global limiter shared by every TheStatsAPI request.

    It spaces request starts over time instead of allowing all scanner workers
    to hit the API at the same instant.
    """

    def __init__(self, max_requests_per_second: float):
        self.max_requests_per_second = max(0.0, float(max_requests_per_second))
        self.min_interval = (
            1.0 / self.max_requests_per_second
            if self.max_requests_per_second > 0
            else 0.0
        )
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed_at - now)

            if wait_seconds > 0:
                time.sleep(wait_seconds)

            started_at = time.monotonic()
            self._next_allowed_at = started_at + self.min_interval


# ============================================================================
# STATS API CLIENT
# ============================================================================

class StatsAPIClient:

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 20,
    ):

        normalized_api_key = str(api_key or "").strip()

        # TheStatsAPI expects Bearer authentication. Accept both a raw key
        # and an accidentally prefixed "Bearer ..." value in .env so we
        # never send "Bearer Bearer ...".
        if normalized_api_key.lower().startswith("bearer "):
            normalized_api_key = normalized_api_key[7:].strip()

        self.api_key = normalized_api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Une Session requests par thread : le scanner peut analyser plusieurs
        # matchs en parallèle sans partager un objet Session mutable.
        self._thread_local = threading.local()

        # IMPORTANT: le limiteur est partagé par toutes les threads via ce
        # client unique. Les workers restent parallèles, mais les départs de
        # requêtes API sont contrôlés globalement.
        self._rate_limiter = GlobalRateLimiter(
            API_MAX_REQUESTS_PER_SECOND
        )

    def _get_session(self) -> requests.Session:

        session = getattr(
            self._thread_local,
            "session",
            None,
        )

        if session is None:
            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            })
            self._thread_local.session = session

        return session

    # ----------------------------------------------------------------------

    def _retry_after_seconds(
        self,
        response: requests.Response,
        attempt: int,
    ) -> float:

        raw = str(
            response.headers.get("Retry-After", "")
            or ""
        ).strip()

        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(raw)

                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(
                            tzinfo=timezone.utc
                        )

                    now = datetime.now(
                        timezone.utc
                    )

                    return max(
                        0.0,
                        (retry_at - now).total_seconds(),
                    )

                except Exception:
                    pass

        # Backoff exponentiel si l'API ne fournit pas Retry-After.
        return min(
            60.0,
            API_429_DEFAULT_BACKOFF_SECONDS
            * (2 ** max(0, attempt - 1)),
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:

        url = (
            f"{self.base_url}/"
            f"{path.lstrip('/')}"
        )

        max_attempts = max(
            1,
            API_429_MAX_RETRIES + 1,
        )

        last_response = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            # Toutes les threads passent ici avant de toucher TheStatsAPI.
            self._rate_limiter.wait()

            response = self._get_session().request(
                method=method,
                url=url,
                params=dict(params or {}),
                timeout=self.timeout,
            )

            last_response = response

            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            if attempt >= max_attempts:
                logger.error(
                    "RATE LIMIT 429 | abandon après %d tentative(s) | %s",
                    attempt,
                    path,
                )
                response.raise_for_status()

            wait_seconds = self._retry_after_seconds(
                response,
                attempt,
            )

            logger.warning(
                "RATE LIMIT 429 | tentative=%d/%d | path=%s | attente=%.1fs",
                attempt,
                max_attempts,
                path,
                wait_seconds,
            )

            time.sleep(
                wait_seconds
            )

        # Sécurité théorique : le flux ne devrait jamais arriver ici.
        if last_response is not None:
            last_response.raise_for_status()

        raise requests.HTTPError(
            f"Request failed without response: {path}"
        )

    # ----------------------------------------------------------------------

    def get(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Compatibility wrapper used by pre_match_feature_engineering.py.

        The PRE-MATCH builder expects a client exposing ``client.get(...)``.
        The scanner historically exposed only ``_request(...)``.  Keep the
        scanner's authenticated session and fail softly here so that an
        unavailable auxiliary endpoint produces missing features rather than
        aborting the whole PRE-MATCH build.
        """

        try:
            return self._request(
                "GET",
                path,
                params,
            )

        except requests.RequestException as exc:
            logger.warning(
                "TheStatsAPI GET failed | path=%s | error=%s",
                path,
                exc,
            )
            return None

        except Exception as exc:
            logger.warning(
                "TheStatsAPI GET unexpected failure | path=%s | error=%s",
                path,
                exc,
            )
            return None

    # ----------------------------------------------------------------------

    def test_auth(self) -> bool:

        try:

            self._request(
                "GET",
                "/football/matches",
                {
                    "status": "live",
                    "page": 1,
                    "per_page": 1,
                },
            )

            logger.info(
                "TheStatsAPI authentication : OK"
            )

            return True

        except Exception as exc:

            logger.error(
                "TheStatsAPI authentication FAILED : %s",
                exc,
            )

            return False

    # ----------------------------------------------------------------------

    def get_live_matches(self) -> List[Dict[str, Any]]:

        all_matches: List[
            Dict[str, Any]
        ] = []

        seen_ids: Set[str] = set()

        total_announced = None
        total_pages_announced = None

        for page in range(
            1,
            LIVE_MATCHES_MAX_PAGES + 1,
        ):

            payload = self._request(
                "GET",
                "/football/matches",
                {
                    "status": "live",
                    "page": page,
                    "per_page": LIVE_MATCHES_PER_PAGE,
                },
            )

            if isinstance(
                payload,
                Mapping,
            ):

                data = payload.get(
                    "data",
                    []
                )

                meta = payload.get(
                    "meta",
                    {}
                )

            else:

                data = []
                meta = {}

            if not isinstance(
                data,
                list,
            ):
                data = []

            if isinstance(
                meta,
                Mapping,
            ):

                total_announced = first_not_none(
                    meta.get("total"),
                    total_announced,
                )

                total_pages_announced = first_not_none(
                    meta.get("total_pages"),
                    total_pages_announced,
                )

                current_page = safe_int(
                    meta.get("page")
                ) or page

                total_pages = (
                    safe_int(
                        meta.get(
                            "total_pages"
                        )
                    )
                    or 1
                )

                per_page = (
                    safe_int(
                        meta.get(
                            "per_page"
                        )
                    )
                    or LIVE_MATCHES_PER_PAGE
                )

            else:

                current_page = page
                total_pages = 1
                per_page = (
                    LIVE_MATCHES_PER_PAGE
                )

            new_count = 0

            for match in data:

                if not isinstance(
                    match,
                    Mapping,
                ):
                    continue

                match_id = first_not_none(
                    match.get("match_id"),
                    match.get("id"),
                )

                if match_id is None:
                    continue

                match_id = str(
                    match_id
                )

                if match_id in seen_ids:
                    continue

                seen_ids.add(
                    match_id
                )

                all_matches.append(
                    dict(match)
                )

                new_count += 1

            logger.info(
                "TheStatsAPI LIVE pagination: "
                "page=%d/%d | reçus=%d | total=%s | per_page=%d",
                current_page,
                total_pages,
                len(data),
                (
                    total_announced
                    if total_announced is not None
                    else "?"
                ),
                per_page,
            )

            logger.info(
                "Page LIVE %d: %d nouveaux matchs "
                "après déduplication.",
                current_page,
                new_count,
            )

            if current_page >= total_pages:
                break

            if len(data) == 0:
                break

        logger.info(
            "TOTAL LIVE récupéré après pagination : %d",
            len(all_matches),
        )

        logger.info(
            "TOTAL LIVE annoncé par TheStatsAPI : %s",
            (
                total_announced
                if total_announced is not None
                else "?"
            ),
        )

        logger.info(
            "Pages LIVE annoncées par TheStatsAPI : %s",
            (
                total_pages_announced
                if total_pages_announced is not None
                else "?"
            ),
        )

        return all_matches

    # ----------------------------------------------------------------------

    def get_match(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}",
        )

    # ----------------------------------------------------------------------

    def get_live_stats(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/live-stats",
        )

    # ----------------------------------------------------------------------

    def get_stats(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/stats",
        )

    # ----------------------------------------------------------------------

    def get_timeline(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/timeline",
        )

    # ----------------------------------------------------------------------

    def get_shotmap(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/shotmap",
        )

    # ----------------------------------------------------------------------

    def get_odds(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/odds",
        )

    # ----------------------------------------------------------------------

    def get_live_odds(
        self,
        match_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/matches/{match_id}/odds/live",
        )

    # ----------------------------------------------------------------------

    def get_team_stats(
        self,
        team_id: str,
    ) -> Any:

        return self._request(
            "GET",
            f"/football/teams/{team_id}/stats",
        )


# ============================================================================
# LIVE-STATS EXTRACTION
# ============================================================================

STAT_ALIASES: Dict[str, List[str]] = {

    "shots":
        [
            "total_shots",
            "shots",
            "total_attempts",
        ],

    "shots_on_target":
        [
            "shots_on_target",
            "on_target",
            "shots_on_goal",
        ],

    "shots_off_target":
        [
            "shots_off_target",
            "off_target",
            "shots_off_goal",
        ],

    "shots_blocked":
        [
            "shots_blocked",
            "blocked_shots",
        ],

    "corners":
        [
            "corners",
            "corner_kicks",
        ],

    "fouls":
        [
            "fouls",
        ],

    "yellow_cards":
        [
            "yellow_cards",
            "yellow",
            "yellow_card",
        ],

    "red_cards":
        [
            "red_cards",
            "red",
            "red_card",
        ],

    "offsides":
        [
            "offsides",
            "offside",
        ],
}


def extract_pair(
    value: Any,
) -> Tuple[
    Optional[float],
    Optional[float],
]:

    if not isinstance(
        value,
        Mapping,
    ):
        return None, None

    home = first_not_none(
        value.get("home"),
        value.get("Home"),
        value.get("1"),
    )

    away = first_not_none(
        value.get("away"),
        value.get("Away"),
        value.get("2"),
    )

    return (
        safe_float(home),
        safe_float(away),
    )


def extract_stat_pair(
    stats: Mapping[str, Any],
    aliases: Sequence[str],
) -> Tuple[
    Optional[float],
    Optional[float],
]:

    for alias in aliases:

        value = stats.get(
            alias
        )

        if value is None:
            continue

        # Typical TheStatsAPI:
        #
        # {
        #   "all": {
        #       "home": 10,
        #       "away": 5
        #   }
        # }
        if isinstance(
            value,
            Mapping,
        ):

            all_value = value.get(
                "all"
            )

            if isinstance(
                all_value,
                Mapping,
            ):

                home, away = (
                    extract_pair(
                        all_value
                    )
                )

                if (
                    home is not None
                    or away is not None
                ):
                    return home, away

            home, away = (
                extract_pair(
                    value
                )
            )

            if (
                home is not None
                or away is not None
            ):
                return home, away

    return None, None


def extract_live_stats(
    payload: Any,
) -> Dict[str, Any]:

    features: Dict[str, Any] = {}

    if not isinstance(
        payload,
        Mapping,
    ):
        return features

    data = payload.get(
        "data",
        payload,
    )

    if not isinstance(
        data,
        Mapping,
    ):
        return features

    stats = data.get(
        "stats",
        data,
    )

    if not isinstance(
        stats,
        Mapping,
    ):
        return features

    feature_pairs = {

        "shots":
            (
                "home_shots_87",
                "away_shots_87",
            ),

        "shots_on_target":
            (
                "home_shots_on_target_87",
                "away_shots_on_target_87",
            ),

        "shots_off_target":
            (
                "home_shots_off_target_87",
                "away_shots_off_target_87",
            ),

        "shots_blocked":
            (
                "home_shots_blocked_87",
                "away_shots_blocked_87",
            ),

        "corners":
            (
                "home_corners_87",
                "away_corners_87",
            ),

        "fouls":
            (
                "home_fouls_87",
                "away_fouls_87",
            ),

        "yellow_cards":
            (
                "home_yellow_cards_87",
                "away_yellow_cards_87",
            ),

        "red_cards":
            (
                "home_red_cards_87",
                "away_red_cards_87",
            ),

        "offsides":
            (
                "home_offsides_87",
                "away_offsides_87",
            ),
    }

    for stat_name, feature_names in (
        feature_pairs.items()
    ):

        home, away = extract_stat_pair(
            stats,
            STAT_ALIASES[
                stat_name
            ],
        )

        if home is not None:

            features[
                feature_names[0]
            ] = home

        if away is not None:

            features[
                feature_names[1]
            ] = away

    return features


def extract_match_stats_features(
    payload: Any,
) -> Dict[str, Any]:
    """
    Extrait les features LIVE depuis /football/matches/{id}/stats.

    Cet endpoint expose notamment des objets data.home et data.away.
    Aucune absence n'est convertie en zéro.
    """

    features: Dict[str, Any] = {}

    if not isinstance(payload, Mapping):
        return features

    data = payload.get("data", payload)

    if not isinstance(data, Mapping):
        return features

    home = data.get("home")
    away = data.get("away")

    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        return features

    aliases = {
        "shots": (
            "home_shots_87",
            "away_shots_87",
            ("shots", "total_shots", "total_attempts"),
        ),
        "shots_on_target": (
            "home_shots_on_target_87",
            "away_shots_on_target_87",
            ("shots_on_target", "on_target", "shots_on_goal"),
        ),
        "shots_off_target": (
            "home_shots_off_target_87",
            "away_shots_off_target_87",
            ("shots_off_target", "off_target", "shots_off_goal"),
        ),
        "shots_blocked": (
            "home_shots_blocked_87",
            "away_shots_blocked_87",
            ("shots_blocked", "blocked_shots"),
        ),
        "corners": (
            "home_corners_87",
            "away_corners_87",
            ("corners", "corner_kicks"),
        ),
        "fouls": (
            "home_fouls_87",
            "away_fouls_87",
            ("fouls",),
        ),
        "yellow_cards": (
            "home_yellow_cards_87",
            "away_yellow_cards_87",
            ("yellow_cards", "yellow", "yellow_card"),
        ),
        "red_cards": (
            "home_red_cards_87",
            "away_red_cards_87",
            ("red_cards", "red", "red_card"),
        ),
        "offsides": (
            "home_offsides_87",
            "away_offsides_87",
            ("offsides", "offside"),
        ),
    }

    def side_value(
        side: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Optional[float]:
        for key in keys:
            value = side.get(key)
            parsed = safe_float(value)
            if parsed is not None:
                return parsed
        return None

    for _, (home_feature, away_feature, keys) in aliases.items():
        home_value = side_value(home, keys)
        away_value = side_value(away, keys)

        if home_value is not None:
            features[home_feature] = home_value

        if away_value is not None:
            features[away_feature] = away_value

    return features


def derive_shots_off_target(
    features: Dict[str, Any],
) -> None:
    """
    Dérive shots_off_target uniquement lorsque l'identité comptable
    total = on_target + off_target + blocked est entièrement déterminée.
    """

    for side in ("home", "away"):
        destination = f"{side}_shots_off_target_87"

        if features.get(destination) is not None:
            continue

        total = safe_float(features.get(f"{side}_shots_87"))
        on_target = safe_float(features.get(f"{side}_shots_on_target_87"))
        blocked = safe_float(features.get(f"{side}_shots_blocked_87"))

        if total is None or on_target is None or blocked is None:
            continue

        derived = total - on_target - blocked

        if derived < 0:
            continue

        features[destination] = derived


# ============================================================================
# TIMELINE
# ============================================================================

def extract_events(
    payload: Any,
) -> List[Mapping[str, Any]]:

    if payload is None:
        return []

    if isinstance(
        payload,
        list,
    ):

        return [
            item
            for item in payload
            if isinstance(
                item,
                Mapping,
            )
        ]

    if not isinstance(
        payload,
        Mapping,
    ):
        return []

    candidates = [
        payload.get("events"),
        payload.get("timeline"),
    ]

    data = payload.get(
        "data"
    )

    if isinstance(
        data,
        Mapping,
    ):

        candidates.extend([
            data.get("events"),
            data.get("timeline"),
        ])

    for candidate in candidates:

        if isinstance(
            candidate,
            list,
        ):

            return [
                item
                for item in candidate
                if isinstance(
                    item,
                    Mapping,
                )
            ]

    return []


def extract_event_minute(
    event: Mapping[str, Any],
) -> Optional[int]:

    candidates = [
        event.get("minute"),
        event.get("minutes"),
        event.get("elapsed_minutes"),
        event.get("elapsed"),
        event.get("time"),
    ]

    time_data = event.get(
        "time"
    )

    if isinstance(
        time_data,
        Mapping,
    ):

        candidates.extend([
            time_data.get("minute"),
            time_data.get("minutes"),
            time_data.get("elapsed"),
            time_data.get("elapsed_minutes"),
        ])

    for candidate in candidates:

        if isinstance(
            candidate,
            Mapping,
        ):

            candidate = first_not_none(
                candidate.get("minute"),
                candidate.get("minutes"),
                candidate.get("elapsed"),
            )

        minute = parse_match_minute(
            candidate
        )

        if minute is not None:
            return minute

    return None


def extract_event_type(
    event: Mapping[str, Any],
) -> str:

    values = [
        event.get("type"),
        event.get("event_type"),
        event.get("incident_type"),
        event.get("kind"),
        event.get("name"),
        event.get("incident"),
    ]

    for value in values:

        if isinstance(
            value,
            Mapping,
        ):

            value = first_not_none(
                value.get("type"),
                value.get("name"),
                value.get("value"),
            )

        if value is not None:

            return str(
                value
            ).strip().lower()

    return ""


def extract_event_team(
    event: Mapping[str, Any],
) -> Tuple[
    Optional[str],
    Optional[str],
]:

    team_id = None
    team_name = None

    candidates = [
        event.get("team"),
        event.get("team_id"),
        event.get("team_name"),
        event.get("side"),
    ]

    for candidate in candidates:

        if isinstance(
            candidate,
            Mapping,
        ):

            team_id = first_not_none(
                team_id,
                candidate.get("id"),
                candidate.get("team_id"),
            )

            team_name = first_not_none(
                team_name,
                candidate.get("name"),
                candidate.get("team_name"),
            )

        elif candidate is not None:

            text = str(
                candidate
            ).strip()

            if text:

                if (
                    team_id is None
                    and (
                        text.startswith("tm_")
                        or text.isdigit()
                    )
                ):
                    team_id = text

                if team_name is None:
                    team_name = text

    return (
        (
            str(team_id)
            if team_id is not None
            else None
        ),
        team_name,
    )


def event_belongs_to_home(
    event: Mapping[str, Any],
    home_team_id: Optional[str],
    away_team_id: Optional[str],
    home_team_name: Optional[str],
    away_team_name: Optional[str],
) -> Optional[bool]:

    team_id, team_name = (
        extract_event_team(
            event
        )
    )

    if team_id is not None:

        if (
            home_team_id is not None
            and str(team_id) == str(home_team_id)
        ):
            return True

        if (
            away_team_id is not None
            and str(team_id) == str(away_team_id)
        ):
            return False

    if team_name is not None:

        normalized = team_name.lower()

        if (
            home_team_name
            and normalized
            == home_team_name.lower()
        ):
            return True

        if (
            away_team_name
            and normalized
            == away_team_name.lower()
        ):
            return False

    # Some APIs expose "home"/"away" directly.
    for key in (
        "side",
        "team_side",
        "participant",
    ):

        value = event.get(
            key
        )

        if value is None:
            continue

        if isinstance(
            value,
            Mapping,
        ):

            value = first_not_none(
                value.get("side"),
                value.get("name"),
                value.get("type"),
            )

        if value is None:
            continue

        text = str(
            value
        ).lower()

        if text in (
            "home",
            "h",
            "1",
        ):
            return True

        if text in (
            "away",
            "a",
            "2",
        ):
            return False

    return None


def normalize_event_type(
    event_type: str,
) -> str:

    text = event_type.lower()

    text = text.replace(
        "-",
        "_",
    )

    text = text.replace(
        " ",
        "_",
    )

    return text


def extract_red_cards_from_match_detail(
    payload: Any,
) -> Dict[str, Optional[float]]:
    """
    Cherche uniquement des compteurs EXPLICITES de cartons rouges
    dans le détail du match. Une absence n'est jamais convertie en zéro.
    """

    result = {
        "home_red_cards_87": None,
        "away_red_cards_87": None,
    }

    if not isinstance(payload, Mapping):
        return result

    data = payload.get("data")

    if isinstance(data, Mapping):
        nested_match = data.get("match")
        root = (
            nested_match
            if isinstance(nested_match, Mapping)
            else data
        )
    else:
        root = payload

    home_candidates = [
        mapping_get_nested(root, "home", "red_cards"),
        mapping_get_nested(root, "home_team", "red_cards"),
        mapping_get_nested(root, "home", "stats", "red_cards"),
        mapping_get_nested(root, "home_team", "stats", "red_cards"),
        mapping_get_nested(root, "stats", "red_cards", "home"),
        mapping_get_nested(root, "red_cards", "home"),
        root.get("home_red_cards")
        if isinstance(root, Mapping)
        else None,
    ]

    away_candidates = [
        mapping_get_nested(root, "away", "red_cards"),
        mapping_get_nested(root, "away_team", "red_cards"),
        mapping_get_nested(root, "away", "stats", "red_cards"),
        mapping_get_nested(root, "away_team", "stats", "red_cards"),
        mapping_get_nested(root, "stats", "red_cards", "away"),
        mapping_get_nested(root, "red_cards", "away"),
        root.get("away_red_cards")
        if isinstance(root, Mapping)
        else None,
    ]

    for value in home_candidates:
        parsed = safe_float(value)
        if parsed is not None:
            result["home_red_cards_87"] = parsed
            break

    for value in away_candidates:
        parsed = safe_float(value)
        if parsed is not None:
            result["away_red_cards_87"] = parsed
            break

    return result


def timeline_card_debug(
    events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Résumé compact des événements potentiellement liés aux cartes.
    """

    rows: List[Dict[str, Any]] = []

    for event in events:

        if not isinstance(event, Mapping):
            continue

        event_type_raw = extract_event_type(event)
        event_type = normalize_event_type(
            event_type_raw
        )

        raw_text = json.dumps(
            event,
            ensure_ascii=False,
            default=str,
        ).lower()

        if not (
            "card" in event_type
            or "yellow" in event_type
            or "red" in event_type
            or "card" in raw_text
            or "yellow" in raw_text
            or "red" in raw_text
        ):
            continue

        rows.append({
            "minute": extract_event_minute(event),
            "type": event_type_raw,
            "team": extract_event_team(event),
            "raw": event,
        })

    return rows


def analyze_timeline(
    events: Sequence[Mapping[str, Any]],
    current_minute: int,
    home_team_id: Optional[str] = None,
    away_team_id: Optional[str] = None,
    home_team_name: Optional[str] = None,
    away_team_name: Optional[str] = None,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    if not events:
        return result

    side_counters = {

        "shots":
            [None, None],

        "shots_on_target":
            [None, None],

        "shots_off_target":
            [None, None],

        "shots_blocked":
            [None, None],

        "corners":
            [None, None],

        "fouls":
            [None, None],

        "yellow_cards":
            [None, None],

        "red_cards":
            [None, None],

        "offsides":
            [None, None],
    }

    period_counters = {
        "shots_80_87": 0,
        "sot_80_87": 0,
        "corners_80_87": 0,
        "fouls_80_87": 0,
        "cards_80_87": 0,
        "subs_80_87": 0,
    }

    last_goal = None
    last_shot = None
    last_sot = None
    last_corner = None

    side_unknown_for_stats = False

    # Zéro prouvé pour les cartes seulement si la timeline démontre
    # qu'elle transporte effectivement des événements de cartes.
    card_feed_active = False

    for event in events:

        minute = extract_event_minute(
            event
        )

        if minute is None:
            continue

        # SHADOW V1:
        # Interdire tout événement à la minute
        # de décision ou après celle-ci.
        if minute >= current_minute:
            continue

        event_type = normalize_event_type(
            extract_event_type(
                event
            )
        )

        if not event_type:
            continue

        home_side = event_belongs_to_home(
            event,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
        )

        # ------------------------------------------------------------------
        # CLASSIFICATION
        # ------------------------------------------------------------------

        is_goal = any(
            token in event_type
            for token in (
                "goal",
                "score",
            )
        )

        is_corner = (
            "corner" in event_type
        )

        is_foul = (
            "foul" in event_type
        )

        is_yellow = (
            "yellow" in event_type
            or event_type in (
                "card",
                "yellow_card",
            )
        )

        is_red = (
            "red" in event_type
            or "second_yellow" in event_type
            or "2nd_yellow" in event_type
            or "yellow_red" in event_type
            or "sending_off" in event_type
            or "sent_off" in event_type
        )

        is_card = (
            is_yellow
            or is_red
            or "card" in event_type
        )

        is_sub = any(
            token in event_type
            for token in (
                "substitution",
                "sub",
                "substituted",
            )
        )

        is_offside = (
            "offside" in event_type
            or "offsides" in event_type
        )

        # IMPORTANT:
        # Specialized shot event types are counted exactly once.
        # We do not count a generic "shot" in addition to a subtype.
        is_shot = (
            event_type in {
                "shot",
                "shots",
                "attempt",
                "total_shot",
            }
            or "shot" in event_type
        )

        is_sot = (
            "shot_on_target" in event_type
            or "on_target" in event_type
            or "saved" in event_type
        )

        is_off_target = (
            "shot_off_target" in event_type
            or "off_target" in event_type
        )

        is_blocked = (
            "shot_blocked" in event_type
            or "blocked_shot" in event_type
        )

        # ------------------------------------------------------------------
        # LAST EVENT MINUTES
        # ------------------------------------------------------------------

        if is_goal:

            if (
                last_goal is None
                or minute >= last_goal
            ):
                last_goal = minute

        if is_shot:

            if (
                last_shot is None
                or minute >= last_shot
            ):
                last_shot = minute

        if is_sot:

            if (
                last_sot is None
                or minute >= last_sot
            ):
                last_sot = minute

        if is_corner:

            if (
                last_corner is None
                or minute >= last_corner
            ):
                last_corner = minute

        # ------------------------------------------------------------------
        # SIDE-SPECIFIC COUNTERS
        # ------------------------------------------------------------------

        side_index = None

        if home_side is True:
            side_index = 0

        elif home_side is False:
            side_index = 1

        if is_shot:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "shots"
                    ][side_index]
                )

                side_counters[
                    "shots"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_sot:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "shots_on_target"
                    ][side_index]
                )

                side_counters[
                    "shots_on_target"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_off_target:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "shots_off_target"
                    ][side_index]
                )

                side_counters[
                    "shots_off_target"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_blocked:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "shots_blocked"
                    ][side_index]
                )

                side_counters[
                    "shots_blocked"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_corner:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "corners"
                    ][side_index]
                )

                side_counters[
                    "corners"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_foul:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "fouls"
                    ][side_index]
                )

                side_counters[
                    "fouls"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_yellow:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "yellow_cards"
                    ][side_index]
                )

                side_counters[
                    "yellow_cards"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_red:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "red_cards"
                    ][side_index]
                )

                side_counters[
                    "red_cards"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        if is_offside:

            if side_index is None:

                side_unknown_for_stats = True

            else:

                current = (
                    side_counters[
                        "offsides"
                    ][side_index]
                )

                side_counters[
                    "offsides"
                ][side_index] = (
                    1
                    if current is None
                    else current + 1
                )

        # ------------------------------------------------------------------
        # CARD FEED EVIDENCE
        # ------------------------------------------------------------------
        # A card at ANY minute proves the timeline carries card events.
        # If the team is resolvable, missing yellow/red counters may then
        # safely become evidence-based zeros.
        if (
            is_card
            and side_index is not None
        ):
            card_feed_active = True

        # ------------------------------------------------------------------
        # 80 -> 87
        # ------------------------------------------------------------------

        if 80 <= minute <= 87:

            if is_shot:
                period_counters[
                    "shots_80_87"
                ] += 1

            if is_sot:
                period_counters[
                    "sot_80_87"
                ] += 1

            if is_corner:
                period_counters[
                    "corners_80_87"
                ] += 1

            if is_foul:
                period_counters[
                    "fouls_80_87"
                ] += 1

            if is_card:

                period_counters[
                    "cards_80_87"
                ] += 1

            if is_sub:
                period_counters[
                    "subs_80_87"
                ] += 1

    # ------------------------------------------------------------------------
    # Side-specific values
    #
    # We only write zeros when the timeline has been sufficiently parsed.
    # If team ownership cannot be resolved, we leave the feature missing.
    # ------------------------------------------------------------------------

    if not side_unknown_for_stats:

        # Evidence-based zero :
        # dès qu'au moins une carte a été reconnue avec son équipe,
        # la timeline prouve que le flux cartes est actif pour ce match.
        # Les compteurs absents jaune/rouge deviennent alors 0.
        if card_feed_active:

            for stat_name in (
                "yellow_cards",
                "red_cards",
            ):

                for side_index in (0, 1):

                    if (
                        side_counters[
                            stat_name
                        ][side_index]
                        is None
                    ):
                        side_counters[
                            stat_name
                        ][side_index] = 0

        mapping = {

            "shots":
                (
                    "home_shots_87",
                    "away_shots_87",
                ),

            "shots_on_target":
                (
                    "home_shots_on_target_87",
                    "away_shots_on_target_87",
                ),

            "shots_off_target":
                (
                    "home_shots_off_target_87",
                    "away_shots_off_target_87",
                ),

            "shots_blocked":
                (
                    "home_shots_blocked_87",
                    "away_shots_blocked_87",
                ),

            "corners":
                (
                    "home_corners_87",
                    "away_corners_87",
                ),

            "fouls":
                (
                    "home_fouls_87",
                    "away_fouls_87",
                ),

            "yellow_cards":
                (
                    "home_yellow_cards_87",
                    "away_yellow_cards_87",
                ),

            "red_cards":
                (
                    "home_red_cards_87",
                    "away_red_cards_87",
                ),

            "offsides":
                (
                    "home_offsides_87",
                    "away_offsides_87",
                ),
        }

        for stat_name, feature_names in (
            mapping.items()
        ):

            home_value, away_value = (
                side_counters[
                    stat_name
                ]
            )

            if home_value is not None:

                result[
                    feature_names[0]
                ] = home_value

            if away_value is not None:

                result[
                    feature_names[1]
                ] = away_value

    # ------------------------------------------------------------------------
    # Period counters
    # ------------------------------------------------------------------------

    result.update(
        period_counters
    )

    # ------------------------------------------------------------------------
    # Temporal features
    #
    # Missing event => missing feature.
    # We do NOT turn absence of an event into zero.
    # ------------------------------------------------------------------------

    if last_goal is not None:

        result[
            "last_goal_minute"
        ] = last_goal

        result[
            "minutes_since_last_goal"
        ] = max(
            0,
            current_minute - last_goal,
        )

    if last_shot is not None:

        result[
            "last_shot_minute"
        ] = last_shot

        result[
            "minutes_since_last_shot"
        ] = max(
            0,
            current_minute - last_shot,
        )

    if last_sot is not None:

        result[
            "last_shot_on_target_minute"
        ] = last_sot

        result[
            "minutes_since_last_shot_on_target"
        ] = max(
            0,
            current_minute - last_sot,
        )

    if last_corner is not None:

        result[
            "last_corner_minute"
        ] = last_corner

        result[
            "minutes_since_last_corner"
        ] = max(
            0,
            current_minute - last_corner,
        )

    return result


# ============================================================================
# ODDS EXTRACTION
# ============================================================================

def extract_opening_price(
    market_value: Any,
) -> Optional[float]:

    if not isinstance(
        market_value,
        Mapping,
    ):
        return None

    # STRICT:
    # only "opening" is allowed.
    opening = market_value.get(
        "opening"
    )

    return safe_float(
        opening
    )


def bookmaker_name(
    bookmaker: Mapping[str, Any],
) -> str:

    return str(
        first_not_none(
            bookmaker.get(
                "bookmaker"
            ),
            bookmaker.get(
                "name"
            ),
            bookmaker.get(
                "title"
            ),
            "unknown",
        )
    )


def find_bookmakers(
    payload: Any,
) -> List[Mapping[str, Any]]:

    if not isinstance(
        payload,
        Mapping,
    ):
        return []

    data = payload.get(
        "data",
        payload,
    )

    if not isinstance(
        data,
        Mapping,
    ):
        return []

    bookmakers = data.get(
        "bookmakers",
        []
    )

    if not isinstance(
        bookmakers,
        list,
    ):
        return []

    return [
        item
        for item in bookmakers
        if isinstance(
            item,
            Mapping,
        )
    ]


def bookmaker_priority(
    bookmakers: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:

    preferred = [
        bookmaker
        for bookmaker in bookmakers
        if bookmaker_name(
            bookmaker
        ).lower()
        not in (
            "",
            "unknown",
        )
    ]

    def rank(
        bookmaker: Mapping[str, Any],
    ) -> int:

        name = bookmaker_name(
            bookmaker
        ).lower()

        if "pinnacle" in name:
            return 0

        if "bet365" in name:
            return 1

        return 2

    return sorted(
        preferred,
        key=rank,
    )


def market_path_value(
    markets: Mapping[str, Any],
    market_name: str,
    selection: str,
) -> Optional[float]:

    market = markets.get(
        market_name
    )

    if not isinstance(
        market,
        Mapping,
    ):
        return None

    selection_value = market.get(
        selection
    )

    return extract_opening_price(
        selection_value
    )


def total_goals_market_value(
    markets: Mapping[str, Any],
    line: str,
    side: str,
) -> Optional[float]:

    total_goals = markets.get(
        "total_goals"
    )

    if not isinstance(
        total_goals,
        Mapping,
    ):
        return None

    line_data = (
        total_goals.get(line)
    )

    if not isinstance(
        line_data,
        Mapping,
    ):
        return None

    selection = line_data.get(
        side
    )

    return extract_opening_price(
        selection
    )


def find_opening_market_price(
    bookmakers: Sequence[Mapping[str, Any]],
    market_type: str,
    selection: Optional[str] = None,
    line: Optional[str] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[float]:

    diagnostics = (
        diagnostics
        if diagnostics is not None
        else {}
    )

    for bookmaker in bookmaker_priority(
        bookmakers
    ):

        name = bookmaker_name(
            bookmaker
        )

        markets = bookmaker.get(
            "markets",
            {}
        )

        if not isinstance(
            markets,
            Mapping,
        ):
            continue

        value = None
        path = None

        if market_type == "match_odds":

            if selection is None:
                continue

            value = market_path_value(
                markets,
                "match_odds",
                selection,
            )

            path = (
                f"match_odds."
                f"{selection}.opening"
            )

        elif market_type == "btts":

            if selection is None:
                continue

            value = market_path_value(
                markets,
                "btts",
                selection,
            )

            path = (
                f"btts."
                f"{selection}.opening"
            )

        elif market_type == "total_goals":

            if line is None or selection is None:
                continue

            value = total_goals_market_value(
                markets,
                line,
                selection,
            )

            path = (
                f"total_goals."
                f"{line}."
                f"{selection}.opening"
            )

        if value is not None:

            diagnostics.update({
                "status": "available",
                "bookmaker": name,
                "path": path,
                "value": value,
            })

            return value

    diagnostics.update({
        "status":
            "no_valid_opening_price",
        "bookmaker": None,
        "path": None,
        "value": None,
    })

    return None


def flatten_opening_odds(
    payload: Any,
    diagnostics: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    diagnostic_store = (
        diagnostics
        if diagnostics is not None
        else {}
    )

    bookmakers = find_bookmakers(
        payload
    )

    if not bookmakers:

        for feature in ODDS_FEATURES:

            diagnostic_store[
                feature
            ] = {
                "status":
                    "no_bookmakers",
                "endpoint":
                    "/football/matches/{id}/odds",
                "path": None,
                "bookmaker": None,
            }

        return result

    definitions = {

        "odds_home":
            (
                "match_odds",
                "home",
                None,
            ),

        "odds_draw":
            (
                "match_odds",
                "draw",
                None,
            ),

        "odds_away":
            (
                "match_odds",
                "away",
                None,
            ),

        "odds_over05":
            (
                "total_goals",
                "over",
                "0.5",
            ),

        "odds_over15":
            (
                "total_goals",
                "over",
                "1.5",
            ),

        "odds_over25":
            (
                "total_goals",
                "over",
                "2.5",
            ),

        "odds_over35":
            (
                "total_goals",
                "over",
                "3.5",
            ),

        "odds_under25":
            (
                "total_goals",
                "under",
                "2.5",
            ),

        "odds_btts_yes":
            (
                "btts",
                "yes",
                None,
            ),

        "odds_btts_no":
            (
                "btts",
                "no",
                None,
            ),
    }

    for feature, definition in (
        definitions.items()
    ):

        market_type, selection, line = (
            definition
        )

        detail: Dict[str, Any] = {}

        value = find_opening_market_price(
            bookmakers,
            market_type,
            selection=selection,
            line=line,
            diagnostics=detail,
        )

        detail["endpoint"] = (
            "/football/matches/{id}/odds"
        )

        diagnostic_store[
            feature
        ] = detail

        if value is not None:

            result[
                feature
            ] = value

    return result


# ============================================================================
# FEATURE VECTOR
# ============================================================================

def empty_feature_vector() -> Dict[str, Any]:

    return {
        feature: None
        for feature in EXPECTED_FEATURES
    }


def add_match_state_features(
    features: Dict[str, Any],
    match: LiveMatch,
) -> None:

    if (
        match.home_score is None
        or match.away_score is None
    ):
        return

    features[
        "score_87_home"
    ] = match.home_score

    features[
        "score_87_away"
    ] = match.away_score

    if match.home_score > match.away_score:

        features[
            "leading_team_87"
        ] = 0

    elif match.away_score > match.home_score:

        features[
            "leading_team_87"
        ] = 1

    else:

        features[
            "leading_team_87"
        ] = 2

    features[
        "goal_diff_87"
    ] = abs(
        match.home_score
        - match.away_score
    )

    features[
        "is_draw_87"
    ] = int(
        match.home_score
        == match.away_score
    )


# ============================================================================
# PRE-MATCH FEATURES
# ============================================================================

def normalize_feature_dict(
    data: Any,
) -> Dict[str, Any]:

    if data is None:
        return {}

    if isinstance(
        data,
        pd.DataFrame,
    ):

        if len(data) == 0:
            return {}

        return data.iloc[
            0
        ].to_dict()

    if isinstance(
        data,
        Mapping,
    ):

        return dict(data)

    return {}


def call_pre_match_builder(
    match: LiveMatch,
    client: StatsAPIClient,
    diagnostics: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    diagnostics = (
        diagnostics
        if diagnostics is not None
        else {}
    )

    if pre_match is None:

        diagnostics[
            "_builder"
        ] = {
            "status":
                "module_unavailable",
        }

        logger.error(
            "pre_match_feature_engineering indisponible."
        )

        return {}

    builder_names = [
        "build_pre_match_features",
        "build_real_pre_match_features",
        "build_features",
        "construct_pre_match_features",
        "get_pre_match_features",
    ]

    builder = None

    for name in builder_names:

        candidate = getattr(
            pre_match,
            name,
            None,
        )

        if callable(candidate):

            builder = candidate

            diagnostics[
                "_builder_name"
            ] = name

            break

    if builder is None:

        diagnostics[
            "_builder"
        ] = {
            "status":
                "function_unavailable",
        }

        logger.error(
            "Aucune fonction de construction PRE-MATCH "
            "compatible trouvée."
        )

        return {}

    # Le builder officiel de pre_match_feature_engineering.py utilise :
    # build_pre_match_features(client, match, preferred_bookmaker=None)
    # `match.raw` contient la structure API complète du match LIVE.
    # On essaie cette signature en priorité.
    attempts = [
        ((), {
            "client": client,
            "match": match.raw,
        }),
        ((client, match.raw), {}),
        ((), {
            "client": client,
            "match": match.raw,
            "preferred_bookmaker": None,
        }),

        # Compatibilité avec d'anciens builders éventuels.
        ((), {
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "competition_id": match.competition_id,
            "season_id": match.season_id,
            "api_client": client,
        }),
        ((), {
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "competition_id": match.competition_id,
            "season_id": match.season_id,
        }),
        ((), {
            "match_id": match.match_id,
            "client": client,
        }),
    ]

    last_type_error = None

    for attempt_index, (args, kwargs) in enumerate(
        attempts,
        start=1,
    ):

        try:

            result = builder(
                *args,
                **kwargs,
            )

            normalized = (
                normalize_feature_dict(
                    result
                )
            )

            if normalized:

                diagnostics[
                    "_builder"
                ] = {
                    "status":
                        "available",
                    "attempt":
                        attempt_index,
                    "returned_features":
                        len(normalized),
                }

                logger.info(
                    "Builder PRE-MATCH appelé : %s | tentative=%d | features=%d",
                    builder.__name__,
                    attempt_index,
                    len(normalized),
                )

                return normalized

            diagnostics[
                "_builder"
            ] = {
                "status":
                    "empty_result",
                "attempt":
                    attempt_index,
            }

        except TypeError as exc:

            last_type_error = str(exc)
            continue

        except Exception as exc:

            diagnostics[
                "_builder"
            ] = {
                "status":
                    "error",
                "error":
                    str(exc),
                "attempt":
                    attempt_index,
            }

            logger.exception(
                "Erreur PRE-MATCH avec %s : %s",
                builder.__name__,
                exc,
            )

            return {}

    diagnostics[
        "_builder"
    ] = {
        "status":
            "no_compatible_signature",
        "last_type_error":
            last_type_error,
    }

    logger.error(
        "Impossible d'appeler le builder PRE-MATCH | builder=%s | dernier TypeError=%s",
        getattr(builder, "__name__", "unknown"),
        last_type_error,
    )

    return {}


def build_real_pre_match_features(
    match: LiveMatch,
    client: StatsAPIClient,
    odds_payload: Any = None,
    diagnostics: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    diagnostics = (
        diagnostics
        if diagnostics is not None
        else {}
    )

    logger.info(
        "Construction des features PRE-MATCH..."
    )

    logger.info(
        "Match : %s vs %s",
        match.home_team,
        match.away_team,
    )

    logger.info(
        "Match ID : %s",
        match.match_id,
    )

    logger.info(
        "Competition : %s",
        match.competition_id,
    )

    logger.info(
        "Season : %s",
        match.season_id,
    )

    features = call_pre_match_builder(
        match,
        client,
        diagnostics=diagnostics,
    )

    normalized: Dict[str, Any] = {}

    for feature in PRE_MATCH_FEATURES:

        value = features.get(
            feature
        )

        if value is not None:

            normalized[
                feature
            ] = value

    # Never trust odds returned by the generic PRE-MATCH builder. The
    # production schema requires OPENING prices from the dedicated odds
    # endpoint only. If the endpoint cannot provide a market, that feature
    # must remain missing rather than silently falling back to builder data.
    for feature in ODDS_FEATURES:
        normalized.pop(feature, None)

    # ------------------------------------------------------------------------
    # ODDS
    #
    # We ALWAYS query the dedicated odds endpoint.
    # Builder-provided odds are not trusted because they could be last_seen.
    # ------------------------------------------------------------------------

    odds_diagnostics: Dict[
        str,
        Any,
    ] = {}

    if odds_payload is None:

        try:

            logger.info(
                "Récupération cotes pré-match..."
            )

            odds_payload = client.get_odds(
                match.match_id
            )

        except Exception as exc:

            logger.warning(
                "Impossible de récupérer les cotes : %s",
                exc,
            )

            diagnostics[
                "_odds"
            ] = {
                "status":
                    "api_error",
                "error":
                    str(exc),
            }

            odds_payload = None

    if odds_payload is not None:

        odds = flatten_opening_odds(
            odds_payload,
            diagnostics=odds_diagnostics,
        )

        # STRICT:
        # Only values returned by the opening-odds extractor are accepted.
        for feature in ODDS_FEATURES:

            value = odds.get(
                feature
            )

            if value is not None:

                normalized[
                    feature
                ] = value

        diagnostics[
            "_odds"
        ] = odds_diagnostics

    # ------------------------------------------------------------------------
    # PRE-MATCH diagnostics
    # ------------------------------------------------------------------------

    for feature in PRE_MATCH_FEATURES:

        if normalized.get(
            feature
        ) is not None:

            diagnostics[
                feature
            ] = {
                "status":
                    "available",
                "source":
                    FEATURE_SOURCE_MAP.get(
                        feature,
                        "unknown",
                    ),
            }

        else:

            if feature in ODDS_FEATURES:

                odds_detail = (
                    odds_diagnostics.get(
                        feature,
                        {},
                    )
                )

                diagnostics[
                    feature
                ] = {
                    "status":
                        odds_detail.get(
                            "status",
                            "missing",
                        ),
                    "source":
                        FEATURE_SOURCE_MAP.get(
                            feature,
                            "unknown",
                        ),
                    "endpoint":
                        odds_detail.get(
                            "endpoint",
                            "/football/matches/{id}/odds",
                        ),
                    "path":
                        odds_detail.get(
                            "path"
                        ),
                    "bookmaker":
                        odds_detail.get(
                            "bookmaker"
                        ),
                }

            else:

                diagnostics[
                    feature
                ] = {
                    "status":
                        "builder_no_value",
                    "source":
                        FEATURE_SOURCE_MAP.get(
                            feature,
                            "unknown",
                        ),
                }

    available = [
        feature
        for feature in PRE_MATCH_FEATURES
        if normalized.get(feature) is not None
    ]

    missing = [
        feature
        for feature in PRE_MATCH_FEATURES
        if normalized.get(feature) is None
    ]

    logger.info(
        "PRE-MATCH disponibles : %d/%d",
        len(available),
        len(PRE_MATCH_FEATURES),
    )

    if missing:

        logger.warning(
            "PRE-MATCH manquantes : %d/%d",
            len(missing),
            len(PRE_MATCH_FEATURES),
        )

        logger.warning(
            "PRE-MATCH : %s",
            ", ".join(missing),
        )

    return normalized


# ============================================================================
# MISSING FEATURE DIAGNOSTICS
# ============================================================================

def diagnose_missing_features(
    features: Mapping[str, Any],
    match_id: Optional[str] = None,
    diagnostics: Optional[
        Mapping[str, Any]
    ] = None,
) -> List[str]:

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if (
            feature not in features
            or is_missing(
                features.get(feature)
            )
        )
    ]

    if not missing:

        logger.info(
            "Aucune feature manquante."
        )

        return []

    logger.error(
        "DIAGNOSTIC FEATURES MANQUANTES | "
        "match=%s | %d feature(s)",
        match_id or "?",
        len(missing),
    )

    for feature in missing:

        source = FEATURE_SOURCE_MAP.get(
            feature,
            "source inconnue",
        )

        detail = {}

        if diagnostics is not None:

            value = diagnostics.get(
                feature
            )

            if isinstance(
                value,
                Mapping,
            ):
                detail = dict(
                    value
                )

        status = detail.get(
            "status",
            "missing",
        )

        endpoint = detail.get(
            "endpoint"
        )

        path = detail.get(
            "path"
        )

        bookmaker = detail.get(
            "bookmaker"
        )

        logger.error(
            "  MISSING | %-35s | "
            "source=%s | status=%s",
            feature,
            source,
            status,
        )

        if endpoint:

            logger.error(
                "           endpoint=%s",
                endpoint,
            )

        if path:

            logger.error(
                "           path=%s",
                path,
            )

        if bookmaker:

            logger.error(
                "           bookmaker=%s",
                bookmaker,
            )

    return missing


# ============================================================================
# BUILD ALL FEATURES
# ============================================================================

def build_features(
    match: LiveMatch,
    client: StatsAPIClient,
    pre_match_features: Optional[
        Mapping[str, Any]
    ] = None,
    diagnostics: Optional[
        Dict[str, Any]
    ] = None,
    required_features: Optional[
        Sequence[str]
    ] = None,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:

    diagnostics = (
        diagnostics
        if diagnostics is not None
        else {}
    )

    features = empty_feature_vector()

    # Les fallbacks coûteux ne sont lancés que s'ils peuvent compléter une
    # feature réellement attendue par le modèle chargé.
    required_feature_set: Set[str] = set(
        required_features
        if required_features is not None
        else EXPECTED_FEATURES
    )

    required_live_features: Set[str] = (
        required_feature_set
        & set(LIVE_FEATURES)
    )

    # ------------------------------------------------------------------------
    # PRE-MATCH
    # ------------------------------------------------------------------------

    if pre_match_features is None:

        pre_features = (
            build_real_pre_match_features(
                match,
                client,
                diagnostics=diagnostics,
            )
        )

    else:

        pre_features = dict(
            pre_match_features
        )

    for feature in PRE_MATCH_FEATURES:

        value = pre_features.get(
            feature
        )

        if value is not None:

            features[
                feature
            ] = value

    # ------------------------------------------------------------------------
    # MATCH STATE
    # ------------------------------------------------------------------------

    add_match_state_features(
        features,
        match,
    )

    score_available = (
        match.home_score is not None
        and match.away_score is not None
    )

    if score_available:

        for feature in (
            "score_87_home",
            "score_87_away",
            "leading_team_87",
            "goal_diff_87",
            "is_draw_87",
        ):

            diagnostics[
                feature
            ] = {
                "status":
                    "available",
                "source":
                    FEATURE_SOURCE_MAP[
                        feature
                    ],
            }

    else:

        for feature in (
            "score_87_home",
            "score_87_away",
            "leading_team_87",
            "goal_diff_87",
            "is_draw_87",
        ):

            diagnostics[
                feature
            ] = {
                "status":
                    "score_unavailable",
                "source":
                    FEATURE_SOURCE_MAP[
                        feature
                    ],
            }

    # ------------------------------------------------------------------------
    # LIVE-STATS
    # ------------------------------------------------------------------------

    live_stats_features: Dict[
        str,
        Any,
    ] = {}

    live_stats_available = False

    try:

        logger.info(
            "Récupération LIVE-STATS..."
        )

        live_payload = (
            client.get_live_stats(
                match.match_id
            )
        )

        live_stats_features = (
            extract_live_stats(
                live_payload
            )
        )

        if live_stats_features:

            live_stats_available = True

        for feature in LIVE_FEATURES:

            if feature in (
                "shots_80_87",
                "sot_80_87",
                "corners_80_87",
                "fouls_80_87",
                "cards_80_87",
                "subs_80_87",
                "last_goal_minute",
                "last_shot_minute",
                "last_shot_on_target_minute",
                "last_corner_minute",
                "minutes_since_last_goal",
                "minutes_since_last_shot",
                "minutes_since_last_shot_on_target",
                "minutes_since_last_corner",
            ):
                continue

            value = (
                live_stats_features.get(
                    feature
                )
            )

            if value is not None:

                features[
                    feature
                ] = value

                diagnostics[
                    feature
                ] = {
                    "status":
                        "available",
                    "source":
                        "LIVE → /football/matches/"
                        f"{match.match_id}/live-stats",
                }

    except Exception as exc:

        diagnostics[
            "_live_stats"
        ] = {
            "status":
                "api_error",
            "error":
                str(exc),
        }

        logger.warning(
            "LIVE-STATS indisponibles : %s",
            exc,
        )

    # ------------------------------------------------------------------------
    # MATCH DETAIL FALLBACK - explicit red-card counters only
    # ------------------------------------------------------------------------

    try:

        missing_red_features = [
            feature
            for feature in (
                "home_red_cards_87",
                "away_red_cards_87",
            )
            if (
                feature in required_live_features
                and features.get(feature) is None
            )
        ]

        if missing_red_features:

            logger.info(
                "Récupération MATCH DETAIL fallback pour red_cards..."
            )

            match_detail_payload = client.get_match(
                match.match_id
            )

            detail_red_cards = extract_red_cards_from_match_detail(
                match_detail_payload
            )

            for feature in missing_red_features:

                value = detail_red_cards.get(
                    feature
                )

                if value is None:
                    continue

                features[feature] = value

                diagnostics[feature] = {
                    "status":
                        "available_match_detail_fallback",
                    "source":
                        "LIVE → /football/matches/"
                        f"{match.match_id} → red_cards",
                }

    except Exception as exc:

        diagnostics["_match_detail_red_cards"] = {
            "status": "api_error",
            "error": str(exc),
        }

        logger.warning(
            "MATCH DETAIL red_cards indisponible : %s",
            exc,
        )

    # ------------------------------------------------------------------------
    # MATCH STATS SECONDARY FALLBACK
    # ------------------------------------------------------------------------

    try:

        missing_before_stats = [
            feature
            for feature in required_live_features
            if features.get(feature) is None
            and feature not in (
                "shots_80_87",
                "sot_80_87",
                "corners_80_87",
                "fouls_80_87",
                "cards_80_87",
                "subs_80_87",
                "last_goal_minute",
                "last_shot_minute",
                "last_shot_on_target_minute",
                "last_corner_minute",
                "minutes_since_last_goal",
                "minutes_since_last_shot",
                "minutes_since_last_shot_on_target",
                "minutes_since_last_corner",
            )
        ]

        if missing_before_stats:

            logger.info(
                "Récupération MATCH-STATS fallback..."
            )

            stats_payload = client.get_stats(
                match.match_id
            )

            stats_features = (
                extract_match_stats_features(
                    stats_payload
                )
            )

            for feature, value in stats_features.items():

                if (
                    features.get(feature) is None
                    and value is not None
                ):
                    features[feature] = value
                    diagnostics[feature] = {
                        "status": "available_match_stats_fallback",
                        "source": (
                            "LIVE → /football/matches/"
                            f"{match.match_id}/stats"
                        ),
                    }

            derive_shots_off_target(
                features
            )

            for side in ("home", "away"):
                feature = f"{side}_shots_off_target_87"
                if (
                    features.get(feature) is not None
                    and feature not in stats_features
                    and diagnostics.get(feature, {}).get("status")
                    not in ("available", "available_match_stats_fallback")
                ):
                    diagnostics[feature] = {
                        "status": "available_derived",
                        "source": (
                            "DERIVED → total_shots - "
                            "shots_on_target - shots_blocked"
                        ),
                    }

    except Exception as exc:

        diagnostics["_match_stats_fallback"] = {
            "status": "api_error",
            "error": str(exc),
        }

        logger.warning(
            "MATCH-STATS fallback indisponible : %s",
            exc,
        )

    # ------------------------------------------------------------------------
    # TIMELINE
    # ------------------------------------------------------------------------

    timeline_result: Dict[
        str,
        Any,
    ] = {}

    timeline_available = False

    missing_required_live_before_timeline = [
        feature
        for feature in required_live_features
        if features.get(feature) is None
    ]

    try:

        if not missing_required_live_before_timeline:
            logger.info(
                "TIMELINE ignorée : toutes les features LIVE modèle requises sont déjà disponibles."
            )
            timeline_payload = None
            events = []
        else:
            logger.info(
                "Récupération TIMELINE fallback | manque modèle=%d",
                len(missing_required_live_before_timeline),
            )

            timeline_payload = (
                client.get_timeline(
                    match.match_id
                )
            )

            events = extract_events(
                timeline_payload
            )

        # ``events`` est déjà initialisé lorsque la timeline est évitée.


        logger.info(
            "Timeline : %d événements",
            len(events),
        )

        if events:

            timeline_available = True

        home_team_id = (
            extract_team_id(
                match.raw,
                "home",
            )
        )

        away_team_id = (
            extract_team_id(
                match.raw,
                "away",
            )
        )

        timeline_result = analyze_timeline(
            events,
            current_minute=match.minute,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=match.home_team,
            away_team_name=match.away_team,
        )

        # --------------------------------------------------------------------
        # FALLBACK:
        # only fill a feature if live-stats did not provide it.
        # --------------------------------------------------------------------

        for feature in LIVE_FEATURES:

            current = features.get(
                feature
            )

            fallback = (
                timeline_result.get(
                    feature
                )
            )

            if (
                current is None
                and fallback is not None
            ):

                features[
                    feature
                ] = fallback

                diagnostics[
                    feature
                ] = {
                    "status":
                        "available_timeline_fallback",
                    "source":
                        "LIVE → timeline fallback",
                }

                if (
                    feature in (
                        "home_red_cards_87",
                        "away_red_cards_87",
                        "home_yellow_cards_87",
                        "away_yellow_cards_87",
                    )
                    and fallback == 0
                ):
                    logger.info(
                        "TIMELINE PROVEN ZERO | %s = 0",
                        feature,
                    )

        missing_red_after_timeline = [
            feature
            for feature in (
                "home_red_cards_87",
                "away_red_cards_87",
            )
            if features.get(feature) is None
        ]

        if (
            missing_red_after_timeline
            and events
        ):

            card_events = timeline_card_debug(
                events
            )

            logger.warning(
                "DEBUG RED_CARDS | manquantes=%s | card_events=%d",
                missing_red_after_timeline,
                len(card_events),
            )

            for card_event in card_events[:12]:

                logger.warning(
                    "DEBUG CARD EVENT | %s",
                    json.dumps(
                        card_event,
                        ensure_ascii=False,
                        default=str,
                    ),
                )

    except Exception as exc:

        diagnostics[
            "_timeline"
        ] = {
            "status":
                "api_error",
            "error":
                str(exc),
        }

        logger.warning(
            "Timeline indisponible : %s",
            exc,
        )

    # ------------------------------------------------------------------------
    # Mark remaining LIVE features as explicitly missing.
    # ------------------------------------------------------------------------

    for feature in LIVE_FEATURES:

        if features.get(
            feature
        ) is not None:
            continue

        existing = diagnostics.get(
            feature
        )

        if isinstance(
            existing,
            Mapping,
        ) and existing.get(
            "status"
        ) in (
            "available",
            "available_match_stats_fallback",
            "available_derived",
            "available_timeline_fallback",
        ):
            continue

        if live_stats_available:

            status = (
                "live_stats_no_value"
            )

        elif timeline_available:

            status = (
                "live_stats_missing_and_"
                "timeline_no_value"
            )

        else:

            status = (
                "live_sources_unavailable"
            )

        diagnostics[
            feature
        ] = {
            "status":
                status,
            "source":
                FEATURE_SOURCE_MAP.get(
                    feature,
                    "LIVE",
                ),
            "live_stats":
                (
                    "available"
                    if live_stats_available
                    else "unavailable"
                ),
            "timeline":
                (
                    "available"
                    if timeline_available
                    else "unavailable"
                ),
        }

    # ------------------------------------------------------------------------
    # FINAL DATAFRAME
    # ------------------------------------------------------------------------

    X = pd.DataFrame(
        [[
            features[
                feature
            ]
            for feature in EXPECTED_FEATURES
        ]],
        columns=EXPECTED_FEATURES,
    )

    if list(
        X.columns
    ) != EXPECTED_FEATURES:

        raise ValueError(
            "Ordre des features invalide."
        )

    if X.shape[1] != 96:

        raise ValueError(
            f"Nombre de features invalide: "
            f"{X.shape[1]}/96."
        )

    return X, features


# ============================================================================
# STRICT BUILT-IN VALIDATOR
# ============================================================================

def strict_builtin_validator(
    X: pd.DataFrame,
) -> ValidationResult:

    errors: List[str] = []
    warnings: List[str] = []
    missing: List[str] = []

    # ------------------------------------------------------------------------
    # DATAFRAME TYPE
    # ------------------------------------------------------------------------

    if not isinstance(
        X,
        pd.DataFrame,
    ):

        return ValidationResult(
            status="FAIL",
            ready_for_model=False,
            errors=[
                "input_not_dataframe"
            ],
            warnings=[],
            missing=EXPECTED_FEATURES.copy(),
            coverage=0,
        )

    # ------------------------------------------------------------------------
    # ROW COUNT
    # ------------------------------------------------------------------------

    if len(X) != 1:

        errors.append(
            f"wrong_row_count:{len(X)}"
        )

    # ------------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------------

    if list(
        X.columns
    ) != EXPECTED_FEATURES:

        errors.append(
            "schema_mismatch"
        )

    if X.shape[1] != 96:

        errors.append(
            f"wrong_feature_count:"
            f"{X.shape[1]}"
        )

    # ------------------------------------------------------------------------
    # MISSING
    # ------------------------------------------------------------------------

    for feature in EXPECTED_FEATURES:

        if feature not in X.columns:

            missing.append(
                feature
            )

            continue

        value = X.iloc[
            0
        ][feature]

        if is_missing(
            value
        ):

            missing.append(
                feature
            )

    if missing:

        errors.append(
            f"missing_features:"
            f"{len(missing)}"
        )

    # ------------------------------------------------------------------------
    # NUMERIC + FINITE
    # ------------------------------------------------------------------------

    for feature in EXPECTED_FEATURES:

        if feature not in X.columns:
            continue

        value = X.iloc[
            0
        ][feature]

        if is_missing(
            value
        ):
            continue

        numeric = safe_float(
            value
        )

        if numeric is None:

            errors.append(
                f"non_numeric:"
                f"{feature}"
            )

            continue

        if not math.isfinite(
            numeric
        ):

            errors.append(
                f"non_finite:"
                f"{feature}"
            )

    # ------------------------------------------------------------------------
    # FREQUENCIES
    # ------------------------------------------------------------------------

    for feature in FREQUENCY_FEATURES:

        if feature not in X.columns:
            continue

        value = safe_float(
            X.iloc[
                0
            ][feature]
        )

        if value is None:
            continue

        if not 0 <= value <= 1:

            errors.append(
                f"frequency_out_of_range:"
                f"{feature}={value}"
            )

    # ------------------------------------------------------------------------
    # ODDS
    # ------------------------------------------------------------------------

    for feature in ODDS_FEATURES:

        if feature not in X.columns:
            continue

        value = safe_float(
            X.iloc[
                0
            ][feature]
        )

        if value is None:
            continue

        if value <= 1:

            errors.append(
                f"invalid_odds:"
                f"{feature}={value}"
            )

    # ------------------------------------------------------------------------
    # NON-NEGATIVE
    # ------------------------------------------------------------------------

    for feature in NON_NEGATIVE_FEATURES:

        if feature not in X.columns:
            continue

        value = safe_float(
            X.iloc[
                0
            ][feature]
        )

        if value is None:
            continue

        if value < 0:

            errors.append(
                f"negative_value:"
                f"{feature}={value}"
            )

    # ------------------------------------------------------------------------
    # SCORE LOGIC
    # ------------------------------------------------------------------------

    home_score = safe_float(
        X.iloc[
            0
        ].get(
            "score_87_home"
        )
    )

    away_score = safe_float(
        X.iloc[
            0
        ].get(
            "score_87_away"
        )
    )

    goal_diff = safe_float(
        X.iloc[
            0
        ].get(
            "goal_diff_87"
        )
    )

    is_draw = safe_float(
        X.iloc[
            0
        ].get(
            "is_draw_87"
        )
    )

    leading = safe_float(
        X.iloc[
            0
        ].get(
            "leading_team_87"
        )
    )

    if (
        home_score is not None
        and away_score is not None
        and goal_diff is not None
    ):

        expected_goal_diff = abs(
            home_score
            - away_score
        )

        if not math.isclose(
            goal_diff,
            expected_goal_diff,
            rel_tol=0,
            abs_tol=1e-9,
        ):

            errors.append(
                "score_goal_diff_inconsistent:"
                f"goal_diff_87={goal_diff},"
                f"expected={expected_goal_diff}"
            )

    if (
        home_score is not None
        and away_score is not None
        and is_draw is not None
    ):

        expected_draw = int(
            home_score == away_score
        )

        if is_draw != expected_draw:

            errors.append(
                "score_draw_inconsistent:"
                f"is_draw_87={is_draw},"
                f"expected={expected_draw}"
            )

    if (
        home_score is not None
        and away_score is not None
        and leading is not None
    ):

        if home_score > away_score:

            expected_leading = 0

        elif away_score > home_score:

            expected_leading = 1

        else:

            expected_leading = 2

        if leading != expected_leading:

            errors.append(
                "leading_team_inconsistent:"
                f"leading_team_87={leading},"
                f"expected={expected_leading}"
            )

    # ------------------------------------------------------------------------
    # LIVE SHOTS COHERENCE
    # ------------------------------------------------------------------------

    home_shots = safe_float(
        X.iloc[
            0
        ].get(
            "home_shots_87"
        )
    )

    away_shots = safe_float(
        X.iloc[
            0
        ].get(
            "away_shots_87"
        )
    )

    home_sot = safe_float(
        X.iloc[
            0
        ].get(
            "home_shots_on_target_87"
        )
    )

    away_sot = safe_float(
        X.iloc[
            0
        ].get(
            "away_shots_on_target_87"
        )
    )

    if (
        home_shots is not None
        and home_sot is not None
        and home_sot > home_shots
    ):

        errors.append(
            "home_sot_greater_than_total_shots"
        )

    if (
        away_shots is not None
        and away_sot is not None
        and away_sot > away_shots
    ):

        errors.append(
            "away_sot_greater_than_total_shots"
        )

    # ------------------------------------------------------------------------
    # COVERAGE
    #
    # CRITICAL:
    # 0/96 must NEVER be READY.
    # ------------------------------------------------------------------------

    coverage = (
        96 - len(
            set(missing)
        )
    )

    ready = (
        coverage == 96
        and len(errors) == 0
        and len(missing) == 0
    )

    status = (
        "OK"
        if ready
        else "FAIL"
    )

    return ValidationResult(
        status=status,
        ready_for_model=ready,
        errors=errors,
        warnings=warnings,
        missing=missing,
        coverage=coverage,
    )


# ============================================================================
# EXTERNAL FEATURE VALIDATOR
# ============================================================================

def _find_external_validator() -> Optional[Any]:

    module_names = [
        "features.feature_validator",
        "feature_validator",
    ]

    function_names = [
        "validate_feature_vector",
        "validate_features",
        "validate",
        "run_validation",
    ]

    for module_name in module_names:

        try:

            module = importlib.import_module(
                module_name
            )

        except ImportError:

            continue

        for function_name in function_names:

            function = getattr(
                module,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Feature validator externe détectée : "
                    "%s.%s",
                    module_name,
                    function_name,
                )

                return function

    logger.warning(
        "Feature validator externe non trouvée. "
        "Validator interne strict utilisé."
    )

    return None


def _normalize_external_validation(
    result: Any,
    X: pd.DataFrame,
) -> ValidationResult:

    builtin = strict_builtin_validator(
        X
    )

    # ------------------------------------------------------------------------
    # Boolean
    # ------------------------------------------------------------------------

    if isinstance(
        result,
        bool,
    ):

        return ValidationResult(
            status=(
                "OK"
                if result
                else "FAIL"
            ),
            ready_for_model=(
                result
                and builtin.coverage == 96
                and not builtin.errors
                and not builtin.missing
            ),
            errors=list(
                builtin.errors
            ),
            warnings=list(
                builtin.warnings
            ),
            missing=list(
                builtin.missing
            ),
            coverage=builtin.coverage,
        )

    # ------------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------------

    if isinstance(
        result,
        Mapping,
    ):

        status = str(
            result.get(
                "status",
                (
                    "OK"
                    if result.get("valid") is True
                    else "FAIL"
                ),
            )
        )

        explicit_ready = result.get(
            "ready_for_model",
            result.get(
                "ready",
                None,
            ),
        )

        errors = result.get(
            "errors",
            [],
        )

        warnings = result.get(
            "warnings",
            [],
        )

        # Plusieurs formats de validator sont supportés :
        # - missing: [...]
        # - missing_features: [...]
        # - coverage: {total, available, missing, percentage}
        missing = result.get(
            "missing",
            result.get(
                "missing_features",
                [],
            ),
        )

        if not isinstance(
            errors,
            list,
        ):
            errors = [errors]

        if not isinstance(
            warnings,
            list,
        ):
            warnings = [warnings]

        if not isinstance(
            missing,
            list,
        ):
            missing = [missing]

        coverage_payload = result.get(
            "coverage",
            None,
        )

        coverage = None

        # Rapport imbriqué du feature_validator.py.
        if isinstance(
            coverage_payload,
            Mapping,
        ):
            coverage = safe_int(
                coverage_payload.get(
                    "available",
                    None,
                )
            )

            if not missing:
                nested_missing_count = safe_int(
                    coverage_payload.get(
                        "missing",
                        None,
                    )
                )

                # Si le validator ne donne pas la liste mais confirme 0
                # manquante, on conserve missing=[].
                if (
                    nested_missing_count is not None
                    and nested_missing_count == 0
                ):
                    missing = []

        # Rapport plat.
        if coverage is None:

            available_count = safe_int(
                result.get(
                    "available",
                    None,
                )
            )

            total_count = safe_int(
                result.get(
                    "total",
                    None,
                )
            )

            if (
                available_count is not None
                and total_count is not None
                and total_count > 0
            ):
                coverage = available_count

        # Coverage numérique ou ratio.
        if (
            coverage is None
            and coverage_payload is not None
            and not isinstance(
                coverage_payload,
                Mapping,
            )
        ):

            try:
                raw_coverage_float = float(
                    coverage_payload
                )
            except (
                TypeError,
                ValueError,
            ):
                raw_coverage_float = None

            if (
                raw_coverage_float is not None
                and 0.0 <= raw_coverage_float <= 1.0
            ):
                coverage = int(
                    round(
                        raw_coverage_float
                        * len(EXPECTED_FEATURES)
                    )
                )
            else:
                coverage = safe_int(
                    coverage_payload
                )

        if coverage is None:
            coverage = (
                len(EXPECTED_FEATURES)
                - len(missing)
            )

        # Readiness explicite si fournie, sinon on la déduit strictement.
        ready = (
            bool(explicit_ready)
            if explicit_ready is not None
            else (
                coverage == len(EXPECTED_FEATURES)
                and not errors
                and not missing
                and status.upper() == "OK"
            )
        )

        return ValidationResult(
            status=(
                "OK"
                if (
                    ready
                    and coverage == len(EXPECTED_FEATURES)
                    and not errors
                    and not missing
                )
                else "FAIL"
            ),
            ready_for_model=(
                ready
                and coverage == len(EXPECTED_FEATURES)
                and not errors
                and not missing
            ),
            errors=[
                str(x)
                for x in errors
                if x not in (None, "")
            ],
            warnings=[
                str(x)
                for x in warnings
                if x not in (None, "")
            ],
            missing=[
                str(x)
                for x in missing
                if x not in (None, "")
            ],
            coverage=coverage,
        )

    # ------------------------------------------------------------------------
    # Object
    # ------------------------------------------------------------------------

    if result is not None:

        status = str(
            getattr(
                result,
                "status",
                "OK",
            )
        )

        ready = bool(
            getattr(
                result,
                "ready_for_model",
                getattr(
                    result,
                    "ready",
                    False,
                ),
            )
        )

        errors = list(
            getattr(
                result,
                "errors",
                [],
            )
            or []
        )

        warnings = list(
            getattr(
                result,
                "warnings",
                [],
            )
            or []
        )

        missing = list(
            getattr(
                result,
                "missing",
                [],
            )
            or []
        )

        coverage = safe_int(
            getattr(
                result,
                "coverage",
                96 - len(missing),
            )
        )

        if coverage is None:
            coverage = 0

        return ValidationResult(
            status=status,
            ready_for_model=(
                ready
                and coverage == 96
                and not errors
                and not missing
            ),
            errors=[
                str(x)
                for x in errors
            ],
            warnings=[
                str(x)
                for x in warnings
            ],
            missing=[
                str(x)
                for x in missing
            ],
            coverage=coverage,
        )

    return ValidationResult(
        status="FAIL",
        ready_for_model=False,
        errors=[
            "external_validator_returned_none"
        ],
        warnings=[],
        missing=[],
        coverage=0,
    )


# ============================================================================
# VALIDATE MODEL INPUT
# ============================================================================

def validate_model_input(
    X: pd.DataFrame,
) -> ValidationResult:

    # ------------------------------------------------------------------------
    # INTERNAL VALIDATOR
    # ------------------------------------------------------------------------

    builtin = strict_builtin_validator(
        X
    )

    logger.info(
        "VALIDATOR INTERNE | "
        "status=%s | coverage=%d/96 | ready=%s",
        builtin.status,
        builtin.coverage,
        builtin.ready_for_model,
    )

    # ------------------------------------------------------------------------
    # EXTERNAL VALIDATOR
    #
    # We run it even when builtin fails so that the diagnostic is complete.
    # ------------------------------------------------------------------------

    validator = _find_external_validator()

    if validator is None:

        if not builtin.ready_for_model:

            logger.error(
                "VALIDATION FAIL | coverage=%d/96",
                builtin.coverage,
            )

            for error in builtin.errors:

                logger.error(
                    "  VALIDATION ERROR | %s",
                    error,
                )

        return builtin

    try:

        # Le validator externe historique validate_features(features)
        # attend généralement un dictionnaire de features.
        #
        # Important :
        # un DataFrame peut être accepté sans TypeError mais être considéré
        # comme un type invalide par le validator, ce qui produit artificiellement
        # coverage=0. On appelle donc le format dict EN PREMIER.
        feature_dict = X.to_dict(
            orient="records"
        )[0]

        try:

            external_raw = validator(
                feature_dict
            )

        except (
            TypeError,
            AttributeError,
        ):

            # Compatibilité avec un éventuel validator conçu pour pandas.
            external_raw = validator(
                X
            )

        logger.debug(
            "VALIDATOR EXTERNE RAW | type=%s | value=%s",
            type(external_raw).__name__,
            external_raw,
        )

        external = (
            _normalize_external_validation(
                external_raw,
                X,
            )
        )

        logger.info(
            "VALIDATOR EXTERNE | "
            "status=%s | coverage=%d/96 | ready=%s",
            external.status,
            external.coverage,
            external.ready_for_model,
        )

        # --------------------------------------------------------------------
        # MERGE
        # --------------------------------------------------------------------

        merged_errors = list(
            dict.fromkeys(
                builtin.errors
                + external.errors
            )
        )

        merged_warnings = list(
            dict.fromkeys(
                builtin.warnings
                + external.warnings
            )
        )

        merged_missing = list(
            dict.fromkeys(
                builtin.missing
                + external.missing
            )
        )

        coverage = min(
            builtin.coverage,
            external.coverage,
        )

        ready = (
            builtin.ready_for_model
            and external.ready_for_model
            and coverage == 96
            and not merged_errors
            and not merged_missing
        )

        result = ValidationResult(
            status=(
                "OK"
                if ready
                else "FAIL"
            ),
            ready_for_model=ready,
            errors=merged_errors,
            warnings=merged_warnings,
            missing=merged_missing,
            coverage=coverage,
        )

        if ready:

            logger.info(
                "VALIDATION OK | 96/96 | "
                "READY FOR MODEL : YES"
            )

        else:

            logger.error(
                "VALIDATION FAIL | "
                "coverage=%d/96 | "
                "READY FOR MODEL : NO",
                coverage,
            )

            if merged_missing:

                logger.error(
                    "Features manquantes : %s",
                    ", ".join(
                        merged_missing
                    ),
                )

            for error in merged_errors:

                logger.error(
                    "  VALIDATION ERROR | %s",
                    error,
                )

        return result

    except Exception as exc:

        logger.exception(
            "Erreur validator externe : %s",
            exc,
        )

        # If external validator crashes, model remains blocked unless
        # internal validation itself is valid. We deliberately fail closed.
        return ValidationResult(
            status="FAIL",
            ready_for_model=False,
            errors=(
                list(
                    builtin.errors
                )
                + [
                    "external_validator_error:"
                    f"{exc}"
                ]
            ),
            warnings=list(
                builtin.warnings
            ),
            missing=list(
                builtin.missing
            ),
            coverage=builtin.coverage,
        )


# ============================================================================
# PRODUCTION MODEL
# ============================================================================

class ProductionModel:

    def __init__(self):

        self.model = None

        self.features: List[str] = []

        self.metadata: Dict[
            str,
            Any,
        ] = {}

    # ----------------------------------------------------------------------

    def load(self) -> None:

        logger.info(
            "Chargement modèle : %s",
            MODEL_PATH,
        )

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Modèle introuvable : "
                f"{MODEL_PATH}"
            )

        if not MODEL_FEATURES_PATH.exists():

            raise FileNotFoundError(
                "Fichier features modèle "
                "introuvable : "
                f"{MODEL_FEATURES_PATH}"
            )

        self.model = joblib.load(
            MODEL_PATH
        )

        with open(
            MODEL_FEATURES_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            Mapping,
        ):

            # V3-C schema uses `features`. Keep the other aliases only for
            # schema-format compatibility, never as a V2 model fallback.
            self.features = list(
                data.get(
                    "features_used",
                    data.get(
                        "core_features",
                        data.get(
                            "features",
                            data.get(
                                "feature_names",
                                [],
                            ),
                        ),
                    ),
                )
            )

        elif isinstance(
            data,
            list,
        ):

            self.features = list(
                data
            )

        else:

            raise ValueError(
                "Format features JSON inconnu."
            )

        self.features = [
            str(feature).strip()
            for feature in self.features
            if str(feature).strip()
        ]

        if MODEL_METADATA_PATH.exists():

            try:

                with open(
                    MODEL_METADATA_PATH,
                    "r",
                    encoding="utf-8",
                ) as file:

                    self.metadata = json.load(
                        file
                    )

            except Exception as exc:

                logger.warning(
                    "Impossible de lire metadata : %s",
                    exc,
                )

        if not self.features:

            raise ValueError(
                "Aucune feature modèle trouvée dans le fichier features/schema."
            )

        unexpected_features = [
            feature
            for feature in self.features
            if feature not in EXPECTED_FEATURES
        ]

        if unexpected_features:

            raise ValueError(
                "Le modèle contient des features inconnues du scanner : "
                + ", ".join(unexpected_features[:20])
            )

        if len(self.features) != 38:

            raise ValueError(
                "V3-C exige exactement 38 features, "
                f"schéma chargé={len(self.features)}."
            )

        model_name = str(
            self.metadata.get("model_name", "")
            if isinstance(self.metadata, Mapping)
            else ""
        ).strip()

        logger.info(
            "Mode production : V3_C_AGGRESSIVE | 38 features strictes"
        )

        logger.info(
            "Modèle chargé : OK"
        )

        logger.info(
            "Features modèle : %d",
            len(self.features),
        )

        logger.info(
            "Fichier features/schema : %s",
            MODEL_FEATURES_PATH,
        )

    # ----------------------------------------------------------------------

    def predict_probability(
        self,
        X: pd.DataFrame,
        allow_missing: bool = False,
    ) -> float:

        if self.model is None:

            raise RuntimeError(
                "Modèle non chargé."
            )

        # FINAL SAFETY GATE
        # ------------------------------------------------------------------

        expected_features = list(self.features)

        if list(
            X.columns
        ) != expected_features:

            raise ValueError(
                "Schema incompatible avec le modèle. "
                f"attendu={len(expected_features)} reçu={X.shape[1]}"
            )

        if X.shape[1] != len(expected_features):

            raise ValueError(
                f"Le modèle exige {len(expected_features)} features, "
                f"reçu={X.shape[1]}"
            )

        if len(X) != 1:

            raise ValueError(
                "Le modèle exige exactement "
                "1 ligne."
            )

        missing_columns = list(
            X.columns[
                X.isna().any()
            ]
        )

        if (
            missing_columns
            and not allow_missing
        ):

            raise ValueError(
                "PRÉDICTION BLOQUÉE : "
                f"features manquantes={missing_columns}"
            )

        # ------------------------------------------------------------------
        # Verify all PRESENT values numeric + finite.
        # In DEGRADED mode, NaN is deliberately preserved so that the
        # SimpleImputer already inside the production sklearn Pipeline
        # performs the imputation.
        # ------------------------------------------------------------------

        for feature in expected_features:

            raw_value = X.iloc[
                0
            ][feature]

            if pd.isna(
                raw_value
            ):

                if allow_missing:
                    continue

                raise ValueError(
                    "PRÉDICTION BLOQUÉE : "
                    f"feature manquante={feature}"
                )

            value = safe_float(
                raw_value
            )

            if value is None:

                raise ValueError(
                    "PRÉDICTION BLOQUÉE : "
                    f"feature non numérique="
                    f"{feature}"
                )

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    "PRÉDICTION BLOQUÉE : "
                    f"feature non finie="
                    f"{feature}"
                )

        probabilities = (
            self.model.predict_proba(
                X
            )
        )

        classes = getattr(
            self.model,
            "classes_",
            None,
        )

        if classes is None:

            classifier = getattr(
                self.model,
                "named_steps",
                {},
            ).get(
                "classifier"
            )

            classes = getattr(
                classifier,
                "classes_",
                None,
            )

        if classes is None:

            raise RuntimeError(
                "Impossible de déterminer les classes "
                "du modèle."
            )

        classes = list(
            classes
        )

        if 1 not in classes:

            raise RuntimeError(
                "La classe positive 1 est absente "
                "du modèle."
            )

        positive_index = classes.index(
            1
        )

        probability = float(
            probabilities[
                0
            ][positive_index]
        )

        if not math.isfinite(
            probability
        ):

            raise RuntimeError(
                "Probabilité modèle non finie."
            )

        if not 0 <= probability <= 1:

            raise RuntimeError(
                "Probabilité modèle invalide : "
                f"{probability}"
            )

        return probability


# ============================================================================
# MACOS NOTIFICATIONS
# ============================================================================

def _escape_applescript_text(
    value: Any,
) -> str:
    """
    Escape text safely for AppleScript string literals.
    """

    return (
        str(
            value
        )
        .replace(
            "\\",
            "\\\\",
        )
        .replace(
            '"',
            '\\"',
        )
        .replace(
            "\n",
            " ",
        )
    )


def send_macos_notification(
    title: str,
    message: str,
    sound: str = "Glass",
) -> bool:
    """
    Send a native macOS notification using osascript.

    Returns True if the command was successfully sent.
    """

    if not MACOS_NOTIFICATIONS_ENABLED:
        return False

    escaped_title = _escape_applescript_text(
        title
    )

    escaped_message = _escape_applescript_text(
        message
    )

    escaped_sound = _escape_applescript_text(
        sound
    )

    script = (
        f'display notification "{escaped_message}" '
        f'with title "{escaped_title}" '
        f'sound name "{escaped_sound}"'
    )

    try:

        subprocess.run(
            [
                "osascript",
                "-e",
                script,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True

    except Exception as exc:

        logger.warning(
            "Notification macOS indisponible : %s",
            exc,
        )

        return False


# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

def send_telegram_notification(
    message: str,
) -> bool:
    """
    Send an alert to the configured Telegram private chat using Telegram Bot API.

    The bot token is never logged. Returns True only when Telegram confirms
    the message was accepted.
    """

    if not TELEGRAM_NOTIFICATIONS_ENABLED:
        return False

    if not TELEGRAM_BOT_TOKEN:
        logger.warning(
            "Notification Telegram impossible : TELEGRAM_BOT_TOKEN manquant."
        )
        return False

    if not TELEGRAM_CHAT_ID:
        logger.warning(
            "Notification Telegram impossible : TELEGRAM_CHAT_ID manquant."
        )
        return False

    endpoint = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": str(message),
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            endpoint,
            data=payload,
            timeout=TELEGRAM_API_TIMEOUT,
        )
        response.raise_for_status()

        body = response.json()
        if not body.get("ok"):
            logger.warning(
                "Notification Telegram refusée par l'API : %s",
                body.get("description", "réponse inconnue"),
            )
            return False

        return True

    except requests.RequestException as exc:
        logger.warning(
            "Notification Telegram indisponible : %s",
            exc,
        )
        return False

    except (ValueError, TypeError) as exc:
        logger.warning(
            "Réponse Telegram invalide : %s",
            exc,
        )
        return False


# ============================================================================
# SIGNAL LOGIC
# ============================================================================

def fair_odds(
    probability: float,
) -> float:

    if probability <= 0:

        return float(
            "inf"
        )

    return 1.0 / probability


def calculate_edge(
    probability: float,
    odds: float,
) -> float:

    return (
        probability
        * odds
        - 1.0
    )


def probability_zone(
    probability: float,
) -> str:

    if probability >= 0.80:
        return "V3C_80_PLUS"

    if probability >= 0.75:
        return "V3C_75_80"

    if probability >= 0.70:
        return "V3C_70_75"

    if probability >= 0.65:
        return "V3C_65_70_OBSERVATION"

    if probability >= 0.60:
        return "V3C_PRIORITY_60_65"

    if probability >= 0.55:
        return "V3C_PRIORITY_55_60"

    if probability >= 0.50:
        return "V3C_50_55_OBSERVATION"

    return "V3C_BELOW_50"


def signal_decision(
    probability: float,
    odds: float,
) -> str:

    if odds <= 1:

        return "IGNORE"

    break_even = (
        1.0 / odds
    )

    if probability < (
        MIN_DISPLAY_PROBABILITY
    ):

        return "IGNORE"

    if probability >= break_even:

        return "VALUE"

    return "IGNORE"


# ============================================================================
# LIVE SCANNER
# ============================================================================

class LiveScanner:

    def __init__(
        self,
        client: StatsAPIClient,
        model: ProductionModel,
    ):

        self.client = client
        self.model = model

        self.pre_match_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self.last_processed_minute: Dict[
            str,
            int,
        ] = {}

        self.expired_matches: Set[
            str
        ] = set()

        self.run_count = 0

        # Matchs déjà préchargés à 83'-84'. Le PRE-MATCH reste ensuite dans
        # pre_match_cache et sera réutilisé instantanément à 85'-87'.
        self.preloaded_match_ids: Set[str] = set()

        # Mémoire locale de "readiness" LIVE observée lors d'une analyse
        # précédente (85' ou 86'). Elle permet de prioriser à 87' les matchs
        # dont les features LIVE modèle étaient déjà complètes, SANS ajouter
        # de nouvel appel API uniquement pour classer la file.
        #
        # Format:
        #   match_id -> {
        #       "minute": int,
        #       "available": int,
        #       "total": int,
        #       "complete": bool,
        #   }
        self.live_readiness_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # Une seule analyse lourde par match/minute. À 87', les retries sont
        # réalisés immédiatement à l'intérieur de analyze_match().
        self.analysis_attempted_keys: Set[
            Tuple[str, int]
        ] = set()

        # Les caches sont partagés entre les workers.
        self._state_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._pre_match_locks: Dict[str, threading.Lock] = {}

        # Persistent deduplication for the historical snapshot collector.
        # We save every valid 85/86/87 snapshot, not only alerts, so future
        # datasets are not biased toward high-probability matches.
        self.history_snapshot_keys: Set[
            Tuple[str, int]
        ] = set()

        if (
            HISTORY_CAPTURE_ENABLED
            and HISTORY_SNAPSHOTS_PATH.exists()
        ):

            try:

                existing_history = pd.read_csv(
                    HISTORY_SNAPSHOTS_PATH,
                    usecols=[
                        "match_id",
                        "minute",
                    ],
                )

                for row in existing_history.itertuples(
                    index=False
                ):

                    try:
                        self.history_snapshot_keys.add(
                            (
                                str(row.match_id),
                                int(row.minute),
                            )
                        )
                    except (TypeError, ValueError):
                        continue

                logger.info(
                    "Historique snapshots chargé : %d clé(s)",
                    len(self.history_snapshot_keys),
                )

            except Exception as exc:

                logger.warning(
                    "Historique snapshots existant illisible : %s",
                    exc,
                )

    # ----------------------------------------------------------------------
    # CANDIDATES
    # ----------------------------------------------------------------------

    def find_candidates(
        self,
    ) -> Tuple[
        List[LiveMatch],
        List[LiveMatch],
    ]:

        raw_matches = (
            self.client.get_live_matches()
        )

        candidates: List[
            LiveMatch
        ] = []

        preload_candidates: List[
            LiveMatch
        ] = []

        minute_distribution: Dict[
            int,
            int,
        ] = {}

        unparsed_minutes = 0

        for raw in raw_matches:

            match = parse_live_match(
                raw
            )

            if match is None:

                unparsed_minutes += 1
                continue

            minute_distribution[
                match.minute
            ] = (
                minute_distribution.get(
                    match.minute,
                    0,
                )
                + 1
            )

            if (
                PRELOAD_MINUTE_START
                <= match.minute
                <= PRELOAD_MINUTE_END
                and match.match_id
                not in self.preloaded_match_ids
                and match.match_id
                not in self.pre_match_cache
            ):
                preload_candidates.append(
                    match
                )

            if (
                match.minute
                < MIN_SCAN_MINUTE
            ):
                continue

            if (
                match.minute
                > MAX_SCAN_MINUTE
            ):

                self.expired_matches.add(
                    match.match_id
                )

                continue

            candidates.append(
                match
            )

        logger.info(
            "Matchs LIVE récupérés : %d",
            len(raw_matches),
        )

        if minute_distribution:

            logger.info(
                "Distribution minutes LIVE : %s",
                sorted(
                    minute_distribution.items()
                ),
            )

        if unparsed_minutes:

            logger.warning(
                "Matchs LIVE sans minute exploitable : %d",
                unparsed_minutes,
            )

        logger.info(
            "Candidats analyse %d-%d' : %d",
            MIN_SCAN_MINUTE,
            MAX_SCAN_MINUTE,
            len(candidates),
        )

        logger.info(
            "Candidats PRELOAD %d-%d' : %d",
            PRELOAD_MINUTE_START,
            PRELOAD_MINUTE_END,
            len(preload_candidates),
        )

        return (
            candidates,
            preload_candidates,
        )

    # ----------------------------------------------------------------------
    # PRELOAD 83'-84'
    # ----------------------------------------------------------------------

    def preload_pre_match(
        self,
        matches: Sequence[LiveMatch],
    ) -> None:
        """
        Prépare le PRE-MATCH des rencontres à 83'-84' sans lancer le modèle.

        Le preload est volontairement traité APRES les candidats 85'-87' du
        scan courant : il ne doit jamais ralentir une analyse urgente à 87'.
        """

        if not matches:
            return

        # 84' avant 83' : le match le plus proche de la fenêtre modèle passe
        # en premier.
        ordered = sorted(
            matches,
            key=lambda match: (
                -match.minute,
                match.match_id,
            ),
        )

        for match in ordered:

            with self._state_lock:
                already_ready = (
                    match.match_id in self.preloaded_match_ids
                    or match.match_id in self.pre_match_cache
                )

            if already_ready:
                continue

            try:
                diagnostics: Dict[str, Any] = {}

                logger.info(
                    "PRELOAD PRE-MATCH | %s vs %s | %d'",
                    match.home_team,
                    match.away_team,
                    match.minute,
                )

                features = self.get_pre_match_features(
                    match,
                    diagnostics=diagnostics,
                )

                coverage = sum(
                    features.get(feature) is not None
                    for feature in PRE_MATCH_FEATURES
                )

                with self._state_lock:
                    self.preloaded_match_ids.add(
                        match.match_id
                    )

                logger.info(
                    "PRELOAD TERMINÉ | %s | PRE=%d/%d",
                    match.match_id,
                    coverage,
                    len(PRE_MATCH_FEATURES),
                )

            except Exception as exc:
                logger.warning(
                    "PRELOAD ÉCHOUÉ | %s vs %s | %d' | %s",
                    match.home_team,
                    match.away_team,
                    match.minute,
                    exc,
                )

    # ----------------------------------------------------------------------
    # PRE-MATCH CACHE
    # ----------------------------------------------------------------------

    def get_pre_match_features(
        self,
        match: LiveMatch,
        diagnostics: Optional[
            Dict[str, Any]
        ] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:

        # Empêche deux workers de reconstruire simultanément le PRE-MATCH du
        # même match. Les matchs différents restent parallélisés.
        with self._state_lock:
            lock = self._pre_match_locks.setdefault(
                match.match_id,
                threading.Lock(),
            )

        with lock:
            cached = self.pre_match_cache.get(
                match.match_id
            )

            if cached is not None and not force_rebuild:
                logger.info(
                    "PRE-MATCH: cache HIT | %d/%d",
                    sum(cached.get(f) is not None for f in PRE_MATCH_FEATURES),
                    len(PRE_MATCH_FEATURES),
                )
                return cached

            logger.info(
                "PRE-MATCH: %s",
                "FORCE REBUILD" if force_rebuild else "cache MISS",
            )

            features = build_real_pre_match_features(
                match,
                self.client,
                diagnostics=diagnostics,
            )

            self.pre_match_cache[match.match_id] = features
            return features

    # ----------------------------------------------------------------------
    # SAME-MINUTE PRIORITY
    # ----------------------------------------------------------------------

    def _pre_match_cache_coverage(
        self,
        match_id: str,
    ) -> int:

        cached = self.pre_match_cache.get(
            match_id
        )

        if not cached:
            return 0

        return sum(
            cached.get(feature) is not None
            for feature in PRE_MATCH_FEATURES
        )

    def _same_minute_priority_key(
        self,
        match: LiveMatch,
    ) -> Tuple[
        int,
        int,
        int,
        int,
        str,
    ]:
        """
        Priorité à l'intérieur d'une même minute.

        1) PRE-MATCH déjà préchargé / en cache
        2) LIVE modèle déjà observé complet lors d'une minute précédente
        3) meilleure couverture PRE locale
        4) meilleure couverture LIVE observée
        5) match_id pour un ordre déterministe

        IMPORTANT:
        Aucun appel API supplémentaire n'est fait pour calculer cette priorité.
        """

        preloaded = int(
            match.match_id in self.preloaded_match_ids
            or match.match_id in self.pre_match_cache
        )

        pre_coverage = self._pre_match_cache_coverage(
            match.match_id
        )

        live_info = self.live_readiness_cache.get(
            match.match_id,
            {},
        )

        live_complete = int(
            bool(
                live_info.get(
                    "complete",
                    False,
                )
            )
        )

        live_available = safe_int(
            live_info.get(
                "available",
                0,
            )
        ) or 0

        # sorted() est ascendant : valeurs négatives = meilleure priorité.
        return (
            -preloaded,
            -live_complete,
            -pre_coverage,
            -live_available,
            str(match.match_id),
        )

    def _log_priority_batch(
        self,
        minute: int,
        batch: Sequence[LiveMatch],
    ) -> None:

        for rank, match in enumerate(
            batch,
            start=1,
        ):

            preloaded = (
                match.match_id in self.preloaded_match_ids
                or match.match_id in self.pre_match_cache
            )

            pre_coverage = self._pre_match_cache_coverage(
                match.match_id
            )

            live_info = self.live_readiness_cache.get(
                match.match_id,
                {},
            )

            live_complete = bool(
                live_info.get(
                    "complete",
                    False,
                )
            )

            live_available = safe_int(
                live_info.get(
                    "available",
                    0,
                )
            ) or 0

            live_total = safe_int(
                live_info.get(
                    "total",
                    0,
                )
            ) or 0

            logger.info(
                "PRIORITY %d' #%d | %s vs %s | PRELOAD=%s | PRE=%d/%d | LIVE_READY=%s | LIVE_PREV=%d/%d",
                minute,
                rank,
                match.home_team,
                match.away_team,
                "YES" if preloaded else "NO",
                pre_coverage,
                len(PRE_MATCH_FEATURES),
                "YES" if live_complete else "NO",
                live_available,
                live_total,
            )

    # ----------------------------------------------------------------------
    # ANALYZE MATCH
    # ----------------------------------------------------------------------

    def analyze_match(
        self,
        match: LiveMatch,
    ) -> Optional[Prediction]:

        logger.info(
            "=" * 80
        )

        logger.info(
            "ANALYSE %s vs %s | %d'",
            match.home_team,
            match.away_team,
            match.minute,
        )

        if (
            match.home_score is None
            or match.away_score is None
        ):

            logger.error(
                "MODEL BLOCKED | score indisponible "
                "pour %s",
                match.match_id,
            )

            logger.error(
                "AUCUNE PRÉDICTION EFFECTUÉE."
            )

            return None

        logger.info(
            "Score : %d-%d",
            match.home_score,
            match.away_score,
        )

        # ------------------------------------------------------------------
        # Detailed diagnostics object
        # ------------------------------------------------------------------

        diagnostics: Dict[
            str,
            Any,
        ] = {}

        # ------------------------------------------------------------------
        # PRE-MATCH
        # ------------------------------------------------------------------

        pre_match_features = (
            self.get_pre_match_features(
                match,
                diagnostics=diagnostics,
            )
        )

        # ------------------------------------------------------------------
        # BUILD 96 FEATURES
        # ------------------------------------------------------------------

        model_features = list(self.model.features)

        X, raw_features = build_features(
            match,
            self.client,
            pre_match_features=(
                pre_match_features
            ),
            diagnostics=diagnostics,
            required_features=model_features,
        )

        # ------------------------------------------------------------------
        # COVERAGE / MISSING
        # ------------------------------------------------------------------

        # Diagnostic complet du pool source 96 features.
        # La décision V3-C est prise uniquement sur les 38 features du schéma
        # V3_C_AGGRESSIVE chargé.
        missing_all = (
            diagnose_missing_features(
                raw_features,
                match_id=match.match_id,
                diagnostics=diagnostics,
            )
        )

        available_all = 96 - len(missing_all)

        pre_available = sum(
            raw_features.get(feature) is not None
            for feature in PRE_MATCH_FEATURES
        )

        live_available = sum(
            raw_features.get(feature) is not None
            for feature in LIVE_FEATURES
        )

        model_feature_count = len(model_features)

        missing = [
            feature
            for feature in model_features
            if (
                feature not in raw_features
                or is_missing(raw_features.get(feature))
            )
        ]

        available = model_feature_count - len(missing)

        model_pre_available = sum(
            raw_features.get(feature) is not None
            for feature in model_features
            if feature in PRE_MATCH_FEATURES
        )

        model_live_available = sum(
            raw_features.get(feature) is not None
            for feature in model_features
            if feature in LIVE_FEATURES
        )

        model_pre_total = sum(
            1
            for feature in model_features
            if feature in PRE_MATCH_FEATURES
        )

        model_live_total = sum(
            1
            for feature in model_features
            if feature in LIVE_FEATURES
        )

        logger.info(
            "Features disponibles 96 brut : %d/96",
            available_all,
        )

        logger.info(
            "PRE-MATCH brut : %d/%d",
            pre_available,
            len(PRE_MATCH_FEATURES),
        )

        logger.info(
            "LIVE brut : %d/%d",
            live_available,
            len(LIVE_FEATURES),
        )

        logger.info(
            "Features modèle disponibles : %d/%d",
            available,
            model_feature_count,
        )

        logger.info(
            "PRE-MATCH modèle : %d/%d",
            model_pre_available,
            model_pre_total,
        )

        logger.info(
            "LIVE modèle : %d/%d",
            model_live_available,
            model_live_total,
        )

        # ------------------------------------------------------------------
        # MODEL COVERAGE GATE V3-C
        #
        # Production strict:
        #   - exact V3-C schema = 38 features
        #   - 38/38 required
        #   - no degraded prediction
        #   - no silent imputation
        # ------------------------------------------------------------------

        critical_missing = [
            feature
            for feature in missing
            if feature in CRITICAL_MODEL_FEATURES
        ]

        if model_feature_count != 38:
            raise RuntimeError(
                f"V3-C schema invalide : {model_feature_count}/38 features."
            )

        required_model_coverage = model_feature_count

        # ------------------------------------------------------------------
        # PRIORITÉ 87' : retries LIVE immédiats, sans attendre le prochain scan.
        # On peut aussi reconstruire une fois le PRE-MATCH s'il est incomplet.
        # ------------------------------------------------------------------

        if (
            match.minute == 87
            and (
                available < required_model_coverage
                or critical_missing
            )
            and MAX_87_LIVE_RETRIES > 0
        ):
            cached_pre_coverage = sum(
                pre_match_features.get(feature) is not None
                for feature in PRE_MATCH_FEATURES
            )

            if cached_pre_coverage < PRE_MATCH_REBUILD_MIN_COVERAGE:
                logger.warning(
                    "87' PRIORITY | PRE-MATCH cache incomplet=%d/%d | reconstruction immédiate",
                    cached_pre_coverage,
                    len(PRE_MATCH_FEATURES),
                )
                diagnostics = {}
                pre_match_features = self.get_pre_match_features(
                    match,
                    diagnostics=diagnostics,
                    force_rebuild=True,
                )

            for retry_index in range(1, MAX_87_LIVE_RETRIES + 1):
                logger.warning(
                    "87' PRIORITY | retry LIVE immédiat %d/%d | %s vs %s",
                    retry_index,
                    MAX_87_LIVE_RETRIES,
                    match.home_team,
                    match.away_team,
                )

                retry_diagnostics: Dict[str, Any] = {}
                X_retry, raw_retry = build_features(
                    match,
                    self.client,
                    pre_match_features=pre_match_features,
                    diagnostics=retry_diagnostics,
                    required_features=model_features,
                )

                retry_missing = [
                    feature
                    for feature in model_features
                    if (
                        feature not in raw_retry
                        or is_missing(raw_retry.get(feature))
                    )
                ]
                retry_available = model_feature_count - len(retry_missing)
                retry_critical_missing = [
                    feature
                    for feature in retry_missing
                    if feature in CRITICAL_MODEL_FEATURES
                ]

                # Toujours garder la tentative ayant la meilleure couverture.
                if retry_available >= available:
                    X = X_retry
                    raw_features = raw_retry
                    diagnostics = retry_diagnostics
                    missing = retry_missing
                    available = retry_available
                    critical_missing = retry_critical_missing
                    available_all = sum(
                        raw_features.get(feature) is not None
                        for feature in EXPECTED_FEATURES
                    )
                    pre_available = sum(
                        raw_features.get(feature) is not None
                        for feature in PRE_MATCH_FEATURES
                    )
                    live_available = sum(
                        raw_features.get(feature) is not None
                        for feature in LIVE_FEATURES
                    )
                    model_pre_available = sum(
                        raw_features.get(feature) is not None
                        for feature in model_features
                        if feature in PRE_MATCH_FEATURES
                    )
                    model_live_available = sum(
                        raw_features.get(feature) is not None
                        for feature in model_features
                        if feature in LIVE_FEATURES
                    )

                if (
                    available >= required_model_coverage
                    and not critical_missing
                ):
                    logger.info(
                        "87' PRIORITY | retry réussi | couverture=%d/%d",
                        available,
                        model_feature_count,
                    )
                    break

        # Mémorise la qualité LIVE réellement observée pour pouvoir donner
        # une priorité plus intelligente au même match lorsqu'il atteindra
        # la minute suivante, notamment 87'.
        with self._state_lock:
            self.live_readiness_cache[
                match.match_id
            ] = {
                "minute":
                    match.minute,
                "available":
                    model_live_available,
                "total":
                    model_live_total,
                "complete":
                    (
                        model_live_total > 0
                        and model_live_available
                        == model_live_total
                    ),
            }

        prediction_quality = (
            "FULL_V3C"
            if not missing
            else "BLOCKED_V3C"
        )

        if available < required_model_coverage or critical_missing:

            if available < required_model_coverage:
                logger.error(
                    "MODEL BLOCKED | couverture modèle minimale=%d/%d | actuel=%d/%d",
                    required_model_coverage,
                    model_feature_count,
                    available,
                    model_feature_count,
                )

            if critical_missing:
                logger.error(
                    "MODEL BLOCKED | feature(s) critique(s) modèle manquante(s) : %s",
                    ", ".join(critical_missing),
                )

            return self._blocked_prediction_log(
                match,
                missing,
                model_pre_available,
                model_live_available,
                diagnostics=diagnostics,
                feature_count=model_feature_count,
                raw_96_coverage=available_all,
            )

        # ------------------------------------------------------------------
        # Build exact model input from the loaded model schema.
        # ------------------------------------------------------------------

        X_model = X.loc[:, model_features].copy()

        if missing:
            # This branch should already have been blocked by the strict 38/38
            # gate above. Fail closed if execution ever reaches it.
            logger.error(
                "V3-C STRICT GATE FAILED | missing=%s",
                ", ".join(missing),
            )
            return None

        numeric_X = X_model.apply(
            pd.to_numeric,
            errors="coerce",
        )

        actual_nan = set(
            numeric_X.columns[
                numeric_X.isna().any()
            ]
        )

        expected_nan: Set[str] = set()
        present_values = numeric_X.to_numpy(dtype=float)

        final_gate_failed = (
            X_model.shape[1] != model_feature_count
            or list(X_model.columns) != model_features
            or actual_nan != expected_nan
            or not np.isfinite(present_values).all()
        )

        if final_gate_failed:
            logger.error(
                "FINAL MODEL GATE FAILED | expected_nan=%s | actual_nan=%s",
                sorted(expected_nan),
                sorted(actual_nan),
            )
            return None

        # V3-C uses its own exact 38-feature schema. The legacy external
        # validator is a 96-feature validator and must not be allowed to reject a
        # valid V3-C row merely because 58 unused source features are absent.
        validation = ValidationResult(
            status="FULL_V3C",
            ready_for_model=(
                available == model_feature_count
                and model_feature_count == 38
                and not missing
            ),
            errors=[],
            warnings=[],
            missing=[],
            coverage=available,
        )

        logger.info(
            "Validator modèle : status=%s | coverage=%d/%d | ready=%s",
            validation.status,
            validation.coverage,
            model_feature_count,
            validation.ready_for_model,
        )

        if validation.errors:
            for error in validation.errors:
                logger.error(
                    "Validator error : %s",
                    error,
                )

        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(
                    "Validator warning : %s",
                    warning,
                )

        if not validation.ready_for_model:
            logger.error(
                "PRÉDICTION BLOQUÉE PAR LE VALIDATOR."
            )
            logger.error(
                "READY FOR MODEL : NO"
            )
            return None

        # ------------------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------------------

        probability = (
            self.model.predict_probability(
                X_model,
                allow_missing=False,
            )
        )

        odds = SIMULATED_ODDS

        fair = fair_odds(
            probability
        )

        edge = calculate_edge(
            probability,
            odds,
        )

        zone = probability_zone(
            probability
        )

        decision = signal_decision(
            probability,
            odds,
        )


        alert = (
            PRIORITY_PROBABILITY_MIN
            <= probability
            < PRIORITY_PROBABILITY_MAX
        )

        prediction = Prediction(
            match_id=match.match_id,

            home_team=match.home_team,

            away_team=match.away_team,

            minute=match.minute,

            score_home=match.home_score,

            score_away=match.away_score,

            probability=probability,

            fair_odds=fair,

            simulated_odds=odds,

            edge=edge,

            zone=zone,

            decision=decision,

            validation_status=(
                validation.status
            ),

            coverage=(
                validation.coverage
            ),

            feature_count=model_feature_count,

            prediction_quality=(
                prediction_quality
            ),

            missing_features=list(
                missing
            ),

            alert=alert,

            alert_threshold=(
                ALERT_PROBABILITY
            ),
        )

        self.save_history_snapshot(
            match=match,
            raw_features=raw_features,
            prediction=prediction,
        )

        logger.info(
            "P(goal_after_87) = %.4f",
            probability,
        )

        logger.info(
            "Fair odds = %.2f",
            fair,
        )

        logger.info(
            "Simulated odds = %.2f",
            odds,
        )

        logger.info(
            "Edge = %.2f%%",
            edge * 100,
        )

        logger.info(
            "Zone = %s",
            zone,
        )

        logger.info(
            "Decision = %s",
            decision,
        )


        logger.info(
            "Prediction quality = %s | coverage=%d/%d",
            prediction_quality,
            validation.coverage,
            model_feature_count,
        )

        if missing:

            logger.info(
                "Imputed by model pipeline = %s",
                ", ".join(
                    missing
                ),
            )


        if alert:

            logger.warning(
                "🚨 V3-C PRIORITY %.1f%%-<%.1f%% | %s vs %s | %d' | "
                "Score %d-%d | P(goal_after_87)=%.2f%% | "
                "Quality=%s %d/%d | Decision=%s",
                PRIORITY_PROBABILITY_MIN * 100.0,
                PRIORITY_PROBABILITY_MAX * 100.0,
                match.home_team,
                match.away_team,
                match.minute,
                match.home_score,
                match.away_score,
                probability * 100.0,
                prediction_quality,
                validation.coverage,
                model_feature_count,
                decision,
            )

            notification_sent = send_macos_notification(
                title=(
                    f"🚨 V3-C Goal After 87 | "
                    f"{PRIORITY_PROBABILITY_MIN * 100.0:.0f}-"
                    f"<{PRIORITY_PROBABILITY_MAX * 100.0:.0f}%"
                ),
                message=(
                    f"{match.home_team} "
                    f"{match.home_score}-"
                    f"{match.away_score} "
                    f"{match.away_team} | "
                    f"{match.minute}' | "
                    f"P={probability * 100.0:.1f}% | "
                    f"{prediction_quality} "
                    f"{validation.coverage}/{model_feature_count} | "
                    f"{decision}"
                ),
                sound="Glass",
            )

            logger.info(
                "Notification macOS = %s",
                (
                    "ENVOYÉE"
                    if notification_sent
                    else "NON ENVOYÉE"
                ),
            )

            break_even = (
                1.0 / odds
                if odds > 1
                else float("inf")
            )

            telegram_message = (
                "🚨 GOAL AFTER 87 — V3-C PRIORITY\n\n"
                f"{match.home_team} {match.home_score}-"
                f"{match.away_score} {match.away_team}\n"
                f"Minute : {match.minute}'\n"
                f"Probabilité : {probability * 100.0:.2f}%\n"
                f"Fair odds : {fair:.2f}\n"
                f"Cote simulée : {odds:.2f}\n"
                f"Break-even : {break_even * 100.0:.2f}%\n"
                f"Edge : {edge * 100.0:+.2f}%\n"
                f"Zone : {zone}\n"
                f"Qualité : {prediction_quality} "
                f"{validation.coverage}/{model_feature_count}\n"
                f"Décision : {decision}"
            )

            telegram_sent = send_telegram_notification(
                telegram_message
            )

            logger.info(
                "Notification Telegram = %s",
                (
                    "ENVOYÉE"
                    if telegram_sent
                    else "NON ENVOYÉE"
                ),
            )

            # Terminal bell when supported by the terminal.
            print("\a", end="", flush=True)

        return prediction

    # ----------------------------------------------------------------------
    # HISTORICAL SNAPSHOT COLLECTOR
    # ----------------------------------------------------------------------

    def save_history_snapshot(
        self,
        match: LiveMatch,
        raw_features: Mapping[str, Any],
        prediction: Prediction,
    ) -> None:

        if not HISTORY_CAPTURE_ENABLED:
            return

        key = (
            str(match.match_id),
            int(match.minute),
        )

        if key in self.history_snapshot_keys:

            logger.debug(
                "HISTORY snapshot déjà sauvegardé : %s | %d'",
                match.match_id,
                match.minute,
            )

            return

        HISTORY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # IMPORTANT: model_probability / alert / decision are metadata only.
        # They must NEVER become model features when a future dataset is built.
        row: Dict[str, Any] = {
            "snapshot_timestamp_utc":
                pd.Timestamp.utcnow().isoformat(),

            "schema_version":
                HISTORY_SCHEMA_VERSION,

            "match_id":
                match.match_id,

            "competition_id":
                match.competition_id,

            "season_id":
                match.season_id,

            "home_team":
                match.home_team,

            "away_team":
                match.away_team,

            "minute":
                match.minute,

            "score_home_snapshot":
                match.home_score,

            "score_away_snapshot":
                match.away_score,
        }

        # Save the exact 96 model inputs in their exact production order.
        # Missing tolerated values remain blank/NaN; they are never replaced
        # silently by zero.
        for feature in EXPECTED_FEATURES:
            row[feature] = raw_features.get(
                feature,
                np.nan,
            )

        row.update(
            {
                "model_probability":
                    prediction.probability,

                "model_fair_odds":
                    prediction.fair_odds,

                "simulated_odds":
                    prediction.simulated_odds,

                "model_edge":
                    prediction.edge,

                "probability_zone":
                    prediction.zone,

                "decision":
                    prediction.decision,

                "alert":
                    prediction.alert,

                "alert_threshold":
                    prediction.alert_threshold,

                "prediction_quality":
                    prediction.prediction_quality,

                "coverage":
                    prediction.coverage,

                "missing_features":
                    "|".join(
                        prediction.missing_features
                    ),

                # Filled later from the finalized match history/timeline.
                # Keeping the target pending prevents accidental leakage.
                "goal_after_87":
                    np.nan,

                "target_status":
                    "PENDING",
            }
        )

        metadata_columns = [
            "snapshot_timestamp_utc",
            "schema_version",
            "match_id",
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            "minute",
            "score_home_snapshot",
            "score_away_snapshot",
        ]

        model_metadata_columns = [
            "model_probability",
            "model_fair_odds",
            "simulated_odds",
            "model_edge",
            "probability_zone",
            "decision",
            "alert",
            "alert_threshold",
            "prediction_quality",
            "coverage",
            "missing_features",
            "goal_after_87",
            "target_status",
        ]

        ordered_columns = (
            metadata_columns
            + EXPECTED_FEATURES
            + model_metadata_columns
        )

        try:

            # Plusieurs workers peuvent terminer presque simultanément.
            # Le verrou évite deux append concurrents dans le même CSV.
            with self._history_lock:
                if key in self.history_snapshot_keys:
                    return

                history_df = pd.DataFrame(
                    [row],
                    columns=ordered_columns,
                )

                file_exists = (
                    HISTORY_SNAPSHOTS_PATH.exists()
                )

                history_df.to_csv(
                    HISTORY_SNAPSHOTS_PATH,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )

                self.history_snapshot_keys.add(
                    key
                )

            logger.info(
                "HISTORY snapshot sauvegardé : %s | %d' | %s | %d/96",
                match.match_id,
                match.minute,
                prediction.prediction_quality,
                prediction.coverage,
            )

        except Exception as exc:

            # Historical capture must never stop the live scanner.
            logger.error(
                "HISTORY snapshot non sauvegardé : %s",
                exc,
            )

    # ----------------------------------------------------------------------
    # BLOCKED DIAGNOSTIC
    # ----------------------------------------------------------------------

    def _blocked_prediction_log(
        self,
        match: LiveMatch,
        missing: Sequence[str],
        pre_available: int,
        live_available: int,
        diagnostics: Optional[
            Mapping[str, Any]
        ] = None,
        feature_count: int = 96,
        raw_96_coverage: Optional[int] = None,
    ) -> None:

        logger.error(
            "MODEL BLOCKED | %s vs %s | "
            "%d' | model_coverage=%d/%d | raw_96=%s",
            match.home_team,
            match.away_team,
            match.minute,
            feature_count - len(missing),
            feature_count,
            (
                f"{raw_96_coverage}/96"
                if raw_96_coverage is not None
                else "?"
            ),
        )

        logger.error(
            "PRE-MATCH modèle : %d",
            pre_available,
        )

        logger.error(
            "LIVE modèle : %d",
            live_available,
        )

        logger.error(
            "MISSING FEATURES :"
        )

        for feature in missing:

            source = FEATURE_SOURCE_MAP.get(
                feature,
                "unknown",
            )

            detail = {}

            if diagnostics is not None:

                candidate = diagnostics.get(
                    feature
                )

                if isinstance(
                    candidate,
                    Mapping,
                ):

                    detail = dict(
                        candidate
                    )

            status = detail.get(
                "status",
                "missing",
            )

            endpoint = detail.get(
                "endpoint"
            )

            path = detail.get(
                "path"
            )

            bookmaker = detail.get(
                "bookmaker"
            )

            logger.error(
                "  - %s | source=%s | status=%s",
                feature,
                source,
                status,
            )

            if endpoint:

                logger.error(
                    "      endpoint=%s",
                    endpoint,
                )

            if path:

                logger.error(
                    "      path=%s",
                    path,
                )

            if bookmaker:

                logger.error(
                    "      bookmaker=%s",
                    bookmaker,
                )

        logger.error(
            "AUCUNE PRÉDICTION EFFECTUÉE."
        )

        return None

    # ----------------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------------

    def save_predictions(
        self,
        predictions: Sequence[
            Prediction
        ],
    ) -> None:

        if not predictions:
            return

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for prediction in predictions:

            rows.append(
                {
                    "timestamp":
                        pd.Timestamp.utcnow(),

                    "model_version":
                        "V3_C_AGGRESSIVE",

                    "match_id":
                        prediction.match_id,

                    "home_team":
                        prediction.home_team,

                    "away_team":
                        prediction.away_team,

                    "minute":
                        prediction.minute,

                    "score_home":
                        prediction.score_home,

                    "score_away":
                        prediction.score_away,

                    "probability":
                        prediction.probability,

                    "fair_odds":
                        prediction.fair_odds,

                    "simulated_odds":
                        prediction.simulated_odds,

                    "edge":
                        prediction.edge,

                    "zone":
                        prediction.zone,

                    "decision":
                        prediction.decision,

                    "validation_status":
                        prediction.validation_status,

                    "coverage":
                        prediction.coverage,

                    "feature_count":
                        prediction.feature_count,

                    "prediction_quality":
                        prediction.prediction_quality,

                    "missing_features":
                        "|".join(
                            prediction.missing_features
                        ),


                    "alert":
                        prediction.alert,

                    "alert_threshold":
                        prediction.alert_threshold,

                    "priority_probability_min":
                        PRIORITY_PROBABILITY_MIN,

                    "priority_probability_max_exclusive":
                        PRIORITY_PROBABILITY_MAX,

                    "is_priority_55_65":
                        prediction.alert,
                }
            )

        df = pd.DataFrame(
            rows
        )

        file_exists = (
            PREDICTIONS_PATH.exists()
        )

        df.to_csv(
            PREDICTIONS_PATH,
            mode="a",
            header=not file_exists,
            index=False,
        )

        logger.info(
            "Predictions sauvegardées : %s",
            PREDICTIONS_PATH,
        )

    # ----------------------------------------------------------------------
    # DISPLAY
    # ----------------------------------------------------------------------

    def display_results(
        self,
        predictions: Sequence[
            Prediction
        ],
    ) -> None:

        if not predictions:

            logger.info(
                "Aucune prédiction valide à afficher."
            )

            return

        print()
        print(
            "=" * 100
        )

        print(
            "LIVE SCANNER V3-C - RESULTS"
        )

        print(
            "=" * 100
        )

        for index, prediction in enumerate(
            predictions,
            start=1,
        ):

            print(
                f"{index:02d}. "
                f"{prediction.home_team} "
                f"{prediction.score_home}-"
                f"{prediction.score_away} "
                f"{prediction.away_team} | "
                f"{prediction.minute}' | "
                f"P={prediction.probability:.2%} | "
                f"Fair={prediction.fair_odds:.2f} | "
                f"Edge={prediction.edge:.2%} | "
                f"{prediction.zone} | "
                f"{prediction.decision} | "
                f"{prediction.prediction_quality} "
                f"{prediction.coverage}/{prediction.feature_count}"
                + (
                    f" | 🚨 V3-C PRIORITY "
                    f"{PRIORITY_PROBABILITY_MIN * 100.0:.0f}-"
                    f"<{PRIORITY_PROBABILITY_MAX * 100.0:.0f}%"
                    if prediction.alert
                    else ""
                )
            )

        print()

        print(
            f"Cote simulée : "
            f"{SIMULATED_ODDS:.2f}"
        )

        print(
            f"Break-even : "
            f"{1 / SIMULATED_ODDS:.2%}"
        )

        alerts = [
            prediction
            for prediction in predictions
            if prediction.alert
        ]

        if alerts:

            print()
            print(
                f"🚨 V3-C PRIORITÉ : "
                f"{PRIORITY_PROBABILITY_MIN * 100.0:.0f}% <= P < "
                f"{PRIORITY_PROBABILITY_MAX * 100.0:.0f}%"
            )
            print("-" * 100)

            for alert_prediction in alerts:

                print(
                    f"🚨 {alert_prediction.home_team} "
                    f"{alert_prediction.score_home}-"
                    f"{alert_prediction.score_away} "
                    f"{alert_prediction.away_team} | "
                    f"{alert_prediction.minute}' | "
                    f"P={alert_prediction.probability:.2%} | "
                    f"{alert_prediction.prediction_quality} "
                    f"{alert_prediction.coverage}/{alert_prediction.feature_count} | "
                    f"{alert_prediction.decision}"
                )

            print("-" * 100)

        print(
            "=" * 100
        )

        print()

    # ----------------------------------------------------------------------
    # SINGLE SCAN
    # ----------------------------------------------------------------------

    def scan_once(
        self,
    ) -> List[Prediction]:

        self.run_count += 1

        logger.info(
            ""
        )

        logger.info(
            "########################################"
        )

        logger.info(
            "SCAN #%d",
            self.run_count,
        )

        logger.info(
            "########################################"
        )

        candidates, preload_candidates = self.find_candidates()

        # File de priorité stricte : 87' > 86' > 85'.
        # On ne soumet plus tous les matchs dans une seule longue file au pool.
        candidates = sorted(
            candidates,
            key=lambda match: (
                -match.minute,
                match.match_id,
            ),
        )

        eligible: List[LiveMatch] = []

        with self._state_lock:
            for match in candidates:
                key = (
                    str(match.match_id),
                    int(match.minute),
                )

                if key in self.analysis_attempted_keys:
                    logger.debug(
                        "Analyse lourde déjà tentée à %d' : %s",
                        match.minute,
                        match.match_id,
                    )
                    continue

                # Réservation de la clé AVANT soumission au pool : même si la
                # prédiction échoue, 85'/86' ne seront pas retraités en boucle.
                self.analysis_attempted_keys.add(key)
                eligible.append(match)

        if eligible:
            logger.info(
                "Ordonnancement analyse : %d match(s) | priorité=87>86>85 | workers=%d",
                len(eligible),
                SCANNER_WORKERS,
            )

        predictions: List[Prediction] = []

        if eligible:
            # Traitement par niveau de priorité. Un match à 85' ne peut donc
            # plus être placé devant un match à 87' dans la file des workers.
            priority_minutes = sorted(
                {
                    match.minute
                    for match in eligible
                },
                reverse=True,
            )

            for priority_minute in priority_minutes:

                batch = [
                    match
                    for match in eligible
                    if match.minute == priority_minute
                ]

                if not batch:
                    continue

                # À minute égale, on ne fait plus un ordre arbitraire:
                # PRELOAD/cache d'abord, puis LIVE déjà observé complet,
                # puis meilleure couverture locale connue.
                batch = sorted(
                    batch,
                    key=self._same_minute_priority_key,
                )

                self._log_priority_batch(
                    priority_minute,
                    batch,
                )

                worker_count = max(
                    1,
                    min(
                        SCANNER_WORKERS,
                        len(batch),
                    ),
                )

                logger.info(
                    "PRIORITY QUEUE | minute=%d' | batch=%d | workers=%d",
                    priority_minute,
                    len(batch),
                    worker_count,
                )

                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix=(
                        f"goal87_m{priority_minute}"
                    ),
                ) as executor:

                    future_to_match = {
                        executor.submit(
                            self.analyze_match,
                            match,
                        ): match
                        for match in batch
                    }

                    for future in as_completed(
                        future_to_match
                    ):
                        match = future_to_match[
                            future
                        ]

                        try:
                            prediction = future.result()
                        except Exception as exc:
                            logger.exception(
                                "Analyse worker échouée | %s vs %s | %d' | %s",
                                match.home_team,
                                match.away_team,
                                match.minute,
                                exc,
                            )
                            prediction = None

                        if prediction is not None:
                            with self._state_lock:
                                self.last_processed_minute[
                                    match.match_id
                                ] = match.minute

                            predictions.append(
                                prediction
                            )
                        else:
                            logger.info(
                                "Analyse terminée sans prédiction | %s | %d' | pas de nouveau traitement lourd à cette minute",
                                match.match_id,
                                match.minute,
                            )

        # Le PRELOAD n'est exécuté qu'après les analyses urgentes du scan.
        # Il prépare les matchs 83'-84' pour les futurs passages 85'-87'.
        self.preload_pre_match(
            preload_candidates
        )

        # Affichage déterministe malgré l'exécution parallèle.
        predictions.sort(
            key=lambda prediction: (
                -prediction.minute,
                prediction.home_team,
                prediction.away_team,
            )
        )

        # ------------------------------------------------------------------
        # Expire matches after 87'
        # ------------------------------------------------------------------

        with self._state_lock:
            for match_id in list(
                self.last_processed_minute.keys()
            ):
                if match_id in self.expired_matches:
                    self.last_processed_minute.pop(
                        match_id,
                        None,
                    )

            if self.expired_matches:
                self.analysis_attempted_keys = {
                    key
                    for key in self.analysis_attempted_keys
                    if key[0] not in self.expired_matches
                }

                for match_id in self.expired_matches:
                    self.live_readiness_cache.pop(
                        match_id,
                        None,
                    )
                    self.preloaded_match_ids.discard(
                        match_id
                    )

        self.display_results(
            predictions
        )

        self.save_predictions(
            predictions
        )

        return predictions

    # ----------------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------------

    def run(
        self,
    ) -> None:

        logger.info(
            "LIVE SCANNER START"
        )

        logger.info(
            "Fenêtre d'analyse : %d' → %d'",
            MIN_SCAN_MINUTE,
            MAX_SCAN_MINUTE,
        )

        logger.info(
            "Intervalle : %ds",
            SCAN_INTERVAL_SECONDS,
        )

        while True:

            try:

                self.scan_once()

            except KeyboardInterrupt:

                logger.info(
                    "Scanner arrêté par utilisateur."
                )

                break

            except Exception as exc:

                logger.exception(
                    "Erreur globale scanner : %s",
                    exc,
                )

            logger.info(
                "Prochain scan dans %ds...",
                SCAN_INTERVAL_SECONDS,
            )

            time.sleep(
                SCAN_INTERVAL_SECONDS
            )

    # ----------------------------------------------------------------------
    # CONFIG VALIDATION
    # ----------------------------------------------------------------------

    def validate_configuration(
        self,
    ) -> bool:

        logger.info(
            "Validation configuration..."
        )

        valid = True

        if not STATS_API_KEY:

            logger.error(
                "STATS_API_KEY manquante."
            )

            valid = False

        if not MODEL_PATH.exists():

            logger.error(
                "Modèle absent : %s",
                MODEL_PATH,
            )

            valid = False

        if not MODEL_FEATURES_PATH.exists():

            logger.error(
                "Feature JSON absent : %s",
                MODEL_FEATURES_PATH,
            )

            valid = False

        if pre_match is None:

            logger.error(
                "Module PRE-MATCH indisponible."
            )

            valid = False

        if len(
            EXPECTED_FEATURES
        ) != 96:

            logger.error(
                "Schema interne invalide : %d/96",
                len(EXPECTED_FEATURES),
            )

            valid = False

        if not (
            1 <= LIVE_MATCHES_PER_PAGE <= 100
        ):

            logger.error(
                "LIVE_MATCHES_PER_PAGE invalide : %d",
                LIVE_MATCHES_PER_PAGE,
            )

            valid = False

        if SCANNER_WORKERS < 1:

            logger.error(
                "SCANNER_WORKERS doit être >= 1."
            )

            valid = False

        if API_MAX_REQUESTS_PER_SECOND <= 0:

            logger.error(
                "API_MAX_REQUESTS_PER_SECOND doit être > 0."
            )

            valid = False

        if API_429_MAX_RETRIES < 0:

            logger.error(
                "API_429_MAX_RETRIES doit être >= 0."
            )

            valid = False

        if (
            PRELOAD_MINUTE_START < 0
            or PRELOAD_MINUTE_END > 120
            or PRELOAD_MINUTE_START > PRELOAD_MINUTE_END
            or PRELOAD_MINUTE_END >= MIN_SCAN_MINUTE
        ):

            logger.error(
                "Fenêtre PRELOAD invalide : %d -> %d | elle doit être avant MIN_SCAN_MINUTE=%d",
                PRELOAD_MINUTE_START,
                PRELOAD_MINUTE_END,
                MIN_SCAN_MINUTE,
            )

            valid = False

        if MAX_87_LIVE_RETRIES < 0:

            logger.error(
                "MAX_87_LIVE_RETRIES doit être >= 0."
            )

            valid = False

        if not (
            0 <= PRE_MATCH_REBUILD_MIN_COVERAGE <= len(PRE_MATCH_FEATURES)
        ):

            logger.error(
                "PRE_MATCH_REBUILD_MIN_COVERAGE invalide : %d",
                PRE_MATCH_REBUILD_MIN_COVERAGE,
            )

            valid = False

        if (
            LIVE_MATCHES_MAX_PAGES < 1
        ):

            logger.error(
                "LIVE_MATCHES_MAX_PAGES invalide."
            )

            valid = False

        if (
            MIN_SCAN_MINUTE
            < 0
            or MAX_SCAN_MINUTE
            > 120
            or MIN_SCAN_MINUTE
            > MAX_SCAN_MINUTE
        ):

            logger.error(
                "Fenêtre de scan invalide : "
                "%d -> %d",
                MIN_SCAN_MINUTE,
                MAX_SCAN_MINUTE,
            )

            valid = False

        if SIMULATED_ODDS <= 1:

            logger.error(
                "SIMULATED_ODDS doit être > 1."
            )

            valid = False

        if not (
            0 <= MIN_DISPLAY_PROBABILITY <= 1
        ):

            logger.error(
                "MIN_DISPLAY_PROBABILITY doit "
                "être entre 0 et 1."
            )

            valid = False


        if not (
            0 <= PRIORITY_PROBABILITY_MIN
            < PRIORITY_PROBABILITY_MAX
            <= 1
        ):

            logger.error(
                "Zone V3-C prioritaire invalide : %.4f -> %.4f",
                PRIORITY_PROBABILITY_MIN,
                PRIORITY_PROBABILITY_MAX,
            )

            valid = False


        if TELEGRAM_API_TIMEOUT <= 0:

            logger.error(
                "TELEGRAM_API_TIMEOUT doit être > 0."
            )

            valid = False

        if MIN_MODEL_COVERAGE != 38:

            logger.error(
                "V3-C exige MIN_MODEL_COVERAGE=38, actuel=%d.",
                MIN_MODEL_COVERAGE,
            )

            valid = False

        if len(self.model.features) != 38:

            logger.error(
                "Schéma V3-C chargé invalide : %d/38.",
                len(self.model.features),
            )

            valid = False

        logger.info(
            "API base URL : %s",
            STATS_API_BASE_URL,
        )

        logger.info(
            "Model path : %s",
            MODEL_PATH,
        )

        logger.info(
            "Model mode : V3_C_AGGRESSIVE ONLY"
        )

        logger.info(
            "Predictions V3-C path : %s",
            PREDICTIONS_PATH,
        )

        logger.info(
            "Features : %d/96",
            len(EXPECTED_FEATURES),
        )

        logger.info(
            "Scan window : %d' -> %d'",
            MIN_SCAN_MINUTE,
            MAX_SCAN_MINUTE,
        )

        logger.info(
            "Scanner workers : %d | priorité=87>86>85",
            SCANNER_WORKERS,
        )

        logger.info(
            "PRELOAD PRE-MATCH : %d' -> %d'",
            PRELOAD_MINUTE_START,
            PRELOAD_MINUTE_END,
        )

        logger.info(
            "API global rate limit : %.2f req/s | 429 retries=%d",
            API_MAX_REQUESTS_PER_SECOND,
            API_429_MAX_RETRIES,
        )

        logger.info(
            "87' immediate LIVE retries : %d",
            MAX_87_LIVE_RETRIES,
        )

        logger.info(
            "87' PRE-MATCH rebuild threshold : %d/%d",
            PRE_MATCH_REBUILD_MIN_COVERAGE,
            len(PRE_MATCH_FEATURES),
        )

        logger.info(
            "V3-C model coverage minimum : %d/38",
            MIN_MODEL_COVERAGE,
        )


        logger.info(
            "V3-C priority probability zone : %.1f%% <= P < %.1f%%",
            PRIORITY_PROBABILITY_MIN * 100.0,
            PRIORITY_PROBABILITY_MAX * 100.0,
        )


        logger.info(
            "macOS notifications : %s",
            (
                "ON"
                if MACOS_NOTIFICATIONS_ENABLED
                else "OFF"
            ),
        )

        logger.info(
            "Telegram notifications : %s",
            (
                "ON"
                if TELEGRAM_NOTIFICATIONS_ENABLED
                else "OFF"
            ),
        )

        if TELEGRAM_NOTIFICATIONS_ENABLED:
            telegram_config_ok = bool(
                TELEGRAM_BOT_TOKEN
                and TELEGRAM_CHAT_ID
            )
            logger.info(
                "Telegram configuration : %s",
                (
                    "OK"
                    if telegram_config_ok
                    else "INCOMPLÈTE"
                ),
            )

            if not TELEGRAM_BOT_TOKEN:
                logger.warning(
                    "TELEGRAM_BOT_TOKEN manquant dans .env"
                )

            if not TELEGRAM_CHAT_ID:
                logger.warning(
                    "TELEGRAM_CHAT_ID manquant dans .env"
                )

        logger.info(
            "Historical capture : %s",
            (
                "ON"
                if HISTORY_CAPTURE_ENABLED
                else "OFF"
            ),
        )

        if HISTORY_CAPTURE_ENABLED:
            logger.info(
                "Historical snapshots path : %s",
                HISTORY_SNAPSHOTS_PATH,
            )

        logger.info(
            "LIVE pagination : per_page=%d | max_pages=%d",
            LIVE_MATCHES_PER_PAGE,
            LIVE_MATCHES_MAX_PAGES,
        )

        external_validator = (
            _find_external_validator()
        )

        if external_validator is None:

            logger.warning(
                "Validator externe non disponible. "
                "Le validator interne strict reste actif."
            )

        else:

            logger.info(
                "Validator externe : AVAILABLE"
            )

        logger.info(
            "Validation configuration : %s",
            "OK"
            if valid
            else "FAIL",
        )

        return valid


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()

    print(
        "=============================================="
    )

    print(
        " LIVE SCANNER ENGINE V3-C - GOAL AFTER 87'"
    )

    print(
        "=============================================="
    )

    print()

    if not STATS_API_KEY:

        logger.error(
            "STATS_API_KEY non définie dans .env"
        )

        sys.exit(1)

    client = StatsAPIClient(
        api_key=STATS_API_KEY,
        base_url=STATS_API_BASE_URL,
        timeout=STATS_API_TIMEOUT,
    )

    model = ProductionModel()

    model.load()

    scanner = LiveScanner(
        client=client,
        model=model,
    )

    if not scanner.validate_configuration():

        logger.error(
            "Configuration invalide."
        )

        sys.exit(1)

    if not client.test_auth():

        logger.error(
            "Connexion TheStatsAPI impossible."
        )

        sys.exit(1)

    scanner.run()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()