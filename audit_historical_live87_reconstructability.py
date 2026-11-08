#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_historical_live87_reconstructability.py

AUDIT — PEUT-ON RECONSTRUIRE HISTORIQUEMENT L'ÉTAT LIVE À 87' ?
================================================================

BUT
---
Avant de collecter des milliers de matchs passés, ce script vérifie si
TheStatsAPI conserve suffisamment d'événements HORODATÉS pour reconstruire
fidèlement les features LIVE telles qu'elles auraient existé à la 87e minute.

Le script ne construit PAS encore le dataset final.
Il mesure la reconstructibilité historique.

PRINCIPE DE SÉCURITÉ
--------------------
Une statistique finale de match n'est PAS automatiquement une statistique à 87'.

Exemple:
    /stats dit 18 tirs à la fin du match
    => cela ne prouve PAS qu'il y avait 18 tirs à 87'.

Une feature est considérée "SAFE_RECONSTRUCTABLE" uniquement si elle peut être
calculée à partir d'événements portant une minute exploitable et filtrés avec :

    event_minute <= 87

Les endpoints de stats finales servent seulement de contrôle de cohérence.

SOURCES TESTÉES
---------------
- /football/matches/{id}
- /football/matches/{id}/timeline
- /football/matches/{id}/shotmap
- /football/matches/{id}/live-stats
- /football/matches/{id}/stats

SORTIES
-------
outputs/historical_live87_audit/
├── historical_live87_audit_by_match.csv
├── historical_live87_feature_coverage.csv
├── historical_live87_endpoint_coverage.csv
├── historical_live87_source_diagnostics.csv
├── historical_live87_audit_report.md
└── historical_live87_debug_samples.json

UTILISATION RECOMMANDÉE
-----------------------
Audit rapide sur 30 matchs du dataset prospectif/reconstruit:
    python3 audit_historical_live87_reconstructability.py --limit 30

Audit plus sérieux:
    python3 audit_historical_live87_reconstructability.py --limit 100

Audit d'un CSV précis:
    python3 audit_historical_live87_reconstructability.py \
        --source-csv outputs/rebuilt_dataset_v2/Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv \
        --limit 100

IMPORTANT
---------
- Ce script ne modifie PAS le scanner de production.
- Il ne modifie PAS le modèle.
- Il ne fait AUCUNE imputation.
- Il ne transforme JAMAIS une donnée finale en donnée à 87' sans preuve temporelle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


# =============================================================================
# PATHS / CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "historical_live87_audit"
)

DEFAULT_SOURCE_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "rebuilt_dataset_v2"
    / "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv",

    BASE_DIR
    / "outputs"
    / "clean_dataset_v2"
    / "Clean_Dataset_Goal_After_87_RAW_v2.csv",

    BASE_DIR
    / "outputs"
    / "history"
    / "live_feature_snapshots_labeled_v3.csv",
]

DEFAULT_SCHEMA_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
    / "model_schema_v2.json",
]

API_BASE_URL_DEFAULT = (
    "https://api.thestatsapi.com/api"
)

SNAPSHOT_MINUTE = 87

DEFAULT_REQUEST_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 5
DEFAULT_MIN_INTERVAL = 0.20
DEFAULT_BACKOFF = 2.0


# =============================================================================
# LIVE SCHEMA
# =============================================================================

ALL_LIVE_FEATURES = [
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
]

# Fallback exact CORE LIVE schema from Model Schema V2 currently validated.
FALLBACK_CORE_LIVE_FEATURES = [
    "goal_diff_87",
    "is_draw_87",
    "leading_team_87",
    "score_87_away",
    "score_87_home",
    "away_corners_87",
    "away_fouls_87",
    "away_shots_87",
    "away_shots_on_target_87",
    "cards_80_87",
    "corners_80_87",
    "fouls_80_87",
    "home_corners_87",
    "home_fouls_87",
    "home_shots_87",
    "home_shots_on_target_87",
    "last_corner_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "minutes_since_last_corner",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "shots_80_87",
    "sot_80_87",
    "subs_80_87",
    "away_yellow_cards_87",
    "home_yellow_cards_87",
]


# =============================================================================
# TEXT / NUMERIC HELPERS
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass

    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()

    text = (
        text
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = clean_text(value)

    if not text:
        return None

    try:
        result = float(
            text
        )
    except Exception:
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def safe_int(value: Any) -> Optional[int]:
    value = safe_float(
        value
    )

    if value is None:
        return None

    return int(
        value
    )


def now_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def first_not_none(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def dict_get_nested(
    mapping: Any,
    paths: Sequence[
        Sequence[str]
    ],
) -> Any:
    if not isinstance(
        mapping,
        Mapping,
    ):
        return None

    for path in paths:
        current: Any = mapping

        ok = True

        for key in path:
            if not isinstance(
                current,
                Mapping,
            ):
                ok = False
                break

            if key not in current:
                ok = False
                break

            current = current[
                key
            ]

        if ok:
            return current

    return None


# =============================================================================
# API CLIENT
# =============================================================================

class StatsAPIClient:

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        backoff: float = DEFAULT_BACKOFF,
    ):
        self.api_key = clean_text(
            api_key
        )

        self.base_url = clean_text(
            base_url
        ).rstrip("/")

        self.timeout = int(
            timeout
        )

        self.max_retries = int(
            max_retries
        )

        self.min_interval = float(
            min_interval
        )

        self.backoff = float(
            backoff
        )

        self.session = requests.Session()

        self.session.headers.update({
            "Authorization":
                f"Bearer {self.api_key}",
            "Accept":
                "application/json",
            "User-Agent":
                "GoalAfter87-HistoricalAudit/2.1",
        })

        self._last_request_at = 0.0

        self.request_count = 0
        self.retry_429_count = 0

    def request(
        self,
        endpoint: str,
        params: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Tuple[
        Optional[Any],
        Optional[int],
        str,
    ]:
        url = (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
        )

        last_error = ""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            elapsed = (
                time.monotonic()
                - self._last_request_at
            )

            if elapsed < self.min_interval:
                time.sleep(
                    self.min_interval
                    - elapsed
                )

            try:
                response = (
                    self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout,
                    )
                )

                self._last_request_at = (
                    time.monotonic()
                )

                self.request_count += 1

            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if attempt >= self.max_retries:
                    return (
                        None,
                        None,
                        last_error,
                    )

                wait = min(
                    self.backoff
                    * (
                        2
                        ** (
                            attempt
                            - 1
                        )
                    ),
                    30.0,
                )

                time.sleep(
                    wait
                )

                continue

            status = int(
                response.status_code
            )

            if status == 429:
                self.retry_429_count += 1

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                try:
                    wait = float(
                        retry_after
                    )
                except Exception:
                    wait = min(
                        self.backoff
                        * (
                            2
                            ** (
                                attempt
                                - 1
                            )
                        ),
                        30.0,
                    )

                last_error = (
                    "HTTP 429"
                )

                if attempt >= self.max_retries:
                    return (
                        None,
                        status,
                        last_error,
                    )

                time.sleep(
                    max(
                        0.5,
                        wait,
                    )
                )

                continue

            if status >= 400:
                return (
                    None,
                    status,
                    f"HTTP {status}",
                )

            try:
                payload = response.json()

            except Exception as exc:
                return (
                    None,
                    status,
                    (
                        "invalid_json:"
                        f"{type(exc).__name__}"
                    ),
                )

            return (
                payload,
                status,
                "",
            )

        return (
            None,
            None,
            last_error,
        )


# =============================================================================
# GENERIC PAYLOAD UNWRAPPING
# =============================================================================

def payload_data(
    payload: Any,
) -> Any:
    if isinstance(
        payload,
        Mapping,
    ):
        return payload.get(
            "data",
            payload,
        )

    return payload


def as_list(
    payload: Any,
) -> List[
    Mapping[str, Any]
]:
    data = payload_data(
        payload
    )

    if isinstance(
        data,
        list,
    ):
        return [
            item
            for item in data
            if isinstance(
                item,
                Mapping,
            )
        ]

    if isinstance(
        data,
        Mapping,
    ):
        for key in (
            "items",
            "matches",
            "events",
            "timeline",
            "shots",
            "data",
            "results",
        ):
            value = data.get(
                key
            )

            if isinstance(
                value,
                list,
            ):
                return [
                    item
                    for item in value
                    if isinstance(
                        item,
                        Mapping,
                    )
                ]

    return []


def unwrap_match(
    payload: Any,
) -> Dict[str, Any]:
    data = payload_data(
        payload
    )

    if isinstance(
        data,
        Mapping,
    ):
        match = data.get(
            "match"
        )

        if isinstance(
            match,
            Mapping,
        ):
            return dict(
                match
            )

        return dict(
            data
        )

    return {}


# =============================================================================
# MATCH / TEAM EXTRACTION
# =============================================================================

def extract_match_id(
    match: Mapping[str, Any],
) -> Optional[str]:
    value = first_not_none(
        match.get(
            "match_id"
        ),
        match.get(
            "id"
        ),
        match.get(
            "event_id"
        ),
    )

    if value is None:
        return None

    return clean_text(
        value
    )


def extract_team_object(
    match: Mapping[str, Any],
    side: str,
) -> Mapping[str, Any]:
    side = normalize_text(
        side
    )

    candidates = [
        match.get(
            f"{side}_team"
        ),
        match.get(
            side
        ),
    ]

    teams = match.get(
        "teams"
    )

    if isinstance(
        teams,
        Mapping,
    ):
        candidates.append(
            teams.get(
                side
            )
        )

    for candidate in candidates:
        if isinstance(
            candidate,
            Mapping,
        ):
            return candidate

    return {}


def extract_team_id(
    match: Mapping[str, Any],
    side: str,
) -> Optional[str]:
    team = extract_team_object(
        match,
        side,
    )

    value = first_not_none(
        team.get(
            "id"
        ),
        team.get(
            "team_id"
        ),
        match.get(
            f"{side}_team_id"
        ),
    )

    if value is None:
        return None

    return clean_text(
        value
    )


def extract_team_name(
    match: Mapping[str, Any],
    side: str,
) -> str:
    team = extract_team_object(
        match,
        side,
    )

    value = first_not_none(
        team.get(
            "name"
        ),
        team.get(
            "team_name"
        ),
        match.get(
            f"{side}_team_name"
        ),
        match.get(
            f"{side}_name"
        ),
    )

    return clean_text(
        value
    )


def extract_final_score(
    match: Mapping[str, Any],
) -> Tuple[
    Optional[int],
    Optional[int],
]:
    score = match.get(
        "score"
    )

    if isinstance(
        score,
        Mapping,
    ):
        regulation = score.get(
            "regulation"
        )

        if isinstance(
            regulation,
            Mapping,
        ):
            home = safe_int(
                regulation.get(
                    "home"
                )
            )

            away = safe_int(
                regulation.get(
                    "away"
                )
            )

            if (
                home is not None
                and away is not None
            ):
                return (
                    home,
                    away,
                )

        home = safe_int(
            score.get(
                "home"
            )
        )

        away = safe_int(
            score.get(
                "away"
            )
        )

        if (
            home is not None
            and away is not None
        ):
            return (
                home,
                away,
            )

    return (
        safe_int(
            match.get(
                "home_score"
            )
        ),
        safe_int(
            match.get(
                "away_score"
            )
        ),
    )


# =============================================================================
# EVENT EXTRACTION
# =============================================================================

def extract_event_minute(
    event: Mapping[str, Any],
) -> Optional[int]:
    values = [
        event.get(
            "minute"
        ),
        event.get(
            "elapsed"
        ),
        event.get(
            "time"
        ),
        event.get(
            "match_minute"
        ),
        event.get(
            "event_minute"
        ),
    ]

    time_obj = event.get(
        "time"
    )

    if isinstance(
        time_obj,
        Mapping,
    ):
        values = [
            time_obj.get(
                "minute"
            ),
            time_obj.get(
                "elapsed"
            ),
        ] + values

    for value in values:
        if isinstance(
            value,
            Mapping,
        ):
            continue

        if isinstance(
            value,
            str,
        ):
            text = value.strip()

            # "90+4", "87'", "45 + 2"
            match = re.search(
                r"(\d+)\s*(?:\+\s*(\d+))?",
                text,
            )

            if match:
                base = int(
                    match.group(
                        1
                    )
                )

                extra = int(
                    match.group(
                        2
                    )
                    or 0
                )

                return (
                    base
                    + extra
                )

        parsed = safe_int(
            value
        )

        if parsed is not None:
            return parsed

    return None


def extract_event_type(
    event: Mapping[str, Any],
) -> str:
    values = [
        event.get(
            "type"
        ),
        event.get(
            "event_type"
        ),
        event.get(
            "incident_type"
        ),
        event.get(
            "name"
        ),
        event.get(
            "action"
        ),
        event.get(
            "detail"
        ),
    ]

    for value in values:
        if isinstance(
            value,
            Mapping,
        ):
            nested = first_not_none(
                value.get(
                    "name"
                ),
                value.get(
                    "type"
                ),
                value.get(
                    "value"
                ),
            )

            if nested is not None:
                return normalize_text(
                    nested
                )

        elif value is not None:
            text = normalize_text(
                value
            )

            if text:
                return text

    return ""


def extract_event_team_id(
    event: Mapping[str, Any],
) -> Optional[str]:
    team = event.get(
        "team"
    )

    if isinstance(
        team,
        Mapping,
    ):
        value = first_not_none(
            team.get(
                "id"
            ),
            team.get(
                "team_id"
            ),
        )

        if value is not None:
            return clean_text(
                value
            )

    value = first_not_none(
        event.get(
            "team_id"
        ),
        event.get(
            "participant_id"
        ),
        event.get(
            "competitor_id"
        ),
    )

    if value is None:
        return None

    return clean_text(
        value
    )


def extract_event_team_name(
    event: Mapping[str, Any],
) -> str:
    team = event.get(
        "team"
    )

    if isinstance(
        team,
        Mapping,
    ):
        value = first_not_none(
            team.get(
                "name"
            ),
            team.get(
                "team_name"
            ),
        )

        if value is not None:
            return clean_text(
                value
            )

    return clean_text(
        first_not_none(
            event.get(
                "team_name"
            ),
            event.get(
                "participant_name"
            ),
        )
    )


def identify_side(
    event: Mapping[str, Any],
    home_team_id: Optional[str],
    away_team_id: Optional[str],
    home_team_name: str,
    away_team_name: str,
) -> Optional[str]:
    team_id = extract_event_team_id(
        event
    )

    if (
        team_id
        and home_team_id
        and team_id
        == home_team_id
    ):
        return "home"

    if (
        team_id
        and away_team_id
        and team_id
        == away_team_id
    ):
        return "away"

    team_name = normalize_text(
        extract_event_team_name(
            event
        )
    )

    if (
        team_name
        and normalize_text(
            home_team_name
        )
        == team_name
    ):
        return "home"

    if (
        team_name
        and normalize_text(
            away_team_name
        )
        == team_name
    ):
        return "away"

    side = normalize_text(
        first_not_none(
            event.get(
                "side"
            ),
            event.get(
                "team_side"
            ),
            event.get(
                "home_away"
            ),
        )
    )

    if side in (
        "home",
        "h",
        "1",
    ):
        return "home"

    if side in (
        "away",
        "a",
        "2",
    ):
        return "away"

    return None


# =============================================================================
# EVENT CLASSIFICATION
# =============================================================================

DISALLOWED_HINTS = {
    "disallowed",
    "cancelled",
    "canceled",
    "overturned",
    "no_goal",
    "var_cancel",
    "goal_cancel",
    "goal_disallow",
}

GOAL_HINTS = {
    "goal",
    "own_goal",
    "penalty_goal",
    "scored",
}

SHOT_HINTS = {
    "shot",
    "attempt",
    "total_shot",
}

SOT_HINTS = {
    "shot_on_target",
    "on_target",
    "saved",
    "goal",
}

OFF_TARGET_HINTS = {
    "shot_off_target",
    "off_target",
    "missed",
}

BLOCKED_HINTS = {
    "shot_blocked",
    "blocked_shot",
}

CORNER_HINTS = {
    "corner",
}

FOUL_HINTS = {
    "foul",
}

YELLOW_HINTS = {
    "yellow",
    "yellow_card",
    "card_yellow",
}

RED_HINTS = {
    "red",
    "red_card",
    "second_yellow",
    "yellow_red",
    "sending_off",
    "sent_off",
}

OFFSIDE_HINTS = {
    "offside",
}

SUB_HINTS = {
    "substitution",
    "substituted",
    "sub",
}


def contains_hint(
    text: str,
    hints: Iterable[str],
) -> bool:
    return any(
        hint in text
        for hint in hints
    )


def event_flags(
    event: Mapping[str, Any],
) -> Dict[str, bool]:
    """
    Classifie un événement timeline OU shotmap.

    IMPORTANT — shotmap TheStatsAPI
    --------------------------------
    Les objets shotmap historiques exposent directement :
      - result: "goal" | "save" | "miss" | "block" | ...
      - is_goal: bool
      - is_on_target: bool
      - is_blocked_shot: bool

    Ces champs sont prioritaires pour déterminer SOT / blocked / off-target.
    On ne dépend donc plus seulement du texte du champ `type`.
    """

    event_type = extract_event_type(
        event
    )

    raw_text = normalize_text(
        json.dumps(
            dict(
                event
            ),
            ensure_ascii=False,
            default=str,
        )
    )

    combined = (
        f"{event_type}_{raw_text}"
    )

    result_text = normalize_text(
        event.get(
            "result"
        )
    )

    explicit_is_goal = event.get(
        "is_goal"
    )

    explicit_is_on_target = event.get(
        "is_on_target"
    )

    explicit_is_blocked = event.get(
        "is_blocked_shot"
    )

    disallowed = contains_hint(
        combined,
        DISALLOWED_HINTS,
    )

    text_goal = contains_hint(
        event_type,
        GOAL_HINTS,
    ) or result_text == "goal"

    is_goal = (
        (
            explicit_is_goal is True
            or text_goal
        )
        and not disallowed
    )

    is_corner = contains_hint(
        event_type,
        CORNER_HINTS,
    )

    is_foul = contains_hint(
        event_type,
        FOUL_HINTS,
    )

    is_yellow = contains_hint(
        event_type,
        YELLOW_HINTS,
    )

    is_red = contains_hint(
        event_type,
        RED_HINTS,
    )

    is_offside = contains_hint(
        event_type,
        OFFSIDE_HINTS,
    )

    is_sub = contains_hint(
        event_type,
        SUB_HINTS,
    )

    # Every shotmap object is already a shot. For timeline events, use text hints.
    looks_like_shotmap = any(
        key in event
        for key in (
            "result",
            "is_goal",
            "is_on_target",
            "is_blocked_shot",
            "expected_goals",
            "goal_mouth_location",
        )
    )

    is_shot = (
        looks_like_shotmap
        or contains_hint(
            event_type,
            SHOT_HINTS,
        )
        or is_goal
    )

    # TheStatsAPI shotmap gives the exact boolean we need.
    if isinstance(
        explicit_is_on_target,
        bool,
    ):
        is_sot = (
            explicit_is_on_target
            or is_goal
        )
    else:
        is_sot = (
            result_text in {
                "save",
                "saved",
                "goal",
            }
            or contains_hint(
                event_type,
                SOT_HINTS,
            )
            or is_goal
        )

    if isinstance(
        explicit_is_blocked,
        bool,
    ):
        is_blocked = explicit_is_blocked
    else:
        is_blocked = (
            result_text in {
                "block",
                "blocked",
            }
            or contains_hint(
                event_type,
                BLOCKED_HINTS,
            )
        )

    # "miss" = off target.
    # Un tir bloqué ne doit PAS être compté comme off target.
    is_off_target = (
        (
            result_text in {
                "miss",
                "missed",
                "off_target",
                "offtarget",
            }
            or contains_hint(
                event_type,
                OFF_TARGET_HINTS,
            )
        )
        and not is_blocked
        and not is_sot
        and not is_goal
    )

    return {
        "goal":
            is_goal,
        "shot":
            is_shot,
        "sot":
            is_sot,
        "off_target":
            is_off_target,
        "blocked":
            is_blocked,
        "corner":
            is_corner,
        "foul":
            is_foul,
        "yellow":
            is_yellow,
        "red":
            is_red,
        "offside":
            is_offside,
        "sub":
            is_sub,
    }


# =============================================================================
# TIMESTAMPABILITY AUDIT
# =============================================================================

def event_source_diagnostics(
    events: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    total = len(
        events
    )

    timestamped = 0
    side_known = 0

    type_counts: Counter = Counter()

    for event in events:
        if extract_event_minute(
            event
        ) is not None:
            timestamped += 1

        if (
            extract_event_team_id(
                event
            )
            or extract_event_team_name(
                event
            )
        ):
            side_known += 1

        event_type = extract_event_type(
            event
        )

        if event_type:
            type_counts[
                event_type
            ] += 1

    return {
        "event_count":
            total,
        "timestamped_count":
            timestamped,
        "timestamp_rate":
            (
                timestamped / total
                if total
                else 0.0
            ),
        "team_identified_count":
            side_known,
        "team_identified_rate":
            (
                side_known / total
                if total
                else 0.0
            ),
        "top_event_types":
            "|".join(
                f"{name}:{count}"
                for name, count
                in type_counts.most_common(
                    15
                )
            ),
    }


# =============================================================================
# RECONSTRUCTION @87
# =============================================================================

def initialize_counter() -> Dict[str, int]:
    return {
        "shots": 0,
        "sot": 0,
        "off_target": 0,
        "blocked": 0,
        "corners": 0,
        "fouls": 0,
        "yellow": 0,
        "red": 0,
        "offsides": 0,
    }


def reconstruct_from_events(
    timeline_events: Sequence[
        Mapping[str, Any]
    ],
    shot_events: Sequence[
        Mapping[str, Any]
    ],
    home_team_id: Optional[str],
    away_team_id: Optional[str],
    home_team_name: str,
    away_team_name: str,
    snapshot_minute: int = SNAPSHOT_MINUTE,
) -> Tuple[
    Dict[str, Any],
    Dict[str, str],
]:
    """
    Retourne:
      - features reconstruites
      - source/provenance par feature

    Stratégie:
      * score, corners, fouls, cards, offsides, subs, last events:
        timeline horodatée
      * shots / SOT:
        shotmap horodatée prioritaire
        timeline seulement si elle transporte explicitement des tirs
    """

    features: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}

    timeline_before = []

    for event in timeline_events:
        minute = extract_event_minute(
            event
        )

        if minute is None:
            continue

        if minute > snapshot_minute:
            continue

        timeline_before.append(
            event
        )

    shots_before = []

    for event in shot_events:
        minute = extract_event_minute(
            event
        )

        if minute is None:
            continue

        if minute > snapshot_minute:
            continue

        shots_before.append(
            event
        )

    home = initialize_counter()
    away = initialize_counter()

    score_home = 0
    score_away = 0

    last_goal = None
    last_shot = None
    last_sot = None
    last_corner = None

    period = {
        "shots_80_87": 0,
        "sot_80_87": 0,
        "corners_80_87": 0,
        "fouls_80_87": 0,
        "cards_80_87": 0,
        "subs_80_87": 0,
    }

    timeline_type_seen = Counter()
    timeline_side_unknown = Counter()

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------

    for event in timeline_before:
        minute = extract_event_minute(
            event
        )

        if minute is None:
            continue

        flags = event_flags(
            event
        )

        side = identify_side(
            event,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
        )

        bucket = (
            home
            if side == "home"
            else (
                away
                if side == "away"
                else None
            )
        )

        for key, active in flags.items():
            if active:
                timeline_type_seen[
                    key
                ] += 1

                if (
                    bucket is None
                    and key
                    in (
                        "corner",
                        "foul",
                        "yellow",
                        "red",
                        "offside",
                    )
                ):
                    timeline_side_unknown[
                        key
                    ] += 1

        if flags[
            "goal"
        ]:
            if side == "home":
                score_home += 1
            elif side == "away":
                score_away += 1

            if (
                last_goal is None
                or minute > last_goal
            ):
                last_goal = minute

        if flags[
            "corner"
        ]:
            if bucket is not None:
                bucket[
                    "corners"
                ] += 1

            if (
                last_corner is None
                or minute > last_corner
            ):
                last_corner = minute

        if flags[
            "foul"
        ] and bucket is not None:
            bucket[
                "fouls"
            ] += 1

        if flags[
            "yellow"
        ] and bucket is not None:
            bucket[
                "yellow"
            ] += 1

        if flags[
            "red"
        ] and bucket is not None:
            bucket[
                "red"
            ] += 1

        if flags[
            "offside"
        ] and bucket is not None:
            bucket[
                "offsides"
            ] += 1

        if 80 <= minute <= snapshot_minute:
            if flags[
                "corner"
            ]:
                period[
                    "corners_80_87"
                ] += 1

            if flags[
                "foul"
            ]:
                period[
                    "fouls_80_87"
                ] += 1

            if (
                flags[
                    "yellow"
                ]
                or flags[
                    "red"
                ]
            ):
                period[
                    "cards_80_87"
                ] += 1

            if flags[
                "sub"
            ]:
                period[
                    "subs_80_87"
                ] += 1

    # Score features are safe only if timeline includes goal events with known side.
    goal_events_seen = timeline_type_seen[
        "goal"
    ]

    if goal_events_seen > 0:
        known_goals = (
            score_home
            + score_away
        )

        if known_goals == goal_events_seen:
            features[
                "score_87_home"
            ] = score_home

            features[
                "score_87_away"
            ] = score_away

            features[
                "goal_diff_87"
            ] = abs(
                score_home
                - score_away
            )

            features[
                "is_draw_87"
            ] = int(
                score_home
                == score_away
            )

            features[
                "leading_team_87"
            ] = (
                0
                if score_home
                > score_away
                else (
                    1
                    if score_away
                    > score_home
                    else 2
                )
            )

            for feature in (
                "score_87_home",
                "score_87_away",
                "goal_diff_87",
                "is_draw_87",
                "leading_team_87",
            ):
                provenance[
                    feature
                ] = "timeline_goals_le87"

    # Special case 0-0:
    # We cannot prove 0-0 merely because no goal event exists unless the timeline
    # itself looks active/complete. This audit deliberately stays conservative.
    elif len(
        timeline_before
    ) >= 10:
        features[
            "score_87_home"
        ] = 0

        features[
            "score_87_away"
        ] = 0

        features[
            "goal_diff_87"
        ] = 0

        features[
            "is_draw_87"
        ] = 1

        features[
            "leading_team_87"
        ] = 2

        for feature in (
            "score_87_home",
            "score_87_away",
            "goal_diff_87",
            "is_draw_87",
            "leading_team_87",
        ):
            provenance[
                feature
            ] = (
                "timeline_active_no_goal_le87"
            )

    # Timeline side-specific stats.
    mapping = [
        (
            "corner",
            "corners",
            "home_corners_87",
            "away_corners_87",
        ),
        (
            "foul",
            "fouls",
            "home_fouls_87",
            "away_fouls_87",
        ),
        (
            "yellow",
            "yellow",
            "home_yellow_cards_87",
            "away_yellow_cards_87",
        ),
        (
            "red",
            "red",
            "home_red_cards_87",
            "away_red_cards_87",
        ),
        (
            "offside",
            "offsides",
            "home_offsides_87",
            "away_offsides_87",
        ),
    ]

    for (
        seen_key,
        counter_key,
        home_feature,
        away_feature,
    ) in mapping:
        if (
            timeline_type_seen[
                seen_key
            ] > 0
            and timeline_side_unknown[
                seen_key
            ] == 0
        ):
            features[
                home_feature
            ] = home[
                counter_key
            ]

            features[
                away_feature
            ] = away[
                counter_key
            ]

            provenance[
                home_feature
            ] = (
                "timeline_events_le87"
            )

            provenance[
                away_feature
            ] = (
                "timeline_events_le87"
            )

    for feature in (
        "corners_80_87",
        "fouls_80_87",
        "cards_80_87",
        "subs_80_87",
    ):
        category = (
            "corner"
            if feature
            == "corners_80_87"
            else (
                "foul"
                if feature
                == "fouls_80_87"
                else (
                    "yellow"
                    if feature
                    == "cards_80_87"
                    else "sub"
                )
            )
        )

        # cards can be red-only too
        active = (
            timeline_type_seen[
                category
            ] > 0
        )

        if feature == "cards_80_87":
            active = (
                timeline_type_seen[
                    "yellow"
                ] > 0
                or timeline_type_seen[
                    "red"
                ] > 0
            )

        if active:
            features[
                feature
            ] = period[
                feature
            ]

            provenance[
                feature
            ] = (
                "timeline_events_80_87"
            )

    if last_goal is not None:
        features[
            "last_goal_minute"
        ] = last_goal

        features[
            "minutes_since_last_goal"
        ] = (
            snapshot_minute
            - last_goal
        )

        provenance[
            "last_goal_minute"
        ] = (
            "timeline_events_le87"
        )

        provenance[
            "minutes_since_last_goal"
        ] = (
            "derived_from_timeline"
        )

    if last_corner is not None:
        features[
            "last_corner_minute"
        ] = last_corner

        features[
            "minutes_since_last_corner"
        ] = (
            snapshot_minute
            - last_corner
        )

        provenance[
            "last_corner_minute"
        ] = (
            "timeline_events_le87"
        )

        provenance[
            "minutes_since_last_corner"
        ] = (
            "derived_from_timeline"
        )

    # -------------------------------------------------------------------------
    # Shotmap — preferred source for shots
    # -------------------------------------------------------------------------

    shotmap_usable = (
        len(
            shots_before
        ) > 0
    )

    if shotmap_usable:
        shot_home = initialize_counter()
        shot_away = initialize_counter()

        shot_side_unknown = 0
        shot_count = 0
        sot_count = 0

        for event in shots_before:
            minute = extract_event_minute(
                event
            )

            if minute is None:
                continue

            flags = event_flags(
                event
            )

            # Shotmap entries can be shot records even if type label is unusual.
            is_shot = True

            side = identify_side(
                event,
                home_team_id,
                away_team_id,
                home_team_name,
                away_team_name,
            )

            bucket = (
                shot_home
                if side == "home"
                else (
                    shot_away
                    if side == "away"
                    else None
                )
            )

            if bucket is None:
                shot_side_unknown += 1
                continue

            bucket[
                "shots"
            ] += 1

            shot_count += 1

            is_goal = flags[
                "goal"
            ]

            is_sot = flags[
                "sot"
            ]

            if is_sot:
                bucket[
                    "sot"
                ] += 1

                sot_count += 1

            if flags[
                "off_target"
            ]:
                bucket[
                    "off_target"
                ] += 1

            if flags[
                "blocked"
            ]:
                bucket[
                    "blocked"
                ] += 1

            if (
                last_shot is None
                or minute > last_shot
            ):
                last_shot = minute

            if (
                is_sot
                and (
                    last_sot is None
                    or minute > last_sot
                )
            ):
                last_sot = minute

            if 80 <= minute <= snapshot_minute:
                period[
                    "shots_80_87"
                ] += 1

                if is_sot:
                    period[
                        "sot_80_87"
                    ] += 1

        if (
            shot_count > 0
            and shot_side_unknown == 0
        ):
            features[
                "home_shots_87"
            ] = shot_home[
                "shots"
            ]

            features[
                "away_shots_87"
            ] = shot_away[
                "shots"
            ]

            features[
                "home_shots_on_target_87"
            ] = shot_home[
                "sot"
            ]

            features[
                "away_shots_on_target_87"
            ] = shot_away[
                "sot"
            ]

            features[
                "home_shots_off_target_87"
            ] = shot_home[
                "off_target"
            ]

            features[
                "away_shots_off_target_87"
            ] = shot_away[
                "off_target"
            ]

            features[
                "home_shots_blocked_87"
            ] = shot_home[
                "blocked"
            ]

            features[
                "away_shots_blocked_87"
            ] = shot_away[
                "blocked"
            ]

            provenance[
                "home_shots_87"
            ] = "shotmap_le87"

            provenance[
                "away_shots_87"
            ] = "shotmap_le87"

            provenance[
                "home_shots_on_target_87"
            ] = "shotmap_le87"

            provenance[
                "away_shots_on_target_87"
            ] = "shotmap_le87"

            provenance[
                "home_shots_off_target_87"
            ] = "shotmap_le87"

            provenance[
                "away_shots_off_target_87"
            ] = "shotmap_le87"

            provenance[
                "home_shots_blocked_87"
            ] = "shotmap_le87"

            provenance[
                "away_shots_blocked_87"
            ] = "shotmap_le87"

            features[
                "shots_80_87"
            ] = period[
                "shots_80_87"
            ]

            features[
                "sot_80_87"
            ] = period[
                "sot_80_87"
            ]

            provenance[
                "shots_80_87"
            ] = "shotmap_80_87"

            provenance[
                "sot_80_87"
            ] = "shotmap_80_87"

        if last_shot is not None:
            features[
                "last_shot_minute"
            ] = last_shot

            features[
                "minutes_since_last_shot"
            ] = (
                snapshot_minute
                - last_shot
            )

            provenance[
                "last_shot_minute"
            ] = "shotmap_le87"

            provenance[
                "minutes_since_last_shot"
            ] = "derived_from_shotmap"

        if last_sot is not None:
            features[
                "last_shot_on_target_minute"
            ] = last_sot

            features[
                "minutes_since_last_shot_on_target"
            ] = (
                snapshot_minute
                - last_sot
            )

            provenance[
                "last_shot_on_target_minute"
            ] = "shotmap_le87"

            provenance[
                "minutes_since_last_shot_on_target"
            ] = "derived_from_shotmap"

    # -------------------------------------------------------------------------
    # Timeline shot fallback if shotmap unavailable
    # -------------------------------------------------------------------------

    if not shotmap_usable:
        timeline_shot_events = []

        for event in timeline_before:
            flags = event_flags(
                event
            )

            if flags[
                "shot"
            ]:
                timeline_shot_events.append(
                    event
                )

        if timeline_shot_events:
            shot_home = 0
            shot_away = 0
            sot_home = 0
            sot_away = 0
            unknown = 0
            recent_shots = 0
            recent_sot = 0

            for event in timeline_shot_events:
                minute = extract_event_minute(
                    event
                )

                if minute is None:
                    continue

                flags = event_flags(
                    event
                )

                side = identify_side(
                    event,
                    home_team_id,
                    away_team_id,
                    home_team_name,
                    away_team_name,
                )

                if side is None:
                    unknown += 1
                    continue

                if side == "home":
                    shot_home += 1
                else:
                    shot_away += 1

                if flags[
                    "sot"
                ]:
                    if side == "home":
                        sot_home += 1
                    else:
                        sot_away += 1

                if (
                    last_shot is None
                    or minute > last_shot
                ):
                    last_shot = minute

                if (
                    flags[
                        "sot"
                    ]
                    and (
                        last_sot is None
                        or minute > last_sot
                    )
                ):
                    last_sot = minute

                if 80 <= minute <= snapshot_minute:
                    recent_shots += 1

                    if flags[
                        "sot"
                    ]:
                        recent_sot += 1

            if unknown == 0:
                features[
                    "home_shots_87"
                ] = shot_home

                features[
                    "away_shots_87"
                ] = shot_away

                features[
                    "home_shots_on_target_87"
                ] = sot_home

                features[
                    "away_shots_on_target_87"
                ] = sot_away

                features[
                    "shots_80_87"
                ] = recent_shots

                features[
                    "sot_80_87"
                ] = recent_sot

                for feature in (
                    "home_shots_87",
                    "away_shots_87",
                    "home_shots_on_target_87",
                    "away_shots_on_target_87",
                    "shots_80_87",
                    "sot_80_87",
                ):
                    provenance[
                        feature
                    ] = (
                        "timeline_shot_fallback"
                    )

            if last_shot is not None:
                features[
                    "last_shot_minute"
                ] = last_shot

                features[
                    "minutes_since_last_shot"
                ] = (
                    snapshot_minute
                    - last_shot
                )

                provenance[
                    "last_shot_minute"
                ] = (
                    "timeline_shot_fallback"
                )

                provenance[
                    "minutes_since_last_shot"
                ] = (
                    "derived_timeline_shot_fallback"
                )

            if last_sot is not None:
                features[
                    "last_shot_on_target_minute"
                ] = last_sot

                features[
                    "minutes_since_last_shot_on_target"
                ] = (
                    snapshot_minute
                    - last_sot
                )

                provenance[
                    "last_shot_on_target_minute"
                ] = (
                    "timeline_shot_fallback"
                )

                provenance[
                    "minutes_since_last_shot_on_target"
                ] = (
                    "derived_timeline_shot_fallback"
                )

    return (
        features,
        provenance,
    )


# =============================================================================
# FINAL-STATS EXTRACTION — SANITY CHECK ONLY
# =============================================================================

STAT_ALIASES = {
    "shots": (
        "shots",
        "total_shots",
        "total_attempts",
    ),
    "shots_on_target": (
        "shots_on_target",
        "on_target",
        "shots_on_goal",
    ),
    "corners": (
        "corners",
        "corner_kicks",
    ),
    "fouls": (
        "fouls",
    ),
    "yellow_cards": (
        "yellow_cards",
        "yellow",
        "yellow_card",
    ),
}


def side_value(
    side: Mapping[str, Any],
    aliases: Sequence[str],
) -> Optional[float]:
    for key in aliases:
        value = safe_float(
            side.get(
                key
            )
        )

        if value is not None:
            return value

    return None


def extract_final_stats(
    payload: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    data = payload_data(
        payload
    )

    if not isinstance(
        data,
        Mapping,
    ):
        return result

    home = data.get(
        "home"
    )

    away = data.get(
        "away"
    )

    if not isinstance(
        home,
        Mapping,
    ) or not isinstance(
        away,
        Mapping,
    ):
        stats = data.get(
            "stats"
        )

        if isinstance(
            stats,
            Mapping,
        ):
            home = stats.get(
                "home"
            )

            away = stats.get(
                "away"
            )

    if not isinstance(
        home,
        Mapping,
    ) or not isinstance(
        away,
        Mapping,
    ):
        return result

    for stat_name, aliases in (
        STAT_ALIASES.items()
    ):
        home_value = side_value(
            home,
            aliases,
        )

        away_value = side_value(
            away,
            aliases,
        )

        if home_value is not None:
            result[
                f"final_home_{stat_name}"
            ] = home_value

        if away_value is not None:
            result[
                f"final_away_{stat_name}"
            ] = away_value

    return result


def reconstructed_not_above_final(
    reconstructed: Mapping[str, Any],
    final_stats: Mapping[str, Any],
) -> Tuple[
    int,
    int,
]:
    comparisons = [
        (
            "home_shots_87",
            "final_home_shots",
        ),
        (
            "away_shots_87",
            "final_away_shots",
        ),
        (
            "home_shots_on_target_87",
            "final_home_shots_on_target",
        ),
        (
            "away_shots_on_target_87",
            "final_away_shots_on_target",
        ),
        (
            "home_corners_87",
            "final_home_corners",
        ),
        (
            "away_corners_87",
            "final_away_corners",
        ),
        (
            "home_fouls_87",
            "final_home_fouls",
        ),
        (
            "away_fouls_87",
            "final_away_fouls",
        ),
    ]

    checked = 0
    violations = 0

    for live_feature, final_feature in comparisons:
        live = safe_float(
            reconstructed.get(
                live_feature
            )
        )

        final = safe_float(
            final_stats.get(
                final_feature
            )
        )

        if (
            live is None
            or final is None
        ):
            continue

        checked += 1

        if live > final:
            violations += 1

    return (
        checked,
        violations,
    )


# =============================================================================
# INPUT MATCH IDS
# =============================================================================

def resolve_source_csv(
    explicit: Optional[str],
) -> Optional[Path]:
    if explicit:
        path = Path(
            explicit
        ).expanduser()

        if not path.is_absolute():
            path = (
                BASE_DIR
                / path
            )

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"CSV introuvable : {path}"
            )

        return path

    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    return None


def load_match_ids_from_csv(
    path: Path,
    limit: int,
) -> List[str]:
    df = pd.read_csv(
        path
    )

    if "match_id" not in df.columns:
        raise ValueError(
            f"`match_id` absent de {path}"
        )

    ids = []

    seen = set()

    for value in df[
        "match_id"
    ].tolist():
        match_id = clean_text(
            value
        )

        if not match_id:
            continue

        if match_id in seen:
            continue

        seen.add(
            match_id
        )

        ids.append(
            match_id
        )

        if (
            limit > 0
            and len(
                ids
            ) >= limit
        ):
            break

    return ids


def discover_finished_match_ids(
    client: StatsAPIClient,
    limit: int,
    competition_id: Optional[str],
) -> List[str]:
    params: Dict[str, Any] = {
        "status":
            "finished",
        "per_page":
            min(
                max(
                    limit,
                    50,
                ),
                100,
            ),
    }

    if competition_id:
        params[
            "competition_id"
        ] = competition_id

    payload, _, error = client.request(
        "/football/matches",
        params=params,
    )

    if error:
        raise RuntimeError(
            f"Impossible de récupérer les matchs terminés : {error}"
        )

    matches = as_list(
        payload
    )

    ids = []

    for match in matches:
        match_id = extract_match_id(
            match
        )

        if match_id:
            ids.append(
                match_id
            )

        if (
            limit > 0
            and len(
                ids
            ) >= limit
        ):
            break

    return ids


# =============================================================================
# SCHEMA LOADING
# =============================================================================

def load_core_live_schema(
    explicit: Optional[str],
) -> List[str]:
    candidates: List[Path] = []

    if explicit:
        path = Path(
            explicit
        ).expanduser()

        if not path.is_absolute():
            path = (
                BASE_DIR
                / path
            )

        candidates.append(
            path
        )

    candidates.extend(
        DEFAULT_SCHEMA_CANDIDATES
    )

    for path in candidates:
        if not path.exists():
            continue

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            features = payload.get(
                "core_features",
                [],
            )

            live = [
                feature
                for feature in features
                if feature
                in ALL_LIVE_FEATURES
            ]

            if live:
                print(
                    f"Schema CORE LIVE chargé : "
                    f"{path} "
                    f"({len(live)} features)"
                )

                return live

        except Exception as exc:
            print(
                f"Schema illisible "
                f"{path}: {exc}"
            )

    print(
        "Schema JSON non trouvé/utilisable. "
        "Fallback CORE LIVE intégré utilisé."
    )

    return list(
        FALLBACK_CORE_LIVE_FEATURES
    )


# =============================================================================
# ENDPOINT AUDIT
# =============================================================================

def endpoint_summary_row(
    name: str,
    payload: Any,
    status: Optional[int],
    error: str,
) -> Dict[str, Any]:
    items = as_list(
        payload
    )

    data = payload_data(
        payload
    )

    nonempty = False

    if isinstance(
        data,
        Mapping,
    ):
        nonempty = bool(
            data
        )

    elif isinstance(
        data,
        list,
    ):
        nonempty = bool(
            data
        )

    return {
        f"{name}_http_status":
            status,
        f"{name}_available":
            int(
                payload is not None
            ),
        f"{name}_nonempty":
            int(
                nonempty
            ),
        f"{name}_item_count":
            len(
                items
            ),
        f"{name}_error":
            error,
    }


# =============================================================================
# MAIN AUDIT PER MATCH
# =============================================================================

def audit_match(
    client: StatsAPIClient,
    match_id: str,
    core_live_features: Sequence[str],
    debug_samples: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    row: Dict[str, Any] = {
        "match_id":
            match_id,
    }

    diagnostics: List[
        Dict[str, Any]
    ] = []

    detail_payload, detail_status, detail_error = (
        client.request(
            f"/football/matches/{match_id}"
        )
    )

    row.update(
        endpoint_summary_row(
            "detail",
            detail_payload,
            detail_status,
            detail_error,
        )
    )

    match = unwrap_match(
        detail_payload
    )

    home_team_id = extract_team_id(
        match,
        "home",
    )

    away_team_id = extract_team_id(
        match,
        "away",
    )

    home_team_name = extract_team_name(
        match,
        "home",
    )

    away_team_name = extract_team_name(
        match,
        "away",
    )

    final_home, final_away = (
        extract_final_score(
            match
        )
    )

    row[
        "home_team"
    ] = home_team_name

    row[
        "away_team"
    ] = away_team_name

    row[
        "final_score_home"
    ] = final_home

    row[
        "final_score_away"
    ] = final_away

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------

    timeline_payload, timeline_status, timeline_error = (
        client.request(
            f"/football/matches/{match_id}/timeline"
        )
    )

    row.update(
        endpoint_summary_row(
            "timeline",
            timeline_payload,
            timeline_status,
            timeline_error,
        )
    )

    timeline_events = as_list(
        timeline_payload
    )

    timeline_diag = event_source_diagnostics(
        timeline_events
    )

    for key, value in timeline_diag.items():
        row[
            f"timeline_{key}"
        ] = value

    # -------------------------------------------------------------------------
    # Shotmap
    # -------------------------------------------------------------------------

    shotmap_payload, shotmap_status, shotmap_error = (
        client.request(
            f"/football/matches/{match_id}/shotmap"
        )
    )

    row.update(
        endpoint_summary_row(
            "shotmap",
            shotmap_payload,
            shotmap_status,
            shotmap_error,
        )
    )

    shot_events = as_list(
        shotmap_payload
    )

    shotmap_diag = event_source_diagnostics(
        shot_events
    )

    for key, value in shotmap_diag.items():
        row[
            f"shotmap_{key}"
        ] = value

    # -------------------------------------------------------------------------
    # live-stats historical availability
    # -------------------------------------------------------------------------

    live_stats_payload, live_stats_status, live_stats_error = (
        client.request(
            f"/football/matches/{match_id}/live-stats"
        )
    )

    row.update(
        endpoint_summary_row(
            "live_stats",
            live_stats_payload,
            live_stats_status,
            live_stats_error,
        )
    )

    # -------------------------------------------------------------------------
    # final stats — SANITY ONLY
    # -------------------------------------------------------------------------

    stats_payload, stats_status, stats_error = (
        client.request(
            f"/football/matches/{match_id}/stats"
        )
    )

    row.update(
        endpoint_summary_row(
            "stats",
            stats_payload,
            stats_status,
            stats_error,
        )
    )

    final_stats = extract_final_stats(
        stats_payload
    )

    # -------------------------------------------------------------------------
    # Reconstruct <=87 only
    # -------------------------------------------------------------------------

    reconstructed, provenance = (
        reconstruct_from_events(
            timeline_events=timeline_events,
            shot_events=shot_events,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=home_team_name,
            away_team_name=away_team_name,
            snapshot_minute=SNAPSHOT_MINUTE,
        )
    )

    available_all = 0
    available_core = 0

    for feature in ALL_LIVE_FEATURES:
        value = reconstructed.get(
            feature
        )

        safe = (
            value is not None
        )

        if safe:
            available_all += 1

        if (
            feature
            in core_live_features
            and safe
        ):
            available_core += 1

        diagnostics.append({
            "match_id":
                match_id,
            "feature":
                feature,
            "is_core_live":
                int(
                    feature
                    in core_live_features
                ),
            "safe_reconstructable":
                int(
                    safe
                ),
            "value":
                value,
            "provenance":
                provenance.get(
                    feature,
                    "",
                ),
        })

    row[
        "safe_live_all_count"
    ] = available_all

    row[
        "safe_live_all_total"
    ] = len(
        ALL_LIVE_FEATURES
    )

    row[
        "safe_live_all_rate"
    ] = (
        available_all
        / len(
            ALL_LIVE_FEATURES
        )
    )

    row[
        "safe_live_core_count"
    ] = available_core

    row[
        "safe_live_core_total"
    ] = len(
        core_live_features
    )

    row[
        "safe_live_core_rate"
    ] = (
        available_core
        / len(
            core_live_features
        )
        if core_live_features
        else 0.0
    )

    # Sanity: @87 cumulative counts must not exceed final totals.
    checked, violations = (
        reconstructed_not_above_final(
            reconstructed,
            final_stats,
        )
    )

    row[
        "final_stat_sanity_checks"
    ] = checked

    row[
        "final_stat_sanity_violations"
    ] = violations

    row[
        "final_stat_sanity_ok"
    ] = int(
        violations == 0
    )

    # Timeline leak guard.
    future_reconstructed = 0

    for feature in (
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ):
        value = safe_float(
            reconstructed.get(
                feature
            )
        )

        if (
            value is not None
            and value > SNAPSHOT_MINUTE
        ):
            future_reconstructed += 1

    row[
        "future_event_leak_count"
    ] = future_reconstructed

    # Conservative verdict.
    core_rate = row[
        "safe_live_core_rate"
    ]

    if (
        core_rate >= 0.95
        and violations == 0
        and future_reconstructed == 0
    ):
        verdict = (
            "HISTORICALLY_RECONSTRUCTABLE"
        )

    elif (
        core_rate >= 0.80
        and violations == 0
        and future_reconstructed == 0
    ):
        verdict = (
            "PARTIALLY_RECONSTRUCTABLE"
        )

    else:
        verdict = (
            "INSUFFICIENT_HISTORICAL_LIVE87"
        )

    row[
        "verdict"
    ] = verdict

    # Keep tiny raw samples, never every payload.
    if len(
        debug_samples
    ) < 5:
        debug_samples[
            match_id
        ] = {
            "home_team":
                home_team_name,
            "away_team":
                away_team_name,
            "timeline_first_5":
                [
                    dict(
                        event
                    )
                    for event in timeline_events[
                        :5
                    ]
                ],
            "shotmap_first_5":
                [
                    dict(
                        event
                    )
                    for event in shot_events[
                        :5
                    ]
                ],
            "reconstructed":
                reconstructed,
            "provenance":
                provenance,
        }

    return (
        row,
        diagnostics,
    )


# =============================================================================
# AGGREGATES
# =============================================================================

def build_feature_coverage(
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "is_core_live",
                "matches",
                "safe_reconstructable",
                "coverage_rate",
                "top_provenance",
            ]
        )

    rows = []

    for feature, group in diagnostics_df.groupby(
        "feature",
        sort=False,
    ):
        matches = len(
            group
        )

        safe_count = int(
            group[
                "safe_reconstructable"
            ].sum()
        )

        provenances = (
            group[
                "provenance"
            ]
            .fillna(
                ""
            )
        )

        provenance_counts = Counter(
            value
            for value in provenances
            if value
        )

        top_provenance = "|".join(
            f"{name}:{count}"
            for name, count
            in provenance_counts.most_common(
                5
            )
        )

        rows.append({
            "feature":
                feature,
            "is_core_live":
                int(
                    group[
                        "is_core_live"
                    ].max()
                ),
            "matches":
                matches,
            "safe_reconstructable":
                safe_count,
            "missing":
                matches
                - safe_count,
            "coverage_rate":
                (
                    safe_count
                    / matches
                    if matches
                    else 0.0
                ),
            "top_provenance":
                top_provenance,
        })

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        by=[
            "is_core_live",
            "coverage_rate",
            "feature",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def build_endpoint_coverage(
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    endpoints = [
        "detail",
        "timeline",
        "shotmap",
        "live_stats",
        "stats",
    ]

    rows = []

    total = len(
        matches_df
    )

    for endpoint in endpoints:
        available_col = (
            f"{endpoint}_available"
        )

        nonempty_col = (
            f"{endpoint}_nonempty"
        )

        item_col = (
            f"{endpoint}_item_count"
        )

        available = (
            int(
                matches_df[
                    available_col
                ].sum()
            )
            if available_col
            in matches_df.columns
            else 0
        )

        nonempty = (
            int(
                matches_df[
                    nonempty_col
                ].sum()
            )
            if nonempty_col
            in matches_df.columns
            else 0
        )

        mean_items = (
            float(
                matches_df[
                    item_col
                ].mean()
            )
            if item_col
            in matches_df.columns
            else 0.0
        )

        rows.append({
            "endpoint":
                endpoint,
            "matches":
                total,
            "http_available":
                available,
            "http_available_rate":
                (
                    available
                    / total
                    if total
                    else 0.0
                ),
            "nonempty":
                nonempty,
            "nonempty_rate":
                (
                    nonempty
                    / total
                    if total
                    else 0.0
                ),
            "mean_item_count":
                mean_items,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    matches_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    core_live_features: Sequence[str],
    client: StatsAPIClient,
    source_description: str,
    output_path: Path,
) -> None:
    total = len(
        matches_df
    )

    verdict_counts = (
        matches_df[
            "verdict"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
        if (
            total
            and "verdict"
            in matches_df.columns
        )
        else {}
    )

    core_features_df = feature_df[
        feature_df[
            "feature"
        ].isin(
            core_live_features
        )
    ].copy()

    core95 = int(
        (
            core_features_df[
                "coverage_rate"
            ]
            >= 0.95
        ).sum()
    ) if not core_features_df.empty else 0

    core80 = int(
        (
            core_features_df[
                "coverage_rate"
            ]
            >= 0.80
        ).sum()
    ) if not core_features_df.empty else 0

    mean_core_rate = (
        float(
            matches_df[
                "safe_live_core_rate"
            ].mean()
        )
        if total
        else 0.0
    )

    leak_total = (
        int(
            matches_df[
                "future_event_leak_count"
            ].sum()
        )
        if total
        else 0
    )

    sanity_violations = (
        int(
            matches_df[
                "final_stat_sanity_violations"
            ].sum()
        )
        if total
        else 0
    )

    lines = [
        "# Audit Historical LIVE87 Reconstructability",
        "",
        "## Question",
        "",
        "Peut-on utiliser les matchs passés conservés par l'API pour reconstruire "
        "l'état du match exactement à 87' sans utiliser les statistiques finales comme fuite temporelle ?",
        "",
        "## Source",
        "",
        f"- {source_description}",
        f"- Matchs audités : **{total}**",
        f"- Requêtes API : **{client.request_count}**",
        f"- HTTP 429 retentés : **{client.retry_429_count}**",
        "",
        "## Règle stricte",
        "",
        "Une feature n'est déclarée reconstructible que si elle provient d'événements "
        "horodatés filtrables avec `minute <= 87`.",
        "",
        "`/stats` et `/live-stats` sur un match terminé ne sont jamais considérés comme "
        "preuves de l'état à 87' lorsqu'ils exposent seulement des cumuls finaux.",
        "",
        "## Résumé",
        "",
        f"- CORE LIVE attendues : **{len(core_live_features)}**",
        f"- CORE LIVE à >=95% de reconstructibilité : **{core95}/{len(core_live_features)}**",
        f"- CORE LIVE à >=80% : **{core80}/{len(core_live_features)}**",
        f"- Couverture CORE LIVE moyenne par match : **{mean_core_rate:.2%}**",
        f"- Fuites temporelles reconstruites >87 détectées : **{leak_total}**",
        f"- Violations `stat@87 > stat_finale` : **{sanity_violations}**",
        "",
        "## Verdicts par match",
        "",
    ]

    for verdict, count in sorted(
        verdict_counts.items(),
        key=lambda item: (
            -item[1],
            str(
                item[0]
            ),
        ),
    ):
        lines.append(
            f"- `{verdict}` : {count}"
        )

    lines.extend([
        "",
        "## Couverture des endpoints",
        "",
    ])

    for _, row in endpoint_df.iterrows():
        lines.append(
            f"- `{row['endpoint']}` : "
            f"HTTP {row['http_available_rate']:.2%}, "
            f"non-vide {row['nonempty_rate']:.2%}, "
            f"items moyens {row['mean_item_count']:.1f}"
        )

    lines.extend([
        "",
        "## CORE LIVE — reconstructibilité historique",
        "",
    ])

    if core_features_df.empty:
        lines.append(
            "- Aucune feature CORE trouvée."
        )
    else:
        for _, row in core_features_df.iterrows():
            lines.append(
                f"- `{row['feature']}` : "
                f"**{row['coverage_rate']:.2%}** "
                f"({int(row['safe_reconstructable'])}/{int(row['matches'])}) "
                f"— {row['top_provenance'] or 'aucune source sûre'}"
            )

    lines.extend([
        "",
        "## Interprétation",
        "",
        "### Si la majorité des CORE LIVE est >=95%",
        "",
        "On peut construire un collecteur HISTORIQUE et récupérer des centaines ou milliers "
        "de matchs passés sans attendre la collecte prospective.",
        "",
        "### Si les scores/tirs récents sont reconstructibles mais pas corners/fautes/cartons",
        "",
        "On peut créer un schéma historique réduit et entraîner un modèle uniquement avec les "
        "features réellement reconstructibles.",
        "",
        "### Si les endpoints historiques n'exposent que les stats finales",
        "",
        "Il ne faut pas les utiliser comme état à 87'. La collecte prospective reste nécessaire "
        "pour ces variables.",
        "",
        "## Prochaine décision",
        "",
        "Ne construire `collect_historical_dataset_v2.py` que si cet audit démontre une "
        "reconstructibilité suffisante et sans fuite temporelle.",
    ])

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit de reconstructibilité historique des features LIVE à 87'."
        )
    )

    parser.add_argument(
        "--source-csv",
        default=None,
        help=(
            "CSV contenant match_id. "
            "Par défaut, cherche les datasets déjà reconstruits."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help=(
            "Nombre de matchs à auditer. Défaut: 30."
        ),
    )

    parser.add_argument(
        "--competition-id",
        default=None,
        help=(
            "Si aucun CSV n'est disponible, limite la découverte API à cette compétition."
        ),
    )

    parser.add_argument(
        "--schema",
        default=None,
        help=(
            "Chemin vers model_schema_v2.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Dossier de sortie."
        ),
    )

    parser.add_argument(
        "--sleep-between-matches",
        type=float,
        default=0.5,
        help=(
            "Pause entre matchs. Défaut: 0.5s."
        ),
    )

    args = parser.parse_args()

    env_path = (
        BASE_DIR
        / ".env"
    )

    load_dotenv(
        env_path
        if env_path.exists()
        else None
    )

    api_key = clean_text(
        os.getenv(
            "STATS_API_KEY"
        )
        or os.getenv(
            "THESTATSAPI_KEY"
        )
        or os.getenv(
            "THE_STATS_API_KEY"
        )
    )

    if not api_key:
        print(
            "ERREUR : clé API introuvable dans .env "
            "(STATS_API_KEY / THESTATSAPI_KEY / THE_STATS_API_KEY)."
        )

        return 2

    base_url = clean_text(
        os.getenv(
            "STATS_API_BASE_URL",
            API_BASE_URL_DEFAULT,
        )
    )

    client = StatsAPIClient(
        api_key=api_key,
        base_url=base_url,
    )

    core_live_features = load_core_live_schema(
        args.schema
    )

    print(
        "Version audit : 2.1 — shotmap natif "
        "(is_on_target / is_blocked_shot / result)"
    )

    source_csv = resolve_source_csv(
        args.source_csv
    )

    if source_csv is not None:
        match_ids = load_match_ids_from_csv(
            source_csv,
            limit=max(
                0,
                args.limit,
            ),
        )

        source_description = (
            f"CSV : `{source_csv}`"
        )

    else:
        match_ids = discover_finished_match_ids(
            client=client,
            limit=max(
                1,
                args.limit,
            ),
            competition_id=(
                args.competition_id
            ),
        )

        source_description = (
            "Découverte API `/football/matches?status=finished`"
        )

    if not match_ids:
        print(
            "Aucun match_id à auditer."
        )

        return 3

    output_dir = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        "AUDIT HISTORICAL LIVE87 RECONSTRUCTABILITY"
    )
    print("=" * 100)

    print(
        f"Source : {source_description}"
    )

    print(
        f"Matchs : {len(match_ids)}"
    )

    print(
        f"CORE LIVE : {len(core_live_features)}"
    )

    print(
        "Règle : uniquement événements horodatés <= 87'."
    )

    match_rows: List[
        Dict[str, Any]
    ] = []

    diagnostic_rows: List[
        Dict[str, Any]
    ] = []

    debug_samples: Dict[
        str,
        Any
    ] = {}

    for index, match_id in enumerate(
        match_ids,
        start=1,
    ):
        print()
        print("-" * 100)

        print(
            f"[{index}/{len(match_ids)}] "
            f"{match_id}"
        )

        try:
            row, diagnostics = audit_match(
                client=client,
                match_id=match_id,
                core_live_features=(
                    core_live_features
                ),
                debug_samples=(
                    debug_samples
                ),
            )

            match_rows.append(
                row
            )

            diagnostic_rows.extend(
                diagnostics
            )

            print(
                f"{row.get('home_team', '')} "
                f"vs "
                f"{row.get('away_team', '')}"
            )

            print(
                f"timeline="
                f"{row.get('timeline_item_count', 0)} "
                f"| shotmap="
                f"{row.get('shotmap_item_count', 0)} "
                f"| CORE SAFE="
                f"{row.get('safe_live_core_count', 0)}/"
                f"{row.get('safe_live_core_total', 0)} "
                f"({row.get('safe_live_core_rate', 0.0):.1%})"
            )

            print(
                f"Verdict : "
                f"{row.get('verdict', '')}"
            )

        except KeyboardInterrupt:
            print()
            print(
                "Interruption utilisateur."
            )
            break

        except Exception as exc:
            print(
                f"ERREUR {match_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            match_rows.append({
                "match_id":
                    match_id,
                "verdict":
                    "AUDIT_ERROR",
                "audit_error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            })

        if args.sleep_between_matches > 0:
            time.sleep(
                args.sleep_between_matches
            )

    matches_df = pd.DataFrame(
        match_rows
    )

    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )

    feature_df = build_feature_coverage(
        diagnostics_df
    )

    endpoint_df = build_endpoint_coverage(
        matches_df
    )

    match_path = (
        output_dir
        / "historical_live87_audit_by_match.csv"
    )

    feature_path = (
        output_dir
        / "historical_live87_feature_coverage.csv"
    )

    endpoint_path = (
        output_dir
        / "historical_live87_endpoint_coverage.csv"
    )

    diagnostics_path = (
        output_dir
        / "historical_live87_source_diagnostics.csv"
    )

    report_path = (
        output_dir
        / "historical_live87_audit_report.md"
    )

    debug_path = (
        output_dir
        / "historical_live87_debug_samples.json"
    )

    matches_df.to_csv(
        match_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_df.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    endpoint_df.to_csv(
        endpoint_path,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
        encoding="utf-8-sig",
    )

    debug_path.write_text(
        json.dumps(
            debug_samples,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    write_report(
        matches_df=matches_df,
        feature_df=feature_df,
        endpoint_df=endpoint_df,
        core_live_features=(
            core_live_features
        ),
        client=client,
        source_description=(
            source_description
        ),
        output_path=report_path,
    )

    reconstructable = int(
        (
            matches_df.get(
                "verdict",
                pd.Series(
                    dtype=str
                ),
            )
            == "HISTORICALLY_RECONSTRUCTABLE"
        ).sum()
    )

    partial = int(
        (
            matches_df.get(
                "verdict",
                pd.Series(
                    dtype=str
                ),
            )
            == "PARTIALLY_RECONSTRUCTABLE"
        ).sum()
    )

    insufficient = int(
        (
            matches_df.get(
                "verdict",
                pd.Series(
                    dtype=str
                ),
            )
            == "INSUFFICIENT_HISTORICAL_LIVE87"
        ).sum()
    )

    print()
    print("=" * 100)
    print(
        "AUDIT TERMINÉ"
    )
    print("=" * 100)

    print(
        f"Historically reconstructable : "
        f"{reconstructable}"
    )

    print(
        f"Partially reconstructable : "
        f"{partial}"
    )

    print(
        f"Insufficient : "
        f"{insufficient}"
    )

    print(
        f"Requêtes API : "
        f"{client.request_count}"
    )

    print(
        f"HTTP 429 retentés : "
        f"{client.retry_429_count}"
    )

    print()
    print(
        f"Rapport : {report_path}"
    )

    print(
        f"Feature coverage : {feature_path}"
    )

    print(
        f"Endpoint coverage : {endpoint_path}"
    )

    print()
    print(
        "Le scanner de production n'a pas été modifié."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
