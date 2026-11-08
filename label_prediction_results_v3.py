#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
label_prediction_results_v3.py

AUDIT CONSERVATEUR MULTI-SOURCE "GOAL AFTER 87" — V3

OBJECTIF
--------
Corriger la logique de labellisation précédente.

Cette version combine TIMELINE OFFICIELLE + contrôle du score final pour déterminer
si un but valide a été marqué strictement après la 87e minute.

Le score final n'est plus utilisé pour décider directement du target.
Il sert uniquement de contrôle de cohérence.

RÈGLE CIBLE
-----------
goal_after_87 = 1
    si la timeline contient au moins un BUT VALIDÉ de temps réglementaire
    avec minute effective strictement > 87.

goal_after_87 = 0
    si le match est terminé, la timeline est exploitable, et aucun but valide
    de temps réglementaire > 87 n'est trouvé.

PENDING
    si le match n'est pas terminé, si la timeline est absente/inexploitable,
    ou si la résolution est ambiguë.

EXCLUSIONS
----------
- buts refusés / annulés / overturned / VAR disallowed
- tirs au but (shootout)
- prolongation / extra time
- événements dont la minute n'est pas exploitable

IMPORTANT
---------
- Ce script NE MODIFIE PAS live_scanner_engine.py.
- Il NE MODIFIE PAS le fichier historique original.
- Il écrit de nouveaux fichiers *_v2.csv.

SORTIES
-------
outputs/history/live_feature_snapshots_labeled_v3.csv
outputs/history/prediction_results_by_match_v3.csv
outputs/history/signal87_audit_v3.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_HISTORY = (
    BASE_DIR / "outputs" / "history" / "live_feature_snapshots.csv"
)

DEFAULT_LABELED = (
    BASE_DIR / "outputs" / "history" / "live_feature_snapshots_labeled_v3.csv"
)

DEFAULT_SUMMARY = (
    BASE_DIR / "outputs" / "history" / "prediction_results_by_match_v3.csv"
)

DEFAULT_SIGNAL_AUDIT = (
    BASE_DIR / "outputs" / "history" / "signal87_audit_v3.csv"
)

DEFAULT_THRESHOLD = 0.34
DEFAULT_REQUEST_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 4
DEFAULT_SLEEP = 0.25


# =============================================================================
# MATCH STATUS
# =============================================================================

FINISHED_STATUSES = {
    "finished",
    "complete",
    "completed",
    "ended",
    "full_time",
    "fulltime",
    "ft",
    "after_extra_time",
    "after_penalties",
}

PENDING_STATUSES = {
    "scheduled",
    "not_started",
    "live",
    "in_play",
    "first_half",
    "halftime",
    "second_half",
    "extra_time",
    "paused",
    "postponed",
    "suspended",
}


# =============================================================================
# EVENT RULES
# =============================================================================

GOAL_POSITIVE_HINTS = {
    "goal",
    "own_goal",
    "penalty_goal",
    "scored",
}

GOAL_NEGATIVE_HINTS = {
    "disallowed",
    "disallow",
    "cancelled",
    "canceled",
    "overturned",
    "no_goal",
    "goal_cancelled",
    "goal_canceled",
    "goal_disallowed",
    "var_disallowed",
    "var_cancel",
    "offside_goal",
}

SHOOTOUT_HINTS = {
    "shootout",
    "penalty_shootout",
    "penalty_shoot_out",
    "penalties",
}

EXTRA_TIME_HINTS = {
    "extra_time",
    "after_extra_time",
    "aet",
    "extra_time_first_half",
    "extra_time_second_half",
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return text


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None

    text = clean_text(value)
    if not text:
        return None

    # "90+4" -> 94
    match = re.fullmatch(r"(\d+)\s*\+\s*(\d+)", text)
    if match:
        return int(match.group(1)) + int(match.group(2))

    # "90'" / "89 min"
    match = re.search(r"\d+", text)
    if match:
        try:
            return int(match.group(0))
        except Exception:
            return None

    return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def nested_get(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_payload_objects(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        yield payload

        for key in ("data", "match", "result", "response"):
            value = payload.get(key)

            if isinstance(value, Mapping):
                yield value

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        yield item

    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item


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
    ) -> None:

        api_key = clean_text(api_key)

        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()

        if not api_key:
            raise ValueError("STATS_API_KEY absente.")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    wait = min(2 ** attempt, 20)
                    print(
                        f"    429 API -> attente {wait}s "
                        f"({attempt}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.RequestException as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                wait = min(2 ** (attempt - 1), 10)
                time.sleep(wait)

        raise RuntimeError(
            f"GET {path} impossible: {last_error}"
        )

    def get_match(self, match_id: str) -> Any:
        return self._get(
            f"/football/matches/{match_id}"
        )

    def get_timeline(self, match_id: str) -> Any:
        return self._get(
            f"/football/matches/{match_id}/timeline"
        )


# =============================================================================
# MATCH DETAIL
# =============================================================================

def extract_status(payload: Any) -> str:
    values: List[Any] = []

    for obj in iter_payload_objects(payload):
        values.extend([
            obj.get("status"),
            obj.get("match_status"),
            nested_get(obj, "live", "match_status"),
            nested_get(obj, "state", "status"),
        ])

    for value in values:
        text = normalize_text(value)
        if text:
            return text

    return ""


def _score_pair(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(value, Mapping):
        return None, None

    home = safe_int(
        first_not_none(
            value.get("home"),
            value.get("home_score"),
            value.get("homeScore"),
        )
    )
    away = safe_int(
        first_not_none(
            value.get("away"),
            value.get("away_score"),
            value.get("awayScore"),
        )
    )

    return home, away


def extract_final_score(
    payload: Any,
) -> Tuple[Optional[int], Optional[int], str]:

    for obj in iter_payload_objects(payload):

        score = obj.get("score")

        if isinstance(score, Mapping):

            # On préfère le score de temps réglementaire.
            for key in (
                "regulation",
                "final_score",
                "final",
            ):
                home, away = _score_pair(
                    score.get(key)
                )

                if (
                    home is not None
                    and away is not None
                ):
                    return (
                        home,
                        away,
                        f"score.{key}",
                    )

            home, away = _score_pair(score)

            if (
                home is not None
                and away is not None
            ):
                return home, away, "score"

        for key in (
            "regulation",
            "final_score",
            "final",
        ):
            home, away = _score_pair(
                obj.get(key)
            )

            if (
                home is not None
                and away is not None
            ):
                return home, away, key

    return None, None, ""


def match_is_finished(
    detail: Any,
    final_home: Optional[int],
    final_away: Optional[int],
    score_source: str,
) -> bool:

    status = extract_status(detail)

    if status in FINISHED_STATUSES:
        return True

    if status in PENDING_STATUSES:
        return False

    # Score final/réglementaire explicite.
    if (
        final_home is not None
        and final_away is not None
        and score_source in {
            "score.regulation",
            "score.final_score",
            "score.final",
            "regulation",
            "final_score",
            "final",
        }
    ):
        return True

    return False


# =============================================================================
# TIMELINE PARSING
# =============================================================================

def extract_events(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, Mapping)
        ]

    if not isinstance(payload, Mapping):
        return []

    for key in (
        "events",
        "timeline",
        "data",
        "result",
        "response",
    ):
        value = payload.get(key)

        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, Mapping)
            ]

        if isinstance(value, Mapping):

            for nested_key in (
                "events",
                "timeline",
                "data",
            ):
                nested = value.get(nested_key)

                if isinstance(nested, list):
                    return [
                        item
                        for item in nested
                        if isinstance(item, Mapping)
                    ]

    return []


def extract_event_minute(
    event: Mapping[str, Any],
) -> Optional[int]:

    # 1) Objet time
    time_obj = event.get("time")

    if isinstance(time_obj, Mapping):

        base = safe_int(
            first_not_none(
                time_obj.get("elapsed"),
                time_obj.get("minute"),
                time_obj.get("minutes"),
            )
        )

        extra = safe_int(
            first_not_none(
                time_obj.get("extra"),
                time_obj.get("added"),
                time_obj.get("stoppage"),
            )
        )

        if base is not None:
            return base + (extra or 0)

    # 2) Objet minute
    minute_obj = event.get("minute")

    if isinstance(minute_obj, Mapping):

        base = safe_int(
            first_not_none(
                minute_obj.get("regular"),
                minute_obj.get("minute"),
                minute_obj.get("elapsed"),
            )
        )

        extra = safe_int(
            first_not_none(
                minute_obj.get("extra"),
                minute_obj.get("added"),
            )
        )

        if base is not None:
            return base + (extra or 0)

    # 3) Champs séparés
    base = safe_int(
        first_not_none(
            event.get("elapsed"),
            event.get("elapsed_minutes"),
            event.get("minute"),
            event.get("match_minute"),
        )
    )

    extra = safe_int(
        first_not_none(
            event.get("extra"),
            event.get("extra_minute"),
            event.get("added_time"),
            event.get("stoppage_time"),
        )
    )

    if base is not None:
        return base + (extra or 0)

    return None


def extract_period(event: Mapping[str, Any]) -> str:
    candidates = [
        event.get("period"),
        event.get("match_period"),
        event.get("phase"),
        event.get("stage"),
        nested_get(event, "time", "period"),
    ]

    for value in candidates:
        text = normalize_text(value)
        if text:
            return text

    return ""


def event_text(event: Mapping[str, Any]) -> str:
    parts: List[str] = []

    for key in (
        "type",
        "event_type",
        "event",
        "name",
        "detail",
        "description",
        "status",
        "outcome",
        "result",
        "reason",
        "var",
    ):
        value = event.get(key)

        if isinstance(value, Mapping):
            for inner in value.values():
                if not isinstance(
                    inner,
                    (Mapping, list),
                ):
                    parts.append(
                        clean_text(inner)
                    )

        elif isinstance(value, list):
            for item in value:
                if not isinstance(
                    item,
                    (Mapping, list),
                ):
                    parts.append(
                        clean_text(item)
                    )

        else:
            parts.append(
                clean_text(value)
            )

    return normalize_text(
        " ".join(parts)
    )


def is_shootout_event(
    event: Mapping[str, Any],
) -> bool:

    text = event_text(event)
    period = extract_period(event)

    return (
        any(
            hint in text
            for hint in SHOOTOUT_HINTS
        )
        or any(
            hint in period
            for hint in SHOOTOUT_HINTS
        )
    )


def is_extra_time_event(
    event: Mapping[str, Any],
) -> bool:

    text = event_text(event)
    period = extract_period(event)

    if any(
        hint in period
        for hint in EXTRA_TIME_HINTS
    ):
        return True

    if any(
        hint in text
        for hint in EXTRA_TIME_HINTS
    ):
        return True

    # Une minute > 90 n'est PAS automatiquement extra time.
    # 90+5 peut devenir minute effective 95, donc on ne l'exclut pas par minute seule.
    return False


def is_disallowed_goal(
    event: Mapping[str, Any],
) -> bool:

    text = event_text(event)

    return any(
        hint in text
        for hint in GOAL_NEGATIVE_HINTS
    )


def is_goal_event(
    event: Mapping[str, Any],
) -> bool:

    if is_shootout_event(event):
        return False

    if is_disallowed_goal(event):
        return False

    if event.get("is_goal") is True:
        return True

    if event.get("goal") is True:
        return True

    text = event_text(event)

    return any(
        hint in text
        for hint in GOAL_POSITIVE_HINTS
    )


def is_regulation_goal_event(
    event: Mapping[str, Any],
) -> bool:

    if not is_goal_event(event):
        return False

    if is_shootout_event(event):
        return False

    if is_extra_time_event(event):
        return False

    return True


def audit_timeline(
    payload: Any,
) -> Dict[str, Any]:

    events = extract_events(payload)

    if not events:
        return {
            "timeline_available": False,
            "valid_goal_minutes": [],
            "excluded_goal_like_events": [],
            "all_event_count": 0,
        }

    valid_goal_minutes: List[int] = []
    excluded_goal_like_events: List[str] = []

    for event in events:

        text = event_text(event)
        minute = extract_event_minute(event)
        period = extract_period(event)

        looks_like_goal = (
            event.get("is_goal") is True
            or event.get("goal") is True
            or "goal" in text
            or "scored" in text
        )

        if is_regulation_goal_event(event):

            if minute is not None:
                valid_goal_minutes.append(
                    minute
                )

        elif looks_like_goal:

            reason = []

            if is_shootout_event(event):
                reason.append("shootout")

            if is_extra_time_event(event):
                reason.append("extra_time")

            if is_disallowed_goal(event):
                reason.append("disallowed")

            if minute is None:
                reason.append(
                    "minute_unparseable"
                )

            excluded_goal_like_events.append(
                json.dumps(
                    {
                        "minute": minute,
                        "period": period,
                        "reason": reason,
                        "text": text[:250],
                    },
                    ensure_ascii=False,
                )
            )

    return {
        "timeline_available": True,
        "valid_goal_minutes":
            sorted(valid_goal_minutes),
        "excluded_goal_like_events":
            excluded_goal_like_events,
        "all_event_count": len(events),
    }


# =============================================================================
# CSV HELPERS
# =============================================================================

def read_csv_rows(
    path: Path,
) -> Tuple[List[str], List[Dict[str, str]]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        fieldnames = list(
            reader.fieldnames or []
        )

        rows = [
            dict(row)
            for row in reader
        ]

    if not fieldnames:
        raise ValueError(
            f"CSV sans en-tête: {path}"
        )

    return fieldnames, rows


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })


# =============================================================================
# SNAPSHOT HELPERS
# =============================================================================

def latest_row_at_minute(
    rows: Sequence[Mapping[str, str]],
    minute: int,
) -> Optional[Mapping[str, str]]:

    candidates = [
        row
        for row in rows
        if safe_int(
            row.get("minute")
        ) == minute
    ]

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda row: clean_text(
            row.get(
                "snapshot_timestamp_utc"
            )
        ),
    )[-1]


def probability_from_row(
    row: Optional[Mapping[str, str]],
) -> Optional[float]:

    if row is None:
        return None

    if clean_text(
        row.get("prediction_quality")
    ).upper() == "BLOCKED":
        return None

    return safe_float(
        row.get("model_probability")
    )


def quality_label(
    row: Optional[Mapping[str, str]],
) -> str:

    if row is None:
        return ""

    quality = clean_text(
        row.get("prediction_quality")
    )

    coverage = safe_int(
        row.get("coverage")
    )

    if quality and coverage is not None:
        return (
            f"{quality} {coverage}/96"
        )

    return quality


# =============================================================================
# ONE MATCH RESOLUTION
# =============================================================================

def resolve_match(
    client: StatsAPIClient,
    match_id: str,
    match_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    """
    V3 — résolution multi-source conservatrice.

    Priorité:
      1) Timeline officielle: au moins un but VALIDÉ de temps réglementaire >87
         => POSITIF certain.
      2) Timeline exploitable, aucun but >87, score final identique au snapshot 87
         => NEGATIF certain.
      3) Timeline exploitable, aucun but >87, MAIS score final supérieur au score87
         => AMBIGUOUS (ne jamais transformer cela en NON).
      4) Timeline absente/inexploitable
         => PENDING, même si le score final a changé.

    Pourquoi:
    - La timeline peut manquer un événement.
    - Le snapshot marqué "87" peut être reçu avec retard API.
    - On refuse donc de fabriquer un 0 lorsqu'une autre source indique qu'un but
      a probablement eu lieu après le snapshot.
    """

    detail = client.get_match(match_id)

    final_home, final_away, final_source = extract_final_score(detail)
    api_status = extract_status(detail)

    finished = match_is_finished(
        detail,
        final_home,
        final_away,
        final_source,
    )

    row87 = latest_row_at_minute(
        match_rows,
        87,
    )

    score87_home: Optional[int] = None
    score87_away: Optional[int] = None
    score_delta: Optional[int] = None

    if row87 is not None:
        score87_home = safe_int(
            row87.get("score_home_snapshot")
        )
        score87_away = safe_int(
            row87.get("score_away_snapshot")
        )

    if (
        score87_home is not None
        and score87_away is not None
        and final_home is not None
        and final_away is not None
    ):
        score_delta = (
            final_home
            + final_away
            - score87_home
            - score87_away
        )

    base_result: Dict[str, Any] = {
        "match_id": match_id,
        "target_status": "PENDING",
        "goal_after_87": "",
        "target_source": "",
        "final_score_home":
            "" if final_home is None else final_home,
        "final_score_away":
            "" if final_away is None else final_away,
        "api_status": api_status,
        "valid_goal_minutes_all": "",
        "valid_goal_minutes_after_87": "",
        "excluded_goal_like_events": "",
        "timeline_event_count": 0,
        "resolution_note": "",
        "coherence_warning": "",
        "resolution_class": "PENDING",
        "target_confidence": "",
        "score87_home":
            "" if score87_home is None else score87_home,
        "score87_away":
            "" if score87_away is None else score87_away,
        "score_delta_after_snapshot87":
            "" if score_delta is None else score_delta,
    }

    if not finished:
        base_result["resolution_note"] = "match_not_confirmed_finished"
        return base_result

    try:
        timeline_payload = client.get_timeline(match_id)
    except Exception as exc:
        base_result["resolution_note"] = f"timeline_api_error:{exc}"
        return base_result

    audit = audit_timeline(timeline_payload)

    base_result["timeline_event_count"] = audit["all_event_count"]
    base_result["excluded_goal_like_events"] = " || ".join(
        audit["excluded_goal_like_events"]
    )

    if not audit["timeline_available"]:
        base_result["resolution_note"] = "timeline_empty_or_unparseable"

        if score_delta is not None and score_delta > 0:
            base_result["coherence_warning"] = (
                "score_increased_after_snapshot87_but_timeline_unavailable"
            )
            base_result["resolution_class"] = "AMBIGUOUS"
        return base_result

    valid_goal_minutes = audit["valid_goal_minutes"]

    after87 = [
        minute
        for minute in valid_goal_minutes
        if minute > 87
    ]

    base_result["valid_goal_minutes_all"] = "|".join(
        str(minute)
        for minute in valid_goal_minutes
    )
    base_result["valid_goal_minutes_after_87"] = "|".join(
        str(minute)
        for minute in after87
    )

    # ----------------------------------------------------------------------
    # POSITIVE — timeline explicitly proves a valid regulation goal >87.
    # ----------------------------------------------------------------------
    if after87:
        base_result["goal_after_87"] = 1
        base_result["target_status"] = "LABELED"
        base_result["target_source"] = "official_timeline_regulation_goal_after87"
        base_result["resolution_note"] = "timeline_positive"
        base_result["resolution_class"] = "POSITIVE"
        base_result["target_confidence"] = "HIGH"

        if score_delta is not None and score_delta <= 0:
            # Important audit flag: timeline says goal, score comparison disagrees.
            # Keep the positive target because the event has an official minute,
            # but expose the conflict for manual review.
            base_result["coherence_warning"] = (
                "timeline_goal_after87_but_final_total_not_higher_than_snapshot87"
            )
            base_result["target_confidence"] = "MEDIUM"

        return base_result

    # ----------------------------------------------------------------------
    # NO timeline goal >87.
    # Never automatically call this NON if score changed after snapshot 87.
    # ----------------------------------------------------------------------
    if score_delta is not None:

        if score_delta > 0:
            base_result["target_status"] = "PENDING"
            base_result["goal_after_87"] = ""
            base_result["target_source"] = (
                "timeline_no_goal_but_score_changed_after_snapshot87"
            )
            base_result["resolution_note"] = (
                "ambiguous_timeline_score_conflict"
            )
            base_result["resolution_class"] = "AMBIGUOUS"
            base_result["target_confidence"] = ""
            base_result["coherence_warning"] = (
                "final_total_higher_than_snapshot87_but_no_valid_timeline_goal_after87"
            )
            return base_result

        if score_delta == 0:
            base_result["goal_after_87"] = 0
            base_result["target_status"] = "LABELED"
            base_result["target_source"] = (
                "official_timeline_no_goal_after87_plus_unchanged_score"
            )
            base_result["resolution_note"] = "timeline_negative_score_confirmed"
            base_result["resolution_class"] = "NEGATIVE"
            base_result["target_confidence"] = "HIGH"
            return base_result

        # Final total lower than snapshot total should not normally happen.
        base_result["target_status"] = "PENDING"
        base_result["resolution_class"] = "AMBIGUOUS"
        base_result["resolution_note"] = "final_score_lower_than_snapshot87"
        base_result["coherence_warning"] = (
            "final_total_lower_than_snapshot87"
        )
        return base_result

    # ----------------------------------------------------------------------
    # Timeline usable but no score87/final-score comparison is available.
    # A negative timeline result alone is not strong enough for V3.
    # ----------------------------------------------------------------------
    base_result["target_status"] = "PENDING"
    base_result["resolution_class"] = "AMBIGUOUS"
    base_result["resolution_note"] = (
        "timeline_no_goal_after87_but_score_crosscheck_unavailable"
    )
    base_result["target_source"] = "timeline_only_insufficient_for_negative_v3"

    return base_result


# =============================================================================
# MATCH SUMMARY
# =============================================================================

def build_summary_rows(
    grouped_rows: Mapping[
        str,
        Sequence[Mapping[str, str]],
    ],
    resolutions: Mapping[
        str,
        Mapping[str, Any],
    ],
    threshold: float,
) -> List[Dict[str, Any]]:

    output: List[
        Dict[str, Any]
    ] = []

    for match_id, rows in grouped_rows.items():

        ordered = sorted(
            rows,
            key=lambda row: clean_text(
                row.get(
                    "snapshot_timestamp_utc"
                )
            ),
        )

        first = ordered[0]

        row85 = latest_row_at_minute(
            ordered,
            85,
        )

        row86 = latest_row_at_minute(
            ordered,
            86,
        )

        row87 = latest_row_at_minute(
            ordered,
            87,
        )

        p85 = probability_from_row(
            row85
        )
        p86 = probability_from_row(
            row86
        )
        p87 = probability_from_row(
            row87
        )

        if row87 is None:
            signal_final = "NON_ANALYSE"

        elif clean_text(
            row87.get(
                "prediction_quality"
            )
        ).upper() == "BLOCKED":
            signal_final = "BLOCKED"

        elif p87 is None:
            signal_final = "INCONNU"

        else:
            signal_final = (
                "OUI"
                if p87 >= threshold
                else "NON"
            )

        score87 = ""

        if row87 is not None:

            h = safe_int(
                row87.get(
                    "score_home_snapshot"
                )
            )

            a = safe_int(
                row87.get(
                    "score_away_snapshot"
                )
            )

            if (
                h is not None
                and a is not None
            ):
                score87 = f"{h}-{a}"

        resolution = resolutions.get(
            match_id,
            {},
        )

        target_status = clean_text(
            resolution.get(
                "target_status"
            )
        )

        if target_status == "LABELED":
            result_after87 = (
                "OUI"
                if int(
                    resolution.get(
                        "goal_after_87",
                        0,
                    )
                ) == 1
                else "NON"
            )
        else:
            result_after87 = "PENDING"

        final_score = ""

        fh = resolution.get(
            "final_score_home",
            "",
        )

        fa = resolution.get(
            "final_score_away",
            "",
        )

        if (
            fh != ""
            and fa != ""
        ):
            final_score = f"{fh}-{fa}"

        output.append({
            "match_id": match_id,
            "home_team": first.get(
                "home_team",
                "",
            ),
            "away_team": first.get(
                "away_team",
                "",
            ),
            "P85": (
                "" if p85 is None
                else p85
            ),
            "Qualite85":
                quality_label(row85),
            "P86": (
                "" if p86 is None
                else p86
            ),
            "Qualite86":
                quality_label(row86),
            "P87": (
                "" if p87 is None
                else p87
            ),
            "Qualite87":
                quality_label(row87),
            "Score87": score87,
            "Signal_final":
                signal_final,
            "Resultat_apres_87":
                result_after87,
            "Final_score":
                final_score,
            "Goal_minutes_after_87":
                resolution.get(
                    "valid_goal_minutes_after_87",
                    "",
                ),
            "Target_source":
                resolution.get(
                    "target_source",
                    "",
                ),
            "API_status":
                resolution.get(
                    "api_status",
                    "",
                ),
            "Resolution_class":
                resolution.get(
                    "resolution_class",
                    "",
                ),
            "Target_confidence":
                resolution.get(
                    "target_confidence",
                    "",
                ),
            "Score_delta_after_snapshot87":
                resolution.get(
                    "score_delta_after_snapshot87",
                    "",
                ),
            "Coherence_warning":
                resolution.get(
                    "coherence_warning",
                    "",
                ),
            "Resolution_note":
                resolution.get(
                    "resolution_note",
                    "",
                ),
        })

    output.sort(
        key=lambda row: (
            clean_text(
                row["home_team"]
            ).lower(),
            clean_text(
                row["away_team"]
            ).lower(),
        )
    )

    return output


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Audit strict des résultats Goal After 87 "
            "avec timeline officielle comme source principale."
        )
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LABELED,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--signal-audit",
        type=Path,
        default=DEFAULT_SIGNAL_AUDIT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> int:

    load_dotenv()

    args = parse_args()

    api_key = clean_text(
        os.getenv(
            "STATS_API_KEY"
        )
    )

    api_base = clean_text(
        os.getenv(
            "STATS_API_BASE_URL",
            "https://api.thestatsapi.com/api",
        )
    ).rstrip("/")

    history_path = (
        args.history
        if args.history.is_absolute()
        else BASE_DIR / args.history
    )

    output_path = (
        args.output
        if args.output.is_absolute()
        else BASE_DIR / args.output
    )

    summary_path = (
        args.summary
        if args.summary.is_absolute()
        else BASE_DIR / args.summary
    )

    signal_audit_path = (
        args.signal_audit
        if args.signal_audit.is_absolute()
        else BASE_DIR / args.signal_audit
    )

    if not history_path.exists():
        print(
            f"ERREUR: fichier introuvable: "
            f"{history_path}"
        )
        return 2

    if not api_key:
        print(
            "ERREUR: STATS_API_KEY absente du .env"
        )
        return 2

    fieldnames, rows = (
        read_csv_rows(
            history_path
        )
    )

    grouped: Dict[
        str,
        List[Dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        match_id = clean_text(
            row.get("match_id")
        )

        if match_id:
            grouped[
                match_id
            ].append(row)

    match_ids = list(
        grouped.keys()
    )

    if args.limit > 0:
        match_ids = (
            match_ids[:args.limit]
        )

    print("=" * 86)
    print(
        "GOAL AFTER 87 — STRICT MULTI-SOURCE LABELING V3"
    )
    print("=" * 86)
    print(
        f"Historique : {history_path}"
    )
    print(
        f"Snapshots  : {len(rows)}"
    )
    print(
        f"Matchs     : {len(grouped)}"
    )
    print(
        f"À traiter  : {len(match_ids)}"
    )
    print(
        f"Seuil      : {args.threshold:.2%}"
    )
    print("=" * 86)

    client = StatsAPIClient(
        api_key=api_key,
        base_url=api_base,
    )

    resolutions: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for index, match_id in enumerate(
        match_ids,
        start=1,
    ):

        match_rows = grouped[
            match_id
        ]

        first = match_rows[0]

        home = clean_text(
            first.get("home_team")
        )

        away = clean_text(
            first.get("away_team")
        )

        print(
            f"[{index:03d}/{len(match_ids):03d}] "
            f"{home} vs {away} | {match_id}"
        )

        try:
            resolution = resolve_match(
                client=client,
                match_id=match_id,
                match_rows=match_rows,
            )

        except Exception as exc:
            resolution = {
                "match_id": match_id,
                "target_status": "PENDING",
                "goal_after_87": "",
                "target_source": "",
                "final_score_home": "",
                "final_score_away": "",
                "api_status": "",
                "valid_goal_minutes_all": "",
                "valid_goal_minutes_after_87": "",
                "excluded_goal_like_events": "",
                "timeline_event_count": 0,
                "resolution_note":
                    f"unhandled_error:{exc}",
                "coherence_warning": "",
                "resolution_class": "PENDING",
                "target_confidence": "",
                "score87_home": "",
                "score87_away": "",
                "score_delta_after_snapshot87": "",
            }

        resolutions[
            match_id
        ] = resolution

        status = resolution.get(
            "target_status"
        )

        target = resolution.get(
            "goal_after_87"
        )

        goals = resolution.get(
            "valid_goal_minutes_after_87",
            "",
        )

        warning = resolution.get(
            "coherence_warning",
            "",
        )

        resolution_class = resolution.get(
            "resolution_class",
            "",
        )

        confidence = resolution.get(
            "target_confidence",
            "",
        )

        if status == "LABELED":

            print(
                f"    -> LABELED={target} "
                f"| goals>87={goals or '-'} "
                f"| class={resolution_class or '-'} "
                f"| confidence={confidence or '-'} "
                f"| warning={warning or '-'}"
            )

        else:

            print(
                f"    -> PENDING "
                f"| {resolution.get('resolution_note', '')}"
            )

        if args.sleep > 0:
            time.sleep(
                args.sleep
            )

    # =========================================================================
    # Update snapshot copy
    # =========================================================================

    extra_columns = [
        "target_source",
        "final_score_home",
        "final_score_away",
        "valid_goal_minutes_all",
        "goal_minutes_after_87",
        "timeline_event_count",
        "excluded_goal_like_events",
        "target_resolved_at_utc",
        "target_resolution_note",
        "target_coherence_warning",
        "target_resolution_class",
        "target_confidence",
        "score_delta_after_snapshot87",
    ]

    output_fieldnames = list(
        fieldnames
    )

    for column in extra_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(
                column
            )

    resolved_at = (
        now_utc_iso()
    )

    updated_rows: List[
        Dict[str, Any]
    ] = []

    for row in rows:

        new_row = dict(row)

        match_id = clean_text(
            row.get("match_id")
        )

        resolution = (
            resolutions.get(
                match_id
            )
        )

        if resolution:

            if resolution.get(
                "target_status"
            ) == "LABELED":

                new_row[
                    "goal_after_87"
                ] = resolution[
                    "goal_after_87"
                ]

                new_row[
                    "target_status"
                ] = "LABELED_V3"

            new_row[
                "target_source"
            ] = resolution.get(
                "target_source",
                "",
            )

            new_row[
                "final_score_home"
            ] = resolution.get(
                "final_score_home",
                "",
            )

            new_row[
                "final_score_away"
            ] = resolution.get(
                "final_score_away",
                "",
            )

            new_row[
                "valid_goal_minutes_all"
            ] = resolution.get(
                "valid_goal_minutes_all",
                "",
            )

            new_row[
                "goal_minutes_after_87"
            ] = resolution.get(
                "valid_goal_minutes_after_87",
                "",
            )

            new_row[
                "timeline_event_count"
            ] = resolution.get(
                "timeline_event_count",
                0,
            )

            new_row[
                "excluded_goal_like_events"
            ] = resolution.get(
                "excluded_goal_like_events",
                "",
            )

            new_row[
                "target_resolved_at_utc"
            ] = resolved_at

            new_row[
                "target_resolution_note"
            ] = resolution.get(
                "resolution_note",
                "",
            )

            new_row[
                "target_coherence_warning"
            ] = resolution.get(
                "coherence_warning",
                "",
            )

            new_row[
                "target_resolution_class"
            ] = resolution.get(
                "resolution_class",
                "",
            )

            new_row[
                "target_confidence"
            ] = resolution.get(
                "target_confidence",
                "",
            )

            new_row[
                "score_delta_after_snapshot87"
            ] = resolution.get(
                "score_delta_after_snapshot87",
                "",
            )

        updated_rows.append(
            new_row
        )

    write_csv_rows(
        output_path,
        output_fieldnames,
        updated_rows,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    summary_rows = (
        build_summary_rows(
            grouped_rows=grouped,
            resolutions=resolutions,
            threshold=args.threshold,
        )
    )

    summary_fields = [
        "match_id",
        "home_team",
        "away_team",
        "P85",
        "Qualite85",
        "P86",
        "Qualite86",
        "P87",
        "Qualite87",
        "Score87",
        "Signal_final",
        "Resultat_apres_87",
        "Final_score",
        "Goal_minutes_after_87",
        "Target_source",
        "API_status",
        "Resolution_class",
        "Target_confidence",
        "Score_delta_after_snapshot87",
        "Coherence_warning",
        "Resolution_note",
    ]

    write_csv_rows(
        summary_path,
        summary_fields,
        summary_rows,
    )

    # =========================================================================
    # Strict signal87 audit
    # =========================================================================

    signal87_rows = [
        row
        for row in summary_rows
        if row[
            "Signal_final"
        ] == "OUI"
    ]

    write_csv_rows(
        signal_audit_path,
        summary_fields,
        signal87_rows,
    )

    resolved_all = [
        row
        for row in summary_rows
        if row[
            "Resultat_apres_87"
        ] in {
            "OUI",
            "NON",
        }
    ]

    final_signals_resolved = [
        row
        for row in signal87_rows
        if row[
            "Resultat_apres_87"
        ] in {
            "OUI",
            "NON",
        }
    ]

    wins = sum(
        row[
            "Resultat_apres_87"
        ] == "OUI"
        for row in final_signals_resolved
    )

    losses = (
        len(final_signals_resolved)
        - wins
    )

    hit_rate = (
        wins
        / len(final_signals_resolved)
        if final_signals_resolved
        else None
    )

    warnings = [
        row
        for row in summary_rows
        if clean_text(
            row[
                "Coherence_warning"
            ]
        )
    ]

    pending = [
        row
        for row in summary_rows
        if row[
            "Resultat_apres_87"
        ] == "PENDING"
    ]

    print()
    print("=" * 86)
    print(
        "RÉSULTATS V3 — TIMELINE + CONTRÔLE SCORE"
    )
    print("=" * 86)

    print(
        f"Matchs résolus : "
        f"{len(resolved_all)} / {len(summary_rows)}"
    )

    print(
        f"Matchs PENDING : "
        f"{len(pending)}"
    )

    print(
        f"Warnings cohérence : "
        f"{len(warnings)}"
    )

    print()

    print(
        f"Signaux finaux 87' "
        f"P >= {args.threshold:.2%} : "
        f"{len(signal87_rows)}"
    )

    print(
        f"Signaux avec résultat : "
        f"{len(final_signals_resolved)}"
    )

    print(
        f"Gagnants : {wins}"
    )

    print(
        f"Perdants : {losses}"
    )

    if hit_rate is not None:
        print(
            f"Taux de réussite : "
            f"{hit_rate:.2%}"
        )

    print()
    print(
        f"Historique labellisé V3 : "
        f"{output_path}"
    )
    print(
        f"Résumé par match V3     : "
        f"{summary_path}"
    )
    print(
        f"Audit signaux 87' V3    : "
        f"{signal_audit_path}"
    )
    print("=" * 86)

    # =========================================================================
    # Display strict signal list
    # =========================================================================

    if signal87_rows:

        print()
        print(
            "SIGNALS 87' — DÉTAIL"
        )
        print("-" * 86)

        for row in signal87_rows:

            p87 = safe_float(
                row["P87"]
            )

            p_text = (
                f"{p87:.2%}"
                if p87 is not None
                else "?"
            )

            print(
                f"{row['home_team']} vs "
                f"{row['away_team']} | "
                f"Score87={row['Score87']} | "
                f"P87={p_text} | "
                f"{row['Qualite87']} | "
                f"Résultat={row['Resultat_apres_87']} | "
                f"But(s)>87={row['Goal_minutes_after_87'] or '-'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
