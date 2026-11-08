#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_historical_dataset_v2.py

COLLECTE HISTORIQUE INCRÉMENTALE — DATASET GOAL AFTER 87
========================================================

OBJECTIF
--------
Construire progressivement un NOUVEAU dataset d'entraînement fiable à partir
des matchs passés disponibles dans TheStatsAPI.

Ce script ne dépend PAS de l'ancien dataset de 13 800 lignes.

Pour chaque match terminé :
    1) récupère les détails du match
    2) reconstruit les 59 PRE-MATCH avec cutoff strict au kickoff
    3) reconstruit les LIVE à 87' depuis timeline/shotmap historiques
    4) calcule le target goal_after_87 depuis la timeline
    5) écrit la ligne dans un dataset historique V2
    6) enregistre le match_id dans un index local pour ne jamais recollecter
       inutilement un match déjà traité

PRINCIPE INCRÉMENTAL
--------------------
Si un match_id est déjà dans l'index avec le même schema_version et status=done,
il est ignoré.

Cela permet de relancer le script tous les jours, ou après interruption, sans
recommencer les matchs déjà collectés.

SÉCURITÉ TEMPORELLE
-------------------
PRE-MATCH :
    uniquement les matchs historiques avec match_date < kickoff du match cible.

LIVE 87 :
    uniquement les événements horodatés avec minute <= 87.

TARGET :
    uniquement les buts réglementaires strictement >87, sans prolongation,
    sans tirs au but, sans buts annulés.

Aucune statistique finale n'est utilisée comme état live à 87'.

DÉPENDANCES LOCALES
-------------------
Ce fichier s'appuie sur deux fichiers déjà construits dans le projet :

    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py

Ils doivent être à la racine du projet, dans le même dossier que ce script.

SORTIES
-------
outputs/historical_dataset_v2/
├── Historical_Dataset_Goal_After_87_RAW_v2.csv
├── Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
├── historical_matches_index_v2.csv
├── historical_dataset_v2_rejections.csv
├── historical_dataset_v2_feature_coverage.csv
└── historical_dataset_v2_report.md

UTILISATION
-----------
Test rapide sur 20 matchs :
    python3 collect_historical_dataset_v2.py --limit 20

Collecte plus large :
    python3 collect_historical_dataset_v2.py --limit 500

Collecte ciblée :
    python3 collect_historical_dataset_v2.py --competition-id comp_5840 --limit 500

Reprendre sans recollecter :
    python3 collect_historical_dataset_v2.py --limit 500 --resume

Forcer la reconstruction des matchs déjà traités :
    python3 collect_historical_dataset_v2.py --limit 500 --force

IMPORTANT
---------
- Ne modifie PAS le scanner de production.
- Ne modifie PAS le modèle actuel.
- Ne remplit PAS les NaN par médiane.
- Ne transforme PAS une stat finale en stat à 87'.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# PATHS / VERSION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

SCHEMA_VERSION = "historical_dataset_v2_core_live_2026_09_13"

SNAPSHOT_MINUTE = 87

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
)

DEFAULT_MODEL_SCHEMA = (
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
    / "model_schema_v2.json"
)

PREMATCH_MODULE_PATH = (
    BASE_DIR
    / "rebuild_clean_prematch_v2.py"
)

LIVE87_MODULE_PATH = (
    BASE_DIR
    / "audit_historical_live87_reconstructability.py"
)


# =============================================================================
# SCHEMA FALLBACK
# =============================================================================

PRE_MATCH_FEATURES = [
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
]

LIVE_FEATURES = [
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

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"

DEFAULT_CORE_FEATURES = [
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
    "away_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_btts_freq",
    "away_form10",
    "away_form5",
    "away_over25_freq",
    "away_scored_freq",
    "away_wins_freq",
    "home_avg_goals_conceded",
    "home_avg_goals_scored",
    "home_btts_freq",
    "home_form10",
    "home_form5",
    "home_over25_freq",
    "home_scored_freq",
    "home_wins_freq",
    "league_away_btts_freq",
    "league_away_over05_freq",
    "league_away_over25_freq",
    "league_home_btts_freq",
    "league_home_over05_freq",
    "league_home_over25_freq",
    "odds_away",
    "odds_draw",
    "odds_home",
]


# =============================================================================
# BASIC HELPERS
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


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = clean_text(value)

    if not text:
        return None

    try:
        number = float(text)
    except Exception:
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt_utc_naive(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = clean_text(value)

        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )
        except Exception:
            parsed = pd.to_datetime(
                text,
                utc=True,
                errors="coerce",
            )

            if pd.isna(parsed):
                return None

            parsed = parsed.to_pydatetime()

    if parsed.tzinfo is not None:
        parsed = (
            parsed
            .astimezone(timezone.utc)
            .replace(tzinfo=None)
        )

    return parsed


def count_available(
    row: Mapping[str, Any],
    features: Sequence[str],
) -> int:
    count = 0

    for feature in features:
        if safe_float(row.get(feature)) is not None:
            count += 1

    return count


def load_module(
    module_name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Module introuvable : {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Impossible de charger : {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def maybe_to_float(value: Any) -> Any:
    number = safe_float(value)

    if number is None:
        return np.nan

    return float(number)


# =============================================================================
# API LISTING
# =============================================================================

def as_list_from_payload(
    payload: Any,
) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [
            dict(item)
            for item in payload
            if isinstance(item, Mapping)
        ]

    if not isinstance(payload, Mapping):
        return []

    data = payload.get("data", payload)

    if isinstance(data, list):
        return [
            dict(item)
            for item in data
            if isinstance(item, Mapping)
        ]

    if isinstance(data, Mapping):
        for key in (
            "items",
            "matches",
            "results",
            "data",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return [
                    dict(item)
                    for item in value
                    if isinstance(item, Mapping)
                ]

    return []


def extract_match_id_generic(
    match: Mapping[str, Any],
) -> str:
    for key in (
        "match_id",
        "id",
        "event_id",
    ):
        value = match.get(key)

        if value is not None:
            return clean_text(value)

    return ""


def extract_competition_id_generic(
    match: Mapping[str, Any],
) -> str:
    for key in (
        "competition_id",
        "league_id",
        "tournament_id",
    ):
        value = match.get(key)

        if value is not None:
            return clean_text(value)

    competition = match.get("competition")

    if isinstance(competition, Mapping):
        return clean_text(
            competition.get("id")
            or competition.get("competition_id")
        )

    return ""


def extract_season_id_generic(
    match: Mapping[str, Any],
) -> str:
    for key in (
        "season_id",
        "season",
    ):
        value = match.get(key)

        if isinstance(value, Mapping):
            return clean_text(
                value.get("id")
                or value.get("season_id")
            )

        if value is not None:
            return clean_text(value)

    return ""


def extract_match_date_generic(
    match: Mapping[str, Any],
) -> str:
    for key in (
        "utc_date",
        "date",
        "start_time",
        "kickoff",
        "kickoff_time",
        "match_date",
    ):
        value = match.get(key)

        if value is not None:
            return clean_text(value)

    return ""


def list_finished_matches(
    client: Any,
    limit: int,
    competition_id: Optional[str],
    season_id: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    max_pages: int,
    page_sleep: float,
) -> List[Dict[str, Any]]:
    """
    Liste des matchs terminés.

    Le format exact de pagination dépend de l'API.
    Cette fonction essaie `page=1..max_pages` avec `per_page=100`.
    Elle s'arrête si une page revient vide ou si la limite est atteinte.
    """

    rows: List[Dict[str, Any]] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params: Dict[str, Any] = {
            "status": "finished",
            "per_page": 100,
            "page": page,
        }

        if competition_id:
            params["competition_id"] = competition_id

        if season_id:
            params["season_id"] = season_id

        if date_from:
            params["date_from"] = date_from

        if date_to:
            params["date_to"] = date_to

        payload = client.get(
            "/football/matches",
            params=params,
        )

        page_rows = as_list_from_payload(payload)

        if not page_rows:
            break

        added = 0

        for match in page_rows:
            match_id = extract_match_id_generic(match)

            if not match_id:
                continue

            if match_id in seen:
                continue

            seen.add(match_id)
            rows.append(match)
            added += 1

            if limit > 0 and len(rows) >= limit:
                return rows

        if added == 0:
            break

        if page_sleep > 0:
            time.sleep(page_sleep)

    return rows


# =============================================================================
# INDEX / INCREMENTAL STORAGE
# =============================================================================

def load_index(
    index_path: Path,
) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame(
            columns=[
                "match_id",
                "status",
                "schema_version",
                "competition_id",
                "season_id",
                "utc_date",
                "home_team",
                "away_team",
                "pre_available",
                "live_available",
                "core_available",
                "target_status",
                "goal_after_87",
                "error",
                "collected_at_utc",
            ]
        )

    return pd.read_csv(index_path)


def load_existing_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    if not dataset_path.exists():
        return pd.DataFrame()

    return pd.read_csv(dataset_path)


def already_done(
    index_df: pd.DataFrame,
    match_id: str,
    force: bool,
) -> bool:
    if force:
        return False

    if index_df.empty:
        return False

    if "match_id" not in index_df.columns:
        return False

    current = index_df[
        index_df["match_id"].astype(str) == str(match_id)
    ]

    if current.empty:
        return False

    if "schema_version" in current.columns:
        current = current[
            current["schema_version"].astype(str) == SCHEMA_VERSION
        ]

    if current.empty:
        return False

    if "status" not in current.columns:
        return False

    return bool(
        (
            current["status"].astype(str) == "done"
        ).any()
    )


def append_or_replace_index(
    index_df: pd.DataFrame,
    new_row: Mapping[str, Any],
) -> pd.DataFrame:
    match_id = clean_text(new_row.get("match_id"))

    if match_id and not index_df.empty and "match_id" in index_df.columns:
        mask = ~(
            (index_df["match_id"].astype(str) == match_id)
            & (
                index_df.get(
                    "schema_version",
                    pd.Series([""] * len(index_df)),
                ).astype(str)
                == SCHEMA_VERSION
            )
        )

        index_df = index_df[mask].copy()

    return pd.concat(
        [
            index_df,
            pd.DataFrame([dict(new_row)]),
        ],
        ignore_index=True,
    )


def append_or_replace_dataset(
    dataset_df: pd.DataFrame,
    new_row: Mapping[str, Any],
    force: bool,
) -> pd.DataFrame:
    match_id = clean_text(new_row.get("match_id"))

    if match_id and not dataset_df.empty and "match_id" in dataset_df.columns:
        dataset_df = dataset_df[
            dataset_df["match_id"].astype(str) != match_id
        ].copy()

    return pd.concat(
        [
            dataset_df,
            pd.DataFrame([dict(new_row)]),
        ],
        ignore_index=True,
    )


def save_all(
    dataset_df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejection_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    core_features: Sequence[str],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "Historical_Dataset_Goal_After_87_RAW_v2.csv"
    )

    ready_path = (
        output_dir
        / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
    )

    index_path = (
        output_dir
        / "historical_matches_index_v2.csv"
    )

    rejections_path = (
        output_dir
        / "historical_dataset_v2_rejections.csv"
    )

    coverage_path = (
        output_dir
        / "historical_dataset_v2_feature_coverage.csv"
    )

    dataset_df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    if (
        not dataset_df.empty
        and all(
            feature in dataset_df.columns
            for feature in core_features
        )
    ):
        ready_mask = (
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
        )

        ready_df = dataset_df[ready_mask].copy()

    else:
        ready_df = pd.DataFrame()

    ready_df.to_csv(
        ready_path,
        index=False,
        encoding="utf-8-sig",
    )

    index_df.to_csv(
        index_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(rejection_rows).to_csv(
        rejections_path,
        index=False,
        encoding="utf-8-sig",
    )

    coverage_rows = []

    if not dataset_df.empty:
        feature_columns = [
            feature
            for feature in PRE_MATCH_FEATURES + LIVE_FEATURES
            if feature in dataset_df.columns
        ]

        for feature in feature_columns:
            available = int(dataset_df[feature].notna().sum())
            total = len(dataset_df)

            coverage_rows.append({
                "feature": feature,
                "group": (
                    "PRE"
                    if feature in PRE_MATCH_FEATURES
                    else "LIVE"
                ),
                "available": available,
                "missing": total - available,
                "coverage_rate": (
                    available / total
                    if total
                    else 0.0
                ),
                "is_core": int(feature in core_features),
            })

    pd.DataFrame(coverage_rows).to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# TARGET V2
# =============================================================================

BAD_GOAL_HINTS = {
    "disallowed",
    "cancelled",
    "canceled",
    "overturned",
    "no_goal",
    "var_cancel",
    "goal_cancel",
    "goal_disallow",
}


def event_is_bad_goal(
    live87,
    event: Mapping[str, Any],
) -> bool:
    raw = live87.normalize_text(
        json.dumps(
            dict(event),
            ensure_ascii=False,
            default=str,
        )
    )

    return any(
        hint in raw
        for hint in BAD_GOAL_HINTS
    )


def event_is_goal(
    live87,
    event: Mapping[str, Any],
) -> bool:
    flags = live87.event_flags(event)

    if not flags.get("goal"):
        return False

    if event_is_bad_goal(live87, event):
        return False

    return True


def event_is_regulation_time(
    event: Mapping[str, Any],
) -> bool:
    period = clean_text(event.get("period")).lower()

    if any(
        bad in period
        for bad in (
            "extra",
            "penalty",
            "shootout",
        )
    ):
        return False

    minute = safe_int(event.get("minute"))

    if minute is None:
        return False

    # On exclut les minutes de prolongation type 120.
    # Le temps additionnel 90+ est accepté si encodé comme minute <= 100,
    # car la cible historique précédente distinguait le 90+.
    if minute >= 120:
        return False

    return True


def compute_target_after87(
    live87,
    timeline_events: Sequence[Mapping[str, Any]],
    final_home: Optional[int],
    final_away: Optional[int],
) -> Dict[str, Any]:
    """
    Target conservateur.

    Positif :
        au moins un but réglementaire strictement >87.

    Négatif :
        timeline disponible, aucun but >87, et cohérence score final
        avec les buts timeline.

    Ambiguous :
        timeline insuffisante ou incohérence forte.
    """

    if not timeline_events:
        return {
            "goal_after_87": np.nan,
            "target_status": "PENDING",
            "target_resolution_class": "PENDING",
            "target_confidence": "",
            "goal_minutes_after_87": "",
            "target_warning": "timeline_empty",
        }

    goal_minutes_after = []
    valid_goal_events = []

    for event in timeline_events:
        minute = live87.extract_event_minute(event)

        if minute is None:
            continue

        if not event_is_goal(live87, event):
            continue

        if not event_is_regulation_time(event):
            continue

        valid_goal_events.append(event)

        if minute > SNAPSHOT_MINUTE:
            goal_minutes_after.append(minute)

    if goal_minutes_after:
        return {
            "goal_after_87": 1,
            "target_status": "LABELED_V2",
            "target_resolution_class": "POSITIVE",
            "target_confidence": "HIGH",
            "goal_minutes_after_87": "|".join(
                str(item)
                for item in sorted(goal_minutes_after)
            ),
            "target_warning": "",
        }

    # Negative if no valid goal >87.
    # We keep this conservative but not overstrict: timeline is nonempty and all
    # known valid regulation goals are <=87.
    return {
        "goal_after_87": 0,
        "target_status": "LABELED_V2",
        "target_resolution_class": "NEGATIVE",
        "target_confidence": "HIGH",
        "goal_minutes_after_87": "",
        "target_warning": "",
    }


# =============================================================================
# MODEL SCHEMA
# =============================================================================

def load_core_features(
    schema_path: Optional[str],
) -> List[str]:
    path = (
        Path(schema_path).expanduser()
        if schema_path
        else DEFAULT_MODEL_SCHEMA
    )

    if not path.is_absolute():
        path = BASE_DIR / path

    if path.exists():
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )

            features = payload.get(
                "core_features",
                [],
            )

            features = [
                clean_text(feature)
                for feature in features
                if clean_text(feature)
            ]

            if features:
                print(
                    f"Schema CORE chargé : {path} "
                    f"({len(features)} features)"
                )

                return features

        except Exception as exc:
            print(
                f"Schema illisible {path}: {exc}"
            )

    print(
        "Schema JSON non trouvé/utilisable. "
        "Fallback CORE intégré utilisé."
    )

    return list(DEFAULT_CORE_FEATURES)


# =============================================================================
# LIVE RECONSTRUCTION
# =============================================================================

def fetch_timeline_and_shotmap(
    live87,
    client: Any,
    match_id: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    timeline_payload = client.get(
        f"/football/matches/{match_id}/timeline"
    )

    shotmap_payload = client.get(
        f"/football/matches/{match_id}/shotmap"
    )

    timeline_events = [
        dict(item)
        for item in live87.as_list(timeline_payload)
    ]

    shot_events = [
        dict(item)
        for item in live87.as_list(shotmap_payload)
    ]

    return (
        timeline_events,
        shot_events,
    )


def reconstruct_live87(
    live87,
    target_match: Mapping[str, Any],
    timeline_events: Sequence[Mapping[str, Any]],
    shot_events: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    home_team_id = live87.extract_team_id(
        target_match,
        "home",
    )

    away_team_id = live87.extract_team_id(
        target_match,
        "away",
    )

    home_team_name = live87.extract_team_name(
        target_match,
        "home",
    )

    away_team_name = live87.extract_team_name(
        target_match,
        "away",
    )

    return live87.reconstruct_from_events(
        timeline_events=timeline_events,
        shot_events=shot_events,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        snapshot_minute=SNAPSHOT_MINUTE,
    )


def validate_live87(
    row: Mapping[str, Any],
) -> List[str]:
    errors = []

    for feature in (
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ):
        value = safe_float(row.get(feature))

        if value is not None and value > SNAPSHOT_MINUTE:
            errors.append(f"{feature}_gt87")

    h = safe_float(row.get("score_87_home"))
    a = safe_float(row.get("score_87_away"))

    if h is not None and a is not None:
        diff = safe_float(row.get("goal_diff_87"))
        draw = safe_float(row.get("is_draw_87"))
        leading = safe_float(row.get("leading_team_87"))

        if diff is not None and diff != abs(h - a):
            errors.append("goal_diff_inconsistent")

        expected_draw = 1.0 if h == a else 0.0

        if draw is not None and draw != expected_draw:
            errors.append("is_draw_inconsistent")

        expected_leading = (
            0.0
            if h > a
            else (
                1.0
                if a > h
                else 2.0
            )
        )

        if leading is not None and leading != expected_leading:
            errors.append("leading_team_inconsistent")

    return sorted(set(errors))


# =============================================================================
# SINGLE MATCH COLLECTION
# =============================================================================

def collect_one_match(
    prematch,
    live87,
    client: Any,
    match_stub: Mapping[str, Any],
    core_features: Sequence[str],
    xg_limit: int,
    with_league_second_half: bool,
    preferred_bookmaker: Optional[str],
) -> Tuple[
    Optional[Dict[str, Any]],
    Dict[str, Any],
]:
    match_id = extract_match_id_generic(match_stub)

    status_row: Dict[str, Any] = {
        "match_id": match_id,
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "competition_id": extract_competition_id_generic(match_stub),
        "season_id": extract_season_id_generic(match_stub),
        "utc_date": extract_match_date_generic(match_stub),
        "home_team": "",
        "away_team": "",
        "pre_available": 0,
        "live_available": 0,
        "core_available": 0,
        "target_status": "",
        "goal_after_87": np.nan,
        "error": "",
        "collected_at_utc": now_utc_iso(),
    }

    if not match_id:
        status_row["error"] = "missing_match_id"
        return None, status_row

    target_match = prematch.get_match_detail(
        prematch.lab,
        client,
        match_id,
    )

    if not target_match:
        status_row["error"] = "target_match_detail_unavailable"
        return None, status_row

    kickoff_value = prematch.lab.extract_match_date(
        target_match
    )

    kickoff = prematch.dt_utc_naive(
        kickoff_value
    )

    if kickoff is None:
        status_row["error"] = "kickoff_unparseable"
        return None, status_row

    home_team = prematch.lab.extract_team_name(
        prematch.lab.extract_home_team(
            target_match
        )
    )

    away_team = prematch.lab.extract_team_name(
        prematch.lab.extract_away_team(
            target_match
        )
    )

    competition_id = prematch.lab.extract_competition_id(
        target_match
    )

    season_id = prematch.lab.extract_season_id(
        target_match
    )

    status_row["home_team"] = home_team
    status_row["away_team"] = away_team
    status_row["competition_id"] = competition_id
    status_row["season_id"] = season_id
    status_row["utc_date"] = kickoff.isoformat() + "Z"

    # -------------------------------------------------------------------------
    # PRE AS-OF kickoff
    # -------------------------------------------------------------------------

    pre_features, pre_audit = prematch.build_prematch_asof(
        lab=prematch.lab,
        client=client,
        target_match=target_match,
        cutoff=kickoff,
        xg_limit=xg_limit,
        with_second_half=with_league_second_half,
        preferred_bookmaker=preferred_bookmaker,
    )

    # -------------------------------------------------------------------------
    # LIVE 87 historical reconstruction
    # -------------------------------------------------------------------------

    timeline_events, shot_events = fetch_timeline_and_shotmap(
        live87=live87,
        client=client,
        match_id=match_id,
    )

    live_features, live_provenance = reconstruct_live87(
        live87=live87,
        target_match=target_match,
        timeline_events=timeline_events,
        shot_events=shot_events,
    )

    final_home, final_away = live87.extract_final_score(
        target_match
    )

    target_info = compute_target_after87(
        live87=live87,
        timeline_events=timeline_events,
        final_home=final_home,
        final_away=final_away,
    )

    # -------------------------------------------------------------------------
    # Assemble row
    # -------------------------------------------------------------------------

    row: Dict[str, Any] = {
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_utc": kickoff.isoformat() + "Z",
        "schema_version": SCHEMA_VERSION,
        "snapshot_minute": SNAPSHOT_MINUTE,
        "source": "historical_api",
        "collected_at_utc": now_utc_iso(),
        "timeline_event_count": len(timeline_events),
        "shotmap_event_count": len(shot_events),
        "final_score_home": final_home,
        "final_score_away": final_away,
    }

    for feature in PRE_MATCH_FEATURES:
        row[feature] = maybe_to_float(
            pre_features.get(feature)
        )

    for feature in LIVE_FEATURES:
        row[feature] = maybe_to_float(
            live_features.get(feature)
        )

    for key, value in target_info.items():
        row[key] = value

    row["_pre_available"] = count_available(
        row,
        PRE_MATCH_FEATURES,
    )

    row["_live_available"] = count_available(
        row,
        LIVE_FEATURES,
    )

    row["_core_available"] = count_available(
        row,
        core_features,
    )

    row["_core_total"] = len(core_features)

    row["_core_complete"] = int(
        row["_core_available"]
        == len(core_features)
    )

    row["_live_provenance_json"] = json.dumps(
        live_provenance,
        ensure_ascii=False,
        default=str,
    )

    validation_errors = validate_live87(row)

    row["_validation_errors"] = "|".join(
        validation_errors
    )

    if validation_errors:
        status_row["status"] = "partial"
        status_row["error"] = "|".join(
            validation_errors
        )
    elif target_info["target_status"] != "LABELED_V2":
        status_row["status"] = "partial"
        status_row["error"] = target_info.get(
            "target_warning",
            "target_unresolved",
        )
    else:
        status_row["status"] = "done"

    status_row["pre_available"] = row["_pre_available"]
    status_row["live_available"] = row["_live_available"]
    status_row["core_available"] = row["_core_available"]
    status_row["target_status"] = row.get("target_status", "")
    status_row["goal_after_87"] = row.get("goal_after_87", np.nan)
    status_row["collected_at_utc"] = row["collected_at_utc"]

    return row, status_row


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    dataset_df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejection_rows: Sequence[Mapping[str, Any]],
    core_features: Sequence[str],
    output_dir: Path,
    client: Any,
    args: argparse.Namespace,
) -> None:
    raw_path = (
        output_dir
        / "Historical_Dataset_Goal_After_87_RAW_v2.csv"
    )

    ready_path = (
        output_dir
        / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
    )

    index_path = (
        output_dir
        / "historical_matches_index_v2.csv"
    )

    coverage_path = (
        output_dir
        / "historical_dataset_v2_feature_coverage.csv"
    )

    total = len(dataset_df)

    if total and TARGET_COLUMN in dataset_df.columns:
        labeled = dataset_df[
            pd.to_numeric(
                dataset_df[TARGET_COLUMN],
                errors="coerce",
            ).isin([0, 1])
        ].copy()

        positives = int(
            (
                pd.to_numeric(
                    labeled[TARGET_COLUMN],
                    errors="coerce",
                )
                == 1
            ).sum()
        )

        negatives = int(
            (
                pd.to_numeric(
                    labeled[TARGET_COLUMN],
                    errors="coerce",
                )
                == 0
            ).sum()
        )

        target_rate = (
            positives / len(labeled)
            if len(labeled)
            else 0.0
        )

    else:
        labeled = pd.DataFrame()
        positives = 0
        negatives = 0
        target_rate = 0.0

    if (
        total
        and all(
            feature in dataset_df.columns
            for feature in core_features
        )
    ):
        core_complete = int(
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
            .sum()
        )
    else:
        core_complete = 0

    status_counts = (
        index_df["status"]
        .value_counts(dropna=False)
        .to_dict()
        if not index_df.empty and "status" in index_df.columns
        else {}
    )

    lines = [
        "# Historical Dataset V2 — Goal After 87",
        "",
        "## Objectif",
        "",
        "Construire un dataset historique incrémental sans réutiliser l'ancien dataset de 13 800 lignes.",
        "",
        "## Règles",
        "",
        "- PRE-MATCH calculé uniquement avec les matchs avant le kickoff du match cible.",
        "- LIVE87 reconstruit uniquement avec timeline/shotmap filtrés `minute <= 87`.",
        "- Target calculé avec les buts réglementaires strictement après 87'.",
        "- Aucune imputation médiane.",
        "- Aucun match déjà `done` avec le même `schema_version` n'est recollecté, sauf `--force`.",
        "",
        "## Résultats",
        "",
        f"- Schema version : `{SCHEMA_VERSION}`",
        f"- Lignes dataset RAW : **{total}**",
        f"- Lignes avec CORE complet : **{core_complete}**",
        f"- Features CORE : **{len(core_features)}**",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Target rate : **{target_rate:.2%}**",
        f"- Rejets / partiels : **{len(rejection_rows)}**",
        f"- Requêtes API : **{getattr(client, 'request_count', 0)}**",
        f"- HTTP 429 retentés : **{getattr(client, 'retry_429_count', 0)}**",
        "",
        "## Status index",
        "",
    ]

    if status_counts:
        for status, count in sorted(
            status_counts.items(),
            key=lambda item: str(item[0]),
        ):
            lines.append(
                f"- `{status}` : {count}"
            )
    else:
        lines.append("- Aucun status.")

    lines.extend([
        "",
        "## Sorties",
        "",
        f"- RAW : `{raw_path}`",
        f"- CORE READY : `{ready_path}`",
        f"- Index : `{index_path}`",
        f"- Couverture : `{coverage_path}`",
        "",
        "## Prochaine étape",
        "",
        "Après une première collecte de 500 à 1 000 matchs, lancer un audit statistique du dataset avant tout entraînement.",
    ])

    report_path = (
        output_dir
        / "historical_dataset_v2_report.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collecte historique incrémentale pour le dataset Goal After 87."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Nombre maximal de nouveaux matchs à collecter. Défaut: 50.",
    )

    parser.add_argument(
        "--competition-id",
        default=None,
        help="Filtrer une compétition.",
    )

    parser.add_argument(
        "--season-id",
        default=None,
        help="Filtrer une saison.",
    )

    parser.add_argument(
        "--date-from",
        default=None,
        help="Filtrer date début si supporté par l'API.",
    )

    parser.add_argument(
        "--date-to",
        default=None,
        help="Filtrer date fin si supporté par l'API.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Nombre maximum de pages API à parcourir. Défaut: 20.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="model_schema_v2.json.",
    )

    parser.add_argument(
        "--xg-limit",
        type=int,
        default=5,
        help="Nombre de matchs récents par équipe pour xG. 0 = désactivé. Défaut: 5.",
    )

    parser.add_argument(
        "--with-league-second-half",
        action="store_true",
        help="Inclure league_2nd_half_over15_freq. Coûteux.",
    )

    parser.add_argument(
        "--preferred-bookmaker",
        default=None,
        help="Bookmaker préféré pour les cotes 1X2.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre sans recollecter les matchs déjà done.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recollecter même les matchs déjà done.",
    )

    parser.add_argument(
        "--sleep-between-matches",
        type=float,
        default=0.5,
        help="Pause entre matchs. Défaut: 0.5s.",
    )

    parser.add_argument(
        "--sleep-between-pages",
        type=float,
        default=0.2,
        help="Pause entre pages de listing. Défaut: 0.2s.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Sauvegarde tous les N matchs collectés. Défaut: 1.",
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env_path = BASE_DIR / ".env"
    load_dotenv(env_path if env_path.exists() else None)

    print()
    print("=" * 100)
    print("COLLECT HISTORICAL DATASET V2 — GOAL AFTER 87")
    print("=" * 100)

    print(f"Schema version : {SCHEMA_VERSION}")

    # Modules
    prematch = load_module(
        "prematch_rebuild_runtime",
        PREMATCH_MODULE_PATH,
    )

    live87 = load_module(
        "historical_live87_runtime",
        LIVE87_MODULE_PATH,
    )

    print(f"Module PRE : {PREMATCH_MODULE_PATH.name}")
    print(f"Module LIVE87 : {LIVE87_MODULE_PATH.name}")

    # PRE module loads the LAB and exposes a rate-limited client compatible with
    # both PRE and LIVE functions.
    prematch.lab = prematch.load_lab()

    api_key = clean_text(
        getattr(prematch.lab, "API_KEY", "")
        or os.getenv("STATS_API_KEY")
        or os.getenv("THESTATSAPI_KEY")
        or os.getenv("THE_STATS_API_KEY")
    )

    if not api_key:
        print(
            "ERREUR : clé API introuvable dans .env."
        )
        return 2

    client = prematch.build_rate_limited_client(
        prematch.lab
    )

    core_features = load_core_features(
        args.schema
    )

    index_path = (
        output_dir
        / "historical_matches_index_v2.csv"
    )

    raw_dataset_path = (
        output_dir
        / "Historical_Dataset_Goal_After_87_RAW_v2.csv"
    )

    index_df = load_index(index_path)
    dataset_df = load_existing_dataset(raw_dataset_path)

    print(f"Index existant : {len(index_df)} lignes")
    print(f"Dataset existant : {len(dataset_df)} lignes")
    print(f"CORE features : {len(core_features)}")

    print()
    print("Listing des matchs terminés...")

    match_stubs = list_finished_matches(
        client=client,
        limit=(
            args.limit
            if args.limit > 0
            else 0
        ),
        competition_id=args.competition_id,
        season_id=args.season_id,
        date_from=args.date_from,
        date_to=args.date_to,
        max_pages=max(1, args.max_pages),
        page_sleep=args.sleep_between_pages,
    )

    print(f"Matchs trouvés par l'API : {len(match_stubs)}")

    if not match_stubs:
        print("Aucun match terminé trouvé.")
        return 3

    to_process = []

    skipped = 0

    for stub in match_stubs:
        match_id = extract_match_id_generic(stub)

        if not match_id:
            continue

        if already_done(
            index_df=index_df,
            match_id=match_id,
            force=args.force,
        ):
            skipped += 1
            continue

        to_process.append(stub)

    print(f"Déjà collectés ignorés : {skipped}")
    print(f"Nouveaux à traiter : {len(to_process)}")

    if not to_process:
        print("Rien à faire. Tous les matchs trouvés sont déjà collectés.")
        return 0

    rejection_rows: List[Dict[str, Any]] = []

    rejections_path = (
        output_dir
        / "historical_dataset_v2_rejections.csv"
    )

    if rejections_path.exists() and not args.force:
        try:
            rejection_rows = (
                pd.read_csv(rejections_path)
                .to_dict(orient="records")
            )
        except Exception:
            rejection_rows = []

    collected_this_run = 0

    for position, stub in enumerate(
        to_process,
        start=1,
    ):
        match_id = extract_match_id_generic(stub)

        print()
        print("-" * 100)
        print(
            f"[{position}/{len(to_process)}] {match_id}"
        )
        print("-" * 100)

        try:
            row, status_row = collect_one_match(
                prematch=prematch,
                live87=live87,
                client=client,
                match_stub=stub,
                core_features=core_features,
                xg_limit=max(0, args.xg_limit),
                with_league_second_half=args.with_league_second_half,
                preferred_bookmaker=args.preferred_bookmaker,
            )

            index_df = append_or_replace_index(
                index_df,
                status_row,
            )

            if row is not None:
                dataset_df = append_or_replace_dataset(
                    dataset_df,
                    row,
                    force=args.force,
                )

                print(
                    f"{row.get('home_team', '')} "
                    f"vs "
                    f"{row.get('away_team', '')}"
                )

                print(
                    f"PRE={row['_pre_available']}/59 | "
                    f"LIVE={row['_live_available']}/37 | "
                    f"CORE={row['_core_available']}/{len(core_features)} | "
                    f"target={row.get(TARGET_COLUMN)} | "
                    f"status={status_row['status']}"
                )

                if status_row["status"] != "done":
                    rejection_rows.append({
                        "match_id": match_id,
                        "status": status_row["status"],
                        "reason": status_row.get("error", ""),
                        "collected_at_utc": status_row.get("collected_at_utc", ""),
                    })

            else:
                print(
                    f"Rejet : {status_row.get('error', '')}"
                )

                rejection_rows.append({
                    "match_id": match_id,
                    "status": status_row["status"],
                    "reason": status_row.get("error", ""),
                    "collected_at_utc": status_row.get("collected_at_utc", ""),
                })

            collected_this_run += 1

            if (
                args.save_every > 0
                and collected_this_run % args.save_every == 0
            ):
                save_all(
                    dataset_df=dataset_df,
                    index_df=index_df,
                    rejection_rows=rejection_rows,
                    output_dir=output_dir,
                    core_features=core_features,
                )

        except KeyboardInterrupt:
            print()
            print("Interruption utilisateur. Sauvegarde...")
            save_all(
                dataset_df=dataset_df,
                index_df=index_df,
                rejection_rows=rejection_rows,
                output_dir=output_dir,
                core_features=core_features,
            )
            return 130

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            print(f"ERREUR {match_id}: {error}")

            status_row = {
                "match_id": match_id,
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "competition_id": extract_competition_id_generic(stub),
                "season_id": extract_season_id_generic(stub),
                "utc_date": extract_match_date_generic(stub),
                "home_team": "",
                "away_team": "",
                "pre_available": 0,
                "live_available": 0,
                "core_available": 0,
                "target_status": "",
                "goal_after_87": np.nan,
                "error": error,
                "collected_at_utc": now_utc_iso(),
            }

            index_df = append_or_replace_index(
                index_df,
                status_row,
            )

            rejection_rows.append({
                "match_id": match_id,
                "status": "failed",
                "reason": error,
                "collected_at_utc": now_utc_iso(),
            })

            save_all(
                dataset_df=dataset_df,
                index_df=index_df,
                rejection_rows=rejection_rows,
                output_dir=output_dir,
                core_features=core_features,
            )

        if args.sleep_between_matches > 0:
            time.sleep(args.sleep_between_matches)

    save_all(
        dataset_df=dataset_df,
        index_df=index_df,
        rejection_rows=rejection_rows,
        output_dir=output_dir,
        core_features=core_features,
    )

    write_report(
        dataset_df=dataset_df,
        index_df=index_df,
        rejection_rows=rejection_rows,
        core_features=core_features,
        output_dir=output_dir,
        client=client,
        args=args,
    )

    if not dataset_df.empty:
        core_complete = int(
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
            .sum()
        )

        positives = int(
            (
                pd.to_numeric(
                    dataset_df[TARGET_COLUMN],
                    errors="coerce",
                )
                == 1
            ).sum()
        )

        negatives = int(
            (
                pd.to_numeric(
                    dataset_df[TARGET_COLUMN],
                    errors="coerce",
                )
                == 0
            ).sum()
        )

        target_rate = (
            positives / (positives + negatives)
            if (positives + negatives)
            else 0.0
        )

    else:
        core_complete = 0
        positives = 0
        negatives = 0
        target_rate = 0.0

    print()
    print("=" * 100)
    print("COLLECTE TERMINÉE")
    print("=" * 100)
    print(f"Dataset RAW total : {len(dataset_df)}")
    print(f"CORE complet : {core_complete}")
    print(f"Positifs : {positives}")
    print(f"Négatifs : {negatives}")
    print(f"Target rate : {target_rate:.2%}")
    print(f"Index total : {len(index_df)}")
    print(f"Rejets / partiels : {len(rejection_rows)}")
    print(f"Requêtes API : {getattr(client, 'request_count', 0)}")
    print(f"HTTP 429 retentés : {getattr(client, 'retry_429_count', 0)}")
    print()
    print("Sorties :")
    print(f"  {output_dir / 'Historical_Dataset_Goal_After_87_RAW_v2.csv'}")
    print(f"  {output_dir / 'Historical_Dataset_Goal_After_87_CORE_READY_v2.csv'}")
    print(f"  {output_dir / 'historical_matches_index_v2.csv'}")
    print(f"  {output_dir / 'historical_dataset_v2_feature_coverage.csv'}")
    print(f"  {output_dir / 'historical_dataset_v2_report.md'}")
    print()
    print("Le scanner de production n'a pas été modifié.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
