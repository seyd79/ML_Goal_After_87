#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_historical_dataset_v2.py

AUDIT DU DATASET HISTORIQUE V2 — GOAL AFTER 87
==============================================

OBJECTIF
--------
Contrôler la qualité du dataset construit par :

    collect_historical_dataset_v2.py

avant toute étape d'entraînement.

Ce script vérifie :

1) la taille du dataset
2) les doublons de match_id
3) l'index incrémental
4) la distribution du target goal_after_87
5) la couverture PRE / LIVE / CORE
6) les lignes CORE complètes
7) les lignes avec target manquant
8) les matchs partiels / failed
9) les compétitions problématiques
10) les incohérences logiques LIVE87
11) les fuites temporelles évidentes >87
12) les features à garder / surveiller / retirer

IMPORTANT
---------
- Ce script ne modifie PAS le scanner de production.
- Il ne modifie PAS le modèle.
- Il ne modifie PAS le dataset.
- Il ne fait AUCUNE imputation.
- Il sert uniquement à auditer.

ENTRÉES PAR DÉFAUT
------------------
outputs/historical_dataset_v2/
├── Historical_Dataset_Goal_After_87_RAW_v2.csv
├── Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
├── historical_matches_index_v2.csv
├── historical_dataset_v2_rejections.csv
└── historical_dataset_v2_feature_coverage.csv

Schema :
outputs/model_schema_v2/model_schema_v2.json

SORTIES
-------
outputs/historical_dataset_v2/audit/
├── audit_summary_v2.csv
├── audit_feature_coverage_v2.csv
├── audit_core_missing_by_row_v2.csv
├── audit_problem_rows_v2.csv
├── audit_competition_quality_v2.csv
├── audit_target_distribution_v2.csv
├── audit_index_quality_v2.csv
└── audit_historical_dataset_v2_report.md

UTILISATION
-----------
    python3 audit_historical_dataset_v2.py

Avec un autre dossier :
    python3 audit_historical_dataset_v2.py --dataset-dir outputs/historical_dataset_v2

Avec un autre schema :
    python3 audit_historical_dataset_v2.py --schema outputs/model_schema_v2/model_schema_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_DIR = (
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
)

DEFAULT_SCHEMA_PATH = (
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
    / "model_schema_v2.json"
)

RAW_FILENAME = "Historical_Dataset_Goal_After_87_RAW_v2.csv"
READY_FILENAME = "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
INDEX_FILENAME = "historical_matches_index_v2.csv"
REJECTIONS_FILENAME = "historical_dataset_v2_rejections.csv"
COVERAGE_FILENAME = "historical_dataset_v2_feature_coverage.csv"

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"


# =============================================================================
# FEATURES
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

ALL_FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES

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
# HELPERS
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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_dataset_dir(value: Optional[str]) -> Path:
    path = (
        Path(value).expanduser()
        if value
        else DEFAULT_DATASET_DIR
    )

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def resolve_schema_path(value: Optional[str]) -> Path:
    path = (
        Path(value).expanduser()
        if value
        else DEFAULT_SCHEMA_PATH
    )

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def load_core_features(schema_path: Path) -> List[str]:
    if schema_path.exists():
        try:
            payload = json.loads(
                schema_path.read_text(
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
                return features

        except Exception:
            pass

    return list(DEFAULT_CORE_FEATURES)


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            [np.nan] * len(df),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def count_available_row(
    row: Mapping[str, Any],
    features: Sequence[str],
) -> int:
    count = 0

    for feature in features:
        if safe_float(row.get(feature)) is not None:
            count += 1

    return count


def percent(value: float) -> str:
    return f"{value:.2%}"


def value_counts_dict(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, int]:
    if df.empty or column not in df.columns:
        return {}

    values = (
        df[column]
        .fillna("")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        str(key): int(value)
        for key, value in values.items()
    }


# =============================================================================
# FEATURE COVERAGE
# =============================================================================

def build_feature_coverage(
    dataset_df: pd.DataFrame,
    core_features: Sequence[str],
) -> pd.DataFrame:
    rows = []

    if dataset_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "group",
                "is_core",
                "available",
                "missing",
                "coverage_rate",
                "mean",
                "std",
                "min",
                "max",
                "mode",
                "mode_share",
                "class",
            ]
        )

    total = len(dataset_df)

    for feature in ALL_FEATURES:
        if feature not in dataset_df.columns:
            rows.append({
                "feature": feature,
                "group": "PRE" if feature in PRE_MATCH_FEATURES else "LIVE",
                "is_core": int(feature in core_features),
                "available": 0,
                "missing": total,
                "coverage_rate": 0.0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "mode": "",
                "mode_share": 0.0,
                "class": "MISSING_COLUMN",
            })
            continue

        values = numeric_series(
            dataset_df,
            feature,
        )

        available = int(values.notna().sum())
        missing = total - available
        coverage = available / total if total else 0.0

        non_na = values.dropna()

        if len(non_na):
            mode_value = non_na.mode()
            if len(mode_value):
                mode = float(mode_value.iloc[0])
                mode_share = float((non_na == mode).mean())
            else:
                mode = np.nan
                mode_share = 0.0

            mean = float(non_na.mean())
            std = float(non_na.std(ddof=0))
            min_value = float(non_na.min())
            max_value = float(non_na.max())
        else:
            mode = np.nan
            mode_share = 0.0
            mean = np.nan
            std = np.nan
            min_value = np.nan
            max_value = np.nan

        if coverage >= 0.95:
            klass = "CORE_COVERAGE"
        elif coverage >= 0.80:
            klass = "WATCH"
        else:
            klass = "LOW_COVERAGE"

        rows.append({
            "feature": feature,
            "group": "PRE" if feature in PRE_MATCH_FEATURES else "LIVE",
            "is_core": int(feature in core_features),
            "available": available,
            "missing": missing,
            "coverage_rate": coverage,
            "mean": mean,
            "std": std,
            "min": min_value,
            "max": max_value,
            "mode": mode,
            "mode_share": mode_share,
            "class": klass,
        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        by=[
            "is_core",
            "coverage_rate",
            "group",
            "feature",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)


# =============================================================================
# PROBLEM ROWS
# =============================================================================

def row_missing_features(
    row: Mapping[str, Any],
    features: Sequence[str],
) -> List[str]:
    missing = []

    for feature in features:
        if safe_float(row.get(feature)) is None:
            missing.append(feature)

    return missing


def validate_live_row(
    row: Mapping[str, Any],
) -> List[str]:
    errors = []

    # No future leakage for recency features.
    for feature in (
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ):
        value = safe_float(row.get(feature))

        if value is not None and value > 87:
            errors.append(f"{feature}_gt87")

    # Score consistency.
    home = safe_float(row.get("score_87_home"))
    away = safe_float(row.get("score_87_away"))

    if home is not None and away is not None:
        expected_diff = abs(home - away)
        expected_draw = 1.0 if home == away else 0.0

        if home > away:
            expected_leading = 0.0
        elif away > home:
            expected_leading = 1.0
        else:
            expected_leading = 2.0

        diff = safe_float(row.get("goal_diff_87"))
        draw = safe_float(row.get("is_draw_87"))
        leading = safe_float(row.get("leading_team_87"))

        if diff is not None and not math.isclose(
            diff,
            expected_diff,
            abs_tol=1e-9,
            rel_tol=0,
        ):
            errors.append("goal_diff_inconsistent")

        if draw is not None and not math.isclose(
            draw,
            expected_draw,
            abs_tol=1e-9,
            rel_tol=0,
        ):
            errors.append("is_draw_inconsistent")

        if leading is not None and not math.isclose(
            leading,
            expected_leading,
            abs_tol=1e-9,
            rel_tol=0,
        ):
            errors.append("leading_team_inconsistent")

    # SOT <= shots.
    pairs = [
        ("home_shots_on_target_87", "home_shots_87"),
        ("away_shots_on_target_87", "away_shots_87"),
        ("home_shots_off_target_87", "home_shots_87"),
        ("away_shots_off_target_87", "away_shots_87"),
        ("home_shots_blocked_87", "home_shots_87"),
        ("away_shots_blocked_87", "away_shots_87"),
        ("sot_80_87", "shots_80_87"),
    ]

    for child, parent in pairs:
        child_value = safe_float(row.get(child))
        parent_value = safe_float(row.get(parent))

        if (
            child_value is not None
            and parent_value is not None
            and child_value > parent_value
        ):
            errors.append(f"{child}_gt_{parent}")

    # Recent events cannot exceed cumulative totals.
    recent_pairs = [
        ("shots_80_87", "home_shots_87", "away_shots_87"),
        ("sot_80_87", "home_shots_on_target_87", "away_shots_on_target_87"),
        ("corners_80_87", "home_corners_87", "away_corners_87"),
        ("fouls_80_87", "home_fouls_87", "away_fouls_87"),
        ("cards_80_87", "home_yellow_cards_87", "away_yellow_cards_87"),
    ]

    for recent, home_col, away_col in recent_pairs:
        recent_value = safe_float(row.get(recent))
        home_value = safe_float(row.get(home_col))
        away_value = safe_float(row.get(away_col))

        if (
            recent_value is not None
            and home_value is not None
            and away_value is not None
            and recent_value > home_value + away_value
        ):
            errors.append(f"{recent}_gt_total_{home_col}_{away_col}")

    return sorted(set(errors))


def validate_pre_row(
    row: Mapping[str, Any],
) -> List[str]:
    errors = []

    for side in ("home", "away"):
        mp = safe_float(row.get(f"{side}_team_matches_played"))
        wins = safe_float(row.get(f"{side}_team_wins"))
        draws = safe_float(row.get(f"{side}_team_draws"))
        losses = safe_float(row.get(f"{side}_team_losses"))
        points = safe_float(row.get(f"{side}_team_points_standing"))
        gf = safe_float(row.get(f"{side}_team_goals_for"))
        ga = safe_float(row.get(f"{side}_team_goals_against"))
        gd = safe_float(row.get(f"{side}_team_goal_difference"))

        if all(value is not None for value in (mp, wins, draws, losses)):
            if not math.isclose(
                mp,
                wins + draws + losses,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                errors.append(f"{side}_standing_matches_inconsistent")

        if all(value is not None for value in (points, wins, draws)):
            if not math.isclose(
                points,
                3 * wins + draws,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                errors.append(f"{side}_standing_points_inconsistent")

        if all(value is not None for value in (gf, ga, gd)):
            if not math.isclose(
                gd,
                gf - ga,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                errors.append(f"{side}_standing_gd_inconsistent")

    # Frequency ranges.
    frequency_features = [
        feature
        for feature in PRE_MATCH_FEATURES
        if (
            feature.endswith("_freq")
            or feature.startswith("league_")
            or feature.startswith("h2h5_")
        )
    ]

    for feature in frequency_features:
        value = safe_float(row.get(feature))

        if value is None:
            continue

        if value < 0 or value > 1:
            errors.append(f"{feature}_outside_0_1")

    # Form ranges if using standard points.
    for feature, max_value in (
        ("home_form5", 15),
        ("away_form5", 15),
        ("home_form10", 30),
        ("away_form10", 30),
    ):
        value = safe_float(row.get(feature))

        if value is None:
            continue

        if value < 0 or value > max_value:
            errors.append(f"{feature}_outside_expected_range")

    return sorted(set(errors))


def build_problem_rows(
    dataset_df: pd.DataFrame,
    core_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    problem_rows = []
    missing_rows = []

    if dataset_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in dataset_df.iterrows():
        row_dict = row.to_dict()
        match_id = clean_text(row_dict.get(ID_COLUMN))

        pre_available = count_available_row(
            row_dict,
            PRE_MATCH_FEATURES,
        )

        live_available = count_available_row(
            row_dict,
            LIVE_FEATURES,
        )

        core_available = count_available_row(
            row_dict,
            core_features,
        )

        core_missing = row_missing_features(
            row_dict,
            core_features,
        )

        pre_errors = validate_pre_row(row_dict)
        live_errors = validate_live_row(row_dict)

        target = safe_float(row_dict.get(TARGET_COLUMN))
        target_ok = target in (0.0, 1.0)

        if core_missing:
            missing_rows.append({
                "match_id": match_id,
                "home_team": row_dict.get("home_team", ""),
                "away_team": row_dict.get("away_team", ""),
                "competition_id": row_dict.get("competition_id", ""),
                "season_id": row_dict.get("season_id", ""),
                "pre_available": pre_available,
                "live_available": live_available,
                "core_available": core_available,
                "core_total": len(core_features),
                "core_missing_count": len(core_missing),
                "core_missing_features": "|".join(core_missing),
                "target": row_dict.get(TARGET_COLUMN, np.nan),
            })

        reasons = []

        if not target_ok:
            reasons.append("target_missing_or_invalid")

        if core_missing:
            reasons.append("core_incomplete")

        if pre_errors:
            reasons.extend(pre_errors)

        if live_errors:
            reasons.extend(live_errors)

        raw_validation = clean_text(
            row_dict.get("_validation_errors", "")
        )

        if raw_validation:
            reasons.append(f"collector_validation:{raw_validation}")

        if reasons:
            problem_rows.append({
                "match_id": match_id,
                "home_team": row_dict.get("home_team", ""),
                "away_team": row_dict.get("away_team", ""),
                "competition_id": row_dict.get("competition_id", ""),
                "season_id": row_dict.get("season_id", ""),
                "pre_available": pre_available,
                "live_available": live_available,
                "core_available": core_available,
                "core_total": len(core_features),
                "target": row_dict.get(TARGET_COLUMN, np.nan),
                "problem_reasons": "|".join(sorted(set(reasons))),
            })

    return (
        pd.DataFrame(problem_rows),
        pd.DataFrame(missing_rows),
    )


# =============================================================================
# COMPETITION QUALITY
# =============================================================================

def build_competition_quality(
    dataset_df: pd.DataFrame,
    core_features: Sequence[str],
) -> pd.DataFrame:
    if dataset_df.empty or "competition_id" not in dataset_df.columns:
        return pd.DataFrame()

    rows = []

    for competition_id, group in dataset_df.groupby("competition_id", dropna=False):
        competition_id = clean_text(competition_id)

        total = len(group)

        if not total:
            continue

        core_complete = 0
        live_mean = 0.0
        pre_mean = 0.0
        core_mean = 0.0

        pre_counts = []
        live_counts = []
        core_counts = []

        for _, row in group.iterrows():
            row_dict = row.to_dict()

            pre_counts.append(
                count_available_row(row_dict, PRE_MATCH_FEATURES)
            )

            live_counts.append(
                count_available_row(row_dict, LIVE_FEATURES)
            )

            core_count = count_available_row(row_dict, core_features)
            core_counts.append(core_count)

            if core_count == len(core_features):
                core_complete += 1

        target = pd.to_numeric(
            group[TARGET_COLUMN],
            errors="coerce",
        ) if TARGET_COLUMN in group.columns else pd.Series(dtype=float)

        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())
        labeled = positives + negatives

        sample_home = ""
        sample_away = ""

        if "home_team" in group.columns and len(group):
            sample_home = clean_text(group["home_team"].iloc[0])

        if "away_team" in group.columns and len(group):
            sample_away = clean_text(group["away_team"].iloc[0])

        rows.append({
            "competition_id": competition_id,
            "matches": total,
            "core_complete": core_complete,
            "core_complete_rate": core_complete / total,
            "avg_pre_available": float(np.mean(pre_counts)) if pre_counts else 0.0,
            "avg_live_available": float(np.mean(live_counts)) if live_counts else 0.0,
            "avg_core_available": float(np.mean(core_counts)) if core_counts else 0.0,
            "positives": positives,
            "negatives": negatives,
            "labeled": labeled,
            "target_rate": positives / labeled if labeled else 0.0,
            "sample_match": f"{sample_home} vs {sample_away}",
        })

    return pd.DataFrame(rows).sort_values(
        by=[
            "core_complete_rate",
            "matches",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

def build_target_distribution(
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    if dataset_df.empty or TARGET_COLUMN not in dataset_df.columns:
        return pd.DataFrame(
            columns=[
                "target",
                "count",
                "share",
            ]
        )

    target = pd.to_numeric(
        dataset_df[TARGET_COLUMN],
        errors="coerce",
    )

    rows = []

    total = len(dataset_df)

    for label, mask in (
        ("0_negative", target == 0),
        ("1_positive", target == 1),
        ("nan_unresolved", target.isna()),
        ("other_invalid", ~(target.isin([0, 1]) | target.isna())),
    ):
        count = int(mask.sum())

        rows.append({
            "target": label,
            "count": count,
            "share": count / total if total else 0.0,
        })

    return pd.DataFrame(rows)


# =============================================================================
# INDEX QUALITY
# =============================================================================

def build_index_quality(
    index_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if index_df.empty:
        return pd.DataFrame([
            {
                "metric": "index_rows",
                "value": 0,
            }
        ])

    rows.append({
        "metric": "index_rows",
        "value": len(index_df),
    })

    if ID_COLUMN in index_df.columns:
        rows.append({
            "metric": "index_unique_match_ids",
            "value": int(index_df[ID_COLUMN].astype(str).nunique()),
        })

        rows.append({
            "metric": "index_duplicate_match_ids",
            "value": int(index_df[ID_COLUMN].astype(str).duplicated().sum()),
        })

    if "status" in index_df.columns:
        for status, count in index_df["status"].fillna("").astype(str).value_counts().items():
            rows.append({
                "metric": f"index_status_{status or 'blank'}",
                "value": int(count),
            })

    if not dataset_df.empty and ID_COLUMN in dataset_df.columns and ID_COLUMN in index_df.columns:
        dataset_ids = set(dataset_df[ID_COLUMN].dropna().astype(str))
        index_ids = set(index_df[ID_COLUMN].dropna().astype(str))

        rows.append({
            "metric": "dataset_rows",
            "value": len(dataset_df),
        })

        rows.append({
            "metric": "dataset_unique_match_ids",
            "value": len(dataset_ids),
        })

        rows.append({
            "metric": "dataset_ids_missing_from_index",
            "value": len(dataset_ids - index_ids),
        })

        rows.append({
            "metric": "index_done_ids_missing_from_dataset",
            "value": len(
                set(
                    index_df[
                        index_df.get("status", pd.Series([""] * len(index_df))).astype(str)
                        == "done"
                    ][ID_COLUMN].dropna().astype(str)
                )
                - dataset_ids
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY / REPORT
# =============================================================================

def build_summary(
    dataset_df: pd.DataFrame,
    ready_df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    problem_rows: pd.DataFrame,
    missing_rows: pd.DataFrame,
    core_features: Sequence[str],
) -> pd.DataFrame:
    rows = []

    raw_rows = len(dataset_df)
    ready_rows = len(ready_df)

    rows.append({
        "metric": "raw_rows",
        "value": raw_rows,
    })

    rows.append({
        "metric": "core_ready_rows_file",
        "value": ready_rows,
    })

    if not dataset_df.empty and ID_COLUMN in dataset_df.columns:
        rows.append({
            "metric": "unique_match_ids",
            "value": int(dataset_df[ID_COLUMN].astype(str).nunique()),
        })

        rows.append({
            "metric": "duplicate_match_ids",
            "value": int(dataset_df[ID_COLUMN].astype(str).duplicated().sum()),
        })

    if not dataset_df.empty and TARGET_COLUMN in dataset_df.columns:
        target = pd.to_numeric(
            dataset_df[TARGET_COLUMN],
            errors="coerce",
        )

        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())
        unresolved = int(target.isna().sum())
        labeled = positives + negatives

        rows.extend([
            {
                "metric": "positives",
                "value": positives,
            },
            {
                "metric": "negatives",
                "value": negatives,
            },
            {
                "metric": "unresolved_target",
                "value": unresolved,
            },
            {
                "metric": "target_rate_labeled",
                "value": positives / labeled if labeled else 0.0,
            },
        ])

    if (
        not dataset_df.empty
        and all(feature in dataset_df.columns for feature in core_features)
    ):
        core_complete = int(
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
            .sum()
        )

        rows.append({
            "metric": "core_complete_recomputed",
            "value": core_complete,
        })

        rows.append({
            "metric": "core_complete_rate",
            "value": core_complete / raw_rows if raw_rows else 0.0,
        })

    rows.extend([
        {
            "metric": "core_feature_count",
            "value": len(core_features),
        },
        {
            "metric": "index_rows",
            "value": len(index_df),
        },
        {
            "metric": "rejection_rows",
            "value": len(rejections_df),
        },
        {
            "metric": "problem_rows",
            "value": len(problem_rows),
        },
        {
            "metric": "core_missing_rows",
            "value": len(missing_rows),
        },
    ])

    if not feature_coverage.empty:
        core_cov = feature_coverage[
            feature_coverage["is_core"] == 1
        ]

        rows.append({
            "metric": "core_features_coverage_95_plus",
            "value": int((core_cov["coverage_rate"] >= 0.95).sum()),
        })

        rows.append({
            "metric": "core_features_coverage_100",
            "value": int((core_cov["coverage_rate"] == 1.0).sum()),
        })

        rows.append({
            "metric": "lowest_core_coverage",
            "value": float(core_cov["coverage_rate"].min()) if len(core_cov) else 0.0,
        })

    return pd.DataFrame(rows)


def top_low_coverage_features(
    feature_coverage: pd.DataFrame,
    core_only: bool,
    n: int = 15,
) -> pd.DataFrame:
    if feature_coverage.empty:
        return pd.DataFrame()

    df = feature_coverage.copy()

    if core_only:
        df = df[df["is_core"] == 1].copy()

    return df.sort_values(
        by=[
            "coverage_rate",
            "missing",
        ],
        ascending=[
            True,
            False,
        ],
    ).head(n)


def write_report(
    output_path: Path,
    dataset_df: pd.DataFrame,
    ready_df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    problem_rows: pd.DataFrame,
    missing_rows: pd.DataFrame,
    competition_quality: pd.DataFrame,
    target_distribution: pd.DataFrame,
    index_quality: pd.DataFrame,
    core_features: Sequence[str],
    dataset_dir: Path,
    schema_path: Path,
) -> None:
    raw_rows = len(dataset_df)
    ready_rows = len(ready_df)

    target = (
        pd.to_numeric(dataset_df[TARGET_COLUMN], errors="coerce")
        if not dataset_df.empty and TARGET_COLUMN in dataset_df.columns
        else pd.Series(dtype=float)
    )

    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    unresolved = int(target.isna().sum())
    labeled = positives + negatives
    target_rate = positives / labeled if labeled else 0.0

    if (
        not dataset_df.empty
        and all(feature in dataset_df.columns for feature in core_features)
    ):
        core_complete = int(
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
            .sum()
        )
    else:
        core_complete = 0

    status_counts = value_counts_dict(index_df, "status")

    low_core = top_low_coverage_features(
        feature_coverage,
        core_only=True,
        n=20,
    )

    low_all = top_low_coverage_features(
        feature_coverage,
        core_only=False,
        n=20,
    )

    problem_reason_counts = Counter()

    if not problem_rows.empty and "problem_reasons" in problem_rows.columns:
        for value in problem_rows["problem_reasons"].fillna("").astype(str):
            for reason in value.split("|"):
                if reason:
                    problem_reason_counts[reason] += 1

    lines = [
        "# Audit Historical Dataset V2 — Goal After 87",
        "",
        "## Source",
        "",
        f"- Dataset dir : `{dataset_dir}`",
        f"- Schema : `{schema_path}`",
        "",
        "## Résumé",
        "",
        f"- RAW rows : **{raw_rows}**",
        f"- CORE_READY rows fichier : **{ready_rows}**",
        f"- CORE_READY recomputé : **{core_complete}**",
        f"- Features CORE : **{len(core_features)}**",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Targets non résolus : **{unresolved}**",
        f"- Target rate sur labels résolus : **{target_rate:.2%}**",
        f"- Index rows : **{len(index_df)}**",
        f"- Rejections/partials file rows : **{len(rejections_df)}**",
        f"- Problem rows détectées : **{len(problem_rows)}**",
        "",
        "## Status index",
        "",
    ]

    if status_counts:
        for key, value in status_counts.items():
            lines.append(f"- `{key or 'blank'}` : {value}")
    else:
        lines.append("- Index vide ou absent.")

    lines.extend([
        "",
        "## Distribution target",
        "",
    ])

    if not target_distribution.empty:
        for _, row in target_distribution.iterrows():
            lines.append(
                f"- `{row['target']}` : {int(row['count'])} "
                f"({float(row['share']):.2%})"
            )
    else:
        lines.append("- Aucune distribution disponible.")

    lines.extend([
        "",
        "## Features CORE les plus fragiles",
        "",
    ])

    if low_core.empty:
        lines.append("- Aucune.")
    else:
        for _, row in low_core.iterrows():
            lines.append(
                f"- `{row['feature']}` ({row['group']}) : "
                f"{float(row['coverage_rate']):.2%} "
                f"— missing {int(row['missing'])}/{int(row['available']) + int(row['missing'])}"
            )

    lines.extend([
        "",
        "## Features globales les moins disponibles",
        "",
    ])

    if low_all.empty:
        lines.append("- Aucune.")
    else:
        for _, row in low_all.iterrows():
            lines.append(
                f"- `{row['feature']}` ({row['group']}, core={int(row['is_core'])}) : "
                f"{float(row['coverage_rate']):.2%}"
            )

    lines.extend([
        "",
        "## Raisons des problèmes",
        "",
    ])

    if problem_reason_counts:
        for reason, count in problem_reason_counts.most_common():
            lines.append(f"- `{reason}` : {count}")
    else:
        lines.append("- Aucune incohérence détectée.")

    lines.extend([
        "",
        "## Compétitions à surveiller",
        "",
    ])

    if competition_quality.empty:
        lines.append("- Pas assez d'information compétition.")
    else:
        watch = competition_quality.head(15)

        for _, row in watch.iterrows():
            lines.append(
                f"- `{row['competition_id']}` : "
                f"{int(row['matches'])} match(s), "
                f"CORE complet {float(row['core_complete_rate']):.2%}, "
                f"PRE moyen {float(row['avg_pre_available']):.1f}/59, "
                f"LIVE moyen {float(row['avg_live_available']):.1f}/37, "
                f"ex: {row['sample_match']}"
            )

    lines.extend([
        "",
        "## Décision recommandée",
        "",
    ])

    if raw_rows < 100:
        lines.extend([
            "Le dataset est encore petit. Utilise cet audit pour corriger les problèmes évidents,",
            "puis lance une collecte plus large avec `--limit 100 --resume`, puis `--limit 500 --resume`.",
        ])
    elif core_complete / raw_rows >= 0.90 if raw_rows else False:
        lines.extend([
            "La couverture CORE est bonne. Tu peux continuer la collecte historique incrémentale.",
            "Après 1 000+ lignes CORE_READY, il faudra auditer la distribution des features et préparer `train_model_v2.py`.",
        ])
    else:
        lines.extend([
            "La couverture CORE est encore trop instable. Il faut identifier les features CORE manquantes",
            "et/ou certaines compétitions problématiques avant de lancer une collecte massive.",
        ])

    lines.extend([
        "",
        "## Fichiers produits",
        "",
        "- `audit_summary_v2.csv`",
        "- `audit_feature_coverage_v2.csv`",
        "- `audit_core_missing_by_row_v2.csv`",
        "- `audit_problem_rows_v2.csv`",
        "- `audit_competition_quality_v2.csv`",
        "- `audit_target_distribution_v2.csv`",
        "- `audit_index_quality_v2.csv`",
        "",
        "Le scanner de production n'a pas été modifié.",
    ])

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit du dataset historique V2 Goal After 87."
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dossier outputs/historical_dataset_v2.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Chemin vers model_schema_v2.json.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie de l'audit.",
    )

    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    schema_path = resolve_schema_path(args.schema)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = dataset_dir / RAW_FILENAME
    ready_path = dataset_dir / READY_FILENAME
    index_path = dataset_dir / INDEX_FILENAME
    rejections_path = dataset_dir / REJECTIONS_FILENAME

    print()
    print("=" * 100)
    print("AUDIT HISTORICAL DATASET V2 — GOAL AFTER 87")
    print("=" * 100)

    print(f"Dataset dir : {dataset_dir}")
    print(f"Schema : {schema_path}")

    dataset_df = read_csv_if_exists(raw_path)
    ready_df = read_csv_if_exists(ready_path)
    index_df = read_csv_if_exists(index_path)
    rejections_df = read_csv_if_exists(rejections_path)

    core_features = load_core_features(schema_path)

    print(f"RAW rows : {len(dataset_df)}")
    print(f"READY rows : {len(ready_df)}")
    print(f"Index rows : {len(index_df)}")
    print(f"Rejections rows : {len(rejections_df)}")
    print(f"CORE features : {len(core_features)}")

    feature_coverage = build_feature_coverage(
        dataset_df,
        core_features,
    )

    problem_rows, missing_rows = build_problem_rows(
        dataset_df,
        core_features,
    )

    competition_quality = build_competition_quality(
        dataset_df,
        core_features,
    )

    target_distribution = build_target_distribution(
        dataset_df,
    )

    index_quality = build_index_quality(
        index_df,
        dataset_df,
    )

    summary = build_summary(
        dataset_df=dataset_df,
        ready_df=ready_df,
        index_df=index_df,
        rejections_df=rejections_df,
        feature_coverage=feature_coverage,
        problem_rows=problem_rows,
        missing_rows=missing_rows,
        core_features=core_features,
    )

    summary_path = output_dir / "audit_summary_v2.csv"
    feature_path = output_dir / "audit_feature_coverage_v2.csv"
    missing_path = output_dir / "audit_core_missing_by_row_v2.csv"
    problems_path = output_dir / "audit_problem_rows_v2.csv"
    competitions_path = output_dir / "audit_competition_quality_v2.csv"
    target_path = output_dir / "audit_target_distribution_v2.csv"
    index_quality_path = output_dir / "audit_index_quality_v2.csv"
    report_path = output_dir / "audit_historical_dataset_v2_report.md"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_coverage.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    missing_rows.to_csv(
        missing_path,
        index=False,
        encoding="utf-8-sig",
    )

    problem_rows.to_csv(
        problems_path,
        index=False,
        encoding="utf-8-sig",
    )

    competition_quality.to_csv(
        competitions_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_distribution.to_csv(
        target_path,
        index=False,
        encoding="utf-8-sig",
    )

    index_quality.to_csv(
        index_quality_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_report(
        output_path=report_path,
        dataset_df=dataset_df,
        ready_df=ready_df,
        index_df=index_df,
        rejections_df=rejections_df,
        feature_coverage=feature_coverage,
        problem_rows=problem_rows,
        missing_rows=missing_rows,
        competition_quality=competition_quality,
        target_distribution=target_distribution,
        index_quality=index_quality,
        core_features=core_features,
        dataset_dir=dataset_dir,
        schema_path=schema_path,
    )

    target = (
        pd.to_numeric(dataset_df[TARGET_COLUMN], errors="coerce")
        if not dataset_df.empty and TARGET_COLUMN in dataset_df.columns
        else pd.Series(dtype=float)
    )

    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    unresolved = int(target.isna().sum())
    labeled = positives + negatives

    if (
        not dataset_df.empty
        and all(feature in dataset_df.columns for feature in core_features)
    ):
        core_complete = int(
            dataset_df[list(core_features)]
            .notna()
            .all(axis=1)
            .sum()
        )
    else:
        core_complete = 0

    print()
    print("-" * 100)
    print(f"RAW rows : {len(dataset_df)}")
    print(f"CORE complet recomputé : {core_complete}")
    print(f"Positifs : {positives}")
    print(f"Négatifs : {negatives}")
    print(f"Targets non résolus : {unresolved}")
    print(
        "Target rate : "
        f"{(positives / labeled if labeled else 0.0):.2%}"
    )
    print(f"Problem rows : {len(problem_rows)}")
    print(f"Core missing rows : {len(missing_rows)}")
    print("-" * 100)

    print("Sorties :")
    print(f"  {summary_path}")
    print(f"  {feature_path}")
    print(f"  {missing_path}")
    print(f"  {problems_path}")
    print(f"  {competitions_path}")
    print(f"  {target_path}")
    print(f"  {index_quality_path}")
    print(f"  {report_path}")

    print()
    print("Le scanner de production n'a pas été modifié.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
