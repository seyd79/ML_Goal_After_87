#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_clean_dataset_v2.py

CONSTRUCTION D'UN NOUVEAU DATASET PROSPECTIF PROPRE
===================================================

But
---
Construire un dataset d'entraînement propre à partir des snapshots LIVE réellement
capturés par le scanner, et NON à partir de l'ancien dataset historique de 13 800 matchs.

Principe
--------
- état du match capturé EXACTEMENT à 87'
- 96 features brutes conservées telles qu'observées
- aucune imputation artificielle
- aucune valeur manquante remplacée par 0
- cible issue uniquement d'un résultat résolu avec forte confiance
- positifs = but de temps réglementaire strictement APRÈS 87'
- négatifs = timeline sans but >87 + score final inchangé
- ambigu / pending / non résolu = rejeté
- une seule ligne par match
- contrôle anti-fuite temporelle : aucune feature "last_*_minute" > 87
- contrôle des invariants du score à 87'
- production de plusieurs sorties :
    1) dataset FULL 96/96 strict -> recommandé pour l'entraînement initial
    2) dataset raw labellisé -> conserve les lignes incomplètes pour études futures
    3) rejets + raisons détaillées
    4) rapport qualité

IMPORTANT
---------
Ce script ne modifie PAS :
- le scanner de production
- le modèle actuel
- les snapshots source
- les labels V3

Il construit un NOUVEAU dataset séparé.

Entrées supportées
------------------
Mode A — fichier déjà labellisé par snapshot :
    outputs/history/live_feature_snapshots_labeled_v3.csv

Mode B — snapshots + labels séparés :
    outputs/history/live_feature_snapshots.csv
    prediction_results_by_match_v3.csv
ou
    outputs/history/live_feature_snapshots.csv
    signal87_audit_v3.csv

Utilisation
-----------
Automatique :
    python3 build_clean_dataset_v2.py

Entrées explicites :
    python3 build_clean_dataset_v2.py \
        --snapshots outputs/history/live_feature_snapshots.csv \
        --labels prediction_results_by_match_v3.csv

Fichier labellisé directement :
    python3 build_clean_dataset_v2.py \
        --labeled outputs/history/live_feature_snapshots_labeled_v3.csv

Sorties
-------
outputs/clean_dataset_v2/
├── Clean_Dataset_Goal_After_87_FULL_v2.csv
├── Clean_Dataset_Goal_After_87_RAW_v2.csv
├── Clean_Dataset_Goal_After_87_Rejected_v2.csv
├── clean_dataset_v2_summary.csv
└── clean_dataset_v2_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "clean_dataset_v2"
)

LABELED_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "history"
    / "live_feature_snapshots_labeled_v3.csv",

    BASE_DIR
    / "live_feature_snapshots_labeled_v3.csv",

    BASE_DIR
    / "outputs"
    / "live_feature_snapshots_labeled_v3.csv",
]

SNAPSHOT_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "history"
    / "live_feature_snapshots.csv",

    BASE_DIR
    / "outputs"
    / "history"
    / "live_feature_snapshots_labeled_v3.csv",

    BASE_DIR
    / "live_feature_snapshots.csv",
]

LABEL_CANDIDATES = [
    BASE_DIR
    / "prediction_results_by_match_v3.csv",

    BASE_DIR
    / "signal87_audit_v3.csv",

    BASE_DIR
    / "outputs"
    / "history"
    / "prediction_results_by_match_v3.csv",

    BASE_DIR
    / "outputs"
    / "history"
    / "signal87_audit_v3.csv",
]


# =============================================================================
# SCHEMA
# =============================================================================

ID_COLUMN = "match_id"
TARGET_COLUMN = "goal_after_87"

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

FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES

assert len(PRE_MATCH_FEATURES) == 59
assert len(LIVE_FEATURES) == 37
assert len(FEATURES) == 96


# =============================================================================
# COLUMN ALIASES
# =============================================================================

MINUTE_ALIASES = [
    "minute",
    "snapshot_minute",
    "live_minute",
    "match_minute",
    "current_minute",
]

CAPTURE_TIME_ALIASES = [
    "captured_at_utc",
    "captured_at",
    "timestamp",
    "created_at",
    "scan_timestamp",
    "snapshot_time",
]

QUALITY_ALIASES = [
    "quality",
    "quality_status",
    "data_quality",
    "model_quality",
    "coverage_quality",
]

COVERAGE_ALIASES = [
    "coverage",
    "feature_coverage",
    "available_features",
    "features_available",
    "coverage_count",
]

TARGET_STATUS_ALIASES = [
    "target_status",
]

RESOLUTION_CLASS_ALIASES = [
    "resolution_class",
    "Resolution_class",
]

CONFIDENCE_ALIASES = [
    "target_confidence",
    "Target_confidence",
]

TARGET_ALIASES = [
    "goal_after_87",
]

FIRST_GOAL_AFTER87_ALIASES = [
    "valid_goal_minutes_after_87",
    "first_goal_after_87_minute",
    "goal_minute_after_87",
]

HOME_NAME_ALIASES = [
    "home_team",
    "home_team_name",
]

AWAY_NAME_ALIASES = [
    "away_team",
    "away_team_name",
]

COMPETITION_ALIASES = [
    "competition_id",
]

SEASON_ALIASES = [
    "season_id",
]


# =============================================================================
# HELPERS
# =============================================================================

def first_existing(
    df: pd.DataFrame,
    names: Sequence[str],
) -> Optional[str]:

    for name in names:
        if name in df.columns:
            return name

    return None


def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def norm_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def upper_text(
    value: Any,
) -> str:

    return norm_text(
        value
    ).upper()


def resolve_existing_path(
    explicit: Optional[str],
    candidates: Sequence[Path],
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
                f"Fichier introuvable : {path}"
            )

        return path

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def parse_coverage_value(
    value: Any,
) -> Optional[int]:

    if value is None:
        return None

    text = norm_text(
        value
    )

    if not text:
        return None

    # "FULL 96/96", "DEGRADED 95/96", "96", etc.
    import re

    found = re.search(
        r"(\d+)\s*/\s*96",
        text,
    )

    if found:
        return int(
            found.group(1)
        )

    found = re.search(
        r"\b(\d{1,2})\b",
        text,
    )

    if found:
        parsed = int(
            found.group(1)
        )

        if 0 <= parsed <= 96:
            return parsed

    try:
        parsed = int(
            float(text)
        )

        if 0 <= parsed <= 96:
            return parsed

    except Exception:
        pass

    return None


def extract_first_goal_after87(
    value: Any,
) -> Optional[int]:

    text = norm_text(
        value
    )

    if not text:
        return None

    import re

    numbers = [
        int(item)
        for item in re.findall(
            r"\d+",
            text,
        )
    ]

    after87 = [
        minute
        for minute in numbers
        if minute > 87
    ]

    if not after87:
        return None

    return min(
        after87
    )


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_inputs(
    labeled_path: Optional[Path],
    snapshots_path: Optional[Path],
    labels_path: Optional[Path],
) -> Tuple[pd.DataFrame, str]:

    if labeled_path is not None:
        df = pd.read_csv(
            labeled_path
        )

        return (
            df,
            f"labeled:{labeled_path}",
        )

    if snapshots_path is None:
        raise FileNotFoundError(
            "Aucun fichier snapshots trouvé."
        )

    snapshots = pd.read_csv(
        snapshots_path
    )

    # If snapshots already contain target columns, no external merge needed.
    if (
        TARGET_COLUMN in snapshots.columns
        and first_existing(
            snapshots,
            TARGET_STATUS_ALIASES,
        )
        is not None
    ):
        return (
            snapshots,
            f"snapshots_already_labeled:{snapshots_path}",
        )

    if labels_path is None:
        raise FileNotFoundError(
            "Les snapshots ne sont pas labellisés et aucun fichier labels V3 n'a été trouvé."
        )

    labels = pd.read_csv(
        labels_path
    )

    if ID_COLUMN not in snapshots.columns:
        raise ValueError(
            f"{ID_COLUMN} absent des snapshots."
        )

    if ID_COLUMN not in labels.columns:
        raise ValueError(
            f"{ID_COLUMN} absent des labels."
        )

    # Keep only label/audit fields from labels to avoid duplicate feature names.
    label_columns = [
        column
        for column in labels.columns
        if (
            column == ID_COLUMN
            or column not in snapshots.columns
        )
    ]

    labels_small = labels[
        label_columns
    ].copy()

    # Deduplicate labels by match.
    labels_small = (
        labels_small
        .drop_duplicates(
            subset=[ID_COLUMN],
            keep="last",
        )
    )

    merged = snapshots.merge(
        labels_small,
        how="left",
        on=ID_COLUMN,
        validate="many_to_one",
    )

    return (
        merged,
        (
            f"snapshots:{snapshots_path}"
            f" + labels:{labels_path}"
        ),
    )


# =============================================================================
# NORMALIZATION
# =============================================================================

def ensure_feature_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    missing_columns = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Features absentes du fichier source :\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_columns
            )
        )

    for feature in FEATURES:
        df[
            feature
        ] = safe_numeric(
            df[feature]
        )

    return df


def add_normalized_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    minute_col = first_existing(
        df,
        MINUTE_ALIASES,
    )

    if minute_col is None:
        raise ValueError(
            "Impossible de trouver la colonne minute du snapshot."
        )

    df[
        "_snapshot_minute"
    ] = safe_numeric(
        df[
            minute_col
        ]
    )

    capture_col = first_existing(
        df,
        CAPTURE_TIME_ALIASES,
    )

    if capture_col is not None:
        df[
            "_capture_time"
        ] = pd.to_datetime(
            df[
                capture_col
            ],
            errors="coerce",
            utc=True,
        )
    else:
        df[
            "_capture_time"
        ] = pd.NaT

    quality_col = first_existing(
        df,
        QUALITY_ALIASES,
    )

    if quality_col is not None:
        df[
            "_quality_text"
        ] = (
            df[
                quality_col
            ]
            .astype(str)
        )
    else:
        df[
            "_quality_text"
        ] = ""

    coverage_col = first_existing(
        df,
        COVERAGE_ALIASES,
    )

    coverage_values: List[
        Optional[int]
    ] = []

    for index, row in df.iterrows():

        coverage = None

        if coverage_col is not None:
            coverage = parse_coverage_value(
                row.get(
                    coverage_col
                )
            )

        if coverage is None:
            coverage = parse_coverage_value(
                row.get(
                    "_quality_text"
                )
            )

        if coverage is None:
            coverage = int(
                row[
                    FEATURES
                ]
                .notna()
                .sum()
            )

        coverage_values.append(
            coverage
        )

    df[
        "_coverage"
    ] = coverage_values

    target_status_col = first_existing(
        df,
        TARGET_STATUS_ALIASES,
    )

    resolution_col = first_existing(
        df,
        RESOLUTION_CLASS_ALIASES,
    )

    confidence_col = first_existing(
        df,
        CONFIDENCE_ALIASES,
    )

    target_col = first_existing(
        df,
        TARGET_ALIASES,
    )

    if target_col is None:
        raise ValueError(
            "goal_after_87 absent."
        )

    df[
        "_target_numeric"
    ] = safe_numeric(
        df[
            target_col
        ]
    )

    if target_status_col is not None:
        df[
            "_target_status"
        ] = df[
            target_status_col
        ].map(
            upper_text
        )
    else:
        df[
            "_target_status"
        ] = ""

    if resolution_col is not None:
        df[
            "_resolution_class"
        ] = df[
            resolution_col
        ].map(
            upper_text
        )
    else:
        df[
            "_resolution_class"
        ] = ""

    if confidence_col is not None:
        df[
            "_target_confidence"
        ] = df[
            confidence_col
        ].map(
            upper_text
        )
    else:
        df[
            "_target_confidence"
        ] = ""

    goal_minute_col = first_existing(
        df,
        FIRST_GOAL_AFTER87_ALIASES,
    )

    if goal_minute_col is not None:
        df[
            "_first_goal_after87"
        ] = df[
            goal_minute_col
        ].map(
            extract_first_goal_after87
        )
    else:
        df[
            "_first_goal_after87"
        ] = np.nan

    return df


# =============================================================================
# QUALITY / VALIDATION
# =============================================================================

def row_rejection_reasons(
    row: pd.Series,
    require_high_confidence: bool,
) -> List[str]:

    reasons: List[str] = []

    # -------------------------------------------------------------------------
    # Exact minute 87 only
    # -------------------------------------------------------------------------

    minute = row.get(
        "_snapshot_minute"
    )

    if pd.isna(
        minute
    ) or int(minute) != 87:
        reasons.append(
            "snapshot_not_exactly_87"
        )

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    target = row.get(
        "_target_numeric"
    )

    if (
        pd.isna(
            target
        )
        or float(target) not in (
            0.0,
            1.0,
        )
    ):
        reasons.append(
            "target_not_binary_or_missing"
        )

    target_status = upper_text(
        row.get(
            "_target_status"
        )
    )

    if (
        target_status
        and target_status != "LABELED"
    ):
        reasons.append(
            f"target_status_{target_status.lower()}"
        )

    resolution = upper_text(
        row.get(
            "_resolution_class"
        )
    )

    if (
        resolution
        and resolution
        not in (
            "POSITIVE",
            "NEGATIVE",
        )
    ):
        reasons.append(
            f"resolution_{resolution.lower()}"
        )

    confidence = upper_text(
        row.get(
            "_target_confidence"
        )
    )

    if require_high_confidence:

        if confidence != "HIGH":
            reasons.append(
                (
                    "target_confidence_not_high"
                    if confidence
                    else "target_confidence_missing"
                )
            )

    # -------------------------------------------------------------------------
    # Temporal leakage
    # -------------------------------------------------------------------------

    last_minute_features = [
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ]

    for feature in last_minute_features:

        value = row.get(
            feature
        )

        if pd.notna(
            value
        ) and float(
            value
        ) > 87:
            reasons.append(
                f"future_leak_{feature}_gt87"
            )

    # -------------------------------------------------------------------------
    # Score invariants
    # -------------------------------------------------------------------------

    sh = row.get(
        "score_87_home"
    )

    sa = row.get(
        "score_87_away"
    )

    diff = row.get(
        "goal_diff_87"
    )

    draw = row.get(
        "is_draw_87"
    )

    leading = row.get(
        "leading_team_87"
    )

    if (
        pd.notna(
            sh
        )
        and pd.notna(
            sa
        )
    ):
        expected_diff = abs(
            float(sh)
            - float(sa)
        )

        if (
            pd.notna(
                diff
            )
            and not np.isclose(
                float(diff),
                expected_diff,
                atol=1e-9,
            )
        ):
            reasons.append(
                "invalid_goal_diff_87"
            )

        expected_draw = (
            1.0
            if float(sh) == float(sa)
            else 0.0
        )

        if (
            pd.notna(
                draw
            )
            and not np.isclose(
                float(draw),
                expected_draw,
                atol=1e-9,
            )
        ):
            reasons.append(
                "invalid_is_draw_87"
            )

        if float(sh) > float(sa):
            expected_leading = 0.0
        elif float(sa) > float(sh):
            expected_leading = 1.0
        else:
            expected_leading = 2.0

        if (
            pd.notna(
                leading
            )
            and not np.isclose(
                float(leading),
                expected_leading,
                atol=1e-9,
            )
        ):
            reasons.append(
                "invalid_leading_team_87"
            )

    # -------------------------------------------------------------------------
    # Basic impossible LIVE states
    # -------------------------------------------------------------------------

    for side in (
        "home",
        "away",
    ):
        shots = row.get(
            f"{side}_shots_87"
        )

        sot = row.get(
            f"{side}_shots_on_target_87"
        )

        if (
            pd.notna(
                shots
            )
            and pd.notna(
                sot
            )
            and float(
                sot
            ) > float(
                shots
            )
        ):
            reasons.append(
                f"invalid_{side}_sot_gt_shots"
            )

    total_shots = None

    if (
        pd.notna(
            row.get(
                "home_shots_87"
            )
        )
        and pd.notna(
            row.get(
                "away_shots_87"
            )
        )
    ):
        total_shots = (
            float(
                row[
                    "home_shots_87"
                ]
            )
            + float(
                row[
                    "away_shots_87"
                ]
            )
        )

    recent_shots = row.get(
        "shots_80_87"
    )

    if (
        total_shots is not None
        and pd.notna(
            recent_shots
        )
        and float(
            recent_shots
        ) > total_shots
    ):
        reasons.append(
            "invalid_shots_80_87_gt_total"
        )

    total_sot = None

    if (
        pd.notna(
            row.get(
                "home_shots_on_target_87"
            )
        )
        and pd.notna(
            row.get(
                "away_shots_on_target_87"
            )
        )
    ):
        total_sot = (
            float(
                row[
                    "home_shots_on_target_87"
                ]
            )
            + float(
                row[
                    "away_shots_on_target_87"
                ]
            )
        )

    recent_sot = row.get(
        "sot_80_87"
    )

    if (
        total_sot is not None
        and pd.notna(
            recent_sot
        )
        and float(
            recent_sot
        ) > total_sot
    ):
        reasons.append(
            "invalid_sot_80_87_gt_total"
        )

    # -------------------------------------------------------------------------
    # Positive target should have a verified goal minute >87 when available.
    # -------------------------------------------------------------------------

    if (
        pd.notna(
            target
        )
        and int(
            target
        ) == 1
    ):
        goal_minute = row.get(
            "_first_goal_after87"
        )

        # If V3 provided the field but no minute survived, reject.
        if (
            "_first_goal_after87"
            in row.index
            and pd.isna(
                goal_minute
            )
        ):
            # Allow only when the label source itself is explicit official timeline.
            source = upper_text(
                row.get(
                    "target_source"
                )
            )

            if (
                "OFFICIAL_TIMELINE"
                not in source
            ):
                reasons.append(
                    "positive_without_verified_goal_minute"
                )

    return sorted(
        set(
            reasons
        )
    )


def compute_missing_features(
    row: pd.Series,
) -> List[str]:

    return [
        feature
        for feature in FEATURES
        if pd.isna(
            row.get(
                feature
            )
        )
    ]


def add_row_diagnostics(
    df: pd.DataFrame,
    require_high_confidence: bool,
) -> pd.DataFrame:

    df = df.copy()

    reasons_col = []
    missing_col = []
    observed_col = []

    for _, row in df.iterrows():

        reasons = row_rejection_reasons(
            row,
            require_high_confidence=(
                require_high_confidence
            ),
        )

        missing = compute_missing_features(
            row
        )

        reasons_col.append(
            "|".join(
                reasons
            )
        )

        missing_col.append(
            "|".join(
                missing
            )
        )

        observed_col.append(
            96 - len(
                missing
            )
        )

    df[
        "_rejection_reasons"
    ] = reasons_col

    df[
        "_missing_features"
    ] = missing_col

    df[
        "_observed_features"
    ] = observed_col

    return df


# =============================================================================
# DEDUPLICATION
# =============================================================================

def choose_one_snapshot_per_match(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Only exact 87 candidates before dedupe.
    exact = df[
        df[
            "_snapshot_minute"
        ]
        == 87
    ].copy()

    if exact.empty:
        return exact

    # Prefer:
    # 1. no semantic rejection
    # 2. most observed features
    # 3. highest reported coverage
    # 4. earliest capture at minute 87 to reduce risk of API minute lag / future leakage.
    exact[
        "_semantic_ok_rank"
    ] = (
        exact[
            "_rejection_reasons"
        ]
        == ""
    ).astype(
        int
    )

    exact[
        "_capture_missing_rank"
    ] = exact[
        "_capture_time"
    ].isna().astype(
        int
    )

    exact = exact.sort_values(
        by=[
            ID_COLUMN,
            "_semantic_ok_rank",
            "_observed_features",
            "_coverage",
            "_capture_missing_rank",
            "_capture_time",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
        ],
        na_position="last",
    )

    return (
        exact
        .drop_duplicates(
            subset=[
                ID_COLUMN
            ],
            keep="first",
        )
        .copy()
    )


# =============================================================================
# OUTPUT PREPARATION
# =============================================================================

def metadata_columns(
    df: pd.DataFrame,
) -> List[str]:

    candidates: List[str] = [
        ID_COLUMN,
    ]

    for aliases in (
        HOME_NAME_ALIASES,
        AWAY_NAME_ALIASES,
        COMPETITION_ALIASES,
        SEASON_ALIASES,
        CAPTURE_TIME_ALIASES,
        QUALITY_ALIASES,
        COVERAGE_ALIASES,
    ):
        col = first_existing(
            df,
            aliases,
        )

        if (
            col
            and col not in candidates
            and col not in FEATURES
        ):
            candidates.append(
                col
            )

    return candidates


def build_export_columns(
    df: pd.DataFrame,
    include_diagnostics: bool,
) -> List[str]:

    columns = (
        metadata_columns(
            df
        )
        + FEATURES
        + [
            TARGET_COLUMN,
        ]
    )

    if include_diagnostics:

        optional = [
            "target_status",
            "target_source",
            "resolution_class",
            "target_confidence",
            "valid_goal_minutes_after_87",
            "final_score_home",
            "final_score_away",
            "score_delta_after_snapshot87",
            "coherence_warning",
            "_snapshot_minute",
            "_coverage",
            "_observed_features",
            "_missing_features",
            "_rejection_reasons",
        ]

        columns.extend(
            [
                column
                for column in optional
                if column in df.columns
            ]
        )

    # de-duplicate while preserving order
    seen = set()
    result = []

    for column in columns:
        if (
            column in df.columns
            and column not in seen
        ):
            seen.add(
                column
            )
            result.append(
                column
            )

    return result


# =============================================================================
# REPORT
# =============================================================================

def make_report(
    source_description: str,
    original_rows: int,
    exact87_rows: int,
    deduped_rows: int,
    raw_clean_rows: int,
    full_rows: int,
    rejected_rows: int,
    full_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> str:

    if full_df.empty:
        positives = 0
        negatives = 0
        target_rate = 0.0
    else:
        positives = int(
            (
                full_df[
                    TARGET_COLUMN
                ]
                == 1
            ).sum()
        )

        negatives = int(
            (
                full_df[
                    TARGET_COLUMN
                ]
                == 0
            ).sum()
        )

        target_rate = float(
            full_df[
                TARGET_COLUMN
            ].mean()
        )

    reason_counts: Dict[
        str,
        int
    ] = {}

    if not rejected_df.empty:

        for value in rejected_df[
            "_rejection_reasons"
        ].fillna(
            ""
        ):
            for reason in norm_text(
                value
            ).split(
                "|"
            ):
                if not reason:
                    continue

                reason_counts[
                    reason
                ] = (
                    reason_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

        for value in rejected_df[
            "_missing_features"
        ].fillna(
            ""
        ):
            if value:
                reason_counts[
                    "not_full_96_96"
                ] = (
                    reason_counts.get(
                        "not_full_96_96",
                        0,
                    )
                    + 1
                )

    lines = [
        "# Clean Dataset Goal After 87 — V2",
        "",
        "## Principe",
        "",
        "Ce dataset est construit à partir des snapshots prospectifs réels du scanner.",
        "L'ancien dataset historique de 13 800 matchs n'est pas utilisé comme source d'entraînement.",
        "",
        "Règles :",
        "",
        "- snapshot exactement à 87'",
        "- 96 features brutes",
        "- aucune imputation",
        "- aucune valeur manquante remplacée par 0",
        "- target V3 résolu",
        "- confiance HIGH par défaut",
        "- positif uniquement si but réglementaire strictement après 87'",
        "- négatif uniquement si résolution V3 fiable",
        "- contrôle anti-fuite des événements après 87'",
        "- une seule ligne par match",
        "",
        "## Volumes",
        "",
        f"- Source : `{source_description}`",
        f"- Lignes source : **{original_rows}**",
        f"- Snapshots exactement à 87' : **{exact87_rows}**",
        f"- Matchs uniques après sélection du meilleur snapshot : **{deduped_rows}**",
        f"- Lignes labellisées sémantiquement propres : **{raw_clean_rows}**",
        f"- Lignes FULL 96/96 finales : **{full_rows}**",
        f"- Rejets : **{rejected_rows}**",
        "",
        "## Dataset FULL",
        "",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Taux de but après 87' : **{target_rate:.2%}**",
        "",
        "## Raisons principales de rejet",
        "",
    ]

    if reason_counts:
        for reason, count in sorted(
            reason_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            lines.append(
                f"- `{reason}` : {count}"
            )
    else:
        lines.append(
            "- Aucun rejet."
        )

    lines.extend([
        "",
        "## Recommandation",
        "",
        "`Clean_Dataset_Goal_After_87_FULL_v2.csv` est le dataset strict à utiliser "
        "pour un premier nouvel entraînement lorsque le volume devient suffisant.",
        "",
        "`Clean_Dataset_Goal_After_87_RAW_v2.csv` sert à étudier la disponibilité réelle "
        "des features et éventuellement concevoir plus tard une stratégie d'imputation propre.",
        "",
        "Ne pas fusionner ces données avec l'ancien dataset historique audité.",
    ])

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Construit un nouveau dataset propre Goal After 87 "
            "à partir des snapshots prospectifs."
        )
    )

    parser.add_argument(
        "--labeled",
        default=None,
        help=(
            "CSV de snapshots déjà labellisés V3."
        ),
    )

    parser.add_argument(
        "--snapshots",
        default=None,
        help=(
            "CSV brut live_feature_snapshots.csv."
        ),
    )

    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "CSV labels V3 par match."
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
        "--allow-medium-confidence",
        action="store_true",
        help=(
            "Autorise les labels MEDIUM. "
            "Par défaut seuls HIGH sont retenus."
        ),
    )

    args = parser.parse_args()

    labeled_path = resolve_existing_path(
        args.labeled,
        LABELED_CANDIDATES,
    )

    snapshots_path = None
    labels_path = None

    if labeled_path is None:

        snapshots_path = resolve_existing_path(
            args.snapshots,
            SNAPSHOT_CANDIDATES,
        )

        labels_path = resolve_existing_path(
            args.labels,
            LABEL_CANDIDATES,
        )

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
        "BUILD CLEAN DATASET V2 — GOAL AFTER 87"
    )
    print("=" * 100)

    df, source_description = load_inputs(
        labeled_path=labeled_path,
        snapshots_path=snapshots_path,
        labels_path=labels_path,
    )

    original_rows = len(
        df
    )

    print(
        f"Source : {source_description}"
    )

    print(
        f"Lignes chargées : {original_rows}"
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{ID_COLUMN} absent."
        )

    df = ensure_feature_schema(
        df
    )

    df = add_normalized_metadata(
        df
    )

    df = add_row_diagnostics(
        df,
        require_high_confidence=(
            not args.allow_medium_confidence
        ),
    )

    exact87_rows = int(
        (
            df[
                "_snapshot_minute"
            ]
            == 87
        ).sum()
    )

    # Keep pre-87 and post-87 rows in rejected audit later.
    non87 = df[
        df[
            "_snapshot_minute"
        ]
        != 87
    ].copy()

    selected = choose_one_snapshot_per_match(
        df
    )

    deduped_rows = len(
        selected
    )

    # -------------------------------------------------------------------------
    # Semantic clean set: exact 87 + reliable target + no invariant/leak issue.
    # Missing features are allowed here and kept RAW.
    # -------------------------------------------------------------------------

    semantic_clean = selected[
        selected[
            "_rejection_reasons"
        ]
        == ""
    ].copy()

    raw_clean_rows = len(
        semantic_clean
    )

    # -------------------------------------------------------------------------
    # FULL: all 96 values observed.
    # -------------------------------------------------------------------------

    full = semantic_clean[
        semantic_clean[
            "_observed_features"
        ]
        == 96
    ].copy()

    # Enforce target as integer.
    full[
        TARGET_COLUMN
    ] = full[
        "_target_numeric"
    ].astype(
        int
    )

    semantic_clean[
        TARGET_COLUMN
    ] = semantic_clean[
        "_target_numeric"
    ].astype(
        int
    )

    # -------------------------------------------------------------------------
    # Rejects
    # -------------------------------------------------------------------------

    selected_rejected = selected[
        selected[
            "_rejection_reasons"
        ]
        != ""
    ].copy()

    incomplete = semantic_clean[
        semantic_clean[
            "_observed_features"
        ]
        < 96
    ].copy()

    # For reporting, incomplete clean rows are not "bad labels" but rejected from FULL.
    if not incomplete.empty:
        incomplete[
            "_rejection_reasons"
        ] = incomplete[
            "_rejection_reasons"
        ].astype(
            str
        )

        incomplete[
            "_rejection_reasons"
        ] = (
            incomplete[
                "_rejection_reasons"
            ]
            .replace(
                "nan",
                "",
            )
        )

    rejected = pd.concat(
        [
            non87,
            selected_rejected,
            incomplete,
        ],
        ignore_index=True,
        sort=False,
    )

    # -------------------------------------------------------------------------
    # Exports
    # -------------------------------------------------------------------------

    full_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_FULL_v2.csv"
    )

    raw_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_RAW_v2.csv"
    )

    rejected_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_Rejected_v2.csv"
    )

    summary_path = (
        output_dir
        / "clean_dataset_v2_summary.csv"
    )

    report_path = (
        output_dir
        / "clean_dataset_v2_report.md"
    )

    full_columns = (
        metadata_columns(
            full
        )
        + FEATURES
        + [
            TARGET_COLUMN
        ]
    )

    # Remove duplicate column names.
    full_columns = list(
        dict.fromkeys(
            [
                col
                for col in full_columns
                if col in full.columns
            ]
        )
    )

    full[
        full_columns
    ].to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    raw_columns = build_export_columns(
        semantic_clean,
        include_diagnostics=True,
    )

    semantic_clean[
        raw_columns
    ].to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    rejected_columns = build_export_columns(
        rejected,
        include_diagnostics=True,
    )

    rejected[
        rejected_columns
    ].to_csv(
        rejected_path,
        index=False,
        encoding="utf-8-sig",
    )

    positives = (
        int(
            (
                full[
                    TARGET_COLUMN
                ]
                == 1
            ).sum()
        )
        if not full.empty
        else 0
    )

    negatives = (
        int(
            (
                full[
                    TARGET_COLUMN
                ]
                == 0
            ).sum()
        )
        if not full.empty
        else 0
    )

    summary = pd.DataFrame([
        {
            "metric":
                "source_rows",
            "value":
                original_rows,
        },
        {
            "metric":
                "exact_87_rows",
            "value":
                exact87_rows,
        },
        {
            "metric":
                "unique_match_rows_after_selection",
            "value":
                deduped_rows,
        },
        {
            "metric":
                "semantic_clean_rows",
            "value":
                raw_clean_rows,
        },
        {
            "metric":
                "full_96_96_rows",
            "value":
                len(
                    full
                ),
        },
        {
            "metric":
                "full_positive_rows",
            "value":
                positives,
        },
        {
            "metric":
                "full_negative_rows",
            "value":
                negatives,
        },
        {
            "metric":
                "full_target_rate",
            "value":
                (
                    float(
                        full[
                            TARGET_COLUMN
                        ].mean()
                    )
                    if not full.empty
                    else np.nan
                ),
        },
        {
            "metric":
                "rejected_rows",
            "value":
                len(
                    rejected
                ),
        },
    ])

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = make_report(
        source_description=(
            source_description
        ),
        original_rows=(
            original_rows
        ),
        exact87_rows=(
            exact87_rows
        ),
        deduped_rows=(
            deduped_rows
        ),
        raw_clean_rows=(
            raw_clean_rows
        ),
        full_rows=(
            len(
                full
            )
        ),
        rejected_rows=(
            len(
                rejected
            )
        ),
        full_df=(
            full
        ),
        rejected_df=(
            rejected
        ),
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------

    print()
    print("-" * 100)
    print(
        f"Snapshots exactement à 87' : "
        f"{exact87_rows}"
    )

    print(
        f"Matchs uniques sélectionnés : "
        f"{deduped_rows}"
    )

    print(
        f"Lignes sémantiquement propres : "
        f"{raw_clean_rows}"
    )

    print(
        f"FULL 96/96 : "
        f"{len(full)}"
    )

    print(
        f"  Positifs : {positives}"
    )

    print(
        f"  Négatifs : {negatives}"
    )

    if not full.empty:
        print(
            f"  Target rate : "
            f"{full[TARGET_COLUMN].mean():.2%}"
        )

    print(
        f"Rejets / hors FULL : "
        f"{len(rejected)}"
    )

    print("-" * 100)
    print(
        f"Dataset FULL : {full_path}"
    )
    print(
        f"Dataset RAW : {raw_path}"
    )
    print(
        f"Rejets : {rejected_path}"
    )
    print(
        f"Résumé : {summary_path}"
    )
    print(
        f"Rapport : {report_path}"
    )
    print("=" * 100)

    print()
    print(
        "Aucune imputation n'a été appliquée."
    )
    print(
        "L'ancien dataset historique n'a pas été utilisé."
    )
    print(
        "Le scanner de production n'a pas été modifié."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
