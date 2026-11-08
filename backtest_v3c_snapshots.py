#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BACKTEST V3-C SUR LES SNAPSHOTS HISTORIQUES
===========================================

Objectif
--------
Tester le modèle V3_C_AGGRESSIVE sur des snapshots réellement postérieurs
à sa période d'entraînement, sans réentraîner le modèle.

Par défaut, le script :
1. cherche tous les fichiers:
       outputs/history/live_feature_snapshots*.csv
2. fusionne les différentes versions de snapshots ;
3. conserve uniquement les snapshots à la 87e minute ;
4. conserve uniquement les matchs strictement POSTÉRIEURS au cutoff
   d'entraînement V3-C ;
5. charge le schéma exact V3-C (38 features) ;
6. exige 38/38 features numériques et finies ;
7. déduplique à 1 observation par match ;
8. calcule P(goal_after_87) avec V3-C ;
9. récupère automatiquement les targets déjà fiables quand elles existent ;
10. laisse les autres targets à compléter manuellement ;
11. produit un fichier compact à labelliser et un fichier complet avec les
    38 features ;
12. calcule automatiquement ROC-AUC, PR-AUC, Brier, LogLoss, lift, recall,
    precision et les performances par seuil/tranche dès que des labels fiables
    sont disponibles.

IMPORTANT
---------
- Aucun réentraînement.
- Aucun remplissage silencieux des valeurs manquantes.
- Un match = une observation principale (87').
- Les matchs utilisés avant / pendant l'entraînement V3-C sont exclus par
  défaut afin de conserver un vrai test externe.
- Le modèle V3-C refit-all est supposé avoir été entraîné jusqu'au
  2026-09-13 16:30:00 UTC.
- Les probabilités V3-C sont analysées comme des scores ; elles ne sont pas
  supposées parfaitement calibrées.

Utilisation standard
--------------------
Depuis la racine du projet:

    python3 backtest_v3c_snapshots.py

Après avoir rempli manuellement les résultats manquants dans:

    outputs/v3c_external_test/v3c_predictions_to_label.csv

relancer simplement:

    python3 backtest_v3c_snapshots.py

Le script préservera les labels manuels déjà présents dans ce fichier.

Options utiles
--------------
    python3 backtest_v3c_snapshots.py --help

Exemple avec un autre cutoff:

    python3 backtest_v3c_snapshots.py \
        --cutoff "2026-09-13T16:30:00Z"

Exemple avec un fichier de labels manuel externe:

    python3 backtest_v3c_snapshots.py \
        --labels-csv "/chemin/mes_labels.csv"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit(
        "scikit-learn est requis pour ce script. "
        "Active ton .venv puis installe/valide scikit-learn."
    ) from exc


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("backtest_v3c_snapshots")


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_HISTORY_DIR = BASE_DIR / "outputs" / "history"

DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "model_v3_experiments"
    / "candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)

DEFAULT_SCHEMA_PATH = (
    BASE_DIR
    / "outputs"
    / "model_v3_experiments"
    / "schemas"
    / "V3_C_AGGRESSIVE_schema.json"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "v3c_external_test"
)

DEFAULT_SNAPSHOT_GLOB = "live_feature_snapshots*.csv"

# Fin de la période utilisée pour construire le dataset de V3-C.
DEFAULT_TRAINING_CUTOFF_UTC = "2026-09-13T16:30:00Z"

DEFAULT_MINUTE = 87

EXPECTED_V3C_FEATURE_COUNT = 38

PRIORITY_MIN = 0.55
PRIORITY_MAX = 0.65

THRESHOLDS = [
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
]

BANDS: List[Tuple[float, Optional[float], str]] = [
    (0.00, 0.50, "<50%"),
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 0.80, "75-80%"),
    (0.80, None, "80%+"),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SnapshotLoadResult:
    data: pd.DataFrame
    files: List[Path]
    unreadable_files: List[Tuple[Path, str]]


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def parse_bool(value: Any) -> Optional[bool]:
    text = normalize_text(value).lower()

    if text in {"1", "true", "yes", "y", "oui"}:
        return True

    if text in {"0", "false", "no", "n", "non"}:
        return False

    return None


def parse_utc_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return parsed


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lecture stricte mais compatible avec les CSV historiques.

    Les fichiers snapshots vus dans le projet sont des CSV séparés par virgule.
    `low_memory=False` évite les inférences de type incohérentes.
    """
    return pd.read_csv(
        path,
        low_memory=False,
    )


# =============================================================================
# SCHEMA / MODEL
# =============================================================================

def load_v3c_schema(schema_path: Path) -> List[str]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schéma V3-C introuvable : {schema_path}"
        )

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        features = payload

    elif isinstance(payload, Mapping):
        features = (
            payload.get("features")
            or payload.get("features_used")
            or payload.get("core_features")
            or payload.get("feature_names")
            or []
        )

    else:
        raise ValueError(
            "Format JSON du schéma V3-C non reconnu."
        )

    features = [
        str(feature).strip()
        for feature in features
        if str(feature).strip()
    ]

    if len(features) != EXPECTED_V3C_FEATURE_COUNT:
        raise ValueError(
            "Le schéma V3-C doit contenir exactement "
            f"{EXPECTED_V3C_FEATURE_COUNT} features, reçu={len(features)}."
        )

    if len(set(features)) != len(features):
        duplicates = sorted(
            feature
            for feature in set(features)
            if features.count(feature) > 1
        )

        raise ValueError(
            "Features dupliquées dans le schéma V3-C : "
            + ", ".join(duplicates)
        )

    return features


def load_v3c_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle V3-C introuvable : {model_path}"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Le modèle chargé ne fournit pas predict_proba()."
        )

    return model


def positive_class_index(model: Any) -> int:
    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        named_steps = getattr(
            model,
            "named_steps",
            {},
        )

        if isinstance(named_steps, Mapping):
            classifier = named_steps.get("classifier")
            classes = getattr(
                classifier,
                "classes_",
                None,
            )

    if classes is None:
        raise RuntimeError(
            "Impossible de déterminer les classes du modèle."
        )

    classes = list(classes)

    if 1 not in classes:
        raise RuntimeError(
            f"Classe positive 1 absente du modèle. classes={classes}"
        )

    return classes.index(1)


# =============================================================================
# SNAPSHOT DISCOVERY / LOADING
# =============================================================================

def discover_snapshot_files(
    history_dir: Path,
    snapshot_glob: str,
) -> List[Path]:
    if not history_dir.exists():
        raise FileNotFoundError(
            f"Dossier history introuvable : {history_dir}"
        )

    files = sorted(
        path
        for path in history_dir.glob(snapshot_glob)
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            "Aucun fichier snapshot trouvé avec le motif "
            f"{snapshot_glob!r} dans {history_dir}"
        )

    return files


def load_all_snapshots(
    files: Sequence[Path],
) -> SnapshotLoadResult:
    frames: List[pd.DataFrame] = []
    unreadable: List[Tuple[Path, str]] = []

    for path in files:
        try:
            df = read_csv_robust(path)
        except Exception as exc:
            unreadable.append(
                (path, str(exc))
            )
            logger.warning(
                "Fichier ignoré car illisible | %s | %s",
                path.name,
                exc,
            )
            continue

        required_identity = {
            "match_id",
            "minute",
            "snapshot_timestamp_utc",
        }

        if not required_identity.issubset(df.columns):
            logger.warning(
                "Fichier ignoré : colonnes snapshot essentielles absentes | %s",
                path.name,
            )
            continue

        df = df.copy()
        df["_source_file"] = path.name
        df["_source_row"] = np.arange(
            len(df),
            dtype=int,
        )

        frames.append(df)

        logger.info(
            "Snapshot chargé | %-45s | lignes=%d | colonnes=%d",
            path.name,
            len(df),
            len(df.columns),
        )

    if not frames:
        raise RuntimeError(
            "Aucun fichier snapshot exploitable n'a pu être chargé."
        )

    # Union de colonnes: les différentes versions de fichiers peuvent avoir
    # des colonnes de target additionnelles.
    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return SnapshotLoadResult(
        data=combined,
        files=list(files),
        unreadable_files=unreadable,
    )


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

LABEL_METADATA_COLUMNS = [
    "goal_after_87",
    "target_status",
    "target_source",
    "target_confidence",
    "target_resolution_class",
    "target_resolution_note",
    "target_coherence_warning",
    "target_resolved_at_utc",
    "final_score_home",
    "final_score_away",
    "valid_goal_minutes_all",
    "goal_minutes_after_87",
    "score_delta_after_snapshot87",
]


def normalized_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None

    text = normalize_text(value).lower()

    if text in {"", "nan", "none", "null", "pending"}:
        return None

    if text in {"1", "1.0", "true", "yes", "oui", "positive"}:
        return 1

    if text in {"0", "0.0", "false", "no", "non", "negative"}:
        return 0

    number = safe_float(value)

    if number in {0.0, 1.0}:
        return int(number)

    return None


def label_row_quality(row: pd.Series) -> Tuple[int, int, int]:
    """
    Plus le score est élevé, plus la ligne est intéressante comme source
    de vérité terrain.
    """
    status = normalize_text(
        row.get("target_status")
    ).upper()

    confidence = normalize_text(
        row.get("target_confidence")
    ).upper()

    source = normalize_text(
        row.get("target_source")
    ).lower()

    score = 0

    if "LABELED_V3" in status:
        score += 50
    elif "LABELED" in status:
        score += 40

    if confidence == "HIGH":
        score += 20
    elif confidence == "MEDIUM":
        score += 10

    if "official_timeline" in source:
        score += 20
    elif "timeline" in source:
        score += 10
    elif "final_score" in source:
        score += 5

    has_final = (
        safe_int(row.get("final_score_home")) is not None
        and safe_int(row.get("final_score_away")) is not None
    )

    if has_final:
        score += 5

    has_goal_minutes = bool(
        normalize_text(
            row.get("goal_minutes_after_87")
        )
    )

    if has_goal_minutes:
        score += 5

    resolved_at = normalize_text(
        row.get("target_resolved_at_utc")
    )

    return (
        score,
        1 if resolved_at else 0,
        1 if has_final else 0,
    )


def build_label_lookup(
    all_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit une vérité terrain au niveau match.

    - Si plusieurs fichiers donnent le même label 0/1 -> on garde le meilleur
      jeu de métadonnées.
    - Si deux sources donnent des labels contradictoires -> label_conflict=True
      et la target est laissée vide.
    """
    records: List[Dict[str, Any]] = []

    for match_id, group in all_rows.groupby(
        "match_id",
        sort=False,
        dropna=False,
    ):
        candidates: List[Tuple[int, pd.Series]] = []

        for _, row in group.iterrows():
            label = normalized_binary_label(
                row.get("goal_after_87")
            )

            if label is None:
                continue

            score = label_row_quality(row)[0]
            candidates.append(
                (score, row)
            )

        labels = {
            normalized_binary_label(row.get("goal_after_87"))
            for _, row in candidates
        }
        labels.discard(None)

        conflict = len(labels) > 1

        record: Dict[str, Any] = {
            "match_id": str(match_id),
            "existing_goal_after_87": np.nan,
            "existing_target_status": "",
            "existing_target_source": "",
            "existing_target_confidence": "",
            "existing_target_resolution_class": "",
            "existing_target_resolution_note": "",
            "existing_target_coherence_warning": "",
            "existing_target_resolved_at_utc": "",
            "existing_final_score_home": np.nan,
            "existing_final_score_away": np.nan,
            "existing_valid_goal_minutes_all": "",
            "existing_goal_minutes_after_87": "",
            "existing_score_delta_after_snapshot87": np.nan,
            "label_conflict": conflict,
        }

        if candidates and not conflict:
            candidates.sort(
                key=lambda item: label_row_quality(item[1]),
                reverse=True,
            )

            best = candidates[0][1]

            record["existing_goal_after_87"] = (
                normalized_binary_label(
                    best.get("goal_after_87")
                )
            )

            for column in LABEL_METADATA_COLUMNS:
                if column == "goal_after_87":
                    continue

                destination = f"existing_{column}"

                if destination in record:
                    record[destination] = best.get(
                        column,
                        "",
                    )

        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# MANUAL LABEL PRESERVATION
# =============================================================================

MANUAL_LABEL_COLUMNS = [
    "goal_after_87",
    "final_score_home",
    "final_score_away",
    "goal_minutes_after_87",
    "manual_label_note",
]


def read_manual_labels(
    candidate_paths: Sequence[Path],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in candidate_paths:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception as exc:
            logger.warning(
                "Impossible de lire les labels manuels | %s | %s",
                path,
                exc,
            )
            continue

        if "match_id" not in df.columns:
            continue

        keep = ["match_id"] + [
            column
            for column in MANUAL_LABEL_COLUMNS
            if column in df.columns
        ]

        subset = df.loc[:, keep].copy()
        subset["match_id"] = (
            subset["match_id"]
            .astype(str)
            .str.strip()
        )

        subset["_manual_source"] = path.name

        frames.append(subset)

    if not frames:
        return pd.DataFrame(
            columns=[
                "match_id",
                *MANUAL_LABEL_COLUMNS,
                "_manual_source",
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # Garder la dernière valeur renseignée par match.
    combined["_has_manual_target"] = (
        combined["goal_after_87"]
        .map(normalized_binary_label)
        .notna()
        if "goal_after_87" in combined.columns
        else False
    )

    combined = combined.sort_values(
        by=["_has_manual_target"],
        ascending=True,
        kind="stable",
    )

    combined = combined.drop_duplicates(
        subset=["match_id"],
        keep="last",
    )

    return combined


def overlay_manual_labels(
    scored: pd.DataFrame,
    manual: pd.DataFrame,
) -> pd.DataFrame:
    result = scored.copy()

    if manual.empty:
        return result

    renamed = manual.rename(
        columns={
            column: f"manual_{column}"
            for column in MANUAL_LABEL_COLUMNS
            if column in manual.columns
        }
    )

    result = result.merge(
        renamed,
        on="match_id",
        how="left",
    )

    if "manual_goal_after_87" in result.columns:
        manual_labels = result[
            "manual_goal_after_87"
        ].map(normalized_binary_label)

        result.loc[
            manual_labels.notna(),
            "goal_after_87",
        ] = manual_labels[
            manual_labels.notna()
        ].astype(int)

        result.loc[
            manual_labels.notna(),
            "label_origin",
        ] = "MANUAL"

    for column in (
        "final_score_home",
        "final_score_away",
        "goal_minutes_after_87",
        "manual_label_note",
    ):
        manual_column = f"manual_{column}"

        if manual_column not in result.columns:
            continue

        mask = (
            result[manual_column]
            .notna()
            & (
                result[manual_column]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        if column not in result.columns:
            result[column] = np.nan

        result.loc[
            mask,
            column,
        ] = result.loc[
            mask,
            manual_column,
        ]

    return result


# =============================================================================
# FILTER / DEDUP / COVERAGE
# =============================================================================

def coerce_feature_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    missing_columns = [
        feature
        for feature in features
        if feature not in frame.columns
    ]

    if missing_columns:
        # Ajouter explicitement NaN. On ne remplit jamais par zéro.
        for feature in missing_columns:
            frame[feature] = np.nan

    numeric = frame.loc[:, features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return numeric


def v3c_coverage(
    numeric_features: pd.DataFrame,
) -> pd.Series:
    finite = np.isfinite(
        numeric_features.to_numpy(dtype=float)
    )

    return pd.Series(
        finite.sum(axis=1),
        index=numeric_features.index,
        dtype=int,
    )


def select_external_snapshot_rows(
    all_rows: pd.DataFrame,
    features: Sequence[str],
    cutoff: pd.Timestamp,
    minute: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working = all_rows.copy()

    working["match_id"] = (
        working["match_id"]
        .astype(str)
        .str.strip()
    )

    working["_parsed_minute"] = pd.to_numeric(
        working["minute"],
        errors="coerce",
    )

    working["_snapshot_dt"] = parse_utc_series(
        working["snapshot_timestamp_utc"]
    )

    # Initial exclusions for diagnostics.
    exclusion_reasons = pd.Series(
        "",
        index=working.index,
        dtype=object,
    )

    exclusion_reasons.loc[
        working["_snapshot_dt"].isna()
    ] = "invalid_snapshot_timestamp"

    exclusion_reasons.loc[
        working["_parsed_minute"] != minute
    ] = "not_target_minute"

    exclusion_reasons.loc[
        working["_snapshot_dt"].notna()
        & (working["_snapshot_dt"] <= cutoff)
    ] = "not_post_training"

    eligible = working.loc[
        (working["_parsed_minute"] == minute)
        & working["_snapshot_dt"].notna()
        & (working["_snapshot_dt"] > cutoff)
    ].copy()

    numeric = coerce_feature_matrix(
        eligible,
        features,
    )

    eligible["_v3c_coverage"] = v3c_coverage(
        numeric
    )

    eligible["_v3c_complete"] = (
        eligible["_v3c_coverage"]
        == len(features)
    )

    # Nombre de colonnes présentes (utile pour départager des fichiers copies).
    eligible["_non_null_total"] = (
        eligible.notna().sum(axis=1)
    )

    # Déduplication:
    # - même match à la même minute peut être présent dans plusieurs fichiers;
    # - priorité 38/38;
    # - puis plus de métadonnées disponibles;
    # - puis snapshot le plus récent.
    eligible = eligible.sort_values(
        by=[
            "match_id",
            "_v3c_complete",
            "_v3c_coverage",
            "_non_null_total",
            "_snapshot_dt",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    )

    deduped = eligible.drop_duplicates(
        subset=["match_id"],
        keep="first",
    ).copy()

    excluded_rows: List[pd.DataFrame] = []

    incomplete = deduped.loc[
        ~deduped["_v3c_complete"]
    ].copy()

    if not incomplete.empty:
        incomplete["exclusion_reason"] = (
            "v3c_incomplete_features"
        )
        excluded_rows.append(incomplete)

    final = deduped.loc[
        deduped["_v3c_complete"]
    ].copy()

    if excluded_rows:
        excluded = pd.concat(
            excluded_rows,
            ignore_index=True,
            sort=False,
        )
    else:
        excluded = pd.DataFrame()

    return final, excluded


# =============================================================================
# PREDICTION
# =============================================================================

def predict_v3c(
    model: Any,
    rows: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    X = rows.loc[:, features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.shape[1] != EXPECTED_V3C_FEATURE_COUNT:
        raise RuntimeError(
            f"Input V3-C invalide : {X.shape[1]}/38 features."
        )

    if list(X.columns) != list(features):
        raise RuntimeError(
            "Ordre des features différent du schéma V3-C."
        )

    matrix = X.to_numpy(
        dtype=float
    )

    if not np.isfinite(matrix).all():
        bad_columns = list(
            X.columns[
                ~np.isfinite(matrix).all(axis=0)
            ]
        )

        raise RuntimeError(
            "Valeurs manquantes/non finies encore présentes avant predict_proba : "
            + ", ".join(bad_columns)
        )

    probabilities = model.predict_proba(X)

    index = positive_class_index(model)

    positive = np.asarray(
        probabilities[:, index],
        dtype=float,
    )

    if not np.isfinite(positive).all():
        raise RuntimeError(
            "Le modèle a produit des probabilités non finies."
        )

    if ((positive < 0) | (positive > 1)).any():
        raise RuntimeError(
            "Le modèle a produit une probabilité hors [0,1]."
        )

    return positive


def probability_band(
    probability: float,
) -> str:
    for lower, upper, label in BANDS:
        if upper is None:
            if probability >= lower:
                return label
        elif lower <= probability < upper:
            return label

    return "UNKNOWN"


# =============================================================================
# METRICS
# =============================================================================

def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1.0 + (z * z / total)

    center = (
        p
        + z * z / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p * (1 - p) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def safe_metric(
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> float:
    try:
        value = function(
            *args,
            **kwargs,
        )
        return float(value)
    except Exception:
        return float("nan")


def calculate_global_metrics(
    labeled: pd.DataFrame,
) -> Dict[str, Any]:
    y = labeled["goal_after_87"].astype(int).to_numpy()
    p = labeled["v3c_probability"].astype(float).to_numpy()

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    return {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "base_rate": (
            positives / len(y)
            if len(y)
            else float("nan")
        ),
        "roc_auc": safe_metric(
            roc_auc_score,
            y,
            p,
        ),
        "pr_auc": safe_metric(
            average_precision_score,
            y,
            p,
        ),
        "brier": safe_metric(
            brier_score_loss,
            y,
            p,
        ),
        "log_loss": safe_metric(
            log_loss,
            y,
            np.column_stack(
                [1.0 - p, p]
            ),
            labels=[0, 1],
        ),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
        "min_probability": float(np.min(p)),
        "max_probability": float(np.max(p)),
    }


def threshold_metrics(
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    y = labeled["goal_after_87"].astype(int)
    p = labeled["v3c_probability"].astype(float)

    total_positives = int(y.sum())
    base_rate = (
        total_positives / len(labeled)
        if len(labeled)
        else float("nan")
    )

    rows: List[Dict[str, Any]] = []

    for threshold in THRESHOLDS:
        selected_mask = p >= threshold

        selected = int(selected_mask.sum())
        selected_positives = int(
            y[selected_mask].sum()
        ) if selected else 0

        selected_negatives = (
            selected - selected_positives
        )

        false_negatives = (
            total_positives
            - selected_positives
        )

        total_negatives = (
            len(labeled)
            - total_positives
        )

        true_negatives = (
            total_negatives
            - selected_negatives
        )

        precision = (
            selected_positives / selected
            if selected
            else float("nan")
        )

        recall = (
            selected_positives / total_positives
            if total_positives
            else float("nan")
        )

        lift = (
            precision / base_rate
            if (
                selected
                and base_rate
                and math.isfinite(base_rate)
                and base_rate > 0
            )
            else float("nan")
        )

        ci_low, ci_high = wilson_interval(
            selected_positives,
            selected,
        )

        rows.append({
            "threshold": threshold,
            "threshold_percent": threshold * 100.0,
            "selected": selected,
            "selected_rate": (
                selected / len(labeled)
                if len(labeled)
                else float("nan")
            ),
            "positives_selected": selected_positives,
            "negatives_selected": selected_negatives,
            "observed_rate_precision": precision,
            "wilson95_low": ci_low,
            "wilson95_high": ci_high,
            "recall": recall,
            "lift_vs_base": lift,
            "true_positives": selected_positives,
            "false_positives": selected_negatives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "mean_selected_probability": (
                float(p[selected_mask].mean())
                if selected
                else float("nan")
            ),
            "median_selected_probability": (
                float(p[selected_mask].median())
                if selected
                else float("nan")
            ),
            "base_rate": base_rate,
        })

    return pd.DataFrame(rows)


def band_metrics(
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    y = labeled["goal_after_87"].astype(int)
    p = labeled["v3c_probability"].astype(float)

    positives_total = int(y.sum())

    base_rate = (
        positives_total / len(labeled)
        if len(labeled)
        else float("nan")
    )

    rows: List[Dict[str, Any]] = []

    for lower, upper, label in BANDS:
        if upper is None:
            mask = p >= lower
        else:
            mask = (
                (p >= lower)
                & (p < upper)
            )

        n = int(mask.sum())
        wins = int(
            y[mask].sum()
        ) if n else 0

        observed_rate = (
            wins / n
            if n
            else float("nan")
        )

        lift = (
            observed_rate / base_rate
            if (
                n
                and base_rate
                and math.isfinite(base_rate)
                and base_rate > 0
            )
            else float("nan")
        )

        recall_share = (
            wins / positives_total
            if positives_total
            else float("nan")
        )

        ci_low, ci_high = wilson_interval(
            wins,
            n,
        )

        rows.append({
            "band": label,
            "lower": lower,
            "upper_exclusive": upper,
            "n": n,
            "positives": wins,
            "negatives": n - wins,
            "observed_rate": observed_rate,
            "wilson95_low": ci_low,
            "wilson95_high": ci_high,
            "recall_share": recall_share,
            "lift_vs_base": lift,
            "mean_probability": (
                float(p[mask].mean())
                if n
                else float("nan")
            ),
            "median_probability": (
                float(p[mask].median())
                if n
                else float("nan")
            ),
            "base_rate": base_rate,
        })

    # Ajouter explicitement la zone opérationnelle 55-65.
    mask = (
        (p >= PRIORITY_MIN)
        & (p < PRIORITY_MAX)
    )

    n = int(mask.sum())
    wins = int(
        y[mask].sum()
    ) if n else 0

    observed_rate = (
        wins / n
        if n
        else float("nan")
    )

    lift = (
        observed_rate / base_rate
        if (
            n
            and base_rate
            and math.isfinite(base_rate)
            and base_rate > 0
        )
        else float("nan")
    )

    recall_share = (
        wins / positives_total
        if positives_total
        else float("nan")
    )

    ci_low, ci_high = wilson_interval(
        wins,
        n,
    )

    rows.append({
        "band": "55-65% PRIORITY",
        "lower": PRIORITY_MIN,
        "upper_exclusive": PRIORITY_MAX,
        "n": n,
        "positives": wins,
        "negatives": n - wins,
        "observed_rate": observed_rate,
        "wilson95_low": ci_low,
        "wilson95_high": ci_high,
        "recall_share": recall_share,
        "lift_vs_base": lift,
        "mean_probability": (
            float(p[mask].mean())
            if n
            else float("nan")
        ),
        "median_probability": (
            float(p[mask].median())
            if n
            else float("nan")
        ),
        "base_rate": base_rate,
    })

    return pd.DataFrame(rows)


# =============================================================================
# OUTPUT PREPARATION
# =============================================================================

def attach_existing_labels(
    scored: pd.DataFrame,
    label_lookup: pd.DataFrame,
) -> pd.DataFrame:
    result = scored.merge(
        label_lookup,
        on="match_id",
        how="left",
    )

    result["goal_after_87"] = np.nan
    result["label_origin"] = ""

    no_conflict = ~result[
        "label_conflict"
    ].fillna(False)

    existing_label = result[
        "existing_goal_after_87"
    ].map(normalized_binary_label)

    trustworthy = (
        no_conflict
        & existing_label.notna()
    )

    result.loc[
        trustworthy,
        "goal_after_87",
    ] = existing_label[
        trustworthy
    ].astype(int)

    result.loc[
        trustworthy,
        "label_origin",
    ] = "EXISTING_SNAPSHOT_LABEL"

    # Compact aliases.
    result["target_status"] = result.get(
        "existing_target_status",
        "",
    )

    result["target_source"] = result.get(
        "existing_target_source",
        "",
    )

    result["target_confidence"] = result.get(
        "existing_target_confidence",
        "",
    )

    result["final_score_home"] = result.get(
        "existing_final_score_home",
        np.nan,
    )

    result["final_score_away"] = result.get(
        "existing_final_score_away",
        np.nan,
    )

    result["goal_minutes_after_87"] = result.get(
        "existing_goal_minutes_after_87",
        "",
    )

    result["manual_label_note"] = ""

    return result


def make_scored_frame(
    selected: pd.DataFrame,
    probabilities: np.ndarray,
    features: Sequence[str],
) -> pd.DataFrame:
    scored = selected.copy()

    scored["v3c_probability"] = probabilities

    scored["v3c_fair_odds"] = np.where(
        probabilities > 0,
        1.0 / probabilities,
        np.inf,
    )

    scored["v3c_band"] = [
        probability_band(float(value))
        for value in probabilities
    ]

    scored["is_priority_55_65"] = (
        (scored["v3c_probability"] >= PRIORITY_MIN)
        & (scored["v3c_probability"] < PRIORITY_MAX)
    )

    scored["v3c_feature_coverage"] = (
        EXPECTED_V3C_FEATURE_COUNT
    )

    scored["v3c_model"] = "V3_C_AGGRESSIVE"

    # Conserver une liste de sources agrégée au niveau match.
    if "_source_file" in scored.columns:
        scored["selected_source_file"] = scored[
            "_source_file"
        ]

    # Vérification finale.
    X = scored.loc[:, features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Une valeur non finie a survécu au filtre 38/38."
        )

    return scored


def output_column_order(
    frame: pd.DataFrame,
) -> List[str]:
    preferred = [
        "match_id",
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "snapshot_timestamp_utc",
        "minute",
        "score_home_snapshot",
        "score_away_snapshot",
        "v3c_probability",
        "v3c_fair_odds",
        "v3c_band",
        "is_priority_55_65",
        "v3c_feature_coverage",
        "v3c_model",
        "goal_after_87",
        "label_origin",
        "target_status",
        "target_source",
        "target_confidence",
        "final_score_home",
        "final_score_away",
        "goal_minutes_after_87",
        "manual_label_note",
        "label_conflict",
        "selected_source_file",
    ]

    return [
        column
        for column in preferred
        if column in frame.columns
    ]


def write_report(
    path: Path,
    *,
    snapshot_files: Sequence[Path],
    cutoff: pd.Timestamp,
    raw_rows: int,
    unique_matches_raw: int,
    minute87_post_cutoff_rows: int,
    selected_rows: int,
    excluded_incomplete: int,
    labeled_rows: int,
    manual_needed: int,
    global_metrics: Optional[Dict[str, Any]],
    bands: Optional[pd.DataFrame],
) -> None:
    lines: List[str] = []

    lines.append("# V3-C External Snapshot Backtest")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Snapshot files loaded: **{len(snapshot_files)}**")
    lines.append(f"- Raw snapshot rows: **{raw_rows}**")
    lines.append(f"- Raw unique matches: **{unique_matches_raw}**")
    lines.append(f"- Training cutoff (UTC): **{cutoff.isoformat()}**")
    lines.append("- Main evaluation minute: **87'**")
    lines.append(
        f"- Post-cutoff 87' rows before final 38/38 filter: "
        f"**{minute87_post_cutoff_rows}**"
    )
    lines.append(
        f"- Unique 87' matches with exact V3-C 38/38 features: "
        f"**{selected_rows}**"
    )
    lines.append(
        f"- Excluded after dedupe because V3-C features were incomplete: "
        f"**{excluded_incomplete}**"
    )
    lines.append(f"- Already labeled: **{labeled_rows}**")
    lines.append(f"- Still requiring manual label: **{manual_needed}**")
    lines.append("")
    lines.append("## Snapshot files")
    lines.append("")

    for file in snapshot_files:
        lines.append(f"- `{file.name}`")

    if global_metrics is None:
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.append(
            "No reliable labels are currently available for the selected "
            "external sample. Fill `goal_after_87` in "
            "`v3c_predictions_to_label.csv` and rerun the script."
        )
    else:
        lines.append("")
        lines.append("## Current labeled-subset metrics")
        lines.append("")
        lines.append(
            f"- N: **{global_metrics['n']}**"
        )
        lines.append(
            f"- Positives: **{global_metrics['positives']}**"
        )
        lines.append(
            f"- Base rate: **{global_metrics['base_rate']:.2%}**"
        )
        lines.append(
            f"- ROC-AUC: **{global_metrics['roc_auc']:.6f}**"
        )
        lines.append(
            f"- PR-AUC: **{global_metrics['pr_auc']:.6f}**"
        )
        lines.append(
            f"- Brier: **{global_metrics['brier']:.6f}**"
        )
        lines.append(
            f"- LogLoss: **{global_metrics['log_loss']:.6f}**"
        )

        if bands is not None and not bands.empty:
            priority = bands.loc[
                bands["band"]
                == "55-65% PRIORITY"
            ]

            if not priority.empty:
                row = priority.iloc[0]

                lines.append("")
                lines.append("## 55-65% priority zone")
                lines.append("")
                lines.append(
                    f"- N: **{int(row['n'])}**"
                )
                lines.append(
                    f"- Positives: **{int(row['positives'])}**"
                )

                observed = safe_float(
                    row["observed_rate"]
                )

                lift = safe_float(
                    row["lift_vs_base"]
                )

                recall_share = safe_float(
                    row["recall_share"]
                )

                if observed is not None:
                    lines.append(
                        f"- Observed rate: **{observed:.2%}**"
                    )

                if lift is not None:
                    lines.append(
                        f"- Lift vs base: **{lift:.3f}**"
                    )

                if recall_share is not None:
                    lines.append(
                        f"- Recall share: **{recall_share:.2%}**"
                    )

                low = safe_float(
                    row["wilson95_low"]
                )
                high = safe_float(
                    row["wilson95_high"]
                )

                if low is not None and high is not None:
                    lines.append(
                        f"- Wilson 95% CI: **{low:.2%} – {high:.2%}**"
                    )

    lines.append("")
    lines.append("## Methodological note")
    lines.append("")
    lines.append(
        "This script does not retrain V3-C. It scores only snapshots after "
        "the training cutoff and uses one 87' observation per match. This "
        "reduces leakage and prevents 85'/86'/87' snapshots of the same match "
        "from being counted as independent matches."
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest externe de V3_C_AGGRESSIVE sur les snapshots historiques."
        )
    )

    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help=(
            "Dossier contenant les snapshots "
            f"(défaut: {DEFAULT_HISTORY_DIR})"
        ),
    )

    parser.add_argument(
        "--snapshot-glob",
        default=DEFAULT_SNAPSHOT_GLOB,
        help=(
            "Motif de fichiers snapshot "
            f"(défaut: {DEFAULT_SNAPSHOT_GLOB})"
        ),
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin du modèle V3-C refit-all.",
    )

    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Chemin du schéma V3-C 38 features.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Dossier de sortie du backtest.",
    )

    parser.add_argument(
        "--cutoff",
        default=DEFAULT_TRAINING_CUTOFF_UTC,
        help=(
            "Date/heure UTC de fin de période d'entraînement. "
            "Seuls les snapshots strictement postérieurs sont retenus."
        ),
    )

    parser.add_argument(
        "--minute",
        type=int,
        default=DEFAULT_MINUTE,
        help="Minute principale à tester (défaut: 87).",
    )

    parser.add_argument(
        "--labels-csv",
        default="",
        help=(
            "Fichier externe de labels manuels à fusionner. "
            "Doit contenir au minimum match_id et goal_after_87."
        ),
    )

    parser.add_argument(
        "--no-preserve-output-labels",
        action="store_true",
        help=(
            "Ne pas relire automatiquement les labels déjà remplis dans "
            "v3c_predictions_to_label.csv."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    history_dir = resolve_path(
        args.history_dir
    )

    model_path = resolve_path(
        args.model
    )

    schema_path = resolve_path(
        args.schema
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_to_label_path = (
        output_dir
        / "v3c_predictions_to_label.csv"
    )

    full_scored_path = (
        output_dir
        / "v3c_scored_snapshots_full.csv"
    )

    labeled_subset_path = (
        output_dir
        / "v3c_labeled_subset.csv"
    )

    excluded_path = (
        output_dir
        / "v3c_excluded_incomplete.csv"
    )

    threshold_metrics_path = (
        output_dir
        / "v3c_threshold_metrics.csv"
    )

    band_metrics_path = (
        output_dir
        / "v3c_probability_bands.csv"
    )

    global_metrics_path = (
        output_dir
        / "v3c_global_metrics.json"
    )

    report_path = (
        output_dir
        / "v3c_external_test_report.md"
    )

    metadata_path = (
        output_dir
        / "v3c_external_test_metadata.json"
    )

    # -------------------------------------------------------------------------
    # Cutoff
    # -------------------------------------------------------------------------
    cutoff = pd.to_datetime(
        args.cutoff,
        utc=True,
        errors="raise",
    )

    logger.info("=" * 90)
    logger.info("BACKTEST V3-C SNAPSHOTS")
    logger.info("=" * 90)
    logger.info("History dir : %s", history_dir)
    logger.info("Model       : %s", model_path)
    logger.info("Schema      : %s", schema_path)
    logger.info("Cutoff UTC  : %s", cutoff.isoformat())
    logger.info("Minute      : %d'", args.minute)
    logger.info("Output dir  : %s", output_dir)

    # -------------------------------------------------------------------------
    # Model/schema
    # -------------------------------------------------------------------------
    features = load_v3c_schema(
        schema_path
    )

    model = load_v3c_model(
        model_path
    )

    logger.info(
        "Schéma V3-C chargé : %d features",
        len(features),
    )

    # -------------------------------------------------------------------------
    # Snapshot files
    # -------------------------------------------------------------------------
    files = discover_snapshot_files(
        history_dir,
        args.snapshot_glob,
    )

    logger.info(
        "Fichiers snapshots détectés : %d",
        len(files),
    )

    loaded = load_all_snapshots(
        files
    )

    all_rows = loaded.data

    raw_rows = len(all_rows)

    unique_matches_raw = (
        all_rows["match_id"]
        .astype(str)
        .nunique()
    )

    logger.info(
        "Snapshots fusionnés : %d lignes | %d matchs uniques",
        raw_rows,
        unique_matches_raw,
    )

    # -------------------------------------------------------------------------
    # Label lookup from ALL historical copies
    # -------------------------------------------------------------------------
    label_lookup = build_label_lookup(
        all_rows
    )

    label_conflicts = int(
        label_lookup[
            "label_conflict"
        ].fillna(False).sum()
    )

    if label_conflicts:
        logger.warning(
            "Conflits de labels détectés : %d match(s). "
            "Ils resteront à vérifier manuellement.",
            label_conflicts,
        )

    # -------------------------------------------------------------------------
    # Pre-filter count for report
    # -------------------------------------------------------------------------
    all_rows["_tmp_dt"] = parse_utc_series(
        all_rows["snapshot_timestamp_utc"]
    )

    all_rows["_tmp_minute"] = pd.to_numeric(
        all_rows["minute"],
        errors="coerce",
    )

    minute87_post_cutoff_rows = int(
        (
            (all_rows["_tmp_minute"] == args.minute)
            & all_rows["_tmp_dt"].notna()
            & (all_rows["_tmp_dt"] > cutoff)
        ).sum()
    )

    all_rows = all_rows.drop(
        columns=[
            "_tmp_dt",
            "_tmp_minute",
        ],
        errors="ignore",
    )

    # -------------------------------------------------------------------------
    # Select one strict 38/38 row per external match
    # -------------------------------------------------------------------------
    selected, excluded = select_external_snapshot_rows(
        all_rows,
        features,
        cutoff=cutoff,
        minute=args.minute,
    )

    if selected.empty:
        raise RuntimeError(
            "Aucun snapshot post-entraînement 87' avec 38/38 features V3-C."
        )

    logger.info(
        "Matchs externes exploitables V3-C : %d",
        len(selected),
    )

    logger.info(
        "Matchs 87' exclus après dédup car <38/38 : %d",
        len(excluded),
    )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------
    probabilities = predict_v3c(
        model,
        selected,
        features,
    )

    scored = make_scored_frame(
        selected,
        probabilities,
        features,
    )

    # -------------------------------------------------------------------------
    # Existing labels from snapshot variants
    # -------------------------------------------------------------------------
    scored = attach_existing_labels(
        scored,
        label_lookup,
    )

    # -------------------------------------------------------------------------
    # Preserve previous manual labels / merge external labels
    # -------------------------------------------------------------------------
    manual_sources: List[Path] = []

    if (
        not args.no_preserve_output_labels
        and predictions_to_label_path.exists()
    ):
        manual_sources.append(
            predictions_to_label_path
        )

    if args.labels_csv:
        manual_sources.append(
            resolve_path(args.labels_csv)
        )

    manual = read_manual_labels(
        manual_sources
    )

    scored = overlay_manual_labels(
        scored,
        manual,
    )

    # Normalize final target after overlays.
    scored["goal_after_87"] = scored[
        "goal_after_87"
    ].map(normalized_binary_label)

    scored["label_needed"] = (
        scored["goal_after_87"].isna()
    )

    scored["label_status_for_backtest"] = np.where(
        scored["goal_after_87"].notna(),
        "LABELED",
        "TO_LABEL",
    )

    # -------------------------------------------------------------------------
    # Outputs: compact file to label manually
    # -------------------------------------------------------------------------
    compact_columns = output_column_order(
        scored
    )

    extra_manual = [
        "label_needed",
        "label_status_for_backtest",
    ]

    for column in extra_manual:
        if column in scored.columns:
            compact_columns.append(column)

    compact = scored.loc[
        :,
        list(dict.fromkeys(compact_columns)),
    ].copy()

    # Sort priority zone first, then descending probability.
    compact = compact.sort_values(
        by=[
            "is_priority_55_65",
            "v3c_probability",
            "snapshot_timestamp_utc",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    )

    compact.to_csv(
        predictions_to_label_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Full reproducibility file with exact 38 features
    # -------------------------------------------------------------------------
    identity_columns = [
        column
        for column in [
            "match_id",
            "competition_id",
            "season_id",
            "home_team",
            "away_team",
            "snapshot_timestamp_utc",
            "minute",
            "score_home_snapshot",
            "score_away_snapshot",
            "v3c_probability",
            "v3c_fair_odds",
            "v3c_band",
            "is_priority_55_65",
            "goal_after_87",
            "label_origin",
            "label_needed",
            "label_conflict",
            "selected_source_file",
        ]
        if column in scored.columns
    ]

    full_columns = (
        identity_columns
        + [
            feature
            for feature in features
            if feature in scored.columns
        ]
    )

    scored.loc[
        :,
        list(dict.fromkeys(full_columns)),
    ].to_csv(
        full_scored_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Exclusions
    # -------------------------------------------------------------------------
    if not excluded.empty:
        keep_excluded = [
            column
            for column in [
                "match_id",
                "home_team",
                "away_team",
                "snapshot_timestamp_utc",
                "minute",
                "_v3c_coverage",
                "exclusion_reason",
                "_source_file",
            ]
            if column in excluded.columns
        ]

        excluded.loc[
            :,
            keep_excluded,
        ].to_csv(
            excluded_path,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "match_id",
                "exclusion_reason",
            ]
        ).to_csv(
            excluded_path,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Metrics on currently labeled subset
    # -------------------------------------------------------------------------
    labeled = scored.loc[
        scored["goal_after_87"].notna()
    ].copy()

    labeled["goal_after_87"] = (
        labeled["goal_after_87"]
        .astype(int)
    )

    labeled.to_csv(
        labeled_subset_path,
        index=False,
    )

    global_metrics: Optional[Dict[str, Any]]
    thresholds_df: Optional[pd.DataFrame]
    bands_df: Optional[pd.DataFrame]

    if (
        len(labeled) >= 2
        and labeled["goal_after_87"].nunique() == 2
    ):
        global_metrics = calculate_global_metrics(
            labeled
        )

        thresholds_df = threshold_metrics(
            labeled
        )

        bands_df = band_metrics(
            labeled
        )

        thresholds_df.to_csv(
            threshold_metrics_path,
            index=False,
        )

        bands_df.to_csv(
            band_metrics_path,
            index=False,
        )

        global_metrics_path.write_text(
            json.dumps(
                global_metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Metrics disponibles | N=%d | base=%.2f%% | AUC=%.4f | PR-AUC=%.4f",
            global_metrics["n"],
            global_metrics["base_rate"] * 100.0,
            global_metrics["roc_auc"],
            global_metrics["pr_auc"],
        )

    else:
        global_metrics = None
        thresholds_df = None
        bands_df = None

        pd.DataFrame().to_csv(
            threshold_metrics_path,
            index=False,
        )

        pd.DataFrame().to_csv(
            band_metrics_path,
            index=False,
        )

        global_metrics_path.write_text(
            json.dumps(
                {
                    "status": "insufficient_labeled_data",
                    "labeled_rows": int(len(labeled)),
                    "unique_targets": int(
                        labeled["goal_after_87"].nunique()
                    ) if not labeled.empty else 0,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -------------------------------------------------------------------------
    # Metadata / report
    # -------------------------------------------------------------------------
    labeled_rows = int(
        scored["goal_after_87"].notna().sum()
    )

    manual_needed = int(
        scored["goal_after_87"].isna().sum()
    )

    metadata = {
        "model": "V3_C_AGGRESSIVE",
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "feature_count": len(features),
        "features": features,
        "training_cutoff_utc": cutoff.isoformat(),
        "evaluation_minute": args.minute,
        "snapshot_glob": args.snapshot_glob,
        "snapshot_files": [
            str(path)
            for path in loaded.files
        ],
        "raw_snapshot_rows": raw_rows,
        "raw_unique_matches": int(
            unique_matches_raw
        ),
        "post_cutoff_target_minute_rows": (
            minute87_post_cutoff_rows
        ),
        "external_unique_matches_38_of_38": int(
            len(scored)
        ),
        "excluded_incomplete_after_dedup": int(
            len(excluded)
        ),
        "label_conflicts_detected": label_conflicts,
        "currently_labeled_matches": labeled_rows,
        "manual_labels_needed": manual_needed,
        "priority_zone": {
            "lower_inclusive": PRIORITY_MIN,
            "upper_exclusive": PRIORITY_MAX,
        },
        "manual_label_sources_loaded": [
            str(path)
            for path in manual_sources
            if path.exists()
        ],
        "outputs": {
            "predictions_to_label": str(
                predictions_to_label_path
            ),
            "full_scored": str(
                full_scored_path
            ),
            "labeled_subset": str(
                labeled_subset_path
            ),
            "threshold_metrics": str(
                threshold_metrics_path
            ),
            "probability_bands": str(
                band_metrics_path
            ),
            "global_metrics": str(
                global_metrics_path
            ),
            "report": str(
                report_path
            ),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    write_report(
        report_path,
        snapshot_files=loaded.files,
        cutoff=cutoff,
        raw_rows=raw_rows,
        unique_matches_raw=int(
            unique_matches_raw
        ),
        minute87_post_cutoff_rows=(
            minute87_post_cutoff_rows
        ),
        selected_rows=len(scored),
        excluded_incomplete=len(excluded),
        labeled_rows=labeled_rows,
        manual_needed=manual_needed,
        global_metrics=global_metrics,
        bands=bands_df,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    priority_count = int(
        scored["is_priority_55_65"].sum()
    )

    priority_labeled = scored.loc[
        scored["is_priority_55_65"]
        & scored["goal_after_87"].notna()
    ]

    logger.info("=" * 90)
    logger.info("RÉSUMÉ")
    logger.info("=" * 90)
    logger.info(
        "Matchs externes V3-C 38/38 : %d",
        len(scored),
    )
    logger.info(
        "Zone 55-65%% : %d match(s)",
        priority_count,
    )
    logger.info(
        "Déjà labellisés : %d",
        labeled_rows,
    )
    logger.info(
        "À labelliser manuellement : %d",
        manual_needed,
    )

    if not priority_labeled.empty:
        priority_wins = int(
            priority_labeled[
                "goal_after_87"
            ].astype(int).sum()
        )

        priority_rate = (
            priority_wins
            / len(priority_labeled)
        )

        logger.info(
            "Zone 55-65%% déjà labellisée : %d/%d = %.2f%%",
            priority_wins,
            len(priority_labeled),
            priority_rate * 100.0,
        )

    logger.info(
        "Fichier à compléter : %s",
        predictions_to_label_path,
    )

    logger.info(
        "Rapport : %s",
        report_path,
    )

    logger.info(
        "Aucun modèle n'a été réentraîné ni modifié."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        logger.warning("Interruption utilisateur.")
        raise SystemExit(130)
    except Exception as exc:
        logger.exception(
            "ERREUR FATALE | %s",
            exc,
        )
        raise SystemExit(1)
