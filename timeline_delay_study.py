#!/usr/bin/env python3
# =============================================================================
# TIMELINE DELAY STUDY - GOAL AFTER 87'
# =============================================================================
#
# IMPORTANT
# ---------
# Ce script est COMPLETEMENT SEPARE du scanner live.
# Il NE MODIFIE PAS live_scanner_engine.py.
#
# But de l'étude :
#   1) reconstruire des probabilités HORS-ECHANTILLON sur le dataset historique
#      avec EXACTEMENT la même pipeline / les mêmes hyperparamètres que le
#      modèle de production (via sklearn.clone du .joblib de production) ;
#   2) sélectionner les matchs avec P >= seuil (34 % par défaut) ;
#   3) pour les matchs historiques qui ont réellement eu goal_after_87 = 1,
#      récupérer la timeline finale via TheStatsAPI ;
#   4) trouver le PREMIER but après 87' ;
#   5) mesurer le délai approximatif entre 87' et ce premier but ;
#   6) produire un rapport complet par :
#        - tranche de probabilité,
#        - score à 87',
#        - état nul / écart 1 / écart 2+,
#        - minute du premier but,
#        - délai après 87',
#        - qualité / cohérence de la timeline.
#
# Pourquoi "approximatif" ?
# -------------------------
# Les timelines football sont généralement à la minute, pas à la seconde.
# Un but à 88' signifie donc environ 1 minute après 87', pas "exactement
# 60 secondes". Le script conserve aussi minute + extra_time bruts.
#
# IMPORTANT METHODOLOGIQUE
# ------------------------
# Le dataset ML historique contient UNE ligne par match autour de l'état 86/87.
# Il ne contient pas les trois snapshots 85', 86', 87' du scanner actuel.
# Cette étude répond donc à :
#
#   "Pour les situations historiques P >= 34 % autour de 87',
#    quand arrive le premier but après 87' ?"
#
# Elle NE répond PAS encore exactement à :
#
#   "Combien de temps après une alerte réelle reçue à 85' ou 86' ?"
#
# Cette deuxième question sera étudiée séparément avec
# outputs/history/live_feature_snapshots.csv.
#
# ECONOMIE API
# ------------
# Par défaut, le script ne télécharge les timelines que pour les lignes :
#     P >= seuil ET goal_after_87 == 1
#
# Les matchs négatifs sont déjà connus via la cible du dataset et n'ont pas
# besoin d'un appel timeline pour mesurer le délai du but.
#
# Toutes les timelines téléchargées sont mises en cache :
#     outputs/timeline_study/cache/timelines/<match_id>.json
#
# En cas d'interruption, relancer la même commande reprend l'étude sans refaire
# les appels déjà mis en cache.
#
# UTILISATION
# -----------
#
# Depuis la racine ML_Goal_After_87 :
#
#   python3 timeline_delay_study.py
#
# Seuil différent :
#
#   python3 timeline_delay_study.py --threshold 0.34
#
# Seulement calculer les probabilités et préparer la liste (aucun appel API) :
#
#   python3 timeline_delay_study.py --prepare-only
#
# Reprendre après interruption :
#
#   python3 timeline_delay_study.py
#
# Vérifier aussi toutes les timelines négatives (beaucoup plus de requêtes) :
#
#   python3 timeline_delay_study.py --verify-all
#
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()

OUTPUT_DIR = BASE_DIR / "outputs" / "timeline_study"
CACHE_DIR = OUTPUT_DIR / "cache"
TIMELINE_CACHE_DIR = CACHE_DIR / "timelines"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_CANDIDATES = [
    BASE_DIR / "ML_Dataset_PreMatch_Live87.csv",
    BASE_DIR / "data" / "ML_Dataset_PreMatch_Live87.csv",
]

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "production"
    / "ET_500_d15_73_27.joblib"
)

MODEL_FEATURES_PATH = (
    BASE_DIR
    / "outputs"
    / "production"
    / "ET_500_d15_73_27_features.json"
)

OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
SELECTED_PATH = OUTPUT_DIR / "selected_p34_plus.csv"
DELAYS_PATH = OUTPUT_DIR / "timeline_goal_delays.csv"
REPORT_PATH = OUTPUT_DIR / "report_p34_timeline.md"
EXCEL_PATH = OUTPUT_DIR / "report_p34_timeline.xlsx"
RUN_METADATA_PATH = OUTPUT_DIR / "run_metadata.json"


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("timeline_delay_study")


# =============================================================================
# API CONFIG
# =============================================================================

STATS_API_KEY = os.getenv("STATS_API_KEY", "").strip()
if STATS_API_KEY.lower().startswith("bearer "):
    STATS_API_KEY = STATS_API_KEY[7:].strip()

STATS_API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api",
).rstrip("/")

STATS_API_TIMEOUT = int(
    os.getenv("STATS_API_TIMEOUT", "20")
)

# Petite pause entre nouveaux appels pour éviter de marteler l'API.
TIMELINE_REQUEST_DELAY_SECONDS = float(
    os.getenv("TIMELINE_STUDY_REQUEST_DELAY_SECONDS", "0.20")
)

MAX_RETRIES = int(
    os.getenv("TIMELINE_STUDY_MAX_RETRIES", "4")
)


# =============================================================================
# EXPECTED DATASET SCHEMA
# =============================================================================

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"

EXPECTED_MODEL_FEATURE_COUNT = 96


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        pass
    return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        if math.isfinite(result):
            return int(result)
    except Exception:
        pass
    return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_token(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def score_label(home: Any, away: Any) -> str:
    h = safe_int(home)
    a = safe_int(away)
    if h is None or a is None:
        return "UNKNOWN"
    return f"{h}-{a}"


def score_state(home: Any, away: Any) -> str:
    h = safe_int(home)
    a = safe_int(away)

    if h is None or a is None:
        return "UNKNOWN"

    diff = abs(h - a)

    if diff == 0:
        return "DRAW"
    if diff == 1:
        return "ONE_GOAL_GAP"
    return "TWO_PLUS_GOAL_GAP"


def probability_band(p: float) -> str:
    if p >= 0.50:
        return "50%+"
    if p >= 0.45:
        return "45-50%"
    if p >= 0.40:
        return "40-45%"
    if p >= 0.37:
        return "37-40%"
    if p >= 0.35:
        return "35-37%"
    if p >= 0.34:
        return "34-35%"
    return "<34%"


PROB_BAND_ORDER = [
    "34-35%",
    "35-37%",
    "37-40%",
    "40-45%",
    "45-50%",
    "50%+",
]


def delay_bucket(delay_minutes: Optional[float]) -> str:
    if delay_minutes is None or not math.isfinite(delay_minutes):
        return "UNKNOWN"

    # Les données sont à la minute : <=1 signifie typiquement but à 88'
    if delay_minutes <= 1:
        return "<=1 min"
    if delay_minutes <= 2:
        return "1-2 min"
    if delay_minutes <= 3:
        return "2-3 min"
    if delay_minutes <= 5:
        return "3-5 min"
    return ">5 min"


DELAY_BUCKET_ORDER = [
    "<=1 min",
    "1-2 min",
    "2-3 min",
    "3-5 min",
    ">5 min",
    "UNKNOWN",
]


# =============================================================================
# DATASET / MODEL
# =============================================================================

def resolve_dataset_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {path}")
        return path

    for path in DEFAULT_DATASET_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "ML_Dataset_PreMatch_Live87.csv introuvable. "
        "Place-le à la racine du projet ou utilise --dataset CHEMIN."
    )


def load_feature_order(df: pd.DataFrame) -> List[str]:
    """
    Priorité :
      1) JSON exact du modèle de production si disponible ;
      2) sinon dataset : toutes les colonnes sauf match_id + target.
    """

    if MODEL_FEATURES_PATH.exists():
        payload = json.loads(
            MODEL_FEATURES_PATH.read_text(encoding="utf-8")
        )

        if isinstance(payload, list):
            features = [str(x) for x in payload]

        elif isinstance(payload, Mapping):
            candidate = first_not_none(
                payload.get("features"),
                payload.get("feature_names"),
                payload.get("expected_features"),
            )
            if isinstance(candidate, list):
                features = [str(x) for x in candidate]
            else:
                features = []
        else:
            features = []

        if len(features) == EXPECTED_MODEL_FEATURE_COUNT:
            missing = [x for x in features if x not in df.columns]
            if missing:
                raise RuntimeError(
                    "Le JSON features contient des colonnes absentes "
                    f"du dataset : {missing[:10]}"
                )
            return features

    features = [
        col
        for col in df.columns
        if col not in {ID_COLUMN, TARGET_COLUMN}
    ]

    if len(features) != EXPECTED_MODEL_FEATURE_COUNT:
        raise RuntimeError(
            f"Le dataset doit fournir 96 features. Trouvé : {len(features)}"
        )

    return features


def validate_dataset(df: pd.DataFrame, features: Sequence[str]) -> None:
    required = {ID_COLUMN, TARGET_COLUMN, *features}
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Colonnes requises manquantes : {missing}"
        )

    if len(features) != EXPECTED_MODEL_FEATURE_COUNT:
        raise RuntimeError(
            f"Features : {len(features)}/96"
        )

    target_values = set(
        pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1}):
        raise RuntimeError(
            f"Cible invalide : valeurs={sorted(target_values)}"
        )

    if df[ID_COLUMN].isna().any():
        raise RuntimeError("match_id contient des valeurs manquantes.")

    duplicated = int(df[ID_COLUMN].duplicated().sum())
    if duplicated:
        raise RuntimeError(
            f"Dataset contient {duplicated} match_id dupliqués."
        )


def load_production_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle de production introuvable : {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Le modèle chargé ne possède pas predict_proba()."
        )

    return model


def compute_oof_probabilities(
    df: pd.DataFrame,
    features: Sequence[str],
    model: Any,
    folds: int,
    force: bool,
) -> pd.DataFrame:
    """
    OOF = chaque ligne est prédite par un modèle qui n'a pas été entraîné
    sur cette ligne.

    On clone EXACTEMENT la pipeline de production afin de conserver :
      - imputer,
      - ExtraTrees,
      - tous les hyperparamètres du fichier joblib.
    """

    if OOF_PATH.exists() and not force:
        cached = pd.read_csv(OOF_PATH)

        expected = {
            ID_COLUMN,
            TARGET_COLUMN,
            "oof_probability",
            "score_87_home",
            "score_87_away",
        }

        if (
            len(cached) == len(df)
            and expected.issubset(cached.columns)
            and cached[ID_COLUMN].astype(str).tolist()
            == df[ID_COLUMN].astype(str).tolist()
        ):
            logger.info(
                "OOF cache HIT : %s",
                OOF_PATH,
            )
            return cached

        logger.warning(
            "OOF cache incompatible -> recalcul."
        )

    X = df[list(features)].copy()
    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    ).astype(int).to_numpy()

    probabilities = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    logger.info(
        "Calcul OOF : %d folds | %d lignes | %d features",
        folds,
        len(df),
        len(features),
    )

    for fold_index, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        logger.info(
            "OOF fold %d/%d | train=%d | test=%d",
            fold_index,
            folds,
            len(train_idx),
            len(test_idx),
        )

        fold_model = clone(model)

        fold_model.fit(
            X.iloc[train_idx],
            y[train_idx],
        )

        fold_prob = fold_model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

        probabilities[test_idx] = fold_prob

    if np.isnan(probabilities).any():
        raise RuntimeError(
            "OOF incomplet : certaines probabilités sont NaN."
        )

    out = pd.DataFrame({
        ID_COLUMN: df[ID_COLUMN].astype(str),
        TARGET_COLUMN: y,
        "oof_probability": probabilities,
        "score_87_home": df["score_87_home"],
        "score_87_away": df["score_87_away"],
        "goal_diff_87": df["goal_diff_87"],
        "is_draw_87": df["is_draw_87"],
        "last_goal_minute": df["last_goal_minute"],
    })

    out["score_87"] = [
        score_label(h, a)
        for h, a in zip(
            out["score_87_home"],
            out["score_87_away"],
        )
    ]

    out["score_state"] = [
        score_state(h, a)
        for h, a in zip(
            out["score_87_home"],
            out["score_87_away"],
        )
    ]

    out["probability_band"] = out[
        "oof_probability"
    ].map(probability_band)

    out.to_csv(
        OOF_PATH,
        index=False,
    )

    logger.info(
        "OOF sauvegardé : %s",
        OOF_PATH,
    )

    return out


# =============================================================================
# API CLIENT / CACHE
# =============================================================================

class TimelineClient:

    def __init__(self) -> None:
        if not STATS_API_KEY:
            raise RuntimeError(
                "STATS_API_KEY absent du .env."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {STATS_API_KEY}",
            "Accept": "application/json",
        })

        self.api_calls = 0
        self.cache_hits = 0

    def cache_path(self, match_id: str) -> Path:
        safe_id = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(match_id),
        )
        return TIMELINE_CACHE_DIR / f"{safe_id}.json"

    def fetch_timeline(
        self,
        match_id: str,
        force: bool = False,
    ) -> Tuple[Optional[Any], str]:

        cache_path = self.cache_path(match_id)

        if cache_path.exists() and not force:
            try:
                payload = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                self.cache_hits += 1
                return payload, "CACHE"
            except Exception:
                logger.warning(
                    "Cache timeline illisible : %s",
                    cache_path,
                )

        url = (
            f"{STATS_API_BASE_URL}"
            f"/football/matches/{match_id}/timeline"
        )

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.api_calls += 1

                response = self.session.get(
                    url,
                    timeout=STATS_API_TIMEOUT,
                )

                if response.status_code == 429:
                    retry_after = safe_int(
                        response.headers.get("Retry-After")
                    )
                    wait_seconds = retry_after or min(
                        30,
                        3 * attempt,
                    )

                    logger.warning(
                        "429 | match=%s | attente=%ss",
                        match_id,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                # 404/409 : inutile de les retenter plusieurs fois.
                if response.status_code in {404, 409}:
                    return None, f"HTTP_{response.status_code}"

                response.raise_for_status()
                payload = response.json()

                cache_path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                time.sleep(
                    TIMELINE_REQUEST_DELAY_SECONDS
                )

                return payload, "API"

            except requests.RequestException as exc:
                last_error = exc

                if attempt < MAX_RETRIES:
                    time.sleep(min(10, attempt * 2))
                    continue

            except Exception as exc:
                last_error = exc
                break

        logger.warning(
            "Timeline échec | match=%s | %s",
            match_id,
            last_error,
        )
        return None, "ERROR"


# =============================================================================
# TIMELINE PARSING
# =============================================================================

def timeline_events(payload: Any) -> List[Mapping[str, Any]]:
    """
    Tolère plusieurs structures API.
    """

    if payload is None:
        return []

    if isinstance(payload, list):
        return [
            x for x in payload
            if isinstance(x, Mapping)
        ]

    if not isinstance(payload, Mapping):
        return []

    direct_keys = (
        "events",
        "timeline",
        "items",
        "results",
    )

    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [
                x for x in value
                if isinstance(x, Mapping)
            ]

    data = payload.get("data")

    if isinstance(data, list):
        return [
            x for x in data
            if isinstance(x, Mapping)
        ]

    if isinstance(data, Mapping):
        for key in direct_keys:
            value = data.get(key)
            if isinstance(value, list):
                return [
                    x for x in value
                    if isinstance(x, Mapping)
                ]

    return []


def event_type(event: Mapping[str, Any]) -> str:
    return normalize_token(
        first_not_none(
            event.get("type"),
            event.get("event_type"),
            event.get("event"),
            event.get("name"),
            event.get("kind"),
            "",
        )
    )


def is_goal_event(event: Mapping[str, Any]) -> bool:
    t = event_type(event)

    if not t:
        return False

    # Evénements explicitement NON comptés.
    excluded_tokens = (
        "disallowed",
        "cancelled",
        "canceled",
        "goal_kick",
        "goalkeeper",
        "no_goal",
        "var_no_goal",
    )

    if any(token in t for token in excluded_tokens):
        return False

    # Cas usuels.
    if t in {
        "goal",
        "own_goal",
        "penalty_goal",
        "goal_scored",
    }:
        return True

    # Fallback : contient goal mais pas dans les exclusions.
    return "goal" in t


def extract_event_minute(
    event: Mapping[str, Any],
) -> Tuple[Optional[int], int]:
    """
    Retourne :
      base_minute, extra_time

    Ex:
      minute=90, extra_time=3 -> 90+3
    """

    minute = safe_int(
        first_not_none(
            event.get("minute"),
            event.get("elapsed"),
            event.get("time"),
            event.get("match_minute"),
        )
    )

    extra = safe_int(
        first_not_none(
            event.get("extra_time"),
            event.get("stoppage_time"),
            event.get("added_time"),
            event.get("additional_time"),
            0,
        )
    )

    if extra is None:
        extra = 0

    return minute, max(0, extra)


def effective_minute(
    minute: Optional[int],
    extra: int,
) -> Optional[float]:
    if minute is None:
        return None
    return float(minute + max(0, extra))


def display_minute(
    minute: Optional[int],
    extra: int,
) -> str:
    if minute is None:
        return "UNKNOWN"
    if extra > 0:
        return f"{minute}+{extra}"
    return str(minute)


def extract_team_name(event: Mapping[str, Any]) -> Optional[str]:
    team = event.get("team")

    if isinstance(team, Mapping):
        value = first_not_none(
            team.get("name"),
            team.get("team_name"),
        )
        return str(value) if value is not None else None

    if isinstance(team, (list, tuple)) and len(team) >= 2:
        return str(team[1])

    if team is not None:
        return str(team)

    return None


def first_goal_after_87(
    payload: Any,
) -> Dict[str, Any]:

    events = timeline_events(payload)

    goals: List[Dict[str, Any]] = []

    for event in events:
        if not is_goal_event(event):
            continue

        minute, extra = extract_event_minute(event)
        eff = effective_minute(minute, extra)

        if eff is None:
            continue

        # Convention de l'étude :
        # "après 87" => temps effectif strictement > 87.
        if eff <= 87:
            continue

        goals.append({
            "base_minute": minute,
            "extra_time": extra,
            "effective_minute": eff,
            "minute_label": display_minute(
                minute,
                extra,
            ),
            "type": event_type(event),
            "team": extract_team_name(event),
            "raw": event,
        })

    goals.sort(
        key=lambda x: (
            x["effective_minute"],
            x["base_minute"] or 0,
            x["extra_time"],
        )
    )

    if not goals:
        return {
            "found": False,
            "event_count": len(events),
            "goals_after_87_count": 0,
            "first_goal": None,
        }

    return {
        "found": True,
        "event_count": len(events),
        "goals_after_87_count": len(goals),
        "first_goal": goals[0],
    }


# =============================================================================
# TIMELINE STUDY
# =============================================================================

def select_threshold_matches(
    oof: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    selected = oof[
        oof["oof_probability"] >= threshold
    ].copy()

    selected = selected.sort_values(
        ["oof_probability", ID_COLUMN],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected["threshold"] = threshold

    selected.to_csv(
        SELECTED_PATH,
        index=False,
    )

    logger.info(
        "P >= %.2f%% : %d matchs | positives=%d | taux réel=%.2f%%",
        threshold * 100,
        len(selected),
        int(selected[TARGET_COLUMN].sum()),
        (
            selected[TARGET_COLUMN].mean() * 100
            if len(selected)
            else 0.0
        ),
    )

    return selected


def run_timeline_collection(
    selected: pd.DataFrame,
    verify_all: bool,
    force_timelines: bool,
) -> pd.DataFrame:

    client = TimelineClient()

    if verify_all:
        to_fetch = selected.copy()
    else:
        to_fetch = selected[
            selected[TARGET_COLUMN] == 1
        ].copy()

    logger.info(
        "Timelines à étudier : %d | mode=%s",
        len(to_fetch),
        (
            "TOUS LES P>=SEUIL"
            if verify_all
            else "POSITIFS P>=SEUIL UNIQUEMENT"
        ),
    )

    rows: List[Dict[str, Any]] = []

    # Garder les négatifs dans le fichier de sortie même sans API.
    if not verify_all:
        negatives = selected[
            selected[TARGET_COLUMN] == 0
        ]

        for _, row in negatives.iterrows():
            rows.append({
                ID_COLUMN: str(row[ID_COLUMN]),
                TARGET_COLUMN: 0,
                "oof_probability": float(
                    row["oof_probability"]
                ),
                "probability_band": row[
                    "probability_band"
                ],
                "score_87": row["score_87"],
                "score_87_home": row[
                    "score_87_home"
                ],
                "score_87_away": row[
                    "score_87_away"
                ],
                "score_state": row["score_state"],
                "timeline_status":
                    "NOT_REQUESTED_NEGATIVE",
                "timeline_source": "",
                "timeline_event_count": np.nan,
                "timeline_goal_found": False,
                "target_timeline_consistent": True,
                "first_goal_minute_base": np.nan,
                "first_goal_extra_time": np.nan,
                "first_goal_minute_label": "",
                "first_goal_effective_minute": np.nan,
                "delay_from_87_minutes": np.nan,
                "delay_bucket": "NO_GOAL",
                "goal_team": "",
                "goal_event_type": "",
            })

    total = len(to_fetch)

    for index, (_, row) in enumerate(
        to_fetch.iterrows(),
        start=1,
    ):
        match_id = str(row[ID_COLUMN])

        logger.info(
            "[%d/%d] timeline %s | P=%.2f%% | target=%d",
            index,
            total,
            match_id,
            float(row["oof_probability"]) * 100,
            int(row[TARGET_COLUMN]),
        )

        payload, source = client.fetch_timeline(
            match_id,
            force=force_timelines,
        )

        if payload is None:
            rows.append({
                ID_COLUMN: match_id,
                TARGET_COLUMN: int(row[TARGET_COLUMN]),
                "oof_probability": float(
                    row["oof_probability"]
                ),
                "probability_band": row[
                    "probability_band"
                ],
                "score_87": row["score_87"],
                "score_87_home": row[
                    "score_87_home"
                ],
                "score_87_away": row[
                    "score_87_away"
                ],
                "score_state": row["score_state"],
                "timeline_status": source,
                "timeline_source": source,
                "timeline_event_count": np.nan,
                "timeline_goal_found": False,
                "target_timeline_consistent": False,
                "first_goal_minute_base": np.nan,
                "first_goal_extra_time": np.nan,
                "first_goal_minute_label": "",
                "first_goal_effective_minute": np.nan,
                "delay_from_87_minutes": np.nan,
                "delay_bucket": "UNKNOWN",
                "goal_team": "",
                "goal_event_type": "",
            })
            continue

        parsed = first_goal_after_87(
            payload
        )

        first_goal = parsed["first_goal"]
        goal_found = bool(parsed["found"])

        target = int(row[TARGET_COLUMN])
        consistent = (
            (target == 1 and goal_found)
            or (target == 0 and not goal_found)
        )

        if first_goal is not None:
            first_eff = float(
                first_goal["effective_minute"]
            )
            delay = first_eff - 87.0
        else:
            first_eff = np.nan
            delay = np.nan

        rows.append({
            ID_COLUMN: match_id,
            TARGET_COLUMN: target,
            "oof_probability": float(
                row["oof_probability"]
            ),
            "probability_band": row[
                "probability_band"
            ],
            "score_87": row["score_87"],
            "score_87_home": row[
                "score_87_home"
            ],
            "score_87_away": row[
                "score_87_away"
            ],
            "score_state": row["score_state"],
            "timeline_status": "OK",
            "timeline_source": source,
            "timeline_event_count": int(
                parsed["event_count"]
            ),
            "timeline_goal_found": goal_found,
            "target_timeline_consistent": consistent,
            "first_goal_minute_base": (
                first_goal["base_minute"]
                if first_goal else np.nan
            ),
            "first_goal_extra_time": (
                first_goal["extra_time"]
                if first_goal else np.nan
            ),
            "first_goal_minute_label": (
                first_goal["minute_label"]
                if first_goal else ""
            ),
            "first_goal_effective_minute": first_eff,
            "delay_from_87_minutes": delay,
            "delay_bucket": (
                delay_bucket(delay)
                if goal_found
                else "NO_GOAL"
            ),
            "goal_team": (
                first_goal["team"]
                if first_goal else ""
            ),
            "goal_event_type": (
                first_goal["type"]
                if first_goal else ""
            ),
        })

        # Sauvegarde incrémentale à chaque match :
        # si le script est interrompu, les timelines sont déjà en cache.
        if index % 20 == 0:
            pd.DataFrame(rows).to_csv(
                DELAYS_PATH,
                index=False,
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["oof_probability", ID_COLUMN],
            ascending=[False, True],
        ).reset_index(drop=True)

    result.to_csv(
        DELAYS_PATH,
        index=False,
    )

    logger.info(
        "Timeline collection terminée | API=%d | cache=%d",
        client.api_calls,
        client.cache_hits,
    )

    return result


# =============================================================================
# REPORTING
# =============================================================================

def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Aucune donnée._"

    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_float_dtype(
            display[col]
        ):
            display[col] = display[col].map(
                lambda x: (
                    ""
                    if pd.isna(x)
                    else f"{x:.2f}"
                )
            )

    headers = list(display.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for _, row in display.iterrows():
        values = [
            str(row[col]).replace("|", "\\|")
            for col in headers
        ]
        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def make_summary_tables(
    selected: pd.DataFrame,
    delays: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    tables: Dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------
    # Probability bands - ALL selected
    # ---------------------------------------------------------------

    prob_rows = []

    for band in PROB_BAND_ORDER:
        grp = selected[
            selected["probability_band"] == band
        ]

        if grp.empty:
            continue

        prob_rows.append({
            "Tranche P": band,
            "Matchs": len(grp),
            "Buts après 87": int(
                grp[TARGET_COLUMN].sum()
            ),
            "Taux réel %": (
                grp[TARGET_COLUMN].mean() * 100
            ),
            "P moyenne %": (
                grp["oof_probability"].mean() * 100
            ),
        })

    tables["probability"] = pd.DataFrame(
        prob_rows
    )

    # ---------------------------------------------------------------
    # Score exact
    # ---------------------------------------------------------------

    score_table = (
        selected
        .groupby("score_87", dropna=False)
        .agg(
            Matchs=(ID_COLUMN, "count"),
            Buts_apres_87=(
                TARGET_COLUMN,
                "sum",
            ),
            Taux_reel=(
                TARGET_COLUMN,
                "mean",
            ),
            P_moyenne=(
                "oof_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    score_table[
        "Taux réel %"
    ] = score_table["Taux_reel"] * 100

    score_table[
        "P moyenne %"
    ] = score_table["P_moyenne"] * 100

    score_table = (
        score_table[
            [
                "score_87",
                "Matchs",
                "Buts_apres_87",
                "Taux réel %",
                "P moyenne %",
            ]
        ]
        .sort_values(
            ["Matchs", "Taux réel %"],
            ascending=[False, False],
        )
    )

    tables["score"] = score_table

    # ---------------------------------------------------------------
    # Score state
    # ---------------------------------------------------------------

    state_table = (
        selected
        .groupby("score_state", dropna=False)
        .agg(
            Matchs=(ID_COLUMN, "count"),
            Buts_apres_87=(
                TARGET_COLUMN,
                "sum",
            ),
            Taux_reel=(
                TARGET_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    state_table[
        "Taux réel %"
    ] = state_table["Taux_reel"] * 100

    tables["score_state"] = state_table[
        [
            "score_state",
            "Matchs",
            "Buts_apres_87",
            "Taux réel %",
        ]
    ].sort_values(
        "Taux réel %",
        ascending=False,
    )

    # ---------------------------------------------------------------
    # Delay buckets - only timeline-confirmed positive goals
    # ---------------------------------------------------------------

    confirmed = delays[
        (delays[TARGET_COLUMN] == 1)
        & (delays["timeline_goal_found"] == True)
    ].copy()

    delay_rows = []

    for bucket in DELAY_BUCKET_ORDER:
        grp = confirmed[
            confirmed["delay_bucket"] == bucket
        ]

        if grp.empty:
            continue

        delay_rows.append({
            "Délai depuis 87": bucket,
            "Buts": len(grp),
            "% des buts confirmés": (
                len(grp)
                / len(confirmed)
                * 100
                if len(confirmed)
                else 0
            ),
        })

    tables["delay"] = pd.DataFrame(
        delay_rows
    )

    # ---------------------------------------------------------------
    # First goal minute
    # ---------------------------------------------------------------

    if not confirmed.empty:
        minute_table = (
            confirmed
            .groupby(
                "first_goal_minute_label",
                dropna=False,
            )
            .size()
            .reset_index(name="Buts")
        )

        minute_table[
            "% des buts confirmés"
        ] = (
            minute_table["Buts"]
            / minute_table["Buts"].sum()
            * 100
        )

        minute_table["sort_eff"] = (
            confirmed
            .groupby(
                "first_goal_minute_label"
            )["first_goal_effective_minute"]
            .min()
            .reindex(
                minute_table[
                    "first_goal_minute_label"
                ]
            )
            .to_numpy()
        )

        minute_table = (
            minute_table
            .sort_values("sort_eff")
            .drop(columns="sort_eff")
        )
    else:
        minute_table = pd.DataFrame()

    tables["minute"] = minute_table

    # ---------------------------------------------------------------
    # Delay by probability band
    # ---------------------------------------------------------------

    if not confirmed.empty:
        cross = pd.crosstab(
            confirmed["probability_band"],
            confirmed["delay_bucket"],
        )

        cross = cross.reindex(
            index=[
                x for x in PROB_BAND_ORDER
                if x in cross.index
            ],
            columns=[
                x for x in DELAY_BUCKET_ORDER
                if x in cross.columns
            ],
            fill_value=0,
        )

        cross = cross.reset_index()
    else:
        cross = pd.DataFrame()

    tables["delay_by_probability"] = cross

    # ---------------------------------------------------------------
    # Delay by score state
    # ---------------------------------------------------------------

    if not confirmed.empty:
        score_delay = pd.crosstab(
            confirmed["score_state"],
            confirmed["delay_bucket"],
        ).reset_index()
    else:
        score_delay = pd.DataFrame()

    tables["delay_by_score_state"] = (
        score_delay
    )

    # ---------------------------------------------------------------
    # Timeline consistency
    # ---------------------------------------------------------------

    if not delays.empty:
        consistency = (
            delays
            .groupby(
                [
                    "timeline_status",
                    "target_timeline_consistent",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="Matchs")
        )
    else:
        consistency = pd.DataFrame()

    tables["consistency"] = consistency

    return tables


def write_report(
    dataset_path: Path,
    threshold: float,
    full_df: pd.DataFrame,
    selected: pd.DataFrame,
    delays: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    folds: int,
) -> None:

    base_rate = (
        full_df[TARGET_COLUMN].mean() * 100
    )

    selected_rate = (
        selected[TARGET_COLUMN].mean() * 100
        if len(selected)
        else 0.0
    )

    positives = int(
        selected[TARGET_COLUMN].sum()
    )

    confirmed = delays[
        (delays[TARGET_COLUMN] == 1)
        & (delays["timeline_goal_found"] == True)
    ].copy()

    missing_positive_timeline = delays[
        (delays[TARGET_COLUMN] == 1)
        & (delays["timeline_goal_found"] != True)
    ]

    if not confirmed.empty:
        mean_delay = confirmed[
            "delay_from_87_minutes"
        ].mean()

        median_delay = confirmed[
            "delay_from_87_minutes"
        ].median()

        within_1 = (
            confirmed[
                "delay_from_87_minutes"
            ].le(1).mean() * 100
        )

        within_2 = (
            confirmed[
                "delay_from_87_minutes"
            ].le(2).mean() * 100
        )

        within_3 = (
            confirmed[
                "delay_from_87_minutes"
            ].le(3).mean() * 100
        )

        within_5 = (
            confirmed[
                "delay_from_87_minutes"
            ].le(5).mean() * 100
        )
    else:
        mean_delay = median_delay = np.nan
        within_1 = within_2 = within_3 = within_5 = np.nan

    report = f"""# Étude Timeline — Goal After 87

## 1. Objet

Cette étude analyse les matchs historiques dont la probabilité **hors-échantillon**
est supérieure ou égale à **{threshold:.2%}**.

Le scanner live de production n'a pas été modifié.

## 2. Méthode

- Dataset : `{dataset_path.name}`
- Lignes : **{len(full_df):,}**
- Features ML : **96**
- Validation : **{folds}-fold Stratified Cross-Validation**
- Modèle de chaque fold : clone exact de la pipeline `.joblib` de production
- Cible : `goal_after_87`
- Seuil étudié : **P >= {threshold:.2%}**
- Timeline : `/football/matches/{{match_id}}/timeline`
- Délai étudié : première minute de but strictement **> 87'**
- Résolution temporelle : **minute**, pas seconde

## 3. Résultat global du seuil

- Taux naturel du dataset : **{base_rate:.2f}%**
- Matchs P >= {threshold:.2%} : **{len(selected):,}**
- Positifs réels : **{positives:,}**
- Taux réel dans la sélection : **{selected_rate:.2f}%**
- Lift vs base : **{(selected_rate/base_rate if base_rate else np.nan):.2f}x**

## 4. Tranches de probabilité

{markdown_table(tables["probability"])}

## 5. Scores à 87'

{markdown_table(tables["score"])}

## 6. État du score

{markdown_table(tables["score_state"])}

## 7. Délai du premier but après 87'

Timelines positives confirmées : **{len(confirmed):,}/{positives:,}**

Timelines positives non confirmées / indisponibles : **{len(missing_positive_timeline):,}**

- Délai moyen : **{mean_delay:.2f} min**
- Délai médian : **{median_delay:.2f} min**
- But <= 1 min après 87 : **{within_1:.2f}%**
- But <= 2 min après 87 : **{within_2:.2f}%**
- But <= 3 min après 87 : **{within_3:.2f}%**
- But <= 5 min après 87 : **{within_5:.2f}%**

{markdown_table(tables["delay"])}

## 8. Minute exacte du premier but

{markdown_table(tables["minute"])}

## 9. Délai selon la tranche de probabilité

{markdown_table(tables["delay_by_probability"])}

## 10. Délai selon l'état du score

{markdown_table(tables["delay_by_score_state"])}

## 11. Contrôle de cohérence dataset / timeline

{markdown_table(tables["consistency"])}

## 12. Interprétation opérationnelle

Cette étude historique utilise un snapshot proche de 87'. Elle permet d'estimer
à quelle vitesse un but arrive **après 87'** lorsqu'un match se trouve dans une
zone de probabilité élevée.

Elle ne doit pas être interprétée comme le délai exact après une notification
réelle à 85' ou 86', car ces snapshots n'existent pas dans le dataset historique
original.

La seconde étude devra utiliser `outputs/history/live_feature_snapshots.csv`
pour mesurer séparément :

- alerte à 85' -> prochain but ;
- alerte à 86' -> prochain but ;
- alerte à 87' -> prochain but ;
- effet d'un but entre deux alertes ;
- signal reconfirmé / signal perdu.

## 13. Protection du scanner

Aucun code de `live_scanner_engine.py` n'est utilisé ou modifié par cette étude.
Toute expérimentation future devra se faire dans un second scanner/lab scanner
avant toute décision de promotion en production.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # Excel complet
    try:
        with pd.ExcelWriter(
            EXCEL_PATH,
            engine="openpyxl",
        ) as writer:

            selected.to_excel(
                writer,
                sheet_name="P34_Plus",
                index=False,
            )

            delays.to_excel(
                writer,
                sheet_name="Timeline_Delays",
                index=False,
            )

            for name, table in tables.items():
                sheet_name = {
                    "probability": "Prob_Bands",
                    "score": "Scores",
                    "score_state": "Score_State",
                    "delay": "Delay_Buckets",
                    "minute": "Goal_Minutes",
                    "delay_by_probability": "Delay_By_Prob",
                    "delay_by_score_state": "Delay_By_Score",
                    "consistency": "Consistency",
                }.get(name, name[:31])

                table.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False,
                )

        logger.info(
            "Rapport Excel : %s",
            EXCEL_PATH,
        )

    except Exception as exc:
        logger.warning(
            "Excel non généré : %s",
            exc,
        )

    logger.info(
        "Rapport Markdown : %s",
        REPORT_PATH,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Etude historique du délai du premier but "
            "après 87' pour P >= seuil."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Chemin du CSV. Défaut : "
            "ML_Dataset_PreMatch_Live87.csv"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
        help="Seuil de probabilité. Défaut : 0.34",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Nombre de folds OOF. Défaut : 5",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Calcule OOF + sélection mais ne fait "
            "aucun appel timeline."
        ),
    )

    parser.add_argument(
        "--verify-all",
        action="store_true",
        help=(
            "Télécharge aussi les timelines des négatifs "
            "P>=seuil. Coûte plus de requêtes."
        ),
    )

    parser.add_argument(
        "--force-oof",
        action="store_true",
        help="Recalcule les probabilités OOF.",
    )

    parser.add_argument(
        "--force-timelines",
        action="store_true",
        help="Ignore le cache des timelines.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    args = parse_args()

    if not (0 < args.threshold < 1):
        raise SystemExit(
            "--threshold doit être entre 0 et 1."
        )

    if args.folds < 2:
        raise SystemExit(
            "--folds doit être >= 2."
        )

    print()
    print("=" * 74)
    print(" TIMELINE DELAY STUDY - GOAL AFTER 87'")
    print("=" * 74)
    print(" Scanner live : NON MODIFIÉ")
    print(f" Seuil étude : {args.threshold:.2%}")
    print("=" * 74)
    print()

    dataset_path = resolve_dataset_path(
        args.dataset
    )

    logger.info(
        "Dataset : %s",
        dataset_path,
    )

    df = pd.read_csv(
        dataset_path
    )

    features = load_feature_order(
        df
    )

    validate_dataset(
        df,
        features,
    )

    logger.info(
        "Dataset OK | rows=%d | features=%d | target_rate=%.2f%%",
        len(df),
        len(features),
        df[TARGET_COLUMN].mean() * 100,
    )

    model = load_production_model()

    logger.info(
        "Production model chargé : %s",
        MODEL_PATH,
    )

    # Sauvegarde des paramètres exacts utilisés par le .joblib.
    try:
        model_params = model.get_params(
            deep=True
        )
    except Exception:
        model_params = {}

    metadata = {
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "dataset":
            str(dataset_path),
        "model":
            str(MODEL_PATH),
        "threshold":
            args.threshold,
        "folds":
            args.folds,
        "rows":
            len(df),
        "features":
            len(features),
        "target_rate":
            float(df[TARGET_COLUMN].mean()),
        "model_params":
            {
                str(k): str(v)
                for k, v in model_params.items()
            },
        "scanner_modified":
            False,
    }

    RUN_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    oof = compute_oof_probabilities(
        df=df,
        features=features,
        model=model,
        folds=args.folds,
        force=args.force_oof,
    )

    selected = select_threshold_matches(
        oof=oof,
        threshold=args.threshold,
    )

    if selected.empty:
        logger.warning(
            "Aucun match P>=seuil."
        )
        return 0

    if args.prepare_only:
        print()
        print("PREPARE-ONLY terminé.")
        print(f"OOF      : {OOF_PATH}")
        print(f"Sélection: {SELECTED_PATH}")
        print("Aucun appel timeline effectué.")
        return 0

    delays = run_timeline_collection(
        selected=selected,
        verify_all=args.verify_all,
        force_timelines=args.force_timelines,
    )

    tables = make_summary_tables(
        selected=selected,
        delays=delays,
    )

    write_report(
        dataset_path=dataset_path,
        threshold=args.threshold,
        full_df=df,
        selected=selected,
        delays=delays,
        tables=tables,
        folds=args.folds,
    )

    print()
    print("=" * 74)
    print(" ETUDE TERMINEE")
    print("=" * 74)
    print(f"OOF         : {OOF_PATH}")
    print(f"P>=seuil    : {SELECTED_PATH}")
    print(f"Délais      : {DELAYS_PATH}")
    print(f"Rapport MD  : {REPORT_PATH}")
    print(f"Rapport XLSX: {EXCEL_PATH}")
    print(f"Cache       : {TIMELINE_CACHE_DIR}")
    print()
    print("live_scanner_engine.py : NON MODIFIÉ")
    print("=" * 74)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
