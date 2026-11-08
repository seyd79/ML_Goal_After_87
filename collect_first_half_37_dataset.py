#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_first_half_37_dataset.py

DATASET HISTORIQUE — GOAL AFTER 37' BEFORE HALF-TIME
=====================================================

OBJECTIF
--------
Construire un NOUVEAU dataset, totalement séparé du scanner V3-C, pour entraîner
un modèle qui estime à la 37e minute :

    P(au moins un but APRES 37' et AVANT la fin de la 1re mi-temps)

Le script réutilise les informations PRE-MATCH déjà reconstruites proprement dans
le dataset historique Goal After 87, puis reconstruit l'état LIVE à 37' à partir
DES SOURCES HORODATEES :

    - /football/matches/{id}/timeline
    - /football/matches/{id}/shotmap
    - /football/matches/{id}/odds
    - /football/matches/{id}          (identité équipes / contrôle)

IMPORTANT — AUCUNE FUITE TEMPORELLE
-----------------------------------
- Les features LIVE utilisent uniquement des événements dont minute <= 37.
- Les statistiques finales de match ne sont jamais utilisées comme features.
- Les événements > 37' ne servent qu'à construire le TARGET.
- Les cotes utilisent uniquement le champ `opening`; `last_seen` est ignoré.
- Missing != 0 : aucune absence n'est transformée silencieusement en zéro.
- Le target négatif n'est validé que si la timeline prouve la fin de la 1re MT.

DEFINITION DU TARGET
--------------------
Snapshot : état connu jusqu'à 37' INCLUS.

    goal_after_37_before_ht = 1
        si la timeline est complète, contient la fin de la 1re mi-temps,
        et au moins un événement goal de period=first_half a minute > 37.

    goal_after_37_before_ht = 0
        si la timeline est complète, contient la fin de la 1re mi-temps,
        et aucun but first_half n'arrive après 37'.

IMPORTANT : les classes 1 ET 0 exigent donc la même preuve de complétude.
Cela évite un biais de sélection où les positifs pourraient entrer avec une
source partielle alors que les négatifs exigeraient une timeline complète.

Un but à 37' appartient donc aux FEATURES (score à 37'), pas au target.
Un but à 38', 41' ou 45+N appartient au target positif.

DONNEES PRODUITES
-----------------
outputs/first_half_37/
├── cache/
│   └── api/
├── datasets/
│   └── First_Half_37_RAW.csv
└── audit/
    ├── first_half_37_match_audit.csv
    ├── first_half_37_feature_coverage.csv
    ├── first_half_37_rejections.csv
    └── first_half_37_report.md

DEPENDANCES DU PROJET
---------------------
Ce fichier doit être placé à la RACINE de ML_Goal_After_87, à côté de :

    audit_historical_live87_reconstructability.py
    rebuild_clean_prematch_v2.py
    .env

Il ne modifie aucun de ces fichiers et ne touche pas au scanner V3-C.

TEST RECOMMANDE
---------------
    caffeinate -dimsu python3 collect_first_half_37_dataset.py \
        --limit 100 \
        --resume

Pour un test plus court :
    python3 collect_first_half_37_dataset.py --limit 20

Après validation du rapport, on pourra augmenter progressivement :
    caffeinate -dimsu python3 collect_first_half_37_dataset.py \
        --limit 1000 \
        --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"
PREMATCH_MODULE_PATH = BASE_DIR / "rebuild_clean_prematch_v2.py"

DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "first_half_37"
DEFAULT_DATASET_DIR = DEFAULT_OUTPUT_DIR / "datasets"
DEFAULT_AUDIT_DIR = DEFAULT_OUTPUT_DIR / "audit"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache" / "api"

# Cache historique existant : lecture SEULEMENT comme fallback afin d'éviter
# de refaire des appels déjà payés lors de la construction du dataset V2.
LEGACY_CACHE_DIR = BASE_DIR / "outputs" / "historical_dataset_v2" / "api_cache_v2"

DEFAULT_SOURCE_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_RAW_v2.csv",

    BASE_DIR
    / "outputs"
    / "rebuilt_dataset_v2"
    / "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv",
]

RAW_DATASET_NAME = "First_Half_37_RAW.csv"
MATCH_AUDIT_NAME = "first_half_37_match_audit.csv"
FEATURE_COVERAGE_NAME = "first_half_37_feature_coverage.csv"
REJECTIONS_NAME = "first_half_37_rejections.csv"
REPORT_NAME = "first_half_37_report.md"

SCHEMA_VERSION = "first_half_37_historical_v1"
SNAPSHOT_MINUTE = 37
TARGET_COLUMN = "goal_after_37_before_ht"

# 120 requêtes/min = 2 req/s. 0.75 s laisse une marge et réduit le risque de
# 429, surtout si un autre processus utilise la même clé en parallèle.
DEFAULT_MIN_API_INTERVAL = 0.75


# =============================================================================
# GENERIC HELPERS
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

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = clean_text(value)
    if not text:
        return None

    text = text.replace(",", ".")

    try:
        number = float(text)
    except Exception:
        return None

    return number if math.isfinite(number) else None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    return int(number) if number is not None else None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def normalize_endpoint(endpoint: str) -> str:
    endpoint = clean_text(endpoint)

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    while "//" in endpoint:
        endpoint = endpoint.replace("//", "/")

    return endpoint


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dépendance introuvable : {path}\n"
            "Place ce fichier à la racine du projet avec les modules existants."
        )

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module : {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(raw: Optional[str], default: Path) -> Path:
    if not raw:
        return default

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def source_path_from_candidates(raw: Optional[str]) -> Path:
    if raw:
        path = resolve_path(raw, Path(raw))
        if not path.exists():
            raise FileNotFoundError(f"Source CSV introuvable : {path}")
        return path

    for path in DEFAULT_SOURCE_CANDIDATES:
        if path.exists():
            return path

    lines = "\n".join(f"  - {p}" for p in DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(
        "Aucun dataset historique source trouvé. Candidats :\n" + lines
    )


def payload_data(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload.get("data", payload)
    return payload


def payload_list(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]

    if not isinstance(payload, Mapping):
        return []

    data = payload.get("data", payload)

    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, Mapping)]

    if isinstance(data, Mapping):
        for key in ("events", "shots", "items", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]

    return []


def payload_explicitly_contains_event_list(payload: Any) -> bool:
    """True si la réponse expose explicitement une liste d'événements, même vide."""
    if isinstance(payload, list):
        return True

    if not isinstance(payload, Mapping):
        return False

    data = payload.get("data", payload)

    if isinstance(data, list):
        return True

    if isinstance(data, Mapping):
        return any(
            key in data and isinstance(data.get(key), list)
            for key in ("events", "shots", "items", "results", "data")
        )

    return False


def nonempty_count(row: Mapping[str, Any], columns: Sequence[str]) -> int:
    total = 0

    for column in columns:
        value = row.get(column)

        if safe_float(value) is not None:
            total += 1
        elif isinstance(value, str) and value.strip():
            total += 1

    return total


# =============================================================================
# API CACHE
# =============================================================================

class CachedHistoricalClient:
    """
    Cache disque persistant autour de audit.StatsAPIClient.

    - écrit les nouvelles réponses dans outputs/first_half_37/cache/api
    - lit aussi l'ancien cache historical_dataset_v2/api_cache_v2 si disponible
    - ne cache jamais une erreur HTTP comme une donnée valide
    """

    def __init__(
        self,
        base_client: Any,
        cache_dir: Path,
        legacy_cache_dir: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.base_client = base_client
        self.cache_dir = cache_dir
        self.legacy_cache_dir = legacy_cache_dir
        self.enabled = bool(enabled)

        self.cache_hits = 0
        self.legacy_cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def request_count(self) -> int:
        return int(getattr(self.base_client, "request_count", 0))

    @property
    def retry_429_count(self) -> int:
        return int(getattr(self.base_client, "retry_429_count", 0))

    def _cache_key(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> str:
        raw = stable_json({
            "endpoint": normalize_endpoint(endpoint),
            "params": dict(params or {}),
        })

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(
        self,
        root: Path,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> Path:
        key = self._cache_key(endpoint, params)
        return root / key[:2] / f"{key}.json"

    @staticmethod
    def _read_record(path: Path) -> Tuple[bool, Any]:
        if not path.exists():
            return False, None

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False, None

        if not isinstance(record, Mapping):
            return False, None

        if "payload" not in record:
            return False, None

        return True, record.get("payload")

    def _write_record(
        self,
        path: Path,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
        payload: Any,
    ) -> None:
        if payload is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "cached_at_utc": now_utc_iso(),
            "endpoint": normalize_endpoint(endpoint),
            "params": dict(params or {}),
            "payload": payload,
        }

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
        self.cache_writes += 1

    def request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[int], str]:
        endpoint = normalize_endpoint(endpoint)

        if self.enabled:
            primary = self._cache_path(self.cache_dir, endpoint, params)
            ok, payload = self._read_record(primary)

            if ok:
                self.cache_hits += 1
                return payload, 200, ""

            if self.legacy_cache_dir and self.legacy_cache_dir.exists():
                legacy = self._cache_path(
                    self.legacy_cache_dir,
                    endpoint,
                    params,
                )
                ok, payload = self._read_record(legacy)

                if ok:
                    self.legacy_cache_hits += 1
                    # Copie vers le cache du nouveau projet pour le rendre autonome.
                    self._write_record(primary, endpoint, params, payload)
                    return payload, 200, ""

        self.cache_misses += 1

        payload, status, error = self.base_client.request(
            endpoint,
            params=params,
        )

        if self.enabled and status == 200 and payload is not None:
            primary = self._cache_path(self.cache_dir, endpoint, params)
            self._write_record(primary, endpoint, params, payload)

        return payload, status, error


# =============================================================================
# MATCH / EVENT HELPERS
# =============================================================================

def extract_match_id(row: Mapping[str, Any]) -> str:
    for key in ("match_id", "id", "event_id"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def normalize_period(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    aliases = {
        "first_half": "first_half",
        "firsthalf": "first_half",
        "1st_half": "first_half",
        "1h": "first_half",
        "first": "first_half",
        "period_1": "first_half",
        "1": "first_half",
        "second_half": "second_half",
        "secondhalf": "second_half",
        "2nd_half": "second_half",
        "2h": "second_half",
        "second": "second_half",
        "period_2": "second_half",
        "2": "second_half",
    }

    return aliases.get(text, text)


def event_period(event: Mapping[str, Any]) -> str:
    for key in ("period", "match_period", "half"):
        value = event.get(key)

        if isinstance(value, Mapping):
            value = value.get("name") or value.get("value") or value.get("id")

        normalized = normalize_period(value)
        if normalized:
            return normalized

    return ""


def event_base_minute(event: Mapping[str, Any], audit: Any) -> Optional[int]:
    # TheStatsAPI timeline typically exposes integer `minute` plus `extra_time`.
    direct = safe_int(event.get("minute"))
    if direct is not None:
        return direct

    return audit.extract_event_minute(event)


def event_extra_time(event: Mapping[str, Any]) -> int:
    for key in ("extra_time", "minute_extra", "added_time"):
        value = safe_int(event.get(key))
        if value is not None:
            return max(0, value)
    return 0


def display_event_minute(event: Mapping[str, Any], audit: Any) -> str:
    minute = event_base_minute(event, audit)
    if minute is None:
        return ""

    extra = event_extra_time(event)
    return f"{minute}+{extra}" if extra > 0 else str(minute)


def is_first_half_event(event: Mapping[str, Any], audit: Any) -> bool:
    period = event_period(event)

    if period:
        return period == "first_half"

    # Fallback conservative if old payload has no period field.
    minute = event_base_minute(event, audit)
    return minute is not None and minute <= 45


def timeline_coverage(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""

    data = payload.get("data", payload)

    if isinstance(data, Mapping):
        value = data.get("coverage")

        if isinstance(value, Mapping):
            for key in ("timeline", "level", "status", "coverage"):
                if value.get(key) is not None:
                    return clean_text(value.get(key)).lower()

        if value is not None:
            return clean_text(value).lower()

    meta = payload.get("meta")
    if isinstance(meta, Mapping) and meta.get("coverage") is not None:
        return clean_text(meta.get("coverage")).lower()

    return ""


def timeline_is_full(payload: Any) -> bool:
    value = timeline_coverage(payload)
    return value in {"full", "complete", "completed", "1", "true"}


def first_half_period_end_seen(
    events: Sequence[Mapping[str, Any]],
    audit: Any,
) -> bool:
    for event in events:
        if not is_first_half_event(event, audit):
            continue

        event_type = audit.extract_event_type(event)
        if "period_end" in event_type or event_type in {"half_time", "halftime"}:
            return True

    return False


def first_half_period_metadata_rate(
    events: Sequence[Mapping[str, Any]],
) -> float:
    if not events:
        return 0.0

    present = sum(1 for event in events if event_period(event))
    return present / len(events)


def get_detail_team_context(
    detail_payload: Any,
    source_row: Mapping[str, Any],
    audit: Any,
) -> Dict[str, Any]:
    match = audit.unwrap_match(detail_payload)

    home_id = audit.extract_team_id(match, "home")
    away_id = audit.extract_team_id(match, "away")
    home_name = audit.extract_team_name(match, "home")
    away_name = audit.extract_team_name(match, "away")

    if not home_name:
        home_name = clean_text(source_row.get("home_team"))

    if not away_name:
        away_name = clean_text(source_row.get("away_team"))

    return {
        "match": match,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team": home_name,
        "away_team": away_name,
    }


# =============================================================================
# TARGET
# =============================================================================

def resolve_target(
    timeline_events: Sequence[Mapping[str, Any]],
    timeline_payload: Any,
    audit: Any,
) -> Dict[str, Any]:
    future_first_half_goals: List[Mapping[str, Any]] = []
    all_first_half_goals: List[Mapping[str, Any]] = []

    for event in timeline_events:
        if not is_first_half_event(event, audit):
            continue

        flags = audit.event_flags(event)
        if not flags.get("goal"):
            continue

        minute = event_base_minute(event, audit)
        if minute is None:
            continue

        all_first_half_goals.append(event)

        if minute > SNAPSHOT_MINUTE:
            future_first_half_goals.append(event)

    period_end = first_half_period_end_seen(timeline_events, audit)
    full = timeline_is_full(timeline_payload)

    goal_minutes = [
        display_event_minute(event, audit)
        for event in future_first_half_goals
    ]

    goal_minutes = [value for value in goal_minutes if value]

    # Règle symétrique de qualité : aucune classe n'entre dans le dataset ML
    # sans preuve que la timeline couvre entièrement la première mi-temps.
    if full and period_end:
        if future_first_half_goals:
            return {
                TARGET_COLUMN: 1,
                "target_status": "RESOLVED",
                "target_resolution_class": "POSITIVE_FULL_FIRST_HALF_TIMELINE",
                "target_confidence": "HIGH",
                "goal_minutes_after_37_before_ht": "|".join(goal_minutes),
                "first_half_period_end_seen": 1,
                "timeline_full": 1,
                "first_half_goal_count": len(all_first_half_goals),
            }

        return {
            TARGET_COLUMN: 0,
            "target_status": "RESOLVED",
            "target_resolution_class": "NEGATIVE_FULL_FIRST_HALF_TIMELINE",
            "target_confidence": "HIGH",
            "goal_minutes_after_37_before_ht": "",
            "first_half_period_end_seen": 1,
            "timeline_full": 1,
            "first_half_goal_count": len(all_first_half_goals),
        }

    return {
        TARGET_COLUMN: np.nan,
        "target_status": "UNRESOLVED",
        "target_resolution_class": "TIMELINE_NOT_PROVEN_COMPLETE",
        "target_confidence": "LOW",
        "goal_minutes_after_37_before_ht": "|".join(goal_minutes),
        "first_half_period_end_seen": int(period_end),
        "timeline_full": int(full),
        "first_half_goal_count": len(all_first_half_goals),
    }


# =============================================================================
# 37' LIVE FEATURES
# =============================================================================

CUMULATIVE_RENAME = {
    "score_87_home": "score_37_home",
    "score_87_away": "score_37_away",
    "leading_team_87": "leading_team_37",
    "goal_diff_87": "goal_diff_37",
    "is_draw_87": "is_draw_37",
    "home_shots_87": "home_shots_37",
    "away_shots_87": "away_shots_37",
    "home_shots_on_target_87": "home_shots_on_target_37",
    "away_shots_on_target_87": "away_shots_on_target_37",
    "home_shots_off_target_87": "home_shots_off_target_37",
    "away_shots_off_target_87": "away_shots_off_target_37",
    "home_shots_blocked_87": "home_shots_blocked_37",
    "away_shots_blocked_87": "away_shots_blocked_37",
    "home_corners_87": "home_corners_37",
    "away_corners_87": "away_corners_37",
    "home_fouls_87": "home_fouls_37",
    "away_fouls_87": "away_fouls_37",
    "home_yellow_cards_87": "home_yellow_cards_37",
    "away_yellow_cards_87": "away_yellow_cards_37",
    "home_red_cards_87": "home_red_cards_37",
    "away_red_cards_87": "away_red_cards_37",
    "home_offsides_87": "home_offsides_37",
    "away_offsides_87": "away_offsides_37",
    "last_goal_minute": "last_goal_minute",
    "last_shot_minute": "last_shot_minute",
    "last_shot_on_target_minute": "last_shot_on_target_minute",
    "last_corner_minute": "last_corner_minute",
    "minutes_since_last_goal": "minutes_since_last_goal",
    "minutes_since_last_shot": "minutes_since_last_shot",
    "minutes_since_last_shot_on_target": "minutes_since_last_shot_on_target",
    "minutes_since_last_corner": "minutes_since_last_corner",
}

LIVE_37_BASE_FEATURES = list(dict.fromkeys(CUMULATIVE_RENAME.values()))

# Schéma diagnostique FIXE. La première version inférait le dénominateur
# LIVE37 à partir des colonnes effectivement produites pour chaque match,
# ce qui donnait des affichages trompeurs comme 57/57, 66/66, 77/77.
# Ici le dénominateur est constant entre les matchs.
LIVE_37_DERIVED_SCORE_FEATURES = [
    "total_goals_37",
    "has_goal_37",
]

LIVE_37_RECENT_CORE_FEATURES = []
LIVE_37_RECENT_XG_FEATURES = []
for _start in (30, 33, 35):
    LIVE_37_RECENT_CORE_FEATURES.extend([
        f"corners_{_start}_37",
        f"fouls_{_start}_37",
        f"cards_{_start}_37",
        f"subs_{_start}_37",
        f"home_corners_{_start}_37",
        f"away_corners_{_start}_37",
        f"shots_{_start}_37",
        f"sot_{_start}_37",
        f"home_shots_{_start}_37",
        f"away_shots_{_start}_37",
        f"home_sot_{_start}_37",
        f"away_sot_{_start}_37",
    ])
    LIVE_37_RECENT_XG_FEATURES.extend([
        f"xg_{_start}_37",
        f"home_xg_{_start}_37",
        f"away_xg_{_start}_37",
    ])

LIVE_37_CUMULATIVE_XG_FEATURES = [
    "total_xg_37",
    "max_shot_xg_37",
    "last_shot_xg_37",
    "home_xg_37",
    "away_xg_37",
]

LIVE_37_CORE_FEATURES = list(dict.fromkeys(
    LIVE_37_BASE_FEATURES
    + LIVE_37_DERIVED_SCORE_FEATURES
    + LIVE_37_RECENT_CORE_FEATURES
))

LIVE_37_XG_FEATURES = list(dict.fromkeys(
    LIVE_37_RECENT_XG_FEATURES
    + LIVE_37_CUMULATIVE_XG_FEATURES
))

LIVE_37_ALL_FEATURES = list(dict.fromkeys(
    LIVE_37_CORE_FEATURES
    + LIVE_37_XG_FEATURES
))


def reconstruct_base_live37(
    timeline_events: Sequence[Mapping[str, Any]],
    shot_events: Sequence[Mapping[str, Any]],
    team_context: Mapping[str, Any],
    timeline_payload: Any,
    audit: Any,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    # <=37 suffit à exclure structurellement toute donnée de deuxième période.
    reconstructed, provenance_87 = audit.reconstruct_from_events(
        timeline_events=timeline_events,
        shot_events=shot_events,
        home_team_id=team_context.get("home_team_id"),
        away_team_id=team_context.get("away_team_id"),
        home_team_name=clean_text(team_context.get("home_team")),
        away_team_name=clean_text(team_context.get("away_team")),
        snapshot_minute=SNAPSHOT_MINUTE,
    )

    features: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}

    for old_name, new_name in CUMULATIVE_RENAME.items():
        value = reconstructed.get(old_name)
        if value is not None:
            features[new_name] = value

            source = clean_text(provenance_87.get(old_name))
            source = source.replace("87", "37")
            features_source = source or "historical_event_reconstruction_le37"
            provenance[new_name] = features_source

    # Si la timeline déclare explicitement une couverture complète, l'absence de
    # goal <=37 prouve bien 0-0. L'ancien audit 87 restait volontairement encore
    # plus conservateur et exigeait >=10 événements.
    if (
        timeline_is_full(timeline_payload)
        and features.get("score_37_home") is None
        and features.get("score_37_away") is None
    ):
        known_goals = []
        side_unknown = False

        for event in timeline_events:
            if not is_first_half_event(event, audit):
                continue

            minute = event_base_minute(event, audit)
            if minute is None or minute > SNAPSHOT_MINUTE:
                continue

            if not audit.event_flags(event).get("goal"):
                continue

            side = audit.identify_side(
                event,
                team_context.get("home_team_id"),
                team_context.get("away_team_id"),
                clean_text(team_context.get("home_team")),
                clean_text(team_context.get("away_team")),
            )

            if side not in {"home", "away"}:
                side_unknown = True
                break

            known_goals.append(side)

        if not side_unknown:
            home_goals = sum(1 for side in known_goals if side == "home")
            away_goals = sum(1 for side in known_goals if side == "away")

            features.update({
                "score_37_home": home_goals,
                "score_37_away": away_goals,
                "goal_diff_37": abs(home_goals - away_goals),
                "is_draw_37": int(home_goals == away_goals),
                "leading_team_37": (
                    0 if home_goals > away_goals
                    else 1 if away_goals > home_goals
                    else 2
                ),
            })

            for name in (
                "score_37_home",
                "score_37_away",
                "goal_diff_37",
                "is_draw_37",
                "leading_team_37",
            ):
                provenance[name] = "full_timeline_goals_le37"

    if (
        safe_float(features.get("score_37_home")) is not None
        and safe_float(features.get("score_37_away")) is not None
    ):
        features["total_goals_37"] = (
            float(features["score_37_home"])
            + float(features["score_37_away"])
        )
        features["has_goal_37"] = int(features["total_goals_37"] > 0)
        provenance["total_goals_37"] = "derived_score_37"
        provenance["has_goal_37"] = "derived_score_37"

    return features, provenance


def classify_shot_event(event: Mapping[str, Any], audit: Any) -> Dict[str, bool]:
    # Priorité à la classification native shotmap du module d'audit.
    try:
        flags = audit.shotmap_flags(event)
        return {
            "shot": bool(flags.get("shot", True)),
            "sot": bool(flags.get("sot")),
            "goal": bool(flags.get("goal")),
            "blocked": bool(flags.get("blocked")),
            "off_target": bool(flags.get("off_target")),
        }
    except Exception:
        flags = audit.event_flags(event)
        return {
            "shot": bool(flags.get("shot")),
            "sot": bool(flags.get("sot")),
            "goal": bool(flags.get("goal")),
            "blocked": bool(flags.get("blocked")),
            "off_target": bool(flags.get("off_target")),
        }


def extract_shot_xg(event: Mapping[str, Any]) -> Optional[float]:
    for key in ("expected_goals", "xg", "expected_goal", "shot_xg"):
        value = safe_float(event.get(key))
        if value is not None:
            return value

    stats = event.get("stats")
    if isinstance(stats, Mapping):
        for key in ("expected_goals", "xg"):
            value = safe_float(stats.get(key))
            if value is not None:
                return value

    return None


def reconstruct_recent_windows(
    timeline_events: Sequence[Mapping[str, Any]],
    shot_events: Sequence[Mapping[str, Any]],
    team_context: Mapping[str, Any],
    audit: Any,
    timeline_full_flag: bool,
    shotmap_feed_proven: bool,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    features: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}

    windows = (30, 33, 35)

    # ---------------------------------------------------------------------
    # Timeline-based pressure: corners / fouls / cards / substitutions
    # ---------------------------------------------------------------------
    for start in windows:
        counters = Counter()
        home_corners = 0
        away_corners = 0
        corner_side_unknown = 0

        for event in timeline_events:
            if not is_first_half_event(event, audit):
                continue

            minute = event_base_minute(event, audit)
            if minute is None or not (start <= minute <= SNAPSHOT_MINUTE):
                continue

            flags = audit.event_flags(event)

            if flags.get("corner"):
                counters["corners"] += 1

                side = audit.identify_side(
                    event,
                    team_context.get("home_team_id"),
                    team_context.get("away_team_id"),
                    clean_text(team_context.get("home_team")),
                    clean_text(team_context.get("away_team")),
                )

                if side == "home":
                    home_corners += 1
                elif side == "away":
                    away_corners += 1
                else:
                    corner_side_unknown += 1

            if flags.get("foul"):
                counters["fouls"] += 1

            if flags.get("yellow") or flags.get("red"):
                counters["cards"] += 1

            if flags.get("sub"):
                counters["subs"] += 1

        for category in ("corners", "fouls", "cards", "subs"):
            # Missing != 0: un zéro n'est enregistré que si la timeline est
            # explicitement complète. Un compte positif reste observable même
            # avec une couverture non déclarée complète.
            if counters[category] > 0 or timeline_full_flag:
                name = f"{category}_{start}_37"
                features[name] = int(counters[category])
                provenance[name] = f"timeline_first_half_{start}_37"

        if (
            corner_side_unknown == 0
            and (home_corners + away_corners > 0 or timeline_full_flag)
        ):
            features[f"home_corners_{start}_37"] = home_corners
            features[f"away_corners_{start}_37"] = away_corners
            provenance[f"home_corners_{start}_37"] = f"timeline_first_half_{start}_37"
            provenance[f"away_corners_{start}_37"] = f"timeline_first_half_{start}_37"

    # ---------------------------------------------------------------------
    # Shotmap pressure: shots / SOT / xG
    # ---------------------------------------------------------------------
    usable_shots = []

    for event in shot_events:
        minute = event_base_minute(event, audit)
        if minute is None or minute > SNAPSHOT_MINUTE:
            continue
        usable_shots.append(event)

    if not shotmap_feed_proven:
        return features, provenance

    for start in windows:
        shot_count = 0
        sot_count = 0
        home_shots = 0
        away_shots = 0
        home_sot = 0
        away_sot = 0
        side_unknown = 0
        xg_total = 0.0
        xg_known = 0
        home_xg = 0.0
        away_xg = 0.0
        home_xg_known = 0
        away_xg_known = 0

        for event in usable_shots:
            minute = event_base_minute(event, audit)
            if minute is None or not (start <= minute <= SNAPSHOT_MINUTE):
                continue

            flags = classify_shot_event(event, audit)
            if not flags.get("shot"):
                continue

            shot_count += 1

            if flags.get("sot"):
                sot_count += 1

            side = audit.identify_side(
                event,
                team_context.get("home_team_id"),
                team_context.get("away_team_id"),
                clean_text(team_context.get("home_team")),
                clean_text(team_context.get("away_team")),
            )

            if side == "home":
                home_shots += 1
                if flags.get("sot"):
                    home_sot += 1
            elif side == "away":
                away_shots += 1
                if flags.get("sot"):
                    away_sot += 1
            else:
                side_unknown += 1

            xg = extract_shot_xg(event)
            if xg is not None:
                xg_total += xg
                xg_known += 1

                if side == "home":
                    home_xg += xg
                    home_xg_known += 1
                elif side == "away":
                    away_xg += xg
                    away_xg_known += 1

        features[f"shots_{start}_37"] = shot_count
        features[f"sot_{start}_37"] = sot_count
        provenance[f"shots_{start}_37"] = f"shotmap_{start}_37"
        provenance[f"sot_{start}_37"] = f"shotmap_{start}_37"

        if side_unknown == 0:
            for name, value in (
                (f"home_shots_{start}_37", home_shots),
                (f"away_shots_{start}_37", away_shots),
                (f"home_sot_{start}_37", home_sot),
                (f"away_sot_{start}_37", away_sot),
            ):
                features[name] = value
                provenance[name] = f"shotmap_{start}_37"

        if xg_known > 0:
            features[f"xg_{start}_37"] = xg_total
            provenance[f"xg_{start}_37"] = f"shotmap_xg_{start}_37"

        if home_xg_known > 0:
            features[f"home_xg_{start}_37"] = home_xg
            provenance[f"home_xg_{start}_37"] = f"shotmap_xg_{start}_37"

        if away_xg_known > 0:
            features[f"away_xg_{start}_37"] = away_xg
            provenance[f"away_xg_{start}_37"] = f"shotmap_xg_{start}_37"

    return features, provenance


def reconstruct_xg_cumulative(
    shot_events: Sequence[Mapping[str, Any]],
    team_context: Mapping[str, Any],
    audit: Any,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    home_xg = 0.0
    away_xg = 0.0
    home_known = 0
    away_known = 0
    unknown_side_xg = 0
    total_known = 0
    total_xg = 0.0
    max_shot_xg = None
    last_shot_xg = None
    last_shot_minute = None

    for event in shot_events:
        minute = event_base_minute(event, audit)
        if minute is None or minute > SNAPSHOT_MINUTE:
            continue

        xg = extract_shot_xg(event)
        if xg is None:
            continue

        total_xg += xg
        total_known += 1

        if max_shot_xg is None or xg > max_shot_xg:
            max_shot_xg = xg

        if last_shot_minute is None or minute >= last_shot_minute:
            last_shot_minute = minute
            last_shot_xg = xg

        side = audit.identify_side(
            event,
            team_context.get("home_team_id"),
            team_context.get("away_team_id"),
            clean_text(team_context.get("home_team")),
            clean_text(team_context.get("away_team")),
        )

        if side == "home":
            home_xg += xg
            home_known += 1
        elif side == "away":
            away_xg += xg
            away_known += 1
        else:
            unknown_side_xg += 1

    features: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}

    if total_known > 0:
        features["total_xg_37"] = total_xg
        features["max_shot_xg_37"] = max_shot_xg
        features["last_shot_xg_37"] = last_shot_xg

        for name in ("total_xg_37", "max_shot_xg_37", "last_shot_xg_37"):
            provenance[name] = "shotmap_expected_goals_le37"

    if unknown_side_xg == 0 and total_known > 0:
        # A side with no xG-valued shot legitimately receives 0 if other shots
        # establish that the shotmap/xG feed is active for the match.
        features["home_xg_37"] = home_xg
        features["away_xg_37"] = away_xg
        provenance["home_xg_37"] = "shotmap_expected_goals_le37"
        provenance["away_xg_37"] = "shotmap_expected_goals_le37"

    return features, provenance


# =============================================================================
# FIRST-HALF OPENING ODDS
# =============================================================================

def get_bookmakers(odds_payload: Any) -> List[Mapping[str, Any]]:
    data = payload_data(odds_payload)

    if isinstance(data, Mapping):
        bookmakers = data.get("bookmakers")
        if isinstance(bookmakers, list):
            return [x for x in bookmakers if isinstance(x, Mapping)]

    return []


def nested_opening(mapping: Any, *keys: str) -> Optional[float]:
    current = mapping

    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)

    if isinstance(current, Mapping):
        current = current.get("opening")

    return safe_float(current)


def line_market_opening(
    markets: Mapping[str, Any],
    market_name: str,
    line_candidates: Sequence[str],
    side: str,
) -> Optional[float]:
    market = markets.get(market_name)
    if not isinstance(market, Mapping):
        return None

    for line in line_candidates:
        value = nested_opening(market, line, side)
        if value is not None:
            return value

    # Accept numeric keys that stringify differently (0.5 vs "0.5").
    for raw_key, raw_value in market.items():
        if clean_text(raw_key) not in set(line_candidates):
            continue
        if isinstance(raw_value, Mapping):
            value = nested_opening(raw_value, side)
            if value is not None:
                return value

    return None


def parse_first_half_opening_odds(odds_payload: Any) -> Dict[str, Any]:
    """
    Produit des cotes consensus = médiane des OPENING disponibles.
    Jamais de fallback vers last_seen.
    """
    bookmakers = get_bookmakers(odds_payload)

    buckets: Dict[str, List[float]] = {
        "odds_fh_home": [],
        "odds_fh_draw": [],
        "odds_fh_away": [],
        "odds_fh_over05": [],
        "odds_fh_under05": [],
        "odds_fh_over15": [],
        "odds_fh_under15": [],
        "odds_fh_over25": [],
        "odds_fh_under25": [],
        "odds_fh_btts_yes": [],
        "odds_fh_btts_no": [],
    }

    bookmakers_with_fh = 0

    for bookmaker in bookmakers:
        markets = bookmaker.get("markets")
        if not isinstance(markets, Mapping):
            continue

        local_found = 0

        for destination, path in (
            ("odds_fh_home", ("first_half_result", "home")),
            ("odds_fh_draw", ("first_half_result", "draw")),
            ("odds_fh_away", ("first_half_result", "away")),
            ("odds_fh_btts_yes", ("btts_first_half", "yes")),
            ("odds_fh_btts_no", ("btts_first_half", "no")),
        ):
            value = nested_opening(markets, *path)
            if value is not None and value > 1.0:
                buckets[destination].append(value)
                local_found += 1

        line_specs = (
            ("0.5", "odds_fh_over05", "odds_fh_under05"),
            ("1.5", "odds_fh_over15", "odds_fh_under15"),
            ("2.5", "odds_fh_over25", "odds_fh_under25"),
        )

        for line, over_name, under_name in line_specs:
            over = line_market_opening(
                markets,
                "first_half_total_goals",
                (line,),
                "over",
            )
            under = line_market_opening(
                markets,
                "first_half_total_goals",
                (line,),
                "under",
            )

            if over is not None and over > 1.0:
                buckets[over_name].append(over)
                local_found += 1

            if under is not None and under > 1.0:
                buckets[under_name].append(under)
                local_found += 1

        if local_found > 0:
            bookmakers_with_fh += 1

    result: Dict[str, Any] = {
        "fh_odds_bookmakers_available": bookmakers_with_fh,
    }

    for destination, values in buckets.items():
        if values:
            result[destination] = float(np.median(values))
            result[f"{destination}_n"] = len(values)
        else:
            result[destination] = np.nan
            result[f"{destination}_n"] = 0

    return result


FH_ODDS_FEATURES = [
    "odds_fh_home",
    "odds_fh_draw",
    "odds_fh_away",
    "odds_fh_over05",
    "odds_fh_under05",
    "odds_fh_over15",
    "odds_fh_under15",
    "odds_fh_over25",
    "odds_fh_under25",
    "odds_fh_btts_yes",
    "odds_fh_btts_no",
]


# =============================================================================
# PER-MATCH COLLECTION
# =============================================================================

def collect_one_match(
    source_row: Mapping[str, Any],
    client: CachedHistoricalClient,
    audit: Any,
    prematch_features: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    match_id = extract_match_id(source_row)

    audit_row: Dict[str, Any] = {
        "match_id": match_id,
        "collected_at_utc": now_utc_iso(),
        "status": "",
        "error": "",
    }

    if not match_id:
        audit_row.update({
            "status": "failed",
            "error": "missing_match_id",
        })
        return None, audit_row

    # ---------------------------------------------------------------------
    # API — detail / timeline / shotmap / odds
    # ---------------------------------------------------------------------
    detail_payload, detail_status, detail_error = client.request(
        f"/football/matches/{match_id}"
    )

    timeline_payload, timeline_status, timeline_error = client.request(
        f"/football/matches/{match_id}/timeline"
    )

    shotmap_payload, shotmap_status, shotmap_error = client.request(
        f"/football/matches/{match_id}/shotmap"
    )

    odds_payload, odds_status, odds_error = client.request(
        f"/football/matches/{match_id}/odds"
    )

    audit_row.update({
        "detail_http_status": detail_status,
        "timeline_http_status": timeline_status,
        "shotmap_http_status": shotmap_status,
        "odds_http_status": odds_status,
        "detail_error": detail_error,
        "timeline_error": timeline_error,
        "shotmap_error": shotmap_error,
        "odds_error": odds_error,
    })

    if timeline_payload is None:
        audit_row.update({
            "status": "failed",
            "error": f"timeline_unavailable:{timeline_error or timeline_status}",
        })
        return None, audit_row

    timeline_events = payload_list(timeline_payload)
    shot_events = payload_list(shotmap_payload)

    if not timeline_events:
        audit_row.update({
            "status": "failed",
            "error": "empty_timeline",
        })
        return None, audit_row

    team_context = get_detail_team_context(
        detail_payload,
        source_row,
        audit,
    )

    # ---------------------------------------------------------------------
    # PRE-MATCH — reuse the already audited/rebuilt source columns.
    # ---------------------------------------------------------------------
    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_minute": SNAPSHOT_MINUTE,
        "match_id": match_id,
        "competition_id": clean_text(source_row.get("competition_id")),
        "season_id": clean_text(source_row.get("season_id")),
        "home_team": clean_text(team_context.get("home_team")),
        "away_team": clean_text(team_context.get("away_team")),
        "kickoff_utc": source_row.get("kickoff_utc", source_row.get("utc_date", "")),
        "source_dataset": "historical_goal_after_87_prematch_reuse",
        "collected_at_utc": now_utc_iso(),
    }

    for feature in prematch_features:
        row[feature] = source_row.get(feature, np.nan)

    # ---------------------------------------------------------------------
    # LIVE @37
    # ---------------------------------------------------------------------
    base_features, base_provenance = reconstruct_base_live37(
        timeline_events=timeline_events,
        shot_events=shot_events,
        team_context=team_context,
        timeline_payload=timeline_payload,
        audit=audit,
    )

    recent_features, recent_provenance = reconstruct_recent_windows(
        timeline_events=timeline_events,
        shot_events=shot_events,
        team_context=team_context,
        audit=audit,
        timeline_full_flag=timeline_is_full(timeline_payload),
        shotmap_feed_proven=(
            shotmap_status == 200
            and (
                bool(shot_events)
                or payload_explicitly_contains_event_list(shotmap_payload)
            )
        ),
    )

    xg_features, xg_provenance = reconstruct_xg_cumulative(
        shot_events=shot_events,
        team_context=team_context,
        audit=audit,
    )

    row.update(base_features)
    row.update(recent_features)
    row.update(xg_features)

    # Stabilise le schéma entre tous les matchs. Une feature non reconstructible
    # reste NaN : aucune valeur n'est inventée.
    for feature in LIVE_37_ALL_FEATURES:
        row.setdefault(feature, np.nan)

    provenance = {}
    provenance.update(base_provenance)
    provenance.update(recent_provenance)
    provenance.update(xg_provenance)

    # ---------------------------------------------------------------------
    # Opening odds specific to first half
    # ---------------------------------------------------------------------
    fh_odds = parse_first_half_opening_odds(odds_payload)
    row.update(fh_odds)

    # ---------------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------------
    target = resolve_target(
        timeline_events=timeline_events,
        timeline_payload=timeline_payload,
        audit=audit,
    )
    row.update(target)

    # ---------------------------------------------------------------------
    # Diagnostics / no-future-leak proof
    # ---------------------------------------------------------------------
    future_feature_event_count = 0

    # Every timestamp-derived feature above is built through <=37 filters.
    # This explicit scan catches any impossible last-event minute >37.
    for feature in (
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ):
        value = safe_float(row.get(feature))
        if value is not None and value > SNAPSHOT_MINUTE:
            future_feature_event_count += 1

    prematch_available = nonempty_count(row, prematch_features)

    # Schéma LIVE37 fixe pour que les couvertures soient comparables entre matchs.
    live_core_available = nonempty_count(row, LIVE_37_CORE_FEATURES)
    live_xg_available = nonempty_count(row, LIVE_37_XG_FEATURES)
    live_available = nonempty_count(row, LIVE_37_ALL_FEATURES)
    fh_odds_available = nonempty_count(row, FH_ODDS_FEATURES)

    audit_row.update({
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "kickoff_utc": row.get("kickoff_utc", ""),
        "timeline_event_count": len(timeline_events),
        "shotmap_event_count": len(shot_events),
        "timeline_coverage": timeline_coverage(timeline_payload),
        "timeline_full": int(timeline_is_full(timeline_payload)),
        "first_half_period_metadata_rate": first_half_period_metadata_rate(timeline_events),
        "first_half_period_end_seen": int(first_half_period_end_seen(timeline_events, audit)),
        "prematch_available": prematch_available,
        "prematch_total": len(prematch_features),
        "live37_available": live_available,
        "live37_total_fixed_schema": len(LIVE_37_ALL_FEATURES),
        "live37_core_available": live_core_available,
        "live37_core_total": len(LIVE_37_CORE_FEATURES),
        "live37_xg_available": live_xg_available,
        "live37_xg_total": len(LIVE_37_XG_FEATURES),
        "fh_odds_available": fh_odds_available,
        "fh_odds_total": len(FH_ODDS_FEATURES),
        "target_status": row.get("target_status", ""),
        TARGET_COLUMN: row.get(TARGET_COLUMN, np.nan),
        "future_feature_event_count": future_feature_event_count,
        "live_provenance_json": json.dumps(
            provenance,
            ensure_ascii=False,
            sort_keys=True,
        ),
    })

    if future_feature_event_count > 0:
        audit_row.update({
            "status": "failed",
            "error": "future_event_leak_detected",
        })
        return None, audit_row

    # A row with unresolved target is still useful for coverage diagnostics,
    # but it must not enter the ML dataset.
    if row.get("target_status") != "RESOLVED":
        audit_row.update({
            "status": "partial",
            "error": "target_unresolved",
        })
        return None, audit_row

    audit_row["status"] = "done"

    row["_prematch_available"] = prematch_available
    row["_prematch_total"] = len(prematch_features)
    row["_live37_available"] = live_available
    row["_live37_total_fixed_schema"] = len(LIVE_37_ALL_FEATURES)
    row["_live37_core_available"] = live_core_available
    row["_live37_core_total"] = len(LIVE_37_CORE_FEATURES)
    row["_live37_xg_available"] = live_xg_available
    row["_live37_xg_total"] = len(LIVE_37_XG_FEATURES)
    row["_fh_odds_available"] = fh_odds_available
    row["_fh_odds_total"] = len(FH_ODDS_FEATURES)
    row["_live_provenance_json"] = audit_row["live_provenance_json"]

    return row, audit_row


# =============================================================================
# DATAFRAME / OUTPUT HELPERS
# =============================================================================

def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_or_replace(
    df: pd.DataFrame,
    row: Mapping[str, Any],
    key: str = "match_id",
) -> pd.DataFrame:
    new_df = pd.DataFrame([dict(row)])

    if df.empty:
        return new_df

    if key in df.columns and key in new_df.columns:
        value = clean_text(new_df.iloc[0][key])
        if value:
            df = df[df[key].astype(str) != value].copy()

    return pd.concat([df, new_df], ignore_index=True, sort=False)


def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def build_feature_coverage(
    dataset_df: pd.DataFrame,
    prematch_features: Sequence[str],
) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame(
            columns=["feature", "group", "available", "rows", "coverage_pct"]
        )

    metadata = {
        "schema_version",
        "snapshot_minute",
        "match_id",
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        "source_dataset",
        "collected_at_utc",
        TARGET_COLUMN,
        "target_status",
        "target_resolution_class",
        "target_confidence",
        "goal_minutes_after_37_before_ht",
        "first_half_period_end_seen",
        "timeline_full",
        "first_half_goal_count",
        "fh_odds_bookmakers_available",
        "_prematch_available",
        "_prematch_total",
        "_live37_available",
        "_live37_total_fixed_schema",
        "_live37_core_available",
        "_live37_core_total",
        "_live37_xg_available",
        "_live37_xg_total",
        "_fh_odds_available",
        "_fh_odds_total",
        "_live_provenance_json",
    }

    rows = []
    total = len(dataset_df)

    for column in dataset_df.columns:
        if column in metadata or column.endswith("_n"):
            continue

        series = dataset_df[column]
        available = int(series.notna().sum())

        if column in prematch_features:
            group = "PRE_MATCH"
        elif column in FH_ODDS_FEATURES:
            group = "ODDS_FIRST_HALF"
        else:
            group = "LIVE_37"

        rows.append({
            "feature": column,
            "group": group,
            "available": available,
            "rows": total,
            "coverage_pct": (100.0 * available / total) if total else 0.0,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["group", "coverage_pct", "feature"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return result


def write_report(
    path: Path,
    source_path: Path,
    dataset_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    client: CachedHistoricalClient,
) -> None:
    total_done = len(dataset_df)

    positives = 0
    negatives = 0

    if not dataset_df.empty and TARGET_COLUMN in dataset_df.columns:
        target = pd.to_numeric(dataset_df[TARGET_COLUMN], errors="coerce")
        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())

    positive_rate = (
        100.0 * positives / (positives + negatives)
        if (positives + negatives) > 0
        else 0.0
    )

    attempted = len(audit_df)
    failed = 0
    partial = 0

    if not audit_df.empty and "status" in audit_df.columns:
        failed = int((audit_df["status"] == "failed").sum())
        partial = int((audit_df["status"] == "partial").sum())

    def coverage_group(group: str) -> str:
        if coverage_df.empty:
            return "n/a"

        block = coverage_df[coverage_df["group"] == group]
        if block.empty:
            return "n/a"

        return (
            f"médiane {block['coverage_pct'].median():.1f}% | "
            f"min {block['coverage_pct'].min():.1f}% | "
            f"max {block['coverage_pct'].max():.1f}%"
        )

    lines = [
        "# First Half 37 — Historical Dataset Report",
        "",
        f"- Généré : `{now_utc_iso()}`",
        f"- Source PRE-MATCH : `{source_path}`",
        f"- Schéma : `{SCHEMA_VERSION}`",
        f"- Snapshot : **{SNAPSHOT_MINUTE}'**",
        "",
        "## Résumé",
        "",
        f"- Matchs tentés : **{attempted}**",
        f"- Lignes ML résolues : **{total_done}**",
        f"- Positifs goal_after_37_before_ht : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Taux positif : **{positive_rate:.2f}%**",
        f"- Partiels : **{partial}**",
        f"- Échecs : **{failed}**",
        "",
        "## Couverture des groupes de features",
        "",
        f"- PRE_MATCH : {coverage_group('PRE_MATCH')}",
        f"- LIVE_37 : {coverage_group('LIVE_37')}",
        f"- ODDS_FIRST_HALF : {coverage_group('ODDS_FIRST_HALF')}",
        "",
        "## API / cache",
        "",
        f"- Requêtes API réelles : **{client.request_count}**",
        f"- HTTP 429 : **{client.retry_429_count}**",
        f"- Cache hits First Half 37 : **{client.cache_hits}**",
        f"- Cache hits ancien dataset : **{client.legacy_cache_hits}**",
        f"- Cache misses : **{client.cache_misses}**",
        f"- Cache writes : **{client.cache_writes}**",
        "",
        "## Règles anti-fuite",
        "",
        "- Les features LIVE sont filtrées avec `minute <= 37`.",
        "- Les buts après 37' servent uniquement au target.",
        "- Les cotes utilisent uniquement `opening`; jamais `last_seen`.",
        "- Les stats finales ne sont pas utilisées comme features.",
        "- Aucun remplissage silencieux des valeurs manquantes.",
        "",
        "## Étape suivante",
        "",
        "Auditer ce rapport et la couverture. Ensuite construire le schéma ML 37' "
        "et effectuer une validation chronologique avant tout scanner production.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# SOURCE PREPARATION
# =============================================================================

def prepare_source(
    source_path: Path,
    limit: int,
    strategy: str,
    random_seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    if "match_id" not in df.columns:
        raise ValueError("Le source CSV doit contenir `match_id`.")

    df = df.drop_duplicates(subset=["match_id"], keep="last").copy()

    date_column = None
    for candidate in ("kickoff_utc", "utc_date"):
        if candidate in df.columns:
            date_column = candidate
            break

    if strategy == "recent" and date_column:
        parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        df = df.assign(_sort_date=parsed).sort_values(
            "_sort_date",
            ascending=False,
            na_position="last",
        ).drop(columns=["_sort_date"])

    elif strategy == "oldest" and date_column:
        parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        df = df.assign(_sort_date=parsed).sort_values(
            "_sort_date",
            ascending=True,
            na_position="last",
        ).drop(columns=["_sort_date"])

    elif strategy == "random":
        df = df.sample(frac=1.0, random_state=random_seed)

    if limit > 0:
        df = df.head(limit)

    return df.reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit un dataset historique à 37' pour prédire un but "
            "avant la mi-temps."
        )
    )

    parser.add_argument(
        "--source-csv",
        default=None,
        help="Dataset historique source contenant match_id + PRE-MATCH.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Défaut: outputs/first_half_37",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Nombre maximal de matchs source à considérer. Défaut: 100.",
    )

    parser.add_argument(
        "--strategy",
        choices=("recent", "oldest", "random", "source"),
        default="recent",
        help="Ordre de sélection du test. Défaut: recent.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=87,
        help="Seed si --strategy random.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les match_id déjà présents dans le dataset/audit.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcule et remplace les matchs déjà présents.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Sauvegarde tous les N matchs. Défaut: 5.",
    )

    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=DEFAULT_MIN_API_INTERVAL,
        help=(
            "Intervalle minimal entre appels API réels. Défaut: 0.75s "
            "(~1.33 req/s)."
        ),
    )

    parser.add_argument(
        "--cache-disable",
        action="store_true",
        help="Désactive le cache (déconseillé).",
    )

    parser.add_argument(
        "--no-legacy-cache",
        action="store_true",
        help="Ne lit pas l'ancien cache historical_dataset_v2.",
    )

    args = parser.parse_args()

    if ENV_FILE.exists():
        # Ce projet a déjà rencontré des variables shell obsolètes : le .env de
        # la racine doit être la source de vérité.
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()

    api_key = clean_text(os.getenv("STATS_API_KEY"))
    base_url = clean_text(
        os.getenv("STATS_API_BASE_URL", "https://api.thestatsapi.com/api")
    ).rstrip("/")
    timeout = safe_int(os.getenv("STATS_API_TIMEOUT")) or 20

    if not api_key:
        print("ERREUR : STATS_API_KEY absent du .env")
        return 2

    source_path = source_path_from_candidates(args.source_csv)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    dataset_dir = output_dir / "datasets"
    audit_dir = output_dir / "audit"
    cache_dir = output_dir / "cache" / "api"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_path = dataset_dir / RAW_DATASET_NAME
    match_audit_path = audit_dir / MATCH_AUDIT_NAME
    coverage_path = audit_dir / FEATURE_COVERAGE_NAME
    rejections_path = audit_dir / REJECTIONS_NAME
    report_path = audit_dir / REPORT_NAME

    print()
    print("=" * 100)
    print("FIRST HALF 37 — HISTORICAL DATASET COLLECTOR")
    print("=" * 100)
    print(f"Source : {source_path}")
    print(f"Output : {output_dir}")
    print(f"API base : {base_url}")
    print(f"Snapshot : {SNAPSHOT_MINUTE}'")
    print("Target : goal AFTER 37' and BEFORE half-time")
    print(f"API min interval : {max(0.1, args.min_api_interval):.2f}s")
    print()

    audit_module = load_module(
        "historical_live87_audit_for_first_half37",
        AUDIT_MODULE_PATH,
    )

    prematch_module = load_module(
        "prematch_rebuild_for_first_half37",
        PREMATCH_MODULE_PATH,
    )

    prematch_features = list(prematch_module.PRE_MATCH_FEATURES)

    source_df = prepare_source(
        source_path=source_path,
        limit=max(0, args.limit),
        strategy=args.strategy,
        random_seed=args.random_seed,
    )

    print(f"Matchs source sélectionnés : {len(source_df)}")
    print(f"Features PRE réutilisées : {len(prematch_features)}")

    base_client = audit_module.StatsAPIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=6,
        min_interval=max(0.1, args.min_api_interval),
        backoff=2.0,
    )

    legacy_cache = None
    if not args.no_legacy_cache and LEGACY_CACHE_DIR.exists():
        legacy_cache = LEGACY_CACHE_DIR

    client = CachedHistoricalClient(
        base_client=base_client,
        cache_dir=cache_dir,
        legacy_cache_dir=legacy_cache,
        enabled=not args.cache_disable,
    )

    dataset_df = load_existing(raw_path)
    audit_df = load_existing(match_audit_path)
    rejections_df = load_existing(rejections_path)

    # Nettoyage d'un champ diagnostique de la toute première version du
    # collecteur. Il utilisait un dénominateur LIVE37 dynamique par match.
    # On le retire pour ne garder que le schéma fixe de cette version.
    dataset_df = dataset_df.drop(
        columns=["_live37_total_observed_schema"],
        errors="ignore",
    )
    audit_df = audit_df.drop(
        columns=["live37_total_observed_schema"],
        errors="ignore",
    )

    existing_done = set()
    existing_audited = set()

    if not dataset_df.empty and "match_id" in dataset_df.columns:
        existing_done = set(dataset_df["match_id"].astype(str))

    if not audit_df.empty and "match_id" in audit_df.columns:
        existing_audited = set(audit_df["match_id"].astype(str))

    processed_this_run = 0
    skipped = 0

    try:
        for index, source_row_series in source_df.iterrows():
            source_row = source_row_series.to_dict()
            match_id = extract_match_id(source_row)

            if not match_id:
                continue

            if args.resume and not args.force:
                if match_id in existing_done or match_id in existing_audited:
                    skipped += 1
                    continue

            print()
            print("-" * 100)
            print(f"[{index + 1}/{len(source_df)}] {match_id}")

            row, match_audit = collect_one_match(
                source_row=source_row,
                client=client,
                audit=audit_module,
                prematch_features=prematch_features,
            )

            audit_df = append_or_replace(
                audit_df,
                match_audit,
            )

            if row is not None:
                dataset_df = append_or_replace(
                    dataset_df,
                    row,
                )

                target = safe_int(row.get(TARGET_COLUMN))
                print(
                    f"{row.get('home_team', '')} vs {row.get('away_team', '')} | "
                    f"PRE={row.get('_prematch_available', 0)}/{row.get('_prematch_total', 0)} | "
                    f"LIVE37_CORE={row.get('_live37_core_available', 0)}/{row.get('_live37_core_total', 0)} | "
                    f"XG37={row.get('_live37_xg_available', 0)}/{row.get('_live37_xg_total', 0)} | "
                    f"FH_ODDS={row.get('_fh_odds_available', 0)}/{row.get('_fh_odds_total', 0)} | "
                    f"TARGET={target} | done"
                )
            else:
                rejection = {
                    "match_id": match_id,
                    "home_team": clean_text(source_row.get("home_team")),
                    "away_team": clean_text(source_row.get("away_team")),
                    "status": match_audit.get("status", "failed"),
                    "reason": match_audit.get("error", ""),
                    "collected_at_utc": match_audit.get("collected_at_utc", now_utc_iso()),
                }

                rejections_df = append_or_replace(
                    rejections_df,
                    rejection,
                )

                print(
                    f"REJET/PARTIAL | {match_audit.get('status')} | "
                    f"{match_audit.get('error')}"
                )

            processed_this_run += 1

            if processed_this_run % max(1, args.save_every) == 0:
                save_csv_atomic(dataset_df, raw_path)
                save_csv_atomic(audit_df, match_audit_path)
                save_csv_atomic(rejections_df, rejections_path)

                coverage_df = build_feature_coverage(
                    dataset_df,
                    prematch_features,
                )
                save_csv_atomic(coverage_df, coverage_path)

                write_report(
                    report_path,
                    source_path,
                    dataset_df,
                    audit_df,
                    coverage_df,
                    client,
                )

                print(
                    f"Sauvegarde intermédiaire | dataset={len(dataset_df)} | "
                    f"API={client.request_count} | cache={client.cache_hits + client.legacy_cache_hits}"
                )

    except KeyboardInterrupt:
        print("\nInterruption utilisateur — sauvegarde avant sortie...")

    finally:
        save_csv_atomic(dataset_df, raw_path)
        save_csv_atomic(audit_df, match_audit_path)
        save_csv_atomic(rejections_df, rejections_path)

        coverage_df = build_feature_coverage(
            dataset_df,
            prematch_features,
        )
        save_csv_atomic(coverage_df, coverage_path)

        write_report(
            report_path,
            source_path,
            dataset_df,
            audit_df,
            coverage_df,
            client,
        )

    print()
    print("=" * 100)
    print("COLLECTE TERMINEE")
    print("=" * 100)
    print(f"Traités cette exécution : {processed_this_run}")
    print(f"Ignorés (--resume) : {skipped}")
    print(f"Dataset total : {len(dataset_df)}")
    print(f"Audit total : {len(audit_df)}")
    print(f"Requêtes API réelles : {client.request_count}")
    print(f"HTTP 429 : {client.retry_429_count}")
    print(f"Cache hits nouveau : {client.cache_hits}")
    print(f"Cache hits ancien : {client.legacy_cache_hits}")
    print(f"Dataset : {raw_path}")
    print(f"Rapport : {report_path}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
