#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_historical_dataset_v2_hierarchical.py

COLLECTE HISTORIQUE HIÉRARCHIQUE — DATASET GOAL AFTER 87
========================================================

OBJECTIF
--------
Construire le dataset historique V2 en parcourant l'API plus intelligemment :

    Compétitions
        ↓
    Saisons
        ↓
    Matchs terminés
        ↓
    Reconstruction PRE as-of kickoff
        ↓
    Reconstruction LIVE87
        ↓
    Target goal_after_87
        ↓
    Dataset + index incrémental

POURQUOI CE FICHIER
-------------------
Le collecteur simple `collect_historical_dataset_v2.py` liste surtout les matchs
terminés les plus récents. Résultat : beaucoup de matchs 2026.

Ce collecteur hiérarchique vise à explorer l'historique disponible compétition
par compétition et saison par saison, sans recollecter les matchs déjà présents
dans l'index.

SÉCURITÉ
--------
- Ne modifie PAS le scanner de production.
- Ne modifie PAS le modèle.
- Ne modifie PAS `collect_historical_dataset_v2.py`.
- Réutilise le même moteur `collect_one_match()` déjà validé.
- Respecte l'index existant : aucun match déjà `done` avec le même schema_version
  n'est recollecté, sauf avec `--force`.
- Sauvegarde régulièrement.

CACHE
-----
Ce fichier intègre un cache disque optionnel pour réduire les appels répétés :

    outputs/historical_dataset_v2/api_cache_v2/

Le listing `/football/matches` n'est PAS caché.
Les détails, timelines, shotmaps, stats, odds et historiques peuvent être cachés.

UTILISATION
-----------
Test rapide :
    python3 collect_historical_dataset_v2_hierarchical.py --limit-new 50

Avec caffeinate :
    caffeinate -dimsu python3 collect_historical_dataset_v2_hierarchical.py \
        --resume \
        --limit-new 1000 \
        --max-competitions 30 \
        --max-seasons-per-competition 5 \
        --max-pages-per-season 20

Cibler des compétitions précises :
    python3 collect_historical_dataset_v2_hierarchical.py \
        --resume \
        --competition-ids comp_9799 comp_5840 \
        --limit-new 1000

Cibler aussi des saisons :
    python3 collect_historical_dataset_v2_hierarchical.py \
        --resume \
        --competition-ids comp_9799 \
        --season-ids sn_8454787 sn_1234567 \
        --limit-new 500

Si l'endpoint des saisons n'est pas disponible, le script tente quand même
de lister les matchs par compétition sans saison explicite.

SORTIES
-------
Même dossier que le collecteur V2 :

outputs/historical_dataset_v2/
├── Historical_Dataset_Goal_After_87_RAW_v2.csv
├── Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
├── historical_matches_index_v2.csv
├── historical_dataset_v2_rejections.csv
├── historical_dataset_v2_feature_coverage.csv
├── historical_dataset_v2_report.md
└── hierarchical_collection_state_v2.json

IMPORTANT
---------
Ce fichier doit être placé à la racine du projet, dans le même dossier que :

    collect_historical_dataset_v2.py
    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_COLLECTOR_PATH = (
    BASE_DIR
    / "collect_historical_dataset_v2.py"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
)

DEFAULT_CACHE_DIR = (
    DEFAULT_OUTPUT_DIR
    / "api_cache_v2"
)

STATE_FILENAME = "hierarchical_collection_state_v2.json"


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


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(value: Any) -> Optional[int]:
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

    return int(number)


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

    if not endpoint:
        return ""

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    while "//" in endpoint:
        endpoint = endpoint.replace("//", "/")

    return endpoint


def as_list_from_payload(payload: Any) -> List[Dict[str, Any]]:
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
            "competitions",
            "seasons",
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


def extract_id(item: Mapping[str, Any]) -> str:
    for key in (
        "id",
        "competition_id",
        "season_id",
        "match_id",
        "event_id",
    ):
        value = item.get(key)

        if value is not None:
            return clean_text(value)

    return ""


def extract_name(item: Mapping[str, Any]) -> str:
    for key in (
        "name",
        "title",
        "competition_name",
        "league_name",
        "season_name",
    ):
        value = item.get(key)

        if value is not None:
            return clean_text(value)

    return ""


def extract_match_id(match: Mapping[str, Any]) -> str:
    for key in (
        "match_id",
        "id",
        "event_id",
    ):
        value = match.get(key)

        if value is not None:
            return clean_text(value)

    return ""


def extract_competition_id(match: Mapping[str, Any]) -> str:
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


def extract_season_id(match: Mapping[str, Any]) -> str:
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


def path_resolve(value: Optional[str], default: Path) -> Path:
    if value:
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = BASE_DIR / path

        return path.resolve()

    return default.resolve()


# =============================================================================
# ORIGINAL COLLECTOR LOADING
# =============================================================================

def load_original_collector():
    if not ORIGINAL_COLLECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Fichier original introuvable : {ORIGINAL_COLLECTOR_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "collect_historical_dataset_v2_original",
        ORIGINAL_COLLECTOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Impossible de charger : {ORIGINAL_COLLECTOR_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[
        "collect_historical_dataset_v2_original"
    ] = module

    spec.loader.exec_module(module)

    return module


# =============================================================================
# PERSISTENT API CACHE
# =============================================================================

CACHE_EXCLUDED_ENDPOINTS = {
    "/football/matches",
    "football/matches",
}


class PersistentCachedClient:
    """
    Wrapper autour du client API utilisé par les modules existants.

    Le collecteur original attend une méthode :

        client.get(endpoint, params=None)

    Ce wrapper conserve cette interface.
    """

    def __init__(
        self,
        base_client: Any,
        cache_dir: Path,
        enabled: bool = True,
        save_none: bool = False,
        ttl_hours: float = 0.0,
    ):
        self.base_client = base_client
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.save_none = save_none
        self.ttl_seconds = (
            float(ttl_hours) * 3600.0
            if ttl_hours and ttl_hours > 0
            else 0.0
        )

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self.cache_skipped = 0
        self.cache_expired = 0

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(
            self.base_client,
            name,
        )

    @property
    def request_count(self) -> int:
        return int(
            getattr(
                self.base_client,
                "request_count",
                0,
            )
        )

    @property
    def retry_429_count(self) -> int:
        return int(
            getattr(
                self.base_client,
                "retry_429_count",
                0,
            )
        )

    def should_cache(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> bool:
        if not self.enabled:
            return False

        ep = normalize_endpoint(endpoint)
        ep_no = ep.lstrip("/")

        if ep in CACHE_EXCLUDED_ENDPOINTS:
            return False

        if ep_no in CACHE_EXCLUDED_ENDPOINTS:
            return False

        lowered = ep_no.lower()

        if "live-matches" in lowered:
            return False

        return True

    def cache_key(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> str:
        raw = stable_json({
            "endpoint": normalize_endpoint(endpoint),
            "params": dict(params or {}),
        })

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def cache_path(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> Path:
        key = self.cache_key(
            endpoint,
            params,
        )

        return (
            self.cache_dir
            / key[:2]
            / f"{key}.json"
        )

    def is_expired(self, record: Mapping[str, Any]) -> bool:
        if self.ttl_seconds <= 0:
            return False

        cached_at = clean_text(
            record.get("cached_at_utc")
        )

        if not cached_at:
            return True

        try:
            dt = datetime.fromisoformat(
                cached_at.replace("Z", "+00:00")
            )
        except Exception:
            return True

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        age = (
            datetime.now(timezone.utc)
            - dt.astimezone(timezone.utc)
        ).total_seconds()

        return age > self.ttl_seconds

    def read_cache(self, path: Path) -> Tuple[bool, Optional[Any]]:
        if not path.exists():
            return False, None

        try:
            record = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )
        except Exception:
            return False, None

        if not isinstance(record, Mapping):
            return False, None

        if self.is_expired(record):
            self.cache_expired += 1
            return False, None

        if "payload" not in record:
            return False, None

        return True, record.get("payload")

    def write_cache(
        self,
        path: Path,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
        payload: Any,
    ) -> None:
        if payload is None and not self.save_none:
            self.cache_skipped += 1
            return

        record = {
            "cached_at_utc": now_utc_iso(),
            "endpoint": normalize_endpoint(endpoint),
            "params": dict(params or {}),
            "payload": payload,
        }

        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            tmp = path.with_suffix(
                path.suffix + ".tmp"
            )

            tmp.write_text(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            tmp.replace(path)

            self.cache_writes += 1

        except Exception:
            self.cache_skipped += 1

    def get(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        use_cache = self.should_cache(
            endpoint,
            params,
        )

        path = None

        if use_cache:
            path = self.cache_path(
                endpoint,
                params,
            )

            ok, payload = self.read_cache(
                path
            )

            if ok:
                self.cache_hits += 1
                return payload

        self.cache_misses += 1

        payload = self.base_client.get(
            endpoint,
            params=params,
        )

        if use_cache and path is not None:
            self.write_cache(
                path,
                endpoint,
                params,
                payload,
            )

        return payload

    def print_stats(self) -> None:
        print()
        print("=" * 100)
        print("CACHE API V2")
        print("=" * 100)
        print(f"Cache activé : {self.enabled}")
        print(f"Dossier cache : {self.cache_dir}")
        print(f"Cache hits : {self.cache_hits}")
        print(f"Cache misses : {self.cache_misses}")
        print(f"Cache writes : {self.cache_writes}")
        print(f"Cache expired : {self.cache_expired}")
        print(f"Cache skipped : {self.cache_skipped}")
        print(f"Requêtes API réelles : {self.request_count}")
        print(f"HTTP 429 retentés : {self.retry_429_count}")
        print("=" * 100)


# =============================================================================
# STATE
# =============================================================================

def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "created_at_utc": now_utc_iso(),
            "updated_at_utc": now_utc_iso(),
            "competition_cursor": {},
            "completed_competition_seasons": [],
            "runs": [],
        }

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        if isinstance(payload, Mapping):
            result = dict(payload)
        else:
            result = {}

    except Exception:
        result = {}

    result.setdefault("created_at_utc", now_utc_iso())
    result.setdefault("updated_at_utc", now_utc_iso())
    result.setdefault("competition_cursor", {})
    result.setdefault("completed_competition_seasons", [])
    result.setdefault("runs", [])

    return result


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = dict(state)
    payload["updated_at_utc"] = now_utc_iso()

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


# =============================================================================
# COMPETITIONS / SEASONS DISCOVERY
# =============================================================================

def list_competitions(
    client: Any,
    limit: int,
    max_pages: int,
    page_sleep: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()

    endpoints = [
        "/football/competitions",
    ]

    for endpoint in endpoints:
        for page in range(1, max_pages + 1):
            params = {
                "page": page,
                "per_page": 100,
            }

            payload = client.get(
                endpoint,
                params=params,
            )

            items = as_list_from_payload(
                payload
            )

            if not items:
                break

            added = 0

            for item in items:
                competition_id = (
                    clean_text(item.get("id"))
                    or clean_text(item.get("competition_id"))
                )

                if not competition_id:
                    continue

                if competition_id in seen:
                    continue

                seen.add(
                    competition_id
                )

                rows.append(
                    item
                )

                added += 1

                if limit > 0 and len(rows) >= limit:
                    return rows

            if added == 0:
                break

            if page_sleep > 0:
                time.sleep(page_sleep)

    return rows


def fetch_seasons_from_endpoint(
    client: Any,
    endpoint: str,
    competition_id: str,
    max_pages: int,
    page_sleep: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {
            "competition_id": competition_id,
            "page": page,
            "per_page": 100,
        }

        payload = client.get(
            endpoint,
            params=params,
        )

        items = as_list_from_payload(
            payload
        )

        if not items:
            break

        added = 0

        for item in items:
            season_id = (
                clean_text(item.get("id"))
                or clean_text(item.get("season_id"))
            )

            if not season_id:
                continue

            if season_id in seen:
                continue

            seen.add(season_id)
            rows.append(item)
            added += 1

        if added == 0:
            break

        if page_sleep > 0:
            time.sleep(page_sleep)

    return rows


def list_seasons_for_competition(
    client: Any,
    competition_id: str,
    max_pages: int,
    page_sleep: float,
) -> List[Dict[str, Any]]:
    """
    Essaie plusieurs formes d'endpoints car le nom exact peut varier.

    Si rien ne marche, retourne une liste vide.
    Le collecteur utilisera alors un fallback :
        lister les matchs par compétition sans season_id explicite.
    """

    endpoints = [
        f"/football/competitions/{competition_id}/seasons",
        "/football/seasons",
        "/football/competition-seasons",
    ]

    merged: List[Dict[str, Any]] = []
    seen = set()

    for endpoint in endpoints:
        items = fetch_seasons_from_endpoint(
            client=client,
            endpoint=endpoint,
            competition_id=competition_id,
            max_pages=max_pages,
            page_sleep=page_sleep,
        )

        for item in items:
            season_id = (
                clean_text(item.get("id"))
                or clean_text(item.get("season_id"))
            )

            if not season_id:
                continue

            if season_id in seen:
                continue

            seen.add(season_id)
            merged.append(item)

    return merged


def build_manual_competitions(
    competition_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": clean_text(comp_id),
            "name": clean_text(comp_id),
        }
        for comp_id in competition_ids
        if clean_text(comp_id)
    ]


def build_manual_seasons(
    season_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": clean_text(season_id),
            "name": clean_text(season_id),
        }
        for season_id in season_ids
        if clean_text(season_id)
    ]


# =============================================================================
# MATCH LISTING
# =============================================================================

def list_finished_matches_for_scope(
    client: Any,
    competition_id: Optional[str],
    season_id: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    max_pages: int,
    page_sleep: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params: Dict[str, Any] = {
            "status": "finished",
            "page": page,
            "per_page": 100,
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

        items = as_list_from_payload(
            payload
        )

        if not items:
            break

        added = 0

        for item in items:
            match_id = extract_match_id(
                item
            )

            if not match_id:
                continue

            if match_id in seen:
                continue

            seen.add(match_id)
            rows.append(item)
            added += 1

        if added == 0:
            break

        if page_sleep > 0:
            time.sleep(page_sleep)

    return rows


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def prepare_runtime_modules(
    collector: Any,
    cache_dir: Path,
    cache_enabled: bool,
    cache_save_none: bool,
    cache_ttl_hours: float,
):
    prematch = collector.load_module(
        "prematch_rebuild_runtime",
        collector.PREMATCH_MODULE_PATH,
    )

    live87 = collector.load_module(
        "historical_live87_runtime",
        collector.LIVE87_MODULE_PATH,
    )

    prematch.lab = prematch.load_lab()

    base_client = prematch.build_rate_limited_client(
        prematch.lab
    )

    client = PersistentCachedClient(
        base_client=base_client,
        cache_dir=cache_dir,
        enabled=cache_enabled,
        save_none=cache_save_none,
        ttl_hours=cache_ttl_hours,
    )

    return prematch, live87, client


def load_rejections(path: Path, force: bool) -> List[Dict[str, Any]]:
    if path.exists() and not force:
        try:
            return (
                pd.read_csv(path)
                .to_dict(orient="records")
            )
        except Exception:
            return []

    return []


def should_skip_scope(
    state: Mapping[str, Any],
    competition_id: str,
    season_id: str,
    ignore_completed_scopes: bool,
) -> bool:
    if not ignore_completed_scopes:
        return False

    key = f"{competition_id}::{season_id}"

    completed = set(
        clean_text(item)
        for item in state.get(
            "completed_competition_seasons",
            []
        )
    )

    return key in completed


def mark_scope_completed(
    state: Dict[str, Any],
    competition_id: str,
    season_id: str,
) -> None:
    key = f"{competition_id}::{season_id}"

    completed = list(
        state.get(
            "completed_competition_seasons",
            []
        )
    )

    if key not in completed:
        completed.append(key)

    state["completed_competition_seasons"] = completed


def collect_match_stub(
    collector: Any,
    prematch: Any,
    live87: Any,
    client: Any,
    stub: Mapping[str, Any],
    core_features: Sequence[str],
    index_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    rejection_rows: List[Dict[str, Any]],
    output_dir: Path,
    force: bool,
    xg_limit: int,
    with_league_second_half: bool,
    preferred_bookmaker: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], bool]:
    match_id = extract_match_id(
        stub
    )

    if not match_id:
        return (
            index_df,
            dataset_df,
            rejection_rows,
            False,
        )

    if collector.already_done(
        index_df=index_df,
        match_id=match_id,
        force=force,
    ):
        return (
            index_df,
            dataset_df,
            rejection_rows,
            False,
        )

    try:
        row, status_row = collector.collect_one_match(
            prematch=prematch,
            live87=live87,
            client=client,
            match_stub=stub,
            core_features=core_features,
            xg_limit=xg_limit,
            with_league_second_half=with_league_second_half,
            preferred_bookmaker=preferred_bookmaker,
        )

        index_df = collector.append_or_replace_index(
            index_df,
            status_row,
        )

        if row is not None:
            dataset_df = collector.append_or_replace_dataset(
                dataset_df,
                row,
                force=force,
            )

            print(
                f"{row.get('home_team', '')} "
                f"vs "
                f"{row.get('away_team', '')}"
            )

            print(
                f"PRE={row.get('_pre_available', 0)}/59 | "
                f"LIVE={row.get('_live_available', 0)}/37 | "
                f"CORE={row.get('_core_available', 0)}/{len(core_features)} | "
                f"target={row.get(collector.TARGET_COLUMN)} | "
                f"status={status_row.get('status')}"
            )

            if status_row.get("status") != "done":
                rejection_rows.append({
                    "match_id": match_id,
                    "status": status_row.get("status", ""),
                    "reason": status_row.get("error", ""),
                    "collected_at_utc": status_row.get("collected_at_utc", ""),
                })

        else:
            print(
                f"Rejet : {status_row.get('error', '')}"
            )

            rejection_rows.append({
                "match_id": match_id,
                "status": status_row.get("status", "failed"),
                "reason": status_row.get("error", ""),
                "collected_at_utc": status_row.get("collected_at_utc", ""),
            })

        return (
            index_df,
            dataset_df,
            rejection_rows,
            True,
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

        print(
            f"ERREUR {match_id}: {error}"
        )

        status_row = {
            "match_id": match_id,
            "schema_version": collector.SCHEMA_VERSION,
            "status": "failed",
            "competition_id": extract_competition_id(stub),
            "season_id": extract_season_id(stub),
            "utc_date": collector.extract_match_date_generic(stub),
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

        index_df = collector.append_or_replace_index(
            index_df,
            status_row,
        )

        rejection_rows.append({
            "match_id": match_id,
            "status": "failed",
            "reason": error,
            "collected_at_utc": now_utc_iso(),
        })

        return (
            index_df,
            dataset_df,
            rejection_rows,
            True,
        )


# =============================================================================
# REPORT EXTENSION
# =============================================================================

def write_hierarchical_report(
    output_dir: Path,
    state: Mapping[str, Any],
    competitions_seen: int,
    scopes_seen: int,
    listed_matches: int,
    skipped_done: int,
    collected_new: int,
    client: Any,
) -> None:
    path = (
        output_dir
        / "historical_dataset_v2_hierarchical_report.md"
    )

    lines = [
        "# Historical Dataset V2 — Hierarchical Collection",
        "",
        "## Résumé de la dernière exécution",
        "",
        f"- Compétitions parcourues : **{competitions_seen}**",
        f"- Scopes compétition/saison parcourus : **{scopes_seen}**",
        f"- Matchs listés : **{listed_matches}**",
        f"- Matchs déjà `done` ignorés : **{skipped_done}**",
        f"- Nouveaux matchs traités : **{collected_new}**",
        f"- Requêtes API réelles : **{getattr(client, 'request_count', 0)}**",
        f"- HTTP 429 retentés : **{getattr(client, 'retry_429_count', 0)}**",
        "",
        "## Cache",
        "",
        f"- Cache hits : **{getattr(client, 'cache_hits', 0)}**",
        f"- Cache misses : **{getattr(client, 'cache_misses', 0)}**",
        f"- Cache writes : **{getattr(client, 'cache_writes', 0)}**",
        "",
        "## Principe",
        "",
        "Ce collecteur explore l'historique par compétition et saison, au lieu de se limiter "
        "aux pages les plus récentes de `/football/matches`.",
        "",
        "Le scanner de production n'a pas été modifié.",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collecte historique hiérarchique V2 : compétitions → saisons → matchs."
        )
    )

    parser.add_argument(
        "--limit-new",
        type=int,
        default=500,
        help="Nombre maximal de nouveaux matchs à traiter. Défaut: 500.",
    )

    parser.add_argument(
        "--competition-ids",
        nargs="*",
        default=None,
        help="Liste de competition_id à cibler. Si absent, découverte API.",
    )

    parser.add_argument(
        "--season-ids",
        nargs="*",
        default=None,
        help="Liste de season_id à cibler. Si absent, découverte API par compétition.",
    )

    parser.add_argument(
        "--max-competitions",
        type=int,
        default=50,
        help="Nombre max de compétitions à découvrir. Défaut: 50.",
    )

    parser.add_argument(
        "--max-seasons-per-competition",
        type=int,
        default=5,
        help="Nombre max de saisons par compétition. Défaut: 5.",
    )

    parser.add_argument(
        "--max-pages-competitions",
        type=int,
        default=10,
        help="Pages max pour lister les compétitions. Défaut: 10.",
    )

    parser.add_argument(
        "--max-pages-seasons",
        type=int,
        default=5,
        help="Pages max pour lister les saisons. Défaut: 5.",
    )

    parser.add_argument(
        "--max-pages-per-season",
        type=int,
        default=20,
        help="Pages max de matchs par scope compétition/saison. Défaut: 20.",
    )

    parser.add_argument(
        "--date-from",
        default=None,
        help="Filtre date début si supporté par l'API.",
    )

    parser.add_argument(
        "--date-to",
        default=None,
        help="Filtre date fin si supporté par l'API.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Chemin vers model_schema_v2.json.",
    )

    parser.add_argument(
        "--xg-limit",
        type=int,
        default=5,
        help="Nombre de matchs récents par équipe pour xG. Défaut: 5. 0 = désactivé.",
    )

    parser.add_argument(
        "--with-league-second-half",
        action="store_true",
        help="Inclure league_2nd_half_over15_freq. Plus coûteux.",
    )

    parser.add_argument(
        "--preferred-bookmaker",
        default=None,
        help="Bookmaker préféré pour les cotes 1X2.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignorer les matchs déjà done avec le même schema_version.",
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
        default=0.5,
        help="Pause entre pages/scopes. Défaut: 0.5s.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Sauvegarde tous les N matchs traités. Défaut: 1.",
    )

    parser.add_argument(
        "--ignore-completed-scopes",
        action="store_true",
        help="Ignore les scopes compétition/saison déjà marqués terminés dans le state.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Dossier du cache API.",
    )

    parser.add_argument(
        "--cache-disable",
        action="store_true",
        help="Désactive le cache.",
    )

    parser.add_argument(
        "--cache-clear",
        action="store_true",
        help="Vide le cache avant exécution.",
    )

    parser.add_argument(
        "--cache-save-none",
        action="store_true",
        help="Cache aussi les réponses None. Désactivé par défaut.",
    )

    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=0.0,
        help="TTL cache en heures. 0 = pas d'expiration.",
    )

    args = parser.parse_args()

    output_dir = path_resolve(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    cache_dir = path_resolve(
        args.cache_dir,
        DEFAULT_CACHE_DIR,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.cache_clear and cache_dir.exists():
        print(f"Suppression du cache : {cache_dir}")
        shutil.rmtree(
            cache_dir,
            ignore_errors=True,
        )

    load_dotenv(
        BASE_DIR / ".env"
        if (BASE_DIR / ".env").exists()
        else None
    )

    print()
    print("=" * 100)
    print("COLLECT HISTORICAL DATASET V2 — HIÉRARCHIQUE")
    print("=" * 100)

    collector = load_original_collector()

    print(f"Collecteur original : {ORIGINAL_COLLECTOR_PATH.name}")
    print(f"Schema version : {collector.SCHEMA_VERSION}")

    prematch, live87, client = prepare_runtime_modules(
        collector=collector,
        cache_dir=cache_dir,
        cache_enabled=not args.cache_disable,
        cache_save_none=args.cache_save_none,
        cache_ttl_hours=args.cache_ttl_hours,
    )

    print(f"Module PRE : {collector.PREMATCH_MODULE_PATH.name}")
    print(f"Module LIVE87 : {collector.LIVE87_MODULE_PATH.name}")
    print(f"Cache activé : {not args.cache_disable}")
    print(f"Cache dir : {cache_dir}")

    core_features = collector.load_core_features(
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

    rejections_path = (
        output_dir
        / "historical_dataset_v2_rejections.csv"
    )

    state_path = (
        output_dir
        / STATE_FILENAME
    )

    index_df = collector.load_index(
        index_path
    )

    dataset_df = collector.load_existing_dataset(
        raw_dataset_path
    )

    rejection_rows = load_rejections(
        rejections_path,
        force=args.force,
    )

    state = load_state(
        state_path
    )

    print(f"Index existant : {len(index_df)} lignes")
    print(f"Dataset existant : {len(dataset_df)} lignes")
    print(f"CORE features : {len(core_features)}")

    # -------------------------------------------------------------------------
    # Discover competitions
    # -------------------------------------------------------------------------

    if args.competition_ids:
        competitions = build_manual_competitions(
            args.competition_ids
        )
    else:
        print()
        print("Découverte des compétitions...")

        competitions = list_competitions(
            client=client,
            limit=max(0, args.max_competitions),
            max_pages=max(1, args.max_pages_competitions),
            page_sleep=max(0.0, args.sleep_between_pages),
        )

    print(f"Compétitions à parcourir : {len(competitions)}")

    if not competitions:
        print("Aucune compétition trouvée.")
        return 3

    collected_new = 0
    skipped_done = 0
    listed_matches = 0
    scopes_seen = 0
    competitions_seen = 0

    try:
        for comp_index, competition in enumerate(
            competitions,
            start=1,
        ):
            if (
                args.limit_new > 0
                and collected_new >= args.limit_new
            ):
                break

            competition_id = (
                clean_text(competition.get("id"))
                or clean_text(competition.get("competition_id"))
            )

            competition_name = extract_name(
                competition
            )

            if not competition_id:
                continue

            competitions_seen += 1

            print()
            print("=" * 100)
            print(
                f"COMPÉTITION [{comp_index}/{len(competitions)}] "
                f"{competition_id} — {competition_name}"
            )
            print("=" * 100)

            # -----------------------------------------------------------------
            # Seasons
            # -----------------------------------------------------------------

            if args.season_ids:
                seasons = build_manual_seasons(
                    args.season_ids
                )

            else:
                seasons = list_seasons_for_competition(
                    client=client,
                    competition_id=competition_id,
                    max_pages=max(1, args.max_pages_seasons),
                    page_sleep=max(0.0, args.sleep_between_pages),
                )

                if args.max_seasons_per_competition > 0:
                    seasons = seasons[
                        : args.max_seasons_per_competition
                    ]

            # Fallback if seasons endpoint is unavailable.
            if not seasons:
                seasons = [
                    {
                        "id": "",
                        "name": "NO_SEASON_SCOPE",
                    }
                ]

            print(f"Saisons/scopes : {len(seasons)}")

            for season_index, season in enumerate(
                seasons,
                start=1,
            ):
                if (
                    args.limit_new > 0
                    and collected_new >= args.limit_new
                ):
                    break

                season_id = (
                    clean_text(season.get("id"))
                    or clean_text(season.get("season_id"))
                )

                season_name = extract_name(
                    season
                )

                if should_skip_scope(
                    state=state,
                    competition_id=competition_id,
                    season_id=season_id,
                    ignore_completed_scopes=args.ignore_completed_scopes,
                ):
                    print(
                        f"Scope déjà terminé ignoré : "
                        f"{competition_id} / {season_id}"
                    )
                    continue

                scopes_seen += 1

                print()
                print("-" * 100)
                print(
                    f"Scope saison [{season_index}/{len(seasons)}] "
                    f"competition={competition_id} "
                    f"season={season_id or 'ALL'} "
                    f"{season_name}"
                )
                print("-" * 100)

                match_stubs = list_finished_matches_for_scope(
                    client=client,
                    competition_id=competition_id,
                    season_id=season_id or None,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    max_pages=max(1, args.max_pages_per_season),
                    page_sleep=max(0.0, args.sleep_between_pages),
                )

                listed_matches += len(match_stubs)

                print(
                    f"Matchs terminés listés : {len(match_stubs)}"
                )

                if not match_stubs:
                    mark_scope_completed(
                        state,
                        competition_id,
                        season_id,
                    )

                    save_state(
                        state_path,
                        state,
                    )

                    continue

                processed_in_scope = 0

                for stub in match_stubs:
                    if (
                        args.limit_new > 0
                        and collected_new >= args.limit_new
                    ):
                        break

                    match_id = extract_match_id(
                        stub
                    )

                    if not match_id:
                        continue

                    if collector.already_done(
                        index_df=index_df,
                        match_id=match_id,
                        force=args.force,
                    ):
                        skipped_done += 1
                        continue

                    print()
                    print(
                        f"[NEW {collected_new + 1}"
                        f"/{args.limit_new if args.limit_new > 0 else '∞'}] "
                        f"{match_id}"
                    )

                    (
                        index_df,
                        dataset_df,
                        rejection_rows,
                        was_processed,
                    ) = collect_match_stub(
                        collector=collector,
                        prematch=prematch,
                        live87=live87,
                        client=client,
                        stub=stub,
                        core_features=core_features,
                        index_df=index_df,
                        dataset_df=dataset_df,
                        rejection_rows=rejection_rows,
                        output_dir=output_dir,
                        force=args.force,
                        xg_limit=max(0, args.xg_limit),
                        with_league_second_half=args.with_league_second_half,
                        preferred_bookmaker=args.preferred_bookmaker,
                    )

                    if was_processed:
                        collected_new += 1
                        processed_in_scope += 1

                    if (
                        args.save_every > 0
                        and collected_new % args.save_every == 0
                    ):
                        collector.save_all(
                            dataset_df=dataset_df,
                            index_df=index_df,
                            rejection_rows=rejection_rows,
                            output_dir=output_dir,
                            core_features=core_features,
                        )

                        save_state(
                            state_path,
                            state,
                        )

                    if args.sleep_between_matches > 0:
                        time.sleep(
                            args.sleep_between_matches
                        )

                mark_scope_completed(
                    state,
                    competition_id,
                    season_id,
                )

                save_state(
                    state_path,
                    state,
                )

                print(
                    f"Scope terminé : nouveaux traités={processed_in_scope}"
                )

                if args.sleep_between_pages > 0:
                    time.sleep(
                        args.sleep_between_pages
                    )

    except KeyboardInterrupt:
        print()
        print("Interruption utilisateur. Sauvegarde en cours...")

        collector.save_all(
            dataset_df=dataset_df,
            index_df=index_df,
            rejection_rows=rejection_rows,
            output_dir=output_dir,
            core_features=core_features,
        )

        save_state(
            state_path,
            state,
        )

        client.print_stats()

        return 130

    # Final save/report
    collector.save_all(
        dataset_df=dataset_df,
        index_df=index_df,
        rejection_rows=rejection_rows,
        output_dir=output_dir,
        core_features=core_features,
    )

    collector.write_report(
        dataset_df=dataset_df,
        index_df=index_df,
        rejection_rows=rejection_rows,
        core_features=core_features,
        output_dir=output_dir,
        client=client,
        args=args,
    )

    state.setdefault(
        "runs",
        []
    ).append({
        "finished_at_utc": now_utc_iso(),
        "competitions_seen": competitions_seen,
        "scopes_seen": scopes_seen,
        "listed_matches": listed_matches,
        "skipped_done": skipped_done,
        "collected_new": collected_new,
        "request_count": getattr(client, "request_count", 0),
        "retry_429_count": getattr(client, "retry_429_count", 0),
        "cache_hits": getattr(client, "cache_hits", 0),
        "cache_misses": getattr(client, "cache_misses", 0),
    })

    save_state(
        state_path,
        state,
    )

    write_hierarchical_report(
        output_dir=output_dir,
        state=state,
        competitions_seen=competitions_seen,
        scopes_seen=scopes_seen,
        listed_matches=listed_matches,
        skipped_done=skipped_done,
        collected_new=collected_new,
        client=client,
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
                    dataset_df[collector.TARGET_COLUMN],
                    errors="coerce",
                )
                == 1
            ).sum()
        )

        negatives = int(
            (
                pd.to_numeric(
                    dataset_df[collector.TARGET_COLUMN],
                    errors="coerce",
                )
                == 0
            ).sum()
        )

        target_rate = (
            positives / (positives + negatives)
            if positives + negatives
            else 0.0
        )

    else:
        core_complete = 0
        positives = 0
        negatives = 0
        target_rate = 0.0

    print()
    print("=" * 100)
    print("COLLECTE HIÉRARCHIQUE TERMINÉE")
    print("=" * 100)
    print(f"Dataset RAW total : {len(dataset_df)}")
    print(f"CORE complet : {core_complete}")
    print(f"Positifs : {positives}")
    print(f"Négatifs : {negatives}")
    print(f"Target rate : {target_rate:.2%}")
    print(f"Index total : {len(index_df)}")
    print(f"Rejets / partiels : {len(rejection_rows)}")
    print(f"Compétitions parcourues : {competitions_seen}")
    print(f"Scopes parcourus : {scopes_seen}")
    print(f"Matchs listés : {listed_matches}")
    print(f"Déjà done ignorés : {skipped_done}")
    print(f"Nouveaux traités : {collected_new}")
    print(f"Requêtes API : {getattr(client, 'request_count', 0)}")
    print(f"HTTP 429 retentés : {getattr(client, 'retry_429_count', 0)}")
    print()
    print("Sorties :")
    print(f"  {output_dir / 'Historical_Dataset_Goal_After_87_RAW_v2.csv'}")
    print(f"  {output_dir / 'Historical_Dataset_Goal_After_87_CORE_READY_v2.csv'}")
    print(f"  {output_dir / 'historical_matches_index_v2.csv'}")
    print(f"  {output_dir / 'historical_dataset_v2_feature_coverage.csv'}")
    print(f"  {output_dir / 'historical_dataset_v2_report.md'}")
    print(f"  {output_dir / 'historical_dataset_v2_hierarchical_report.md'}")
    print(f"  {state_path}")

    client.print_stats()

    print()
    print("Le scanner de production n'a pas été modifié.")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
