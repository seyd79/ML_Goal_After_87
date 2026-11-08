#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_first_half_37_historical_api.py

BACKFILL HISTORIQUE DIRECT API — FIRST HALF 37'
================================================

OBJECTIF
--------
Agrandir le dataset First Half 37 au-delà des matchs déjà présents dans
`Historical_Dataset_Goal_After_87_RAW_v2.csv`.

Ce collecteur :

1) prend par défaut les compétitions déjà connues dans le dataset First Half 37 ;
2) découvre directement dans TheStatsAPI des matchs terminés PLUS ANCIENS ;
3) reconstruit les 59 features PRE-MATCH strictement AS-OF kickoff ;
4) reconstruit l'état LIVE à 37' via timeline + shotmap ;
5) utilise uniquement les opening odds ;
6) résout le target : but APRES 37' et AVANT la fin de la 1re mi-temps ;
7) rejette toute timeline insuffisante au lieu d'inventer des valeurs ;
8) écrit un backfill séparé puis un dataset combiné dédupliqué ;
9) utilise le même cache disque que le collecteur First Half 37 ;
10) sauvegarde un index et un state pour reprendre après interruption.

IMPORTANT — ANTI-FUITE
----------------------
- Les features LIVE ne voient jamais un événement > 37'.
- Les buts > 37' servent uniquement au target.
- Les données PRE sont calculées uniquement avec des matchs antérieurs au kickoff.
- Pour les historiques équipe/ligue, le client injecte `date_to=<kickoff>` afin
  d'éviter que la page API soit remplie de matchs futurs par rapport au match cible.
- Les caches mémoire PRE dépendant implicitement du cutoff sont vidés entre matchs.
- Les stats finales ne sont jamais utilisées pour backcaster l'état à 37'.
- Missing != 0. Aucun remplissage silencieux.

POURQUOI UN FICHIER SEPARE
--------------------------
Le collecteur existant `collect_first_half_37_dataset.py` part d'une liste locale
préexistante. Ce nouveau fichier ne le remplace pas : il s'en sert comme moteur de
reconstruction validé, mais découvre de NOUVEAUX match_id directement dans l'API.

DEPENDANCES ATTENDUES A LA RACINE DU PROJET
-------------------------------------------
- collect_first_half_37_dataset.py
- rebuild_clean_prematch_v2.py
- audit_historical_live87_reconstructability.py
- pre_match_feature_engineering_fidelity_lab*.py (déjà utilisé par le projet)
- .env avec STATS_API_KEY

SORTIES
-------
outputs/first_half_37/historical_api/
├── First_Half_37_API_BACKFILL.csv
├── First_Half_37_COMBINED.csv
├── historical_api_index.csv
├── historical_api_rejections.csv
├── historical_api_state.json
├── historical_api_feature_coverage.csv
└── historical_api_report.md

Le dataset initial n'est PAS écrasé.

UTILISATION RECOMMANDEE
-----------------------
Test court, 50 tentatives :

    python3 collect_first_half_37_historical_api.py \
        --max-new-attempts 50 \
        --target-new-rows 25 \
        --resume

Backfill principal, en visant 3 000 nouvelles lignes :

    caffeinate -dimsu python3 collect_first_half_37_historical_api.py \
        --target-new-rows 3000 \
        --max-new-attempts 6500 \
        --lookback-days 1100 \
        --window-days 180 \
        --max-api-requests 60000 \
        --resume

Par défaut, les xG PRE historiques sont désactivés (`--xg-limit 0`) pour éviter
une explosion du quota API. Ils pourront être enrichis plus tard si les expériences
montrent qu'ils apportent un vrai gain hors échantillon.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

FIRST_HALF_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
PREMATCH_MODULE_PATH = BASE_DIR / "rebuild_clean_prematch_v2.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

DEFAULT_FIRST_HALF_DIR = BASE_DIR / "outputs" / "first_half_37"
DEFAULT_SEED_CSV = DEFAULT_FIRST_HALF_DIR / "datasets" / "First_Half_37_RAW.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_FIRST_HALF_DIR / "historical_api"
DEFAULT_SHARED_CACHE_DIR = DEFAULT_FIRST_HALF_DIR / "cache" / "api"
DEFAULT_LEGACY_CACHE_DIR = BASE_DIR / "outputs" / "historical_dataset_v2" / "api_cache_v2"

BACKFILL_FILENAME = "First_Half_37_API_BACKFILL.csv"
COMBINED_FILENAME = "First_Half_37_COMBINED.csv"
INDEX_FILENAME = "historical_api_index.csv"
REJECTIONS_FILENAME = "historical_api_rejections.csv"
STATE_FILENAME = "historical_api_state.json"
COVERAGE_FILENAME = "historical_api_feature_coverage.csv"
REPORT_FILENAME = "historical_api_report.md"

SCHEMA_VERSION = "first_half_37_historical_api_v1"
TARGET_COLUMN = "goal_after_37_before_ht"

DEFAULT_MIN_API_INTERVAL = 0.75
DEFAULT_LOOKBACK_DAYS = 1100
DEFAULT_WINDOW_DAYS = 180
DEFAULT_MAX_API_REQUESTS = 60000


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
    value = safe_float(value)

    if value is None:
        return None

    return int(value)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_date(value: Any) -> Optional[date]:
    text = clean_text(value)

    if not text:
        return None

    # Accepte YYYY-MM-DD ou timestamp ISO.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        pass

    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def date_str(value: date) -> str:
    return value.isoformat()


def resolve_path(raw: Optional[str], default: Path) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path.resolve()

    return default.resolve()


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Module requis introuvable : {path}\n"
            "Place ce fichier à la racine du projet avec les autres scripts."
        )

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger : {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_list_from_payload(payload: Any) -> List[Dict[str, Any]]:
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
        for key in (
            "items",
            "matches",
            "competitions",
            "seasons",
            "results",
            "data",
        ):
            values = data.get(key)
            if isinstance(values, list):
                return [dict(x) for x in values if isinstance(x, Mapping)]

    return []


def extract_match_id(match: Mapping[str, Any]) -> str:
    for key in ("match_id", "id", "event_id"):
        value = clean_text(match.get(key))
        if value:
            return value
    return ""


def extract_competition_id(match: Mapping[str, Any]) -> str:
    for key in ("competition_id", "league_id", "tournament_id"):
        value = clean_text(match.get(key))
        if value:
            return value

    competition = match.get("competition")
    if isinstance(competition, Mapping):
        return clean_text(
            competition.get("id") or competition.get("competition_id")
        )

    return ""


def extract_season_id(match: Mapping[str, Any]) -> str:
    value = match.get("season_id")
    if value is not None:
        return clean_text(value)

    season = match.get("season")
    if isinstance(season, Mapping):
        return clean_text(season.get("id") or season.get("season_id"))

    if season is not None:
        return clean_text(season)

    return ""


def atomic_write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def append_or_replace_by_match_id(
    df: pd.DataFrame,
    row: Mapping[str, Any],
) -> pd.DataFrame:
    new = pd.DataFrame([dict(row)])

    if df.empty:
        return new

    match_id = clean_text(row.get("match_id"))

    if match_id and "match_id" in df.columns:
        df = df[df["match_id"].astype(str) != match_id].copy()

    return pd.concat([df, new], ignore_index=True, sort=False)


def merge_seed_and_backfill(seed_df: pd.DataFrame, backfill_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty and backfill_df.empty:
        return pd.DataFrame()

    combined = pd.concat(
        [seed_df.copy(), backfill_df.copy()],
        ignore_index=True,
        sort=False,
    )

    if "match_id" in combined.columns:
        combined["match_id"] = combined["match_id"].astype(str)
        combined = combined.drop_duplicates(subset=["match_id"], keep="last")

    if "kickoff_utc" in combined.columns:
        sort_key = pd.to_datetime(combined["kickoff_utc"], errors="coerce", utc=True)
        combined = combined.assign(_sort_kickoff=sort_key)
        combined = combined.sort_values(
            ["_sort_kickoff", "match_id"],
            ascending=[False, True],
            na_position="last",
        ).drop(columns=["_sort_kickoff"])

    return combined.reset_index(drop=True)


# =============================================================================
# STATE
# =============================================================================

def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "completed_windows": [],
            "created_at_utc": now_utc_iso(),
        }

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        value = {}

    if not isinstance(value, dict):
        value = {}

    value.setdefault("schema_version", SCHEMA_VERSION)
    value.setdefault("completed_windows", [])
    value.setdefault("created_at_utc", now_utc_iso())
    return value


def window_key(competition_id: str, start: date, end: date) -> str:
    return f"{competition_id}::{start.isoformat()}::{end.isoformat()}"


def mark_window_completed(
    state: Dict[str, Any],
    competition_id: str,
    start: date,
    end: date,
) -> None:
    key = window_key(competition_id, start, end)
    completed = list(state.get("completed_windows", []))

    if key not in completed:
        completed.append(key)

    state["completed_windows"] = completed
    state["updated_at_utc"] = now_utc_iso()


# =============================================================================
# API ADAPTER
# =============================================================================

class UnifiedHistoricalClient:
    """
    Adapte le CachedHistoricalClient du collecteur 37' pour deux interfaces :

      - request(endpoint, params) -> (payload, status, error)
      - get(endpoint, params) -> payload

    `rebuild_clean_prematch_v2.py` utilise `.get()`.
    `collect_first_half_37_dataset.py` utilise `.request()`.

    Pendant la reconstruction PRE d'un match, `current_cutoff_date` est injecté
    dans les listings `/football/matches` d'historique équipe/ligue.
    """

    def __init__(self, cached_client: Any):
        self.cached_client = cached_client
        self.current_cutoff_date: Optional[str] = None

    @property
    def request_count(self) -> int:
        return int(getattr(self.cached_client, "request_count", 0))

    @property
    def retry_429_count(self) -> int:
        return int(getattr(self.cached_client, "retry_429_count", 0))

    @property
    def cache_hits(self) -> int:
        return int(getattr(self.cached_client, "cache_hits", 0))

    @property
    def legacy_cache_hits(self) -> int:
        return int(getattr(self.cached_client, "legacy_cache_hits", 0))

    @property
    def cache_misses(self) -> int:
        return int(getattr(self.cached_client, "cache_misses", 0))

    @property
    def cache_writes(self) -> int:
        return int(getattr(self.cached_client, "cache_writes", 0))

    def set_cutoff(self, cutoff: Optional[datetime]) -> None:
        self.current_cutoff_date = (
            cutoff.date().isoformat()
            if cutoff is not None
            else None
        )

    def clear_cutoff(self) -> None:
        self.current_cutoff_date = None

    def _params_with_cutoff(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        result = dict(params or {})
        ep = "/" + endpoint.lstrip("/")

        if (
            self.current_cutoff_date
            and ep == "/football/matches"
            and clean_text(result.get("status")).lower() == "finished"
            and (
                clean_text(result.get("team_id"))
                or clean_text(result.get("competition_id"))
            )
        ):
            # On ne remplace jamais une borne plus restrictive explicitement passée.
            result.setdefault("date_to", self.current_cutoff_date)

        return result

    def request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[int], str]:
        # Important : l'interface request() est utilisée pour detail/timeline/
        # shotmap/odds du match cible. On n'y injecte PAS le cutoff.
        return self.cached_client.request(endpoint, params=params)

    def get(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        adjusted = self._params_with_cutoff(endpoint, params)
        payload, _status, _error = self.cached_client.request(
            endpoint,
            params=adjusted,
        )
        return payload


# =============================================================================
# PRE-MATCH RECONSTRUCTION SAFETY
# =============================================================================

def clear_cutoff_sensitive_prematch_memory_caches(prematch: Any) -> None:
    """
    `rebuild_clean_prematch_v2.py` possède des caches mémoire historiques dont
    certaines clés n'incluent pas le cutoff temporel. Ils sont sûrs dans un run
    centré sur une période récente, mais pas pour un backfill multi-années.

    On vide uniquement les caches dépendants du cutoff avant chaque cible.
    Le cache DISQUE persistant, lui, reste actif et est correctement clé par
    endpoint + paramètres, y compris `date_to` injecté.
    """

    for name in ("TEAM_HISTORY_CACHE", "LEAGUE_HISTORY_CACHE"):
        cache = getattr(prematch, name, None)
        if isinstance(cache, dict):
            cache.clear()


def patch_paginated_league_history(
    prematch: Any,
    max_pages: int,
) -> None:
    """
    Remplace `get_league_history` par une version paginée.

    Raison : reconstruire un classement as-of kickoff avec une seule page de
    100 matchs peut tronquer une saison de grande ligue. La version ci-dessous
    récupère toutes les pages disponibles jusqu'à `max_pages`.
    """

    def get_league_history_paginated(
        lab: Any,
        client: Any,
        competition_id: str,
        season_id: Optional[str],
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for page in range(1, max(1, max_pages) + 1):
            params: Dict[str, Any] = {
                "competition_id": competition_id,
                "status": "finished",
                "page": page,
                "per_page": per_page,
            }

            if season_id:
                params["season_id"] = season_id

            payload = client.get("/football/matches", params=params)
            items = prematch.api_list(lab, payload)

            if not items:
                break

            added = 0

            for item in items:
                match_id = ""
                try:
                    match_id = clean_text(lab.extract_match_id(item))
                except Exception:
                    match_id = extract_match_id(item)

                # Les rares lignes sans id restent utilisables pour le tableau,
                # mais on ne peut pas les dédupliquer proprement.
                dedupe_key = match_id or json.dumps(item, sort_keys=True, default=str)

                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)
                rows.append(dict(item))
                added += 1

            if added == 0:
                break

            # Si l'API renvoie moins que per_page, il n'y a probablement plus de page.
            if len(items) < per_page:
                break

        return rows

    prematch.get_league_history = get_league_history_paginated


# =============================================================================
# COMPETITIONS / DATE WINDOWS / DISCOVERY
# =============================================================================

def load_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Dataset seed introuvable : {seed_path}\n"
            "Lance d'abord le collecteur First Half 37 ou passe --seed-csv."
        )

    df = pd.read_csv(seed_path, low_memory=False)

    if "match_id" not in df.columns:
        raise ValueError(f"Colonne match_id absente de {seed_path}")

    return df


def seed_competition_ids(seed_df: pd.DataFrame) -> List[str]:
    if "competition_id" not in seed_df.columns:
        return []

    values = [
        clean_text(x)
        for x in seed_df["competition_id"].dropna().tolist()
    ]

    return sorted({x for x in values if x})


def discover_api_competitions(client: UnifiedHistoricalClient) -> List[str]:
    client.clear_cutoff()
    payload = client.get("/football/competitions", params={"page": 1, "per_page": 500})
    items = as_list_from_payload(payload)

    ids: Set[str] = set()

    for item in items:
        competition_id = clean_text(
            item.get("id") or item.get("competition_id")
        )
        if competition_id:
            ids.add(competition_id)

    return sorted(ids)


def resolve_competitions(
    args: argparse.Namespace,
    seed_df: pd.DataFrame,
    client: UnifiedHistoricalClient,
) -> List[str]:
    manual = [clean_text(x) for x in (args.competition_ids or []) if clean_text(x)]

    if manual:
        values = sorted(set(manual))
    else:
        seed_ids = seed_competition_ids(seed_df)
        api_ids: List[str] = []

        if args.competition_source in ("api", "both"):
            api_ids = discover_api_competitions(client)

        if args.competition_source == "api":
            values = api_ids
        elif args.competition_source == "both":
            values = sorted(set(seed_ids).union(api_ids))
        else:
            values = seed_ids

    if args.shuffle_competitions:
        rng = random.Random(args.random_seed)
        rng.shuffle(values)

    if args.max_competitions > 0:
        values = values[: args.max_competitions]

    return values


def infer_default_date_to(seed_df: pd.DataFrame) -> date:
    """
    On cible par défaut l'historique antérieur au plus ancien match du seed.
    Cela maximise la diversité temporelle et évite de relister principalement
    les ~980 matchs déjà présents.
    """

    if "kickoff_utc" in seed_df.columns:
        parsed = pd.to_datetime(seed_df["kickoff_utc"], errors="coerce", utc=True)
        parsed = parsed.dropna()

        if len(parsed) > 0:
            oldest = parsed.min().date()
            return oldest - timedelta(days=1)

    return datetime.now(timezone.utc).date()


def build_date_windows(
    start: date,
    end: date,
    window_days: int,
    order: str,
) -> List[Tuple[date, date]]:
    if start > end:
        start, end = end, start

    days = max(1, int(window_days))
    windows: List[Tuple[date, date]] = []

    cursor = start

    while cursor <= end:
        window_end = min(cursor + timedelta(days=days - 1), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)

    if order == "newest":
        windows.reverse()

    return windows


def payload_total_pages(payload: Any) -> Optional[int]:
    if not isinstance(payload, Mapping):
        return None

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        return None

    return safe_int(meta.get("total_pages"))


def iter_finished_matches_window(
    client: UnifiedHistoricalClient,
    competition_id: str,
    start: date,
    end: date,
    max_pages: int,
    per_page: int,
    max_api_requests: int,
) -> Iterable[Tuple[Dict[str, Any], int, int]]:
    """
    Yields (match_stub, page, total_pages_or_0).
    """

    client.clear_cutoff()
    seen: Set[str] = set()

    for page in range(1, max(1, max_pages) + 1):
        if max_api_requests > 0 and client.request_count >= max_api_requests:
            return

        params = {
            "status": "finished",
            "competition_id": competition_id,
            "date_from": date_str(start),
            "date_to": date_str(end),
            "page": page,
            "per_page": per_page,
        }

        payload = client.get("/football/matches", params=params)
        items = as_list_from_payload(payload)
        total_pages = payload_total_pages(payload) or 0

        if not items:
            return

        added = 0

        for item in items:
            match_id = extract_match_id(item)
            if not match_id or match_id in seen:
                continue

            seen.add(match_id)
            added += 1
            yield dict(item), page, total_pages

        if added == 0:
            return

        if total_pages and page >= total_pages:
            return

        if len(items) < per_page:
            return


# =============================================================================
# MATCH BUILD
# =============================================================================

def target_detail_from_payload(prematch: Any, lab: Any, payload: Any) -> Dict[str, Any]:
    try:
        match = prematch.unwrap_match(lab, payload)
        if isinstance(match, Mapping):
            return dict(match)
    except Exception:
        pass

    if isinstance(payload, Mapping):
        data = payload.get("data", payload)
        if isinstance(data, Mapping):
            return dict(data)

    return {}


def build_source_row_for_strict_collector(
    lab: Any,
    target_match: Mapping[str, Any],
    stub: Mapping[str, Any],
    prematch_features: Mapping[str, Any],
) -> Dict[str, Any]:
    match_id = ""
    try:
        match_id = clean_text(lab.extract_match_id(target_match))
    except Exception:
        match_id = extract_match_id(target_match)

    if not match_id:
        match_id = extract_match_id(stub)

    try:
        home = lab.extract_home_team(target_match)
        home_name = clean_text(lab.extract_team_name(home))
    except Exception:
        home_name = ""

    try:
        away = lab.extract_away_team(target_match)
        away_name = clean_text(lab.extract_team_name(away))
    except Exception:
        away_name = ""

    try:
        competition_id = clean_text(lab.extract_competition_id(target_match))
    except Exception:
        competition_id = extract_competition_id(target_match)

    try:
        season_id = clean_text(lab.extract_season_id(target_match))
    except Exception:
        season_id = extract_season_id(target_match)

    try:
        kickoff = lab.extract_match_date(target_match)
    except Exception:
        kickoff = target_match.get("utc_date") or target_match.get("kickoff_utc")

    row: Dict[str, Any] = {
        "match_id": match_id,
        "competition_id": competition_id or extract_competition_id(stub),
        "season_id": season_id or extract_season_id(stub),
        "home_team": home_name or clean_text(stub.get("home_team")),
        "away_team": away_name or clean_text(stub.get("away_team")),
        "kickoff_utc": kickoff or stub.get("utc_date") or stub.get("kickoff_utc") or "",
    }

    row.update(dict(prematch_features))
    return row


def rebuild_one_new_match(
    stub: Mapping[str, Any],
    client: UnifiedHistoricalClient,
    first_half: Any,
    prematch: Any,
    audit: Any,
    lab: Any,
    xg_limit: int,
    with_league_second_half: bool,
    preferred_bookmaker: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    match_id = extract_match_id(stub)

    base_audit: Dict[str, Any] = {
        "match_id": match_id,
        "competition_id": extract_competition_id(stub),
        "season_id": extract_season_id(stub),
        "status": "",
        "error": "",
        "collected_at_utc": now_utc_iso(),
    }

    if not match_id:
        base_audit.update({"status": "failed", "error": "missing_match_id"})
        return None, base_audit

    # ------------------------------------------------------------------
    # 1) Match detail / kickoff
    # ------------------------------------------------------------------
    client.clear_cutoff()
    detail_payload = client.get(f"/football/matches/{match_id}")
    target_match = target_detail_from_payload(prematch, lab, detail_payload)

    if not target_match:
        base_audit.update({
            "status": "failed",
            "error": "target_match_detail_unavailable",
        })
        return None, base_audit

    try:
        kickoff_value = lab.extract_match_date(target_match)
    except Exception:
        kickoff_value = target_match.get("utc_date") or target_match.get("kickoff_utc")

    cutoff = prematch.dt_utc_naive(kickoff_value)

    if cutoff is None:
        base_audit.update({
            "status": "failed",
            "error": "target_kickoff_unparseable",
        })
        return None, base_audit

    # ------------------------------------------------------------------
    # 2) PRE as-of kickoff
    # ------------------------------------------------------------------
    clear_cutoff_sensitive_prematch_memory_caches(prematch)
    client.set_cutoff(cutoff)

    try:
        pre_features, pre_audit = prematch.build_prematch_asof(
            lab=lab,
            client=client,
            target_match=target_match,
            cutoff=cutoff,
            xg_limit=max(0, int(xg_limit)),
            with_second_half=bool(with_league_second_half),
            preferred_bookmaker=preferred_bookmaker,
        )
    except Exception as exc:
        client.clear_cutoff()
        base_audit.update({
            "status": "failed",
            "error": f"prematch_rebuild_failed:{type(exc).__name__}:{exc}",
        })
        return None, base_audit
    finally:
        client.clear_cutoff()

    source_row = build_source_row_for_strict_collector(
        lab=lab,
        target_match=target_match,
        stub=stub,
        prematch_features=pre_features,
    )

    # ------------------------------------------------------------------
    # 3) LIVE37 + FH odds + strict target
    # ------------------------------------------------------------------
    row, live_audit = first_half.collect_one_match(
        source_row=source_row,
        client=client,
        audit=audit,
        prematch_features=list(prematch.PRE_MATCH_FEATURES),
    )

    merged_audit = dict(base_audit)
    merged_audit.update(dict(live_audit or {}))
    merged_audit["prematch_available"] = pre_audit.get("prematch_available", 0)
    merged_audit["prematch_missing"] = pre_audit.get("prematch_missing", "")
    merged_audit["prematch_home_history_before"] = pre_audit.get("home_history_before", 0)
    merged_audit["prematch_away_history_before"] = pre_audit.get("away_history_before", 0)
    merged_audit["prematch_league_history_before"] = pre_audit.get("league_history_before", 0)
    merged_audit["prematch_h2h_before"] = pre_audit.get("h2h_before", 0)
    merged_audit["prematch_cutoff_utc"] = cutoff.isoformat() + "Z"

    if row is None:
        return None, merged_audit

    row["schema_version"] = SCHEMA_VERSION
    row["source_dataset"] = "thestatsapi_direct_historical_backfill"
    row["prematch_rebuild_policy"] = "strict_asof_kickoff_date_to_injected"
    row["prematch_cutoff_utc"] = cutoff.isoformat() + "Z"
    row["_prematch_home_history_before"] = pre_audit.get("home_history_before", 0)
    row["_prematch_away_history_before"] = pre_audit.get("away_history_before", 0)
    row["_prematch_league_history_before"] = pre_audit.get("league_history_before", 0)
    row["_prematch_h2h_before"] = pre_audit.get("h2h_before", 0)

    return row, merged_audit


# =============================================================================
# COVERAGE / REPORT
# =============================================================================

def build_coverage(
    first_half: Any,
    combined_df: pd.DataFrame,
    prematch_features: Sequence[str],
) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame(columns=["feature", "group", "available", "rows", "coverage_pct"])

    try:
        return first_half.build_feature_coverage(combined_df, prematch_features)
    except Exception:
        rows: List[Dict[str, Any]] = []

        for feature in prematch_features:
            available = int(pd.to_numeric(combined_df.get(feature), errors="coerce").notna().sum()) if feature in combined_df.columns else 0
            rows.append({
                "feature": feature,
                "group": "PRE_MATCH",
                "available": available,
                "rows": len(combined_df),
                "coverage_pct": (100.0 * available / len(combined_df)) if len(combined_df) else 0.0,
            })

        return pd.DataFrame(rows)


def date_range_text(df: pd.DataFrame) -> str:
    if df.empty or "kickoff_utc" not in df.columns:
        return "n/a"

    values = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True).dropna()

    if len(values) == 0:
        return "n/a"

    return f"{values.min().date().isoformat()} -> {values.max().date().isoformat()}"


def positive_summary(df: pd.DataFrame) -> Tuple[int, int, float]:
    if df.empty or TARGET_COLUMN not in df.columns:
        return 0, 0, 0.0

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    valid = target.isin([0, 1])
    n = int(valid.sum())
    positives = int((target[valid] == 1).sum())
    rate = (100.0 * positives / n) if n else 0.0
    return positives, n, rate


def write_report(
    path: Path,
    seed_path: Path,
    seed_df: pd.DataFrame,
    backfill_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    index_df: pd.DataFrame,
    client: UnifiedHistoricalClient,
    args: argparse.Namespace,
    competition_count: int,
    window_count: int,
) -> None:
    seed_pos, seed_n, seed_rate = positive_summary(seed_df)
    back_pos, back_n, back_rate = positive_summary(backfill_df)
    comb_pos, comb_n, comb_rate = positive_summary(combined_df)

    statuses: Dict[str, int] = {}
    if not index_df.empty and "status" in index_df.columns:
        statuses = {
            str(k): int(v)
            for k, v in index_df["status"].fillna("").value_counts().to_dict().items()
        }

    lines = [
        "# First Half 37 — Historical API Backfill Report",
        "",
        f"- Généré : `{now_utc_iso()}`",
        f"- Schéma : `{SCHEMA_VERSION}`",
        f"- Seed : `{seed_path}`",
        f"- Compétitions ciblées : **{competition_count}**",
        f"- Fenêtres date prévues : **{window_count}**",
        f"- Lookback : **{args.lookback_days} jours**",
        f"- Fenêtre : **{args.window_days} jours**",
        f"- xG PRE limit : **{args.xg_limit}**",
        "",
        "## Taille / target",
        "",
        f"- Seed : **{len(seed_df)}** lignes | target résolu={seed_n} | positifs={seed_pos} ({seed_rate:.2f}%) | dates {date_range_text(seed_df)}",
        f"- API backfill : **{len(backfill_df)}** lignes | target résolu={back_n} | positifs={back_pos} ({back_rate:.2f}%) | dates {date_range_text(backfill_df)}",
        f"- COMBINED : **{len(combined_df)}** lignes | target résolu={comb_n} | positifs={comb_pos} ({comb_rate:.2f}%) | dates {date_range_text(combined_df)}",
        "",
        "## API / cache",
        "",
        f"- Requêtes API réelles : **{client.request_count}**",
        f"- HTTP 429 : **{client.retry_429_count}**",
        f"- Cache hits First Half 37 : **{client.cache_hits}**",
        f"- Cache hits legacy : **{client.legacy_cache_hits}**",
        f"- Cache misses : **{client.cache_misses}**",
        f"- Cache writes : **{client.cache_writes}**",
        "",
        "## Index",
        "",
        f"- Matchs indexés : **{len(index_df)}**",
        f"- Statuts : `{json.dumps(statuses, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Règles anti-fuite",
        "",
        "- PRE reconstruit uniquement as-of kickoff.",
        "- `date_to=<date kickoff>` injecté dans les historiques équipe/ligue.",
        "- Les caches mémoire PRE dépendants du cutoff sont vidés entre matchs.",
        "- Les league histories sont paginés pour éviter un classement tronqué à 100 matchs.",
        "- LIVE features limitées à minute <= 37.",
        "- Les buts >37 servent uniquement au target.",
        "- Opening odds uniquement ; jamais last_seen.",
        "- Missing != 0.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Découvre directement dans TheStatsAPI des matchs historiques et "
            "construit un backfill First Half 37 strict sans fuite."
        )
    )

    parser.add_argument(
        "--seed-csv",
        default=None,
        help="Dataset First Half 37 existant. Défaut: outputs/first_half_37/datasets/First_Half_37_RAW.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Défaut: outputs/first_half_37/historical_api",
    )
    parser.add_argument(
        "--competition-source",
        choices=("seed", "api", "both"),
        default="seed",
        help="Source des competition_id. Défaut: seed (plus sûr).",
    )
    parser.add_argument(
        "--competition-ids",
        nargs="*",
        default=None,
        help="Liste manuelle de competition_id. Prioritaire sur --competition-source.",
    )
    parser.add_argument(
        "--max-competitions",
        type=int,
        default=0,
        help="Maximum de compétitions. 0 = toutes.",
    )
    parser.add_argument(
        "--shuffle-competitions",
        action="store_true",
        help="Mélange les compétitions avant collecte.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=87,
        help="Seed du mélange des compétitions.",
    )

    parser.add_argument(
        "--date-from",
        default=None,
        help="Date de début YYYY-MM-DD. Si absente: date_to - lookback-days.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help=(
            "Date de fin YYYY-MM-DD. Si absente: veille du plus ancien match du seed, "
            "afin de chercher d'abord de l'historique réellement plus ancien."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Profondeur historique si --date-from absent. Défaut: {DEFAULT_LOOKBACK_DAYS} jours.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Taille d'une fenêtre de découverte. Défaut: {DEFAULT_WINDOW_DAYS} jours.",
    )
    parser.add_argument(
        "--window-order",
        choices=("newest", "oldest"),
        default="newest",
        help="Ordre des fenêtres. Défaut: newest, mais toujours avant le seed.",
    )
    parser.add_argument(
        "--max-pages-per-window",
        type=int,
        default=30,
        help="Pages max de 100 matchs par compétition/fenêtre. Défaut: 30.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="Taille page API. Défaut: 100.",
    )

    parser.add_argument(
        "--target-new-rows",
        type=int,
        default=0,
        help="Arrête après N nouvelles lignes résolues. 0 = pas de cible.",
    )
    parser.add_argument(
        "--max-new-attempts",
        type=int,
        default=0,
        help="Arrête après N nouveaux matchs tentés. 0 = pas de limite.",
    )
    parser.add_argument(
        "--max-attempts-per-scope",
        type=int,
        default=0,
        help=(
            "Maximum de nouveaux matchs tentés par couple compétition/fenêtre. "
            "0 = illimité. Recommandé: 2 à 10 pour éviter qu'une seule compétition "
            "monopolise le backfill."
        ),
    )
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=DEFAULT_MAX_API_REQUESTS,
        help=f"Garde-fou requêtes API réelles. Défaut: {DEFAULT_MAX_API_REQUESTS}.",
    )

    parser.add_argument(
        "--xg-limit",
        type=int,
        default=0,
        help=(
            "Nombre de matchs historiques par équipe à enrichir en xG PRE. "
            "Défaut: 0 (désactivé pour préserver le quota)."
        ),
    )
    parser.add_argument(
        "--with-league-second-half",
        action="store_true",
        help="Reconstruit league_2nd_half_over15_freq. Très coûteux; déconseillé au premier backfill.",
    )
    parser.add_argument(
        "--preferred-bookmaker",
        default=None,
        help="Bookmaker préféré pour les opening odds PRE.",
    )
    parser.add_argument(
        "--league-history-max-pages",
        type=int,
        default=10,
        help="Pages max pour reconstruire le classement de saison as-of kickoff. Défaut: 10.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les matchs déjà seed/backfill/indexés et les fenêtres terminées.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Avec --resume, retente les match_id précédemment failed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retraite les matchs même déjà présents dans le backfill/index. Le seed reste protégé.",
    )
    parser.add_argument(
        "--ignore-completed-windows",
        action="store_true",
        help="Reparcourt les fenêtres déjà marquées terminées dans le state.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Sauvegarde tous les N nouveaux matchs tentés. Défaut: 10.",
    )
    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=DEFAULT_MIN_API_INTERVAL,
        help=f"Intervalle minimal entre appels API réels. Défaut: {DEFAULT_MIN_API_INTERVAL}s.",
    )
    parser.add_argument(
        "--cache-disable",
        action="store_true",
        help="Désactive le cache disque. Déconseillé.",
    )
    parser.add_argument(
        "--no-legacy-cache",
        action="store_true",
        help="N'utilise pas l'ancien cache historical_dataset_v2.",
    )
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="Liste les nouveaux matchs sans reconstruire les features ni consommer les appels par match.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if ENV_FILE.exists():
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

    seed_path = resolve_path(args.seed_csv, DEFAULT_SEED_CSV)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    backfill_path = output_dir / BACKFILL_FILENAME
    combined_path = output_dir / COMBINED_FILENAME
    index_path = output_dir / INDEX_FILENAME
    rejections_path = output_dir / REJECTIONS_FILENAME
    state_path = output_dir / STATE_FILENAME
    coverage_path = output_dir / COVERAGE_FILENAME
    report_path = output_dir / REPORT_FILENAME

    # ------------------------------------------------------------------
    # Modules existants validés
    # ------------------------------------------------------------------
    first_half = load_module("first_half37_engine_for_api_backfill", FIRST_HALF_MODULE_PATH)
    prematch = load_module("prematch_rebuild_for_api_backfill", PREMATCH_MODULE_PATH)
    audit = load_module("historical_live_audit_for_api_backfill", AUDIT_MODULE_PATH)
    lab = prematch.load_lab()

    patch_paginated_league_history(
        prematch=prematch,
        max_pages=max(1, args.league_history_max_pages),
    )

    seed_df = load_seed(seed_path)

    base_client = audit.StatsAPIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=6,
        min_interval=max(0.1, float(args.min_api_interval)),
        backoff=2.0,
    )

    legacy_cache_dir = None
    if not args.no_legacy_cache and DEFAULT_LEGACY_CACHE_DIR.exists():
        legacy_cache_dir = DEFAULT_LEGACY_CACHE_DIR

    cached_client = first_half.CachedHistoricalClient(
        base_client=base_client,
        cache_dir=DEFAULT_SHARED_CACHE_DIR,
        legacy_cache_dir=legacy_cache_dir,
        enabled=not args.cache_disable,
    )

    client = UnifiedHistoricalClient(cached_client)

    competition_ids = resolve_competitions(
        args=args,
        seed_df=seed_df,
        client=client,
    )

    if not competition_ids:
        print("ERREUR : aucune compétition à parcourir.")
        return 2

    inferred_end = infer_default_date_to(seed_df)
    end_date = parse_iso_date(args.date_to) or inferred_end
    start_date = parse_iso_date(args.date_from)

    if start_date is None:
        start_date = end_date - timedelta(days=max(1, args.lookback_days))

    windows = build_date_windows(
        start=start_date,
        end=end_date,
        window_days=max(1, args.window_days),
        order=args.window_order,
    )

    print()
    print("=" * 110)
    print("FIRST HALF 37 — DIRECT HISTORICAL API BACKFILL — BALANCED")
    print("=" * 110)
    print(f"Seed : {seed_path}")
    print(f"Seed rows : {len(seed_df)}")
    print(f"API base : {base_url}")
    print(f"Compétitions : {len(competition_ids)}")
    print(f"Période découverte : {start_date.isoformat()} -> {end_date.isoformat()}")
    print(f"Fenêtres : {len(windows)} x {args.window_days} jours")
    print(f"Window order : {args.window_order}")
    print(f"xG PRE limit : {max(0, args.xg_limit)}")
    print(f"API guard : {max(0, args.max_api_requests)} requêtes réelles")
    print(f"Target new rows : {max(0, args.target_new_rows)}")
    print(f"Max new attempts : {max(0, args.max_new_attempts)}")
    print(f"Max attempts / scope : {max(0, args.max_attempts_per_scope)}")
    print(f"Output : {output_dir}")
    print("=" * 110)

    backfill_df = load_csv(backfill_path)
    index_df = load_csv(index_path)
    rejections_df = load_csv(rejections_path)
    state = load_state(state_path)

    seed_ids: Set[str] = set(seed_df["match_id"].dropna().astype(str))
    backfill_ids: Set[str] = set()
    if not backfill_df.empty and "match_id" in backfill_df.columns:
        backfill_ids = set(backfill_df["match_id"].dropna().astype(str))

    successful_index_ids: Set[str] = set()
    failed_index_ids: Set[str] = set()

    if not index_df.empty and "match_id" in index_df.columns:
        for _, record in index_df.iterrows():
            mid = clean_text(record.get("match_id"))
            status = clean_text(record.get("status")).lower()
            if not mid:
                continue
            if status in ("done", "resolved", "success"):
                successful_index_ids.add(mid)
            else:
                failed_index_ids.add(mid)

    completed_windows: Set[str] = set(
        str(x) for x in state.get("completed_windows", [])
    )

    attempted_this_run = 0
    resolved_this_run = 0
    discovered_new = 0
    skipped_seed = 0
    skipped_existing = 0
    stop_reason = "completed_all_scopes"

    def should_stop() -> bool:
        nonlocal stop_reason

        if args.max_api_requests > 0 and client.request_count >= args.max_api_requests:
            stop_reason = "max_api_requests_reached"
            return True

        if args.max_new_attempts > 0 and attempted_this_run >= args.max_new_attempts:
            stop_reason = "max_new_attempts_reached"
            return True

        if args.target_new_rows > 0 and resolved_this_run >= args.target_new_rows:
            stop_reason = "target_new_rows_reached"
            return True

        return False

    def save_all() -> None:
        combined_df = merge_seed_and_backfill(seed_df, backfill_df)
        coverage_df = build_coverage(
            first_half=first_half,
            combined_df=combined_df,
            prematch_features=list(prematch.PRE_MATCH_FEATURES),
        )

        save_csv_atomic(backfill_df, backfill_path)
        save_csv_atomic(combined_df, combined_path)
        save_csv_atomic(index_df, index_path)
        save_csv_atomic(rejections_df, rejections_path)
        save_csv_atomic(coverage_df, coverage_path)

        state["updated_at_utc"] = now_utc_iso()
        state["last_stop_reason"] = stop_reason
        state["api_request_count_last_run"] = client.request_count
        atomic_write_json(state, state_path)

        write_report(
            path=report_path,
            seed_path=seed_path,
            seed_df=seed_df,
            backfill_df=backfill_df,
            combined_df=combined_df,
            index_df=index_df,
            client=client,
            args=args,
            competition_count=len(competition_ids),
            window_count=len(windows),
        )

    try:
        # Fenêtres en boucle externe : plus de diversité temporelle entre compétitions.
        for window_idx, (window_start, window_end) in enumerate(windows, start=1):
            if should_stop():
                break

            print()
            print("#" * 110)
            print(
                f"WINDOW [{window_idx}/{len(windows)}] "
                f"{window_start.isoformat()} -> {window_end.isoformat()}"
            )
            print("#" * 110)

            for comp_idx, competition_id in enumerate(competition_ids, start=1):
                if should_stop():
                    break

                key = window_key(competition_id, window_start, window_end)

                if (
                    args.resume
                    and not args.ignore_completed_windows
                    and key in completed_windows
                ):
                    continue

                print()
                print("-" * 110)
                print(
                    f"SCOPE comp [{comp_idx}/{len(competition_ids)}] {competition_id} | "
                    f"{window_start.isoformat()} -> {window_end.isoformat()}"
                )
                print("-" * 110)

                scope_fully_consumed = True
                scope_seen = 0
                scope_new = 0
                scope_attempted = 0
                last_page_seen = 0
                last_total_pages = 0

                for stub, page, total_pages in iter_finished_matches_window(
                    client=client,
                    competition_id=competition_id,
                    start=window_start,
                    end=window_end,
                    max_pages=max(1, args.max_pages_per_window),
                    per_page=max(1, min(100, args.per_page)),
                    max_api_requests=max(0, args.max_api_requests),
                ):
                    scope_seen += 1
                    last_page_seen = max(last_page_seen, int(page or 0))
                    last_total_pages = max(last_total_pages, int(total_pages or 0))
                    match_id = extract_match_id(stub)

                    if not match_id:
                        continue

                    if match_id in seed_ids:
                        skipped_seed += 1
                        continue

                    if not args.force:
                        if match_id in backfill_ids or match_id in successful_index_ids:
                            skipped_existing += 1
                            continue

                        if args.resume and match_id in failed_index_ids and not args.retry_failed:
                            skipped_existing += 1
                            continue

                    discovered_new += 1
                    scope_new += 1

                    if args.discovery_only:
                        print(
                            f"DISCOVERY | {match_id} | page={page}/{total_pages or '?'} | "
                            f"comp={competition_id}"
                        )
                        continue

                    if should_stop():
                        scope_fully_consumed = False
                        break

                    attempted_this_run += 1
                    scope_attempted += 1

                    print(
                        f"[{attempted_this_run}] {match_id} | "
                        f"page={page}/{total_pages or '?'} | "
                        f"API={client.request_count}"
                    )

                    try:
                        row, match_audit = rebuild_one_new_match(
                            stub=stub,
                            client=client,
                            first_half=first_half,
                            prematch=prematch,
                            audit=audit,
                            lab=lab,
                            xg_limit=max(0, args.xg_limit),
                            with_league_second_half=args.with_league_second_half,
                            preferred_bookmaker=args.preferred_bookmaker,
                        )
                    except Exception as exc:
                        row = None
                        match_audit = {
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": extract_season_id(stub),
                            "status": "failed",
                            "error": f"unexpected:{type(exc).__name__}:{exc}",
                            "collected_at_utc": now_utc_iso(),
                        }

                    match_audit = dict(match_audit or {})
                    match_audit.setdefault("match_id", match_id)
                    match_audit.setdefault("competition_id", competition_id)
                    match_audit.setdefault("season_id", extract_season_id(stub))
                    match_audit["discovery_window_start"] = window_start.isoformat()
                    match_audit["discovery_window_end"] = window_end.isoformat()
                    match_audit["discovery_page"] = page
                    match_audit["schema_version"] = SCHEMA_VERSION

                    index_df = append_or_replace_by_match_id(index_df, match_audit)

                    status = clean_text(match_audit.get("status")).lower()

                    if row is not None:
                        resolved_this_run += 1
                        backfill_df = append_or_replace_by_match_id(backfill_df, row)
                        backfill_ids.add(match_id)
                        successful_index_ids.add(match_id)
                        failed_index_ids.discard(match_id)

                        print(
                            f"  {row.get('home_team', '')} vs {row.get('away_team', '')} | "
                            f"PRE={row.get('_prematch_available', 0)}/{row.get('_prematch_total', 0)} | "
                            f"LIVE37_CORE={row.get('_live37_core_available', 0)}/{row.get('_live37_core_total', 0)} | "
                            f"XG37={row.get('_live37_xg_available', 0)}/{row.get('_live37_xg_total', 0)} | "
                            f"FH_ODDS={row.get('_fh_odds_available', 0)}/{row.get('_fh_odds_total', 0)} | "
                            f"TARGET={row.get(TARGET_COLUMN)} | DONE"
                        )
                    else:
                        failed_index_ids.add(match_id)

                        rejection = {
                            "match_id": match_id,
                            "competition_id": competition_id,
                            "season_id": extract_season_id(stub),
                            "status": status or "failed",
                            "reason": match_audit.get("error", ""),
                            "discovery_window_start": window_start.isoformat(),
                            "discovery_window_end": window_end.isoformat(),
                            "collected_at_utc": match_audit.get("collected_at_utc", now_utc_iso()),
                        }
                        rejections_df = append_or_replace_by_match_id(rejections_df, rejection)

                        print(
                            f"  REJET | {rejection['status']} | {rejection['reason']}"
                        )

                    if attempted_this_run % max(1, args.save_every) == 0:
                        save_all()
                        print(
                            f"  SAVE | backfill={len(backfill_df)} | "
                            f"new_resolved={resolved_this_run} | "
                            f"attempts={attempted_this_run} | "
                            f"API={client.request_count} | "
                            f"cache={client.cache_hits + client.legacy_cache_hits}"
                        )

                    if should_stop():
                        scope_fully_consumed = False
                        break

                    if (
                        args.max_attempts_per_scope > 0
                        and scope_attempted >= args.max_attempts_per_scope
                    ):
                        scope_fully_consumed = False
                        print(
                            f"  SCOPE CAP | attempts={scope_attempted} | "
                            f"max={args.max_attempts_per_scope} -> compétition suivante"
                        )
                        break

                if args.discovery_only:
                    print(f"  Scope discovery : seen={scope_seen} | new={scope_new}")

                # Ne marque jamais un scope comme terminé si la pagination a été
                # tronquée par --max-pages-per-window. Sinon un --resume futur
                # pourrait sauter des pages historiques jamais explorées.
                if (
                    last_total_pages > 0
                    and last_page_seen > 0
                    and last_page_seen < last_total_pages
                ):
                    scope_fully_consumed = False
                elif (
                    last_total_pages == 0
                    and last_page_seen >= max(1, args.max_pages_per_window)
                    and scope_seen > 0
                ):
                    scope_fully_consumed = False

                if scope_fully_consumed and not should_stop():
                    mark_window_completed(
                        state,
                        competition_id,
                        window_start,
                        window_end,
                    )
                    completed_windows.add(key)
                    atomic_write_json(state, state_path)

                if should_stop():
                    break

            if should_stop():
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\nInterruption utilisateur — sauvegarde en cours...")

    finally:
        save_all()

    combined_df = merge_seed_and_backfill(seed_df, backfill_df)

    print()
    print("=" * 110)
    print("BACKFILL TERMINE")
    print("=" * 110)
    print(f"Stop reason : {stop_reason}")
    print(f"Nouveaux match_id découverts : {discovered_new}")
    print(f"Nouveaux matchs tentés : {attempted_this_run}")
    print(f"Nouvelles lignes résolues : {resolved_this_run}")
    print(f"Backfill total : {len(backfill_df)}")
    print(f"Seed protégé : {len(seed_df)}")
    print(f"Combined total : {len(combined_df)}")
    print(f"Skip seed : {skipped_seed}")
    print(f"Skip existing/index : {skipped_existing}")
    print(f"Requêtes API réelles : {client.request_count}")
    print(f"HTTP 429 : {client.retry_429_count}")
    print(f"Cache hits First Half 37 : {client.cache_hits}")
    print(f"Cache hits legacy : {client.legacy_cache_hits}")
    print(f"Backfill : {backfill_path}")
    print(f"Combined : {combined_path}")
    print(f"Rapport : {report_path}")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
