#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_first_half_37_post_model.py

COLLECTE EXTERNE POST-MODÈLE — FIRST HALF 37
=============================================

But
---
Collecter un NOUVEL échantillon externe de matchs terminés, strictement postérieur
au dataset utilisé pour sélectionner/tuner First_Half37 V1.

Par défaut :
    date_from = 2026-09-14
    date_to   = aujourd'hui UTC
    target    = 300 lignes résolues

Ce script :
- découvre des matchs terminés via TheStatsAPI ;
- se limite aux compétitions déjà présentes dans le dataset First_Half37 combiné ;
- exclut tous les match_id déjà utilisés dans les 1 802 matchs ;
- reconstruit les 59 features PRE strictement as-of kickoff ;
- reconstruit LIVE @37 depuis timeline/shotmap ;
- résout le target goal AFTER 37 and BEFORE HT avec la même logique que le collecteur validé ;
- conserve Missing != 0 ;
- ne modifie PAS le modèle, les 10 features, C=0.01 ni la calibration ;
- écrit un dataset séparé pour validation externe/shadow.

Dépendances locales attendues à la racine du projet :
    collect_first_half_37_dataset.py
    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py
    .env

Exemple recommandé :
    caffeinate -dimsu python3 collect_first_half_37_post_model.py \
      --target-new-rows 300 \
      --max-new-attempts 450 \
      --min-api-interval 0.75 \
      --resume

Si le scanner LIVE tourne en parallèle, utiliser plutôt :
    --min-api-interval 1.25
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

COLLECTOR_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
PREMATCH_PATH = BASE_DIR / "rebuild_clean_prematch_v2.py"
AUDIT_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

DEFAULT_REFERENCE_DATASET = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "historical_api"
    / "First_Half_37_COMBINED.csv"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "post_model_external_v1"
)

DEFAULT_CACHE_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "cache"
    / "api"
)

LEGACY_CACHE_DIR = (
    BASE_DIR
    / "outputs"
    / "historical_dataset_v2"
    / "api_cache_v2"
)

DEFAULT_DATE_FROM = "2026-09-14"
DEFAULT_TARGET_NEW_ROWS = 300
DEFAULT_MAX_NEW_ATTEMPTS = 450
DEFAULT_MAX_PAGES = 50
DEFAULT_PER_PAGE = 100
DEFAULT_MIN_API_INTERVAL = 0.75
DEFAULT_MAX_API_REQUESTS = 20000
DEFAULT_SAVE_EVERY = 10

DATASET_NAME = "First_Half_37_POST_MODEL_EXTERNAL.csv"
INDEX_NAME = "post_model_external_index.csv"
REJECTIONS_NAME = "post_model_external_rejections.csv"
REPORT_NAME = "post_model_external_report.md"
STATE_NAME = "post_model_external_state.json"
COVERAGE_NAME = "post_model_external_feature_coverage.csv"

TARGET_COLUMN = "goal_after_37_before_ht"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def parse_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = clean_text(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            try:
                ts = pd.to_datetime(value, utc=True, errors="coerce")
                if pd.isna(ts):
                    return None
                dt = ts.to_pydatetime()
            except Exception:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module requis absent : {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def match_id_from_row(row: Mapping[str, Any]) -> str:
    for key in ("match_id", "id", "event_id"):
        v = clean_text(row.get(key))
        if v:
            return v
    return ""


def kickoff_from_match(match: Mapping[str, Any]) -> Optional[datetime]:
    candidates = [
        match.get("kickoff_utc"),
        match.get("utc_date"),
        match.get("date"),
        match.get("start_time"),
        match.get("start_at"),
    ]
    for value in candidates:
        dt = parse_utc(value)
        if dt is not None:
            return dt
    return None


def get_team_name(audit, match: Mapping[str, Any], side: str) -> str:
    try:
        team = audit.extract_home_team(match) if side == "home" else audit.extract_away_team(match)
        return clean_text(audit.extract_team_name(team))
    except Exception:
        team = match.get(f"{side}_team") or match.get(side) or {}
        if isinstance(team, Mapping):
            return clean_text(team.get("name") or team.get("team_name"))
        return clean_text(team)


def get_competition_id(audit, match: Mapping[str, Any]) -> str:
    try:
        return clean_text(audit.extract_competition_id(match))
    except Exception:
        comp = match.get("competition")
        if isinstance(comp, Mapping):
            return clean_text(comp.get("id") or comp.get("competition_id"))
        return clean_text(match.get("competition_id"))


def get_season_id(audit, match: Mapping[str, Any]) -> str:
    try:
        return clean_text(audit.extract_season_id(match))
    except Exception:
        season = match.get("season")
        if isinstance(season, Mapping):
            return clean_text(season.get("id") or season.get("season_id"))
        return clean_text(match.get("season_id"))


class CachedGetAdapter:
    """Expose client.get(...) pour rebuild_clean_prematch_v2.py."""

    def __init__(self, cached_client):
        self.cached_client = cached_client

    def get(self, endpoint: str, params: Optional[Mapping[str, Any]] = None):
        payload, _status, _error = self.cached_client.request(endpoint, params=params)
        return payload


class GuardedCachedClient:
    """Proxy qui impose un plafond d'appels API réels."""

    def __init__(self, inner, max_api_requests: int):
        self.inner = inner
        self.max_api_requests = int(max_api_requests)

    def request(self, endpoint: str, params: Optional[Mapping[str, Any]] = None):
        if self.request_count >= self.max_api_requests:
            raise RuntimeError("api_guard_reached")
        return self.inner.request(endpoint, params=params)

    @property
    def request_count(self) -> int:
        return int(self.inner.request_count)

    @property
    def retry_429_count(self) -> int:
        return int(self.inner.retry_429_count)

    @property
    def cache_hits(self) -> int:
        return int(self.inner.cache_hits)

    @property
    def legacy_cache_hits(self) -> int:
        return int(self.inner.legacy_cache_hits)

    @property
    def cache_misses(self) -> int:
        return int(self.inner.cache_misses)

    @property
    def cache_writes(self) -> int:
        return int(self.inner.cache_writes)


class GuardedGetAdapter(CachedGetAdapter):
    def __init__(self, guarded_client: GuardedCachedClient):
        self.cached_client = guarded_client


def clear_prematch_memory_caches(prematch) -> None:
    # Critique : ces caches ne sont pas tous cutoff-aware.
    for name in (
        "MATCH_DETAIL_CACHE",
        "TEAM_HISTORY_CACHE",
        "LEAGUE_HISTORY_CACHE",
        "ODDS_CACHE",
    ):
        obj = getattr(prematch, name, None)
        if hasattr(obj, "clear"):
            obj.clear()


def install_strict_cutoff_history_fetchers(prematch, lab, cutoff_ref: Dict[str, Optional[datetime]]) -> None:
    """
    Remplace les fetchers PRE par des versions date_to-aware.
    Cela évite qu'un appel d'historique récent tronqué à 100 matchs fasse entrer
    des lignes postérieures au kickoff ou masque des lignes anciennes utiles.
    """

    def _mid(item: Mapping[str, Any]) -> str:
        try:
            return clean_text(lab.extract_match_id(item))
        except Exception:
            return match_id_from_row(item)

    def strict_team_history(_lab, client, team_id: str, per_page: int = 100):
        cutoff = cutoff_ref.get("cutoff")
        if cutoff is None:
            raise RuntimeError("strict_cutoff_missing")
        key = f"{team_id}|{cutoff.date().isoformat()}"
        cache = getattr(prematch, "TEAM_HISTORY_CACHE", {})
        if key in cache:
            return cache[key]
        rows: List[Dict[str, Any]] = []
        seen = set()
        # 2 pages = jusqu'à 200 matchs antérieurs, largement suffisant pour form/H2H.
        for page in range(1, 3):
            payload = client.get(
                "/football/matches",
                params={
                    "team_id": team_id,
                    "status": "finished",
                    "date_to": cutoff.date().isoformat(),
                    "page": page,
                    "per_page": min(100, int(per_page)),
                },
            )
            items = prematch.api_list(_lab, payload)
            if not items:
                break
            added = 0
            for item in items:
                mid = _mid(item)
                if mid and mid in seen:
                    continue
                if mid:
                    seen.add(mid)
                rows.append(dict(item))
                added += 1
            if added == 0 or len(items) < min(100, int(per_page)):
                break
        cache[key] = rows
        return rows

    def strict_league_history(_lab, client, competition_id: str, season_id: Optional[str], per_page: int = 100):
        cutoff = cutoff_ref.get("cutoff")
        if cutoff is None:
            raise RuntimeError("strict_cutoff_missing")
        key = (str(competition_id), str(season_id or ""), cutoff.date().isoformat())
        cache = getattr(prematch, "LEAGUE_HISTORY_CACHE", {})
        if key in cache:
            return cache[key]
        rows: List[Dict[str, Any]] = []
        seen = set()
        # Paginer toute la saison antérieure au cutoff, au lieu de tronquer à 100.
        for page in range(1, 21):
            params: Dict[str, Any] = {
                "competition_id": competition_id,
                "status": "finished",
                "date_to": cutoff.date().isoformat(),
                "page": page,
                "per_page": min(100, int(per_page)),
            }
            if season_id:
                params["season_id"] = season_id
            payload = client.get("/football/matches", params=params)
            items = prematch.api_list(_lab, payload)
            if not items:
                break
            added = 0
            for item in items:
                mid = _mid(item)
                if mid and mid in seen:
                    continue
                if mid:
                    seen.add(mid)
                rows.append(dict(item))
                added += 1
            if added == 0 or len(items) < min(100, int(per_page)):
                break
        cache[key] = rows
        return rows

    prematch.get_team_history = strict_team_history
    prematch.get_league_history = strict_league_history


def discover_matches(
    client: GuardedCachedClient,
    collector,
    date_from: str,
    date_to: str,
    allowed_competitions: Sequence[str],
    max_pages: int,
    per_page: int,
) -> List[Dict[str, Any]]:
    allowed = set(clean_text(x) for x in allowed_competitions if clean_text(x))
    found: List[Dict[str, Any]] = []
    seen = set()

    for page in range(1, max_pages + 1):
        payload, status, error = client.request(
            "/football/matches",
            params={
                "status": "finished",
                "date_from": date_from,
                "date_to": date_to,
                "page": page,
                "per_page": per_page,
            },
        )
        if status != 200 or payload is None:
            print(f"DISCOVERY STOP | page={page} | status={status} | error={error}")
            break

        rows = collector.payload_list(payload)
        if not rows:
            break

        added = 0
        for row in rows:
            mid = match_id_from_row(row)
            if not mid or mid in seen:
                continue
            comp_id = clean_text(row.get("competition_id"))
            if not comp_id:
                comp = row.get("competition")
                if isinstance(comp, Mapping):
                    comp_id = clean_text(comp.get("id") or comp.get("competition_id"))
            if allowed and comp_id and comp_id not in allowed:
                continue
            seen.add(mid)
            found.append(dict(row))
            added += 1

        print(f"DISCOVERY page={page} | rows={len(rows)} | retained_total={len(found)} | API={client.request_count}")
        if added == 0 and len(rows) < per_page:
            break
        if len(rows) < per_page:
            break

    # Ordre chronologique pour une lecture propre. La sélection n'utilise aucun target.
    def key(r: Mapping[str, Any]):
        dt = kickoff_from_match(r)
        return dt or datetime.max.replace(tzinfo=timezone.utc)

    found.sort(key=key)
    return found


def make_source_row(
    audit,
    prematch,
    target_match: Mapping[str, Any],
    kickoff: datetime,
    features: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "match_id": clean_text(audit.extract_match_id(target_match)),
        "competition_id": get_competition_id(audit, target_match),
        "season_id": get_season_id(audit, target_match),
        "home_team": get_team_name(audit, target_match, "home"),
        "away_team": get_team_name(audit, target_match, "away"),
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
        "source_dataset": "post_model_external_2026_09_14_plus",
    }
    for name in prematch.PRE_MATCH_FEATURES:
        value = features.get(name)
        f = safe_float(value)
        row[name] = np.nan if f is None else f
    return row


def feature_coverage(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in columns:
        if col not in df.columns:
            available = 0
        else:
            available = int(df[col].notna().sum())
        rows.append({
            "feature": col,
            "available": available,
            "rows": total,
            "coverage_pct": (100.0 * available / total) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["coverage_pct", "feature"], ascending=[False, True])


def write_report(
    path: Path,
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
    client: GuardedCachedClient,
    args,
) -> None:
    target = pd.to_numeric(df.get(TARGET_COLUMN, pd.Series(dtype=float)), errors="coerce")
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    resolved = positives + negatives
    rate = (100.0 * positives / resolved) if resolved else 0.0
    dates = pd.to_datetime(df.get("kickoff_utc", pd.Series(dtype=str)), utc=True, errors="coerce", format="mixed") if not df.empty else pd.Series(dtype="datetime64[ns, UTC]")
    dmin = dates.min() if len(dates) and dates.notna().any() else None
    dmax = dates.max() if len(dates) and dates.notna().any() else None

    lines = [
        "# First Half 37 — Post-model External Collection V1",
        "",
        f"- Généré : `{now_utc_iso()}`",
        f"- Période demandée : `{args.date_from}` -> `{args.date_to}`",
        f"- Target demandé : **{args.target_new_rows}** lignes résolues",
        "",
        "## Résultat",
        "",
        f"- Lignes résolues : **{len(df)}**",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        f"- Taux positif : **{rate:.2f}%**",
        f"- Dates dataset : **{dmin} -> {dmax}**",
        f"- Match IDs uniques : **{df['match_id'].nunique() if 'match_id' in df.columns else 0}**",
        f"- Index total : **{len(index_df)}**",
        f"- Rejets : **{len(rejections_df)}**",
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
        "## Règles",
        "",
        "- Matchs déjà présents dans First_Half_37_COMBINED.csv exclus.",
        "- PRE reconstruit strictement as-of kickoff.",
        "- Caches mémoire PRE dépendants du cutoff vidés entre matchs.",
        "- LIVE features limitées à minute <=37 via le collecteur validé.",
        "- Buts >37 utilisés uniquement pour le target.",
        "- Missing != 0.",
        "- Ce dataset est séparé : ne pas le mélanger au training avant analyse externe.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_all(
    dataset_df: pd.DataFrame,
    index_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
    dataset_path: Path,
    index_path: Path,
    rejections_path: Path,
    coverage_path: Path,
    state_path: Path,
    report_path: Path,
    coverage_columns: Sequence[str],
    client: GuardedCachedClient,
    args,
    stop_reason: str,
) -> None:
    if not dataset_df.empty:
        dataset_df = dataset_df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    if not index_df.empty:
        index_df = index_df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    if not rejections_df.empty:
        rejections_df = rejections_df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)

    atomic_write_csv(dataset_df, dataset_path)
    atomic_write_csv(index_df, index_path)
    atomic_write_csv(rejections_df, rejections_path)
    atomic_write_csv(feature_coverage(dataset_df, coverage_columns), coverage_path)
    atomic_write_json({
        "saved_at_utc": now_utc_iso(),
        "stop_reason": stop_reason,
        "resolved_rows": len(dataset_df),
        "indexed": len(index_df),
        "rejections": len(rejections_df),
        "api_requests": client.request_count,
        "http_429": client.retry_429_count,
        "date_from": args.date_from,
        "date_to": args.date_to,
    }, state_path)
    write_report(report_path, dataset_df, index_df, rejections_df, client, args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collecte externe First_Half37 après le 13 septembre 2026.")
    parser.add_argument("--date-from", default=DEFAULT_DATE_FROM)
    parser.add_argument("--date-to", default=date.today().isoformat())
    parser.add_argument("--target-new-rows", type=int, default=DEFAULT_TARGET_NEW_ROWS)
    parser.add_argument("--max-new-attempts", type=int, default=DEFAULT_MAX_NEW_ATTEMPTS)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    parser.add_argument("--min-api-interval", type=float, default=DEFAULT_MIN_API_INTERVAL)
    parser.add_argument("--max-api-requests", type=int, default=DEFAULT_MAX_API_REQUESTS)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--reference-dataset", default=str(DEFAULT_REFERENCE_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-legacy-cache", action="store_true")
    args = parser.parse_args()

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()

    api_key = clean_text(os.getenv("STATS_API_KEY"))
    base_url = clean_text(os.getenv("STATS_API_BASE_URL", "https://api.thestatsapi.com/api")).rstrip("/")
    timeout = int(float(os.getenv("STATS_API_TIMEOUT", "20")))
    if not api_key:
        print("ERREUR : STATS_API_KEY absent du .env")
        return 2

    date_from_dt = parse_utc(args.date_from + "T00:00:00Z")
    date_to_dt = parse_utc(args.date_to + "T23:59:59Z")
    if date_from_dt is None or date_to_dt is None or date_from_dt > date_to_dt:
        print("ERREUR : plage de dates invalide")
        return 2

    reference_path = Path(args.reference_dataset).expanduser().resolve()
    if not reference_path.exists():
        print(f"ERREUR : dataset référence introuvable : {reference_path}")
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DATASET_NAME
    index_path = output_dir / INDEX_NAME
    rejections_path = output_dir / REJECTIONS_NAME
    report_path = output_dir / REPORT_NAME
    state_path = output_dir / STATE_NAME
    coverage_path = output_dir / COVERAGE_NAME

    collector = load_module("fh37_collector_post_model_runtime", COLLECTOR_PATH)
    prematch = load_module("fh37_prematch_post_model_runtime", PREMATCH_PATH)
    audit = load_module("fh37_audit_post_model_runtime", AUDIT_PATH)
    lab = prematch.load_lab()
    cutoff_ref: Dict[str, Optional[datetime]] = {"cutoff": None}
    install_strict_cutoff_history_fetchers(prematch, lab, cutoff_ref)

    reference_df = pd.read_csv(reference_path)
    if "match_id" not in reference_df.columns:
        print("ERREUR : le dataset référence ne contient pas match_id")
        return 2
    used_ids = set(reference_df["match_id"].astype(str).str.strip())
    competitions = sorted(set(reference_df.get("competition_id", pd.Series(dtype=str)).dropna().astype(str).str.strip()))

    base_client = audit.StatsAPIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=6,
        min_interval=max(0.1, float(args.min_api_interval)),
        backoff=2.0,
    )
    legacy_cache = None if args.no_legacy_cache else (LEGACY_CACHE_DIR if LEGACY_CACHE_DIR.exists() else None)
    cached = collector.CachedHistoricalClient(
        base_client=base_client,
        cache_dir=DEFAULT_CACHE_DIR,
        legacy_cache_dir=legacy_cache,
        enabled=True,
    )
    client = GuardedCachedClient(cached, max_api_requests=args.max_api_requests)
    prematch_client = GuardedGetAdapter(client)

    dataset_df = load_csv(dataset_path) if args.resume else pd.DataFrame()
    index_df = load_csv(index_path) if args.resume else pd.DataFrame()
    rejections_df = load_csv(rejections_path) if args.resume else pd.DataFrame()

    existing_ids = set(dataset_df.get("match_id", pd.Series(dtype=str)).astype(str).str.strip())
    indexed_ids = set(index_df.get("match_id", pd.Series(dtype=str)).astype(str).str.strip())

    print("=" * 108)
    print("FIRST HALF 37 — POST-MODEL EXTERNAL COLLECTION V1")
    print("=" * 108)
    print(f"Reference dataset : {reference_path}")
    print(f"Reference rows    : {len(reference_df)}")
    print(f"Date range        : {args.date_from} -> {args.date_to}")
    print(f"Competitions      : {len(competitions)}")
    print(f"Target new rows   : {args.target_new_rows}")
    print(f"Max attempts      : {args.max_new_attempts}")
    print(f"API guard         : {args.max_api_requests}")
    print(f"Min API interval  : {max(0.1, float(args.min_api_interval)):.2f}s")
    print(f"Output            : {output_dir}")
    print("=" * 108)

    try:
        discovered = discover_matches(
            client=client,
            collector=collector,
            date_from=args.date_from,
            date_to=args.date_to,
            allowed_competitions=competitions,
            max_pages=max(1, args.max_pages),
            per_page=max(1, min(100, args.per_page)),
        )
    except RuntimeError as exc:
        if str(exc) == "api_guard_reached":
            print("STOP : API guard atteint pendant la découverte")
            return 3
        raise

    candidates = []
    for raw in discovered:
        mid = match_id_from_row(raw)
        if not mid:
            continue
        if mid in used_ids:
            continue
        if args.resume and (mid in existing_ids or mid in indexed_ids):
            continue
        dt = kickoff_from_match(raw)
        if dt is None:
            continue
        if not (date_from_dt <= dt <= date_to_dt):
            continue
        candidates.append(raw)

    print(f"Matchs découverts dans la période : {len(discovered)}")
    print(f"Nouveaux candidats après exclusions : {len(candidates)}")

    new_resolved = 0
    attempts = 0
    stop_reason = "candidates_exhausted"
    dataset_rows = dataset_df.to_dict("records") if not dataset_df.empty else []
    index_rows = index_df.to_dict("records") if not index_df.empty else []
    rejection_rows = rejections_df.to_dict("records") if not rejections_df.empty else []

    coverage_columns = list(dict.fromkeys(
        list(prematch.PRE_MATCH_FEATURES)
        + list(getattr(collector, "LIVE_37_ALL_FEATURES", []))
        + list(getattr(collector, "FH_ODDS_FEATURES", []))
    ))

    try:
        for raw in candidates:
            if new_resolved >= args.target_new_rows:
                stop_reason = "target_new_rows_reached"
                break
            if attempts >= args.max_new_attempts:
                stop_reason = "max_new_attempts_reached"
                break
            if client.request_count >= args.max_api_requests:
                stop_reason = "api_guard_reached"
                break

            mid = match_id_from_row(raw)
            attempts += 1
            print()
            print("-" * 108)
            print(f"[{attempts}] {mid} | API={client.request_count}")

            try:
                # Detail canonique
                detail_payload, detail_status, detail_error = client.request(f"/football/matches/{mid}")
                if detail_status != 200 or detail_payload is None:
                    raise RuntimeError(f"detail_unavailable:{detail_status}:{detail_error}")
                target_match = audit.unwrap_match(detail_payload)
                if not target_match:
                    raise RuntimeError("detail_unwrap_empty")

                kickoff = kickoff_from_match(target_match)
                if kickoff is None:
                    raise RuntimeError("kickoff_unparseable")
                if not (date_from_dt <= kickoff <= date_to_dt):
                    raise RuntimeError("outside_requested_period")

                # Anti-fuite PRE : vider les caches mémoire cutoff-dépendants
                # puis injecter date_to=<date kickoff> dans les historiques.
                clear_prematch_memory_caches(prematch)
                cutoff_ref["cutoff"] = kickoff

                pre_features, pre_audit = prematch.build_prematch_asof(
                    lab=lab,
                    client=prematch_client,
                    target_match=target_match,
                    cutoff=kickoff.replace(tzinfo=None),
                    xg_limit=0,
                    with_second_half=False,
                    preferred_bookmaker=None,
                )

                source_row = make_source_row(
                    audit=audit,
                    prematch=prematch,
                    target_match=target_match,
                    kickoff=kickoff,
                    features=pre_features,
                )

                row, live_audit = collector.collect_one_match(
                    source_row=source_row,
                    client=client,
                    audit=audit,
                    prematch_features=prematch.PRE_MATCH_FEATURES,
                )

                status = clean_text(live_audit.get("status"))
                error = clean_text(live_audit.get("error"))
                if row is None or status != "done":
                    rejection_rows.append({
                        "match_id": mid,
                        "kickoff_utc": kickoff.isoformat(),
                        "status": status or "failed",
                        "error": error or "unresolved",
                        "api_requests_so_far": client.request_count,
                    })
                    index_rows.append({
                        "match_id": mid,
                        "kickoff_utc": kickoff.isoformat(),
                        "status": status or "failed",
                        "error": error or "unresolved",
                    })
                    print(f"  REJET | {status or 'failed'} | {error or 'unresolved'}")
                else:
                    # Marquer explicitement ce bloc comme externe post-modèle.
                    row["schema_version"] = "first_half_37_post_model_external_v1"
                    row["source_dataset"] = "post_model_external_2026_09_14_plus"
                    row["prematch_policy"] = "strict_asof_kickoff"
                    row["external_validation_only"] = 1
                    row["collected_at_utc"] = now_utc_iso()
                    row["_prematch_available"] = int(sum(pd.notna(row.get(f)) for f in prematch.PRE_MATCH_FEATURES))

                    dataset_rows.append(row)
                    index_rows.append({
                        "match_id": mid,
                        "kickoff_utc": row.get("kickoff_utc", kickoff.isoformat()),
                        "status": "done",
                        "target": row.get(TARGET_COLUMN, np.nan),
                        "prematch_available": row.get("_prematch_available", np.nan),
                        "live37_core_available": row.get("_live37_core_available", np.nan),
                        "live37_xg_available": row.get("_live37_xg_available", np.nan),
                    })
                    new_resolved += 1
                    print(
                        f"  {row.get('home_team','')} vs {row.get('away_team','')} | "
                        f"PRE={row.get('_prematch_available','?')}/{len(prematch.PRE_MATCH_FEATURES)} | "
                        f"LIVE37_CORE={row.get('_live37_core_available','?')}/{len(getattr(collector,'LIVE_37_CORE_FEATURES',[]))} | "
                        f"XG37={row.get('_live37_xg_available','?')}/{len(getattr(collector,'LIVE_37_XG_FEATURES',[]))} | "
                        f"TARGET={row.get(TARGET_COLUMN)} | DONE"
                    )

            except RuntimeError as exc:
                if str(exc) == "api_guard_reached":
                    stop_reason = "api_guard_reached"
                    break
                rejection_rows.append({
                    "match_id": mid,
                    "kickoff_utc": clean_text(raw.get("kickoff_utc") or raw.get("utc_date") or raw.get("date")),
                    "status": "failed",
                    "error": str(exc),
                    "api_requests_so_far": client.request_count,
                })
                index_rows.append({"match_id": mid, "status": "failed", "error": str(exc)})
                print(f"  REJET | failed | {exc}")
            except Exception as exc:
                rejection_rows.append({
                    "match_id": mid,
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                    "api_requests_so_far": client.request_count,
                })
                index_rows.append({"match_id": mid, "status": "failed", "error": f"{type(exc).__name__}:{exc}"})
                print(f"  REJET | failed | {type(exc).__name__}: {exc}")

            if attempts % max(1, args.save_every) == 0:
                dataset_df = pd.DataFrame(dataset_rows)
                index_df = pd.DataFrame(index_rows)
                rejections_df = pd.DataFrame(rejection_rows)
                save_all(
                    dataset_df, index_df, rejections_df,
                    dataset_path, index_path, rejections_path, coverage_path,
                    state_path, report_path, coverage_columns, client, args,
                    stop_reason="running",
                )
                print(
                    f"  SAVE | dataset={len(dataset_df)} | new_resolved={new_resolved} | "
                    f"attempts={attempts} | API={client.request_count} | cache={client.cache_hits}"
                )

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\nINTERRUPTION CLAVIER — sauvegarde en cours...")

    dataset_df = pd.DataFrame(dataset_rows)
    index_df = pd.DataFrame(index_rows)
    rejections_df = pd.DataFrame(rejection_rows)
    save_all(
        dataset_df, index_df, rejections_df,
        dataset_path, index_path, rejections_path, coverage_path,
        state_path, report_path, coverage_columns, client, args,
        stop_reason=stop_reason,
    )

    print()
    print("=" * 108)
    print("POST-MODEL EXTERNAL COLLECTION TERMINEE")
    print("=" * 108)
    print(f"Stop reason             : {stop_reason}")
    print(f"Nouveaux matchs tentés : {attempts}")
    print(f"Nouvelles lignes résolues : {new_resolved}")
    print(f"Dataset externe total  : {len(dataset_df)}")
    print(f"Requêtes API réelles   : {client.request_count}")
    print(f"HTTP 429                : {client.retry_429_count}")
    print(f"Cache hits First Half37 : {client.cache_hits}")
    print(f"Cache hits legacy       : {client.legacy_cache_hits}")
    print(f"Dataset                 : {dataset_path}")
    print(f"Rapport                 : {report_path}")
    print("IMPORTANT : garder ce dataset séparé du training jusqu'à l'analyse externe.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
