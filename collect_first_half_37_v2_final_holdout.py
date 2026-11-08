#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 FINAL HOLDOUT COLLECTOR (POST-FREEZE)

Collecte aveugle du nouveau holdout FINAL V2.

Fenêtre pré-enregistrée:
    2026-09-20 00:00:00 UTC
    ->
    2026-10-03 23:59:59 UTC

Règles:
- vérifie le FREEZE_LOCK et les SHA256 AVANT collecte;
- refuse un modèle différent du V2 gelé;
- liste TOUS les matchs terminés de la fenêtre, jour par jour;
- reconstruit le PRE-MATCH strictement as-of kickoff;
- reconstruit les features LIVE avec minute <= 37;
- résout le target uniquement avec la timeline complète de 1re mi-temps;
- ne sélectionne jamais les matchs selon le target ou la probabilité;
- n'affiche pas le taux de positifs pendant la collecte;
- ne score jamais le modèle: le scoring appartient à l'évaluateur aveugle séparé.

Dépendances attendues à la racine:
    collect_first_half_37_dataset.py
    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py
    .env
    outputs/first_half_37/v2_frozen/...

Sorties:
    outputs/first_half_37/v2_final_holdout/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

FIRST_HALF_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
PREMATCH_MODULE_PATH = BASE_DIR / "rebuild_clean_prematch_v2.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_frozen"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FREEZE_LOCK.json"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FROZEN.joblib"
SCHEMA_PATH = FROZEN_DIR / "first_half_37_v2_frozen_schema.json"
FREEZE_MANIFEST_PATH = FROZEN_DIR / "first_half_37_v2_freeze_manifest.json"

DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_final_holdout"
DATASET_NAME = "First_Half_37_V2_FINAL_HOLDOUT.csv"
AUDIT_NAME = "first_half_37_v2_final_holdout_collection_audit.csv"
REJECTIONS_NAME = "first_half_37_v2_final_holdout_rejections.csv"
LISTING_NAME = "first_half_37_v2_final_holdout_listed_matches.csv"
MANIFEST_NAME = "first_half_37_v2_final_holdout_collection_manifest.json"

START_UTC = pd.Timestamp("2026-09-20 00:00:00", tz="UTC")
END_UTC = pd.Timestamp("2026-10-03 23:59:59", tz="UTC")

EXPECTED_MODEL_SHA256 = (
    "defcbfadde1a7d15528b74ec3ddd38ca44a415fbc64ade7b73cc09237e09dcba"
)
EXPECTED_MODEL_NAME = "FIRST_HALF37_V2_TOP8_LOGISTIC_L2_OOF150"
EXPECTED_C = 0.01
EXPECTED_ALPHA = -0.173842194905
EXPECTED_WINDOW = 150
EXPECTED_FEATURES = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON invalide: {path}")
    return value


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Module requis introuvable: {path}\n"
            "Place ce script à la racine du projet."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_dt(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def verify_freeze() -> Dict[str, Any]:
    for p in (FREEZE_LOCK_PATH, MODEL_PATH, SCHEMA_PATH, FREEZE_MANIFEST_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Artefact gelé manquant: {p}")

    lock = load_json(FREEZE_LOCK_PATH)

    if lock.get("status") != "LOCKED_BEFORE_NEW_HOLDOUT":
        raise RuntimeError(f"FREEZE_LOCK status invalide: {lock.get('status')}")

    if lock.get("model_name") != EXPECTED_MODEL_NAME:
        raise RuntimeError("FREEZE_LOCK model_name différent du modèle pré-enregistré.")

    if lock.get("features") != EXPECTED_FEATURES:
        raise RuntimeError("FREEZE_LOCK features différentes du TOP_8 pré-enregistré.")

    if abs(float(lock.get("C")) - EXPECTED_C) > 1e-15:
        raise RuntimeError("FREEZE_LOCK C différent de 0.01.")

    if int(lock.get("calibration_window")) != EXPECTED_WINDOW:
        raise RuntimeError("FREEZE_LOCK calibration_window différent de 150.")

    if abs(float(lock.get("calibration_alpha")) - EXPECTED_ALPHA) > 1e-12:
        raise RuntimeError("FREEZE_LOCK alpha différent de l'alpha final pré-enregistré.")

    actual_model_sha = sha256_file(MODEL_PATH)
    if actual_model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "SHA256 DU MODELE DIFFERENT DU GEL PRE-ENREGISTRE.\n"
            f"Attendu: {EXPECTED_MODEL_SHA256}\n"
            f"Trouvé : {actual_model_sha}"
        )

    if lock.get("model_sha256") != actual_model_sha:
        raise RuntimeError("FREEZE_LOCK model_sha256 ne correspond pas au fichier modèle.")

    actual_schema_sha = sha256_file(SCHEMA_PATH)
    if lock.get("schema_sha256") != actual_schema_sha:
        raise RuntimeError("FREEZE_LOCK schema_sha256 ne correspond pas au schéma.")

    actual_manifest_sha = sha256_file(FREEZE_MANIFEST_PATH)
    if lock.get("manifest_sha256") != actual_manifest_sha:
        raise RuntimeError("FREEZE_LOCK manifest_sha256 ne correspond pas au manifest gelé.")

    if lock.get("do_not_modify_after_holdout_open") is not True:
        raise RuntimeError("FREEZE_LOCK ne contient pas la règle do_not_modify.")

    return {
        "lock": lock,
        "model_sha256": actual_model_sha,
        "schema_sha256": actual_schema_sha,
        "freeze_manifest_sha256": actual_manifest_sha,
        "freeze_lock_sha256": sha256_file(FREEZE_LOCK_PATH),
    }


def payload_rows(payload: Any, lab: Any) -> List[Dict[str, Any]]:
    try:
        rows = lab.as_list(payload)
        if isinstance(rows, list):
            return [dict(x) for x in rows if isinstance(x, Mapping)]
    except Exception:
        pass

    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]

    if isinstance(payload, Mapping):
        for key in ("data", "matches", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]
            if isinstance(value, Mapping):
                for sub in ("data", "matches", "results", "items"):
                    rows = value.get(sub)
                    if isinstance(rows, list):
                        return [dict(x) for x in rows if isinstance(x, Mapping)]
    return []


def unwrap_match(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("data", "match"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                if key == "data" and isinstance(nested.get("match"), Mapping):
                    return dict(nested["match"])
                return dict(nested)
        return dict(payload)
    return {}


def extract_match_id(row: Mapping[str, Any], lab: Any) -> str:
    try:
        v = clean(lab.extract_match_id(row))
        if v:
            return v
    except Exception:
        pass
    for key in ("match_id", "id", "event_id"):
        v = clean(row.get(key))
        if v:
            return v
    return ""


def extract_kickoff(row: Mapping[str, Any], lab: Any) -> Optional[pd.Timestamp]:
    try:
        value = lab.extract_match_date(row)
        ts = parse_dt(value)
        if ts is not None:
            return ts
    except Exception:
        pass
    for key in ("kickoff_utc", "utc_date", "date", "start_time", "kickoff"):
        ts = parse_dt(row.get(key))
        if ts is not None:
            return ts
    return None


def extract_competition_id(row: Mapping[str, Any], lab: Any) -> str:
    try:
        return clean(lab.extract_competition_id(row))
    except Exception:
        pass
    return clean(row.get("competition_id"))


def extract_season_id(row: Mapping[str, Any], lab: Any) -> str:
    try:
        return clean(lab.extract_season_id(row))
    except Exception:
        pass
    return clean(row.get("season_id"))


def team_name(row: Mapping[str, Any], side: str, lab: Any) -> str:
    try:
        team = lab.extract_home_team(row) if side == "home" else lab.extract_away_team(row)
        return clean(lab.extract_team_name(team))
    except Exception:
        pass

    t = row.get(f"{side}_team")
    if isinstance(t, Mapping):
        return clean(t.get("name") or t.get("team_name"))
    return clean(row.get(f"{side}_team_name") or row.get(f"{side}_name"))


def list_window_matches(
    client: Any,
    lab: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    per_page: int,
    max_pages_per_day: int,
    page_sleep: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    all_by_id: Dict[str, Dict[str, Any]] = {}
    daily_summary: List[Dict[str, Any]] = []
    listing_complete = True

    day = start.date()
    final_day = end.date()

    while day <= final_day:
        day_text = day.isoformat()
        day_seen = set()
        rows_this_day = 0
        pages_used = 0
        hit_cap_full = False

        for page in range(1, max_pages_per_day + 1):
            params = {
                "status": "finished",
                "date_from": day_text,
                "date_to": day_text,
                "page": page,
                "per_page": per_page,
            }

            payload = client.get("/football/matches", params=params)
            rows = payload_rows(payload, lab)
            pages_used = page

            if not rows:
                break

            new_on_page = 0
            for raw in rows:
                match_id = extract_match_id(raw, lab)
                kickoff = extract_kickoff(raw, lab)

                if not match_id or kickoff is None:
                    continue
                if kickoff < start or kickoff > end:
                    continue

                if match_id not in day_seen:
                    day_seen.add(match_id)
                    rows_this_day += 1

                if match_id not in all_by_id:
                    all_by_id[match_id] = dict(raw)
                    new_on_page += 1

            if len(rows) < per_page:
                break
            if new_on_page == 0 and page > 1:
                break

            if page == max_pages_per_day and len(rows) >= per_page:
                hit_cap_full = True
                listing_complete = False

            if page_sleep > 0:
                time.sleep(page_sleep)

        daily_summary.append({
            "date_utc": day_text,
            "unique_matches_in_window": rows_this_day,
            "pages_used": pages_used,
            "max_pages_per_day": max_pages_per_day,
            "page_cap_hit_while_full": int(hit_cap_full),
        })
        print(
            f"Listing {day_text} | matchs uniques={rows_this_day} | "
            f"pages={pages_used} | complete={'YES' if not hit_cap_full else 'NO'}"
        )

        day += timedelta(days=1)

    matches = list(all_by_id.values())
    matches.sort(
        key=lambda r: (
            extract_kickoff(r, lab) or pd.Timestamp.max.tz_localize("UTC"),
            extract_match_id(r, lab),
        )
    )

    return matches, daily_summary, listing_complete


def listing_dataframe(matches: Sequence[Mapping[str, Any]], lab: Any) -> pd.DataFrame:
    rows = []
    for stub in matches:
        kickoff = extract_kickoff(stub, lab)
        rows.append({
            "match_id": extract_match_id(stub, lab),
            "kickoff_utc": kickoff.isoformat() if kickoff is not None else "",
            "competition_id": extract_competition_id(stub, lab),
            "season_id": extract_season_id(stub, lab),
            "home_team": team_name(stub, "home", lab),
            "away_team": team_name(stub, "away", lab),
        })
    return pd.DataFrame(rows)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def append_replace(df: pd.DataFrame, row: Mapping[str, Any], key: str = "match_id") -> pd.DataFrame:
    new = pd.DataFrame([dict(row)])
    if df.empty:
        return new
    if key in df.columns and key in new.columns:
        value = clean(new.iloc[0][key])
        if value:
            df = df[df[key].astype(str) != value].copy()
    return pd.concat([df, new], ignore_index=True, sort=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collecte aveugle du FINAL HOLDOUT First Half 37 V2."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--min-api-interval", type=float, default=0.25)
    parser.add_argument("--page-sleep", type=float, default=0.15)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages-per-day", type=int, default=25)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="TEST UNIQUEMENT. Si >0, le manifest sera INCOMPLET et l'évaluateur refusera de scorer.",
    )
    parser.add_argument(
        "--xg-history-limit",
        type=int,
        default=0,
        help="PRE-MATCH xG historique. TOP_8 n'en a pas besoin; défaut=0.",
    )
    args = parser.parse_args()

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / "api"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / DATASET_NAME
    audit_path = output_dir / AUDIT_NAME
    rejections_path = output_dir / REJECTIONS_NAME
    listing_path = output_dir / LISTING_NAME
    manifest_path = output_dir / MANIFEST_NAME

    evaluation_lock = output_dir / "evaluation" / "FIRST_HALF37_V2_HOLDOUT_EVALUATION_LOCK.json"
    opened_lock = output_dir / "evaluation" / "FIRST_HALF37_V2_HOLDOUT_OPENED.json"
    if evaluation_lock.exists() or opened_lock.exists():
        raise RuntimeError(
            "Le holdout a déjà été ouvert/évalué. La collecte est verrouillée."
        )

    print("=" * 118)
    print("FIRST HALF 37 — V2 FINAL HOLDOUT COLLECTOR (BLIND)")
    print("=" * 118)
    print(f"Fenêtre UTC : {START_UTC} -> {END_UTC}")
    print(f"Output      : {output_dir}")
    print("Scoring     : DESACTIVE dans ce script")
    print()

    freeze = verify_freeze()
    print("FREEZE_LOCK : PASS")
    print(f"MODEL SHA256: {freeze['model_sha256']}")
    print("Pipeline    : TOP_8 + Logistic L2 C=0.01 + intercept OOF150")
    print()

    api_key = clean(os.getenv("STATS_API_KEY"))
    base_url = clean(
        os.getenv("STATS_API_BASE_URL", "https://api.thestatsapi.com/api")
    ).rstrip("/")
    timeout = int(float(os.getenv("STATS_API_TIMEOUT", "20")))

    if not api_key:
        raise RuntimeError("STATS_API_KEY absent du .env.")

    firsthalf = load_module("first_half37_holdout_runtime", FIRST_HALF_MODULE_PATH)
    prematch = load_module("prematch_holdout_runtime", PREMATCH_MODULE_PATH)
    audit = load_module("audit_holdout_runtime", AUDIT_MODULE_PATH)

    lab = prematch.load_lab()
    prematch.lab = lab
    prematch_client = prematch.build_rate_limited_client(
        lab,
        max_retries=6,
        base_wait=2.0,
        min_interval=max(0.1, args.min_api_interval),
    )

    live_base = audit.StatsAPIClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=6,
        min_interval=max(0.1, args.min_api_interval),
        backoff=2.0,
    )
    live_client = firsthalf.CachedHistoricalClient(
        base_client=live_base,
        cache_dir=cache_dir,
        legacy_cache_dir=None,
        enabled=True,
    )

    matches, day_summary, listing_complete = list_window_matches(
        client=prematch_client,
        lab=lab,
        start=START_UTC,
        end=END_UTC,
        per_page=max(1, args.per_page),
        max_pages_per_day=max(1, args.max_pages_per_day),
        page_sleep=max(0.0, args.page_sleep),
    )

    full_listing_count = len(matches)
    listing_df = listing_dataframe(matches, lab)
    atomic_csv(listing_df, listing_path)

    print()
    print(f"TOTAL MATCHS LISTES DANS LA FENETRE : {full_listing_count}")
    print(f"LISTING COMPLETE : {'YES' if listing_complete else 'NO'}")
    print()

    limited_run = args.limit > 0
    if limited_run:
        matches = matches[: args.limit]
        print(
            f"MODE TEST --limit={args.limit}: l'évaluation finale sera bloquée "
            "jusqu'à une collecte complète."
        )
        print()

    dataset_df = load_existing(dataset_path) if args.resume else pd.DataFrame()
    audit_df = load_existing(audit_path) if args.resume else pd.DataFrame()
    rejections_df = load_existing(rejections_path) if args.resume else pd.DataFrame()

    processed_ids = set()
    if args.resume and not audit_df.empty and "match_id" in audit_df.columns:
        processed_ids = set(audit_df["match_id"].dropna().astype(str))

    prematch_features = list(prematch.PRE_MATCH_FEATURES)
    attempted_this_run = 0

    try:
        for i, stub in enumerate(matches, start=1):
            match_id = extract_match_id(stub, lab)
            if not match_id:
                continue

            if args.resume and match_id in processed_ids:
                continue

            attempted_this_run += 1
            kickoff = extract_kickoff(stub, lab)

            print("-" * 118)
            print(
                f"[{i}/{len(matches)}] {match_id} | "
                f"{team_name(stub, 'home', lab)} vs {team_name(stub, 'away', lab)} | "
                f"{kickoff}"
            )

            if kickoff is None or kickoff < START_UTC or kickoff > END_UTC:
                ar = {
                    "match_id": match_id,
                    "kickoff_utc": "" if kickoff is None else kickoff.isoformat(),
                    "status": "failed",
                    "error": "kickoff_outside_preregistered_window",
                    "collected_at_utc": now_utc_iso(),
                }
                audit_df = append_replace(audit_df, ar)
                rejections_df = append_replace(rejections_df, ar)
                print("REJET | kickoff hors fenêtre")
                continue

            # Detail d'abord: permet au PRE-MATCH d'utiliser les IDs complets.
            detail_payload, detail_status, detail_error = live_client.request(
                f"/football/matches/{match_id}"
            )
            target_match = unwrap_match(detail_payload)
            if not target_match:
                target_match = dict(stub)

            # PRE-MATCH strictement avant kickoff.
            #
            # IMPORTANT:
            # build_prematch_asof() travaille avec un cutoff UTC-naive
            # (comme le script historique original via dt_utc_naive()).
            # Lui passer un datetime timezone-aware peut provoquer:
            #   TypeError: can't compare offset-naive and offset-aware datetimes
            # et transformer silencieusement toutes les 59 PRE en NaN si on
            # avale l'exception.
            prematch_cutoff = prematch.dt_utc_naive(kickoff.to_pydatetime())
            if prematch_cutoff is None:
                ar = {
                    "match_id": match_id,
                    "kickoff_utc": kickoff.isoformat(),
                    "status": "failed",
                    "error": "prematch_cutoff_unparseable",
                    "collected_at_utc": now_utc_iso(),
                }
                audit_df = append_replace(audit_df, ar)
                rejections_df = append_replace(rejections_df, ar)
                print("REJET | cutoff PRE-MATCH non parsable")
                continue

            try:
                pre_features, pre_audit = prematch.build_prematch_asof(
                    lab=lab,
                    client=prematch_client,
                    target_match=target_match,
                    cutoff=prematch_cutoff,
                    xg_limit=max(0, args.xg_history_limit),
                    with_second_half=False,
                    preferred_bookmaker=None,
                )
            except Exception as exc:
                # Un échec technique du builder PRE ne doit JAMAIS être converti
                # en 59 valeurs manquantes puis accepté dans le holdout.
                ar = {
                    "match_id": match_id,
                    "kickoff_utc": kickoff.isoformat(),
                    "status": "failed",
                    "error": f"prematch_rebuild_exception:{type(exc).__name__}:{exc}",
                    "collected_at_utc": now_utc_iso(),
                }
                audit_df = append_replace(audit_df, ar)
                rejections_df = append_replace(rejections_df, ar)
                print(f"REJET | {ar['error']}")
                continue

            pre_available = int(pre_audit.get("prematch_available", 0) or 0)
            if pre_available == 0:
                # Ce cas peut être réellement dû à une absence de données,
                # mais il doit être visible. On ne le confond plus avec une
                # exception technique (qui est rejetée ci-dessus).
                print("WARNING | PRE-MATCH reconstruit sans exception mais couverture=0/59")

            source_row = dict(target_match)
            source_row["match_id"] = match_id
            source_row["competition_id"] = (
                extract_competition_id(target_match, lab)
                or extract_competition_id(stub, lab)
            )
            source_row["season_id"] = (
                extract_season_id(target_match, lab)
                or extract_season_id(stub, lab)
            )
            source_row["home_team"] = (
                target_match.get("home_team")
                if isinstance(target_match.get("home_team"), Mapping)
                else stub.get("home_team")
            )
            source_row["away_team"] = (
                target_match.get("away_team")
                if isinstance(target_match.get("away_team"), Mapping)
                else stub.get("away_team")
            )
            source_row["kickoff_utc"] = kickoff.isoformat()
            source_row.update(pre_features)

            try:
                row, audit_row = firsthalf.collect_one_match(
                    source_row=source_row,
                    client=live_client,
                    audit=audit,
                    prematch_features=prematch_features,
                )
            except Exception as exc:
                row = None
                audit_row = {
                    "match_id": match_id,
                    "kickoff_utc": kickoff.isoformat(),
                    "status": "failed",
                    "error": f"collector_exception:{type(exc).__name__}:{exc}",
                    "collected_at_utc": now_utc_iso(),
                }

            audit_row["holdout_window_start_utc"] = START_UTC.isoformat()
            audit_row["holdout_window_end_utc"] = END_UTC.isoformat()
            audit_row["freeze_model_sha256"] = freeze["model_sha256"]
            audit_row["freeze_lock_sha256"] = freeze["freeze_lock_sha256"]
            audit_row["prematch_policy"] = "strict_asof_kickoff"
            audit_row["prematch_rebuild_error"] = clean(pre_audit.get("error"))
            audit_row["prematch_rebuild_available"] = pre_audit.get(
                "prematch_available", np.nan
            )
            audit_row["detail_prefetch_http_status"] = detail_status
            audit_row["detail_prefetch_error"] = detail_error

            audit_df = append_replace(audit_df, audit_row)

            if row is None:
                rejections_df = append_replace(rejections_df, audit_row)
                print(
                    f"REJET/PARTIAL | {audit_row.get('status')} | "
                    f"{audit_row.get('error')}"
                )
            else:
                row["schema_version"] = "first_half_37_v2_final_holdout_v1"
                row["source_dataset"] = "v2_final_holdout_post_freeze"
                row["holdout_window_start_utc"] = START_UTC.isoformat()
                row["holdout_window_end_utc"] = END_UTC.isoformat()
                row["freeze_model_sha256"] = freeze["model_sha256"]
                row["freeze_lock_sha256"] = freeze["freeze_lock_sha256"]
                row["prematch_policy"] = "strict_asof_kickoff"
                row["blind_collection"] = 1
                row["_prematch_rebuild_error"] = clean(pre_audit.get("error"))

                dataset_df = append_replace(dataset_df, row)
                print(
                    f"ACCEPTE | PRE={row.get('_prematch_available', 'n/a')}/"
                    f"{row.get('_prematch_total', 'n/a')} | "
                    f"LIVE37={row.get('_live37_available', 'n/a')}/"
                    f"{row.get('_live37_total_fixed_schema', 'n/a')} | "
                    "TARGET MASQUE"
                )

            if args.save_every > 0 and attempted_this_run % args.save_every == 0:
                atomic_csv(dataset_df, dataset_path)
                atomic_csv(audit_df, audit_path)
                atomic_csv(rejections_df, rejections_path)

    except KeyboardInterrupt:
        print()
        print("Interruption utilisateur: sauvegarde de l'état partiel...")
        atomic_csv(dataset_df, dataset_path)
        atomic_csv(audit_df, audit_path)
        atomic_csv(rejections_df, rejections_path)
        return 130

    atomic_csv(dataset_df, dataset_path)
    atomic_csv(audit_df, audit_path)
    atomic_csv(rejections_df, rejections_path)

    listed_ids = set(listing_df["match_id"].dropna().astype(str))
    audited_ids = (
        set(audit_df["match_id"].dropna().astype(str))
        if not audit_df.empty and "match_id" in audit_df.columns
        else set()
    )

    all_listed_attempted = listed_ids.issubset(audited_ids)
    collection_complete = bool(
        listing_complete
        and not limited_run
        and all_listed_attempted
    )

    dataset_sha = sha256_file(dataset_path) if dataset_path.exists() else ""
    audit_sha = sha256_file(audit_path) if audit_path.exists() else ""

    manifest = {
        "schema_version": "first_half_37_v2_final_holdout_collection_manifest_v1",
        "created_at_utc": now_utc_iso(),
        "blind_collection": True,
        "scoring_performed": False,
        "target_summary_intentionally_omitted": True,
        "window": {
            "start_utc": START_UTC.isoformat(),
            "end_utc": END_UTC.isoformat(),
            "inclusive": True,
        },
        "selection_rule": (
            "all finished matches returned by TheStatsAPI inside the preregistered "
            "UTC window; no selection by target, probability, competition or result"
        ),
        "data_rules": {
            "snapshot": "minute <= 37",
            "target": "goal_after_37_before_ht",
            "prematch": "strict_asof_kickoff",
            "missing": "missing != 0",
            "target_negative_requires_first_half_end": True,
        },
        "listing": {
            "full_listed_matches": int(full_listing_count),
            "listing_complete": bool(listing_complete),
            "daily_summary": day_summary,
            "listed_matches_file": str(listing_path),
            "listed_matches_sha256": sha256_file(listing_path),
        },
        "collection": {
            "limited_test_run": bool(limited_run),
            "limit": int(args.limit),
            "audited_unique_matches": int(len(audited_ids)),
            "accepted_rows": int(len(dataset_df)),
            "rejected_or_partial_rows": int(len(rejections_df)),
            "all_listed_attempted": bool(all_listed_attempted),
            "collection_complete": bool(collection_complete),
        },
        "freeze": {
            "model_name": EXPECTED_MODEL_NAME,
            "model_sha256": freeze["model_sha256"],
            "freeze_lock_sha256": freeze["freeze_lock_sha256"],
            "schema_sha256": freeze["schema_sha256"],
            "freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
            "C": EXPECTED_C,
            "alpha": EXPECTED_ALPHA,
            "calibration_window": EXPECTED_WINDOW,
            "features": EXPECTED_FEATURES,
        },
        "files": {
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha,
            "audit": str(audit_path),
            "audit_sha256": audit_sha,
            "rejections": str(rejections_path),
            "rejections_sha256": (
                sha256_file(rejections_path) if rejections_path.exists() else ""
            ),
        },
        "api": {
            "prematch_requests": int(getattr(prematch_client, "request_count", 0)),
            "prematch_429": int(getattr(prematch_client, "retry_429_count", 0)),
            "live_requests": int(getattr(live_client, "request_count", 0)),
            "live_429": int(getattr(live_client, "retry_429_count", 0)),
            "live_cache_hits": int(getattr(live_client, "cache_hits", 0)),
            "live_cache_misses": int(getattr(live_client, "cache_misses", 0)),
        },
    }
    atomic_json(manifest, manifest_path)

    print()
    print("=" * 118)
    print("COLLECTE FINAL HOLDOUT V2 TERMINEE")
    print("=" * 118)
    print(f"Matchs listés          : {full_listing_count}")
    print(f"Matchs audités         : {len(audited_ids)}")
    print(f"Lignes acceptées       : {len(dataset_df)}")
    print(f"Rejets/partiels        : {len(rejections_df)}")
    print(f"Listing complet        : {'YES' if listing_complete else 'NO'}")
    print(f"Collection complète    : {'YES' if collection_complete else 'NO'}")
    print(f"Dataset SHA256         : {dataset_sha}")
    print(f"Manifest               : {manifest_path}")
    print("TARGET / TAUX POSITIF  : MASQUES JUSQU'A L'EVALUATION")
    print("MODEL SCORING           : NON EFFECTUE")
    print()

    if not collection_complete:
        print(
            "ATTENTION: l'évaluateur aveugle refusera ce dataset tant que "
            "collection_complete != true."
        )
        return 3

    print(
        "Le holdout est prêt. Ne pas ouvrir/modifier le CSV; "
        "lancer directement l'évaluateur aveugle."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
