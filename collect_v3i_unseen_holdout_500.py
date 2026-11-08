#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_v3i_unseen_holdout_500.py

V3-I — COLLECTE DE 500 MATCHS HISTORIQUES INÉDITS
=================================================

But
---
Construire un HOLDOUT séparé de 500 matchs terminés qui :

1) ne sont présents dans AUCUN des fichiers du dataset historique principal ;
2) sont, par défaut, postérieurs au 13 septembre 2026 ;
3) sont reconstruits avec le moteur historique déjà validé du projet ;
4) possèdent les 38/38 features exactes de V3-C ;
5) possèdent un target goal_after_87 connu (0/1) ;
6) ne sont JAMAIS ajoutés au dataset d'entraînement principal ;
7) ne sont scorés par AUCUN modèle pendant la collecte.

Le script sert à constituer le holdout externe destiné à départager plus tard :
    - V3-C production
    - V3H_ET_LEAF5_PLATT challenger

IMPORTANT
---------
Le dossier principal `outputs/historical_dataset_v2/` est LECTURE SEULE pour ce
script. Toutes les écritures vont dans :

    outputs/v3i_unseen_holdout_500/

Le script s'arrête lorsque 500 lignes V3-C-ready + target connu ont été obtenues.
Les matchs partiels/échoués peuvent être conservés dans le RAW séparé, mais ils
ne comptent pas dans les 500.

Dépendances projet
------------------
Ce fichier doit être placé à la racine du projet, à côté de :
    collect_historical_dataset_v2.py
    collect_historical_dataset_v2_hierarchical.py
    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py

Commande recommandée
--------------------
    cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
    caffeinate -dimsu python3 -u collect_v3i_unseen_holdout_500.py

Le script est resumable : relancer exactement la même commande continue la
collecte sans recollecter les matchs déjà tentés.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

HIERARCHICAL_COLLECTOR_PATH = (
    BASE_DIR / "collect_historical_dataset_v2_hierarchical.py"
)

MASTER_DATASET_DIR = (
    BASE_DIR / "outputs" / "historical_dataset_v2"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "outputs" / "v3i_unseen_holdout_500"
)

DEFAULT_CACHE_DIR = (
    MASTER_DATASET_DIR / "api_cache_v2"
)

DEFAULT_V3C_SCHEMA = (
    BASE_DIR
    / "outputs"
    / "model_v3_experiments"
    / "schemas"
    / "V3_C_AGGRESSIVE_schema.json"
)

DEFAULT_CORE_SCHEMA = (
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
    / "model_schema_v2.json"
)

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"
TARGET_USABLE_DEFAULT = 500
DEFAULT_DATE_FROM = "2026-09-14"

CANDIDATE_MANIFEST = "v3i_candidate_manifest.csv"
COLLECTION_MANIFEST = "v3i_collection_manifest.json"
READY_FILENAME = "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv"
BLIND_FILENAME = "V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv"
TARGETS_FILENAME = "V3I_UNSEEN_HOLDOUT_500_TARGETS.csv"
OVERLAP_FILENAME = "v3i_overlap_audit.csv"
REPORT_FILENAME = "v3i_collection_report.md"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = clean_text(value).lower()
    return text in {"", "nan", "none", "null", "na", "n/a"}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value: Optional[str], default: Path) -> Path:
    if not value:
        return default
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


# =============================================================================
# LOAD PROJECT MODULE
# =============================================================================

def load_hierarchical_module():
    if not HIERARCHICAL_COLLECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Collecteur hiérarchique introuvable : {HIERARCHICAL_COLLECTOR_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "v3i_hierarchical_runtime",
        HIERARCHICAL_COLLECTOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {HIERARCHICAL_COLLECTOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["v3i_hierarchical_runtime"] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# ROBUST ID EXTRACTION — MASTER DATASET IS READ-ONLY
# =============================================================================

def detect_header_and_delimiter(path: Path, id_candidates: Sequence[str] = ("match_id", "fixture_id", "id")):
    delimiters = ["\t", ",", ";", "|"]
    wanted = {x.lower() for x in id_candidates}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        lines = []
        for _ in range(80):
            line = fh.readline()
            if line == "":
                break
            lines.append(line.rstrip("\r\n"))

    for line_no, line in enumerate(lines, start=1):
        for delim in delimiters:
            try:
                row = next(csv.reader([line], delimiter=delim))
            except Exception:
                continue
            header = [clean_text(x) for x in row]
            normalized = [x.lower() for x in header]
            if any(x in wanted for x in normalized):
                return delim, line_no, header

    raise ValueError(f"Header match_id/fixture_id/id introuvable dans {path}")


def read_ids_robust(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return ids

    try:
        delim, header_line, header = detect_header_and_delimiter(path)
    except Exception:
        return ids

    lowered = [x.lower() for x in header]
    id_idx = None
    for candidate in ("match_id", "fixture_id", "id"):
        if candidate in lowered:
            id_idx = lowered.index(candidate)
            break
    if id_idx is None:
        return ids

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for _ in range(header_line - 1):
            fh.readline()
        reader = csv.reader(fh, delimiter=delim)
        try:
            next(reader)
        except StopIteration:
            return ids

        for row in reader:
            if not row or id_idx >= len(row):
                continue
            match_id = clean_text(row[id_idx])
            if match_id and match_id.lower() not in {"match_id", "fixture_id", "id"}:
                ids.add(match_id)

    return ids


def load_master_exclusion_ids(master_dir: Path) -> Tuple[Set[str], List[Dict[str, Any]]]:
    candidates = [
        master_dir / "historical_matches_index_v2.csv",
        master_dir / "Historical_Dataset_Goal_After_87_RAW_v2.csv",
        master_dir / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
        master_dir / "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
        master_dir / "historical_dataset_v2_rejections.csv",
    ]

    all_ids: Set[str] = set()
    audit: List[Dict[str, Any]] = []

    for path in candidates:
        ids = read_ids_robust(path)
        all_ids.update(ids)
        audit.append({
            "path": str(path),
            "exists": path.exists(),
            "ids_found": len(ids),
            "sha256": file_sha256(path),
        })

    return all_ids, audit


# =============================================================================
# V3-C SCHEMA
# =============================================================================

def _find_feature_lists(obj: Any, path: str = "") -> List[Tuple[str, List[str]]]:
    found: List[Tuple[str, List[str]]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                found.append((child_path, [clean_text(x) for x in value]))
            found.extend(_find_feature_lists(value, child_path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_find_feature_lists(value, f"{path}[{i}]"))
    return found


def load_v3c_features(schema_path: Path) -> List[str]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema V3-C introuvable : {schema_path}")

    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    preferred_keys = [
        "features",
        "features_used",
        "feature_names",
        "model_features",
        "core_features",
    ]

    if isinstance(payload, Mapping):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                features = [clean_text(x) for x in value if clean_text(x)]
                if len(features) == 38:
                    return features

    candidates = _find_feature_lists(payload)
    exact_38 = [(p, xs) for p, xs in candidates if len(xs) == 38]
    if len(exact_38) == 1:
        return exact_38[0][1]
    if len(exact_38) > 1:
        # Prefer paths whose names mention features.
        featureish = [(p, xs) for p, xs in exact_38 if "feature" in p.lower()]
        if len(featureish) == 1:
            return featureish[0][1]

    details = ", ".join(f"{p}:{len(xs)}" for p, xs in candidates[:20])
    raise ValueError(
        "Impossible d'identifier exactement les 38 features V3-C dans le schema. "
        f"Listes trouvées: {details}"
    )


# =============================================================================
# FOOTBALL-ONLY FILTER (CONSERVATIVE)
# =============================================================================

def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ñ": "n", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NON_FOOTBALL_HINTS = {
    "handball", "handbold", "haandbold", "handebol", "handboll", "håndbold",
    "basketball", "nba", "wnba", "euroleague",
    "ice hockey", "hockey", "nhl",
    "volleyball", "volley",
    "tennis", " atp ", " wta ",
    "baseball", "mlb",
    "american football", "nfl", "rugby", "cricket",
    "water polo", "futsal", "esports", "e sports",
}


def stringify_item(item: Mapping[str, Any]) -> str:
    try:
        return normalize_text(json.dumps(dict(item), ensure_ascii=False, default=str))
    except Exception:
        return normalize_text(str(item))


def clearly_non_football(item: Mapping[str, Any]) -> bool:
    text = " " + stringify_item(item) + " "
    return any(normalize_text(hint) in text for hint in NON_FOOTBALL_HINTS)


# =============================================================================
# CANDIDATE DISCOVERY
# =============================================================================

def candidate_metadata(hier: Any, collector: Any, stub: Mapping[str, Any]) -> Dict[str, Any]:
    match_id = hier.extract_match_id(stub)
    competition_id = hier.extract_competition_id(stub)
    season_id = hier.extract_season_id(stub)
    kickoff = clean_text(collector.extract_match_date_generic(stub))

    # Team names are only for audit/readability; collection engine resolves the true row later.
    home = ""
    away = ""
    for key in ("home_team_name", "home_name"):
        if stub.get(key) is not None:
            home = clean_text(stub.get(key))
            break
    for key in ("away_team_name", "away_name"):
        if stub.get(key) is not None:
            away = clean_text(stub.get(key))
            break

    return {
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "kickoff_utc": kickoff,
        "home_team_stub": home,
        "away_team_stub": away,
        "stub_json": json.dumps(dict(stub), ensure_ascii=False, default=str, separators=(",", ":")),
    }


def kickoff_sort_key(value: Any) -> Tuple[int, int]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return (1, 0)
    return (0, int(ts.value))


def discover_candidates(
    hier: Any,
    collector: Any,
    client: Any,
    master_ids: Set[str],
    own_ids: Set[str],
    competition_ids: Optional[Sequence[str]],
    season_ids: Optional[Sequence[str]],
    max_competitions: int,
    max_seasons_per_competition: int,
    max_pages_competitions: int,
    max_pages_seasons: int,
    max_pages_per_season: int,
    date_from: Optional[str],
    date_to: Optional[str],
    sleep_between_pages: float,
) -> pd.DataFrame:
    if competition_ids:
        competitions = hier.build_manual_competitions(competition_ids)
    else:
        competitions = hier.list_competitions(
            client=client,
            limit=max_competitions,
            max_pages=max_pages_competitions,
            page_sleep=sleep_between_pages,
        )

    competitions = [c for c in competitions if not clearly_non_football(c)]
    print(f"Compétitions football candidates : {len(competitions)}")

    found: Dict[str, Dict[str, Any]] = {}

    for comp_i, competition in enumerate(competitions, start=1):
        competition_id = (
            clean_text(competition.get("id"))
            or clean_text(competition.get("competition_id"))
        )
        if not competition_id:
            continue

        comp_name = hier.extract_name(competition)
        print(f"[DISCOVERY {comp_i}/{len(competitions)}] {competition_id} — {comp_name}")

        if season_ids:
            seasons = hier.build_manual_seasons(season_ids)
        else:
            seasons = hier.list_seasons_for_competition(
                client=client,
                competition_id=competition_id,
                max_pages=max_pages_seasons,
                page_sleep=sleep_between_pages,
            )
            if max_seasons_per_competition > 0:
                seasons = seasons[:max_seasons_per_competition]

        if not seasons:
            seasons = [{"id": "", "name": "NO_SEASON_SCOPE"}]

        for season in seasons:
            season_id = clean_text(season.get("id") or season.get("season_id"))
            stubs = hier.list_finished_matches_for_scope(
                client=client,
                competition_id=competition_id,
                season_id=season_id or None,
                date_from=date_from,
                date_to=date_to,
                max_pages=max_pages_per_season,
                page_sleep=sleep_between_pages,
            )

            for original_stub in stubs:
                if clearly_non_football(original_stub):
                    continue

                stub = dict(original_stub)
                stub.setdefault("competition_id", competition_id)
                if season_id:
                    stub.setdefault("season_id", season_id)

                match_id = hier.extract_match_id(stub)
                if not match_id:
                    continue
                if match_id in master_ids or match_id in own_ids or match_id in found:
                    continue

                meta = candidate_metadata(hier, collector, stub)

                # Local date guard. If the API ignored date_from/date_to but a parsable
                # kickoff exists, enforce the requested window ourselves.
                ts = pd.to_datetime(meta["kickoff_utc"], utc=True, errors="coerce")
                if not pd.isna(ts):
                    if date_from:
                        lo = pd.to_datetime(date_from, utc=True, errors="coerce")
                        if not pd.isna(lo) and ts < lo:
                            continue
                    if date_to:
                        hi = pd.to_datetime(date_to, utc=True, errors="coerce")
                        if not pd.isna(hi):
                            # Treat a date-only --date-to as inclusive through end-of-day.
                            if len(clean_text(date_to)) <= 10:
                                hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                            if ts > hi:
                                continue

                found[match_id] = meta

    rows = list(found.values())
    rows.sort(key=lambda r: (kickoff_sort_key(r.get("kickoff_utc")), r.get("match_id", "")))

    for i, row in enumerate(rows, start=1):
        row["selection_order"] = i
        row["discovery_status"] = "UNSEEN_AT_DISCOVERY"

    return pd.DataFrame(rows)


# =============================================================================
# READY / AUDIT EXPORTS
# =============================================================================

def usable_mask(df: pd.DataFrame, v3c_features: Sequence[str], target_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    missing_cols = [c for c in v3c_features if c not in df.columns]
    if missing_cols:
        return pd.Series(False, index=df.index)

    feature_ok = pd.Series(True, index=df.index)
    for col in v3c_features:
        feature_ok &= ~df[col].map(is_missing)

    if target_col not in df.columns:
        target_ok = pd.Series(False, index=df.index)
    else:
        target_num = pd.to_numeric(df[target_col], errors="coerce")
        target_ok = target_num.isin([0, 1])

    id_ok = (
        ~df[ID_COLUMN].map(is_missing)
        if ID_COLUMN in df.columns
        else pd.Series(False, index=df.index)
    )

    return feature_ok & target_ok & id_ok


def add_selection_order(df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toujours la colonne d'ordre, même si df contient 0 ligne.

    Important : après un premier match non READY, `ready` peut être un DataFrame
    vide mais avec les colonnes du dataset. La version précédente retournait alors
    immédiatement sans créer `_v3i_selection_order`, puis `sort_values()` levait
    KeyError.
    """
    out = df.copy()

    order_map = {}
    if (
        not candidates_df.empty
        and "match_id" in candidates_df.columns
        and "selection_order" in candidates_df.columns
    ):
        order_map = dict(
            zip(
                candidates_df["match_id"].astype(str),
                pd.to_numeric(candidates_df["selection_order"], errors="coerce"),
            )
        )

    if ID_COLUMN in out.columns:
        out["_v3i_selection_order"] = out[ID_COLUMN].astype(str).map(order_map)
    else:
        # Colonne vide mais présente : garantit que le tri en aval est toujours valide.
        out["_v3i_selection_order"] = pd.Series(index=out.index, dtype="float64")

    return out


def export_holdout_views(
    dataset_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    v3c_features: Sequence[str],
    target_usable: int,
    master_ids: Set[str],
    output_dir: Path,
) -> Tuple[pd.DataFrame, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_df.empty:
        ready = dataset_df.copy()
    else:
        mask = usable_mask(dataset_df, v3c_features, TARGET_COLUMN)
        ready = dataset_df.loc[mask].copy()
        ready = add_selection_order(ready, candidates_df)
        ready = ready.sort_values(
            by=["_v3i_selection_order", ID_COLUMN],
            kind="stable",
            na_position="last",
        )
        ready = ready.drop_duplicates(subset=[ID_COLUMN], keep="first")
        ready = ready.head(target_usable).copy()

    overlap_ids = []
    if not ready.empty:
        overlap_ids = sorted(set(ready[ID_COLUMN].astype(str)) & master_ids)

    overlap_df = pd.DataFrame({"match_id": overlap_ids})
    overlap_df.to_csv(output_dir / OVERLAP_FILENAME, index=False, encoding="utf-8-sig")

    if overlap_ids:
        raise RuntimeError(
            f"SECURITY STOP: {len(overlap_ids)} match(s) du holdout existent maintenant "
            "dans le dataset principal. Aucun export READY n'est autorisé. "
            f"Exemples: {overlap_ids[:10]}"
        )

    ready.to_csv(output_dir / READY_FILENAME, index=False, encoding="utf-8-sig")

    blind = ready.drop(columns=[TARGET_COLUMN], errors="ignore")
    blind.to_csv(output_dir / BLIND_FILENAME, index=False, encoding="utf-8-sig")

    if ready.empty:
        targets = pd.DataFrame(columns=[ID_COLUMN, TARGET_COLUMN])
    else:
        target_cols = [c for c in [ID_COLUMN, "kickoff_utc", "home_team", "away_team", TARGET_COLUMN] if c in ready.columns]
        targets = ready[target_cols].copy()
    targets.to_csv(output_dir / TARGETS_FILENAME, index=False, encoding="utf-8-sig")

    return ready, len(overlap_ids)


def write_report(
    output_dir: Path,
    target_usable: int,
    ready_n: int,
    raw_n: int,
    candidate_n: int,
    master_id_n: int,
    attempted_this_run: int,
    skipped_master_this_run: int,
    v3c_features: Sequence[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> None:
    lines = [
        "# V3-I — Unseen Historical Holdout Collection",
        "",
        "## Etat",
        "",
        f"- Objectif V3-C-ready + target : **{target_usable}**",
        f"- READY actuel : **{ready_n}**",
        f"- RAW séparé : **{raw_n}**",
        f"- Candidats gelés : **{candidate_n}**",
        f"- IDs exclus du dataset principal : **{master_id_n}**",
        f"- Matchs tentés cette exécution : **{attempted_this_run}**",
        f"- Conflits master ignorés cette exécution : **{skipped_master_this_run}**",
        f"- Features V3-C exigées : **{len(v3c_features)}/38**",
        f"- Fenêtre : `{date_from or 'aucun minimum'}` → `{date_to or 'aucune borne haute explicite'}`",
        "",
        "## Règles anti-contamination",
        "",
        "- Le dataset principal est lecture seule.",
        "- Tout match_id déjà présent dans index / RAW / CORE_READY principal est exclu.",
        "- Les candidats sont ordonnés avant utilisation du target.",
        "- Le target n'entre jamais dans le choix d'un candidat, sauf pour vérifier qu'il est connu (0/1).",
        "- Aucun modèle n'est chargé ou scoré pendant cette collecte.",
        "- Le fichier READY est refusé si un overlap avec le dataset principal est détecté.",
        "",
        "## Fichiers",
        "",
        f"- `{READY_FILENAME}` : 500 lignes max, 38/38 + target.",
        f"- `{BLIND_FILENAME}` : même holdout sans target.",
        f"- `{TARGETS_FILENAME}` : targets séparés.",
        f"- `{CANDIDATE_MANIFEST}` : ordre de sélection gelé.",
        f"- `{OVERLAP_FILENAME}` : doit rester vide.",
        "",
        "Ne pas réentraîner ni retuner V3-C/V3-H à partir de ce holdout avant la comparaison finale.",
    ]
    (output_dir / REPORT_FILENAME).write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collecte 500 matchs terminés inédits pour holdout V3-I."
    )

    p.add_argument("--target-usable", type=int, default=TARGET_USABLE_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=2500,
                   help="Max de nouveaux matchs à tenter sur une exécution. Défaut: 2500.")
    p.add_argument("--date-from", default=DEFAULT_DATE_FROM,
                   help="Défaut 2026-09-14 : uniquement après l'univers historique déjà utilisé.")
    p.add_argument("--date-to", default=None,
                   help="Optionnel. Seuls les matchs status=finished sont listés de toute façon.")
    p.add_argument("--competition-ids", nargs="*", default=None)
    p.add_argument("--season-ids", nargs="*", default=None)
    p.add_argument("--max-competitions", type=int, default=50)
    p.add_argument("--max-seasons-per-competition", type=int, default=8)
    p.add_argument("--max-pages-competitions", type=int, default=10)
    p.add_argument("--max-pages-seasons", type=int, default=5)
    p.add_argument("--max-pages-per-season", type=int, default=20)
    p.add_argument("--sleep-between-matches", type=float, default=0.35)
    p.add_argument("--sleep-between-pages", type=float, default=0.25)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--xg-limit", type=int, default=5)
    p.add_argument("--with-league-second-half", action="store_true")
    p.add_argument("--preferred-bookmaker", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--master-dir", default=None)
    p.add_argument("--v3c-schema", default=None)
    p.add_argument("--core-schema", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--cache-disable", action="store_true")
    p.add_argument("--cache-save-none", action="store_true")
    p.add_argument("--cache-ttl-hours", type=float, default=0.0)
    p.add_argument("--cache-clear", action="store_true")
    p.add_argument("--rediscover-candidates", action="store_true",
                   help="Supprime et reconstruit le manifest candidat. A éviter après début de collecte.")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()

    if args.target_usable <= 0:
        raise ValueError("--target-usable doit être > 0")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts doit être > 0")

    master_dir = resolve_path(args.master_dir, MASTER_DATASET_DIR)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    v3c_schema = resolve_path(args.v3c_schema, DEFAULT_V3C_SCHEMA)
    core_schema = resolve_path(args.core_schema, DEFAULT_CORE_SCHEMA)
    cache_dir = resolve_path(args.cache_dir, DEFAULT_CACHE_DIR)

    # Hard safety: no chance of writing into the training dataset directory.
    if output_dir.resolve() == master_dir.resolve():
        raise RuntimeError(
            "SECURITY STOP: output-dir pointe vers le dataset historique principal. "
            "Utilise un dossier séparé."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_clear and cache_dir.exists():
        print(f"Suppression cache : {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    env_path = BASE_DIR / ".env"
    load_dotenv(env_path if env_path.exists() else None)

    print()
    print("=" * 112)
    print("V3-I — COLLECTION 500 MATCHS HISTORIQUES INÉDITS")
    print("=" * 112)
    print(f"Master READ-ONLY : {master_dir}")
    print(f"Sorties séparées : {output_dir}")
    print(f"Fenêtre           : {args.date_from or '-∞'} -> {args.date_to or 'finished/latest'}")
    print(f"Objectif          : {args.target_usable} matchs V3-C 38/38 + target connu")

    hier = load_hierarchical_module()
    collector = hier.load_original_collector()

    v3c_features = load_v3c_features(v3c_schema)
    if len(v3c_features) != 38:
        raise RuntimeError(f"V3-C doit contenir 38 features, trouvé {len(v3c_features)}")

    core_features = collector.load_core_features(str(core_schema))

    print(f"Schema V3-C       : {v3c_schema} ({len(v3c_features)} features)")
    print(f"Schema collecte   : {core_schema} ({len(core_features)} CORE features)")
    print(f"Schema version    : {collector.SCHEMA_VERSION}")

    master_ids, master_sources_audit = load_master_exclusion_ids(master_dir)
    print(f"IDs master exclus : {len(master_ids):,}")

    prematch, live87, client = hier.prepare_runtime_modules(
        collector=collector,
        cache_dir=cache_dir,
        cache_enabled=not args.cache_disable,
        cache_save_none=args.cache_save_none,
        cache_ttl_hours=args.cache_ttl_hours,
    )

    print(f"Cache activé      : {not args.cache_disable}")
    print(f"Cache dir         : {cache_dir}")

    raw_dataset_path = output_dir / "Historical_Dataset_Goal_After_87_RAW_v2.csv"
    index_path = output_dir / "historical_matches_index_v2.csv"
    rejections_path = output_dir / "historical_dataset_v2_rejections.csv"

    dataset_df = collector.load_existing_dataset(raw_dataset_path)
    index_df = collector.load_index(index_path)
    rejection_rows = hier.load_rejections(rejections_path, force=False)

    own_ids = set()
    if not index_df.empty and ID_COLUMN in index_df.columns:
        own_ids.update(index_df[ID_COLUMN].astype(str).str.strip().tolist())
    if not dataset_df.empty and ID_COLUMN in dataset_df.columns:
        own_ids.update(dataset_df[ID_COLUMN].astype(str).str.strip().tolist())
    own_ids.discard("")

    print(f"IDs déjà tentés V3-I : {len(own_ids):,}")

    candidate_path = output_dir / CANDIDATE_MANIFEST

    if args.rediscover_candidates:
        if not dataset_df.empty or not index_df.empty:
            raise RuntimeError(
                "Refus de --rediscover-candidates : la collecte V3-I a déjà commencé. "
                "L'ordre des candidats doit rester gelé."
            )
        candidate_path.unlink(missing_ok=True)

    if candidate_path.exists():
        candidates_df = pd.read_csv(candidate_path, low_memory=False)
        required_cols = {"selection_order", "match_id", "stub_json"}
        if not required_cols.issubset(set(candidates_df.columns)):
            raise RuntimeError(f"Manifest candidat invalide : {candidate_path}")
        print(f"Manifest candidat chargé : {len(candidates_df):,} matchs")
    else:
        print()
        print("Découverte des matchs terminés inédits...")
        candidates_df = discover_candidates(
            hier=hier,
            collector=collector,
            client=client,
            master_ids=master_ids,
            own_ids=own_ids,
            competition_ids=args.competition_ids,
            season_ids=args.season_ids,
            max_competitions=max(1, args.max_competitions),
            max_seasons_per_competition=max(1, args.max_seasons_per_competition),
            max_pages_competitions=max(1, args.max_pages_competitions),
            max_pages_seasons=max(1, args.max_pages_seasons),
            max_pages_per_season=max(1, args.max_pages_per_season),
            date_from=args.date_from,
            date_to=args.date_to,
            sleep_between_pages=max(0.0, args.sleep_between_pages),
        )
        if candidates_df.empty:
            raise RuntimeError(
                "Aucun match inédit trouvé. Elargis --max-competitions / --max-pages-per-season "
                "ou recule --date-from."
            )
        candidates_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
        print(f"Manifest candidat GELÉ : {len(candidates_df):,} matchs -> {candidate_path}")

    # Audit manifest against current master before collection.
    manifest_ids = set(candidates_df["match_id"].astype(str).str.strip())
    manifest_master_overlap = sorted(manifest_ids & master_ids)
    if manifest_master_overlap:
        # This can happen on resume if the master dataset has been expanded since the
        # candidate manifest was frozen. We skip those rows rather than contaminate V3-I.
        print(
            f"ATTENTION : {len(manifest_master_overlap)} candidat(s) sont désormais dans le master; "
            "ils seront ignorés."
        )

    ready_df, _ = export_holdout_views(
        dataset_df=dataset_df,
        candidates_df=candidates_df,
        v3c_features=v3c_features,
        target_usable=args.target_usable,
        master_ids=master_ids,
        output_dir=output_dir,
    )
    ready_ids = set(ready_df[ID_COLUMN].astype(str)) if not ready_df.empty else set()

    print(f"READY avant run    : {len(ready_df):,}/{args.target_usable}")

    if len(ready_df) >= args.target_usable:
        print("Objectif déjà atteint. Aucune nouvelle requête nécessaire.")
        write_report(
            output_dir, args.target_usable, len(ready_df), len(dataset_df),
            len(candidates_df), len(master_ids), 0, 0, v3c_features,
            args.date_from, args.date_to,
        )
        client.print_stats()
        return 0

    attempted_this_run = 0
    skipped_master_this_run = 0
    new_usable_this_run = 0

    # Current output IDs: never retry on normal resume; continue further down frozen order.
    output_seen_ids = set(own_ids)

    try:
        for _, candidate in candidates_df.sort_values("selection_order", kind="stable").iterrows():
            if len(ready_ids) >= args.target_usable:
                break
            if attempted_this_run >= args.max_attempts:
                break

            match_id = clean_text(candidate.get("match_id"))
            if not match_id:
                continue
            if match_id in master_ids:
                skipped_master_this_run += 1
                continue
            if match_id in output_seen_ids:
                continue

            try:
                stub = json.loads(clean_text(candidate.get("stub_json")))
            except Exception as exc:
                print(f"SKIP {match_id}: stub_json invalide ({exc})")
                continue

            print()
            print(
                f"[V3-I TRY {attempted_this_run + 1}/{args.max_attempts}] "
                f"selection={int(candidate['selection_order'])} | {match_id}"
            )

            before_rows = len(dataset_df)

            index_df, dataset_df, rejection_rows, was_processed = hier.collect_match_stub(
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
                force=False,
                xg_limit=max(0, args.xg_limit),
                with_league_second_half=args.with_league_second_half,
                preferred_bookmaker=args.preferred_bookmaker,
            )

            if not was_processed:
                output_seen_ids.add(match_id)
                continue

            attempted_this_run += 1
            output_seen_ids.add(match_id)

            # Recompute READY on the separated dataframe only. This does not score any model.
            refreshed_ready, _ = export_holdout_views(
                dataset_df=dataset_df,
                candidates_df=candidates_df,
                v3c_features=v3c_features,
                target_usable=args.target_usable,
                master_ids=master_ids,
                output_dir=output_dir,
            )
            refreshed_ids = set(refreshed_ready[ID_COLUMN].astype(str)) if not refreshed_ready.empty else set()

            if match_id in refreshed_ids and match_id not in ready_ids:
                new_usable_this_run += 1
                print(
                    f"  -> V3-I READY 38/38 + target | "
                    f"progress={len(refreshed_ready)}/{args.target_usable}"
                )
            else:
                print(
                    f"  -> ne compte pas encore dans les 500 | "
                    f"progress={len(refreshed_ready)}/{args.target_usable}"
                )

            ready_df = refreshed_ready
            ready_ids = refreshed_ids

            if args.save_every > 0 and attempted_this_run % args.save_every == 0:
                collector.save_all(
                    dataset_df=dataset_df,
                    index_df=index_df,
                    rejection_rows=rejection_rows,
                    output_dir=output_dir,
                    core_features=core_features,
                )

                manifest_payload = {
                    "updated_at_utc": now_utc_iso(),
                    "status": "COLLECTING" if len(ready_df) < args.target_usable else "COMPLETE",
                    "target_usable": args.target_usable,
                    "ready_rows": len(ready_df),
                    "raw_rows": len(dataset_df),
                    "attempted_this_run": attempted_this_run,
                    "new_usable_this_run": new_usable_this_run,
                    "candidate_rows": len(candidates_df),
                    "master_ids_excluded": len(master_ids),
                    "date_from": args.date_from,
                    "date_to": args.date_to,
                    "v3c_schema": str(v3c_schema),
                    "v3c_schema_sha256": file_sha256(v3c_schema),
                    "v3c_feature_count": len(v3c_features),
                    "master_sources": master_sources_audit,
                }
                write_json(output_dir / COLLECTION_MANIFEST, manifest_payload)

            if args.sleep_between_matches > 0:
                time.sleep(args.sleep_between_matches)

    except KeyboardInterrupt:
        print()
        print("Interruption utilisateur — sauvegarde V3-I...")

    # Final save no matter whether complete / interrupted / exhausted.
    collector.save_all(
        dataset_df=dataset_df,
        index_df=index_df,
        rejection_rows=rejection_rows,
        output_dir=output_dir,
        core_features=core_features,
    )

    master_ids_final, master_sources_final = load_master_exclusion_ids(master_dir)
    ready_df, overlap_n = export_holdout_views(
        dataset_df=dataset_df,
        candidates_df=candidates_df,
        v3c_features=v3c_features,
        target_usable=args.target_usable,
        master_ids=master_ids_final,
        output_dir=output_dir,
    )

    complete = len(ready_df) >= args.target_usable

    final_manifest = {
        "created_or_updated_at_utc": now_utc_iso(),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "purpose": "Frozen unseen historical/future holdout for V3-C vs V3H_ET_LEAF5_PLATT",
        "target_usable": args.target_usable,
        "ready_rows": len(ready_df),
        "raw_rows": len(dataset_df),
        "index_rows": len(index_df),
        "candidate_rows": len(candidates_df),
        "attempted_this_run": attempted_this_run,
        "new_usable_this_run": new_usable_this_run,
        "master_conflicts_skipped_this_run": skipped_master_this_run,
        "master_ids_excluded_final": len(master_ids_final),
        "overlap_ready_vs_master": overlap_n,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "v3c_schema": str(v3c_schema),
        "v3c_schema_sha256": file_sha256(v3c_schema),
        "v3c_features": list(v3c_features),
        "v3c_feature_count": len(v3c_features),
        "core_schema": str(core_schema),
        "collector_schema_version": collector.SCHEMA_VERSION,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "master_sources": master_sources_final,
        "files": {
            "ready": str(output_dir / READY_FILENAME),
            "blind": str(output_dir / BLIND_FILENAME),
            "targets": str(output_dir / TARGETS_FILENAME),
            "overlap_audit": str(output_dir / OVERLAP_FILENAME),
            "raw": str(raw_dataset_path),
            "index": str(index_path),
        },
        "anti_contamination": {
            "master_is_read_only": True,
            "model_scoring_during_collection": False,
            "selection_uses_target_value": False,
            "ready_requires_target_known_only": True,
            "ready_requires_v3c_38_of_38": True,
        },
    }
    write_json(output_dir / COLLECTION_MANIFEST, final_manifest)

    write_report(
        output_dir=output_dir,
        target_usable=args.target_usable,
        ready_n=len(ready_df),
        raw_n=len(dataset_df),
        candidate_n=len(candidates_df),
        master_id_n=len(master_ids_final),
        attempted_this_run=attempted_this_run,
        skipped_master_this_run=skipped_master_this_run,
        v3c_features=v3c_features,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    print()
    print("=" * 112)
    print("RÉSUMÉ V3-I COLLECTION")
    print("=" * 112)
    print(f"READY V3-C 38/38 + target : {len(ready_df):,}/{args.target_usable}")
    print(f"RAW séparé                  : {len(dataset_df):,}")
    print(f"Candidats gelés             : {len(candidates_df):,}")
    print(f"Tentés cette exécution      : {attempted_this_run:,}")
    print(f"Nouveaux READY cette run    : {new_usable_this_run:,}")
    print(f"Overlap avec master         : {overlap_n} (DOIT être 0)")
    print(f"Statut                      : {'COMPLETE' if complete else 'INCOMPLETE'}")
    print(f"READY                       : {output_dir / READY_FILENAME}")
    print(f"BLIND                       : {output_dir / BLIND_FILENAME}")
    print(f"TARGETS                     : {output_dir / TARGETS_FILENAME}")
    print(f"Manifest                    : {output_dir / COLLECTION_MANIFEST}")
    print()
    print("Le dataset historique principal et le scanner de production n'ont pas été modifiés.")

    client.print_stats()

    if complete:
        return 0

    print()
    print(
        "Objectif non atteint avec les candidats/tentatives disponibles. "
        "Relance la même commande; si le manifest est épuisé, crée un nouveau dossier "
        "V3-I ou élargis la fenêtre avant toute nouvelle validation."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
