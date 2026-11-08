#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3-I BATCH 2 — COLLECTE ÉLARGIE ET SÉPARÉE (221)
=================================================

Pourquoi ce script
------------------
Le collecteur V3-I initial a déjà épuisé ses 589 candidats issus des 50
premières compétitions. Une nouvelle découverte sur ces mêmes 50 compétitions
n'a trouvé qu'un seul match inédit, qui n'était pas V3-C 38/38.

Ce script NE TOUCHE PAS au Batch 1 de 279 matchs.

Il :
1) lance le collecteur hiérarchique football-only dans un dossier SÉPARÉ ;
2) explore jusqu'à 300 compétitions ;
3) garde la fenêtre >= 2026-09-14 ;
4) exclut tous les IDs du master historique ;
5) exclut tous les IDs du Batch 1 ;
6) exige V3-C exact 38/38 + target connu ;
7) ne fige le Batch 2 qu'une fois 221 matchs valides disponibles.

Le choix des 221 est déterministe (kickoff_utc, match_id) et ne dépend jamais
du target ni des probabilités modèles.

Aucun modèle n'est scoré pendant la collecte.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

FOOTBALL_COLLECTOR = (
    ROOT / "collect_historical_dataset_v2_hierarchical_football_only.py"
)

SOURCE_DIR = (
    ROOT / "outputs/v3i_unseen_holdout_500/batch2_hierarchical_source"
)

SOURCE_RAW = SOURCE_DIR / "Historical_Dataset_Goal_After_87_RAW_v2.csv"

MASTER_DIR = ROOT / "outputs/historical_dataset_v2"
MASTER_INDEX = MASTER_DIR / "historical_matches_index_v2.csv"
MASTER_READY = (
    MASTER_DIR / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
)

BATCH1_FROZEN = (
    ROOT
    / "outputs/v3i_unseen_holdout_500/batch1_frozen_279"
    / "V3I_BATCH1_279_FROZEN.csv"
)

V3C_SCHEMA = (
    ROOT
    / "outputs/model_v3_experiments/schemas"
    / "V3_C_AGGRESSIVE_schema.json"
)

CACHE_DIR = MASTER_DIR / "api_cache_v2"

OUT_DIR = (
    ROOT / "outputs/v3i_unseen_holdout_500/batch2_frozen_221"
)

BATCH2_READY = OUT_DIR / "V3I_BATCH2_221_READY.csv"
BATCH2_BLIND = OUT_DIR / "V3I_BATCH2_221_READY_BLIND.csv"
BATCH2_TARGETS = OUT_DIR / "V3I_BATCH2_221_TARGETS.csv"
BATCH2_MANIFEST = OUT_DIR / "V3I_BATCH2_221_MANIFEST.json"
BATCH2_POOL = OUT_DIR / "V3I_BATCH2_ELIGIBLE_POOL.csv"

TARGET = "goal_after_87"
MATCH_ID = "match_id"
KICKOFF = "kickoff_utc"

DATE_FROM = pd.Timestamp("2026-09-14T00:00:00Z")
NEEDED = 221


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def hr(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def read_csv_robust(path: Path) -> pd.DataFrame:
    attempts = []
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            attempts.append((df.shape[1], sep, df))
        except Exception:
            pass

    if not attempts:
        raise RuntimeError(f"Impossible de lire {path}")

    attempts.sort(key=lambda x: x[0], reverse=True)
    return attempts[0][2]


def load_schema_features(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    candidates: List[List[str]] = []

    def walk(obj: Any):
        if isinstance(obj, Mapping):
            for _, v in obj.items():
                if (
                    isinstance(v, list)
                    and len(v) == 38
                    and all(isinstance(x, str) for x in v)
                ):
                    candidates.append(list(v))
                walk(v)
        elif isinstance(obj, list):
            if len(obj) == 38 and all(isinstance(x, str) for x in obj):
                candidates.append(list(obj))
            else:
                for v in obj:
                    walk(v)

    walk(payload)

    uniq = []
    seen = set()
    for c in candidates:
        k = tuple(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)

    if not uniq:
        raise RuntimeError("Liste de 38 features V3-C introuvable dans le schéma.")

    return uniq[0]


def parse_time(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def load_master_ids() -> set:
    ids = set()

    if MASTER_INDEX.exists():
        df = read_csv_robust(MASTER_INDEX)
        if MATCH_ID in df.columns:
            ids.update(df[MATCH_ID].dropna().astype(str))

    if MASTER_READY.exists():
        df = read_csv_robust(MASTER_READY)
        if MATCH_ID in df.columns:
            ids.update(df[MATCH_ID].dropna().astype(str))

    return ids


def load_batch1_ids() -> set:
    if not BATCH1_FROZEN.exists():
        raise FileNotFoundError(
            f"Batch 1 figé introuvable : {BATCH1_FROZEN}\n"
            "Exécuter prepare_v3i_batch2_221.py d'abord."
        )

    df = pd.read_csv(BATCH1_FROZEN, low_memory=False)

    if len(df) != 279:
        raise RuntimeError(
            f"Batch1 frozen={len(df)}, attendu=279."
        )

    if MATCH_ID not in df.columns:
        raise RuntimeError("match_id absent du Batch1 frozen.")

    return set(df[MATCH_ID].astype(str))


def run_source_collector(args) -> int:
    if not FOOTBALL_COLLECTOR.exists():
        raise FileNotFoundError(FOOTBALL_COLLECTOR)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FOOTBALL_COLLECTOR),
        "--priority-competitions-disable",
        "--resume",
        "--limit-new",
        str(args.limit_new),
        "--max-competitions",
        str(args.max_competitions),
        "--max-seasons-per-competition",
        str(args.max_seasons_per_competition),
        "--max-pages-competitions",
        str(args.max_pages_competitions),
        "--max-pages-seasons",
        str(args.max_pages_seasons),
        "--max-pages-per-season",
        str(args.max_pages_per_season),
        "--date-from",
        "2026-09-14",
        "--output-dir",
        str(SOURCE_DIR),
        "--cache-dir",
        str(CACHE_DIR),
        "--sleep-between-matches",
        str(args.sleep_between_matches),
        "--sleep-between-pages",
        str(args.sleep_between_pages),
        "--save-every",
        "1",
    ]

    hr("COLLECTE SOURCE BATCH 2")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    completed = subprocess.run(cmd)
    return int(completed.returncode)


def build_eligible_pool() -> pd.DataFrame:
    if not SOURCE_RAW.exists():
        raise FileNotFoundError(
            f"Dataset source absent : {SOURCE_RAW}"
        )

    features = load_schema_features(V3C_SCHEMA)
    source = read_csv_robust(SOURCE_RAW)

    needed_cols = [MATCH_ID, TARGET, KICKOFF] + features
    missing_cols = [c for c in needed_cols if c not in source.columns]
    if missing_cols:
        raise RuntimeError(
            f"Colonnes absentes de la source : {missing_cols}"
        )

    source[MATCH_ID] = source[MATCH_ID].astype(str)
    source[TARGET] = pd.to_numeric(
        source[TARGET],
        errors="coerce",
    )
    source["_kickoff_parsed"] = parse_time(source[KICKOFF])

    X = source[features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    exact38 = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    target_ok = source[TARGET].isin([0, 1])
    date_ok = (
        source["_kickoff_parsed"].notna()
        & (source["_kickoff_parsed"] >= DATE_FROM)
    )

    master_ids = load_master_ids()
    batch1_ids = load_batch1_ids()

    new_ok = (
        ~source[MATCH_ID].isin(master_ids)
        & ~source[MATCH_ID].isin(batch1_ids)
    )

    pool = source.loc[
        exact38 & target_ok & date_ok & new_ok
    ].copy()

    pool = pool.drop_duplicates(
        subset=[MATCH_ID],
        keep="first",
    )

    pool = pool.sort_values(
        ["_kickoff_parsed", MATCH_ID],
        kind="mergesort",
    ).reset_index(drop=True)

    pool["_v3i_batch"] = 2
    pool["_v3i_batch2_selection_order"] = np.arange(
        1,
        len(pool) + 1,
        dtype=int,
    )

    return pool


def freeze_if_ready(pool: pd.DataFrame) -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Toujours sauvegarder le pool courant pour audit.
    pool.drop(
        columns=["_kickoff_parsed"],
        errors="ignore",
    ).to_csv(
        BATCH2_POOL,
        index=False,
    )

    hr("AUDIT POOL BATCH 2")
    print(f"Éligibles V3-C 38/38 + target + date + IDs inédits : {len(pool)}")
    print(f"Objectif Batch2 : {NEEDED}")

    if BATCH2_READY.exists():
        frozen = pd.read_csv(
            BATCH2_READY,
            low_memory=False,
        )
        if len(frozen) != NEEDED:
            raise RuntimeError(
                f"Batch2 frozen existant={len(frozen)}, attendu={NEEDED}."
            )
        print("Batch 2 déjà figé. Aucun écrasement.")
        return True

    if len(pool) < NEEDED:
        print(
            f"Encore {NEEDED-len(pool)} matchs éligibles nécessaires. "
            "Relancer le même script pour continuer la source."
        )
        return False

    frozen = pool.iloc[:NEEDED].copy()

    if frozen[MATCH_ID].duplicated().any():
        raise RuntimeError("Doublon dans Batch2 avant freeze.")

    # READY complet
    ready = frozen.drop(
        columns=["_kickoff_parsed"],
        errors="ignore",
    )
    ready.to_csv(BATCH2_READY, index=False)

    # TARGETS séparés
    target_cols = [
        c for c in [
            MATCH_ID,
            TARGET,
            "target_status",
            "target_source",
            "target_confidence",
            "target_resolution_class",
            "goal_minutes_after_87",
            "_v3i_batch",
            "_v3i_batch2_selection_order",
        ]
        if c in ready.columns
    ]
    ready[target_cols].to_csv(
        BATCH2_TARGETS,
        index=False,
    )

    # BLIND: retire target et métadonnées qui le révèlent.
    leak_prefixes = ("target_",)
    leak_exact = {
        TARGET,
        "goal_minutes_after_87",
        "goal_after_87_minute",
    }

    blind_cols = [
        c
        for c in ready.columns
        if c not in leak_exact
        and not any(c.startswith(p) for p in leak_prefixes)
    ]
    ready[blind_cols].to_csv(
        BATCH2_BLIND,
        index=False,
    )

    manifest = {
        "created_at_utc": now_utc(),
        "status": "FROZEN_221",
        "selection_rule": (
            "Exact V3-C 38/38, target known, kickoff>=2026-09-14, "
            "not in historical master, not in Batch1; "
            "deterministic sort kickoff_utc then match_id; "
            "first 221. No model probability used."
        ),
        "source_dir": str(SOURCE_DIR),
        "eligible_pool_n_at_freeze": int(len(pool)),
        "batch2_n": int(len(ready)),
        "batch2_positives": int(pd.to_numeric(ready[TARGET]).sum()),
        "date_min": str(pd.to_datetime(ready[KICKOFF], utc=True, errors="coerce").min()),
        "date_max": str(pd.to_datetime(ready[KICKOFF], utc=True, errors="coerce").max()),
        "ready": str(BATCH2_READY),
        "blind": str(BATCH2_BLIND),
        "targets": str(BATCH2_TARGETS),
    }

    BATCH2_MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    hr("BATCH 2 FIGÉ")
    print(f"READY  : {BATCH2_READY}")
    print(f"BLIND  : {BATCH2_BLIND}")
    print(f"TARGETS: {BATCH2_TARGETS}")
    print(f"N      : {len(ready)}")
    print()
    print(
        "Étape suivante : "
        "python3 -u evaluate_v3i_batch2_221_SEPARATE.py"
    )
    return True


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--skip-collect",
        action="store_true",
        help="Ne lance pas l'API; reconstruit seulement le pool depuis la source existante.",
    )
    p.add_argument(
        "--limit-new",
        type=int,
        default=900,
        help="Nombre max de nouveaux matchs source à traiter sur cette exécution.",
    )
    p.add_argument(
        "--max-competitions",
        type=int,
        default=300,
    )
    p.add_argument(
        "--max-seasons-per-competition",
        type=int,
        default=3,
    )
    p.add_argument(
        "--max-pages-competitions",
        type=int,
        default=10,
    )
    p.add_argument(
        "--max-pages-seasons",
        type=int,
        default=3,
    )
    p.add_argument(
        "--max-pages-per-season",
        type=int,
        default=10,
    )
    p.add_argument(
        "--sleep-between-matches",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--sleep-between-pages",
        type=float,
        default=0.5,
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    hr("V3-I BATCH 2 — COLLECTE ÉLARGIE SÉPARÉE")
    print(f"Source séparée : {SOURCE_DIR}")
    print(f"Fenêtre         : >= 2026-09-14")
    print(f"Compétitions    : jusqu'à {args.max_competitions}")
    print(f"Objectif        : {NEEDED} matchs exact 38/38 + target")
    print("Batch1          : READ-ONLY / exclu")

    if not args.skip_collect:
        rc = run_source_collector(args)
        print()
        print(f"Collecteur source terminé avec code={rc}")
        print(
            "Même si le collecteur s'arrête avant 900 nouveaux matchs, "
            "le pool déjà collecté sera audité."
        )

    pool = build_eligible_pool()
    done = freeze_if_ready(pool)

    return 0 if done else 2


if __name__ == "__main__":
    raise SystemExit(main())
