#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PREPARE V3-I BATCH 2 — FREEZE BATCH 1 (279) + FORCE NEW DISCOVERY
=================================================================

But
---
1) Vérifier que le Batch 1 contient exactement 279 matchs uniques.
2) Sauvegarder une copie immuable du Batch 1.
3) Sauvegarder les IDs + hashes + modèle V3-H exact.
4) Archiver le manifest candidat consommé (589 candidats) SANS le supprimer.
5) Laisser le READY principal intact pour que le collecteur continue de 279 -> 500.
6) En retirant seulement le manifest actif, forcer le collecteur à refaire une
   découverte de candidats lors de la prochaine exécution.

Aucun modèle n'est réentraîné.
Aucun target n'est modifié.
Aucune ligne READY n'est supprimée.

Commande
--------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 -u prepare_v3i_batch2_221.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

BASE = ROOT / "outputs/v3i_unseen_holdout_500"
READY = BASE / "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv"
ACTIVE_MANIFEST = BASE / "v3i_candidate_manifest.csv"

EXACT_EVAL_DIR = BASE / "evaluation_batch1_279_EXACT"
V3H_FROZEN = EXACT_EVAL_DIR / "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368.joblib"
VALIDATION_MANIFEST = EXACT_EVAL_DIR / "validation_manifest_EXACT.json"

FREEZE_DIR = BASE / "batch1_frozen_279"
FROZEN_READY = FREEZE_DIR / "V3I_BATCH1_279_FROZEN.csv"
FROZEN_IDS = FREEZE_DIR / "V3I_BATCH1_279_IDS.csv"
FROZEN_CANDIDATES = FREEZE_DIR / "v3i_candidate_manifest_BATCH1_CONSUMED.csv"
FREEZE_MANIFEST = FREEZE_DIR / "batch1_freeze_manifest.json"

EXPECTED_N = 279


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    print("=" * 110)
    print("V3-I — PREPARE BATCH 2 / FREEZE BATCH 1")
    print("=" * 110)

    if not READY.exists():
        raise FileNotFoundError(READY)

    df = pd.read_csv(READY, low_memory=False)

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"READY actuel={len(df)} lignes, attendu={EXPECTED_N}. "
            "Le freeze Batch 1 doit être fait sur exactement 279 matchs."
        )

    if "match_id" not in df.columns:
        raise ValueError("match_id absent du READY.")

    if df["match_id"].duplicated().any():
        raise RuntimeError("Batch 1 contient des match_id dupliqués.")

    if "goal_after_87" not in df.columns:
        raise ValueError("goal_after_87 absent du READY.")

    y = pd.to_numeric(df["goal_after_87"], errors="coerce")
    if not y.isin([0, 1]).all():
        raise RuntimeError("Batch 1 contient des targets non résolus/non binaires.")

    if not V3H_FROZEN.exists():
        raise FileNotFoundError(
            f"Modèle V3-H exact introuvable : {V3H_FROZEN}\n"
            "Exécuter d'abord evaluate_v3i_batch1_279_v3c_v3h_EXACT.py."
        )

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot READY immuable.
    if FROZEN_READY.exists():
        old = pd.read_csv(FROZEN_READY, low_memory=False)
        old_ids = set(old["match_id"].astype(str))
        new_ids = set(df["match_id"].astype(str))
        if len(old) != EXPECTED_N or old_ids != new_ids:
            raise RuntimeError(
                "Un freeze Batch 1 différent existe déjà. "
                "Aucun écrasement automatique."
            )
        print("Freeze READY déjà présent et cohérent.")
    else:
        shutil.copy2(READY, FROZEN_READY)
        print(f"Freeze READY créé : {FROZEN_READY}")

    ids = df[["match_id"]].copy()
    if "_v3i_selection_order" in df.columns:
        ids["_v3i_selection_order"] = df["_v3i_selection_order"]
        ids = ids.sort_values(
            ["_v3i_selection_order", "match_id"],
            kind="mergesort",
        )
    else:
        ids = ids.sort_values("match_id", kind="mergesort")

    ids.to_csv(FROZEN_IDS, index=False)

    # Archive du manifest consommé pour forcer une nouvelle découverte ensuite.
    manifest_archived = False
    if ACTIVE_MANIFEST.exists():
        if FROZEN_CANDIDATES.exists():
            # Vérifie que l'archive existe; ne détruit jamais l'active si ambiguity.
            if sha256(ACTIVE_MANIFEST) != sha256(FROZEN_CANDIDATES):
                raise RuntimeError(
                    "Un manifest Batch 1 archivé différent existe déjà. "
                    "Stop pour éviter un écrasement."
                )
            ACTIVE_MANIFEST.unlink()
            manifest_archived = True
            print(
                "Manifest actif identique à l'archive existante : "
                "manifest actif retiré pour forcer la redécouverte."
            )
        else:
            shutil.move(str(ACTIVE_MANIFEST), str(FROZEN_CANDIDATES))
            manifest_archived = True
            print(f"Manifest Batch 1 archivé : {FROZEN_CANDIDATES}")
    else:
        print("Aucun manifest actif : la prochaine collecte fera déjà une découverte.")

    # Copie de sécurité du modèle exact + validation manifest.
    frozen_model_copy = FREEZE_DIR / V3H_FROZEN.name
    if not frozen_model_copy.exists():
        shutil.copy2(V3H_FROZEN, frozen_model_copy)

    validation_copy = None
    if VALIDATION_MANIFEST.exists():
        validation_copy = FREEZE_DIR / VALIDATION_MANIFEST.name
        if not validation_copy.exists():
            shutil.copy2(VALIDATION_MANIFEST, validation_copy)

    payload = {
        "created_at_utc": utc_now(),
        "status": "BATCH1_FROZEN_BATCH2_COLLECTION_CAN_START",
        "batch1_rows": int(len(df)),
        "batch1_positives": int(y.sum()),
        "batch1_positive_rate": float(y.mean()),
        "batch1_ready_source": str(READY),
        "batch1_frozen_ready": str(FROZEN_READY),
        "batch1_ids": str(FROZEN_IDS),
        "batch1_ready_sha256": sha256(FROZEN_READY),
        "v3h_exact_model_source": str(V3H_FROZEN),
        "v3h_exact_model_frozen_copy": str(frozen_model_copy),
        "v3h_exact_model_sha256": sha256(frozen_model_copy),
        "candidate_manifest_archived": bool(manifest_archived),
        "candidate_manifest_archive": (
            str(FROZEN_CANDIDATES)
            if FROZEN_CANDIDATES.exists()
            else None
        ),
        "collection_goal_total": 500,
        "batch2_needed": 221,
        "rule": (
            "Le READY principal reste intact. "
            "Le collecteur doit uniquement ajouter de nouveaux match_id "
            "jusqu'à 500. Ne pas réentraîner V3-H."
        ),
    }

    FREEZE_MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("FREEZE TERMINÉ")
    print("=" * 110)
    print(f"Batch 1 figé : {len(df)}")
    print(f"Positifs      : {int(y.sum())}")
    print(f"Taux          : {float(y.mean()):.2%}")
    print(f"À collecter   : {500-len(df)}")
    print(f"Freeze dir    : {FREEZE_DIR}")
    print()
    print(
        "PROCHAINE COMMANDE :\n"
        'caffeinate -dimsu python3 -u collect_v3i_unseen_holdout_500.py'
    )
    print()
    print(
        "Le manifest actif a été archivé pour que le collecteur fasse "
        "une nouvelle découverte. Le fichier READY de 279 matchs est conservé."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
