#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path

COLLECTOR = Path("collect_v41_balanced_pilot_v1.py")

if not COLLECTOR.is_file():
    raise SystemExit("Collecteur V4.1 introuvable")

MAX_PASSES = 4
TOTAL_BUDGET = 20

print("=" * 70, flush=True)
print("ÉTAPE 243 — COLLECTE ÉQUILIBRÉE V4.1")
print("Budget maximal :", TOTAL_BUDGET, "requêtes")
print("Scanner V3-C inchangé")
print("=" * 70, flush=True)

completed = 0

for passage in range(1, MAX_PASSES + 1):
    print(
        "\nPASSAGE", passage, "/", MAX_PASSES,
        flush=True
    )

    result = subprocess.run(
        [sys.executable, str(COLLECTOR)],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout, flush=True)

    if result.stderr:
        print(
            result.stderr,
            file=sys.stderr,
            flush=True,
        )

    output = result.stdout

    if result.returncode != 0:
        print("ARRÊT : code de sortie non nul.")
        break

    if "ARRÊT DE SÉCURITÉ" in output:
        print("ARRÊT : anomalie détectée.")
        break

    if "ÉTAPE 237 TERMINÉE." not in output:
        print("ARRÊT : bilan final absent.")
        break

    if "Requêtes : 5" not in output:
        print(
            "ARRÊT : nombre de requêtes "
            "différent du plan attendu."
        )
        break

    completed += 1

print("\n=== BILAN DE L'ORCHESTRATION ===")
print("Passages complets :", completed)
print("Maximum théorique de requêtes :", TOTAL_BUDGET)
print("Aucun entraînement lancé.")
print("Aucun label créé.")
print("Modèles V4 inchangés.")
print("Scanner V3-C inchangé.")
