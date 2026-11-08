#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS HISTORICAL EXTERNAL HOLDOUT COLLECTOR (BLIND)
================================================================================

OBJECTIF
--------
Construire un HOLDOUT HISTORIQUE EXTERNE, séparé temporellement du développement,
sans toucher au Future Pristine Holdout.

Fenêtre verrouillée:
    2024-01-01 00:00:00 UTC
    ->
    2025-04-12 23:59:59 UTC

Pourquoi cette fenêtre:
    Le développement First Half 37 utilisé pour construire la V3 commence
    ensuite. Cette fenêtre est donc entièrement antérieure au bloc de
    développement connu.

Population:
    TOUS les matchs football terminés listés par TheStatsAPI dans la fenêtre.

Inclusion du holdout:
    - timeline 1re mi-temps prouvée complète;
    - fin de première période / halftime observée;
    - score à 37' reconstructible;
    - score_37_home == score_37_away > 0  (draw_with_goals).

Blind protocol:
    - aucun scoring modèle ici;
    - aucune probabilité calculée;
    - target non affiché;
    - target écrit NaN dans le dataset blind;
    - timeline brute sauvegardée dans sealed_target_source;
    - l'évaluateur one-shot ouvrira les targets plus tard;
    - missing != 0;
    - stats finales interdites pour reconstruire les 5 prédicteurs.

Optimisation API:
    1) listing historique par jour;
    2) timeline pour chaque match listé;
    3) shotmap seulement si le match est draw_with_goals à 37';
    4) détail match seulement si les noms ne suffisent pas à reconstruire
       le score ou à identifier les côtés;
    5) cache local + reprise.

Le collecteur est resumable. Un garde-fou d'appels API par exécution est
disponible via --max-api-requests. Le protocole est fixe et ne permet pas
de modifier la fenêtre via CLI.

Dépendances projet à la racine:
    collect_first_half_37_dataset.py
    audit_historical_live87_reconstructability.py
    .env
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

FEATURE_MODULE_PATH = BASE_DIR / "collect_first_half_37_dataset.py"
AUDIT_MODULE_PATH = BASE_DIR / "audit_historical_live87_reconstructability.py"

FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_frozen"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json"
SCHEMA_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_SCHEMA.json"
TRAINING_MANIFEST_PATH = FROZEN_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_TRAINING_MANIFEST.json"

PRIMARY_CONSUMED_DATASET = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_context_benchmark"
    / "First_Half_37_V3_DEVELOPMENT_CONSUMED.csv"
)

TRAINING_DATASET_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_feature_selection"
    / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_historical_external_holdout"
)

PROTOCOL_LOCK_PATH = OUTPUT_DIR / "FIRST_HALF37_V3_HISTORICAL_EXTERNAL_PROTOCOL_LOCK.json"
BLIND_DATASET_PATH = OUTPUT_DIR / "First_Half_37_V3_HISTORICAL_EXTERNAL_HOLDOUT_BLIND.csv"
LISTING_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_listing.csv"
DAY_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_listing_by_day.csv"
MATCH_AUDIT_PATH = OUTPUT_DIR / "first_half_37_v3_historical_external_match_audit.csv"
COLLECTION_MANIFEST_PATH = OUTPUT_DIR / "FIRST_HALF37_V3_HISTORICAL_EXTERNAL_COLLECTION_MANIFEST.json"
RAW_TIMELINE_DIR = OUTPUT_DIR / "sealed_target_source" / "timeline"
API_CACHE_DIR = OUTPUT_DIR / "cache" / "api"

WINDOW_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 4, 12, 23, 59, 59, tzinfo=timezone.utc)

EXPECTED_MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
EXPECTED_MODEL_STATUS = "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT"
EXPECTED_TRAINING_ROWS = 240