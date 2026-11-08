import ast
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

ORIGINAL = Path("live_scanner_engine.py")
SHADOW = Path("live_scanner_engine_shadow_v1.py")

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

SCHEMA = Path(
    "outputs/model_v3_experiments/schemas/"
    "V3_C_AGGRESSIVE_schema.json"
)

HELPERS = {
    "extract_event_minute",
    "extract_event_type",
    "normalize_event_type",
    "event_belongs_to_home",
    "analyze_timeline",
}


def load_functions(path):
    """
    Charge uniquement les définitions de fonctions
    nécessaires, sans importer ni lancer le scanner.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    needed = set(HELPERS)
    pending = list(HELPERS)

    while pending:
        name = pending.pop()

        if name not in definitions:
            raise RuntimeError(
                f"{path}: fonction introuvable : {name}"
            )

        node = definitions[name]

        for child in ast.walk(node):
            if not isinstance(child, ast.Name):
                continue

            dep = child.id

            if dep in definitions and dep not in needed:
                needed.add(dep)
                pending.append(dep)

    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in needed
    ]

    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )

    module = ast.Module(
        body=[future] + selected,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    namespace = {
        "__name__": "__offline_replay__",
        "Any": Any,
        "Dict": Dict,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "datetime": datetime,
        "timezone": timezone,
        "timedelta": timedelta,
        "re": re,
        "math": math,
        "json": json,
        "unicodedata": unicodedata,
        "np": np,
        "pd": pd,
    }

    exec(
        compile(module, str(path), "exec"),
        namespace,
    )

    return namespace


def read_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:
        mid = row["match_id"]

        if not mid or mid in result:
            raise ValueError(f"ID invalide : {path}")

        result[mid] = row

    return result


def equal(a, b):
    if a is None and b is None:
        return True

    try:
        if math.isnan(a) and math.isnan(b):
            return True
    except (TypeError, ValueError):
        pass

    return a == b


with SCHEMA.open(encoding="utf-8") as f:
    schema = json.load(f)

if isinstance(schema, list):
    active = schema
else:
    active = schema.get(
        "features_used",
        schema.get(
            "core_features",
            schema.get(
                "features",
                schema.get("feature_names", [])
            ),
        ),
    )

active = set(active)

if len(active) != 38:
    raise ValueError("Schéma actif V3-C invalide")

print("=== ÉTAPE 144 : REPLAY RÉEL SHADOW ===")
print("Chargement isolé des fonctions...")

original_ns = load_functions(ORIGINAL)
shadow_ns = load_functions(SHADOW)

original = original_ns["analyze_timeline"]
shadow = shadow_ns["analyze_timeline"]
extract_minute = original_ns["extract_event_minute"]

matches = 0
comparisons = 0
reference_checks = 0
future_event_cases = 0

changed_cases = Counter()
changed_features = Counter()
changed_active_features = Counter()
active_cases = Counter()

examples = []
errors = []

for batch in range(2, 12):
    tag = f"BATCH{batch:02d}"

    manifest = read_csv(
        JOIN / f"HISTORICAL_WAVE1_{tag}_MANIFEST_V1.csv"
    )

    dataset = read_csv(
        JOIN / f"{tag}_B_RESEARCH_V1.csv"
    )

    if (
        len(manifest) != 20
        or len(dataset) != 20
        or set(manifest) != set(dataset)
    ):
        raise ValueError(f"Lot incorrect : {tag}")

    for mid in sorted(dataset):
        if dataset[mid].get("research_split") not in (
            "TRAIN", "VALIDATION"
        ):
            raise ValueError(
                f"Split interdit pour {mid}"
            )

        info = manifest[mid]

        with (CACHE / f"{mid}.json").open(
            encoding="utf-8"
        ) as f:
            cache = json.load(f)

        payload = cache.get("data")

        if not isinstance(payload, dict):
            raise ValueError(f"Cache invalide : {mid}")

        if (
            payload.get("match_id") != mid
            or payload.get("coverage") != "full"
            or not isinstance(payload.get("events"), list)
        ):
            raise ValueError(
                f"Timeline invalide : {mid}"
            )

        events = payload["events"]

        kwargs = {
            "home_team_id": (
                info.get("home_team_id")
                or info.get("home_id")
            ),
            "away_team_id": (
                info.get("away_team_id")
                or info.get("away_id")
            ),
            "home_team_name": (
                info.get("home_team")
                or info.get("home_team_name")
            ),
            "away_team_name": (
                info.get("away_team")
                or info.get("away_team_name")
            ),
        }

        for minute in (85, 86, 87):
            prior_events = []
            future_events = 0

            for event in events:
                event_m = extract_minute(event)

                if event_m is None:
                    continue

                if event_m < minute:
                    prior_events.append(event)
                else:
                    future_events += 1

            try:
                baseline = original(
                    events,
                    current_minute=minute,
                    **kwargs,
                )

                corrected = shadow(
                    events,
                    current_minute=minute,
                    **kwargs,
                )

                reference = original(
                    prior_events,
                    current_minute=minute,
                    **kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Replay impossible : {mid} "
                    f"à {minute}' : {exc}"
                ) from exc

            comparisons += 1
            reference_checks += 1

            if future_events:
                future_event_cases += 1

            bad_reference = [
                feature
                for feature in (
                    set(corrected) | set(reference)
                )
                if not equal(
                    corrected.get(feature),
                    reference.get(feature),
                )
            ]

            if bad_reference:
                errors.append(
                    (mid, minute, bad_reference)
                )

            differences = [
                feature
                for feature in (
                    set(baseline) | set(corrected)
                )
                if not equal(
                    baseline.get(feature),
                    corrected.get(feature),
                )
            ]

            if differences:
                changed_cases[minute] += 1

            active_diffs = [
                feature
                for feature in differences
                if feature in active
            ]

            if active_diffs:
                active_cases[minute] += 1

            for feature in differences:
                changed_features[feature] += 1

            for feature in active_diffs:
                changed_active_features[feature] += 1

            if differences and len(examples) < 12:
                examples.append({
                    "match_id": mid,
                    "minute": minute,
                    "future_events": future_events,
                    "active_differences": active_diffs,
                    "all_differences": differences,
                })

        matches += 1

    print(
        f"{tag} | matchs=20 | "
        f"comparaisons={comparisons}"
    )

print("\n=== BILAN ===")
print("Matchs :", matches)
print("Comparaisons original / shadow :", comparisons)
print("Contrôles de référence :", reference_checks)
print("Erreurs de référence :", len(errors))
print(
    "Cas contenant des événements à la minute "
    "de décision ou après :",
    future_event_cases,
)

print("\n=== CHANGEMENTS PAR MINUTE ===")
for minute in (85, 86, 87):
    print(
        f"{minute}' | "
        f"sorties modifiées={changed_cases[minute]} | "
        f"variables actives modifiées="
        f"{active_cases[minute]}"
    )

print("\n=== VARIABLES ACTIVES MODIFIÉES ===")
for feature, count in changed_active_features.most_common():
    print(feature, ":", count)

print("\n=== AUTRES VARIABLES MODIFIÉES ===")
for feature, count in changed_features.most_common():
    if feature not in active:
        print(feature, ":", count)

print("\n=== EXEMPLES ===")
for example in examples:
    print(example)

if errors:
    print("\n=== ERREURS DE RÉFÉRENCE ===")
    for error in errors[:10]:
        print(error)

if (
    matches != 200
    or comparisons != 600
    or reference_checks != 600
    or errors
):
    raise SystemExit(
        "REPLAY SHADOW : CONTRÔLE NON VALIDÉ"
    )

print("\nREPLAY SHADOW : PASS")
print("SHADOW identique au calcul sur événements filtrés.")
print("TEST FUTUR non utilisé.")
print("Aucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier historique modifié.")
print("Scanner LIVE inchangé.")
print("Aucune conclusion de performance prédictive.")
