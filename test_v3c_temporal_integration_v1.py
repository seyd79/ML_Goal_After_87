import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

LIVE = Path("live_scanner_engine.py")
CANDIDATE = Path(
    "live_scanner_engine_temporal_fix_candidate.py"
)
SCHEMA = Path(
    "outputs/model_v3_experiments/schemas/"
    "V3_C_AGGRESSIVE_schema.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError("Réseau interdit dans le test OFFLINE")


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "v3c_temporal_integration_offline",
        CANDIDATE,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def get_live_stats(self, match_id):
        self.calls.append("live_stats")
        return {"home_shots_87": 7}

    def get_stats(self, match_id):
        self.calls.append("stats")
        return {}

    def get_timeline(self, match_id):
        self.calls.append("timeline")
        return {"events": self.events}

    def __getattr__(self, name):
        if name.startswith("get_"):
            def fake(*args, **kwargs):
                self.calls.append(name)
                return {}
            return fake
        raise AttributeError(name)


def run_pipeline(module, match, events, features):
    client = FakeClient(events)

    X, raw = module.build_features(
        match=match,
        client=client,
        pre_match_features={},
        diagnostics={},
        required_features=features,
    )

    if "timeline" not in client.calls:
        raise AssertionError(
            "Le fallback timeline n'a pas été exécuté"
        )

    if X.shape != (1, 96):
        raise AssertionError(
            f"Vecteur inattendu : {X.shape}"
        )

    if list(X.columns) != list(module.EXPECTED_FEATURES):
        raise AssertionError(
            "Ordre des 96 variables incorrect"
        )

    if raw.get("home_shots_87") != 7:
        raise AssertionError(
            "Le fallback a écrasé LIVE-STATS"
        )

    return X, raw


print("=== ÉTAPE 154 : INTÉGRATION OFFLINE V3-C ===")

live_before = sha256(LIVE)
candidate_before = sha256(CANDIDATE)

with SCHEMA.open(encoding="utf-8") as f:
    schema = json.load(f)

if isinstance(schema, list):
    features = schema
else:
    features = next(
        (
            schema[key]
            for key in (
                "features_used",
                "core_features",
                "features",
                "feature_names",
            )
            if isinstance(schema.get(key), list)
        ),
        None,
    )

if not isinstance(features, list) or len(features) != 38:
    raise ValueError("Schéma actif incorrect")

if len(set(features)) != 38:
    raise ValueError("Variables du modèle dupliquées")

events = [
    {"minute": 80, "type": "foul", "side": "away"},
    {"minute": 82, "type": "shot", "side": "home"},
    {"minute": 84, "type": "corner_kick", "side": "away"},
    {"minute": 86, "type": "shot_on_target", "side": "home"},
    {"minute": 90, "type": "shot", "side": "away"},
]

with patch.object(
    socket.socket,
    "connect",
    side_effect=no_network,
):
    module = load_candidate()

    # L'extraction seule est simulée pour fournir
    # les événements choisis. Le reste du pipeline
    # ainsi que analyze_timeline sont exécutés réellement.
    module.extract_events = lambda payload: payload["events"]
    module.extract_live_stats = lambda payload: payload
    module.extract_match_stats_features = lambda payload: {}

    for minute in (85, 86, 87):
        match = module.LiveMatch(
            match_id="OFFLINE_TEST_001",
            home_team="Home",
            away_team="Away",
            minute=minute,
            home_score=0,
            away_score=0,
            raw={},
        )

        full_X, full_raw = run_pipeline(
            module, match, events, features
        )

        prior = [
            event for event in events
            if event["minute"] < minute
        ]

        prior_X, prior_raw = run_pipeline(
            module, match, prior, features
        )

        if not full_X.equals(prior_X):
            different = [
                key
                for key in full_raw
                if not (
                    pd.isna(full_raw[key])
                    and pd.isna(prior_raw.get(key))
                )
                and full_raw[key] != prior_raw.get(key)
            ]
            raise AssertionError(
                f"Divergence à {minute}' : {different}"
            )

        expected_last_shot = 86 if minute == 87 else 82
        expected_sot = 1 if minute == 87 else 0

        if full_raw.get("last_shot_minute") != expected_last_shot:
            raise AssertionError(
                f"Dernier tir incorrect à {minute}'"
            )

        if full_raw.get("sot_80_87") != expected_sot:
            raise AssertionError(
                f"Compteur tirs cadrés incorrect à {minute}'"
            )

        if full_raw.get("corners_80_87") != 1:
            raise AssertionError(
                f"Compteur corners incorrect à {minute}'"
            )

        selected_X = full_X.loc[:, features]

        if selected_X.shape != (1, 38):
            raise AssertionError(
                "Sélection des 38 variables invalide"
            )

        print(
            f"PASS {minute}' | "
            "96 variables | "
            "38 variables actives | "
            "priorité LIVE-STATS | "
            "exclusion des événements futurs"
        )

if sha256(LIVE) != live_before:
    raise AssertionError("Fichier LIVE modifié")

if sha256(CANDIDATE) != candidate_before:
    raise AssertionError("Candidat modifié")

print("\n=== BILAN ===")
print("Tests d'intégration : 3/3")
print("Appels réseau : bloqués")
print("Fichier LIVE : inchangé")
print("Candidat : inchangé")
print("Aucun entraînement.")
print("INTÉGRATION TEMPORELLE : PASS")
