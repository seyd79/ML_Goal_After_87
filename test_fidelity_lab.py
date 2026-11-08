#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_fidelity_lab.py

TESTEUR DU MODULE PRE-MATCH "FIDELITY LAB V3"

But
---
Tester `pre_match_feature_engineering_fidelity_lab.py`
sur des matchs LIVE réels SANS modifier le scanner de production.

Contrairement au test intégré du module PRE-MATCH, ce script
n'exige PAS qu'un match soit entre 85' et 87'.

Il peut :
- afficher tous les matchs LIVE,
- sélectionner automatiquement le match le plus proche de 87',
- tester un match précis avec --match-id,
- tester plusieurs matchs avec --count,
- reconstruire les 59 features PRE-MATCH avec le builder LAB,
- vérifier les corrections form5/form10,
- vérifier league_home_over05_freq / league_away_over05_freq,
- sauvegarder chaque échantillon dans :
    outputs/fidelity_lab/lab_v2_pre_match_samples.csv

IMPORTANT
---------
Ce script importe uniquement le module LAB.
Il ne modifie ni live_scanner_engine.py ni le module PRE-MATCH de production.

Placement attendu
-----------------
Racine du projet :

ML_Goal_After_87/
├── .env
├── test_fidelity_lab.py
├── pre_match_feature_engineering_fidelity_lab.py
└── outputs/

Utilisation
-----------
Test automatique du match LIVE le plus proche de 87' :

    python test_fidelity_lab.py

Afficher les matchs puis tester les 3 plus proches de 87' :

    python test_fidelity_lab.py --count 3

Tester un match précis :

    python test_fidelity_lab.py --match-id mt_123456789

Ne pas sauvegarder le CSV :

    python test_fidelity_lab.py --no-save
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

LAB_MODULE_CANDIDATES = [
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab_v1.py",
]


def resolve_lab_module_path() -> Path:
    for candidate in LAB_MODULE_CANDIDATES:
        if candidate.exists():
            return candidate

    expected = "\n".join(
        f"  - {path.name}"
        for path in LAB_MODULE_CANDIDATES
    )

    raise FileNotFoundError(
        "Aucun module LAB trouvé à la racine du projet.\n"
        "Noms acceptés :\n"
        f"{expected}"
    )

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "fidelity_lab"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "lab_v2_pre_match_samples.csv"
)

EXPECTED_PRE_COUNT = 59


# =============================================================================
# HELPERS
# =============================================================================

def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def safe_int(value: Any) -> Optional[int]:
    value = safe_float(value)

    if value is None:
        return None

    return int(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)

    if value is None:
        return "None"

    return f"{value:.{digits}f}"


def now_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# =============================================================================
# LOAD LAB MODULE
# =============================================================================

def load_lab_module():

    lab_module_path = resolve_lab_module_path()

    print(
        f"Module LAB détecté : {lab_module_path.name}"
    )

    spec = importlib.util.spec_from_file_location(
        "pre_match_feature_engineering_fidelity_lab_runtime",
        lab_module_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Impossible de charger le module LAB."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module



# =============================================================================
# API CLIENT TEST — RETRY 429 + PACING
# =============================================================================

def build_test_client(
    lab,
    max_retries: int = 5,
    base_wait: float = 2.0,
    min_interval: float = 0.20,
):
    """
    Retourne un client LAB qui protège l'audit contre les HTTP 429.

    Important:
    - uniquement pour le test LAB;
    - aucune modification du scanner de production;
    - respecte Retry-After si l'API le fournit;
    - sinon backoff exponentiel: 2s, 4s, 8s, ...
    """

    class FidelityRateLimitedClient(
        lab.StatsAPIClient
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fidelity_last_request_at = 0.0

        def request(
            self,
            method,
            endpoint,
            params=None,
        ):
            url = (
                f"{self.base_url}/"
                f"{endpoint.lstrip('/')}"
            )

            for attempt in range(
                1,
                max_retries + 1,
            ):
                elapsed = (
                    time.monotonic()
                    - self._fidelity_last_request_at
                )

                if elapsed < min_interval:
                    time.sleep(
                        min_interval - elapsed
                    )

                try:
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        timeout=self.timeout,
                    )
                except Exception as exc:
                    if attempt >= max_retries:
                        print(
                            f"Erreur API finale : {exc}"
                        )
                        return None

                    wait = min(
                        base_wait * (2 ** (attempt - 1)),
                        30.0,
                    )

                    print(
                        f"Erreur API temporaire : {exc} "
                        f"| retry dans {wait:.1f}s"
                    )

                    time.sleep(wait)
                    continue

                self._fidelity_last_request_at = (
                    time.monotonic()
                )

                if response.status_code == 429:
                    retry_after = (
                        response.headers.get(
                            "Retry-After"
                        )
                    )

                    try:
                        wait = float(
                            retry_after
                        )
                    except Exception:
                        wait = min(
                            base_wait
                            * (2 ** (attempt - 1)),
                            30.0,
                        )

                    print(
                        f"API 429 : {method} {endpoint} "
                        f"| tentative {attempt}/{max_retries} "
                        f"| attente {wait:.1f}s"
                    )

                    if attempt >= max_retries:
                        return None

                    time.sleep(
                        max(wait, 0.5)
                    )
                    continue

                if response.status_code >= 400:
                    print(
                        f"API {response.status_code} : "
                        f"{method} {endpoint}"
                    )
                    return None

                try:
                    return response.json()
                except ValueError:
                    print(
                        f"Réponse JSON invalide : "
                        f"{endpoint}"
                    )
                    return None

            return None

    return FidelityRateLimitedClient()


# =============================================================================
# MATCH DISPLAY
# =============================================================================

def match_info(
    lab,
    match: Mapping[str, Any],
) -> Dict[str, Any]:

    home = lab.extract_home_team(
        match
    )

    away = lab.extract_away_team(
        match
    )

    home_name = (
        lab.extract_team_name(home)
        or "Home"
    )

    away_name = (
        lab.extract_team_name(away)
        or "Away"
    )

    minute = lab.extract_live_minute(
        match
    )

    score_home, score_away = (
        lab.extract_live_score(
            match
        )
    )

    match_id = lab.extract_match_id(
        match
    )

    return {
        "match_id": match_id,
        "home_team": home_name,
        "away_team": away_name,
        "minute": minute,
        "score_home": score_home,
        "score_away": score_away,
    }


def print_live_matches(
    lab,
    live_matches: Sequence[
        Mapping[str, Any]
    ],
) -> None:

    print()
    print("=" * 100)
    print("MATCHS LIVE DISPONIBLES")
    print("=" * 100)

    if not live_matches:
        print("Aucun match LIVE reçu.")
        return

    enriched = [
        match_info(
            lab,
            match,
        )
        for match in live_matches
    ]

    enriched.sort(
        key=lambda row: (
            row["minute"]
            if row["minute"] is not None
            else -1
        )
    )

    for index, row in enumerate(
        enriched,
        start=1,
    ):

        minute = (
            f"{row['minute']}'"
            if row["minute"] is not None
            else "?"
        )

        score = (
            f"{row['score_home']}-"
            f"{row['score_away']}"
        )

        print(
            f"{index:02d}. "
            f"{row['home_team']} "
            f"{score} "
            f"{row['away_team']} "
            f"| {minute} "
            f"| {row['match_id']}"
        )

    print("=" * 100)


# =============================================================================
# MATCH SELECTION
# =============================================================================

def choose_matches(
    lab,
    live_matches: Sequence[
        Mapping[str, Any]
    ],
    match_id: Optional[str],
    count: int,
) -> List[Mapping[str, Any]]:

    if match_id:

        for match in live_matches:

            current_id = clean(
                lab.extract_match_id(
                    match
                )
            )

            if current_id == clean(
                match_id
            ):
                return [match]

        # Le match n'est peut-être plus LIVE.
        # On tente un GET direct.
        try:
            payload = lab.StatsAPIClient().get(
                f"/football/matches/{match_id}"
            )

            if isinstance(
                payload,
                dict,
            ):

                # API parfois encapsulée.
                for key in (
                    "data",
                    "match",
                    "result",
                ):
                    candidate = payload.get(
                        key
                    )

                    if isinstance(
                        candidate,
                        dict,
                    ):
                        payload = candidate
                        break

                return [payload]

        except Exception as exc:
            print(
                f"Impossible de récupérer "
                f"{match_id} directement : {exc}"
            )

        return []

    scored: List[
        tuple
    ] = []

    for match in live_matches:

        minute = (
            lab.extract_live_minute(
                match
            )
        )

        if minute is None:
            distance = 999
        else:
            distance = abs(
                87 - minute
            )

        scored.append(
            (
                distance,
                -(minute or -1),
                match,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return [
        item[2]
        for item in scored[
            :max(count, 1)
        ]
    ]


# =============================================================================
# VALIDATION LAB
# =============================================================================

def validate_corrected_features(
    features: Mapping[str, Any],
) -> List[str]:

    warnings: List[str] = []

    checks = [
        (
            "home_form5",
            0.0,
            15.0,
        ),
        (
            "away_form5",
            0.0,
            15.0,
        ),
        (
            "home_form10",
            0.0,
            30.0,
        ),
        (
            "away_form10",
            0.0,
            30.0,
        ),
        (
            "league_home_over05_freq",
            0.0,
            1.0,
        ),
        (
            "league_away_over05_freq",
            0.0,
            1.0,
        ),
    ]

    for feature, low, high in checks:

        value = safe_float(
            features.get(
                feature
            )
        )

        if value is None:
            warnings.append(
                f"{feature}=None"
            )
            continue

        if not (
            low
            <= value
            <= high
        ):
            warnings.append(
                f"{feature}={value} hors "
                f"[{low}, {high}]"
            )

    return warnings


def print_feature_focus(
    features: Mapping[str, Any],
) -> None:

    print()
    print("-" * 100)
    print(
        "FEATURES DE FIDÉLITÉ PRIORITAIRES"
    )
    print("-" * 100)

    groups = {
        "FORME": [
            "home_form5",
            "away_form5",
            "home_form10",
            "away_form10",
        ],
        "LEAGUE OVER 0.5": [
            "league_home_over05_freq",
            "league_away_over05_freq",
        ],
        "xG — OBSERVATION UNIQUEMENT": [
            "home_avg_xg",
            "away_avg_xg",
        ],
        "STANDINGS — OBSERVATION UNIQUEMENT": [
            "home_team_rank",
            "away_team_rank",
            "home_team_matches_played",
            "away_team_matches_played",
            "home_team_points_standing",
            "away_team_points_standing",
        ],
    }

    for title, names in groups.items():

        print()
        print(
            f"[{title}]"
        )

        for name in names:

            print(
                f"  {name:<35} "
                f"= {fmt(features.get(name))}"
            )

    print("-" * 100)


# =============================================================================
# SAVE SAMPLE
# =============================================================================

def append_sample(
    lab,
    match: Mapping[str, Any],
    features: Mapping[str, Any],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    info = match_info(
        lab,
        match,
    )

    row: Dict[str, Any] = {
        "captured_at_utc":
            now_utc_iso(),
        "match_id":
            info["match_id"],
        "home_team":
            info["home_team"],
        "away_team":
            info["away_team"],
        "live_minute":
            info["minute"],
        "live_score_home":
            info["score_home"],
        "live_score_away":
            info["score_away"],
    }

    for feature in (
        lab.PRE_MATCH_FEATURES
    ):
        row[
            feature
        ] = features.get(
            feature
        )

    fieldnames = list(
        row.keys()
    )

    file_exists = (
        OUTPUT_CSV.exists()
        and OUTPUT_CSV.stat().st_size > 0
    )

    with OUTPUT_CSV.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# =============================================================================
# ONE MATCH
# =============================================================================

def test_one_match(
    lab,
    client,
    match: Mapping[str, Any],
    save: bool,
    min_coverage: int,
) -> bool:

    info = match_info(
        lab,
        match,
    )

    print()
    print("=" * 100)
    print(
        f"TEST LAB : "
        f"{info['home_team']} "
        f"vs {info['away_team']}"
    )
    print("=" * 100)

    print(
        f"Match ID : "
        f"{info['match_id']}"
    )

    print(
        f"Minute LIVE : "
        f"{info['minute']}"
    )

    print(
        f"Score LIVE : "
        f"{info['score_home']}-"
        f"{info['score_away']}"
    )

    try:

        features = (
            lab.build_pre_match_features(
                client,
                match,
            )
        )

    except Exception as exc:

        print()
        print(
            "ERREUR BUILD PRE-MATCH LAB :"
        )
        print(
            repr(exc)
        )

        return False

    if not isinstance(
        features,
        Mapping,
    ):

        print(
            "ERREUR : le builder n'a pas "
            "retourné un dictionnaire."
        )

        return False

    available = sum(
        1
        for feature
        in lab.PRE_MATCH_FEATURES
        if safe_float(
            features.get(
                feature
            )
        ) is not None
    )

    missing = [
        feature
        for feature
        in lab.PRE_MATCH_FEATURES
        if safe_float(
            features.get(
                feature
            )
        ) is None
    ]

    print()
    print(
        f"Couverture PRE-MATCH LAB : "
        f"{available}/"
        f"{len(lab.PRE_MATCH_FEATURES)}"
    )

    if missing:

        print(
            "Features manquantes :"
        )

        for feature in missing:
            print(
                f"  - {feature}"
            )

    else:

        print(
            "Features manquantes : AUCUNE"
        )

    sample_valid = (
        available >= min_coverage
    )

    if not sample_valid:
        print()
        print(
            f"⚠️ ÉCHANTILLON NON VALIDE POUR L'AUDIT : "
            f"couverture {available}/59 < "
            f"minimum {min_coverage}/59."
        )
        print(
            "Cause probable : rate-limit/API indisponible. "
            "Cet échantillon ne sera pas sauvegardé."
        )

    print_feature_focus(
        features
    )

    warnings = (
        validate_corrected_features(
            features
        )
    )

    print()
    print(
        "VALIDATION DES CORRECTIONS"
    )
    print("-" * 100)

    if warnings:

        print(
            "⚠️ À vérifier :"
        )

        for warning in warnings:
            print(
                f"  - {warning}"
            )

    else:

        print(
            "✅ Les features corrigées "
            "respectent les plages attendues."
        )

    home_o05 = safe_float(
        features.get(
            "league_home_over05_freq"
        )
    )

    away_o05 = safe_float(
        features.get(
            "league_away_over05_freq"
        )
    )

    if (
        home_o05 is not None
        and away_o05 is not None
    ):

        print()
        print(
            "League Over 0.5 différencié : "
            f"HOME={home_o05:.4f} "
            f"| AWAY={away_o05:.4f} "
            f"| écart="
            f"{abs(home_o05-away_o05):.4f}"
        )

    if save and sample_valid:

        append_sample(
            lab,
            match,
            features,
        )

        print()
        print(
            f"Échantillon sauvegardé : "
            f"{OUTPUT_CSV}"
        )

    if not sample_valid:
        return False

    return True


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Teste le PRE-MATCH Fidelity LAB "
            "sur des matchs LIVE réels."
        )
    )

    parser.add_argument(
        "--match-id",
        default=None,
        help=(
            "ID d'un match précis. "
            "Ex: mt_734296041"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help=(
            "Nombre de matchs LIVE à tester "
            "automatiquement. Défaut: 1."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "N'écrit pas "
            "outputs/fidelity_lab/"
            "lab_v2_pre_match_samples.csv"
        ),
    )

    parser.add_argument(
        "--between-matches",
        type=float,
        default=12.0,
        help=(
            "Pause entre deux matchs testés pour éviter les 429. "
            "Défaut: 12 secondes."
        ),
    )

    parser.add_argument(
        "--min-coverage",
        type=int,
        default=45,
        help=(
            "Couverture PRE minimale pour considérer l'échantillon valide. "
            "Défaut: 45/59."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    args = parse_args()

    print()
    print("=" * 100)
    print(
        "TEST FIDELITY LAB V3 — "
        "PRE-MATCH HISTORICAL COMPATIBILITY"
    )
    print("=" * 100)

    try:
        lab = load_lab_module()

    except Exception as exc:
        print(
            f"ERREUR chargement module LAB : "
            f"{exc}"
        )
        return 2

    api_key = clean(
        getattr(
            lab,
            "API_KEY",
            "",
        )
    )

    if not api_key:
        print(
            "ERREUR : STATS_API_KEY "
            "absente du .env"
        )
        return 2

    print(
        "Module LAB : OK"
    )

    print(
        "API KEY : OK"
    )

    print(
        f"PRE-MATCH attendu : "
        f"{len(lab.PRE_MATCH_FEATURES)}"
        f"/{EXPECTED_PRE_COUNT}"
    )

    client = build_test_client(
        lab=lab,
    )

    try:
        live_matches = (
            lab.get_live_matches(
                client
            )
        )

    except Exception as exc:
        print(
            "ERREUR récupération LIVE : "
            f"{exc}"
        )
        return 2

    print(
        f"Matchs LIVE récupérés : "
        f"{len(live_matches)}"
    )

    print_live_matches(
        lab,
        live_matches,
    )

    selected = choose_matches(
        lab=lab,
        live_matches=live_matches,
        match_id=args.match_id,
        count=max(
            args.count,
            1,
        ),
    )

    if not selected:

        print()
        print(
            "Aucun match sélectionnable."
        )

        return 1

    print()
    print(
        f"{len(selected)} match(s) "
        f"sélectionné(s) pour le test."
    )

    success = 0

    for index, match in enumerate(
        selected,
        start=1,
    ):

        if index > 1 and args.between_matches > 0:
            print()
            print(
                f"Pause anti-429 : "
                f"{args.between_matches:.1f}s"
            )
            time.sleep(
                args.between_matches
            )

        if test_one_match(
            lab=lab,
            client=client,
            match=match,
            save=not args.no_save,
            min_coverage=max(
                0,
                min(args.min_coverage, 59),
            ),
        ):
            success += 1

    print()
    print("=" * 100)
    print(
        "FIN TEST FIDELITY LAB"
    )
    print("=" * 100)

    print(
        f"Tests réussis : "
        f"{success}/{len(selected)}"
    )

    if not args.no_save:
        print(
            f"CSV cumulatif : "
            f"{OUTPUT_CSV}"
        )

    print()
    print(
        "Le scanner de production "
        "n'a pas été modifié."
    )

    return (
        0
        if success > 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
