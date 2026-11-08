#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rebuild_clean_prematch_v2.py

RECONSTRUCTION PRE-MATCH "AS-OF" POUR LE NOUVEAU DATASET GOAL AFTER 87
======================================================================

OBJECTIF
--------
Reprendre les matchs prospectifs propres déjà capturés à 87' et RECONSTRUIRE
leurs 59 features PRE-MATCH à partir de l'API, avec une règle stricte :

    aucune information postérieure au coup d'envoi du match cible
    ne peut entrer dans les features PRE-MATCH.

Le script conserve les 37 features LIVE réellement capturées à 87' et le label V3,
mais remplace complètement les anciennes 59 features PRE-MATCH par des valeurs
recalculées "as-of kickoff".

IMPORTANT
---------
Ce script NE MODIFIE PAS :
- live_scanner_engine.py
- le scanner de production
- le modèle actuel
- les snapshots historiques
- le dataset historique de 13 800 matchs

Il crée uniquement de nouveaux fichiers dans :
    outputs/rebuilt_dataset_v2/

ENTRÉE RECOMMANDÉE
------------------
outputs/clean_dataset_v2/Clean_Dataset_Goal_After_87_RAW_v2.csv

Ce fichier contient déjà :
- un snapshot exact à 87'
- une seule ligne par match
- un target V3 résolu
- les 37 LIVE capturées
- éventuellement des PRE anciennes que CE SCRIPT REMPLACE

FALLBACK
--------
Si le fichier ci-dessus n'existe pas, le script peut utiliser :
outputs/history/live_feature_snapshots_labeled_v3.csv

RÈGLES DE RECONSTRUCTION
------------------------
1) Le match cible est récupéré via /football/matches/{id}.
2) Sa date de coup d'envoi devient le CUTOFF.
3) Pour toutes les statistiques historiques :
       historical_match_date < target_kickoff
4) Le match cible lui-même est exclu.
5) Aucun résultat futur n'est autorisé.
6) Les classements sont RECONSTRUITS à partir des matchs terminés de la saison
   avant le kickoff, au lieu d'utiliser le classement actuel.
7) Form5 / Form10 :
       victoire = 3
       nul = 1
       défaite = 0
       somme sur les 5 / 10 derniers matchs disponibles.
8) H2H :
       maximum 5 confrontations terminées AVANT le kickoff.
9) Fréquences de ligue :
       uniquement les matchs terminés AVANT le kickoff.
10) xG :
       uniquement les matchs historiques pré-kickoff pour lesquels l'API fournit
       réellement un xG ; aucune constante n'est inventée.
11) Odds :
       opening odds du match cible si encore disponibles dans l'API.
12) NaN :
       conservé comme NaN. Aucun remplissage médian, aucun zéro inventé.

SORTIES
-------
outputs/rebuilt_dataset_v2/
├── Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv
├── Clean_Dataset_Goal_After_87_REBUILT_FULL_v2.csv
├── prematch_rebuild_audit_v2.csv
├── feature_coverage_v2.csv
├── rejected_rebuild_v2.csv
└── rebuild_clean_prematch_v2_report.md

UTILISATION
-----------
    python3 rebuild_clean_prematch_v2.py

Test court :
    python3 rebuild_clean_prematch_v2.py --limit 5

Reprendre après interruption :
    python3 rebuild_clean_prematch_v2.py --resume

Limiter les appels xG :
    python3 rebuild_clean_prematch_v2.py --xg-limit 10

Désactiver temporairement xG :
    python3 rebuild_clean_prematch_v2.py --xg-limit 0

NOTES
-----
- La reconstruction est volontairement conservatrice.
- Le script écrit progressivement les résultats après chaque match.
- Les HTTP 429 sont retentés avec backoff.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

LAB_MODULE_CANDIDATES = [
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab_v2.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab_v1.py",
]

INPUT_CANDIDATES = [
    BASE_DIR
    / "outputs"
    / "clean_dataset_v2"
    / "Clean_Dataset_Goal_After_87_RAW_v2.csv",

    BASE_DIR
    / "outputs"
    / "history"
    / "live_feature_snapshots_labeled_v3.csv",

    BASE_DIR
    / "live_feature_snapshots_labeled_v3.csv",
]

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "rebuilt_dataset_v2"
)


# =============================================================================
# SCHEMA
# =============================================================================

ID_COLUMN = "match_id"
TARGET_COLUMN = "goal_after_87"

PRE_MATCH_FEATURES = [
    "home_team_rank",
    "away_team_rank",
    "home_team_points_standing",
    "away_team_points_standing",
    "home_team_matches_played",
    "away_team_matches_played",
    "home_team_wins",
    "away_team_wins",
    "home_team_draws",
    "away_team_draws",
    "home_team_losses",
    "away_team_losses",
    "home_team_goals_for",
    "away_team_goals_for",
    "home_team_goals_against",
    "away_team_goals_against",
    "home_team_goal_difference",
    "away_team_goal_difference",
    "home_position",
    "away_position",
    "home_points",
    "away_points",
    "home_form5",
    "away_form5",
    "home_form10",
    "away_form10",
    "home_scored_freq",
    "away_scored_freq",
    "home_wins_freq",
    "away_wins_freq",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",
    "home_avg_xg",
    "away_avg_xg",
    "home_btts_freq",
    "away_btts_freq",
    "home_over25_freq",
    "away_over25_freq",
    "h2h5_over25_freq",
    "h2h5_btts_freq",
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "league_2nd_half_over15_freq",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
]

LIVE_FEATURES = [
    "score_87_home",
    "score_87_away",
    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",
    "home_corners_87",
    "away_corners_87",
    "home_fouls_87",
    "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87",
    "away_red_cards_87",
    "home_offsides_87",
    "away_offsides_87",
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",
    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",
    "minutes_since_last_goal",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "minutes_since_last_corner",
]

FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES

assert len(PRE_MATCH_FEATURES) == 59
assert len(LIVE_FEATURES) == 37
assert len(FEATURES) == 96


# =============================================================================
# HELPERS
# =============================================================================

def clean(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        number = float(value)
    except Exception:
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def now_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def dt_utc_naive(
    value: Any,
) -> Optional[datetime]:
    """
    Parse une date puis la normalise en datetime UTC-naive.
    Cela évite les comparaisons aware/naive.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = clean(value)

        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )
        except Exception:
            try:
                parsed = pd.to_datetime(
                    text,
                    utc=True,
                    errors="coerce",
                )

                if pd.isna(parsed):
                    return None

                parsed = parsed.to_pydatetime()

            except Exception:
                return None

    if parsed.tzinfo is not None:
        parsed = (
            parsed
            .astimezone(
                timezone.utc
            )
            .replace(
                tzinfo=None
            )
        )

    return parsed


def nonempty_count(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> int:

    count = 0

    for key in keys:
        if safe_float(
            mapping.get(
                key
            )
        ) is not None:
            count += 1

    return count


def resolve_path(
    explicit: Optional[str],
    candidates: Sequence[Path],
) -> Path:

    if explicit:
        path = Path(
            explicit
        ).expanduser()

        if not path.is_absolute():
            path = (
                BASE_DIR
                / path
            )

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        return path

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Aucun fichier source compatible trouvé."
    )


# =============================================================================
# LOAD FIDELITY LAB
# =============================================================================

def resolve_lab_path() -> Path:

    for path in LAB_MODULE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Module PRE-MATCH Fidelity LAB introuvable.\n"
        "Noms acceptés :\n"
        + "\n".join(
            f"  - {path.name}"
            for path in LAB_MODULE_CANDIDATES
        )
    )


def load_lab():

    path = resolve_lab_path()

    print(
        f"Module Fidelity LAB : {path.name}"
    )

    spec = importlib.util.spec_from_file_location(
        "prematch_fidelity_lab_runtime",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Impossible de charger le module Fidelity LAB."
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
# RATE-LIMITED CLIENT
# =============================================================================

def build_rate_limited_client(
    lab,
    max_retries: int = 6,
    base_wait: float = 2.0,
    min_interval: float = 0.25,
):

    class HistoricalClient(
        lab.StatsAPIClient
    ):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

            self._last_request_at = 0.0
            self.request_count = 0
            self.retry_429_count = 0

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
                    - self._last_request_at
                )

                if elapsed < min_interval:
                    time.sleep(
                        min_interval
                        - elapsed
                    )

                try:
                    response = (
                        self.session.request(
                            method=method,
                            url=url,
                            params=params,
                            timeout=self.timeout,
                        )
                    )
                except Exception as exc:

                    if attempt >= max_retries:
                        print(
                            f"Erreur API finale : "
                            f"{endpoint} | {exc}"
                        )
                        return None

                    wait = min(
                        base_wait
                        * (
                            2
                            ** (
                                attempt
                                - 1
                            )
                        ),
                        30.0,
                    )

                    print(
                        f"Erreur API temporaire : "
                        f"{endpoint} "
                        f"| attente {wait:.1f}s"
                    )

                    time.sleep(
                        wait
                    )
                    continue

                self._last_request_at = (
                    time.monotonic()
                )

                self.request_count += 1

                if response.status_code == 429:

                    self.retry_429_count += 1

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
                            * (
                                2
                                ** (
                                    attempt
                                    - 1
                                )
                            ),
                            30.0,
                        )

                    print(
                        f"API 429 : "
                        f"{endpoint} "
                        f"| tentative "
                        f"{attempt}/{max_retries} "
                        f"| attente {wait:.1f}s"
                    )

                    if attempt >= max_retries:
                        return None

                    time.sleep(
                        max(
                            wait,
                            0.5,
                        )
                    )

                    continue

                if response.status_code >= 400:

                    print(
                        f"API "
                        f"{response.status_code} : "
                        f"{method} {endpoint}"
                    )

                    return None

                try:
                    return response.json()

                except ValueError:

                    print(
                        f"JSON invalide : "
                        f"{endpoint}"
                    )

                    return None

            return None

    return HistoricalClient()


# =============================================================================
# API CACHES
# =============================================================================

MATCH_DETAIL_CACHE: Dict[
    str,
    Dict[str, Any]
] = {}

TEAM_HISTORY_CACHE: Dict[
    str,
    List[Dict[str, Any]]
] = {}

LEAGUE_HISTORY_CACHE: Dict[
    Tuple[str, str],
    List[Dict[str, Any]]
] = {}

ODDS_CACHE: Dict[
    str,
    Dict[str, Any]
] = {}


# =============================================================================
# API FETCH
# =============================================================================

def unwrap_match(
    lab,
    payload: Any,
) -> Dict[str, Any]:

    try:
        result = (
            lab.unwrap_match_payload(
                payload
            )
        )

        if isinstance(
            result,
            dict,
        ):
            return result

    except Exception:
        pass

    if not isinstance(
        payload,
        dict,
    ):
        return {}

    data = payload.get(
        "data"
    )

    if isinstance(
        data,
        dict,
    ):
        nested = data.get(
            "match"
        )

        if isinstance(
            nested,
            dict,
        ):
            return nested

        return data

    return payload


def get_match_detail(
    lab,
    client,
    match_id: str,
) -> Dict[str, Any]:

    key = str(
        match_id
    )

    if key in MATCH_DETAIL_CACHE:
        return MATCH_DETAIL_CACHE[
            key
        ]

    payload = client.get(
        f"/football/matches/{key}"
    )

    match = unwrap_match(
        lab,
        payload,
    )

    MATCH_DETAIL_CACHE[
        key
    ] = match

    return match


def api_list(
    lab,
    payload: Any,
) -> List[Dict[str, Any]]:

    try:
        values = lab.as_list(
            payload
        )

        if isinstance(
            values,
            list,
        ):
            return [
                item
                for item in values
                if isinstance(
                    item,
                    dict,
                )
            ]

    except Exception:
        pass

    return []


def get_team_history(
    lab,
    client,
    team_id: str,
    per_page: int = 100,
) -> List[Dict[str, Any]]:
    """
    Récupère un historique large de l'équipe.
    On filtre ensuite strictement AVANT le cutoff.
    """

    key = str(
        team_id
    )

    if key in TEAM_HISTORY_CACHE:
        return TEAM_HISTORY_CACHE[
            key
        ]

    payload = client.get(
        "/football/matches",
        params={
            "team_id":
                team_id,
            "status":
                "finished",
            "per_page":
                per_page,
        },
    )

    rows = api_list(
        lab,
        payload,
    )

    TEAM_HISTORY_CACHE[
        key
    ] = rows

    return rows


def get_league_history(
    lab,
    client,
    competition_id: str,
    season_id: Optional[str],
    per_page: int = 100,
) -> List[Dict[str, Any]]:

    cache_key = (
        str(
            competition_id
        ),
        str(
            season_id
            or ""
        ),
    )

    if cache_key in LEAGUE_HISTORY_CACHE:
        return LEAGUE_HISTORY_CACHE[
            cache_key
        ]

    params: Dict[str, Any] = {
        "competition_id":
            competition_id,
        "status":
            "finished",
        "per_page":
            per_page,
    }

    if season_id:
        params[
            "season_id"
        ] = season_id

    payload = client.get(
        "/football/matches",
        params=params,
    )

    rows = api_list(
        lab,
        payload,
    )

    LEAGUE_HISTORY_CACHE[
        cache_key
    ] = rows

    return rows


# =============================================================================
# TEMPORAL FILTER
# =============================================================================

def match_datetime(
    lab,
    match: Mapping[str, Any],
) -> Optional[datetime]:

    try:
        value = (
            lab.extract_match_date(
                match
            )
        )
    except Exception:
        value = None

    return dt_utc_naive(
        value
    )


def filter_before_cutoff(
    lab,
    matches: Sequence[
        Mapping[str, Any]
    ],
    cutoff: datetime,
    target_match_id: Optional[str] = None,
) -> List[Dict[str, Any]]:

    result: List[
        Dict[str, Any]
    ] = []

    for match in matches:

        if not isinstance(
            match,
            Mapping,
        ):
            continue

        try:
            match_id = (
                lab.extract_match_id(
                    match
                )
            )
        except Exception:
            match_id = None

        if (
            target_match_id
            and match_id is not None
            and str(
                match_id
            )
            == str(
                target_match_id
            )
        ):
            continue

        dt = match_datetime(
            lab,
            match,
        )

        if dt is None:
            # Date inconnue = exclusion, jamais inclusion optimiste.
            continue

        if dt >= cutoff:
            continue

        result.append(
            dict(
                match
            )
        )

    result.sort(
        key=lambda item:
            match_datetime(
                lab,
                item,
            )
            or datetime.min
    )

    return result


# =============================================================================
# TEAM SUMMARY
# =============================================================================

def enrich_xg_before_cutoff(
    lab,
    client,
    matches: Sequence[
        Mapping[str, Any]
    ],
    max_matches: int,
) -> List[Dict[str, Any]]:
    """
    Les matchs reçus sont déjà pré-cutoff.
    On enrichit uniquement les plus récents parmi eux.
    """

    if max_matches <= 0:
        return [
            dict(
                item
            )
            for item in matches
        ]

    try:
        return (
            lab.enrich_history_with_xg(
                client,
                list(
                    matches
                ),
                max_matches=(
                    max_matches
                ),
            )
        )

    except Exception as exc:
        print(
            f"  xG enrichment échoué : "
            f"{exc}"
        )

        return [
            dict(
                item
            )
            for item in matches
        ]


def build_team_summary(
    lab,
    history: Sequence[
        Mapping[str, Any]
    ],
    team_id: Optional[str],
    team_name: Optional[str],
) -> Dict[str, Any]:

    return (
        lab.build_team_pre_match_summary(
            list(
                history
            ),
            team_id=team_id,
            team_name=team_name,
        )
    )


# =============================================================================
# STANDINGS RECONSTRUCTION AS-OF
# =============================================================================

def reconstruct_table_asof(
    lab,
    league_matches: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[
    str,
    Dict[str, Any]
]:
    """
    Reconstruit un classement à partir des matchs de saison terminés AVANT cutoff.

    Tri :
      points desc
      goal difference desc
      goals for desc
      team id asc

    Cette table est temporellement sûre.
    Elle peut différer du classement officiel en cas de groupes,
    retraits de points ou règles de tie-break spécifiques.
    Cette différence est explicitement auditée.
    """

    table: Dict[
        str,
        Dict[str, Any]
    ] = {}

    def ensure_team(
        team_id: Any,
        team_name: Optional[str],
    ) -> Optional[
        Dict[str, Any]
    ]:

        if team_id is None:
            return None

        key = str(
            team_id
        )

        if key not in table:
            table[
                key
            ] = {
                "team_id":
                    key,
                "team_name":
                    team_name,
                "matches_played":
                    0,
                "wins":
                    0,
                "draws":
                    0,
                "losses":
                    0,
                "goals_for":
                    0.0,
                "goals_against":
                    0.0,
                "goal_difference":
                    0.0,
                "points":
                    0.0,
                "rank":
                    None,
            }

        return table[
            key
        ]

    for match in league_matches:

        if not isinstance(
            match,
            Mapping,
        ):
            continue

        home = lab.extract_home_team(
            match
        )

        away = lab.extract_away_team(
            match
        )

        home_id = lab.extract_team_id(
            home
        )

        away_id = lab.extract_team_id(
            away
        )

        home_name = lab.extract_team_name(
            home
        )

        away_name = lab.extract_team_name(
            away
        )

        home_score, away_score = (
            lab.extract_score(
                match
            )
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        h = ensure_team(
            home_id,
            home_name,
        )

        a = ensure_team(
            away_id,
            away_name,
        )

        if h is None or a is None:
            continue

        hs = float(
            home_score
        )

        as_ = float(
            away_score
        )

        h[
            "matches_played"
        ] += 1

        a[
            "matches_played"
        ] += 1

        h[
            "goals_for"
        ] += hs

        h[
            "goals_against"
        ] += as_

        a[
            "goals_for"
        ] += as_

        a[
            "goals_against"
        ] += hs

        if hs > as_:

            h[
                "wins"
            ] += 1

            a[
                "losses"
            ] += 1

            h[
                "points"
            ] += 3.0

        elif hs < as_:

            a[
                "wins"
            ] += 1

            h[
                "losses"
            ] += 1

            a[
                "points"
            ] += 3.0

        else:

            h[
                "draws"
            ] += 1

            a[
                "draws"
            ] += 1

            h[
                "points"
            ] += 1.0

            a[
                "points"
            ] += 1.0

    for row in table.values():

        row[
            "goal_difference"
        ] = (
            row[
                "goals_for"
            ]
            - row[
                "goals_against"
            ]
        )

    ordered = sorted(
        table.values(),
        key=lambda row: (
            -float(
                row[
                    "points"
                ]
            ),
            -float(
                row[
                    "goal_difference"
                ]
            ),
            -float(
                row[
                    "goals_for"
                ]
            ),
            str(
                row[
                    "team_id"
                ]
            ),
        ),
    )

    for rank, row in enumerate(
        ordered,
        start=1,
    ):
        row[
            "rank"
        ] = float(
            rank
        )

    return table


def table_row(
    table: Mapping[
        str,
        Mapping[str, Any]
    ],
    team_id: Any,
) -> Dict[str, Any]:

    if team_id is None:
        return {}

    return dict(
        table.get(
            str(
                team_id
            ),
            {},
        )
    )


# =============================================================================
# H2H
# =============================================================================

def build_h2h_from_history(
    lab,
    team_history: Sequence[
        Mapping[str, Any]
    ],
    home_team_id: Any,
    away_team_id: Any,
    limit: int = 5,
) -> List[Dict[str, Any]]:

    target = {
        str(
            home_team_id
        ),
        str(
            away_team_id
        ),
    }

    result: List[
        Dict[str, Any]
    ] = []

    for match in team_history:

        home = lab.extract_home_team(
            match
        )

        away = lab.extract_away_team(
            match
        )

        ids = {
            str(
                lab.extract_team_id(
                    home
                )
            ),
            str(
                lab.extract_team_id(
                    away
                )
            ),
        }

        if ids == target:
            result.append(
                dict(
                    match
                )
            )

    result.sort(
        key=lambda item:
            match_datetime(
                lab,
                item,
            )
            or datetime.min
    )

    return result[
        -limit:
    ]


# =============================================================================
# LEAGUE FEATURES
# =============================================================================

def league_features_safe(
    lab,
    league_matches: Sequence[
        Mapping[str, Any]
    ],
    client,
    with_second_half: bool,
) -> Dict[str, Any]:
    """
    Pour éviter des centaines d'appels timeline par défaut, la 2H peut être désactivée.
    Les autres league features restent calculées sans invention.
    """

    if not league_matches:
        return {
            "league_home_over25_freq":
                None,
            "league_away_over25_freq":
                None,
            "league_home_over05_freq":
                None,
            "league_away_over05_freq":
                None,
            "league_home_btts_freq":
                None,
            "league_away_btts_freq":
                None,
            "league_2nd_half_over15_freq":
                None,
        }

    if with_second_half:
        try:
            return (
                lab.calculate_league_features(
                    list(
                        league_matches
                    ),
                    client=client,
                )
            )
        except Exception as exc:
            print(
                f"  League 2H enrichissement échoué : "
                f"{exc}"
            )

    # Manually reproduce the safe non-2H league features.
    home_over25 = []
    away_over25 = []
    home_over05 = []
    away_over05 = []
    home_btts = []
    away_btts = []

    for match in league_matches:

        home_score, away_score = (
            lab.extract_score(
                match
            )
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        hs = float(
            home_score
        )

        as_ = float(
            away_score
        )

        total = (
            hs
            + as_
        )

        over25 = (
            total > 2.5
        )

        btts = (
            hs > 0
            and as_ > 0
        )

        # OVER 2.5 is a match-level league frequency,
        # intentionally same home/away semantic.
        home_over25.append(
            over25
        )

        away_over25.append(
            over25
        )

        # OVER 0.5 is side-specific in Fidelity LAB V2.
        home_over05.append(
            hs > 0
        )

        away_over05.append(
            as_ > 0
        )

        home_btts.append(
            btts
        )

        away_btts.append(
            btts
        )

    def avg(
        values: Sequence[
            bool
        ],
    ) -> Optional[float]:

        if not values:
            return None

        return float(
            sum(
                values
            )
            / len(
                values
            )
        )

    return {
        "league_home_over25_freq":
            avg(
                home_over25
            ),
        "league_away_over25_freq":
            avg(
                away_over25
            ),
        "league_home_over05_freq":
            avg(
                home_over05
            ),
        "league_away_over05_freq":
            avg(
                away_over05
            ),
        "league_home_btts_freq":
            avg(
                home_btts
            ),
        "league_away_btts_freq":
            avg(
                away_btts
            ),
        "league_2nd_half_over15_freq":
            None,
    }


# =============================================================================
# BUILD 59 PRE FEATURES
# =============================================================================

def build_prematch_asof(
    lab,
    client,
    target_match: Mapping[str, Any],
    cutoff: datetime,
    xg_limit: int,
    with_second_half: bool,
    preferred_bookmaker: Optional[str],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:

    features = {
        name:
            None
        for name in PRE_MATCH_FEATURES
    }

    match_id = lab.extract_match_id(
        target_match
    )

    home = lab.extract_home_team(
        target_match
    )

    away = lab.extract_away_team(
        target_match
    )

    home_id = lab.extract_team_id(
        home
    )

    away_id = lab.extract_team_id(
        away
    )

    home_name = lab.extract_team_name(
        home
    )

    away_name = lab.extract_team_name(
        away
    )

    competition_id = (
        lab.extract_competition_id(
            target_match
        )
    )

    season_id = (
        lab.extract_season_id(
            target_match
        )
    )

    audit: Dict[
        str,
        Any
    ] = {
        "match_id":
            match_id,
        "home_team":
            home_name,
        "away_team":
            away_name,
        "competition_id":
            competition_id,
        "season_id":
            season_id,
        "kickoff_utc":
            cutoff.isoformat()
            + "Z",
        "home_history_raw":
            0,
        "home_history_before":
            0,
        "away_history_raw":
            0,
        "away_history_before":
            0,
        "league_history_raw":
            0,
        "league_history_before":
            0,
        "h2h_before":
            0,
        "prematch_available":
            0,
        "prematch_missing":
            "",
        "standings_source":
            "reconstructed_from_finished_league_matches_before_kickoff",
        "xg_policy":
            (
                f"official_xg_available_only_recent_{xg_limit}"
                if xg_limit > 0
                else "disabled"
            ),
        "league_2h_policy":
            (
                "verified_timeline_fallback"
                if with_second_half
                else "disabled_nan"
            ),
        "error":
            "",
    }

    # -------------------------------------------------------------------------
    # Team histories
    # -------------------------------------------------------------------------

    home_history_raw = get_team_history(
        lab,
        client,
        home_id,
        per_page=100,
    )

    away_history_raw = get_team_history(
        lab,
        client,
        away_id,
        per_page=100,
    )

    audit[
        "home_history_raw"
    ] = len(
        home_history_raw
    )

    audit[
        "away_history_raw"
    ] = len(
        away_history_raw
    )

    home_history = filter_before_cutoff(
        lab,
        home_history_raw,
        cutoff,
        target_match_id=(
            match_id
        ),
    )

    away_history = filter_before_cutoff(
        lab,
        away_history_raw,
        cutoff,
        target_match_id=(
            match_id
        ),
    )

    audit[
        "home_history_before"
    ] = len(
        home_history
    )

    audit[
        "away_history_before"
    ] = len(
        away_history
    )

    home_history = enrich_xg_before_cutoff(
        lab,
        client,
        home_history,
        max_matches=(
            xg_limit
        ),
    )

    away_history = enrich_xg_before_cutoff(
        lab,
        client,
        away_history,
        max_matches=(
            xg_limit
        ),
    )

    home_summary = build_team_summary(
        lab,
        home_history,
        home_id,
        home_name,
    )

    away_summary = build_team_summary(
        lab,
        away_history,
        away_id,
        away_name,
    )

    # -------------------------------------------------------------------------
    # League history BEFORE kickoff
    # -------------------------------------------------------------------------

    league_history_raw = (
        get_league_history(
            lab,
            client,
            competition_id,
            season_id,
            per_page=100,
        )
        if competition_id
        else []
    )

    audit[
        "league_history_raw"
    ] = len(
        league_history_raw
    )

    league_history = filter_before_cutoff(
        lab,
        league_history_raw,
        cutoff,
        target_match_id=(
            match_id
        ),
    )

    audit[
        "league_history_before"
    ] = len(
        league_history
    )

    # -------------------------------------------------------------------------
    # Reconstructed standings AS-OF
    # -------------------------------------------------------------------------

    table = reconstruct_table_asof(
        lab,
        league_history,
    )

    home_table = table_row(
        table,
        home_id,
    )

    away_table = table_row(
        table,
        away_id,
    )

    # Use reconstructed season table for standings-type fields.
    features[
        "home_team_rank"
    ] = home_table.get(
        "rank"
    )

    features[
        "away_team_rank"
    ] = away_table.get(
        "rank"
    )

    features[
        "home_team_points_standing"
    ] = home_table.get(
        "points"
    )

    features[
        "away_team_points_standing"
    ] = away_table.get(
        "points"
    )

    features[
        "home_team_matches_played"
    ] = home_table.get(
        "matches_played"
    )

    features[
        "away_team_matches_played"
    ] = away_table.get(
        "matches_played"
    )

    features[
        "home_team_wins"
    ] = home_table.get(
        "wins"
    )

    features[
        "away_team_wins"
    ] = away_table.get(
        "wins"
    )

    features[
        "home_team_draws"
    ] = home_table.get(
        "draws"
    )

    features[
        "away_team_draws"
    ] = away_table.get(
        "draws"
    )

    features[
        "home_team_losses"
    ] = home_table.get(
        "losses"
    )

    features[
        "away_team_losses"
    ] = away_table.get(
        "losses"
    )

    features[
        "home_team_goals_for"
    ] = home_table.get(
        "goals_for"
    )

    features[
        "away_team_goals_for"
    ] = away_table.get(
        "goals_for"
    )

    features[
        "home_team_goals_against"
    ] = home_table.get(
        "goals_against"
    )

    features[
        "away_team_goals_against"
    ] = away_table.get(
        "goals_against"
    )

    features[
        "home_team_goal_difference"
    ] = home_table.get(
        "goal_difference"
    )

    features[
        "away_team_goal_difference"
    ] = away_table.get(
        "goal_difference"
    )

    # Legacy aliases retained for schema compatibility.
    features[
        "home_position"
    ] = home_table.get(
        "rank"
    )

    features[
        "away_position"
    ] = away_table.get(
        "rank"
    )

    features[
        "home_points"
    ] = home_table.get(
        "points"
    )

    features[
        "away_points"
    ] = away_table.get(
        "points"
    )

    # -------------------------------------------------------------------------
    # Team-form / team-history statistics
    # -------------------------------------------------------------------------

    mapping = {
        "home_form5":
            home_summary.get(
                "form5"
            ),
        "away_form5":
            away_summary.get(
                "form5"
            ),
        "home_form10":
            home_summary.get(
                "form10"
            ),
        "away_form10":
            away_summary.get(
                "form10"
            ),
        "home_scored_freq":
            home_summary.get(
                "scored_freq"
            ),
        "away_scored_freq":
            away_summary.get(
                "scored_freq"
            ),
        "home_wins_freq":
            home_summary.get(
                "wins_freq"
            ),
        "away_wins_freq":
            away_summary.get(
                "wins_freq"
            ),
        "home_avg_goals_scored":
            home_summary.get(
                "avg_goals_scored"
            ),
        "away_avg_goals_scored":
            away_summary.get(
                "avg_goals_scored"
            ),
        "home_avg_goals_conceded":
            home_summary.get(
                "avg_goals_conceded"
            ),
        "away_avg_goals_conceded":
            away_summary.get(
                "avg_goals_conceded"
            ),
        "home_avg_xg":
            home_summary.get(
                "avg_xg"
            ),
        "away_avg_xg":
            away_summary.get(
                "avg_xg"
            ),
        "home_btts_freq":
            home_summary.get(
                "btts_freq"
            ),
        "away_btts_freq":
            away_summary.get(
                "btts_freq"
            ),
        "home_over25_freq":
            home_summary.get(
                "over25_freq"
            ),
        "away_over25_freq":
            away_summary.get(
                "over25_freq"
            ),
    }

    features.update(
        mapping
    )

    # -------------------------------------------------------------------------
    # H2H BEFORE kickoff
    # -------------------------------------------------------------------------

    h2h = build_h2h_from_history(
        lab,
        home_history,
        home_id,
        away_id,
        limit=5,
    )

    audit[
        "h2h_before"
    ] = len(
        h2h
    )

    h2h_features = (
        lab.calculate_h2h_features(
            h2h
        )
    )

    features.update(
        h2h_features
    )

    # -------------------------------------------------------------------------
    # League features BEFORE kickoff
    # -------------------------------------------------------------------------

    league_features = (
        league_features_safe(
            lab,
            league_history,
            client=client,
            with_second_half=(
                with_second_half
            ),
        )
    )

    features.update(
        league_features
    )

    # -------------------------------------------------------------------------
    # Opening odds of target match
    # -------------------------------------------------------------------------

    odds_key = str(
        match_id
    )

    if odds_key in ODDS_CACHE:

        odds = ODDS_CACHE[
            odds_key
        ]

    else:

        try:
            odds = (
                lab.fetch_prematch_odds(
                    client,
                    match_id,
                    preferred_bookmaker=(
                        preferred_bookmaker
                    ),
                )
            )

        except Exception as exc:

            print(
                f"  Odds indisponibles : "
                f"{exc}"
            )

            odds = {}

        ODDS_CACHE[
            odds_key
        ] = dict(
            odds
            or {}
        )

    features.update(
        odds
        or {}
    )

    available = nonempty_count(
        features,
        PRE_MATCH_FEATURES,
    )

    missing = [
        feature
        for feature in PRE_MATCH_FEATURES
        if safe_float(
            features.get(
                feature
            )
        ) is None
    ]

    audit[
        "prematch_available"
    ] = available

    audit[
        "prematch_missing"
    ] = "|".join(
        missing
    )

    return (
        features,
        audit,
    )


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def load_source(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"{ID_COLUMN} absent du fichier source."
        )

    missing_live = [
        feature
        for feature in LIVE_FEATURES
        if feature not in df.columns
    ]

    if missing_live:
        raise ValueError(
            "37 LIVE incomplètes dans le fichier source :\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_live
            )
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} absent du fichier source."
        )

    # Keep only resolved binary target rows.
    target = pd.to_numeric(
        df[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    df = df[
        target.isin(
            [0, 1]
        )
    ].copy()

    df[
        TARGET_COLUMN
    ] = pd.to_numeric(
        df[
            TARGET_COLUMN
        ],
        errors="coerce",
    ).astype(
        int
    )

    # One row per match; input Clean RAW should already have this.
    df = (
        df
        .drop_duplicates(
            subset=[
                ID_COLUMN
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


def metadata_from_source(
    row: Mapping[str, Any],
) -> Dict[str, Any]:

    names = [
        "match_id",
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "captured_at_utc",
        "captured_at",
        "minute",
        "snapshot_minute",
        "quality",
        "quality_status",
        "target_status",
        "target_source",
        "target_resolution_class",
        "resolution_class",
        "target_confidence",
        "goal_minutes_after_87",
        "valid_goal_minutes_after_87",
        "final_score_home",
        "final_score_away",
        "score_delta_after_snapshot87",
    ]

    result: Dict[
        str,
        Any
    ] = {}

    for name in names:

        if name in row:
            result[
                name
            ] = row[
                name
            ]

    return result


# =============================================================================
# VALIDATION OF REBUILT ROW
# =============================================================================

def validate_rebuilt_row(
    row: Mapping[str, Any],
) -> List[str]:

    reasons: List[
        str
    ] = []

    # Temporal LIVE leakage.
    for feature in (
        "last_goal_minute",
        "last_shot_minute",
        "last_shot_on_target_minute",
        "last_corner_minute",
    ):
        value = safe_float(
            row.get(
                feature
            )
        )

        if (
            value is not None
            and value > 87
        ):
            reasons.append(
                f"future_live_{feature}_gt87"
            )

    # Score consistency.
    h = safe_float(
        row.get(
            "score_87_home"
        )
    )

    a = safe_float(
        row.get(
            "score_87_away"
        )
    )

    if (
        h is not None
        and a is not None
    ):

        diff = safe_float(
            row.get(
                "goal_diff_87"
            )
        )

        if (
            diff is not None
            and not math.isclose(
                diff,
                abs(
                    h - a
                ),
                abs_tol=1e-9,
                rel_tol=0,
            )
        ):
            reasons.append(
                "goal_diff_inconsistent"
            )

        draw = safe_float(
            row.get(
                "is_draw_87"
            )
        )

        expected_draw = (
            1.0
            if h == a
            else 0.0
        )

        if (
            draw is not None
            and not math.isclose(
                draw,
                expected_draw,
                abs_tol=1e-9,
                rel_tol=0,
            )
        ):
            reasons.append(
                "is_draw_inconsistent"
            )

        leading = safe_float(
            row.get(
                "leading_team_87"
            )
        )

        expected_leading = (
            0.0
            if h > a
            else (
                1.0
                if a > h
                else 2.0
            )
        )

        if (
            leading is not None
            and not math.isclose(
                leading,
                expected_leading,
                abs_tol=1e-9,
                rel_tol=0,
            )
        ):
            reasons.append(
                "leading_team_inconsistent"
            )

    # Reconstructed standings arithmetic.
    for side in (
        "home",
        "away",
    ):

        mp = safe_float(
            row.get(
                f"{side}_team_matches_played"
            )
        )

        wins = safe_float(
            row.get(
                f"{side}_team_wins"
            )
        )

        draws = safe_float(
            row.get(
                f"{side}_team_draws"
            )
        )

        losses = safe_float(
            row.get(
                f"{side}_team_losses"
            )
        )

        points = safe_float(
            row.get(
                f"{side}_team_points_standing"
            )
        )

        gf = safe_float(
            row.get(
                f"{side}_team_goals_for"
            )
        )

        ga = safe_float(
            row.get(
                f"{side}_team_goals_against"
            )
        )

        gd = safe_float(
            row.get(
                f"{side}_team_goal_difference"
            )
        )

        if all(
            value is not None
            for value in (
                mp,
                wins,
                draws,
                losses,
            )
        ):
            if not math.isclose(
                mp,
                wins
                + draws
                + losses,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                reasons.append(
                    f"{side}_standing_matches_inconsistent"
                )

        if all(
            value is not None
            for value in (
                points,
                wins,
                draws,
            )
        ):
            if not math.isclose(
                points,
                3.0
                * wins
                + draws,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                reasons.append(
                    f"{side}_standing_points_inconsistent"
                )

        if all(
            value is not None
            for value in (
                gf,
                ga,
                gd,
            )
        ):
            if not math.isclose(
                gd,
                gf - ga,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                reasons.append(
                    f"{side}_standing_gd_inconsistent"
                )

    return sorted(
        set(
            reasons
        )
    )


# =============================================================================
# EXPORT
# =============================================================================

def save_progress(
    rebuilt_rows: List[
        Dict[str, Any]
    ],
    audit_rows: List[
        Dict[str, Any]
    ],
    rejected_rows: List[
        Dict[str, Any]
    ],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv"
    )

    full_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_REBUILT_FULL_v2.csv"
    )

    audit_path = (
        output_dir
        / "prematch_rebuild_audit_v2.csv"
    )

    reject_path = (
        output_dir
        / "rejected_rebuild_v2.csv"
    )

    rebuilt_df = pd.DataFrame(
        rebuilt_rows
    )

    if not rebuilt_df.empty:

        rebuilt_df.to_csv(
            raw_path,
            index=False,
            encoding="utf-8-sig",
        )

        full_mask = (
            rebuilt_df[
                FEATURES
            ]
            .notna()
            .sum(
                axis=1
            )
            == 96
        )

        full_df = rebuilt_df[
            full_mask
        ].copy()

        if (
            "_validation_errors"
            in full_df.columns
        ):
            full_df = full_df[
                full_df[
                    "_validation_errors"
                ]
                .fillna(
                    ""
                )
                == ""
            ]

        full_df.to_csv(
            full_path,
            index=False,
            encoding="utf-8-sig",
        )

    else:

        pd.DataFrame().to_csv(
            raw_path,
            index=False,
        )

        pd.DataFrame().to_csv(
            full_path,
            index=False,
        )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        rejected_rows
    ).to_csv(
        reject_path,
        index=False,
        encoding="utf-8-sig",
    )


def build_coverage_report(
    rebuilt_rows: Sequence[
        Mapping[str, Any]
    ],
) -> pd.DataFrame:

    if not rebuilt_rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "group",
                "available",
                "missing",
                "coverage_rate",
            ]
        )

    df = pd.DataFrame(
        rebuilt_rows
    )

    rows = []

    for feature in FEATURES:

        available = int(
            pd.to_numeric(
                df[
                    feature
                ],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        total = len(
            df
        )

        rows.append({
            "feature":
                feature,
            "group":
                (
                    "PRE"
                    if feature in PRE_MATCH_FEATURES
                    else "LIVE"
                ),
            "available":
                available,
            "missing":
                total - available,
            "coverage_rate":
                (
                    available
                    / total
                    if total
                    else 0.0
                ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "group",
                "coverage_rate",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )


def write_report(
    rebuilt_rows: Sequence[
        Mapping[str, Any]
    ],
    rejected_rows: Sequence[
        Mapping[str, Any]
    ],
    audit_rows: Sequence[
        Mapping[str, Any]
    ],
    client,
    source_path: Path,
    output_dir: Path,
) -> None:

    rebuilt_df = pd.DataFrame(
        rebuilt_rows
    )

    total = len(
        rebuilt_df
    )

    if total:

        observed = (
            rebuilt_df[
                FEATURES
            ]
            .notna()
            .sum(
                axis=1
            )
        )

        full = int(
            (
                observed
                == 96
            ).sum()
        )

        pre_full = int(
            (
                rebuilt_df[
                    PRE_MATCH_FEATURES
                ]
                .notna()
                .sum(
                    axis=1
                )
                == 59
            ).sum()
        )

        positive = int(
            (
                rebuilt_df[
                    TARGET_COLUMN
                ]
                == 1
            ).sum()
        )

        negative = int(
            (
                rebuilt_df[
                    TARGET_COLUMN
                ]
                == 0
            ).sum()
        )

        target_rate = float(
            rebuilt_df[
                TARGET_COLUMN
            ].mean()
        )

    else:

        full = 0
        pre_full = 0
        positive = 0
        negative = 0
        target_rate = 0.0

    errors = defaultdict(
        int
    )

    for row in rejected_rows:

        for item in clean(
            row.get(
                "reason"
            )
        ).split(
            "|"
        ):

            if item:
                errors[
                    item
                ] += 1

    lines = [
        "# Rebuild Clean PRE-MATCH V2",
        "",
        "## Objectif",
        "",
        "Reconstruire les 59 PRE-MATCH avec un cutoff strict au kickoff du match cible.",
        "Les 37 LIVE restent celles réellement capturées à 87'.",
        "",
        "## Règles anti-fuite",
        "",
        "- historique équipe : match_date < kickoff",
        "- H2H : match_date < kickoff",
        "- historique ligue : match_date < kickoff",
        "- classement : reconstruit uniquement avec les matchs terminés pré-kickoff",
        "- xG : uniquement valeurs réellement disponibles sur matchs pré-kickoff",
        "- aucune imputation",
        "- aucune valeur post-kickoff utilisée volontairement",
        "",
        "## Résultats",
        "",
        f"- Source : `{source_path}`",
        f"- Matchs reconstruits : **{total}**",
        f"- PRE 59/59 : **{pre_full}**",
        f"- FULL 96/96 : **{full}**",
        f"- Positifs : **{positive}**",
        f"- Négatifs : **{negative}**",
        f"- Target rate : **{target_rate:.2%}**",
        f"- Rejets : **{len(rejected_rows)}**",
        f"- Requêtes API : **{getattr(client, 'request_count', 0)}**",
        f"- Retry HTTP 429 : **{getattr(client, 'retry_429_count', 0)}**",
        "",
        "## Important",
        "",
        "Le classement reconstruit est temporellement sûr mais peut différer du classement officiel "
        "dans les compétitions avec groupes, retraits de points ou tie-breaks spéciaux.",
        "",
        "Les anciennes PRE du snapshot ne sont pas utilisées comme nouvelles valeurs d'entraînement.",
        "",
        "## Rejets",
        "",
    ]

    if errors:

        for reason, count in sorted(
            errors.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            lines.append(
                f"- `{reason}` : {count}"
            )

    else:

        lines.append(
            "- Aucun rejet."
        )

    lines.extend([
        "",
        "## Prochaine étape",
        "",
        "Auditer `Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv` avant tout entraînement.",
        "Le fichier FULL ne doit être utilisé pour l'entraînement qu'après audit statistique et temporel.",
    ])

    report_path = (
        output_dir
        / "rebuild_clean_prematch_v2_report.md"
    )

    report_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit les 59 PRE-MATCH as-of kickoff "
            "pour le nouveau dataset prospectif."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "CSV source. Défaut : Clean_Dataset_Goal_After_87_RAW_v2.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Dossier de sortie."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Nombre maximal de matchs à reconstruire. 0 = tous."
        ),
    )

    parser.add_argument(
        "--xg-limit",
        type=int,
        default=10,
        help=(
            "Nombre maximal de matchs pré-kickoff récents à enrichir en xG par équipe. "
            "0 = désactivé."
        ),
    )

    parser.add_argument(
        "--with-league-second-half",
        action="store_true",
        help=(
            "Tente de reconstruire league_2nd_half_over15_freq via timelines historiques. "
            "Plus lent et plus coûteux en appels API."
        ),
    )

    parser.add_argument(
        "--preferred-bookmaker",
        default=None,
        help=(
            "Bookmaker préféré pour opening odds. Sinon logique Fidelity LAB."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reprend à partir du CSV RAW déjà produit."
        ),
    )

    parser.add_argument(
        "--sleep-between-matches",
        type=float,
        default=1.0,
        help=(
            "Pause entre matchs cibles. Défaut : 1 seconde."
        ),
    )

    args = parser.parse_args()

    source_path = resolve_path(
        args.input,
        INPUT_CANDIDATES,
    )

    output_dir = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        "REBUILD CLEAN PRE-MATCH V2 — AS-OF KICKOFF"
    )
    print("=" * 100)

    lab = load_lab()

    api_key = clean(
        getattr(
            lab,
            "API_KEY",
            "",
        )
    )

    if not api_key:
        print(
            "ERREUR : STATS_API_KEY absente du .env"
        )
        return 2

    client = build_rate_limited_client(
        lab
    )

    source_df = load_source(
        source_path
    )

    if args.limit > 0:
        source_df = source_df.head(
            args.limit
        ).copy()

    print(
        f"Source : {source_path}"
    )

    print(
        f"Matchs à traiter : "
        f"{len(source_df)}"
    )

    rebuilt_rows: List[
        Dict[str, Any]
    ] = []

    audit_rows: List[
        Dict[str, Any]
    ] = []

    rejected_rows: List[
        Dict[str, Any]
    ] = []

    raw_progress_path = (
        output_dir
        / "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv"
    )

    processed_ids = set()

    if (
        args.resume
        and raw_progress_path.exists()
    ):

        try:
            previous = pd.read_csv(
                raw_progress_path
            )

            if ID_COLUMN in previous.columns:

                rebuilt_rows = (
                    previous
                    .to_dict(
                        orient="records"
                    )
                )

                processed_ids = {
                    str(
                        item
                    )
                    for item in previous[
                        ID_COLUMN
                    ].dropna()
                }

                print(
                    f"Resume : "
                    f"{len(processed_ids)} match(s) déjà reconstruits."
                )

        except Exception as exc:

            print(
                f"Resume impossible : {exc}"
            )

    for position, (
        index,
        source_row,
    ) in enumerate(
        source_df.iterrows(),
        start=1,
    ):

        match_id = clean(
            source_row.get(
                ID_COLUMN
            )
        )

        if not match_id:

            rejected_rows.append({
                "match_id":
                    "",
                "reason":
                    "missing_match_id",
            })

            continue

        if match_id in processed_ids:

            continue

        print()
        print("-" * 100)
        print(
            f"[{position}/{len(source_df)}] "
            f"{match_id}"
        )
        print("-" * 100)

        try:

            target_match = get_match_detail(
                lab,
                client,
                match_id,
            )

            if not target_match:

                rejected_rows.append({
                    "match_id":
                        match_id,
                    "reason":
                        "target_match_detail_unavailable",
                })

                continue

            kickoff_value = (
                lab.extract_match_date(
                    target_match
                )
            )

            cutoff = dt_utc_naive(
                kickoff_value
            )

            if cutoff is None:

                rejected_rows.append({
                    "match_id":
                        match_id,
                    "reason":
                        "target_kickoff_unparseable",
                })

                continue

            home_name = (
                lab.extract_team_name(
                    lab.extract_home_team(
                        target_match
                    )
                )
            )

            away_name = (
                lab.extract_team_name(
                    lab.extract_away_team(
                        target_match
                    )
                )
            )

            print(
                f"Match : "
                f"{home_name} vs {away_name}"
            )

            print(
                f"Kickoff cutoff : "
                f"{cutoff.isoformat()}Z"
            )

            prematch, audit = build_prematch_asof(
                lab=lab,
                client=client,
                target_match=target_match,
                cutoff=cutoff,
                xg_limit=max(
                    0,
                    args.xg_limit,
                ),
                with_second_half=(
                    args.with_league_second_half
                ),
                preferred_bookmaker=(
                    args.preferred_bookmaker
                ),
            )

            output_row = metadata_from_source(
                source_row
            )

            # Always use canonical match metadata from target API when available.
            output_row[
                "match_id"
            ] = match_id

            output_row[
                "home_team"
            ] = (
                home_name
                or output_row.get(
                    "home_team"
                )
            )

            output_row[
                "away_team"
            ] = (
                away_name
                or output_row.get(
                    "away_team"
                )
            )

            output_row[
                "competition_id"
            ] = (
                lab.extract_competition_id(
                    target_match
                )
                or output_row.get(
                    "competition_id"
                )
            )

            output_row[
                "season_id"
            ] = (
                lab.extract_season_id(
                    target_match
                )
                or output_row.get(
                    "season_id"
                )
            )

            output_row[
                "kickoff_utc"
            ] = (
                cutoff.isoformat()
                + "Z"
            )

            # Replace all 59 PRE with rebuilt values.
            for feature in PRE_MATCH_FEATURES:

                value = prematch.get(
                    feature
                )

                output_row[
                    feature
                ] = (
                    np.nan
                    if safe_float(
                        value
                    ) is None
                    else float(
                        value
                    )
                )

            # Preserve all 37 LIVE from actual snapshot.
            for feature in LIVE_FEATURES:

                value = safe_float(
                    source_row.get(
                        feature
                    )
                )

                output_row[
                    feature
                ] = (
                    np.nan
                    if value is None
                    else value
                )

            output_row[
                TARGET_COLUMN
            ] = int(
                source_row[
                    TARGET_COLUMN
                ]
            )

            output_row[
                "_prematch_available"
            ] = nonempty_count(
                output_row,
                PRE_MATCH_FEATURES,
            )

            output_row[
                "_live_available"
            ] = nonempty_count(
                output_row,
                LIVE_FEATURES,
            )

            output_row[
                "_total_available"
            ] = nonempty_count(
                output_row,
                FEATURES,
            )

            validation_errors = (
                validate_rebuilt_row(
                    output_row
                )
            )

            output_row[
                "_validation_errors"
            ] = "|".join(
                validation_errors
            )

            output_row[
                "_rebuilt_at_utc"
            ] = now_utc_iso()

            rebuilt_rows.append(
                output_row
            )

            audit[
                "total_available"
            ] = output_row[
                "_total_available"
            ]

            audit[
                "live_available"
            ] = output_row[
                "_live_available"
            ]

            audit[
                "validation_errors"
            ] = output_row[
                "_validation_errors"
            ]

            audit[
                "api_requests_so_far"
            ] = getattr(
                client,
                "request_count",
                0,
            )

            audit_rows.append(
                audit
            )

            if validation_errors:

                rejected_rows.append({
                    "match_id":
                        match_id,
                    "reason":
                        "|".join(
                            validation_errors
                        ),
                })

            print(
                f"PRE : "
                f"{output_row['_prematch_available']}/59 "
                f"| LIVE : "
                f"{output_row['_live_available']}/37 "
                f"| TOTAL : "
                f"{output_row['_total_available']}/96"
            )

            if validation_errors:

                print(
                    "⚠️ Validation : "
                    + ", ".join(
                        validation_errors
                    )
                )

            else:

                print(
                    "✅ Validation structurelle : OK"
                )

            # Persist after every match.
            save_progress(
                rebuilt_rows,
                audit_rows,
                rejected_rows,
                output_dir,
            )

        except KeyboardInterrupt:

            print()
            print(
                "Interruption utilisateur. "
                "Sauvegarde de la progression..."
            )

            save_progress(
                rebuilt_rows,
                audit_rows,
                rejected_rows,
                output_dir,
            )

            return 130

        except Exception as exc:

            print(
                f"ERREUR {match_id} : "
                f"{type(exc).__name__}: {exc}"
            )

            rejected_rows.append({
                "match_id":
                    match_id,
                "reason":
                    (
                        "unhandled_error:"
                        f"{type(exc).__name__}:"
                        f"{exc}"
                    ),
            })

            save_progress(
                rebuilt_rows,
                audit_rows,
                rejected_rows,
                output_dir,
            )

        if args.sleep_between_matches > 0:

            time.sleep(
                args.sleep_between_matches
            )

    # =========================================================================
    # FINAL EXPORT
    # =========================================================================

    save_progress(
        rebuilt_rows,
        audit_rows,
        rejected_rows,
        output_dir,
    )

    coverage_df = build_coverage_report(
        rebuilt_rows
    )

    coverage_path = (
        output_dir
        / "feature_coverage_v2.csv"
    )

    coverage_df.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_report(
        rebuilt_rows=(
            rebuilt_rows
        ),
        rejected_rows=(
            rejected_rows
        ),
        audit_rows=(
            audit_rows
        ),
        client=client,
        source_path=(
            source_path
        ),
        output_dir=(
            output_dir
        ),
    )

    rebuilt_df = pd.DataFrame(
        rebuilt_rows
    )

    if rebuilt_df.empty:

        full_count = 0
        pre_full_count = 0

    else:

        pre_full_count = int(
            (
                rebuilt_df[
                    PRE_MATCH_FEATURES
                ]
                .notna()
                .sum(
                    axis=1
                )
                == 59
            ).sum()
        )

        full_count = int(
            (
                rebuilt_df[
                    FEATURES
                ]
                .notna()
                .sum(
                    axis=1
                )
                == 96
            ).sum()
        )

    print()
    print("=" * 100)
    print(
        "REBUILD TERMINÉ"
    )
    print("=" * 100)

    print(
        f"Matchs reconstruits : "
        f"{len(rebuilt_rows)}"
    )

    print(
        f"PRE 59/59 : "
        f"{pre_full_count}"
    )

    print(
        f"FULL 96/96 : "
        f"{full_count}"
    )

    print(
        f"Rejets / anomalies : "
        f"{len(rejected_rows)}"
    )

    print(
        f"Requêtes API : "
        f"{getattr(client, 'request_count', 0)}"
    )

    print(
        f"HTTP 429 retentés : "
        f"{getattr(client, 'retry_429_count', 0)}"
    )

    print()
    print(
        "Sorties :"
    )

    for name in (
        "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv",
        "Clean_Dataset_Goal_After_87_REBUILT_FULL_v2.csv",
        "prematch_rebuild_audit_v2.csv",
        "feature_coverage_v2.csv",
        "rejected_rebuild_v2.csv",
        "rebuild_clean_prematch_v2_report.md",
    ):

        print(
            f"  {output_dir / name}"
        )

    print()
    print(
        "Aucune imputation n'a été appliquée."
    )

    print(
        "Le scanner de production n'a pas été modifié."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
