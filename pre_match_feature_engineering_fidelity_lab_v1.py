"""
FIDELITY LAB V2 — MODULE EXPÉRIMENTAL

NE PAS UTILISER EN PRODUCTION SANS VALIDATION.

Corrections / améliorations LAB:
1) form5/form10 -> somme de points 3/1/0, échelles 0-15 / 0-30.
2) league_home_over05_freq -> fréquence domicile marque >=1.
3) league_away_over05_freq -> fréquence extérieur marque >=1.
4) league_2nd_half_over15_freq -> fallback par timeline historique vérifiée.

Le reste du module est conservé identique à la version source.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRE-MATCH FEATURE ENGINEERING
=============================

Construit les 59 features PRE-MATCH utilisées par le modèle ML.

Sources TheStatsAPI :
- matchs
- historique des équipes
- standings
- H2H
- historique de ligue
- cotes pré-match

Important :
- Les features sont calculées uniquement avec des informations
  disponibles AVANT le match cible.
- Pour les cotes pré-match, on utilise "opening", jamais "last_seen".
- L'xG historique est enrichi via /football/matches/{match_id}/stats
  lorsque la liste des matchs ne contient pas directement l'xG.
- league_2nd_half_over15_freq est dérivée du score final et du score mi-temps.
"""

import os
import json
import time
import math
import re
import requests
import numpy as np
from datetime import datetime
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE, override=False)

print(f".env chargé : {ENV_FILE}")

API_KEY = os.getenv("STATS_API_KEY")

API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api"
)

API_TIMEOUT = int(
    os.getenv("STATS_API_TIMEOUT", "20")
)

MIN_SCAN_MINUTE = int(
    os.getenv("MIN_SCAN_MINUTE", "85")
)

MAX_SCAN_MINUTE = int(
    os.getenv("MAX_SCAN_MINUTE", "87")
)


# Nombre maximal de matchs historiques récents à enrichir via /stats
# pour calculer l'xG moyen. Le cache évite de rappeler le même match.
XG_HISTORY_MATCH_LIMIT = int(
    os.getenv("XG_HISTORY_MATCH_LIMIT", "10")
)

_MATCH_STATS_CACHE = {}

# Cache des détails de matchs historiques, utilisé pour récupérer
# le score mi-temps lorsque la liste ne l'inclut pas.
_MATCH_DETAILS_CACHE = {}

# FIDELITY LAB V2:
# cache des timelines historiques utilisées uniquement pour reconstruire
# league_2nd_half_over15_freq lorsque le score mi-temps est absent.
_MATCH_TIMELINE_CACHE = {}

LEAGUE_HT_DETAIL_LIMIT = int(
    os.getenv("LEAGUE_HT_DETAIL_LIMIT", "40")
)

# Limite volontaire pour éviter de multiplier les appels API.
LEAGUE_TIMELINE_LIMIT = int(
    os.getenv("LEAGUE_TIMELINE_LIMIT", "30")
)


# =============================================================================
# FEATURES ATTENDUES PAR LE MODÈLE
# =============================================================================

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


# =============================================================================
# UTILITAIRES
# =============================================================================

def safe_float(value):
    """
    Convertit une valeur en float.
    Retourne None si impossible.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

            value = value.replace(",", ".")

        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def safe_int(value):
    """
    Conversion robuste en int.

    Accepte notamment :
    - 85
    - "85"
    - "85'"
    - "85+2"
    - "85+2'"
    - 85.0
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()

    if not text:
        return None

    # Retirer les apostrophes et espaces
    text = text.replace("'", "").replace("′", "")
    text = text.replace(" ", "")

    # Cas : 85+2
    match = re.match(r"^(\d+)\+(\d+)$", text)

    if match:
        try:
            return int(match.group(1)) + int(match.group(2))
        except ValueError:
            return None

    # Cas : 85.0
    try:
        return int(float(text))
    except (TypeError, ValueError):
        pass

    # Chercher le premier entier
    match = re.search(r"\d+", text)

    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None

    return None


def normalize_team_name(name):
    """
    Normalise légèrement un nom d'équipe pour les comparaisons.
    """

    if name is None:
        return ""

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )


def first_not_none(*values):
    """
    Retourne la première valeur non None.
    """

    for value in values:

        if value is not None:
            return value

    return None


def deep_get(data, *keys):
    """
    Accès sécurisé à un dictionnaire imbriqué.
    """

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def as_list(data):
    """
    Transforme différentes structures API en liste.

    Supporte notamment :

    [
        {...},
        {...}
    ]

    {
        "data": [
            {...}
        ]
    }

    {
        "data": {
            "matches": [
                {...}
            ]
        }
    }

    {
        "matches": [
            {...}
        ]
    }
    """

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    # Niveau courant
    for key in [
        "matches",
        "results",
        "items",
        "fixtures",
        "events",
        "bookmakers",
    ]:

        value = data.get(key)

        if isinstance(value, list):
            return value

    # Wrapper "data"
    nested_data = data.get("data")

    if isinstance(nested_data, list):
        return nested_data

    if isinstance(nested_data, dict):

        for key in [
            "matches",
            "results",
            "items",
            "fixtures",
            "events",
            "bookmakers",
        ]:

            value = nested_data.get(key)

            if isinstance(value, list):
                return value

    return []


def parse_date(value):
    """
    Parse une date ISO/API.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).strip()

    if not text:
        return None

    candidates = [
        text,
        text.replace("Z", "+00:00"),
    ]

    for candidate in candidates:

        try:
            return datetime.fromisoformat(candidate)

        except ValueError:
            pass

    return None


# =============================================================================
# CLIENT API
# =============================================================================

class StatsAPIClient:

    def __init__(
        self,
        api_key=None,
        base_url=API_BASE_URL,
        timeout=API_TIMEOUT,
    ):

        self.api_key = api_key or API_KEY
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ML-Goal-After-87/1.1",
        })

    def request(
        self,
        method,
        endpoint,
        params=None,
    ):
        """
        Requête HTTP sécurisée.
        """

        url = (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
        )

        try:

            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
            )

        except requests.RequestException as exc:

            print(
                f"Erreur API : {exc}"
            )

            return None

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

    def get(
        self,
        endpoint,
        params=None,
    ):

        return self.request(
            "GET",
            endpoint,
            params=params,
        )


# =============================================================================
# EXTRACTION MATCH
# =============================================================================

def extract_match_id(match):

    if not isinstance(match, dict):
        return None

    return first_not_none(
        match.get("match_id"),
        match.get("id"),
        deep_get(
            match,
            "data",
            "match_id",
        ),
    )


def extract_team_id(team):

    if not isinstance(team, dict):
        return None

    return first_not_none(
        team.get("team_id"),
        team.get("id"),
    )


def extract_team_name(team):

    if not isinstance(team, dict):
        return None

    return first_not_none(
        team.get("name"),
        team.get("team_name"),
        team.get("short_name"),
    )


def extract_home_team(match):

    if not isinstance(match, dict):
        return None

    home = first_not_none(
        match.get("home_team"),
        match.get("home"),
    )

    if isinstance(home, dict):
        return home

    if isinstance(home, str):
        return {
            "name": home
        }

    return {}


def extract_away_team(match):

    if not isinstance(match, dict):
        return None

    away = first_not_none(
        match.get("away_team"),
        match.get("away"),
    )

    if isinstance(away, dict):
        return away

    if isinstance(away, str):
        return {
            "name": away
        }

    return {}


def extract_match_date(match):

    if not isinstance(match, dict):
        return None

    return first_not_none(
        match.get("utc_date"),
        match.get("kickoff_utc"),
        match.get("date"),
        match.get("match_date"),
        match.get("start_time"),
        match.get("kickoff"),
        match.get("datetime"),
        match.get("scheduled_at"),
    )


def extract_competition_id(match):

    if not isinstance(match, dict):
        return None

    competition = first_not_none(
        match.get("competition"),
        match.get("league"),
    )

    if isinstance(competition, dict):

        return first_not_none(
            competition.get("competition_id"),
            competition.get("id"),
        )

    return first_not_none(
        match.get("competition_id"),
        match.get("league_id"),
    )


def extract_season_id(match):

    if not isinstance(match, dict):
        return None

    season = match.get("season")

    if isinstance(season, dict):

        return first_not_none(
            season.get("season_id"),
            season.get("id"),
        )

    return match.get("season_id")


# =============================================================================
# HISTORIQUE DES MATCHS
# =============================================================================

def extract_score(match):

    if not isinstance(match, dict):
        return None, None

    home_score = first_not_none(
        match.get("home_score"),
        match.get("home_goals"),
        deep_get(match, "home", "score"),
        deep_get(match, "home_team", "score"),
        deep_get(
            match,
            "score",
            "home",
        ),
        deep_get(
            match,
            "scores",
            "home",
        ),
        deep_get(
            match,
            "final_score",
            "home",
        ),
    )

    away_score = first_not_none(
        match.get("away_score"),
        match.get("away_goals"),
        deep_get(match, "away", "score"),
        deep_get(match, "away_team", "score"),
        deep_get(
            match,
            "score",
            "away",
        ),
        deep_get(
            match,
            "scores",
            "away",
        ),
        deep_get(
            match,
            "final_score",
            "away",
        ),
    )

    return (
        safe_float(home_score),
        safe_float(away_score),
    )


def extract_xg(match):

    if not isinstance(match, dict):
        return None, None

    home_xg = first_not_none(
        match.get("home_xg"),
        match.get("home_expected_goals"),
        deep_get(
            match,
            "xg",
            "home",
        ),
        deep_get(
            match,
            "expected_goals",
            "home",
        ),
        deep_get(
            match,
            "stats",
            "expected_goals",
            "home",
        ),
    )

    away_xg = first_not_none(
        match.get("away_xg"),
        match.get("away_expected_goals"),
        deep_get(
            match,
            "xg",
            "away",
        ),
        deep_get(
            match,
            "expected_goals",
            "away",
        ),
        deep_get(
            match,
            "stats",
            "expected_goals",
            "away",
        ),
    )

    return (
        safe_float(home_xg),
        safe_float(away_xg),
    )


def extract_xg_from_stats_response(data):
    """
    Extrait l'xG home/away depuis GET /football/matches/{match_id}/stats.
    """

    if not isinstance(data, dict):
        return None, None

    payload = data.get("data")

    if not isinstance(payload, dict):
        payload = data

    home_xg = first_not_none(
        deep_get(payload, "overview", "expected_goals", "all", "home"),
        deep_get(payload, "stats", "expected_goals", "all", "home"),
        deep_get(payload, "expected_goals", "all", "home"),
        deep_get(payload, "home", "xg"),
        deep_get(payload, "home", "expected_goals"),
    )

    away_xg = first_not_none(
        deep_get(payload, "overview", "expected_goals", "all", "away"),
        deep_get(payload, "stats", "expected_goals", "all", "away"),
        deep_get(payload, "expected_goals", "all", "away"),
        deep_get(payload, "away", "xg"),
        deep_get(payload, "away", "expected_goals"),
    )

    return (
        safe_float(home_xg),
        safe_float(away_xg),
    )


def get_match_stats_cached(
    client,
    match_id,
):
    """
    Récupère les stats d'un match avec cache mémoire.
    """

    if not match_id:
        return None

    key = str(match_id)

    if key in _MATCH_STATS_CACHE:
        return _MATCH_STATS_CACHE[key]

    data = client.get(
        f"/football/matches/{match_id}/stats"
    )

    _MATCH_STATS_CACHE[key] = data

    return data


def enrich_history_with_xg(
    client,
    history,
    max_matches=XG_HISTORY_MATCH_LIMIT,
):
    """
    Enrichit les matchs historiques récents avec leur xG officiel.

    Ne remplit jamais une absence avec 0.
    """

    if not isinstance(history, list):
        return []

    enriched = [
        dict(match)
        if isinstance(match, dict)
        else match
        for match in history
    ]

    indexed = []

    for index, match in enumerate(enriched):

        if not isinstance(match, dict):
            continue

        dt = parse_date(
            extract_match_date(match)
        ) or datetime.min

        indexed.append(
            (
                dt,
                index,
            )
        )

    indexed.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    attempted = 0

    for _, index in indexed:

        if attempted >= max_matches:
            break

        match = enriched[index]

        current_home_xg, current_away_xg = (
            extract_xg(match)
        )

        if (
            current_home_xg is not None
            and current_away_xg is not None
        ):
            continue

        if match.get("xg_available") is False:
            continue

        match_id = extract_match_id(match)

        if not match_id:
            continue

        attempted += 1

        stats = get_match_stats_cached(
            client,
            match_id,
        )

        home_xg, away_xg = (
            extract_xg_from_stats_response(
                stats
            )
        )

        if (
            current_home_xg is None
            and home_xg is not None
        ):
            match["home_xg"] = home_xg

        if (
            current_away_xg is None
            and away_xg is not None
        ):
            match["away_xg"] = away_xg

    return enriched


def extract_half_time_score(match):
    """
    Extrait le score à la mi-temps.
    """

    if not isinstance(match, dict):
        return None, None

    home_team = extract_home_team(match)
    away_team = extract_away_team(match)

    home_ht = first_not_none(
        match.get("_detail_home_ht_score"),
        deep_get(match, "home", "ht_score"),
        deep_get(match, "home", "half_time_score"),
        home_team.get("ht_score")
        if isinstance(home_team, dict)
        else None,
        home_team.get("half_time_score")
        if isinstance(home_team, dict)
        else None,
        match.get("home_ht_score"),
        match.get("home_half_time_score"),
        deep_get(match, "score", "ht_home"),
        deep_get(match, "score", "half_time", "home"),
        deep_get(match, "score", "halftime", "home"),
        deep_get(match, "half_time", "home"),
    )

    away_ht = first_not_none(
        match.get("_detail_away_ht_score"),
        deep_get(match, "away", "ht_score"),
        deep_get(match, "away", "half_time_score"),
        away_team.get("ht_score")
        if isinstance(away_team, dict)
        else None,
        away_team.get("half_time_score")
        if isinstance(away_team, dict)
        else None,
        match.get("away_ht_score"),
        match.get("away_half_time_score"),
        deep_get(match, "score", "ht_away"),
        deep_get(match, "score", "half_time", "away"),
        deep_get(match, "score", "halftime", "away"),
        deep_get(match, "half_time", "away"),
    )

    if home_ht is None or away_ht is None:

        ht_text = first_not_none(
            match.get("ht_score"),
            match.get("half_time_score"),
            deep_get(match, "score", "ht_score"),
        )

        if isinstance(ht_text, str):

            found = re.match(
                r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$",
                ht_text,
            )

            if found:
                home_ht = first_not_none(
                    home_ht,
                    found.group(1),
                )
                away_ht = first_not_none(
                    away_ht,
                    found.group(2),
                )

    return (
        safe_float(home_ht),
        safe_float(away_ht),
    )


def unwrap_match_payload(data):
    """
    Retourne l'objet match principal d'une réponse /football/matches/{id}.
    """

    if not isinstance(data, dict):
        return {}

    payload = data.get("data")

    if isinstance(payload, dict):
        nested_match = payload.get("match")
        if isinstance(nested_match, dict):
            return nested_match
        return payload

    return data


def get_match_details_cached(
    client,
    match_id,
):
    """
    Récupère le détail d'un match historique avec cache.
    """

    if not match_id:
        return None

    key = str(match_id)

    if key in _MATCH_DETAILS_CACHE:
        return _MATCH_DETAILS_CACHE[key]

    data = client.get(
        f"/football/matches/{match_id}"
    )

    _MATCH_DETAILS_CACHE[key] = data
    return data


def get_match_timeline_cached(
    client,
    match_id,
):
    """
    Récupère la timeline d'un match historique avec cache.
    """

    if not match_id:
        return None

    key = str(match_id)

    if key in _MATCH_TIMELINE_CACHE:
        return _MATCH_TIMELINE_CACHE[key]

    data = client.get(
        f"/football/matches/{match_id}/timeline"
    )

    _MATCH_TIMELINE_CACHE[key] = data
    return data


def extract_timeline_events(data):
    return as_list(data)


def extract_historical_event_minute(event):
    if not isinstance(event, dict):
        return None

    return safe_int(
        first_not_none(
            event.get("minute"),
            event.get("elapsed_minutes"),
            event.get("elapsed"),
            deep_get(event, "time", "minute"),
            deep_get(event, "time", "elapsed"),
        )
    )


def extract_historical_event_type(event):
    if not isinstance(event, dict):
        return ""

    value = first_not_none(
        event.get("type"),
        event.get("event_type"),
        event.get("incident_type"),
        event.get("kind"),
        event.get("name"),
    )

    if isinstance(value, dict):
        value = first_not_none(
            value.get("type"),
            value.get("name"),
            value.get("value"),
        )

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def extract_historical_event_period(event):
    if not isinstance(event, dict):
        return ""

    value = first_not_none(
        event.get("period"),
        event.get("match_period"),
        event.get("half"),
        deep_get(event, "time", "period"),
    )

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def second_half_goals_from_verified_timeline(
    client,
    match,
):
    """
    Retourne le nombre de buts de 2e mi-temps uniquement si la timeline
    reproduit exactement le score final total.
    """

    if not isinstance(match, dict):
        return None

    home_score, away_score = extract_score(match)

    if home_score is None or away_score is None:
        return None

    final_total = int(home_score + away_score)

    match_id = extract_match_id(match)

    if not match_id:
        return None

    timeline_response = get_match_timeline_cached(
        client,
        match_id,
    )

    events = extract_timeline_events(
        timeline_response
    )

    if not events:
        return None

    total_valid_goals = 0
    second_half_goal_count = 0

    for event in events:

        if not isinstance(event, dict):
            continue

        event_type = extract_historical_event_type(
            event
        )

        if "goal" not in event_type:
            continue

        if any(
            token in event_type
            for token in (
                "cancel",
                "disallow",
                "disallowed",
                "var_cancel",
                "no_goal",
            )
        ):
            continue

        minute = extract_historical_event_minute(
            event
        )

        if minute is None:
            continue

        period = extract_historical_event_period(
            event
        )

        if any(
            token in period
            for token in (
                "extra",
                "penalt",
                "shootout",
            )
        ):
            continue

        total_valid_goals += 1

        is_second_half = (
            "second_half" in period
            or period in ("2", "2h", "second")
            or (
                not period
                and 46 <= minute <= 90
            )
        )

        if is_second_half:
            second_half_goal_count += 1

    # Acceptation seulement si la timeline explique exactement le score final.
    if total_valid_goals != final_total:
        return None

    return second_half_goal_count


def enrich_league_history_with_second_half_goals(
    client,
    league_matches,
    max_timelines=LEAGUE_TIMELINE_LIMIT,
):
    """
    Enrichit les matchs sans score HT avec un nombre de buts 2H
    dérivé d'une timeline vérifiée.
    """

    if not isinstance(league_matches, list):
        return []

    enriched = [
        dict(match) if isinstance(match, dict) else match
        for match in league_matches
    ]

    timeline_calls = 0
    verified = 0

    for match in enriched:

        if not isinstance(match, dict):
            continue

        home_ht, away_ht = extract_half_time_score(
            match
        )

        if home_ht is not None and away_ht is not None:
            continue

        if timeline_calls >= max_timelines:
            continue

        timeline_calls += 1

        second_half_goals = (
            second_half_goals_from_verified_timeline(
                client,
                match,
            )
        )

        if second_half_goals is None:
            continue

        match["_verified_second_half_goals"] = (
            second_half_goals
        )
        verified += 1

    print(
        "Historique ligue timeline 2H : "
        f"{verified}/{len(enriched)} match(s) vérifiés "
        f"| timelines appelées={timeline_calls}"
    )

    return enriched

def enrich_league_history_with_half_time(
    client,
    league_matches,
    max_details=LEAGUE_HT_DETAIL_LIMIT,
):
    """
    Enrichit les matchs historiques dont le score mi-temps n'est pas présent
    dans la réponse de liste.

    Aucune valeur n'est inventée.
    """

    if not isinstance(league_matches, list):
        return []

    enriched = [
        dict(match) if isinstance(match, dict) else match
        for match in league_matches
    ]

    detail_calls = 0
    resolved = 0

    for match in enriched:

        if not isinstance(match, dict):
            continue

        home_ht, away_ht = extract_half_time_score(match)

        if home_ht is not None and away_ht is not None:
            resolved += 1
            continue

        if detail_calls >= max_details:
            continue

        match_id = extract_match_id(match)

        if not match_id:
            continue

        detail_calls += 1

        detail_response = get_match_details_cached(
            client,
            match_id,
        )

        detail = unwrap_match_payload(
            detail_response
        )

        if not isinstance(detail, dict) or not detail:
            continue

        detail_home_ht, detail_away_ht = extract_half_time_score(
            detail
        )

        if (
            detail_home_ht is not None
            and detail_away_ht is not None
        ):
            match["_detail_home_ht_score"] = detail_home_ht
            match["_detail_away_ht_score"] = detail_away_ht
            resolved += 1

    print(
        "Historique ligue HT : "
        f"{resolved}/{len(enriched)} match(s) avec score mi-temps "
        f"| détails appelés={detail_calls}"
    )

    if enriched and resolved == 0:
        sample = next(
            (
                item for item in enriched
                if isinstance(item, dict)
            ),
            None,
        )
        if sample is not None:
            print("DEBUG HISTORIQUE LIGUE SANS HT - exemple brut :")
            print(
                json.dumps(
                    sample,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )[:5000]
            )

    return enriched


# =============================================================================
# TEAM HISTORY
# =============================================================================

def get_team_history(
    client,
    team_id,
    limit=50,
):

    if not team_id:
        return []

    data = client.get(
        "/football/matches",
        params={
            "team_id": team_id,
            "status": "finished",
            "per_page": limit,
        },
    )

    return as_list(data)


# =============================================================================
# CALCULS HISTORIQUES ÉQUIPE
# =============================================================================

def team_history_rows(
    history,
    target_team_id=None,
    target_team_name=None,
):

    rows = []

    normalized_target_name = (
        normalize_team_name(
            target_team_name
        )
    )

    for match in history:

        if not isinstance(match, dict):
            continue

        home = extract_home_team(match)
        away = extract_away_team(match)

        home_id = extract_team_id(home)
        away_id = extract_team_id(away)

        home_name = extract_team_name(home)
        away_name = extract_team_name(away)

        is_home = False

        if target_team_id is not None:

            if str(home_id) == str(target_team_id):

                is_home = True

            elif str(away_id) == str(target_team_id):

                is_home = False

            else:
                continue

        elif normalized_target_name:

            if (
                normalize_team_name(home_name)
                == normalized_target_name
            ):

                is_home = True

            elif (
                normalize_team_name(away_name)
                == normalized_target_name
            ):

                is_home = False

            else:
                continue

        else:
            continue

        home_score, away_score = (
            extract_score(match)
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        if is_home:

            goals_for = home_score
            goals_against = away_score

        else:

            goals_for = away_score
            goals_against = home_score

        if goals_for > goals_against:

            result = "W"

        elif goals_for == goals_against:

            result = "D"

        else:

            result = "L"

        home_xg, away_xg = (
            extract_xg(match)
        )

        xg_for = (
            home_xg
            if is_home
            else away_xg
        )

        date = extract_match_date(match)

        rows.append({
            "match": match,
            "date": date,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": result,
            "btts": (
                home_score > 0
                and away_score > 0
            ),
            "over25": (
                home_score + away_score > 2.5
            ),
            "over05": (
                home_score + away_score > 0.5
            ),
            "xg_for": xg_for,
        })

    rows.sort(
        key=lambda x:
        parse_date(
            x["date"]
        ) or datetime.min
    )

    return rows


def calculate_form(
    rows,
    n,
):
    """
    FIDELITY LAB V1

    Reproduit l'échelle observée dans le dataset historique:
      - victoire = 3 points
      - nul = 1 point
      - défaite = 0 point
      - SOMME sur les n derniers matchs disponibles

    Plages attendues:
      form5  : 0 -> 15
      form10 : 0 -> 30
    """

    if not rows:
        return None

    recent = rows[-n:]

    if not recent:
        return None

    points = 0.0

    for row in recent:

        if row["result"] == "W":
            points += 3.0

        elif row["result"] == "D":
            points += 1.0

    return points


def calculate_frequency(
    rows,
    key,
):

    if not rows:
        return None

    values = [
        bool(row.get(key))
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return (
        sum(values) / len(values)
    )


def calculate_average(
    rows,
    key,
):

    values = [
        safe_float(
            row.get(key)
        )
        for row in rows
        if safe_float(
            row.get(key)
        ) is not None
    ]

    if not values:
        return None

    return (
        sum(values) / len(values)
    )


def build_team_pre_match_summary(
    history,
    team_id=None,
    team_name=None,
):

    rows = team_history_rows(
        history,
        target_team_id=team_id,
        target_team_name=team_name,
    )

    if not rows:

        return {
            "matches_played": None,
            "wins": None,
            "draws": None,
            "losses": None,
            "goals_for": None,
            "goals_against": None,
            "goal_difference": None,
            "form5": None,
            "form10": None,
            "scored_freq": None,
            "wins_freq": None,
            "avg_goals_scored": None,
            "avg_goals_conceded": None,
            "avg_xg": None,
            "btts_freq": None,
            "over25_freq": None,
        }

    wins = sum(
        row["result"] == "W"
        for row in rows
    )

    draws = sum(
        row["result"] == "D"
        for row in rows
    )

    losses = sum(
        row["result"] == "L"
        for row in rows
    )

    goals_for = sum(
        row["goals_for"]
        for row in rows
    )

    goals_against = sum(
        row["goals_against"]
        for row in rows
    )

    return {

        "matches_played": len(rows),

        "wins": wins,
        "draws": draws,
        "losses": losses,

        "goals_for": goals_for,
        "goals_against": goals_against,

        "goal_difference": (
            goals_for - goals_against
        ),

        "form5": calculate_form(
            rows,
            5,
        ),

        "form10": calculate_form(
            rows,
            10,
        ),

        "scored_freq": calculate_frequency(
            rows,
            "goals_for",
        ),

        "wins_freq": (
            wins / len(rows)
            if rows
            else None
        ),

        "avg_goals_scored":
            calculate_average(
                rows,
                "goals_for",
            ),

        "avg_goals_conceded":
            calculate_average(
                rows,
                "goals_against",
            ),

        "avg_xg":
            calculate_average(
                rows,
                "xg_for",
            ),

        "btts_freq":
            calculate_frequency(
                rows,
                "btts",
            ),

        "over25_freq":
            calculate_frequency(
                rows,
                "over25",
            ),
    }


# =============================================================================
# TEAM STATS ENDPOINT
# =============================================================================

def get_team_stats(
    client,
    team_id,
    competition_id=None,
    season_id=None,
):

    if not team_id:
        return None

    params = {
        "team_id": team_id,
    }

    if competition_id:
        params["competition_id"] = (
            competition_id
        )

    if season_id:
        params["season_id"] = season_id

    data = client.get(
        f"/football/teams/{team_id}/stats",
        params=params,
    )

    if isinstance(data, dict):
        return data

    return None


# =============================================================================
# STANDINGS
# =============================================================================

def get_standings(
    client,
    competition_id,
    season_id,
):

    if (
        not competition_id
        or not season_id
    ):
        return []

    data = client.get(
        f"/football/competitions/"
        f"{competition_id}/seasons/"
        f"{season_id}/standings"
    )

    return as_list(data)


def extract_standing_team_id(row):

    if not isinstance(row, dict):
        return None

    team = row.get("team")

    if isinstance(team, dict):

        return first_not_none(
            team.get("team_id"),
            team.get("id"),
        )

    return first_not_none(
        row.get("team_id"),
        row.get("id"),
    )


def standing_to_features(
    standings,
    team_id=None,
    team_name=None,
):

    normalized_name = (
        normalize_team_name(
            team_name
        )
    )

    for row in standings:

        if not isinstance(row, dict):
            continue

        row_team_id = (
            extract_standing_team_id(
                row
            )
        )

        row_team = row.get("team")

        row_team_name = None

        if isinstance(
            row_team,
            dict
        ):

            row_team_name = first_not_none(
                row_team.get("name"),
                row_team.get("team_name"),
            )

        matches = first_not_none(
            row.get("matches_played"),
            row.get("played"),
            row.get("games_played"),
            row.get("matches"),
        )

        wins = first_not_none(
            row.get("wins"),
            row.get("won"),
        )

        draws = first_not_none(
            row.get("draws"),
            row.get("draw"),
        )

        losses = first_not_none(
            row.get("losses"),
            row.get("lost"),
        )

        points = first_not_none(
            row.get("points"),
            row.get("pts"),
        )

        goals_for = first_not_none(
            row.get("goals_for"),
            row.get("gf"),
            row.get("goals_scored"),
        )

        goals_against = first_not_none(
            row.get("goals_against"),
            row.get("ga"),
            row.get("goals_conceded"),
        )

        position = first_not_none(
            row.get("position"),
            row.get("rank"),
            row.get("place"),
        )

        matched = False

        if (
            team_id is not None
            and row_team_id is not None
        ):

            matched = (
                str(row_team_id)
                == str(team_id)
            )

        elif (
            normalized_name
            and row_team_name
        ):

            matched = (
                normalize_team_name(
                    row_team_name
                )
                == normalized_name
            )

        if matched:

            gf = safe_float(
                goals_for
            )

            ga = safe_float(
                goals_against
            )

            goal_difference = None

            if (
                gf is not None
                and ga is not None
            ):

                goal_difference = (
                    gf - ga
                )

            return {

                "rank":
                    safe_float(position),

                "points":
                    safe_float(points),

                "matches_played":
                    safe_float(matches),

                "wins":
                    safe_float(wins),

                "draws":
                    safe_float(draws),

                "losses":
                    safe_float(losses),

                "goals_for":
                    gf,

                "goals_against":
                    ga,

                "goal_difference":
                    goal_difference,
            }

    return {}


# =============================================================================
# H2H
# =============================================================================

def get_h2h_history(
    client,
    home_team_id,
    away_team_id,
    limit=5,
):

    if (
        not home_team_id
        or not away_team_id
    ):
        return []

    data = client.get(
        "/football/matches",
        params={
            "team_id": home_team_id,
            "limit": 100,
        },
    )

    matches = as_list(data)

    result = []

    for match in matches:

        home = extract_home_team(match)
        away = extract_away_team(match)

        home_id = extract_team_id(home)
        away_id = extract_team_id(away)

        ids = {
            str(home_id),
            str(away_id),
        }

        target_ids = {
            str(home_team_id),
            str(away_team_id),
        }

        if ids == target_ids:
            result.append(match)

    result.sort(
        key=lambda m:
        parse_date(
            extract_match_date(m)
        ) or datetime.min
    )

    return result[-limit:]


def calculate_h2h_features(h2h):

    rows = []

    for match in h2h:

        home_score, away_score = (
            extract_score(match)
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        total = (
            home_score
            + away_score
        )

        rows.append({
            "over25": total > 2.5,
            "btts": (
                home_score > 0
                and away_score > 0
            ),
        })

    if not rows:

        return {
            "h2h5_over25_freq": None,
            "h2h5_btts_freq": None,
        }

    return {

        "h2h5_over25_freq":
            sum(
                x["over25"]
                for x in rows
            ) / len(rows),

        "h2h5_btts_freq":
            sum(
                x["btts"]
                for x in rows
            ) / len(rows),
    }


# =============================================================================
# HISTORIQUE DE LIGUE
# =============================================================================

def get_league_history(
    client,
    competition_id,
    season_id,
    limit=100,
):

    if not competition_id:
        return []

    params = {
        "competition_id":
            competition_id,
        "status": "finished",
        "per_page": limit,
    }

    if season_id:
        params["season_id"] = season_id

    data = client.get(
        "/football/matches",
        params=params,
    )

    return as_list(data)


def calculate_league_features(
    league_matches,
    client=None,
):

    if client is not None:
        league_matches = enrich_league_history_with_half_time(
            client,
            league_matches,
        )

        # FIDELITY LAB V2:
        # si le score mi-temps n'existe pas, tenter une reconstruction
        # conservatrice via une timeline dont les buts expliquent exactement
        # le score final.
        league_matches = enrich_league_history_with_second_half_goals(
            client,
            league_matches,
        )

    home_over25 = []
    away_over25 = []

    home_over05 = []
    away_over05 = []

    home_btts = []
    away_btts = []

    second_half_over15 = []

    for match in league_matches:

        home_score, away_score = (
            extract_score(match)
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        total = (
            home_score
            + away_score
        )

        home_over25.append(
            total > 2.5
        )

        away_over25.append(
            total > 2.5
        )

        # FIDELITY LAB V1
        # Interprétation compatible avec les distributions historiques:
        # home = fréquence où l'équipe à domicile marque au moins 1 but
        # away = fréquence où l'équipe à l'extérieur marque au moins 1 but
        home_over05.append(
            home_score > 0
        )

        away_over05.append(
            away_score > 0
        )

        btts = (
            home_score > 0
            and away_score > 0
        )

        home_btts.append(
            btts
        )

        away_btts.append(
            btts
        )

        home_ht, away_ht = (
            extract_half_time_score(
                match
            )
        )

        second_half_goals = None

        if (
            home_ht is not None
            and away_ht is not None
        ):

            calculated_2h = (
                total
                - home_ht
                - away_ht
            )

            if calculated_2h >= 0:
                second_half_goals = calculated_2h

        elif match.get(
            "_verified_second_half_goals"
        ) is not None:

            second_half_goals = safe_float(
                match.get(
                    "_verified_second_half_goals"
                )
            )

        if second_half_goals is not None:

            second_half_over15.append(
                second_half_goals > 1.5
            )

    print(
        "League 2H Over 1.5 : "
        f"{len(second_half_over15)} match(s) exploitables "
        f"sur {len(league_matches)}"
    )

    def avg(values):

        if not values:
            return None

        return (
            sum(values)
            / len(values)
        )

    return {

        "league_home_over25_freq":
            avg(home_over25),

        "league_away_over25_freq":
            avg(away_over25),

        "league_home_over05_freq":
            avg(home_over05),

        "league_away_over05_freq":
            avg(away_over05),

        "league_home_btts_freq":
            avg(home_btts),

        "league_away_btts_freq":
            avg(away_btts),

        "league_2nd_half_over15_freq":
            avg(second_half_over15),
    }


# =============================================================================
# ODDS
# =============================================================================

def extract_price(value):
    """
    Extrait uniquement la cote OPENING.
    """

    if value is None:
        return None

    if isinstance(value, dict):

        if "opening" not in value:
            return None

        return safe_float(
            value.get("opening")
        )

    return None


def select_bookmaker(
    bookmakers,
    preferred=None,
):

    if not isinstance(
        bookmakers,
        list
    ):
        return None

    if preferred:

        preferred_normalized = (
            normalize_team_name(
                preferred
            )
        )

        for bookmaker in bookmakers:

            if not isinstance(
                bookmaker,
                dict
            ):
                continue

            name = bookmaker.get(
                "bookmaker"
            )

            if (
                normalize_team_name(
                    name
                )
                == preferred_normalized
            ):
                return bookmaker

    priority = [
        "Pinnacle",
        "Bet365",
    ]

    for preferred_name in priority:

        normalized = (
            normalize_team_name(
                preferred_name
            )
        )

        for bookmaker in bookmakers:

            if not isinstance(
                bookmaker,
                dict
            ):
                continue

            name = bookmaker.get(
                "bookmaker"
            )

            if (
                normalize_team_name(
                    name
                )
                == normalized
            ):
                return bookmaker

    for bookmaker in bookmakers:

        if not isinstance(
            bookmaker,
            dict
        ):
            continue

        markets = bookmaker.get(
            "markets"
        )

        if isinstance(
            markets,
            dict
        ):
            return bookmaker

    return None


def extract_opening_market_outcome(
    markets,
    market_name,
    outcome_name,
):

    if not isinstance(
        markets,
        dict
    ):
        return None

    market = markets.get(
        market_name
    )

    if not isinstance(
        market,
        dict
    ):
        return None

    outcome = market.get(
        outcome_name
    )

    return extract_price(
        outcome
    )


def extract_match_odds(
    bookmaker,
):

    if not isinstance(
        bookmaker,
        dict
    ):

        return {
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
        }

    markets = bookmaker.get(
        "markets",
        {}
    )

    return {

        "odds_home":
            extract_opening_market_outcome(
                markets,
                "match_odds",
                "home",
            ),

        "odds_draw":
            extract_opening_market_outcome(
                markets,
                "match_odds",
                "draw",
            ),

        "odds_away":
            extract_opening_market_outcome(
                markets,
                "match_odds",
                "away",
            ),
    }


def extract_total_goals_odds(
    bookmaker,
):

    result = {

        "odds_over05": None,
        "odds_over15": None,
        "odds_over25": None,
        "odds_over35": None,
        "odds_under25": None,
    }

    if not isinstance(
        bookmaker,
        dict
    ):
        return result

    markets = bookmaker.get(
        "markets",
        {}
    )

    if not isinstance(
        markets,
        dict
    ):
        return result

    total_goals = markets.get(
        "total_goals"
    )

    if not isinstance(
        total_goals,
        dict
    ):
        return result

    mapping = {

        "0.5": (
            "over",
            "odds_over05",
        ),

        "1.5": (
            "over",
            "odds_over15",
        ),

        "2.5": (
            "over",
            "odds_over25",
        ),

        "3.5": (
            "over",
            "odds_over35",
        ),
    }

    for line, (
        outcome,
        feature_name,
    ) in mapping.items():

        line_data = total_goals.get(
            line
        )

        if not isinstance(
            line_data,
            dict
        ):
            continue

        result[feature_name] = (
            extract_price(
                line_data.get(
                    outcome
                )
            )
        )

    line_25 = total_goals.get(
        "2.5"
    )

    if isinstance(
        line_25,
        dict
    ):

        result["odds_under25"] = (
            extract_price(
                line_25.get(
                    "under"
                )
            )
        )

    return result


def extract_btts_odds(
    bookmaker,
):

    result = {
        "odds_btts_yes": None,
        "odds_btts_no": None,
    }

    if not isinstance(
        bookmaker,
        dict
    ):
        return result

    markets = bookmaker.get(
        "markets",
        {}
    )

    if not isinstance(
        markets,
        dict
    ):
        return result

    btts = markets.get(
        "btts"
    )

    if not isinstance(
        btts,
        dict
    ):
        return result

    result["odds_btts_yes"] = (
        extract_price(
            btts.get("yes")
        )
    )

    result["odds_btts_no"] = (
        extract_price(
            btts.get("no")
        )
    )

    return result


def fetch_prematch_odds(
    client,
    match_id,
    preferred_bookmaker=None,
):

    empty = {

        "odds_home": None,
        "odds_draw": None,
        "odds_away": None,

        "odds_over05": None,
        "odds_over15": None,
        "odds_over25": None,
        "odds_over35": None,
        "odds_under25": None,

        "odds_btts_yes": None,
        "odds_btts_no": None,
    }

    if not match_id:
        return empty

    data = client.get(
        f"/football/matches/"
        f"{match_id}/odds"
    )

    if not isinstance(
        data,
        dict
    ):
        return empty

    payload = data.get(
        "data"
    )

    if not isinstance(
        payload,
        dict
    ):
        return empty

    bookmakers = payload.get(
        "bookmakers"
    )

    if not isinstance(
        bookmakers,
        list
    ):
        return empty

    bookmaker = select_bookmaker(
        bookmakers,
        preferred=preferred_bookmaker,
    )

    if not bookmaker:
        return empty

    result = {}

    result.update(
        extract_match_odds(
            bookmaker
        )
    )

    result.update(
        extract_total_goals_odds(
            bookmaker
        )
    )

    result.update(
        extract_btts_odds(
            bookmaker
        )
    )

    return result


# =============================================================================
# MERGE TEAM STATS
# =============================================================================

def merge_team_stats_with_history(
    summary,
    stats,
):

    if not isinstance(
        summary,
        dict
    ):
        summary = {}

    if not isinstance(
        stats,
        dict
    ):
        return summary

    candidates = [
        stats,
        stats.get("data"),
        stats.get("stats"),
    ]

    source = None

    for candidate in candidates:

        if isinstance(
            candidate,
            dict
        ):

            source = candidate
            break

    if not source:
        return summary

    mapping = {

        "matches_played": [
            "matches_played",
            "played",
        ],

        "wins": [
            "wins",
            "won",
        ],

        "draws": [
            "draws",
        ],

        "losses": [
            "losses",
            "lost",
        ],

        "points": [
            "points",
            "pts",
        ],

        "goals_for": [
            "goals_for",
            "gf",
        ],

        "goals_against": [
            "goals_against",
            "ga",
        ],
    }

    for destination, keys in mapping.items():

        if (
            summary.get(
                destination
            )
            is not None
        ):
            continue

        for key in keys:

            value = source.get(
                key
            )

            if value is not None:

                summary[destination] = (
                    safe_float(value)
                )

                break

    if (
        summary.get(
            "goal_difference"
        )
        is None
        and summary.get(
            "goals_for"
        )
        is not None
        and summary.get(
            "goals_against"
        )
        is not None
    ):

        summary[
            "goal_difference"
        ] = (
            summary["goals_for"]
            - summary["goals_against"]
        )

    return summary


# =============================================================================
# BUILD PRE-MATCH FEATURES
# =============================================================================

def build_pre_match_features(
    client,
    match,
    preferred_bookmaker=None,
):

    features = {
        feature: None
        for feature in PRE_MATCH_FEATURES
    }

    match_id = extract_match_id(
        match
    )

    home_team = extract_home_team(
        match
    )

    away_team = extract_away_team(
        match
    )

    home_team_id = extract_team_id(
        home_team
    )

    away_team_id = extract_team_id(
        away_team
    )

    home_team_name = extract_team_name(
        home_team
    )

    away_team_name = extract_team_name(
        away_team
    )

    competition_id = (
        extract_competition_id(
            match
        )
    )

    season_id = extract_season_id(
        match
    )

    print(
        f"Match : "
        f"{home_team_name} vs "
        f"{away_team_name}"
    )

    print(
        f"Match ID : {match_id}"
    )

    print(
        f"Competition : "
        f"{competition_id}"
    )

    print(
        f"Season : {season_id}"
    )

    # =========================================================================
    # HISTORIQUE HOME
    # =========================================================================

    print(
        "Récupération historique équipe domicile..."
    )

    home_history = get_team_history(
        client,
        home_team_id,
        limit=50,
    )

    home_history = enrich_history_with_xg(
        client,
        home_history,
        max_matches=XG_HISTORY_MATCH_LIMIT,
    )

    home_summary = (
        build_team_pre_match_summary(
            home_history,
            team_id=home_team_id,
            team_name=home_team_name,
        )
    )

    # =========================================================================
    # HISTORIQUE AWAY
    # =========================================================================

    print(
        "Récupération historique équipe extérieur..."
    )

    away_history = get_team_history(
        client,
        away_team_id,
        limit=50,
    )

    away_history = enrich_history_with_xg(
        client,
        away_history,
        max_matches=XG_HISTORY_MATCH_LIMIT,
    )

    away_summary = (
        build_team_pre_match_summary(
            away_history,
            team_id=away_team_id,
            team_name=away_team_name,
        )
    )

    # =========================================================================
    # TEAM STATS
    # =========================================================================

    home_stats = get_team_stats(
        client,
        home_team_id,
        competition_id,
        season_id,
    )

    away_stats = get_team_stats(
        client,
        away_team_id,
        competition_id,
        season_id,
    )

    home_summary = (
        merge_team_stats_with_history(
            home_summary,
            home_stats,
        )
    )

    away_summary = (
        merge_team_stats_with_history(
            away_summary,
            away_stats,
        )
    )

    # =========================================================================
    # STANDINGS
    # =========================================================================

    print(
        "Récupération standings..."
    )

    standings = get_standings(
        client,
        competition_id,
        season_id,
    )

    home_standing = (
        standing_to_features(
            standings,
            team_id=home_team_id,
            team_name=home_team_name,
        )
    )

    away_standing = (
        standing_to_features(
            standings,
            team_id=away_team_id,
            team_name=away_team_name,
        )
    )

    # =========================================================================
    # TEAM FEATURES
    # =========================================================================

    features[
        "home_team_matches_played"
    ] = first_not_none(
        home_standing.get(
            "matches_played"
        ),
        home_summary.get(
            "matches_played"
        ),
    )

    features[
        "away_team_matches_played"
    ] = first_not_none(
        away_standing.get(
            "matches_played"
        ),
        away_summary.get(
            "matches_played"
        ),
    )

    features[
        "home_team_wins"
    ] = first_not_none(
        home_standing.get(
            "wins"
        ),
        home_summary.get(
            "wins"
        ),
    )

    features[
        "away_team_wins"
    ] = first_not_none(
        away_standing.get(
            "wins"
        ),
        away_summary.get(
            "wins"
        ),
    )

    features[
        "home_team_draws"
    ] = first_not_none(
        home_standing.get(
            "draws"
        ),
        home_summary.get(
            "draws"
        ),
    )

    features[
        "away_team_draws"
    ] = first_not_none(
        away_standing.get(
            "draws"
        ),
        away_summary.get(
            "draws"
        ),
    )

    features[
        "home_team_losses"
    ] = first_not_none(
        home_standing.get(
            "losses"
        ),
        home_summary.get(
            "losses"
        ),
    )

    features[
        "away_team_losses"
    ] = first_not_none(
        away_standing.get(
            "losses"
        ),
        away_summary.get(
            "losses"
        ),
    )

    features[
        "home_team_goals_for"
    ] = first_not_none(
        home_standing.get(
            "goals_for"
        ),
        home_summary.get(
            "goals_for"
        ),
    )

    features[
        "away_team_goals_for"
    ] = first_not_none(
        away_standing.get(
            "goals_for"
        ),
        away_summary.get(
            "goals_for"
        ),
    )

    features[
        "home_team_goals_against"
    ] = first_not_none(
        home_standing.get(
            "goals_against"
        ),
        home_summary.get(
            "goals_against"
        ),
    )

    features[
        "away_team_goals_against"
    ] = first_not_none(
        away_standing.get(
            "goals_against"
        ),
        away_summary.get(
            "goals_against"
        ),
    )

    features[
        "home_team_goal_difference"
    ] = first_not_none(
        home_standing.get(
            "goal_difference"
        ),
        home_summary.get(
            "goal_difference"
        ),
    )

    features[
        "away_team_goal_difference"
    ] = first_not_none(
        away_standing.get(
            "goal_difference"
        ),
        away_summary.get(
            "goal_difference"
        ),
    )

    features[
        "home_team_points_standing"
    ] = home_standing.get(
        "points"
    )

    features[
        "away_team_points_standing"
    ] = away_standing.get(
        "points"
    )

    features[
        "home_team_rank"
    ] = home_standing.get(
        "rank"
    )

    features[
        "away_team_rank"
    ] = away_standing.get(
        "rank"
    )

    # =========================================================================
    # POSITION / POINTS
    # =========================================================================

    features[
        "home_position"
    ] = home_standing.get(
        "rank"
    )

    features[
        "away_position"
    ] = away_standing.get(
        "rank"
    )

    features[
        "home_points"
    ] = home_standing.get(
        "points"
    )

    features[
        "away_points"
    ] = away_standing.get(
        "points"
    )

    # =========================================================================
    # FORM
    # =========================================================================

    features[
        "home_form5"
    ] = home_summary.get(
        "form5"
    )

    features[
        "away_form5"
    ] = away_summary.get(
        "form5"
    )

    features[
        "home_form10"
    ] = home_summary.get(
        "form10"
    )

    features[
        "away_form10"
    ] = away_summary.get(
        "form10"
    )

    # =========================================================================
    # FREQUENCIES
    # =========================================================================

    features[
        "home_scored_freq"
    ] = home_summary.get(
        "scored_freq"
    )

    features[
        "away_scored_freq"
    ] = away_summary.get(
        "scored_freq"
    )

    features[
        "home_wins_freq"
    ] = home_summary.get(
        "wins_freq"
    )

    features[
        "away_wins_freq"
    ] = away_summary.get(
        "wins_freq"
    )

    features[
        "home_avg_goals_scored"
    ] = home_summary.get(
        "avg_goals_scored"
    )

    features[
        "away_avg_goals_scored"
    ] = away_summary.get(
        "avg_goals_scored"
    )

    features[
        "home_avg_goals_conceded"
    ] = home_summary.get(
        "avg_goals_conceded"
    )

    features[
        "away_avg_goals_conceded"
    ] = away_summary.get(
        "avg_goals_conceded"
    )

    features[
        "home_avg_xg"
    ] = home_summary.get(
        "avg_xg"
    )

    features[
        "away_avg_xg"
    ] = away_summary.get(
        "avg_xg"
    )

    features[
        "home_btts_freq"
    ] = home_summary.get(
        "btts_freq"
    )

    features[
        "away_btts_freq"
    ] = away_summary.get(
        "btts_freq"
    )

    features[
        "home_over25_freq"
    ] = home_summary.get(
        "over25_freq"
    )

    features[
        "away_over25_freq"
    ] = away_summary.get(
        "over25_freq"
    )

    # =========================================================================
    # H2H
    # =========================================================================

    print(
        "Récupération H2H..."
    )

    h2h = get_h2h_history(
        client,
        home_team_id,
        away_team_id,
        limit=5,
    )

    h2h_features = (
        calculate_h2h_features(
            h2h
        )
    )

    features.update(
        h2h_features
    )

    # =========================================================================
    # LEAGUE
    # =========================================================================

    print(
        "Récupération historique de ligue..."
    )

    league_history = (
        get_league_history(
            client,
            competition_id,
            season_id,
            limit=100,
        )
    )

    league_features = (
        calculate_league_features(
            league_history,
            client=client,
        )
    )

    features.update(
        league_features
    )

    # =========================================================================
    # ODDS
    # =========================================================================

    print(
        "Récupération cotes pré-match..."
    )

    odds = fetch_prematch_odds(
        client,
        match_id,
        preferred_bookmaker=preferred_bookmaker,
    )

    features.update(
        odds
    )

    return features


# =============================================================================
# VALIDATION
# =============================================================================

def validate_features(features):

    if not isinstance(
        features,
        dict
    ):

        return {
            "total":
                len(PRE_MATCH_FEATURES),

            "available":
                0,

            "missing":
                PRE_MATCH_FEATURES.copy(),

            "coverage":
                0.0,
        }

    available = []
    missing = []

    for feature in PRE_MATCH_FEATURES:

        value = features.get(
            feature
        )

        if value is None:

            missing.append(
                feature
            )

        else:

            available.append(
                feature
            )

    total = len(
        PRE_MATCH_FEATURES
    )

    coverage = (
        len(available) / total
        if total
        else 0.0
    )

    return {

        "total":
            total,

        "available":
            len(available),

        "missing":
            missing,

        "coverage":
            coverage,
    }


def print_feature_report(
    features
):

    validation = validate_features(
        features
    )

    print()
    print("=" * 80)
    print(
        "PRE-MATCH FEATURE REPORT"
    )
    print("=" * 80)

    for index, feature in enumerate(
        PRE_MATCH_FEATURES,
        start=1,
    ):

        value = features.get(
            feature
        )

        print(
            f"{index:02d}. "
            f"{feature:<42} = {value}"
        )

    print()
    print("-" * 80)

    print(
        f"Features PRE-MATCH disponibles : "
        f"{validation['available']}/"
        f"{validation['total']}"
    )

    print(
        f"Coverage : "
        f"{validation['coverage'] * 100:.1f}%"
    )

    print(
        f"Features PRE-MATCH manquantes : "
        f"{len(validation['missing'])}"
    )

    if validation["missing"]:

        print(
            "Manquantes : "
            + ", ".join(
                validation["missing"]
            )
        )

    print("=" * 80)


# =============================================================================
# MODEL READY
# =============================================================================

def get_model_ready_pre_match_features(
    features
):

    return [
        features.get(
            feature
        )
        for feature in PRE_MATCH_FEATURES
    ]


# =============================================================================
# LIVE MATCH SEARCH
# =============================================================================

def get_live_matches(
    client
):

    data = client.get(
        "/football/matches",
        params={
            "status": "live",
        },
    )

    return as_list(data)


# =============================================================================
# LIVE MINUTE — VERSION ROBUSTE
# =============================================================================

def extract_live_minute(match):
    """
    Extrait la minute LIVE de manière robuste.

    Structures supportées :

    {
        "live": {
            "elapsed_minutes": 85
        }
    }

    {
        "live": {
            "minute": 85
        }
    }

    {
        "data": {
            "live": {
                "elapsed_minutes": 85
            }
        }
    }

    {
        "elapsed_minutes": 85
    }

    {
        "minute": "85'"
    }

    {
        "minute": "85+2"
    }

    On privilégie toujours elapsed_minutes.
    """

    if not isinstance(
        match,
        dict
    ):
        return None

    # -------------------------------------------------------------------------
    # Structures LIVE directes
    # -------------------------------------------------------------------------

    live_candidates = []

    live = match.get(
        "live"
    )

    if isinstance(
        live,
        dict
    ):
        live_candidates.append(
            live
        )

    # -------------------------------------------------------------------------
    # Structure data.live
    # -------------------------------------------------------------------------

    data = match.get(
        "data"
    )

    if isinstance(
        data,
        dict
    ):

        data_live = data.get(
            "live"
        )

        if isinstance(
            data_live,
            dict
        ):

            live_candidates.append(
                data_live
            )

        # Parfois data contient directement elapsed_minutes
        live_candidates.append(
            data
        )

    # -------------------------------------------------------------------------
    # Recherche des champs minute
    # -------------------------------------------------------------------------

    minute_keys = [
        "elapsed_minutes",
        "elapsed_minute",
        "minute",
        "minutes",
        "match_minute",
        "current_minute",
        "current_minute_number",
        "elapsed",
    ]

    for source in live_candidates:

        for key in minute_keys:

            if key not in source:
                continue

            raw_value = source.get(
                key
            )

            minute = safe_int(
                raw_value
            )

            if minute is not None:
                return minute

    # -------------------------------------------------------------------------
    # Fallback au niveau match
    # -------------------------------------------------------------------------

    for key in minute_keys:

        if key not in match:
            continue

        raw_value = match.get(
            key
        )

        minute = safe_int(
            raw_value
        )

        if minute is not None:
            return minute

    # -------------------------------------------------------------------------
    # Fallback sur certains champs courants
    # -------------------------------------------------------------------------

    for key in [
        "time",
        "elapsed",
        "clock",
    ]:

        raw_value = match.get(
            key
        )

        minute = safe_int(
            raw_value
        )

        if minute is not None:
            return minute

    return None


# =============================================================================
# LIVE SCORE
# =============================================================================

def extract_live_score(match):

    if not isinstance(
        match,
        dict
    ):
        return None, None

    score = match.get(
        "score"
    )

    if isinstance(
        score,
        dict
    ):

        home = first_not_none(
            score.get("home"),
            score.get("home_score"),
        )

        away = first_not_none(
            score.get("away"),
            score.get("away_score"),
        )

        return (
            safe_int(home),
            safe_int(away),
        )

    live = match.get(
        "live"
    )

    if isinstance(
        live,
        dict
    ):

        home = first_not_none(
            live.get("home_goals"),
            live.get("home_score"),
        )

        away = first_not_none(
            live.get("away_goals"),
            live.get("away_score"),
        )

        if (
            home is not None
            or away is not None
        ):

            return (
                safe_int(home),
                safe_int(away),
            )

    return (
        safe_int(
            match.get("home_score")
        ),
        safe_int(
            match.get("away_score")
        ),
    )


# =============================================================================
# DEBUG LIVE
# =============================================================================

def print_live_matches_debug(
    live_matches
):
    """
    Affiche tous les matchs LIVE avec la minute
    détectée par extract_live_minute().

    Cela permet de vérifier immédiatement que
    les matchs à 85' sont bien reconnus.
    """

    print()
    print("-" * 80)
    print("DEBUG MATCHS LIVE")
    print("-" * 80)

    if not live_matches:

        print(
            "Aucun match LIVE reçu."
        )

        print("-" * 80)

        return

    for index, match in enumerate(
        live_matches,
        start=1,
    ):

        minute = extract_live_minute(
            match
        )

        home = extract_home_team(
            match
        )

        away = extract_away_team(
            match
        )

        home_name = (
            extract_team_name(home)
            or "Home"
        )

        away_name = (
            extract_team_name(away)
            or "Away"
        )

        home_score, away_score = (
            extract_live_score(
                match
            )
        )

        match_id = extract_match_id(
            match
        )

        print(
            f"{index:02d}. "
            f"{home_name} "
            f"{home_score if home_score is not None else '?'}-"
            f"{away_score if away_score is not None else '?'} "
            f"{away_name} | "
            f"minute détectée = "
            f"{minute if minute is not None else 'INCONNUE'}' | "
            f"ID = {match_id}"
        )

    print("-" * 80)


# =============================================================================
# TEST
# =============================================================================

def run_test():

    print()
    print("=" * 80)
    print(
        "TEST PRE-MATCH FEATURE ENGINEERING"
    )
    print("=" * 80)

    if not API_KEY:

        print()
        print(
            "ERREUR : STATS_API_KEY "
            "est absent du .env"
        )

        return

    print(
        "API KEY : OK"
    )

    client = StatsAPIClient()

    live_matches = get_live_matches(
        client
    )

    print(
        f"Matchs LIVE récupérés : "
        f"{len(live_matches)}"
    )

    # =========================================================================
    # DEBUG : AFFICHER LES 7 MATCHS
    # =========================================================================

    print_live_matches_debug(
        live_matches
    )

    # =========================================================================
    # SÉLECTION 85' → 87'
    # =========================================================================

    candidates = []

    for match in live_matches:

        minute = extract_live_minute(
            match
        )

        if minute is None:
            continue

        if (
            MIN_SCAN_MINUTE
            <= minute
            <= MAX_SCAN_MINUTE
        ):

            candidates.append(
                match
            )

    # =========================================================================
    # AUCUN CANDIDAT
    # =========================================================================

    if not candidates:

        print()
        print(
            f"Aucun match actuellement "
            f"entre "
            f"{MIN_SCAN_MINUTE}' "
            f"et "
            f"{MAX_SCAN_MINUTE}'."
        )

        print()
        print(
            "Vérifie la section "
            "'DEBUG MATCHS LIVE' ci-dessus."
        )

        return

    # =========================================================================
    # CANDIDATS TROUVÉS
    # =========================================================================

    print()
    print("=" * 80)
    print(
        f"MATCHS CANDIDATS "
        f"{MIN_SCAN_MINUTE}' → "
        f"{MAX_SCAN_MINUTE}'"
    )
    print("=" * 80)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        home = extract_home_team(
            candidate
        )

        away = extract_away_team(
            candidate
        )

        home_name = (
            extract_team_name(home)
            or "Home"
        )

        away_name = (
            extract_team_name(away)
            or "Away"
        )

        minute = extract_live_minute(
            candidate
        )

        home_score, away_score = (
            extract_live_score(
                candidate
            )
        )

        match_id = extract_match_id(
            candidate
        )

        print(
            f"{index:02d}. "
            f"{home_name} "
            f"{home_score if home_score is not None else '?'}-"
            f"{away_score if away_score is not None else '?'} "
            f"{away_name} | "
            f"{minute}' | "
            f"ID: {match_id}"
        )

    print("=" * 80)

    # =========================================================================
    # ANALYSE DU PREMIER MATCH
    # =========================================================================

    match = candidates[0]

    home = extract_home_team(
        match
    )

    away = extract_away_team(
        match
    )

    home_name = extract_team_name(
        home
    )

    away_name = extract_team_name(
        away
    )

    minute = extract_live_minute(
        match
    )

    home_score, away_score = (
        extract_live_score(
            match
        )
    )

    print()
    print(
        "Analyse du premier candidat..."
    )

    print(
        f"Match sélectionné : "
        f"{home_name} "
        f"{home_score}-{away_score} "
        f"{away_name} | "
        f"{minute}'"
    )

    features = build_pre_match_features(
        client,
        match,
    )

    print_feature_report(
        features
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    run_test()