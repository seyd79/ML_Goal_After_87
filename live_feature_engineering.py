#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE, override=False)

API_KEY = os.getenv("STATS_API_KEY", "").strip()

API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api"
).rstrip("/")

API_TIMEOUT = int(os.getenv("STATS_API_TIMEOUT", "20"))

MIN_SCAN_MINUTE = int(os.getenv("MIN_SCAN_MINUTE", "85"))
MAX_SCAN_MINUTE = int(os.getenv("MAX_SCAN_MINUTE", "87"))


print(f".env chargé : {ENV_FILE}")


class StatsAPIClient:
    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "GoalAfter87-LiveScanner/1.0"
        })

    def get(self, endpoint, params=None):
        url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

        response = self.session.get(
            url,
            params=params,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        return response.json()


def as_list(data):
    """
    Transforme les différentes structures possibles
    de réponse API en liste de matchs.
    """

    if isinstance(data, dict):

        if isinstance(data.get("data"), list):
            return data["data"]

        if isinstance(data.get("matches"), list):
            return data["matches"]

    return []


def extract_live_minute(match):
    """
    Extrait la minute actuelle du match.
    """

    live = match.get("live", {})

    minute = (
        live.get("elapsed_minutes")
        or live.get("minute")
        or match.get("minute")
    )

    if minute is None:
        return None

    try:
        return int(minute)
    except (ValueError, TypeError):
        return None


def extract_score(match):
    """
    Extrait le score actuel du match.
    """

    score = match.get("score", {})

    home_score = score.get("home", 0)
    away_score = score.get("away", 0)

    try:
        home_score = int(home_score)
    except (ValueError, TypeError):
        home_score = 0

    try:
        away_score = int(away_score)
    except (ValueError, TypeError):
        away_score = 0

    return home_score, away_score


def calculate_leading_team_87(home_score, away_score):
    """
    Convention utilisée par le modèle ML :

        0 = Home mène
        1 = Away mène
        2 = Match nul
    """

    if home_score > away_score:
        leading_team_87 = 0

    elif away_score > home_score:
        leading_team_87 = 1

    else:
        leading_team_87 = 2

    return leading_team_87


def calculate_live_state(match):
    """
    Construit les variables principales liées
    à l'état actuel du match.
    """

    home_score, away_score = extract_score(match)

    leading_team_87 = calculate_leading_team_87(
        home_score,
        away_score
    )

    goal_diff_87 = abs(home_score - away_score)

    is_draw_87 = int(home_score == away_score)

    return {
        "score_87_home": home_score,
        "score_87_away": away_score,
        "leading_team_87": leading_team_87,
        "goal_diff_87": goal_diff_87,
        "is_draw_87": is_draw_87,
    }


def get_live_matches(client):
    """
    Récupère les matchs actuellement LIVE.
    """

    data = client.get(
        "/football/matches",
        params={
            "status": "live",
            "per_page": 100
        }
    )

    return as_list(data)


def get_candidate_matches(matches):
    """
    Sélectionne les matchs compris entre
    MIN_SCAN_MINUTE et MAX_SCAN_MINUTE.
    """

    candidates = []

    for match in matches:

        minute = extract_live_minute(match)

        if minute is None:
            continue

        if MIN_SCAN_MINUTE <= minute <= MAX_SCAN_MINUTE:
            candidates.append(match)

    return candidates


def print_match_debug(match, index=None):
    """
    Affiche les informations principales d'un match.
    """

    home_team = match.get("home_team", {}).get(
        "name",
        "Home"
    )

    away_team = match.get("away_team", {}).get(
        "name",
        "Away"
    )

    home_score, away_score = extract_score(match)

    minute = extract_live_minute(match)

    live_state = calculate_live_state(match)

    prefix = ""

    if index is not None:
        prefix = f"{index:02d}. "

    print(
        f"{prefix}"
        f"{home_team} "
        f"{home_score}-{away_score} "
        f"{away_team} "
        f"| {minute}' "
        f"| leading_team_87="
        f"{live_state['leading_team_87']} "
        f"| goal_diff_87="
        f"{live_state['goal_diff_87']} "
        f"| is_draw_87="
        f"{live_state['is_draw_87']} "
        f"| ID="
        f"{match.get('id', 'N/A')}"
    )


def run_test():

    print("\n" + "=" * 80)
    print("TEST LIVE FEATURE ENGINEERING")
    print("=" * 80)

    if not API_KEY:
        print("ERREUR : STATS_API_KEY absent")
        return

    print("API KEY : OK")

    client = StatsAPIClient()

    try:
        matches = get_live_matches(client)

    except requests.RequestException as error:
        print(f"ERREUR API : {error}")
        return

    print(
        f"Matchs LIVE récupérés : {len(matches)}"
    )

    print("\n" + "-" * 80)
    print("DEBUG MATCHS LIVE")
    print("-" * 80)

    candidates = []

    for index, match in enumerate(matches, 1):

        print_match_debug(
            match,
            index
        )

        minute = extract_live_minute(match)

        if (
            minute is not None
            and MIN_SCAN_MINUTE <= minute <= MAX_SCAN_MINUTE
        ):
            candidates.append(match)

    print("-" * 80)

    if not candidates:

        print(
            f"\nAucun match entre "
            f"{MIN_SCAN_MINUTE}' et "
            f"{MAX_SCAN_MINUTE}'."
        )

        return

    print("\n" + "=" * 80)
    print(
        f"MATCHS CANDIDATS "
        f"{MIN_SCAN_MINUTE}' → "
        f"{MAX_SCAN_MINUTE}'"
    )
    print("=" * 80)

    for index, match in enumerate(candidates, 1):

        print_match_debug(
            match,
            index
        )


if __name__ == "__main__":
    run_test()