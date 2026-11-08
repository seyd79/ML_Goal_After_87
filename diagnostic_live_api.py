#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DIAGNOSTIC LIVE API — TheStatsAPI

Objectif :
    Récupérer un match LIVE réel et sauvegarder toutes les
    réponses API utiles afin d'identifier exactement où se
    trouvent :
        - la minute
        - le score
        - les statistiques
        - les événements
        - les tirs
        - les corners
        - les cartons
        - les remplacements
        - les cotes

Le script ne modifie PAS le modèle ML.
Il sert uniquement à inspecter les données réelles de l'API.
"""

import os
import json
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

ENV_PATH = PROJECT_DIR / ".env"

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "api_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENV
# ============================================================

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


STATS_API_KEY = os.getenv(
    "STATS_API_KEY",
    "",
).strip()

STATS_API_BASE_URL = os.getenv(
    "STATS_API_BASE_URL",
    "https://api.thestatsapi.com/api",
).strip().rstrip("/")

STATS_API_TIMEOUT = int(
    os.getenv(
        "STATS_API_TIMEOUT",
        "20",
    )
)


# ============================================================
# VALIDATION
# ============================================================

if not STATS_API_KEY:

    print(
        "\nERREUR : STATS_API_KEY est absent du .env\n"
    )

    sys.exit(1)


# ============================================================
# SESSION API
# ============================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {STATS_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
})


# ============================================================
# API GET
# ============================================================

def api_get(endpoint, params=None):

    url = (
        f"{STATS_API_BASE_URL}"
        f"{endpoint}"
    )

    print()
    print("=" * 70)
    print("GET")
    print(url)

    if params:
        print(
            f"PARAMS : {params}"
        )

    print("=" * 70)

    response = session.get(
        url,
        params=params,
        timeout=STATS_API_TIMEOUT,
    )

    print(
        f"HTTP STATUS : {response.status_code}"
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    filename,
    data,
):

    output_path = (
        OUTPUT_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"✓ Sauvegardé : {output_path}"
    )


# ============================================================
# PRINT JSON SUMMARY
# ============================================================

def print_summary(
    title,
    data,
):

    print()
    print(
        "#" * 70
    )

    print(title)

    print(
        "#" * 70
    )

    print(
        f"Type : {type(data).__name__}"
    )

    if isinstance(data, dict):

        print(
            "Clés principales :"
        )

        for key in data.keys():

            print(
                f"  - {key}"
            )

    elif isinstance(data, list):

        print(
            f"Nombre d'éléments : "
            f"{len(data)}"
        )

        if data:

            first = data[0]

            if isinstance(
                first,
                dict,
            ):

                print(
                    "Clés du premier élément :"
                )

                for key in first.keys():

                    print(
                        f"  - {key}"
                    )


# ============================================================
# EXTRACTION MATCH ID
# ============================================================

def extract_match_id(match):

    if not isinstance(
        match,
        dict,
    ):

        return None

    for key in [
        "id",
        "match_id",
        "fixture_id",
    ]:

        if (
            key in match
            and match[key] is not None
        ):

            return match[key]

    return None


# ============================================================
# EXTRACTION ÉQUIPE
# ============================================================

def extract_team_name(team):

    if isinstance(
        team,
        str,
    ):

        return team

    if not isinstance(
        team,
        dict,
    ):

        return "Unknown"

    for key in [
        "name",
        "team_name",
        "short_name",
        "display_name",
    ]:

        if (
            key in team
            and team[key] is not None
        ):

            return team[key]

    return "Unknown"


# ============================================================
# EXTRACTION TEAMS
# ============================================================

def extract_teams(match):

    home = None
    away = None

    if not isinstance(
        match,
        dict,
    ):

        return (
            "Unknown",
            "Unknown",
        )

    # --------------------------------------------------------
    # Champs directs
    # --------------------------------------------------------

    for key in [
        "home_team",
        "home",
        "team_home",
    ]:

        if key in match:

            home = match[key]

            break

    for key in [
        "away_team",
        "away",
        "team_away",
    ]:

        if key in match:

            away = match[key]

            break

    # --------------------------------------------------------
    # Objet teams
    # --------------------------------------------------------

    if (
        home is None
        and "teams" in match
    ):

        teams = match["teams"]

        if isinstance(
            teams,
            dict,
        ):

            home = teams.get(
                "home"
            )

            away = teams.get(
                "away"
            )

        elif (
            isinstance(
                teams,
                list,
            )
            and len(teams) >= 2
        ):

            home = teams[0]
            away = teams[1]

    return (
        extract_team_name(home),
        extract_team_name(away),
    )


# ============================================================
# EXTRACTION SCORE
# ============================================================

def extract_score(match):

    home_score = None
    away_score = None

    if not isinstance(
        match,
        dict,
    ):

        return (
            None,
            None,
        )

    score = match.get(
        "score"
    )

    if isinstance(
        score,
        dict,
    ):

        for key in [
            "home",
            "home_score",
            "home_goals",
        ]:

            if key in score:

                home_score = score[key]

                break

        for key in [
            "away",
            "away_score",
            "away_goals",
        ]:

            if key in score:

                away_score = score[key]

                break

    # --------------------------------------------------------
    # Direct fields
    # --------------------------------------------------------

    if home_score is None:

        for key in [
            "home_score",
            "home_goals",
            "score_home",
        ]:

            if key in match:

                home_score = match[key]

                break

    if away_score is None:

        for key in [
            "away_score",
            "away_goals",
            "score_away",
        ]:

            if key in match:

                away_score = match[key]

                break

    return (
        home_score,
        away_score,
    )


# ============================================================
# RECHERCHE RÉCURSIVE DES CHAMPS
# ============================================================

def find_keys_recursive(
    data,
    interesting_keys,
    path="root",
    results=None,
):

    if results is None:
        results = []

    if isinstance(
        data,
        dict,
    ):

        for key, value in data.items():

            current_path = (
                f"{path}.{key}"
            )

            if key.lower() in interesting_keys:

                results.append(
                    {
                        "path": current_path,
                        "value": value,
                    }
                )

            find_keys_recursive(
                value,
                interesting_keys,
                current_path,
                results,
            )

    elif isinstance(
        data,
        list,
    ):

        for index, value in enumerate(data):

            current_path = (
                f"{path}[{index}]"
            )

            find_keys_recursive(
                value,
                interesting_keys,
                current_path,
                results,
            )

    return results


# ============================================================
# DIAGNOSTIC DES CHAMPS IMPORTANTS
# ============================================================

def diagnose_fields(
    match_data,
    live_stats,
    stats,
    timeline,
):

    interesting_keys = {
        "minute",
        "minutes",
        "elapsed",
        "time",
        "current_minute",
        "match_minute",
        "status",
        "score",
        "home_score",
        "away_score",
        "shots",
        "shots_on_target",
        "shot",
        "corners",
        "fouls",
        "yellow_cards",
        "red_cards",
        "cards",
        "substitutions",
        "subs",
        "goal",
        "goals",
        "event",
        "type",
        "detail",
        "description",
    }

    print()
    print(
        "=" * 70
    )

    print(
        "RECHERCHE DES CHAMPS IMPORTANTS"
    )

    print(
        "=" * 70
    )

    datasets = {
        "MATCH": match_data,
        "LIVE_STATS": live_stats,
        "STATS": stats,
        "TIMELINE": timeline,
    }

    for dataset_name, data in datasets.items():

        print()
        print(
            f"--- {dataset_name} ---"
        )

        results = find_keys_recursive(
            data,
            interesting_keys,
        )

        if not results:

            print(
                "Aucun champ intéressant trouvé."
            )

            continue

        for result in results[:200]:

            value = result["value"]

            text = repr(value)

            if len(text) > 300:

                text = (
                    text[:300]
                    + "..."
                )

            print(
                f"{result['path']} = {text}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "DIAGNOSTIC THESTATSAPI — LIVE"
    )

    print(
        "=" * 70
    )

    print(
        f"API : {STATS_API_BASE_URL}"
    )

    print(
        f"Output : {OUTPUT_DIR}"
    )

    # ========================================================
    # 1. RÉCUPÉRER MATCHS LIVE
    # ========================================================

    print()
    print(
        "Récupération des matchs LIVE..."
    )

    live_response = api_get(
        "/football/matches",
        params={
            "status": "live",
            "per_page": 100,
        },
    )

    save_json(
        "01_live_matches.json",
        live_response,
    )

    print_summary(
        "RÉPONSE MATCHS LIVE",
        live_response,
    )

    # ========================================================
    # EXTRAIRE LISTE MATCHS
    # ========================================================

    if isinstance(
        live_response,
        dict,
    ):

        live_matches = None

        for key in [
            "data",
            "matches",
            "results",
        ]:

            if key in live_response:

                live_matches = (
                    live_response[key]
                )

                break

        if live_matches is None:

            live_matches = []

    elif isinstance(
        live_response,
        list,
    ):

        live_matches = live_response

    else:

        live_matches = []

    print()
    print(
        f"Matchs LIVE trouvés : "
        f"{len(live_matches)}"
    )

    if not live_matches:

        print()
        print(
            "Aucun match LIVE."
        )

        print(
            "Le diagnostic s'arrête ici."
        )

        return

    # ========================================================
    # AFFICHER LES MATCHS
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "MATCHS LIVE"
    )

    print(
        "=" * 70
    )

    for index, match in enumerate(
        live_matches,
        start=1,
    ):

        match_id = (
            extract_match_id(match)
        )

        home, away = (
            extract_teams(match)
        )

        home_score, away_score = (
            extract_score(match)
        )

        print(
            f"{index:02d}. "
            f"{home} "
            f"{home_score}-{away_score} "
            f"{away} "
            f"| ID = {match_id}"
        )

    # ========================================================
    # PRENDRE LE PREMIER MATCH
    # ========================================================

    selected_match = (
        live_matches[0]
    )

    match_id = extract_match_id(
        selected_match
    )

    if match_id is None:

        print()
        print(
            "Impossible de trouver le match_id."
        )

        return

    home, away = extract_teams(
        selected_match
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MATCH SÉLECTIONNÉ POUR DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    print(
        f"Match : {home} vs {away}"
    )

    print(
        f"ID : {match_id}"
    )

    # ========================================================
    # 2. MATCH DETAILS
    # ========================================================

    match_data = api_get(
        f"/football/matches/{match_id}"
    )

    save_json(
        "02_match_details.json",
        match_data,
    )

    print_summary(
        "MATCH DETAILS",
        match_data,
    )

    # ========================================================
    # 3. LIVE STATS
    # ========================================================

    try:

        live_stats = api_get(
            f"/football/matches/{match_id}/live-stats"
        )

    except Exception as exc:

        print(
            f"Erreur live-stats : {exc}"
        )

        live_stats = {}

    save_json(
        "03_live_stats.json",
        live_stats,
    )

    print_summary(
        "LIVE STATS",
        live_stats,
    )

    # ========================================================
    # 4. STATS
    # ========================================================

    try:

        stats = api_get(
            f"/football/matches/{match_id}/stats"
        )

    except Exception as exc:

        print(
            f"Erreur stats : {exc}"
        )

        stats = {}

    save_json(
        "04_stats.json",
        stats,
    )

    print_summary(
        "MATCH STATS",
        stats,
    )

    # ========================================================
    # 5. TIMELINE
    # ========================================================

    try:

        timeline = api_get(
            f"/football/matches/{match_id}/timeline"
        )

    except Exception as exc:

        print(
            f"Erreur timeline : {exc}"
        )

        timeline = {}

    save_json(
        "05_timeline.json",
        timeline,
    )

    print_summary(
        "TIMELINE",
        timeline,
    )

    # ========================================================
    # 6. ODDS
    # ========================================================

    try:

        odds = api_get(
            f"/football/matches/{match_id}/odds"
        )

    except Exception as exc:

        print(
            f"Erreur odds : {exc}"
        )

        odds = {}

    save_json(
        "06_odds.json",
        odds,
    )

    print_summary(
        "ODDS",
        odds,
    )

    # ========================================================
    # 7. LIVE ODDS
    # ========================================================

    try:

        live_odds = api_get(
            f"/football/matches/{match_id}/odds/live"
        )

    except Exception as exc:

        print(
            f"Erreur live odds : {exc}"
        )

        live_odds = {}

    save_json(
        "07_live_odds.json",
        live_odds,
    )

    print_summary(
        "LIVE ODDS",
        live_odds,
    )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    diagnose_fields(
        match_data=match_data,
        live_stats=live_stats,
        stats=stats,
        timeline=timeline,
    )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "DIAGNOSTIC TERMINÉ"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Fichiers créés dans :"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "01_live_matches.json"
    )

    print(
        "02_match_details.json"
    )

    print(
        "03_live_stats.json"
    )

    print(
        "04_stats.json"
    )

    print(
        "05_timeline.json"
    )

    print(
        "06_odds.json"
    )

    print(
        "07_live_odds.json"
    )

    print()

    print(
        "IMPORTANT :"
    )

    print(
        "Ne partage jamais ta STATS_API_KEY."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Diagnostic interrompu."
        )

    except Exception as exc:

        print()
        print(
            "=" * 70
        )

        print(
            "ERREUR"
        )

        print(
            "=" * 70
        )

        print(
            repr(exc)
        )

        sys.exit(1)