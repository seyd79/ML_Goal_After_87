#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import get_events, event_minute
from v4_pressure_features_research import compute_b6_historical

OUT = Path(
    "outputs/v4_research/V4_LIVE_SHADOW87_CAPTURES_V1.jsonl"
)

API_SOURCE = Path("diagnostic_live_api.py")


def now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def valid_int(value):
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
        if str(value).strip() != str(number):
            return None
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def as_matches(payload):
    if isinstance(payload, list):
        return payload if all(isinstance(x, dict) for x in payload) else None
    if isinstance(payload, dict):
        for key in ("matches", "data", "results", "response", "items"):
            if key in payload:
                result = as_matches(payload[key])
                if result is not None:
                    return result
    return None


def eligible(match):
    if not isinstance(match, dict):
        return False

    live = match.get("live") or {}
    score = match.get("score") or {}
    home = match.get("home_team") or {}
    away = match.get("away_team") or {}

    if not all(isinstance(x, dict) for x in (live, score, home, away)):
        return False

    if match.get("status") != "live":
        return False

    if live.get("period") != "second_half":
        return False

    if valid_int(live.get("elapsed_minutes")) != 87:
        return False

    updated = parse_utc(live.get("updated_at"))
    if updated is None:
        return False

    age = (now() - updated).total_seconds()
    if not -10 <= age <= 120:
        return False

    if valid_int(score.get("home")) is None:
        return False

    if valid_int(score.get("away")) is None:
        return False

    if not (
        match.get("id")
        and home.get("id")
        and away.get("id")
        and home.get("name")
        and away.get("name")
    ):
        return False

    return True


def existing_ids():
    if not OUT.exists():
        return set()

    ids = set()
    with OUT.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                ids.add(json.loads(line)["match_id"])
    return ids


def main():
    print("=" * 72)
    print("ÉTAPE 217 — CAPTURE LIVE SHADOW À 87'")
    print("=" * 72)

    if not API_SOURCE.is_file():
        raise SystemExit("Client API introuvable")

    spec = importlib.util.spec_from_file_location(
        "v4_shadow_api", API_SOURCE
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    def live_matches():
        response = api.api_get(
            "/football/matches",
            params={
                "status": "live",
                "page": 1,
                "per_page": 100,
            },
        )
        result = as_matches(response)
        if result is None:
            raise ValueError("Structure LIVE non reconnue")
        return result

    print("\n=== REQUÊTE 1 : MATCHS LIVE ===")
    matches = live_matches()

    already = existing_ids()

    candidates = [
        m for m in matches
        if eligible(m)
        and str(m["id"]) not in already
    ]

    print("Matchs LIVE :", len(matches))
    print("Nouveaux matchs admissibles à 87' :", len(candidates))

    if not candidates:
        print("\nAucune capture effectuée.")
        print("Ce résultat est normal en l'absence de match à 87'.")
        return

    first = candidates[0]
    mid = str(first["id"])

    print("\n=== MATCH SÉLECTIONNÉ ===")
    print("Match :", first["home_team"]["name"], "-",
          first["away_team"]["name"])
    print("ID :", mid)
    print("Score :", first["score"]["home"],
          "-", first["score"]["away"])
    print("Minute API : 87")

    print("\n=== REQUÊTE 2 : TIMELINE ===")

    timeline_started = now()
    timeline = api.api_get(
        f"/football/matches/{mid}/timeline"
    )
    timeline_received = now()

    events = get_events(timeline)

    if events is None:
        print("REFUS : timeline non reconnue")
        return

    for event in events:
        if event.get("period") != "second_half":
            continue

        minute = event_minute(event)

        if minute is None:
            print("REFUS : événement de deuxième mi-temps sans minute")
            return

        if minute > 87:
            print("REFUS : timeline contenant un événement après 87'")
            return

    print("\n=== REQUÊTE 3 : REVÉRIFICATION LIVE ===")

    second_matches = live_matches()
    second = next(
        (
            m for m in second_matches
            if str(m.get("id")) == mid
        ),
        None,
    )

    if not eligible(second):
        print("REFUS : statut ou minute LIVE non confirmé")
        return

    if (
        first["score"].get("home") != second["score"].get("home")
        or first["score"].get("away") != second["score"].get("away")
    ):
        print("REFUS : score modifié pendant la capture")
        return

    if mid in existing_ids():
        print("REFUS : match déjà enregistré")
        return

    try:
        b6 = compute_b6_historical(
            timeline,
            home_team=first["home_team"]["name"],
            away_team=first["away_team"]["name"],
            home_id=str(first["home_team"]["id"]),
            away_id=str(first["away_team"]["id"]),
        )
    except Exception as exc:
        print("REFUS : caractéristiques non calculables :", exc)
        return

    record = {
        "match_id": mid,
        "status": "RESEARCH_SHADOW_UNVERIFIED_COVERAGE",
        "source": "LIVE_API_OBSERVATION",
        "first_observation_utc": timeline_started.isoformat(),
        "timeline_received_utc": timeline_received.isoformat(),
        "capture_completed_utc": now().isoformat(),
        "minute_first_check": 87,
        "minute_second_check": 87,
        "kickoff_utc": first.get("utc_date"),
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "score_home": valid_int(first["score"]["home"]),
        "score_away": valid_int(first["score"]["away"]),
        "live_first": first["live"],
        "live_second": second["live"],
        "b6": b6,
        "events_count": len(events),
        "events_at_minute_87": sum(
            1 for e in events
            if event_minute(e) == 87
        ),
        "timeline_payload": timeline,
        "coverage_independently_verified": False,
        "prediction_certified": False,
    }

    canonical = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
    )
    record["sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )
        file.flush()

    print("\n=== CAPTURE ENREGISTRÉE ===")
    print("Fichier :", OUT)
    print("Match :", mid)
    print("Score :", record["score_home"],
          "-", record["score_away"])
    print("Nombre d'événements :", record["events_count"])

    print("\n=== CARACTÉRISTIQUES B6 ===")
    for key, value in b6.items():
        print(f"{key:25s} : {value}")

    print("\n=== LIMITES ===")
    print("Couverture des tirs/corners non certifiée.")
    print("Fraîcheur API contrôlée, pas garantie indépendante.")
    print("Aucune prédiction V4 générée.")
    print("Aucun label consulté.")
    print("Modèles figés inchangés.")
    print("Scanner V3-C inchangé.")
    print("ÉTAPE 217 TERMINÉE.")


if __name__ == "__main__":
    main()
