#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from capture_v4_shadow87_v1 import (
    as_matches, eligible, now, parse_utc, valid_int,
)
from v3c_pressure_features import get_events, event_minute
from v4_pressure_features_research import compute_b6_historical

ROOT = Path("outputs/v4_research")
FROZEN = ROOT / "frozen_simple6_v1"
OUTPUT = ROOT / "V4_LIVE_SHADOW87_PREDICTIONS_V1.jsonl"
OLD_CAPTURES = ROOT / "V4_LIVE_SHADOW87_CAPTURES_V1.jsonl"
PROTOCOL = FROZEN / "V4_FROZEN_PROTOCOL.json"

FEATURES = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]


def saved_ids(path):
    if not path.exists():
        return set()

    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["match_id"]))
    return ids


def load_api():
    path = Path("diagnostic_live_api.py")
    spec = importlib.util.spec_from_file_location(
        "v4_independent_api", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_live(api):
    payload = api.api_get(
        "/football/matches",
        params={
            "status": "live",
            "page": 1,
            "per_page": 100,
        },
    )
    matches = as_matches(payload)
    if matches is None:
        raise ValueError("Liste LIVE non reconnue")
    return matches


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print("=" * 72)
    print("ÉTAPE 218 — PRÉDICTIONS V4 LIVE SHADOW À 87'")
    print("=" * 72)

    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    if (
        protocol["features"] != FEATURES
        or float(protocol["threshold"]) != 0.30
        or protocol["primary_model"] != "SIMPLE6_OLD183"
        or protocol["challenger_model"] != "SIMPLE6_NEW278"
    ):
        raise SystemExit("Protocole figé incompatible")

    old_path = FROZEN / "SIMPLE6_OLD183.joblib"
    new_path = FROZEN / "SIMPLE6_NEW278.joblib"

    for path in (old_path, new_path):
        if not path.is_file():
            raise SystemExit(f"Modèle absent : {path}")

    models = {
        "OLD183": joblib.load(old_path),
        "NEW278": joblib.load(new_path),
    }

    excluded = saved_ids(OLD_CAPTURES) | saved_ids(OUTPUT)

    api = load_api()

    print("\n=== REQUÊTE 1 — MATCHS LIVE ===")
    matches = get_live(api)

    candidates = [
        m for m in matches
        if eligible(m)
        and str(m["id"]) not in excluded
    ]

    print("Matchs LIVE :", len(matches))
    print("Candidats nouveaux à 87' :", len(candidates))

    if not candidates:
        print("Aucune prédiction effectuée.")
        return

    match = candidates[0]
    mid = str(match["id"])

    kickoff = parse_utc(match.get("utc_date"))
    frozen_at = parse_utc(protocol["created_utc"])

    if (
        kickoff is None
        or frozen_at is None
        or kickoff <= frozen_at
    ):
        print("REFUS : date du match ou du gel invalide.")
        return

    home = match["home_team"]
    away = match["away_team"]

    print("\nMatch :", home["name"], "-", away["name"])
    print("Minute :", 87)
    print("Score :", match["score"]["home"],
          "-", match["score"]["away"])

    print("\n=== REQUÊTE 2 — TIMELINE LIVE ===")

    started_at = now()
    timeline = api.api_get(
        f"/football/matches/{mid}/timeline"
    )
    received_at = now()

    events = get_events(timeline)

    if events is None:
        print("REFUS : timeline inconnue")
        return

    for event in events:
        if event.get("period") != "second_half":
            continue

        minute = event_minute(event)

        if minute is None:
            print("REFUS : minute d'événement inconnue")
            return

        if minute >= 87:
            print(
                "REFUS : événement à partir de 87' "
                "dans la timeline."
            )
            return

    print("\n=== REQUÊTE 3 — CONFIRMATION LIVE ===")

    confirmation = get_live(api)

    confirmed = next(
        (
            m for m in confirmation
            if str(m.get("id")) == mid
        ),
        None,
    )

    if not eligible(confirmed):
        print("REFUS : minute 87' non reconfirmée")
        return

    if (
        match["score"]["home"] != confirmed["score"]["home"]
        or match["score"]["away"] != confirmed["score"]["away"]
    ):
        print("REFUS : score modifié")
        return

    first_update = parse_utc(match["live"]["updated_at"])
    last_update = parse_utc(confirmed["live"]["updated_at"])

    if (
        first_update is None
        or last_update is None
        or last_update < first_update
    ):
        print("REFUS : mises à jour LIVE incohérentes")
        return

    if (
        str(home["id"]) !=
        str(confirmed["home_team"]["id"])
        or str(away["id"]) !=
        str(confirmed["away_team"]["id"])
    ):
        print("REFUS : identité des équipes modifiée")
        return

    if mid in (saved_ids(OLD_CAPTURES) | saved_ids(OUTPUT)):
        print("REFUS : match déjà enregistré")
        return

    b6 = compute_b6_historical(
        timeline,
        home_team=home["name"],
        away_team=away["name"],
        home_id=str(home["id"]),
        away_id=str(away["id"]),
    )

    sh = valid_int(match["score"]["home"])
    sa = valid_int(match["score"]["away"])

    x = {
        "score_total_87": sh + sa,
        "score_diff_abs_87": abs(sh - sa),
        "is_draw_87": int(sh == sa),
        "shots_total_last5": (
            b6["shots_home_last5"]
            + b6["shots_away_last5"]
        ),
        "sot_total_last5": (
            b6["sot_home_last5"]
            + b6["sot_away_last5"]
        ),
        "corners_total_last5": (
            b6["corners_home_last5"]
            + b6["corners_away_last5"]
        ),
    }

    frame = pd.DataFrame([x], columns=FEATURES)

    p_old = float(
        models["OLD183"].predict_proba(frame)[0, 1]
    )
    p_new = float(
        models["NEW278"].predict_proba(frame)[0, 1]
    )

    predicted_at = now()

    record = {
        "match_id": mid,
        "kickoff_utc": kickoff.isoformat(),
        "home_team": home["name"],
        "away_team": away["name"],
        "minute": 87,
        "score_home": sh,
        "score_away": sa,
        "timeline_requested_at_utc": started_at.isoformat(),
        "timeline_received_at_utc": received_at.isoformat(),
        "predicted_at_utc": predicted_at.isoformat(),
        "first_live_updated_at_utc": first_update.isoformat(),
        "second_live_updated_at_utc": last_update.isoformat(),
        "b6": b6,
        "features": x,
        "p_old183": p_old,
        "p_new278": p_new,
        "alert_old183": int(p_old >= 0.30),
        "alert_new278": int(p_new >= 0.30),
        "events_count": len(events),
        "timeline_payload": timeline,
        "model_old_sha256": sha256(old_path),
        "model_new_sha256": sha256(new_path),
        "protocol_version": protocol["version"],
        "status": "RESEARCH_SHADOW",
        "timeline_coverage_verified": False,
        "prediction_certified": False,
    }

    canonical = json.dumps(
        record, sort_keys=True,
        ensure_ascii=False,
    )

    record["sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print("\n=== PRÉDICTION SHADOW ENREGISTRÉE ===")
    print("Match :", mid)
    print("OLD183 :", f"{p_old:.2%}",
          "| alerte :", int(p_old >= 0.30))
    print("NEW278 :", f"{p_new:.2%}",
          "| alerte :", int(p_new >= 0.30))
    print("Horodatage :", predicted_at.isoformat())
    print("Fichier :", OUTPUT)
    print("\nRESEARCH_SHADOW : couverture non certifiée.")
    print("Aucun label consulté.")
    print("Aucun modèle modifié.")
    print("Scanner V3-C inchangé.")


if __name__ == "__main__":
    main()
