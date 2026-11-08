#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 234 — COLLECTE V4.1 MULTI-JOURNÉES

Python 3.9.
Collecte progressive et reprenable.
Aucun entraînement ni création de labels.
Aucune modification de V3-C.
"""

import argparse
import csv
import importlib.util
import json
import re
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import (
    get_events,
    event_minute,
)

ROOT = Path("outputs/v4_research")

MANIFEST = ROOT / "V41_DIVERSE_DATES_MANIFEST_V1.csv"
PILOT = ROOT / "V41_JULY_PILOT_MANIFEST_V1.csv"
TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"

SPLIT = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

DEST = ROOT / "v41_diverse" / "raw_timelines"

ALLOWED_DATES = {
    "2026-07-10",
    "2026-07-17",
    "2026-07-24",
    "2026-07-31",
    "2026-08-07",
}

RELEVANT_TYPES = {
    "goal",
    "own_goal",
    "shot_on_target",
    "shot_off_target",
    "shot_blocked",
    "shot_woodwork",
    "corner_kick",
}

def read_csv(path):
    if not path.is_file():
        raise SystemExit(f"Fichier absent : {path}")

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def get_utc(value):
    dt = datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError("Date sans fuseau horaire")

    return dt.astimezone(timezone.utc)


def check_manifest():
    rows = read_csv(MANIFEST)

    if len(rows) != 124:
        raise SystemExit(
            f"124 candidats attendus, trouvés : {len(rows)}"
        )

    split = read_csv(SPLIT)
    train = read_csv(TRAIN)
    pilot = read_csv(PILOT)

    if (
        len(split) != 320
        or len(train) != 278
        or len(pilot) != 200
    ):
        raise SystemExit(
            "Effectifs des populations protégées inattendus"
        )

    protected = set()

    for name, population in [
        ("SPLIT", split),
        ("TRAIN", train),
        ("PILOT", pilot),
    ]:
        ids = [r["match_id"] for r in population]

        if len(ids) != len(set(ids)):
            raise SystemExit(f"Doublons dans {name}")

        protected.update(ids)

    reserved = {
        r["match_id"]
        for r in split
        if r["split"] == "TEST_FUTUR"
    }

    if len(reserved) != 183:
        raise SystemExit("TEST_FUTUR inattendu")

    seen = set()

    for row in rows:
        mid = row["match_id"]

        if not re.fullmatch(r"mt_[0-9]+", mid):
            raise SystemExit(f"ID invalide : {mid}")

        if mid in seen or mid in protected:
            raise SystemExit(
                f"Doublon ou chevauchement : {mid}"
            )

        seen.add(mid)

        kickoff = get_utc(row["kickoff_utc"])

        if (
            kickoff.strftime("%Y-%m-%d")
            not in ALLOWED_DATES
        ):
            raise SystemExit(
                f"Date non autorisée : {mid}"
            )

        if row["research_status"] != "DISCOVERED_ONLY":
            raise SystemExit(
                f"Statut inattendu : {mid}"
            )

        if row["label_used_for_selection"] != "0":
            raise SystemExit(
                f"Sélection non conforme : {mid}"
            )

    return rows


def load_api():
    path = Path("diagnostic_live_api.py")

    if not path.is_file():
        raise SystemExit("Client API introuvable")

    spec = importlib.util.spec_from_file_location(
        "v41_diverse_api", path
    )

    if spec is None or spec.loader is None:
        raise SystemExit("Impossible de charger l'API")

    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    if not callable(getattr(api, "api_get", None)):
        raise SystemExit("api_get introuvable")

    return api


def inspect_timeline(payload, mid):
    if not isinstance(payload, dict):
        raise ValueError("Réponse API non structurée")

    inner = payload.get("data")

    source = (
        inner if isinstance(inner, dict) else payload
    )

    response_id = (
        source.get("match_id") or source.get("id")
    )

    if (
        response_id is not None
        and str(response_id) != mid
    ):
        raise ValueError(
            f"ID incompatible : {response_id}"
        )

    events = get_events(payload)

    if not isinstance(events, list):
        raise ValueError(
            "Liste des événements non reconnue"
        )

    if not events:
        raise ValueError("Timeline sans événements")

    periods = Counter()
    types = Counter()

    second_half_relevant = 0
    second_half_unknown_minute = 0
    events_77_86 = 0

    for event in events:
        if not isinstance(event, dict):
            raise ValueError(
                "Événement non structuré"
            )

        period = str(
            event.get("period") or ""
        ).strip().lower()

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        periods[period] += 1
        types[typ] += 1

        if period != "second_half":
            continue

        if typ not in RELEVANT_TYPES:
            continue

        second_half_relevant += 1
        minute = event_minute(event)

        if minute is None:
            second_half_unknown_minute += 1

        elif 77 <= minute < 87:
            events_77_86 += 1

    return {
        "events_total": len(events),
        "periods": dict(periods),
        "types": dict(types),
        "second_half_present": (
            periods["second_half"] > 0
        ),
        "second_half_relevant": second_half_relevant,
        "second_half_unknown_minute": (
            second_half_unknown_minute
        ),
        "relevant_events_77_86": events_77_86,
        "coverage_certified": False,
        "label_certified": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit", type=int, default=5
    )
    parser.add_argument(
        "--delay", type=float, default=2.5
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        parser.error("--limit doit être entre 1 et 20")

    if args.delay < 2:
        parser.error("--delay doit être >= 2 secondes")

    print("=" * 72)
    print("ÉTAPE 234 — COLLECTE HISTORIQUE V4.1")
    print("=" * 72)

    rows = check_manifest()
    api = load_api()

    DEST.mkdir(parents=True, exist_ok=True)

    pending = [
        row for row in rows
        if not (
            DEST / (row["match_id"] + ".json")
        ).exists()
    ]

    print("\n=== INVENTAIRE ===")
    print("Candidats :", len(rows))
    print("Déjà enregistrés :", len(rows)-len(pending))
    print("Restants :", len(pending))
    print("Budget maximal API :", args.limit)
    print("Délai :", args.delay, "secondes")

    requested = 0
    saved = 0
    last_request = None

    for row in pending:
        if requested >= args.limit:
            break

        mid = row["match_id"]
        dest = DEST / f"{mid}.json"

        if last_request is not None:
            elapsed = time.monotonic() - last_request
            time.sleep(
                max(0, args.delay - elapsed)
            )

        print(f"\n=== {mid} ===", flush=True)

        requested += 1
        last_request = time.monotonic()

        try:
            payload = api.api_get(
                f"/football/matches/{mid}/timeline"
            )

            diagnostic = inspect_timeline(
                payload, mid
            )

            record = {
                "match_id": mid,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_id": row["home_id"],
                "away_id": row["away_id"],
                "kickoff_utc": row["kickoff_utc"],
                "retrieved_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "research_status": (
                    "RAW_TIMELINE_AUDIT_REQUIRED"
                ),
                "training_certified": False,
                "live_certified": False,
                "label_used_for_selection": False,
                "diagnostic": diagnostic,
                "timeline_response": payload,
            }

            # Création exclusive : aucun écrasement.
            with dest.open(
                "x", encoding="utf-8"
            ) as f:
                json.dump(
                    record, f, ensure_ascii=False
                )
                f.flush()

            saved += 1

            print("ENREGISTRÉ :", dest)
            print(
                "Événements :",
                diagnostic["events_total"]
            )
            print(
                "Deuxième mi-temps :",
                diagnostic["second_half_present"]
            )
            print(
                "Événements 77'–86' :",
                diagnostic["relevant_events_77_86"]
            )
            print(
                "Minutes pertinentes inconnues :",
                diagnostic[
                    "second_half_unknown_minute"
                ]
            )

            if not diagnostic["second_half_present"]:
                print(
                    "ATTENTION : deuxième mi-temps absente."
                )

        except KeyboardInterrupt:
            print("\nArrêt manuel.")
            break

        except Exception as exc:
            print(
                "ERREUR — collecte interrompue :",
                type(exc).__name__,
                str(exc),
            )
            print(
                "Aucune requête supplémentaire "
                "ne sera effectuée dans ce lancement."
            )
            break

    total_saved = sum(
        (DEST / (r["match_id"] + ".json")).exists()
        for r in rows
    )

    print("\n" + "=" * 72)
    print("BILAN")
    print("=" * 72)
    print("Requêtes de ce lancement :", requested)
    print("Nouvelles timelines :", saved)
    print("Total enregistré :", total_saved, "/124")
    print("Restants :", len(rows) - total_saved)
    print("Destination :", DEST)
    print("\nAucun label créé.")
    print("Aucun entraînement.")
    print("VALIDATION / TEST_FUTUR protégés.")
    print("Scanner V3-C inchangé.")
    print("ÉTAPE 234 TERMINÉE.")


if __name__ == "__main__":
    main()
