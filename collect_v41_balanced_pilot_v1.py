#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 237 — PILOTE ÉQUILIBRÉ V4.1

- Une nouvelle rencontre par journée
- Cinq appels API maximum
- Reprise sans retélécharger les réponses enregistrées
- Quarantaine des réponses sans couverture complète
- Aucun label, entraînement ou changement V3-C
"""

import json
import os
import time

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import collect_v41_diverse_timelines_v1 as base
from v3c_pressure_features import get_events

ROOT = Path("outputs/v4_research/v41_diverse")
VALID = ROOT / "raw_timelines"
REJECTED = ROOT / "rejected_timelines"

DAYS = [
    "2026-07-10",
    "2026-07-17",
    "2026-07-24",
    "2026-07-31",
    "2026-08-07",
]

DELAY_SECONDS = 2.5
MAX_REQUESTS = 5


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def check_response(payload, mid):
    if not isinstance(payload, dict):
        raise ValueError("Réponse API non structurée")

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError("Bloc data absent ou invalide")

    if str(data.get("match_id")) != mid:
        raise ValueError("match_id API incompatible")

    coverage = data.get("coverage")
    raw_events = data.get("events")

    if not isinstance(raw_events, list):
        raise ValueError("events n'est pas une liste")

    events = get_events(payload)

    if not isinstance(events, list):
        raise ValueError("Format des événements non reconnu")

    if len(events) != len(raw_events):
        raise ValueError("Nombre d'événements incohérent")

    if coverage == "none" and len(events) == 0:
        return "QUARANTINE", "NO_COVERAGE", None

    if coverage != "full":
        return "QUARANTINE", "COVERAGE_NOT_FULL", None

    if not events:
        return "QUARANTINE", "EMPTY_DESPITE_FULL", None

    diagnostic = base.inspect_timeline(payload, mid)

    if not diagnostic["second_half_present"]:
        return "QUARANTINE", "NO_SECOND_HALF", diagnostic

    if diagnostic["second_half_unknown_minute"] > 0:
        return "QUARANTINE", "UNKNOWN_MINUTES", diagnostic

    return "SAVED", "FULL_AUDIT_REQUIRED", diagnostic


def main():
    print("=" * 72)
    print("ÉTAPE 237 — PILOTE ÉQUILIBRÉ V4.1")
    print("=" * 72)

    rows = base.check_manifest()
    api = base.load_api()

    VALID.mkdir(parents=True, exist_ok=True)
    REJECTED.mkdir(parents=True, exist_ok=True)

    by_day = defaultdict(list)

    for row in rows:
        mid = row["match_id"]
        day = row["kickoff_utc"][:10]

        good = VALID / f"{mid}.json"
        bad = REJECTED / f"{mid}.json"

        if good.exists() and bad.exists():
            raise SystemExit(
                f"Fichiers contradictoires : {mid}"
            )

        if not good.exists() and not bad.exists():
            by_day[day].append(row)

    selected = []

    for day in DAYS:
        pending = by_day[day]
        if pending:
            selected.append(pending[0])

    print("\n=== PLAN DE COLLECTE ===")

    for row in selected:
        print(
            row["kickoff_utc"][:10],
            "|", row["match_id"],
            "|", row["home_team"],
            "vs", row["away_team"],
        )

    print("\nRequêtes maximales :", MAX_REQUESTS)
    print("Délai :", DELAY_SECONDS, "secondes")
    print("Sélection indépendante du résultat final.")

    if len(selected) > MAX_REQUESTS:
        raise SystemExit("Budget API dépassé")

    counts = defaultdict(
        lambda: {
            "SAVED": 0,
            "QUARANTINE": 0,
        }
    )

    requested = 0
    last_request = None

    for row in selected:
        mid = row["match_id"]
        day = row["kickoff_utc"][:10]

        if last_request is not None:
            wait = DELAY_SECONDS - (
                time.monotonic() - last_request
            )

            if wait > 0:
                time.sleep(wait)

        print("\n" + "=" * 65)
        print("MATCH :", mid, "|", day, flush=True)
        print("=" * 65)

        requested += 1
        last_request = time.monotonic()

        try:
            payload = api.api_get(
                f"/football/matches/{mid}/timeline"
            )

            status, reason, diagnostic = check_response(
                payload, mid
            )

            record = {
                "match_id": mid,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_id": row["home_id"],
                "away_id": row["away_id"],
                "kickoff_utc": row["kickoff_utc"],
                "retrieved_at_utc": now(),
                "research_status": (
                    "RAW_TIMELINE_AUDIT_REQUIRED"
                    if status == "SAVED"
                    else "QUARANTINED"
                ),
                "reason": reason,
                "training_certified": False,
                "label_certified": False,
                "live_certified": False,
                "label_used_for_selection": False,
                "diagnostic": diagnostic,
                "timeline_response": payload,
            }

            dest_dir = (
                VALID if status == "SAVED"
                else REJECTED
            )

            dest = dest_dir / f"{mid}.json"
            save_new(dest, record)

            counts[day][status] += 1

            print("STATUT :", status)
            print("RAISON :", reason)

            if diagnostic:
                print(
                    "Événements :",
                    diagnostic["events_total"]
                )

            print("Enregistré :", dest)

        except Exception as exc:
            print(
                "\nARRÊT DE SÉCURITÉ :",
                type(exc).__name__,
                str(exc),
            )
            print(
                "Aucune autre requête dans ce lancement."
            )
            break

    print("\n" + "=" * 72)
    print("BILAN — ÉTAPE 237")
    print("=" * 72)

    print("Requêtes :", requested)

    for day in DAYS:
        print(
            day,
            "| nouveaux full :",
            counts[day]["SAVED"],
            "| nouvelles quarantaines :",
            counts[day]["QUARANTINE"],
        )

    total_valid = sum(
        (VALID / f"{r['match_id']}.json").exists()
        for r in rows
    )

    total_rejected = sum(
        (REJECTED / f"{r['match_id']}.json").exists()
        for r in rows
    )

    print("\nTOTAL ENREGISTRÉ :", total_valid)
    print("TOTAL QUARANTAINE :", total_rejected)
    print(
        "NON TRAITÉS :",
        len(rows) - total_valid - total_rejected
    )

    print("\nAucun label créé.")
    print("Aucun modèle entraîné.")
    print("Aucune modification de V3-C.")
    print("ÉTAPE 237 TERMINÉE.")


if __name__ == "__main__":
    main()
