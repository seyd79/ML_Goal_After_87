#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import collect_v41_diverse_timelines_v1 as previous

ROOT = Path("outputs/v4_research/v41_diverse")
VALID = ROOT / "raw_timelines"
REJECTED = ROOT / "rejected_timelines"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def save_exclusive(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as f:
        json.dump(
            record, f,
            ensure_ascii=False,
            indent=2,
        )
        f.flush()


def describe_payload(payload):
    info = {
        "response_type": type(payload).__name__,
    }

    if isinstance(payload, dict):
        info["top_level_keys"] = list(payload.keys())

        data = payload.get("data")

        info["data_type"] = type(data).__name__

        if isinstance(data, dict):
            info["data_keys"] = list(data.keys())

        elif isinstance(data, list):
            info["data_length"] = len(data)

    elif isinstance(payload, list):
        info["response_length"] = len(payload)

    return info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        parser.error("Limite autorisée : 1 à 20")

    if args.delay < 2:
        parser.error("Délai minimum : 2 secondes")

    print("=" * 72)
    print("ÉTAPE 235 — COLLECTEUR V4.1 V2")
    print("=" * 72)

    rows = previous.check_manifest()
    api = previous.load_api()

    VALID.mkdir(parents=True, exist_ok=True)
    REJECTED.mkdir(parents=True, exist_ok=True)

    # Une réponse rejetée est conservée pour
    # investigation et n'est pas redemandée
    # automatiquement.
    pending = []

    for row in rows:
        mid = row["match_id"]

        good = VALID / f"{mid}.json"
        bad = REJECTED / f"{mid}.json"

        if good.exists() and bad.exists():
            raise SystemExit(
                f"Statuts contradictoires : {mid}"
            )

        if not good.exists() and not bad.exists():
            pending.append(row)

    print("\n=== INVENTAIRE ===")
    print("Candidats :", len(rows))
    print("En attente :", len(pending))
    print("Budget API :", args.limit)
    print("Délai :", args.delay, "secondes")

    requests = 0
    saved = 0
    rejected = 0
    errors = 0
    last_request = None

    for row in pending:
        if requests >= args.limit:
            break

        mid = row["match_id"]

        if last_request is not None:
            remaining = (
                args.delay
                - (time.monotonic() - last_request)
            )

            if remaining > 0:
                time.sleep(remaining)

        print("\n" + "=" * 65)
        print(
            "MATCH :",
            mid,
            "|",
            row["kickoff_utc"][:10],
            flush=True,
        )
        print("=" * 65)

        requests += 1
        last_request = time.monotonic()

        # Les erreurs de requête sont bloquantes.
        try:
            payload = api.api_get(
                f"/football/matches/{mid}/timeline"
            )

        except Exception as exc:
            errors += 1

            print(
                "ERREUR API — ARRÊT :",
                type(exc).__name__,
                str(exc),
            )
            break

        # Seule une timeline sans événements
        # reconnus déclenche une quarantaine.
        try:
            diagnostic = previous.inspect_timeline(
                payload, mid
            )

        except ValueError as exc:
            if str(exc) != "Timeline sans événements":
                errors += 1
                print(
                    "ERREUR STRUCTURE — ARRÊT :",
                    str(exc),
                )
                break

            record = {
                "match_id": mid,
                "kickoff_utc": row["kickoff_utc"],
                "retrieved_at_utc": now_utc(),
                "research_status": (
                    "QUARANTINED_EMPTY_EVENTS"
                ),
                "reason": "NO_RECOGNIZED_EVENTS",
                "diagnostic": describe_payload(
                    payload
                ),
                "training_certified": False,
                "label_certified": False,
                "timeline_response": payload,
            }

            try:
                save_exclusive(
                    REJECTED / f"{mid}.json",
                    record,
                )
            except Exception as save_error:
                errors += 1
                print(
                    "ERREUR SAUVEGARDE — ARRÊT :",
                    save_error,
                )
                break

            rejected += 1

            print(
                "QUARANTAINE : aucun événement reconnu"
            )
            print(
                "Structure API :",
                record["diagnostic"],
            )
            print(
                "Réponse conservée :",
                REJECTED / f"{mid}.json",
            )

            continue

        except Exception as exc:
            errors += 1
            print(
                "ERREUR INATTENDUE — ARRÊT :",
                type(exc).__name__,
                str(exc),
            )
            break

        record = {
            "match_id": mid,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_id": row["home_id"],
            "away_id": row["away_id"],
            "kickoff_utc": row["kickoff_utc"],
            "retrieved_at_utc": now_utc(),
            "research_status": (
                "RAW_TIMELINE_AUDIT_REQUIRED"
            ),
            "training_certified": False,
            "live_certified": False,
            "label_used_for_selection": False,
            "diagnostic": diagnostic,
            "timeline_response": payload,
        }

        try:
            save_exclusive(
                VALID / f"{mid}.json",
                record,
            )

        except Exception as exc:
            errors += 1
            print(
                "ERREUR SAUVEGARDE — ARRÊT :",
                str(exc),
            )
            break

        saved += 1

        print("ENREGISTRÉ :", mid)
        print(
            "Événements :",
            diagnostic["events_total"],
        )
        print(
            "Deuxième mi-temps :",
            diagnostic["second_half_present"],
        )
        print(
            "Minutes pertinentes inconnues :",
            diagnostic[
                "second_half_unknown_minute"
            ],
        )

    total_valid = sum(
        (VALID / f"{r['match_id']}.json").exists()
        for r in rows
    )

    total_rejected = sum(
        (REJECTED / f"{r['match_id']}.json").exists()
        for r in rows
    )

    print("\n" + "=" * 72)
    print("BILAN — ÉTAPE 235")
    print("=" * 72)
    print("Requêtes effectuées :", requests)
    print("Nouvelles timelines enregistrées :", saved)
    print("Nouvelles quarantaines :", rejected)
    print("Erreurs bloquantes :", errors)
    print("Total timelines enregistrées :", total_valid)
    print("Total quarantaines :", total_rejected)
    print(
        "Candidats encore non traités :",
        len(rows) - total_valid - total_rejected,
    )

    print("\nAucun label créé.")
    print("Aucun modèle entraîné.")
    print("Aucune quarantaine utilisée comme négatif.")
    print("Scanner V3-C inchangé.")
    print("ÉTAPE 235 TERMINÉE.")


if __name__ == "__main__":
    main()
