import argparse
import csv
import json
import os
import time
import urllib.error

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import rebuild_historical_pressure as original
import safe_historical_batch02_v1 as safe

ROOT = Path("outputs/pressure_historical")
CACHE = ROOT / "timeline_cache"
JOIN = ROOT / "pressure_v2_dataset_join"
OUT = Path("outputs/v4_research")

PLAN = OUT / "V4_FETCH83_MANIFEST_V1.csv"
LOG = OUT / "V4_FETCH83_SAFE_COLLECTION_LOG_V1.jsonl"

CUTOFF = pd.Timestamp("2026-08-25T00:00:00Z")


def log_result(mid, status, detail=""):
    LOG.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "match_id": mid,
        "status": status,
        "detail": detail,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_and_verify():
    if not PLAN.is_file():
        raise SystemExit("Manifeste FETCH83 absent")

    with PLAN.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    ids = [r["match_id"] for r in rows]

    if len(rows) != 83 or len(set(ids)) != 83:
        raise SystemExit("83 IDs uniques attendus")

    dates = pd.to_datetime(
        [r["kickoff_utc"] for r in rows],
        utc=True,
        errors="raise",
    )

    if not (dates < CUTOFF).all():
        raise SystemExit("Match postérieur au cutoff TRAIN")

    split = pd.read_csv(
        JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
        dtype=str,
        usecols=["match_id", "split"],
    )

    existing = set(split.match_id)
    reserved = set(
        split.loc[
            split["split"] == "TEST_FUTUR",
            "match_id"
        ]
    )

    batch_paths = sorted(
        JOIN.glob("BATCH*_B_RESEARCH_V1.csv")
    )

    if len(batch_paths) != 10:
        raise SystemExit("10 lots historiques attendus")

    for path in batch_paths:
        part = pd.read_csv(
            path,
            dtype=str,
            usecols=["match_id"],
        )
        part_ids = set(part.match_id)

        if len(part_ids) != len(part):
            raise SystemExit(f"Doublons : {path}")

        if part_ids & existing:
            raise SystemExit(f"Chevauchement : {path}")

        existing.update(part_ids)

    if len(existing) != 520 or len(reserved) != 183:
        raise SystemExit("Population historique inattendue")

    if set(ids) & existing:
        raise SystemExit(
            "FETCH83 chevauche une population historique"
        )

    for mid in ids:
        original.safe_match_id(mid)
        cache = original.raw_path(ROOT, mid)

        if cache.parent.resolve() != CACHE.resolve():
            raise SystemExit(
                f"Chemin cache inattendu : {cache}"
            )

    return rows


def find_cache(mid):
    choices = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]
    return next(
        (path for path in choices if path.is_file()),
        None,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit", type=int, default=3
    )
    parser.add_argument(
        "--max-requests", type=int, default=5
    )
    parser.add_argument(
        "--delay", type=float, default=3.0
    )
    parser.add_argument(
        "--execute", action="store_true"
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        parser.error("--limit doit être entre 1 et 20")

    if not 1 <= args.max_requests <= 30:
        parser.error(
            "--max-requests doit être entre 1 et 30"
        )

    if args.delay < 1:
        parser.error("--delay doit être >= 1")

    rows = load_and_verify()

    pending = [
        r for r in rows
        if find_cache(r["match_id"]) is None
    ]

    print("=" * 72)
    print("ÉTAPE 199 — COLLECTE SÉCURISÉE V4 FETCH83")
    print("=" * 72)

    print("\n=== INVENTAIRE ===")
    print("Matchs du manifeste :", len(rows))
    print("Caches déjà présents :", len(rows)-len(pending))
    print("Matchs restant à récupérer :", len(pending))

    print("\n=== PARAMÈTRES ===")
    print("Limite de nouveaux matchs :", args.limit)
    print("Budget requêtes :", args.max_requests)
    print("Délai :", args.delay, "secondes")

    if not args.execute:
        print("\n=== SIMULATION ===")

        for row in pending[:args.limit]:
            print(
                "WOULD_FETCH",
                row["match_id"],
                row["kickoff_utc"],
            )

        print("\nAppels API : 0")
        print("Exécution réelle : NON")
        print("Aucun fichier modifié")
        return

    original.load_dot_env(Path(".env"))

    key = (os.environ.get("STATS_API_KEY") or "").strip()

    if key.lower().startswith("bearer "):
        key = key[7:].strip()

    if not key:
        raise SystemExit("STATS_API_KEY manquante")

    base = (
        os.environ.get("STATS_API_BASE_URL")
        or "https://api.thestatsapi.com/api"
    ).rstrip("/")

    if not base.startswith("https://"):
        raise SystemExit("HTTPS obligatoire")

    requests = 0
    success = 0
    last_request = None
    stop_reason = ""

    print("\n=== COLLECTE RÉELLE ===")

    for row in pending:
        if success >= args.limit:
            break

        mid = original.safe_match_id(row["match_id"])

        if find_cache(mid) is not None:
            print("SKIP_CACHE", mid)
            continue

        attempt = 0

        while True:
            if requests >= args.max_requests:
                stop_reason = "BUDGET_REQUÊTES_ATTEINT"
                break

            if last_request is not None:
                remaining = (
                    args.delay
                    - (time.monotonic() - last_request)
                )

                if remaining > 0:
                    time.sleep(remaining)

            requests += 1
            last_request = time.monotonic()

            try:
                payload = original.api_get(
                    mid, key, base, 20
                )

                safe.validate_payload(payload, mid)

                if find_cache(mid) is not None:
                    print("SKIP_ALREADY_CREATED", mid)
                    break

                cache = original.raw_path(ROOT, mid)

                try:
                    safe.save_new_cache(cache, payload)
                except FileExistsError:
                    print("SKIP_ALREADY_CREATED", mid)
                    break

                success += 1

                log_result(mid, "OK_FULL")

                print(
                    "OK_FULL",
                    mid,
                    "| événements :",
                    len(payload["data"]["events"]),
                )

                break

            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    stop_reason = f"HTTP_{exc.code}"
                    log_result(mid, stop_reason)
                    break

                if attempt >= 2:
                    stop_reason = "HTTP_429_RETRIES_EXHAUSTED"
                    log_result(mid, stop_reason)
                    break

                delay = safe.retry_after_seconds(
                    exc.headers, attempt
                )

                if delay > 60:
                    stop_reason = "HTTP_429_LONG_RETRY_AFTER"
                    log_result(mid, stop_reason)
                    break

                if requests >= args.max_requests:
                    stop_reason = "BUDGET_REQUÊTES_ATTEINT"
                    break

                print(
                    "HTTP_429",
                    mid,
                    "| attente :",
                    round(delay, 1),
                    "secondes",
                )

                time.sleep(delay)
                attempt += 1

            except Exception as exc:
                stop_reason = type(exc).__name__
                log_result(
                    mid,
                    stop_reason,
                    str(exc)[:150],
                )

                print("ARRÊT", mid, stop_reason)
                break

        if stop_reason:
            break

    remaining = sum(
        find_cache(r["match_id"]) is None
        for r in rows
    )

    print("\n=== BILAN ===")
    print("Requêtes HTTP :", requests)
    print("Nouveaux caches FULL :", success)
    print("Matchs encore sans cache :", remaining)
    print("Motif d'arrêt :", stop_reason or "NORMAL")
    print("Journal :", LOG)

    print("\nScanner V3-C inchangé.")
    print("VALIDATION et TEST_FUTUR préservés.")
    print("Aucun entraînement effectué.")

    if stop_reason:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
