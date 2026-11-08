import argparse
import csv
import json
import os
import tempfile
import time
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import rebuild_historical_pressure as original

PLAN = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "HISTORICAL_WAVE1_BATCH02_MANIFEST_V1.csv"
)

OUTDIR = Path("outputs/pressure_historical")
LOG = OUTDIR / "batch02_safe_fetch_log_v1.jsonl"


def retry_after_seconds(headers, attempt):
    raw = (headers.get("Retry-After") or "").strip()
    fallback = min(5 * (2 ** attempt), 60)

    if not raw:
        return fallback

    try:
        return max(0, float(raw))
    except ValueError:
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(
                0,
                (dt - datetime.now(timezone.utc)).total_seconds()
            )
        except (TypeError, ValueError):
            return fallback


def validate_payload(payload, mid):
    if not isinstance(payload, dict):
        raise ValueError("JSON principal invalide")

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError("data invalide")

    if str(data.get("match_id")) != mid:
        raise ValueError("match_id différent")

    if data.get("coverage") != "full":
        raise ValueError("coverage non full")

    events = data.get("events")

    if not isinstance(events, list) or not events:
        raise ValueError("Timeline vide ou invalide")


def save_new_cache(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Écriture dans le même dossier : création atomique,
    # sans remplacement d'un cache existant.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".pilot_",
            suffix=".tmp",
            delete=False
        ) as f:
            temp_path = Path(f.name)
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.link(temp_path, path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def log_result(mid, status):
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "match_id": mid,
            "status": status,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat()
        }) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if not (1 <= args.limit <= 20):
        parser.error("limit doit être entre 1 et 20")

    if not (1 <= args.max_requests <= 30):
        parser.error("max-requests doit être entre 1 et 30")

    if args.delay < 1:
        parser.error("delay doit être >= 1 seconde")

    with PLAN.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    ids = [r["match_id"] for r in rows]

    if len(rows) != 20 or len(set(ids)) != 20:
        raise ValueError("Manifeste pilote incorrect")

    if not args.execute:
        print("=== SIMULATION SÉCURISÉE ===")

        for row in rows[:args.limit]:
            print("WOULD_FETCH", row["match_id"])

        print("Appels API : 0")
        print("Exécution réelle : NON")
        return

    original.load_dot_env(Path(".env"))

    key = (os.environ.get("STATS_API_KEY") or "").strip()

    if key.lower().startswith("bearer "):
        key = key[7:].strip()

    if not key:
        raise ValueError("STATS_API_KEY manquante")

    base = (
        os.environ.get("STATS_API_BASE_URL")
        or "https://api.thestatsapi.com/api"
    ).rstrip("/")

    if not base.startswith("https://"):
        raise ValueError("HTTPS obligatoire")

    requests = 0
    success = 0
    skipped = 0
    last_request = None
    stopped_reason = ""

    print("\n=== COLLECTE RÉELLE LIMITÉE ===")

    for row in rows:
        if success + skipped >= args.limit:
            break

        mid = original.safe_match_id(row["match_id"])
        cache = original.raw_path(OUTDIR, mid)

        if cache.exists():
            skipped += 1
            print("SKIP_CACHE", mid)
            continue

        attempt = 0

        while True:
            if requests >= args.max_requests:
                stopped_reason = "BUDGET_REQUÊTES_ATTEINT"
                break

            if last_request is not None:
                remaining = (
                    args.delay - (time.monotonic() - last_request)
                )
                if remaining > 0:
                    time.sleep(remaining)

            requests += 1
            last_request = time.monotonic()

            try:
                payload = original.api_get(
                    mid, key, base, 20
                )

                validate_payload(payload, mid)

                try:
                    save_new_cache(cache, payload)
                    success += 1
                    log_result(mid, "OK_FULL")
                    print("OK_FULL", mid, "| events:",
                          len(payload["data"]["events"]))
                except FileExistsError:
                    skipped += 1
                    print("SKIP_ALREADY_CREATED", mid)

                break

            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    stopped_reason = f"HTTP_{exc.code}"
                    log_result(mid, stopped_reason)
                    break

                if attempt >= 2:
                    stopped_reason = "HTTP_429_RETRIES_EXHAUSTED"
                    log_result(mid, stopped_reason)
                    break

                wait = retry_after_seconds(
                    exc.headers, attempt
                )

                if wait > 60:
                    stopped_reason = (
                        "HTTP_429_LONG_RETRY_AFTER"
                    )
                    log_result(mid, stopped_reason)
                    break

                if requests >= args.max_requests:
                    stopped_reason = "BUDGET_REQUÊTES_ATTEINT"
                    break

                print(
                    "HTTP_429", mid,
                    "| attente:", round(wait, 1), "s"
                )

                time.sleep(wait)
                attempt += 1

            except Exception as exc:
                stopped_reason = type(exc).__name__
                log_result(mid, stopped_reason)
                print(
                    "ARRÊT", mid,
                    "|", stopped_reason
                )
                break

        if stopped_reason:
            break

    print("\n=== BILAN ===")
    print("Requêtes HTTP :", requests)
    print("Timelines FULL enregistrées :", success)
    print("Caches ignorés :", skipped)
    print("Motif d'arrêt :", stopped_reason or "NORMAL")
    print("Log :", LOG)

    if stopped_reason:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
