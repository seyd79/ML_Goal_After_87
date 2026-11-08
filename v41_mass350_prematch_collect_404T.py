import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import rebuild_clean_prematch_v2 as pre

BASE = Path("outputs/v4_research")
RAW = BASE / "v41_mass350/raw_timelines"
MATRIX = BASE / "v41_mass350_blind_features_404K/V41_MASS350_FEATURES_BLIND_PREPARATION.csv"
PILOT = BASE / "v41_prematch_pilot_404S/V41_PREMATCH_PILOT20_NO_TARGET_LABEL.csv"
OUT = BASE / "v41_prematch_collection_404T"
CHECKPOINTS = OUT / "checkpoints"

EXPECTED_MATRIX = "0b1d4155b5dd70655b64721c4ca70524eb545f5e0bacdadac95a1f0ed7278074"
EXPECTED_PILOT = "47507f812a52d9054f5988e5f03fc71c18afd1a28810235cc9280da9225e1b51"

FIELDS = [
    "match_id", "team_id", "side", "kickoff_utc",
    "history_raw_count", "history_before_count",
    "avg_goals_scored", "method",
]

LIVE = [
    "score_87_home", "score_87_away",
    "shots_80_87", "sot_80_87",
    "corners_80_87", "home_shots_on_target_87",
    "away_shots_on_target_87",
    "minutes_since_last_shot_on_target",
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def checkpoint_path(tid):
    if not tid.startswith("tm_") or not tid[3:].isdigit():
        raise RuntimeError("STOP : ID equipe incorrect")
    return CHECKPOINTS / (tid + ".csv")

def validate_records(tid, records, expected):
    if len(records) != len(expected):
        raise RuntimeError("STOP : nombre de lignes incorrect " + tid)
    if len({(r["match_id"], r["side"]) for r in records}) != len(records):
        raise RuntimeError("STOP : doublons " + tid)

    actual_keys = {(r["match_id"], r["side"]) for r in records}
    expected_keys = {(mid, side) for mid, side, _, _ in expected}
    if actual_keys != expected_keys:
        raise RuntimeError("STOP : couverture incorrecte " + tid)

    for r in records:
        if set(r) != set(FIELDS):
            raise RuntimeError("STOP : schema incorrect")
        if r["team_id"] != tid:
            raise RuntimeError("STOP : equipe incorrecte")
        if r["method"] != "ORIGINAL_SINGLE_PAGE_100_ASOF":
            raise RuntimeError("STOP : methode incorrecte")
        if r["avg_goals_scored"] != "" and not valid_number(r["avg_goals_scored"]):
            raise RuntimeError("STOP : moyenne invalide")
        if not 0 <= int(r["history_before_count"]) <= int(r["history_raw_count"]) <= 100:
            raise RuntimeError("STOP : historique incoherent")

def write_new(path, records):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 50:
        raise SystemExit("STOP : limite autorisee 1 a 50")

    if sha(MATRIX) != EXPECTED_MATRIX:
        raise SystemExit("STOP : matrice 404K modifiee")
    if sha(PILOT) != EXPECTED_PILOT:
        raise SystemExit("STOP : pilote 404S modifie")

    matrix_rows = read(MATRIX)
    matrix = {r["match_id"]: r for r in matrix_rows}
    if len(matrix) != 350 or len(matrix_rows) != 350:
        raise SystemExit("STOP : matrice invalide")

    jobs = defaultdict(list)

    files = sorted(RAW.glob("mt_*.json"))
    if len(files) != 350:
        raise SystemExit("STOP : timelines incompletes")

    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        mid = path.stem
        if obj.get("match_id") != mid or mid not in matrix:
            raise SystemExit("STOP : identifiant timeline incorrect")

        row = matrix[mid]
        if not all(str(row.get(k, "")).strip() for k in LIVE):
            continue

        for side in ("home", "away"):
            field = side + "_avg_goals_scored"
            if str(row.get(field, "")).strip():
                continue

            tid = str(obj[side + "_id"])
            jobs[tid].append((
                mid, side, obj["kickoff_utc"], obj[side + "_team"]
            ))

    if len(jobs) != 510:
        raise SystemExit("STOP : 510 equipes attendues")

    if len({mid for j in jobs.values() for mid, _, _, _ in j}) != 338:
        raise SystemExit("STOP : 338 matchs attendus")

    pilot = read(PILOT)
    pilot_by_team = defaultdict(list)
    for r in pilot:
        pilot_by_team[r["team_id"]].append(r)

    for tid, records in pilot_by_team.items():
        validate_records(tid, records, jobs[tid])

    print("=" * 72)
    print("404T — COLLECTE PREMATCH REPRENABLE")
    print("=" * 72)
    print("Equipes requises :", len(jobs))
    print("Pilote existant :", len(pilot_by_team), "equipes")

    # Le mode status ne cree aucun dossier et ne contacte pas l'API.
    if not args.status:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    completed = set()

    for tid in sorted(jobs):
        path = checkpoint_path(tid)
        if path.exists():
            validate_records(tid, read(path), jobs[tid])
            completed.add(tid)
        elif tid in pilot_by_team:
            completed.add(tid)

    remaining = sorted(set(jobs) - completed)

    print("Equipes terminees :", len(completed))
    print("Equipes restantes :", len(remaining))

    if args.status:
        print("Mode STATUS : zero appel API, zero ecriture")
        return

    if not remaining:
        print("COLLECTE TERMINEE")
        return

    lab = pre.load_lab()
    client = pre.build_rate_limited_client(
        lab, max_retries=1, min_interval=1.0
    )

    processed = 0

    for tid in remaining[:args.limit]:
        print("\nEQUIPE :", tid, flush=True)

        payload = client.get(
            "/football/matches",
            params={
                "team_id": tid,
                "status": "finished",
                "per_page": 100,
            },
        )

        if not isinstance(payload, dict):
            print("STOP : reponse API invalide", tid)
            break

        history = pre.api_list(lab, payload)

        if not history or len(history) > 100:
            print("STOP : historique vide ou invalide", tid)
            break

        ids = [str(lab.extract_match_id(m)) for m in history]
        if len(ids) != len(set(ids)):
            print("STOP : doublons API", tid)
            break

        results = []
        failed = False

        for mid, side, date, team_name in sorted(
            jobs[tid], key=lambda x: (x[2], x[0])
        ):
            cutoff = pre.dt_utc_naive(date)
            if cutoff is None:
                failed = True
                break

            before = pre.filter_before_cutoff(
                lab, history, cutoff, target_match_id=mid
            )

            summary = pre.build_team_summary(
                lab, before, tid, team_name
            )
            avg = summary.get("avg_goals_scored")

            if not valid_number(avg):
                print("MANQUANT : moyenne indisponible", tid, mid)
                avg = ""

            results.append({
                "match_id": mid,
                "team_id": tid,
                "side": side,
                "kickoff_utc": date,
                "history_raw_count": len(history),
                "history_before_count": len(before),
                "avg_goals_scored": float(avg) if avg != "" else "",
                "method": "ORIGINAL_SINGLE_PAGE_100_ASOF",
            })

        if failed:
            break

        validate_records(tid, results, jobs[tid])
        write_new(checkpoint_path(tid), results)

        processed += 1
        print(
            "SAUVEGARDE :", len(results),
            "observations |", tid,
            flush=True
        )

    print("\n=== BILAN ===")
    print("Nouvelles equipes :", processed)
    print("Requetes API :", client.request_count)
    print("HTTP 429 :", client.retry_429_count)
    print("Equipes encore a traiter :", len(remaining)-processed)
    print("Aucun label cible consulte.")
    print("Aucune prediction calculee.")
    print("Mode : RECHERCHE EXPLORATOIRE UNIQUEMENT")

if __name__ == "__main__":
    main()
