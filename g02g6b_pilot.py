#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

B = Path("outputs/v4_research")
PLAN = B/"v42_g02g6_prematch_pilot/G02G6_PILOT6_TEAMS_FROZEN.csv"
TARGETS = B/"v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv"
TEAMS = B/"v42_g02g5_prematch_plan/G02G5_TEAMS_FROZEN.csv"
MANIFEST = B/"v41_mass350/manifest.csv"
RAW = B/"v42_g02g6_prematch_pilot/raw_history"

HASHES = {
    PLAN: "a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0",
    TARGETS: "4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2",
    TEAMS: "1b31c5763e6aa2c0fd27d1699d83c6ce1db72a4a0431d6d5cd4a7ce16aff0bb1",
    MANIFEST: "61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e",
}

PILOT = {
    "tm_37445": 2,
    "tm_087341": 2,
    "tm_279303": 2,
    "tm_46516": 4,
    "tm_19704": 6,
    "tm_6047": 6,
}

def stop(message):
    raise SystemExit("STOP : " + message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def cache(tid):
    path = RAW/(tid + ".json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("team_id") != tid
        or data.get("endpoint") != "/football/matches"
        or data.get("params", {}).get("team_id") != tid
        or not isinstance(data.get("response"), (dict, list))
    ):
        stop("cache invalide : " + str(path))
    return data

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, choices=[0, 6], default=0)
args = parser.parse_args()

print("=" * 76)
print("G02-G6B | PILOTE PREMATCH")
print("Budget API :", args.budget)
print("=" * 76)

for path, expected in HASHES.items():
    if not path.is_file() or digest(path) != expected:
        stop("fichier absent ou SHA256 different : " + str(path))
    print("SHA OK :", path)

selected = rows(PLAN)
observations = rows(TARGETS)
teams = rows(TEAMS)
manifest = rows(MANIFEST)

selected_ids = [r["team_id"] for r in selected]

if (
    len(selected) != 6
    or len(set(selected_ids)) != 6
    or set(selected_ids) != set(PILOT)
):
    stop("selection pilote differente des 6 equipes figees")

if (
    len(observations) != 3000
    or len(teams) != 1116
    or len(manifest) != 12415
):
    stop("effectifs des fichiers geles invalides")

if len({r["team_id"] for r in teams}) != 1116:
    stop("IDs equipe dupliques")

manifest_map = {r["match_id"]: r for r in manifest}

if len(manifest_map) != 12415:
    stop("IDs match dupliques")

grouped = {tid: [] for tid in PILOT}

for obs in observations:
    tid = obs["team_id"]
    if tid not in grouped:
        continue

    mid = obs["match_id"]
    side = obs["side"]

    if mid not in manifest_map or side not in ("home", "away"):
        stop("observation invalide : " + repr(obs))

    if manifest_map[mid][side + "_id"] != tid:
        stop("ID equipe incoherent : " + mid)

    cutoff = datetime.fromisoformat(
        obs["safe_cutoff_utc"].replace("Z", "+00:00")
    )

    if cutoff.tzinfo is None:
        stop("cutoff sans fuseau horaire : " + mid)

    grouped[tid].append(obs)

if {tid: len(v) for tid, v in grouped.items()} != PILOT:
    stop("les 22 observations ne correspondent pas au plan fige")

keys = [
    (obs["match_id"], obs["side"])
    for group in grouped.values()
    for obs in group
]

if len(set(keys)) != 22:
    stop("observations pilote dupliquees")

missing = []

print("\n=== PLAN DES 6 EQUIPES ===")

for tid, count in PILOT.items():
    present = cache(tid) is not None
    print(
        tid,
        "| observations =", count,
        "| cache =", "OUI" if present else "NON"
    )
    if not present:
        missing.append(tid)

print("\nObservations :", sum(PILOT.values()))
print("Appels API encore necessaires :", len(missing))

if args.budget == 0:
    print("\nG02-G6B PLAN OK")
    print("ZERO APPEL API")
    print("ZERO MODIFICATION DES DONNEES")
    print("AUCUN ENTRAINEMENT")
    raise SystemExit(0)

import rebuild_clean_prematch_v2 as pre

lab = pre.load_lab()
client = pre.build_rate_limited_client(
    lab, max_retries=1, min_interval=2.5
)

calls = 0

for tid in missing:
    if calls >= args.budget:
        break

    params = {
        "team_id": tid,
        "status": "finished",
        "per_page": 100,
    }

    calls += 1
    print("\nAPPEL", calls, "/", args.budget, "|", tid)

    try:
        response = client.get(
            "/football/matches", params=params
        )

        if not isinstance(response, (dict, list)):
            raise ValueError("type de reponse inattendu")

        if isinstance(response, dict) and response.get("error"):
            raise ValueError(
                "erreur API : " + repr(response["error"])
            )

        page = pre.api_list(lab, response)

        if not isinstance(page, list):
            raise ValueError("reponse API non exploitable")

        record = {
            "team_id": tid,
            "endpoint": "/football/matches",
            "params": params,
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "RAW_NOT_CERTIFIED",
            "response": response,
        }

        RAW.mkdir(parents=True, exist_ok=True)

        final = RAW/(tid + ".json")
        temp = RAW/(tid + ".json.tmp")

        if final.exists() or temp.exists():
            stop("refus ecrasement : " + str(final))

        temp.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp.replace(final)

        print(
            "CACHE SAUVE :", final,
            "| matchs :", len(page),
            "| SHA256 :", digest(final)
        )

    except Exception as exc:
        print(
            "ARRET :",
            type(exc).__name__,
            str(exc)
        )
        break

print("\n=== INSPECTION DES CACHES ===")

for tid in PILOT:
    data = cache(tid)

    if data is None:
        print(tid, ": cache absent")
        continue

    response = data["response"]

    try:
        page = pre.api_list(lab, response)
        dates = sorted(
            str(pre.match_datetime(lab, match))
            for match in page
            if pre.match_datetime(lab, match) is not None
        )

        print(
            tid,
            "| matchs :", len(page),
            "| plus ancien :", dates[0] if dates else "?",
            "| plus recent :", dates[-1] if dates else "?"
        )

        if len(page) >= 100:
            print(
                "ALERTE : page limitee a 100 matchs,"
                " pagination a verifier"
            )

        for obs in sorted(
            grouped[tid],
            key=lambda x: (
                x["safe_cutoff_utc"],
                x["match_id"]
            ),
        ):
            mid = obs["match_id"]
            side = obs["side"]

            cutoff = datetime.fromisoformat(
                obs["safe_cutoff_utc"].replace(
                    "Z", "+00:00"
                )
            ).astimezone(timezone.utc).replace(
                tzinfo=None
            )

            team_name = manifest_map[mid][
                side + "_team"
            ]

            try:
                prior = pre.filter_before_cutoff(
                    lab,
                    page,
                    cutoff,
                    target_match_id=mid,
                )

                summary = pre.build_team_summary(
                    lab, prior, tid, team_name
                )

                print(
                    mid, side,
                    "| historiques :", len(prior),
                    "| moyenne exploratoire :",
                    summary.get("avg_goals_scored")
                )

            except Exception as exc:
                print(
                    mid,
                    "| calcul exploratoire impossible :",
                    type(exc).__name__,
                    str(exc)[:100]
                )

        if isinstance(response, dict):
            for key in ("meta", "pagination", "links"):
                if key in response:
                    print(
                        key, ":",
                        json.dumps(
                            response[key],
                            default=str,
                            ensure_ascii=False
                        )[:800]
                    )

    except Exception as exc:
        print(
            tid,
            "| inspection impossible :",
            type(exc).__name__,
            str(exc)
        )

print("\nG02-G6B TERMINE")
print("Appels API effectues :", calls)
print("DONNEES BRUTES NON CERTIFIEES")
print("AUCUN ENTRAINEMENT")
