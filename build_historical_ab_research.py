import csv
import json
import math
import re
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/pressure_historical")
CACHE = ROOT / "timeline_cache"
JOIN = ROOT / "pressure_v2_dataset_join"

MANIFEST = Path("historical_1710_manifest.csv")
PLAN = ROOT / "HISTORICAL_BALANCED_FETCH_PLAN.csv"

OUT_A = JOIN / "NEW_HISTORICAL_A_RESEARCH.csv"
OUT_B = JOIN / "NEW_HISTORICAL_B_RESEARCH.csv"
OUT_Q = JOIN / "NEW_HISTORICAL_QUALITY_AUDIT.csv"

REGULATION = {"first_half", "second_half"}

EXTRA_TIME = {
    "extra_time_first_half",
    "extra_time_second_half"
}

SHOOTOUT = {
    "penalties",
    "shootout",
    "penalty_shootout",
    "penalty_shoot-out"
}

GOALS = {
    "goal",
    "own_goal",
    "penalty_scored"
}

SHOTS = {
    "shot_on_target",
    "shot_off_target",
    "shot_blocked",
    "shot_woodwork",
    "goal",
    "penalty_scored",
    "penalty_saved",
    "penalty_missed"
}

SOT = {
    "shot_on_target",
    "goal",
    "penalty_scored",
    "penalty_saved"
}

A_FEATURES = [
    "score_87_home",
    "score_87_away",
    "goal_diff_87",
    "is_draw_87",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",
    "home_corners_87",
    "away_corners_87",
    "home_fouls_87",
    "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87",
    "away_red_cards_87",
    "home_offsides_87",
    "away_offsides_87",
    "goal_diff_signed_87"
]

B_EXTRA = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5"
]

META = [
    "match_id",
    "kickoff_utc",
    "goal_after_87",
    "goal_at_87",
    "extra_time_events_excluded",
    "extra_time_goals_excluded",
    "score_independently_verified",
    "live_data_certified",
    "training_certified",
    "research_status"
]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def norm(value):
    return " ".join(
        str(value or "").casefold().split()
    )

def match_side(event, info):
    team = event.get("team") or {}

    if isinstance(team, dict):
        tid = str(team.get("id") or "")
        name = norm(team.get("name"))
    else:
        tid = ""
        name = norm(team)

    home_id = str(info.get("home_id") or "")
    away_id = str(info.get("away_id") or "")

    if tid and tid == home_id:
        return "home"

    if tid and tid == away_id:
        return "away"

    if name and name == norm(info.get("home_team")):
        return "home"

    if name and name == norm(info.get("away_team")):
        return "away"

    return None

def event_minute(event):
    value = event.get("minute")

    if value is None:
        raise ValueError("Minute absente")

    minute = float(value)

    if not math.isfinite(minute) or minute < 0:
        raise ValueError("Minute invalide")

    return minute

def build_match(mid, info, data):
    issues = []
    stats = Counter()

    if data.get("match_id") != mid:
        issues.append("MATCH_ID_MISMATCH")

    if data.get("coverage") != "full":
        issues.append("COVERAGE_NOT_FULL")

    events = data.get("events")

    if not isinstance(events, list) or not events:
        return None, None, {
            "match_id": mid,
            "status": "QUARANTINE",
            "issues": "EMPTY_OR_INVALID_TIMELINE"
        }

    goal_at_87 = 0
    late_goals = 0
    extra_time_goals = 0
    extra_time_events = 0

    for event in events:
        if not isinstance(event, dict):
            issues.append("INVALID_EVENT")
            continue

        typ = str(event.get("type") or "").lower()
        period = str(event.get("period") or "").lower()

        if period in EXTRA_TIME:
            extra_time_events += 1

            if typ in GOALS:
                extra_time_goals += 1

            continue

        if period in SHOOTOUT:
            continue

        if period not in REGULATION:
            issues.append(f"UNKNOWN_PERIOD:{period}")
            continue

        relevant = (
            typ in SHOTS
            or typ in GOALS
            or typ in {
                "corner_kick",
                "foul",
                "offside",
                "yellow_card",
                "red_card",
                "yellow_red_card"
            }
        )

        if not relevant:
            continue

        try:
            minute = event_minute(event)
        except (ValueError, TypeError):
            issues.append("INVALID_MINUTE")
            continue

        # Label strictement après 87 minutes.
        # Seulement en temps réglementaire.
        if typ in GOALS and period == "second_half":
            if minute == 87:
                goal_at_87 += 1
            elif minute > 87:
                late_goals += 1

        # Les features excluent la minute 87
        # et tous les événements ultérieurs.
        if minute >= 87:
            continue

        side = match_side(event, info)

        if side is None:
            issues.append(f"UNKNOWN_TEAM:{typ}")
            continue

        if typ in GOALS:
            # Convention expérimentale :
            # team = bénéficiaire du but.
            # À vérifier avec une source indépendante,
            # particulièrement pour own_goal.
            stats[f"score_{side}"] += 1

        if typ in SHOTS:
            stats[f"{side}_shots_87"] += 1

        if typ in SOT:
            stats[f"{side}_shots_on_target_87"] += 1

        if typ == "shot_off_target":
            stats[f"{side}_shots_off_target_87"] += 1

        if typ == "shot_blocked":
            stats[f"{side}_shots_blocked_87"] += 1

        if typ == "corner_kick":
            # Convention historique constatée :
            # attribution inversée de team.
            credited = (
                "away" if side == "home"
                else "home"
            )
            stats[f"{credited}_corners_87"] += 1

        if typ == "foul":
            stats[f"{side}_fouls_87"] += 1

        if typ == "offside":
            stats[f"{side}_offsides_87"] += 1

        if typ in {"yellow_card", "yellow_red_card"}:
            stats[f"{side}_yellow_cards_87"] += 1

        if typ in {"red_card", "yellow_red_card"}:
            stats[f"{side}_red_cards_87"] += 1

        if period == "second_half" and 82 <= minute < 87:
            if typ in SHOTS:
                stats[f"shots_{side}_last5"] += 1

            if typ in SOT:
                stats[f"sot_{side}_last5"] += 1

            if typ == "corner_kick":
                credited = (
                    "away" if side == "home"
                    else "home"
                )
                stats[f"corners_{credited}_last5"] += 1

    home = stats["score_home"]
    away = stats["score_away"]

    stats["score_87_home"] = home
    stats["score_87_away"] = away
    stats["goal_diff_87"] = abs(home - away)
    stats["goal_diff_signed_87"] = home - away
    stats["is_draw_87"] = int(home == away)

    for side in ("home", "away"):
        shots = stats[f"{side}_shots_87"]
        sot = stats[f"{side}_shots_on_target_87"]
        off = stats[f"{side}_shots_off_target_87"]
        blocked = stats[f"{side}_shots_blocked_87"]

        if sot > shots:
            issues.append(f"SOT_EXCEEDS_SHOTS:{side}")

        if off + blocked > shots:
            issues.append(f"DETAILS_EXCEED_SHOTS:{side}")

        if stats[f"shots_{side}_last5"] > shots:
            issues.append(f"LAST5_EXCEEDS_TOTAL:{side}")

        if stats[f"sot_{side}_last5"] > sot:
            issues.append(f"LAST5_SOT_EXCEEDS_TOTAL:{side}")

    issues = sorted(set(issues))

    status = (
        "QUARANTINE"
        if issues
        else "RESEARCH_ONLY"
    )

    common = {
        "match_id": mid,
        "kickoff_utc": info["kickoff_utc"],
        "goal_after_87": int(late_goals > 0),
        "goal_at_87": int(goal_at_87 > 0),
        "extra_time_events_excluded": extra_time_events,
        "extra_time_goals_excluded": extra_time_goals,
        "score_independently_verified": 0,
        "live_data_certified": 0,
        "training_certified": 0,
        "research_status": status
    }

    row_a = {
        **common,
        **{col: stats[col] for col in A_FEATURES}
    }

    row_b = {
        **row_a,
        **{col: stats[col] for col in B_EXTRA}
    }

    quality = {
        "match_id": mid,
        "status": status,
        "issues": ";".join(issues),
        "goal_after_87": common["goal_after_87"],
        "goal_at_87": common["goal_at_87"],
        "extra_time_events_excluded": extra_time_events,
        "extra_time_goals_excluded": extra_time_goals,
        "score_independently_verified": 0,
        "training_certified": 0
    }

    return row_a, row_b, quality

def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def main():
    JOIN.mkdir(parents=True, exist_ok=True)

    for path in (OUT_A, OUT_B, OUT_Q):
        if path.exists():
            raise FileExistsError(
                f"Fichier déjà présent : {path}"
            )

    manifest_rows = read_csv(MANIFEST)
    manifest = {
        r["match_id"]: r for r in manifest_rows
    }

    plan_rows = read_csv(PLAN)

    paths_by_id = {}

    for path in CACHE.rglob("*.json"):
        for mid in re.findall(r"mt_\d+", path.name):
            paths_by_id.setdefault(mid, []).append(path)

    rows_a = []
    rows_b = []
    quality_rows = []
    counts = Counter()

    for item in plan_rows:
        mid = item["match_id"]
        paths = paths_by_id.get(mid, [])

        if not paths:
            continue

        if len(paths) != 1:
            raise ValueError(
                f"Cache ambigu : {mid}, {paths}"
            )

        if mid not in manifest:
            raise ValueError(
                f"Match absent du manifeste : {mid}"
            )

        payload = json.loads(
            paths[0].read_text(encoding="utf-8")
        )

        data = payload.get("data")

        if not isinstance(data, dict):
            raise ValueError(
                f"Payload invalide : {mid}"
            )

        a, b, quality = build_match(
            mid, manifest[mid], data
        )

        quality_rows.append(quality)
        counts[quality["status"]] += 1
        counts["TIMELINES"] += 1

        if a is not None:
            rows_a.append(a)
            rows_b.append(b)
            counts["POSITIVES"] += a["goal_after_87"]
            counts["EXTRA_TIME_EVENTS"] += (
                a["extra_time_events_excluded"]
            )

        if quality["status"] != "RESEARCH_ONLY":
            print(
                "À EXAMINER :",
                mid,
                quality["issues"]
            )

    rows_a.sort(
        key=lambda r: (r["kickoff_utc"], r["match_id"])
    )

    rows_b.sort(
        key=lambda r: (r["kickoff_utc"], r["match_id"])
    )

    if len(rows_a) != len(rows_b):
        raise ValueError("A/B : nombres de lignes différents")

    for a, b in zip(rows_a, rows_b):
        if a["match_id"] != b["match_id"]:
            raise ValueError("Ordre A/B incohérent")

        for col in A_FEATURES:
            if a[col] != b[col]:
                raise ValueError(
                    f"A/B différent : {a['match_id']} {col}"
                )

    write_csv(
        OUT_A,
        META + A_FEATURES,
        rows_a
    )

    write_csv(
        OUT_B,
        META + A_FEATURES + B_EXTRA,
        rows_b
    )

    write_csv(
        OUT_Q,
        [
            "match_id", "status", "issues",
            "goal_after_87", "goal_at_87",
            "extra_time_events_excluded",
            "extra_time_goals_excluded",
            "score_independently_verified",
            "training_certified"
        ],
        quality_rows
    )

    print("\n=== ÉTAPE 58 : PIPELINE A/B ===")
    print("Timelines traitées :", counts["TIMELINES"])
    print("Statut RESEARCH_ONLY :", counts["RESEARCH_ONLY"])
    print("Quarantaine :", counts["QUARANTINE"])
    print("Lignes dans A :", len(rows_a))
    print("Lignes dans B :", len(rows_b))
    print("Labels positifs provisoires :", counts["POSITIVES"])
    print("Variables A :", len(A_FEATURES))
    print("Variables B :", len(A_FEATURES) + len(B_EXTRA))
    print(
        "Événements de prolongation exclus :",
        counts["EXTRA_TIME_EVENTS"]
    )

    print("\nFICHIERS CRÉÉS :")
    print(OUT_A)
    print(OUT_B)
    print(OUT_Q)

    print("\nAucune requête API.")
    print("Anciens datasets inchangés.")
    print("Aucun entraînement lancé.")
    print("Scores non vérifiés indépendamment.")
    print("Données LIVE non certifiées.")

if __name__ == "__main__":
    main()
