from pathlib import Path
import json
import re
import unicodedata
from collections import defaultdict

import pandas as pd

ROOT = Path("outputs/pressure_historical")
CACHE = ROOT / "timeline_cache"
OUT = Path("outputs/v4_research")

MANIFEST = OUT / "V4_CACHED12_MANIFEST_V1.csv"
AUDIT = OUT / "V4_CACHED_CANDIDATES_AUDIT_V1.csv"
DEST = OUT / "V4_CACHED12_B6_LABELS_RESEARCH_V1.csv"

SHOTS = {
    "shot_on_target",
    "shot_off_target",
    "shot_blocked",
    "shot_woodwork",
    "goal",
    "penalty_scored",
    "penalty_saved",
    "penalty_missed",
}

SOT = {
    "shot_on_target",
    "goal",
    "penalty_scored",
    "penalty_saved",
}

GOALS = {
    "goal",
    "own_goal",
    "penalty_scored",
}

def normalize(value):
    if value is None or pd.isna(value):
        return ""

    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(
        c for c in s if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", s.casefold())

def read_cache(mid):
    choices = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]

    path = next((p for p in choices if p.is_file()), None)

    if path is None:
        raise ValueError(f"Cache absent : {mid}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)

    if not isinstance(payload, dict):
        raise ValueError(f"Payload invalide : {mid}")

    if str(payload.get("match_id")) != mid:
        raise ValueError(f"ID différent : {mid}")

    if payload.get("coverage") != "full":
        raise ValueError(f"Couverture incomplète : {mid}")

    events = payload.get("events")

    if not isinstance(events, list) or not events:
        raise ValueError(f"Timeline invalide : {mid}")

    if not any(
        isinstance(e, dict)
        and e.get("period") == "second_half"
        and e.get("type") == "period_end"
        for e in events
    ):
        raise ValueError(f"Fin de seconde période absente : {mid}")

    return events

def event_minute(event):
    try:
        minute = float(event["minute"])
        extra = float(event.get("extra_time", 0) or 0)
        return minute, extra
    except (KeyError, TypeError, ValueError):
        return None, None

def resolve_teams(events, home_name, away_name, mid):
    names_by_id = defaultdict(set)

    for event in events:
        if not isinstance(event, dict):
            continue

        team = event.get("team")

        if isinstance(team, dict):
            tid = team.get("id")
            name = team.get("name")

            if tid and name:
                names_by_id[str(tid)].add(normalize(name))

    home = normalize(home_name)
    away = normalize(away_name)

    home_ids = [
        tid for tid, names in names_by_id.items()
        if home in names
    ]
    away_ids = [
        tid for tid, names in names_by_id.items()
        if away in names
    ]

    if (
        len(home_ids) != 1
        or len(away_ids) != 1
        or home_ids[0] == away_ids[0]
    ):
        raise ValueError(
            f"Équipes non résolues sans ambiguïté : {mid}"
        )

    return home_ids[0], away_ids[0]

def calculate(events, home_id, away_id, mid):
    counts = {
        "shots_home_last5": 0,
        "shots_away_last5": 0,
        "sot_home_last5": 0,
        "sot_away_last5": 0,
        "corners_home_last5": 0,
        "corners_away_last5": 0,
    }

    late_goals = []
    goals_at87 = []
    overtime_goals = 0

    for event in events:
        if not isinstance(event, dict):
            continue

        typ = event.get("type")
        period = event.get("period")
        minute, extra = event_minute(event)

        if typ in GOALS:
            if period in {
                "extra_time_first_half",
                "extra_time_second_half",
                "penalties",
            }:
                overtime_goals += 1

            if period == "second_half":
                if minute is None:
                    raise ValueError(
                        f"Minute de but invalide : {mid}"
                    )

                if minute == 87:
                    goals_at87.append(typ)
                elif minute > 87:
                    late_goals.append(
                        f"{typ}@{minute:g}+{extra:g}"
                    )

        if period != "second_half":
            continue

        if minute is None or extra is None:
            continue

        if not (82 <= minute < 87 and extra == 0):
            continue

        if typ not in (SHOTS | SOT | {"corner_kick"}):
            continue

        team = event.get("team")
        tid = (
            str(team.get("id"))
            if isinstance(team, dict)
            else ""
        )

        if tid == home_id:
            side = "home"
        elif tid == away_id:
            side = "away"
        else:
            raise ValueError(
                f"Équipe d'événement inconnue : {mid}, {typ}"
            )

        if typ in SHOTS:
            counts[f"shots_{side}_last5"] += 1

        if typ in SOT:
            counts[f"sot_{side}_last5"] += 1

        if typ == "corner_kick":
            opposite = (
                "away" if side == "home" else "home"
            )
            counts[f"corners_{opposite}_last5"] += 1

    return {
        **counts,
        "goal_after_87": int(len(late_goals) > 0),
        "late_goal_events": "|".join(late_goals),
        "goals_exactly_87": len(goals_at87),
        "overtime_goal_events": overtime_goals,
    }

print("=" * 72)
print("ÉTAPE 195 — RECONSTRUCTION CACHED12")
print("=" * 72)

if DEST.exists():
    raise SystemExit(
        "Fichier de sortie existant : remplacement refusé"
    )

manifest = pd.read_csv(MANIFEST, dtype=str)
audit = pd.read_csv(AUDIT, dtype=str)

if len(manifest) != 12 or manifest.match_id.duplicated().any():
    raise SystemExit("Manifest CACHED12 incorrect")

if not set(manifest.match_id).issubset(
    set(audit.loc[audit.status == "BASIC_CHECK_PASS", "match_id"])
):
    raise SystemExit("Tous les matchs n'ont pas passé l'audit 193")

kickoffs = pd.to_datetime(
    manifest.kickoff_utc, utc=True, errors="raise"
)

if not (kickoffs < pd.Timestamp("2026-08-25T00:00:00Z")).all():
    raise SystemExit("Match postérieur à TRAIN")

rows = []

for _, row in manifest.iterrows():
    mid = row["match_id"]

    events = read_cache(mid)

    home_id, away_id = resolve_teams(
        events,
        row["home_team"],
        row["away_team"],
        mid,
    )

    values = calculate(events, home_id, away_id, mid)

    rows.append({
        "match_id": mid,
        "kickoff_utc": row["kickoff_utc"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "research_split": "TRAIN",
        "research_origin": "V4_CACHED12",
        **values,
        "training_certified": 0,
        "live_certified": 0,
        "independent_verified": 0,
    })

result = pd.DataFrame(rows)

if len(result) != 12 or result.match_id.duplicated().any():
    raise SystemExit("Reconstruction incomplète")

print("\n=== MATCHS RECONSTRUITS ===")
print(
    result[
        [
            "match_id",
            "goal_after_87",
            "shots_home_last5",
            "shots_away_last5",
            "sot_home_last5",
            "sot_away_last5",
            "corners_home_last5",
            "corners_away_last5",
        ]
    ].to_string(index=False)
)

print("\n=== LABELS ===")
print("Matchs :", len(result))
print("Positifs :", int(result.goal_after_87.sum()))
print("Négatifs :", int((result.goal_after_87 == 0).sum()))
print(
    "Buts exactement à 87' :",
    int(result.goals_exactly_87.sum())
)
print(
    "Événements de but en prolongation :",
    int(result.overtime_goal_events.sum())
)

OUT.mkdir(parents=True, exist_ok=True)
result.to_csv(DEST, index=False)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("12 matchs déjà sélectionnés avant lecture des labels.")
print("Reconstruction B6 + labels seulement.")
print("A23 non reconstruit à cette étape.")
print("Aucun appel API.")
print("TRAIN original inchangé.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Aucun entraînement.")
print("Scanner LIVE inchangé.")
print("Statut : RESEARCH_ONLY")
