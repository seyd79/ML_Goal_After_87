from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

B6 = [
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

TYPES = {
    "shots": {
        "shot_off_target", "shot_blocked",
        "shot_on_target", "shot_woodwork",
        "goal", "penalty_scored",
    },
    "sot": {
        "shot_on_target", "goal", "penalty_scored",
    },
    "corners": {"corner_kick"},
}

PAIRS = {
    "shots": ("shots_home_last5", "shots_away_last5"),
    "sot": ("sot_home_last5", "sot_away_last5"),
    "corners": ("corners_home_last5", "corners_away_last5"),
}

print("=" * 72)
print("ÉTAPE 179 — ORIENTATION DOMICILE / EXTÉRIEUR V4")
print("=" * 72)

split = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype=str,
    usecols=["match_id", "split"],
)

ids = set(split.loc[split.split == "TRAIN", "match_id"])

base = pd.read_csv(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)
base = base[base.match_id.isin(ids)].copy()
base["source"] = "ALL_320"

parts = []
for path in sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv")):
    part = pd.read_csv(path, dtype={"match_id": str})
    part = part[part.research_split == "TRAIN"].copy()
    part["source"] = "BATCH02_11"
    parts.append(part)

if len(parts) != 10:
    raise SystemExit("Nombre de lots inattendu")

data = pd.concat([base] + parts, ignore_index=True)

if len(data) != 183 or data.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

metadata_path = ROOT / "HISTORICAL_BALANCED_FETCH_PLAN.csv"

metadata = {}

if metadata_path.is_file():
    m = pd.read_csv(metadata_path, dtype=str)

    required = {"match_id", "home_id", "away_id"}
    if required.issubset(m.columns):
        for _, row in m.iterrows():
            mid = row["match_id"]
            if pd.notna(mid):
                metadata[mid] = (
                    row.get("home_id"),
                    row.get("away_id"),
                )

def team_id(obj):
    if isinstance(obj, dict):
        return obj.get("id")
    return obj if isinstance(obj, str) else None

def resolve_ids(mid, payload):
    home, away = metadata.get(mid, (None, None))

    if pd.notna(home) and pd.notna(away):
        return str(home), str(away), "FETCH_PLAN"

    for parent in (payload, payload.get("match", {})):
        if not isinstance(parent, dict):
            continue

        h = parent.get("home_team_id")
        a = parent.get("away_team_id")

        if h is None:
            h = team_id(parent.get("home_team"))
        if a is None:
            a = team_id(parent.get("away_team"))

        if h is not None and a is not None:
            return str(h), str(a), "CACHE"

    return None, None, "INCONNU"

def read_cache(mid):
    candidates = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]

    path = next(
        (p for p in candidates if p.exists()),
        None,
    )

    if path is None:
        raise RuntimeError(f"Cache absent : {mid}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Cache invalide : {mid}")

    return payload

def recent(event):
    if not isinstance(event, dict):
        return False

    if event.get("period") != "second_half":
        return False

    try:
        minute = float(event.get("minute"))
        extra = float(event.get("extra_time", 0) or 0)
    except (ValueError, TypeError):
        return False

    return 82 <= minute < 87 and extra == 0

results = []
unresolved = []
unknown_teams = Counter()

for _, row in data.iterrows():
    mid = str(row.match_id)
    payload = read_cache(mid)
    home_id, away_id, origin = resolve_ids(mid, payload)

    if not home_id or not away_id or home_id == away_id:
        unresolved.append((mid, row.source, list(payload.keys())[:20]))
        continue

    counts = {
        category: {"home": 0, "away": 0}
        for category in TYPES
    }

    for event in payload.get("events", []):
        if not recent(event):
            continue

        typ = event.get("type")
        eid = team_id(event.get("team"))

        if typ not in set().union(*TYPES.values()):
            continue

        if eid is None:
            unknown_teams[row.source] += 1
            continue

        eid = str(eid)

        if eid == home_id:
            side = "home"
        elif eid == away_id:
            side = "away"
        else:
            unknown_teams[row.source] += 1
            continue

        for category, accepted in TYPES.items():
            if typ in accepted:
                counts[category][side] += 1

    record = {
        "match_id": mid,
        "source": row.source,
        "metadata_origin": origin,
    }

    for category, (hcol, acol) in PAIRS.items():
        csv_home = pd.to_numeric(row[hcol], errors="coerce")
        csv_away = pd.to_numeric(row[acol], errors="coerce")

        if pd.isna(csv_home) or pd.isna(csv_away):
            raise RuntimeError(f"CSV invalide : {mid}")

        raw_home = counts[category]["home"]
        raw_away = counts[category]["away"]

        record[f"{category}_direct"] = (
            csv_home == raw_home and csv_away == raw_away
        )
        record[f"{category}_inverse"] = (
            csv_home == raw_away and csv_away == raw_home
        )
        record[f"{category}_informatif"] = (
            raw_home != raw_away
        )
        record[f"{category}_csv"] = (
            int(csv_home), int(csv_away)
        )
        record[f"{category}_raw"] = (
            raw_home, raw_away
        )

    results.append(record)

df = pd.DataFrame(results)

print("\n=== MÉTADONNÉES D'ÉQUIPE ===")
print("Matchs résolus :", len(df), "/183")
print("Matchs non résolus :", len(unresolved))
print("Événements avec équipe non attribuable :", dict(unknown_teams))

if len(df):
    print("Origine des IDs :")
    print(df.metadata_origin.value_counts().to_string())

for item in unresolved[:5]:
    print("ID NON RÉSOLU :", item)

for source in ("ALL_320", "BATCH02_11"):
    group = df[df.source == source]

    print("\n" + "=" * 60)
    print("SOURCE :", source, "| matchs résolus :", len(group))
    print("=" * 60)

    if group.empty:
        continue

    for category in PAIRS:
        informative = group[group[f"{category}_informatif"]]

        print(f"\n{category.upper()}")
        print(
            "Correspondance directe :",
            int(group[f"{category}_direct"].sum()),
            "/", len(group)
        )
        print(
            "Correspondance inversée :",
            int(group[f"{category}_inverse"].sum()),
            "/", len(group)
        )
        print("Matchs discriminants :", len(informative))

        if len(informative):
            print(
                "Directe sur matchs discriminants :",
                int(informative[f"{category}_direct"].sum()),
                "/", len(informative)
            )
            print(
                "Inversée sur matchs discriminants :",
                int(informative[f"{category}_inverse"].sum()),
                "/", len(informative)
            )

        mismatches = group[
            ~group[f"{category}_direct"]
            & ~group[f"{category}_inverse"]
        ]

        print("Ni directe ni inversée :", len(mismatches))

        for _, item in mismatches.head(3).iterrows():
            print(
                "ÉCART :", item.match_id,
                "| CSV H/A :", item[f"{category}_csv"],
                "| RAW H/A :", item[f"{category}_raw"],
            )

print("\n=== BILAN ===")
print("Matchs TRAIN uniquement.")
print("Comparaison des six caractéristiques B.")
print("Aucune orientation corrigée automatiquement.")
print("Aucun label évalué.")
print("VALIDATION et TEST_FUTUR exclus.")
print("Aucun entraînement.")
print("Scanner LIVE inchangé.")
