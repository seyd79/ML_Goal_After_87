from pathlib import Path
from collections import Counter, defaultdict
import difflib
import json
import re
import unicodedata

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
    "sot": {"shot_on_target", "goal", "penalty_scored"},
    "corners": {"corner_kick"},
}

PAIRS = {
    "shots": ("shots_home_last5", "shots_away_last5"),
    "sot": ("sot_home_last5", "sot_away_last5"),
    "corners": ("corners_home_last5", "corners_away_last5"),
}

def norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value.casefold())

def load_csv(path):
    return pd.read_csv(path, dtype={"match_id": str})

plan = load_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

base = load_csv(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv"
)
base = base.merge(
    plan[["match_id", "split"]],
    on="match_id",
    validate="one_to_one",
)
base = base[base["split"] == "TRAIN"].copy()
base["source"] = "ALL_320"

parts = []
for path in sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv")):
    p = load_csv(path)
    p = p[p["research_split"] == "TRAIN"].copy()
    p["source"] = "BATCH02_11"
    parts.append(p)

if len(parts) != 10:
    raise SystemExit("10 lots B attendus")

train = pd.concat([base] + parts, ignore_index=True)

if len(train) != 183 or train.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

# Sources documentant explicitement domicile et extérieur.
meta_paths = [
    ROOT / "HISTORICAL_BALANCED_FETCH_PLAN.csv",
    *sorted(JOIN.glob("HISTORICAL_WAVE1_BATCH*_MANIFEST_V1.csv")),
    ROOT / "timeline_stats_crosscheck.csv",
    JOIN / "SCORE_87_QUALITY_AUDIT.csv",
    ROOT / "pressure_v2_experimental"
         / "pressure_v2_ft_crosscheck_same_provider.csv",
    ROOT / "pressure_v2_experimental"
         / "pressure_v2_85_86_87_EXPERIMENTAL.csv",
]

metadata = defaultdict(set)
train_ids = set(train.match_id)

for path in meta_paths:
    if not path.is_file():
        continue

    m = load_csv(path)
    necessary = {"match_id", "home_team", "away_team"}

    if not necessary.issubset(m.columns):
        continue

    for _, row in m.iterrows():
        mid = row["match_id"]

        if mid not in train_ids:
            continue

        home = row["home_team"]
        away = row["away_team"]

        if norm(home) and norm(away):
            metadata[mid].add((str(home), str(away)))

def read_events(mid):
    candidates = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]
    path = next(
        (p for p in candidates if p.is_file()),
        None,
    )
    if path is None:
        raise RuntimeError(f"Cache manquant : {mid}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)

    events = payload.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError(f"Timeline invalide : {mid}")
    return events

def is_recent(e):
    if not isinstance(e, dict):
        return False
    if e.get("period") != "second_half":
        return False

    try:
        minute = float(e.get("minute"))
        extra = float(e.get("extra_time", 0) or 0)
    except (TypeError, ValueError):
        return False

    return 82 <= minute < 87 and extra == 0

resolved = []
unresolved = []
conflicts = []
unknown_event_teams = Counter()

print("=" * 72)
print("ÉTAPE 181 — ORIENTATION B29 PAR NOMS DES ÉQUIPES")
print("=" * 72)

for _, row in train.iterrows():
    mid = row["match_id"]
    events = read_events(mid)

    possible = metadata.get(mid, set())
    normalized_pairs = {
        (norm(h), norm(a))
        for h, a in possible
    }

    if len(normalized_pairs) != 1:
        reason = (
            "aucune métadonnée"
            if not possible
            else "métadonnées contradictoires"
        )
        unresolved.append((mid, row["source"], reason))
        if len(normalized_pairs) > 1:
            conflicts.append((mid, sorted(possible)))
        continue

    home_name, away_name = sorted(possible)[0]
    home_norm, away_norm = next(iter(normalized_pairs))

    names_by_id = defaultdict(set)

    for e in events:
        if not isinstance(e, dict):
            continue

        team = e.get("team")
        if not isinstance(team, dict):
            continue

        tid = team.get("id")
        name = team.get("name")

        if tid and name:
            names_by_id[str(tid)].add(norm(name))

    home_ids = [
        tid for tid, names in names_by_id.items()
        if home_norm in names
    ]
    away_ids = [
        tid for tid, names in names_by_id.items()
        if away_norm in names
    ]

    if (
        len(home_ids) != 1
        or len(away_ids) != 1
        or home_ids[0] == away_ids[0]
    ):
        suggestions = []

        for target in (home_norm, away_norm):
            choices = [
                (difflib.SequenceMatcher(
                    None, target, name
                ).ratio(), tid, name)
                for tid, names in names_by_id.items()
                for name in names
            ]
            suggestions.append(
                sorted(choices, reverse=True)[:2]
            )

        unresolved.append((
            mid, row["source"],
            f"HOME={home_name} AWAY={away_name}",
            suggestions,
        ))
        continue

    home_id = home_ids[0]
    away_id = away_ids[0]

    counts = {
        category: {"home": 0, "away": 0}
        for category in TYPES
    }

    for e in events:
        if not is_recent(e):
            continue

        typ = e.get("type")
        if not any(typ in types for types in TYPES.values()):
            continue

        team = e.get("team")
        tid = str(team.get("id")) if isinstance(team, dict) else ""

        if tid == home_id:
            side = "home"
        elif tid == away_id:
            side = "away"
        else:
            unknown_event_teams[row["source"]] += 1
            continue

        for category, accepted in TYPES.items():
            if typ in accepted:
                counts[category][side] += 1

    rec = {
        "match_id": mid,
        "source": row["source"],
    }

    for category, (hcol, acol) in PAIRS.items():
        h = pd.to_numeric(row[hcol], errors="coerce")
        a = pd.to_numeric(row[acol], errors="coerce")

        if pd.isna(h) or pd.isna(a):
            raise RuntimeError(f"Valeur manquante : {mid}")

        rh = counts[category]["home"]
        ra = counts[category]["away"]

        rec[f"{category}_direct"] = h == rh and a == ra
        rec[f"{category}_inverse"] = h == ra and a == rh
        rec[f"{category}_informatif"] = rh != ra
        rec[f"{category}_total_ok"] = h + a == rh + ra
        rec[f"{category}_csv"] = (int(h), int(a))
        rec[f"{category}_raw"] = (rh, ra)

    resolved.append(rec)

print("\n=== RÉSOLUTION DES ÉQUIPES ===")
print("Matchs résolus :", len(resolved), "/183")
print("Matchs non résolus :", len(unresolved))
print("Conflits de métadonnées :", len(conflicts))
print("Événements non attribués :", dict(unknown_event_teams))

for item in unresolved[:10]:
    print("NON RÉSOLU :", item)

for item in conflicts[:5]:
    print("CONFLIT :", item)

df = pd.DataFrame(resolved)

for source in ("ALL_320", "BATCH02_11"):
    if df.empty:
        print("\nAucun match résolu pour analyse.")
        break

    g = df[df["source"] == source]

    print("\n" + "=" * 60)
    print("SOURCE :", source, "| matchs :", len(g))
    print("=" * 60)

    for category in PAIRS:
        informative = g[g[f"{category}_informatif"]]

        direct = int(informative[f"{category}_direct"].sum())
        inverse = int(informative[f"{category}_inverse"].sum())
        neither = informative[
            ~informative[f"{category}_direct"]
            & ~informative[f"{category}_inverse"]
        ]

        print(f"\n{category.upper()}")
        print("Totaux corrects :", int(g[f"{category}_total_ok"].sum()), "/", len(g))
        print("Matchs informatifs :", len(informative))
        print("Orientation directe :", direct, "/", len(informative))
        print("Orientation inversée :", inverse, "/", len(informative))
        print("Ni directe ni inversée :", len(neither))

        for _, item in neither.head(3).iterrows():
            print(
                "ÉCART :", item["match_id"],
                "| CSV H/A :", item[f"{category}_csv"],
                "| RAW H/A :", item[f"{category}_raw"],
            )

print("\n=== BILAN ===")
print("Uniquement les 183 matchs TRAIN.")
print("Aucune correspondance approximative appliquée.")
print("Aucune inversion automatique.")
print("Aucun label recalculé.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Aucun fichier de données modifié.")
print("Scanner V3-C inchangé.")
