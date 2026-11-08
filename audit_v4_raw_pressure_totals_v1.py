from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

print("=" * 72)
print("ÉTAPE 178 — RECONSTRUCTION INDÉPENDANTE DE LA PRESSION")
print("=" * 72)

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

base = pd.read_csv(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)

base = base.merge(
    plan, on="match_id", validate="one_to_one"
)
base = base[base["split"] == "TRAIN"].copy()
base["source"] = "ALL_320"

paths = sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(paths) != 10:
    raise SystemExit("Nombre de lots inattendu")

parts = []
for path in paths:
    p = pd.read_csv(path, dtype={"match_id": str})
    p = p[p["research_split"] == "TRAIN"].copy()
    p["source"] = "BATCH02_11"
    parts.append(p)

data = pd.concat([base] + parts, ignore_index=True)

if len(data) != 183 or data.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

RAW_SHOTS = {
    "shot_off_target",
    "shot_blocked",
    "shot_on_target",
}

RAW_GOALS = {"goal", "penalty_scored"}

SHOT_OPTIONS = {
    "tirs_3_types": RAW_SHOTS,
    "tirs_plus_boismontant": RAW_SHOTS | {"shot_woodwork"},
    "tirs_plus_buts": RAW_SHOTS | RAW_GOALS,
    "tirs_plus_buts_bois": (
        RAW_SHOTS | RAW_GOALS | {"shot_woodwork"}
    ),
    "tirs_toutes_tentatives": (
        RAW_SHOTS | RAW_GOALS |
        {"shot_woodwork", "penalty_saved", "penalty_missed"}
    ),
}

SOT_OPTIONS = {
    "cadrés_seuls": {"shot_on_target"},
    "cadrés_plus_buts": {"shot_on_target"} | RAW_GOALS,
    "cadrés_buts_penalty_saved": (
        {"shot_on_target", "goal", "penalty_scored", "penalty_saved"}
    ),
}

def read_events(match_id):
    name = str(match_id)

    candidates = [
        CACHE / f"{name}.json",
        CACHE / f"mt_{name}.json",
    ]

    path = next(
        (p for p in candidates if p.is_file()),
        None
    )

    if path is None:
        raise RuntimeError(f"Cache absent : {name}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)
    events = payload.get("events", [])

    if not isinstance(events, list):
        raise RuntimeError(f"Events invalides : {name}")

    return events

def valid_before_87(event):
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

rows = []

for _, match in data.iterrows():
    events = read_events(match["match_id"])
    recent = [e for e in events if valid_before_87(e)]

    types = Counter(str(e.get("type")) for e in recent)

    def count(names):
        return sum(types[t] for t in names)

    def csv_total(home, away):
        x = pd.to_numeric(match[home], errors="coerce")
        y = pd.to_numeric(match[away], errors="coerce")
        if pd.isna(x) or pd.isna(y):
            raise RuntimeError(
                f"Valeur CSV manquante : {match['match_id']}"
            )
        return float(x + y)

    row = {
        "match_id": match["match_id"],
        "source": match["source"],
        "events_window": len(recent),
        "csv_shots": csv_total(
            "shots_home_last5", "shots_away_last5"
        ),
        "csv_sot": csv_total(
            "sot_home_last5", "sot_away_last5"
        ),
        "csv_corners": csv_total(
            "corners_home_last5", "corners_away_last5"
        ),
        "raw_corners": count({"corner_kick"}),
    }

    for name, event_types in SHOT_OPTIONS.items():
        row[name] = count(event_types)

    for name, event_types in SOT_OPTIONS.items():
        row[name] = count(event_types)

    rows.append(row)

df = pd.DataFrame(rows)

print("\n=== POPULATION ===")
print("Matchs :", len(df))
print("Événements admissibles 82' à 86' :", df.events_window.sum())

def show_comparison(group, actual, candidates):
    for candidate in candidates:
        diff = group[candidate] - group[actual]
        exact = int((diff == 0).sum())

        print(
            f"{candidate:28s}",
            f"identiques={exact:3d}/{len(group)}",
            f"écart_absolu_total={abs(diff).sum():.0f}",
            f"écart_max={abs(diff).max():.0f}"
        )

for source, group in df.groupby("source"):
    print("\n" + "=" * 60)
    print("SOURCE :", source, "| matchs :", len(group))
    print("=" * 60)

    print("\n=== CORNERS ===")
    show_comparison(
        group, "csv_corners", ["raw_corners"]
    )

    print("\n=== TIRS ===")
    show_comparison(
        group, "csv_shots", list(SHOT_OPTIONS)
    )

    print("\n=== TIRS CADRÉS ===")
    show_comparison(
        group, "csv_sot", list(SOT_OPTIONS)
    )

    bad_corners = group[
        group["csv_corners"] != group["raw_corners"]
    ]

    print("\nÉCARTS CORNERS — 5 EXEMPLES MAXIMUM")
    if bad_corners.empty:
        print("Aucun écart")
    else:
        print(
            bad_corners[
                ["match_id", "csv_corners", "raw_corners"]
            ].head(5).to_string(index=False)
        )

print("\n=== BILAN ===")
print("Uniquement les 183 matchs TRAIN.")
print("Fenêtre : 82 <= minute < 87, seconde période.")
print("Comparaison des totaux, sans attribution domicile/extérieur.")
print("Aucun label recalculé.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Aucun fichier existant modifié.")
print("Aucun modèle entraîné.")
print("Scanner V3-C inchangé.")
