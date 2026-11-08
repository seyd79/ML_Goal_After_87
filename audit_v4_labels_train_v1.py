from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

print("=" * 72)
print("ÉTAPE 184 — AUDIT INDÉPENDANT DES LABELS V4")
print("=" * 72)

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

train_ids = set(
    plan.loc[plan["split"] == "TRAIN", "match_id"]
)

base = pd.read_csv(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
    usecols=["match_id", "goal_after_87"],
)
base = base[base.match_id.isin(train_ids)].copy()
base["source"] = "ALL_320"

parts = []
for path in sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv")):
    p = pd.read_csv(
        path,
        dtype={"match_id": str},
        usecols=[
            "match_id", "research_split",
            "goal_after_87",
        ],
    )
    p = p[p.research_split == "TRAIN"].copy()
    p["source"] = "BATCH02_11"
    parts.append(
        p[["match_id", "goal_after_87", "source"]]
    )

if len(parts) != 10:
    raise SystemExit("Nombre de lots inattendu")

data = pd.concat([base] + parts, ignore_index=True)

if len(data) != 183 or data.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

data["goal_after_87"] = pd.to_numeric(
    data.goal_after_87, errors="raise"
)

if not data.goal_after_87.isin([0, 1]).all():
    raise SystemExit("Labels non binaires")

def read_events(mid):
    paths = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]
    path = next((p for p in paths if p.is_file()), None)

    if path is None:
        raise RuntimeError(f"Cache absent : {mid}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)

    if payload.get("coverage") != "full":
        raise RuntimeError(f"Couverture non complète : {mid}")

    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError(f"Timeline invalide : {mid}")

    return events

GOAL_TYPES = {"goal", "penalty_scored", "own_goal"}
STANDARD_TYPES = {"goal", "penalty_scored"}

records = []
issues = Counter()

for _, row in data.iterrows():
    mid = row.match_id
    events = read_events(mid)

    goals_after = []
    goals_at87 = []
    goals_extra_period = []
    invalid_goal_events = []
    var_after87 = 0

    for event in events:
        if not isinstance(event, dict):
            continue

        typ = event.get("type")
        period = event.get("period")

        if typ == "var" and period == "second_half":
            try:
                if float(event.get("minute")) > 87:
                    var_after87 += 1
            except (TypeError, ValueError):
                pass

        if typ not in GOAL_TYPES:
            continue

        try:
            minute = float(event.get("minute"))
            extra = float(event.get("extra_time", 0) or 0)
        except (TypeError, ValueError):
            invalid_goal_events.append(event)
            continue

        info = {
            "type": typ,
            "minute": minute,
            "extra_time": extra,
            "period": period,
            "sequence": event.get("sequence"),
        }

        if period != "second_half":
            if period in {
                "extra_time_first_half",
                "extra_time_second_half",
                "penalties",
            }:
                goals_extra_period.append(info)
            continue

        if minute == 87:
            goals_at87.append(info)
        elif minute > 87:
            goals_after.append(info)

    label = int(row.goal_after_87)

    reconstructed = int(bool(goals_after))
    without_own = int(any(
        e["type"] in STANDARD_TYPES
        for e in goals_after
    ))

    records.append({
        "match_id": mid,
        "source": row.source,
        "csv_label": label,
        "raw_label_all_goals": reconstructed,
        "raw_label_without_own": without_own,
        "goals_after87": goals_after,
        "goals_at87": goals_at87,
        "goals_extra_period": goals_extra_period,
        "invalid_goals": invalid_goal_events,
        "var_after87": var_after87,
    })

    if goals_at87:
        issues["matchs_avec_but_a_87"] += 1
    if goals_extra_period:
        issues["matchs_avec_but_prolongation"] += 1
    if invalid_goal_events:
        issues["matchs_avec_minute_but_invalide"] += 1
    if var_after87:
        issues["matchs_avec_var_apres87"] += 1

df = pd.DataFrame(records)

print("\n=== POPULATION ===")
print("Matchs :", len(df))
print("Labels CSV positifs :", int(df.csv_label.sum()))
print("Labels reconstruits positifs :",
      int(df.raw_label_all_goals.sum()))

print("\n=== COMPARAISON PAR SOURCE ===")

for source, group in df.groupby("source"):
    print("\nSOURCE :", source, "| n =", len(group))

    for col in (
        "raw_label_all_goals",
        "raw_label_without_own",
    ):
        same = group.csv_label == group[col]

        print(
            col,
            "| identiques =", int(same.sum()),
            "/", len(group),
            "| divergences =", int((~same).sum()),
        )

    print(
        "Labels CSV positifs :",
        int(group.csv_label.sum())
    )

    print(
        "Labels reconstruits positifs :",
        int(group.raw_label_all_goals.sum())
    )

print("\n=== TABLE DE CONFUSION DES LABELS ===")
print(
    pd.crosstab(
        df.csv_label,
        df.raw_label_all_goals,
        rownames=["CSV"],
        colnames=["Timeline brute"],
    ).to_string()
)

print("\n=== CAS PARTICULIERS ===")
for name, count in issues.items():
    print(name, ":", count)

mismatches = df[
    df.csv_label != df.raw_label_all_goals
]

print("\n=== DIVERGENCES : 20 PREMIERS CAS ===")
print("Total divergences :", len(mismatches))

for _, r in mismatches.head(20).iterrows():
    print(
        "\nMATCH :", r.match_id,
        "| source :", r.source,
        "| CSV :", r.csv_label,
        "| timeline :", r.raw_label_all_goals,
    )
    print("Buts après 87 :", r.goals_after87)
    print("Buts à 87 :", r.goals_at87)
    print("Buts en prolongation :", r.goals_extra_period)
    print("VAR après 87 :", r.var_after87)
    print("Buts avec minute invalide :", r.invalid_goals)

print("\n=== CONCLUSION AUTOMATIQUE ===")

if len(mismatches) == 0:
    print("COHÉRENCE DES LABELS BRUTS : PASS")
    print("Pas de divergence selon la convention testée.")
else:
    print("COHÉRENCE DES LABELS BRUTS : INVESTIGATION")
    print("Ne pas corriger les labels sans audit manuel.")

print("\n=== LIMITES ===")
print("Les événements VAR ne sont pas interprétés automatiquement.")
print("La validation indépendante des buts reste nécessaire.")
print("TRAIN uniquement : 183 matchs.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Aucune donnée modifiée.")
print("Aucun entraînement.")
print("Scanner LIVE inchangé.")
