from pathlib import Path
import json
import unicodedata
import re
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

SOURCES = [
    (
        ROOT / "timeline_stats_crosscheck.csv",
        "stats_home_corners_ft",
        "stats_away_corners_ft",
    ),
    (
        ROOT / "pressure_v2_experimental"
        / "pressure_v2_ft_crosscheck_same_provider.csv",
        "stats_home_corners",
        "stats_away_corners",
    ),
]

def norm(value):
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(
        c for c in s if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", s.casefold())

def load_events(mid):
    path = CACHE / f"{mid}.json"
    if not path.is_file():
        path = CACHE / f"mt_{mid}.json"

    if not path.is_file():
        raise FileNotFoundError(mid)

    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("data", obj)
    return payload["events"]

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    usecols=["match_id", "split"],
    dtype=str,
)

train_ids = set(
    plan.loc[plan["split"] == "TRAIN", "match_id"]
)

for path in sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv")):
    df = pd.read_csv(
        path,
        dtype=str,
        usecols=["match_id", "research_split"],
    )
    train_ids.update(
        df.loc[
            df["research_split"] == "TRAIN",
            "match_id",
        ]
    )

if len(train_ids) != 183:
    raise SystemExit("Population TRAIN inattendue")

print("=" * 72)
print("ÉTAPE 183 — CORNERS : TIMELINE CONTRE STATISTIQUES FOURNISSEUR")
print("=" * 72)

examined = {}
statuses = []

for path, home_col, away_col in SOURCES:
    print("\nSOURCE :", path)

    if not path.is_file():
        print("ABSENTE")
        continue

    df = pd.read_csv(path, dtype={"match_id": str})

    needed = {
        "match_id", "home_team", "away_team",
        home_col, away_col,
    }

    if not needed.issubset(df.columns):
        print("Colonnes manquantes :", sorted(needed - set(df.columns)))
        continue

    df = df[df["match_id"].isin(train_ids)].copy()

    print("Lignes TRAIN :", len(df))

    for _, row in df.iterrows():
        mid = row["match_id"]

        stats_h = pd.to_numeric(row[home_col], errors="coerce")
        stats_a = pd.to_numeric(row[away_col], errors="coerce")

        if pd.isna(stats_h) or pd.isna(stats_a):
            print(mid, "| statistiques FT manquantes")
            continue

        events = load_events(mid)

        names = {}
        for event in events:
            team = event.get("team") if isinstance(event, dict) else None
            if isinstance(team, dict) and team.get("id") and team.get("name"):
                names.setdefault(str(team["id"]), set()).add(
                    norm(team["name"])
                )

        hn = norm(row["home_team"])
        an = norm(row["away_team"])

        hid = [tid for tid, ns in names.items() if hn in ns]
        aid = [tid for tid, ns in names.items() if an in ns]

        if len(hid) != 1 or len(aid) != 1 or hid[0] == aid[0]:
            print(mid, "| orientation non résolue")
            continue

        raw_h = 0
        raw_a = 0
        unknown = 0

        for event in events:
            if not isinstance(event, dict):
                continue

            if event.get("type") != "corner_kick":
                continue

            if event.get("period") not in {
                "first_half", "second_half"
            }:
                continue

            team = event.get("team")
            tid = (
                str(team.get("id"))
                if isinstance(team, dict)
                else ""
            )

            if tid == hid[0]:
                raw_h += 1
            elif tid == aid[0]:
                raw_a += 1
            else:
                unknown += 1

        direct = (
            raw_h == stats_h
            and raw_a == stats_a
        )

        inverse = (
            raw_a == stats_h
            and raw_h == stats_a
        )

        if direct and inverse:
            status = "AMBIGU"
        elif direct:
            status = "DIRECT"
        elif inverse:
            status = "INVERSE"
        else:
            status = "AUCUNE_CORRESPONDANCE"

        informative = raw_h != raw_a

        statuses.append({
            "file": path.name,
            "match_id": mid,
            "status": status,
            "informative": informative,
        })

        examined.setdefault(mid, []).append(status)

        print(
            f"{mid} | "
            f"RAW H/A={raw_h}/{raw_a} | "
            f"STATS H/A={stats_h:g}/{stats_a:g} | "
            f"{status} | "
            f"informatif={informative} | "
            f"équipes inconnues={unknown}"
        )

print("\n=== BILAN ===")

if statuses:
    result = pd.DataFrame(statuses)

    print(
        "\nPar fichier :\n",
        pd.crosstab(
            result["file"],
            result["status"]
        ).to_string()
    )

    informative = result[result["informative"]]

    print("\nComparaisons informatives :", len(informative))
    print(
        informative["status"]
        .value_counts()
        .to_string()
    )

    print("Matchs distincts :", len(examined))

    conflicts = {
        mid: values
        for mid, values in examined.items()
        if len(set(values)) > 1
    }
    print("Désaccords entre références :", len(conflicts))

    for mid, values in list(conflicts.items())[:5]:
        print("DÉSACCORD :", mid, values)
else:
    print("Aucune comparaison exploitable.")

print("\nTRAIN uniquement.")
print("Temps réglementaire, temps additionnel compris.")
print("Prolongations exclues.")
print("Aucun label recalculé.")
print("VALIDATION et TEST_FUTUR exclus.")
print("Aucun fichier existant modifié.")
print("Aucun entraînement.")
print("Scanner LIVE inchangé.")
