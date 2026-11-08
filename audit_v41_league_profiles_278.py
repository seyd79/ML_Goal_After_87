import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median

ROOT = Path("outputs/v4_research")
POOL = ROOT / "v41_context_pool_266/CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv"
RAW = Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv")
OUT = ROOT / "v41_league_profiles_278"

FEATURES = [
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
]

def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def period(date):
    if date < "2026-08-01":
        return "DISCOVERY"
    if date < "2026-08-20":
        return "AUG01_19"
    if date < "2026-08-25":
        return "AUG20_24"
    return "OUTSIDE"

pool_rows = load(POOL)

if len(pool_rows) != 1131:
    raise SystemExit("Population du pool incorrecte")

selected = {
    r["match_id"]: r
    for r in pool_rows
    if r.get("source_research") == "RAW_V2"
}

if len(selected) != 534:
    raise SystemExit("534 matchs RAW attendus")

raw = {
    r["match_id"]: r
    for r in load(RAW)
    if r.get("match_id") in selected
}

if len(raw) != 534:
    raise SystemExit("Correspondances RAW incompletes")

groups = defaultdict(list)
all_values = defaultdict(list)

for mid, p in selected.items():
    r = raw[mid]

    # Verification de correspondance : score et label.
    for field in [
        "score_87_home", "score_87_away", "goal_after_87"
    ]:
        if number(r.get(field)) != number(p.get(field)):
            raise SystemExit("Incoherence RAW/POOL : " + mid)

    league = str(r.get("competition_id", "")).strip()
    if not league:
        raise SystemExit("competition_id manquant : " + mid)

    date = p.get("kickoff_utc", "")[:10]
    block = period(date)

    if block == "OUTSIDE":
        raise SystemExit("Date hors population : " + mid)

    values = {
        name: number(r.get(name))
        for name in FEATURES
    }

    for name, value in values.items():
        if value is not None:
            all_values[name].append(value)

    shots = number(r.get("shots_80_87"))

    groups[(league, block)].append({
        "id": mid,
        "date": date,
        "values": values,
        "shots_available": shots is not None
    })

print("=" * 76)
print("ETAPE 278 - AUDIT PROFILS STATISTIQUES CHAMPIONNATS")
print("=" * 76)
print("Matchs RAW :", len(selected))
print("Groupes competition/periode :", len(groups))

print("\n=== QUALITE ET ECHELLE DES INDICATEURS ===")

for name in FEATURES:
    vals = all_values[name]

    if vals:
        print(
            f"{name:32s} "
            f"n={len(vals):3d} "
            f"min={min(vals):.3f} "
            f"mediane={median(vals):.3f} "
            f"max={max(vals):.3f}"
        )
    else:
        print(name, "AUCUNE VALEUR")

rows = []

for (league, block), matches in sorted(groups.items()):
    output = {
        "competition_id": league,
        "period": block,
        "n_matches": len(matches),
        "n_days": len({x["date"] for x in matches}),
        "first_date": min(x["date"] for x in matches),
        "last_date": max(x["date"] for x in matches),
        "n_shots_80_87_available": sum(
            x["shots_available"] for x in matches
        ),
        "profile_certified_asof": "NO",
        "profile_aggregation": "ROW_MEAN_EXPLORATORY"
    }

    for feature in FEATURES:
        vals = [
            x["values"][feature]
            for x in matches
            if x["values"][feature] is not None
        ]

        output[feature + "_n"] = len(vals)
        output[feature + "_mean_raw"] = (
            round(mean(vals), 6) if vals else ""
        )
        output[feature + "_median_raw"] = (
            round(median(vals), 6) if vals else ""
        )

    rows.append(output)

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "LEAGUE_PROFILE_AUDIT_V1.csv"

columns = list(rows[0].keys())

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

periods = ["DISCOVERY", "AUG01_19", "AUG20_24"]

for block in periods:
    current = [
        r for r in rows if r["period"] == block
    ]
    current.sort(
        key=lambda r: (-r["n_matches"], r["competition_id"])
    )

    print("\n=== ", block, " ===", sep="")
    print("Competitions :", len(current))
    print(
        "Matchs :", sum(r["n_matches"] for r in current)
    )

    for r in current[:15]:
        over = r["league_home_over25_freq_mean_raw"]
        btts = r["league_home_btts_freq_mean_raw"]

        print(
            f'{r["competition_id"]:13s} '
            f'n={r["n_matches"]:3d} '
            f'jours={r["n_days"]:3d} '
            f'over25_home_moy={str(over):9s} '
            f'btts_home_moy={str(btts):9s}'
        )

discovery_leagues = {
    league for league, block in groups
    if block == "DISCOVERY"
}

candidate_leagues = {
    league for league, block in groups
    if block == "AUG20_24"
} - discovery_leagues

print("\n=== COUVERTURE ===")
print("Championnats de decouverte :",
      sorted(discovery_leagues))
print("Nouvelles competitions du 20-24 aout :",
      len(candidate_leagues))

print("\nFICHIER :", path)
print("\nAVERTISSEMENTS :")
print("- Moyennes de colonnes disponibles, pas statistiques")
print("  officielles certifiees des championnats")
print("- Disponibilite avant match non verifiee")
print("- Aucun championnat classe selon les labels")
print("- Aucun score de similarite encore calcule")
print("- Aucune collecte API ni entrainement")
print("- Aucun holdout ouvert, TEST_FUTUR non consulte")
print("- Scanner V3-C inchange")
print("ETAPE 278 TERMINEE.")
