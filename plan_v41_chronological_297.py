import csv
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date

ROOT = Path("outputs/v4_research")
INVENTORY = ROOT / "v41_chronological_inventory_296/MATCHES_METADATA_ONLY.csv"
RAW = Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv")
POOL = ROOT / "v41_context_pool_266/CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv"
OUT = ROOT / "v41_chronological_plan_297"

QUOTA_PER_MONTH = 12
COMPETITIONS = {"comp_8531", "comp_623699"}

def read(path):
    if not path.is_file():
        raise SystemExit("Fichier absent : " + str(path))
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

if OUT.exists():
    raise SystemExit("STOP : dossier de sortie deja existant")

inventory = read(INVENTORY)
old_ids = {
    r["match_id"] for r in read(RAW)
}
pool_ids = {
    r["match_id"] for r in read(POOL)
}

seen = set()
groups = defaultdict(list)
duplicates = 0

for r in inventory:
    comp = r.get("competition_id", "").strip()
    mid = r.get("match_id", "").strip()
    day = r.get("kickoff_utc", "")[:10]

    if comp not in COMPETITIONS or not mid:
        continue

    if mid in seen:
        duplicates += 1
        continue
    seen.add(mid)

    if mid in old_ids or mid in pool_ids:
        continue

    try:
        parsed = date.fromisoformat(day)
    except ValueError:
        continue

    week = parsed.isocalendar()

    item = dict(r)
    item["month"] = day[:7]
    item["day"] = day
    item["week"] = f"{week.year}-W{week.week:02d}"

    groups[(comp, day[:7])].append(item)

selected = []
summaries = []

print("=" * 76)
print("ETAPE 297 - PLAN DE COLLECTE CHRONOLOGIQUE")
print("=" * 76)
print("Inventaire initial :", len(inventory))
print("Doublons dans inventaire :", duplicates)
print("Matchs existants RAW :", len(old_ids))
print("Matchs existants POOL :", len(pool_ids))
print("Quota maximal par mois :", QUOTA_PER_MONTH)

for comp, month in sorted(groups):
    matches = groups[(comp, month)]
    by_week = defaultdict(list)

    for r in sorted(
        matches, key=lambda x: (x["day"], x["match_id"])
    ):
        by_week[r["week"]].append(r)

    picked = []
    picked_days = set()
    team_counts = Counter()

    # Tourner entre les semaines, en preferant
    # des jours et equipes encore peu representes.
    while len(picked) < QUOTA_PER_MONTH:
        progress = False

        for week in sorted(by_week):
            available = by_week[week]
            if not available:
                continue

            available.sort(
                key=lambda r: (
                    r["day"] in picked_days,
                    team_counts[r.get("home_team", "")]
                    + team_counts[r.get("away_team", "")],
                    r["day"],
                    r["match_id"]
                )
            )

            r = available.pop(0)
            picked.append(r)
            picked_days.add(r["day"])

            for field in ("home_team", "away_team"):
                if r.get(field):
                    team_counts[r[field]] += 1

            progress = True

            if len(picked) >= QUOTA_PER_MONTH:
                break

        if not progress:
            break

    for r in picked:
        selected.append({
            "competition_id": comp,
            "season_id": r["season_id"],
            "match_id": r["match_id"],
            "kickoff_utc": r["kickoff_utc"],
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "month": month,
            "week": r["week"],
            "selection_status": "RESEARCH_COLLECTION_CANDIDATE",
            "protected_clearance": "NOT_VERIFIED",
            "label_status": "NOT_CHECKED",
        })

    summaries.append({
        "competition_id": comp,
        "month": month,
        "new_metadata_available": len(matches),
        "selected": len(picked),
        "selected_days": len(picked_days),
        "selected_weeks": len({r["week"] for r in picked}),
        "selected_seasons": len({r["season_id"] for r in picked}),
    })

OUT.mkdir(parents=True, exist_ok=False)

detail = OUT / "MATCHES_SELECTED_CHRONOLOGICALLY.csv"
summary = OUT / "MONTHLY_SELECTION_REPORT.csv"

with detail.open("w", encoding="utf-8", newline="") as f:
    if selected:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(sorted(
            selected,
            key=lambda r: (
                r["competition_id"],
                r["kickoff_utc"],
                r["match_id"]
            )
        ))

with summary.open("w", encoding="utf-8", newline="") as f:
    if summaries:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

print("\n=== COUVERTURE PAR CHAMPIONNAT ===")

for comp in sorted(COMPETITIONS):
    rows = [r for r in selected if r["competition_id"] == comp]
    months = Counter(r["month"] for r in rows)
    seasons = Counter(r["season_id"] for r in rows)

    print("\n", comp)
    print(" Selectionnes :", len(rows))
    print(" Mois distincts :", len(months))
    print(" Semaines distinctes :", len({r["week"] for r in rows}))
    print(" Jours distincts :", len({r["kickoff_utc"][:10] for r in rows}))
    print(" Saisons :", dict(seasons))
    print(" Repartition :", dict(sorted(months.items())))

print("\n=== SORTIES ===")
print(detail)
print(summary)

print("\nAVERTISSEMENTS")
print("- Selection independante des labels")
print("- Matchs existants exclus")
print("- Selection exploratoire, non exhaustive")
print("- Exclusions protegees non encore certifiees")
print("- Aucune collecte de timelines")
print("- Aucun appel API")
print("- Aucun match admis automatiquement au pool")
print("- TEST_FUTUR non consulte")
print("- V3-C inchange")
print("ETAPE 297 TERMINEE.")
