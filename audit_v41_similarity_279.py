import csv
import math
from pathlib import Path

ROOT = Path("outputs/v4_research")
SOURCE = ROOT / "v41_league_profiles_278/LEAGUE_PROFILE_AUDIT_V1.csv"
OUT = ROOT / "v41_league_similarity_279"

REFERENCE = "comp_8814"

FEATURES = [
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
]

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def integer(value):
    x = number(value)
    if x is None or x < 0 or not x.is_integer():
        raise ValueError("Entier invalide : " + str(value))
    return int(x)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    profiles = list(csv.DictReader(f))

reference_rows = [
    r for r in profiles
    if r.get("competition_id") == REFERENCE
    and r.get("period") == "DISCOVERY"
]

if len(reference_rows) != 1:
    raise SystemExit("Reference historique absente ou dupliquee")

reference = reference_rows[0]
ref_values = {}

for feature in FEATURES:
    value = number(reference.get(feature + "_mean_raw"))
    count = integer(reference.get(feature + "_n"))

    if value is None or not 0 <= value <= 1 or count < 30:
        raise SystemExit(
            "Caracteristique reference insuffisante : " + feature
        )

    ref_values[feature] = value

historical_competitions = {
    r["competition_id"]
    for r in profiles
    if r.get("period") == "DISCOVERY"
}

candidates = [
    r for r in profiles
    if r.get("period") == "AUG20_24"
    and r.get("competition_id") not in historical_competitions
]

if len({r["competition_id"] for r in candidates}) != len(candidates):
    raise SystemExit("Doublons de competitions candidates")

results = []

for r in candidates:
    n = integer(r["n_matches"])
    days = integer(r["n_days"])

    differences = []
    coverages = []
    values = {}

    for feature in FEATURES:
        value = number(r.get(feature + "_mean_raw"))
        available = integer(r.get(feature + "_n"))
        coverage = available / n if n else 0

        values[feature] = value
        coverages.append(coverage)

        if (
            value is not None
            and 0 <= value <= 1
            and coverage >= 0.80
        ):
            differences.append(
                abs(value - ref_values[feature])
            )

    comparable = len(differences) == len(FEATURES)

    if not comparable:
        category = "C_NON_COMPARABLE"
    elif n >= 5 and days >= 3:
        category = "A_PRIORITE_A_DOCUMENTER"
    elif n >= 3 and days >= 2:
        category = "B_PEU_DE_DONNEES"
    else:
        category = "C_ECHANTILLON_TROP_PETIT"

    distance = (
        sum(differences) / len(FEATURES)
        if comparable else None
    )

    item = {
        "competition_id": r["competition_id"],
        "category": category,
        "n_matches": n,
        "n_days": days,
        "features_comparables": len(differences),
        "coverage_min": round(min(coverages), 3),
        "distance": (
            round(distance, 6)
            if distance is not None else ""
        ),
        "reference": REFERENCE,
        "status": "RESEARCH_ONLY_ASOF_UNVERIFIED"
    }

    for feature in FEATURES:
        item[feature + "_mean"] = (
            values[feature]
            if values[feature] is not None else ""
        )

    results.append(item)

order = {
    "A_PRIORITE_A_DOCUMENTER": 0,
    "B_PEU_DE_DONNEES": 1,
    "C_ECHANTILLON_TROP_PETIT": 2,
    "C_NON_COMPARABLE": 3
}

results.sort(
    key=lambda r: (
        order[r["category"]],
        float(r["distance"]) if r["distance"] != "" else 999,
        -r["n_matches"],
        r["competition_id"]
    )
)

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "LEAGUE_SIMILARITY_EXPLORATORY_V1.csv"

if not results:
    raise SystemExit("Aucun championnat candidat")

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print("=" * 76)
print("ETAPE 279 - RECHERCHE DES CHAMPIONNATS SIMILAIRES")
print("=" * 76)

print("\n=== CHAMPIONNAT DE REFERENCE ===")
print("Competition :", REFERENCE)

for feature in FEATURES:
    print(
        f"{feature:32s} : "
        f"{100 * ref_values[feature]:.1f}%"
    )

print("\n=== QUALITE DES CANDIDATS ===")
print("Nouvelles competitions :", len(candidates))

for category in order:
    count = sum(
        r["category"] == category
        for r in results
    )
    print(category, ":", count)

print("\n=== CLASSEMENT DES CANDIDATS ===")

comparable = [
    r for r in results
    if r["category"].startswith(("A_", "B_"))
]

for r in comparable[:20]:
    distance_pp = 100 * float(r["distance"])

    print(
        f'{r["competition_id"]:13s} '
        f'| {r["category"]:26s} '
        f'| ecart={distance_pp:.1f} points '
        f'| matchs={r["n_matches"]} '
        f'| jours={r["n_days"]} '
        f'| couverture={r["coverage_min"]:.0%}'
    )

if not comparable:
    print("Aucun candidat suffisamment documente.")

print("\n=== FICHIER CREE ===")
print(path)

print("\nLIMITES :")
print("- Similarite statistique, pas probabilite de reussite")
print("- Variables historiques non certifiees as-of")
print("- Faibles effectifs de nombreux championnats")
print("- Aucun label candidat utilise pour le classement")
print("- Aucune collecte ou evaluation")
print("- Aucun holdout ouvert, TEST_FUTUR non consulte")
print("- V3-C inchange")

print("\nETAPE 279 TERMINEE.")
