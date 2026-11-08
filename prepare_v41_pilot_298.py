import ast
import csv
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("outputs/v4_research")
SOURCE = (
    ROOT / "v41_chronological_plan_297"
    / "MATCHES_SELECTED_CHRONOLOGICALLY.csv"
)
RAW = Path(
    "outputs/historical_dataset_v2/"
    "Historical_Dataset_Goal_After_87_RAW_v2.csv"
)
OUT = ROOT / "v41_pilot_plan_298"

COMPETITIONS = ["comp_8531", "comp_623699"]
MONTHS_PER_COMPETITION = 6

def load(path):
    if not path.is_file():
        raise SystemExit("Fichier absent : " + str(path))
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

if OUT.exists():
    raise SystemExit(
        "STOP : repertoire de sortie deja existant."
    )

rows = load(SOURCE)
historical_ids = {
    r["match_id"] for r in load(RAW)
}

if len(rows) != 227:
    raise SystemExit(
        f"Population inattendue : {len(rows)} au lieu de 227"
    )

ids = [r.get("match_id", "") for r in rows]

if len(ids) != len(set(ids)) or not all(ids):
    raise SystemExit("IDs du manifeste invalides ou dupliques")

overlap = set(ids) & historical_ids

if overlap:
    raise SystemExit(
        f"STOP : {len(overlap)} doublons historiques"
    )

groups = defaultdict(list)

for row in rows:
    comp = row["competition_id"]
    if comp not in COMPETITIONS:
        raise SystemExit("Competition inattendue : " + comp)

    if row.get("protected_clearance") != "NOT_VERIFIED":
        raise SystemExit(
            "Statut de protection inattendu"
        )

    groups[comp].append(row)

pilot = []

for comp in COMPETITIONS:
    matches = groups[comp]
    by_month = defaultdict(list)

    for row in matches:
        by_month[row["month"]].append(row)

    months = sorted(by_month)

    if len(months) < MONTHS_PER_COMPETITION:
        raise SystemExit(
            "Diversite mensuelle insuffisante : " + comp
        )

    # Six positions reparties de maniere reguliere
    # dans les mois disponibles.
    indexes = sorted({
        round(
            i * (len(months) - 1)
            / (MONTHS_PER_COMPETITION - 1)
        )
        for i in range(MONTHS_PER_COMPETITION)
    })

    if len(indexes) != MONTHS_PER_COMPETITION:
        raise SystemExit("Selection de mois incorrecte")

    for index in indexes:
        month = months[index]

        choices = sorted(
            by_month[month],
            key=lambda r: (
                r["kickoff_utc"],
                r["match_id"]
            )
        )

        # Selection deterministe sans consulter les labels.
        chosen = choices[len(choices) // 2]

        pilot.append({
            "competition_id": comp,
            "season_id": chosen["season_id"],
            "match_id": chosen["match_id"],
            "kickoff_utc": chosen["kickoff_utc"],
            "home_team": chosen["home_team"],
            "away_team": chosen["away_team"],
            "month": month,
            "selection_method": "TEMPORAL_SPREAD_6_MONTHS",
            "historical_raw_overlap": "NO",
            "protected_clearance": "NOT_VERIFIED",
            "collection_authorized": "NO",
            "label_consulted": "NO",
        })

assert len(pilot) == 12
assert len({r["match_id"] for r in pilot}) == 12

source_hash = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()

print("=" * 76)
print("ETAPE 298 - PILOTE CHRONOLOGIQUE V4.1")
print("=" * 76)

print("\n=== SOURCE ===")
print("Matchs selectionnes a l'etape 297 :", len(rows))
print("SHA256 du manifeste :", source_hash)

print("\n=== PILOTE DE 12 MATCHS ===")

for comp in COMPETITIONS:
    subset = sorted(
        [r for r in pilot if r["competition_id"] == comp],
        key=lambda r: r["kickoff_utc"]
    )

    print("\nCOMPETITION :", comp)
    print("Matchs :", len(subset))
    print("Mois :", len({r["month"] for r in subset}))
    print("Saisons :", dict(Counter(
        r["season_id"] for r in subset
    )))

    for r in subset:
        print(
            r["kickoff_utc"][:10],
            "|", r["match_id"],
            "|", r["home_team"],
            "-", r["away_team"]
        )

print("\n=== CAPACITE DE COLLECTE PAR ID ===")

path = Path("collect_historical_dataset_v2.py")
tree = ast.parse(
    path.read_text(encoding="utf-8"),
    filename=str(path)
)

for node in ast.walk(tree):
    if not isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        continue

    if node.name not in (
        "collect_one_match",
        "list_finished_matches",
    ):
        continue

    arguments = [
        a.arg for a in node.args.posonlyargs
    ] + [
        a.arg for a in node.args.args
    ] + [
        a.arg for a in node.args.kwonlyargs
    ]

    print(
        node.name,
        "| ligne", node.lineno,
        "| arguments :", arguments
    )

OUT.mkdir(parents=True, exist_ok=False)

destination = OUT / "PILOT12_METADATA_ONLY.csv"

with destination.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(pilot[0].keys())
    )
    writer.writeheader()
    writer.writerows(pilot)

print("\n=== FICHIER ===")
print(destination)

print("\nAVERTISSEMENTS")
print("- 12 matchs choisis sans labels")
print("- Repartition sur plusieurs mois")
print("- Chevauchement historique RAW : zero")
print("- Protection holdout non certifiee")
print("- Collecte non autorisee par ce manifeste")
print("- Aucun appel API")
print("- Aucun match reconstruit")
print("- TEST_FUTUR non consulte")
print("- V3-C inchange")
print("ETAPE 298 TERMINEE.")
