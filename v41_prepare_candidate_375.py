import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path("outputs/v4_research")

SRC_DIR = BASE / "v41_train_dataset_corrected_366"
SOURCE = SRC_DIR / "V41_TRAIN112_OWN_GOAL_CORRECTED.csv"
SOURCE_MANIFEST = SRC_DIR / "V41_OWN_GOAL_CORRECTION_MANIFEST.json"

DEST = BASE / "v41_candidate_a_375"
OUTPUT = DEST / "V41_A_TRAIN112_CORE43_PROVISIONAL.csv"
MANIFEST = DEST / "V41_A_CORE43_MANIFEST.json"

EXCLUDE = {
    "league_away_btts_freq",
    "league_away_over05_freq",
    "league_away_over25_freq",
    "league_home_btts_freq",
    "league_home_over05_freq",
    "league_home_over25_freq",
}

META = {
    "match_id",
    "competition_id",
    "season_id",
    "kickoff_utc",
    "goal_after_87",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def missing(value):
    return str(value).strip().lower() in {
        "", "nan", "none", "null", "na", "n/a"
    }

if DEST.exists():
    raise RuntimeError("STOP : dossier 375 deja existant")

with SOURCE_MANIFEST.open(encoding="utf-8") as f:
    source_manifest = json.load(f)

if sha(SOURCE) != source_manifest["corrected_file_sha256"]:
    raise RuntimeError("STOP : SHA256 source incorrect")

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)

if not columns:
    raise RuntimeError("STOP : colonnes absentes")

features = [c for c in columns if c not in META]

if len(features) != 49 or not EXCLUDE.issubset(features):
    raise RuntimeError("STOP : schema source inattendu")

selected = [c for c in features if c not in EXCLUDE]

if len(selected) != 43:
    raise RuntimeError("STOP : 43 features attendues")

if len(rows) != 112:
    raise RuntimeError("STOP : 112 matchs attendus")

if len({r["match_id"] for r in rows}) != 112:
    raise RuntimeError("STOP : IDs dupliques")

labels = Counter(r["goal_after_87"] for r in rows)

if labels != {"0": 74, "1": 38}:
    raise RuntimeError("STOP : distribution des labels modifiee")

output_columns = [
    "match_id",
    "competition_id",
    "season_id",
    "kickoff_utc",
] + selected + ["goal_after_87"]

missing_counts = {
    feature: sum(missing(r[feature]) for r in rows)
    for feature in selected
}

# Contrôle des valeurs numériques non manquantes.
for row in rows:
    for feature in selected:
        value = row[feature]
        if missing(value):
            continue
        try:
            number = float(value)
        except ValueError:
            raise RuntimeError(
                f"STOP : valeur non numerique {row['match_id']} / {feature}"
            )
        if not __import__("math").isfinite(number):
            raise RuntimeError(
                f"STOP : valeur non finie {row['match_id']} / {feature}"
            )

DEST.mkdir(parents=True, exist_ok=False)

with OUTPUT.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row[c] for c in output_columns})

manifest = {
    "model_candidate": "V4.1-A",
    "status": "PROVISIONAL_FEATURE_MATRIX",
    "source_sha256": sha(SOURCE),
    "output_sha256": sha(OUTPUT),
    "rows": 112,
    "features": selected,
    "excluded_features": sorted(EXCLUDE),
    "missing_counts": missing_counts,
    "positive_labels": 38,
    "negative_labels": 74,
    "labels_independently_certified": False,
    "temporal_availability_fully_certified": False,
    "protected_sets_fully_certified": False,
    "training_authorized": False,
}

with MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("ETAPE 375 - MATRICE CANDIDATE V4.1-A")
print("=" * 70)
print("Matchs TRAIN :", len(rows))
print("Caracteristiques :", len(selected))
print("Variables de ligue exclues :", len(EXCLUDE))
print("Positifs :", labels["1"])
print("Negatifs :", labels["0"])

print("\n=== VALEURS MANQUANTES RESTANTES ===")
for feature, count in sorted(
    missing_counts.items(),
    key=lambda item: (-item[1], item[0])
):
    if count:
        print(f"{feature}: {count}/112")

print("\nFichier :", OUTPUT)
print("Manifest :", MANIFEST)
print("SHA256 :", sha(OUTPUT))
print("Dataset original : INCHANGE")
print("Validation / HOLDOUT : NON CONSULTES")
print("Entrainement : NON AUTORISE")
print("Aucun appel API")
print("ETAPE 375 TERMINEE")
