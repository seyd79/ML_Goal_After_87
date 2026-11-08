from pathlib import Path
import pandas as pd

ROOT = Path("outputs/pressure_historical")
OUT = Path("outputs/v4_research")

PLAN = OUT / "V4_TRAIN_EXPANSION_PLAN_V1.csv"
MANIFEST = Path("historical_1710_manifest.csv")
CACHE = ROOT / "timeline_cache"

print("=" * 72)
print("ÉTAPE 194 — MANIFESTS DE COLLECTE V4")
print("=" * 72)

plan = pd.read_csv(PLAN, dtype={"match_id": str})
manifest = pd.read_csv(MANIFEST, dtype={"match_id": str})

if len(plan) != 95 or plan.match_id.duplicated().any():
    raise SystemExit("Plan V4 incorrect")

if len(manifest) != 1710 or manifest.match_id.duplicated().any():
    raise SystemExit("Manifeste historique incorrect")

required = [
    "match_id",
    "home_team",
    "away_team",
    "home_id",
    "away_id",
    "kickoff_utc",
]

if not set(required).issubset(manifest.columns):
    raise SystemExit("Colonnes du manifeste insuffisantes")

joined = plan[["match_id"]].merge(
    manifest[required],
    on="match_id",
    how="left",
    validate="one_to_one",
)

if joined["kickoff_utc"].isna().any():
    raise SystemExit("Matchs absents du manifeste")

dates = pd.to_datetime(
    joined["kickoff_utc"],
    utc=True,
    errors="raise"
)

cutoff = pd.Timestamp("2026-08-25T00:00:00Z")

if not (dates < cutoff).all():
    raise SystemExit("Un match est postérieur à la limite TRAIN")

def has_cache(mid):
    return (
        (CACHE / f"{mid}.json").is_file()
        or (CACHE / f"mt_{mid}.json").is_file()
    )

joined["cached"] = joined["match_id"].map(has_cache)

ready = joined[joined.cached].copy()
missing = joined[~joined.cached].copy()

if len(ready) != 12 or len(missing) != 83:
    raise SystemExit(
        f"Répartition inattendue : "
        f"{len(ready)} caches, {len(missing)} manquants"
    )

path12 = OUT / "V4_CACHED12_MANIFEST_V1.csv"
path83 = OUT / "V4_FETCH83_MANIFEST_V1.csv"

if path12.exists() or path83.exists():
    raise SystemExit(
        "Un fichier de sortie existe déjà. "
        "Aucun remplacement effectué."
    )

ready[required].to_csv(path12, index=False)
missing[required].to_csv(path83, index=False)

print("\n=== MANIFEST DES 12 MATCHS DISPONIBLES ===")
print("Matchs :", len(ready))
print("Fichier :", path12)

print("\n=== MANIFEST DES 83 MATCHS À COLLECTER ===")
print("Matchs :", len(missing))
print("Fichier :", path83)

print("\n=== RÉPARTITION PAR JOUR ===")

for label, group in [
    ("CACHED12", ready),
    ("FETCH83", missing),
]:
    print("\n", label)
    days = pd.to_datetime(
        group["kickoff_utc"], utc=True
    ).dt.strftime("%Y-%m-%d")

    print(days.value_counts().sort_index().to_string())

print("\n=== VÉRIFICATION DES IDs ===")
print("Doublons entre les deux lots :",
      len(set(ready.match_id) & set(missing.match_id)))

print("\n=== BILAN ===")
print("Deux manifests V4 créés.")
print("Aucun appel API.")
print("Aucun label consulté.")
print("Aucun dataset existant modifié.")
print("VALIDATION et TEST_FUTUR préservés.")
print("Scanner LIVE inchangé.")
print("ÉTAPE 194 TERMINÉE.")
