from pathlib import Path
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"
MANIFEST = Path("historical_1710_manifest.csv")

print("=" * 72)
print("ÉTAPE 192 — INVENTAIRE DES MATCHS SUPPLÉMENTAIRES V4")
print("=" * 72)

manifest = pd.read_csv(
    MANIFEST,
    dtype={"match_id": str},
    usecols=["match_id", "kickoff_utc"],
)

if len(manifest) != 1710:
    raise SystemExit("Manifeste initial inattendu")

if manifest.match_id.duplicated().any():
    raise SystemExit("Doublons dans le manifeste")

manifest["kickoff"] = pd.to_datetime(
    manifest.kickoff_utc, utc=True, errors="coerce"
)

if manifest.kickoff.isna().any():
    raise SystemExit("Dates invalides")

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

if len(plan) != 320:
    raise SystemExit("Plan ALL_320 incorrect")

existing = set(plan.match_id)
test_reserved = set(
    plan.loc[plan["split"] == "TEST_FUTUR", "match_id"]
)

for path in sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv")):
    part = pd.read_csv(
        path,
        dtype={"match_id": str},
        usecols=["match_id"],
    )

    if existing.intersection(set(part.match_id)):
        raise SystemExit("Chevauchement des datasets")

    existing.update(part.match_id)

if len(existing) != 520 or len(test_reserved) != 183:
    raise SystemExit("Effectifs historiques inattendus")

start_validation = pd.Timestamp(
    "2026-08-25T00:00:00Z"
)

end_validation = pd.Timestamp(
    "2026-09-12T00:00:00Z"
)

manifest["already_used"] = manifest.match_id.isin(existing)
manifest["test_reserved"] = manifest.match_id.isin(test_reserved)

manifest["cache_available"] = manifest.match_id.map(
    lambda mid: (CACHE / f"{mid}.json").is_file()
    or (CACHE / f"mt_{mid}.json").is_file()
)

def period(timestamp):
    if timestamp < start_validation:
        return "AVANT_VALIDATION"
    if timestamp < end_validation:
        return "PENDANT_VALIDATION"
    return "APRES_VALIDATION"

manifest["period"] = manifest.kickoff.map(period)

print("\n=== INVENTAIRE GLOBAL ===")
print("Matchs du manifeste :", len(manifest))
print("Matchs déjà utilisés :", len(existing))
print("TEST_FUTUR réservés :", len(test_reserved))

print("\n=== DISPONIBILITÉ PAR PÉRIODE ===")

for name, group in manifest.groupby("period", sort=True):
    unused = group[~group.already_used]
    cached_unused = unused[unused.cache_available]

    print(f"\n{name}")
    print("Matchs dans le manifeste :", len(group))
    print("Déjà dans les 520 :", int(group.already_used.sum()))
    print("Non utilisés :", len(unused))
    print("Non utilisés avec cache :", len(cached_unused))
    print(
        "Non utilisés sans cache :",
        int((~unused.cache_available).sum())
    )

print("\n=== CANDIDATS TRAIN CHRONOLOGIQUES ===")

candidates = manifest[
    (manifest.period == "AVANT_VALIDATION")
    & (~manifest.already_used)
    & (~manifest.test_reserved)
].copy()

candidates["day"] = candidates.kickoff.dt.strftime("%Y-%m-%d")

print("Candidats potentiels :", len(candidates))
print(
    "Avec cache existant :",
    int(candidates.cache_available.sum())
)
print(
    "Sans cache :",
    int((~candidates.cache_available).sum())
)

print("\nRépartition par journée :")

if len(candidates):
    print(
        candidates.groupby("day")
        .agg(
            matchs=("match_id", "size"),
            caches=("cache_available", "sum"),
        )
        .to_string()
    )
else:
    print("Aucun candidat supplémentaire.")

print("\n=== EXEMPLES DE CANDIDATS ===")

if len(candidates):
    print(
        candidates.sort_values("kickoff")[
            ["match_id", "kickoff_utc", "cache_available"]
        ].head(15).to_string(index=False)
    )

print("\n=== BILAN ===")
print("Inventaire uniquement.")
print("Aucun label lu.")
print("Aucun appel API.")
print("Aucun dataset modifié.")
print("Aucun entraînement.")
print("TEST_FUTUR non évalué.")
print("Scanner V3-C inchangé.")
