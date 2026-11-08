from pathlib import Path
import json
import pandas as pd

from v3c_pressure_features import build_features

ROOT = Path("outputs/pressure_historical")
OUT = Path("outputs/v4_research")
CACHE = ROOT / "timeline_cache"

MANIFEST = OUT / "V4_CACHED12_MANIFEST_V1.csv"
REFERENCE = OUT / "V4_CACHED12_B29_RESEARCH_V1.csv"

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

print("=" * 72)
print("ÉTAPE 212 — PARITÉ HISTORIQUE / CALCULATEUR LIVE")
print("=" * 72)

manifest = pd.read_csv(MANIFEST, dtype=str)
reference = pd.read_csv(
    REFERENCE,
    dtype={"match_id": str}
)

if (
    len(manifest) != 12
    or len(reference) != 12
    or manifest.match_id.duplicated().any()
    or reference.match_id.duplicated().any()
):
    raise SystemExit("Population CACHED12 incorrecte")

if set(manifest.match_id) != set(reference.match_id):
    raise SystemExit("Identifiants incompatibles")

reference = reference.set_index("match_id")

differences = {name: 0 for name in B6}
exact_matches = 0
errors = []

print("\n=== COMPARAISON MATCH PAR MATCH ===")

for _, match in manifest.iterrows():
    mid = match["match_id"]

    path = CACHE / f"{mid}.json"

    if not path.is_file():
        path = CACHE / f"mt_{mid}.json"

    if not path.is_file():
        errors.append(f"{mid}: cache absent")
        continue

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        result = build_features(
            payload=payload,
            match_id=mid,
            snapshot_minute=87,
            home_team=match["home_team"],
            away_team=match["away_team"],
            source="HISTORICAL_DIAGNOSTIC_ONLY",
            home_id=(
                match["home_id"]
                if pd.notna(match.get("home_id"))
                else None
            ),
            away_id=(
                match["away_id"]
                if pd.notna(match.get("away_id"))
                else None
            ),
            schema_confirmed=False,
        )

        mismatch = []

        for feature in B6:
            expected = float(reference.loc[mid, feature])
            actual = result.get(feature)

            if actual is None:
                mismatch.append(
                    f"{feature}: attendu={expected:g}, "
                    "obtenu=NULL"
                )
                differences[feature] += 1
                continue

            actual = float(actual)

            if abs(expected - actual) > 1e-9:
                differences[feature] += 1
                mismatch.append(
                    f"{feature}: attendu={expected:g}, "
                    f"obtenu={actual:g}"
                )

        if mismatch:
            print(f"\nDIFFÉRENCES {mid}")

            for item in mismatch:
                print("  ", item)
        else:
            exact_matches += 1
            print(f"IDENTIQUE {mid}")

        print(
            "  status:",
            result.get("pressure_status")
        )

    except Exception as exc:
        errors.append(f"{mid}: {exc}")
        print(f"ERREUR {mid}: {exc}")

print("\n=== SYNTHÈSE ===")
print("Matchs comparés :", 12 - len(errors))
print("Matchs identiques sur B6 :", exact_matches)
print("Erreurs de lecture ou de calcul :", len(errors))

print("\n=== ÉCARTS PAR CARACTÉRISTIQUE ===")

for feature, count in differences.items():
    print(f"{feature:25s} : {count}")

if errors:
    print("\n=== ERREURS ===")
    for error in errors:
        print(error)

print("\n=== CONCLUSION ===")

if exact_matches == 12 and not errors:
    print("Parité B6 observée sur les 12 matchs.")
    print("La compatibilité LIVE reste à prouver.")
else:
    print("Parité B6 non démontrée.")
    print("Adaptateur V4 spécifique nécessaire.")

print("\n=== GARANTIES ===")
print("Test historique hors ligne uniquement.")
print("Aucun appel API.")
print("Aucun label utilisé.")
print("Aucun dataset modifié.")
print("Aucun modèle réentraîné.")
print("Scanner V3-C inchangé.")
