import csv
import json
from pathlib import Path

from build_historical_ab_research_v3 import event_minute
from shadow_b29_v1 import calculate_b29, FEATURES

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:
        mid = row["match_id"]

        if not mid or mid in result:
            raise ValueError(f"ID invalide : {path}")

        result[mid] = row

    return result


matches = 0
feature_checks = 0
temporal_checks = 0
errors = []

print("=== ÉTAPE 132 : SHADOW B COMPLET ===")
print("Variables :", len(FEATURES))
print("TEST FUTUR : NON UTILISÉ")

for number in range(2, 12):
    tag = f"BATCH{number:02d}"

    manifest = read_csv(
        JOIN / f"HISTORICAL_WAVE1_{tag}_MANIFEST_V1.csv"
    )

    dataset = read_csv(
        JOIN / f"{tag}_B_RESEARCH_V1.csv"
    )

    if (
        len(manifest) != 20
        or len(dataset) != 20
        or set(manifest) != set(dataset)
    ):
        raise ValueError(f"Lot incorrect : {tag}")

    for mid, row in dataset.items():

        if row.get("research_split") not in (
            "TRAIN", "VALIDATION"
        ):
            raise ValueError(f"Split interdit : {mid}")

        path = CACHE / f"{mid}.json"

        with path.open(encoding="utf-8") as f:
            cache = json.load(f)

        payload = cache.get("data", {})

        if (
            payload.get("match_id") != mid
            or payload.get("coverage") != "full"
            or not isinstance(payload.get("events"), list)
        ):
            raise ValueError(f"Cache incorrect : {mid}")

        events = payload["events"]
        info = manifest[mid]

        # Vérifier les 29 variables à 87'.
        calculated = calculate_b29(
            events, info, decision_minute=87
        )

        for feature in FEATURES:
            expected = int(row[feature])
            actual = int(calculated[feature])

            feature_checks += 1

            if expected != actual:
                errors.append(
                    f"{mid}: {feature}, "
                    f"CSV={expected}, shadow={actual}"
                )

        # Contrôler l'exclusion des événements futurs.
        for decision_minute in (85, 86, 87):

            full = calculate_b29(
                events, info, decision_minute
            )

            snapshot = []

            for event in events:
                try:
                    minute = event_minute(event)
                except (ValueError, TypeError):
                    continue

                if minute < decision_minute:
                    snapshot.append(event)

            restricted = calculate_b29(
                snapshot, info, decision_minute
            )

            temporal_checks += 1

            if full != restricted:
                errors.append(
                    f"{mid}: divergence temporelle "
                    f"à {decision_minute}'"
                )

        matches += 1

    print(f"{tag}: 20 matchs analysés")

print("\n=== BILAN ===")
print("Matchs contrôlés :", matches)
print("Variables :", len(FEATURES))
print("Comparaisons CSV :", feature_checks)
print("Tests temporels :", temporal_checks)
print("Erreurs :", len(errors))

for error in errors[:20]:
    print("ERREUR :", error)

if (
    matches != 200
    or feature_checks != 5800
    or temporal_checks != 600
    or errors
):
    raise SystemExit("SHADOW B29 : ÉCHEC")

print("\nSHADOW B29 : PASS")
print("Aucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Scanner original inchangé.")
print("TEST FUTUR non utilisé.")
print("Validation LIVE non accordée.")
