import csv
import json
from pathlib import Path

from build_historical_ab_research_v3 import event_minute
from shadow_b_last5_v1 import (
    calculate_b_last5,
    B_EXTRA,
)

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:
        mid = row["match_id"]

        if not mid or mid in result:
            raise ValueError(f"ID invalide : {path}")

        result[mid] = row

    return result


matches_checked = 0
feature_checks = 0
temporal_checks = 0
errors = []

print("=== ÉTAPE 131 : CALCULATEUR B EXPÉRIMENTAL ===")

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

        with (CACHE / f"{mid}.json").open(
            encoding="utf-8"
        ) as f:
            cache = json.load(f)

        payload = cache.get("data")

        if not isinstance(payload, dict):
            raise ValueError(f"Payload absent : {mid}")

        if (
            payload.get("match_id") != mid
            or payload.get("coverage") != "full"
            or not isinstance(payload.get("events"), list)
        ):
            raise ValueError(f"Cache invalide : {mid}")

        events = payload["events"]
        info = manifest[mid]

        # Parité des six variables à 87'.
        calculated = calculate_b_last5(
            events, info, decision_minute=87
        )

        for feature in B_EXTRA:
            expected = int(row[feature])
            actual = calculated[feature]

            feature_checks += 1

            if actual != expected:
                errors.append(
                    f"{mid}: {feature}, "
                    f"CSV={expected}, shadow={actual}"
                )

        # Invariance temporelle aux minutes 85, 86, 87.
        for decision_minute in (85, 86, 87):
            full_result = calculate_b_last5(
                events, info, decision_minute
            )

            snapshot = []

            for event in events:
                if not isinstance(event, dict):
                    continue

                try:
                    minute = event_minute(event)
                except (ValueError, TypeError):
                    continue

                if minute < decision_minute:
                    snapshot.append(event)

            snapshot_result = calculate_b_last5(
                snapshot, info, decision_minute
            )

            temporal_checks += 1

            if full_result != snapshot_result:
                errors.append(
                    f"{mid}: fuite temporelle "
                    f"à {decision_minute}'"
                )

        matches_checked += 1

    print(f"{tag}: 20 matchs vérifiés")

print("\n=== BILAN ===")
print("Matchs :", matches_checked)
print("Comparaisons CSV :", feature_checks)
print("Tests temporels :", temporal_checks)
print("Erreurs :", len(errors))

for error in errors[:20]:
    print("ERREUR :", error)

if (
    matches_checked != 200
    or feature_checks != 1200
    or temporal_checks != 600
    or errors
):
    raise SystemExit("AUDIT SHADOW B : ÉCHEC")

print("\nSHADOW B : PASS")
print("Aucun match TEST utilisé.")
print("Aucun appel API.")
print("Aucune modification du scanner LIVE.")
print("Aucun modèle déployé.")
