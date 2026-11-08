import csv
import json
from pathlib import Path
from collections import Counter

from build_historical_ab_research_v3 import (
    SHOTS, SOT, match_side, event_minute
)

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

FEATURES = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

def read_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    result = {}
    for row in rows:
        mid = row["match_id"]
        if not mid or mid in result:
            raise ValueError(f"ID absent ou dupliqué : {path}")
        result[mid] = row

    return result

total_matches = 0
feature_checks = 0
csv_differences = Counter()
convention_differences = Counter()
batch_results = []
unknown_sides = 0
examples = []

print("=== ÉTAPE 130 : AUDIT B SUR 200 MATCHS ===")
print("Lots : BATCH02 à BATCH11")
print("TEST FUTUR : EXCLU")
print("Aucun appel API")

for n in range(2, 12):
    tag = f"BATCH{n:02d}"

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
        raise ValueError(f"Effectifs ou IDs invalides : {tag}")

    batch_mismatches = 0
    batch_corner_differences = 0

    for mid, row in dataset.items():

        if row.get("research_split") not in (
            "TRAIN", "VALIDATION"
        ):
            raise ValueError(
                f"Split interdit : {tag} / {mid}"
            )

        with (CACHE / f"{mid}.json").open(
            encoding="utf-8"
        ) as f:
            cache = json.load(f)

        payload = cache.get("data", {})

        if (
            payload.get("match_id") != mid
            or payload.get("coverage") != "full"
            or not isinstance(payload.get("events"), list)
        ):
            raise ValueError(f"Timeline invalide : {mid}")

        historical = Counter()
        direct = Counter()
        issues = []

        info = manifest[mid]

        for event in payload["events"]:
            if not isinstance(event, dict):
                continue

            try:
                minute = event_minute(event)
            except (ValueError, TypeError):
                continue

            if (
                event.get("period") != "second_half"
                or not (82 <= minute < 87)
            ):
                continue

            typ = str(
                event.get("type") or ""
            ).strip().lower()

            is_hist_shot = typ in SHOTS
            is_hist_sot = typ in SOT
            is_corner = typ == "corner_kick"

            # Approximation des règles de classification LIVE.
            is_direct_shot = (
                typ in {
                    "shot", "shots",
                    "attempt", "total_shot"
                }
                or "shot" in typ
            )

            is_direct_sot = (
                "shot_on_target" in typ
                or "on_target" in typ
                or "saved" in typ
            )

            if not (
                is_hist_shot or is_hist_sot
                or is_corner or is_direct_shot
                or is_direct_sot
            ):
                continue

            side = match_side(event, info)

            if side not in ("home", "away"):
                unknown_sides += 1
                issues.append(
                    f"Équipe inconnue : {typ} à {minute}'"
                )
                continue

            opposite = (
                "away" if side == "home" else "home"
            )

            # Règles historiques du modèle B.
            if is_hist_shot:
                historical[
                    f"shots_{side}_last5"
                ] += 1

            if is_hist_sot:
                historical[
                    f"sot_{side}_last5"
                ] += 1

            if is_corner:
                historical[
                    f"corners_{opposite}_last5"
                ] += 1

            # Règles directes approximant le scanner LIVE.
            if is_direct_shot:
                direct[
                    f"shots_{side}_last5"
                ] += 1

            if is_direct_sot:
                direct[
                    f"sot_{side}_last5"
                ] += 1

            if is_corner:
                direct[
                    f"corners_{side}_last5"
                ] += 1

        csv_mismatch = False
        corner_difference = False

        for feature in FEATURES:
            expected = int(row[feature])
            calculated = historical[feature]

            feature_checks += 1

            if expected != calculated:
                csv_differences[feature] += 1
                csv_mismatch = True
                issues.append(
                    f"{feature}: CSV={expected}, "
                    f"historique={calculated}"
                )

            if direct[feature] != calculated:
                convention_differences[feature] += 1

                if feature.startswith("corners_"):
                    corner_difference = True

        if csv_mismatch:
            batch_mismatches += 1

        if corner_difference:
            batch_corner_differences += 1

        if issues and len(examples) < 20:
            examples.append((mid, issues))

        total_matches += 1

    batch_results.append((
        tag,
        batch_mismatches,
        batch_corner_differences
    ))

    print(
        f"{tag} | matchs=20 | "
        f"écarts CSV={batch_mismatches} | "
        f"corners différents={batch_corner_differences}"
    )

print("\n=== BILAN GLOBAL ===")
print("Matchs :", total_matches)
print("Comparaisons CSV :", feature_checks)
print("Attributions inconnues :", unknown_sides)

print("\n=== ÉCARTS CSV / RECONSTRUCTION B ===")
for feature in FEATURES:
    print(
        feature,
        ":",
        csv_differences[feature]
    )

print("\n=== ÉCARTS DE CONVENTION DIRECTE ===")
for feature in FEATURES:
    print(
        feature,
        ":",
        convention_differences[feature]
    )

print("\n=== EXEMPLES D'ANOMALIES ===")
for mid, issues in examples:
    print(mid, "|", "; ".join(issues))

if total_matches != 200 or feature_checks != 1200:
    raise ValueError(
        "Effectifs de l'audit incomplets"
    )

if unknown_sides:
    raise ValueError(
        f"{unknown_sides} événements sans équipe identifiable"
    )

if sum(csv_differences.values()):
    raise ValueError(
        "La reconstruction B présente des divergences"
    )

print("\n=== AUDIT TERMINÉ ===")
print("Reconstruction des six variables : PASS")
print("Aucune donnée TEST utilisée")
print("Aucun fichier modifié")
print("Scanner LIVE inchangé")
print("Aucun modèle déployé")
print(
    "Attention : règles LIVE approximées, "
    "non équivalentes à un test du scanner complet."
)
