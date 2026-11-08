import csv
import json
import importlib.util
from pathlib import Path
from collections import Counter
from decimal import Decimal, InvalidOperation

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
BUILDER = Path("build_historical_ab_research_v3.py")
SPLIT_PLAN = JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"

SOURCES = [
    ("ORIGINAL_320", 320,
     "ALL_320_A_SCORE_CORRECTED_V3_RESEARCH.csv",
     "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv"),
    ("EXTRA_5", 5,
     "EXTRA_5_TRAIN_VAL_A_RESEARCH_V1.csv",
     "EXTRA_5_TRAIN_VAL_B_RESEARCH_V1.csv"),
    ("PILOT_20", 20,
     "PILOT_20_A_RESEARCH_V1.csv",
     "PILOT_20_B_RESEARCH_V1.csv"),
]

for number in range(2, 12):
    tag = f"BATCH{number:02d}"
    SOURCES.append((
        tag, 20,
        f"{tag}_A_RESEARCH_V1.csv",
        f"{tag}_B_RESEARCH_V1.csv"
    ))

EXPECTED_COUNTS = {
    "TRAIN": 195,
    "VALIDATION": 167,
    "TEST_FUTUR": 183
}

EXPECTED_POSITIVES = {
    "TRAIN": 53,
    "VALIDATION": 42,
    "TEST_FUTUR": 55
}

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output = {}

    for row in rows:
        mid = row["match_id"]

        if not mid or mid in output:
            raise ValueError(f"ID absent ou doublon : {path}, {mid}")

        output[mid] = row

    return output

def number(value, context):
    try:
        d = Decimal(str(value))
        if not d.is_finite():
            raise ValueError("Non finite")
        return d
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(
            f"Valeur numérique invalide : {context} = {value!r}"
        )

spec = importlib.util.spec_from_file_location(
    "historical_builder_v3", BUILDER
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

FEATURES_A = list(builder.A_FEATURES)
FEATURES_B = FEATURES_A + list(builder.B_EXTRA)

if len(FEATURES_A) != 23 or len(FEATURES_B) != 29:
    raise ValueError("Dimensions A/B invalides")

split_plan = read_csv(SPLIT_PLAN)

if len(split_plan) != 320:
    raise ValueError("Plan original de 320 matchs attendu")

all_ids = set()
groups = {
    "TRAIN": {"A": [], "B": []},
    "VALIDATION": {"A": [], "B": []},
    "TEST_FUTUR": {"A": [], "B": []}
}

manifest_rows = []
counts = Counter()
positives = Counter()

for origin, expected, file_a, file_b in SOURCES:
    data_a = read_csv(JOIN / file_a)
    data_b = read_csv(JOIN / file_b)

    if len(data_a) != expected or set(data_a) != set(data_b):
        raise ValueError(f"Sources A/B incohérentes : {origin}")

    for mid in sorted(data_a):
        if mid in all_ids:
            raise ValueError(f"Doublon entre sources : {mid}")

        all_ids.add(mid)

        a = data_a[mid]
        b = data_b[mid]

        if a.get("kickoff_utc") != b.get("kickoff_utc"):
            raise ValueError(f"Kickoff A/B différent : {mid}")

        if origin == "ORIGINAL_320":
            if mid not in split_plan:
                raise ValueError(f"Match absent du split : {mid}")

            group = split_plan[mid]["split"]

            if a["kickoff_utc"] != split_plan[mid]["kickoff_utc"]:
                raise ValueError(f"Kickoff/split différent : {mid}")
        else:
            group = a.get("research_split")

            if group != b.get("research_split"):
                raise ValueError(f"Split A/B différent : {mid}")

        if group not in groups:
            raise ValueError(f"Split invalide : {mid}: {group}")

        label_a = number(
            a.get("goal_after_87"),
            f"{mid}:A:label"
        )
        label_b = number(
            b.get("goal_after_87"),
            f"{mid}:B:label"
        )

        if label_a != label_b or label_a not in (0, 1):
            raise ValueError(f"Label A/B invalide : {mid}")

        label = int(label_a)

        for feature in FEATURES_A:
            va = number(
                a.get(feature),
                f"{mid}:A:{feature}"
            )
            vb = number(
                b.get(feature),
                f"{mid}:B:{feature}"
            )

            if va != vb:
                raise ValueError(
                    f"Variable commune A/B différente : "
                    f"{mid}, {feature}"
                )

        for feature in FEATURES_B:
            number(
                b.get(feature),
                f"{mid}:B:{feature}"
            )

        source_quality = (
            a.get("v3_quality_gate")
            or a.get("research_quality_gate")
            or "UNKNOWN"
        )

        common = {
            "match_id": mid,
            "kickoff_utc": a["kickoff_utc"],
            "goal_after_87": label,
            "research_split": group,
            "research_origin": origin,
            "source_quality_gate": source_quality,
            "independent_verified": 0,
            "live_certified": 0,
            "training_certified": 0
        }

        groups[group]["A"].append({
            **common,
            **{k: a[k] for k in FEATURES_A}
        })

        groups[group]["B"].append({
            **common,
            **{k: b[k] for k in FEATURES_B}
        })

        manifest_rows.append(common)

        counts[group] += 1
        positives[group] += label

print("\n=== ÉTAPE 118 : CONSOLIDATION ===")

for group in EXPECTED_COUNTS:
    print(
        group,
        "| matchs:", counts[group],
        "| positifs:", positives[group]
    )

    if counts[group] != EXPECTED_COUNTS[group]:
        raise ValueError(f"Effectif incorrect : {group}")

    if positives[group] != EXPECTED_POSITIVES[group]:
        raise ValueError(f"Labels positifs incorrects : {group}")

if len(all_ids) != 545:
    raise ValueError(
        f"545 IDs attendus, {len(all_ids)} obtenus"
    )

if sum(positives.values()) != 150:
    raise ValueError("150 positifs attendus")

outputs = []

for group in groups:
    for model in ("A", "B"):
        outputs.append((
            JOIN / f"V3C_545_{model}_{group}_RESEARCH_V1.csv",
            groups[group][model]
        ))

outputs.append((
    JOIN / "V3C_545_CONSOLIDATION_MANIFEST_V1.csv",
    manifest_rows
))

summary_path = JOIN / "V3C_545_CONSOLIDATION_SUMMARY_V1.json"

# Vérifier toutes les destinations avant toute écriture.
for path, _ in outputs:
    if path.exists():
        raise FileExistsError(path)

if summary_path.exists():
    raise FileExistsError(summary_path)

def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Fichier vide : {path}")

    rows = sorted(
        rows,
        key=lambda r: (
            r["kickoff_utc"],
            r["match_id"]
        )
    )

    fields = list(rows[0])

    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

for path, rows in outputs:
    save_csv(path, rows)

summary = {
    "version": "V1_RESEARCH_ONLY",
    "total_matches": len(all_ids),
    "total_positives": sum(positives.values()),
    "features_a": FEATURES_A,
    "features_b": FEATURES_B,
    "counts": dict(counts),
    "positives": dict(positives),
    "independent_verified": False,
    "live_certified": False,
    "training_certified": False,
    "test_future_is_separate": True
}

with summary_path.open("x", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n=== VALIDATION FINALE ===")
print("Matchs uniques :", len(all_ids))
print("Labels positifs :", sum(positives.values()))
print("Variables A :", len(FEATURES_A))
print("Variables B :", len(FEATURES_B))
print("Doublons : 0")
print("Divergences A/B : 0")
print("TEST FUTUR séparé : OUI")

print("\n=== FICHIERS CRÉÉS ===")
for path, _ in outputs:
    print(path)

print(summary_path)

print("\nAucun appel API.")
print("Aucun entraînement.")
print("Aucun dataset existant modifié.")
print("Aucune certification indépendante accordée.")
