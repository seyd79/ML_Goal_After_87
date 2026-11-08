import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

JOIN = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)
CACHE = Path("outputs/pressure_historical/timeline_cache")
MANIFEST = Path("historical_1710_manifest.csv")
PLAN = JOIN / "HISTORICAL_COLLECTION_WAVE1_PLAN_V1.csv"

# 8 lots de 9 TRAIN + 11 VALIDATION,
# puis 1 lot de 8 TRAIN + 12 VALIDATION.
SIZES = [(9, 11)] * 8 + [(8, 12)]

def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    ids = [r["match_id"] for r in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(f"Doublons : {path}")

    return fields, rows

fields, manifest_rows = load(MANIFEST)
_, plan_rows = load(PLAN)

manifest = {r["match_id"]: r for r in manifest_rows}

if len(manifest) != 1710 or len(plan_rows) != 220:
    raise ValueError("Manifeste ou plan incorrect")

remaining = {
    "TRAIN": defaultdict(list),
    "VALIDATION": defaultdict(list)
}

cached = Counter()

for row in plan_rows:
    mid = row["match_id"]
    group = row["target_split"]
    day = row["day_utc"]

    if mid not in manifest:
        raise ValueError(f"Match absent : {mid}")

    if group not in remaining:
        raise ValueError(f"Split interdit : {mid}")

    if (CACHE / f"{mid}.json").exists():
        cached[group] += 1
    else:
        remaining[group][day].append(mid)

if cached != Counter({
    "TRAIN": 20,
    "VALIDATION": 20
}):
    raise ValueError(
        f"40 caches du plan attendus, obtenus : {cached}"
    )

def stable_key(mid):
    return hashlib.sha256(
        ("V3C_WAVE1_REMAINING|" + mid).encode()
    ).hexdigest()

def balanced_order(group):
    by_day = remaining[group]
    days = sorted(by_day)

    for day in days:
        by_day[day].sort(key=stable_key)

    ordered = []

    while any(by_day.values()):
        for day in days:
            if by_day[day]:
                ordered.append(by_day[day].pop(0))

    return ordered

train = balanced_order("TRAIN")
validation = balanced_order("VALIDATION")

if len(train) != 80 or len(validation) != 100:
    raise ValueError(
        f"Attendu 80 TRAIN + 100 VALIDATION ; "
        f"obtenu {len(train)} + {len(validation)}"
    )

targets = [
    JOIN / (
        f"HISTORICAL_WAVE1_BATCH{number:02d}"
        "_MANIFEST_V1.csv"
    )
    for number in range(3, 12)
]

for path in targets:
    if path.exists():
        raise FileExistsError(path)

used = set()
offset_train = 0
offset_validation = 0

for index, (n_train, n_val) in enumerate(SIZES):
    number = index + 3

    selected_train = train[
        offset_train:offset_train + n_train
    ]
    selected_val = validation[
        offset_validation:offset_validation + n_val
    ]

    offset_train += n_train
    offset_validation += n_val

    ids = selected_train + selected_val

    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError(f"Lot incorrect : {number}")

    if used.intersection(ids):
        raise ValueError("Doublon entre les lots")

    used.update(ids)

    rows = [manifest[mid] for mid in ids]
    rows.sort(
        key=lambda r: (
            r["kickoff_utc"],
            r["match_id"]
        )
    )

    path = targets[index]

    with path.open(
        "x", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"BATCH{number:02d}",
        "| TRAIN:", n_train,
        "| VALIDATION:", n_val,
        "| TOTAL:", len(rows),
        "| fichier:", path
    )

assert len(used) == 180
assert offset_train == 80
assert offset_validation == 100

print("\n=== ÉTAPE 106 VALIDÉE ===")
print("Lots créés :", len(targets))
print("Matchs planifiés :", len(used))
print("TRAIN restant :", offset_train)
print("VALIDATION restante :", offset_validation)
print("TEST FUTUR : 0")
print("Aucun appel API.")
print("Aucun entraînement.")
