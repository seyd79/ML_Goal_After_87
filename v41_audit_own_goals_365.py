import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch
import v41_live87_own_goal_364 as corrected
from v41_targeted_collector_328 import load_live87

BASE = Path("outputs/v4_research/v41_train_dataset_358")
DATA = BASE / "V41_TRAIN112_LABELED_PROVISIONAL.csv"
REPORT = BASE / "V41_TRAIN112_PROVENANCE.json"

OUT = Path("outputs/v4_research/v41_own_goal_audit_365")
CHECKPOINTS = OUT / "checkpoints"

FIELDS = [
    "score_87_home",
    "score_87_away",
    "goal_diff_87",
    "is_draw_87",
    "leading_team_87",
]

def numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def equal(a, b):
    x, y = numeric(a), numeric(b)
    return x == y if x is not None or y is not None else str(a) == str(b)

with REPORT.open(encoding="utf-8") as f:
    provenance = json.load(f)

if hashlib.sha256(DATA.read_bytes()).hexdigest() != provenance["train_dataset_sha256"]:
    raise RuntimeError("STOP : dataset TRAIN modifie")

with DATA.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 112
assert len({r["match_id"] for r in rows}) == 112
assert Counter(r["goal_after_87"] for r in rows) == {"0": 74, "1": 38}

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

lab = prematch.load_lab()
client = prematch.build_rate_limited_client(lab)
original = load_live87()

print("=" * 72)
print("ETAPE 365 - AUDIT OWN_GOAL SUR 112 MATCHS TRAIN")
print("=" * 72)

for i, row in enumerate(rows, 1):
    mid = row["match_id"]
    checkpoint = CHECKPOINTS / (mid + ".json")

    if checkpoint.exists():
        print(f"[{i}/112] DEJA AUDITE : {mid}")
        continue

    detail = prematch.get_match_detail(lab, client, mid)

    if not detail or str(lab.extract_match_id(detail)) != mid:
        raise RuntimeError(f"STOP : detail API invalide : {mid}")

    timeline, shots = collector.fetch_timeline_and_shotmap(
        original, client, mid
    )

    before, _ = collector.reconstruct_live87(
        original, detail, timeline, shots
    )

    after, _ = collector.reconstruct_live87(
        corrected, detail, timeline, shots
    )

    final_home, final_away = original.extract_final_score(detail)

    differences = {
        field: {
            "dataset": row.get(field),
            "original": before.get(field),
            "corrected": after.get(field),
        }
        for field in FIELDS
        if not equal(before.get(field), after.get(field))
    }

    frozen_mismatch = [
        field for field in FIELDS
        if not equal(row.get(field), before.get(field))
    ]

    own_goals = [
        {
            "minute": corrected.extract_event_minute(event),
            "team": event.get("team"),
        }
        for event in timeline
        if isinstance(event, dict)
        and str(event.get("type", "")).lower() == "own_goal"
        and corrected.extract_event_minute(event) is not None
        and corrected.extract_event_minute(event) <= 87
    ]

    h87 = numeric(after.get("score_87_home"))
    a87 = numeric(after.get("score_87_away"))
    fh = numeric(final_home)
    fa = numeric(final_away)

    monotone = (
        None not in (h87, a87, fh, fa)
        and fh >= h87
        and fa >= a87
    )

    label_agrees = None
    if monotone:
        expected = int((fh - h87) + (fa - a87) > 0)
        label_agrees = expected == int(row["goal_after_87"])

    result = {
        "match_id": mid,
        "split": "TRAIN_PROVISIONAL",
        "label_original": int(row["goal_after_87"]),
        "own_goals_le87": own_goals,
        "changed_fields": differences,
        "frozen_vs_original_mismatch": frozen_mismatch,
        "final_score": [final_home, final_away],
        "corrected_score_87": [h87, a87],
        "score_monotone": monotone,
        "label_score_agrees": label_agrees,
        "status": "AUDIT_ONLY_NOT_CERTIFIED",
    }

    with checkpoint.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    if differences or frozen_mismatch or not monotone or label_agrees is False:
        print(
            f"[{i}/112] A EXAMINER : {mid}"
            f" | own_goals={len(own_goals)}"
            f" | champs_modifies={len(differences)}"
            f" | decalage_dataset={len(frozen_mismatch)}"
            f" | score_coherent={monotone}"
            f" | label_coherent={label_agrees}"
        )
    else:
        print(f"[{i}/112] OK : {mid}")

reports = []
for row in rows:
    path = CHECKPOINTS / (row["match_id"] + ".json")
    if path.exists():
        with path.open(encoding="utf-8") as f:
            reports.append(json.load(f))

changed = [r for r in reports if r["changed_fields"]]
drift = [r for r in reports if r["frozen_vs_original_mismatch"]]
non_monotone = [r for r in reports if not r["score_monotone"]]
label_disagreements = [
    r for r in reports if r["label_score_agrees"] is False
]

print("\n" + "=" * 72)
print("BILAN ETAPE 365")
print("=" * 72)
print("Matchs audites :", len(reports), "/112")
print("Matchs avec own_goal <=87 :", sum(bool(r["own_goals_le87"]) for r in reports))
print("Matchs avec caracteristiques modifiees :", len(changed))
print("Matchs avec decalage du dataset original :", len(drift))
print("Scores non monotones apres correction :", len(non_monotone))
print("Desaccords label / score :", len(label_disagreements))

for title, group in [
    ("CARACTERISTIQUES MODIFIEES", changed),
    ("DECALAGES AVEC DATASET FIGE", drift),
    ("SCORES NON MONOTONES", non_monotone),
    ("DESACCORDS DE LABELS", label_disagreements),
]:
    print("\n" + title + " :")
    for r in group:
        print(" ", r["match_id"])

print("\nResultats :", CHECKPOINTS)
print("Dataset fige et labels : INCHANGES")
print("Validation / HOLDOUT : NON CONSULTES")
print("Entrainement : NON AUTORISE")
print("ETAPE 365 TERMINEE.")
