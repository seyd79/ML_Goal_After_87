import csv
import json
import math
import hashlib
from pathlib import Path
from collections import Counter

BASE = Path("outputs/v4_research")
RAW = BASE / "v41_mass350/raw_timelines"
MATRIX = BASE / "v41_mass350_blind_features_404K/V41_MASS350_FEATURES_BLIND_PREPARATION.csv"
PILOT = BASE / "v41_prematch_pilot_404S/V41_PREMATCH_PILOT20_NO_TARGET_LABEL.csv"
CHECKPOINTS = BASE / "v41_prematch_collection_404T/checkpoints"
OUT = BASE / "v41_mass350_final_features_404W"

EXPECTED = {
    MATRIX: "0b1d4155b5dd70655b64721c4ca70524eb545f5e0bacdadac95a1f0ed7278074",
    PILOT: "47507f812a52d9054f5988e5f03fc71c18afd1a28810235cc9280da9225e1b51",
}

LIVE = [
    "score_87_home", "score_87_away",
    "shots_80_87", "sot_80_87", "corners_80_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "minutes_since_last_shot_on_target",
]
PRE = ["home_avg_goals_scored", "away_avg_goals_scored"]
FIELDS = ["match_id", "kickoff_utc"] + LIVE + PRE
META = ["match_id", "team_id", "side", "kickoff_utc",
        "history_raw_count", "history_before_count",
        "avg_goals_scored", "method"]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def read(path, required):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise RuntimeError("STOP : schema incorrect " + str(path))
        return [{k: r.get(k, "") for k in required} for r in reader]

def number(x):
    if str(x).strip() == "":
        return None
    v = float(x)
    if not math.isfinite(v) or v < 0:
        raise RuntimeError("STOP : valeur numerique invalide")
    return v

for path, digest in EXPECTED.items():
    if sha(path) != digest:
        raise SystemExit("STOP : empreinte modifiee " + str(path))

if OUT.exists():
    raise SystemExit("STOP : dossier 404W deja existant")

original = read(MATRIX, FIELDS)
if len(original) != 350 or len({r["match_id"] for r in original}) != 350:
    raise SystemExit("STOP : matrice 404K invalide")

by_id = {r["match_id"]: r.copy() for r in original}
timeline = {}
files = sorted(RAW.glob("mt_*.json"))
if len(files) != 350:
    raise SystemExit("STOP : nombre de timelines incorrect")

for path in files:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("match_id") != path.stem or path.stem not in by_id:
        raise SystemExit("STOP : timeline incoherente")
    timeline[path.stem] = obj

pilot = read(PILOT, META)
checkpoints = sorted(CHECKPOINTS.glob("tm_*.csv"))
if len(checkpoints) != 505:
    raise SystemExit("STOP : 505 checkpoints attendus")

observations = list(pilot)
for path in checkpoints:
    rows = read(path, META)
    if not rows or any(r["team_id"] + ".csv" != path.name for r in rows):
        raise SystemExit("STOP : checkpoint invalide")
    observations.extend(rows)

if len(observations) != 676:
    raise SystemExit("STOP : 676 observations attendues")

used = set()
teams = set()
missing = []

for r in observations:
    mid = r["match_id"]
    side = r["side"]
    tid = r["team_id"]
    key = (mid, side)

    if key in used or mid not in timeline or side not in ("home", "away"):
        raise SystemExit("STOP : observation inconnue ou en double")
    used.add(key)
    teams.add(tid)

    obj = timeline[mid]
    row = by_id[mid]
    if str(obj[side + "_id"]) != tid:
        raise SystemExit("STOP : equipe incorrecte")
    if r["kickoff_utc"] != str(obj["kickoff_utc"]):
        raise SystemExit("STOP : date incorrecte")
    if r["method"] != "ORIGINAL_SINGLE_PAGE_100_ASOF":
        raise SystemExit("STOP : methode inattendue")

    count_raw = int(r["history_raw_count"])
    count_before = int(r["history_before_count"])
    if not (0 <= count_before <= count_raw <= 100):
        raise SystemExit("STOP : historique incoherent")

    field = side + "_avg_goals_scored"
    if str(row[field]).strip():
        raise SystemExit("STOP : tentative d'ecraser une valeur originale")

    value = number(r["avg_goals_scored"])
    if value is None:
        missing.append((mid, side, tid))
        row[field] = ""
    else:
        row[field] = str(value)

if len(teams) != 510 or len(used) != 676:
    raise SystemExit("STOP : couverture incomplete")

full = partial = live_incomplete = 0
missing_features = Counter()

for mid, row in by_id.items():
    if row["kickoff_utc"] != str(timeline[mid]["kickoff_utc"]):
        raise SystemExit("STOP : date matrice / timeline differente")

    values = {f: number(row[f]) for f in LIVE + PRE}

    for f, v in values.items():
        if v is None:
            missing_features[f] += 1

    if not all(values[f] is not None for f in LIVE):
        live_incomplete += 1
    elif all(values[f] is not None for f in PRE):
        full += 1
    else:
        partial += 1

if (full, partial, live_incomplete) != (344, 2, 4):
    raise SystemExit(
        f"STOP : couverture inattendue {full}/{partial}/{live_incomplete}"
    )

OUT.mkdir(parents=True, exist_ok=False)
result = OUT / "V41_MASS350_FINAL_FEATURES_BLIND.csv"
manifest = OUT / "V41_MASS350_FINAL_FEATURES_MANIFEST.json"

with result.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(by_id[mid] for mid in sorted(by_id))

report = {
    "status": "EXPLORATORY_FEATURES_FROZEN_NOT_CERTIFIED",
    "matches": 350,
    "complete_10_of_10": full,
    "prematch_partial_with_live_complete": partial,
    "live_incomplete": live_incomplete,
    "teams_collected": len(teams),
    "prematch_observations": len(observations),
    "missing_by_feature": dict(missing_features),
    "missing_prematch_observations": [
        {"match_id": mid, "side": side, "team_id": tid}
        for mid, side, tid in missing
    ],
    "prematch_method": "ORIGINAL_SINGLE_PAGE_100_ASOF",
    "prematch_temporal_certified": False,
    "live_equivalence_certified": False,
    "fully_independent_holdout_certified": False,
    "predictions_generated": False,
    "source_sha256": {str(p): sha(p) for p in EXPECTED},
    "checkpoint_sha256": {
        p.name: sha(p) for p in checkpoints
    },
    "output_sha256": sha(result),
}

with manifest.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=" * 74)
print("404W — MATRICE MASS350 FIGEE")
print("=" * 74)
print("Matchs :", len(by_id))
print("10/10 complets :", full)
print("Prematch partiel :", partial)
print("Live incomplet :", live_incomplete)
print("Moyennes prematch absentes :", len(missing))
print("SHA256 final :", report["output_sha256"])
print("Matrice :", result)
print("Manifest :", manifest)
print("Aucun label consulte.")
print("Aucune prediction calculee.")
