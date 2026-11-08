import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("outputs/v4_research")

OOF = (
    BASE / "v41_experiment_380"
    / "V41_A2_OOF60_PREDICTIONS.csv"
)
OOF_REPORT = (
    BASE / "v41_experiment_380"
    / "V41_A2_EXPERIMENT_REPORT.json"
)

VALID = (
    BASE / "v41_validation_predictions_387"
    / "V41_A2_VALIDATION46_PREDICTIONS_BLIND.csv"
)
VALID_REPORT = (
    BASE / "v41_validation_predictions_387"
    / "V41_A2_VALIDATION46_PREDICTIONS_FREEZE.json"
)

FEATURES = (
    BASE / "v41_validation_own_goal_385"
    / "V41_A2_VALIDATION46_FEATURES_BLIND.csv"
)
FEATURES_REPORT = (
    BASE / "v41_validation_own_goal_385"
    / "V41_A2_VALIDATION_FEATURES_MANIFEST.json"
)

FREEZE = (
    BASE / "v41_candidate_freeze_382"
    / "V41_A2_EXPLORATORY_FREEZE.json"
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def check(condition, message):
    if not condition:
        raise RuntimeError("STOP : " + message)

oof_report = read_json(OOF_REPORT)
valid_report = read_json(VALID_REPORT)
features_report = read_json(FEATURES_REPORT)
freeze = read_json(FREEZE)

check(sha(OOF) == oof_report["predictions_sha256"],
      "OOF modifie")
check(sha(VALID) == valid_report["predictions_sha256"],
      "validation predictions modifiees")
check(sha(FEATURES) == features_report["output_sha256"],
      "validation features modifiees")
check(sha(FREEZE) == valid_report["candidate_freeze_sha256"],
      "gel 382 modifie")
check(sha(FEATURES) == valid_report["validation_features_sha256"],
      "source validation incoherente")

check(freeze["candidate"] == "V4.1-A2",
      "candidat incorrect")
check(freeze["threshold"] == 0.60,
      "seuil incorrect")
check(valid_report["labels_accessed"] is False,
      "statut labels inattendu")
check(valid_report["outcomes_evaluated"] is False,
      "statut evaluation inattendu")

oof = read_csv(OOF)
valid = read_csv(VALID)
features = read_csv(FEATURES)

check(len(oof) == 60, "OOF different de 60")
check(len(valid) == 46, "validation differente de 46")
check(len(features) == 46, "features differentes de 46")

oof_ids = [r["match_id"] for r in oof]
valid_ids = [r["match_id"] for r in valid]
feature_ids = [r["match_id"] for r in features]

check(len(set(oof_ids)) == 60, "doublons OOF")
check(len(set(valid_ids)) == 46, "doublons validation")
check(set(valid_ids) == set(feature_ids),
      "IDs validation/features differents")
check(not set(oof_ids).intersection(valid_ids),
      "chevauchement OOF / validation")

for rows in (oof, valid):
    for r in rows:
        p = float(r["probability"])
        check(math.isfinite(p) and 0 <= p <= 1,
              "probabilite invalide")
        check(int(r["alert"]) == int(p >= 0.60),
              "alerte incoherente")

check(sum(int(r["alert"]) for r in oof) == 4,
      "nombre alertes OOF inattendu")
check(sum(int(r["alert"]) for r in valid) == 1,
      "nombre alertes validation inattendu")

def summary(name, rows):
    p = np.array(
        [float(r["probability"]) for r in rows]
    )

    print(f"\n=== {name} ===")
    print("Matchs :", len(rows))
    print("Moyenne :", f"{p.mean():.4f}")
    print("Mediane :", f"{np.median(p):.4f}")
    print("Minimum :", f"{p.min():.4f}")
    print("Maximum :", f"{p.max():.4f}")
    print("Percentile 75 :", f"{np.percentile(p, 75):.4f}")
    print("Percentile 90 :", f"{np.percentile(p, 90):.4f}")
    print("Alertes >=0.60 :", int((p >= 0.60).sum()))
    print("Taux de selection :",
          f"{100*(p >= 0.60).mean():.2f}%")

    print("\nTranches de probabilite :")

    bands = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.8),
        (0.8, 1.000001),
    ]

    for low, high in bands:
        n = int(((p >= low) & (p < high)).sum())
        print(f"  [{low:.1f}, {min(high, 1):.1f}) : {n}")

print("=" * 72)
print("ETAPE 388 - DIAGNOSTIC DES PROBABILITES V4.1-A2")
print("=" * 72)

summary("OOF EXPLORATOIRE", oof)
summary("VALIDATION AVEUGLE", valid)

print("\n=== VALIDATION PAR MOIS ===")

by_month = defaultdict(list)

for r in valid:
    month = r["kickoff_utc"][:7]
    by_month[month].append(r)

for month, group in sorted(by_month.items()):
    p = [float(r["probability"]) for r in group]
    alerts = sum(int(r["alert"]) for r in group)

    print(
        f"{month} | "
        f"matchs={len(group)} | "
        f"proba_moyenne={np.mean(p):.4f} | "
        f"alertes={alerts}"
    )

print("\n=== INTEGRITE ===")
print("OOF original : INCHANGE")
print("Validation : INCHANGEE")
print("Seuil fige : 0.60")
print("Labels validation : NON CONSULTES")
print("Aucun entrainement")
print("Aucun appel API")
print("Aucun fichier modifie")
print("HOLDOUT : NON CONSULTE")
print("ETAPE 388 TERMINEE")
