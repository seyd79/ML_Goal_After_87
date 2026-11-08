import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path("outputs/v4_research/v41_experiment_377")
PRED = BASE / "V41_A_OOF60_PREDICTIONS.csv"
REPORT = BASE / "V41_A_EXPERIMENT_REPORT.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

if sha(PRED) != report["predictions_sha256"]:
    raise RuntimeError("STOP : predictions modifiees")

with PRED.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 60:
    raise RuntimeError("STOP : 60 predictions attendues")

if len({r["match_id"] for r in rows}) != 60:
    raise RuntimeError("STOP : IDs dupliques")

THRESHOLD = 0.60

for r in rows:
    r["fold"] = int(r["fold"])
    r["label"] = int(r["label"])
    r["probability"] = float(r["probability"])
    r["alert"] = int(r["alert"])

    if not 0 <= r["probability"] <= 1:
        raise RuntimeError("STOP : probabilite invalide")

    if r["alert"] != int(r["probability"] >= THRESHOLD):
        raise RuntimeError("STOP : seuil incoherent")

def mean(values):
    return sum(values) / len(values) if values else None

def brier(group):
    return mean([
        (r["probability"] - r["label"]) ** 2
        for r in group
    ])

print("=" * 72)
print("ETAPE 379 - DIAGNOSTIC DES ERREURS V4.1-A")
print("=" * 72)

tp = [r for r in rows if r["alert"] == 1 and r["label"] == 1]
fp = [r for r in rows if r["alert"] == 1 and r["label"] == 0]
fn = [r for r in rows if r["alert"] == 0 and r["label"] == 1]
tn = [r for r in rows if r["alert"] == 0 and r["label"] == 0]

print("\n=== MATRICE DE CONFUSION ===")
print("Vrais positifs :", len(tp))
print("Faux positifs :", len(fp))
print("Faux negatifs :", len(fn))
print("Vrais negatifs :", len(tn))

print("\n=== BRIER PAR FOLD ===")
for fold in (1, 2, 3):
    group = [r for r in rows if r["fold"] == fold]
    print(
        f"FOLD {fold} | n={len(group)} | "
        f"Brier={brier(group):.5f} | "
        f"proba moyenne={mean([r['probability'] for r in group]):.3f} | "
        f"taux reel={mean([r['label'] for r in group]):.3f}"
    )

print("\n=== PROBABILITES PAR RESULTAT ===")
for name, group in [
    ("Vrais positifs", tp),
    ("Faux positifs", fp),
    ("Faux negatifs", fn),
    ("Vrais negatifs", tn),
]:
    avg = mean([r["probability"] for r in group])
    print(
        f"{name:18s} | n={len(group):2d} | "
        f"probabilite moyenne={avg:.4f}"
        if avg is not None else
        f"{name:18s} | n=0"
    )

print("\n=== TRANCHES DE PROBABILITE ===")
bands = [
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.01),
]

for low, high in bands:
    group = [
        r for r in rows
        if low <= r["probability"] < high
    ]
    n = len(group)
    wins = sum(r["label"] for r in group)
    avg = mean([r["probability"] for r in group])
    rate = wins / n if n else None

    label = (
        f"{low:.2f}-{min(high, 1):.2f}"
    )

    if n:
        print(
            f"{label} | n={n:2d} | "
            f"positifs={wins:2d} | "
            f"proba moyenne={avg:.3f} | "
            f"taux reel={rate:.3f}"
        )
    else:
        print(f"{label} | n=0")

print("\n=== FAUX POSITIFS ===")
for r in sorted(
    fp, key=lambda x: -x["probability"]
):
    print(
        f"{r['match_id']} | "
        f"fold={r['fold']} | "
        f"p={r['probability']:.4f}"
    )

print("\n=== FAUX NEGATIFS ===")
for r in sorted(
    fn, key=lambda x: -x["probability"]
):
    print(
        f"{r['match_id']} | "
        f"fold={r['fold']} | "
        f"p={r['probability']:.4f}"
    )

print("\nAucun seuil optimise.")
print("Aucun modele entraine.")
print("Aucun fichier modifie.")
print("Validation / HOLDOUT : NON CONSULTES")
print("ETAPE 379 TERMINEE.")
