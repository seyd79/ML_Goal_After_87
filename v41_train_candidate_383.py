import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("outputs/v4_research")

DATA = (
    BASE / "v41_candidate_a_375"
    / "V41_A_TRAIN112_CORE43_PROVISIONAL.csv"
)

SOURCE_MANIFEST = (
    BASE / "v41_candidate_a_375"
    / "V41_A_CORE43_MANIFEST.json"
)

FREEZE = (
    BASE / "v41_candidate_freeze_382"
    / "V41_A2_EXPLORATORY_FREEZE.json"
)

OUT = BASE / "v41_fitted_candidate_383"
MODEL = OUT / "V41_A2_RESEARCH_TRAIN112.joblib"
REPORT = OUT / "V41_A2_FITTED_RESEARCH_MANIFEST.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def numeric(value):
    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null", "na", "n/a"
    }:
        return np.nan

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Valeur numerique non finie")

    return number

print("=" * 70)
print("ETAPE 383 - ENTRAINEMENT RECHERCHE V4.1-A2")
print("=" * 70)

if os.getenv("V41_AUTORISE_EXPERIENCE_PROVISOIRE") != "OUI":
    raise RuntimeError(
        "STOP : autorisation exploratoire manquante"
    )

if OUT.exists():
    raise RuntimeError("STOP : dossier 383 deja existant")

source = load_json(SOURCE_MANIFEST)
freeze = load_json(FREEZE)

if sha(DATA) != source["output_sha256"]:
    raise RuntimeError("STOP : dataset modifie")

if sha(DATA) != freeze["research_dataset_sha256"]:
    raise RuntimeError("STOP : dataset different du gel 382")

if freeze["status"] != "EXPLORATORY_CANDIDATE_FROZEN":
    raise RuntimeError("STOP : statut du gel incorrect")

if source["training_authorized"] is not False:
    raise RuntimeError("STOP : statut source inattendu")

if freeze["official_training_authorized"] is not False:
    raise RuntimeError("STOP : statut gel inattendu")

if freeze["production_authorized"] is not False:
    raise RuntimeError("STOP : autorisation production inattendue")

features = freeze["features"]

if (
    len(features) != 10
    or len(set(features)) != 10
    or not set(features).issubset(source["features"])
):
    raise RuntimeError("STOP : features non conformes")

if freeze["model"] != "LogisticRegression_L2":
    raise RuntimeError("STOP : modele inattendu")

if freeze["regularization_C"] != 0.1:
    raise RuntimeError("STOP : regularisation modifiee")

if freeze["threshold"] != 0.60:
    raise RuntimeError("STOP : seuil modifie")

rows = load_csv(DATA)

if len(rows) != 112:
    raise RuntimeError("STOP : 112 observations attendues")

ids = [r["match_id"] for r in rows]

if len(set(ids)) != 112:
    raise RuntimeError("STOP : IDs dupliques")

labels = Counter(r["goal_after_87"] for r in rows)

if labels != {"0": 74, "1": 38}:
    raise RuntimeError("STOP : repartition des labels incorrecte")

if any(
    not r["kickoff_utc"].startswith("2025-")
    for r in rows
):
    raise RuntimeError("STOP : date hors TRAIN attendu")

X = np.asarray([
    [numeric(r[feature]) for feature in features]
    for r in rows
], dtype=float)

y = np.asarray([
    int(r["goal_after_87"]) for r in rows
], dtype=int)

if X.shape != (112, 10):
    raise RuntimeError("STOP : matrice inattendue")

if np.isnan(X).all(axis=0).any():
    raise RuntimeError("STOP : feature totalement manquante")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="liblinear",
        class_weight=None,
        max_iter=2000,
        random_state=41,
    )),
])

# Uniquement les 112 matchs TRAIN.
model.fit(X, y)

if list(model.named_steps["classifier"].classes_) != [0, 1]:
    raise RuntimeError("STOP : classes inattendues")

OUT.mkdir(parents=True, exist_ok=False)
joblib.dump(model, MODEL)

report = {
    "candidate": "V4.1-A2",
    "status": "FITTED_EXPLORATORY_ONLY",
    "training_matches": 112,
    "positive_labels": 38,
    "negative_labels": 74,
    "features": features,
    "model_type": "LogisticRegression_L2",
    "regularization_C": 0.1,
    "threshold": 0.60,
    "imputation": "median_train_only",
    "scaling": "standard_train_only",
    "source_sha256": sha(DATA),
    "freeze_382_sha256": sha(FREEZE),
    "model_sha256": sha(MODEL),
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "joblib_version": joblib.__version__,
    "exploratory_only": True,
    "independent_validation_completed": False,
    "labels_independently_certified": False,
    "protected_sets_fully_certified": False,
    "official_training_authorized": False,
    "production_authorized": False,
    "validation_accessed": False,
    "holdout_accessed": False,
    "v3c_modified": False,
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Candidat : V4.1-A2")
print("Entrainement : 112 matchs TRAIN")
print("Features :", len(features))
print("Positifs :", labels["1"])
print("Negatifs :", labels["0"])
print("Regularisation C : 0.1")
print("Seuil fige : 0.60")
print("Modele sauvegarde :", MODEL)
print("SHA256 modele :", sha(MODEL))
print("Manifest :", REPORT)
print("Validation / HOLDOUT : NON CONSULTES")
print("Entrainement officiel : NON AUTORISE")
print("Production : NON AUTORISEE")
print("V3-C : INCHANGE")
print("ETAPE 383 TERMINEE")
