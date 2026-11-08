import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = Path("outputs/v4_research")

DATA_DIR = BASE / "v41_candidate_a_375"
DATA = DATA_DIR / "V41_A_TRAIN112_CORE43_PROVISIONAL.csv"
DATA_MANIFEST = DATA_DIR / "V41_A_CORE43_MANIFEST.json"

PROTOCOL_DIR = BASE / "v41_protocol_376"
FOLDS = PROTOCOL_DIR / "V41_A_TRAIN_ONLY_FOLDS.csv"
PROTOCOL = PROTOCOL_DIR / "V41_A_PROTOCOL_FREEZE.json"

OUT = BASE / "v41_experiment_377"
PREDICTIONS = OUT / "V41_A_OOF60_PREDICTIONS.csv"
REPORT = OUT / "V41_A_EXPERIMENT_REPORT.json"

THRESHOLD = 0.60
C = 0.1

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def vector(row, features):
    result = []
    for feature in features:
        value = str(row[feature]).strip()
        if value.lower() in {
            "", "nan", "none", "null", "na", "n/a"
        }:
            result.append(np.nan)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("Valeur infinie : " + feature)
            result.append(number)
    return result

def metrics(predictions):
    y = np.array(
        [int(r["label"]) for r in predictions],
        dtype=int
    )
    p = np.array(
        [float(r["probability"]) for r in predictions],
        dtype=float
    )

    selected = p >= THRESHOLD

    n_alerts = int(selected.sum())
    wins = int(y[selected].sum())
    losses = n_alerts - wins

    precision = (
        wins / n_alerts if n_alerts else None
    )
    recall = (
        wins / int(y.sum()) if y.sum() else None
    )

    auc = (
        float(roc_auc_score(y, p))
        if len(set(y)) == 2
        else None
    )

    brier = float(brier_score_loss(y, p))

    return {
        "matches": len(y),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "alerts": n_alerts,
        "winning_alerts": wins,
        "losing_alerts": losses,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "brier_score": brier,
    }

print("=" * 72)
print("ETAPE 377 - PREMIERE EXPERIENCE V4.1-A")
print("=" * 72)

if os.getenv("V41_AUTORISE_EXPERIENCE_PROVISOIRE") != "OUI":
    raise RuntimeError(
        "STOP : autorisation de recherche provisoire manquante"
    )

if OUT.exists():
    raise RuntimeError(
        "STOP : dossier 377 deja existant"
    )

data_manifest = load_json(DATA_MANIFEST)
protocol = load_json(PROTOCOL)

if sha(DATA) != data_manifest["output_sha256"]:
    raise RuntimeError("STOP : dataset 375 modifie")

if sha(DATA) != protocol["source_sha256"]:
    raise RuntimeError("STOP : source protocole incoherente")

if sha(FOLDS) != protocol["fold_file_sha256"]:
    raise RuntimeError("STOP : protocole 376 modifie")

if data_manifest.get("training_authorized") is not False:
    raise RuntimeError("STOP : statut source inattendu")

if protocol.get("training_authorized") is not False:
    raise RuntimeError("STOP : statut protocole inattendu")

if protocol.get("status") != "DESIGN_ONLY_NOT_TRAINED":
    raise RuntimeError("STOP : protocole inattendu")

features = data_manifest["features"]

if len(features) != 43 or len(set(features)) != 43:
    raise RuntimeError("STOP : schema CORE43 incorrect")

rows = load_csv(DATA)
assignments = load_csv(FOLDS)

if len(rows) != 112:
    raise RuntimeError("STOP : 112 matchs attendus")

by_id = {r["match_id"]: r for r in rows}

if len(by_id) != 112:
    raise RuntimeError("STOP : doublons TRAIN")

if Counter(r["goal_after_87"] for r in rows) != {
    "0": 74, "1": 38
}:
    raise RuntimeError("STOP : labels inattendus")

if not all(
    all(feature in row for feature in features)
    for row in rows
):
    raise RuntimeError("STOP : caracteristique manquante")

fold_numbers = sorted(
    {int(r["fold"]) for r in assignments}
)

if fold_numbers != [1, 2, 3]:
    raise RuntimeError("STOP : folds inattendus")

predictions = []
fold_summaries = []
all_oof_ids = []

for fold in fold_numbers:
    fit_ids = [
        r["match_id"]
        for r in assignments
        if int(r["fold"]) == fold
        and r["role"] == "FIT"
    ]
    oof_ids = [
        r["match_id"]
        for r in assignments
        if int(r["fold"]) == fold
        and r["role"] == "OUT_OF_FOLD"
    ]

    if (
        not fit_ids
        or not oof_ids
        or len(set(fit_ids)) != len(fit_ids)
        or len(set(oof_ids)) != len(oof_ids)
        or set(fit_ids) & set(oof_ids)
        or not set(fit_ids + oof_ids).issubset(by_id)
    ):
        raise RuntimeError(
            f"STOP : partition invalide fold {fold}"
        )

    fit = [by_id[mid] for mid in fit_ids]
    oof = [by_id[mid] for mid in oof_ids]

    if max(r["kickoff_utc"] for r in fit) >= min(
        r["kickoff_utc"] for r in oof
    ):
        raise RuntimeError(
            f"STOP : fuite chronologique fold {fold}"
        )

    X_fit = np.array(
        [vector(r, features) for r in fit],
        dtype=float
    )
    X_oof = np.array(
        [vector(r, features) for r in oof],
        dtype=float
    )

    y_fit = np.array(
        [int(r["goal_after_87"]) for r in fit],
        dtype=int
    )

    if set(y_fit) != {0, 1}:
        raise RuntimeError("STOP : classe absente")

    # Ne pas entrainer une variable totalement absente du FIT.
    if np.any(np.all(np.isnan(X_fit), axis=0)):
        raise RuntimeError(
            f"STOP : feature entierement manquante fold {fold}"
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight=None,
            max_iter=2000,
            random_state=41
        )),
    ])

    # Le pipeline est ajuste sur FIT seulement.
    model.fit(X_fit, y_fit)

    classes = list(
        model.named_steps["classifier"].classes_
    )
    positive_index = classes.index(1)

    probabilities = model.predict_proba(X_oof)[
        :, positive_index
    ]

    fold_predictions = []

    for row, probability in zip(oof, probabilities):
        record = {
            "fold": fold,
            "match_id": row["match_id"],
            "kickoff_utc": row["kickoff_utc"],
            "label": int(row["goal_after_87"]),
            "probability": float(probability),
            "alert": int(probability >= THRESHOLD),
        }
        predictions.append(record)
        fold_predictions.append(record)

    result = metrics(fold_predictions)
    result["fold"] = fold
    result["fit_matches"] = len(fit)

    fold_summaries.append(result)
    all_oof_ids.extend(oof_ids)

    print(
        f"FOLD {fold} | "
        f"FIT={len(fit):3d} | "
        f"OOF={len(oof):2d} | "
        f"alertes={result['alerts']:2d} | "
        f"gagnantes={result['winning_alerts']:2d} | "
        f"perdantes={result['losing_alerts']:2d}"
    )

if (
    len(all_oof_ids) != 60
    or len(set(all_oof_ids)) != 60
    or len(predictions) != 60
):
    raise RuntimeError(
        "STOP : 60 predictions OOF uniques attendues"
    )

predictions.sort(
    key=lambda r: (r["kickoff_utc"], r["match_id"])
)

overall = metrics(predictions)

OUT.mkdir(parents=True, exist_ok=False)

with PREDICTIONS.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold", "match_id", "kickoff_utc",
            "label", "probability", "alert"
        ]
    )
    writer.writeheader()
    writer.writerows(predictions)

report = {
    "status": "EXPLORATORY_ONLY_NOT_CERTIFIED",
    "candidate": "V4.1-A",
    "dataset_sha256": sha(DATA),
    "folds_sha256": sha(FOLDS),
    "predictions_sha256": sha(PREDICTIONS),
    "model": "LogisticRegression_L2",
    "regularization_C": C,
    "threshold_predefined": THRESHOLD,
    "features": features,
    "fold_results": fold_summaries,
    "oof_results": overall,
    "labels_independently_certified": False,
    "protected_sets_fully_certified": False,
    "official_training_authorized": False,
    "production_authorized": False,
    "official_validation_accessed": False,
    "holdout_accessed": False,
    "model_saved": False
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 72)
print("RESULTATS GLOBAUX OOF - V4.1-A")
print("=" * 72)
print("Matchs evalues :", overall["matches"])
print("Positifs reels :", overall["positives"])
print("Alertes emises :", overall["alerts"])
print("Alertes gagnantes :", overall["winning_alerts"])
print("Alertes perdantes :", overall["losing_alerts"])

if overall["precision"] is None:
    print("Precision : NON DEFINIE (aucune alerte)")
else:
    print(
        "Precision : {:.2f}%".format(
            overall["precision"] * 100
        )
    )

if overall["recall"] is not None:
    print(
        "Rappel : {:.2f}%".format(
            overall["recall"] * 100
        )
    )

if overall["roc_auc"] is not None:
    print("ROC AUC : {:.4f}".format(overall["roc_auc"]))

print("Brier score : {:.4f}".format(overall["brier_score"]))
print("Predictions :", PREDICTIONS)
print("Rapport :", REPORT)
print("Modele non sauvegarde.")
print("Validation officielle et HOLDOUT : NON CONSULTES")
print("V3-C : INCHANGE")
print("Resultats exploratoires non certifies.")
print("ETAPE 377 TERMINEE.")
