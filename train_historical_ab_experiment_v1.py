import csv
import json
import math
from pathlib import Path
from datetime import datetime
from statistics import NormalDist

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)

JOIN = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)

SUMMARY = JOIN / "V3C_545_CONSOLIDATION_SUMMARY_V1.json"

OUT_PRED = JOIN / "V3C_AB_VALIDATION_PREDICTIONS_V1.csv"
OUT_REPORT = JOIN / "V3C_AB_FIRST_EXPERIMENT_V1.json"

TOP_K = (10, 20, 30, 50)
REGULARIZATION_C = 0.2

for path in (OUT_PRED, OUT_REPORT):
    if path.exists():
        raise FileExistsError(
            f"Fichier déjà présent : {path}"
        )


def load(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:
        mid = row["match_id"]

        if not mid or mid in result:
            raise ValueError(
                f"ID absent ou dupliqué : {path}"
            )

        result[mid] = row

    return result


def parse_date(raw):
    dt = datetime.fromisoformat(
        raw.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError(
            f"Date sans fuseau : {raw}"
        )

    return dt


def parse_label(raw):
    value = float(raw)

    if not math.isfinite(value):
        raise ValueError("Label non fini")

    if value not in (0.0, 1.0):
        raise ValueError(
            f"Label incorrect : {raw}"
        )

    return int(value)


def matrix(rows, ids, features):
    result = []

    for mid in ids:
        row = []

        for feature in features:
            try:
                value = float(rows[mid][feature])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Feature invalide : {mid}, {feature}"
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Feature non finie : "
                    f"{mid}, {feature}"
                )

            row.append(value)

        result.append(row)

    return np.asarray(result, dtype=float)


def wilson_interval(wins, total):
    if total == 0:
        return None

    z = NormalDist().inv_cdf(0.975)
    p = wins / total
    z2 = z * z

    center = (
        p + z2 / (2 * total)
    ) / (
        1 + z2 / total
    )

    margin = (
        z / (1 + z2 / total)
        * math.sqrt(
            p * (1 - p) / total
            + z2 / (4 * total * total)
        )
    )

    return [
        round(max(0, center - margin), 4),
        round(min(1, center + margin), 4)
    ]


summary = json.loads(
    SUMMARY.read_text(encoding="utf-8")
)

fa = summary["features_a"]
fb = summary["features_b"]

if len(fa) != 23 or len(fb) != 29:
    raise ValueError(
        "Dimensions A/B incorrectes"
    )

if len(set(fa)) != 23 or len(set(fb)) != 29:
    raise ValueError(
        "Features répétées"
    )

if fb[:23] != fa:
    raise ValueError(
        "Les 23 variables communes ne correspondent pas"
    )

data = {}

for model in ("A", "B"):
    for group in ("TRAIN", "VALIDATION"):
        path = JOIN / (
            f"V3C_545_{model}_{group}_RESEARCH_V1.csv"
        )

        data[(model, group)] = load(path)

train_a = data[("A", "TRAIN")]
train_b = data[("B", "TRAIN")]
val_a = data[("A", "VALIDATION")]
val_b = data[("B", "VALIDATION")]

if (
    len(train_a) != 195
    or len(train_b) != 195
    or len(val_a) != 167
    or len(val_b) != 167
):
    raise ValueError(
        "Effectifs TRAIN/VALIDATION incorrects"
    )

if set(train_a) != set(train_b):
    raise ValueError(
        "Les matchs TRAIN A/B diffèrent"
    )

if set(val_a) != set(val_b):
    raise ValueError(
        "Les matchs VALIDATION A/B diffèrent"
    )

if set(train_a) & set(val_a):
    raise ValueError(
        "Fuite : match présent dans les deux ensembles"
    )

for group, a_rows, b_rows in (
    ("TRAIN", train_a, train_b),
    ("VALIDATION", val_a, val_b)
):
    for mid in a_rows:
        a = a_rows[mid]
        b = b_rows[mid]

        if (
            a["research_split"] != group
            or b["research_split"] != group
        ):
            raise ValueError(
                f"Split incorrect : {mid}"
            )

        if (
            a["kickoff_utc"] != b["kickoff_utc"]
            or parse_label(a["goal_after_87"])
            != parse_label(b["goal_after_87"])
        ):
            raise ValueError(
                f"Incohérence A/B : {mid}"
            )

train_ids = sorted(
    train_a,
    key=lambda mid: (
        parse_date(train_a[mid]["kickoff_utc"]),
        mid
    )
)

val_ids = sorted(
    val_a,
    key=lambda mid: (
        parse_date(val_a[mid]["kickoff_utc"]),
        mid
    )
)

train_dates = [
    parse_date(train_a[mid]["kickoff_utc"])
    for mid in train_ids
]

val_dates = [
    parse_date(val_a[mid]["kickoff_utc"])
    for mid in val_ids
]

if max(train_dates) >= min(val_dates):
    raise ValueError(
        "ERREUR : TRAIN et VALIDATION "
        "ne sont pas chronologiques"
    )

y_train = np.asarray([
    parse_label(train_a[mid]["goal_after_87"])
    for mid in train_ids
], dtype=int)

y_val = np.asarray([
    parse_label(val_a[mid]["goal_after_87"])
    for mid in val_ids
], dtype=int)

if (
    int(y_train.sum()) != 53
    or int(y_val.sum()) != 42
):
    raise ValueError(
        "Nombre de positifs inattendu"
    )

results = {}
predictions = {}
rankings = {}

print("\n=== ÉTAPE 119 : PREMIER ENTRAÎNEMENT A/B ===")

print("TRAIN :", len(train_ids))
print("VALIDATION :", len(val_ids))
print("Positifs TRAIN :", int(y_train.sum()))
print("Positifs VALIDATION :", int(y_val.sum()))

print(
    "TRAIN :",
    min(train_dates).date(),
    "->",
    max(train_dates).date()
)

print(
    "VALIDATION :",
    min(val_dates).date(),
    "->",
    max(val_dates).date()
)

print("\nTEST FUTUR : NON CHARGÉ")

baseline_rate = float(y_train.mean())
baseline_predictions = np.full(
    len(val_ids),
    baseline_rate
)

print("\n=== BASELINE ===")
print("Prévalence TRAIN :", round(baseline_rate, 4))
print(
    "Prévalence VALIDATION :",
    round(float(y_val.mean()), 4)
)
print(
    "Brier baseline :",
    round(
        float(brier_score_loss(
            y_val, baseline_predictions
        )),
        4
    )
)

for model_name, features in (
    ("A", fa),
    ("B", fb)
):
    train_rows = data[(model_name, "TRAIN")]
    val_rows = data[(model_name, "VALIDATION")]

    X_train = matrix(
        train_rows, train_ids, features
    )

    X_val = matrix(
        val_rows, val_ids, features
    )

    # Configuration fixée à l'avance :
    # aucun ajustement sur les résultats VALIDATION.
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=REGULARIZATION_C,
            solver="lbfgs",
            max_iter=3000,
            class_weight=None,
            random_state=42
        )
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(
        X_val
    )[:, 1]

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            f"Probabilités invalides : {model_name}"
        )

    predictions[model_name] = probabilities

    # Classement reproductible :
    # plus forte probabilité d'abord,
    # puis ID pour départager les égalités.
    order = sorted(
        range(len(val_ids)),
        key=lambda i: (
            -float(probabilities[i]),
            val_ids[i]
        )
    )

    rankings[model_name] = {
        val_ids[i]: rank
        for rank, i in enumerate(order, 1)
    }

    auc = float(
        roc_auc_score(y_val, probabilities)
    )

    ap = float(
        average_precision_score(
            y_val, probabilities
        )
    )

    brier = float(
        brier_score_loss(
            y_val, probabilities
        )
    )

    loss = float(
        log_loss(
            y_val,
            probabilities,
            labels=[0, 1]
        )
    )

    top_results = {}

    print(f"\n=== MODÈLE {model_name} ===")
    print("Features :", len(features))
    print("ROC AUC :", round(auc, 4))
    print("Average Precision :", round(ap, 4))
    print("Brier :", round(brier, 4))
    print("Log Loss :", round(loss, 4))

    print("\n=== PRÉCISION PAR SÉLECTION ===")

    for k in TOP_K:
        chosen = order[:k]
        wins = int(
            sum(int(y_val[i]) for i in chosen)
        )

        precision = wins / k
        recall = wins / int(y_val.sum())
        interval = wilson_interval(wins, k)

        top_results[str(k)] = {
            "alerts": k,
            "wins": wins,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "wilson_95": interval
        }

        print(
            f"TOP {k:2d} | "
            f"wins={wins:2d} | "
            f"precision={precision:.1%} | "
            f"recall={recall:.1%} | "
            f"IC95={interval}"
        )

    results[model_name] = {
        "features": len(features),
        "roc_auc": round(auc, 6),
        "average_precision": round(ap, 6),
        "brier": round(brier, 6),
        "log_loss": round(loss, 6),
        "top_k": top_results
    }

# Sauvegarde des prédictions uniquement
# sur VALIDATION. Aucun résultat TEST ici.
rows = []

for i, mid in enumerate(val_ids):
    rows.append({
        "match_id": mid,
        "kickoff_utc": val_a[mid]["kickoff_utc"],
        "research_split": "VALIDATION",
        "goal_after_87": int(y_val[i]),
        "probability_a": round(
            float(predictions["A"][i]), 8
        ),
        "probability_b": round(
            float(predictions["B"][i]), 8
        ),
        "rank_a": rankings["A"][mid],
        "rank_b": rankings["B"][mid],
        "research_only": 1,
        "live_certified": 0
    })

report = {
    "experiment": "V3C_AB_FIRST_EXPERIMENT_V1",
    "research_only": True,
    "training_certified": False,
    "live_certified": False,
    "algorithm": "L2_LOGISTIC_REGRESSION",
    "regularization_c": REGULARIZATION_C,
    "feature_scaling": "TRAIN_ONLY",
    "train_matches": len(train_ids),
    "train_positives": int(y_train.sum()),
    "validation_matches": len(val_ids),
    "validation_positives": int(y_val.sum()),
    "train_last_kickoff_utc": max(
        train_dates
    ).isoformat(),
    "validation_first_kickoff_utc": min(
        val_dates
    ).isoformat(),
    "test_future_loaded": False,
    "test_future_evaluated": False,
    "model_selection_finalized": False,
    "notes": (
        "Validation exploratoire. "
        "Labels et statistiques non tous vérifiés "
        "indépendamment. Pas de preuve de performance LIVE."
    ),
    "results": results
}

# Vérifications avant écriture.
if len(rows) != 167:
    raise ValueError(
        "Prédictions VALIDATION incomplètes"
    )

with OUT_PRED.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

with OUT_REPORT.open(
    "x", encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n=== EXPÉRIENCE TERMINÉE ===")
print("Prédictions :", OUT_PRED)
print("Rapport :", OUT_REPORT)
print("Aucun fichier TEST ouvert.")
print("Aucune sélection automatique du gagnant.")
print("Aucun changement du scanner LIVE.")
print("Aucun appel API.")
print("Aucun modèle déployé.")
