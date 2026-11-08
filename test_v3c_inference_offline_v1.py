import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL = Path(
    "outputs/model_v3_experiments/"
    "candidates_refit_all/"
    "V3_C_AGGRESSIVE_refit_all.joblib"
)

SCHEMA = Path(
    "outputs/model_v3_experiments/"
    "schemas/V3_C_AGGRESSIVE_schema.json"
)

CSV = Path(
    "outputs/history/live_feature_snapshots_v3c.csv"
)

print("=== ÉTAPE 151 : REPRODUCTION OFFLINE V3-C ===")

with SCHEMA.open(encoding="utf-8") as f:
    schema = json.load(f)

if isinstance(schema, list):
    features = schema
else:
    features = next(
        (
            schema[k]
            for k in (
                "features_used",
                "core_features",
                "features",
                "feature_names",
            )
            if isinstance(schema.get(k), list)
        ),
        None,
    )

if not isinstance(features, list):
    raise ValueError("Schéma non reconnu")

if len(features) != 38 or len(set(features)) != 38:
    raise ValueError("Schéma de 38 variables invalide")

model = joblib.load(MODEL)
df = pd.read_csv(CSV, low_memory=False)

missing = [
    feature for feature in features
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Variables absentes : {missing}"
    )

if "model_probability" not in df.columns:
    raise ValueError(
        "Colonne model_probability absente"
    )

X = df.loc[:, features].copy()

print("Snapshots :", len(df))
print("Variables :", X.shape[1])
print(
    "Snapshots avec valeurs manquantes :",
    int(X.isna().any(axis=1).sum())
)

print("\n=== PRÉDICTION OFFLINE ===")

try:
    probabilities = model.predict_proba(X)[:, 1]
except Exception as exc:
    print("ÉCHEC DE L'INFÉRENCE :", repr(exc))
    print(
        "Vérifier les types des variables et le "
        "prétraitement de production."
    )
    raise SystemExit(1)

recorded = pd.to_numeric(
    df["model_probability"],
    errors="coerce"
).to_numpy(dtype=float)

valid = (
    np.isfinite(probabilities)
    & np.isfinite(recorded)
)

if not valid.any():
    raise ValueError(
        "Aucune probabilité comparable"
    )

diff = np.abs(probabilities[valid] - recorded[valid])

print("Probabilités comparables :", int(valid.sum()))
print("Différence absolue moyenne :", float(diff.mean()))
print("Différence absolue maximale :", float(diff.max()))
print(
    "Identiques à 1e-6 près :",
    int((diff <= 1e-6).sum()),
)
print(
    "Différence supérieure à 0.01 :",
    int((diff > 0.01).sum()),
)

print("\n=== CINQ EXEMPLES ===")

indices = np.flatnonzero(valid)[:5]

for i in indices:
    print(
        f"match={df.iloc[i].get('match_id')} "
        f"minute={df.iloc[i].get('minute')} "
        f"enregistrée={recorded[i]:.6f} "
        f"recalculée={probabilities[i]:.6f}"
    )

print("\n=== CONCLUSION ===")

if np.all(diff <= 1e-6):
    print("REPRODUCTION DES PROBABILITÉS : PASS")
else:
    print("REPRODUCTION NON IDENTIQUE")
    print(
        "Une différence peut indiquer un écart de "
        "modèle, de version ou de prétraitement."
    )

print("Aucun appel API.")
print("Aucun réentraînement.")
print("Aucun fichier historique modifié.")
print("Scanner LIVE inchangé.")
print(
    "Aucune correction temporelle appliquée "
    "dans ce test."
)
