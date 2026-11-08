import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("outputs")

MODEL = ROOT / (
    "model_v3_experiments/candidates_refit_all/"
    "V3_C_AGGRESSIVE_refit_all.joblib"
)

SCHEMA = ROOT / (
    "model_v3_experiments/schemas/"
    "V3_C_AGGRESSIVE_schema.json"
)

SNAPSHOTS = ROOT / (
    "history/live_feature_snapshots_v3c.csv"
)

DEPENDENCIES = {
    "last_shot_minute": [
        "last_shot_minute",
        "minutes_since_last_shot",
    ],
    "last_shot_on_target_minute": [
        "minutes_since_last_shot_on_target",
    ],
    "last_corner_minute": [
        "last_corner_minute",
        "minutes_since_last_corner",
    ],
}

print("=== ETAPE 152 : SENSIBILITE TEMPORELLE V3-C ===")

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

if not isinstance(features, list) or len(features) != 38:
    raise ValueError("Schema actif invalide")

model = joblib.load(MODEL)
df = pd.read_csv(SNAPSHOTS, low_memory=False)

X = df.loc[:, features].copy()

baseline = model.predict_proba(X)[:, 1]

recorded = pd.to_numeric(
    df["model_probability"],
    errors="coerce",
).to_numpy(dtype=float)

if not np.allclose(
    baseline, recorded,
    atol=1e-6,
    rtol=0,
):
    raise ValueError(
        "Le modele ne reproduit plus les probabilites"
    )

minute = pd.to_numeric(
    df["minute"], errors="coerce"
)

X_test = X.copy()
exposed = pd.Series(
    False, index=df.index
)

modified_features = {}

for source, destinations in DEPENDENCIES.items():
    event_minute = pd.to_numeric(
        df[source], errors="coerce"
    )

    invalid = (
        event_minute.notna()
        & minute.notna()
        & (event_minute > minute)
    )

    exposed |= invalid

    for feature in destinations:
        if feature not in features:
            continue

        X_test.loc[invalid, feature] = np.nan

        modified_features[feature] = (
            modified_features.get(feature, 0)
            + int(invalid.sum())
        )

print("Snapshots analyses :", len(df))
print("Snapshots exposes :", int(exposed.sum()))
print(
    "Matchs exposes :",
    df.loc[exposed, "match_id"].nunique()
)

print("\n=== VARIABLES MASQUEES ===")
for feature, count in modified_features.items():
    print(feature, ":", count)

print("\n=== TEST HYPOTHETIQUE ===")
print(
    "Les valeurs suspectes sont remplacees "
    "par NaN, jamais par zero."
)

try:
    alternative = model.predict_proba(
        X_test
    )[:, 1]
except Exception as exc:
    print("SCENARIO NON SUPPORTE :", repr(exc))
    print(
        "Le pipeline ne peut pas traiter ce "
        "scenario de valeurs manquantes."
    )
    raise SystemExit(0)

diff = np.abs(alternative - baseline)
affected = diff[exposed.to_numpy()]

print("\n=== IMPACT SUR LES PROBABILITES ===")

print(
    "Difference absolue moyenne :",
    float(affected.mean()) if len(affected) else 0
)

print(
    "Difference absolue mediane :",
    float(np.median(affected)) if len(affected) else 0
)

print(
    "Difference absolue maximale :",
    float(affected.max()) if len(affected) else 0
)

print(
    "Captures avec variation > 1 point :",
    int((affected > 0.01).sum())
)

print(
    "Captures avec variation > 5 points :",
    int((affected > 0.05).sum())
)

alert = (
    df["alert"].astype(str).str.lower()
    .isin(["true", "1", "yes", "oui"])
)

mask = exposed & alert

print("\n=== ALERTES EXPOSEES ===")
print("Captures alertees :", int(mask.sum()))
print(
    "Matchs alertes distincts :",
    df.loc[mask, "match_id"].nunique()
)

result = df.loc[mask, [
    "match_id",
    "minute",
    "snapshot_timestamp_utc",
]].copy()

result["original"] = baseline[mask.to_numpy()]
result["scenario_nan"] = alternative[mask.to_numpy()]
result["variation"] = (
    result["scenario_nan"] - result["original"]
)

first_alerts = (
    result.sort_values("snapshot_timestamp_utc")
    .drop_duplicates("match_id")
)

print("\n=== PREMIERE ALERTE EXPOSEE PAR MATCH ===")

for row in first_alerts.itertuples(index=False):
    print(
        f"{row.match_id} | "
        f"{row.minute}' | "
        f"original={row.original:.4f} | "
        f"scenario={row.scenario_nan:.4f} | "
        f"variation={row.variation:+.4f}"
    )

print("\n=== LIMITES ===")
print("Scenario NaN uniquement : pas une correction reelle.")
print("Aucune decision d'alerte recalculee.")
print("Aucune performance predictive estimee.")
print("Aucun appel API.")
print("Aucun fichier modifie.")
print("Scanner LIVE inchange.")
