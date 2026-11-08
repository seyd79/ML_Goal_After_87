from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path("outputs/pressure_historical/pressure_v2_dataset_join")
OUT = Path("outputs/v4_research")

A23 = [
    "score_87_home", "score_87_away",
    "goal_diff_87", "is_draw_87",
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87", "away_shots_on_target_87",
    "home_shots_off_target_87", "away_shots_off_target_87",
    "home_shots_blocked_87", "away_shots_blocked_87",
    "home_corners_87", "away_corners_87",
    "home_fouls_87", "away_fouls_87",
    "home_yellow_cards_87", "away_yellow_cards_87",
    "home_red_cards_87", "away_red_cards_87",
    "home_offsides_87", "away_offsides_87",
    "goal_diff_signed_87",
]

B6 = [
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]
B29 = A23 + B6

print("=" * 70)
print("ETAPE 170 — V4 REGULARISATION CHRONOLOGIQUE")
print("=" * 70)

base = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str}
)
plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"]
)

if plan.match_id.duplicated().any():
    raise SystemExit("Doublons dans le plan")

base = base.merge(plan, on="match_id", validate="one_to_one")
base["source"] = "ALL_320"

paths = sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv"))
if len(paths) != 10:
    raise SystemExit("10 lots B attendus")

parts = []
for path in paths:
    p = pd.read_csv(path, dtype={"match_id": str})
    p["split"] = p["research_split"]
    p["source"] = "BATCH02_11"
    parts.append(p)

columns = [
    "match_id", "kickoff_utc", "split",
    "goal_after_87", "source",
] + B29

data = pd.concat(
    [base[columns]] + [p[columns] for p in parts],
    ignore_index=True
)

data["split"] = data["split"].astype(str).str.strip().str.upper()

if len(data) != 520 or data.match_id.duplicated().any():
    raise SystemExit("Population inattendue")

if data["split"].value_counts().to_dict() != {
    "TRAIN": 183, "VALIDATION": 154, "TEST_FUTUR": 183
}:
    raise SystemExit("Répartition des partitions inattendue")

# Retrait de TEST_FUTUR avant toute analyse statistique.
data = data[data["split"].isin(["TRAIN", "VALIDATION"])].copy()

data["kickoff"] = pd.to_datetime(
    data["kickoff_utc"], utc=True, errors="raise"
)
data["y"] = pd.to_numeric(
    data["goal_after_87"], errors="raise"
)

if not data["y"].isin([0, 1]).all():
    raise SystemExit("Labels invalides")

for col in B29:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)

train = data[data.split == "TRAIN"].sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

valid = data[data.split == "VALIDATION"].sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

if train.kickoff.max() >= valid.kickoff.min():
    raise SystemExit("Séparation chronologique incorrecte")

train["day"] = train.kickoff.dt.floor("D")
days = sorted(train.day.unique())

if len(days) < 9:
    raise SystemExit("Période TRAIN trop courte pour ce protocole")

configs = {
    "A23_C1": (A23, 1.0),
    "B29_C003": (B29, 0.03),
    "B29_C01": (B29, 0.1),
    "B29_C03": (B29, 0.3),
    "B29_C1": (B29, 1.0),
}

def make_model(c):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=c,
            max_iter=3000,
            random_state=42
        )),
    ])

# Validation chronologique progressive à l'intérieur de TRAIN.
# Aucun entraînement n'utilise les journées futures.
# Embargo de 4 heures avant le premier match du bloc évalué.
folds = []

for start in (4, 6, 8, 10):
    if start >= len(days):
        continue

    test_days = days[start:start + 2]
    mask_test = train.day.isin(test_days).to_numpy()

    if not mask_test.any():
        continue

    first_kickoff = train.loc[mask_test, "kickoff"].min()
    cutoff = first_kickoff - pd.Timedelta(hours=4)

    mask_fit = (train.kickoff < cutoff).to_numpy()

    yy = train.loc[mask_fit, "y"]

    if (
        mask_fit.sum() < 30
        or yy.nunique() != 2
        or yy.sum() < 5
        or (yy == 0).sum() < 5
    ):
        print("Bloc ignoré : historique insuffisant")
        continue

    folds.append((mask_fit, mask_test))

if len(folds) < 3:
    raise SystemExit("Moins de trois blocs chronologiques utilisables")

oof = {
    name: np.full(len(train), np.nan)
    for name in configs
}

print("\n=== BLOCS CHRONOLOGIQUES TRAIN ===")

for i, (fit_mask, test_mask) in enumerate(folds, 1):
    print(
        f"BLOC {i} | entraînement={fit_mask.sum()} "
        f"| évaluation={test_mask.sum()} "
        f"| premier match={train.loc[test_mask, 'kickoff'].min()}"
    )

    for name, (cols, c) in configs.items():
        model = make_model(c)
        model.fit(
            train.loc[fit_mask, cols],
            train.loc[fit_mask, "y"].astype(int)
        )
        oof[name][test_mask] = model.predict_proba(
            train.loc[test_mask, cols]
        )[:, 1]

common = np.isfinite(oof["A23_C1"])

if common.sum() < 40:
    raise SystemExit("Trop peu de prédictions OOF")

for name in configs:
    if not np.array_equal(common, np.isfinite(oof[name])):
        raise SystemExit("Prédictions OOF non alignées")

y_oof = train.loc[common, "y"].astype(int).to_numpy()

print("\n=== SELECTION SUR TRAIN UNIQUEMENT ===")
print("Matchs OOF :", len(y_oof))
print("Taux positif OOF :", f"{y_oof.mean():.1%}")

scores = []

for name in configs:
    p = oof[name][common]
    auc = roc_auc_score(y_oof, p)
    ap = average_precision_score(y_oof, p)
    brier = brier_score_loss(y_oof, p)

    scores.append({
        "model": name,
        "oof_n": len(y_oof),
        "auc": auc,
        "ap": ap,
        "brier": brier,
    })

    print(
        f"{name:10s} | AUC={auc:.4f} "
        f"| AP={ap:.4f} | Brier={brier:.4f}"
    )

# Choix du C exclusivement par Average Precision OOF.
b_candidates = [
    r for r in scores if r["model"].startswith("B29_")
]
winner = max(b_candidates, key=lambda r: r["ap"])
selected_name = winner["model"]
selected_cols, selected_c = configs[selected_name]

print("\nCANDIDAT B29 RETENU SUR TRAIN :", selected_name)
print("C retenu :", selected_c)

# Règle de sélection indépendante des résultats VALIDATION :
# retenir environ les 15 % les mieux notés en OOF TRAIN.
threshold = float(
    np.quantile(oof[selected_name][common], 0.85)
)

print("Seuil dérivé de TRAIN OOF :", f"{threshold:.4f}")

def summarize(label, y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    selected = p >= threshold
    n = int(selected.sum())
    wins = int(y[selected].sum())
    rate = wins / n if n else float("nan")

    print(
        f"{label:22s} | {wins:2d}/{n:2d} "
        f"| réussite={rate:.1%} "
        f"| couverture={n/len(y):.1%}"
    )

print("\n=== SEUIL FIGE — TRAIN OOF ===")
summarize(
    selected_name,
    y_oof,
    oof[selected_name][common],
    threshold
)

# Refit du candidat sur les 183 matchs TRAIN.
selected_model = make_model(selected_c)
selected_model.fit(
    train[selected_cols],
    train["y"].astype(int)
)

p_valid = selected_model.predict_proba(
    valid[selected_cols]
)[:, 1]

baseline_model = make_model(1.0)
baseline_model.fit(train[A23], train["y"].astype(int))
p_base = baseline_model.predict_proba(valid[A23])[:, 1]

y_valid = valid["y"].astype(int).to_numpy()

print("\n=== VALIDATION — DIAGNOSTIC EXPLORATOIRE ===")

for name, p in [
    ("A23_C1", p_base),
    (selected_name, p_valid),
]:
    print(
        name,
        "| AUC =", f"{roc_auc_score(y_valid, p):.4f}",
        "| AP =", f"{average_precision_score(y_valid, p):.4f}",
        "| Brier =", f"{brier_score_loss(y_valid, p):.4f}"
    )

print("\n=== SEUIL FIGE SUR TRAIN, APPLIQUE EN VALIDATION ===")
summarize(selected_name, y_valid, p_valid, threshold)

print("\n=== SEUIL 0.50 — COMPARAISON DE REFERENCE ===")
summarize(selected_name, y_valid, p_valid, 0.50)

print("\n=== STABILITE DES SELECTIONS ===")

half = len(valid) // 2
groups = [
    ("Première moitié", valid.iloc[:half], p_valid[:half]),
    ("Deuxième moitié", valid.iloc[half:], p_valid[half:]),
    ("Source ALL_320",
     valid.loc[valid.source == "ALL_320"],
     p_valid[(valid.source == "ALL_320").to_numpy()]),
    ("Source BATCH02_11",
     valid.loc[valid.source == "BATCH02_11"],
     p_valid[(valid.source == "BATCH02_11").to_numpy()]),
]

for name, group, probabilities in groups:
    summarize(
        name,
        group["y"].astype(int).to_numpy(),
        probabilities,
        threshold
    )

OUT.mkdir(parents=True, exist_ok=True)

pd.DataFrame(scores).to_csv(
    OUT / "V4_CHRONO_OOF_METRICS_V1.csv",
    index=False
)

oof_output = train.loc[
    common, ["match_id", "kickoff_utc", "y"]
].copy()

for name in configs:
    oof_output[name] = oof[name][common]

oof_output.to_csv(
    OUT / "V4_CHRONO_OOF_PREDICTIONS_V1.csv",
    index=False
)

valid_output = valid[
    ["match_id", "kickoff_utc", "source", "y"]
].copy()

valid_output["A23_C1"] = p_base
valid_output[selected_name] = p_valid
valid_output["selected_train_oof"] = p_valid >= threshold

valid_output.to_csv(
    OUT / "V4_CHRONO_VALIDATION_V1.csv",
    index=False
)

print("\n=== BILAN ===")
print("Candidat B29 :", selected_name)
print("Seuil figé depuis TRAIN OOF :", round(threshold, 4))
print("Modèle de production : NON MODIFIE")
print("TEST_FUTUR : NON EVALUE")
print("Certification : RESEARCH_ONLY")
print("ETAPE 170 TERMINEE")
