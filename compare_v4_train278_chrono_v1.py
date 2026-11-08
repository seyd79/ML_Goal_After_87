from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

OUT = Path("outputs/v4_research")

TRAIN_PATH = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
OOF_PATH = OUT / "V4_SIMPLE_CHRONO_OOF_V1.csv"
DEST = OUT / "V4_TRAIN278_PAIRED_OOF_V1.csv"

EMBARGO = pd.Timedelta(hours=4)

print("=" * 72)
print("ÉTAPE 204 — COMPARAISON CHRONOLOGIQUE SIMPLE6")
print("=" * 72)

if DEST.exists():
    raise SystemExit("Sortie déjà existante : remplacement refusé")

train = pd.read_csv(TRAIN_PATH, dtype={"match_id": str})
oof = pd.read_csv(OOF_PATH, dtype={"match_id": str})

required_train = {
    "match_id", "kickoff_utc", "goal_after_87",
    "research_origin", "score_87_home", "score_87_away",
    "is_draw_87", "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
}

required_oof = {
    "match_id", "kickoff_utc", "y",
    "fold", "SIMPLE6_C01",
}

if not required_train.issubset(train.columns):
    raise SystemExit("Colonnes TRAIN insuffisantes")

if not required_oof.issubset(oof.columns):
    raise SystemExit("Colonnes OOF insuffisantes")

if len(train) != 278 or train.match_id.duplicated().any():
    raise SystemExit("TRAIN278 incorrect")

if len(oof) != 121 or oof.match_id.duplicated().any():
    raise SystemExit("OOF historique incorrect")

train["kickoff"] = pd.to_datetime(
    train.kickoff_utc, utc=True, errors="raise"
)

oof["kickoff"] = pd.to_datetime(
    oof.kickoff_utc, utc=True, errors="raise"
)

old = train[
    ~train.research_origin.isin(
        ["V4_CACHED12", "V4_FETCH83"]
    )
].copy()

if len(old) != 183:
    raise SystemExit("TRAIN historique différent de 183")

if not set(oof.match_id).issubset(set(old.match_id)):
    raise SystemExit("OOF contient des IDs hors TRAIN historique")

lookup = train.set_index("match_id")
labels = lookup.loc[oof.match_id, "goal_after_87"].to_numpy()

if not np.array_equal(
    labels.astype(int),
    oof.y.to_numpy().astype(int)
):
    raise SystemExit("Labels OOF différents du TRAIN")

if oof.fold.nunique() != 4:
    raise SystemExit("Quatre blocs OOF attendus")

if int(oof.y.sum()) != 31:
    raise SystemExit("31 positifs OOF attendus")


def features(frame):
    f = pd.DataFrame(index=frame.index)

    home = pd.to_numeric(frame.score_87_home)
    away = pd.to_numeric(frame.score_87_away)

    f["score_total_87"] = home + away
    f["score_diff_abs_87"] = (home - away).abs()
    f["is_draw_87"] = pd.to_numeric(frame.is_draw_87)

    f["shots_total_last5"] = (
        pd.to_numeric(frame.shots_home_last5)
        + pd.to_numeric(frame.shots_away_last5)
    )

    f["sot_total_last5"] = (
        pd.to_numeric(frame.sot_home_last5)
        + pd.to_numeric(frame.sot_away_last5)
    )

    f["corners_total_last5"] = (
        pd.to_numeric(frame.corners_home_last5)
        + pd.to_numeric(frame.corners_away_last5)
    )

    return f


def fit_predict(fit_rows, test_rows):
    x_train = features(fit_rows)
    x_test = features(test_rows)
    y_train = fit_rows.goal_after_87.astype(int)

    if y_train.nunique() != 2:
        raise ValueError("Une seule classe dans TRAIN")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=3000,
            solver="lbfgs",
            random_state=42,
        ),
    )

    model.fit(x_train, y_train)

    return model.predict_proba(x_test)[:, 1]


predictions = []
expected_old_sizes = [50, 76, 114, 128]

print("\n=== RECONSTRUCTION DES BLOCS ===")

for number, (fold, block) in enumerate(
    oof.groupby("fold", sort=True), start=1
):
    block = block.copy()

    cutoff = block.kickoff.min() - EMBARGO

    fit_old = old[old.kickoff < cutoff].copy()
    fit_new = train[train.kickoff < cutoff].copy()

    n_old = len(fit_old)
    n_new = len(fit_new)

    print(
        f"BLOC {number} | OOF={len(block)} "
        f"| TRAIN ancien={n_old} "
        f"| TRAIN enrichi={n_new} "
        f"| cutoff={cutoff}"
    )

    if n_old != expected_old_sizes[number-1]:
        raise SystemExit(
            f"BLOC {number} : attendu "
            f"{expected_old_sizes[number-1]} exemples anciens, "
            f"obtenu {n_old}. "
            "Définition des folds à vérifier avant comparaison."
        )

    if n_new < n_old:
        raise SystemExit("TRAIN enrichi plus petit")

    if set(block.match_id) & set(fit_new.match_id):
        raise SystemExit("Fuite temporelle : ID OOF dans TRAIN")

    if fit_new.kickoff.max() >= cutoff:
        raise SystemExit("Fuite temporelle détectée")

    test_rows = lookup.loc[block.match_id].reset_index()

    p_old = fit_predict(fit_old, test_rows)
    p_new = fit_predict(fit_new, test_rows)

    block["P_OLD183"] = p_old
    block["P_NEW278"] = p_new

    predictions.append(block)

result = pd.concat(
    predictions,
    ignore_index=True
).sort_values(["kickoff", "match_id"])

print("\n=== PERFORMANCES GLOBALES ===")


def report(name, data, column):
    y = data.y.astype(int)
    p = data[column].astype(float)

    print(
        f"{name:17s} | "
        f"AUC={roc_auc_score(y, p):.4f} | "
        f"AP={average_precision_score(y, p):.4f} | "
        f"Brier={brier_score_loss(y, p):.4f}"
    )


report("ANCIEN 183", result, "P_OLD183")
report("ENRICHI 278", result, "P_NEW278")
report("RÉFÉRENCE 186", result, "SIMPLE6_C01")

print("\n=== TOP 10 PAR BLOC ===")

for number, (fold, block) in enumerate(
    result.groupby("fold", sort=True), start=1
):
    print(
        f"\nBLOC {number} | n={len(block)} "
        f"| positifs={int(block.y.sum())}"
    )

    for model in ["P_OLD183", "P_NEW278"]:
        top = block.nlargest(10, model)
        wins = int(top.y.sum())

        print(
            f"  {model:12s} : "
            f"{wins}/10 = {wins/10:.0%}"
        )

print("\n=== SEUIL FIXE 0.30 ===")

for model in ["P_OLD183", "P_NEW278"]:
    selected = result[result[model] >= 0.30]
    n = len(selected)
    wins = int(selected.y.sum())

    print(
        f"{model:12s} | "
        f"sélections={n} | "
        f"réussites={wins} | "
        f"taux={wins/n:.1%}" if n
        else f"{model:12s} | aucune sélection"
    )

columns = [
    "match_id", "kickoff_utc", "fold", "y",
    "SIMPLE6_C01", "P_OLD183", "P_NEW278",
]

result[columns].to_csv(DEST, index=False)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("Même population OOF pour les deux modèles.")
print("Embargo chronologique : 4 heures.")
print("Même architecture et même régularisation.")
print("Aucun réglage de seuil.")
print("Aucune utilisation de VALIDATION ou TEST_FUTUR.")
print("Scanner V3-C inchangé.")
print("Résultats exploratoires, pas certification LIVE.")
