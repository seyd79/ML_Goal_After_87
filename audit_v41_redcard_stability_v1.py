from pathlib import Path
import pandas as pd

SOURCE = Path(
    "outputs/v4_research/V41_REDCARD_CHRONO_OOF_V1.csv"
)

print("=" * 72)
print("ÉTAPE 224 — STABILITÉ TEMPORELLE DES CARTONS ROUGES")
print("=" * 72)

df = pd.read_csv(SOURCE, dtype={"match_id": str})

required = {
    "match_id", "kickoff_utc", "fold", "y",
    "red_card_imbalance_87",
    "SIMPLE6", "SIMPLE7_RED",
}

if not required.issubset(df.columns):
    raise SystemExit("Colonnes manquantes")

if len(df) != 121 or df.match_id.duplicated().any():
    raise SystemExit("Population OOF inattendue")

for col in ("y", "red_card_imbalance_87"):
    if not df[col].isin([0, 1]).all():
        raise SystemExit(f"Valeurs incorrectes : {col}")

if int(df.y.sum()) != 31:
    raise SystemExit("31 positifs attendus")

df["date"] = pd.to_datetime(
    df.kickoff_utc, utc=True, errors="raise"
).dt.strftime("%Y-%m-%d")

print("\n=== CARTONS ROUGES PAR JOURNÉE UTC ===")
print(
    f"{'Date':12s} {'Matchs':>7s} "
    f"{'Rouges':>7s} {'Buts':>6s} {'Taux':>8s}"
)

for date, group in df.groupby("date", sort=True):
    red = group[group.red_card_imbalance_87 == 1]

    n = len(red)
    wins = int(red.y.sum())

    rate = f"{wins/n:.1%}" if n else "-"

    print(
        f"{date:12s} {len(group):7d} "
        f"{n:7d} {wins:6d} {rate:>8s}"
    )

print("\n=== BILAN DES DEUX GROUPES ===")

for value in (0, 1):
    sub = df[df.red_card_imbalance_87 == value]
    n = len(sub)
    wins = int(sub.y.sum())

    print(
        f"Déséquilibre={value} | "
        f"matchs={n} | buts={wins} | "
        f"taux={wins/n:.1%}"
    )

print("\n=== PERFORMANCE PAR BLOC ===")

for fold, group in df.groupby("fold", sort=True):
    print(f"\nBloc {fold} | matchs={len(group)}")

    for name in ("SIMPLE6", "SIMPLE7_RED"):
        selected = group[group[name] >= 0.30]
        n = len(selected)
        wins = int(selected.y.sum())

        rate = f"{wins/n:.1%}" if n else "-"

        print(
            f"  {name:12s} | alertes={n:2d} "
            f"| buts={wins:2d} | taux={rate}"
        )

print("\n=== COUVERTURE TEMPORELLE ===")

red = df[df.red_card_imbalance_87 == 1]

print("Journées OOF :", df.date.nunique())
print("Journées avec déséquilibre :", red.date.nunique())
print("Matchs avec déséquilibre :", len(red))
print("Buts tardifs avec déséquilibre :", int(red.y.sum()))

print("\n=== GARANTIES ===")
print("Analyse descriptive uniquement.")
print("Aucun entraînement.")
print("Aucun nouveau seuil.")
print("VALIDATION non consultée.")
print("TEST_FUTUR non consulté.")
print("Modèles figés inchangés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 224 TERMINÉE.")
