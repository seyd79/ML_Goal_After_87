from pathlib import Path
import pandas as pd

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

FILE = PROJECT_DIR / "outputs/model_optimization/model_optimization_predictions.csv"

print("=" * 70)
print("INSPECTION DU FICHIER DE PREDICTIONS")
print("=" * 70)

print(f"\nFichier : {FILE}")

if not FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {FILE}")

# Lecture
df = pd.read_csv(FILE)

print("\n" + "=" * 70)
print("DIMENSIONS")
print("=" * 70)

print(f"Lignes    : {len(df):,}")
print(f"Colonnes  : {len(df.columns)}")

print("\n" + "=" * 70)
print("COLONNES EXACTES")
print("=" * 70)

for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col!r}")

print("\n" + "=" * 70)
print("TYPES")
print("=" * 70)

print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("5 PREMIÈRES LIGNES")
print("=" * 70)

print(df.head(5).to_string(index=False))

print("\n" + "=" * 70)
print("VALEURS UNIQUES DES COLONNES")
print("=" * 70)

for col in df.columns:
    n_unique = df[col].nunique(dropna=False)

    if n_unique <= 20:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(20).to_string())

print("\n" + "=" * 70)
print("VALEURS MANQUANTES")
print("=" * 70)

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("Aucune valeur manquante.")
else:
    print(missing.to_string())

print("\n" + "=" * 70)
print("FIN")
print("=" * 70)