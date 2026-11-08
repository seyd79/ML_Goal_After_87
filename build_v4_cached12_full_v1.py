from pathlib import Path
import csv
import json
import importlib.util

import pandas as pd

ROOT = Path("outputs/pressure_historical")
CACHE = ROOT / "timeline_cache"
OUT = Path("outputs/v4_research")

SOURCE = Path("build_historical_ab_research_v3.py")
MANIFEST = OUT / "V4_CACHED12_MANIFEST_V1.csv"
REFERENCE = OUT / "V4_CACHED12_B6_LABELS_RESEARCH_V1.csv"

OUT_A = OUT / "V4_CACHED12_A23_RESEARCH_V1.csv"
OUT_B = OUT / "V4_CACHED12_B29_RESEARCH_V1.csv"
OUT_Q = OUT / "V4_CACHED12_QUALITY_RESEARCH_V1.csv"

print("=" * 72)
print("ÉTAPE 197 — RECONSTRUCTION COMPLÈTE V4 CACHED12")
print("=" * 72)

# Ne jamais remplacer un résultat existant.
for path in (OUT_A, OUT_B, OUT_Q):
    if path.exists():
        raise SystemExit(
            f"Sortie déjà existante, remplacement refusé : {path}"
        )

# Importer le constructeur sans exécuter son main().
spec = importlib.util.spec_from_file_location(
    "historical_v3_builder_v4_research",
    SOURCE,
)

if spec is None or spec.loader is None:
    raise SystemExit("Impossible de charger le constructeur")

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

if not hasattr(builder, "build_match"):
    raise SystemExit("Fonction build_match introuvable")

manifest = pd.read_csv(MANIFEST, dtype=str)
reference = pd.read_csv(REFERENCE, dtype={"match_id": str})

if (
    len(manifest) != 12
    or manifest.match_id.duplicated().any()
    or len(reference) != 12
    or reference.match_id.duplicated().any()
):
    raise SystemExit("Population CACHED12 invalide")

if set(manifest.match_id) != set(reference.match_id):
    raise SystemExit("IDs différents entre manifest et référence")

reference = reference.set_index("match_id")

cutoff = pd.Timestamp("2026-08-25T00:00:00Z")
dates = pd.to_datetime(manifest.kickoff_utc, utc=True)

if not (dates < cutoff).all():
    raise SystemExit("Date hors période TRAIN")

rows_a = []
rows_b = []
rows_q = []
errors = []

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

for _, row in manifest.iterrows():
    mid = row["match_id"]

    path = CACHE / f"{mid}.json"

    if not path.exists():
        path = CACHE / f"mt_{mid}.json"

    if not path.exists():
        errors.append(f"{mid} : cache introuvable")
        continue

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw.get("data", raw)

        if not isinstance(payload, dict):
            raise ValueError("Payload invalide")

        if str(payload.get("match_id")) != mid:
            raise ValueError("ID différent dans le cache")

        info = {
            key: ("" if pd.isna(value) else str(value))
            for key, value in row.to_dict().items()
        }

        a, b, quality = builder.build_match(
            mid, info, payload
        )

        if a is None or b is None:
            raise ValueError(
                f"Constructeur a rejeté le match : {quality}"
            )

        if quality.get("status") != "RESEARCH_ONLY":
            raise ValueError(
                f"Qualité à examiner : {quality}"
            )

        for col in builder.A_FEATURES:
            if col not in a or col not in b:
                raise ValueError(f"Variable A absente : {col}")

            if float(a[col]) != float(b[col]):
                raise ValueError(
                    f"Variable A différente entre A et B : {col}"
                )

        for col in B6:
            if col not in b:
                raise ValueError(f"Variable B absente : {col}")

            expected = float(reference.loc[mid, col])
            actual = float(b[col])

            if abs(actual - expected) > 1e-9:
                raise ValueError(
                    f"{col}: ancien={expected}, nouveau={actual}"
                )

        label_expected = int(
            reference.loc[mid, "goal_after_87"]
        )
        label_actual = int(a["goal_after_87"])

        if label_actual != label_expected:
            raise ValueError(
                f"Label différent : "
                f"{label_expected} vs {label_actual}"
            )

        if int(b["goal_after_87"]) != label_actual:
            raise ValueError("Label A/B incohérent")

        rows_a.append(a)
        rows_b.append(b)
        rows_q.append(quality)

        print(
            f"OK {mid} | label={label_actual} "
            f"| A={len(builder.A_FEATURES)} "
            f"| B={len(builder.A_FEATURES) + len(builder.B_EXTRA)}"
        )

    except Exception as exc:
        errors.append(f"{mid} : {exc}")
        print(f"ERREUR {mid} : {exc}")

print("\n=== CONTRÔLE FINAL ===")

print("A reconstruits :", len(rows_a))
print("B reconstruits :", len(rows_b))
print("Qualité reconstruite :", len(rows_q))
print("Erreurs :", len(errors))

if errors:
    print("\n=== ANOMALIES À CORRIGER ===")
    for error in errors:
        print(error)

    raise SystemExit(
        "Reconstruction non validée. Aucun CSV créé."
    )

if not (
    len(rows_a) == len(rows_b) == len(rows_q) == 12
):
    raise SystemExit(
        "Reconstruction incomplète. Aucun CSV créé."
    )

if sum(int(a["goal_after_87"]) for a in rows_a) != 4:
    raise SystemExit("Nombre de positifs inattendu")

rows_a.sort(key=lambda r: (r["kickoff_utc"], r["match_id"]))
rows_b.sort(key=lambda r: (r["kickoff_utc"], r["match_id"]))
rows_q.sort(key=lambda r: r["match_id"])

def write_new(path, rows):
    columns = list(rows[0].keys())

    if any(set(row.keys()) != set(columns) for row in rows):
        raise ValueError(f"Colonnes incohérentes : {path}")

    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

write_new(OUT_A, rows_a)
write_new(OUT_B, rows_b)
write_new(OUT_Q, rows_q)

print("\n=== RÉSULTAT ===")
print("Matchs :", len(rows_a))
print(
    "Positifs :",
    sum(int(row["goal_after_87"]) for row in rows_a)
)
print("Variables A :", len(builder.A_FEATURES))
print(
    "Variables B :",
    len(builder.A_FEATURES) + len(builder.B_EXTRA)
)
print("Contrôle B6 : 12/12 identiques")
print("Contrôle labels : 12/12 identiques")

print("\n=== FICHIERS CRÉÉS ===")
print(OUT_A)
print(OUT_B)
print(OUT_Q)

print("\n=== GARANTIES ===")
print("Données RESEARCH_ONLY.")
print("Aucun appel API.")
print("Ancien constructeur non modifié.")
print("TRAIN original inchangé.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Aucun entraînement.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 197 TERMINÉE.")
