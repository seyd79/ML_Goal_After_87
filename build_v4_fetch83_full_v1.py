from pathlib import Path
import csv
import json
import math
import importlib.util

import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"
OUT = Path("outputs/v4_research")

MANIFEST = OUT / "V4_FETCH83_MANIFEST_V1.csv"
BUILDER = Path("build_historical_ab_research_v3.py")

OUT_A = OUT / "V4_FETCH83_A23_RESEARCH_V1.csv"
OUT_B = OUT / "V4_FETCH83_B29_RESEARCH_V1.csv"
OUT_Q = OUT / "V4_FETCH83_QUALITY_RESEARCH_V1.csv"

print("=" * 72)
print("ÉTAPE 202 — RECONSTRUCTION COMPLÈTE FETCH83")
print("=" * 72)

# Préserver les anciens résultats.
for path in (OUT_A, OUT_B, OUT_Q):
    if path.exists():
        raise SystemExit(
            f"Sortie existante, remplacement interdit : {path}"
        )

# Importer le constructeur sans lancer main().
spec = importlib.util.spec_from_file_location(
    "v4_historical_builder",
    BUILDER
)

if spec is None or spec.loader is None:
    raise SystemExit("Constructeur introuvable")

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

manifest = pd.read_csv(MANIFEST, dtype=str)

if len(manifest) != 83 or manifest.match_id.duplicated().any():
    raise SystemExit("Manifeste FETCH83 invalide")

dates = pd.to_datetime(
    manifest["kickoff_utc"],
    utc=True,
    errors="raise"
)

if not (
    dates < pd.Timestamp("2026-08-25T00:00:00Z")
).all():
    raise SystemExit("Match hors période TRAIN")

# Vérifier l'absence de chevauchement.
original = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype=str,
    usecols=["match_id", "split"]
)

existing = set(original.match_id)

batch_paths = sorted(
    JOIN.glob("BATCH*_B_RESEARCH_V1.csv")
)

if len(batch_paths) != 10:
    raise SystemExit("Nombre de lots historiques incorrect")

for path in batch_paths:
    part = pd.read_csv(
        path, dtype=str, usecols=["match_id"]
    )
    ids = set(part.match_id)

    if len(ids) != len(part) or ids & existing:
        raise SystemExit(f"Anomalie historique : {path}")

    existing.update(ids)

cached12 = pd.read_csv(
    OUT / "V4_CACHED12_MANIFEST_V1.csv",
    dtype=str,
    usecols=["match_id"]
)

existing.update(cached12.match_id)

if set(manifest.match_id) & existing:
    raise SystemExit("Chevauchement avec les datasets existants")

rows_a = []
rows_b = []
rows_q = []
errors = []

print("\n=== RECONSTRUCTION ===")

for _, row in manifest.iterrows():
    mid = row["match_id"]

    try:
        cache = CACHE / f"{mid}.json"

        if not cache.is_file():
            cache = CACHE / f"mt_{mid}.json"

        if not cache.is_file():
            raise ValueError("Cache absent")

        raw = json.loads(
            cache.read_text(encoding="utf-8")
        )
        payload = raw.get("data", raw)

        if not isinstance(payload, dict):
            raise ValueError("Payload invalide")

        if str(payload.get("match_id")) != mid:
            raise ValueError("ID du cache différent")

        if payload.get("coverage") != "full":
            raise ValueError("Couverture non FULL")

        events = payload.get("events")

        if not isinstance(events, list) or not events:
            raise ValueError("Événements absents")

        if not any(
            isinstance(e, dict)
            and e.get("period") == "second_half"
            and e.get("type") == "period_end"
            for e in events
        ):
            raise ValueError("Fin de seconde période absente")

        info = {
            key: "" if pd.isna(value) else str(value)
            for key, value in row.to_dict().items()
        }

        a, b, quality = builder.build_match(
            mid, info, payload
        )

        if a is None or b is None:
            raise ValueError(
                f"Reconstruction rejetée : {quality}"
            )

        if quality.get("status") != "RESEARCH_ONLY":
            raise ValueError(
                f"Qualité insuffisante : {quality}"
            )

        if a["match_id"] != mid or b["match_id"] != mid:
            raise ValueError("ID reconstruit incohérent")

        label_a = int(a["goal_after_87"])
        label_b = int(b["goal_after_87"])

        if label_a not in (0, 1) or label_a != label_b:
            raise ValueError("Labels A/B incohérents")

        for col in builder.A_FEATURES:
            va = float(a[col])
            vb = float(b[col])

            if not (
                math.isfinite(va)
                and math.isfinite(vb)
                and va == vb
            ):
                raise ValueError(
                    f"Caractéristique A incohérente : {col}"
                )

        for col in builder.B_EXTRA:
            value = float(b[col])

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Caractéristique B invalide : {col}"
                )

        for side in ("home", "away"):
            shots = float(b[f"shots_{side}_last5"])
            sot = float(b[f"sot_{side}_last5"])

            if sot > shots:
                raise ValueError(
                    f"Tirs cadrés > tirs : {side}"
                )

        rows_a.append(a)
        rows_b.append(b)
        rows_q.append(quality)

        print(
            f"OK {mid} | label={label_a} "
            f"| événements={len(events)}"
        )

    except Exception as exc:
        errors.append((mid, str(exc)))
        print(f"ERREUR {mid} | {exc}")

print("\n=== BILAN DE RECONSTRUCTION ===")
print("Matchs attendus :", len(manifest))
print("Matchs valides :", len(rows_a))
print("Erreurs :", len(errors))

if errors:
    print("\n=== MATCHS À EXAMINER ===")
    for mid, message in errors:
        print(mid, "|", message)

    raise SystemExit(
        "Reconstruction incomplète. Aucun CSV créé."
    )

if not (
    len(rows_a) == len(rows_b) == len(rows_q) == 83
):
    raise SystemExit(
        "Effectifs incohérents. Aucun CSV créé."
    )

rows_a.sort(
    key=lambda r: (r["kickoff_utc"], r["match_id"])
)
rows_b.sort(
    key=lambda r: (r["kickoff_utc"], r["match_id"])
)
rows_q.sort(key=lambda r: r["match_id"])

def write_new(path, rows):
    columns = list(rows[0].keys())

    if any(set(r.keys()) != set(columns) for r in rows):
        raise ValueError("Colonnes de sortie incohérentes")

    with path.open(
        "x", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)

write_new(OUT_A, rows_a)
write_new(OUT_B, rows_b)
write_new(OUT_Q, rows_q)

positives = sum(
    int(r["goal_after_87"]) for r in rows_a
)

print("\n=== RÉSULTAT FINAL ===")
print("Matchs reconstruits :", len(rows_a))
print("Positifs :", positives)
print("Négatifs :", len(rows_a) - positives)
print("Taux positif :", f"{positives/83:.1%}")
print("Caractéristiques A :", len(builder.A_FEATURES))
print(
    "Caractéristiques B :",
    len(builder.A_FEATURES) + len(builder.B_EXTRA)
)

print("\n=== FICHIERS CRÉÉS ===")
print(OUT_A)
print(OUT_B)
print(OUT_Q)

print("\n=== GARANTIES ===")
print("Statut RESEARCH_ONLY.")
print("Aucun appel API.")
print("Aucun ancien dataset modifié.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Aucun entraînement.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 202 TERMINÉE.")
