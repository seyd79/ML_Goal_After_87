from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import re

import joblib
import pandas as pd

ROOT = Path("outputs/v4_research")
FROZEN = ROOT / "frozen_simple6_v1"
PROTOCOL = FROZEN / "V4_FROZEN_PROTOCOL.json"

INPUT = ROOT / "V4_PROSPECTIVE_SNAPSHOTS_87_V1.csv"
OUTPUT = ROOT / "V4_PROSPECTIVE_PREDICTIONS_V1.csv"

COLUMNS = [
    "match_id",
    "kickoff_utc",
    "captured_at_utc",
    "snapshot_minute",
    "snapshot_source",
    "score_87_home",
    "score_87_away",
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

FEATURES = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def initialize():
    ROOT.mkdir(parents=True, exist_ok=True)

    with INPUT.open("x", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(COLUMNS)

    print("TEMPLATE CRÉÉ :", INPUT)
    print("Ajouter uniquement des snapshots réellement capturés à 87'.")
    print("Aucun match prérempli. Aucun appel API.")

def score():
    if not PROTOCOL.is_file() or not INPUT.is_file():
        raise SystemExit("Protocole ou fichier de snapshots absent")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    if protocol["features"] != FEATURES:
        raise SystemExit("Caractéristiques incompatibles")

    if protocol["threshold"] != 0.30:
        raise SystemExit("Seuil figé différent")

    frozen_at = pd.Timestamp(protocol["created_utc"])

    df = pd.read_csv(INPUT, dtype={"match_id": str})

    if not set(COLUMNS).issubset(df.columns):
        raise SystemExit("Colonnes obligatoires absentes")

    forbidden = {"goal_after_87", "target", "label", "result"}
    if set(df.columns) & forbidden:
        raise SystemExit("Le fichier de prédiction contient des labels")

    if df.empty:
        print("Aucun snapshot enregistré.")
        return

    if df.match_id.duplicated().any():
        raise SystemExit("IDs dupliqués dans les snapshots")

    if not df.match_id.str.fullmatch(r"mt_[0-9]+").all():
        raise SystemExit("IDs de match invalides")

    if df.snapshot_source.isna().any() or (
        df.snapshot_source.astype(str).str.strip() == ""
    ).any():
        raise SystemExit("Source des snapshots obligatoire")

    kickoff = pd.to_datetime(
        df.kickoff_utc, utc=True, errors="raise"
    )
    captured = pd.to_datetime(
        df.captured_at_utc, utc=True, errors="raise"
    )

    if (kickoff <= frozen_at).any():
        raise SystemExit("Match antérieur au gel des modèles")

    if (captured < frozen_at).any() or (captured < kickoff).any():
        raise SystemExit("Horodatage de capture incohérent")

    if not (
        pd.to_numeric(df.snapshot_minute, errors="raise") == 87
    ).all():
        raise SystemExit("Snapshot hors minute 87")

    numeric_columns = COLUMNS[5:]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

        if (
            df[col].isna().any()
            or (df[col] < 0).any()
            or (df[col] % 1 != 0).any()
        ):
            raise SystemExit(f"Valeurs invalides : {col}")

    for side in ("home", "away"):
        if (
            df[f"sot_{side}_last5"]
            > df[f"shots_{side}_last5"]
        ).any():
            raise SystemExit(f"Tirs cadrés > tirs : {side}")

    home = df.score_87_home
    away = df.score_87_away

    X = pd.DataFrame({
        "score_total_87": home + away,
        "score_diff_abs_87": (home-away).abs(),
        "is_draw_87": (home == away).astype(int),
        "shots_total_last5":
            df.shots_home_last5 + df.shots_away_last5,
        "sot_total_last5":
            df.sot_home_last5 + df.sot_away_last5,
        "corners_total_last5":
            df.corners_home_last5 + df.corners_away_last5,
    })[FEATURES]

    model_paths = {
        "old183": FROZEN / "SIMPLE6_OLD183.joblib",
        "new278": FROZEN / "SIMPLE6_NEW278.joblib",
    }

    predictions = {}

    for name, path in model_paths.items():
        if not path.is_file():
            raise SystemExit(f"Modèle absent : {path}")

        model = joblib.load(path)
        predictions[name] = model.predict_proba(X)[:, 1]

    existing = set()

    if OUTPUT.exists():
        with OUTPUT.open(
            encoding="utf-8", newline=""
        ) as f:
            existing = {
                r["match_id"] for r in csv.DictReader(f)
            }

    fields = COLUMNS + [
        "p_old183", "p_new278",
        "alert_old183", "alert_new278",
        "model_old_sha256", "model_new_sha256",
        "protocol_version",
    ]

    pending = []

    for i, row in df.iterrows():
        if row.match_id in existing:
            continue

        record = {col: row[col] for col in COLUMNS}
        record["p_old183"] = float(predictions["old183"][i])
        record["p_new278"] = float(predictions["new278"][i])
        record["alert_old183"] = int(record["p_old183"] >= 0.30)
        record["alert_new278"] = int(record["p_new278"] >= 0.30)
        record["model_old_sha256"] = sha256(model_paths["old183"])
        record["model_new_sha256"] = sha256(model_paths["new278"])
        record["protocol_version"] = protocol["version"]

        pending.append(record)

    if not pending:
        print("Aucun nouveau match à prédire.")
        return

    new_file = not OUTPUT.exists()

    with OUTPUT.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if new_file:
            writer.writeheader()

        writer.writerows(pending)

    print("Nouvelles prédictions :", len(pending))
    print("Alertes OLD183 :", sum(r["alert_old183"] for r in pending))
    print("Alertes NEW278 :", sum(r["alert_new278"] for r in pending))
    print("Journal :", OUTPUT)
    print("Aucun label consulté.")
    print("Aucun modèle réentraîné.")
    print("Scanner LIVE inchangé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["init", "score"]
    )
    args = parser.parse_args()

    if args.action == "init":
        initialize()
    else:
        score()
