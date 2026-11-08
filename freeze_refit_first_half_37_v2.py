#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 FINAL FREEZE / REFIT

Pipeline figé AVANT ouverture du nouveau holdout:
  Features     : TOP_8 robustes
  Modèle       : LogisticRegression L2
  C            : 0.01
  Calibration  : intercept-only, fenêtre = 150 dernières prédictions OOF temporelles
  Snapshot     : minute <= 37
  Target       : goal_after_37_before_ht

Discipline:
  - entraîne uniquement sur les 2 105 matchs DEVELOPMENT V2 déjà consommés;
  - refuse tout match postérieur au 19 septembre 2026;
  - ne lit aucun nouveau holdout;
  - vérifie exactement les 8 features gelées;
  - calcule l'alpha final uniquement à partir des 150 dernières prédictions OOF
    temporelles du DEVELOPMENT V2;
  - refit ensuite le modèle de base sur les 2 105 matchs;
  - sauvegarde modèle, calibration, schéma, manifest, hashes et lock de gel.

Le holdout futur doit rester fermé jusqu'après exécution réussie de ce script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DEV = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "First_Half_37_V2_DEVELOPMENT.csv"
)
DEFAULT_RANKING = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "first_half_37_v2_robust_feature_ranking.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_frozen"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
STAGE_COL = "_v2_stage"

MODEL_NAME = "FIRST_HALF37_V2_TOP8_LOGISTIC_L2_OOF150"
MODEL_VERSION = "V2_FROZEN_2026-10"
C_FIXED = 0.01
CALIBRATION_WINDOW = 150
OOF_SPLITS = 5
EPS = 1e-8

EXPECTED_ROWS = 2105
EXPECTED_POSITIVES = 610
EXPECTED_ORIGINAL = 1802
EXPECTED_EXTERNAL = 303
MAX_ALLOWED_KICKOFF = pd.Timestamp("2026-09-19 23:59:59", tz="UTC")

EXPECTED_TOP8 = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrated_probability(raw_probability, alpha: float):
    return sigmoid(logit(raw_probability) + alpha)


def build_base_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=C_FIXED,
                solver="liblinear",
                class_weight=None,
                max_iter=5000,
                random_state=87,
            ),
        ),
    ])


def load_and_validate_development(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DEVELOPMENT V2 introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {TARGET, DATE_COL, ID_COL, STAGE_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if df[DATE_COL].isna().any():
        raise ValueError("kickoff_utc contient des valeurs non parsables.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target non résolu présent dans DEVELOPMENT V2.")

    if not set(df[TARGET].unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Target contient des valeurs autres que 0/1.")

    if df[ID_COL].duplicated().any():
        dup = df.loc[df[ID_COL].duplicated(), ID_COL].astype(str).head(10).tolist()
        raise ValueError(f"match_id dupliqué(s): {dup}")

    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    n_rows = len(df)
    n_pos = int(df[TARGET].sum())
    n_original = int(df[STAGE_COL].astype(str).eq("original_consumed").sum())
    n_external = int(df[STAGE_COL].astype(str).eq("external_consumed").sum())

    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Gel refusé: DEVELOPMENT V2={n_rows} lignes, attendu={EXPECTED_ROWS}."
        )
    if n_pos != EXPECTED_POSITIVES:
        raise RuntimeError(
            f"Gel refusé: positifs={n_pos}, attendu={EXPECTED_POSITIVES}."
        )
    if n_original != EXPECTED_ORIGINAL:
        raise RuntimeError(
            f"Gel refusé: original_consumed={n_original}, attendu={EXPECTED_ORIGINAL}."
        )
    if n_external != EXPECTED_EXTERNAL:
        raise RuntimeError(
            f"Gel refusé: external_consumed={n_external}, attendu={EXPECTED_EXTERNAL}."
        )

    max_date = df[DATE_COL].max()
    if max_date > MAX_ALLOWED_KICKOFF:
        raise RuntimeError(
            "Gel refusé: le DEVELOPMENT contient un match postérieur au "
            f"19 septembre 2026 ({max_date})."
        )

    return df


def load_and_validate_top8(ranking_path: Path, df: pd.DataFrame) -> List[str]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking V2 introuvable: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    required = {"feature", "selected_v2", "robust_selection_score"}
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"Ranking incomplet: {missing}")

    selected = ranking[ranking["selected_v2"] == 1].copy()
    selected = selected.sort_values("robust_selection_score", ascending=False)
    top8 = selected["feature"].astype(str).tolist()[:8]

    if top8 != EXPECTED_TOP8:
        raise RuntimeError(
            "Gel refusé: le TOP_8 courant diffère du TOP_8 déjà validé.\n"
            f"Attendu: {EXPECTED_TOP8}\n"
            f"Trouvé : {top8}"
        )

    absent = [f for f in top8 if f not in df.columns]
    if absent:
        raise RuntimeError(f"Features TOP_8 absentes du DEVELOPMENT: {absent}")

    return top8


def generate_temporal_oof(
    df: pd.DataFrame,
    features: List[str],
    n_splits: int = OOF_SPLITS,
) -> pd.DataFrame:
    """
    OOF expanding time series sur l'ensemble des 2 105 matchs connus.
    Chaque prédiction OOF est produite par un modèle entraîné uniquement
    sur des lignes chronologiquement antérieures.
    """
    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int).to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold_no, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        if train_idx.max() >= val_idx.min():
            raise RuntimeError("Violation temporelle détectée dans TimeSeriesSplit.")

        model = build_base_model()
        model.fit(X.iloc[train_idx], y[train_idx])

        p = model.predict_proba(X.iloc[val_idx])[:, 1]

        train_max_date = df.iloc[train_idx][DATE_COL].max()
        val_min_date = df.iloc[val_idx][DATE_COL].min()

        if train_max_date > val_min_date:
            raise RuntimeError(
                f"Leakage temporel OOF fold {fold_no}: "
                f"train_max={train_max_date}, val_min={val_min_date}"
            )

        for idx, prob in zip(val_idx, p):
            rows.append({
                "row_index": int(idx),
                ID_COL: str(df.iloc[idx][ID_COL]),
                DATE_COL: df.iloc[idx][DATE_COL],
                TARGET: int(y[idx]),
                "raw_oof_probability": float(prob),
                "oof_fold": int(fold_no),
                "train_rows_for_oof_model": int(len(train_idx)),
                "train_max_kickoff_for_oof_model": train_max_date,
            })

    oof = pd.DataFrame(rows).sort_values([DATE_COL, "row_index"]).reset_index(drop=True)

    if len(oof) < CALIBRATION_WINDOW:
        raise RuntimeError(
            f"Seulement {len(oof)} prédictions OOF; "
            f"fenêtre calibration={CALIBRATION_WINDOW} impossible."
        )

    return oof


def fit_intercept_shift(p: np.ndarray, y: np.ndarray) -> float:
    """
    Fit alpha dans:
        calibrated_p = sigmoid(logit(raw_p) + alpha)
    pente strictement figée à 1.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    y = np.asarray(y, dtype=float)

    if len(p) != len(y):
        raise ValueError("p/y longueur différente.")
    if len(p) == 0:
        raise ValueError("Calibration vide.")

    z = logit(p)
    target_rate = float(y.mean())

    lo, hi = -8.0, 8.0
    for _ in range(160):
        mid = (lo + hi) / 2.0
        mean_pred = float(sigmoid(z + mid).mean())
        if mean_pred < target_rate:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)


def freeze_final_pipeline(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[Pipeline, float, pd.DataFrame, Dict]:
    oof = generate_temporal_oof(df, features, n_splits=OOF_SPLITS)

    calibration_slice = oof.tail(CALIBRATION_WINDOW).copy()

    alpha = fit_intercept_shift(
        calibration_slice["raw_oof_probability"].to_numpy(),
        calibration_slice[TARGET].to_numpy(),
    )

    calibration_slice["calibrated_oof_probability"] = calibrated_probability(
        calibration_slice["raw_oof_probability"].to_numpy(),
        alpha,
    )

    X_all = df[features].apply(pd.to_numeric, errors="coerce")
    y_all = df[TARGET].astype(int).to_numpy()

    final_model = build_base_model()
    final_model.fit(X_all, y_all)

    raw_train = final_model.predict_proba(X_all)[:, 1]
    calibrated_train = calibrated_probability(raw_train, alpha)

    diag = {
        "development_rows": int(len(df)),
        "development_positives": int(y_all.sum()),
        "development_positive_rate": float(y_all.mean()),
        "development_date_min": str(df[DATE_COL].min()),
        "development_date_max": str(df[DATE_COL].max()),
        "oof_rows_total": int(len(oof)),
        "oof_splits": OOF_SPLITS,
        "calibration_window": CALIBRATION_WINDOW,
        "calibration_date_min": str(calibration_slice[DATE_COL].min()),
        "calibration_date_max": str(calibration_slice[DATE_COL].max()),
        "calibration_rows": int(len(calibration_slice)),
        "calibration_actual_rate": float(calibration_slice[TARGET].mean()),
        "calibration_raw_mean_probability": float(
            calibration_slice["raw_oof_probability"].mean()
        ),
        "calibration_alpha_intercept_shift": float(alpha),
        "calibration_calibrated_mean_probability": float(
            calibration_slice["calibrated_oof_probability"].mean()
        ),
        "final_refit_raw_mean_probability_on_development": float(raw_train.mean()),
        "final_refit_calibrated_mean_probability_on_development": float(
            calibrated_train.mean()
        ),
    }

    return final_model, alpha, oof, diag


def model_coefficients(
    fitted_pipeline: Pipeline,
    features: List[str],
) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]
    rows = []
    for f, coef in zip(features, model.coef_[0]):
        rows.append({
            "feature": f,
            "coefficient_standardized": float(coef),
            "abs_coefficient": abs(float(coef)),
        })
    return pd.DataFrame(rows).sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def write_report(
    path: Path,
    features: List[str],
    diag: Dict,
    coef_df: pd.DataFrame,
    dev_sha: str,
    ranking_sha: str,
):
    lines = []
    lines.append("# First Half 37 — V2 FINAL FREEZE / REFIT")
    lines.append("")
    lines.append("## Frozen pipeline")
    lines.append("")
    lines.append(f"- Model name: `{MODEL_NAME}`")
    lines.append(f"- Version: `{MODEL_VERSION}`")
    lines.append("- Base model: `LogisticRegression L2`")
    lines.append(f"- C: `{C_FIXED}`")
    lines.append("- Class weight: `None`")
    lines.append("- Imputation: median")
    lines.append("- Scaling: StandardScaler")
    lines.append(
        f"- Calibration: intercept-only, last `{CALIBRATION_WINDOW}` temporal OOF predictions"
    )
    lines.append(
        f"- Final alpha: `{diag['calibration_alpha_intercept_shift']:+.12f}`"
    )
    lines.append("")
    lines.append("## Features")
    lines.append("")
    for i, f in enumerate(features, 1):
        lines.append(f"{i}. `{f}`")
    lines.append("")
    lines.append("## Development frozen")
    lines.append("")
    lines.append(
        f"- Rows: **{diag['development_rows']}** | "
        f"positives: **{diag['development_positives']}** "
        f"({diag['development_positive_rate']:.2%})"
    )
    lines.append(
        f"- Dates: `{diag['development_date_min']}` -> `{diag['development_date_max']}`"
    )
    lines.append(f"- DEVELOPMENT SHA256: `{dev_sha}`")
    lines.append(f"- Ranking SHA256: `{ranking_sha}`")
    lines.append("")
    lines.append("## Final OOF150 calibration")
    lines.append("")
    lines.append(
        f"- OOF rows total: **{diag['oof_rows_total']}** across **{diag['oof_splits']}** temporal splits"
    )
    lines.append(
        f"- Calibration rows: **{diag['calibration_rows']}**"
    )
    lines.append(
        f"- Calibration dates: `{diag['calibration_date_min']}` -> `{diag['calibration_date_max']}`"
    )
    lines.append(
        f"- Actual rate in OOF150: **{diag['calibration_actual_rate']:.2%}**"
    )
    lines.append(
        f"- Mean raw OOF probability: **{diag['calibration_raw_mean_probability']:.2%}**"
    )
    lines.append(
        f"- Mean calibrated OOF probability: **{diag['calibration_calibrated_mean_probability']:.2%}**"
    )
    lines.append("")
    lines.append("## Standardized coefficients")
    lines.append("")
    lines.append("```")
    lines.append(coef_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Freeze rule")
    lines.append("")
    lines.append(
        "À partir de ce gel, aucun choix de feature, C, calibration, fenêtre, "
        "prétraitement ou seuil ne doit être modifié après consultation du nouveau holdout."
    )
    lines.append(
        "Le prochain holdout doit contenir uniquement des matchs postérieurs au "
        "19 septembre 2026 et doit rester intact jusqu'au premier scoring final."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze/refit final du First Half 37 V2 avant nouveau holdout."
    )
    parser.add_argument("--development-csv", default=str(DEFAULT_DEV))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise explicitement le remplacement d'un gel déjà présent.",
    )
    args = parser.parse_args()

    dev_path = Path(args.development_csv)
    ranking_path = Path(args.ranking_csv)
    out_dir = Path(args.output_dir)

    lock_path = out_dir / "FIRST_HALF37_V2_FREEZE_LOCK.json"

    if lock_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Gel déjà présent: {lock_path}\n"
            "Par sécurité, le script refuse de le remplacer.\n"
            "Utiliser --overwrite uniquement si le remplacement est volontaire "
            "AVANT ouverture du futur holdout."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V2 FINAL FREEZE / REFIT")
    print("=" * 120)
    print(f"DEVELOPMENT : {dev_path}")
    print(f"Ranking     : {ranking_path}")
    print(f"Output      : {out_dir}")
    print(f"Model       : {MODEL_NAME}")
    print(f"C           : {C_FIXED}")
    print(f"Calibration : intercept-only | last {CALIBRATION_WINDOW} temporal OOF")
    print("IMPORTANT   : aucun match postérieur au 19 septembre 2026 ne sera accepté.")
    print("IMPORTANT   : aucun futur holdout n'est lu.")
    print()

    df = load_and_validate_development(dev_path)
    features = load_and_validate_top8(ranking_path, df)

    dev_sha = sha256_file(dev_path)
    ranking_sha = sha256_file(ranking_path)

    print(
        f"DEVELOPMENT validé : {len(df)} matchs | "
        f"{int(df[TARGET].sum())} positifs ({df[TARGET].mean():.2%})"
    )
    print(
        f"Dates : {df[DATE_COL].min()} -> {df[DATE_COL].max()}"
    )
    print("TOP_8 validé :")
    for i, f in enumerate(features, 1):
        print(f"  {i:02d}. {f}")
    print()

    final_model, alpha, oof, diag = freeze_final_pipeline(df, features)
    coef_df = model_coefficients(final_model, features)

    calibration_slice = oof.tail(CALIBRATION_WINDOW).copy()
    calibration_slice["calibrated_oof_probability"] = calibrated_probability(
        calibration_slice["raw_oof_probability"].to_numpy(),
        alpha,
    )

    model_path = out_dir / "FIRST_HALF37_V2_FROZEN.joblib"
    calibration_path = out_dir / "first_half_37_v2_frozen_calibration.json"
    schema_path = out_dir / "first_half_37_v2_frozen_schema.json"
    manifest_path = out_dir / "first_half_37_v2_freeze_manifest.json"
    oof_path = out_dir / "first_half_37_v2_final_temporal_oof.csv"
    oof150_path = out_dir / "first_half_37_v2_final_oof150_calibration.csv"
    coef_path = out_dir / "first_half_37_v2_final_coefficients.csv"
    report_path = out_dir / "first_half_37_v2_freeze_report.md"

    bundle = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "base_model": final_model,
        "features": features,
        "target": TARGET,
        "C": C_FIXED,
        "calibration": {
            "type": "intercept_only",
            "alpha": float(alpha),
            "slope": 1.0,
            "window": CALIBRATION_WINDOW,
            "source": "last_150_temporal_oof_predictions_on_frozen_development",
        },
        "prediction_formula": "sigmoid(logit(base_model_probability) + alpha)",
        "development_sha256": dev_sha,
        "ranking_sha256": ranking_sha,
    }
    joblib.dump(bundle, model_path)

    calibration = {
        "type": "intercept_only",
        "alpha": float(alpha),
        "slope": 1.0,
        "window": CALIBRATION_WINDOW,
        "oof_splits": OOF_SPLITS,
        "oof_rows_total": int(len(oof)),
        "calibration_rows": int(len(calibration_slice)),
        "calibration_date_min": str(calibration_slice[DATE_COL].min()),
        "calibration_date_max": str(calibration_slice[DATE_COL].max()),
        "calibration_actual_rate": float(calibration_slice[TARGET].mean()),
        "calibration_raw_mean_probability": float(
            calibration_slice["raw_oof_probability"].mean()
        ),
        "calibration_calibrated_mean_probability": float(
            calibration_slice["calibrated_oof_probability"].mean()
        ),
        "formula": "p_cal = sigmoid(logit(p_raw) + alpha)",
    }
    calibration_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = {
        "schema_version": "first_half_37_v2_frozen_schema_v1",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "snapshot_cutoff": "minute <= 37",
        "target": TARGET,
        "target_definition": (
            "1 iff a valid first-half goal occurs at minute >37 before first-half "
            "period end, including stoppage time; minute 37 belongs to predictors."
        ),
        "feature_order": features,
        "n_features": len(features),
        "missing_policy": "missing remains missing until pipeline median imputation",
        "imputer": "SimpleImputer(strategy='median') fit on all 2105 frozen development rows",
        "scaler": "StandardScaler fit on all 2105 frozen development rows",
        "model": {
            "class": "LogisticRegression",
            "penalty": "l2",
            "C": C_FIXED,
            "solver": "liblinear",
            "class_weight": None,
            "max_iter": 5000,
            "random_state": 87,
        },
        "calibration": calibration,
        "opening_odds_policy": "not used in frozen TOP_8",
        "future_holdout_used": False,
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    oof.to_csv(oof_path, index=False)
    calibration_slice.to_csv(oof150_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    created_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": "first_half_37_v2_freeze_manifest_v1",
        "created_at_utc": created_at,
        "status": "FROZEN_BEFORE_NEW_HOLDOUT",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "development": {
            "path": str(dev_path),
            "sha256": dev_sha,
            "rows": int(len(df)),
            "positives": int(df[TARGET].sum()),
            "positive_rate": float(df[TARGET].mean()),
            "date_min": str(df[DATE_COL].min()),
            "date_max": str(df[DATE_COL].max()),
            "original_consumed_rows": int(
                df[STAGE_COL].astype(str).eq("original_consumed").sum()
            ),
            "external_consumed_rows": int(
                df[STAGE_COL].astype(str).eq("external_consumed").sum()
            ),
        },
        "ranking": {
            "path": str(ranking_path),
            "sha256": ranking_sha,
        },
        "pipeline": {
            "features": features,
            "model": "LogisticRegression L2",
            "C": C_FIXED,
            "class_weight": None,
            "imputer": "median",
            "scaler": "StandardScaler",
            "calibration": calibration,
        },
        "diagnostics": diag,
        "holdout_policy": {
            "future_holdout_used": False,
            "minimum_kickoff_for_new_holdout": "strictly after 2026-09-19",
            "rule": (
                "No feature/model/C/calibration/window/preprocessing changes "
                "after opening the new holdout."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "model_bundle": str(model_path),
            "calibration": str(calibration_path),
            "schema": str(schema_path),
            "manifest": str(manifest_path),
            "temporal_oof": str(oof_path),
            "oof150_calibration": str(oof150_path),
            "coefficients": str(coef_path),
            "report": str(report_path),
            "freeze_lock": str(lock_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        report_path,
        features=features,
        diag=diag,
        coef_df=coef_df,
        dev_sha=dev_sha,
        ranking_sha=ranking_sha,
    )

    # Reload self-test.
    reloaded = joblib.load(model_path)
    sample = df.head(32)
    X_sample = sample[features].apply(pd.to_numeric, errors="coerce")

    p_a = final_model.predict_proba(X_sample)[:, 1]
    p_b = reloaded["base_model"].predict_proba(X_sample)[:, 1]

    if not np.allclose(p_a, p_b, atol=1e-12, rtol=0):
        raise RuntimeError("Self-test reload échoué: probabilités RAW différentes.")

    p_cal_a = calibrated_probability(p_a, alpha)
    p_cal_b = calibrated_probability(
        p_b,
        float(reloaded["calibration"]["alpha"]),
    )
    if not np.allclose(p_cal_a, p_cal_b, atol=1e-12, rtol=0):
        raise RuntimeError("Self-test reload échoué: probabilités calibrées différentes.")

    model_sha = sha256_file(model_path)
    schema_sha = sha256_file(schema_path)
    manifest_sha = sha256_file(manifest_path)

    lock = {
        "lock_version": "first_half_37_v2_freeze_lock_v1",
        "created_at_utc": created_at,
        "status": "LOCKED_BEFORE_NEW_HOLDOUT",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_sha256": model_sha,
        "schema_sha256": schema_sha,
        "manifest_sha256": manifest_sha,
        "development_sha256": dev_sha,
        "ranking_sha256": ranking_sha,
        "features": features,
        "C": C_FIXED,
        "calibration_type": "intercept_only",
        "calibration_window": CALIBRATION_WINDOW,
        "calibration_alpha": float(alpha),
        "future_holdout_used": False,
        "do_not_modify_after_holdout_open": True,
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 120)
    print("V2 FINAL FREEZE / REFIT TERMINE")
    print("=" * 120)
    print(f"Rows DEVELOPMENT : {len(df)}")
    print(f"Positifs         : {int(df[TARGET].sum())} ({df[TARGET].mean():.2%})")
    print(f"Features         : {len(features)}/8")
    print(f"C                : {C_FIXED}")
    print(f"OOF total        : {len(oof)}")
    print(f"Calibration      : last {CALIBRATION_WINDOW} OOF")
    print(
        f"OOF150 dates     : {calibration_slice[DATE_COL].min()} "
        f"-> {calibration_slice[DATE_COL].max()}"
    )
    print(f"OOF150 réel      : {calibration_slice[TARGET].mean():.2%}")
    print(
        f"OOF150 RAW mean  : "
        f"{calibration_slice['raw_oof_probability'].mean():.2%}"
    )
    print(
        f"OOF150 CAL mean  : "
        f"{calibration_slice['calibrated_oof_probability'].mean():.2%}"
    )
    print(f"Alpha final      : {alpha:+.12f}")
    print(f"Model            : {model_path}")
    print(f"Schema           : {schema_path}")
    print(f"Calibration      : {calibration_path}")
    print(f"Manifest         : {manifest_path}")
    print(f"Freeze lock      : {lock_path}")
    print(f"Model SHA256     : {model_sha}")
    print()
    print("SELF-TEST RELOAD : PASS")
    print("FUTUR HOLDOUT    : TOUJOURS FERME")
    print("IMPORTANT        : ne plus modifier features/C/calibration/window après ouverture du holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
