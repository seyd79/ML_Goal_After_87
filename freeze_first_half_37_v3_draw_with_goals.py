#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

FS_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_feature_selection"
BENCH_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_model_benchmark"
CAL_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_calibration_audit"
OUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_frozen"

DATASET_PATH = FS_DIR / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
FEATURE_SCHEMA_PATH = FS_DIR / "first_half_37_v3_draw_with_goals_selected_features.json"
FS_MANIFEST_PATH = FS_DIR / "first_half_37_v3_draw_with_goals_feature_selection_manifest.json"
BENCH_RANKING_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_model_ranking.csv"
BENCH_MANIFEST_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_model_benchmark_manifest.json"
CAL_SUMMARY_PATH = CAL_DIR / "first_half_37_v3_draw_with_goals_calibration_summary.csv"
CAL_MANIFEST_PATH = CAL_DIR / "first_half_37_v3_draw_with_goals_calibration_manifest.json"

MODEL_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FROZEN.joblib"
FREEZE_LOCK_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FREEZE_LOCK.json"
SCHEMA_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_SCHEMA.json"
TRAINING_MANIFEST_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_TRAINING_MANIFEST.json"
OOF_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_OOF_CALIBRATION.csv"
IMPORTANCE_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_FEATURE_IMPORTANCE.csv"
SELF_TEST_PATH = OUT_DIR / "FIRST_HALF37_V3_DRAW_WITH_GOALS_SELF_TEST.json"

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
CONTEXT_COL = "score_context_37"

MODEL_NAME = "FIRST_HALF37_V3_DRAW_WITH_GOALS_TOP5_EXTRATREES_INTERCEPT"
MODEL_VERSION = "V3_DWG_1"
EXPECTED_CONFIG = "NEW_TOP5__extra_trees_balanced"
EXPECTED_CALIBRATION = "ROLLING_INTERCEPT"

FEATURES = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
]

MODEL_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=87,
    n_jobs=-1,
)

EXPECTED_ROWS = 240
EXPECTED_POSITIVES = 86
N_OOF_SPLITS = 5
EPS = 1e-10


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON invalide: {path}")
    return value


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_alpha(y, p) -> float:
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    lo, hi = -12.0, 12.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        q = sigmoid(lp + mid)
        f = float(np.sum(y - q))
        if f > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def safe_roc(y, p) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr(y, p) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def ece(y, p, bins=10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (p <= hi if i == bins-1 else p < hi)
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pr = safe_pr(y, p)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "roc_auc": safe_roc(y, p),
        "pr_auc": pr,
        "pr_lift_vs_prevalence": pr - float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0, 1])),
        "ece_10": ece(y, p),
    }


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(**MODEL_PARAMS)),
    ])


def validate_and_load():
    for p in [
        DATASET_PATH, FEATURE_SCHEMA_PATH, FS_MANIFEST_PATH,
        BENCH_RANKING_PATH, BENCH_MANIFEST_PATH,
        CAL_SUMMARY_PATH, CAL_MANIFEST_PATH,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    feature_schema = load_json(FEATURE_SCHEMA_PATH)
    fs_manifest = load_json(FS_MANIFEST_PATH)
    bench_manifest = load_json(BENCH_MANIFEST_PATH)
    cal_manifest = load_json(CAL_MANIFEST_PATH)

    if fs_manifest.get("new_future_holdout_used") is not False:
        raise RuntimeError("Feature selection manifest invalide.")
    if bench_manifest.get("new_future_holdout_used") is not False:
        raise RuntimeError("Benchmark manifest invalide.")
    if cal_manifest.get("new_future_holdout_used") is not False:
        raise RuntimeError("Calibration manifest invalide.")

    selected = list(feature_schema.get("selected_features", []))
    if selected[:5] != FEATURES:
        raise RuntimeError(f"TOP5 inattendu: {selected[:5]}")

    bench = pd.read_csv(BENCH_RANKING_PATH, low_memory=False)
    b = bench[bench["config"].astype(str).eq(EXPECTED_CONFIG)]
    if len(b) != 1 or int(b.iloc[0]["passes_research_gate"]) != 1:
        raise RuntimeError("Le candidat benchmark choisi n'est pas un PASS unique.")

    cal = pd.read_csv(CAL_SUMMARY_PATH, low_memory=False)
    passed = cal[cal["passes_calibration_gate"] == 1]
    if len(passed) != 1:
        raise RuntimeError(f"Calibration PASS attendue=1, trouvée={len(passed)}.")

    r = passed.iloc[0]
    if str(r["config"]) != EXPECTED_CONFIG or str(r["strategy"]) != EXPECTED_CALIBRATION:
        raise RuntimeError(
            f"Calibration gagnante inattendue: {r['config']} + {r['strategy']}"
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Rows={len(df)}, attendu={EXPECTED_ROWS}.")

    required = {
        ID_COL, DATE_COL, TARGET, CONTEXT_COL,
        "score_37_home", "score_37_away", *FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Colonnes absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if df[DATE_COL].isna().any() or df[TARGET].isna().any():
        raise RuntimeError("Date ou target invalide.")

    df[TARGET] = df[TARGET].astype(int)
    if int(df[TARGET].sum()) != EXPECTED_POSITIVES:
        raise RuntimeError(f"Positifs={int(df[TARGET].sum())}, attendu={EXPECTED_POSITIVES}.")
    if df[ID_COL].astype(str).duplicated().any():
        raise RuntimeError("match_id dupliqué.")
    if not df[CONTEXT_COL].astype(str).eq("draw_with_goals").all():
        raise RuntimeError("Contexte inattendu.")

    h = pd.to_numeric(df["score_37_home"], errors="coerce")
    a = pd.to_numeric(df["score_37_away"], errors="coerce")
    if (h.isna() | a.isna() | (h != a) | (h <= 0)).any():
        raise RuntimeError("Règle draw_with_goals violée.")

    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    hashes = {
        "dataset_sha256": sha256_file(DATASET_PATH),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
        "feature_selection_manifest_sha256": sha256_file(FS_MANIFEST_PATH),
        "benchmark_ranking_sha256": sha256_file(BENCH_RANKING_PATH),
        "benchmark_manifest_sha256": sha256_file(BENCH_MANIFEST_PATH),
        "calibration_summary_sha256": sha256_file(CAL_SUMMARY_PATH),
        "calibration_manifest_sha256": sha256_file(CAL_MANIFEST_PATH),
    }
    return df, hashes


def temporal_oof(df: pd.DataFrame) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=N_OOF_SPLITS)
    pieces: List[pd.DataFrame] = []

    for fold_idx, (tr, va) in enumerate(splitter.split(df), start=1):
        train = df.iloc[tr]
        val = df.iloc[va]

        ytr = train[TARGET].to_numpy(dtype=int)
        yva = val[TARGET].to_numpy(dtype=int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            raise RuntimeError(f"OOF fold {fold_idx}: une seule classe.")

        model = make_pipeline()
        Xtr = train[FEATURES].apply(pd.to_numeric, errors="coerce")
        Xva = val[FEATURES].apply(pd.to_numeric, errors="coerce")
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]

        part = val[[ID_COL, DATE_COL, TARGET]].copy()
        part["oof_fold"] = fold_idx
        part["train_n"] = len(train)
        part["raw_probability"] = p
        pieces.append(part)

    return pd.concat(pieces, ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS FINAL REFIT + FREEZE")
    print("=" * 120)
    print("AUCUN NOUVEAU HOLDOUT FUTUR")
    print()

    df, upstream_hashes = validate_and_load()
    y = df[TARGET].to_numpy(dtype=int)

    print(f"TRAIN final : {len(df)} | positifs={int(y.sum())} ({y.mean():.2%})")
    print("Features    : " + ", ".join(FEATURES))
    print()

    oof = temporal_oof(df)
    y_oof = oof[TARGET].to_numpy(dtype=int)
    p_raw = oof["raw_probability"].to_numpy(dtype=float)

    alpha = fit_alpha(y_oof, p_raw)
    p_cal = sigmoid(logit(p_raw) + alpha)
    oof["calibrated_probability"] = p_cal

    raw_m = metrics(y_oof, p_raw)
    cal_m = metrics(y_oof, p_cal)

    print(f"OOF N       : {len(oof)}")
    print(f"Alpha       : {alpha:+.12f}")
    print(
        f"OOF RAW     : ROC={raw_m['roc_auc']:.4f} | PR={raw_m['pr_auc']:.4f} | "
        f"Brier={raw_m['brier']:.4f} | LL={raw_m['log_loss']:.4f} | ECE={raw_m['ece_10']:.4f}"
    )
    print(
        f"OOF CAL     : ROC={cal_m['roc_auc']:.4f} | PR={cal_m['pr_auc']:.4f} | "
        f"Brier={cal_m['brier']:.4f} | LL={cal_m['log_loss']:.4f} | ECE={cal_m['ece_10']:.4f}"
    )
    print()

    final_model = make_pipeline()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    final_model.fit(X, y)

    bundle = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at_utc": now_utc_iso(),
        "status": "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT",
        "context": {
            "name": "draw_with_goals",
            "cutoff_minute": 37,
            "rule": "score_37_home == score_37_away and score_37_home > 0",
        },
        "target": TARGET,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "base_model": final_model,
        "base_model_type": "ExtraTreesClassifier",
        "base_model_params": MODEL_PARAMS,
        "calibration": {
            "selected_strategy_from_audit": EXPECTED_CALIBRATION,
            "deployment_form": "fixed_temporal_oof_intercept_only",
            "formula": "sigmoid(logit(p_raw) + alpha)",
            "alpha": float(alpha),
            "oof_splits": N_OOF_SPLITS,
            "oof_rows": int(len(oof)),
        },
        "training": {
            "rows": int(len(df)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "date_min_utc": df[DATE_COL].min().isoformat(),
            "date_max_utc": df[DATE_COL].max().isoformat(),
            "dataset_sha256": upstream_hashes["dataset_sha256"],
        },
        "upstream_hashes": upstream_hashes,
    }

    joblib.dump(bundle, MODEL_PATH)
    model_sha = sha256_file(MODEL_PATH)

    oof.to_csv(OOF_PATH, index=False)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_model.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    SCHEMA_PATH.write_text(json.dumps({
        "schema_version": "first_half_37_v3_draw_with_goals_schema_v1",
        "model_name": MODEL_NAME,
        "context": "draw_with_goals",
        "cutoff_minute": 37,
        "score_context_rule": "score_37_home == score_37_away and score_37_home > 0",
        "target": TARGET,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "missing_value_policy": "NaN remains missing; frozen median imputer inside pipeline",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    TRAINING_MANIFEST_PATH.write_text(json.dumps({
        "schema_version": "first_half_37_v3_draw_with_goals_training_manifest_v1",
        "created_at_utc": now_utc_iso(),
        "model_name": MODEL_NAME,
        "model_sha256": model_sha,
        "training_rows": int(len(df)),
        "training_positives": int(y.sum()),
        "training_positive_rate": float(y.mean()),
        "features": FEATURES,
        "base_model_params": MODEL_PARAMS,
        "calibration_alpha": float(alpha),
        "oof_raw_metrics": raw_m,
        "oof_calibrated_metrics": cal_m,
        "oof_rows": int(len(oof)),
        "new_future_holdout_used": False,
        "upstream_hashes": upstream_hashes,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reload self-test.
    reloaded = joblib.load(MODEL_PATH)
    test = df.iloc[[-1]]
    test_X = test[FEATURES].apply(pd.to_numeric, errors="coerce")

    raw_before = float(final_model.predict_proba(test_X)[0, 1])
    raw_after = float(reloaded["base_model"].predict_proba(test_X)[0, 1])
    cal_before = float(sigmoid(logit([raw_before]) + alpha)[0])
    cal_after = float(sigmoid(logit([raw_after]) + float(reloaded["calibration"]["alpha"]))[0])

    if abs(raw_before - raw_after) > 1e-12 or abs(cal_before - cal_after) > 1e-12:
        raise RuntimeError("SELF-TEST reload FAIL.")

    SELF_TEST_PATH.write_text(json.dumps({
        "status": "PASS",
        "tested_match_id": str(test.iloc[0][ID_COL]),
        "raw_probability_before_save": raw_before,
        "raw_probability_after_reload": raw_after,
        "calibrated_probability_before_save": cal_before,
        "calibrated_probability_after_reload": cal_after,
    }, indent=2), encoding="utf-8")

    FREEZE_LOCK_PATH.write_text(json.dumps({
        "schema_version": "first_half_37_v3_draw_with_goals_freeze_lock_v1",
        "frozen_at_utc": now_utc_iso(),
        "status": "FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selected_benchmark_config": EXPECTED_CONFIG,
        "selected_calibration_strategy": EXPECTED_CALIBRATION,
        "features": FEATURES,
        "model_sha256": model_sha,
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "training_manifest_sha256": sha256_file(TRAINING_MANIFEST_PATH),
        "oof_calibration_sha256": sha256_file(OOF_PATH),
        "feature_importance_sha256": sha256_file(IMPORTANCE_PATH),
        "self_test_sha256": sha256_file(SELF_TEST_PATH),
        "training_dataset_sha256": upstream_hashes["dataset_sha256"],
        "training_rows": int(len(df)),
        "training_positives": int(y.sum()),
        "training_positive_rate": float(y.mean()),
        "calibration_alpha": float(alpha),
        "future_holdout_consumed": False,
        "rules": [
            "Do not modify frozen model, schema, features, params or alpha.",
            "Do not tune on the next pristine holdout.",
            "Next evaluation must use only future draw_with_goals matches not present in training.",
            "Holdout becomes consumed immediately after first target-open evaluation.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS FREEZE TERMINE")
    print("=" * 120)
    print(f"Nom modèle          : {MODEL_NAME}")
    print(f"Train final         : {len(df)}")
    print(f"Positifs            : {int(y.sum())} ({y.mean():.2%})")
    print(f"Features            : {len(FEATURES)}")
    print("                      " + ", ".join(FEATURES))
    print("Calibration         : fixed temporal OOF intercept-only")
    print(f"Alpha               : {alpha:+.12f}")
    print(f"OOF calibration N   : {len(oof)}")
    print("Self-test           : PASS")
    print(f"Model SHA256        : {model_sha}")
    print()
    print(f"Model               : {MODEL_PATH}")
    print(f"Freeze lock         : {FREEZE_LOCK_PATH}")
    print()
    print("STATUT: FROZEN_AWAITING_PRISTINE_FUTURE_HOLDOUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
