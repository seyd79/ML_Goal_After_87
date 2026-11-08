#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS CALIBRATION AUDIT
====================================================

But:
Auditer la calibration des finalistes PASS du benchmark multivarié, sans
ouvrir aucun nouveau holdout et sans modifier le ranking-model lui-même.

Entrées:
  outputs/first_half_37/v3_draw_with_goals_model_benchmark/
    first_half_37_v3_draw_with_goals_predictions.csv
    first_half_37_v3_draw_with_goals_fold_metrics.csv
    first_half_37_v3_draw_with_goals_model_ranking.csv
    first_half_37_v3_draw_with_goals_model_benchmark_manifest.json

Finalistes:
  uniquement passes_research_gate == 1.

Stratégies de calibration:
  RAW
  ROLLING_INTERCEPT
  ROLLING_PLATT
  ROLLING_ISOTONIC

Principe anti-leakage:
  - Pour INTERNAL_1: pas d'historique OOF précédent => RAW fallback.
  - Pour INTERNAL_k: calibrateur appris seulement sur les folds OOF antérieurs.
  - Pour LATE_STRESS_50: calibrateur appris sur toutes les prédictions OOF
    INTERNAL_1..4 du même candidat.
  - Aucun target du fold courant ne sert à calibrer ce fold.

Gate calibration:
  - discrimination non dégradée: mean internal ROC delta >= -0.005
    et late ROC delta >= -0.005
  - mean internal Brier improvement vs baseline > 0
  - mean internal LogLoss improvement vs baseline > 0
  - late Brier improvement vs baseline > 0
  - late LogLoss improvement vs baseline > 0
  - mean internal ECE <= RAW
  - late ECE <= RAW

IMPORTANT:
Ceci reste du DEVELOPMENT sur données consommées.
Un futur holdout entièrement neuf sera obligatoire après freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

BENCH_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_model_benchmark"
)

PRED_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_predictions.csv"
FOLD_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_fold_metrics.csv"
RANKING_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_model_ranking.csv"
MANIFEST_PATH = BENCH_DIR / "first_half_37_v3_draw_with_goals_model_benchmark_manifest.json"

DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_calibration_audit"
)

TARGET = "goal_after_37_before_ht"
P_COL = "probability"
EPS = 1e-10

INTERNAL_FOLDS = [
    "INTERNAL_1",
    "INTERNAL_2",
    "INTERNAL_3",
    "INTERNAL_4",
]
LATE_FOLD = "LATE_STRESS_50"

STRATEGIES = [
    "RAW",
    "ROLLING_INTERCEPT",
    "ROLLING_PLATT",
    "ROLLING_ISOTONIC",
]

MIN_CALIBRATION_N = 25
MIN_ISOTONIC_N = 40

MAX_ROC_DEGRADATION = 0.005


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
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON invalide: {path}")
    return obj


def safe_roc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr(y, p) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def ece_binary(y, p, n_bins=10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not mask.any():
            continue

        out += mask.mean() * abs(
            float(y[mask].mean()) - float(p[mask].mean())
        )

    return float(out)


def fit_intercept_alpha(y, p) -> float:
    """
    Résout sum(y - sigmoid(logit(p)+alpha)) = 0 par dichotomie.
    C'est une calibration intercept-only, pente fixée à 1.
    """
    y = np.asarray(y, dtype=float)
    lp = logit(p)

    lo, hi = -12.0, 12.0

    for _ in range(100):
        mid = (lo + hi) / 2.0
        q = sigmoid(lp + mid)
        f = float(np.sum(y - q))

        if f > 0:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)


def apply_strategy(
    strategy: str,
    history_y: np.ndarray,
    history_p: np.ndarray,
    current_p: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    current_p = np.asarray(current_p, dtype=float)

    if strategy == "RAW":
        return current_p.copy(), {"status": "raw"}

    if (
        len(history_y) < MIN_CALIBRATION_N
        or len(np.unique(history_y)) < 2
    ):
        return current_p.copy(), {
            "status": "raw_fallback_insufficient_history",
            "history_n": int(len(history_y)),
        }

    if strategy == "ROLLING_INTERCEPT":
        alpha = fit_intercept_alpha(history_y, history_p)
        q = sigmoid(logit(current_p) + alpha)
        return q, {
            "status": "calibrated",
            "alpha": float(alpha),
            "history_n": int(len(history_y)),
        }

    if strategy == "ROLLING_PLATT":
        X = logit(history_p).reshape(-1, 1)

        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
            random_state=87,
        )
        model.fit(X, history_y)

        slope = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])

        # Une pente négative inverse le ranking; rejet => fallback RAW.
        if slope <= 0:
            return current_p.copy(), {
                "status": "raw_fallback_negative_slope",
                "history_n": int(len(history_y)),
                "slope": slope,
                "intercept": intercept,
            }

        q = model.predict_proba(
            logit(current_p).reshape(-1, 1)
        )[:, 1]

        return q, {
            "status": "calibrated",
            "history_n": int(len(history_y)),
            "slope": slope,
            "intercept": intercept,
        }

    if strategy == "ROLLING_ISOTONIC":
        if len(history_y) < MIN_ISOTONIC_N:
            return current_p.copy(), {
                "status": "raw_fallback_insufficient_isotonic_history",
                "history_n": int(len(history_y)),
            }

        model = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            increasing=True,
            out_of_bounds="clip",
        )
        model.fit(history_p, history_y)
        q = model.predict(current_p)

        return np.asarray(q, dtype=float), {
            "status": "calibrated",
            "history_n": int(len(history_y)),
        }

    raise ValueError(strategy)


def fold_metrics(
    y,
    p,
    baseline_rate: float,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    baseline = np.full(
        len(y),
        float(baseline_rate),
        dtype=float,
    )

    b = float(brier_score_loss(y, p))
    b0 = float(brier_score_loss(y, baseline))

    ll = float(
        log_loss(
            y,
            np.clip(p, EPS, 1-EPS),
            labels=[0, 1],
        )
    )
    ll0 = float(
        log_loss(
            y,
            np.clip(baseline, EPS, 1-EPS),
            labels=[0, 1],
        )
    )

    pr = safe_pr(y, p)
    prevalence = float(y.mean())

    return {
        "n": int(len(y)),
        "positive_rate": prevalence,
        "mean_probability": float(p.mean()),
        "gap_meanp_minus_actual": float(p.mean() - prevalence),
        "roc_auc": safe_roc(y, p),
        "pr_auc": pr,
        "pr_lift": pr - prevalence,
        "brier": b,
        "baseline_brier": b0,
        "brier_improvement": b0 - b,
        "log_loss": ll,
        "baseline_logloss": ll0,
        "logloss_improvement": ll0 - ll,
        "ece_10": ece_binary(y, p, 10),
    }


def load_inputs():
    for path in (PRED_PATH, FOLD_PATH, RANKING_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_json(MANIFEST_PATH)

    if manifest.get("new_future_holdout_used") is not False:
        raise RuntimeError("Benchmark manifest incohérent.")

    pred = pd.read_csv(PRED_PATH, low_memory=False)
    folds = pd.read_csv(FOLD_PATH, low_memory=False)
    ranking = pd.read_csv(RANKING_PATH, low_memory=False)

    required_pred = {
        "match_id", TARGET, "config", "fold", "kind", P_COL
    }
    missing = sorted(required_pred - set(pred.columns))
    if missing:
        raise RuntimeError(f"Predictions: colonnes absentes: {missing}")

    pred[TARGET] = pd.to_numeric(pred[TARGET], errors="coerce")
    pred[P_COL] = pd.to_numeric(pred[P_COL], errors="coerce")

    if pred[TARGET].isna().any() or pred[P_COL].isna().any():
        raise RuntimeError("Target/probability manquant.")

    pred[TARGET] = pred[TARGET].astype(int)

    finalists = (
        ranking[ranking["passes_research_gate"] == 1]["config"]
        .astype(str)
        .tolist()
    )

    if not finalists:
        raise RuntimeError("Aucun finaliste PASS à calibrer.")

    return pred, folds, ranking, finalists, manifest


def evaluate_config_strategy(
    pred: pd.DataFrame,
    fold_meta: pd.DataFrame,
    config: str,
    strategy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = pred[pred["config"] == config].copy()
    meta = fold_meta[fold_meta["config"] == config].copy()

    rows = []
    out_preds = []

    history_parts = []

    for fold in INTERNAL_FOLDS:
        cur = cfg[cfg["fold"] == fold].copy()

        if cur.empty:
            continue

        if history_parts:
            history = pd.concat(history_parts, ignore_index=True)
            hy = history[TARGET].to_numpy(dtype=int)
            hp = history[P_COL].to_numpy(dtype=float)
        else:
            hy = np.array([], dtype=int)
            hp = np.array([], dtype=float)

        raw = cur[P_COL].to_numpy(dtype=float)
        y = cur[TARGET].to_numpy(dtype=int)

        calibrated, calib_meta = apply_strategy(
            strategy,
            hy,
            hp,
            raw,
        )

        fm = meta[meta["fold"] == fold]
        if len(fm) != 1:
            raise RuntimeError(
                f"{config}/{fold}: fold metadata absent ou dupliqué."
            )
        baseline_rate = float(
            fm.iloc[0]["train_positive_rate"]
        )

        m = fold_metrics(
            y=y,
            p=calibrated,
            baseline_rate=baseline_rate,
        )

        rows.append({
            "config": config,
            "strategy": strategy,
            "fold": fold,
            "kind": "internal",
            "calibration_status": calib_meta.get("status"),
            "calibration_history_n": int(calib_meta.get("history_n", 0)),
            "calibration_alpha": calib_meta.get("alpha", np.nan),
            "calibration_slope": calib_meta.get("slope", np.nan),
            "calibration_intercept": calib_meta.get("intercept", np.nan),
            **m,
        })

        op = cur[["match_id", TARGET, "config", "fold", "kind"]].copy()
        op["strategy"] = strategy
        op["raw_probability"] = raw
        op["calibrated_probability"] = calibrated
        out_preds.append(op)

        # L'historique de calibration utilise les prédictions OOF RAW du fold passé.
        history_parts.append(cur[[TARGET, P_COL]].copy())

    # LATE: calibrateur appris sur tous les OOF internes.
    late = cfg[cfg["fold"] == LATE_FOLD].copy()

    if len(late):
        history = pd.concat(history_parts, ignore_index=True)

        hy = history[TARGET].to_numpy(dtype=int)
        hp = history[P_COL].to_numpy(dtype=float)

        raw = late[P_COL].to_numpy(dtype=float)
        y = late[TARGET].to_numpy(dtype=int)

        calibrated, calib_meta = apply_strategy(
            strategy,
            hy,
            hp,
            raw,
        )

        fm = meta[meta["fold"] == LATE_FOLD]
        if len(fm) != 1:
            raise RuntimeError(
                f"{config}/{LATE_FOLD}: fold metadata absent ou dupliqué."
            )

        baseline_rate = float(
            fm.iloc[0]["train_positive_rate"]
        )

        m = fold_metrics(
            y=y,
            p=calibrated,
            baseline_rate=baseline_rate,
        )

        rows.append({
            "config": config,
            "strategy": strategy,
            "fold": LATE_FOLD,
            "kind": "late_stress",
            "calibration_status": calib_meta.get("status"),
            "calibration_history_n": int(calib_meta.get("history_n", 0)),
            "calibration_alpha": calib_meta.get("alpha", np.nan),
            "calibration_slope": calib_meta.get("slope", np.nan),
            "calibration_intercept": calib_meta.get("intercept", np.nan),
            **m,
        })

        op = late[["match_id", TARGET, "config", "fold", "kind"]].copy()
        op["strategy"] = strategy
        op["raw_probability"] = raw
        op["calibrated_probability"] = calibrated
        out_preds.append(op)

    return (
        pd.DataFrame(rows),
        pd.concat(out_preds, ignore_index=True),
    )


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (config, strategy), block in audit.groupby(
        ["config", "strategy"],
        sort=False,
    ):
        internal = block[block["kind"] == "internal"].copy()
        late = block[block["kind"] == "late_stress"].copy()

        if len(late) != 1:
            raise RuntimeError(
                f"{config}/{strategy}: late stress absent."
            )

        s = late.iloc[0]

        raw_block = audit[
            (audit["config"] == config)
            & (audit["strategy"] == "RAW")
        ]
        raw_internal = raw_block[
            raw_block["kind"] == "internal"
        ]
        raw_late = raw_block[
            raw_block["kind"] == "late_stress"
        ].iloc[0]

        mean_roc = float(internal["roc_auc"].mean())
        raw_mean_roc = float(raw_internal["roc_auc"].mean())

        mean_ece = float(internal["ece_10"].mean())
        raw_mean_ece = float(raw_internal["ece_10"].mean())

        late_roc = float(s["roc_auc"])
        raw_late_roc = float(raw_late["roc_auc"])

        passes = int(
            strategy != "RAW"
            and mean_roc - raw_mean_roc >= -MAX_ROC_DEGRADATION
            and late_roc - raw_late_roc >= -MAX_ROC_DEGRADATION
            and float(internal["brier_improvement"].mean()) > 0
            and float(internal["logloss_improvement"].mean()) > 0
            and float(s["brier_improvement"]) > 0
            and float(s["logloss_improvement"]) > 0
            and mean_ece <= raw_mean_ece
            and float(s["ece_10"]) <= float(raw_late["ece_10"])
        )

        calibration_score = (
            float(internal["brier_improvement"].mean())
            + float(internal["logloss_improvement"].mean())
            + float(s["brier_improvement"])
            + float(s["logloss_improvement"])
            - 0.2 * mean_ece
            - 0.2 * float(s["ece_10"])
        )

        rows.append({
            "config": config,
            "strategy": strategy,
            "passes_calibration_gate": passes,
            "mean_internal_roc_auc": mean_roc,
            "delta_internal_roc_vs_raw": mean_roc - raw_mean_roc,
            "mean_internal_pr_lift": float(internal["pr_lift"].mean()),
            "mean_internal_brier_improvement": float(
                internal["brier_improvement"].mean()
            ),
            "mean_internal_logloss_improvement": float(
                internal["logloss_improvement"].mean()
            ),
            "mean_internal_ece": mean_ece,
            "raw_mean_internal_ece": raw_mean_ece,
            "late_stress_roc_auc": late_roc,
            "delta_late_roc_vs_raw": late_roc - raw_late_roc,
            "late_stress_pr_lift": float(s["pr_lift"]),
            "late_stress_brier_improvement": float(
                s["brier_improvement"]
            ),
            "late_stress_logloss_improvement": float(
                s["logloss_improvement"]
            ),
            "late_stress_ece": float(s["ece_10"]),
            "raw_late_stress_ece": float(raw_late["ece_10"]),
            "late_mean_probability": float(s["mean_probability"]),
            "late_positive_rate": float(s["positive_rate"]),
            "calibration_score": calibration_score,
        })

    out = pd.DataFrame(rows)

    # RAW ne peut pas "PASS calibration", mais reste visible comme référence.
    out = out.sort_values(
        [
            "passes_calibration_gate",
            "calibration_score",
            "late_stress_brier_improvement",
            "mean_internal_brier_improvement",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(
    path: Path,
    finalists: List[str],
    summary: pd.DataFrame,
    audit: pd.DataFrame,
):
    lines = [
        "# First Half 37 — V3 DRAW_WITH_GOALS Calibration Audit",
        "",
        "## Discipline",
        "",
        "- Aucun nouveau holdout futur utilisé.",
        "- Aucun ranking-model n'est réentraîné ou modifié.",
        "- Calibration INTERNAL = rolling, seulement avec folds OOF antérieurs.",
        "- Calibration LATE_STRESS = calibrateur appris sur tous les OOF internes.",
        "- Les 50 LATE_STRESS restent des données consommées de DEVELOPMENT.",
        "",
        "## Finalistes",
        "",
    ]

    for f in finalists:
        lines.append(f"- `{f}`")

    lines += [
        "",
        "## Gate calibration",
        "",
        "- ROC internal et late: dégradation maximale tolérée = 0.005",
        "- Brier improvement > 0 en moyenne internal ET late",
        "- LogLoss improvement > 0 en moyenne internal ET late",
        "- ECE internal <= RAW",
        "- ECE late <= RAW",
        "",
        "## Ranking calibration",
        "",
        "```",
    ]

    show = [
        "rank",
        "config",
        "strategy",
        "passes_calibration_gate",
        "mean_internal_roc_auc",
        "mean_internal_brier_improvement",
        "mean_internal_logloss_improvement",
        "mean_internal_ece",
        "late_stress_roc_auc",
        "late_stress_pr_lift",
        "late_stress_brier_improvement",
        "late_stress_logloss_improvement",
        "late_stress_ece",
        "late_mean_probability",
        "late_positive_rate",
    ]

    lines.append(summary[show].to_string(index=False))
    lines += [
        "```",
        "",
        "## Fold detail",
        "",
        "```",
    ]

    fold_show = [
        "config",
        "strategy",
        "fold",
        "calibration_status",
        "calibration_history_n",
        "roc_auc",
        "pr_lift",
        "brier_improvement",
        "logloss_improvement",
        "ece_10",
        "mean_probability",
        "positive_rate",
    ]

    lines.append(
        audit
        .sort_values(["config", "strategy", "kind", "fold"])
        [fold_show]
        .to_string(index=False)
    )

    lines += [
        "```",
        "",
        "## Suite",
        "",
        "Si au moins une stratégie passe, le prochain choix doit privilégier "
        "un pipeline simple, stable en ranking et correctement calibré. "
        "Ensuite seulement: refit sur les 240 matchs consommés, freeze complet, "
        "puis nouveau holdout futur entièrement neuf.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit de calibration V3 draw_with_goals."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS CALIBRATION AUDIT")
    print("=" * 120)
    print("AUCUN NOUVEAU HOLDOUT FUTUR")
    print("ROLLING CALIBRATION SANS LEAKAGE DU FOLD COURANT")
    print()

    pred, fold_meta, ranking, finalists, manifest = load_inputs()

    print(f"Finalistes PASS benchmark : {len(finalists)}")
    for f in finalists:
        print(f"  - {f}")
    print()

    audit_frames = []
    pred_frames = []

    for config in finalists:
        print("-" * 120)
        print(config)

        for strategy in STRATEGIES:
            fm, pm = evaluate_config_strategy(
                pred=pred,
                fold_meta=fold_meta,
                config=config,
                strategy=strategy,
            )
            audit_frames.append(fm)
            pred_frames.append(pm)

            internal = fm[fm["kind"] == "internal"]
            late = fm[fm["kind"] == "late_stress"].iloc[0]

            print(
                f"  {strategy:<20} | "
                f"INT BrierImp={internal['brier_improvement'].mean():+.4f} | "
                f"INT LLImp={internal['logloss_improvement'].mean():+.4f} | "
                f"INT ECE={internal['ece_10'].mean():.4f} | "
                f"LATE BrierImp={late['brier_improvement']:+.4f} | "
                f"LATE LLImp={late['logloss_improvement']:+.4f} | "
                f"LATE ECE={late['ece_10']:.4f}"
            )

    audit = pd.concat(audit_frames, ignore_index=True)
    calibrated_predictions = pd.concat(pred_frames, ignore_index=True)
    summary = summarize(audit)

    audit_path = out_dir / "first_half_37_v3_draw_with_goals_calibration_fold_metrics.csv"
    pred_out = out_dir / "first_half_37_v3_draw_with_goals_calibrated_predictions.csv"
    summary_path = out_dir / "first_half_37_v3_draw_with_goals_calibration_summary.csv"
    report_path = out_dir / "first_half_37_v3_draw_with_goals_calibration_report.md"
    manifest_path = out_dir / "first_half_37_v3_draw_with_goals_calibration_manifest.json"

    audit.to_csv(audit_path, index=False)
    calibrated_predictions.to_csv(pred_out, index=False)
    summary.to_csv(summary_path, index=False)

    write_report(
        report_path,
        finalists=finalists,
        summary=summary,
        audit=audit,
    )

    manifest_out = {
        "schema_version": "first_half_37_v3_draw_with_goals_calibration_audit_v1",
        "created_at_utc": now_utc_iso(),
        "mode": "research_only_consumed_data",
        "new_future_holdout_used": False,
        "source_predictions_sha256": sha256_file(PRED_PATH),
        "source_fold_metrics_sha256": sha256_file(FOLD_PATH),
        "source_ranking_sha256": sha256_file(RANKING_PATH),
        "finalists": finalists,
        "strategies": STRATEGIES,
        "min_calibration_n": MIN_CALIBRATION_N,
        "min_isotonic_n": MIN_ISOTONIC_N,
        "max_roc_degradation": MAX_ROC_DEGRADATION,
    }

    manifest_path.write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed = summary[
        summary["passes_calibration_gate"] == 1
    ].copy()

    print()
    print("=" * 120)
    print("DRAW_WITH_GOALS CALIBRATION AUDIT TERMINE")
    print("=" * 120)
    print(f"Finalistes benchmark      : {len(finalists)}")
    print(f"Combinaisons calibration  : {len(summary)}")
    print(f"Calibration PASS          : {len(passed)}")

    if len(passed):
        best = passed.iloc[0]
        print(
            f"#1 calibration            : {best['config']} + {best['strategy']} | "
            f"INT BrierImp={best['mean_internal_brier_improvement']:+.4f} | "
            f"INT LLImp={best['mean_internal_logloss_improvement']:+.4f} | "
            f"LATE BrierImp={best['late_stress_brier_improvement']:+.4f} | "
            f"LATE LLImp={best['late_stress_logloss_improvement']:+.4f}"
        )
    else:
        print("#1 calibration            : AUCUNE STRATEGIE PASS")

    print()
    print("TOP 8")
    cols = [
        "rank",
        "config",
        "strategy",
        "passes_calibration_gate",
        "mean_internal_roc_auc",
        "mean_internal_brier_improvement",
        "mean_internal_logloss_improvement",
        "late_stress_roc_auc",
        "late_stress_pr_lift",
        "late_stress_brier_improvement",
        "late_stress_logloss_improvement",
        "late_stress_ece",
    ]
    print(summary[cols].head(8).to_string(index=False))
    print()
    print(f"Rapport                    : {report_path}")
    print(f"Summary                    : {summary_path}")
    print(f"Fold metrics               : {audit_path}")
    print()
    print(
        "IMPORTANT: même en cas de PASS, ceci reste du DEVELOPMENT consommé. "
        "Le vrai test final devra être un nouveau bloc futur."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
