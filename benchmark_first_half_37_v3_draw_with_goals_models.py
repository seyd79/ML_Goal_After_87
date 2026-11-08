#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS MULTIVARIATE BENCHMARK
Research only: 190 anciens matchs + 50 LATE_STRESS consommés.
Aucun nouveau holdout futur. Aucun modèle n'est gelé ici.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

FS_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_feature_selection"
DATASET_PATH = FS_DIR / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
FEATURE_SCHEMA_PATH = FS_DIR / "first_half_37_v3_draw_with_goals_selected_features.json"
FEATURE_RANKING_PATH = FS_DIR / "first_half_37_v3_draw_with_goals_feature_ranking.csv"
FS_MANIFEST_PATH = FS_DIR / "first_half_37_v3_draw_with_goals_feature_selection_manifest.json"

DEFAULT_OUTPUT = BASE_DIR / "outputs" / "first_half_37" / "v3_draw_with_goals_model_benchmark"

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
PARTITION_COL = "selection_partition"

SELECTION_PARTITION = "FEATURE_SELECTION_DEV_190"
LATE_PARTITION = "CONSUMED_LATE_STRESS_50"

EXPECTED_TOTAL_ROWS = 240
EXPECTED_SELECTION_ROWS = 190
EXPECTED_LATE_ROWS = 50

LEGACY_TOP8 = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]

EXPECTED_NEW_TOP8 = [
    "total_xg_37",
    "xg_30_37",
    "away_xg_37",
    "home_shots_37",
    "home_corners_30_37",
    "home_over25_freq",
    "home_team_points_standing",
    "last_shot_on_target_minute",
]

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
OUTER_SPLITS = 4
INNER_SPLITS = 3
RANDOM_STATE = 87
EPS = 1e-10

# Gate research, défini avant de voir les résultats de ce benchmark.
GATE_MIN_MEAN_ROC = 0.58
GATE_MIN_MEDIAN_ROC = 0.58
GATE_MIN_ROC_POS_FRAC = 0.75
GATE_MIN_MEAN_PR_LIFT = 0.08
GATE_MIN_PR_POS_FRAC = 0.75
GATE_MIN_LATE_ROC = 0.55
GATE_MIN_LATE_PR_LIFT = 0.05


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


def load_inputs():
    for p in (DATASET_PATH, FEATURE_SCHEMA_PATH, FEATURE_RANKING_PATH, FS_MANIFEST_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    schema = load_json(FEATURE_SCHEMA_PATH)
    manifest = load_json(FS_MANIFEST_PATH)
    ranking = pd.read_csv(FEATURE_RANKING_PATH, low_memory=False)
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    if manifest.get("new_future_holdout_used") is not False:
        raise RuntimeError("Manifest feature selection incohérent: nouveau holdout indiqué.")

    if manifest.get("selection_uses_late_stress_targets") is not False:
        raise RuntimeError("La sélection précédente ne devait pas utiliser les targets LATE_STRESS.")

    selected = list(schema.get("selected_features", []))
    if selected != EXPECTED_NEW_TOP8:
        raise RuntimeError(
            "TOP8 spécialisé inattendu.\n"
            f"Attendu: {EXPECTED_NEW_TOP8}\n"
            f"Trouvé : {selected}"
        )

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Dataset draw={len(df)}, attendu={EXPECTED_TOTAL_ROWS}.")

    required = {ID_COL, DATE_COL, TARGET, PARTITION_COL, *LEGACY_TOP8, *EXPECTED_NEW_TOP8}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Colonnes requises absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
    if df[DATE_COL].isna().any():
        raise RuntimeError("kickoff_utc non parsable.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise RuntimeError("Target non résolu.")
    df[TARGET] = df[TARGET].astype(int)

    if df[ID_COL].astype(str).duplicated().any():
        raise RuntimeError("match_id dupliqué.")

    n_sel = int(df[PARTITION_COL].eq(SELECTION_PARTITION).sum())
    n_late = int(df[PARTITION_COL].eq(LATE_PARTITION).sum())

    if n_sel != EXPECTED_SELECTION_ROWS:
        raise RuntimeError(f"Selection rows={n_sel}, attendu={EXPECTED_SELECTION_ROWS}.")
    if n_late != EXPECTED_LATE_ROWS:
        raise RuntimeError(f"Late rows={n_late}, attendu={EXPECTED_LATE_ROWS}.")

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True), selected, ranking, manifest


def feature_sets(selected: Sequence[str]) -> Dict[str, List[str]]:
    s = list(selected)
    return {
        "LEGACY_TOP8": list(LEGACY_TOP8),
        "NEW_TOP3": s[:3],
        "NEW_TOP5": s[:5],
        "NEW_TOP8": s[:8],
    }


def logistic_pipeline(C: float, balanced: bool) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=float(C),
            solver="liblinear",
            class_weight="balanced" if balanced else None,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def extra_trees_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def hgb_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_depth=2,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


def tune_logistic_c(train_df: pd.DataFrame, features: Sequence[str], balanced: bool) -> float:
    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    if len(train_df) < 60:
        return 0.03

    X = train_df[list(features)].apply(pd.to_numeric, errors="coerce")
    y = train_df[TARGET].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return 0.03

    splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)
    rows = []

    for C in C_GRID:
        lifts, rocs = [], []

        for tr, va in splitter.split(X):
            ytr = y[tr]
            yva = y[va]

            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue

            model = logistic_pipeline(C, balanced)
            model.fit(X.iloc[tr], ytr)
            p = model.predict_proba(X.iloc[va])[:, 1]

            pr = safe_pr(yva, p)
            roc = safe_roc(yva, p)

            if np.isfinite(pr):
                lifts.append(pr - float(yva.mean()))
            if np.isfinite(roc):
                rocs.append(roc)

        if lifts:
            rows.append({
                "C": C,
                "mean_pr_lift": float(np.mean(lifts)),
                "mean_roc": float(np.mean(rocs)) if rocs else -np.inf,
                "positive_pr_fraction": float(np.mean(np.asarray(lifts) > 0)),
            })

    if not rows:
        return 0.03

    r = pd.DataFrame(rows).sort_values(
        ["positive_pr_fraction", "mean_pr_lift", "mean_roc", "C"],
        ascending=[False, False, False, True],
    )
    return float(r.iloc[0]["C"])


def build_model(model_name: str, train_df: pd.DataFrame, features: Sequence[str]):
    if model_name == "logistic_l2":
        C = tune_logistic_c(train_df, features, False)
        return logistic_pipeline(C, False), {"selected_C": C}

    if model_name == "logistic_l2_balanced":
        C = tune_logistic_c(train_df, features, True)
        return logistic_pipeline(C, True), {"selected_C": C}

    if model_name == "extra_trees_balanced":
        return extra_trees_pipeline(), {"selected_C": np.nan}

    if model_name == "hist_gradient_boosting":
        return hgb_pipeline(), {"selected_C": np.nan}

    raise ValueError(model_name)


def metric_row(y, p, train_rate: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    prevalence = float(y.mean())
    pr = safe_pr(y, p)
    roc = safe_roc(y, p)

    baseline = np.full(len(y), float(train_rate), dtype=float)

    b = float(brier_score_loss(y, p))
    b0 = float(brier_score_loss(y, baseline))

    ll = float(log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0, 1]))
    ll0 = float(log_loss(y, np.clip(baseline, EPS, 1-EPS), labels=[0, 1]))

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prevalence,
        "train_positive_rate": float(train_rate),
        "mean_probability": float(p.mean()),
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift": pr - prevalence if np.isfinite(pr) else np.nan,
        "brier": b,
        "baseline_brier_train_rate": b0,
        "brier_improvement_vs_train_rate": b0 - b,
        "log_loss": ll,
        "baseline_logloss_train_rate": ll0,
        "logloss_improvement_vs_train_rate": ll0 - ll,
    }


def evaluate_configuration(config_name: str, model_name: str, features: Sequence[str],
                           selection: pd.DataFrame, late: pd.DataFrame):
    fold_rows = []
    pred_frames = []

    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    selection = selection.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    for fold_idx, (tr, va) in enumerate(splitter.split(selection), start=1):
        train = selection.iloc[tr].copy()
        val = selection.iloc[va].copy()

        ytr = train[TARGET].astype(int).to_numpy()
        yva = val[TARGET].astype(int).to_numpy()

        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue

        model, meta = build_model(model_name, train, features)

        Xtr = train[list(features)].apply(pd.to_numeric, errors="coerce")
        Xva = val[list(features)].apply(pd.to_numeric, errors="coerce")

        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]

        fold_rows.append({
            "config": config_name,
            "feature_set": config_name.split("__", 1)[0],
            "model": model_name,
            "n_features": len(features),
            "fold": f"INTERNAL_{fold_idx}",
            "kind": "internal",
            "selected_C": meta.get("selected_C"),
            **metric_row(yva, p, float(ytr.mean())),
        })

        pred = val[[ID_COL, DATE_COL, TARGET]].copy()
        pred["config"] = config_name
        pred["fold"] = f"INTERNAL_{fold_idx}"
        pred["kind"] = "internal"
        pred["probability"] = p
        pred_frames.append(pred)

    # LATE_STRESS
    ytr = selection[TARGET].astype(int).to_numpy()
    yva = late[TARGET].astype(int).to_numpy()

    model, meta = build_model(model_name, selection, features)

    Xtr = selection[list(features)].apply(pd.to_numeric, errors="coerce")
    Xva = late[list(features)].apply(pd.to_numeric, errors="coerce")

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]

    fold_rows.append({
        "config": config_name,
        "feature_set": config_name.split("__", 1)[0],
        "model": model_name,
        "n_features": len(features),
        "fold": "LATE_STRESS_50",
        "kind": "late_stress",
        "selected_C": meta.get("selected_C"),
        **metric_row(yva, p, float(ytr.mean())),
    })

    pred = late[[ID_COL, DATE_COL, TARGET]].copy()
    pred["config"] = config_name
    pred["fold"] = "LATE_STRESS_50"
    pred["kind"] = "late_stress"
    pred["probability"] = p
    pred_frames.append(pred)

    return pd.DataFrame(fold_rows), pd.concat(pred_frames, ignore_index=True)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for config, block in fold_metrics.groupby("config", sort=False):
        internal = block[block["kind"] == "internal"].copy()
        late = block[block["kind"] == "late_stress"].copy()

        if len(late) != 1:
            raise RuntimeError(f"{config}: LATE_STRESS absent ou dupliqué.")

        s = late.iloc[0]

        mean_roc = float(internal["roc_auc"].mean())
        median_roc = float(internal["roc_auc"].median())
        roc_pos_frac = float((internal["roc_auc"] > 0.5).mean())

        mean_pr_lift = float(internal["pr_lift"].mean())
        median_pr_lift = float(internal["pr_lift"].median())
        pr_pos_frac = float((internal["pr_lift"] > 0).mean())

        mean_brier_imp = float(internal["brier_improvement_vs_train_rate"].mean())
        mean_ll_imp = float(internal["logloss_improvement_vs_train_rate"].mean())

        passes = int(
            len(internal) >= OUTER_SPLITS
            and mean_roc >= GATE_MIN_MEAN_ROC
            and median_roc >= GATE_MIN_MEDIAN_ROC
            and roc_pos_frac >= GATE_MIN_ROC_POS_FRAC
            and mean_pr_lift >= GATE_MIN_MEAN_PR_LIFT
            and pr_pos_frac >= GATE_MIN_PR_POS_FRAC
            and float(s["roc_auc"]) >= GATE_MIN_LATE_ROC
            and float(s["pr_lift"]) >= GATE_MIN_LATE_PR_LIFT
        )

        robustness_score = (
            mean_roc
            + 0.60 * mean_pr_lift
            + 0.35 * float(s["roc_auc"])
            + 0.35 * float(s["pr_lift"])
            + 0.05 * roc_pos_frac
            + 0.05 * pr_pos_frac
        )

        rows.append({
            "config": config,
            "feature_set": block.iloc[0]["feature_set"],
            "model": block.iloc[0]["model"],
            "n_features": int(block.iloc[0]["n_features"]),
            "passes_research_gate": passes,
            "mean_internal_roc_auc": mean_roc,
            "median_internal_roc_auc": median_roc,
            "internal_roc_above_050_fraction": roc_pos_frac,
            "mean_internal_pr_auc": float(internal["pr_auc"].mean()),
            "mean_internal_pr_lift": mean_pr_lift,
            "median_internal_pr_lift": median_pr_lift,
            "internal_pr_lift_positive_fraction": pr_pos_frac,
            "mean_internal_brier_improvement": mean_brier_imp,
            "mean_internal_logloss_improvement": mean_ll_imp,
            "late_stress_roc_auc": float(s["roc_auc"]),
            "late_stress_pr_auc": float(s["pr_auc"]),
            "late_stress_positive_rate": float(s["positive_rate"]),
            "late_stress_pr_lift": float(s["pr_lift"]),
            "late_stress_brier_improvement": float(s["brier_improvement_vs_train_rate"]),
            "late_stress_logloss_improvement": float(s["logloss_improvement_vs_train_rate"]),
            "late_stress_mean_probability": float(s["mean_probability"]),
            "selected_C_values": "|".join(str(x) for x in block["selected_C"].dropna().tolist()),
            "robustness_score": robustness_score,
        })

    out = pd.DataFrame(rows).sort_values(
        ["passes_research_gate", "robustness_score", "mean_internal_roc_auc", "late_stress_roc_auc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(path: Path, fs: Mapping[str, Sequence[str]], ranking: pd.DataFrame,
                 fold_metrics: pd.DataFrame, selection: pd.DataFrame, late: pd.DataFrame):
    lines = [
        "# First Half 37 — V3 DRAW_WITH_GOALS Multivariate Benchmark",
        "",
        "## Discipline",
        "",
        "- Aucun nouveau holdout futur utilisé.",
        "- 190 anciens matchs = validation temporelle interne principale.",
        "- 50 matchs tardifs consommés = stress test de DEVELOPMENT.",
        "- Aucun candidat n'est encore validé production.",
        "",
        "## Dataset",
        "",
        f"- INTERNAL DEVELOPMENT : **{len(selection)}** | positifs={int(selection[TARGET].sum())} ({selection[TARGET].mean():.2%})",
        f"- LATE_STRESS : **{len(late)}** | positifs={int(late[TARGET].sum())} ({late[TARGET].mean():.2%})",
        "",
        "## Feature sets",
        "",
    ]

    for name, features in fs.items():
        lines.append(f"### {name} — {len(features)} features")
        lines.extend([f"- `{f}`" for f in features])
        lines.append("")

    lines += [
        "## Research gate",
        "",
        f"- mean internal ROC >= {GATE_MIN_MEAN_ROC:.2f}",
        f"- median internal ROC >= {GATE_MIN_MEDIAN_ROC:.2f}",
        f"- ROC > .50 dans >= {GATE_MIN_ROC_POS_FRAC:.0%} folds",
        f"- mean internal PR lift >= {GATE_MIN_MEAN_PR_LIFT:+.2f}",
        f"- PR lift > 0 dans >= {GATE_MIN_PR_POS_FRAC:.0%} folds",
        f"- LATE_STRESS ROC >= {GATE_MIN_LATE_ROC:.2f}",
        f"- LATE_STRESS PR lift >= {GATE_MIN_LATE_PR_LIFT:+.2f}",
        "",
        "Brier/LogLoss sont diagnostics seulement à cette étape.",
        "",
        "## Ranking",
        "",
        "```",
    ]

    show = [
        "rank", "config", "passes_research_gate",
        "mean_internal_roc_auc", "median_internal_roc_auc",
        "internal_roc_above_050_fraction", "mean_internal_pr_lift",
        "internal_pr_lift_positive_fraction", "late_stress_roc_auc",
        "late_stress_pr_lift", "mean_internal_brier_improvement",
        "late_stress_brier_improvement", "robustness_score",
    ]
    lines.append(ranking[show].to_string(index=False))
    lines += ["```", "", "## Fold metrics", "", "```"]

    fold_show = [
        "config", "fold", "kind", "n", "positive_rate", "roc_auc",
        "pr_auc", "pr_lift", "brier_improvement_vs_train_rate",
        "logloss_improvement_vs_train_rate", "selected_C",
    ]
    lines.append(
        fold_metrics[fold_show]
        .sort_values(["config", "kind", "fold"])
        .to_string(index=False)
    )
    lines += [
        "```",
        "",
        "## Suite",
        "",
        "Les candidats PASS sont seulement des finalistes de DEVELOPMENT. "
        "La prochaine étape doit auditer leur calibration, puis choisir au plus "
        "un pipeline à geler avant un nouveau holdout futur.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark multivarié spécialisé draw_with_goals."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS MULTIVARIATE BENCHMARK")
    print("=" * 120)
    print("AUCUN NOUVEAU HOLDOUT FUTUR")
    print("190 INTERNAL + 50 LATE_STRESS CONSOMMES")
    print()

    df, selected_features, feature_ranking, fs_manifest = load_inputs()

    selection = (
        df[df[PARTITION_COL] == SELECTION_PARTITION]
        .copy()
        .sort_values([DATE_COL, ID_COL])
        .reset_index(drop=True)
    )
    late = (
        df[df[PARTITION_COL] == LATE_PARTITION]
        .copy()
        .sort_values([DATE_COL, ID_COL])
        .reset_index(drop=True)
    )

    fs = feature_sets(selected_features)

    print(
        f"INTERNAL DEVELOPMENT : {len(selection)} | "
        f"positifs={int(selection[TARGET].sum())} ({selection[TARGET].mean():.2%})"
    )
    print(
        f"LATE_STRESS          : {len(late)} | "
        f"positifs={int(late[TARGET].sum())} ({late[TARGET].mean():.2%})"
    )
    print()

    for name, feats in fs.items():
        print(f"{name:<14} : {len(feats)} features | " + ", ".join(feats))
    print()

    model_names = [
        "logistic_l2",
        "logistic_l2_balanced",
        "extra_trees_balanced",
        "hist_gradient_boosting",
    ]

    fold_frames = []
    pred_frames = []
    total = len(fs) * len(model_names)
    done = 0

    for fs_name, features in fs.items():
        for model_name in model_names:
            done += 1
            config_name = f"{fs_name}__{model_name}"

            fm, pm = evaluate_configuration(
                config_name, model_name, features, selection, late
            )
            fold_frames.append(fm)
            pred_frames.append(pm)

            internal = fm[fm["kind"] == "internal"]
            stress = fm[fm["kind"] == "late_stress"].iloc[0]

            print(
                f"[{done:02d}/{total:02d}] {config_name:<48} | "
                f"INT ROC={internal['roc_auc'].mean():.4f} | "
                f"INT PRlift={internal['pr_lift'].mean():+.4f} | "
                f"LATE ROC={stress['roc_auc']:.4f} | "
                f"LATE PRlift={stress['pr_lift']:+.4f}"
            )

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    ranking = summarize(fold_metrics)

    fold_path = out_dir / "first_half_37_v3_draw_with_goals_fold_metrics.csv"
    pred_path = out_dir / "first_half_37_v3_draw_with_goals_predictions.csv"
    ranking_path = out_dir / "first_half_37_v3_draw_with_goals_model_ranking.csv"
    report_path = out_dir / "first_half_37_v3_draw_with_goals_model_benchmark_report.md"
    manifest_path = out_dir / "first_half_37_v3_draw_with_goals_model_benchmark_manifest.json"

    fold_metrics.to_csv(fold_path, index=False)
    predictions.to_csv(pred_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    write_report(report_path, fs, ranking, fold_metrics, selection, late)

    manifest = {
        "schema_version": "first_half_37_v3_draw_with_goals_model_benchmark_v1",
        "created_at_utc": now_utc_iso(),
        "mode": "research_only_consumed_data",
        "new_future_holdout_used": False,
        "selection_rows": len(selection),
        "late_stress_rows": len(late),
        "source_dataset": str(DATASET_PATH),
        "source_dataset_sha256": sha256_file(DATASET_PATH),
        "feature_schema": str(FEATURE_SCHEMA_PATH),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
        "feature_sets": fs,
        "models": model_names,
        "gate": {
            "min_mean_internal_roc": GATE_MIN_MEAN_ROC,
            "min_median_internal_roc": GATE_MIN_MEDIAN_ROC,
            "min_internal_roc_positive_fraction": GATE_MIN_ROC_POS_FRAC,
            "min_mean_internal_pr_lift": GATE_MIN_MEAN_PR_LIFT,
            "min_internal_pr_positive_fraction": GATE_MIN_PR_POS_FRAC,
            "min_late_roc": GATE_MIN_LATE_ROC,
            "min_late_pr_lift": GATE_MIN_LATE_PR_LIFT,
        },
        "brier_logloss_role": "diagnostic_only_before_calibration_audit",
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pass_count = int(ranking["passes_research_gate"].sum())
    winner = ranking.iloc[0]

    print()
    print("=" * 120)
    print("DRAW_WITH_GOALS MULTIVARIATE BENCHMARK TERMINE")
    print("=" * 120)
    print(f"Configurations testées  : {len(ranking)}")
    print(f"Configurations PASS      : {pass_count}")
    print(
        f"#1                       : {winner['config']} | "
        f"INT ROC={winner['mean_internal_roc_auc']:.4f} | "
        f"INT PRlift={winner['mean_internal_pr_lift']:+.4f} | "
        f"LATE ROC={winner['late_stress_roc_auc']:.4f} | "
        f"LATE PRlift={winner['late_stress_pr_lift']:+.4f} | "
        f"gate={'PASS' if int(winner['passes_research_gate']) else 'FAIL'}"
    )
    print()
    print("TOP 5")
    cols = [
        "rank", "config", "passes_research_gate",
        "mean_internal_roc_auc", "mean_internal_pr_lift",
        "late_stress_roc_auc", "late_stress_pr_lift",
        "mean_internal_brier_improvement", "late_stress_brier_improvement",
    ]
    print(ranking[cols].head(5).to_string(index=False))
    print()
    print(f"Rapport                 : {report_path}")
    print(f"Ranking                 : {ranking_path}")
    print(f"Fold metrics            : {fold_path}")
    print()
    print(
        "IMPORTANT: aucun candidat n'est encore gelé. "
        "Les finalistes PASS doivent passer un audit de calibration."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
