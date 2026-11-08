#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 SUBSET + MODEL COMPARISON

Compare les sous-ensembles V2 sur les 2 105 matchs désormais connus.

Design:
  - DEVELOPMENT V2 = 1 802 historiques + 303 externes déjà consommés.
  - 5 folds temporels internes expansifs sur les 1 802 historiques.
  - 1 stress fold: TRAIN=1 802 historiques, VAL=303 externes consommés.
  - Aucun match postérieur au 19 septembre 2026 n'est lu.
  - Les sous-ensembles sont dérivés du ranking robuste V2 déjà produit.
  - Pour les régressions logistiques, C est choisi uniquement dans le TRAIN
    de chaque fold via mini-validation temporelle interne.
  - Aucun tuning sur un futur holdout.

Sorties:
  outputs/first_half_37/v2_model_comparison/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

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
DEFAULT_SCHEMA = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "first_half_37_v2_robust_schema.json"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_model_comparison"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
STAGE_COL = "_v2_stage"

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

SUBSET_SIZES = [5, 8, 10, 12, 15, 19]


@dataclass
class FoldDef:
    name: str
    kind: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def safe_roc(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_pr(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def baseline_brier(rate: float) -> float:
    return rate * (1.0 - rate)


def baseline_logloss(rate: float) -> float:
    eps = 1e-12
    r = min(1 - eps, max(eps, rate))
    return -(r * math.log(r) + (1-r) * math.log(1-r))


def load_dev(path: Path) -> pd.DataFrame:
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
        raise ValueError("kickoff_utc non parsable dans DEVELOPMENT V2.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target non résolu présent.")

    if df[ID_COL].duplicated().any():
        raise ValueError("match_id dupliqué dans DEVELOPMENT V2.")

    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def load_selected_features(ranking_path: Path) -> List[str]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking V2 introuvable: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    needed = {"feature", "selected_v2", "robust_selection_score"}
    if not needed.issubset(ranking.columns):
        raise ValueError(
            f"Ranking V2 doit contenir {sorted(needed)}."
        )

    selected = ranking[ranking["selected_v2"] == 1].copy()
    selected = selected.sort_values(
        "robust_selection_score",
        ascending=False,
    )

    features = selected["feature"].astype(str).tolist()

    if len(features) < 5:
        raise RuntimeError(
            f"Seulement {len(features)} features V2 sélectionnées; comparaison impossible."
        )
    return features


def build_subsets(features: List[str]) -> Dict[str, List[str]]:
    subsets: Dict[str, List[str]] = {}

    for size in SUBSET_SIZES:
        if len(features) >= size:
            subsets[f"TOP_{size}"] = features[:size]

    if f"TOP_{len(features)}" not in subsets:
        subsets[f"ALL_{len(features)}"] = features.copy()

    return subsets


def make_outer_folds(df: pd.DataFrame, n_internal: int = 5) -> List[FoldDef]:
    orig_mask = df[STAGE_COL].astype(str).eq("original_consumed")
    ext_mask = df[STAGE_COL].astype(str).eq("external_consumed")

    orig_idx = np.where(orig_mask.to_numpy())[0]
    ext_idx = np.where(ext_mask.to_numpy())[0]

    if len(orig_idx) == 0 or len(ext_idx) == 0:
        raise RuntimeError(
            f"Stages inattendus: original={len(orig_idx)}, external={len(ext_idx)}"
        )

    # L'ordre du dataframe est déjà chronologique.
    orig_local = np.arange(len(orig_idx))
    tscv = TimeSeriesSplit(n_splits=n_internal)

    folds: List[FoldDef] = []
    for i, (tr_local, va_local) in enumerate(tscv.split(orig_local), start=1):
        folds.append(
            FoldDef(
                name=f"INTERNAL_{i}",
                kind="internal",
                train_idx=orig_idx[tr_local],
                val_idx=orig_idx[va_local],
            )
        )

    folds.append(
        FoldDef(
            name="EXTERNAL_STRESS",
            kind="external_stress",
            train_idx=orig_idx,
            val_idx=ext_idx,
        )
    )
    return folds


def build_logistic(C: float, balanced: bool) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                class_weight="balanced" if balanced else None,
                max_iter=5000,
                random_state=87,
            ),
        ),
    ])


def build_hist_gradient_balanced():
    # HistGradientBoosting n'accepte pas class_weight dans toutes les versions sklearn.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=87,
            ),
        ),
    ])


def build_extra_trees():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=6,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=87,
                n_jobs=-1,
            ),
        ),
    ])


def tune_c_temporal(
    train_df: pd.DataFrame,
    features: List[str],
    balanced: bool,
    c_grid: List[float],
) -> Tuple[float, pd.DataFrame]:
    """
    Choisit C uniquement dans le TRAIN du fold externe.
    Critère:
      1) fraction de mini-folds avec PR lift > 0
      2) mean PR lift
      3) mean PR-AUC
      4) mean ROC-AUC
      5) plus petit C en tie-break
    """
    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    # Petit TRAIN: on évite des sous-splits trop maigres.
    if len(train_df) < 250:
        return 0.1, pd.DataFrame()

    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    Xall = train_df[features].apply(pd.to_numeric, errors="coerce")
    yall = train_df[TARGET].astype(int).to_numpy()

    for C in c_grid:
        mini = []
        for inner_i, (tr_idx, va_idx) in enumerate(tscv.split(Xall), start=1):
            Xtr = Xall.iloc[tr_idx]
            Xva = Xall.iloc[va_idx]
            ytr = yall[tr_idx]
            yva = yall[va_idx]

            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue

            model = build_logistic(C=C, balanced=balanced)
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xva)[:, 1]

            pr = safe_pr(yva, p)
            roc = safe_roc(yva, p)
            base = float(np.mean(yva))
            mini.append({
                "inner_fold": inner_i,
                "C": C,
                "pr_auc": pr,
                "roc_auc": roc,
                "val_rate": base,
                "pr_lift": pr - base,
            })

        if mini:
            mdf = pd.DataFrame(mini)
            rows.append({
                "C": C,
                "n_inner_folds": len(mdf),
                "pr_positive_fraction": float((mdf["pr_lift"] > 0).mean()),
                "mean_pr_lift": float(mdf["pr_lift"].mean()),
                "mean_pr_auc": float(mdf["pr_auc"].mean()),
                "mean_roc_auc": float(mdf["roc_auc"].mean()),
            })

    if not rows:
        return 0.1, pd.DataFrame()

    tuning = pd.DataFrame(rows).sort_values(
        [
            "pr_positive_fraction",
            "mean_pr_lift",
            "mean_pr_auc",
            "mean_roc_auc",
            "C",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return float(tuning.iloc[0]["C"]), tuning


def evaluate_one(
    df: pd.DataFrame,
    fold: FoldDef,
    subset_name: str,
    features: List[str],
    model_name: str,
    c_grid: List[float],
) -> Tuple[Dict, pd.DataFrame]:
    train_df = df.iloc[fold.train_idx].copy()
    val_df = df.iloc[fold.val_idx].copy()

    Xtr = train_df[features].apply(pd.to_numeric, errors="coerce")
    Xva = val_df[features].apply(pd.to_numeric, errors="coerce")
    ytr = train_df[TARGET].astype(int).to_numpy()
    yva = val_df[TARGET].astype(int).to_numpy()

    tuning = pd.DataFrame()
    selected_c = np.nan

    if model_name == "logistic_l2":
        selected_c, tuning = tune_c_temporal(
            train_df, features, balanced=False, c_grid=c_grid
        )
        model = build_logistic(selected_c, balanced=False)

    elif model_name == "logistic_l2_balanced":
        selected_c, tuning = tune_c_temporal(
            train_df, features, balanced=True, c_grid=c_grid
        )
        model = build_logistic(selected_c, balanced=True)

    elif model_name == "hist_gradient_boosting":
        model = build_hist_gradient_balanced()

    elif model_name == "extra_trees_balanced":
        model = build_extra_trees()

    else:
        raise ValueError(model_name)

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]

    roc = safe_roc(yva, p)
    pr = safe_pr(yva, p)
    val_rate = float(np.mean(yva))
    brier = float(brier_score_loss(yva, p))
    ll = float(log_loss(yva, np.clip(p, 1e-8, 1-1e-8)))

    # Baselines utilisent le taux de TRAIN: réaliste pour prédire le futur.
    train_rate = float(np.mean(ytr))
    bb = baseline_brier(train_rate)
    bll = baseline_logloss(train_rate)

    row = {
        "subset": subset_name,
        "n_features": len(features),
        "model": model_name,
        "fold": fold.name,
        "kind": fold.kind,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_positive_rate": train_rate,
        "val_positive_rate": val_rate,
        "selected_C": selected_c,
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift_vs_val_rate": pr - val_rate,
        "mean_probability": float(np.mean(p)),
        "brier": brier,
        "brier_baseline_train_rate": bb,
        "brier_improvement_vs_train_baseline": bb - brier,
        "log_loss": ll,
        "logloss_baseline_train_rate": bll,
        "logloss_improvement_vs_train_baseline": bll - ll,
    }

    if not tuning.empty:
        tuning = tuning.copy()
        tuning.insert(0, "outer_fold", fold.name)
        tuning.insert(0, "model", model_name)
        tuning.insert(0, "subset", subset_name)

    return row, tuning


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["subset", "n_features", "model"]

    for keys, block in metrics.groupby(group_cols, sort=False):
        subset, n_features, model = keys
        internal = block[block["kind"] == "internal"].copy()
        stress = block[block["kind"] == "external_stress"].copy()

        if len(stress) != 1:
            raise RuntimeError(f"Stress fold absent pour {subset}/{model}")

        s = stress.iloc[0]

        mean_internal_roc = float(internal["roc_auc"].mean())
        mean_internal_pr = float(internal["pr_auc"].mean())
        mean_internal_lift = float(internal["pr_lift_vs_val_rate"].mean())
        internal_pr_pos_frac = float(
            (internal["pr_lift_vs_val_rate"] > 0).mean()
        )
        internal_roc_pos_frac = float(
            (internal["roc_auc"] > 0.5).mean()
        )
        internal_brier_better_frac = float(
            (internal["brier_improvement_vs_train_baseline"] > 0).mean()
        )

        stress_roc = float(s["roc_auc"])
        stress_pr = float(s["pr_auc"])
        stress_lift = float(s["pr_lift_vs_val_rate"])

        # Gate V2 de comparaison. Calibration finale sera une étape séparée.
        passes_gate = int(
            mean_internal_roc > 0.50
            and mean_internal_lift > 0
            and internal_pr_pos_frac >= 0.60
            and internal_roc_pos_frac >= 0.60
            and stress_roc > 0.50
            and stress_lift >= 0
        )

        # Score de ranking: on donne beaucoup de poids au stress fold,
        # puisque c'est précisément là que V1 a cassé.
        robustness_score = (
            0.30 * mean_internal_roc
            + 0.25 * mean_internal_pr
            + 0.20 * stress_roc
            + 0.20 * stress_pr
            + 0.05 * min(internal_pr_pos_frac, internal_roc_pos_frac)
        )

        c_vals = block["selected_C"].dropna().tolist()

        rows.append({
            "subset": subset,
            "n_features": int(n_features),
            "model": model,
            "passes_v2_model_gate": passes_gate,
            "mean_internal_roc_auc": mean_internal_roc,
            "mean_internal_pr_auc": mean_internal_pr,
            "mean_internal_pr_lift": mean_internal_lift,
            "internal_pr_above_base_fraction": internal_pr_pos_frac,
            "internal_roc_above_050_fraction": internal_roc_pos_frac,
            "internal_brier_better_fraction": internal_brier_better_frac,
            "stress_roc_auc": stress_roc,
            "stress_pr_auc": stress_pr,
            "stress_val_rate": float(s["val_positive_rate"]),
            "stress_pr_lift": stress_lift,
            "stress_brier": float(s["brier"]),
            "stress_brier_improvement_vs_train_baseline": float(
                s["brier_improvement_vs_train_baseline"]
            ),
            "stress_log_loss": float(s["log_loss"]),
            "stress_logloss_improvement_vs_train_baseline": float(
                s["logloss_improvement_vs_train_baseline"]
            ),
            "robustness_score": robustness_score,
            "selected_C_values": "|".join(
                f"{x:g}" for x in c_vals
            ) if c_vals else "",
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "passes_v2_model_gate",
            "stress_pr_lift",
            "stress_roc_auc",
            "mean_internal_pr_lift",
            "mean_internal_roc_auc",
            "n_features",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(
    path: Path,
    df: pd.DataFrame,
    subsets: Dict[str, List[str]],
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# First Half 37 — V2 Subset + Model Comparison")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append(
        "- Les 303 matchs externes précédents sont déjà consommés et servent ici de stress fold."
    )
    lines.append(
        "- Aucun match postérieur au 19 septembre 2026 n'est utilisé."
    )
    lines.append(
        "- Aucun futur holdout n'est utilisé pour choisir subset, modèle ou C."
    )
    lines.append("")
    lines.append("## DEVELOPMENT V2")
    lines.append("")
    lines.append(
        f"- Matchs : **{len(df)}** | positifs **{int(df[TARGET].sum())}** "
        f"({df[TARGET].mean():.2%})"
    )
    lines.append(
        f"- Dates : `{df[DATE_COL].min()}` -> `{df[DATE_COL].max()}`"
    )
    lines.append("")
    lines.append("## Sous-ensembles")
    lines.append("")
    for name, features in subsets.items():
        lines.append(
            f"- `{name}` : **{len(features)}** features — "
            + ", ".join(f"`{f}`" for f in features)
        )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("```")
    cols = [
        "rank",
        "subset",
        "n_features",
        "model",
        "passes_v2_model_gate",
        "mean_internal_roc_auc",
        "mean_internal_pr_auc",
        "mean_internal_pr_lift",
        "internal_pr_above_base_fraction",
        "stress_roc_auc",
        "stress_pr_auc",
        "stress_pr_lift",
        "stress_brier_improvement_vs_train_baseline",
        "robustness_score",
        "selected_C_values",
    ]
    lines.append(ranking[cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Rang #1")
    lines.append("")
    if not ranking.empty:
        r = ranking.iloc[0]
        lines.append(
            f"- `{r['subset']} + {r['model']}` | "
            f"features={int(r['n_features'])} | "
            f"gate={'PASS' if int(r['passes_v2_model_gate']) else 'FAIL'}"
        )
        lines.append(
            f"- Internal ROC={r['mean_internal_roc_auc']:.4f} | "
            f"Internal PR={r['mean_internal_pr_auc']:.4f} | "
            f"Internal lift={r['mean_internal_pr_lift']:+.4f}"
        )
        lines.append(
            f"- Stress ROC={r['stress_roc_auc']:.4f} | "
            f"Stress PR={r['stress_pr_auc']:.4f} | "
            f"Stress lift={r['stress_pr_lift']:+.4f}"
        )
    lines.append("")
    lines.append("## Suite")
    lines.append("")
    lines.append(
        "Si au moins un pipeline passe le gate V2, l'étape suivante est un audit/calibration "
        "temporelle du ou des meilleurs candidats, toujours sans utiliser de nouveaux matchs futurs."
    )
    lines.append(
        "Le futur holdout V2 devra être constitué uniquement de matchs postérieurs au "
        "19 septembre 2026 et restera fermé jusqu'au gel définitif du pipeline."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare les subsets/models First Half 37 V2 sur les 2 105 matchs connus."
    )
    parser.add_argument("--development-csv", default=str(DEFAULT_DEV))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING))
    parser.add_argument("--schema-json", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--internal-folds", type=int, default=5)
    parser.add_argument(
        "--c-grid",
        default="0.01,0.03,0.1,0.3,1,3,10",
        help="Grille C logistic séparée par virgules.",
    )
    args = parser.parse_args()

    dev_path = Path(args.development_csv)
    ranking_path = Path(args.ranking_csv)
    schema_path = Path(args.schema_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_grid = [float(x.strip()) for x in args.c_grid.split(",") if x.strip()]

    print("=" * 116)
    print("FIRST HALF 37 — V2 SUBSET + MODEL COMPARISON")
    print("=" * 116)
    print(f"DEVELOPMENT : {dev_path}")
    print(f"Ranking     : {ranking_path}")
    print(f"Schema      : {schema_path}")
    print(f"Output      : {out_dir}")
    print("IMPORTANT   : aucun match postérieur au 19 septembre 2026 n'est lu.")
    print()

    df = load_dev(dev_path)
    features = load_selected_features(ranking_path)
    subsets = build_subsets(features)
    folds = make_outer_folds(df, n_internal=args.internal_folds)

    print(
        f"Dataset={len(df)} | positifs={int(df[TARGET].sum())} "
        f"({df[TARGET].mean():.2%})"
    )
    print(
        f"Dates={df[DATE_COL].min()} -> {df[DATE_COL].max()}"
    )
    print()
    print("Sous-ensembles:")
    for name, feats in subsets.items():
        print(f"  {name:<10} {len(feats):2d} features")
    print()

    (out_dir / "first_half_37_v2_subset_definitions.json").write_text(
        json.dumps(subsets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    models = [
        "logistic_l2",
        "logistic_l2_balanced",
        "hist_gradient_boosting",
        "extra_trees_balanced",
    ]

    metric_rows = []
    tuning_frames = []

    for subset_name, subset_features in subsets.items():
        print("-" * 116)
        print(f"SUBSET {subset_name} | n_features={len(subset_features)}")

        for model_name in models:
            fold_rows = []

            for fold in folds:
                row, tuning = evaluate_one(
                    df=df,
                    fold=fold,
                    subset_name=subset_name,
                    features=subset_features,
                    model_name=model_name,
                    c_grid=c_grid,
                )
                metric_rows.append(row)
                fold_rows.append(row)

                if not tuning.empty:
                    tuning_frames.append(tuning)

            block = pd.DataFrame(fold_rows)
            internal = block[block["kind"] == "internal"]
            stress = block[block["kind"] == "external_stress"].iloc[0]

            print(
                f"  {model_name:<28} "
                f"INT ROC={internal['roc_auc'].mean():.4f} "
                f"PR={internal['pr_auc'].mean():.4f} "
                f"lift={internal['pr_lift_vs_val_rate'].mean():+.4f} | "
                f"STRESS ROC={stress['roc_auc']:.4f} "
                f"PR={stress['pr_auc']:.4f} "
                f"lift={stress['pr_lift_vs_val_rate']:+.4f}"
            )

    metrics = pd.DataFrame(metric_rows)
    ranking = summarize(metrics)

    metrics_path = out_dir / "first_half_37_v2_fold_metrics.csv"
    ranking_path_out = out_dir / "first_half_37_v2_model_ranking.csv"
    tuning_path = out_dir / "first_half_37_v2_c_tuning.csv"
    report_path = out_dir / "first_half_37_v2_model_comparison_report.md"
    manifest_path = out_dir / "first_half_37_v2_model_comparison_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    ranking.to_csv(ranking_path_out, index=False)

    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(
            tuning_path, index=False
        )
    else:
        pd.DataFrame().to_csv(tuning_path, index=False)

    write_report(
        report_path,
        df=df,
        subsets=subsets,
        metrics=metrics,
        ranking=ranking,
    )

    manifest = {
        "schema_version": "first_half_37_v2_model_comparison_v1",
        "development_rows": int(len(df)),
        "development_positive_rate": float(df[TARGET].mean()),
        "date_min": str(df[DATE_COL].min()),
        "date_max": str(df[DATE_COL].max()),
        "subset_definitions": subsets,
        "models": models,
        "c_grid": c_grid,
        "internal_folds": args.internal_folds,
        "external_stress_fold": True,
        "future_holdout_used": False,
        "ranking_file": str(ranking_path_out),
        "metrics_file": str(metrics_path),
        "report_file": str(report_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pass_count = int(ranking["passes_v2_model_gate"].sum())
    winner = ranking.iloc[0]

    print()
    print("=" * 116)
    print("COMPARAISON V2 TERMINEE")
    print("=" * 116)
    print(f"Configurations testées : {len(ranking)}")
    print(f"Pipelines passant le gate V2 : {pass_count}")
    print(
        f"#1 interne : {winner['subset']} + {winner['model']} | "
        f"features={int(winner['n_features'])} | "
        f"INT ROC={winner['mean_internal_roc_auc']:.4f} | "
        f"INT PR={winner['mean_internal_pr_auc']:.4f} | "
        f"INT lift={winner['mean_internal_pr_lift']:+.4f} | "
        f"STRESS ROC={winner['stress_roc_auc']:.4f} | "
        f"STRESS PR={winner['stress_pr_auc']:.4f} | "
        f"STRESS lift={winner['stress_pr_lift']:+.4f} | "
        f"gate={'PASS' if int(winner['passes_v2_model_gate']) else 'FAIL'}"
    )
    print(f"Rapport  : {report_path}")
    print(f"Ranking  : {ranking_path_out}")
    print(f"Metrics  : {metrics_path}")
    print(f"Manifest : {manifest_path}")
    print("IMPORTANT: aucun futur holdout n'a été utilisé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
