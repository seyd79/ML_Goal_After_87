#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 DRAW_WITH_GOALS ROBUST FEATURE SELECTION
===========================================================

OBJECTIF
--------
Créer le DEVELOPMENT spécialisé `draw_with_goals` puis sélectionner un PETIT
ensemble de features robustes pour la prochaine V3 spécialisée.

IMPORTANT — DISCIPLINE
----------------------
- Aucun nouveau match futur n'est lu.
- Les 2 699 matchs du V3 DEVELOPMENT sont déjà consommés.
- Le sous-ensemble draw_with_goals attendu est:
      190 lignes anciennes (feature-selection DEV)
       50 lignes du holdout V2 consommé (LATE_STRESS diagnostic)
      240 lignes au total.
- Les 50 LATE_STRESS NE SONT PAS utilisés pour décider quelles features passent.
  Ils sont seulement reportés comme diagnostic de stabilité tardive.
- Aucun modèle final n'est gelé ici.
- Aucun seuil de pari / zone d'alerte n'est choisi ici.

SELECTION
---------
Pool volontairement conservateur:
1) les 19 features robustes déjà retenues dans le V2;
2) quelques features LIVE <=37 explicitement reconstructibles et pertinentes
   pour la pression récente.

Pour chaque feature:
- couverture;
- 4 folds temporels expanding sur les 190 lignes anciennes uniquement;
- Logistic L2 univariée, C fixe=0.1;
- ROC, PR lift, signe du coefficient;
- stabilité de direction;
- diagnostic séparé sur les 50 LATE_STRESS.

Gate interne (190 anciennes lignes seulement):
- coverage >= 80%
- >= 4 folds éligibles
- mean ROC >= 0.54
- median ROC >= 0.54
- ROC > 0.50 dans >= 75% des folds
- mean PR lift >= +0.02
- PR lift > 0 dans >= 75% des folds
- stabilité du signe coefficient >= 75%

Puis pruning de redondance Spearman |rho| >= 0.90.
Maximum final: 8 features.

SORTIES
-------
outputs/first_half_37/v3_draw_with_goals_feature_selection/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

V3_DEV_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_context_benchmark"
    / "First_Half_37_V3_DEVELOPMENT_CONSUMED.csv"
)

V2_RANKING_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
    / "first_half_37_v2_robust_feature_ranking.csv"
)

CONTEXT_AUDIT_VERDICT_PATH = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_context_stability_audit"
    / "first_half_37_v3_context_stability_verdict.json"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v3_draw_with_goals_feature_selection"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
CONTEXT_COL = "score_context_37"
SOURCE_COL = "_v3_source"

EXPECTED_TOTAL_V3_ROWS = 2699
EXPECTED_DRAW_TOTAL = 240
EXPECTED_DRAW_SELECTION_DEV = 190
EXPECTED_DRAW_LATE_STRESS = 50

SELECTION_SOURCE = "v2_development_consumed"
LATE_SOURCE = "v2_final_holdout_consumed"

N_SPLITS = 4
UNIVARIATE_C = 0.1

MIN_COVERAGE = 0.80
MIN_FOLDS = 4
MIN_MEAN_ROC = 0.54
MIN_MEDIAN_ROC = 0.54
MIN_ROC_POSITIVE_FRACTION = 0.75
MIN_MEAN_PR_LIFT = 0.02
MIN_PR_POSITIVE_FRACTION = 0.75
MIN_SIGN_STABILITY = 0.75

REDUNDANCY_RHO = 0.90
MAX_SELECTED = 8

# Le TOP19 original sera relu du ranking V2.
# Ajouts conservateurs: uniquement stats historiques reconstructibles à <=37.
CURATED_EXTRA_FEATURES = [
    # pression cumulée
    "away_shots_37",
    "home_shots_on_target_37",
    "away_shots_on_target_37",
    "home_xg_37",

    # activité récente
    "shots_30_37",
    "shots_33_37",
    "sot_30_37",
    "sot_33_37",
    "sot_35_37",
    "xg_30_37",
    "xg_33_37",
    "xg_35_37",

    # corners / fautes / substitutions récents
    "corners_30_37",
    "corners_33_37",
    "corners_35_37",
    "fouls_30_37",
    "fouls_33_37",
    "fouls_35_37",
    "subs_30_37",
    "subs_33_37",

    # récence actions
    "last_corner_minute",
    "last_shot_minute",
    "minutes_since_last_shot",
    "last_shot_on_target_minute",
    "minutes_since_last_shot_on_target",
]


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


def safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def build_univariate_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=UNIVARIATE_C,
                solver="liblinear",
                class_weight=None,
                max_iter=5000,
                random_state=87,
            ),
        ),
    ])


def load_inputs() -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    for path in (V3_DEV_PATH, V2_RANKING_PATH, CONTEXT_AUDIT_VERDICT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    verdict = load_json(CONTEXT_AUDIT_VERDICT_PATH)

    if verdict.get("verdict") != "CONTEXT_SIGNAL_EXISTS":
        raise RuntimeError(
            "Le contexte spécialisé ne doit être développé que si "
            "l'audit temporel conclut CONTEXT_SIGNAL_EXISTS."
        )

    signals = verdict.get("research_signal_contexts", [])
    if "draw_with_goals" not in signals:
        raise RuntimeError(
            "draw_with_goals n'est pas présent dans research_signal_contexts."
        )

    df = pd.read_csv(V3_DEV_PATH, low_memory=False)

    if len(df) != EXPECTED_TOTAL_V3_ROWS:
        raise RuntimeError(
            f"V3 DEVELOPMENT={len(df)}, attendu={EXPECTED_TOTAL_V3_ROWS}."
        )

    required = {
        ID_COL,
        DATE_COL,
        TARGET,
        CONTEXT_COL,
        SOURCE_COL,
        "score_37_home",
        "score_37_away",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"V3 DEVELOPMENT: colonnes absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if df[DATE_COL].isna().any():
        raise RuntimeError("kickoff_utc non parsable.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise RuntimeError("Target non résolu.")
    df[TARGET] = df[TARGET].astype(int)

    if df[ID_COL].astype(str).duplicated().any():
        raise RuntimeError("V3 DEVELOPMENT contient des match_id dupliqués.")

    ranking = pd.read_csv(V2_RANKING_PATH, low_memory=False)
    required_rank = {"feature", "selected_v2", "robust_selection_score"}
    if not required_rank.issubset(ranking.columns):
        raise RuntimeError(
            f"Ranking V2 incomplet: "
            f"{sorted(required_rank - set(ranking.columns))}"
        )

    top19 = (
        ranking[ranking["selected_v2"] == 1]
        .sort_values("robust_selection_score", ascending=False)["feature"]
        .astype(str)
        .tolist()
    )

    if len(top19) != 19:
        raise RuntimeError(f"Attendu TOP19=19, trouvé={len(top19)}.")

    return df, top19, verdict


def build_draw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    draw = df[df[CONTEXT_COL].astype(str).eq("draw_with_goals")].copy()
    draw = draw.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    if len(draw) != EXPECTED_DRAW_TOTAL:
        raise RuntimeError(
            f"draw_with_goals={len(draw)}, attendu={EXPECTED_DRAW_TOTAL}."
        )

    h = pd.to_numeric(draw["score_37_home"], errors="coerce")
    a = pd.to_numeric(draw["score_37_away"], errors="coerce")

    bad = h.isna() | a.isna() | (h != a) | (h <= 0)
    if bad.any():
        sample = draw.loc[
            bad,
            [ID_COL, "score_37_home", "score_37_away"],
        ].head(10)
        raise RuntimeError(
            "Des lignes draw_with_goals ne respectent pas score_home=score_away>0:\n"
            + sample.to_string(index=False)
        )

    selection_n = int(draw[SOURCE_COL].eq(SELECTION_SOURCE).sum())
    late_n = int(draw[SOURCE_COL].eq(LATE_SOURCE).sum())

    if selection_n != EXPECTED_DRAW_SELECTION_DEV:
        raise RuntimeError(
            f"Selection DEV draw={selection_n}, "
            f"attendu={EXPECTED_DRAW_SELECTION_DEV}."
        )

    if late_n != EXPECTED_DRAW_LATE_STRESS:
        raise RuntimeError(
            f"LATE_STRESS draw={late_n}, "
            f"attendu={EXPECTED_DRAW_LATE_STRESS}."
        )

    unknown_sources = sorted(
        set(draw[SOURCE_COL].dropna().astype(str))
        - {SELECTION_SOURCE, LATE_SOURCE}
    )
    if unknown_sources:
        raise RuntimeError(
            f"Sources V3 inattendues dans draw_with_goals: {unknown_sources}"
        )

    draw["selection_partition"] = np.where(
        draw[SOURCE_COL].eq(SELECTION_SOURCE),
        "FEATURE_SELECTION_DEV_190",
        "CONSUMED_LATE_STRESS_50",
    )

    # Score exact utile comme audit descriptif.
    draw["draw_score_37"] = (
        h.astype(int).astype(str) + "-" + a.astype(int).astype(str)
    )

    return draw


def build_candidate_pool(
    draw: pd.DataFrame,
    top19: Sequence[str],
) -> Tuple[List[str], List[str]]:
    requested = list(dict.fromkeys([*top19, *CURATED_EXTRA_FEATURES]))

    available = []
    absent = []

    for feature in requested:
        if feature not in draw.columns:
            absent.append(feature)
            continue

        numeric = pd.to_numeric(draw[feature], errors="coerce")
        # Une feature entièrement absente ou constante ne sert pas.
        if numeric.notna().sum() < 20:
            continue
        if numeric.nunique(dropna=True) < 2:
            continue

        available.append(feature)

    return available, absent


def evaluate_feature_internal(
    selection_df: pd.DataFrame,
    feature: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    selection_df = selection_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    x_all = pd.to_numeric(selection_df[feature], errors="coerce")
    coverage = float(x_all.notna().mean())

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    rows: List[Dict[str, Any]] = []
    coefficient_signs: List[int] = []

    for fold_idx, (tr, va) in enumerate(
        splitter.split(selection_df),
        start=1,
    ):
        train = selection_df.iloc[tr]
        val = selection_df.iloc[va]

        Xtr = train[[feature]].apply(pd.to_numeric, errors="coerce")
        Xva = val[[feature]].apply(pd.to_numeric, errors="coerce")
        ytr = train[TARGET].astype(int).to_numpy()
        yva = val[TARGET].astype(int).to_numpy()

        eligible = (
            len(np.unique(ytr)) >= 2
            and len(np.unique(yva)) >= 2
            and Xtr[feature].notna().sum() >= 10
            and Xva[feature].notna().sum() >= 5
        )

        if not eligible:
            rows.append({
                "feature": feature,
                "fold": fold_idx,
                "eligible": 0,
                "train_n": len(train),
                "val_n": len(val),
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "pr_lift": np.nan,
                "coefficient": np.nan,
                "coefficient_sign": 0,
            })
            continue

        model = build_univariate_model()
        model.fit(Xtr, ytr)

        p = model.predict_proba(Xva)[:, 1]
        roc = safe_roc(yva, p)
        pr = safe_pr(yva, p)
        base = float(yva.mean())

        coef = float(
            model.named_steps["model"].coef_[0][0]
        )
        sign = int(np.sign(coef))
        if sign != 0:
            coefficient_signs.append(sign)

        rows.append({
            "feature": feature,
            "fold": fold_idx,
            "eligible": 1,
            "train_n": len(train),
            "val_n": len(val),
            "train_positive_rate": float(ytr.mean()),
            "val_positive_rate": base,
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_lift": pr - base,
            "coefficient": coef,
            "coefficient_sign": sign,
        })

    folds = pd.DataFrame(rows)
    eligible = folds[folds["eligible"] == 1].copy()

    if len(eligible):
        mean_roc = float(eligible["roc_auc"].mean())
        median_roc = float(eligible["roc_auc"].median())
        min_roc = float(eligible["roc_auc"].min())
        roc_pos_frac = float((eligible["roc_auc"] > 0.5).mean())

        mean_pr_lift = float(eligible["pr_lift"].mean())
        median_pr_lift = float(eligible["pr_lift"].median())
        min_pr_lift = float(eligible["pr_lift"].min())
        pr_pos_frac = float((eligible["pr_lift"] > 0).mean())
    else:
        mean_roc = median_roc = min_roc = np.nan
        roc_pos_frac = np.nan
        mean_pr_lift = median_pr_lift = min_pr_lift = np.nan
        pr_pos_frac = np.nan

    if coefficient_signs:
        pos_frac = float(
            np.mean(np.asarray(coefficient_signs) > 0)
        )
        neg_frac = float(
            np.mean(np.asarray(coefficient_signs) < 0)
        )
        sign_stability = max(pos_frac, neg_frac)
        dominant_sign = 1 if pos_frac >= neg_frac else -1
    else:
        sign_stability = np.nan
        dominant_sign = 0

    passes = int(
        coverage >= MIN_COVERAGE
        and len(eligible) >= MIN_FOLDS
        and mean_roc >= MIN_MEAN_ROC
        and median_roc >= MIN_MEDIAN_ROC
        and roc_pos_frac >= MIN_ROC_POSITIVE_FRACTION
        and mean_pr_lift >= MIN_MEAN_PR_LIFT
        and pr_pos_frac >= MIN_PR_POSITIVE_FRACTION
        and sign_stability >= MIN_SIGN_STABILITY
    )

    # Score exclusivement basé sur la sélection DEV 190.
    robust_score = (
        max(0.0, mean_roc - 0.5) * 2.0
        + max(0.0, mean_pr_lift) * 1.5
        + (roc_pos_frac if np.isfinite(roc_pos_frac) else 0.0) * 0.15
        + (pr_pos_frac if np.isfinite(pr_pos_frac) else 0.0) * 0.15
        + (sign_stability if np.isfinite(sign_stability) else 0.0) * 0.10
        + coverage * 0.05
    )

    summary = {
        "feature": feature,
        "selection_dev_coverage": coverage,
        "eligible_folds": int(len(eligible)),
        "mean_roc_auc": mean_roc,
        "median_roc_auc": median_roc,
        "min_roc_auc": min_roc,
        "roc_above_050_fraction": roc_pos_frac,
        "mean_pr_lift": mean_pr_lift,
        "median_pr_lift": median_pr_lift,
        "min_pr_lift": min_pr_lift,
        "pr_lift_positive_fraction": pr_pos_frac,
        "coefficient_sign_stability": sign_stability,
        "dominant_coefficient_sign": dominant_sign,
        "passes_internal_gate": passes,
        "robust_score": robust_score,
    }

    return summary, folds


def evaluate_feature_late(
    selection_df: pd.DataFrame,
    late_df: pd.DataFrame,
    feature: str,
) -> Dict[str, Any]:
    Xtr = selection_df[[feature]].apply(pd.to_numeric, errors="coerce")
    Xva = late_df[[feature]].apply(pd.to_numeric, errors="coerce")

    ytr = selection_df[TARGET].astype(int).to_numpy()
    yva = late_df[TARGET].astype(int).to_numpy()

    coverage = float(Xva[feature].notna().mean())

    if (
        len(np.unique(ytr)) < 2
        or len(np.unique(yva)) < 2
        or Xtr[feature].notna().sum() < 10
        or Xva[feature].notna().sum() < 5
    ):
        return {
            "late_coverage": coverage,
            "late_roc_auc": np.nan,
            "late_pr_auc": np.nan,
            "late_pr_lift": np.nan,
            "late_coefficient": np.nan,
            "late_direction_supports_internal": np.nan,
        }

    model = build_univariate_model()
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xva)[:, 1]
    roc = safe_roc(yva, p)
    pr = safe_pr(yva, p)
    base = float(yva.mean())
    coef = float(model.named_steps["model"].coef_[0][0])

    return {
        "late_coverage": coverage,
        "late_roc_auc": roc,
        "late_pr_auc": pr,
        "late_pr_lift": pr - base,
        "late_coefficient": coef,
    }


def add_late_direction_support(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    out = ranking.copy()

    support = []
    for _, row in out.iterrows():
        internal_sign = int(row["dominant_coefficient_sign"])
        late_coef = row["late_coefficient"]

        if internal_sign == 0 or pd.isna(late_coef):
            support.append(np.nan)
        else:
            support.append(
                int(np.sign(float(late_coef)) == internal_sign)
            )

    out["late_direction_supports_internal"] = support
    return out


def prune_redundancy(
    selection_df: pd.DataFrame,
    ranking: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame]:
    passed = ranking[
        ranking["passes_internal_gate"] == 1
    ].copy()

    passed = passed.sort_values(
        [
            "robust_score",
            "mean_roc_auc",
            "mean_pr_lift",
        ],
        ascending=[False, False, False],
    )

    if passed.empty:
        return [], pd.DataFrame(
            columns=[
                "feature",
                "action",
                "kept_feature",
                "abs_spearman_rho",
            ]
        )

    features = passed["feature"].tolist()

    numeric = selection_df[features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    corr = numeric.corr(method="spearman").abs()

    kept: List[str] = []
    audit_rows: List[Dict[str, Any]] = []

    for feature in features:
        conflict_feature = None
        conflict_rho = np.nan

        for existing in kept:
            rho = corr.loc[feature, existing]
            if pd.notna(rho) and rho >= REDUNDANCY_RHO:
                conflict_feature = existing
                conflict_rho = float(rho)
                break

        if conflict_feature is None:
            kept.append(feature)
            audit_rows.append({
                "feature": feature,
                "action": "KEEP",
                "kept_feature": feature,
                "abs_spearman_rho": np.nan,
            })
        else:
            audit_rows.append({
                "feature": feature,
                "action": "DROP_REDUNDANT",
                "kept_feature": conflict_feature,
                "abs_spearman_rho": conflict_rho,
            })

        if len(kept) >= MAX_SELECTED:
            # On continue l'audit pour expliquer les features restantes,
            # mais elles ne peuvent plus entrer.
            pass

    # Limite finale explicite.
    final = kept[:MAX_SELECTED]

    if len(kept) > MAX_SELECTED:
        for feature in kept[MAX_SELECTED:]:
            audit_rows.append({
                "feature": feature,
                "action": "DROP_MAX_SELECTED_CAP",
                "kept_feature": "",
                "abs_spearman_rho": np.nan,
            })

    audit = pd.DataFrame(audit_rows)
    return final, audit


def score_distribution_table(draw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score, g in draw.groupby("draw_score_37", sort=True):
        rows.append({
            "draw_score_37": score,
            "n": int(len(g)),
            "positives": int(g[TARGET].sum()),
            "positive_rate": float(g[TARGET].mean()),
            "selection_dev_n": int(
                g[SOURCE_COL].eq(SELECTION_SOURCE).sum()
            ),
            "late_stress_n": int(
                g[SOURCE_COL].eq(LATE_SOURCE).sum()
            ),
        })
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    draw: pd.DataFrame,
    candidate_pool: Sequence[str],
    absent_features: Sequence[str],
    ranking: pd.DataFrame,
    selected_features: Sequence[str],
    redundancy: pd.DataFrame,
    score_dist: pd.DataFrame,
) -> None:
    selection = draw[
        draw["selection_partition"] == "FEATURE_SELECTION_DEV_190"
    ]
    late = draw[
        draw["selection_partition"] == "CONSUMED_LATE_STRESS_50"
    ]

    lines: List[str] = []

    lines.append("# First Half 37 — V3 DRAW_WITH_GOALS Robust Feature Selection")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- Aucun nouveau match futur utilisé.")
    lines.append("- Les 240 matchs sont déjà consommés.")
    lines.append(
        "- La sélection de features utilise **uniquement les 190 anciens "
        "draw_with_goals**."
    )
    lines.append(
        "- Les 50 matchs tardifs sont diagnostics seulement et ne changent "
        "pas le PASS interne."
    )
    lines.append("- Aucun modèle final n'est gelé ici.")
    lines.append("")
    lines.append("## Dataset spécialisé")
    lines.append("")
    lines.append(
        f"- Total : **{len(draw)}** | positifs={int(draw[TARGET].sum())} "
        f"({draw[TARGET].mean():.2%})"
    )
    lines.append(
        f"- FEATURE_SELECTION_DEV : **{len(selection)}** | "
        f"positifs={int(selection[TARGET].sum())} "
        f"({selection[TARGET].mean():.2%})"
    )
    lines.append(
        f"- CONSUMED_LATE_STRESS : **{len(late)}** | "
        f"positifs={int(late[TARGET].sum())} "
        f"({late[TARGET].mean():.2%})"
    )
    lines.append("")
    lines.append("## Scores à 37'")
    lines.append("")
    lines.append("```")
    lines.append(score_dist.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Pool candidat")
    lines.append("")
    lines.append(f"- Features testées : **{len(candidate_pool)}**")
    lines.append(
        "- Absentes du dataset (ignorées) : "
        + (", ".join(f"`{x}`" for x in absent_features) if absent_features else "aucune")
    )
    lines.append("")
    lines.append("## Gate interne")
    lines.append("")
    lines.append(f"- coverage >= {MIN_COVERAGE:.0%}")
    lines.append(f"- folds éligibles >= {MIN_FOLDS}")
    lines.append(f"- mean ROC >= {MIN_MEAN_ROC:.2f}")
    lines.append(f"- median ROC >= {MIN_MEDIAN_ROC:.2f}")
    lines.append(
        f"- ROC > .50 fraction >= {MIN_ROC_POSITIVE_FRACTION:.0%}"
    )
    lines.append(f"- mean PR lift >= {MIN_MEAN_PR_LIFT:+.2f}")
    lines.append(
        f"- PR lift > 0 fraction >= {MIN_PR_POSITIVE_FRACTION:.0%}"
    )
    lines.append(
        f"- coefficient sign stability >= {MIN_SIGN_STABILITY:.0%}"
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    show = [
        "feature",
        "passes_internal_gate",
        "robust_score",
        "selection_dev_coverage",
        "mean_roc_auc",
        "median_roc_auc",
        "roc_above_050_fraction",
        "mean_pr_lift",
        "pr_lift_positive_fraction",
        "coefficient_sign_stability",
        "late_roc_auc",
        "late_pr_lift",
        "late_direction_supports_internal",
    ]
    lines.append("```")
    lines.append(ranking[show].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Redondance")
    lines.append("")
    lines.append("```")
    lines.append(
        redundancy.to_string(index=False)
        if not redundancy.empty
        else "AUCUNE FEATURE PASSE LE GATE"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Features retenues pour le prochain benchmark")
    lines.append("")
    if selected_features:
        for feature in selected_features:
            lines.append(f"- `{feature}`")
    else:
        lines.append("**AUCUNE** — signal feature insuffisamment robuste.")
    lines.append("")
    lines.append("## Suite")
    lines.append("")
    lines.append(
        "Ces features sont des candidates de DEVELOPMENT uniquement. "
        "La prochaine étape sera une comparaison multivariée limitée "
        "(Logistic L2 baseline + quelques challengers simples) sur les "
        "données consommées, puis seulement un freeze avant un nouveau "
        "holdout futur."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sélection robuste de features pour V3 draw_with_goals "
            "sans nouveau holdout."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 DRAW_WITH_GOALS ROBUST FEATURE SELECTION")
    print("=" * 120)
    print("AUCUN NOUVEAU HOLDOUT FUTUR")
    print("SELECTION SUR 190 ANCIENS MATCHS UNIQUEMENT")
    print("LATE_STRESS 50 = DIAGNOSTIC SEULEMENT")
    print()

    df, top19, context_verdict = load_inputs()
    draw = build_draw_dataset(df)

    selection = draw[
        draw["selection_partition"] == "FEATURE_SELECTION_DEV_190"
    ].copy()

    late = draw[
        draw["selection_partition"] == "CONSUMED_LATE_STRESS_50"
    ].copy()

    print(
        f"DRAW_WITH_GOALS TOTAL : {len(draw)} | "
        f"positifs={int(draw[TARGET].sum())} ({draw[TARGET].mean():.2%})"
    )
    print(
        f"SELECTION DEV         : {len(selection)} | "
        f"positifs={int(selection[TARGET].sum())} ({selection[TARGET].mean():.2%})"
    )
    print(
        f"LATE STRESS DIAG      : {len(late)} | "
        f"positifs={int(late[TARGET].sum())} ({late[TARGET].mean():.2%})"
    )
    print()

    candidate_pool, absent = build_candidate_pool(draw, top19)

    print(f"Features candidates disponibles : {len(candidate_pool)}")
    if absent:
        print(f"Features demandées absentes     : {len(absent)}")
    print()

    summaries: List[Dict[str, Any]] = []
    fold_frames: List[pd.DataFrame] = []

    for i, feature in enumerate(candidate_pool, start=1):
        summary, folds = evaluate_feature_internal(
            selection,
            feature,
        )

        late_metrics = evaluate_feature_late(
            selection,
            late,
            feature,
        )

        summary.update(late_metrics)
        summaries.append(summary)
        fold_frames.append(folds)

        print(
            f"[{i:02d}/{len(candidate_pool):02d}] "
            f"{feature:<34} | "
            f"ROC={summary['mean_roc_auc']:.4f} | "
            f"PRlift={summary['mean_pr_lift']:+.4f} | "
            f"stable={summary['coefficient_sign_stability']:.0%} | "
            f"LATE ROC={summary['late_roc_auc']:.4f} | "
            f"{'PASS' if summary['passes_internal_gate'] else 'FAIL'}"
        )

    ranking = pd.DataFrame(summaries)
    ranking = add_late_direction_support(ranking)

    ranking = ranking.sort_values(
        [
            "passes_internal_gate",
            "robust_score",
            "mean_roc_auc",
            "mean_pr_lift",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    ranking.insert(0, "rank", np.arange(1, len(ranking)+1))

    fold_metrics = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames
        else pd.DataFrame()
    )

    selected_features, redundancy = prune_redundancy(
        selection,
        ranking,
    )

    score_dist = score_distribution_table(draw)

    dataset_path = (
        out_dir
        / "First_Half_37_V3_DRAW_WITH_GOALS_DEVELOPMENT_CONSUMED.csv"
    )
    ranking_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_feature_ranking.csv"
    )
    fold_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_feature_fold_metrics.csv"
    )
    redundancy_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_redundancy_audit.csv"
    )
    score_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_score_distribution.csv"
    )
    schema_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_selected_features.json"
    )
    report_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_feature_selection_report.md"
    )
    manifest_path = (
        out_dir
        / "first_half_37_v3_draw_with_goals_feature_selection_manifest.json"
    )

    draw.to_csv(dataset_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    redundancy.to_csv(redundancy_path, index=False)
    score_dist.to_csv(score_path, index=False)

    schema_payload = {
        "schema_version": "first_half_37_v3_draw_with_goals_features_v1",
        "context": "draw_with_goals",
        "selection_rows": len(selection),
        "late_stress_rows": len(late),
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "selection_uses_late_stress_targets": False,
        "redundancy_threshold_abs_spearman": REDUNDANCY_RHO,
        "max_selected": MAX_SELECTED,
    }
    schema_path.write_text(
        json.dumps(
            schema_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(
        report_path,
        draw=draw,
        candidate_pool=candidate_pool,
        absent_features=absent,
        ranking=ranking,
        selected_features=selected_features,
        redundancy=redundancy,
        score_dist=score_dist,
    )

    manifest = {
        "schema_version": (
            "first_half_37_v3_draw_with_goals_feature_selection_manifest_v1"
        ),
        "created_at_utc": now_utc_iso(),
        "mode": "research_only_consumed_data",
        "new_future_holdout_used": False,
        "context_audit_verdict": context_verdict.get("verdict"),
        "context": "draw_with_goals",
        "v3_development_source": str(V3_DEV_PATH),
        "v3_development_source_sha256": sha256_file(V3_DEV_PATH),
        "v2_ranking_source": str(V2_RANKING_PATH),
        "v2_ranking_source_sha256": sha256_file(V2_RANKING_PATH),
        "rows_total": len(draw),
        "selection_dev_rows": len(selection),
        "late_stress_rows": len(late),
        "selection_uses_late_stress_targets": False,
        "candidate_pool": candidate_pool,
        "selected_features": selected_features,
        "gate": {
            "min_coverage": MIN_COVERAGE,
            "min_folds": MIN_FOLDS,
            "min_mean_roc": MIN_MEAN_ROC,
            "min_median_roc": MIN_MEDIAN_ROC,
            "min_roc_positive_fraction": MIN_ROC_POSITIVE_FRACTION,
            "min_mean_pr_lift": MIN_MEAN_PR_LIFT,
            "min_pr_positive_fraction": MIN_PR_POSITIVE_FRACTION,
            "min_sign_stability": MIN_SIGN_STABILITY,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed_count = int(ranking["passes_internal_gate"].sum())

    print()
    print("=" * 120)
    print("DRAW_WITH_GOALS FEATURE SELECTION TERMINEE")
    print("=" * 120)
    print(f"Features testées         : {len(ranking)}")
    print(f"Features PASS interne    : {passed_count}")
    print(f"Features retenues        : {len(selected_features)}")
    print(
        "Sélection finale         : "
        + (
            ", ".join(selected_features)
            if selected_features
            else "AUCUNE"
        )
    )
    print()
    print(f"Dataset spécialisé       : {dataset_path}")
    print(f"Ranking                  : {ranking_path}")
    print(f"Rapport                  : {report_path}")
    print(f"Schema features          : {schema_path}")
    print()
    print(
        "IMPORTANT: les 50 LATE_STRESS n'ont PAS servi à décider "
        "quelles features passent le gate interne."
    )

    if not selected_features:
        print(
            "VERDICT RESEARCH: INSUFFICIENT_ROBUST_FEATURES — "
            "ne pas forcer un modèle spécialisé."
        )
        return 3

    print(
        "Prochaine étape: benchmark multivarié limité sur les features retenues."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
