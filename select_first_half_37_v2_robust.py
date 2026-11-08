#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 ROBUST DEVELOPMENT + FEATURE SELECTION

But:
  1) fusionner les 1 802 matchs historiques + les 303 matchs externes déjà consommés;
  2) considérer ces 2 105 matchs comme DEVELOPMENT V2;
  3) refaire une sélection de features beaucoup plus stricte et temporelle;
  4) utiliser le bloc externe 14–19 septembre 2026 comme STRESS FOLD connu;
  5) NE PAS créer ni regarder de nouveau holdout futur dans ce script.

Important:
  - aucune donnée postérieure au dataset externe n'est utilisée;
  - les targets/résolutions futures ne sont jamais candidates;
  - Missing != 0;
  - pas de random CV;
  - pas de tuning de modèle final ici;
  - les 303 matchs externes sont désormais "consommés pour développement V2".

Sorties:
  outputs/first_half_37/v2_robust_selection/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGINAL = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "historical_api"
    / "First_Half_37_COMBINED.csv"
)
DEFAULT_EXTERNAL = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "post_model_external_v1"
    / "First_Half_37_POST_MODEL_EXTERNAL.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "v2_robust_selection"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"

V1_COMPACT10 = [
    "home_over25_freq",
    "away_wins_freq",
    "home_shots_37",
    "away_xg_37",
    "away_shots_on_target_37",
    "last_corner_minute",
    "away_form10",
    "league_home_over25_freq",
    "home_form5",
    "away_shots_37",
]

# Champs interdits: target, résolution future, diagnostics, métadonnées,
# ou information connue seulement après la 37e / la mi-temps.
EXCLUDE_EXACT = {
    TARGET,
    "snapshot_minute",
    "external_validation_only",
    "first_half_period_end_seen",
    "timeline_full",
    "first_half_goal_count",
    "goal_minutes_after_37_before_ht",
    "target_status",
    "target_resolution_class",
    "target_confidence",
    "target_warning",
    "final_score_home",
    "final_score_away",
    "timeline_event_count",
    "shotmap_event_count",
    "collected_at_utc",
}

EXCLUDE_PREFIXES = (
    "_prematch_",
    "_live37_",
    "_fh_odds_",
    "_pre_",
    "_live_",
    "_core_",
)

META_COLUMNS = {
    "schema_version",
    "match_id",
    "competition_id",
    "season_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "source_dataset",
    "source",
    "_v2_stage",
    "_v2_original_row_order",
    "prematch_policy",
    "_live_provenance_json",
}

ODDS_TOKENS = ("odds_", "fh_odds_")


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{100.0*x:.2f}%"


def safe_auc(y: pd.Series, s: pd.Series) -> float:
    mask = y.notna() & s.notna()
    yy = y.loc[mask]
    ss = s.loc[mask]
    if len(yy) < 20 or yy.nunique() < 2 or ss.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy.astype(int), ss.astype(float)))
    except Exception:
        return float("nan")


def safe_pr(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def safe_roc(y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def load_dataset(path: Path, stage: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {ID_COL, DATE_COL, TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name}: colonnes requises absentes: {missing}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if df[DATE_COL].isna().any():
        bad = int(df[DATE_COL].isna().sum())
        raise ValueError(f"{path.name}: {bad} kickoff_utc non parsables.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError(f"{path.name}: target non résolu présent.")
    if not set(df[TARGET].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: target contient autre chose que 0/1.")

    if df[ID_COL].isna().any():
        raise ValueError(f"{path.name}: match_id manquant.")

    before = len(df)
    df = df.drop_duplicates(subset=[ID_COL], keep="last").copy()
    if len(df) != before:
        print(f"WARNING: {before-len(df)} doublon(s) supprimé(s) dans {path.name}")

    df["_v2_stage"] = stage
    return df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def build_development(
    original: pd.DataFrame,
    external: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    overlap = set(original[ID_COL].astype(str)) & set(external[ID_COL].astype(str))
    if overlap:
        raise ValueError(f"Overlap original/externe détecté: {len(overlap)} match_id.")

    original_max = original[DATE_COL].max()
    external_min = external[DATE_COL].min()

    combined = pd.concat([original, external], ignore_index=True, sort=False)
    combined = combined.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    combined["_v2_original_row_order"] = np.arange(len(combined), dtype=int)

    meta = {
        "original_rows": int(len(original)),
        "external_rows": int(len(external)),
        "combined_rows": int(len(combined)),
        "original_positive_rate": float(original[TARGET].mean()),
        "external_positive_rate": float(external[TARGET].mean()),
        "combined_positive_rate": float(combined[TARGET].mean()),
        "original_date_min": str(original[DATE_COL].min()),
        "original_date_max": str(original_max),
        "external_date_min": str(external_min),
        "external_date_max": str(external[DATE_COL].max()),
        "external_strictly_after_original": bool(external_min > original_max),
        "overlap_match_id": 0,
    }
    return combined, meta


def is_candidate_name(col: str, include_odds: bool) -> bool:
    if col in META_COLUMNS or col in EXCLUDE_EXACT:
        return False
    if col.startswith(EXCLUDE_PREFIXES):
        return False
    low = col.lower()
    if not include_odds and any(tok in low for tok in ODDS_TOKENS):
        return False
    if "target" in low:
        return False
    if "goal_minutes_after" in low:
        return False
    if "period_end" in low or "timeline_full" in low:
        return False
    return True


def numeric_candidates(
    combined: pd.DataFrame,
    original: pd.DataFrame,
    external: pd.DataFrame,
    min_coverage: float,
    max_missing_delta: float,
    include_odds: bool,
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    candidates = []

    common_cols = [c for c in combined.columns if c in original.columns and c in external.columns]

    for col in common_cols:
        if not is_candidate_name(col, include_odds=include_odds):
            continue

        orig_num = pd.to_numeric(original[col], errors="coerce")
        ext_num = pd.to_numeric(external[col], errors="coerce")
        comb_num = pd.to_numeric(combined[col], errors="coerce")

        orig_cov = float(orig_num.notna().mean())
        ext_cov = float(ext_num.notna().mean())
        comb_cov = float(comb_num.notna().mean())
        missing_delta = abs((1-orig_cov) - (1-ext_cov))
        nunique = int(comb_num.nunique(dropna=True))

        numeric_enough = comb_cov >= min_coverage
        quality_pass = (
            orig_cov >= min_coverage
            and ext_cov >= min_coverage
            and missing_delta <= max_missing_delta
            and nunique >= 2
        )

        reason = "PASS"
        if orig_cov < min_coverage:
            reason = "LOW_COVERAGE_ORIGINAL"
        elif ext_cov < min_coverage:
            reason = "LOW_COVERAGE_EXTERNAL"
        elif missing_delta > max_missing_delta:
            reason = "MISSINGNESS_SHIFT"
        elif nunique < 2:
            reason = "CONSTANT_OR_SINGLE_VALUE"

        rows.append({
            "feature": col,
            "original_coverage": orig_cov,
            "external_coverage": ext_cov,
            "combined_coverage": comb_cov,
            "missing_rate_delta_abs": missing_delta,
            "nunique": nunique,
            "quality_pass": int(quality_pass),
            "quality_reason": reason,
            "v1_compact10": int(col in V1_COMPACT10),
        })

        if quality_pass and numeric_enough:
            candidates.append(col)

    audit = pd.DataFrame(rows).sort_values(
        ["quality_pass", "combined_coverage", "feature"],
        ascending=[False, False, True],
    )
    return candidates, audit


@dataclass
class FoldDef:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    kind: str


def make_folds(original: pd.DataFrame, external: pd.DataFrame, n_internal: int = 5) -> List[FoldDef]:
    original = original.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    external = external.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    if len(original) < 500:
        raise ValueError("Dataset original trop petit pour les folds temporels demandés.")

    folds: List[FoldDef] = []

    tscv = TimeSeriesSplit(n_splits=n_internal)
    dummy = np.arange(len(original))
    for i, (train_idx, val_idx) in enumerate(tscv.split(dummy), start=1):
        tr = original.iloc[train_idx].copy()
        va = original.iloc[val_idx].copy()
        folds.append(FoldDef(
            name=f"INTERNAL_{i}",
            train=tr,
            val=va,
            kind="internal",
        ))

    # Stress fold explicite: tout l'original -> bloc externe déjà consommé.
    folds.append(FoldDef(
        name="EXTERNAL_STRESS",
        train=original.copy(),
        val=external.copy(),
        kind="external_stress",
    ))
    return folds


def make_l2_pipeline(features: List[str], c: float = 0.1) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=c,
            solver="liblinear",
            max_iter=5000,
            random_state=87,
        )),
    ])


def make_l1_pipeline(features: List[str], c: float = 0.1) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1",
            C=c,
            solver="liblinear",
            max_iter=5000,
            random_state=87,
        )),
    ])


def evaluate_folds(
    folds: List[FoldDef],
    features: List[str],
    perm_repeats: int,
    l2_c: float,
    l1_c: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    perm_rows = []
    l1_rows = []

    for idx, fold in enumerate(folds, start=1):
        tr = fold.train.copy()
        va = fold.val.copy()

        Xtr = tr[features].apply(pd.to_numeric, errors="coerce")
        Xva = va[features].apply(pd.to_numeric, errors="coerce")
        ytr = tr[TARGET].astype(int).to_numpy()
        yva = va[TARGET].astype(int).to_numpy()

        l2 = make_l2_pipeline(features, c=l2_c)
        l2.fit(Xtr, ytr)
        p = l2.predict_proba(Xva)[:, 1]

        roc = safe_roc(yva, p)
        pr = safe_pr(yva, p)
        val_rate = float(np.mean(yva))
        brier = float(brier_score_loss(yva, p))
        ll = float(log_loss(yva, np.clip(p, 1e-8, 1-1e-8)))

        metric_rows.append({
            "fold": fold.name,
            "kind": fold.kind,
            "train_rows": len(tr),
            "val_rows": len(va),
            "train_date_min": tr[DATE_COL].min(),
            "train_date_max": tr[DATE_COL].max(),
            "val_date_min": va[DATE_COL].min(),
            "val_date_max": va[DATE_COL].max(),
            "train_positive_rate": float(np.mean(ytr)),
            "val_positive_rate": val_rate,
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_lift_vs_val_rate": pr - val_rate if not pd.isna(pr) else np.nan,
            "brier": brier,
            "log_loss": ll,
        })

        print(
            f"{fold.name:<16} | TRAIN={len(tr):4d} VAL={len(va):4d} | "
            f"ROC={roc:.4f} PR={pr:.4f} base={val_rate:.4f}"
        )

        # Permutation importance PR-AUC sur le bloc temporel de validation.
        perm = permutation_importance(
            l2,
            Xva,
            yva,
            scoring="average_precision",
            n_repeats=perm_repeats,
            random_state=87 + idx,
            n_jobs=-1,
        )

        # Associations univariées sur validation.
        yva_s = pd.Series(yva, index=Xva.index)
        for j, feature in enumerate(features):
            auc_uni = safe_auc(yva_s, Xva[feature])
            perm_rows.append({
                "fold": fold.name,
                "kind": fold.kind,
                "feature": feature,
                "permutation_pr_importance_mean": float(perm.importances_mean[j]),
                "permutation_pr_importance_std": float(perm.importances_std[j]),
                "univariate_auc": auc_uni,
                "univariate_direction": (
                    np.nan if pd.isna(auc_uni)
                    else (1 if auc_uni > 0.5 else (-1 if auc_uni < 0.5 else 0))
                ),
                "univariate_auc_distance": (
                    np.nan if pd.isna(auc_uni) else abs(auc_uni - 0.5)
                ),
            })

        # L1 stability sur le TRAIN du fold uniquement.
        l1 = make_l1_pipeline(features, c=l1_c)
        l1.fit(Xtr, ytr)
        coefs = l1.named_steps["model"].coef_[0]
        for feature, coef in zip(features, coefs):
            l1_rows.append({
                "fold": fold.name,
                "kind": fold.kind,
                "feature": feature,
                "coefficient": float(coef),
                "selected": int(abs(coef) > 1e-8),
                "sign": int(np.sign(coef)),
            })

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(perm_rows),
        pd.DataFrame(l1_rows),
    )


def aggregate_feature_stability(
    candidates: List[str],
    quality_audit: pd.DataFrame,
    perm_df: pd.DataFrame,
    l1_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    quality_map = quality_audit.set_index("feature").to_dict("index")

    for feature in candidates:
        p = perm_df[perm_df["feature"] == feature].copy()
        l = l1_df[l1_df["feature"] == feature].copy()

        p_int = p[p["kind"] == "internal"]
        p_ext = p[p["kind"] == "external_stress"]
        l_int = l[l["kind"] == "internal"]

        internal_perm_mean = float(p_int["permutation_pr_importance_mean"].mean())
        internal_perm_median = float(p_int["permutation_pr_importance_mean"].median())
        internal_perm_positive_fraction = float(
            (p_int["permutation_pr_importance_mean"] > 0).mean()
        )
        external_perm = (
            float(p_ext["permutation_pr_importance_mean"].iloc[0])
            if len(p_ext) else np.nan
        )

        # Direction univariée dominante interne.
        valid_dirs = p_int["univariate_direction"].dropna()
        valid_dirs = valid_dirs[valid_dirs != 0]
        if len(valid_dirs):
            pos_frac = float((valid_dirs > 0).mean())
            neg_frac = float((valid_dirs < 0).mean())
            dominant_dir = 1 if pos_frac >= neg_frac else -1
            univariate_direction_consistency = max(pos_frac, neg_frac)
        else:
            dominant_dir = 0
            univariate_direction_consistency = np.nan

        ext_auc = (
            float(p_ext["univariate_auc"].iloc[0])
            if len(p_ext) else np.nan
        )
        if pd.isna(ext_auc) or dominant_dir == 0:
            external_oriented_auc = np.nan
        else:
            external_oriented_auc = ext_auc if dominant_dir > 0 else 1.0 - ext_auc

        # L1.
        selected_int = l_int[l_int["selected"] == 1]
        l1_selection_fraction = float(l_int["selected"].mean()) if len(l_int) else np.nan

        if len(selected_int):
            pos = float((selected_int["sign"] > 0).mean())
            neg = float((selected_int["sign"] < 0).mean())
            l1_sign_consistency = max(pos, neg)
            l1_dominant_sign = 1 if pos >= neg else -1
        else:
            l1_sign_consistency = 0.0
            l1_dominant_sign = 0

        q = quality_map.get(feature, {})

        direction_break = int(
            not pd.isna(external_oriented_auc)
            and external_oriented_auc < 0.47
            and (not pd.isna(univariate_direction_consistency))
            and univariate_direction_consistency >= 0.75
        )

        # Hard gate de robustesse. C'est volontairement plus strict que V1.
        robust_pass = int(
            float(q.get("original_coverage", 0)) >= 0.80
            and float(q.get("external_coverage", 0)) >= 0.80
            and float(q.get("missing_rate_delta_abs", 1)) <= 0.10
            and l1_selection_fraction >= 0.40
            and l1_sign_consistency >= 0.75
            and internal_perm_positive_fraction >= 0.40
            and (pd.isna(external_perm) or external_perm >= -0.002)
            and not direction_break
        )

        rows.append({
            "feature": feature,
            "v1_compact10": int(feature in V1_COMPACT10),
            "original_coverage": q.get("original_coverage", np.nan),
            "external_coverage": q.get("external_coverage", np.nan),
            "missing_rate_delta_abs": q.get("missing_rate_delta_abs", np.nan),
            "internal_perm_mean": internal_perm_mean,
            "internal_perm_median": internal_perm_median,
            "internal_perm_positive_fraction": internal_perm_positive_fraction,
            "external_stress_perm_pr_importance": external_perm,
            "internal_univariate_direction": dominant_dir,
            "internal_univariate_direction_consistency": univariate_direction_consistency,
            "external_univariate_auc": ext_auc,
            "external_oriented_auc": external_oriented_auc,
            "direction_break_flag": direction_break,
            "l1_selection_fraction": l1_selection_fraction,
            "l1_sign_consistency": l1_sign_consistency,
            "l1_dominant_sign": l1_dominant_sign,
            "robust_pass_pre_redundancy": robust_pass,
        })

    out = pd.DataFrame(rows)

    # Score composite par rang percentilé. Le score sert à ordonner les PASS,
    # pas à rendre une feature PASS.
    score_cols = [
        "internal_perm_mean",
        "internal_perm_positive_fraction",
        "external_stress_perm_pr_importance",
        "l1_selection_fraction",
        "l1_sign_consistency",
        "external_oriented_auc",
    ]
    parts = []
    for col in score_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        parts.append(s.rank(pct=True, method="average").fillna(0.5))
    out["robust_selection_score"] = pd.concat(parts, axis=1).mean(axis=1)

    return out.sort_values(
        ["robust_pass_pre_redundancy", "robust_selection_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def remove_redundancy(
    dev: pd.DataFrame,
    feature_table: pd.DataFrame,
    rho_threshold: float,
    max_features: int,
) -> Tuple[List[str], pd.DataFrame]:
    pass_df = feature_table[
        feature_table["robust_pass_pre_redundancy"] == 1
    ].copy()

    if pass_df.empty:
        return [], pd.DataFrame(columns=[
            "kept_feature", "dropped_feature", "abs_spearman", "reason"
        ])

    ordered = pass_df.sort_values(
        "robust_selection_score", ascending=False
    )["feature"].tolist()

    numeric = dev[ordered].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()

    kept: List[str] = []
    drops = []

    for feat in ordered:
        drop_against = None
        drop_rho = np.nan
        for k in kept:
            rho = corr.loc[feat, k]
            if pd.notna(rho) and rho >= rho_threshold:
                drop_against = k
                drop_rho = float(rho)
                break

        if drop_against is None:
            kept.append(feat)
        else:
            drops.append({
                "kept_feature": drop_against,
                "dropped_feature": feat,
                "abs_spearman": drop_rho,
                "reason": f"abs_spearman>={rho_threshold:.2f}",
            })

        if len(kept) >= max_features:
            break

    return kept, pd.DataFrame(drops)


def write_report(
    path: Path,
    meta: Dict,
    quality_audit: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_table: pd.DataFrame,
    selected: List[str],
    redundancy: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# First Half 37 — V2 Robust Feature Selection")
    lines.append("")
    lines.append("## Development V2")
    lines.append("")
    lines.append(
        f"- Original consommé : **{meta['original_rows']}** matchs | "
        f"{pct(meta['original_positive_rate'])} positifs"
    )
    lines.append(
        f"- Externe consommé : **{meta['external_rows']}** matchs | "
        f"{pct(meta['external_positive_rate'])} positifs"
    )
    lines.append(
        f"- DEVELOPMENT V2 : **{meta['combined_rows']}** matchs | "
        f"{pct(meta['combined_positive_rate'])} positifs"
    )
    lines.append(
        f"- Dates : `{meta['original_date_min']}` -> `{meta['external_date_max']}`"
    )
    lines.append("- Aucun nouveau holdout futur n'est utilisé ici.")
    lines.append("")
    lines.append("## Folds")
    lines.append("")
    lines.append("```")
    lines.append(fold_metrics.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Filtrage qualité")
    lines.append("")
    lines.append(
        f"- Features auditées : **{len(quality_audit)}**"
    )
    lines.append(
        f"- Features passant qualité : **{int(quality_audit['quality_pass'].sum())}**"
    )
    lines.append("")
    lines.append("## Robustesse")
    lines.append("")
    pass_count = int(feature_table["robust_pass_pre_redundancy"].sum())
    lines.append(f"- PASS avant redondance : **{pass_count}**")
    lines.append(f"- Sélectionnées après redondance : **{len(selected)}**")
    lines.append("")
    lines.append("### Features V2 sélectionnées")
    lines.append("")
    if selected:
        for i, f in enumerate(selected, start=1):
            lines.append(f"{i}. `{f}`")
    else:
        lines.append("**Aucune feature n'a passé le gate robuste.**")
    lines.append("")
    lines.append("### Statut des 10 features V1")
    lines.append("")
    v1 = feature_table[feature_table["v1_compact10"] == 1].copy()
    if not v1.empty:
        cols = [
            "feature",
            "robust_pass_pre_redundancy",
            "robust_selection_score",
            "internal_perm_positive_fraction",
            "external_stress_perm_pr_importance",
            "l1_selection_fraction",
            "l1_sign_consistency",
            "external_oriented_auc",
            "direction_break_flag",
        ]
        lines.append("```")
        lines.append(v1[cols].to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append(
        "Ces 2 105 matchs sont désormais du DEVELOPMENT V2. "
        "Toute version V2 construite à partir de cet audit devra être validée "
        "sur des matchs postérieurs au 19 septembre 2026 qui n'ont jamais été utilisés ici."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construit DEVELOPMENT V2 et sélectionne des features robustes temporellement."
    )
    parser.add_argument("--original-csv", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--external-csv", default=str(DEFAULT_EXTERNAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--max-missing-delta", type=float, default=0.10)
    parser.add_argument("--rho-threshold", type=float, default=0.92)
    parser.add_argument("--max-selected-features", type=int, default=30)
    parser.add_argument("--internal-folds", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=5)
    parser.add_argument("--l2-c", type=float, default=0.10)
    parser.add_argument("--l1-c", type=float, default=0.10)
    parser.add_argument(
        "--include-odds",
        action="store_true",
        help="Inclure les colonnes odds si elles passent les règles qualité. Défaut: exclues.",
    )
    args = parser.parse_args()

    original_path = Path(args.original_csv)
    external_path = Path(args.external_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("FIRST HALF 37 — V2 ROBUST DEVELOPMENT + FEATURE SELECTION")
    print("=" * 108)
    print(f"Original : {original_path}")
    print(f"Externe  : {external_path}")
    print(f"Output   : {out_dir}")
    print("IMPORTANT: les 303 matchs externes sont désormais consommés pour DEVELOPMENT V2.")
    print("IMPORTANT: aucun nouveau holdout futur n'est lu.")
    print()

    original = load_dataset(original_path, "original_consumed")
    external = load_dataset(external_path, "external_consumed")
    dev, meta = build_development(original, external)

    print(
        f"Original={len(original)} | rate={original[TARGET].mean():.2%} | "
        f"{original[DATE_COL].min()} -> {original[DATE_COL].max()}"
    )
    print(
        f"External={len(external)} | rate={external[TARGET].mean():.2%} | "
        f"{external[DATE_COL].min()} -> {external[DATE_COL].max()}"
    )
    print(
        f"DEVELOPMENT V2={len(dev)} | positives={int(dev[TARGET].sum())} "
        f"({dev[TARGET].mean():.2%})"
    )
    print()

    dev_path = out_dir / "First_Half_37_V2_DEVELOPMENT.csv"
    dev.to_csv(dev_path, index=False)

    candidates, quality_audit = numeric_candidates(
        dev,
        original,
        external,
        min_coverage=args.min_coverage,
        max_missing_delta=args.max_missing_delta,
        include_odds=args.include_odds,
    )
    quality_path = out_dir / "first_half_37_v2_feature_quality.csv"
    quality_audit.to_csv(quality_path, index=False)

    print(
        f"Features candidates après qualité: {len(candidates)} "
        f"(min coverage={args.min_coverage:.0%}, max missing delta={args.max_missing_delta:.0%})"
    )
    if not candidates:
        raise RuntimeError("Aucune feature candidate après filtres qualité.")

    folds = make_folds(
        original=original,
        external=external,
        n_internal=args.internal_folds,
    )

    print()
    print("Validation temporelle robuste...")
    fold_metrics, perm_df, l1_df = evaluate_folds(
        folds=folds,
        features=candidates,
        perm_repeats=args.perm_repeats,
        l2_c=args.l2_c,
        l1_c=args.l1_c,
    )

    fold_metrics.to_csv(out_dir / "first_half_37_v2_fold_metrics.csv", index=False)
    perm_df.to_csv(out_dir / "first_half_37_v2_permutation_by_fold.csv", index=False)
    l1_df.to_csv(out_dir / "first_half_37_v2_l1_stability_by_fold.csv", index=False)

    feature_table = aggregate_feature_stability(
        candidates=candidates,
        quality_audit=quality_audit,
        perm_df=perm_df,
        l1_df=l1_df,
    )

    selected, redundancy = remove_redundancy(
        dev=dev,
        feature_table=feature_table,
        rho_threshold=args.rho_threshold,
        max_features=args.max_selected_features,
    )

    feature_table["selected_v2"] = feature_table["feature"].isin(selected).astype(int)
    feature_table.to_csv(
        out_dir / "first_half_37_v2_robust_feature_ranking.csv",
        index=False,
    )
    redundancy.to_csv(
        out_dir / "first_half_37_v2_redundancy_drops.csv",
        index=False,
    )

    schema = {
        "schema_version": "first_half_37_v2_robust_selection_v1",
        "target": TARGET,
        "snapshot_minute": 37,
        "development_rows": int(len(dev)),
        "development_positive_rate": float(dev[TARGET].mean()),
        "original_rows": int(len(original)),
        "external_consumed_rows": int(len(external)),
        "selected_features": selected,
        "n_selected_features": int(len(selected)),
        "selection_rules": {
            "min_coverage_each_block": args.min_coverage,
            "max_missing_delta_abs": args.max_missing_delta,
            "l1_selection_fraction_min": 0.40,
            "l1_sign_consistency_min": 0.75,
            "internal_perm_positive_fraction_min": 0.40,
            "external_stress_perm_min": -0.002,
            "direction_break_external_oriented_auc_lt": 0.47,
            "spearman_redundancy_threshold": args.rho_threshold,
            "max_selected_features": args.max_selected_features,
            "odds_included": bool(args.include_odds),
        },
        "validation_design": {
            "internal_expanding_folds": args.internal_folds,
            "external_stress_fold": True,
            "external_stress_dates": [
                str(external[DATE_COL].min()),
                str(external[DATE_COL].max()),
            ],
            "future_holdout_used": False,
        },
        "v1_compact10_reference": V1_COMPACT10,
    }

    schema_path = out_dir / "first_half_37_v2_robust_schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "meta": meta,
        "outputs": {
            "development_csv": str(dev_path),
            "quality_csv": str(quality_path),
            "fold_metrics_csv": str(out_dir / "first_half_37_v2_fold_metrics.csv"),
            "permutation_csv": str(out_dir / "first_half_37_v2_permutation_by_fold.csv"),
            "l1_csv": str(out_dir / "first_half_37_v2_l1_stability_by_fold.csv"),
            "ranking_csv": str(out_dir / "first_half_37_v2_robust_feature_ranking.csv"),
            "redundancy_csv": str(out_dir / "first_half_37_v2_redundancy_drops.csv"),
            "schema_json": str(schema_path),
        },
    }
    (out_dir / "first_half_37_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path = out_dir / "first_half_37_v2_robust_selection_report.md"
    write_report(
        report_path,
        meta=meta,
        quality_audit=quality_audit,
        fold_metrics=fold_metrics,
        feature_table=feature_table,
        selected=selected,
        redundancy=redundancy,
    )

    print()
    print("=" * 108)
    print("V2 ROBUST SELECTION TERMINEE")
    print("=" * 108)
    print(f"DEVELOPMENT V2 : {len(dev)} matchs | {int(dev[TARGET].sum())} positifs ({dev[TARGET].mean():.2%})")
    print(f"Candidates qualité : {len(candidates)}")
    print(f"PASS robustesse avant redondance : {int(feature_table['robust_pass_pre_redundancy'].sum())}")
    print(f"Features V2 sélectionnées : {len(selected)}")
    for i, f in enumerate(selected, start=1):
        print(f"  {i:02d}. {f}")

    print()
    print("STATUT DES FEATURES V1 COMPACT10")
    v1_view = feature_table[feature_table["v1_compact10"] == 1][
        [
            "feature",
            "robust_pass_pre_redundancy",
            "internal_perm_positive_fraction",
            "external_stress_perm_pr_importance",
            "l1_selection_fraction",
            "l1_sign_consistency",
            "external_oriented_auc",
            "direction_break_flag",
            "selected_v2",
        ]
    ]
    if len(v1_view):
        print(v1_view.to_string(index=False))
    else:
        print("Aucune feature V1 retrouvée parmi les candidates qualité.")

    print()
    print(f"Rapport : {report_path}")
    print(f"Schema  : {schema_path}")
    print(f"Ranking : {out_dir / 'first_half_37_v2_robust_feature_ranking.csv'}")
    print(f"DEV V2  : {dev_path}")
    print("IMPORTANT : le prochain holdout V2 devra être postérieur au 19 septembre 2026.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
