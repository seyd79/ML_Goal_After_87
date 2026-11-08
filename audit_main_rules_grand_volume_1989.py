#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOAL AFTER 87 — GRAND VOLUME AUDIT DES RÈGLES PRINCIPALES
=========================================================

Objectif
--------
Appliquer les règles principales figées à tout le volume actuellement
disponible, SANS mélanger les niveaux de preuve.

Volumes produits
----------------
1) HIST_DEV_1368_FINAL_MODEL
   - descriptif / fit historique
   - V3-H final a été entraîné sur ces 1368 lignes
   - V3-C production est un refit_all

2) HIST_SEEN_TAIL_342_FINAL_MODEL
   - historique hors DEV pour V3-H, mais déjà inspecté auparavant
   - descriptif, pas holdout pristine

3) HIST_ALL_1710_FINAL_MODEL
   - les 1710 historiques complets
   - descriptif uniquement

4) ROLLING_OOS_684_EXACT
   - vrai bloc rolling OOS historique
   - V3-C et V3-H recalculés fold par fold
   - bloc de validation historique propre au protocole

5) V3I_EXTERNAL_279
   - validation externe 14 sept -> 7 oct 2026
   - modèles figés

6) OOS_PLUS_EXTERNAL_963
   - 684 rolling OOS + 279 V3-I
   - meilleur grand bloc de validation disponible

7) ALL_UNIQUE_1989_FINAL_MODEL
   - 1710 historiques + 279 V3-I
   - grand volume descriptif
   - NE PAS confondre avec une validation indépendante

Règles principales gelées
-------------------------
V3-C principal : 55 <= p < 65 %
V3-H principal : 31 <= p < 34 %
V3-H fort      : 32 <= p < 34 %

Règles secondaires conservées pour référence
---------------------------------------------
V3-C ciblé     : 58 <= p < 61 %
V3-H resserré  : 32.5 <= p < 34 %

Aucune optimisation de seuil n'est effectuée.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:",
    )
)

ORIGINAL_V3H_SCRIPT = ROOT / "train_compare_v3c_v3h_calibration.py"

V3C_MODEL_PATH = (
    ROOT
    / "outputs/model_v3_experiments/candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)

V3H_MODEL_CANDIDATES = [
    ROOT
    / "outputs/v3i_unseen_holdout_500/batch1_frozen_279"
    / "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368.joblib",
    ROOT
    / "outputs/v3i_unseen_holdout_500/evaluation_batch1_279_EXACT"
    / "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368.joblib",
]

V3I_PATH = (
    ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv"
)

OUT_DIR = ROOT / "outputs/grand_volume_rules_audit"

TARGET = "goal_after_87"
MATCH_ID = "match_id"
DEV_N = 1368
TAIL_N = 342
ROLLING_OOS_N = 684
V3I_EXPECTED_N = 279

EPS = 1e-12

RULES = [
    ("V3C_MAIN_55_65", "p_v3c", 0.55, 0.65, "MAIN"),
    ("V3H_MAIN_31_34", "p_v3h", 0.31, 0.34, "MAIN"),
    ("V3H_STRONG_32_34", "p_v3h", 0.32, 0.34, "MAIN"),
    ("V3C_SECONDARY_58_61", "p_v3c", 0.58, 0.61, "SECONDARY"),
    ("V3H_SECONDARY_32p5_34", "p_v3h", 0.325, 0.34, "SECONDARY"),
]

# Fidélité historique connue avant ce grand audit.
EXPECTED_OOS_COUNTS = {
    "V3C_MAIN_55_65": (157, 49),
    "V3C_SECONDARY_58_61": (54, 19),
    "V3H_MAIN_31_34": (126, 50),
    "V3H_STRONG_32_34": (55, 23),
    "V3H_SECONDARY_32p5_34": (39, 16),
}

EXPECTED_ET_RAW_ROC = [
    0.559719,
    0.506460,
    0.598338,
    0.591112,
]

ROC_TOL = 5e-6


def hr(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_original_module():
    if not ORIGINAL_V3H_SCRIPT.exists():
        raise FileNotFoundError(ORIGINAL_V3H_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "goal87_v3h_original_grand_volume",
        ORIGINAL_V3H_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible d'importer le script V3-H original.")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def original_defaults(mod):
    old = list(sys.argv)
    try:
        sys.argv = [str(ORIGINAL_V3H_SCRIPT)]
        args = mod.parse_args()
    finally:
        sys.argv = old
    return args


def resolve_inputs(mod, args):
    template_path = mod.resolve_explicit_or_default(
        ROOT,
        args.template,
        mod.DEFAULT_TEMPLATE_REL,
        "Template V3-C",
    )
    schema_path = mod.resolve_explicit_or_default(
        ROOT,
        args.schema,
        mod.DEFAULT_SCHEMA_REL,
        "Schéma V3-C",
    )
    dataset_path = mod.resolve_dataset(ROOT, args.dataset)
    return template_path, schema_path, dataset_path


def load_v3h_package():
    for p in V3H_MODEL_CANDIDATES:
        if p.exists():
            package = joblib.load(p)
            print(f"V3-H frozen : {p}")
            return p, package
    raise FileNotFoundError(
        "Modèle V3-H exact introuvable. Candidats:\n"
        + "\n".join(str(p) for p in V3H_MODEL_CANDIDATES)
    )


def prepare_historical_exact(mod, args, schema_path: Path, dataset_path: Path):
    features = mod.load_schema(schema_path)

    df, target_col, time_col, match_id_col, csv_audit = mod.load_dataset_robust(
        dataset_path,
        features,
        args.target,
        args.time_col,
    )

    y = mod.normalize_target(df[target_col])
    times = mod.parse_time_series(df[time_col])

    if times.isna().any():
        raise RuntimeError(
            f"{int(times.isna().sum())} dates historiques non parsées."
        )

    X = pd.DataFrame(index=df.index)
    for feat in features:
        X[feat] = pd.to_numeric(df[feat], errors="coerce")

    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    df2 = df.loc[finite].copy()
    X = X.loc[finite].copy()
    y = y.loc[finite].copy()
    times = times.loc[finite].copy()

    order = np.argsort(times.to_numpy())

    df2 = df2.iloc[order].reset_index(drop=True)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    times = times.iloc[order].reset_index(drop=True)

    if len(X) != 1710:
        raise RuntimeError(
            f"Historique complet={len(X)}, attendu=1710."
        )

    if df2[match_id_col].astype(str).duplicated().any():
        raise RuntimeError("match_id historiques dupliqués.")

    seen_tail_n = max(
        1,
        int(math.ceil(len(X) * float(args.seen_tail_fraction))),
    )
    dev_n = len(X) - seen_tail_n

    if dev_n != DEV_N or seen_tail_n != TAIL_N:
        raise RuntimeError(
            f"Split historique inattendu : DEV={dev_n}, TAIL={seen_tail_n}."
        )

    return {
        "features": list(features),
        "df": df2,
        "X": X,
        "y": y.reset_index(drop=True),
        "times": times,
        "match_id_col": match_id_col,
        "target_col": target_col,
        "time_col": time_col,
        "dev_n": dev_n,
        "tail_n": seen_tail_n,
        "csv_audit": csv_audit,
    }


def safe_roc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_pr(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def ece10(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    total = len(y)
    if total == 0:
        return float("nan")

    out = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & ((p <= hi) if i == 9 else (p < hi))
        n = int(mask.sum())
        if n:
            out += (n/total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def wilson(pos: int, n: int, z: float = 1.96):
    if n <= 0:
        return float("nan"), float("nan")

    p = pos / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = (
        z
        * math.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
        / denom
    )
    return center-margin, center+margin


def apply_saved_platt(package: Mapping[str, Any], raw_p: np.ndarray) -> np.ndarray:
    raw_p = np.clip(np.asarray(raw_p, dtype=float), EPS, 1-EPS)
    x = np.log(raw_p/(1-raw_p)).reshape(-1, 1)
    return package["platt_logistic_model"].predict_proba(x)[:, 1]


def score_with_frozen_final(
    df: pd.DataFrame,
    X: pd.DataFrame,
    package: Mapping[str, Any],
    sample_name: str,
    match_id_col: str,
) -> pd.DataFrame:
    v3c = joblib.load(V3C_MODEL_PATH)

    p_v3c = np.asarray(
        v3c.predict_proba(X),
        dtype=float,
    )[:, 1]

    p_v3h_raw = np.asarray(
        package["et_model"].predict_proba(X),
        dtype=float,
    )[:, 1]

    p_v3h = apply_saved_platt(
        package,
        p_v3h_raw,
    )

    out = pd.DataFrame({
        "sample": sample_name,
        "match_id": df[match_id_col].astype(str).to_numpy(),
        TARGET: pd.to_numeric(df[TARGET], errors="coerce").astype(int).to_numpy(),
        "p_v3c": p_v3c,
        "p_v3h_raw": p_v3h_raw,
        "p_v3h": p_v3h,
    })

    for c in [
        "competition_id",
        "season_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        "score_87_home",
        "score_87_away",
        "_v3i_selection_order",
    ]:
        if c in df.columns:
            out[c] = df[c].to_numpy()

    return out


def build_rolling_oos_exact(mod, args, hist):
    X = hist["X"]
    y = hist["y"]
    df = hist["df"]
    times = hist["times"]
    dev_n = hist["dev_n"]

    X_dev = X.iloc[:dev_n].reset_index(drop=True)
    y_dev = y.iloc[:dev_n].reset_index(drop=True)
    df_dev = df.iloc[:dev_n].reset_index(drop=True)
    t_dev = times.iloc[:dev_n].reset_index(drop=True)

    outer_splits = mod.rolling_splits(
        dev_n,
        int(args.rolling_folds),
        float(args.rolling_initial_train_fraction),
    )

    balanced_factory = mod.make_balanced_v3c_factory(
        joblib.load(V3C_MODEL_PATH),
        int(args.n_jobs),
    )

    et_factory = mod.make_et_leaf5_factory(
        int(args.random_state),
        int(args.n_jobs),
        int(args.et_n_estimators),
    )

    frames: List[pd.DataFrame] = []

    hr("REPRODUCTION ROLLING OOS 684 — FIDÉLITÉ")

    for fold_no, (train_idx, val_idx) in enumerate(outer_splits, start=1):
        Xtr = X_dev.iloc[train_idx].reset_index(drop=True)
        ytr = y_dev.iloc[train_idx].reset_index(drop=True)
        Xv = X_dev.iloc[val_idx]
        yv = y_dev.iloc[val_idx].to_numpy(dtype=int)

        v3c_fold = balanced_factory()
        v3c_fold.fit(Xtr, ytr)
        p_v3c = mod.predict_probability(v3c_fold, Xv)

        et = et_factory()
        et.fit(Xtr, ytr)
        p_raw = mod.predict_probability(et, Xv)

        raw_roc = safe_roc(yv, p_raw)
        expected = EXPECTED_ET_RAW_ROC[fold_no-1]

        if abs(raw_roc - expected) > ROC_TOL:
            raise RuntimeError(
                f"Fidélité V3-H F{fold_no} échouée : "
                f"attendu={expected:.6f}, obtenu={raw_roc:.6f}"
            )

        y_oof, p_oof, _ = mod.temporal_oof_probabilities(
            et_factory,
            Xtr,
            ytr,
            int(args.inner_calibration_folds),
            float(args.inner_calibration_initial_train_fraction),
        )

        platt = mod.fit_calibrator(
            "platt",
            p_oof,
            y_oof,
        )
        p_v3h = platt.transform(p_raw)

        block = pd.DataFrame({
            "sample": "ROLLING_OOS_684_EXACT",
            "match_id": (
                df_dev.iloc[val_idx][hist["match_id_col"]]
                .astype(str)
                .to_numpy()
            ),
            TARGET: yv,
            "p_v3c": p_v3c,
            "p_v3h_raw": p_raw,
            "p_v3h": p_v3h,
            "fold": fold_no,
            "kickoff_utc": t_dev.iloc[val_idx].astype(str).to_numpy(),
        })

        frames.append(block)

        print(
            f"F{fold_no} | train={len(train_idx)} | val={len(val_idx)} | "
            f"ET raw ROC={raw_roc:.6f} OK"
        )

    out = pd.concat(frames, ignore_index=True)

    if len(out) != ROLLING_OOS_N:
        raise RuntimeError(
            f"Rolling OOS={len(out)}, attendu={ROLLING_OOS_N}."
        )

    # Fidelity gate sur les règles déjà connues.
    print()
    print("CHECK DES COMPTAGES HISTORIQUES CONNUS")

    for rule, col, lo, hi, _tier in RULES:
        z = out[
            (out[col] >= lo)
            & (out[col] < hi)
        ]
        n = len(z)
        pos = int(z[TARGET].sum())

        expected = EXPECTED_OOS_COUNTS.get(rule)
        if expected is None:
            continue

        exp_n, exp_pos = expected

        print(
            f"{rule:<28} attendu={exp_pos}/{exp_n} | "
            f"obtenu={pos}/{n} | "
            f"{'OK' if (n == exp_n and pos == exp_pos) else 'FAIL'}"
        )

        if n != exp_n or pos != exp_pos:
            raise RuntimeError(
                f"Fidélité règle {rule} échouée."
            )

    return out


def load_v3i_external(features: Sequence[str]):
    if not V3I_PATH.exists():
        raise FileNotFoundError(V3I_PATH)

    df = pd.read_csv(V3I_PATH, low_memory=False)

    # Ce grand audit est figé sur le Batch 1 externe de 279.
    if len(df) < V3I_EXPECTED_N:
        raise RuntimeError(
            f"V3-I contient {len(df)} lignes, attendu au moins {V3I_EXPECTED_N}."
        )

    # Si la collecte Batch2 a déjà commencé, isoler les 279 du Batch1 par ordre.
    if len(df) > V3I_EXPECTED_N:
        if "_v3i_selection_order" not in df.columns:
            raise RuntimeError(
                "V3-I >279 mais _v3i_selection_order absent : "
                "impossible d'isoler proprement le Batch1."
            )
        work = df.copy()
        work["_v3i_selection_order"] = pd.to_numeric(
            work["_v3i_selection_order"],
            errors="coerce",
        )
        work = work.sort_values(
            ["_v3i_selection_order", MATCH_ID],
            kind="mergesort",
        )
        df = work.iloc[:V3I_EXPECTED_N].copy()

    if df[MATCH_ID].astype(str).duplicated().any():
        raise RuntimeError("match_id V3-I dupliqués.")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"Features absentes de V3-I : {missing}"
        )

    X = df[list(features)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise RuntimeError("V3-I Batch1 contient des lignes non 38/38.")

    return df.reset_index(drop=True), X.reset_index(drop=True)


def global_metrics(scored: pd.DataFrame, sample: str):
    y = scored[TARGET].astype(int).to_numpy()

    rows = []
    for model, col in [
        ("V3C", "p_v3c"),
        ("V3H_RAW", "p_v3h_raw"),
        ("V3H_PLATT", "p_v3h"),
    ]:
        p = scored[col].astype(float).to_numpy()

        rows.append({
            "sample": sample,
            "model": model,
            "n": len(y),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "roc_auc": safe_roc(y, p),
            "pr_auc": safe_pr(y, p),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(
                log_loss(
                    y,
                    np.clip(p, EPS, 1-EPS),
                    labels=[0, 1],
                )
            ),
            "ece_10": ece10(y, p),
        })

    return pd.DataFrame(rows)


def rule_row(
    scored: pd.DataFrame,
    sample: str,
    rule: str,
    col: str,
    lo: float,
    hi: float,
    tier: str,
):
    z = scored[
        (scored[col] >= lo)
        & (scored[col] < hi)
    ]

    n = len(z)
    pos = int(z[TARGET].sum()) if n else 0
    rate = pos/n if n else float("nan")
    ci_lo, ci_hi = wilson(pos, n)
    base_rate = float(scored[TARGET].mean())
    mean_p = float(z[col].mean()) if n else float("nan")

    return {
        "sample": sample,
        "evidence_class": EVIDENCE_CLASS.get(sample, ""),
        "tier": tier,
        "rule": rule,
        "sample_n": len(scored),
        "sample_positives": int(scored[TARGET].sum()),
        "sample_base_rate": base_rate,
        "selected_n": n,
        "selected_positives": pos,
        "selected_rate": rate,
        "lift_pp_vs_sample": (
            rate - base_rate
            if n
            else float("nan")
        ),
        "lift_ratio_vs_sample": (
            rate/base_rate
            if n and base_rate > 0
            else float("nan")
        ),
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "mean_probability": mean_p,
        "fair_odds_empirical": (
            1/rate
            if n and rate > 0
            else float("nan")
        ),
        "roi_at_3.20": (
            rate*3.20 - 1
            if n
            else float("nan")
        ),
    }


def rules_table(scored: pd.DataFrame, sample: str):
    return pd.DataFrame([
        rule_row(
            scored,
            sample,
            rule,
            col,
            lo,
            hi,
            tier,
        )
        for rule, col, lo, hi, tier in RULES
    ])


EVIDENCE_CLASS = {
    "HIST_DEV_1368_FINAL_MODEL": "DESCRIPTIVE_IN_SAMPLE_OR_REFIT",
    "HIST_SEEN_TAIL_342_FINAL_MODEL": "DESCRIPTIVE_SEEN_HISTORICAL",
    "HIST_ALL_1710_FINAL_MODEL": "DESCRIPTIVE_MIXED",
    "ROLLING_OOS_684_EXACT": "VALIDATION_ROLLING_OOS",
    "V3I_EXTERNAL_279": "VALIDATION_EXTERNAL",
    "OOS_PLUS_EXTERNAL_963": "VALIDATION_COMBINED",
    "ALL_UNIQUE_1989_FINAL_MODEL": "DESCRIPTIVE_GRAND_VOLUME",
}


def fmt_pct(x):
    try:
        x = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(x) else f"{x:.2%}"


def fmt_num(x, d=2):
    try:
        x = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(x) else f"{x:.{d}f}"


def make_report(
    rule_results: pd.DataFrame,
    metrics: pd.DataFrame,
    hist_start: str,
    hist_end: str,
    v3i_start: str,
    v3i_end: str,
):
    main = rule_results[
        rule_results["tier"] == "MAIN"
    ].copy()

    lines = [
        "# Goal After 87 — Grand Volume Rules Audit",
        "",
        f"Généré : `{utc_now()}`",
        "",
        "## Volumes",
        "",
        "- Historique complet : **1710 matchs**.",
        "- Rolling OOS exact : **684 matchs**.",
        "- V3-I externe : **279 matchs**.",
        "- OOS + externe : **963 matchs**.",
        "- Total unique disponible : **1989 matchs**.",
        "",
        f"- Historique : `{hist_start}` -> `{hist_end}`.",
        f"- V3-I : `{v3i_start}` -> `{v3i_end}`.",
        "",
        "## Règle d'interprétation",
        "",
        "Le bloc **OOS_PLUS_EXTERNAL_963** est le meilleur grand bloc de validation. "
        "Le bloc **ALL_UNIQUE_1989_FINAL_MODEL** sert à observer le comportement "
        "sur le maximum de volume, mais il contient de l'in-sample / historique déjà vu "
        "et ne doit pas être présenté comme une validation indépendante.",
        "",
        "## Règles principales",
        "",
        main.to_csv(index=False),
        "",
        "## Métriques modèles",
        "",
        metrics.to_csv(index=False),
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hr("GOAL AFTER 87 — GRAND VOLUME AUDIT")

    mod = load_original_module()
    args = original_defaults(mod)

    _template_path, schema_path, dataset_path = resolve_inputs(
        mod,
        args,
    )

    v3h_path, package = load_v3h_package()

    hist = prepare_historical_exact(
        mod,
        args,
        schema_path,
        dataset_path,
    )

    features = list(hist["features"])

    if list(package["features"]) != features:
        raise RuntimeError(
            "Le schéma du V3-H frozen ne correspond pas au schéma historique."
        )

    print(f"Historique : {len(hist['df'])} matchs")
    print(f"DEV        : {hist['dev_n']}")
    print(f"TAIL       : {hist['tail_n']}")
    print(f"V3-H       : {v3h_path}")

    # ----------------------------------------------------------------------
    # 1) Scoring final-model sur tout l'historique : descriptif.
    # ----------------------------------------------------------------------
    scored_hist_all = score_with_frozen_final(
        hist["df"],
        hist["X"],
        package,
        "HIST_ALL_1710_FINAL_MODEL",
        hist["match_id_col"],
    )

    scored_hist_dev = scored_hist_all.iloc[:DEV_N].copy()
    scored_hist_dev["sample"] = "HIST_DEV_1368_FINAL_MODEL"

    scored_hist_tail = scored_hist_all.iloc[DEV_N:].copy()
    scored_hist_tail["sample"] = "HIST_SEEN_TAIL_342_FINAL_MODEL"

    # ----------------------------------------------------------------------
    # 2) Rolling OOS exact.
    # ----------------------------------------------------------------------
    scored_oos = build_rolling_oos_exact(
        mod,
        args,
        hist,
    )

    # ----------------------------------------------------------------------
    # 3) V3-I externe 279.
    # ----------------------------------------------------------------------
    v3i_df, v3i_X = load_v3i_external(features)

    hist_ids = set(
        hist["df"][hist["match_id_col"]]
        .astype(str)
    )
    v3i_ids = set(v3i_df[MATCH_ID].astype(str))

    overlap = hist_ids.intersection(v3i_ids)
    if overlap:
        raise RuntimeError(
            f"Overlap historique/V3-I : {len(overlap)} match_id."
        )

    scored_v3i = score_with_frozen_final(
        v3i_df,
        v3i_X,
        package,
        "V3I_EXTERNAL_279",
        MATCH_ID,
    )

    if len(scored_v3i) != V3I_EXPECTED_N:
        raise RuntimeError("V3-I externe != 279.")

    # ----------------------------------------------------------------------
    # 4) Grand bloc propre = OOS + externe.
    # ----------------------------------------------------------------------
    scored_963 = pd.concat(
        [
            scored_oos.copy(),
            scored_v3i.copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    scored_963["sample"] = "OOS_PLUS_EXTERNAL_963"

    if len(scored_963) != 963:
        raise RuntimeError(
            f"OOS+external={len(scored_963)}, attendu=963."
        )

    if scored_963["match_id"].astype(str).duplicated().any():
        raise RuntimeError("Doublons dans OOS+external 963.")

    # ----------------------------------------------------------------------
    # 5) Tous les matchs uniques = 1710 + 279.
    #    Final-model scoring descriptif.
    # ----------------------------------------------------------------------
    scored_1989 = pd.concat(
        [
            scored_hist_all.copy(),
            scored_v3i.copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    scored_1989["sample"] = "ALL_UNIQUE_1989_FINAL_MODEL"

    if len(scored_1989) != 1989:
        raise RuntimeError(
            f"ALL UNIQUE={len(scored_1989)}, attendu=1989."
        )

    if scored_1989["match_id"].astype(str).duplicated().any():
        raise RuntimeError("Doublons dans ALL UNIQUE 1989.")

    samples = {
        "HIST_DEV_1368_FINAL_MODEL": scored_hist_dev,
        "HIST_SEEN_TAIL_342_FINAL_MODEL": scored_hist_tail,
        "HIST_ALL_1710_FINAL_MODEL": scored_hist_all,
        "ROLLING_OOS_684_EXACT": scored_oos,
        "V3I_EXTERNAL_279": scored_v3i,
        "OOS_PLUS_EXTERNAL_963": scored_963,
        "ALL_UNIQUE_1989_FINAL_MODEL": scored_1989,
    }

    # ----------------------------------------------------------------------
    # Tables.
    # ----------------------------------------------------------------------
    rule_results = pd.concat(
        [
            rules_table(df, sample)
            for sample, df in samples.items()
        ],
        ignore_index=True,
    )

    metrics = pd.concat(
        [
            global_metrics(df, sample)
            for sample, df in samples.items()
        ],
        ignore_index=True,
    )

    # Comparaison centrée sur règles principales.
    main_rules = rule_results[
        rule_results["tier"] == "MAIN"
    ].copy()

    # ----------------------------------------------------------------------
    # Sauvegardes.
    # ----------------------------------------------------------------------
    scored_oos.to_csv(
        OUT_DIR / "rolling_oos_684_exact_predictions.csv",
        index=False,
    )
    scored_v3i.to_csv(
        OUT_DIR / "v3i_external_279_predictions.csv",
        index=False,
    )
    scored_hist_all.to_csv(
        OUT_DIR / "historical_1710_final_model_predictions.csv",
        index=False,
    )
    scored_963.to_csv(
        OUT_DIR / "oos_plus_external_963_predictions.csv",
        index=False,
    )
    scored_1989.to_csv(
        OUT_DIR / "all_unique_1989_final_model_predictions.csv",
        index=False,
    )

    rule_results.to_csv(
        OUT_DIR / "rule_rates_all_samples.csv",
        index=False,
    )
    main_rules.to_csv(
        OUT_DIR / "main_rules_summary.csv",
        index=False,
    )
    metrics.to_csv(
        OUT_DIR / "model_metrics_all_samples.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # Terminal output.
    # ----------------------------------------------------------------------
    hr("VOLUMES")
    print("Historique complet        : 1710")
    print("Rolling OOS exact         : 684")
    print("V3-I externe              : 279")
    print("OOS + externe             : 963")
    print("Total unique disponible   : 1989")

    hr("RÈGLES PRINCIPALES — MEILLEUR BLOC DE VALIDATION (963)")
    show = main_rules[
        main_rules["sample"] == "OOS_PLUS_EXTERNAL_963"
    ].copy()

    for c in [
        "sample_base_rate",
        "selected_rate",
        "lift_pp_vs_sample",
        "ci95_low",
        "ci95_high",
        "mean_probability",
        "roi_at_3.20",
    ]:
        show[c] = show[c].map(fmt_pct)

    show["fair_odds_empirical"] = show[
        "fair_odds_empirical"
    ].map(lambda x: fmt_num(x, 2))

    print(
        show[
            [
                "rule",
                "sample_n",
                "selected_n",
                "selected_positives",
                "selected_rate",
                "sample_base_rate",
                "lift_pp_vs_sample",
                "ci95_low",
                "ci95_high",
                "fair_odds_empirical",
                "roi_at_3.20",
            ]
        ].to_string(index=False)
    )

    hr("RÈGLES PRINCIPALES — GRAND VOLUME DESCRIPTIF 1989")
    show2 = main_rules[
        main_rules["sample"] == "ALL_UNIQUE_1989_FINAL_MODEL"
    ].copy()

    for c in [
        "sample_base_rate",
        "selected_rate",
        "lift_pp_vs_sample",
        "ci95_low",
        "ci95_high",
        "mean_probability",
        "roi_at_3.20",
    ]:
        show2[c] = show2[c].map(fmt_pct)

    show2["fair_odds_empirical"] = show2[
        "fair_odds_empirical"
    ].map(lambda x: fmt_num(x, 2))

    print(
        show2[
            [
                "rule",
                "sample_n",
                "selected_n",
                "selected_positives",
                "selected_rate",
                "sample_base_rate",
                "lift_pp_vs_sample",
                "ci95_low",
                "ci95_high",
                "fair_odds_empirical",
                "roi_at_3.20",
            ]
        ].to_string(index=False)
    )

    hr("STABILITÉ — OOS 684 vs V3-I 279 vs COMBINÉ 963")
    stable = main_rules[
        main_rules["sample"].isin(
            [
                "ROLLING_OOS_684_EXACT",
                "V3I_EXTERNAL_279",
                "OOS_PLUS_EXTERNAL_963",
            ]
        )
    ].copy()

    for c in [
        "selected_rate",
        "sample_base_rate",
        "lift_pp_vs_sample",
        "ci95_low",
        "ci95_high",
    ]:
        stable[c] = stable[c].map(fmt_pct)

    print(
        stable[
            [
                "sample",
                "rule",
                "selected_n",
                "selected_positives",
                "selected_rate",
                "sample_base_rate",
                "lift_pp_vs_sample",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    report = make_report(
        rule_results,
        metrics,
        str(hist["times"].iloc[0]),
        str(hist["times"].iloc[-1]),
        (
            str(v3i_df["kickoff_utc"].min())
            if "kickoff_utc" in v3i_df.columns
            else ""
        ),
        (
            str(v3i_df["kickoff_utc"].max())
            if "kickoff_utc" in v3i_df.columns
            else ""
        ),
    )
    (OUT_DIR / "grand_volume_rules_report.md").write_text(
        report,
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": utc_now(),
        "volumes": {
            "historical_complete": 1710,
            "rolling_oos": 684,
            "v3i_external": 279,
            "oos_plus_external": 963,
            "all_unique": 1989,
        },
        "main_rules": [
            "V3C_MAIN_55_65",
            "V3H_MAIN_31_34",
            "V3H_STRONG_32_34",
        ],
        "methodology": {
            "best_validation_block": "OOS_PLUS_EXTERNAL_963",
            "max_volume_descriptive_block": "ALL_UNIQUE_1989_FINAL_MODEL",
            "warning": (
                "ALL_UNIQUE_1989 contains historical training/seen data. "
                "Use it descriptively, not as independent validation."
            ),
        },
        "v3h_frozen_model": str(v3h_path),
        "v3c_model": str(V3C_MODEL_PATH),
    }

    (OUT_DIR / "grand_volume_audit_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    hr("SORTIES")
    print(OUT_DIR)
    print()
    print("Fichiers principaux :")
    print("  main_rules_summary.csv")
    print("  rule_rates_all_samples.csv")
    print("  model_metrics_all_samples.csv")
    print("  oos_plus_external_963_predictions.csv")
    print("  all_unique_1989_final_model_predictions.csv")
    print("  grand_volume_rules_report.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
