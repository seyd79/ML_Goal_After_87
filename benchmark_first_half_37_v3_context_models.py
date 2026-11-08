#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 CONTEXT MODEL BENCHMARK
==========================================

But:
Tester si une architecture PAR CONTEXTE DE SCORE à 37' est plus robuste que
le V2 global, sans prétendre à une nouvelle validation externe.

Données autorisées:
  - 2 105 matchs DEVELOPMENT V2 déjà consommés
  - 594 matchs FINAL HOLDOUT V2 désormais consommés après FINAL_FAIL

Total V3 DEVELOPMENT potentiel: 2 699 matchs.

IMPORTANT:
  - Les 594 matchs FINAL HOLDOUT sont désormais du DEVELOPMENT / RESEARCH.
  - Ils servent ici de LATE_STRESS consommé.
  - Aucun nouveau match futur n'est lu.
  - Aucun pipeline produit ici n'est "validé production".
  - Toute future V3 devra être gelée puis évaluée sur un AUTRE bloc futur neuf.

Architectures comparées:
  1) GLOBAL_TOP8
  2) GLOBAL_TOP19
  3) GLOBAL_CONTEXT_TOP8
  4) GLOBAL_INTERACTIONS_TOP8
  5) SEPARATE4_TOP8
  6) SEPARATE4_TOP19

Contextes:
  - 0-0
  - home_lead
  - away_lead
  - draw_with_goals

Validation DEVELOPMENT:
  - 5 folds TimeSeriesSplit sur les 2 105 matchs pré-holdout
  - LATE_STRESS: TRAIN=2 105, VAL=594 (holdout V2 consommé)
  - C logistic choisi uniquement dans le TRAIN de chaque fold, via mini-CV temporelle
  - aucune calibration additionnelle à cette étape

Sorties:
  outputs/first_half_37/v3_context_benchmark/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent

DEV_V2_PATH = BASE_DIR / "outputs" / "first_half_37" / "v2_robust_selection" / "First_Half_37_V2_DEVELOPMENT.csv"
RANKING_V2_PATH = BASE_DIR / "outputs" / "first_half_37" / "v2_robust_selection" / "first_half_37_v2_robust_feature_ranking.csv"
CONSUMED_HOLDOUT_PATH = BASE_DIR / "outputs" / "first_half_37" / "v2_final_holdout" / "First_Half_37_V2_FINAL_HOLDOUT.csv"
EVAL_LOCK_PATH = BASE_DIR / "outputs" / "first_half_37" / "v2_final_holdout" / "evaluation" / "FIRST_HALF37_V2_HOLDOUT_EVALUATION_LOCK.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "first_half_37" / "v3_context_benchmark"

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"

EXPECTED_DEV_ROWS = 2105
EXPECTED_HOLDOUT_ROWS = 594
EXPECTED_HOLDOUT_VERDICT = "FINAL_FAIL"

TOP8 = [
    "home_over25_freq",
    "shots_35_37",
    "away_shots_35_37",
    "away_xg_37",
    "subs_35_37",
    "home_corners_30_37",
    "total_xg_37",
    "score_37_away",
]

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
MIN_CONTEXT_TRAIN = 40
MIN_CONTEXT_VAL_FOR_METRICS = 15

CONTEXT_ORDER = ["0-0", "home_lead", "away_lead", "draw_with_goals"]
CONTEXT_METRIC_ORDER = [*CONTEXT_ORDER, "missing"]
CONTEXT_DUMMIES = ["ctx_home_lead", "ctx_away_lead", "ctx_draw_with_goals", "ctx_missing"]


@dataclass
class FoldDef:
    name: str
    kind: str
    train_idx: np.ndarray
    val_idx: np.ndarray


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


def load_inputs(dev_path: Path, holdout_path: Path, ranking_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    for p in (dev_path, holdout_path, ranking_path, EVAL_LOCK_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    lock = load_json(EVAL_LOCK_PATH)
    if lock.get("status") != "FINAL_HOLDOUT_CONSUMED":
        raise RuntimeError("Le holdout V2 n'est pas marqué FINAL_HOLDOUT_CONSUMED.")
    if lock.get("verdict") != EXPECTED_HOLDOUT_VERDICT:
        raise RuntimeError(
            f"Verdict holdout inattendu: {lock.get('verdict')} "
            f"(attendu {EXPECTED_HOLDOUT_VERDICT})."
        )

    dev = pd.read_csv(dev_path, low_memory=False)
    hold = pd.read_csv(holdout_path, low_memory=False)

    if len(dev) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"DEVELOPMENT V2={len(dev)}, attendu={EXPECTED_DEV_ROWS}.")
    if len(hold) != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError(f"Holdout consommé={len(hold)}, attendu={EXPECTED_HOLDOUT_ROWS}.")

    required_base = {TARGET, DATE_COL, ID_COL, "score_37_home", "score_37_away"}

    for name, df in (("DEV", dev), ("LATE_STRESS", hold)):
        missing = sorted(required_base - set(df.columns))
        if missing:
            raise RuntimeError(f"{name}: colonnes requises absentes: {missing}")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
        if df[DATE_COL].isna().any():
            raise RuntimeError(f"{name}: kickoff_utc non parsable.")

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        if df[TARGET].isna().any():
            raise RuntimeError(f"{name}: target non résolu.")
        df[TARGET] = df[TARGET].astype(int)

    overlap = set(dev[ID_COL].astype(str)) & set(hold[ID_COL].astype(str))
    if overlap:
        raise RuntimeError(f"Overlap DEV/HOLDOUT consommé: {len(overlap)} matchs.")

    ranking = pd.read_csv(ranking_path)
    needed = {"feature", "selected_v2", "robust_selection_score"}
    if not needed.issubset(ranking.columns):
        raise RuntimeError(f"Ranking V2 incomplet: {sorted(needed - set(ranking.columns))}")

    selected = ranking[ranking["selected_v2"] == 1].copy()
    selected = selected.sort_values("robust_selection_score", ascending=False)
    top19 = selected["feature"].astype(str).tolist()

    if len(top19) != 19:
        raise RuntimeError(f"Attendu 19 features robustes V2, trouvé {len(top19)}.")
    if top19[:8] != TOP8:
        raise RuntimeError("Le TOP_8 du ranking ne correspond plus au TOP_8 consommé.")

    missing_features = [f for f in top19 if f not in dev.columns or f not in hold.columns]
    if missing_features:
        raise RuntimeError(f"Features robustes absentes d'un bloc: {missing_features}")

    dev = dev.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    hold = hold.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    if dev[DATE_COL].max() >= hold[DATE_COL].min():
        raise RuntimeError("Ordre temporel inattendu: le bloc 594 n'est pas strictement après DEV.")

    return dev, hold, top19


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le contexte de score à 37' sans fabriquer de score.

    Politique:
      1) utilise score_37_home / score_37_away si présents;
      2) tente uniquement des alias explicitement "37";
      3) tente un score composite contenant "37" (ex. "1-0");
      4) si le score reste incomplet, contexte="missing" et les modèles
         SEPARATE4 utilisent le fallback global pour ces lignes.

    Aucune ligne n'est silencieusement supprimée.
    """
    out = df.copy()

    def num_col(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce")

    h = num_col("score_37_home")
    a = num_col("score_37_away")

    # Alias strictement liés à la minute 37.
    home_aliases = [
        "home_score_37",
        "home_goals_37",
        "score_home_37",
        "home_score_at_37",
    ]
    away_aliases = [
        "away_score_37",
        "away_goals_37",
        "score_away_37",
        "away_score_at_37",
    ]

    for col in home_aliases:
        if col in out.columns:
            candidate = pd.to_numeric(out[col], errors="coerce")
            h = h.where(h.notna(), candidate)

    for col in away_aliases:
        if col in out.columns:
            candidate = pd.to_numeric(out[col], errors="coerce")
            a = a.where(a.notna(), candidate)

    # Scores composites dont le nom indique explicitement la minute 37.
    composite_candidates = [
        c for c in out.columns
        if "score" in c.lower() and "37" in c.lower()
        and c not in {
            "score_37_home", "score_37_away",
            *home_aliases, *away_aliases,
        }
    ]

    need = h.isna() | a.isna()
    if need.any():
        score_re = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")
        for col in composite_candidates:
            if not need.any():
                break

            parsed_h = pd.Series(np.nan, index=out.index, dtype=float)
            parsed_a = pd.Series(np.nan, index=out.index, dtype=float)

            for idx, value in out.loc[need, col].items():
                if pd.isna(value):
                    continue
                m = score_re.match(str(value))
                if m:
                    parsed_h.loc[idx] = float(m.group(1))
                    parsed_a.loc[idx] = float(m.group(2))

            h = h.where(h.notna(), parsed_h)
            a = a.where(a.notna(), parsed_a)
            need = h.isna() | a.isna()

    # Ecrit les versions numériques récupérées pour les candidats V3.
    out["score_37_home"] = h
    out["score_37_away"] = a

    complete = h.notna() & a.notna()

    states = pd.Series("missing", index=out.index, dtype=object)
    states.loc[complete & (h == 0) & (a == 0)] = "0-0"
    states.loc[complete & (h > a)] = "home_lead"
    states.loc[complete & (a > h)] = "away_lead"
    states.loc[complete & (h == a) & (h > 0)] = "draw_with_goals"

    invalid = complete & ~states.isin(CONTEXT_ORDER)
    if invalid.any():
        bad = out.loc[invalid, [ID_COL, "score_37_home", "score_37_away"]].head(10)
        raise RuntimeError(f"Contexte de score invalide détecté:\n{bad}")

    out["score_context_37"] = states

    out["goal_diff_37_ctx"] = h - a
    out["total_goals_37_ctx"] = h + a
    out["is_draw_37_ctx"] = np.where(
        complete,
        (h == a).astype(float),
        np.nan,
    )

    out["ctx_home_lead"] = (states == "home_lead").astype(int)
    out["ctx_away_lead"] = (states == "away_lead").astype(int)
    out["ctx_draw_with_goals"] = (states == "draw_with_goals").astype(int)
    out["ctx_missing"] = (states == "missing").astype(int)

    return out


def add_interactions(df: pd.DataFrame, base_features: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    interaction_base = [f for f in base_features if f != "score_37_away"]

    interaction_features = []
    for f in interaction_base:
        x = pd.to_numeric(out[f], errors="coerce")
        for d in CONTEXT_DUMMIES:
            name = f"{f}__x__{d}"
            out[name] = x * pd.to_numeric(out[d], errors="coerce")
            interaction_features.append(name)

    return out, interaction_features


def build_model(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            class_weight=None,
            max_iter=5000,
            random_state=87,
        )),
    ])


def tune_c_temporal(train_df: pd.DataFrame, features: Sequence[str], c_grid: Sequence[float] = C_GRID) -> float:
    train_df = train_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    if len(train_df) < 120:
        return 0.03

    splitter = TimeSeriesSplit(n_splits=3)
    X = train_df[list(features)].apply(pd.to_numeric, errors="coerce")
    y = train_df[TARGET].astype(int).to_numpy()

    rows = []
    for C in c_grid:
        lifts = []
        rocs = []

        for tr, va in splitter.split(X):
            ytr = y[tr]
            yva = y[va]
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue

            model = build_model(C)
            model.fit(X.iloc[tr], ytr)
            p = model.predict_proba(X.iloc[va])[:, 1]

            pr = safe_pr(yva, p)
            roc = safe_roc(yva, p)
            base = float(yva.mean())

            if np.isfinite(pr):
                lifts.append(pr - base)
            if np.isfinite(roc):
                rocs.append(roc)

        if lifts:
            rows.append({
                "C": C,
                "mean_pr_lift": float(np.mean(lifts)),
                "mean_roc": float(np.mean(rocs)) if rocs else -np.inf,
                "positive_lift_fraction": float(np.mean(np.asarray(lifts) > 0)),
            })

    if not rows:
        return 0.03

    rank = pd.DataFrame(rows).sort_values(
        ["positive_lift_fraction", "mean_pr_lift", "mean_roc", "C"],
        ascending=[False, False, False, True],
    )
    return float(rank.iloc[0]["C"])


def make_folds(dev: pd.DataFrame, hold: pd.DataFrame) -> List[FoldDef]:
    folds = []
    splitter = TimeSeriesSplit(n_splits=5)
    local = np.arange(len(dev))

    for i, (tr, va) in enumerate(splitter.split(local), start=1):
        folds.append(FoldDef(f"INTERNAL_{i}", "internal", tr, va))

    folds.append(FoldDef(
        "LATE_STRESS_594",
        "late_stress",
        np.arange(len(dev)),
        np.arange(len(hold)),
    ))
    return folds


def candidate_definitions(top19: Sequence[str], interaction_features: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    context_numeric = [
        "score_37_home",
        "goal_diff_37_ctx",
        "total_goals_37_ctx",
        "is_draw_37_ctx",
        *CONTEXT_DUMMIES,
    ]

    global_context_features = list(dict.fromkeys([*TOP8, *context_numeric]))
    interaction_set = list(dict.fromkeys([*global_context_features, *interaction_features]))

    return {
        "GLOBAL_TOP8": {
            "kind": "global",
            "features": TOP8,
        },
        "GLOBAL_TOP19": {
            "kind": "global",
            "features": list(top19),
        },
        "GLOBAL_CONTEXT_TOP8": {
            "kind": "global",
            "features": global_context_features,
        },
        "GLOBAL_INTERACTIONS_TOP8": {
            "kind": "global",
            "features": interaction_set,
        },
        "SEPARATE4_TOP8": {
            "kind": "separate4",
            "features": list(dict.fromkeys([
                *TOP8,
                "score_37_home",
                "goal_diff_37_ctx",
                "total_goals_37_ctx",
            ])),
        },
        "SEPARATE4_TOP19": {
            "kind": "separate4",
            "features": list(dict.fromkeys([
                *top19,
                "score_37_home",
                "goal_diff_37_ctx",
                "total_goals_37_ctx",
            ])),
        },
    }


def metric_row(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    prevalence = float(y.mean())
    base_p = np.full(len(y), prevalence, dtype=float)

    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, base_p))

    ll = float(log_loss(y, np.clip(p, 1e-10, 1-1e-10), labels=[0, 1]))
    base_ll = float(log_loss(y, np.clip(base_p, 1e-10, 1-1e-10), labels=[0, 1]))

    pr = safe_pr(y, p)

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prevalence,
        "mean_probability": float(p.mean()),
        "roc_auc": safe_roc(y, p),
        "pr_auc": pr,
        "pr_lift_vs_val_prevalence": pr - prevalence,
        "brier": brier,
        "brier_improvement_vs_val_prevalence": base_brier - brier,
        "log_loss": ll,
        "logloss_improvement_vs_val_prevalence": base_ll - ll,
    }


def predict_global(train_df: pd.DataFrame, val_df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    C = tune_c_temporal(train_df, features)

    model = build_model(C)
    Xtr = train_df[list(features)].apply(pd.to_numeric, errors="coerce")
    Xva = val_df[list(features)].apply(pd.to_numeric, errors="coerce")
    ytr = train_df[TARGET].astype(int).to_numpy()

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]

    return p, {
        "selected_C": C,
        "fallback_rows": 0,
        "context_C": {},
    }


def predict_separate4(train_df: pd.DataFrame, val_df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = np.full(len(val_df), np.nan, dtype=float)

    global_C = tune_c_temporal(train_df, features)
    global_model = build_model(global_C)
    global_model.fit(
        train_df[list(features)].apply(pd.to_numeric, errors="coerce"),
        train_df[TARGET].astype(int).to_numpy(),
    )

    fallback_rows = 0
    context_C: Dict[str, float] = {}

    for context in CONTEXT_ORDER:
        tr = train_df[train_df["score_context_37"] == context].copy()
        mask = val_df["score_context_37"].eq(context).to_numpy()
        n_val = int(mask.sum())

        if n_val == 0:
            continue

        Xva = val_df.loc[mask, list(features)].apply(pd.to_numeric, errors="coerce")

        if len(tr) < MIN_CONTEXT_TRAIN or tr[TARGET].nunique() < 2:
            p[mask] = global_model.predict_proba(Xva)[:, 1]
            fallback_rows += n_val
            context_C[context] = float("nan")
            continue

        C = tune_c_temporal(tr, features)
        context_C[context] = C

        model = build_model(C)
        model.fit(
            tr[list(features)].apply(pd.to_numeric, errors="coerce"),
            tr[TARGET].astype(int).to_numpy(),
        )
        p[mask] = model.predict_proba(Xva)[:, 1]

    # Les lignes dont le score à 37' reste réellement manquant ne sont pas
    # inventées. Elles utilisent le fallback global et sont comptabilisées.
    unresolved = np.isnan(p)
    if unresolved.any():
        Xva_missing = val_df.loc[unresolved, list(features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        p[unresolved] = global_model.predict_proba(Xva_missing)[:, 1]
        fallback_rows += int(unresolved.sum())

    if np.isnan(p).any():
        raise RuntimeError("SEPARATE4: certaines lignes validation non scorées après fallback.")

    return p, {
        "selected_C": global_C,
        "fallback_rows": fallback_rows,
        "context_C": context_C,
    }


def evaluate_candidate(name: str, spec: Mapping[str, Any], dev: pd.DataFrame, hold: pd.DataFrame, folds: Sequence[FoldDef]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    state_rows = []
    pred_frames = []

    for fold in folds:
        if fold.kind == "late_stress":
            train_df = dev.iloc[fold.train_idx].copy()
            val_df = hold.iloc[fold.val_idx].copy()
        else:
            train_df = dev.iloc[fold.train_idx].copy()
            val_df = dev.iloc[fold.val_idx].copy()

        features = list(spec["features"])

        if spec["kind"] == "global":
            p, meta = predict_global(train_df, val_df, features)
        elif spec["kind"] == "separate4":
            p, meta = predict_separate4(train_df, val_df, features)
        else:
            raise ValueError(spec["kind"])

        y = val_df[TARGET].astype(int).to_numpy()
        m = metric_row(y, p)

        fold_rows.append({
            "model": name,
            "model_kind": spec["kind"],
            "n_features": len(features),
            "fold": fold.name,
            "kind": fold.kind,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "selected_C_global_or_fallback": meta["selected_C"],
            "fallback_rows": meta["fallback_rows"],
            "fallback_fraction": meta["fallback_rows"] / len(val_df) if len(val_df) else 0.0,
            "context_C_json": json.dumps(meta["context_C"], ensure_ascii=False),
            **m,
        })

        pred = val_df[[ID_COL, DATE_COL, TARGET, "score_context_37"]].copy()
        pred["model"] = name
        pred["fold"] = fold.name
        pred["kind"] = fold.kind
        pred["probability"] = p
        pred_frames.append(pred)

        for context in CONTEXT_METRIC_ORDER:
            mask = val_df["score_context_37"].eq(context).to_numpy()
            n = int(mask.sum())
            if n < MIN_CONTEXT_VAL_FOR_METRICS:
                continue

            sm = metric_row(y[mask], p[mask])
            state_rows.append({
                "model": name,
                "fold": fold.name,
                "kind": fold.kind,
                "score_context_37": context,
                **sm,
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(state_rows), pd.concat(pred_frames, ignore_index=True)


def summarize_models(fold_metrics: pd.DataFrame, state_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    baseline_stress = fold_metrics[
        (fold_metrics["model"] == "GLOBAL_TOP8")
        & (fold_metrics["kind"] == "late_stress")
    ].iloc[0]

    for model, block in fold_metrics.groupby("model", sort=False):
        internal = block[block["kind"] == "internal"].copy()
        stress = block[block["kind"] == "late_stress"].copy()

        if len(stress) != 1:
            raise RuntimeError(f"Stress fold absent pour {model}")

        s = stress.iloc[0]

        stress_states = state_metrics[
            (state_metrics["model"] == model)
            & (state_metrics["kind"] == "late_stress")
            & (state_metrics["score_context_37"] != "missing")
        ].copy()

        min_state_roc = float(stress_states["roc_auc"].min()) if not stress_states.empty else np.nan
        states_above_050 = float((stress_states["roc_auc"] > 0.5).mean()) if not stress_states.empty else np.nan

        mean_internal_roc = float(internal["roc_auc"].mean())
        mean_internal_pr_lift = float(internal["pr_lift_vs_val_prevalence"].mean())
        pr_positive_fraction = float((internal["pr_lift_vs_val_prevalence"] > 0).mean())
        roc_positive_fraction = float((internal["roc_auc"] > 0.5).mean())

        gate = int(
            mean_internal_roc > 0.53
            and mean_internal_pr_lift > 0
            and pr_positive_fraction >= 0.60
            and roc_positive_fraction >= 0.60
            and float(s["roc_auc"]) > 0.53
            and float(s["pr_lift_vs_val_prevalence"]) > 0.02
        )

        rows.append({
            "model": model,
            "model_kind": block.iloc[0]["model_kind"],
            "n_features": int(block.iloc[0]["n_features"]),
            "passes_v3_context_research_gate": gate,
            "mean_internal_roc_auc": mean_internal_roc,
            "mean_internal_pr_auc": float(internal["pr_auc"].mean()),
            "mean_internal_pr_lift": mean_internal_pr_lift,
            "internal_pr_lift_positive_fraction": pr_positive_fraction,
            "internal_roc_above_050_fraction": roc_positive_fraction,
            "late_stress_roc_auc": float(s["roc_auc"]),
            "late_stress_pr_auc": float(s["pr_auc"]),
            "late_stress_positive_rate": float(s["positive_rate"]),
            "late_stress_pr_lift": float(s["pr_lift_vs_val_prevalence"]),
            "late_stress_brier_improvement": float(s["brier_improvement_vs_val_prevalence"]),
            "late_stress_logloss_improvement": float(s["logloss_improvement_vs_val_prevalence"]),
            "late_stress_fallback_fraction": float(s["fallback_fraction"]),
            "late_stress_min_state_roc": min_state_roc,
            "late_stress_states_roc_above_050_fraction": states_above_050,
            "delta_stress_roc_vs_global_top8": float(s["roc_auc"]) - float(baseline_stress["roc_auc"]),
            "delta_stress_pr_lift_vs_global_top8": float(s["pr_lift_vs_val_prevalence"]) - float(baseline_stress["pr_lift_vs_val_prevalence"]),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "passes_v3_context_research_gate",
            "late_stress_roc_auc",
            "late_stress_pr_lift",
            "mean_internal_roc_auc",
            "mean_internal_pr_lift",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def write_report(path: Path, dev: pd.DataFrame, hold: pd.DataFrame, candidates: Mapping[str, Mapping[str, Any]], ranking: pd.DataFrame, state_metrics: pd.DataFrame) -> None:
    lines = []

    lines.append("# First Half 37 — V3 Context Model Benchmark")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- Les 594 matchs FINAL_FAIL V2 sont désormais **consommés**.")
    lines.append("- Ils servent uniquement de `LATE_STRESS_594` de DEVELOPMENT V3.")
    lines.append("- Aucun nouveau holdout futur n'est lu.")
    lines.append("- Aucun candidat de ce rapport n'est validé production.")
    lines.append("")
    lines.append("## Données")
    lines.append("")
    lines.append(
        f"- DEV pré-holdout : **{len(dev)}** matchs | "
        f"{int(dev[TARGET].sum())} positifs ({dev[TARGET].mean():.2%})"
    )
    lines.append(
        f"- LATE_STRESS consommé : **{len(hold)}** matchs | "
        f"{int(hold[TARGET].sum())} positifs ({hold[TARGET].mean():.2%})"
    )
    lines.append(
        f"- V3 DEVELOPMENT potentiel : **{len(dev)+len(hold)}** matchs"
    )
    lines.append("")
    lines.append("## Contextes")
    lines.append("")
    for context in CONTEXT_METRIC_ORDER:
        nd = int((dev["score_context_37"] == context).sum())
        nh = int((hold["score_context_37"] == context).sum())
        lines.append(f"- `{context}` : DEV={nd} | LATE_STRESS={nh}")
    lines.append("")
    lines.append("## Candidats")
    lines.append("")
    for name, spec in candidates.items():
        lines.append(f"- `{name}` — `{spec['kind']}` — {len(spec['features'])} features")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    show = [
        "rank",
        "model",
        "passes_v3_context_research_gate",
        "mean_internal_roc_auc",
        "mean_internal_pr_lift",
        "late_stress_roc_auc",
        "late_stress_pr_lift",
        "late_stress_brier_improvement",
        "late_stress_logloss_improvement",
        "late_stress_min_state_roc",
        "delta_stress_roc_vs_global_top8",
        "delta_stress_pr_lift_vs_global_top8",
    ]
    lines.append("```")
    lines.append(ranking[show].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## LATE_STRESS par état du score")
    lines.append("")
    stress_states = state_metrics[state_metrics["kind"] == "late_stress"].copy()
    show_state = [
        "model",
        "score_context_37",
        "n",
        "positive_rate",
        "roc_auc",
        "pr_auc",
        "pr_lift_vs_val_prevalence",
        "brier_improvement_vs_val_prevalence",
        "logloss_improvement_vs_val_prevalence",
    ]
    lines.append("```")
    lines.append(
        stress_states[show_state]
        .sort_values(["model", "score_context_37"])
        .to_string(index=False)
    )
    lines.append("```")
    lines.append("")
    lines.append("## Suite")
    lines.append("")
    lines.append(
        "Si une architecture contextuelle améliore clairement GLOBAL_TOP8 "
        "sur les folds internes ET sur LATE_STRESS_594, la prochaine étape "
        "sera une sélection de features spécifique à chaque contexte."
    )
    lines.append(
        "Le futur holdout final V3 devra être un bloc entièrement neuf."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark V3 par contexte de score à 37'.")
    parser.add_argument("--dev-v2", default=str(DEV_V2_PATH))
    parser.add_argument("--consumed-holdout", default=str(CONSUMED_HOLDOUT_PATH))
    parser.add_argument("--ranking-v2", default=str(RANKING_V2_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    dev_path = Path(args.dev_v2)
    holdout_path = Path(args.consumed_holdout)
    ranking_path = Path(args.ranking_v2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 CONTEXT MODEL BENCHMARK")
    print("=" * 120)
    print("RESEARCH ONLY — 594 FINAL_FAIL déjà consommés")
    print("AUCUN NOUVEAU HOLDOUT FUTUR")
    print()

    dev, hold, top19 = load_inputs(dev_path, holdout_path, ranking_path)

    dev = add_context_columns(dev)
    hold = add_context_columns(hold)

    dev_missing_ctx = int((dev["score_context_37"] == "missing").sum())
    hold_missing_ctx = int((hold["score_context_37"] == "missing").sum())

    print(
        f"Scores à 37' incomplets après récupération | "
        f"DEV={dev_missing_ctx}/{len(dev)} | "
        f"LATE_STRESS={hold_missing_ctx}/{len(hold)}"
    )
    if dev_missing_ctx or hold_missing_ctx:
        print(
            "POLITIQUE : aucune ligne supprimée; contexte='missing'. "
            "SEPARATE4 utilisera le fallback global pour ces lignes."
        )
    print()

    dev, interaction_features = add_interactions(dev, TOP8)
    hold, _ = add_interactions(hold, TOP8)

    candidates = candidate_definitions(top19, interaction_features)

    all_feature_names = set()
    for spec in candidates.values():
        all_feature_names.update(spec["features"])

    missing = sorted(f for f in all_feature_names if f not in dev.columns or f not in hold.columns)
    if missing:
        raise RuntimeError(f"Features candidat absentes: {missing}")

    print(
        f"DEV pré-holdout : {len(dev)} | "
        f"positifs={int(dev[TARGET].sum())} ({dev[TARGET].mean():.2%})"
    )
    print(
        f"LATE_STRESS     : {len(hold)} | "
        f"positifs={int(hold[TARGET].sum())} ({hold[TARGET].mean():.2%})"
    )
    print(f"V3 DEVELOPMENT : {len(dev)+len(hold)} matchs désormais consommés")
    print()

    print("Répartition des contextes:")
    for context in CONTEXT_METRIC_ORDER:
        print(
            f"  {context:<18} "
            f"DEV={int((dev['score_context_37'] == context).sum()):4d} | "
            f"LATE_STRESS={int((hold['score_context_37'] == context).sum()):3d}"
        )
    print()

    dev_out = dev.copy()
    dev_out["_v3_source"] = "v2_development_consumed"

    hold_out = hold.copy()
    hold_out["_v3_source"] = "v2_final_holdout_consumed"

    v3_development = pd.concat(
        [dev_out, hold_out],
        ignore_index=True,
        sort=False,
    ).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    v3_dev_path = out_dir / "First_Half_37_V3_DEVELOPMENT_CONSUMED.csv"
    v3_development.to_csv(v3_dev_path, index=False)

    folds = make_folds(dev, hold)

    fold_frames = []
    state_frames = []
    pred_frames = []

    for name, spec in candidates.items():
        print("-" * 120)
        print(f"{name} | kind={spec['kind']} | features={len(spec['features'])}")

        fm, sm, pm = evaluate_candidate(name, spec, dev, hold, folds)

        fold_frames.append(fm)
        state_frames.append(sm)
        pred_frames.append(pm)

        internal = fm[fm["kind"] == "internal"]
        stress = fm[fm["kind"] == "late_stress"].iloc[0]

        print(
            f"  INT ROC={internal['roc_auc'].mean():.4f} | "
            f"INT PR lift={internal['pr_lift_vs_val_prevalence'].mean():+.4f} | "
            f"LATE ROC={stress['roc_auc']:.4f} | "
            f"LATE PR lift={stress['pr_lift_vs_val_prevalence']:+.4f} | "
            f"fallback={stress['fallback_fraction']:.1%}"
        )

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    state_metrics = pd.concat(state_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)

    ranking = summarize_models(fold_metrics, state_metrics)

    fold_path = out_dir / "first_half_37_v3_context_fold_metrics.csv"
    state_path = out_dir / "first_half_37_v3_context_state_metrics.csv"
    pred_path = out_dir / "first_half_37_v3_context_predictions.csv"
    ranking_out = out_dir / "first_half_37_v3_context_model_ranking.csv"
    definitions_path = out_dir / "first_half_37_v3_context_definitions.json"
    report_path = out_dir / "first_half_37_v3_context_benchmark_report.md"
    manifest_path = out_dir / "first_half_37_v3_context_benchmark_manifest.json"

    fold_metrics.to_csv(fold_path, index=False)
    state_metrics.to_csv(state_path, index=False)
    predictions.to_csv(pred_path, index=False)
    ranking.to_csv(ranking_out, index=False)

    definitions_path.write_text(
        json.dumps(
            {
                "contexts": CONTEXT_ORDER,
                "context_metric_order": CONTEXT_METRIC_ORDER,
                "missing_score_policy": "explicit missing context; global fallback for SEPARATE4; no row dropped",
                "top8": TOP8,
                "top19": top19,
                "interaction_features": interaction_features,
                "candidates": candidates,
                "C_grid": C_GRID,
                "min_context_train": MIN_CONTEXT_TRAIN,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(report_path, dev, hold, candidates, ranking, state_metrics)

    manifest = {
        "schema_version": "first_half_37_v3_context_benchmark_v1",
        "created_at_utc": now_utc_iso(),
        "mode": "research_only_consumed_data",
        "future_holdout_used": False,
        "v2_final_holdout_status": "CONSUMED_AND_USED_AS_LATE_STRESS",
        "dev_v2_rows": int(len(dev)),
        "late_stress_rows": int(len(hold)),
        "v3_development_rows": int(len(v3_development)),
        "dev_v2_sha256": sha256_file(dev_path),
        "late_stress_sha256": sha256_file(holdout_path),
        "ranking_v2_sha256": sha256_file(ranking_path),
        "candidates": {
            k: {
                "kind": v["kind"],
                "n_features": len(v["features"]),
                "features": v["features"],
            }
            for k, v in candidates.items()
        },
        "outputs": {
            "v3_development": str(v3_dev_path),
            "fold_metrics": str(fold_path),
            "state_metrics": str(state_path),
            "predictions": str(pred_path),
            "ranking": str(ranking_out),
            "definitions": str(definitions_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    winner = ranking.iloc[0]
    pass_count = int(ranking["passes_v3_context_research_gate"].sum())

    print()
    print("=" * 120)
    print("V3 CONTEXT BENCHMARK TERMINE")
    print("=" * 120)
    print(f"Candidats testés           : {len(ranking)}")
    print(f"Candidats PASS research    : {pass_count}")
    print(
        f"#1                        : {winner['model']} | "
        f"INT ROC={winner['mean_internal_roc_auc']:.4f} | "
        f"INT PR lift={winner['mean_internal_pr_lift']:+.4f} | "
        f"LATE ROC={winner['late_stress_roc_auc']:.4f} | "
        f"LATE PR lift={winner['late_stress_pr_lift']:+.4f} | "
        f"gate={'PASS' if int(winner['passes_v3_context_research_gate']) else 'FAIL'}"
    )
    print(
        f"Delta vs GLOBAL_TOP8      : "
        f"ROC={winner['delta_stress_roc_vs_global_top8']:+.4f} | "
        f"PR lift={winner['delta_stress_pr_lift_vs_global_top8']:+.4f}"
    )
    print(f"V3 DEVELOPMENT consommé   : {v3_dev_path}")
    print(f"Rapport                    : {report_path}")
    print(f"Ranking                    : {ranking_out}")
    print(f"State metrics              : {state_path}")
    print()
    print(
        "IMPORTANT: ceci est du DEVELOPMENT. "
        "Aucune architecture n'est encore validée sur un nouveau futur holdout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
