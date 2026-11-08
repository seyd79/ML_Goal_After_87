#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOAL AFTER 87 — BATCH 2 FINAL DIAGNOSTICS
==========================================

Objectif
--------
Faire, SANS réentraînement et SANS retuning :

1) le contrôle des règles figées, notamment la bande confirmatoire V3-H 32–36
   qui manquait dans l'évaluateur précédent ;
2) les métriques globales V3-C / V3-H raw / V3-H Platt sur Batch1, Batch2 et 500 ;
3) le test prospectif du contexte ligue sur Batch2, en utilisant UNIQUEMENT
   le mapping ligue appris auparavant sur ROLLING_OOS_684 ;
4) un résumé propre avant toute décision sur les 168 matchs de réserve.

Important
---------
- Aucun modèle n'est réentraîné.
- Aucun seuil n'est recherché dans Batch2.
- Le fichier de tiers ligue est lu tel quel depuis l'audit OOS antérieur.
- Batch2_221 est la partie décisive pour les nouvelles règles de contexte ligue.
- COMBINED_500 est descriptif / supportif, pas un nouveau jeu indépendant.
- Le scanner de production n'est jamais modifié.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/v3i_unseen_holdout_500"

BATCH1 = BASE / "batch1_frozen_279/V3I_BATCH1_279_FROZEN.csv"
BATCH2 = BASE / "batch2_frozen_221/V3I_BATCH2_221_READY.csv"

V3H = (
    BASE
    / "batch1_frozen_279"
    / "V3H_ET_LEAF5_PLATT_EXACT_FINAL_DEV1368.joblib"
)

V3C = (
    ROOT
    / "outputs/model_v3_experiments/candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)

# Mapping de ligue figé AVANT Batch2, appris uniquement sur OOS 684.
LEAGUE_SCORES = (
    ROOT
    / "outputs/league_partial_pooling_external"
    / "oos_league_partial_pooling_scores.csv"
)

OUT = BASE / "evaluation_batch2_FINAL_DIAGNOSTICS"

TARGET = "goal_after_87"
COMP = "competition_id"
EPS = 1e-12

# -----------------------------------------------------------------------------
# RÈGLES
# -----------------------------------------------------------------------------
# Confirmatoires/frozen avant V3-I :
# - V3C 55–65
# - V3H 32–34
# - V3H 32–36
#
# V3H 31–34 était exploratoire historiquement, puis prospectivement conservée
# pour les validations externes et pour le contexte ligue.
BANDS = [
    ("V3C_MAIN_55_65", "p_v3c", 0.55, 0.65, "FROZEN_PRIMARY"),
    ("V3H_STRONG_32_34", "p_v3h", 0.32, 0.34, "FROZEN_PRIMARY"),
    ("V3H_BROAD_32_36", "p_v3h", 0.32, 0.36, "FROZEN_PRIMARY"),
    ("V3H_MAIN_31_34", "p_v3h", 0.31, 0.34, "FROZEN_SECONDARY"),
]

# Contexte ligue gelé après l'analyse OOS -> Batch1.
LEAGUE_RULE_LO = 0.31
LEAGUE_RULE_HI = 0.34


def hr(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_roc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_pr(y, p) -> float:
    y = np.asarray(y, dtype=int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def wilson(pos: int, n: int, z: float = 1.96):
    if n <= 0:
        return float("nan"), float("nan")
    p = pos / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-m, c+m


def ece10(y, p) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, 11)
    out = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & ((p <= hi) if i == 9 else (p < hi))
        n = int(mask.sum())
        if n:
            out += (n/len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def platt(package: Mapping, p_raw):
    p_raw = np.clip(np.asarray(p_raw, dtype=float), EPS, 1-EPS)
    x = np.log(p_raw/(1-p_raw)).reshape(-1, 1)
    return package["platt_logistic_model"].predict_proba(x)[:, 1]


def score(df: pd.DataFrame, package: Mapping, features: Sequence[str], v3c_model) -> pd.DataFrame:
    X = df[list(features)].apply(pd.to_numeric, errors="coerce")

    # On exige le schéma modèle exact : 38 features présentes.
    if X.shape[1] != len(features):
        raise RuntimeError(f"Features={X.shape[1]}, attendu={len(features)}")

    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad_rows = np.where(~np.isfinite(arr).all(axis=1))[0]
        raise RuntimeError(
            f"Bloc contient des lignes non exactes sur les 38 features. "
            f"Exemples index={bad_rows[:10].tolist()}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if not y.isin([0, 1]).all():
        raise RuntimeError("Targets non binaires/incomplets.")

    p_c = np.asarray(v3c_model.predict_proba(X), dtype=float)[:, 1]
    p_raw = np.asarray(package["et_model"].predict_proba(X), dtype=float)[:, 1]
    p_h = platt(package, p_raw)

    out = df.copy()
    out[TARGET] = y.astype(int)
    out[COMP] = out[COMP].astype(str)
    out["p_v3c"] = p_c
    out["p_v3h_raw"] = p_raw
    out["p_v3h"] = p_h
    return out


def band_table(scored: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    base = float(scored[TARGET].mean())

    for rule, col, lo, hi, status in BANDS:
        z = scored[(scored[col] >= lo) & (scored[col] < hi)]
        n = len(z)
        pos = int(z[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        a, b = wilson(pos, n)

        rows.append({
            "sample": sample,
            "rule": rule,
            "rule_status": status,
            "probability_column": col,
            "lo_inclusive": lo,
            "hi_exclusive": hi,
            "sample_n": len(scored),
            "sample_positives": int(scored[TARGET].sum()),
            "sample_base_rate": base,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "lift_pp": rate-base if n else float("nan"),
            "ci95_low": a,
            "ci95_high": b,
            "fair_odds_empirical": 1/rate if n and rate > 0 else float("nan"),
            "roi_at_3.20": rate*3.20 - 1 if n else float("nan"),
        })

    return pd.DataFrame(rows)


def global_metrics(scored: pd.DataFrame, sample: str) -> pd.DataFrame:
    y = scored[TARGET].astype(int).to_numpy()
    rows = []

    for name, col in [
        ("V3C", "p_v3c"),
        ("V3H_RAW", "p_v3h_raw"),
        ("V3H_PLATT", "p_v3h"),
    ]:
        p = scored[col].to_numpy(dtype=float)
        rows.append({
            "sample": sample,
            "model": name,
            "n": len(y),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "roc_auc": safe_roc(y, p),
            "pr_auc": safe_pr(y, p),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0, 1])),
            "ece_10": ece10(y, p),
        })

    return pd.DataFrame(rows)


def load_frozen_league_mapping(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, low_memory=False)

    required = {
        COMP,
        "league_tier",
        "oos_conservative_league_score",
        "oos_league_n",
        "oos_selected_n",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise RuntimeError(f"Colonnes ligue manquantes : {missing}")

    scores = scores.copy()
    scores[COMP] = scores[COMP].astype(str)

    if scores[COMP].duplicated().any():
        dup = scores.loc[scores[COMP].duplicated(), COMP].head(10).tolist()
        raise RuntimeError(f"competition_id dupliqués dans mapping ligue : {dup}")

    # Aucune recalcul de tier ici : on utilise exactement le tier déjà figé.
    return scores


def attach_league_context(scored: pd.DataFrame, league_scores: pd.DataFrame) -> pd.DataFrame:
    cols = [
        COMP,
        "league_tier",
        "oos_conservative_league_score",
        "oos_league_n",
        "oos_selected_n",
    ]

    out = scored.merge(
        league_scores[cols],
        on=COMP,
        how="left",
        validate="many_to_one",
    )

    out["league_status"] = np.where(
        out["league_tier"].notna(),
        "KNOWN_OOS_LEAGUE",
        "UNSEEN_LEAGUE",
    )
    out["league_tier_external"] = out["league_tier"].fillna("UNSEEN_LEAGUE")
    return out


def league_context_table(scored: pd.DataFrame, sample: str) -> pd.DataFrame:
    # IMPORTANT : le contexte ligue a été défini pour V3H 31–34.
    z = scored[
        (scored["p_v3h"] >= LEAGUE_RULE_LO)
        & (scored["p_v3h"] < LEAGUE_RULE_HI)
    ].copy()

    scenarios = [
        (
            "ALL_V3H_31_34",
            pd.Series(True, index=z.index),
            "FROZEN_BASELINE",
        ),
        (
            "KNOWN_OOS_LEAGUES_ONLY",
            z["league_status"] == "KNOWN_OOS_LEAGUE",
            "PROSPECTIVE_LEAGUE_TEST",
        ),
        (
            "HIGH_OR_MID_OOS_LEAGUE_SCORE",
            z["league_tier_external"].isin(["HIGH", "MID"]),
            "PROSPECTIVE_LEAGUE_TEST",
        ),
        (
            "HIGH_OOS_LEAGUE_SCORE_ONLY",
            z["league_tier_external"] == "HIGH",
            "DIAGNOSTIC",
        ),
        (
            "MID_OOS_LEAGUE_SCORE_ONLY",
            z["league_tier_external"] == "MID",
            "DIAGNOSTIC",
        ),
        (
            "LOW_OOS_LEAGUE_SCORE_ONLY",
            z["league_tier_external"] == "LOW",
            "DIAGNOSTIC",
        ),
        (
            "UNSEEN_LEAGUE_ONLY",
            z["league_status"] == "UNSEEN_LEAGUE",
            "DIAGNOSTIC",
        ),
    ]

    rows = []
    sample_base = float(scored[TARGET].mean())
    rule_base = float(z[TARGET].mean()) if len(z) else float("nan")

    for name, mask, status in scenarios:
        g = z[mask]
        n = len(g)
        pos = int(g[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        rows.append({
            "sample": sample,
            "scenario": name,
            "scenario_status": status,
            "sample_n": len(scored),
            "sample_base_rate": sample_base,
            "v3h31_34_n": len(z),
            "v3h31_34_rate": rule_base,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "lift_vs_sample_pp": rate-sample_base if n else float("nan"),
            "lift_vs_v3h31_34_pp": rate-rule_base if n else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
            "fair_odds_empirical": 1/rate if n and rate > 0 else float("nan"),
            "roi_at_3.20": rate*3.20 - 1 if n else float("nan"),
        })

    return pd.DataFrame(rows)


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int = 42, B: int = 5000):
    mask = np.isfinite(score)
    y = np.asarray(y)[mask].astype(int)
    score = np.asarray(score)[mask].astype(float)

    if len(y) < 5 or len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return float("nan"), float("nan"), float("nan"), 0, len(y)

    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        ss = score[idx]
        if len(np.unique(yy)) < 2 or len(np.unique(ss)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, ss)))

    if not vals:
        return auc, float("nan"), float("nan"), 0, len(y)

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi), len(vals), len(y)


def league_auc_table(scored: pd.DataFrame, sample: str) -> pd.DataFrame:
    z = scored[
        (scored["p_v3h"] >= LEAGUE_RULE_LO)
        & (scored["p_v3h"] < LEAGUE_RULE_HI)
    ].copy()

    score = pd.to_numeric(
        z["oos_conservative_league_score"], errors="coerce"
    ).to_numpy(dtype=float)

    auc, lo, hi, valid_b, n_known = bootstrap_auc(
        z[TARGET].to_numpy(), score
    )

    return pd.DataFrame([{
        "sample": sample,
        "v3h31_34_selected_n": len(z),
        "selected_n_with_known_oos_league_score": n_known,
        "auc_oos_league_score": auc,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "bootstrap_valid_replicates": valid_b,
    }])


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(v) else f"{v:.2%}"


def f4(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(v) else f"{v:.4f}"


def print_band_results(bands: pd.DataFrame) -> None:
    show = bands.copy()
    for c in [
        "sample_base_rate", "selected_rate", "lift_pp", "ci95_low",
        "ci95_high", "roi_at_3.20"
    ]:
        show[c] = show[c].map(pct)

    show["fair_odds_empirical"] = show["fair_odds_empirical"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    )

    cols = [
        "sample", "rule", "rule_status", "selected_n",
        "selected_positives", "selected_rate", "sample_base_rate",
        "lift_pp", "ci95_low", "ci95_high", "fair_odds_empirical",
        "roi_at_3.20",
    ]
    print(show[cols].to_string(index=False))


def print_metrics(metrics: pd.DataFrame) -> None:
    show = metrics.copy()
    for c in ["positive_rate", "mean_probability", "ece_10"]:
        show[c] = show[c].map(pct)
    for c in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        show[c] = show[c].map(f4)

    print(show[
        [
            "sample", "model", "n", "positive_rate", "mean_probability",
            "roc_auc", "pr_auc", "brier", "log_loss", "ece_10",
        ]
    ].to_string(index=False))


def print_league(league: pd.DataFrame) -> None:
    show = league.copy()
    for c in [
        "sample_base_rate", "v3h31_34_rate", "selected_rate",
        "lift_vs_sample_pp", "lift_vs_v3h31_34_pp", "ci95_low",
        "ci95_high", "roi_at_3.20",
    ]:
        show[c] = show[c].map(pct)
    show["fair_odds_empirical"] = show["fair_odds_empirical"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    )

    print(show[
        [
            "sample", "scenario", "scenario_status", "selected_n",
            "selected_positives", "selected_rate", "v3h31_34_rate",
            "lift_vs_v3h31_34_pp", "sample_base_rate",
            "lift_vs_sample_pp", "ci95_low", "ci95_high",
            "fair_odds_empirical", "roi_at_3.20",
        ]
    ].to_string(index=False))


def main() -> int:
    required_paths = [BATCH1, BATCH2, V3H, V3C, LEAGUE_SCORES]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    b1 = pd.read_csv(BATCH1, low_memory=False)
    b2 = pd.read_csv(BATCH2, low_memory=False)

    if len(b1) != 279:
        raise RuntimeError(f"Batch1={len(b1)}, attendu=279.")
    if len(b2) != 221:
        raise RuntimeError(f"Batch2={len(b2)}, attendu=221.")

    ids1 = set(b1["match_id"].astype(str))
    ids2 = set(b2["match_id"].astype(str))
    overlap = ids1 & ids2
    if overlap:
        raise RuntimeError(f"Overlap Batch1/Batch2 : {len(overlap)}")

    package = joblib.load(V3H)
    features = list(package["features"])
    if len(features) != 38:
        raise RuntimeError(f"V3-H package features={len(features)}, attendu=38")

    v3c_model = joblib.load(V3C)
    league_scores = load_frozen_league_mapping(LEAGUE_SCORES)
    league_hash = sha256_file(LEAGUE_SCORES)

    s1 = score(b1, package, features, v3c_model)
    s2 = score(b2, package, features, v3c_model)
    s500 = pd.concat([s1, s2], ignore_index=True, sort=False)

    s1 = attach_league_context(s1, league_scores)
    s2 = attach_league_context(s2, league_scores)
    s500 = attach_league_context(s500, league_scores)

    OUT.mkdir(parents=True, exist_ok=True)

    # Copie exacte du mapping ligue utilisé pour rendre le test audit-able.
    frozen_copy = OUT / "FROZEN_oos_league_partial_pooling_scores_USED.csv"
    shutil.copy2(LEAGUE_SCORES, frozen_copy)

    bands = pd.concat([
        band_table(s1, "BATCH1_279"),
        band_table(s2, "BATCH2_221"),
        band_table(s500, "COMBINED_500"),
    ], ignore_index=True)

    metrics = pd.concat([
        global_metrics(s1, "BATCH1_279"),
        global_metrics(s2, "BATCH2_221"),
        global_metrics(s500, "COMBINED_500"),
    ], ignore_index=True)

    league = pd.concat([
        league_context_table(s1, "BATCH1_279"),
        league_context_table(s2, "BATCH2_221"),
        league_context_table(s500, "COMBINED_500"),
    ], ignore_index=True)

    league_auc = pd.concat([
        league_auc_table(s1, "BATCH1_279"),
        league_auc_table(s2, "BATCH2_221"),
        league_auc_table(s500, "COMBINED_500"),
    ], ignore_index=True)

    # Sorties détaillées.
    s1.to_csv(OUT / "batch1_279_predictions_with_league_context.csv", index=False)
    s2.to_csv(OUT / "batch2_221_predictions_with_league_context.csv", index=False)
    s500.to_csv(OUT / "combined_500_predictions_with_league_context.csv", index=False)
    bands.to_csv(OUT / "frozen_rule_stability_including_v3h_32_36.csv", index=False)
    metrics.to_csv(OUT / "global_model_metrics_batch1_batch2_500.csv", index=False)
    league.to_csv(OUT / "frozen_league_context_test_batch1_batch2_500.csv", index=False)
    league_auc.to_csv(OUT / "frozen_league_score_auc_batch1_batch2_500.csv", index=False)

    # ------------------------------------------------------------------
    # Affichage terminal
    # ------------------------------------------------------------------
    hr("1) RÈGLES FIGÉES — INCLUT MAINTENANT V3-H 32–36")
    print_band_results(bands)

    hr("2) MÉTRIQUES GLOBALES — V3-C / V3-H RAW / V3-H PLATT")
    print_metrics(metrics)

    hr("3) CONTEXTE LIGUE FIGÉ OOS — TEST PROSPECTIF SUR BATCH2")
    print(f"Mapping ligue utilisé : {LEAGUE_SCORES}")
    print(f"SHA256 mapping ligue  : {league_hash}")
    print()
    print_league(league)

    hr("4) DISCRIMINATION CONTINUE DU SCORE LIGUE")
    la = league_auc.copy()
    for c in ["auc_oos_league_score", "bootstrap_ci95_low", "bootstrap_ci95_high"]:
        la[c] = la[c].map(f4)
    print(la.to_string(index=False))

    # ------------------------------------------------------------------
    # Résumé strictement factuel, sans retuning.
    # ------------------------------------------------------------------
    def pick_band(sample: str, rule: str):
        r = bands[(bands["sample"] == sample) & (bands["rule"] == rule)]
        return None if r.empty else r.iloc[0]

    def pick_league(sample: str, scenario: str):
        r = league[(league["sample"] == sample) & (league["scenario"] == scenario)]
        return None if r.empty else r.iloc[0]

    summary = {
        "created_at_utc": now_utc(),
        "no_retrain": True,
        "no_retuning": True,
        "production_scanner_modified": False,
        "batch1_n": 279,
        "batch2_n": 221,
        "combined_n": 500,
        "v3h_features": len(features),
        "league_mapping_source": str(LEAGUE_SCORES),
        "league_mapping_sha256": league_hash,
        "rules": {
            "V3C_MAIN_55_65": "frozen primary",
            "V3H_STRONG_32_34": "frozen primary",
            "V3H_BROAD_32_36": "frozen primary previously omitted from Batch2 evaluator",
            "V3H_MAIN_31_34": "frozen secondary / basis for league-context test",
        },
        "league_context_tests": {
            "KNOWN_OOS_LEAGUES_ONLY": "prospectively frozen before Batch2 labels",
            "HIGH_OR_MID_OOS_LEAGUE_SCORE": "prospectively frozen before Batch2 labels",
        },
    }

    for rule in ["V3C_MAIN_55_65", "V3H_STRONG_32_34", "V3H_BROAD_32_36", "V3H_MAIN_31_34"]:
        r = pick_band("BATCH2_221", rule)
        if r is not None:
            summary.setdefault("batch2_rules", {})[rule] = {
                "selected_n": int(r["selected_n"]),
                "positives": int(r["selected_positives"]),
                "rate": float(r["selected_rate"]),
                "sample_base_rate": float(r["sample_base_rate"]),
                "lift_pp": float(r["lift_pp"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
            }

    for scenario in ["KNOWN_OOS_LEAGUES_ONLY", "HIGH_OR_MID_OOS_LEAGUE_SCORE"]:
        r = pick_league("BATCH2_221", scenario)
        if r is not None:
            summary.setdefault("batch2_league_context", {})[scenario] = {
                "selected_n": int(r["selected_n"]),
                "positives": int(r["selected_positives"]),
                "rate": float(r["selected_rate"]),
                "v3h31_34_rate": float(r["v3h31_34_rate"]),
                "lift_vs_v3h31_34_pp": float(r["lift_vs_v3h31_34_pp"]),
                "sample_base_rate": float(r["sample_base_rate"]),
            }

    (OUT / "FINAL_DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": now_utc(),
        "status": "FINAL_BATCH2_DIAGNOSTICS_COMPLETE",
        "inputs": {
            "batch1": str(BATCH1),
            "batch2": str(BATCH2),
            "v3c": str(V3C),
            "v3h": str(V3H),
            "league_scores": str(LEAGUE_SCORES),
        },
        "league_mapping_sha256": league_hash,
        "methodological_note": (
            "Batch2_221 is the decisive prospective replication block. "
            "Combined_500 is supporting/descriptive. No threshold or tier is retuned here. "
            "The 168 reserve matches are not touched by this script."
        ),
    }

    (OUT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hr("5) SORTIES")
    print(OUT)
    print("  frozen_rule_stability_including_v3h_32_36.csv")
    print("  global_model_metrics_batch1_batch2_500.csv")
    print("  frozen_league_context_test_batch1_batch2_500.csv")
    print("  frozen_league_score_auc_batch1_batch2_500.csv")
    print("  batch2_221_predictions_with_league_context.csv")
    print("  FINAL_DIAGNOSTIC_SUMMARY.json")
    print("  FROZEN_oos_league_partial_pooling_scores_USED.csv")
    print("  evaluation_manifest.json")

    hr("IMPORTANT")
    print(
        "Les 168 matchs de réserve ne sont pas chargés et ne sont pas évalués par ce script.\n"
        "On lira d'abord Batch2 pour décider quelles règles, s'il y en a, méritent d'être\n"
        "figées pour le test final sur cette réserve."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
