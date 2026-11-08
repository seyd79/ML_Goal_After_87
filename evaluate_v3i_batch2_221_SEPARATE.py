#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3-I — ÉVALUATION BATCH 2 SÉPARÉ (221) + COMBINÉ 500
=====================================================

Aucun réentraînement.
Aucune nouvelle recherche de seuil.

Utilise :
- Batch1 frozen 279 ;
- Batch2 frozen 221 ;
- V3-H exact déjà figé après Batch1 ;
- V3-C production.

Compare les règles déjà figées sur :
- Batch1 279
- Batch2 221
- Combined 500
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

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

BATCH1 = (
    BASE / "batch1_frozen_279/V3I_BATCH1_279_FROZEN.csv"
)
BATCH2 = (
    BASE / "batch2_frozen_221/V3I_BATCH2_221_READY.csv"
)

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

OUT = (
    BASE / "evaluation_batch2_221_separate_and_combined500"
)

TARGET = "goal_after_87"
EPS = 1e-12

BANDS = [
    ("V3C_MAIN_55_65", "p_v3c", 0.55, 0.65),
    ("V3H_MAIN_31_34", "p_v3h", 0.31, 0.34),
    ("V3H_STRONG_32_34", "p_v3h", 0.32, 0.34),
    ("V3C_SECONDARY_58_61", "p_v3c", 0.58, 0.61),
    ("V3H_SECONDARY_32p5_34", "p_v3h", 0.325, 0.34),
]


def hr(s: str):
    print()
    print("=" * 112)
    print(s)
    print("=" * 112)


def safe_roc(y, p):
    y = np.asarray(y, dtype=int)
    return (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan")
    )


def safe_pr(y, p):
    y = np.asarray(y, dtype=int)
    return (
        float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan")
    )


def wilson(pos, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    p = pos/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    m = z*math.sqrt(
        p*(1-p)/n + z*z/(4*n*n)
    )/d
    return c-m, c+m


def ece10(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, 11)
    out = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (
            (p <= hi) if i == 9 else (p < hi)
        )
        n = int(mask.sum())
        if n:
            out += (n/len(y))*abs(
                float(y[mask].mean())
                - float(p[mask].mean())
            )
    return float(out)


def platt(package: Mapping, p_raw):
    p_raw = np.clip(
        np.asarray(p_raw, dtype=float),
        EPS,
        1-EPS,
    )
    x = np.log(p_raw/(1-p_raw)).reshape(-1, 1)
    return package[
        "platt_logistic_model"
    ].predict_proba(x)[:, 1]


def score(df, package, features: Sequence[str]):
    X = df[list(features)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Bloc contient une ligne non 38/38."
        )

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )
    if not y.isin([0, 1]).all():
        raise RuntimeError(
            "Targets non binaires/incomplets."
        )

    v3c = joblib.load(V3C)
    p_c = np.asarray(
        v3c.predict_proba(X),
        dtype=float,
    )[:, 1]

    p_raw = np.asarray(
        package["et_model"].predict_proba(X),
        dtype=float,
    )[:, 1]

    p_h = platt(package, p_raw)

    out = df.copy()
    out["p_v3c"] = p_c
    out["p_v3h_raw"] = p_raw
    out["p_v3h"] = p_h
    return out


def band_table(scored, sample):
    rows = []
    base = float(scored[TARGET].mean())

    for rule, col, lo, hi in BANDS:
        z = scored[
            (scored[col] >= lo)
            & (scored[col] < hi)
        ]
        n = len(z)
        pos = int(z[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        a, b = wilson(pos, n)

        rows.append({
            "sample": sample,
            "rule": rule,
            "sample_n": len(scored),
            "sample_base_rate": base,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "lift_pp": (
                rate-base
                if n
                else float("nan")
            ),
            "ci95_low": a,
            "ci95_high": b,
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
        })

    return pd.DataFrame(rows)


def global_metrics(scored, sample):
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
            "positive_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "roc_auc": safe_roc(y, p),
            "pr_auc": safe_pr(y, p),
            "brier": float(
                brier_score_loss(y, p)
            ),
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


def pct(x):
    try:
        x = float(x)
    except Exception:
        return ""
    return (
        ""
        if not math.isfinite(x)
        else f"{x:.2%}"
    )


def main():
    for p in [BATCH1, BATCH2, V3H, V3C]:
        if not p.exists():
            raise FileNotFoundError(p)

    b1 = pd.read_csv(BATCH1, low_memory=False)
    b2 = pd.read_csv(BATCH2, low_memory=False)

    if len(b1) != 279:
        raise RuntimeError(
            f"Batch1={len(b1)}, attendu=279."
        )
    if len(b2) != 221:
        raise RuntimeError(
            f"Batch2={len(b2)}, attendu=221."
        )

    ids1 = set(b1["match_id"].astype(str))
    ids2 = set(b2["match_id"].astype(str))

    if ids1.intersection(ids2):
        raise RuntimeError(
            "Overlap Batch1/Batch2."
        )

    package = joblib.load(V3H)
    features = list(package["features"])

    s1 = score(b1, package, features)
    s2 = score(b2, package, features)
    s500 = score(
        pd.concat(
            [b1, b2],
            ignore_index=True,
            sort=False,
        ),
        package,
        features,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    s2.to_csv(
        OUT / "batch2_221_predictions.csv",
        index=False,
    )
    s500.to_csv(
        OUT / "combined_500_predictions.csv",
        index=False,
    )
    bands.to_csv(
        OUT / "frozen_rule_rates_batch1_batch2_500.csv",
        index=False,
    )
    metrics.to_csv(
        OUT / "model_metrics_batch1_batch2_500.csv",
        index=False,
    )

    hr("BATCH 2 — STABILITÉ DES RÈGLES FIGÉES")

    show = bands[
        bands["rule"].isin([
            "V3C_MAIN_55_65",
            "V3H_MAIN_31_34",
            "V3H_STRONG_32_34",
        ])
    ].copy()

    for c in [
        "sample_base_rate",
        "selected_rate",
        "lift_pp",
        "ci95_low",
        "ci95_high",
        "roi_at_3.20",
    ]:
        show[c] = show[c].map(pct)

    show["fair_odds_empirical"] = show[
        "fair_odds_empirical"
    ].map(
        lambda x: (
            ""
            if pd.isna(x)
            else f"{x:.2f}"
        )
    )

    print(
        show[
            [
                "sample",
                "rule",
                "selected_n",
                "selected_positives",
                "selected_rate",
                "sample_base_rate",
                "lift_pp",
                "ci95_low",
                "ci95_high",
                "fair_odds_empirical",
                "roi_at_3.20",
            ]
        ].to_string(index=False)
    )

    hr("SORTIES")
    print(OUT)

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "BATCH2_EVALUATED_WITH_FROZEN_MODELS",
        "batch1_n": 279,
        "batch2_n": 221,
        "combined_n": 500,
        "v3h_frozen": str(V3H),
        "v3c": str(V3C),
        "note": (
            "No retrain, no retuning, "
            "frozen probability bands only."
        ),
    }

    (
        OUT / "evaluation_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
