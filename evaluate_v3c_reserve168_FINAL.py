#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, binomtest
except Exception:
    fisher_exact = None
    binomtest = None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "v3i_unseen_holdout_500" / "reserve168_final_v3c_only"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = ROOT / "outputs" / "model_v3_experiments" / "candidates_refit_all" / "V3_C_AGGRESSIVE_refit_all.joblib"
SCHEMA = ROOT / "outputs" / "model_v3_experiments" / "schemas" / "V3_C_AGGRESSIVE_schema.json"
FREEZE = ROOT / "outputs" / "v3c_league_context_freeze_before_reserve168" / "FROZEN_RULES_BEFORE_RESERVE168.json"

# The 389-row eligible pool created by the Batch2 collector.
POOL_CANDIDATES = [
    ROOT / "outputs" / "v3i_unseen_holdout_500" / "batch2_frozen_221" / "V3I_BATCH2_ELIGIBLE_POOL.csv",
    ROOT / "outputs" / "v3i_unseen_holdout_500" / "batch2_frozen_221" / "V3I_BATCH2_389_ELIGIBLE_POOL.csv",
    ROOT / "outputs" / "v3i_unseen_holdout_500" / "batch2_hierarchical_source" / "V3I_BATCH2_ELIGIBLE_POOL.csv",
]

# Historical external results already consumed; used only for stability comparison.
BATCH1_N, BATCH1_SEL, BATCH1_POS = 279, 55, 20
BATCH2_N, BATCH2_SEL, BATCH2_POS = 221, 58, 23
PREV_EXT_SEL = BATCH1_SEL + BATCH2_SEL
PREV_EXT_POS = BATCH1_POS + BATCH2_POS
PREV_EXT_RATE = PREV_EXT_POS / PREV_EXT_SEL

LOW, HIGH = 0.55, 0.65
BREAK_EVEN_320 = 1 / 3.20


def read_csv_robust(path: Path) -> pd.DataFrame:
    attempts = []
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        for sep in (",", "\t", ";"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                attempts.append((enc, sep, repr(e)))
    raise RuntimeError(f"Impossible de lire {path}; essais={attempts[-4:]}")


def find_pool() -> Path:
    for p in POOL_CANDIDATES:
        if p.exists():
            return p
    # Safe fallback: locate candidate pools with >= 389 rows and selection-order column.
    for p in (ROOT / "outputs").rglob("*.csv"):
        name = p.name.lower()
        if "eligible" not in name and "batch2" not in name:
            continue
        try:
            df = read_csv_robust(p)
        except Exception:
            continue
        if "_v3i_batch2_selection_order" in df.columns and len(df) >= 389:
            return p
    raise FileNotFoundError("Pool Batch2 389 introuvable sous outputs/.")


def load_features() -> list[str]:
    if not SCHEMA.exists():
        raise FileNotFoundError(SCHEMA)
    obj = json.loads(SCHEMA.read_text(encoding="utf-8"))
    for key in ("features", "feature_names", "columns", "input_features"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 38:
            return [str(x) for x in v]
    # recursively inspect values
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            if len(x) == 38 and all(isinstance(z, str) for z in x):
                yield x
            for z in x:
                yield from walk(z)
    for cand in walk(obj):
        return list(cand)
    raise RuntimeError("Impossible d'extraire les 38 features du schema V3-C.")


def get_predictor(obj):
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "estimator", "clf"):
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise RuntimeError("Artefact V3-C sans predict_proba exploitable.")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return max(0.0, ctr-half), min(1.0, ctr+half)


def pct(x):
    return "" if pd.isna(x) else f"{100*x:.2f}%"


def row_summary(sample: str, y: np.ndarray, sel: np.ndarray) -> dict:
    n = len(y)
    s = int(sel.sum())
    pos = int(y[sel].sum()) if s else 0
    rate = pos/s if s else float("nan")
    base = float(y.mean()) if n else float("nan")
    lo, hi = wilson(pos, s)
    non = ~sel
    non_n = int(non.sum())
    non_pos = int(y[non].sum()) if non_n else 0
    non_rate = non_pos/non_n if non_n else float("nan")
    odds_ratio = p_fisher = float("nan")
    if fisher_exact is not None and s and non_n:
        table = [[pos, s-pos], [non_pos, non_n-non_pos]]
        odds_ratio, p_fisher = fisher_exact(table, alternative="greater")
    p_vs_be = float("nan")
    if binomtest is not None and s:
        p_vs_be = binomtest(pos, s, BREAK_EVEN_320, alternative="greater").pvalue
    return {
        "sample": sample,
        "n": n,
        "positives": int(y.sum()),
        "base_rate": base,
        "selected_n": s,
        "selected_positives": pos,
        "selected_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "lift_vs_sample_pp": 100*(rate-base) if s else float("nan"),
        "nonselected_n": non_n,
        "nonselected_positives": non_pos,
        "nonselected_rate": non_rate,
        "lift_vs_nonselected_pp": 100*(rate-non_rate) if s and non_n else float("nan"),
        "fisher_one_sided_p": p_fisher,
        "odds_ratio_selected_vs_nonselected": odds_ratio,
        "fair_odds_empirical": 1/rate if s and rate > 0 else float("nan"),
        "roi_at_3.20": rate*3.20-1 if s else float("nan"),
        "p_one_sided_vs_break_even_31_25": p_vs_be,
    }


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ("base_rate", "selected_rate", "ci95_low", "ci95_high", "nonselected_rate", "roi_at_3.20"):
        if c in x:
            x[c] = x[c].map(pct)
    for c in ("lift_vs_sample_pp", "lift_vs_nonselected_pp"):
        if c in x:
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f} pp")
    for c in ("fisher_one_sided_p", "p_one_sided_vs_break_even_31_25"):
        if c in x:
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    if "odds_ratio_selected_vs_nonselected" in x:
        x["odds_ratio_selected_vs_nonselected"] = x["odds_ratio_selected_vs_nonselected"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    if "fair_odds_empirical" in x:
        x["fair_odds_empirical"] = x["fair_odds_empirical"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return x


def main():
    print("="*126)
    print("V3-C 55–65 — TEST FINAL SUR RÉSERVE 168")
    print("="*126)
    print("Aucun retuning. Aucun filtre ligue. Règle figée: 0.55 <= P(V3-C) < 0.65")

    if FREEZE.exists():
        freeze_obj = json.loads(FREEZE.read_text(encoding="utf-8"))
        print(f"Freeze manifest: {FREEZE}")
    else:
        freeze_obj = None
        print(f"AVERTISSEMENT: manifest de freeze absent: {FREEZE}")

    pool_path = find_pool()
    pool = read_csv_robust(pool_path)
    print(f"Pool: {pool_path}")
    print(f"Pool rows: {len(pool)}")

    order_col = "_v3i_batch2_selection_order"
    if order_col not in pool.columns:
        raise RuntimeError(f"Colonne absente: {order_col}")
    pool[order_col] = pd.to_numeric(pool[order_col], errors="coerce")
    reserve = pool.loc[pool[order_col].between(222, 389, inclusive="both")].copy()
    reserve = reserve.sort_values(order_col).reset_index(drop=True)
    if len(reserve) != 168:
        raise RuntimeError(f"Réserve attendue=168, obtenue={len(reserve)}")
    if reserve[order_col].tolist() != list(range(222,390)):
        raise RuntimeError("La réserve n'est pas exactement ordres 222..389.")

    if "goal_after_87" not in reserve.columns:
        raise RuntimeError("goal_after_87 absent.")
    y = pd.to_numeric(reserve["goal_after_87"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0,1}):
        raise RuntimeError("Targets réserve invalides/incomplets.")
    y = y.astype(int).to_numpy()

    features = load_features()
    missing = [c for c in features if c not in reserve.columns]
    if missing:
        raise RuntimeError(f"Features V3-C absentes: {missing}")
    print(f"Features V3-C: {len(features)}/38")

    model_obj = joblib.load(MODEL)
    model = get_predictor(model_obj)
    X = reserve[features].apply(pd.to_numeric, errors="coerce")
    p = model.predict_proba(X)[:,1]
    reserve["v3c_probability"] = p
    reserve["v3c55_65_selected"] = (p >= LOW) & (p < HIGH)
    sel = reserve["v3c55_65_selected"].to_numpy(bool)

    # Independent final holdout summary.
    final = row_summary("RESERVE168_FINAL", y, sel)

    # Stability reference from the already-consumed 500 external matches.
    prior = {
        "sample": "PREVIOUS_EXTERNAL500",
        "n": 500,
        "positives": int(round(0.2980*500)),
        "base_rate": 0.2980,
        "selected_n": PREV_EXT_SEL,
        "selected_positives": PREV_EXT_POS,
        "selected_rate": PREV_EXT_RATE,
        "ci95_low": wilson(PREV_EXT_POS, PREV_EXT_SEL)[0],
        "ci95_high": wilson(PREV_EXT_POS, PREV_EXT_SEL)[1],
        "lift_vs_sample_pp": 100*(PREV_EXT_RATE-0.2980),
        "nonselected_n": np.nan,
        "nonselected_positives": np.nan,
        "nonselected_rate": np.nan,
        "lift_vs_nonselected_pp": np.nan,
        "fisher_one_sided_p": np.nan,
        "odds_ratio_selected_vs_nonselected": np.nan,
        "fair_odds_empirical": 1/PREV_EXT_RATE,
        "roi_at_3.20": PREV_EXT_RATE*3.20-1,
        "p_one_sided_vs_break_even_31_25": binomtest(PREV_EXT_POS, PREV_EXT_SEL, BREAK_EVEN_320, alternative="greater").pvalue if binomtest is not None else np.nan,
    }

    # Combined external 668 is descriptive; reserve remains the decisive final test.
    comb_sel = PREV_EXT_SEL + final["selected_n"]
    comb_pos = PREV_EXT_POS + final["selected_positives"]
    comb_rate = comb_pos/comb_sel if comb_sel else np.nan
    combined = {
        "sample": "COMBINED_EXTERNAL668_DESCRIPTIVE",
        "n": 668,
        "positives": 149 + int(y.sum()),  # 29.8% of 500 was exactly 149 positives
        "base_rate": (149 + int(y.sum())) / 668,
        "selected_n": comb_sel,
        "selected_positives": comb_pos,
        "selected_rate": comb_rate,
        "ci95_low": wilson(comb_pos, comb_sel)[0],
        "ci95_high": wilson(comb_pos, comb_sel)[1],
        "lift_vs_sample_pp": 100*(comb_rate - ((149 + int(y.sum()))/668)),
        "nonselected_n": np.nan,
        "nonselected_positives": np.nan,
        "nonselected_rate": np.nan,
        "lift_vs_nonselected_pp": np.nan,
        "fisher_one_sided_p": np.nan,
        "odds_ratio_selected_vs_nonselected": np.nan,
        "fair_odds_empirical": 1/comb_rate if comb_rate > 0 else np.nan,
        "roi_at_3.20": comb_rate*3.20-1,
        "p_one_sided_vs_break_even_31_25": binomtest(comb_pos, comb_sel, BREAK_EVEN_320, alternative="greater").pvalue if binomtest is not None else np.nan,
    }

    summary = pd.DataFrame([final, prior, combined])

    print("\n" + "="*126)
    print("RÉSULTAT FINAL — V3-C 55–65")
    print("="*126)
    cols = [
        "sample","n","positives","base_rate","selected_n","selected_positives","selected_rate",
        "lift_vs_sample_pp","ci95_low","ci95_high","fair_odds_empirical","roi_at_3.20",
        "p_one_sided_vs_break_even_31_25"
    ]
    print(fmt_table(summary[cols]).to_string(index=False))

    print("\n" + "="*126)
    print("DIAGNOSTIC INTERNE RÉSERVE — SÉLECTIONNÉS VS NON-SÉLECTIONNÉS")
    print("="*126)
    diag_cols = [
        "sample","selected_n","selected_positives","selected_rate",
        "nonselected_n","nonselected_positives","nonselected_rate","lift_vs_nonselected_pp",
        "odds_ratio_selected_vs_nonselected","fisher_one_sided_p"
    ]
    print(fmt_table(pd.DataFrame([final])[diag_cols]).to_string(index=False))

    reserve_out = OUT / "reserve168_predictions_v3c_final.csv"
    summary_out = OUT / "reserve168_v3c_final_summary.csv"
    manifest_out = OUT / "FINAL_RESERVE168_MANIFEST.json"
    reserve.to_csv(reserve_out, index=False)
    summary.to_csv(summary_out, index=False)

    manifest = {
        "rule": "V3C_MAIN_55_65",
        "rule_definition": "0.55 <= p_v3c < 0.65",
        "league_filter": None,
        "reserve_selection_order": [222, 389],
        "reserve_n": 168,
        "pool_path": str(pool_path),
        "model_path": str(MODEL),
        "schema_path": str(SCHEMA),
        "freeze_manifest_path": str(FREEZE) if FREEZE.exists() else None,
        "final_holdout_result": final,
        "previous_external500": {
            "selected_n": PREV_EXT_SEL,
            "selected_positives": PREV_EXT_POS,
            "selected_rate": PREV_EXT_RATE,
        },
        "combined_external668_descriptive": {
            "selected_n": comb_sel,
            "selected_positives": comb_pos,
            "selected_rate": comb_rate,
        },
        "notes": [
            "Reserve168 is the decisive final holdout.",
            "Combined668 is descriptive only.",
            "No league filter and no retuning are applied.",
            "ROI@3.20 is theoretical and does not represent realized bookmaker odds."
        ]
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "="*126)
    print("SORTIES")
    print("="*126)
    print(OUT)
    print(f"  {reserve_out.name}")
    print(f"  {summary_out.name}")
    print(f"  {manifest_out.name}")
    print("\nScanner de production: NON MODIFIÉ")

if __name__ == "__main__":
    raise SystemExit(main())
