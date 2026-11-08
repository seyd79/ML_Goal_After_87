#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_first_half_37_dataset_shift.py

FIRST HALF 37 — DATASET / FEATURE DRIFT AUDIT V1
================================================

Objectif
--------
Comparer le dataset original utilisé pour construire First_Half37 V1 avec le
nouvel échantillon externe post-modèle, SANS réentraîner ni modifier le modèle.

Le script audite :
- dérive des 10 features COMPACT10 ;
- couverture / missingness ;
- PSI, KS, déplacement moyen standardisé ;
- stabilité du sens de relation feature -> target ;
- provenance LIVE (shotmap/timeline/fallback) ;
- dérive des compétitions ;
- état du score à 37' ;
- taux de target par jour ;
- si disponibles : dérive de la distribution des probabilités du modèle entre
  le FINAL HOLDOUT consommé et l'échantillon externe.

IMPORTANT
---------
- AUCUNE sélection de feature.
- AUCUN tuning.
- AUCUN refit.
- AUCUNE modification du modèle.
- Les seuils de drift servent au diagnostic seulement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None


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

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "drift_audit_v1"
)

DEFAULT_FINAL_HOLDOUT_PRED = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "final_holdout_v1"
    / "first_half_37_final_holdout_predictions.csv"
)

DEFAULT_EXTERNAL_PRED = (
    BASE_DIR
    / "outputs"
    / "first_half_37"
    / "post_model_external_v1"
    / "evaluation_v1"
    / "first_half_37_post_model_predictions.csv"
)

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
MATCH_COL = "match_id"

COMPACT10 = [
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

LIVE_COMPACT10 = {
    "home_shots_37",
    "away_xg_37",
    "away_shots_on_target_37",
    "last_corner_minute",
    "away_shots_37",
}

DRIFT_MODERATE_PSI = 0.10
DRIFT_SEVERE_PSI = 0.25
DRIFT_MODERATE_KS = 0.10
DRIFT_SEVERE_KS = 0.20
DRIFT_MODERATE_MISSING_DELTA = 0.05
DRIFT_SEVERE_MISSING_DELTA = 0.10
DRIFT_MODERATE_SMD = 0.30
DRIFT_SEVERE_SMD = 0.50
EPS = 1e-6


def clean_path(value: Optional[str], default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default.resolve()


def read_dataset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} introuvable : {path}")

    df = pd.read_csv(path, low_memory=False)

    if MATCH_COL not in df.columns:
        raise ValueError(f"{label}: colonne {MATCH_COL!r} absente")
    if TARGET not in df.columns:
        raise ValueError(f"{label}: colonne {TARGET!r} absente")

    df = df.drop_duplicates(subset=[MATCH_COL], keep="last").copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if DATE_COL in df.columns:
        # Accepte les deux formats observés : .000Z et Z.
        try:
            df["_kickoff"] = pd.to_datetime(
                df[DATE_COL], utc=True, errors="coerce", format="mixed"
            )
        except TypeError:
            df["_kickoff"] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    else:
        df["_kickoff"] = pd.NaT

    return df


def pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{100*x:.2f}%"


def finite_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_mean(s: pd.Series) -> float:
    x = finite_num(s).dropna()
    return float(x.mean()) if len(x) else float("nan")


def safe_std(s: pd.Series) -> float:
    x = finite_num(s).dropna()
    return float(x.std(ddof=1)) if len(x) >= 2 else float("nan")


def safe_quantile(s: pd.Series, q: float) -> float:
    x = finite_num(s).dropna()
    return float(x.quantile(q)) if len(x) else float("nan")


def safe_corr_feature_target(feature: pd.Series, target: pd.Series) -> float:
    x = finite_num(feature)
    y = pd.to_numeric(target, errors="coerce")
    mask = x.notna() & y.isin([0, 1])
    if mask.sum() < 5:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return float("nan")
    return float(xx.corr(yy))


def safe_univariate_auc(feature: pd.Series, target: pd.Series) -> float:
    if roc_auc_score is None:
        return float("nan")
    x = finite_num(feature)
    y = pd.to_numeric(target, errors="coerce")
    mask = x.notna() & y.isin([0, 1])
    if mask.sum() < 10:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, xx))
    except Exception:
        return float("nan")


def empirical_ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")

    if ks_2samp is not None:
        try:
            r = ks_2samp(a, b, alternative="two-sided", mode="auto")
            return float(r.statistic), float(r.pvalue)
        except Exception:
            pass

    values = np.unique(np.concatenate([a, b]))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, values, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, values, side="right") / len(b_sorted)
    stat = float(np.max(np.abs(cdf_a - cdf_b)))
    return stat, float("nan")


def build_psi_bins(reference: pd.Series, n_bins: int = 10) -> np.ndarray:
    x = finite_num(reference).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        v = float(edges[0]) if len(edges) else 0.0
        return np.array([-np.inf, v, np.inf], dtype=float)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    ref = finite_num(reference)
    cur = finite_num(current)
    bins = build_psi_bins(ref, n_bins=n_bins)

    ref_valid = ref.dropna().to_numpy(dtype=float)
    cur_valid = cur.dropna().to_numpy(dtype=float)

    ref_counts, _ = np.histogram(ref_valid, bins=bins)
    cur_counts, _ = np.histogram(cur_valid, bins=bins)

    # Ajout d'un bin missing pour conserver Missing != 0.
    ref_counts = np.append(ref_counts, int(ref.isna().sum()))
    cur_counts = np.append(cur_counts, int(cur.isna().sum()))

    ref_prop = ref_counts / max(1, ref_counts.sum())
    cur_prop = cur_counts / max(1, cur_counts.sum())

    ref_prop = np.clip(ref_prop, EPS, None)
    cur_prop = np.clip(cur_prop, EPS, None)

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def drift_severity(
    psi_value: float,
    ks_value: float,
    missing_delta: float,
    smd: float,
    direction_flip: bool,
) -> str:
    severe = (
        (np.isfinite(psi_value) and psi_value >= DRIFT_SEVERE_PSI)
        or (np.isfinite(ks_value) and ks_value >= DRIFT_SEVERE_KS)
        or abs(missing_delta) >= DRIFT_SEVERE_MISSING_DELTA
        or (np.isfinite(smd) and abs(smd) >= DRIFT_SEVERE_SMD)
        or direction_flip
    )
    if severe:
        return "SEVERE"

    moderate = (
        (np.isfinite(psi_value) and psi_value >= DRIFT_MODERATE_PSI)
        or (np.isfinite(ks_value) and ks_value >= DRIFT_MODERATE_KS)
        or abs(missing_delta) >= DRIFT_MODERATE_MISSING_DELTA
        or (np.isfinite(smd) and abs(smd) >= DRIFT_MODERATE_SMD)
    )
    return "MODERATE" if moderate else "LOW"


def summarize_dataset(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    target = pd.to_numeric(df[TARGET], errors="coerce")
    resolved = target.isin([0, 1])
    dates = df["_kickoff"].dropna()
    return {
        "dataset": label,
        "rows": int(len(df)),
        "unique_match_id": int(df[MATCH_COL].nunique(dropna=True)),
        "target_resolved": int(resolved.sum()),
        "positives": int((target == 1).sum()),
        "positive_rate": float(target[resolved].mean()) if resolved.any() else float("nan"),
        "date_min": dates.min().isoformat() if len(dates) else "",
        "date_max": dates.max().isoformat() if len(dates) else "",
        "competitions": int(df["competition_id"].nunique(dropna=True)) if "competition_id" in df.columns else 0,
    }


def feature_drift_table(original: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    missing_features = [f for f in COMPACT10 if f not in original.columns or f not in external.columns]
    if missing_features:
        raise ValueError(f"Features COMPACT10 absentes : {missing_features}")

    for feature in COMPACT10:
        o = finite_num(original[feature])
        e = finite_num(external[feature])

        o_non = o.dropna()
        e_non = e.dropna()

        o_mean = safe_mean(o)
        e_mean = safe_mean(e)
        o_std = safe_std(o)
        e_std = safe_std(e)
        smd = (e_mean - o_mean) / o_std if np.isfinite(o_std) and o_std > 0 else float("nan")

        ks_stat, ks_p = empirical_ks(o_non.to_numpy(dtype=float), e_non.to_numpy(dtype=float))
        psi_value = psi(o, e)

        corr_o = safe_corr_feature_target(o, original[TARGET])
        corr_e = safe_corr_feature_target(e, external[TARGET])
        auc_o = safe_univariate_auc(o, original[TARGET])
        auc_e = safe_univariate_auc(e, external[TARGET])

        direction_flip = False
        if np.isfinite(corr_o) and np.isfinite(corr_e):
            # Évite de flagger un flip quand les deux effets sont quasi nuls.
            direction_flip = (corr_o * corr_e < 0) and (abs(corr_o) >= 0.03) and (abs(corr_e) >= 0.03)

        missing_o = float(o.isna().mean())
        missing_e = float(e.isna().mean())
        missing_delta = missing_e - missing_o

        severity = drift_severity(
            psi_value=psi_value,
            ks_value=ks_stat,
            missing_delta=missing_delta,
            smd=smd,
            direction_flip=direction_flip,
        )

        rows.append({
            "feature": feature,
            "kind": "LIVE37" if feature in LIVE_COMPACT10 else "PREMATCH",
            "original_n": int(o_non.size),
            "external_n": int(e_non.size),
            "original_missing_rate": missing_o,
            "external_missing_rate": missing_e,
            "missing_delta": missing_delta,
            "original_mean": o_mean,
            "external_mean": e_mean,
            "mean_delta": e_mean - o_mean if np.isfinite(o_mean) and np.isfinite(e_mean) else float("nan"),
            "original_std": o_std,
            "external_std": e_std,
            "standardized_mean_shift": smd,
            "original_median": safe_quantile(o, 0.50),
            "external_median": safe_quantile(e, 0.50),
            "original_p10": safe_quantile(o, 0.10),
            "external_p10": safe_quantile(e, 0.10),
            "original_p90": safe_quantile(o, 0.90),
            "external_p90": safe_quantile(e, 0.90),
            "psi": psi_value,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "corr_target_original": corr_o,
            "corr_target_external": corr_e,
            "univariate_auc_original": auc_o,
            "univariate_auc_external": auc_e,
            "target_direction_flip": int(direction_flip),
            "drift_severity": severity,
        })

    out = pd.DataFrame(rows)
    severity_order = pd.Categorical(out["drift_severity"], categories=["SEVERE", "MODERATE", "LOW"], ordered=True)
    out = out.assign(_sev=severity_order).sort_values(
        ["_sev", "psi", "ks_stat"], ascending=[True, False, False]
    ).drop(columns=["_sev"]).reset_index(drop=True)
    return out


def parse_provenance(value: Any) -> Dict[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def provenance_table(original: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    if "_live_provenance_json" not in original.columns or "_live_provenance_json" not in external.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def counts(df: pd.DataFrame, feature: str) -> Counter:
        c: Counter = Counter()
        for value in df["_live_provenance_json"]:
            m = parse_provenance(value)
            prov = m.get(feature, "<missing_provenance>")
            c[prov] += 1
        return c

    for feature in sorted(LIVE_COMPACT10):
        co = counts(original, feature)
        ce = counts(external, feature)
        keys = sorted(set(co) | set(ce))
        no = max(1, sum(co.values()))
        ne = max(1, sum(ce.values()))
        for source in keys:
            so = co[source] / no
            se = ce[source] / ne
            rows.append({
                "feature": feature,
                "provenance": source,
                "original_count": int(co[source]),
                "external_count": int(ce[source]),
                "original_share": so,
                "external_share": se,
                "share_delta": se - so,
            })

    return pd.DataFrame(rows).sort_values(
        ["feature", "external_share", "original_share"], ascending=[True, False, False]
    ).reset_index(drop=True)


def categorical_distribution_drift(
    original: pd.Series,
    external: pd.Series,
) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    o = original.fillna("<NA>").astype(str)
    e = external.fillna("<NA>").astype(str)
    cats = sorted(set(o.unique()) | set(e.unique()))
    po = o.value_counts(normalize=True)
    pe = e.value_counts(normalize=True)

    p = np.array([float(po.get(c, 0.0)) for c in cats], dtype=float)
    q = np.array([float(pe.get(c, 0.0)) for c in cats], dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], EPS, None))))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    dist = {c: (float(po.get(c, 0.0)), float(pe.get(c, 0.0))) for c in cats}
    return float(jsd), dist


def competition_table(original: pd.DataFrame, external: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if "competition_id" not in original.columns or "competition_id" not in external.columns:
        return pd.DataFrame(), float("nan")

    jsd, _ = categorical_distribution_drift(original["competition_id"], external["competition_id"])

    o_count = original["competition_id"].astype(str).value_counts(dropna=False)
    e_count = external["competition_id"].astype(str).value_counts(dropna=False)
    o_share = o_count / max(1, len(original))
    e_share = e_count / max(1, len(external))

    rows = []
    for comp in sorted(set(o_count.index) | set(e_count.index)):
        ext_mask = external["competition_id"].astype(str) == str(comp)
        tar = pd.to_numeric(external.loc[ext_mask, TARGET], errors="coerce")
        resolved = tar.isin([0, 1])
        rows.append({
            "competition_id": comp,
            "original_n": int(o_count.get(comp, 0)),
            "external_n": int(e_count.get(comp, 0)),
            "original_share": float(o_share.get(comp, 0.0)),
            "external_share": float(e_share.get(comp, 0.0)),
            "share_delta": float(e_share.get(comp, 0.0) - o_share.get(comp, 0.0)),
            "external_positive_rate": float(tar[resolved].mean()) if resolved.any() else float("nan"),
        })

    out = pd.DataFrame(rows).sort_values(
        ["external_n", "share_delta"], ascending=[False, False]
    ).reset_index(drop=True)
    return out, jsd


def score_state_series(df: pd.DataFrame) -> pd.Series:
    if "score_37_home" not in df.columns or "score_37_away" not in df.columns:
        return pd.Series(["<missing>"] * len(df), index=df.index)

    h = pd.to_numeric(df["score_37_home"], errors="coerce")
    a = pd.to_numeric(df["score_37_away"], errors="coerce")

    state = pd.Series("<missing>", index=df.index, dtype="object")
    mask = h.notna() & a.notna()
    state.loc[mask & (h == 0) & (a == 0)] = "0-0"
    state.loc[mask & (h == a) & ~((h == 0) & (a == 0))] = "draw_with_goals"
    state.loc[mask & (h > a)] = "home_lead"
    state.loc[mask & (h < a)] = "away_lead"
    return state


def score_state_table(original: pd.DataFrame, external: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    os = score_state_series(original)
    es = score_state_series(external)
    jsd, _ = categorical_distribution_drift(os, es)

    rows = []
    for state in sorted(set(os.unique()) | set(es.unique())):
        om = os == state
        em = es == state
        et = pd.to_numeric(external.loc[em, TARGET], errors="coerce")
        er = et.isin([0, 1])
        rows.append({
            "score_state_37": state,
            "original_n": int(om.sum()),
            "external_n": int(em.sum()),
            "original_share": float(om.mean()),
            "external_share": float(em.mean()),
            "share_delta": float(em.mean() - om.mean()),
            "external_positive_rate": float(et[er].mean()) if er.any() else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("external_n", ascending=False).reset_index(drop=True), jsd


def daily_external_table(external: pd.DataFrame) -> pd.DataFrame:
    if "_kickoff" not in external.columns:
        return pd.DataFrame()
    tmp = external.copy()
    tmp["date"] = tmp["_kickoff"].dt.date.astype(str)
    rows = []
    for date, g in tmp.groupby("date", dropna=False):
        t = pd.to_numeric(g[TARGET], errors="coerce")
        r = t.isin([0, 1])
        rows.append({
            "date": date,
            "n": int(len(g)),
            "positives": int((t == 1).sum()),
            "positive_rate": float(t[r].mean()) if r.any() else float("nan"),
            "competitions": int(g["competition_id"].nunique(dropna=True)) if "competition_id" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def source_table(original: pd.DataFrame) -> pd.DataFrame:
    if "source_dataset" not in original.columns:
        return pd.DataFrame()
    rows = []
    for src, g in original.groupby("source_dataset", dropna=False):
        t = pd.to_numeric(g[TARGET], errors="coerce")
        r = t.isin([0, 1])
        rows.append({
            "source_dataset": src,
            "n": int(len(g)),
            "positives": int((t == 1).sum()),
            "positive_rate": float(t[r].mean()) if r.any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def find_probability_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "probability",
        "probability_calibrated",
        "calibrated_probability",
        "p_goal_after_37",
        "predicted_probability",
        "prob",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    numeric_candidates = []
    for col in df.columns:
        low = col.lower()
        if "prob" in low and "band" not in low:
            numeric_candidates.append(col)
    return numeric_candidates[0] if numeric_candidates else None


def prediction_shift_table(final_path: Path, external_path: Path) -> pd.DataFrame:
    if not final_path.exists() or not external_path.exists():
        return pd.DataFrame()
    try:
        f = pd.read_csv(final_path, low_memory=False)
        e = pd.read_csv(external_path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    fc = find_probability_column(f)
    ec = find_probability_column(e)
    if not fc or not ec:
        return pd.DataFrame()

    rows = []
    for label, df, col in [("final_holdout", f, fc), ("external", e, ec)]:
        p = finite_num(df[col])
        rows.append({
            "dataset": label,
            "probability_column": col,
            "n": int(p.notna().sum()),
            "mean_probability": safe_mean(p),
            "std_probability": safe_std(p),
            "p10": safe_quantile(p, 0.10),
            "median": safe_quantile(p, 0.50),
            "p90": safe_quantile(p, 0.90),
            "min": float(p.min()) if p.notna().any() else float("nan"),
            "max": float(p.max()) if p.notna().any() else float("nan"),
        })

    # drift stats between probability vectors
    fp = finite_num(f[fc])
    ep = finite_num(e[ec])
    ks_stat, ks_p = empirical_ks(fp.dropna().to_numpy(float), ep.dropna().to_numpy(float))
    pv = psi(fp, ep)
    rows.append({
        "dataset": "shift_external_vs_final",
        "probability_column": f"{fc} vs {ec}",
        "n": int(ep.notna().sum()),
        "mean_probability": float("nan"),
        "std_probability": float("nan"),
        "p10": float("nan"),
        "median": float("nan"),
        "p90": float("nan"),
        "min": ks_stat,
        "max": pv,
    })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_Aucune donnée disponible._"
    view = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()

    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit de dataset shift First_Half37 V1")
    parser.add_argument("--original-csv", default=None)
    parser.add_argument("--external-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--final-holdout-predictions", default=None)
    parser.add_argument("--external-predictions", default=None)
    args = parser.parse_args()

    original_path = clean_path(args.original_csv, DEFAULT_ORIGINAL)
    external_path = clean_path(args.external_csv, DEFAULT_EXTERNAL)
    output_dir = clean_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    final_pred_path = clean_path(args.final_holdout_predictions, DEFAULT_FINAL_HOLDOUT_PRED)
    external_pred_path = clean_path(args.external_predictions, DEFAULT_EXTERNAL_PRED)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("FIRST HALF 37 — DATASET / FEATURE DRIFT AUDIT V1")
    print("=" * 108)
    print(f"Original : {original_path}")
    print(f"Externe  : {external_path}")
    print(f"Output   : {output_dir}")
    print("IMPORTANT: aucun refit, aucun tuning, aucune modification du modèle.")
    print()

    original = read_dataset(original_path, "Dataset original")
    external = read_dataset(external_path, "Dataset externe")

    overlap = set(original[MATCH_COL].astype(str)) & set(external[MATCH_COL].astype(str))
    if overlap:
        raise RuntimeError(f"Overlap original/externe détecté : {len(overlap)} match_id")

    overview = pd.DataFrame([
        summarize_dataset(original, "original_1802"),
        summarize_dataset(external, "external_post_model"),
    ])
    overview.to_csv(output_dir / "first_half_37_drift_overview.csv", index=False)

    print(
        f"Original : {len(original)} matchs | positifs={(pd.to_numeric(original[TARGET], errors='coerce') == 1).sum()} "
        f"({pct(pd.to_numeric(original[TARGET], errors='coerce').mean())})"
    )
    print(
        f"Externe  : {len(external)} matchs | positifs={(pd.to_numeric(external[TARGET], errors='coerce') == 1).sum()} "
        f"({pct(pd.to_numeric(external[TARGET], errors='coerce').mean())})"
    )
    print(f"Overlap  : {len(overlap)}")
    print()

    drift = feature_drift_table(original, external)
    drift.to_csv(output_dir / "first_half_37_compact10_drift.csv", index=False)

    prov = provenance_table(original, external)
    prov.to_csv(output_dir / "first_half_37_live_provenance_drift.csv", index=False)

    comp, comp_jsd = competition_table(original, external)
    comp.to_csv(output_dir / "first_half_37_competition_drift.csv", index=False)

    score, score_jsd = score_state_table(original, external)
    score.to_csv(output_dir / "first_half_37_score_state_drift.csv", index=False)

    daily = daily_external_table(external)
    daily.to_csv(output_dir / "first_half_37_external_daily_target.csv", index=False)

    sources = source_table(original)
    sources.to_csv(output_dir / "first_half_37_original_sources.csv", index=False)

    pred_shift = prediction_shift_table(final_pred_path, external_pred_path)
    if not pred_shift.empty:
        pred_shift.to_csv(output_dir / "first_half_37_prediction_distribution_shift.csv", index=False)

    severe = int((drift["drift_severity"] == "SEVERE").sum())
    moderate = int((drift["drift_severity"] == "MODERATE").sum())
    low = int((drift["drift_severity"] == "LOW").sum())
    flips = int(drift["target_direction_flip"].sum())

    print("COMPACT10 — DRIFT")
    print(
        drift[[
            "feature", "kind", "psi", "ks_stat", "standardized_mean_shift",
            "missing_delta", "corr_target_original", "corr_target_external",
            "target_direction_flip", "drift_severity",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print()
    print(f"Résumé drift : SEVERE={severe} | MODERATE={moderate} | LOW={low} | direction flips={flips}")
    print(f"JSD compétitions : {comp_jsd:.4f}")
    print(f"JSD score state  : {score_jsd:.4f}")

    # Diagnostic simple, non décisionnel.
    if severe >= 3 or flips >= 2:
        verdict = "DRIFT_IMPORTANT"
    elif severe >= 1 or moderate >= 3:
        verdict = "DRIFT_PRESENT"
    else:
        verdict = "DRIFT_LIMITED"

    manifest = {
        "verdict": verdict,
        "original_rows": int(len(original)),
        "external_rows": int(len(external)),
        "overlap": int(len(overlap)),
        "compact10_severe": severe,
        "compact10_moderate": moderate,
        "compact10_low": low,
        "target_direction_flips": flips,
        "competition_jsd": comp_jsd,
        "score_state_jsd": score_jsd,
        "compact10": COMPACT10,
        "note": "Diagnostic only. Do not use external sample for tuning without declaring a new experiment.",
    }
    (output_dir / "first_half_37_drift_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    o_rate = float(pd.to_numeric(original[TARGET], errors="coerce").dropna().mean())
    e_rate = float(pd.to_numeric(external[TARGET], errors="coerce").dropna().mean())

    report_lines = [
        "# First Half 37 — Dataset / Feature Drift Audit V1",
        "",
        "## Règle",
        "",
        "Cet audit compare le dataset original au nouvel échantillon externe. Il ne réentraîne pas le modèle et ne modifie ni features, ni C, ni calibration.",
        "",
        "## Vue d'ensemble",
        "",
        f"- Original : **{len(original)}** matchs | taux positif **{100*o_rate:.2f}%**",
        f"- Externe : **{len(external)}** matchs | taux positif **{100*e_rate:.2f}%**",
        f"- Delta base rate : **{100*(e_rate-o_rate):+.2f} points**",
        f"- Overlap match_id : **{len(overlap)}**",
        f"- Diagnostic global : **`{verdict}`**",
        "",
        "## Drift COMPACT10",
        "",
        f"- SEVERE : **{severe}**",
        f"- MODERATE : **{moderate}**",
        f"- LOW : **{low}**",
        f"- Flips du sens feature→target : **{flips}**",
        "",
        markdown_table(
            drift,
            [
                "feature", "kind", "psi", "ks_stat", "standardized_mean_shift",
                "original_missing_rate", "external_missing_rate", "missing_delta",
                "corr_target_original", "corr_target_external", "target_direction_flip",
                "drift_severity",
            ],
            max_rows=20,
        ),
        "",
        "## Composition des compétitions",
        "",
        f"- Jensen-Shannon divergence : **{comp_jsd:.4f}**",
        "",
        markdown_table(
            comp,
            ["competition_id", "original_n", "external_n", "original_share", "external_share", "share_delta", "external_positive_rate"],
            max_rows=20,
        ),
        "",
        "## Etat du score à 37'",
        "",
        f"- Jensen-Shannon divergence : **{score_jsd:.4f}**",
        "",
        markdown_table(
            score,
            ["score_state_37", "original_n", "external_n", "original_share", "external_share", "share_delta", "external_positive_rate"],
            max_rows=20,
        ),
        "",
        "## Taux externe par jour",
        "",
        markdown_table(daily, ["date", "n", "positives", "positive_rate", "competitions"], max_rows=30),
        "",
        "## Provenance LIVE COMPACT10",
        "",
        markdown_table(
            prov,
            ["feature", "provenance", "original_count", "external_count", "original_share", "external_share", "share_delta"],
            max_rows=50,
        ),
        "",
        "## Interprétation des seuils",
        "",
        "Les seuils PSI/KS/SMD de ce script sont des heuristiques de diagnostic, pas des règles de production. Un drift marqué indique où enquêter; il ne prouve pas à lui seul qu'une feature est mauvaise.",
        "",
        "## Discipline expérimentale",
        "",
        "Le nouvel échantillon externe est désormais un échantillon d'audit observé. Si l'on modifie le modèle à partir de ce diagnostic, cela devient une nouvelle expérience et il faudra valider la nouvelle version sur des matchs futurs non utilisés ici.",
    ]

    if not pred_shift.empty:
        report_lines.extend([
            "",
            "## Distribution des probabilités (optionnel)",
            "",
            markdown_table(
                pred_shift,
                ["dataset", "probability_column", "n", "mean_probability", "std_probability", "p10", "median", "p90", "min", "max"],
                max_rows=10,
            ),
        ])

    report_path = output_dir / "first_half_37_drift_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("=" * 108)
    print("AUDIT DRIFT TERMINE")
    print("=" * 108)
    print(f"Verdict : {verdict}")
    print(f"Rapport : {report_path}")
    print(f"Features: {output_dir / 'first_half_37_compact10_drift.csv'}")
    print(f"Provenance: {output_dir / 'first_half_37_live_provenance_drift.csv'}")
    print(f"Competitions: {output_dir / 'first_half_37_competition_drift.csv'}")
    print(f"Score state: {output_dir / 'first_half_37_score_state_drift.csv'}")
    print("IMPORTANT : diagnostic seulement — aucun modèle modifié.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
