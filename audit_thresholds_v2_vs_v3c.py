#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
AUDIT COMPLET DES SEUILS — V2 vs V3-C
===============================================================================

Objectif
--------
Comparer V2_BASELINE_52 et V3_C_AGGRESSIVE sur EXACTEMENT les mêmes matchs
du TEST temporel déjà généré par train_compare_v2_v3.py.

Le script utilise :
    outputs/model_v3_experiments/predictions_test.csv

Il NE réentraîne aucun modèle et NE modifie aucun fichier de production.

Comparaisons générées
---------------------
1) Seuils :
   50.0 %
   52.5 %
   55.0 %
   57.5 %
   60.0 %
   62.5 %
   65.0 %
   67.5 %
   70.0 %
   72.5 %
   75.0 %
   77.5 %
   80.0 %

2) Tranches de probabilité :
   50–55 %
   55–60 %
   60–65 %
   65–70 %
   70–75 %
   75–80 %
   80 %+ 

3) Pour chaque seuil :
   - nombre de matchs sélectionnés
   - nombre de positifs
   - nombre de négatifs
   - taux observé
   - IC 95 % Wilson du taux observé
   - recall
   - lift vs taux de base du TEST
   - probabilité moyenne / médiane
   - calibration gap = probabilité moyenne - taux observé

4) Comparaison V3-C - V2 :
   - différence de volume
   - différence de taux observé
   - différence de recall
   - différence de lift

Sorties
-------
outputs/model_v3_experiments/threshold_audit/
    threshold_metrics_v2_v3c.csv
    threshold_comparison_v3c_minus_v2.csv
    probability_bands_v2_v3c.csv
    probability_bands_comparison.csv
    overlap_by_threshold.csv
    audit_thresholds_report.md
    audit_thresholds_metadata.json

Commande
--------
cd "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
python3 audit_thresholds_v2_vs_v3c.py
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(
    os.getenv(
        "GOAL87_PROJECT_ROOT",
        "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
    )
)

DEFAULT_PREDICTIONS_REL = (
    "outputs/model_v3_experiments/predictions_test.csv"
)

DEFAULT_OUTPUT_REL = (
    "outputs/model_v3_experiments/threshold_audit"
)

MODEL_V2 = "V2_BASELINE_52"
MODEL_V3C = "V3_C_AGGRESSIVE"

THRESHOLDS = [
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
]

BANDS = [
    (0.50, 0.55, "50–55%"),
    (0.55, 0.60, "55–60%"),
    (0.60, 0.65, "60–65%"),
    (0.65, 0.70, "65–70%"),
    (0.70, 0.75, "70–75%"),
    (0.75, 0.80, "75–80%"),
    (0.80, None, "80%+"),
]


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def resolve_path(root: Path, raw: Optional[str], default_rel: str) -> Path:
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = root / p
    else:
        p = root / default_rel
    return p


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """
    Intervalle de confiance Wilson 95 % pour une proportion.
    Plus approprié qu'une approximation normale pour les petits volumes.
    """
    if n <= 0:
        return float("nan"), float("nan")

    p = successes / n
    z2 = z * z

    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n)
            + (z2 / (4.0 * n * n))
        )
        / denom
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100.0 * x:.{digits}f}%"


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier predictions_test.csv introuvable : {path}"
        )

    df = pd.read_csv(path)

    required = {
        "model_name",
        "row_index_sorted",
        "target",
        "probability",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans predictions_test.csv : {missing}"
        )

    df = df.copy()

    df["model_name"] = df["model_name"].astype(str)
    df["row_index_sorted"] = pd.to_numeric(
        df["row_index_sorted"],
        errors="raise",
    ).astype(int)

    df["target"] = pd.to_numeric(
        df["target"],
        errors="raise",
    ).astype(int)

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="raise",
    ).astype(float)

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Target non binaire : {sorted(df['target'].unique().tolist())}"
        )

    if ((df["probability"] < 0) | (df["probability"] > 1)).any():
        raise ValueError("Probabilités hors intervalle [0,1].")

    return df


def validate_same_test_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v2 = df[df["model_name"] == MODEL_V2].copy()
    v3 = df[df["model_name"] == MODEL_V3C].copy()

    if v2.empty:
        raise ValueError(f"Aucune ligne pour {MODEL_V2}.")
    if v3.empty:
        raise ValueError(f"Aucune ligne pour {MODEL_V3C}.")

    # Un seul enregistrement par ligne TEST / modèle.
    if v2["row_index_sorted"].duplicated().any():
        raise ValueError("Doublons détectés dans V2 sur row_index_sorted.")
    if v3["row_index_sorted"].duplicated().any():
        raise ValueError("Doublons détectés dans V3-C sur row_index_sorted.")

    v2 = v2.sort_values("row_index_sorted").reset_index(drop=True)
    v3 = v3.sort_values("row_index_sorted").reset_index(drop=True)

    if len(v2) != len(v3):
        raise ValueError(
            f"Volumes TEST différents : V2={len(v2)}, V3-C={len(v3)}"
        )

    if not np.array_equal(
        v2["row_index_sorted"].to_numpy(),
        v3["row_index_sorted"].to_numpy(),
    ):
        raise ValueError(
            "V2 et V3-C ne portent pas sur les mêmes lignes TEST."
        )

    if not np.array_equal(
        v2["target"].to_numpy(),
        v3["target"].to_numpy(),
    ):
        raise ValueError(
            "Les targets diffèrent entre V2 et V3-C sur le même TEST."
        )

    # Si match_id existe, on le vérifie aussi.
    if "match_id" in v2.columns and "match_id" in v3.columns:
        a = v2["match_id"].astype(str).to_numpy()
        b = v3["match_id"].astype(str).to_numpy()

        if not np.array_equal(a, b):
            raise ValueError(
                "Les match_id diffèrent entre V2 et V3-C."
            )

    return v2, v3


# =============================================================================
# THRESHOLD METRICS
# =============================================================================

def threshold_row(
    model_name: str,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    mask = p >= threshold

    selected = int(mask.sum())
    positives = int(y[mask].sum()) if selected else 0
    negatives = selected - positives

    total_positives = int(y.sum())
    base_rate = float(y.mean())

    if selected:
        observed = positives / selected
        mean_p = float(p[mask].mean())
        median_p = float(np.median(p[mask]))
        min_p = float(p[mask].min())
        max_p = float(p[mask].max())
        ci_low, ci_high = wilson_interval(positives, selected)
    else:
        observed = float("nan")
        mean_p = float("nan")
        median_p = float("nan")
        min_p = float("nan")
        max_p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    recall = (
        positives / total_positives
        if total_positives > 0
        else float("nan")
    )

    lift = (
        observed / base_rate
        if selected and base_rate > 0
        else float("nan")
    )

    calibration_gap = (
        mean_p - observed
        if selected
        else float("nan")
    )

    return {
        "model_name": model_name,
        "threshold": threshold,
        "threshold_pct": threshold * 100.0,
        "test_rows": len(y),
        "test_positives": total_positives,
        "test_base_rate": base_rate,
        "selected": selected,
        "selected_rate": selected / len(y),
        "positives": positives,
        "negatives": negatives,
        "observed_win_rate": observed,
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "recall": recall,
        "lift_vs_base": lift,
        "mean_probability_selected": mean_p,
        "median_probability_selected": median_p,
        "min_probability_selected": min_p,
        "max_probability_selected": max_p,
        "calibration_gap_mean_p_minus_observed": calibration_gap,
    }


def build_threshold_metrics(
    v2: pd.DataFrame,
    v3: pd.DataFrame,
) -> pd.DataFrame:
    y = v2["target"].to_numpy(dtype=int)

    rows: List[Dict[str, float]] = []

    for model_name, frame in [
        (MODEL_V2, v2),
        (MODEL_V3C, v3),
    ]:
        p = frame["probability"].to_numpy(dtype=float)

        for threshold in THRESHOLDS:
            rows.append(
                threshold_row(
                    model_name,
                    y,
                    p,
                    threshold,
                )
            )

    return pd.DataFrame(rows)


def build_threshold_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    v2 = (
        metrics[metrics["model_name"] == MODEL_V2]
        .set_index("threshold")
        .sort_index()
    )
    v3 = (
        metrics[metrics["model_name"] == MODEL_V3C]
        .set_index("threshold")
        .sort_index()
    )

    thresholds = sorted(set(v2.index) & set(v3.index))

    rows = []

    for t in thresholds:
        a = v2.loc[t]
        b = v3.loc[t]

        rows.append({
            "threshold": t,
            "threshold_pct": t * 100.0,

            "v2_selected": int(a["selected"]),
            "v3c_selected": int(b["selected"]),
            "delta_selected_v3c_minus_v2": int(b["selected"] - a["selected"]),

            "v2_observed_win_rate": a["observed_win_rate"],
            "v3c_observed_win_rate": b["observed_win_rate"],
            "delta_win_rate_v3c_minus_v2": (
                b["observed_win_rate"] - a["observed_win_rate"]
                if pd.notna(a["observed_win_rate"]) and pd.notna(b["observed_win_rate"])
                else float("nan")
            ),

            "v2_recall": a["recall"],
            "v3c_recall": b["recall"],
            "delta_recall_v3c_minus_v2": b["recall"] - a["recall"],

            "v2_lift": a["lift_vs_base"],
            "v3c_lift": b["lift_vs_base"],
            "delta_lift_v3c_minus_v2": (
                b["lift_vs_base"] - a["lift_vs_base"]
                if pd.notna(a["lift_vs_base"]) and pd.notna(b["lift_vs_base"])
                else float("nan")
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# PROBABILITY BANDS
# =============================================================================

def band_row(
    model_name: str,
    y: np.ndarray,
    p: np.ndarray,
    low: float,
    high: Optional[float],
    label: str,
) -> Dict[str, float]:
    if high is None:
        mask = p >= low
    else:
        mask = (p >= low) & (p < high)

    n = int(mask.sum())
    wins = int(y[mask].sum()) if n else 0
    losses = n - wins

    if n:
        win_rate = wins / n
        mean_p = float(p[mask].mean())
        median_p = float(np.median(p[mask]))
        ci_low, ci_high = wilson_interval(wins, n)
    else:
        win_rate = float("nan")
        mean_p = float("nan")
        median_p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    base_rate = float(y.mean())

    return {
        "model_name": model_name,
        "band": label,
        "band_low": low,
        "band_high": high if high is not None else float("nan"),
        "matches": n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "mean_probability": mean_p,
        "median_probability": median_p,
        "lift_vs_base": (
            win_rate / base_rate
            if n and base_rate > 0
            else float("nan")
        ),
        "calibration_gap_mean_p_minus_observed": (
            mean_p - win_rate
            if n
            else float("nan")
        ),
    }


def build_bands(v2: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    y = v2["target"].to_numpy(dtype=int)

    rows = []

    for model_name, frame in [
        (MODEL_V2, v2),
        (MODEL_V3C, v3),
    ]:
        p = frame["probability"].to_numpy(dtype=float)

        for low, high, label in BANDS:
            rows.append(
                band_row(
                    model_name,
                    y,
                    p,
                    low,
                    high,
                    label,
                )
            )

    return pd.DataFrame(rows)


def build_band_comparison(bands: pd.DataFrame) -> pd.DataFrame:
    v2 = (
        bands[bands["model_name"] == MODEL_V2]
        .set_index("band")
    )
    v3 = (
        bands[bands["model_name"] == MODEL_V3C]
        .set_index("band")
    )

    rows = []

    for _, _, label in BANDS:
        a = v2.loc[label]
        b = v3.loc[label]

        rows.append({
            "band": label,

            "v2_matches": int(a["matches"]),
            "v3c_matches": int(b["matches"]),
            "delta_matches_v3c_minus_v2": int(
                b["matches"] - a["matches"]
            ),

            "v2_wins": int(a["wins"]),
            "v3c_wins": int(b["wins"]),

            "v2_win_rate": a["win_rate"],
            "v3c_win_rate": b["win_rate"],
            "delta_win_rate_v3c_minus_v2": (
                b["win_rate"] - a["win_rate"]
                if pd.notna(a["win_rate"]) and pd.notna(b["win_rate"])
                else float("nan")
            ),

            "v2_mean_probability": a["mean_probability"],
            "v3c_mean_probability": b["mean_probability"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# OVERLAP
# =============================================================================

def build_overlap(
    v2: pd.DataFrame,
    v3: pd.DataFrame,
) -> pd.DataFrame:
    y = v2["target"].to_numpy(dtype=int)
    p2 = v2["probability"].to_numpy(dtype=float)
    p3 = v3["probability"].to_numpy(dtype=float)

    rows = []

    for t in THRESHOLDS:
        s2 = p2 >= t
        s3 = p3 >= t

        both = s2 & s3
        only_v2 = s2 & ~s3
        only_v3 = s3 & ~s2
        neither = ~s2 & ~s3

        def rate(mask: np.ndarray) -> float:
            n = int(mask.sum())
            return float(y[mask].mean()) if n else float("nan")

        rows.append({
            "threshold": t,
            "threshold_pct": t * 100.0,

            "both_selected": int(both.sum()),
            "both_win_rate": rate(both),

            "only_v2_selected": int(only_v2.sum()),
            "only_v2_win_rate": rate(only_v2),

            "only_v3c_selected": int(only_v3.sum()),
            "only_v3c_win_rate": rate(only_v3),

            "neither_selected": int(neither.sum()),
            "neither_win_rate": rate(neither),
        })

    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================

def build_report(
    threshold_metrics: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    bands: pd.DataFrame,
    band_comparison: pd.DataFrame,
    overlap: pd.DataFrame,
    test_rows: int,
    test_positives: int,
    source_path: Path,
) -> str:
    base_rate = test_positives / test_rows

    # Compact threshold tables.
    tm = threshold_metrics.copy()
    tm["threshold"] = tm["threshold"].map(lambda x: f"{x:.3f}")
    tm["observed_win_rate"] = tm["observed_win_rate"].map(
        lambda x: pct(x) if pd.notna(x) else "NA"
    )
    tm["recall"] = tm["recall"].map(
        lambda x: pct(x) if pd.notna(x) else "NA"
    )
    tm["selected_rate"] = tm["selected_rate"].map(
        lambda x: pct(x) if pd.notna(x) else "NA"
    )
    tm["lift_vs_base"] = tm["lift_vs_base"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
    )

    threshold_view = tm[
        [
            "model_name",
            "threshold",
            "selected",
            "positives",
            "observed_win_rate",
            "recall",
            "lift_vs_base",
            "selected_rate",
        ]
    ]

    bc = band_comparison.copy()
    for col in [
        "v2_win_rate",
        "v3c_win_rate",
        "delta_win_rate_v3c_minus_v2",
    ]:
        bc[col] = bc[col].map(
            lambda x: pct(x) if pd.notna(x) else "NA"
        )

    lines = [
        "# AUDIT COMPLET DES SEUILS — V2 vs V3-C",
        "",
        "## Base de comparaison",
        f"- Source : `{source_path}`",
        f"- Matchs TEST : **{test_rows}**",
        f"- Positifs TEST : **{test_positives}**",
        f"- Taux de base TEST : **{base_rate:.2%}**",
        "- Les deux modèles sont vérifiés sur exactement les mêmes lignes TEST.",
        "",
        "## Tous les seuils",
        "",
        md_table(threshold_view),
        "",
        "## Comparaison V3-C moins V2",
        "",
        md_table(threshold_comparison),
        "",
        "## Tranches de probabilité",
        "",
        md_table(band_comparison),
        "",
        "## Chevauchement des sélections",
        "",
        md_table(overlap),
        "",
        "## Lecture méthodologique",
        "",
        "- Un seuil n'est pas jugé uniquement sur son taux observé.",
        "- Un petit volume peut donner un taux très instable.",
        "- Le recall montre combien de vrais positifs du TEST sont capturés.",
        "- Le lift compare le taux observé au taux de base du TEST.",
        "- Les IC 95 % Wilson sont disponibles dans le CSV détaillé.",
        "- Le script n'altère ni V2 ni V3-C et ne choisit pas automatiquement un seuil.",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit complet des seuils V2 vs V3-C "
            "sur les mêmes prédictions TEST."
        )
    )

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
    )

    parser.add_argument(
        "--predictions",
        default=None,
        help="Chemin vers predictions_test.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()

    root = Path(args.project_root).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"Racine projet introuvable : {root}"
        )

    predictions_path = resolve_path(
        root,
        args.predictions,
        DEFAULT_PREDICTIONS_REL,
    )

    output_dir = resolve_path(
        root,
        args.output_dir,
        DEFAULT_OUTPUT_REL,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 88)
    log("AUDIT COMPLET DES SEUILS — V2 vs V3-C")
    log("=" * 88)
    log(f"Source  : {predictions_path}")
    log(f"Sorties : {output_dir}")
    log()

    df = load_predictions(predictions_path)
    v2, v3 = validate_same_test_set(df)

    test_rows = len(v2)
    test_positives = int(v2["target"].sum())
    base_rate = float(v2["target"].mean())

    log(
        f"TEST vérifié : {test_rows} matchs | "
        f"{test_positives} positifs | base={base_rate:.2%}"
    )
    log(
        f"V2 et V3-C utilisent exactement les mêmes {test_rows} lignes."
    )
    log()

    threshold_metrics = build_threshold_metrics(v2, v3)
    threshold_comparison = build_threshold_comparison(
        threshold_metrics
    )

    bands = build_bands(v2, v3)
    band_comparison = build_band_comparison(bands)

    overlap = build_overlap(v2, v3)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    p_threshold = (
        output_dir /
        "threshold_metrics_v2_v3c.csv"
    )
    p_compare = (
        output_dir /
        "threshold_comparison_v3c_minus_v2.csv"
    )
    p_bands = (
        output_dir /
        "probability_bands_v2_v3c.csv"
    )
    p_bands_compare = (
        output_dir /
        "probability_bands_comparison.csv"
    )
    p_overlap = (
        output_dir /
        "overlap_by_threshold.csv"
    )
    p_report = (
        output_dir /
        "audit_thresholds_report.md"
    )
    p_meta = (
        output_dir /
        "audit_thresholds_metadata.json"
    )

    threshold_metrics.to_csv(
        p_threshold,
        index=False,
    )

    threshold_comparison.to_csv(
        p_compare,
        index=False,
    )

    bands.to_csv(
        p_bands,
        index=False,
    )

    band_comparison.to_csv(
        p_bands_compare,
        index=False,
    )

    overlap.to_csv(
        p_overlap,
        index=False,
    )

    report = build_report(
        threshold_metrics=threshold_metrics,
        threshold_comparison=threshold_comparison,
        bands=bands,
        band_comparison=band_comparison,
        overlap=overlap,
        test_rows=test_rows,
        test_positives=test_positives,
        source_path=predictions_path,
    )

    p_report.write_text(
        report,
        encoding="utf-8",
    )

    metadata = {
        "source_predictions": str(predictions_path),
        "models": [MODEL_V2, MODEL_V3C],
        "test_rows": test_rows,
        "test_positives": test_positives,
        "test_negatives": test_rows - test_positives,
        "test_base_rate": base_rate,
        "thresholds": THRESHOLDS,
        "bands": [
            {
                "low": low,
                "high": high,
                "label": label,
            }
            for low, high, label in BANDS
        ],
        "same_test_set_verified": True,
    }

    p_meta.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # CONSOLE SUMMARY
    # -------------------------------------------------------------------------
    log("=" * 88)
    log("RÉSULTATS PAR SEUIL")
    log("=" * 88)

    console = threshold_metrics[
        [
            "model_name",
            "threshold_pct",
            "selected",
            "positives",
            "observed_win_rate",
            "recall",
            "lift_vs_base",
            "mean_probability_selected",
        ]
    ].copy()

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(
            console.to_string(
                index=False,
                formatters={
                    "threshold_pct": lambda x: f"{x:5.1f}%",
                    "observed_win_rate": lambda x: (
                        f"{100*x:6.2f}%"
                        if pd.notna(x)
                        else "   NA"
                    ),
                    "recall": lambda x: (
                        f"{100*x:6.2f}%"
                        if pd.notna(x)
                        else "   NA"
                    ),
                    "lift_vs_base": lambda x: (
                        f"{x:6.3f}"
                        if pd.notna(x)
                        else "   NA"
                    ),
                    "mean_probability_selected": lambda x: (
                        f"{100*x:6.2f}%"
                        if pd.notna(x)
                        else "   NA"
                    ),
                },
            )
        )

    log()
    log("=" * 88)
    log("TRANCHES DE PROBABILITÉ")
    log("=" * 88)

    bands_console = bands[
        [
            "model_name",
            "band",
            "matches",
            "wins",
            "losses",
            "win_rate",
            "lift_vs_base",
            "mean_probability",
        ]
    ].copy()

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(
            bands_console.to_string(
                index=False,
                formatters={
                    "win_rate": lambda x: (
                        f"{100*x:6.2f}%"
                        if pd.notna(x)
                        else "   NA"
                    ),
                    "lift_vs_base": lambda x: (
                        f"{x:6.3f}"
                        if pd.notna(x)
                        else "   NA"
                    ),
                    "mean_probability": lambda x: (
                        f"{100*x:6.2f}%"
                        if pd.notna(x)
                        else "   NA"
                    ),
                },
            )
        )

    log()
    log("Fichiers générés :")
    for p in [
        p_threshold,
        p_compare,
        p_bands,
        p_bands_compare,
        p_overlap,
        p_report,
        p_meta,
    ]:
        log(f"  - {p}")

    log()
    log(
        "Audit terminé. Aucun modèle ni seuil de production n'a été modifié."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrompu par l'utilisateur.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERREUR AUDIT SEUILS V2/V3-C : {exc}",
            file=sys.stderr,
        )
        raise
