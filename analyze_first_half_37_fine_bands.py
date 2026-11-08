#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_first_half_37_fine_bands.py

Analyse DESCRIPTIVE du FINAL HOLDOUT First_Half37 déjà consommé.
Ce script ne réentraîne rien, ne modifie aucun modèle et ne doit pas servir à
retuner le pipeline sur ce holdout.

Il découpe les probabilités finales en tranches plus fines, notamment :
25-27 %, 27-29 %, 29-31 %, 31-33 %, 33-35 %.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRED = (
    BASE_DIR / "outputs" / "first_half_37" / "final_holdout_v1"
    / "first_half_37_final_holdout_predictions.csv"
)
DEFAULT_OUT = (
    BASE_DIR / "outputs" / "first_half_37" / "final_holdout_v1"
    / "first_half_37_final_probability_bands_fine.csv"
)
DEFAULT_REPORT = (
    BASE_DIR / "outputs" / "first_half_37" / "final_holdout_v1"
    / "first_half_37_final_probability_bands_fine.md"
)

# Tranches resserrées autour de la zone réellement occupée par le modèle.
BINS = [0.0, 0.20, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.40, 0.50, 1.000001]
LABELS = [
    "<20%",
    "20-23%",
    "23-25%",
    "25-27%",
    "27-29%",
    "29-31%",
    "31-33%",
    "33-35%",
    "35-40%",
    "40-50%",
    "50%+",
]


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    if "final_probability" not in df.columns:
        raise ValueError("Colonne `final_probability` absente.")
    if "goal_after_37_before_ht" not in df.columns:
        raise ValueError("Colonne `goal_after_37_before_ht` absente.")

    p = pd.to_numeric(df["final_probability"], errors="coerce")
    y = pd.to_numeric(df["goal_after_37_before_ht"], errors="coerce")

    valid = p.notna() & y.isin([0, 1])
    p = p[valid].to_numpy(dtype=float)
    y = y[valid].to_numpy(dtype=int)

    cats = pd.cut(
        p,
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    rows = []
    for label in LABELS:
        mask = np.asarray(cats == label)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "probability_band": label,
                "n": 0,
                "positives": 0,
                "observed_rate": np.nan,
                "mean_probability": np.nan,
                "calibration_gap": np.nan,
            })
            continue

        positives = int(y[mask].sum())
        observed = float(y[mask].mean())
        mean_p = float(p[mask].mean())
        rows.append({
            "probability_band": label,
            "n": n,
            "positives": positives,
            "observed_rate": observed,
            "mean_probability": mean_p,
            "calibration_gap": observed - mean_p,
        })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PRED))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.output)
    report_path = Path(args.report)

    if not pred_path.exists():
        print(f"ERREUR : fichier introuvable : {pred_path}")
        return 2

    df = pd.read_csv(pred_path)
    table = build_table(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print("=" * 100)
    print("FIRST HALF 37 — FINAL HOLDOUT — TRANCHES FINES")
    print("=" * 100)
    print(f"Predictions : {pred_path}")
    print(f"Matchs      : {len(df)}")
    print()

    display = table.copy()
    for c in ["observed_rate", "mean_probability", "calibration_gap"]:
        display[c] = display[c].map(lambda x: "-" if pd.isna(x) else f"{100*x:.2f}%")
    print(display.to_string(index=False))

    nonempty = table[table["n"] > 0].copy()
    if not nonempty.empty:
        best = nonempty.sort_values(["observed_rate", "n"], ascending=[False, False]).iloc[0]
        print()
        print(
            f"Meilleure tranche descriptive : {best['probability_band']} | "
            f"{int(best['positives'])}/{int(best['n'])} = {100*best['observed_rate']:.2f}%"
        )

    lines = [
        "# First Half 37 — FINAL HOLDOUT — tranches fines",
        "",
        "> Analyse descriptive uniquement. Le FINAL HOLDOUT est déjà consommé; ces tranches ne doivent pas servir à retuner le modèle.",
        "",
        f"- Predictions : `{pred_path}`",
        f"- Matchs : **{len(df)}**",
        "",
        "| Tranche | n | Positifs | Taux observé | Probabilité moyenne | Écart calibration |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        def pct(v):
            return "—" if pd.isna(v) else f"{100*float(v):.2f}%"
        lines.append(
            f"| {r['probability_band']} | {int(r['n'])} | {int(r['positives'])} | "
            f"{pct(r['observed_rate'])} | {pct(r['mean_probability'])} | {pct(r['calibration_gap'])} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"CSV     : {out_path}")
    print(f"Rapport : {report_path}")
    print("IMPORTANT : descriptif seulement; aucun tuning sur le FINAL HOLDOUT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
