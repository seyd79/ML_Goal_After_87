#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_v3c_shots35_probability_cross.py

V3-C — CROSS AUDIT: PROBABILITÉ 55–65% × TOTAL SHOTS >=35 @87'
================================================================

Mesure, sur le bloc OOS V3-C de predictions_test.csv, les taux réels de
goal_after_87 pour:
- 55–60%
- 60–65%
- 55–65%
- SHOTS35
- SHOTS35 ∩ 55–60%
- SHOTS35 ∩ 60–65%
- SHOTS35 ∩ 55–65%
- <35 shots ∩ 55–65%

Aucun retrain. Aucun changement du scanner.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    BASE_DIR / "outputs" / "model_v3_experiments" / "predictions_test.csv"
)

DATASET_CANDIDATES = [
    BASE_DIR / "outputs" / "historical_dataset_v2" / "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv",
    BASE_DIR / "outputs" / "historical_dataset_v2" / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv",
    BASE_DIR / "outputs" / "historical_dataset_v2" / "Historical_Dataset_Goal_After_87_RAW_v2.csv",
]

OUTPUT_DIR = BASE_DIR / "outputs" / "v3c_shots35_cross"
MODEL_NAME = "V3_C_AGGRESSIVE"


def clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def probability_series(s: pd.Series) -> pd.Series:
    out = (
        s.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    p = pd.to_numeric(out, errors="coerce")
    return p.where(p <= 1.0, p / 100.0)


def target_series(s: pd.Series) -> pd.Series:
    def f(v: Any) -> float:
        t = clean(v).lower()
        if t in {"1", "1.0", "true", "yes", "oui", "positive"}:
            return 1.0
        if t in {"0", "0.0", "false", "no", "non", "negative"}:
            return 0.0
        try:
            x = float(t.replace(",", "."))
            if x in (0.0, 1.0):
                return x
        except Exception:
            pass
        return float("nan")
    return s.map(f)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return max(0.0, center-half), min(1.0, center+half)


def detect_header_and_delimiter(path: Path) -> Tuple[str, int, List[str]]:
    delimiters = ["\t", ",", ";", "|"]
    required = {"match_id", "home_shots_87", "away_shots_87"}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        physical: List[str] = []
        for _ in range(100):
            line = fh.readline()
            if line == "":
                break
            physical.append(line.rstrip("\r\n"))

    best = None
    for idx, line in enumerate(physical):
        for delim in delimiters:
            try:
                fields = next(csv.reader([line], delimiter=delim))
            except Exception:
                continue
            fields = [clean(x) for x in fields]
            if required.issubset(set(fields)):
                bonus = sum(
                    x in fields for x in [
                        "home_team", "away_team", "competition_id",
                        "kickoff_utc", "score_87_home", "score_87_away"
                    ]
                )
                cand = (bonus, idx, delim, fields)
                if best is None or cand[0] > best[0]:
                    best = cand

    if best is None:
        raise ValueError(
            f"Header contenant match_id/home_shots_87/away_shots_87 introuvable dans {path}"
        )

    _, idx, delim, fields = best
    return delim, idx, fields


def stream_shots_context(path: Path, wanted_ids: Sequence[str]) -> pd.DataFrame:
    wanted = {clean(x) for x in wanted_ids if clean(x)}
    delim, header_idx, header = detect_header_and_delimiter(path)
    index = {name: i for i, name in enumerate(header)}

    optional = [
        "home_team", "away_team", "competition_id", "season_id",
        "kickoff_utc", "score_87_home", "score_87_away"
    ]

    found: Dict[str, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)

        for physical_idx, row in enumerate(reader):
            if physical_idx <= header_idx or not row:
                continue

            if len(row) <= index["match_id"]:
                continue

            mid = clean(row[index["match_id"]])
            if mid not in wanted:
                continue

            max_idx = max(index["match_id"], index["home_shots_87"], index["away_shots_87"])
            if len(row) <= max_idx:
                raise RuntimeError(f"{path.name}: ligne ciblée trop courte pour {mid}")

            rec: Dict[str, Any] = {
                "match_id": mid,
                "home_shots_87": row[index["home_shots_87"]],
                "away_shots_87": row[index["away_shots_87"]],
            }

            for c in optional:
                if c in index and len(row) > index[c]:
                    rec[c] = row[index[c]]

            if mid in found:
                prior = found[mid]
                for c in ("home_shots_87", "away_shots_87"):
                    a = clean(prior.get(c))
                    b = clean(rec.get(c))
                    if a and b and a != b:
                        raise RuntimeError(
                            f"{path.name}: doublon incohérent pour {mid}: {c}={a} vs {b}"
                        )
            else:
                found[mid] = rec

    return pd.DataFrame([found[mid] for mid in wanted if mid in found])


def resolve_context_source(wanted_ids: Sequence[str]) -> Tuple[Path, pd.DataFrame]:
    wanted = {clean(x) for x in wanted_ids if clean(x)}
    errors = []

    for path in DATASET_CANDIDATES:
        if not path.exists():
            continue

        try:
            ctx = stream_shots_context(path, wanted_ids)
            got = set(ctx["match_id"].astype(str)) if not ctx.empty else set()
            missing = wanted - got
            if not missing:
                return path, ctx
            errors.append(f"{path.name}: {len(missing)} match_id manquants")
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Aucune source historique ne permet la jointure complète.\n- "
        + "\n- ".join(errors)
    )


def summarize(df: pd.DataFrame, name: str, mask: pd.Series, base_rate: float) -> Dict[str, Any]:
    g = df.loc[mask].copy()
    n = len(g)
    positives = int(g["target"].sum()) if n else 0
    rate = positives / n if n else float("nan")
    lo, hi = wilson_ci(positives, n)
    mean_p = float(g["probability"].mean()) if n else float("nan")

    return {
        "group": name,
        "n": n,
        "positives": positives,
        "positive_rate": rate,
        "rate_percent": 100 * rate if n else float("nan"),
        "ci95_low": lo,
        "ci95_high": hi,
        "mean_v3c_probability": mean_p,
        "uplift_vs_all_oos_pp": 100 * (rate - base_rate) if n else float("nan"),
    }


def pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{100*x:.2f}%"


def main() -> int:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    pred = pd.read_csv(PREDICTIONS_PATH)

    needed = {"model_name", "match_id", "target", "probability"}
    missing = needed - set(pred.columns)
    if missing:
        raise ValueError(f"predictions_test.csv colonnes manquantes: {sorted(missing)}")

    pred = pred[pred["model_name"].astype(str) == MODEL_NAME].copy()

    pred["match_id"] = pred["match_id"].astype(str).str.strip()
    if pred["match_id"].duplicated().any():
        raise RuntimeError("match_id dupliqués dans predictions_test V3-C")

    pred["target"] = target_series(pred["target"])
    pred["probability"] = probability_series(pred["probability"])

    if pred[["target", "probability"]].isna().any().any():
        raise RuntimeError("Target ou probabilité invalide dans le bloc OOS V3-C")

    source_path, ctx = resolve_context_source(pred["match_id"].tolist())
    ctx["match_id"] = ctx["match_id"].astype(str).str.strip()

    df = pred.merge(ctx, on="match_id", how="left", validate="one_to_one")

    for c in ("home_shots_87", "away_shots_87"):
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

    if df[["home_shots_87", "away_shots_87"]].isna().any().any():
        bad = df.loc[
            df[["home_shots_87", "away_shots_87"]].isna().any(axis=1),
            "match_id"
        ].head(10).tolist()
        raise RuntimeError(f"Shots @87 manquants: {bad}")

    df["total_shots_87"] = df["home_shots_87"] + df["away_shots_87"]
    df["shots35"] = df["total_shots_87"] >= 35

    p = df["probability"]
    m55_60 = (p >= 0.55) & (p < 0.60)
    m60_65 = (p >= 0.60) & (p < 0.65)
    m55_65 = (p >= 0.55) & (p < 0.65)
    s35 = df["shots35"]

    base_rate = float(df["target"].mean())

    groups = [
        ("ALL_V3C_OOS", pd.Series(True, index=df.index)),
        ("V3C_55_60", m55_60),
        ("V3C_60_65", m60_65),
        ("V3C_55_65", m55_65),
        ("SHOTS35_ALL_PROBS", s35),
        ("SHOTS35_AND_55_60", s35 & m55_60),
        ("SHOTS35_AND_60_65", s35 & m60_65),
        ("SHOTS35_AND_55_65", s35 & m55_65),
        ("LT35_SHOTS_AND_55_65", (~s35) & m55_65),
    ]

    summary = pd.DataFrame([
        summarize(df, name, mask, base_rate)
        for name, mask in groups
    ])

    r_band = summary.loc[summary["group"] == "V3C_55_65"].iloc[0]
    r_cross = summary.loc[summary["group"] == "SHOTS35_AND_55_65"].iloc[0]
    r_non = summary.loc[summary["group"] == "LT35_SHOTS_AND_55_65"].iloc[0]

    direct_delta = float("nan")
    risk_ratio = float("nan")

    if r_cross["n"] > 0 and r_non["n"] > 0:
        direct_delta = float(r_cross["positive_rate"] - r_non["positive_rate"])
        if r_non["positive_rate"] > 0:
            risk_ratio = float(r_cross["positive_rate"] / r_non["positive_rate"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "v3c_shots35_cross_summary.csv",
        index=False
    )

    inter = df.loc[s35 & m55_65].copy()
    inter["result"] = np.where(
        inter["target"].eq(1),
        "GOAL_AFTER_87",
        "NO_GOAL_AFTER_87"
    )

    keep = [
        c for c in [
            "match_id", "kickoff_time_utc", "kickoff_utc", "competition_id",
            "home_team", "away_team", "score_87_home", "score_87_away",
            "home_shots_87", "away_shots_87", "total_shots_87",
            "probability", "target", "result"
        ]
        if c in inter.columns
    ]

    inter = inter.sort_values(
        ["probability", "total_shots_87"],
        ascending=[False, False]
    )

    inter[keep].to_csv(
        OUTPUT_DIR / "v3c_shots35_intersection_matches.csv",
        index=False
    )

    report = [
        "# V3-C — Cross audit 55–65% × SHOTS35",
        "",
        f"- OOS V3-C: **{len(df)} matchs**",
        f"- Base rate OOS: **{pct(base_rate)}**",
        f"- Source shots: `{source_path}`",
        "",
        "## Résultats",
        "",
        "| Groupe | n | Buts >87 | Taux | IC95% | P moyenne V3-C | Uplift vs global |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary.itertuples(index=False):
        report.append(
            f"| {r.group} | {int(r.n)} | {int(r.positives)} | "
            f"{pct(r.positive_rate)} | {pct(r.ci95_low)}–{pct(r.ci95_high)} | "
            f"{pct(r.mean_v3c_probability)} | {r.uplift_vs_all_oos_pp:+.2f} pp |"
        )

    report += [
        "",
        "## Comparaison directe dans 55–65%",
        "",
        f"- 55–65 global: **{int(r_band['positives'])}/{int(r_band['n'])} = {pct(float(r_band['positive_rate']))}**.",
        f"- 55–65 + SHOTS35: **{int(r_cross['positives'])}/{int(r_cross['n'])} = {pct(float(r_cross['positive_rate']))}**.",
        f"- 55–65 + <35 tirs: **{int(r_non['positives'])}/{int(r_non['n'])} = {pct(float(r_non['positive_rate']))}**.",
    ]

    if np.isfinite(direct_delta):
        report.append(
            f"- Effet brut SHOTS35 dans la bande 55–65: **{direct_delta*100:+.2f} points**."
        )
    if np.isfinite(risk_ratio):
        report.append(
            f"- Risk ratio SHOTS35 vs <35 tirs dans 55–65: **{risk_ratio:.3f}×**."
        )

    report += [
        "",
        "Important: analyse descriptive sur un bloc OOS déjà consommé. "
        "Aucun filtre n'est ajouté automatiquement au scanner."
    ]

    (OUTPUT_DIR / "v3c_shots35_cross_report.md").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("=" * 104)
    print("V3-C — CROSS 55–65% × SHOTS35 @87'")
    print("=" * 104)
    print(f"OOS V3-C      : {len(df)}")
    print(f"Source shots  : {source_path.name}")
    print(f"Base OOS      : {base_rate:.2%}")
    print()

    disp = summary[
        [
            "group", "n", "positives", "positive_rate",
            "ci95_low", "ci95_high", "mean_v3c_probability",
            "uplift_vs_all_oos_pp"
        ]
    ].copy()

    for c in [
        "positive_rate", "ci95_low", "ci95_high", "mean_v3c_probability"
    ]:
        disp[c] = disp[c].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else "NA"
        )

    print(disp.to_string(index=False))
    print()
    print("COMPARAISON DIRECTE 55–65%:")
    print(
        f"55–65 global       : {int(r_band['positives'])}/{int(r_band['n'])} "
        f"= {float(r_band['positive_rate']):.2%}"
    )
    print(
        f"55–65 + SHOTS35    : {int(r_cross['positives'])}/{int(r_cross['n'])} "
        f"= {float(r_cross['positive_rate']):.2%}"
    )
    print(
        f"55–65 + <35 shots  : {int(r_non['positives'])}/{int(r_non['n'])} "
        f"= {float(r_non['positive_rate']):.2%}"
    )

    if np.isfinite(direct_delta):
        print(f"Delta SHOTS35      : {direct_delta*100:+.2f} points")
    if np.isfinite(risk_ratio):
        print(f"Risk ratio         : {risk_ratio:.3f}x")

    print()
    print(f"Sorties             : {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
