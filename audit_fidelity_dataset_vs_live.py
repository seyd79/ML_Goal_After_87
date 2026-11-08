#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_fidelity_dataset_vs_live.py

AUDIT DE FIDÉLITÉ
=================
Compare le dataset historique d'entraînement et les snapshots réellement
produits par le scanner live à 87'.

Ce script NE MODIFIE PAS le scanner de production.

ENTRÉES PAR DÉFAUT
------------------
ML_Dataset_PreMatch_Live87.csv
outputs/history/live_feature_snapshots.csv
outputs/production/ET_500_d15_73_27_features.json
outputs/production/ET_500_d15_73_27.joblib   (optionnel)

SORTIES
-------
outputs/fidelity_audit/feature_fidelity_audit.csv
outputs/fidelity_audit/fidelity_summary.csv
outputs/fidelity_audit/live_invariant_violations.csv
outputs/fidelity_audit/fidelity_report.md

CE QUE L'AUDIT TESTE
--------------------
1. Schéma : mêmes 96 features et même ordre.
2. Complétude : historique vs live, FULL vs DEGRADED.
3. Distribution : moyenne, médiane, quantiles, SMD, KS, PSI.
4. Hors-distribution : part des valeurs live hors min/max historique.
5. Cohérence logique des features live à 87'.
6. Priorisation selon l'importance du modèle si l'artefact joblib est disponible.

IMPORTANT
---------
Cet audit mesure la fidélité STATISTIQUE et LOGIQUE.

Il ne peut pas prouver à lui seul que chaque feature historique a été construite
avec exactement le même endpoint, la même fenêtre temporelle et la même formule
que le scanner live. Pour cette fidélité SÉMANTIQUE, il faut comparer le code
original ayant généré ML_Dataset_PreMatch_Live87.csv avec le scanner actuel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_CANDIDATES = [
    BASE_DIR / "ML_Dataset_PreMatch_Live87.csv",
    BASE_DIR / "data" / "ML_Dataset_PreMatch_Live87.csv",
]

DEFAULT_LIVE_HISTORY = (
    BASE_DIR / "outputs" / "history" / "live_feature_snapshots.csv"
)

DEFAULT_FEATURES_JSON = (
    BASE_DIR
    / "outputs"
    / "production"
    / "ET_500_d15_73_27_features.json"
)

DEFAULT_MODEL = (
    BASE_DIR
    / "outputs"
    / "production"
    / "ET_500_d15_73_27.joblib"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "outputs" / "fidelity_audit"
)

ID_COLUMN = "match_id"
TARGET_COLUMN = "goal_after_87"

EXPECTED_FEATURE_COUNT = 96
PRE_COUNT = 59
LIVE_COUNT = 37


# =============================================================================
# HELPERS
# =============================================================================

def resolve_dataset(cli_path: Optional[str]) -> Path:
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {path}")
        return path

    for path in DEFAULT_DATASET_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "ML_Dataset_PreMatch_Live87.csv introuvable. "
        "Place-le à la racine du projet ou utilise --dataset CHEMIN."
    )


def resolve_path(cli_path: Optional[str], default: Path) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    return default


def load_feature_order(
    dataset: pd.DataFrame,
    features_json_path: Path,
) -> List[str]:
    if features_json_path.exists():
        payload = json.loads(
            features_json_path.read_text(encoding="utf-8")
        )

        if isinstance(payload, list):
            features = [str(x) for x in payload]

        elif isinstance(payload, Mapping):
            candidate = (
                payload.get("features")
                or payload.get("feature_names")
                or payload.get("expected_features")
            )
            features = (
                [str(x) for x in candidate]
                if isinstance(candidate, list)
                else []
            )

        else:
            features = []

        if len(features) == EXPECTED_FEATURE_COUNT:
            return features

    features = [
        c
        for c in dataset.columns
        if c not in {ID_COLUMN, TARGET_COLUMN}
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Impossible de déterminer les 96 features. Trouvé : {len(features)}"
        )

    return features


def numeric_series(
    df: pd.DataFrame,
    feature: str,
) -> pd.Series:
    return pd.to_numeric(
        df[feature],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def safe_stat(
    s: pd.Series,
    fn: str,
) -> float:
    clean = s.dropna()
    if clean.empty:
        return float("nan")

    if fn == "mean":
        return float(clean.mean())
    if fn == "std":
        return float(clean.std(ddof=1))
    if fn == "min":
        return float(clean.min())
    if fn == "max":
        return float(clean.max())
    if fn == "median":
        return float(clean.median())

    raise ValueError(fn)


def q(s: pd.Series, value: float) -> float:
    clean = s.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(value))


def standardized_mean_difference(
    hist: pd.Series,
    live: pd.Series,
) -> float:
    h = hist.dropna()
    l = live.dropna()

    if len(h) < 2 or len(l) < 2:
        return float("nan")

    h_var = float(h.var(ddof=1))
    l_var = float(l.var(ddof=1))

    pooled = math.sqrt(max((h_var + l_var) / 2.0, 0.0))

    if pooled == 0:
        return 0.0 if float(h.mean()) == float(l.mean()) else float("inf")

    return float((l.mean() - h.mean()) / pooled)


def ks_statistic(
    hist: pd.Series,
    live: pd.Series,
) -> Tuple[float, float]:
    if ks_2samp is None:
        return float("nan"), float("nan")

    h = hist.dropna().to_numpy(dtype=float)
    l = live.dropna().to_numpy(dtype=float)

    if len(h) < 2 or len(l) < 2:
        return float("nan"), float("nan")

    try:
        result = ks_2samp(h, l)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def psi(
    hist: pd.Series,
    live: pd.Series,
    bins: int = 10,
) -> float:
    """
    Population Stability Index.
    Bins fondés sur les quantiles historiques.
    """
    h = hist.dropna().to_numpy(dtype=float)
    l = live.dropna().to_numpy(dtype=float)

    if len(h) < 10 or len(l) < 5:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)

    try:
        edges = np.unique(
            np.quantile(h, quantiles)
        )
    except Exception:
        return float("nan")

    if len(edges) < 3:
        return 0.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    h_counts, _ = np.histogram(h, bins=edges)
    l_counts, _ = np.histogram(l, bins=edges)

    eps = 1e-6

    h_pct = h_counts / max(h_counts.sum(), 1)
    l_pct = l_counts / max(l_counts.sum(), 1)

    h_pct = np.clip(h_pct, eps, None)
    l_pct = np.clip(l_pct, eps, None)

    return float(
        np.sum(
            (l_pct - h_pct)
            * np.log(l_pct / h_pct)
        )
    )


def severity_from_metrics(
    psi_value: float,
    smd_abs: float,
    ks_value: float,
    out_of_range_rate: float,
    live_missing_rate: float,
) -> str:
    severe = False
    moderate = False

    if math.isfinite(psi_value):
        severe |= psi_value >= 0.25
        moderate |= psi_value >= 0.10

    if math.isfinite(smd_abs):
        severe |= smd_abs >= 0.80
        moderate |= smd_abs >= 0.30

    if math.isfinite(ks_value):
        severe |= ks_value >= 0.30
        moderate |= ks_value >= 0.15

    if math.isfinite(out_of_range_rate):
        severe |= out_of_range_rate >= 0.10
        moderate |= out_of_range_rate >= 0.03

    if math.isfinite(live_missing_rate):
        severe |= live_missing_rate >= 0.10
        moderate |= live_missing_rate >= 0.03

    if severe:
        return "SEVERE"
    if moderate:
        return "MODERATE"
    return "LOW"


def extract_feature_importances(
    model_path: Path,
    features: Sequence[str],
) -> Dict[str, float]:
    if not model_path.exists():
        return {}

    try:
        model = joblib.load(model_path)
    except Exception:
        return {}

    estimator = model

    # sklearn Pipeline
    if hasattr(model, "named_steps"):
        named_steps = getattr(model, "named_steps")
        for candidate_name in (
            "model",
            "classifier",
            "clf",
            "extratreesclassifier",
        ):
            if candidate_name in named_steps:
                estimator = named_steps[candidate_name]
                break
        else:
            # dernier step
            try:
                estimator = list(named_steps.values())[-1]
            except Exception:
                estimator = model

    importances = getattr(
        estimator,
        "feature_importances_",
        None,
    )

    if importances is None:
        return {}

    arr = np.asarray(importances, dtype=float)

    if len(arr) != len(features):
        return {}

    return {
        feature: float(value)
        for feature, value in zip(features, arr)
    }


# =============================================================================
# LOGICAL INVARIANTS
# =============================================================================

def check_live_invariants(
    live87: pd.DataFrame,
) -> pd.DataFrame:
    violations: List[Dict[str, Any]] = []

    def add(
        row: pd.Series,
        rule: str,
        details: str,
    ) -> None:
        violations.append({
            "match_id": row.get("match_id", ""),
            "minute": row.get("minute", ""),
            "prediction_quality": row.get(
                "prediction_quality",
                "",
            ),
            "coverage": row.get("coverage", ""),
            "rule": rule,
            "details": details,
        })

    for _, row in live87.iterrows():
        def val(name: str) -> Optional[float]:
            x = pd.to_numeric(
                pd.Series([row.get(name)]),
                errors="coerce",
            ).iloc[0]

            return (
                None
                if pd.isna(x)
                else float(x)
            )

        sh = val("score_87_home")
        sa = val("score_87_away")
        lead = val("leading_team_87")
        diff = val("goal_diff_87")
        draw = val("is_draw_87")

        if sh is not None and sa is not None:
            expected_diff = abs(sh - sa)

            if diff is not None and abs(diff - expected_diff) > 1e-9:
                add(
                    row,
                    "goal_diff_87",
                    f"observed={diff}; expected_abs_diff={expected_diff}",
                )

            expected_draw = 1.0 if sh == sa else 0.0

            if draw is not None and abs(draw - expected_draw) > 1e-9:
                add(
                    row,
                    "is_draw_87",
                    f"observed={draw}; expected={expected_draw}",
                )

            expected_lead = (
                2.0
                if sh == sa
                else (
                    0.0
                    if sh > sa
                    else 1.0
                )
            )

            if lead is not None and abs(lead - expected_lead) > 1e-9:
                add(
                    row,
                    "leading_team_87",
                    f"observed={lead}; expected={expected_lead}",
                )

        # Temporal identities at an exact 87 snapshot.
        temporal_pairs = [
            (
                "last_goal_minute",
                "minutes_since_last_goal",
            ),
            (
                "last_shot_minute",
                "minutes_since_last_shot",
            ),
            (
                "last_shot_on_target_minute",
                "minutes_since_last_shot_on_target",
            ),
            (
                "last_corner_minute",
                "minutes_since_last_corner",
            ),
        ]

        for last_name, since_name in temporal_pairs:
            last = val(last_name)
            since = val(since_name)

            if last is None or since is None:
                continue

            expected_since = max(87.0 - last, 0.0)

            if abs(since - expected_since) > 1.01:
                add(
                    row,
                    f"{since_name}_identity",
                    (
                        f"{last_name}={last}; "
                        f"{since_name}={since}; "
                        f"expected≈{expected_since}"
                    ),
                )

        # Window features should be non-negative.
        for feature in (
            "shots_80_87",
            "sot_80_87",
            "corners_80_87",
            "fouls_80_87",
            "cards_80_87",
            "subs_80_87",
        ):
            x = val(feature)

            if x is not None and x < 0:
                add(
                    row,
                    f"{feature}_nonnegative",
                    f"value={x}",
                )

        # Recent shots cannot logically exceed total shots if both exist.
        hs = val("home_shots_87")
        aw = val("away_shots_87")
        recent_shots = val("shots_80_87")

        if (
            hs is not None
            and aw is not None
            and recent_shots is not None
            and recent_shots > hs + aw + 1e-9
        ):
            add(
                row,
                "shots_80_87_le_total",
                (
                    f"recent={recent_shots}; "
                    f"total={hs + aw}"
                ),
            )

        hsot = val("home_shots_on_target_87")
        asot = val("away_shots_on_target_87")
        recent_sot = val("sot_80_87")

        if (
            hsot is not None
            and asot is not None
            and recent_sot is not None
            and recent_sot > hsot + asot + 1e-9
        ):
            add(
                row,
                "sot_80_87_le_total",
                (
                    f"recent={recent_sot}; "
                    f"total={hsot + asot}"
                ),
            )

    return pd.DataFrame(violations)


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit de fidélité statistique dataset historique vs scanner live."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Chemin vers ML_Dataset_PreMatch_Live87.csv",
    )

    parser.add_argument(
        "--live-history",
        default=None,
        help="Chemin vers live_feature_snapshots.csv",
    )

    parser.add_argument(
        "--features-json",
        default=None,
        help="Chemin vers ET_500_d15_73_27_features.json",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Chemin vers ET_500_d15_73_27.joblib",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie",
    )

    args = parser.parse_args()

    dataset_path = resolve_dataset(args.dataset)

    live_path = resolve_path(
        args.live_history,
        DEFAULT_LIVE_HISTORY,
    )

    features_json_path = resolve_path(
        args.features_json,
        DEFAULT_FEATURES_JSON,
    )

    model_path = resolve_path(
        args.model,
        DEFAULT_MODEL,
    )

    output_dir = resolve_path(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    if not live_path.exists():
        raise FileNotFoundError(
            f"Historique live introuvable : {live_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("AUDIT FIDÉLITÉ — DATASET HISTORIQUE ↔ SCANNER LIVE")
    print("=" * 88)
    print(f"Dataset historique : {dataset_path}")
    print(f"Snapshots live      : {live_path}")
    print(f"Features JSON       : {features_json_path}")
    print(f"Modèle              : {model_path}")
    print("=" * 88)

    historical = pd.read_csv(dataset_path)
    live = pd.read_csv(live_path)

    features = load_feature_order(
        historical,
        features_json_path,
    )

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Feature count invalide : {len(features)}/96"
        )

    missing_hist_schema = [
        f for f in features
        if f not in historical.columns
    ]

    missing_live_schema = [
        f for f in features
        if f not in live.columns
    ]

    if missing_hist_schema:
        raise RuntimeError(
            "Features absentes du dataset historique : "
            + ", ".join(missing_hist_schema)
        )

    if missing_live_schema:
        raise RuntimeError(
            "Features absentes des snapshots live : "
            + ", ".join(missing_live_schema)
        )

    live["minute_numeric"] = pd.to_numeric(
        live.get("minute"),
        errors="coerce",
    )

    live87 = live[
        live["minute_numeric"] == 87
    ].copy()

    if live87.empty:
        raise RuntimeError(
            "Aucun snapshot à 87' trouvé."
        )

    # FULL only is our highest-fidelity production subset.
    if "prediction_quality" in live87.columns:
        live87_full = live87[
            live87["prediction_quality"]
            .astype(str)
            .str.upper()
            .eq("FULL")
        ].copy()
    else:
        live87_full = live87.copy()

    # Use all 87 rows as a second comparison, including DEGRADED/BLOCKED observed values.
    live87_all = live87.copy()

    importances = extract_feature_importances(
        model_path,
        features,
    )

    feature_rows: List[Dict[str, Any]] = []

    for idx, feature in enumerate(features):
        hist_s = numeric_series(
            historical,
            feature,
        )

        live_full_s = numeric_series(
            live87_full,
            feature,
        )

        live_all_s = numeric_series(
            live87_all,
            feature,
        )

        h_clean = hist_s.dropna()
        lf_clean = live_full_s.dropna()
        la_clean = live_all_s.dropna()

        h_min = safe_stat(hist_s, "min")
        h_max = safe_stat(hist_s, "max")

        if lf_clean.empty or not math.isfinite(h_min) or not math.isfinite(h_max):
            out_rate = float("nan")
        else:
            out_rate = float(
                (
                    (lf_clean < h_min)
                    | (lf_clean > h_max)
                ).mean()
            )

        smd = standardized_mean_difference(
            hist_s,
            live_full_s,
        )

        ks_value, ks_p = ks_statistic(
            hist_s,
            live_full_s,
        )

        psi_value = psi(
            hist_s,
            live_full_s,
        )

        hist_missing_rate = float(
            hist_s.isna().mean()
        )

        live_full_missing_rate = float(
            live_full_s.isna().mean()
        ) if len(live87_full) else float("nan")

        live_all_missing_rate = float(
            live_all_s.isna().mean()
        )

        severity = severity_from_metrics(
            psi_value=psi_value,
            smd_abs=abs(smd) if math.isfinite(smd) else float("nan"),
            ks_value=ks_value,
            out_of_range_rate=out_rate,
            live_missing_rate=live_full_missing_rate,
        )

        feature_rows.append({
            "feature_index": idx + 1,
            "feature": feature,
            "group": (
                "PRE_MATCH"
                if idx < PRE_COUNT
                else "LIVE_87"
            ),
            "model_importance": importances.get(
                feature,
                float("nan"),
            ),
            "historical_n": int(h_clean.shape[0]),
            "historical_missing_rate": hist_missing_rate,
            "historical_mean": safe_stat(hist_s, "mean"),
            "historical_std": safe_stat(hist_s, "std"),
            "historical_min": h_min,
            "historical_q05": q(hist_s, 0.05),
            "historical_q25": q(hist_s, 0.25),
            "historical_median": q(hist_s, 0.50),
            "historical_q75": q(hist_s, 0.75),
            "historical_q95": q(hist_s, 0.95),
            "historical_max": h_max,
            "live87_full_n": int(lf_clean.shape[0]),
            "live87_full_missing_rate": live_full_missing_rate,
            "live87_full_mean": safe_stat(live_full_s, "mean"),
            "live87_full_std": safe_stat(live_full_s, "std"),
            "live87_full_min": safe_stat(live_full_s, "min"),
            "live87_full_q05": q(live_full_s, 0.05),
            "live87_full_q25": q(live_full_s, 0.25),
            "live87_full_median": q(live_full_s, 0.50),
            "live87_full_q75": q(live_full_s, 0.75),
            "live87_full_q95": q(live_full_s, 0.95),
            "live87_full_max": safe_stat(live_full_s, "max"),
            "live87_all_n": int(la_clean.shape[0]),
            "live87_all_missing_rate": live_all_missing_rate,
            "smd_livefull_vs_hist": smd,
            "abs_smd": abs(smd) if math.isfinite(smd) else float("nan"),
            "ks_stat": ks_value,
            "ks_pvalue": ks_p,
            "psi": psi_value,
            "livefull_outside_historical_range_rate": out_rate,
            "severity": severity,
        })

    feature_audit = pd.DataFrame(feature_rows)

    # Weighted drift priority.
    if (
        "model_importance" in feature_audit.columns
        and feature_audit["model_importance"].notna().any()
    ):
        importance = feature_audit["model_importance"].fillna(0.0)

        drift_component = (
            feature_audit["abs_smd"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(upper=3.0)
        )

        feature_audit["importance_x_drift"] = (
            importance
            * drift_component
        )
    else:
        feature_audit["importance_x_drift"] = np.nan

    severity_rank = {
        "SEVERE": 3,
        "MODERATE": 2,
        "LOW": 1,
    }

    feature_audit["_severity_rank"] = (
        feature_audit["severity"]
        .map(severity_rank)
        .fillna(0)
    )

    feature_audit = feature_audit.sort_values(
        by=[
            "_severity_rank",
            "importance_x_drift",
            "abs_smd",
            "psi",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).drop(columns=["_severity_rank"])

    feature_csv = (
        output_dir
        / "feature_fidelity_audit.csv"
    )

    feature_audit.to_csv(
        feature_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Logical invariants.
    violations = check_live_invariants(
        live87
    )

    violations_csv = (
        output_dir
        / "live_invariant_violations.csv"
    )

    violations.to_csv(
        violations_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Summary.
    severity_counts = (
        feature_audit["severity"]
        .value_counts()
        .to_dict()
    )

    pre = feature_audit[
        feature_audit["group"] == "PRE_MATCH"
    ]

    lv = feature_audit[
        feature_audit["group"] == "LIVE_87"
    ]

    summary_rows = [
        {
            "metric": "historical_rows",
            "value": len(historical),
        },
        {
            "metric": "live_snapshots_total",
            "value": len(live),
        },
        {
            "metric": "live_snapshots_87",
            "value": len(live87),
        },
        {
            "metric": "live_full_87",
            "value": len(live87_full),
        },
        {
            "metric": "features_total",
            "value": len(features),
        },
        {
            "metric": "historical_total_missing_values_96",
            "value": int(
                historical[features]
                .isna()
                .sum()
                .sum()
            ),
        },
        {
            "metric": "severe_features",
            "value": int(
                severity_counts.get(
                    "SEVERE",
                    0,
                )
            ),
        },
        {
            "metric": "moderate_features",
            "value": int(
                severity_counts.get(
                    "MODERATE",
                    0,
                )
            ),
        },
        {
            "metric": "low_drift_features",
            "value": int(
                severity_counts.get(
                    "LOW",
                    0,
                )
            ),
        },
        {
            "metric": "pre_match_severe_features",
            "value": int(
                (pre["severity"] == "SEVERE").sum()
            ),
        },
        {
            "metric": "live87_severe_features",
            "value": int(
                (lv["severity"] == "SEVERE").sum()
            ),
        },
        {
            "metric": "invariant_violations",
            "value": len(violations),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        output_dir
        / "fidelity_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------------------

    severe = feature_audit[
        feature_audit["severity"]
        == "SEVERE"
    ].copy()

    moderate = feature_audit[
        feature_audit["severity"]
        == "MODERATE"
    ].copy()

    top_priority = feature_audit.sort_values(
        by=[
            "importance_x_drift",
            "abs_smd",
        ],
        ascending=False,
        na_position="last",
    ).head(20)

    def fmt(x: Any, digits: int = 3) -> str:
        try:
            value = float(x)
            if not math.isfinite(value):
                return ""
            return f"{value:.{digits}f}"
        except Exception:
            return ""

    report: List[str] = [
        "# Audit de fidélité — dataset historique ↔ scanner live",
        "",
        "## 1. Périmètre",
        "",
        f"- Dataset historique : `{dataset_path.name}`",
        f"- Lignes historiques : **{len(historical)}**",
        f"- Features modèle : **{len(features)}**",
        f"- Snapshots live totaux : **{len(live)}**",
        f"- Snapshots exactement à 87' : **{len(live87)}**",
        f"- Snapshots FULL 96/96 à 87' : **{len(live87_full)}**",
        "",
        "La comparaison de distribution principale utilise les snapshots **FULL 96/96 à 87'**.",
        "Les snapshots DEGRADED/BLOCKED sont conservés comme diagnostic secondaire.",
        "",
        "## 2. Verdict de schéma",
        "",
        f"- Features historiques absentes : **{len(missing_hist_schema)}**",
        f"- Features live absentes : **{len(missing_live_schema)}**",
        f"- Valeurs manquantes historiques sur les 96 features : "
        f"**{int(historical[features].isna().sum().sum())}**",
        "",
        "## 3. Drift détecté",
        "",
        f"- Features SEVERE : **{len(severe)} / 96**",
        f"- Features MODERATE : **{len(moderate)} / 96**",
        f"- PRE-MATCH sévères : **{int((pre['severity'] == 'SEVERE').sum())} / 59**",
        f"- LIVE sévères : **{int((lv['severity'] == 'SEVERE').sum())} / 37**",
        "",
        "### Features prioritaires",
        "",
        "| Feature | Groupe | Importance modèle | SMD abs | KS | PSI | Hors plage hist. | Sévérité |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in top_priority.iterrows():
        report.append(
            "| "
            + " | ".join([
                str(row["feature"]),
                str(row["group"]),
                fmt(row["model_importance"], 4),
                fmt(row["abs_smd"], 3),
                fmt(row["ks_stat"], 3),
                fmt(row["psi"], 3),
                (
                    ""
                    if pd.isna(
                        row[
                            "livefull_outside_historical_range_rate"
                        ]
                    )
                    else f"{float(row['livefull_outside_historical_range_rate']):.1%}"
                ),
                str(row["severity"]),
            ])
            + " |"
        )

    report.extend([
        "",
        "## 4. Cohérence logique des snapshots live",
        "",
        f"- Violations détectées : **{len(violations)}**",
        "",
    ])

    if not violations.empty:
        rule_counts = (
            violations["rule"]
            .value_counts()
            .head(15)
        )

        report.extend([
            "| Règle | Violations |",
            "|---|---:|",
        ])

        for rule, count in rule_counts.items():
            report.append(
                f"| {rule} | {int(count)} |"
            )

    report.extend([
        "",
        "## 5. Lecture des métriques",
        "",
        "- **SMD** : différence de moyenne standardisée. "
        "≈0 = distributions proches; >0,3 = à surveiller; >0,8 = fort écart.",
        "- **KS** : différence maximale entre les distributions cumulées.",
        "- **PSI** : <0,10 faible; 0,10–0,25 changement notable; >0,25 fort drift.",
        "- **Hors plage historique** : valeurs live situées en dehors du min/max observé sur les 13 800 matchs.",
        "",
        "## 6. Limite essentielle",
        "",
        "Cet audit peut montrer qu'une feature live ne ressemble pas statistiquement à la feature historique, "
        "mais il ne peut pas prouver que les deux ont été calculées avec la même formule.",
        "",
        "Pour une preuve de fidélité **sémantique**, il faut le code original qui a généré "
        "`ML_Dataset_PreMatch_Live87.csv` puis comparer feature par feature : endpoint, champ JSON, "
        "fenêtre temporelle, agrégation, conventions de zéro/manquant et heure de coupure.",
        "",
        "## 7. Fichiers produits",
        "",
        f"- `{feature_csv.name}`",
        f"- `{summary_csv.name}`",
        f"- `{violations_csv.name}`",
    ])

    report_path = (
        output_dir
        / "fidelity_report.md"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("RÉSULTAT")
    print("-" * 88)
    print(f"Historique            : {len(historical)} lignes")
    print(f"Live @87              : {len(live87)} snapshots")
    print(f"FULL @87              : {len(live87_full)} snapshots")
    print(f"SEVERE                 : {len(severe)}/96")
    print(f"MODERATE               : {len(moderate)}/96")
    print(f"Violations invariants  : {len(violations)}")
    print()
    print(f"Rapport : {report_path}")
    print(f"Features: {feature_csv}")
    print(f"Résumé  : {summary_csv}")
    print(f"Invariants: {violations_csv}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
