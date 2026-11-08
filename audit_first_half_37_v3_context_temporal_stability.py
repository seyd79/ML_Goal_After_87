#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V3 CONTEXT TEMPORAL STABILITY AUDIT
===================================================

But:
Déterminer si AU MOINS UN contexte de score à 37' possède un signal temporel
répétable, sans entraîner, retuner ou sélectionner quoi que ce soit.

AUCUN NOUVEAU MODELE.
AUCUN TUNING.
AUCUNE SELECTION DE FEATURES.
AUCUN NOUVEAU HOLDOUT.

Le script relit uniquement les prédictions déjà produites par:
    benchmark_first_half_37_v3_context_models.py

Entrées:
    outputs/first_half_37/v3_context_benchmark/
        first_half_37_v3_context_predictions.csv
        first_half_37/v3_context_state_metrics.csv (si présent)
        first_half_37_v3_context_model_ranking.csv
        first_half_37_v3_context_benchmark_manifest.json

Question principale:
    "Existe-t-il au moins un contexte parmi
     0-0 / home_lead / away_lead / draw_with_goals
     qui reste utile dans plusieurs périodes temporelles ?"

Décision diagnostique (RESEARCH ONLY, pré-définie avant cet audit):
Un contexte est classé RESEARCH_SIGNAL si, pour au moins un modèle déjà testé:
    - >= 5 périodes éligibles parmi INTERNAL_1..5 + LATE_STRESS_594
    - ROC > 0.50 dans >= 80% des périodes éligibles
    - PR lift > 0 dans >= 80% des périodes éligibles
    - médiane ROC >= 0.55
    - moyenne PR lift > 0
    - LATE_STRESS ROC >= 0.53
    - LATE_STRESS PR lift > 0.02

Ces critères sont diagnostiques uniquement.
Ils NE valident PAS un futur modèle de production.

Sorties:
    outputs/first_half_37/v3_context_stability_audit/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent

BENCH_DIR = BASE_DIR / "outputs" / "first_half_37" / "v3_context_benchmark"
PREDICTIONS_PATH = BENCH_DIR / "first_half_37_v3_context_predictions.csv"
STATE_METRICS_PATH = BENCH_DIR / "first_half_37_v3_context_state_metrics.csv"
RANKING_PATH = BENCH_DIR / "first_half_37_v3_context_model_ranking.csv"
MANIFEST_PATH = BENCH_DIR / "first_half_37_v3_context_benchmark_manifest.json"

DEFAULT_OUTPUT = (
    BASE_DIR / "outputs" / "first_half_37" / "v3_context_stability_audit"
)

TARGET = "goal_after_37_before_ht"
CONTEXT_COL = "score_context_37"
MODEL_COL = "model"
FOLD_COL = "fold"
KIND_COL = "kind"
P_COL = "probability"

CONTEXTS = ["0-0", "home_lead", "away_lead", "draw_with_goals"]
PERIODS = [
    "INTERNAL_1",
    "INTERNAL_2",
    "INTERNAL_3",
    "INTERNAL_4",
    "INTERNAL_5",
    "LATE_STRESS_594",
]

# Minimum raisonnable pour afficher ROC/PR contextuels.
MIN_N_PERIOD = 15

# Gate diagnostique pré-défini.
MIN_ELIGIBLE_PERIODS = 5
MIN_POSITIVE_FRACTION = 0.80
MIN_MEDIAN_ROC = 0.55
MIN_LATE_ROC = 0.53
MIN_LATE_PR_LIFT = 0.02

EPS = 1e-10


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


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    prevalence = float(y.mean())
    pr = safe_pr(y, p)
    roc = safe_roc(y, p)

    base = np.full(len(y), prevalence, dtype=float)
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, base))

    ll = float(
        log_loss(
            y,
            np.clip(p, EPS, 1-EPS),
            labels=[0, 1],
        )
    )
    base_ll = float(
        log_loss(
            y,
            np.clip(base, EPS, 1-EPS),
            labels=[0, 1],
        )
    )

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prevalence,
        "mean_probability": float(p.mean()),
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift": pr - prevalence if np.isfinite(pr) else np.nan,
        "brier": brier,
        "brier_improvement": base_brier - brier,
        "log_loss": ll,
        "logloss_improvement": base_ll - ll,
    }


def validate_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    for path in (PREDICTIONS_PATH, RANKING_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_json(MANIFEST_PATH)

    if manifest.get("future_holdout_used") is not False:
        raise RuntimeError("Manifest inattendu: future_holdout_used != false.")

    if manifest.get("v2_final_holdout_status") != "CONSUMED_AND_USED_AS_LATE_STRESS":
        raise RuntimeError("Manifest inattendu: statut du bloc 594 incorrect.")

    pred = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    ranking = pd.read_csv(RANKING_PATH, low_memory=False)

    required = {
        MODEL_COL, FOLD_COL, KIND_COL, CONTEXT_COL,
        TARGET, P_COL,
    }
    missing = sorted(required - set(pred.columns))
    if missing:
        raise RuntimeError(f"Predictions: colonnes manquantes: {missing}")

    pred[TARGET] = pd.to_numeric(pred[TARGET], errors="coerce")
    pred[P_COL] = pd.to_numeric(pred[P_COL], errors="coerce")

    if pred[TARGET].isna().any():
        raise RuntimeError("Target manquant dans predictions.")
    if pred[P_COL].isna().any():
        raise RuntimeError("Probability manquante dans predictions.")

    pred[TARGET] = pred[TARGET].astype(int)

    unexpected_context = sorted(
        set(pred[CONTEXT_COL].dropna().astype(str))
        - set(CONTEXTS)
        - {"missing"}
    )
    if unexpected_context:
        raise RuntimeError(f"Contextes inattendus: {unexpected_context}")

    return pred, ranking, manifest


def period_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in pred[MODEL_COL].dropna().unique():
        mdf = pred[pred[MODEL_COL] == model]

        for context in CONTEXTS:
            cdf = mdf[mdf[CONTEXT_COL] == context]

            for period in PERIODS:
                g = cdf[cdf[FOLD_COL] == period]

                if g.empty:
                    continue

                y = g[TARGET].to_numpy(dtype=int)
                p = g[P_COL].to_numpy(dtype=float)

                row = {
                    "model": model,
                    "context": context,
                    "period": period,
                    "kind": (
                        "late_stress"
                        if period == "LATE_STRESS_594"
                        else "internal"
                    ),
                    "eligible": int(
                        len(g) >= MIN_N_PERIOD
                        and len(np.unique(y)) >= 2
                    ),
                }

                if row["eligible"]:
                    row.update(metrics(y, p))
                else:
                    row.update({
                        "n": int(len(g)),
                        "positives": int(y.sum()),
                        "positive_rate": float(y.mean()) if len(y) else np.nan,
                        "mean_probability": float(p.mean()) if len(p) else np.nan,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "pr_lift": np.nan,
                        "brier": np.nan,
                        "brier_improvement": np.nan,
                        "log_loss": np.nan,
                        "logloss_improvement": np.nan,
                    })

                rows.append(row)

    return pd.DataFrame(rows)


def summarize_contexts(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, context), g in periods.groupby(["model", "context"], sort=False):
        eligible = g[g["eligible"] == 1].copy()
        late = g[
            (g["period"] == "LATE_STRESS_594")
            & (g["eligible"] == 1)
        ].copy()

        n_eligible = len(eligible)

        if n_eligible:
            roc_positive_fraction = float((eligible["roc_auc"] > 0.5).mean())
            pr_positive_fraction = float((eligible["pr_lift"] > 0).mean())
            median_roc = float(eligible["roc_auc"].median())
            mean_roc = float(eligible["roc_auc"].mean())
            min_roc = float(eligible["roc_auc"].min())
            mean_pr_lift = float(eligible["pr_lift"].mean())
            median_pr_lift = float(eligible["pr_lift"].median())
            brier_positive_fraction = float(
                (eligible["brier_improvement"] > 0).mean()
            )
            logloss_positive_fraction = float(
                (eligible["logloss_improvement"] > 0).mean()
            )
        else:
            roc_positive_fraction = np.nan
            pr_positive_fraction = np.nan
            median_roc = np.nan
            mean_roc = np.nan
            min_roc = np.nan
            mean_pr_lift = np.nan
            median_pr_lift = np.nan
            brier_positive_fraction = np.nan
            logloss_positive_fraction = np.nan

        if len(late) == 1:
            late_roc = float(late.iloc[0]["roc_auc"])
            late_pr_lift = float(late.iloc[0]["pr_lift"])
            late_brier_imp = float(late.iloc[0]["brier_improvement"])
            late_logloss_imp = float(late.iloc[0]["logloss_improvement"])
            late_n = int(late.iloc[0]["n"])
        else:
            late_roc = np.nan
            late_pr_lift = np.nan
            late_brier_imp = np.nan
            late_logloss_imp = np.nan
            late_n = 0

        research_signal = int(
            n_eligible >= MIN_ELIGIBLE_PERIODS
            and roc_positive_fraction >= MIN_POSITIVE_FRACTION
            and pr_positive_fraction >= MIN_POSITIVE_FRACTION
            and median_roc >= MIN_MEDIAN_ROC
            and mean_pr_lift > 0
            and np.isfinite(late_roc)
            and late_roc >= MIN_LATE_ROC
            and late_pr_lift > MIN_LATE_PR_LIFT
        )

        rows.append({
            "model": model,
            "context": context,
            "research_signal": research_signal,
            "eligible_periods": n_eligible,
            "roc_above_050_fraction": roc_positive_fraction,
            "pr_lift_positive_fraction": pr_positive_fraction,
            "median_roc_auc": median_roc,
            "mean_roc_auc": mean_roc,
            "min_roc_auc": min_roc,
            "mean_pr_lift": mean_pr_lift,
            "median_pr_lift": median_pr_lift,
            "brier_improvement_positive_fraction": brier_positive_fraction,
            "logloss_improvement_positive_fraction": logloss_positive_fraction,
            "late_stress_n": late_n,
            "late_stress_roc_auc": late_roc,
            "late_stress_pr_lift": late_pr_lift,
            "late_stress_brier_improvement": late_brier_imp,
            "late_stress_logloss_improvement": late_logloss_imp,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "research_signal",
            "late_stress_roc_auc",
            "roc_above_050_fraction",
            "median_roc_auc",
            "mean_pr_lift",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out)+1))
    return out


def context_consensus(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for context in CONTEXTS:
        g = summary[summary["context"] == context].copy()

        rows.append({
            "context": context,
            "models_evaluated": int(len(g)),
            "models_with_research_signal": int(g["research_signal"].sum()),
            "best_model": (
                str(g.iloc[0]["model"]) if len(g) else ""
            ),
            "best_late_stress_roc": (
                float(g["late_stress_roc_auc"].max())
                if len(g) else np.nan
            ),
            "best_late_stress_pr_lift": (
                float(g["late_stress_pr_lift"].max())
                if len(g) else np.nan
            ),
            "median_across_models_of_median_roc": (
                float(g["median_roc_auc"].median())
                if len(g) else np.nan
            ),
            "median_across_models_roc_positive_fraction": (
                float(g["roc_above_050_fraction"].median())
                if len(g) else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(
        [
            "models_with_research_signal",
            "best_late_stress_roc",
            "median_across_models_of_median_roc",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def automatic_verdict(
    summary: pd.DataFrame,
    consensus: pd.DataFrame,
) -> Dict[str, Any]:
    signals = summary[summary["research_signal"] == 1].copy()

    if len(signals):
        contexts = sorted(signals["context"].unique().tolist())
        verdict = "CONTEXT_SIGNAL_EXISTS"
    else:
        contexts = []
        verdict = "NO_REPRODUCIBLE_CONTEXT_SIGNAL"

    best = summary.iloc[0].to_dict() if len(summary) else {}

    return {
        "verdict": verdict,
        "research_signal_contexts": contexts,
        "research_signal_model_context_pairs": int(len(signals)),
        "best_pair": {
            "model": best.get("model"),
            "context": best.get("context"),
            "late_stress_roc_auc": best.get("late_stress_roc_auc"),
            "late_stress_pr_lift": best.get("late_stress_pr_lift"),
            "median_roc_auc": best.get("median_roc_auc"),
            "roc_above_050_fraction": best.get("roc_above_050_fraction"),
        },
    }


def write_report(
    path: Path,
    verdict: Mapping[str, Any],
    summary: pd.DataFrame,
    consensus: pd.DataFrame,
    periods: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("# First Half 37 — V3 Context Temporal Stability Audit")
    lines.append("")
    lines.append("## Discipline")
    lines.append("")
    lines.append("- Aucun nouveau modèle entraîné.")
    lines.append("- Aucun hyperparamètre retuné.")
    lines.append("- Aucune feature ajoutée/retirée.")
    lines.append("- Aucun nouveau holdout utilisé.")
    lines.append("- Audit uniquement sur les prédictions déjà produites.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict['verdict']}**")
    lines.append("")
    lines.append(
        f"- Contextes RESEARCH_SIGNAL : "
        f"`{verdict['research_signal_contexts']}`"
    )
    lines.append(
        f"- Paires modèle/contexte RESEARCH_SIGNAL : "
        f"`{verdict['research_signal_model_context_pairs']}`"
    )
    lines.append("")
    lines.append("## Critères diagnostiques")
    lines.append("")
    lines.append(f"- périodes éligibles >= {MIN_ELIGIBLE_PERIODS}")
    lines.append(f"- ROC > 0.50 dans >= {MIN_POSITIVE_FRACTION:.0%}")
    lines.append(f"- PR lift > 0 dans >= {MIN_POSITIVE_FRACTION:.0%}")
    lines.append(f"- médiane ROC >= {MIN_MEDIAN_ROC:.2f}")
    lines.append(f"- LATE_STRESS ROC >= {MIN_LATE_ROC:.2f}")
    lines.append(f"- LATE_STRESS PR lift > {MIN_LATE_PR_LIFT:+.2f}")
    lines.append("")
    lines.append("## Consensus par contexte")
    lines.append("")
    lines.append("```")
    lines.append(consensus.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Ranking modèle × contexte")
    lines.append("")
    show = [
        "rank",
        "model",
        "context",
        "research_signal",
        "eligible_periods",
        "roc_above_050_fraction",
        "pr_lift_positive_fraction",
        "median_roc_auc",
        "mean_roc_auc",
        "mean_pr_lift",
        "late_stress_roc_auc",
        "late_stress_pr_lift",
    ]
    lines.append("```")
    lines.append(summary[show].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Détail temporel")
    lines.append("")
    show_p = [
        "model",
        "context",
        "period",
        "n",
        "positive_rate",
        "roc_auc",
        "pr_auc",
        "pr_lift",
        "brier_improvement",
        "logloss_improvement",
    ]
    lines.append("```")
    lines.append(
        periods[show_p]
        .sort_values(["context", "model", "period"])
        .to_string(index=False)
    )
    lines.append("```")
    lines.append("")
    lines.append("## Interprétation")
    lines.append("")
    lines.append(
        "Si le verdict est NO_REPRODUCIBLE_CONTEXT_SIGNAL, la branche "
        "First Half 37 doit être considérée comme non suffisamment robuste "
        "avec les données et features actuelles."
    )
    lines.append(
        "Si CONTEXT_SIGNAL_EXISTS apparaît, cela autorise seulement une "
        "dernière phase de développement spécialisée sur ce contexte. "
        "Un futur bloc entièrement neuf restera obligatoire pour validation."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit temporel par contexte sans nouvel entraînement."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FIRST HALF 37 — V3 CONTEXT TEMPORAL STABILITY AUDIT")
    print("=" * 120)
    print("AUCUN MODELE / AUCUN TUNING / AUCUNE FEATURE NOUVELLE")
    print()

    pred, ranking, manifest = validate_inputs()

    print(f"Prédictions chargées : {len(pred)}")
    print(f"Modèles déjà testés  : {pred[MODEL_COL].nunique()}")
    print(f"Contextes étudiés     : {', '.join(CONTEXTS)}")
    print()

    periods = period_table(pred)
    summary = summarize_contexts(periods)
    consensus = context_consensus(summary)
    verdict = automatic_verdict(summary, consensus)

    periods_path = out_dir / "first_half_37_v3_context_stability_periods.csv"
    summary_path = out_dir / "first_half_37_v3_context_stability_summary.csv"
    consensus_path = out_dir / "first_half_37_v3_context_stability_consensus.csv"
    verdict_path = out_dir / "first_half_37_v3_context_stability_verdict.json"
    report_path = out_dir / "first_half_37_v3_context_stability_report.md"
    manifest_out = out_dir / "first_half_37_v3_context_stability_manifest.json"

    periods.to_csv(periods_path, index=False)
    summary.to_csv(summary_path, index=False)
    consensus.to_csv(consensus_path, index=False)
    verdict_path.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        report_path,
        verdict=verdict,
        summary=summary,
        consensus=consensus,
        periods=periods,
    )

    manifest_payload = {
        "schema_version": "first_half_37_v3_context_stability_audit_v1",
        "created_at_utc": now_utc_iso(),
        "mode": "diagnostic_only_no_training_no_tuning",
        "source_predictions": str(PREDICTIONS_PATH),
        "source_predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "source_benchmark_manifest": str(MANIFEST_PATH),
        "source_benchmark_manifest_sha256": sha256_file(MANIFEST_PATH),
        "new_holdout_used": False,
        "criteria": {
            "min_n_period": MIN_N_PERIOD,
            "min_eligible_periods": MIN_ELIGIBLE_PERIODS,
            "min_positive_fraction": MIN_POSITIVE_FRACTION,
            "min_median_roc": MIN_MEDIAN_ROC,
            "min_late_roc": MIN_LATE_ROC,
            "min_late_pr_lift": MIN_LATE_PR_LIFT,
        },
        "verdict": verdict,
    }

    manifest_out.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 120)
    print("CONTEXT TEMPORAL STABILITY AUDIT TERMINE")
    print("=" * 120)
    print(f"VERDICT                  : {verdict['verdict']}")
    print(
        "Contextes signal         : "
        + (
            ", ".join(verdict["research_signal_contexts"])
            if verdict["research_signal_contexts"]
            else "aucun"
        )
    )
    print(
        f"Paires modèle/contexte   : "
        f"{verdict['research_signal_model_context_pairs']}"
    )
    print()

    if len(summary):
        print("TOP 8 PAIRES MODELE × CONTEXTE")
        cols = [
            "rank",
            "model",
            "context",
            "research_signal",
            "eligible_periods",
            "roc_above_050_fraction",
            "median_roc_auc",
            "mean_pr_lift",
            "late_stress_roc_auc",
            "late_stress_pr_lift",
        ]
        print(summary[cols].head(8).to_string(index=False))

    print()
    print(f"Rapport                  : {report_path}")
    print(f"Summary                  : {summary_path}")
    print(f"Periods                  : {periods_path}")
    print(f"Consensus                : {consensus_path}")
    print()
    print(
        "IMPORTANT: résultat diagnostique uniquement. "
        "Aucun candidat n'est validé production."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
