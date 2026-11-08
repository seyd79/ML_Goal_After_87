#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOAL AFTER 87 — V3-C 55–65 : CONTEXTE LIGUE + FREEZE AVANT RÉSERVE 168
=============================================================================

Objectif
--------
Construire un contexte ligue SPÉCIFIQUE à V3-C 55–65 en utilisant uniquement
ROLLING_OOS_684, puis le tester sur les 500 matchs externes déjà consommés
(Batch1=279 + Batch2=221), sans jamais charger les 168 matchs de réserve.

Méthode
-------
1) ROLLING_OOS_684 :
   - règle de base : 0.55 <= p_v3c < 0.65 ;
   - estimation par ligue du taux naturel et du taux dans la règle ;
   - shrinkage vers les taux globaux OOS ;
   - score conservateur = (taux sélectionné shrunk - base shrunk) * fiabilité ;
   - tiers LOW / MID / HIGH appris UNIQUEMENT sur OOS.

2) Externe consommé :
   - Batch1 279 et Batch2 221 sont scorés avec l'artefact V3-C final ;
   - aucune borne n'est retunée ;
   - le mapping ligue OOS est appliqué tel quel ;
   - on mesure la stabilité de règles de contexte prédéclarées.

3) Freeze avant réserve 168 :
   - V3-C 55–65 reste toujours la règle baseline figée ;
   - au maximum UNE règle de contexte ligue peut être promue comme challenger ;
   - la promotion suit des critères écrits dans ce fichier AVANT lecture des
     résultats externes par ce script ;
   - la réserve 168 n'est jamais ouverte ni lue.

Le scanner de production n'est jamais modifié.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

# ------------------------------ ENTRÉES -------------------------------------
OOS_PRED = (
    ROOT
    / "outputs/grand_volume_rules_audit"
    / "rolling_oos_684_exact_predictions.csv"
)
HIST = (
    ROOT
    / "outputs/historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
)
BATCH1 = (
    ROOT
    / "outputs/v3i_unseen_holdout_500/batch1_frozen_279"
    / "V3I_BATCH1_279_FROZEN.csv"
)
BATCH2 = (
    ROOT
    / "outputs/v3i_unseen_holdout_500/batch2_frozen_221"
    / "V3I_BATCH2_221_READY.csv"
)
V3C_MODEL = (
    ROOT
    / "outputs/model_v3_experiments/candidates_refit_all"
    / "V3_C_AGGRESSIVE_refit_all.joblib"
)
V3C_SCHEMA = (
    ROOT
    / "outputs/model_v3_experiments/schemas"
    / "V3_C_AGGRESSIVE_schema.json"
)

OUT = ROOT / "outputs/v3c_league_context_freeze_before_reserve168"

# ------------------------------ CONSTANTES ---------------------------------
TARGET = "goal_after_87"
MATCH_ID = "match_id"
COMP = "competition_id"
P_COL = "p_v3c"
RULE_LO = 0.55
RULE_HI = 0.65
PRIOR_STRENGTH = 20.0
BOOTSTRAPS = 5000
RANDOM_STATE = 42

# Fidélité connue avant ce script.
EXPECTED_OOS_N = 684
EXPECTED_OOS_SELECTED_N = 157
EXPECTED_OOS_SELECTED_POS = 49
EXPECTED_BATCH1_N = 279
EXPECTED_BATCH1_SELECTED_N = 55
EXPECTED_BATCH1_SELECTED_POS = 20
EXPECTED_BATCH2_N = 221
EXPECTED_BATCH2_SELECTED_N = 58
EXPECTED_BATCH2_SELECTED_POS = 23

# ---------------------------------------------------------------------------
# CANDIDATS DE CONTEXTE PRÉDÉCLARÉS
# ---------------------------------------------------------------------------
# Aucun nouveau seuil de probabilité V3-C n'est recherché.
# Ces trois scénarios sont définis uniquement par le mapping appris sur OOS.
FREEZE_CANDIDATES = [
    "KNOWN_OOS_LEAGUES_ONLY",
    "POSITIVE_OOS_LEAGUE_SCORE_ONLY",
    "HIGH_OR_MID_OOS_LEAGUE_SCORE",
]

# Critères de promotion d'UN seul challenger ligue pour le test final réserve.
# Le but est d'éviter de promouvoir un petit sous-groupe spectaculaire mais
# instable.
FREEZE_MIN_COMBINED_N = 40
FREEZE_MIN_EACH_BATCH_N = 12
FREEZE_MIN_COMBINED_LIFT_VS_V3C = 0.02     # +2 points minimum vs V3-C baseline
FREEZE_MIN_EACH_BATCH_LIFT_VS_V3C = 0.00   # ne doit pas dégrader B1 ou B2


def hr(title: str) -> None:
    print()
    print("=" * 126)
    print(title)
    print("=" * 126)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Lecture CSV/TSV tolérante sans masquer les erreurs de contenu.

    Le fichier OOS historique a déjà existé sous plusieurs encodages / séparateurs
    selon les scripts qui l'ont produit. On essaie donc plusieurs combinaisons,
    mais on ne saute AUCUNE ligne : la fidélité 684/157/49 reste obligatoire.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise IsADirectoryError(path)

    attempts = []
    errors = []
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "utf-16"]
    seps = [",", "\t", ";", None]

    for enc in encodings:
        for sep in seps:
            # sep=None nécessite le moteur python pour le sniffing.
            engines = ["python"] if sep is None else ["c", "python"]
            for engine in engines:
                try:
                    kwargs = dict(
                        filepath_or_buffer=path,
                        sep=sep,
                        engine=engine,
                        encoding=enc,
                    )
                    if engine == "c":
                        kwargs["low_memory"] = False
                    d = pd.read_csv(**kwargs)
                    if d.shape[1] >= 2:
                        attempts.append((d.shape[1], d, enc, sep, engine))
                except Exception as e:
                    errors.append(f"enc={enc} sep={repr(sep)} engine={engine}: {type(e).__name__}: {e}")

    if not attempts:
        sample = "\n".join(errors[-12:])
        raise RuntimeError(
            f"Impossible de lire {path}.\n"
            f"Dernières erreurs de parsing :\n{sample}"
        )

    attempts.sort(key=lambda x: x[0], reverse=True)
    return attempts[0][1]


def _normalize_oos_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise uniquement les alias connus, sans modifier les valeurs."""
    out = df.copy()

    if TARGET not in out.columns:
        target_aliases = ["target", "label", "y", "y_true"]
        for c in target_aliases:
            if c in out.columns:
                out = out.rename(columns={c: TARGET})
                break

    if P_COL not in out.columns:
        p_aliases = [
            "v3c_probability", "v3c_prob", "prob_v3c", "probability_v3c",
            "p_v3c_oos", "v3c_p", "probability", "p_goal_after_87_v3c",
        ]
        for c in p_aliases:
            if c in out.columns:
                out = out.rename(columns={c: P_COL})
                break

    return out


def locate_and_read_oos_predictions() -> Tuple[pd.DataFrame, Path]:
    """Trouve le fichier exact de prédictions rolling OOS 684.

    On privilégie le chemin historique attendu. S'il est illisible, on cherche
    automatiquement une copie équivalente dans outputs/ au lieu de bloquer sur
    un ancien fichier mal encodé.
    """
    candidates: List[Path] = []
    if OOS_PRED.exists():
        candidates.append(OOS_PRED)

    outputs = ROOT / "outputs"
    patterns = [
        "**/rolling_oos_684_exact_predictions.csv",
        "**/*rolling*oos*684*pred*.csv",
        "**/*oos*684*pred*.csv",
        "**/*rolling*oos*pred*.csv",
    ]
    if outputs.exists():
        for pat in patterns:
            for q in outputs.glob(pat):
                if q.is_file() and q not in candidates:
                    candidates.append(q)

    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier de prédictions rolling OOS trouvé sous outputs/. "
            f"Chemin attendu : {OOS_PRED}"
        )

    diagnostics = []
    for q in candidates:
        try:
            d = _normalize_oos_prediction_columns(read_csv_robust(q))
        except Exception as e:
            diagnostics.append(f"{q}: lecture échouée -> {e}")
            continue

        required = {MATCH_ID, TARGET, P_COL}
        missing = required - set(d.columns)
        if missing:
            diagnostics.append(
                f"{q}: colonnes manquantes {sorted(missing)}; "
                f"colonnes vues={list(d.columns)[:20]}"
            )
            continue
        if len(d) != EXPECTED_OOS_N:
            diagnostics.append(f"{q}: {len(d)} lignes, attendu {EXPECTED_OOS_N}")
            continue

        return d, q

    msg = "\n".join(diagnostics[-20:])
    raise RuntimeError(
        "Impossible d'identifier un fichier rolling OOS 684 exploitable.\n"
        "Diagnostics :\n" + msg
    )


def wilson(pos: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = pos / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-m, c+m


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(v) else f"{v:.2%}"


def qcut_tertiles(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(
        ranked,
        q=3,
        labels=["LOW", "MID", "HIGH"],
        duplicates="drop",
    ).astype(str)


def feature_list_from_model_or_schema(model, schema_path: Path) -> List[str]:
    # 1) sklearn Pipeline/DataFrame fit : meilleure source si disponible.
    if hasattr(model, "feature_names_in_"):
        f = [str(x) for x in model.feature_names_in_.tolist()]
        if len(f) == 38:
            return f

    # 2) Certains pipelines exposent les noms via une étape.
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                f = [str(x) for x in step.feature_names_in_.tolist()]
                if len(f) == 38:
                    return f

    # 3) Schéma JSON officiel.
    if not schema_path.exists():
        raise RuntimeError(
            "Impossible de récupérer les 38 features depuis le modèle et schéma absent : "
            f"{schema_path}"
        )

    obj = json.loads(schema_path.read_text(encoding="utf-8"))

    preferred_keys = [
        "features", "feature_names", "columns", "model_features", "selected_features"
    ]
    if isinstance(obj, dict):
        for k in preferred_keys:
            v = obj.get(k)
            if isinstance(v, list) and len(v) == 38 and all(isinstance(x, str) for x in v):
                return list(v)

    # Recherche récursive très conservatrice : première liste de 38 strings.
    found: List[List[str]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if len(x) == 38 and all(isinstance(y, str) for y in x):
                found.append(list(x))
            else:
                for v in x:
                    walk(v)

    walk(obj)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        # Si plusieurs listes identiques, c'est acceptable.
        uniq = []
        for f in found:
            if f not in uniq:
                uniq.append(f)
        if len(uniq) == 1:
            return uniq[0]

    raise RuntimeError("Impossible d'identifier sans ambiguïté les 38 features V3-C.")


def score_external_v3c(df: pd.DataFrame, model, features: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"Features V3-C manquantes : {missing}")

    X = df[list(features)].apply(pd.to_numeric, errors="coerce")
    # Même exigence que l'évaluation externe précédente : bloc exact exploitable.
    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr).all(axis=1))[0]
        raise RuntimeError(
            "Lignes externes non finies sur les 38 features. "
            f"Exemples index={bad[:10].tolist()}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if not y.isin([0, 1]).all():
        raise RuntimeError("Targets externes incomplets/non binaires.")

    out = df.copy()
    out[TARGET] = y.astype(int)
    out[COMP] = out[COMP].astype(str)
    out[P_COL] = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return out


def build_oos_with_league() -> Tuple[pd.DataFrame, Path]:
    pred, pred_path = locate_and_read_oos_predictions()
    hist = read_csv_robust(HIST)

    required_pred = {MATCH_ID, TARGET, P_COL}
    miss = required_pred - set(pred.columns)
    if miss:
        raise RuntimeError(f"OOS predictions : colonnes manquantes {sorted(miss)}")
    if MATCH_ID not in hist.columns or COMP not in hist.columns:
        raise RuntimeError("Dataset historique : match_id/competition_id absent.")

    meta_cols = [MATCH_ID, COMP]
    for c in ["home_team", "away_team", "kickoff_utc", "season_id"]:
        if c in hist.columns:
            meta_cols.append(c)

    meta = hist[meta_cols].drop_duplicates(subset=[MATCH_ID], keep="last").copy()
    out = pred.merge(meta, on=MATCH_ID, how="left", validate="one_to_one")

    if len(out) != EXPECTED_OOS_N:
        raise RuntimeError(f"OOS={len(out)}, attendu={EXPECTED_OOS_N}.")
    if out[COMP].isna().any():
        n = int(out[COMP].isna().sum())
        raise RuntimeError(f"competition_id manquant pour {n} lignes OOS.")

    out[COMP] = out[COMP].astype(str)
    out[TARGET] = pd.to_numeric(out[TARGET], errors="raise").astype(int)
    out[P_COL] = pd.to_numeric(out[P_COL], errors="raise").astype(float)
    return out, pred_path


def selected_mask(df: pd.DataFrame) -> pd.Series:
    return (df[P_COL] >= RULE_LO) & (df[P_COL] < RULE_HI)


def check_fidelity(oos: pd.DataFrame, b1: pd.DataFrame, b2: pd.DataFrame) -> None:
    checks = []
    for label, d, exp_n, exp_sel, exp_pos in [
        ("OOS_684", oos, EXPECTED_OOS_N, EXPECTED_OOS_SELECTED_N, EXPECTED_OOS_SELECTED_POS),
        ("BATCH1_279", b1, EXPECTED_BATCH1_N, EXPECTED_BATCH1_SELECTED_N, EXPECTED_BATCH1_SELECTED_POS),
        ("BATCH2_221", b2, EXPECTED_BATCH2_N, EXPECTED_BATCH2_SELECTED_N, EXPECTED_BATCH2_SELECTED_POS),
    ]:
        m = selected_mask(d)
        n = len(d)
        sel = int(m.sum())
        pos = int(d.loc[m, TARGET].sum())
        checks.append((label, n, sel, pos))
        if (n, sel, pos) != (exp_n, exp_sel, exp_pos):
            raise RuntimeError(
                f"FIDÉLITÉ ÉCHOUÉE {label}: obtenu n/sel/pos={n}/{sel}/{pos}, "
                f"attendu={exp_n}/{exp_sel}/{exp_pos}."
            )

    hr("FIDÉLITÉ V3-C")
    for label, n, sel, pos in checks:
        print(f"{label:<12} n={n:>3} | V3C55-65={pos}/{sel} = {pos/sel:.2%}")


def build_oos_league_mapping(oos: pd.DataFrame) -> pd.DataFrame:
    global_base = float(oos[TARGET].mean())
    sel = oos[selected_mask(oos)].copy()
    global_selected = float(sel[TARGET].mean())

    rows: List[Dict[str, Any]] = []
    for cid, g in oos.groupby(COMP, dropna=False):
        cid = str(cid)
        gs = g[selected_mask(g)]

        n_all = len(g)
        pos_all = int(g[TARGET].sum())
        n_sel = len(gs)
        pos_sel = int(gs[TARGET].sum()) if n_sel else 0

        base_raw = pos_all / n_all if n_all else float("nan")
        selected_raw = pos_sel / n_sel if n_sel else float("nan")

        base_shrunk = (
            pos_all + PRIOR_STRENGTH * global_base
        ) / (n_all + PRIOR_STRENGTH)

        selected_shrunk = (
            pos_sel + PRIOR_STRENGTH * global_selected
        ) / (n_sel + PRIOR_STRENGTH)

        effect = selected_shrunk - base_shrunk
        reliability = n_sel / (n_sel + PRIOR_STRENGTH)
        score = effect * reliability

        row: Dict[str, Any] = {
            COMP: cid,
            "oos_league_n": n_all,
            "oos_league_positives": pos_all,
            "oos_league_base_rate": base_raw,
            "oos_v3c_selected_n": n_sel,
            "oos_v3c_selected_positives": pos_sel,
            "oos_v3c_selected_rate": selected_raw,
            "oos_league_base_rate_shrunk": base_shrunk,
            "oos_v3c_selected_rate_shrunk": selected_shrunk,
            "oos_v3c_league_effect_shrunk": effect,
            "oos_v3c_reliability": reliability,
            "oos_v3c_conservative_league_score": score,
            "oos_global_base_rate": global_base,
            "oos_global_v3c_selected_rate": global_selected,
        }

        for c in ["home_team", "away_team", "season_id"]:
            if c in g.columns:
                # metadata illustrative uniquement, non utilisée pour le score.
                row[f"example_{c}"] = str(g[c].iloc[0])

        rows.append(row)

    scores = pd.DataFrame(rows)
    trainable = scores["oos_v3c_selected_n"] >= 1
    scores["v3c_league_tier"] = "NO_OOS_V3C_SELECTED_HISTORY"
    if int(trainable.sum()) >= 3:
        scores.loc[trainable, "v3c_league_tier"] = qcut_tertiles(
            scores.loc[trainable, "oos_v3c_conservative_league_score"]
        ).values

    scores["v3c_positive_oos_score"] = (
        scores["oos_v3c_conservative_league_score"] > 0
    ) & trainable

    return scores.sort_values(
        ["v3c_league_tier", "oos_v3c_conservative_league_score", "oos_v3c_selected_n"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def apply_mapping(scored: pd.DataFrame, mapping: pd.DataFrame, sample: str) -> pd.DataFrame:
    cols = [
        COMP,
        "oos_league_n",
        "oos_v3c_selected_n",
        "oos_v3c_selected_positives",
        "oos_league_base_rate",
        "oos_v3c_selected_rate",
        "oos_v3c_league_effect_shrunk",
        "oos_v3c_reliability",
        "oos_v3c_conservative_league_score",
        "v3c_league_tier",
        "v3c_positive_oos_score",
    ]
    m = mapping[cols].copy()
    m[COMP] = m[COMP].astype(str)

    out = scored.copy()
    out[COMP] = out[COMP].astype(str)
    out = out.merge(m, on=COMP, how="left", validate="many_to_one")
    out["sample"] = sample
    out["v3c_league_known_oos"] = out["oos_league_n"].notna()
    out["v3c_league_tier_external"] = out["v3c_league_tier"].fillna("UNSEEN_LEAGUE")
    out["v3c_positive_oos_score_external"] = out["v3c_positive_oos_score"].fillna(False).astype(bool)
    return out


def scenario_masks(selected: pd.DataFrame) -> Dict[str, pd.Series]:
    idx = selected.index
    return {
        "ALL_V3C_55_65": pd.Series(True, index=idx),
        "KNOWN_OOS_LEAGUES_ONLY": selected["v3c_league_known_oos"].astype(bool),
        "POSITIVE_OOS_LEAGUE_SCORE_ONLY": selected["v3c_positive_oos_score_external"].astype(bool),
        "HIGH_OR_MID_OOS_LEAGUE_SCORE": selected["v3c_league_tier_external"].isin(["HIGH", "MID"]),
        "HIGH_OOS_LEAGUE_SCORE_ONLY": selected["v3c_league_tier_external"] == "HIGH",
        "MID_OOS_LEAGUE_SCORE_ONLY": selected["v3c_league_tier_external"] == "MID",
        "LOW_OOS_LEAGUE_SCORE_ONLY": selected["v3c_league_tier_external"] == "LOW",
        "NO_OOS_V3C_SELECTED_HISTORY_ONLY": selected["v3c_league_tier_external"] == "NO_OOS_V3C_SELECTED_HISTORY",
        "UNSEEN_LEAGUE_ONLY": selected["v3c_league_tier_external"] == "UNSEEN_LEAGUE",
    }


def scenario_table(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    selected = df[selected_mask(df)].copy()
    baseline_n = len(selected)
    baseline_pos = int(selected[TARGET].sum()) if baseline_n else 0
    baseline_rate = baseline_pos / baseline_n if baseline_n else float("nan")
    sample_base = float(df[TARGET].mean())

    rows = []
    for scenario, mask in scenario_masks(selected).items():
        g = selected[mask]
        n = len(g)
        pos = int(g[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        status = (
            "FROZEN_BASELINE" if scenario == "ALL_V3C_55_65"
            else "FREEZE_CANDIDATE" if scenario in FREEZE_CANDIDATES
            else "DIAGNOSTIC"
        )

        rows.append({
            "sample": sample,
            "scenario": scenario,
            "scenario_status": status,
            "sample_n": len(df),
            "sample_base_rate": sample_base,
            "v3c55_65_selected_n": baseline_n,
            "v3c55_65_selected_positives": baseline_pos,
            "v3c55_65_rate": baseline_rate,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "lift_vs_v3c55_65_pp": rate - baseline_rate if n else float("nan"),
            "lift_vs_sample_pp": rate - sample_base if n else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
            "fair_odds_empirical": 1/rate if n and rate > 0 else float("nan"),
            "roi_at_3.20": rate*3.20 - 1 if n else float("nan"),
        })

    return pd.DataFrame(rows)


def bootstrap_auc(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, float, int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]

    if len(y) < 5 or len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return float("nan"), float("nan"), float("nan"), 0

    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(RANDOM_STATE)
    vals = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        ss = score[idx]
        if len(np.unique(yy)) < 2 or len(np.unique(ss)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, ss)))

    if not vals:
        return auc, float("nan"), float("nan"), 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi), len(vals)


def auc_table(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    sel = df[selected_mask(df)].copy()
    score = pd.to_numeric(sel["oos_v3c_conservative_league_score"], errors="coerce").to_numpy()
    auc, lo, hi, b = bootstrap_auc(sel[TARGET].to_numpy(), score)
    return pd.DataFrame([{
        "sample": sample,
        "v3c55_65_selected_n": len(sel),
        "selected_n_with_known_oos_v3c_league_score": int(np.isfinite(score).sum()),
        "auc_oos_v3c_league_score": auc,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "bootstrap_valid_replicates": b,
    }])


def choose_challenger(all_scenarios: pd.DataFrame) -> Dict[str, Any]:
    def row(sample: str, scenario: str) -> pd.Series:
        z = all_scenarios[
            (all_scenarios["sample"] == sample)
            & (all_scenarios["scenario"] == scenario)
        ]
        if len(z) != 1:
            raise RuntimeError(f"Résultat manquant/dupliqué : {sample} / {scenario}")
        return z.iloc[0]

    considered = []
    eligible = []

    for s in FREEZE_CANDIDATES:
        b1 = row("BATCH1_279", s)
        b2 = row("BATCH2_221", s)
        co = row("COMBINED_500", s)

        rec = {
            "scenario": s,
            "batch1_n": int(b1["selected_n"]),
            "batch1_rate": float(b1["selected_rate"]) if pd.notna(b1["selected_rate"]) else None,
            "batch1_lift_vs_v3c": float(b1["lift_vs_v3c55_65_pp"]) if pd.notna(b1["lift_vs_v3c55_65_pp"]) else None,
            "batch2_n": int(b2["selected_n"]),
            "batch2_rate": float(b2["selected_rate"]) if pd.notna(b2["selected_rate"]) else None,
            "batch2_lift_vs_v3c": float(b2["lift_vs_v3c55_65_pp"]) if pd.notna(b2["lift_vs_v3c55_65_pp"]) else None,
            "combined_n": int(co["selected_n"]),
            "combined_rate": float(co["selected_rate"]) if pd.notna(co["selected_rate"]) else None,
            "combined_lift_vs_v3c": float(co["lift_vs_v3c55_65_pp"]) if pd.notna(co["lift_vs_v3c55_65_pp"]) else None,
        }

        passes = (
            rec["combined_n"] >= FREEZE_MIN_COMBINED_N
            and rec["batch1_n"] >= FREEZE_MIN_EACH_BATCH_N
            and rec["batch2_n"] >= FREEZE_MIN_EACH_BATCH_N
            and rec["combined_lift_vs_v3c"] is not None
            and rec["combined_lift_vs_v3c"] >= FREEZE_MIN_COMBINED_LIFT_VS_V3C
            and rec["batch1_lift_vs_v3c"] is not None
            and rec["batch1_lift_vs_v3c"] >= FREEZE_MIN_EACH_BATCH_LIFT_VS_V3C
            and rec["batch2_lift_vs_v3c"] is not None
            and rec["batch2_lift_vs_v3c"] >= FREEZE_MIN_EACH_BATCH_LIFT_VS_V3C
        )
        rec["passes_freeze_criteria"] = bool(passes)
        considered.append(rec)
        if passes:
            eligible.append(rec)

    chosen = None
    if eligible:
        # Priorité à la stabilité : meilleur pire lift entre Batch1 et Batch2,
        # puis lift combiné, puis volume combiné.
        eligible.sort(
            key=lambda r: (
                min(r["batch1_lift_vs_v3c"], r["batch2_lift_vs_v3c"]),
                r["combined_lift_vs_v3c"],
                r["combined_n"],
            ),
            reverse=True,
        )
        chosen = eligible[0]

    return {
        "baseline_frozen": {
            "name": "V3C_MAIN_55_65",
            "rule": "0.55 <= p_v3c < 0.65",
        },
        "league_challenger_frozen": chosen,
        "all_candidates_considered": considered,
        "criteria": {
            "min_combined_n": FREEZE_MIN_COMBINED_N,
            "min_each_batch_n": FREEZE_MIN_EACH_BATCH_N,
            "min_combined_lift_vs_v3c": FREEZE_MIN_COMBINED_LIFT_VS_V3C,
            "min_each_batch_lift_vs_v3c": FREEZE_MIN_EACH_BATCH_LIFT_VS_V3C,
        },
        "interpretation": (
            "Baseline V3-C + challenger ligue figés prospectivement pour réserve 168."
            if chosen is not None
            else "Aucun contexte ligue ne passe les critères; seule la baseline V3-C est figée pour réserve 168."
        ),
    }


def pretty_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in [
        "sample_base_rate", "v3c55_65_rate", "selected_rate",
        "lift_vs_v3c55_65_pp", "lift_vs_sample_pp", "ci95_low", "ci95_high",
        "roi_at_3.20",
    ]:
        x[c] = x[c].map(pct)
    x["fair_odds_empirical"] = x["fair_odds_empirical"].map(
        lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    )
    return x


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    required = [HIST, BATCH1, BATCH2, V3C_MODEL]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    hr("V3-C 55–65 — CONTEXTE LIGUE APPRIS UNIQUEMENT SUR OOS 684")
    print("Réserve 168 : NON CHARGÉE / NON LUE")

    model = joblib.load(V3C_MODEL)
    features = feature_list_from_model_or_schema(model, V3C_SCHEMA)
    if len(features) != 38:
        raise RuntimeError(f"Schéma V3-C={len(features)} features, attendu=38.")
    print(f"Features V3-C : {len(features)}/38")

    oos, oos_pred_path = build_oos_with_league()
    print(f"Prédictions OOS utilisées : {oos_pred_path}")
    b1_raw = read_csv_robust(BATCH1)
    b2_raw = read_csv_robust(BATCH2)

    if len(b1_raw) != EXPECTED_BATCH1_N:
        raise RuntimeError(f"Batch1={len(b1_raw)}, attendu={EXPECTED_BATCH1_N}")
    if len(b2_raw) != EXPECTED_BATCH2_N:
        raise RuntimeError(f"Batch2={len(b2_raw)}, attendu={EXPECTED_BATCH2_N}")

    for name, d in [("Batch1", b1_raw), ("Batch2", b2_raw)]:
        for c in [MATCH_ID, COMP, TARGET]:
            if c not in d.columns:
                raise RuntimeError(f"{name}: colonne {c} absente")

    b1 = score_external_v3c(b1_raw, model, features)
    b2 = score_external_v3c(b2_raw, model, features)

    # Strictement aucun overlap B1/B2.
    overlap = set(b1[MATCH_ID].astype(str)) & set(b2[MATCH_ID].astype(str))
    if overlap:
        raise RuntimeError(f"Overlap Batch1/Batch2 : {list(overlap)[:10]}")

    check_fidelity(oos, b1, b2)

    mapping = build_oos_league_mapping(oos)
    mapping_path = OUT / "oos_v3c55_65_league_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    mapping_hash = sha256_file(mapping_path)

    b1c = apply_mapping(b1, mapping, "BATCH1_279")
    b2c = apply_mapping(b2, mapping, "BATCH2_221")
    combined = pd.concat([b1c, b2c], ignore_index=True)
    combined["sample"] = "COMBINED_500"

    tables = pd.concat([
        scenario_table(b1c, "BATCH1_279"),
        scenario_table(b2c, "BATCH2_221"),
        scenario_table(combined, "COMBINED_500"),
    ], ignore_index=True)

    aucs = pd.concat([
        auc_table(b1c, "BATCH1_279"),
        auc_table(b2c, "BATCH2_221"),
        auc_table(combined, "COMBINED_500"),
    ], ignore_index=True)

    decision = choose_challenger(tables)
    decision.update({
        "created_at_utc": now_utc(),
        "mapping_sha256": mapping_hash,
        "mapping_source": "ROLLING_OOS_684 only",
        "external_development_set": "Batch1_279 + Batch2_221 = 500 already-consumed matches",
        "reserve168_loaded": False,
        "reserve168_evaluated": False,
        "production_scanner_modified": False,
    })

    # Sorties détaillées.
    tables.to_csv(OUT / "external500_v3c_league_scenarios.csv", index=False)
    aucs.to_csv(OUT / "external500_v3c_league_score_auc.csv", index=False)
    b1c[selected_mask(b1c)].to_csv(
        OUT / "batch1_v3c55_65_selected_with_oos_league_context.csv", index=False
    )
    b2c[selected_mask(b2c)].to_csv(
        OUT / "batch2_v3c55_65_selected_with_oos_league_context.csv", index=False
    )
    combined[selected_mask(combined)].to_csv(
        OUT / "combined500_v3c55_65_selected_with_oos_league_context.csv", index=False
    )

    (OUT / "FROZEN_RULES_BEFORE_RESERVE168.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "created_at_utc": now_utc(),
        "method": "V3-C55-65 OOS684 league partial pooling -> external500 development -> prospective freeze for reserve168",
        "rule": "0.55 <= p_v3c < 0.65",
        "prior_strength": PRIOR_STRENGTH,
        "oos_n": len(oos),
        "batch1_n": len(b1),
        "batch2_n": len(b2),
        "combined_external_n": len(combined),
        "features_n": len(features),
        "oos_competitions": int(oos[COMP].nunique()),
        "batch1_competitions": int(b1[COMP].nunique()),
        "batch2_competitions": int(b2[COMP].nunique()),
        "mapping_sha256": mapping_hash,
        "freeze_candidates_predeclared": FREEZE_CANDIDATES,
        "reserve168_loaded": False,
        "production_scanner_modified": False,
        "inputs": {
            "oos_predictions": str(oos_pred_path),
            "historical_metadata": str(HIST),
            "batch1": str(BATCH1),
            "batch2": str(BATCH2),
            "v3c_model": str(V3C_MODEL),
        },
    }
    (OUT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    hr("CONTEXTE LIGUE V3-C — TEST SUR LES 500 EXTERNES DÉJÀ CONSOMMÉS")
    show_cols = [
        "sample", "scenario", "scenario_status", "selected_n", "selected_positives",
        "selected_rate", "v3c55_65_rate", "lift_vs_v3c55_65_pp", "sample_base_rate",
        "lift_vs_sample_pp", "ci95_low", "ci95_high", "fair_odds_empirical", "roi_at_3.20",
    ]
    print(pretty_scenarios(tables[show_cols]).to_string(index=False))

    hr("DISCRIMINATION CONTINUE DU SCORE LIGUE V3-C")
    a = aucs.copy()
    for c in ["auc_oos_v3c_league_score", "bootstrap_ci95_low", "bootstrap_ci95_high"]:
        a[c] = a[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    print(a.to_string(index=False))

    hr("FREEZE AVANT RÉSERVE 168")
    print("Baseline figée : V3-C 55–65")
    chosen = decision["league_challenger_frozen"]
    if chosen is None:
        print("Challenger ligue : AUCUN — aucun candidat ne passe les critères pré-déclarés.")
    else:
        print(f"Challenger ligue figé : {chosen['scenario']}")
        print(
            f"Combined500 : n={chosen['combined_n']} | rate={chosen['combined_rate']:.2%} | "
            f"lift vs V3-C={chosen['combined_lift_vs_v3c']:+.2%}"
        )
        print(
            f"Batch1 lift={chosen['batch1_lift_vs_v3c']:+.2%} | "
            f"Batch2 lift={chosen['batch2_lift_vs_v3c']:+.2%}"
        )

    print("\nIMPORTANT : la réserve 168 n'a pas été chargée par ce script.")
    print("Le scanner de production n'a pas été modifié.")

    hr("SORTIES")
    print(OUT)
    for name in [
        "oos_v3c55_65_league_mapping.csv",
        "external500_v3c_league_scenarios.csv",
        "external500_v3c_league_score_auc.csv",
        "batch1_v3c55_65_selected_with_oos_league_context.csv",
        "batch2_v3c55_65_selected_with_oos_league_context.csv",
        "combined500_v3c55_65_selected_with_oos_league_context.csv",
        "FROZEN_RULES_BEFORE_RESERVE168.json",
        "evaluation_manifest.json",
    ]:
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
