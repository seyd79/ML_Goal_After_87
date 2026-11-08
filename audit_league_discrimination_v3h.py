#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOAL AFTER 87 — AUDIT DISCRIMINATION PAR LIGUE
================================================

But
---
Mesurer si la performance des règles principales varie réellement par ligue,
sans modifier le modèle et sans confondre taux brut de la ligue avec valeur
ajoutée du filtre.

Blocs utilisés
--------------
- ROLLING_OOS_684_EXACT
- V3I_EXTERNAL_279
- OOS_PLUS_EXTERNAL_963

Règles
------
- V3H_MAIN_31_34   : 0.31 <= p_v3h < 0.34
- V3H_STRONG_32_34 : 0.32 <= p_v3h < 0.34
- V3C_MAIN_55_65   : 0.55 <= p_v3c < 0.65

Méthode
-------
A) Audit combiné 963:
   par compétition, calcul de:
   - N total
   - base rate de la ligue
   - N sélectionné
   - taux de réussite de la règle
   - uplift en points vs base rate de la ligue
   - lift relatif
   - IC95% Wilson
   - taux "shrunk" (empirical Bayes simple) vers le taux global de la règle

B) Stabilité:
   même tableau séparé sur OOS684 et V3I279.

C) Test méthodologiquement plus propre d'un filtre ligue:
   1. découverte des ligues favorables UNIQUEMENT sur OOS684;
   2. gel de cette whitelist/blacklist;
   3. test sur V3I279 sans regarder ses labels pour choisir les ligues.

IMPORTANT
---------
La whitelist obtenue sur OOS est une hypothèse de filtre.
Le test V3-I est la partie confirmatoire.
Aucune ligue n'est ajoutée au scanner automatiquement.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

AUDIT_DIR = ROOT / "outputs/grand_volume_rules_audit"

OOS_PATH = AUDIT_DIR / "rolling_oos_684_exact_predictions.csv"
V3I_PRED_PATH = AUDIT_DIR / "v3i_external_279_predictions.csv"

HIST_PATH = (
    ROOT
    / "outputs/historical_dataset_v2"
    / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"
)

V3I_READY_PATH = (
    ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv"
)

BATCH1_MANIFEST_CANDIDATES = [
    ROOT
    / "outputs/v3i_unseen_holdout_500/batch1_frozen_279"
    / "v3i_candidate_manifest_BATCH1_CONSUMED.csv",
    ROOT
    / "outputs/v3i_unseen_holdout_500"
    / "v3i_candidate_manifest.csv",
]

COMP_AUDIT_PATH = (
    ROOT
    / "outputs/historical_dataset_v2/audit"
    / "audit_competition_quality_v2.csv"
)

OUT_DIR = ROOT / "outputs/league_discrimination_audit"

TARGET = "goal_after_87"
MATCH_ID = "match_id"
COMP_ID = "competition_id"

RULES = [
    ("V3H_MAIN_31_34", "p_v3h", 0.31, 0.34),
    ("V3H_STRONG_32_34", "p_v3h", 0.32, 0.34),
    ("V3C_MAIN_55_65", "p_v3c", 0.55, 0.65),
]

# Pré-déclaré AVANT inspection des résultats ligue.
MIN_COMBINED_LEAGUE_N = 30
MIN_COMBINED_SELECTED_N = 15
MIN_BLOCK_SELECTED_N = 5

# Découverte OOS -> test externe.
DISCOVERY_MIN_LEAGUE_N = 20
DISCOVERY_MIN_SELECTED_N = 8
DISCOVERY_MIN_UPLIFT_PP = 0.05

# Shrinkage simple: 20 pseudo-observations au taux global de la règle.
EB_PRIOR_STRENGTH = 20.0


def hr(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    attempts = []
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            attempts.append((df.shape[1], sep, df))
        except Exception:
            pass

    if not attempts:
        raise RuntimeError(f"Impossible de lire {path}")

    attempts.sort(key=lambda x: x[0], reverse=True)
    return attempts[0][2]


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def wilson(pos: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = pos / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-m, c+m


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def build_competition_name_map(
    hist: pd.DataFrame,
    v3i: pd.DataFrame,
) -> Dict[str, str]:
    """
    Cherche un nom lisible de compétition dans plusieurs sources.
    Si aucun nom n'est disponible, competition_id reste l'identifiant officiel.
    """
    mapping: Dict[str, str] = {}

    name_candidates = [
        "competition_name",
        "league_name",
        "league",
        "competition",
        "tournament_name",
    ]

    def ingest(df: pd.DataFrame):
        if COMP_ID not in df.columns:
            return
        name_col = pick_first_existing(df, name_candidates)
        if not name_col:
            return

        for cid, name in zip(df[COMP_ID], df[name_col]):
            cid_s = safe_text(cid)
            name_s = safe_text(name)
            if cid_s and name_s and cid_s not in mapping:
                mapping[cid_s] = name_s

    ingest(hist)
    ingest(v3i)

    for path in BATCH1_MANIFEST_CANDIDATES:
        if path.exists():
            try:
                ingest(read_csv_robust(path))
            except Exception:
                pass

    if COMP_AUDIT_PATH.exists():
        try:
            ingest(read_csv_robust(COMP_AUDIT_PATH))
        except Exception:
            pass

    return mapping


def metadata_frame(
    df: pd.DataFrame,
    name_map: Mapping[str, str],
) -> pd.DataFrame:
    if MATCH_ID not in df.columns:
        raise RuntimeError("match_id absent du dataset metadata.")
    if COMP_ID not in df.columns:
        raise RuntimeError("competition_id absent du dataset metadata.")

    keep = [
        c for c in [
            MATCH_ID,
            COMP_ID,
            "competition_name",
            "league_name",
            "country",
            "country_name",
            "season_id",
            "home_team",
            "away_team",
            "kickoff_utc",
        ]
        if c in df.columns
    ]

    out = df[keep].copy()
    out[MATCH_ID] = normalize_id_series(out[MATCH_ID])
    out[COMP_ID] = normalize_id_series(out[COMP_ID])

    # Resolve a single readable name.
    direct_name_col = pick_first_existing(
        out,
        ["competition_name", "league_name"],
    )

    if direct_name_col:
        direct = out[direct_name_col].map(safe_text)
    else:
        direct = pd.Series("", index=out.index)

    out["competition_name_resolved"] = [
        direct.iloc[i]
        or name_map.get(safe_text(cid), "")
        or safe_text(cid)
        for i, cid in enumerate(out[COMP_ID])
    ]

    out = out.drop_duplicates(subset=[MATCH_ID], keep="first")
    return out


def attach_metadata(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    sample: str,
) -> pd.DataFrame:
    if MATCH_ID not in pred.columns:
        raise RuntimeError(f"{sample}: match_id absent des prédictions.")

    pred = pred.copy()
    pred[MATCH_ID] = normalize_id_series(pred[MATCH_ID])

    # If competition_id already present, still merge readable name.
    cols_to_merge = [
        c for c in meta.columns
        if c != MATCH_ID and c not in pred.columns
    ]

    out = pred.merge(
        meta[[MATCH_ID] + cols_to_merge],
        on=MATCH_ID,
        how="left",
        validate="many_to_one",
    )

    if COMP_ID not in out.columns:
        raise RuntimeError(
            f"{sample}: competition_id introuvable après jointure."
        )

    out[COMP_ID] = normalize_id_series(out[COMP_ID])

    if "competition_name_resolved" not in out.columns:
        out["competition_name_resolved"] = out[COMP_ID]

    missing_comp = (
        out[COMP_ID].isna()
        | (out[COMP_ID].astype(str).str.strip() == "")
        | (out[COMP_ID].astype(str) == "nan")
    )

    if missing_comp.any():
        raise RuntimeError(
            f"{sample}: {int(missing_comp.sum())} lignes sans competition_id."
        )

    out["sample"] = sample
    return out


def league_base_table(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    for cid, g in df.groupby(COMP_ID, dropna=False):
        n = len(g)
        pos = int(g[TARGET].sum())
        rate = pos/n if n else float("nan")
        rows.append({
            "sample": sample,
            COMP_ID: cid,
            "competition_name": (
                g["competition_name_resolved"].dropna().astype(str).iloc[0]
                if len(g)
                else str(cid)
            ),
            "league_n": n,
            "league_positives": pos,
            "league_base_rate": rate,
        })
    return pd.DataFrame(rows)


def rule_rows_for_sample(
    df: pd.DataFrame,
    sample: str,
    global_rule_rates: Mapping[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    base_by_league = league_base_table(df, sample).set_index(COMP_ID)

    for rule, col, lo, hi in RULES:
        global_rule_rate = float(global_rule_rates[rule])

        for cid, g in df.groupby(COMP_ID, dropna=False):
            base_row = base_by_league.loc[cid]
            league_n = int(base_row["league_n"])
            league_base = float(base_row["league_base_rate"])

            selected = g[
                (g[col] >= lo)
                & (g[col] < hi)
            ]

            n = len(selected)
            pos = int(selected[TARGET].sum()) if n else 0
            rate = pos/n if n else float("nan")
            ci_lo, ci_hi = wilson(pos, n)

            # Empirical Bayes / shrinkage toward global rule rate.
            shrunk_rate = (
                (pos + EB_PRIOR_STRENGTH * global_rule_rate)
                / (n + EB_PRIOR_STRENGTH)
                if n >= 0
                else float("nan")
            )

            rows.append({
                "sample": sample,
                COMP_ID: cid,
                "competition_name": base_row["competition_name"],
                "rule": rule,
                "league_n": league_n,
                "league_base_rate": league_base,
                "selected_n": n,
                "selected_positives": pos,
                "selected_rate": rate,
                "uplift_pp": (
                    rate - league_base
                    if n
                    else float("nan")
                ),
                "lift_ratio": (
                    rate / league_base
                    if n and league_base > 0
                    else float("nan")
                ),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "global_rule_rate": global_rule_rate,
                "shrunk_rate": shrunk_rate,
                "shrunk_uplift_vs_league_base": (
                    shrunk_rate - league_base
                    if math.isfinite(shrunk_rate)
                    else float("nan")
                ),
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


def global_rule_rates(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for rule, col, lo, hi in RULES:
        z = df[(df[col] >= lo) & (df[col] < hi)]
        out[rule] = float(z[TARGET].mean()) if len(z) else float("nan")
    return out


def classify_combined(
    combined_rules: pd.DataFrame,
    oos_rules: pd.DataFrame,
    ext_rules: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classification découverte sur le combined 963.
    Ce n'est PAS une promotion production.
    """
    rows = []

    keys = [COMP_ID, "rule"]

    oos_idx = oos_rules.set_index(keys)
    ext_idx = ext_rules.set_index(keys)

    for _, r in combined_rules.iterrows():
        cid = r[COMP_ID]
        rule = r["rule"]
        key = (cid, rule)

        league_n = int(r["league_n"])
        selected_n = int(r["selected_n"])
        uplift = float(r["uplift_pp"]) if pd.notna(r["uplift_pp"]) else float("nan")

        if league_n < MIN_COMBINED_LEAGUE_N or selected_n < MIN_COMBINED_SELECTED_N:
            cls = "INSUFFICIENT"
        else:
            oo = oos_idx.loc[key] if key in oos_idx.index else None
            ee = ext_idx.loc[key] if key in ext_idx.index else None

            o_n = int(oo["selected_n"]) if oo is not None else 0
            e_n = int(ee["selected_n"]) if ee is not None else 0
            o_u = float(oo["uplift_pp"]) if oo is not None and pd.notna(oo["uplift_pp"]) else float("nan")
            e_u = float(ee["uplift_pp"]) if ee is not None and pd.notna(ee["uplift_pp"]) else float("nan")

            both_testable = (
                o_n >= MIN_BLOCK_SELECTED_N
                and e_n >= MIN_BLOCK_SELECTED_N
                and math.isfinite(o_u)
                and math.isfinite(e_u)
            )

            if both_testable and uplift >= 0.05 and o_u > 0 and e_u > 0:
                cls = "FAVORABLE_CONFIRMED"
            elif both_testable and uplift <= -0.05 and o_u < 0 and e_u < 0:
                cls = "UNFAVORABLE_CONFIRMED"
            elif both_testable and (o_u > 0) != (e_u > 0):
                cls = "MIXED"
            elif uplift >= 0.05:
                cls = "FAVORABLE_SIGNAL"
            elif uplift <= -0.05:
                cls = "UNFAVORABLE_SIGNAL"
            else:
                cls = "NEUTRAL"

        row = dict(r)
        row["classification"] = cls
        rows.append(row)

    return pd.DataFrame(rows)


def discover_oos_league_filter(
    oos_rules: pd.DataFrame,
    rule_name: str = "V3H_MAIN_31_34",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = oos_rules[oos_rules["rule"] == rule_name].copy()

    eligible = r[
        (r["league_n"] >= DISCOVERY_MIN_LEAGUE_N)
        & (r["selected_n"] >= DISCOVERY_MIN_SELECTED_N)
    ].copy()

    whitelist = eligible[
        (eligible["uplift_pp"] >= DISCOVERY_MIN_UPLIFT_PP)
        & (eligible["shrunk_uplift_vs_league_base"] > 0)
    ].copy()

    blacklist = eligible[
        (eligible["uplift_pp"] <= -DISCOVERY_MIN_UPLIFT_PP)
        & (eligible["shrunk_uplift_vs_league_base"] < 0)
    ].copy()

    whitelist = whitelist.sort_values(
        ["shrunk_uplift_vs_league_base", "selected_n"],
        ascending=[False, False],
    )

    blacklist = blacklist.sort_values(
        ["shrunk_uplift_vs_league_base", "selected_n"],
        ascending=[True, False],
    )

    return whitelist, blacklist


def evaluate_filter_on_external(
    ext: pd.DataFrame,
    whitelist_ids: set,
    blacklist_ids: set,
) -> pd.DataFrame:
    """
    Test externe des listes décidées uniquement depuis OOS684.
    """
    rule_mask = (
        (ext["p_v3h"] >= 0.31)
        & (ext["p_v3h"] < 0.34)
    )

    scenarios = {
        "BASE_V3H_31_34_ALL_LEAGUES": rule_mask,
        "OOS_WHITELIST_ONLY": (
            rule_mask
            & ext[COMP_ID].astype(str).isin(whitelist_ids)
        ),
        "EXCLUDE_OOS_BLACKLIST": (
            rule_mask
            & ~ext[COMP_ID].astype(str).isin(blacklist_ids)
        ),
    }

    rows = []
    sample_base = float(ext[TARGET].mean())

    for name, mask in scenarios.items():
        z = ext[mask]
        n = len(z)
        pos = int(z[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        rows.append({
            "scenario": name,
            "external_sample_n": len(ext),
            "external_base_rate": sample_base,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "uplift_pp_vs_external_base": (
                rate - sample_base
                if n
                else float("nan")
            ),
            "ci95_low": lo,
            "ci95_high": hi,
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


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(v) else f"{v:.2%}"


def format_table(df: pd.DataFrame, pct_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(pct)
    for c in ["fair_odds_empirical"]:
        if c in out.columns:
            out[c] = out[c].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hr("AUDIT DISCRIMINATION PAR LIGUE")

    oos = pd.read_csv(OOS_PATH, low_memory=False)
    ext_pred = pd.read_csv(V3I_PRED_PATH, low_memory=False)

    hist = read_csv_robust(HIST_PATH)
    v3i_ready = read_csv_robust(V3I_READY_PATH)

    # Batch1 only if collection already extended.
    if len(v3i_ready) > 279 and "_v3i_selection_order" in v3i_ready.columns:
        v3i_ready["_v3i_selection_order"] = pd.to_numeric(
            v3i_ready["_v3i_selection_order"],
            errors="coerce",
        )
        v3i_ready = (
            v3i_ready
            .sort_values(["_v3i_selection_order", MATCH_ID], kind="mergesort")
            .iloc[:279]
            .copy()
        )

    if len(oos) != 684:
        raise RuntimeError(f"OOS={len(oos)}, attendu=684.")
    if len(ext_pred) != 279:
        raise RuntimeError(f"V3-I={len(ext_pred)}, attendu=279.")

    name_map = build_competition_name_map(hist, v3i_ready)

    hist_meta = metadata_frame(hist, name_map)
    v3i_meta = metadata_frame(v3i_ready, name_map)

    oos = attach_metadata(
        oos,
        hist_meta,
        "ROLLING_OOS_684_EXACT",
    )
    ext = attach_metadata(
        ext_pred,
        v3i_meta,
        "V3I_EXTERNAL_279",
    )

    combined = pd.concat(
        [oos.copy(), ext.copy()],
        ignore_index=True,
        sort=False,
    )
    combined["sample"] = "OOS_PLUS_EXTERNAL_963"

    if len(combined) != 963:
        raise RuntimeError("Combined != 963.")
    if combined[MATCH_ID].astype(str).duplicated().any():
        raise RuntimeError("Doublon match_id dans Combined963.")

    hr("COUVERTURE MÉTADONNÉES")
    print(f"OOS avec competition_id : {oos[COMP_ID].notna().sum()}/684")
    print(f"V3-I avec competition_id: {ext[COMP_ID].notna().sum()}/279")
    print(f"Compétitions distinctes OOS : {oos[COMP_ID].nunique()}")
    print(f"Compétitions distinctes V3-I: {ext[COMP_ID].nunique()}")
    print(f"Compétitions distinctes 963 : {combined[COMP_ID].nunique()}")

    # Global rates by each sample.
    oos_global = global_rule_rates(oos)
    ext_global = global_rule_rates(ext)
    combined_global = global_rule_rates(combined)

    oos_rules = rule_rows_for_sample(
        oos,
        "ROLLING_OOS_684_EXACT",
        oos_global,
    )
    ext_rules = rule_rows_for_sample(
        ext,
        "V3I_EXTERNAL_279",
        ext_global,
    )
    combined_rules = rule_rows_for_sample(
        combined,
        "OOS_PLUS_EXTERNAL_963",
        combined_global,
    )

    all_rules = pd.concat(
        [oos_rules, ext_rules, combined_rules],
        ignore_index=True,
    )

    classified = classify_combined(
        combined_rules,
        oos_rules,
        ext_rules,
    )

    # Focus V3H main.
    focus = classified[
        classified["rule"] == "V3H_MAIN_31_34"
    ].copy()

    focus = focus.sort_values(
        ["classification", "shrunk_uplift_vs_league_base", "selected_n"],
        ascending=[True, False, False],
    )

    # OOS-only discovery and external test.
    whitelist, blacklist = discover_oos_league_filter(
        oos_rules,
        "V3H_MAIN_31_34",
    )

    whitelist_ids = set(
        whitelist[COMP_ID].astype(str)
    )
    blacklist_ids = set(
        blacklist[COMP_ID].astype(str)
    )

    external_filter_test = evaluate_filter_on_external(
        ext,
        whitelist_ids,
        blacklist_ids,
    )

    # Save all.
    oos.to_csv(
        OUT_DIR / "rolling_oos_684_with_league.csv",
        index=False,
    )
    ext.to_csv(
        OUT_DIR / "v3i_external_279_with_league.csv",
        index=False,
    )
    all_rules.to_csv(
        OUT_DIR / "league_rule_performance_long.csv",
        index=False,
    )
    classified.to_csv(
        OUT_DIR / "league_rule_classification_combined963.csv",
        index=False,
    )
    focus.to_csv(
        OUT_DIR / "v3h31_34_league_ranking_combined963.csv",
        index=False,
    )
    whitelist.to_csv(
        OUT_DIR / "oos_discovered_v3h31_34_whitelist.csv",
        index=False,
    )
    blacklist.to_csv(
        OUT_DIR / "oos_discovered_v3h31_34_blacklist.csv",
        index=False,
    )
    external_filter_test.to_csv(
        OUT_DIR / "external_test_oos_league_filter.csv",
        index=False,
    )

    # Stability pivot for V3H main.
    stab = all_rules[
        all_rules["rule"] == "V3H_MAIN_31_34"
    ][
        [
            "sample",
            COMP_ID,
            "competition_name",
            "league_n",
            "league_base_rate",
            "selected_n",
            "selected_positives",
            "selected_rate",
            "uplift_pp",
            "shrunk_rate",
            "shrunk_uplift_vs_league_base",
        ]
    ].copy()

    stab.to_csv(
        OUT_DIR / "v3h31_34_league_stability_oos_external_combined.csv",
        index=False,
    )

    # Terminal.
    hr("V3-H 31–34 — LIGUES AVEC VOLUME SUFFISANT (COMBINÉ 963)")
    show = focus[
        (focus["league_n"] >= MIN_COMBINED_LEAGUE_N)
        & (focus["selected_n"] >= MIN_COMBINED_SELECTED_N)
    ].copy()

    if len(show):
        formatted = format_table(
            show,
            [
                "league_base_rate",
                "selected_rate",
                "uplift_pp",
                "ci95_low",
                "ci95_high",
                "shrunk_rate",
                "shrunk_uplift_vs_league_base",
                "roi_at_3.20",
            ],
        )

        cols = [
            COMP_ID,
            "competition_name",
            "classification",
            "league_n",
            "league_base_rate",
            "selected_n",
            "selected_positives",
            "selected_rate",
            "uplift_pp",
            "shrunk_rate",
            "ci95_low",
            "ci95_high",
        ]

        print(
            formatted[cols]
            .sort_values(
                ["classification", "selected_n"],
                ascending=[True, False],
            )
            .to_string(index=False)
        )
    else:
        print(
            "Aucune ligue n'atteint encore "
            f"N ligue>={MIN_COMBINED_LEAGUE_N} ET "
            f"N sélectionné>={MIN_COMBINED_SELECTED_N}."
        )

    hr("WHITELIST DÉCOUVERTE UNIQUEMENT SUR OOS684")
    if len(whitelist):
        wf = format_table(
            whitelist,
            [
                "league_base_rate",
                "selected_rate",
                "uplift_pp",
                "shrunk_rate",
                "shrunk_uplift_vs_league_base",
            ],
        )
        print(
            wf[
                [
                    COMP_ID,
                    "competition_name",
                    "league_n",
                    "league_base_rate",
                    "selected_n",
                    "selected_positives",
                    "selected_rate",
                    "uplift_pp",
                    "shrunk_rate",
                ]
            ].to_string(index=False)
        )
    else:
        print("Aucune whitelist OOS ne passe les critères pré-déclarés.")

    hr("BLACKLIST DÉCOUVERTE UNIQUEMENT SUR OOS684")
    if len(blacklist):
        bf = format_table(
            blacklist,
            [
                "league_base_rate",
                "selected_rate",
                "uplift_pp",
                "shrunk_rate",
                "shrunk_uplift_vs_league_base",
            ],
        )
        print(
            bf[
                [
                    COMP_ID,
                    "competition_name",
                    "league_n",
                    "league_base_rate",
                    "selected_n",
                    "selected_positives",
                    "selected_rate",
                    "uplift_pp",
                    "shrunk_rate",
                ]
            ].to_string(index=False)
        )
    else:
        print("Aucune blacklist OOS ne passe les critères pré-déclarés.")

    hr("TEST EXTERNE V3-I 279 DU FILTRE LIGUE DÉCOUVERT SUR OOS")
    ef = format_table(
        external_filter_test,
        [
            "external_base_rate",
            "selected_rate",
            "uplift_pp_vs_external_base",
            "ci95_low",
            "ci95_high",
            "roi_at_3.20",
        ],
    )
    print(ef.to_string(index=False))

    # Report.
    report_lines = [
        "# Goal After 87 — Audit discrimination par ligue",
        "",
        f"Généré: `{now_utc()}`",
        "",
        "## Discipline",
        "",
        "- Les probabilités V3-H/V3-C ne sont pas réentraînées.",
        "- Les bornes des règles ne sont pas modifiées.",
        "- Le classement combiné 963 est discovery-only.",
        "- Le test le plus propre d'un filtre ligue est: découverte OOS684 -> test V3-I279.",
        "",
        "## Critères pré-déclarés",
        "",
        f"- Combined league N >= {MIN_COMBINED_LEAGUE_N}",
        f"- Combined selected N >= {MIN_COMBINED_SELECTED_N}",
        f"- OOS discovery league N >= {DISCOVERY_MIN_LEAGUE_N}",
        f"- OOS discovery selected N >= {DISCOVERY_MIN_SELECTED_N}",
        f"- OOS discovery |uplift| >= {DISCOVERY_MIN_UPLIFT_PP:.0%}",
        f"- EB prior strength = {EB_PRIOR_STRENGTH:.0f}",
        "",
        "## Fichiers principaux",
        "",
        "- `v3h31_34_league_ranking_combined963.csv`",
        "- `v3h31_34_league_stability_oos_external_combined.csv`",
        "- `oos_discovered_v3h31_34_whitelist.csv`",
        "- `oos_discovered_v3h31_34_blacklist.csv`",
        "- `external_test_oos_league_filter.csv`",
        "",
    ]

    (OUT_DIR / "league_discrimination_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": now_utc(),
        "samples": {
            "rolling_oos": 684,
            "v3i_external": 279,
            "combined": 963,
        },
        "rules": [
            {
                "name": rule,
                "column": col,
                "lower_inclusive": lo,
                "upper_exclusive": hi,
            }
            for rule, col, lo, hi in RULES
        ],
        "thresholds": {
            "min_combined_league_n": MIN_COMBINED_LEAGUE_N,
            "min_combined_selected_n": MIN_COMBINED_SELECTED_N,
            "min_block_selected_n": MIN_BLOCK_SELECTED_N,
            "discovery_min_league_n": DISCOVERY_MIN_LEAGUE_N,
            "discovery_min_selected_n": DISCOVERY_MIN_SELECTED_N,
            "discovery_min_uplift_pp": DISCOVERY_MIN_UPLIFT_PP,
            "eb_prior_strength": EB_PRIOR_STRENGTH,
        },
        "methodology": (
            "League ranking on combined963 is exploratory. "
            "OOS684 whitelist/blacklist is frozen before external V3I279 evaluation."
        ),
    }

    (OUT_DIR / "league_discrimination_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    hr("SORTIES")
    print(OUT_DIR)
    print()
    print("Fichiers clés:")
    print("  v3h31_34_league_ranking_combined963.csv")
    print("  v3h31_34_league_stability_oos_external_combined.csv")
    print("  oos_discovered_v3h31_34_whitelist.csv")
    print("  oos_discovered_v3h31_34_blacklist.csv")
    print("  external_test_oos_league_filter.csv")
    print()
    print(
        "IMPORTANT: aucune whitelist/blacklist ne doit être mise en production "
        "avant lecture du test externe V3-I."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
