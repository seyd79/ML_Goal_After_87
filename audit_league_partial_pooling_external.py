#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOAL AFTER 87 — LEAGUE SIGNAL WITH PARTIAL POOLING
==================================================

Pourquoi
--------
L'audit par ligue "dur" n'a trouvé aucune ligue avec assez de volume pour une
whitelist/blacklist individuelle. Ce script ne baisse PAS les seuils a posteriori.

Il change de méthode pour traiter correctement la sparsité :
- apprend un effet ligue UNIQUEMENT sur ROLLING_OOS_684 ;
- applique un shrinkage Beta-Binomial vers les taux globaux OOS ;
- transforme les ligues en 3 groupes (LOW / MID / HIGH) selon le score OOS ;
- teste ces groupes tels quels sur V3-I_EXTERNAL_279 ;
- garde UNSEEN pour les ligues absentes de l'OOS ;
- n'utilise jamais les labels V3-I pour définir les groupes.

Règle étudiée
-------------
V3H_MAIN_31_34 : 0.31 <= p_v3h < 0.34

Le test principal est effectué uniquement sur les matchs V3-I qui passent déjà
la règle V3-H 31-34. La question devient :

    "Parmi les matchs que V3-H sélectionnerait déjà, le score ligue appris
     sur OOS améliore-t-il encore la discrimination sur V3-I ?"

Aucun modèle n'est réentraîné.
Aucune borne de probabilité n'est modifiée.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


ROOT = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")
IN_DIR = ROOT / "outputs/league_discrimination_audit"
OUT_DIR = ROOT / "outputs/league_partial_pooling_external"

OOS_PATH = IN_DIR / "rolling_oos_684_with_league.csv"
EXT_PATH = IN_DIR / "v3i_external_279_with_league.csv"

TARGET = "goal_after_87"
COMP = "competition_id"
NAME = "competition_name_resolved"

PRIOR_STRENGTH = 20.0
RANDOM_STATE = 42
BOOTSTRAPS = 5000

RULE_LO = 0.31
RULE_HI = 0.34


def hr(title: str) -> None:
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson(pos: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = pos / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-m, c+m


def qcut_tertiles(values: pd.Series) -> pd.Series:
    """
    Terciles déterminés uniquement depuis les scores OOS.
    Duplicates='drop' évite un crash si plusieurs scores sont identiques.
    """
    ranked = values.rank(method="first")
    q = pd.qcut(
        ranked,
        q=3,
        labels=["LOW", "MID", "HIGH"],
        duplicates="drop",
    )
    return q.astype(str)


def build_oos_league_scores(oos: pd.DataFrame) -> pd.DataFrame:
    base_global = float(oos[TARGET].mean())

    sel_mask = (
        (oos["p_v3h"] >= RULE_LO)
        & (oos["p_v3h"] < RULE_HI)
    )
    sel = oos[sel_mask].copy()
    selected_global = float(sel[TARGET].mean())

    rows: List[Dict[str, Any]] = []

    for cid, g in oos.groupby(COMP, dropna=False):
        cid = str(cid)
        name = (
            str(g[NAME].iloc[0])
            if NAME in g.columns and len(g)
            else cid
        )

        n_all = len(g)
        pos_all = int(g[TARGET].sum())
        base_rate = pos_all / n_all if n_all else float("nan")

        gs = g[
            (g["p_v3h"] >= RULE_LO)
            & (g["p_v3h"] < RULE_HI)
        ]
        n_sel = len(gs)
        pos_sel = int(gs[TARGET].sum()) if n_sel else 0
        sel_rate = pos_sel / n_sel if n_sel else float("nan")

        # Shrink both the league's base rate and its selected rate.
        base_shrunk = (
            pos_all + PRIOR_STRENGTH * base_global
        ) / (
            n_all + PRIOR_STRENGTH
        )

        sel_shrunk = (
            pos_sel + PRIOR_STRENGTH * selected_global
        ) / (
            n_sel + PRIOR_STRENGTH
        )

        league_effect = sel_shrunk - base_shrunk

        # Reliability of the selected-rule evidence in the league.
        reliability = n_sel / (n_sel + PRIOR_STRENGTH)

        # Conservative score: effect x reliability.
        conservative_score = league_effect * reliability

        rows.append({
            COMP: cid,
            "competition_name": name,
            "oos_league_n": n_all,
            "oos_league_positives": pos_all,
            "oos_league_base_rate": base_rate,
            "oos_selected_n": n_sel,
            "oos_selected_positives": pos_sel,
            "oos_selected_rate": sel_rate,
            "oos_base_rate_shrunk": base_shrunk,
            "oos_selected_rate_shrunk": sel_shrunk,
            "oos_league_effect_shrunk": league_effect,
            "oos_reliability": reliability,
            "oos_conservative_league_score": conservative_score,
            "oos_global_base_rate": base_global,
            "oos_global_selected_rate": selected_global,
        })

    scores = pd.DataFrame(rows)

    # Only leagues that had at least one OOS selected match define the score tertiles.
    trainable = scores["oos_selected_n"] >= 1

    scores["league_tier"] = "NO_OOS_SELECTED_HISTORY"

    if int(trainable.sum()) >= 3:
        scores.loc[trainable, "league_tier"] = qcut_tertiles(
            scores.loc[trainable, "oos_conservative_league_score"]
        ).values

    return scores.sort_values(
        [
            "league_tier",
            "oos_conservative_league_score",
            "oos_selected_n",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def external_group_table(ext_selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tier, g in ext_selected.groupby("league_tier_external", dropna=False):
        n = len(g)
        pos = int(g[TARGET].sum())
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        rows.append({
            "league_tier_external": tier,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "mean_oos_league_score": (
                float(g["oos_conservative_league_score"].mean())
                if "oos_conservative_league_score" in g.columns
                else float("nan")
            ),
            "fair_odds_empirical": (
                1/rate if n and rate > 0 else float("nan")
            ),
            "roi_at_3.20": (
                rate*3.20 - 1 if n else float("nan")
            ),
        })

    order = {
        "HIGH": 0,
        "MID": 1,
        "LOW": 2,
        "NO_OOS_SELECTED_HISTORY": 3,
        "UNSEEN_LEAGUE": 4,
    }

    out = pd.DataFrame(rows)
    if len(out):
        out["_order"] = out["league_tier_external"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order")

    return out.reset_index(drop=True)


def external_binary_tests(ext_selected: pd.DataFrame) -> pd.DataFrame:
    tests = []

    def one(name: str, mask: pd.Series):
        g = ext_selected[mask]
        n = len(g)
        pos = int(g[TARGET].sum()) if n else 0
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        tests.append({
            "scenario": name,
            "selected_n": n,
            "selected_positives": pos,
            "selected_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "fair_odds_empirical": (
                1/rate if n and rate > 0 else float("nan")
            ),
            "roi_at_3.20": (
                rate*3.20 - 1 if n else float("nan")
            ),
        })

    one(
        "ALL_V3H_31_34",
        pd.Series(True, index=ext_selected.index),
    )
    one(
        "HIGH_OOS_LEAGUE_SCORE_ONLY",
        ext_selected["league_tier_external"] == "HIGH",
    )
    one(
        "HIGH_OR_MID_OOS_LEAGUE_SCORE",
        ext_selected["league_tier_external"].isin(["HIGH", "MID"]),
    )
    one(
        "LOW_OOS_LEAGUE_SCORE_ONLY",
        ext_selected["league_tier_external"] == "LOW",
    )
    one(
        "KNOWN_OOS_LEAGUES_ONLY",
        ext_selected["league_tier_external"] != "UNSEEN_LEAGUE",
    )

    return pd.DataFrame(tests)


def bootstrap_auc(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, float, int]:
    mask = np.isfinite(score)
    y = np.asarray(y)[mask].astype(int)
    score = np.asarray(score)[mask].astype(float)

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


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not math.isfinite(v) else f"{v:.2%}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hr("LEAGUE SIGNAL — PARTIAL POOLING OOS -> TEST EXTERNE")

    oos = pd.read_csv(OOS_PATH, low_memory=False)
    ext = pd.read_csv(EXT_PATH, low_memory=False)

    if len(oos) != 684:
        raise RuntimeError(f"OOS={len(oos)}, attendu=684.")
    if len(ext) != 279:
        raise RuntimeError(f"V3-I={len(ext)}, attendu=279.")

    if COMP not in oos.columns or COMP not in ext.columns:
        raise RuntimeError("competition_id absent.")

    # Intersection mathematically expected: 65 + 21 - 69 = 17, verify directly.
    oos_ids = set(oos[COMP].astype(str))
    ext_ids = set(ext[COMP].astype(str))
    overlap_ids = oos_ids & ext_ids

    print(f"Compétitions OOS      : {len(oos_ids)}")
    print(f"Compétitions V3-I     : {len(ext_ids)}")
    print(f"Compétitions communes : {len(overlap_ids)}")
    print(f"Compétitions V3-I nouvelles : {len(ext_ids - oos_ids)}")

    scores = build_oos_league_scores(oos)

    # External selected matches only.
    ext_sel = ext[
        (ext["p_v3h"] >= RULE_LO)
        & (ext["p_v3h"] < RULE_HI)
    ].copy()

    if len(ext_sel) != 50:
        raise RuntimeError(
            f"V3-I V3H31-34={len(ext_sel)}, attendu=50."
        )

    score_cols = [
        COMP,
        "competition_name",
        "oos_league_n",
        "oos_selected_n",
        "oos_selected_positives",
        "oos_league_base_rate",
        "oos_selected_rate",
        "oos_base_rate_shrunk",
        "oos_selected_rate_shrunk",
        "oos_league_effect_shrunk",
        "oos_reliability",
        "oos_conservative_league_score",
        "league_tier",
    ]

    ext_sel[COMP] = ext_sel[COMP].astype(str)
    scores[COMP] = scores[COMP].astype(str)

    ext_sel = ext_sel.merge(
        scores[score_cols],
        on=COMP,
        how="left",
        validate="many_to_one",
        suffixes=("", "_oos"),
    )

    ext_sel["league_tier_external"] = ext_sel["league_tier"].fillna(
        "UNSEEN_LEAGUE"
    )

    groups = external_group_table(ext_sel)
    binary = external_binary_tests(ext_sel)

    known_score = pd.to_numeric(
        ext_sel["oos_conservative_league_score"],
        errors="coerce",
    ).to_numpy()

    auc, auc_lo, auc_hi, auc_boot_n = bootstrap_auc(
        ext_sel[TARGET].to_numpy(),
        known_score,
    )

    auc_table = pd.DataFrame([{
        "external_selected_n_total": len(ext_sel),
        "external_selected_n_with_oos_league_score": int(
            np.isfinite(known_score).sum()
        ),
        "auc_oos_league_score_on_external_selected_outcome": auc,
        "bootstrap_ci95_low": auc_lo,
        "bootstrap_ci95_high": auc_hi,
        "bootstrap_valid_replicates": auc_boot_n,
    }])

    # Per-league external detail, but no hard conclusions from small n.
    ext_league_rows = []
    for cid, g in ext_sel.groupby(COMP):
        n = len(g)
        pos = int(g[TARGET].sum())
        rate = pos/n if n else float("nan")
        lo, hi = wilson(pos, n)

        ext_league_rows.append({
            COMP: cid,
            "competition_name": (
                str(g["competition_name"].iloc[0])
                if "competition_name" in g.columns
                else cid
            ),
            "league_tier_external": g["league_tier_external"].iloc[0],
            "oos_conservative_league_score": (
                float(g["oos_conservative_league_score"].iloc[0])
                if pd.notna(g["oos_conservative_league_score"].iloc[0])
                else float("nan")
            ),
            "oos_league_n": (
                int(g["oos_league_n"].iloc[0])
                if pd.notna(g["oos_league_n"].iloc[0])
                else 0
            ),
            "oos_selected_n": (
                int(g["oos_selected_n"].iloc[0])
                if pd.notna(g["oos_selected_n"].iloc[0])
                else 0
            ),
            "external_selected_n": n,
            "external_selected_positives": pos,
            "external_selected_rate": rate,
            "external_ci95_low": lo,
            "external_ci95_high": hi,
        })

    ext_league = pd.DataFrame(ext_league_rows).sort_values(
        [
            "league_tier_external",
            "oos_conservative_league_score",
            "external_selected_n",
        ],
        ascending=[True, False, False],
    )

    # Save.
    scores.to_csv(
        OUT_DIR / "oos_league_partial_pooling_scores.csv",
        index=False,
    )
    ext_sel.to_csv(
        OUT_DIR / "external_v3h31_34_with_oos_league_score.csv",
        index=False,
    )
    groups.to_csv(
        OUT_DIR / "external_results_by_oos_league_tier.csv",
        index=False,
    )
    binary.to_csv(
        OUT_DIR / "external_oos_league_score_scenarios.csv",
        index=False,
    )
    auc_table.to_csv(
        OUT_DIR / "external_oos_league_score_auc.csv",
        index=False,
    )
    ext_league.to_csv(
        OUT_DIR / "external_per_league_detail.csv",
        index=False,
    )

    hr("RÉSULTATS EXTERNES PAR TIER APPRIS SUR OOS")
    show = groups.copy()
    for c in ["selected_rate", "ci95_low", "ci95_high", "roi_at_3.20"]:
        show[c] = show[c].map(pct)
    show["fair_odds_empirical"] = show["fair_odds_empirical"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    )
    show["mean_oos_league_score"] = show["mean_oos_league_score"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):+.4f}"
    )
    print(show.to_string(index=False))

    hr("SCÉNARIOS EXTERNES — V3-H 31–34")
    b = binary.copy()
    for c in ["selected_rate", "ci95_low", "ci95_high", "roi_at_3.20"]:
        b[c] = b[c].map(pct)
    b["fair_odds_empirical"] = b["fair_odds_empirical"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    )
    print(b.to_string(index=False))

    hr("DISCRIMINATION CONTINUE DU SCORE LIGUE SUR V3-I")
    print(
        f"N externe sélectionné              : {len(ext_sel)}\n"
        f"N avec score ligue appris sur OOS  : {int(np.isfinite(known_score).sum())}\n"
        f"AUC score ligue -> résultat externe: "
        f"{auc:.4f}" if math.isfinite(auc) else
        "AUC non calculable"
    )
    if math.isfinite(auc_lo):
        print(
            f"IC95 bootstrap AUC                 : "
            f"[{auc_lo:.4f}, {auc_hi:.4f}] "
            f"(B={auc_boot_n})"
        )

    hr("INTERPRÉTATION AUTOMATIQUE")
    high = binary[binary["scenario"] == "HIGH_OOS_LEAGUE_SCORE_ONLY"]
    allr = binary[binary["scenario"] == "ALL_V3H_31_34"]

    if len(high) and len(allr):
        hn = int(high.iloc[0]["selected_n"])
        hrte = float(high.iloc[0]["selected_rate"]) if hn else float("nan")
        ar = float(allr.iloc[0]["selected_rate"])

        if hn >= 10 and math.isfinite(hrte) and hrte > ar:
            print(
                "Signal encourageant : le tier HIGH appris uniquement sur OOS "
                "surperforme la règle globale sur V3-I."
            )
        elif hn > 0 and math.isfinite(hrte):
            print(
                "Le tier HIGH ne montre pas encore d'amélioration externe claire."
            )
        else:
            print(
                "Volume externe insuffisant dans le tier HIGH."
            )

    print(
        "\nNe pas mettre un filtre ligue en production sur cette seule analyse. "
        "Le Batch 2 externe reste la prochaine confirmation indépendante."
    )

    manifest = {
        "created_at_utc": now_utc(),
        "method": "OOS league partial pooling -> frozen tertiles -> external V3I test",
        "rule": "0.31 <= p_v3h < 0.34",
        "prior_strength": PRIOR_STRENGTH,
        "oos_n": 684,
        "external_n": 279,
        "external_rule_selected_n": len(ext_sel),
        "oos_competitions": len(oos_ids),
        "external_competitions": len(ext_ids),
        "overlap_competitions": len(overlap_ids),
        "external_unseen_competitions": len(ext_ids - oos_ids),
        "no_retuning": True,
    }

    (OUT_DIR / "league_partial_pooling_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    hr("SORTIES")
    print(OUT_DIR)
    print("  oos_league_partial_pooling_scores.csv")
    print("  external_results_by_oos_league_tier.csv")
    print("  external_oos_league_score_scenarios.csv")
    print("  external_oos_league_score_auc.csv")
    print("  external_per_league_detail.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
