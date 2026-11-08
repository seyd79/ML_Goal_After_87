#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_dataset_integrity.py

Audit d'intégrité de ML_Dataset_PreMatch_Live87.csv.
Ne modifie ni le dataset, ni le modèle, ni le scanner.

Sorties par défaut : outputs/dataset_integrity/
- dataset_integrity_report.md
- dataset_summary.csv
- feature_integrity_audit.csv
- invariant_violations.csv
- suspicious_mode_imputation.csv
- suspicious_duplicate_features.csv
- target_association_audit.csv
- duplicate_rows.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_CANDIDATES = [
    BASE_DIR / "ML_Dataset_PreMatch_Live87.csv",
    BASE_DIR / "data" / "ML_Dataset_PreMatch_Live87.csv",
]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "dataset_integrity"

ID_COLUMN = "match_id"
TARGET_COLUMN = "goal_after_87"

PRE_MATCH_FEATURES = [
    "home_team_rank", "away_team_rank",
    "home_team_points_standing", "away_team_points_standing",
    "home_team_matches_played", "away_team_matches_played",
    "home_team_wins", "away_team_wins",
    "home_team_draws", "away_team_draws",
    "home_team_losses", "away_team_losses",
    "home_team_goals_for", "away_team_goals_for",
    "home_team_goals_against", "away_team_goals_against",
    "home_team_goal_difference", "away_team_goal_difference",
    "home_position", "away_position",
    "home_points", "away_points",
    "home_form5", "away_form5",
    "home_form10", "away_form10",
    "home_scored_freq", "away_scored_freq",
    "home_wins_freq", "away_wins_freq",
    "home_avg_goals_scored", "away_avg_goals_scored",
    "home_avg_goals_conceded", "away_avg_goals_conceded",
    "home_avg_xg", "away_avg_xg",
    "home_btts_freq", "away_btts_freq",
    "home_over25_freq", "away_over25_freq",
    "h2h5_over25_freq", "h2h5_btts_freq",
    "league_home_over25_freq", "league_away_over25_freq",
    "league_home_over05_freq", "league_away_over05_freq",
    "league_home_btts_freq", "league_away_btts_freq",
    "league_2nd_half_over15_freq",
    "odds_home", "odds_draw", "odds_away",
    "odds_over05", "odds_over15", "odds_over25", "odds_over35",
    "odds_under25", "odds_btts_yes", "odds_btts_no",
]

LIVE_FEATURES = [
    "score_87_home", "score_87_away", "leading_team_87", "goal_diff_87", "is_draw_87",
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87", "away_shots_on_target_87",
    "home_shots_off_target_87", "away_shots_off_target_87",
    "home_shots_blocked_87", "away_shots_blocked_87",
    "home_corners_87", "away_corners_87",
    "home_fouls_87", "away_fouls_87",
    "home_yellow_cards_87", "away_yellow_cards_87",
    "home_red_cards_87", "away_red_cards_87",
    "home_offsides_87", "away_offsides_87",
    "shots_80_87", "sot_80_87", "corners_80_87", "fouls_80_87", "cards_80_87", "subs_80_87",
    "last_goal_minute", "last_shot_minute", "last_shot_on_target_minute", "last_corner_minute",
    "minutes_since_last_goal", "minutes_since_last_shot", "minutes_since_last_shot_on_target", "minutes_since_last_corner",
]

ALL_FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES

FREQUENCY_FEATURES = [
    "home_scored_freq", "away_scored_freq", "home_wins_freq", "away_wins_freq",
    "home_btts_freq", "away_btts_freq", "home_over25_freq", "away_over25_freq",
    "h2h5_over25_freq", "h2h5_btts_freq",
    "league_home_over25_freq", "league_away_over25_freq",
    "league_home_over05_freq", "league_away_over05_freq",
    "league_home_btts_freq", "league_away_btts_freq",
    "league_2nd_half_over15_freq",
]

ODDS_FEATURES = [
    "odds_home", "odds_draw", "odds_away", "odds_over05", "odds_over15",
    "odds_over25", "odds_over35", "odds_under25", "odds_btts_yes", "odds_btts_no",
]

NONNEGATIVE_FEATURES = [
    c for c in ALL_FEATURES
    if c not in {
        "home_team_goal_difference", "away_team_goal_difference"
    }
]


def resolve_dataset(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset introuvable : {p}")
        return p
    for p in DEFAULT_DATASET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("ML_Dataset_PreMatch_Live87.csv introuvable.")


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def series_equal_rate(a: pd.Series, b: pd.Series, atol: float = 1e-9) -> float:
    av = pd.to_numeric(a, errors="coerce")
    bv = pd.to_numeric(b, errors="coerce")
    both_na = av.isna() & bv.isna()
    both_num = av.notna() & bv.notna()
    eq = pd.Series(False, index=av.index)
    eq.loc[both_na] = True
    if both_num.any():
        eq.loc[both_num] = np.isclose(av.loc[both_num], bv.loc[both_num], atol=atol, rtol=0)
    return float(eq.mean())


def feature_audit(df: pd.DataFrame, mode_share_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in ALL_FEATURES:
        raw = df[feature]
        n = pd.to_numeric(raw, errors="coerce")
        inf_count = int(np.isinf(n.fillna(0).to_numpy(dtype=float)).sum())
        s = n.replace([np.inf, -np.inf], np.nan)
        clean = s.dropna()
        missing_count = int(s.isna().sum())
        if clean.empty:
            stats = dict(min=np.nan, q01=np.nan, q05=np.nan, mean=np.nan, median=np.nan,
                         q95=np.nan, q99=np.nan, max=np.nan, std=np.nan,
                         unique_count=0, mode_value=np.nan, mode_count=0, mode_share=0.0)
        else:
            vc = clean.value_counts()
            stats = dict(
                min=float(clean.min()), q01=float(clean.quantile(.01)), q05=float(clean.quantile(.05)),
                mean=float(clean.mean()), median=float(clean.median()), q95=float(clean.quantile(.95)),
                q99=float(clean.quantile(.99)), max=float(clean.max()), std=float(clean.std(ddof=0)),
                unique_count=int(clean.nunique()), mode_value=float(vc.index[0]), mode_count=int(vc.iloc[0]),
                mode_share=float(vc.iloc[0] / len(df)),
            )

        low = high = None
        if feature in FREQUENCY_FEATURES:
            low, high = 0.0, 1.0
        elif feature in ODDS_FEATURES:
            low = 1.0
        elif feature in ("home_form5", "away_form5"):
            low, high = 0.0, 15.0
        elif feature in ("home_form10", "away_form10"):
            low, high = 0.0, 30.0
        elif feature == "leading_team_87":
            low, high = 0.0, 2.0
        elif feature == "is_draw_87":
            low, high = 0.0, 1.0
        elif feature in NONNEGATIVE_FEATURES:
            low = 0.0

        out_of_range = 0
        if not clean.empty:
            mask = pd.Series(False, index=clean.index)
            if low is not None:
                mask |= clean < low
            if high is not None:
                mask |= clean > high
            out_of_range = int(mask.sum())

        missing_rate = missing_count / len(df)
        status = "OK"
        if inf_count > 0 or out_of_range > 0 or missing_rate > .20:
            status = "SEVERE"
        elif missing_rate > .05:
            status = "MODERATE"
        elif stats["unique_count"] <= 1:
            status = "CONSTANT"
        elif stats["mode_share"] >= .70:
            status = "SUSPECT_IMPUTATION"
        elif stats["mode_share"] >= mode_share_threshold:
            status = "MODERATE"

        rows.append({
            "feature": feature,
            "group": "PRE" if feature in PRE_MATCH_FEATURES else "LIVE",
            "dtype_raw": str(raw.dtype),
            "rows": len(df),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "inf_count": inf_count,
            **stats,
            "expected_low": low,
            "expected_high": high,
            "out_of_expected_range": out_of_range,
            "suspicious_mode": stats["mode_share"] >= mode_share_threshold,
            "status": status,
        })
    return pd.DataFrame(rows)


def invariant_audit(df: pd.DataFrame) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    ids = df[ID_COLUMN] if ID_COLUMN in df.columns else pd.Series(df.index, index=df.index)

    def add(rule: str, severity: str, idx: Any, details: str):
        out.append({"rule": rule, "severity": severity, "row_index": idx,
                    "match_id": ids.loc[idx], "details": details})

    def c(name: str) -> pd.Series:
        return numeric(df, name)

    for side in ("home", "away"):
        mp, w, d, l = [c(f"{side}_team_{x}") for x in ("matches_played", "wins", "draws", "losses")]
        valid = mp.notna() & w.notna() & d.notna() & l.notna()
        bad = valid & ~np.isclose(mp, w + d + l, atol=1e-9)
        for idx in df.index[bad]:
            add(f"{side}_matches=wins+draws+losses", "SEVERE", idx,
                f"matches={mp.loc[idx]}, W={w.loc[idx]}, D={d.loc[idx]}, L={l.loc[idx]}")

        pts = c(f"{side}_team_points_standing")
        exp_pts = 3*w + d
        bad = pts.notna() & w.notna() & d.notna() & ~np.isclose(pts, exp_pts, atol=1e-9)
        for idx in df.index[bad]:
            add(f"{side}_points=3*wins+draws", "SEVERE", idx,
                f"points={pts.loc[idx]}, expected={exp_pts.loc[idx]}")

        gf = c(f"{side}_team_goals_for")
        ga = c(f"{side}_team_goals_against")
        gd = c(f"{side}_team_goal_difference")
        exp_gd = gf - ga
        bad = gf.notna() & ga.notna() & gd.notna() & ~np.isclose(gd, exp_gd, atol=1e-9)
        for idx in df.index[bad]:
            add(f"{side}_goal_difference=GF-GA", "SEVERE", idx,
                f"gd={gd.loc[idx]}, expected={exp_gd.loc[idx]}")

    hs, as_ = c("score_87_home"), c("score_87_away")
    gd, draw, lead = c("goal_diff_87"), c("is_draw_87"), c("leading_team_87")
    valid = hs.notna() & as_.notna()
    exp_gd = (hs - as_).abs()
    bad = valid & gd.notna() & ~np.isclose(gd, exp_gd, atol=1e-9)
    for idx in df.index[bad]:
        add("goal_diff_87=abs(score_home-score_away)", "SEVERE", idx,
            f"goal_diff={gd.loc[idx]}, expected={exp_gd.loc[idx]}")

    exp_draw = (hs == as_).astype(float)
    bad = valid & draw.notna() & ~np.isclose(draw, exp_draw, atol=1e-9)
    for idx in df.index[bad]:
        add("is_draw_87 coherent avec score", "SEVERE", idx,
            f"is_draw={draw.loc[idx]}, expected={exp_draw.loc[idx]}")

    exp_lead = pd.Series(np.nan, index=df.index)
    exp_lead.loc[valid & (hs > as_)] = 0
    exp_lead.loc[valid & (as_ > hs)] = 1
    exp_lead.loc[valid & (hs == as_)] = 2
    bad = valid & lead.notna() & ~np.isclose(lead, exp_lead, atol=1e-9)
    for idx in df.index[bad]:
        add("leading_team_87 coherent avec score", "SEVERE", idx,
            f"leading={lead.loc[idx]}, expected={exp_lead.loc[idx]}")

    for side in ("home", "away"):
        shots = c(f"{side}_shots_87")
        sot = c(f"{side}_shots_on_target_87")
        off = c(f"{side}_shots_off_target_87")
        blk = c(f"{side}_shots_blocked_87")
        bad = shots.notna() & sot.notna() & (sot > shots)
        for idx in df.index[bad]:
            add(f"{side}_SOT<=shots", "SEVERE", idx, f"SOT={sot.loc[idx]}, shots={shots.loc[idx]}")
        bad = shots.notna() & sot.notna() & off.notna() & blk.notna() & ((sot+off+blk) > shots)
        for idx in df.index[bad]:
            add(f"{side}_shot_parts<=shots", "SEVERE", idx,
                f"parts={(sot+off+blk).loc[idx]}, shots={shots.loc[idx]}")

    total_shots = c("home_shots_87") + c("away_shots_87")
    recent_shots = c("shots_80_87")
    bad = total_shots.notna() & recent_shots.notna() & (recent_shots > total_shots)
    for idx in df.index[bad]:
        add("shots_80_87<=total_shots_87", "SEVERE", idx,
            f"recent={recent_shots.loc[idx]}, total={total_shots.loc[idx]}")

    total_sot = c("home_shots_on_target_87") + c("away_shots_on_target_87")
    recent_sot = c("sot_80_87")
    bad = total_sot.notna() & recent_sot.notna() & (recent_sot > total_sot)
    for idx in df.index[bad]:
        add("sot_80_87<=total_sot_87", "SEVERE", idx,
            f"recent={recent_sot.loc[idx]}, total={total_sot.loc[idx]}")

    for last_name, since_name in [
        ("last_goal_minute", "minutes_since_last_goal"),
        ("last_shot_minute", "minutes_since_last_shot"),
        ("last_shot_on_target_minute", "minutes_since_last_shot_on_target"),
        ("last_corner_minute", "minutes_since_last_corner"),
    ]:
        last = c(last_name)
        since = c(since_name)
        bad_future = last.notna() & (last > 87)
        for idx in df.index[bad_future]:
            add(f"{last_name}<=87", "SEVERE", idx, f"{last_name}={last.loc[idx]}")
        exp = 87 - last
        bad = last.notna() & since.notna() & (last <= 87) & ~np.isclose(since, exp, atol=1e-9)
        for idx in df.index[bad]:
            add(f"{since_name}=87-{last_name}", "MODERATE", idx,
                f"since={since.loc[idx]}, expected={exp.loc[idx]}, last={last.loc[idx]}")

    return pd.DataFrame(out)


def duplicate_feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(ALL_FEATURES):
        for b in ALL_FEATURES[i+1:]:
            rate = series_equal_rate(df[a], df[b])
            if rate >= .95:
                rows.append({"feature_a": a, "feature_b": b, "equal_rate": rate,
                             "status": "IDENTICAL" if rate >= .999999 else "NEAR_IDENTICAL"})
    return pd.DataFrame(rows)


def target_association_audit(df: pd.DataFrame) -> pd.DataFrame:
    y = numeric(df, TARGET_COLUMN)
    rows = []
    for feature in ALL_FEATURES:
        x = numeric(df, feature)
        valid = x.notna() & y.notna()
        xv, yv = x[valid], y[valid]
        if len(xv) < 20:
            continue
        corr = np.nan
        if xv.std(ddof=0) > 0 and yv.std(ddof=0) > 0:
            corr = float(np.corrcoef(xv, yv)[0, 1])
        auc = np.nan
        if roc_auc_score is not None and yv.nunique() == 2 and xv.nunique() > 1:
            try:
                raw_auc = float(roc_auc_score(yv, xv))
                auc = max(raw_auc, 1 - raw_auc)
            except Exception:
                pass
        status = "OK"
        if not pd.isna(auc) and auc >= .90:
            status = "CHECK_LEAKAGE"
        elif not pd.isna(auc) and auc >= .80:
            status = "STRONG_ASSOCIATION"
        rows.append({
            "feature": feature,
            "group": "PRE" if feature in PRE_MATCH_FEATURES else "LIVE",
            "valid_rows": len(xv),
            "corr_with_target": corr,
            "univariate_auc_abs": auc,
            "mean_target_0": float(xv[yv == 0].mean()) if (yv == 0).any() else np.nan,
            "mean_target_1": float(xv[yv == 1].mean()) if (yv == 1).any() else np.nan,
            "status": status,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("univariate_auc_abs", ascending=False, na_position="last")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit complet d'intégrité du dataset ML Goal After 87")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode-share-threshold", type=float, default=.50)
    args = parser.parse_args()

    dataset_path = resolve_dataset(args.dataset)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("AUDIT D'INTÉGRITÉ DATASET — GOAL AFTER 87")
    print("=" * 100)
    print(f"Dataset : {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Lignes : {len(df):,}")
    print(f"Colonnes : {len(df.columns)}")

    required = [ID_COLUMN] + ALL_FEATURES + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERREUR SCHÉMA — colonnes absentes :")
        for c in missing:
            print(f"  - {c}")
        return 2

    print(f"Schéma : {len(ALL_FEATURES)}/96 features présentes")

    feature_df = feature_audit(df, args.mode_share_threshold)
    violations_df = invariant_audit(df)
    duplicate_features_df = duplicate_feature_audit(df)
    target_assoc_df = target_association_audit(df)

    duplicate_rows_df = df[df.duplicated(keep=False)].copy()
    target = numeric(df, TARGET_COLUMN)
    feature_missing = int(df[ALL_FEATURES].isna().sum().sum())

    summary_df = pd.DataFrame([
        {"metric": "rows", "value": len(df)},
        {"metric": "columns", "value": len(df.columns)},
        {"metric": "feature_count", "value": len(ALL_FEATURES)},
        {"metric": "pre_feature_count", "value": len(PRE_MATCH_FEATURES)},
        {"metric": "live_feature_count", "value": len(LIVE_FEATURES)},
        {"metric": "feature_missing_values", "value": feature_missing},
        {"metric": "duplicate_exact_rows", "value": int(df.duplicated().sum())},
        {"metric": "duplicate_match_ids", "value": int(df[ID_COLUMN].duplicated().sum())},
        {"metric": "target_rate", "value": float(target.mean())},
    ])

    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_dir / "feature_integrity_audit.csv", index=False, encoding="utf-8-sig")
    violations_df.to_csv(output_dir / "invariant_violations.csv", index=False, encoding="utf-8-sig")
    feature_df[feature_df.mode_share >= args.mode_share_threshold].sort_values("mode_share", ascending=False).to_csv(
        output_dir / "suspicious_mode_imputation.csv", index=False, encoding="utf-8-sig")
    duplicate_features_df.to_csv(output_dir / "suspicious_duplicate_features.csv", index=False, encoding="utf-8-sig")
    target_assoc_df.to_csv(output_dir / "target_association_audit.csv", index=False, encoding="utf-8-sig")
    duplicate_rows_df.to_csv(output_dir / "duplicate_rows.csv", index=False, encoding="utf-8-sig")

    severe_inv = int((violations_df.severity == "SEVERE").sum()) if not violations_df.empty else 0
    severe_features = int((feature_df.status == "SEVERE").sum())
    suspicious_modes = int((feature_df.mode_share >= args.mode_share_threshold).sum())
    leakage = int((target_assoc_df.status == "CHECK_LEAKAGE").sum()) if not target_assoc_df.empty else 0

    report = []
    report += [
        "# Audit d'intégrité — ML_Dataset_PreMatch_Live87", "",
        "## Résumé", "",
        f"- Lignes : **{len(df):,}**",
        f"- Colonnes : **{len(df.columns)}**",
        f"- Features présentes : **{len(ALL_FEATURES)}/96**",
        f"- Valeurs manquantes sur les 96 features : **{feature_missing}**",
        f"- Duplicatas exacts : **{int(df.duplicated().sum())}**",
        f"- Match IDs dupliqués : **{int(df[ID_COLUMN].duplicated().sum())}**",
        f"- Violations d'invariants sévères : **{severe_inv}**",
        f"- Features avec valeur dominante >= {args.mode_share_threshold:.0%} : **{suspicious_modes}**",
        f"- Paires de features >=95% identiques : **{len(duplicate_features_df)}**",
        f"- Features univariées AUC >=0.90 : **{leakage}**", "",
        "## Cible", "",
        f"- Classe 1 : **{int((target == 1).sum()):,}** ({(target == 1).mean():.2%})",
        f"- Classe 0 : **{int((target == 0).sum()):,}** ({(target == 0).mean():.2%})", "",
        "## Features à forte valeur dominante", "",
    ]

    suspicious = feature_df[feature_df.mode_share >= args.mode_share_threshold].sort_values("mode_share", ascending=False)
    if suspicious.empty:
        report.append("Aucune.")
    else:
        report += ["| Feature | Groupe | Mode | Part | Unique | Statut |", "|---|---:|---:|---:|---:|---|"]
        for _, r in suspicious.head(30).iterrows():
            report.append(f"| {r.feature} | {r.group} | {r.mode_value} | {r.mode_share:.2%} | {int(r.unique_count)} | {r.status} |")

    report += ["", "## Invariants", ""]
    if violations_df.empty:
        report.append("Aucune violation détectée.")
    else:
        counts = violations_df.groupby(["severity", "rule"]).size().reset_index(name="count").sort_values("count", ascending=False)
        report += ["| Sévérité | Règle | Nombre |", "|---|---|---:|"]
        for _, r in counts.iterrows():
            report.append(f"| {r.severity} | {r.rule} | {int(r['count'])} |")

    report += ["", "## Redondances >=95%", ""]
    if duplicate_features_df.empty:
        report.append("Aucune.")
    else:
        report += ["| Feature A | Feature B | Taux égalité | Statut |", "|---|---|---:|---|"]
        for _, r in duplicate_features_df.head(40).iterrows():
            report.append(f"| {r.feature_a} | {r.feature_b} | {r.equal_rate:.2%} | {r.status} |")

    report += ["", "## Association univariée avec la cible", ""]
    if not target_assoc_df.empty:
        report += ["| Feature | Groupe | AUC abs | Corr | Mean y=0 | Mean y=1 | Statut |", "|---|---:|---:|---:|---:|---:|---|"]
        for _, r in target_assoc_df.head(30).iterrows():
            auc = "" if pd.isna(r.univariate_auc_abs) else f"{r.univariate_auc_abs:.4f}"
            corr = "" if pd.isna(r.corr_with_target) else f"{r.corr_with_target:.4f}"
            report.append(f"| {r.feature} | {r.group} | {auc} | {corr} | {r.mean_target_0:.4f} | {r.mean_target_1:.4f} | {r.status} |")

    report += [
        "", "## Lecture", "",
        "Priorité : violations mathématiques, valeurs hors plage, valeurs dominantes massives, colonnes quasi identiques, puis éventuelle fuite de cible.",
        "Aucun signal SUSPECT ne prouve seul qu'une feature est fausse : il faut remonter au générateur historique.",
    ]

    report_path = output_dir / "dataset_integrity_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print()
    print("=" * 100)
    print("RÉSULTAT AUDIT DATASET")
    print("=" * 100)
    print(f"Valeurs manquantes : {feature_missing}")
    print(f"Features SEVERE : {severe_features}/96")
    print(f"Violations invariants SEVERE : {severe_inv}")
    print(f"Features avec valeur dominante >= {args.mode_share_threshold:.0%} : {suspicious_modes}")
    print(f"Paires de features >=95% identiques : {len(duplicate_features_df)}")
    print(f"Features AUC >=0.90 : {leakage}")
    print(f"Rapport : {report_path}")
    print("=" * 100)
    print("Aucun fichier de production n'a été modifié.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
