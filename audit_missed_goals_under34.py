#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "outputs" / "history" / "live_feature_snapshots.csv"
RESULTS_V2_PATH = BASE_DIR / "outputs" / "history" / "prediction_results_by_match_v2.csv"
OUT_DIR = BASE_DIR / "outputs" / "history"
MISSED_PATH = OUT_DIR / "audit_under34_missed_goals.csv"
BANDS_PATH = OUT_DIR / "audit_probability_bands.csv"
GROUPS_PATH = OUT_DIR / "audit_threshold_groups.csv"
TRAJ_PATH = OUT_DIR / "audit_probability_trajectories.csv"
REPORT_PATH = OUT_DIR / "audit_under34_report.md"
THRESHOLD = 0.34

BANDS = [
    ("<25%", None, 0.25),
    ("25-28%", 0.25, 0.28),
    ("28-30%", 0.28, 0.30),
    ("30-32%", 0.30, 0.32),
    ("32-34%", 0.32, 0.34),
    ("34-37%", 0.34, 0.37),
    ("37-40%", 0.37, 0.40),
    ("40-45%", 0.40, 0.45),
    ("45-50%", 0.45, 0.50),
    ("50%+", 0.50, None),
]


def clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def fnum(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def inum(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def band_of(p: Optional[float]) -> str:
    if p is None:
        return "NO_P87"
    for name, lo, hi in BANDS:
        if (lo is None or p >= lo) and (hi is None or p < hi):
            return name
    return "NO_P87"


def trajectory(p85: Optional[float], p86: Optional[float], p87: Optional[float]) -> str:
    vals = [p for p in (p85, p86, p87) if p is not None]
    if len(vals) < 2:
        return "INSUFFICIENT"
    first, last = vals[0], vals[-1]
    delta = last - first
    if abs(delta) < 0.005:
        return "STABLE"
    if delta > 0:
        return "RISING_TOWARD_34" if p87 is not None and 0.32 <= p87 < THRESHOLD else "RISING"
    if p85 is not None and p85 >= THRESHOLD and (p87 is None or p87 < THRESHOLD):
        return "FELL_BELOW_34"
    return "FALLING"


def is_draw(score: str) -> Optional[int]:
    if "-" not in clean(score):
        return None
    try:
        a, b = clean(score).split("-", 1)
        return int(a) == int(b)
    except Exception:
        return None


def main() -> int:
    if not HISTORY_PATH.exists():
        print(f"ERREUR: {HISTORY_PATH} introuvable")
        return 2
    if not RESULTS_V2_PATH.exists():
        print("ERREUR: prediction_results_by_match_v2.csv introuvable.")
        print("Exécute d'abord: python label_prediction_results_v2.py")
        return 2

    _, history = read_csv(HISTORY_PATH)
    _, results = read_csv(RESULTS_V2_PATH)

    by_match: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in history:
        mid = clean(r.get("match_id"))
        if mid:
            by_match[mid].append(r)

    def latest(mid: str, minute: int) -> Optional[Dict[str, str]]:
        rows = [r for r in by_match.get(mid, []) if inum(r.get("minute")) == minute]
        if not rows:
            return None
        return sorted(rows, key=lambda r: clean(r.get("snapshot_timestamp_utc")))[-1]

    enriched: List[Dict[str, Any]] = []

    for r in results:
        mid = clean(r.get("match_id"))
        p85, p86, p87 = fnum(r.get("P85")), fnum(r.get("P86")), fnum(r.get("P87"))
        actual = clean(r.get("Resultat_apres_87")).upper()
        y = 1 if actual == "OUI" else 0 if actual == "NON" else None
        pred = None if p87 is None else int(p87 >= THRESHOLD)

        if y == 1 and pred == 1:
            group = "A_CAPTURED_GOAL"
        elif y == 0 and pred == 1:
            group = "B_FALSE_SIGNAL"
        elif y == 1 and pred == 0:
            group = "C_MISSED_GOAL_UNDER34"
        elif y == 0 and pred == 0:
            group = "D_CORRECT_REJECTION"
        elif y == 1 and pred is None:
            group = "E_BLOCKED_OR_UNANALYZED_GOAL"
        elif y == 0 and pred is None:
            group = "F_BLOCKED_OR_UNANALYZED_NO_GOAL"
        else:
            group = "PENDING"

        r87 = latest(mid, 87) or {}
        enriched.append({
            "match_id": mid,
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "P85": "" if p85 is None else p85,
            "P86": "" if p86 is None else p86,
            "P87": "" if p87 is None else p87,
            "delta_P85_P87": "" if p85 is None or p87 is None else p87 - p85,
            "delta_P86_P87": "" if p86 is None or p87 is None else p87 - p86,
            "trajectory": trajectory(p85, p86, p87),
            "probability_band": band_of(p87),
            "Qualite85": r.get("Qualite85", ""),
            "Qualite86": r.get("Qualite86", ""),
            "Qualite87": r.get("Qualite87", ""),
            "coverage87": r87.get("coverage", ""),
            "missing_features87": r87.get("missing_features", ""),
            "Score87": r.get("Score87", ""),
            "draw87": "" if is_draw(r.get("Score87", "")) is None else int(bool(is_draw(r.get("Score87", "")))),
            "Resultat_apres_87": actual,
            "group": group,
            "Final_score": r.get("Final_score", ""),
            "Goal_minutes_after_87": r.get("Goal_minutes_after_87", ""),
            "home_shots_87": r87.get("home_shots_87", ""),
            "away_shots_87": r87.get("away_shots_87", ""),
            "home_sot_87": r87.get("home_shots_on_target_87", ""),
            "away_sot_87": r87.get("away_shots_on_target_87", ""),
            "shots_80_87": r87.get("shots_80_87", ""),
            "sot_80_87": r87.get("sot_80_87", ""),
            "corners_80_87": r87.get("corners_80_87", ""),
            "last_goal_minute": r87.get("last_goal_minute", ""),
            "last_shot_minute": r87.get("last_shot_minute", ""),
            "last_shot_on_target_minute": r87.get("last_shot_on_target_minute", ""),
            "minutes_since_last_goal": r87.get("minutes_since_last_goal", ""),
            "minutes_since_last_shot": r87.get("minutes_since_last_shot", ""),
            "minutes_since_last_shot_on_target": r87.get("minutes_since_last_shot_on_target", ""),
            "Target_source": r.get("Target_source", ""),
            "Coherence_warning": r.get("Coherence_warning", ""),
        })

    missed = [r for r in enriched if r["group"] == "C_MISSED_GOAL_UNDER34"]
    missed.sort(key=lambda r: -(fnum(r["P87"]) or -1))

    missed_fields = list(missed[0].keys()) if missed else [
        "match_id","home_team","away_team","P85","P86","P87","trajectory",
        "probability_band","Qualite87","Score87","Final_score","Goal_minutes_after_87"
    ]
    write_csv(MISSED_PATH, missed_fields, missed)

    # Group summary
    counts = Counter(r["group"] for r in enriched)
    resolved = [r for r in enriched if r["Resultat_apres_87"] in {"OUI", "NON"}]
    actual_goals = sum(r["Resultat_apres_87"] == "OUI" for r in resolved)

    group_rows = []
    for name in [
        "A_CAPTURED_GOAL","B_FALSE_SIGNAL","C_MISSED_GOAL_UNDER34",
        "D_CORRECT_REJECTION","E_BLOCKED_OR_UNANALYZED_GOAL",
        "F_BLOCKED_OR_UNANALYZED_NO_GOAL","PENDING"
    ]:
        count = counts.get(name, 0)
        goal_count = sum(1 for r in enriched if r["group"] == name and r["Resultat_apres_87"] == "OUI")
        group_rows.append({
            "group": name,
            "count": count,
            "share_of_resolved": "" if not resolved else count / len(resolved),
            "share_of_actual_goals": "" if not actual_goals else goal_count / actual_goals,
        })
    write_csv(GROUPS_PATH, ["group","count","share_of_resolved","share_of_actual_goals"], group_rows)

    # Probability bands
    band_rows = []
    for name, _, _ in BANDS + [("NO_P87", None, None)]:
        rows = [r for r in resolved if r["probability_band"] == name]
        goals = sum(r["Resultat_apres_87"] == "OUI" for r in rows)
        full = sum(clean(r["Qualite87"]).startswith("FULL") for r in rows)
        degraded = sum(clean(r["Qualite87"]).startswith("DEGRADED") for r in rows)
        draws = sum(r["draw87"] == 1 for r in rows)
        band_rows.append({
            "band": name,
            "matches": len(rows),
            "goals_after87": goals,
            "no_goal_after87": len(rows) - goals,
            "hit_rate": "" if not rows else goals / len(rows),
            "share_of_all_late_goals": "" if not actual_goals else goals / actual_goals,
            "FULL96": full,
            "DEGRADED": degraded,
            "draw87": draws,
            "draw87_rate": "" if not rows else draws / len(rows),
        })
    write_csv(BANDS_PATH, ["band","matches","goals_after87","no_goal_after87","hit_rate","share_of_all_late_goals","FULL96","DEGRADED","draw87","draw87_rate"], band_rows)

    # Trajectories
    traj_names = sorted(set(r["trajectory"] for r in resolved))
    traj_rows = []
    for name in traj_names:
        rows = [r for r in resolved if r["trajectory"] == name]
        goals = sum(r["Resultat_apres_87"] == "OUI" for r in rows)
        under34_goals = sum(r["Resultat_apres_87"] == "OUI" and fnum(r["P87"]) is not None and fnum(r["P87"]) < THRESHOLD for r in rows)
        traj_rows.append({
            "trajectory": name,
            "matches": len(rows),
            "goals_after87": goals,
            "hit_rate": "" if not rows else goals / len(rows),
            "under34_goals": under34_goals,
        })
    write_csv(TRAJ_PATH, ["trajectory","matches","goals_after87","hit_rate","under34_goals"], traj_rows)

    captured = counts.get("A_CAPTURED_GOAL", 0)
    false_signals = counts.get("B_FALSE_SIGNAL", 0)
    missed_count = counts.get("C_MISSED_GOAL_UNDER34", 0)
    blocked_goals = counts.get("E_BLOCKED_OR_UNANALYZED_GOAL", 0)
    signal_total = captured + false_signals
    hit_rate = captured / signal_total if signal_total else None
    recall = captured / actual_goals if actual_goals else None

    lines = [
        "# Audit des buts après 87' sous le seuil de 34 %",
        "",
        f"- Matchs résolus: **{len(resolved)}**",
        f"- Matchs avec but après 87': **{actual_goals}**",
        f"- Buts capturés par P87 >= 34 %: **{captured}**",
        f"- Buts ratés avec P87 < 34 %: **{missed_count}**",
        f"- Buts dans matchs BLOCKED/non analysés: **{blocked_goals}**",
        f"- Faux signaux >=34 %: **{false_signals}**",
    ]
    if hit_rate is not None:
        lines.append(f"- Taux de réussite du seuil 34 %: **{hit_rate:.2%}**")
    if recall is not None:
        lines.append(f"- Part des buts tardifs capturés par le seuil: **{recall:.2%}**")

    lines += [
        "",
        "## Tranches de P87",
        "",
        "| Tranche | Matchs | Buts >87 | Taux réel | Part de tous les buts tardifs |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in band_rows:
        hr = "" if r["hit_rate"] == "" else f"{float(r['hit_rate']):.2%}"
        sh = "" if r["share_of_all_late_goals"] == "" else f"{float(r['share_of_all_late_goals']):.2%}"
        lines.append(f"| {r['band']} | {r['matches']} | {r['goals_after87']} | {hr} | {sh} |")

    lines += [
        "",
        "## Lecture",
        "",
        "Les matchs P87 < 34 % avec but >87 sont les faux négatifs du seuil opérationnel, pas nécessairement des erreurs du modèle.",
        "L'analyse doit comparer le nombre de buts récupérés en abaissant le seuil au nombre de matchs perdants supplémentaires ajoutés.",
        "Les trajectoires P85→P86→P87 permettent de distinguer un match qui monte vers 34 % d'un ancien signal qui s'affaiblit.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("AUDIT UNDER 34 — PRÊT")
    print("=" * 80)
    print(f"Matchs résolus                  : {len(resolved)}")
    print(f"Buts après 87'                  : {actual_goals}")
    print(f"Capturés >=34                   : {captured}")
    print(f"Ratés <34                       : {missed_count}")
    print(f"Buts BLOCKED/non analysés       : {blocked_goals}")
    if hit_rate is not None:
        print(f"Taux réussite >=34              : {hit_rate:.2%}")
    if recall is not None:
        print(f"Part des buts capturés >=34     : {recall:.2%}")
    print("\nFichiers:")
    for p in [MISSED_PATH, BANDS_PATH, GROUPS_PATH, TRAJ_PATH, REPORT_PATH]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
