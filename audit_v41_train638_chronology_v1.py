#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("outputs/v4_research")
TRAIN = ROOT / "V41_TRAIN638_B29_RESEARCH_V1.csv"
MASS = ROOT / "V41_MASS344_B29_RESEARCH_V1.csv"
OLD = ROOT / "V41_TRAIN294_B29_RESEARCH_V1.csv"

FEATURES = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]

COMPONENTS = [
    "score_87_home", "score_87_away",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def date_utc(value):
    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )
    if dt.tzinfo is None:
        raise ValueError("Date sans fuseau")
    return dt.astimezone(timezone.utc)


def number(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("Valeur non finie")
    return n


def expected_features(row):
    h = number(row["score_87_home"])
    a = number(row["score_87_away"])

    return {
        "score_total_87": h + a,
        "score_diff_abs_87": abs(h - a),
        "is_draw_87": int(h == a),
        "shots_total_last5": (
            number(row["shots_home_last5"])
            + number(row["shots_away_last5"])
        ),
        "sot_total_last5": (
            number(row["sot_home_last5"])
            + number(row["sot_away_last5"])
        ),
        "corners_total_last5": (
            number(row["corners_home_last5"])
            + number(row["corners_away_last5"])
        ),
    }


print("=" * 72)
print("ETAPE 252 — AUDIT CHRONOLOGIQUE TRAIN638")
print("=" * 72)

rows = read_rows(TRAIN)
mass = read_rows(MASS)
old = read_rows(OLD)

if (len(rows), len(mass), len(old)) != (638, 344, 294):
    raise SystemExit("Effectifs inattendus")

ids = [r["match_id"] for r in rows]

if len(set(ids)) != 638:
    raise SystemExit("IDs TRAIN638 dupliqués")

mass_ids = {r["match_id"] for r in mass}
old_ids = {r["match_id"] for r in old}

if mass_ids & old_ids:
    raise SystemExit("Doublons MASS / OLD")

if set(ids) != mass_ids | old_ids:
    raise SystemExit("Population TRAIN638 incorrecte")

months = Counter()
days = Counter()
origins = Counter()
positive_by_origin = Counter()
positive_by_month = Counter()
verified = Counter()
errors = []
dates = []

for row in rows:
    mid = row["match_id"]

    try:
        dt = date_utc(row["kickoff_utc"])
        date = dt.strftime("%Y-%m-%d")
        month = dt.strftime("%Y-%m")

        label = number(row["goal_after_87"])

        if label not in (0, 1):
            raise ValueError("Label non binaire")

        for col in COMPONENTS + FEATURES:
            value = number(row[col])
            if value < 0:
                raise ValueError(
                    "Valeur négative : " + col
                )

        for col, expected in expected_features(row).items():
            if not math.isclose(
                number(row[col]),
                expected,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "SIMPLE6 incohérent : " + col
                )

        origin = row.get("research_origin", "").strip()

        months[month] += 1
        days[date] += 1
        origins[origin] += 1

        positive_by_origin[origin] += int(label)
        positive_by_month[month] += int(label)

        verified[
            row.get("independent_verified", "").strip()
        ] += 1

        dates.append((dt, mid))

    except Exception as exc:
        errors.append((mid, str(exc)))

if errors:
    print("\nANOMALIES :")
    for mid, reason in errors[:30]:
        print(mid, reason)
    raise SystemExit(
        f"{len(errors)} lignes incorrectes"
    )

if sum(positive_by_month.values()) != 162:
    raise SystemExit("162 positifs attendus")

print("\n=== POPULATION ===")
print("Matchs :", len(rows))
print("Positifs :", sum(positive_by_month.values()))
print("Négatifs :", len(rows) - 162)
print("Journées :", len(days))
print("Première date :", min(dates)[0].isoformat())
print("Dernière date :", max(dates)[0].isoformat())
print("Formules SIMPLE6 : OK")

print("\n=== PAR MOIS ===")
for month in sorted(months):
    n = months[month]
    p = positive_by_month[month]
    print(
        f"{month} | n={n:3d} | positifs={p:3d} "
        f"| fréquence={p/n:.1%}"
    )

print("\n=== PAR ORIGINE ===")
for origin, n in origins.most_common():
    p = positive_by_origin[origin]
    print(
        f"{origin or '(vide)'} | n={n} "
        f"| positifs={p} | fréquence={p/n:.1%}"
    )

print("\n=== DATES LES PLUS REPRÉSENTÉES ===")
for day, n in days.most_common(10):
    print(day, "|", n, "matchs")

print(
    "Part des 2 plus grosses journées :",
    f"{sum(n for _, n in days.most_common(2))/638:.1%}"
)

print("\n=== FENÊTRES CHRONOLOGIQUES ===")
windows = [
    ("Avant 2026-08-01", None, "2026-08-01"),
    ("2026-08-01 à 2026-08-19",
     "2026-08-01", "2026-08-20"),
    ("2026-08-20 à 2026-08-24",
     "2026-08-20", "2026-08-25"),
]

for name, start, end in windows:
    selected = [
        r for r in rows
        if (start is None or r["kickoff_utc"][:10] >= start)
        and r["kickoff_utc"][:10] < end
    ]

    p = sum(int(float(r["goal_after_87"]))
            for r in selected)

    print(name, "| n=", len(selected), "| positifs=", p)

print("\n=== VERIFICATION INDEPENDANTE ===")
print("Valeurs independent_verified :", dict(verified))

print("\nAucun appel API.")
print("Aucun fichier modifié.")
print("Aucun modèle entraîné.")
print("TEST_FUTUR non consulté.")
print("Scanner V3-C inchangé.")
print("ETAPE 252 TERMINEE.")
