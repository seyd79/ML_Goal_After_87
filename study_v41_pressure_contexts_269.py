import csv
import math
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/v4_research")
POOL = ROOT / "v41_context_pool_266/CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv"
RAW = Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv")
OUT = ROOT / "v41_pressure_contexts_269"

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def integer(v):
    try:
        n = float(v)
        if math.isfinite(n) and n >= 0 and n.is_integer():
            return int(n)
    except (ValueError, TypeError):
        pass
    return None

def period(date):
    if date < "2026-08-01":
        return "DISCOVERY"
    if date < "2026-08-20":
        return "CHECK_AUG01_19"
    if date < "2026-08-25":
        return "CHECK_AUG20_24"
    return None

def wilson_low(w, n):
    if not n:
        return None
    z = 1.96
    p = w / n
    d = 1 + z*z/n
    return (
        p + z*z/(2*n)
        - z*math.sqrt(p*(1-p)/n + z*z/(4*n))
    ) / d

pool = read(POOL)

if len(pool) != 1131:
    raise SystemExit("Pool strict inattendu")

selected = {
    r["match_id"]: r
    for r in pool if r["source_research"] == "RAW_V2"
}

if len(selected) != 534:
    raise SystemExit("534 matchs RAW attendus")

raw = {
    r["match_id"]: r
    for r in read(RAW)
    if r.get("match_id") in selected
}

data = []
quality = Counter()

for mid, p in selected.items():
    r = raw.get(mid)
    if r is None:
        raise SystemExit("RAW introuvable : " + mid)

    h = integer(p["score_87_home"])
    a = integer(p["score_87_away"])
    y = integer(p["goal_after_87"])

    rh = integer(r["score_87_home"])
    ra = integer(r["score_87_away"])
    ry = integer(r["goal_after_87"])

    if (h, a, y) != (rh, ra, ry):
        raise SystemExit("Incoherence RAW/POOL : " + mid)

    s = integer(r.get("shots_80_87"))
    t = integer(r.get("sot_80_87"))
    c = integer(r.get("corners_80_87"))

    if None in (s, t, c):
        quality["PRESSION_INCOMPLETE"] += 1
        continue

    if t > s:
        quality["SOT_SUPERIEUR_SHOTS"] += 1
        continue

    date = p["kickoff_utc"][:10]
    block = period(date)

    if block is None:
        raise SystemExit("Date hors periode : " + mid)

    data.append(dict(
        id=mid, date=date, period=block,
        h=h, a=a, y=y, shots=s, sot=t, corners=c
    ))

# Regles et seuils fixes avant calcul des resultats.
# "Pression" designe ici l'activite TOTALE des deux equipes.
RULES = {
    "SHOTS_GE2": lambda x: x["shots"] >= 2,
    "SHOTS_GE4": lambda x: x["shots"] >= 4,
    "SHOTS_GE6": lambda x: x["shots"] >= 6,
    "SOT_GE1": lambda x: x["sot"] >= 1,
    "SOT_GE2": lambda x: x["sot"] >= 2,
    "SOT_GE3": lambda x: x["sot"] >= 3,
    "CORNERS_GE1": lambda x: x["corners"] >= 1,
    "CORNERS_GE2": lambda x: x["corners"] >= 2,
    "SHOTS_GE4_SOT_GE1":
        lambda x: x["shots"] >= 4 and x["sot"] >= 1,
    "SHOTS_GE6_SOT_GE2":
        lambda x: x["shots"] >= 6 and x["sot"] >= 2,
    "DRAW_SOT_GE1":
        lambda x: x["h"] == x["a"] and x["sot"] >= 1,
    "DRAW_SOT_GE2":
        lambda x: x["h"] == x["a"] and x["sot"] >= 2,
    "DRAW_SHOTS_GE4":
        lambda x: x["h"] == x["a"] and x["shots"] >= 4,
    "DRAW_0_0_SOT_GE1":
        lambda x: x["h"] == x["a"] == 0 and x["sot"] >= 1,
    "DRAW_1_1_SOT_GE1":
        lambda x: x["h"] == x["a"] == 1 and x["sot"] >= 1,
    "HOME_TRAIL_1_SOT_GE1":
        lambda x: x["a"]-x["h"] == 1 and x["sot"] >= 1,
    "HOME_TRAIL_1_SOT_GE2":
        lambda x: x["a"]-x["h"] == 1 and x["sot"] >= 2,
    "HOME_TRAIL_1_SHOTS_GE4":
        lambda x: x["a"]-x["h"] == 1 and x["shots"] >= 4,
    "HOME_TRAIL_1_SCORED_SOT_GE1":
        lambda x: x["a"]-x["h"] == 1 and x["h"] >= 1
        and x["sot"] >= 1,
    "HOME_TRAIL_1_SCORED_SOT_GE2":
        lambda x: x["a"]-x["h"] == 1 and x["h"] >= 1
        and x["sot"] >= 2,
    "AWAY_TRAIL_1_SOT_GE1":
        lambda x: x["h"]-x["a"] == 1 and x["sot"] >= 1,
    "AWAY_TRAIL_1_SOT_GE2":
        lambda x: x["h"]-x["a"] == 1 and x["sot"] >= 2,
    "MARGIN_1_SOT_GE1":
        lambda x: abs(x["h"]-x["a"]) == 1 and x["sot"] >= 1,
    "MARGIN_1_SOT_GE2":
        lambda x: abs(x["h"]-x["a"]) == 1 and x["sot"] >= 2,
    "HOME_TRAIL_ANY_SOT_GE1":
        lambda x: x["h"] < x["a"] and x["sot"] >= 1,
    "HOME_TRAIL_ANY_SOT_GE2":
        lambda x: x["h"] < x["a"] and x["sot"] >= 2,
    "GOALS_3PLUS_SOT_GE1":
        lambda x: x["h"]+x["a"] >= 3 and x["sot"] >= 1,
    "SCORE_0_0_SHOTS_GE4":
        lambda x: x["h"] == x["a"] == 0 and x["shots"] >= 4,
}

PERIODS = [
    "DISCOVERY", "CHECK_AUG01_19", "CHECK_AUG20_24"
]

def stats(items):
    n = len(items)
    w = sum(x["y"] for x in items)
    return {
        "n": n,
        "wins": w,
        "rate": w/n if n else None,
        "days": len({x["date"] for x in items}),
        "low95": wilson_low(w, n),
    }

baseline = {
    p: stats([x for x in data if x["period"] == p])
    for p in PERIODS
}

results = []

for name, rule in RULES.items():
    for p in PERIODS:
        matches = [
            x for x in data
            if x["period"] == p and rule(x)
        ]
        res = stats(matches)
        res["context"] = name
        res["period"] = p
        res["uplift_pp"] = (
            100*(res["rate"]-baseline[p]["rate"])
            if res["rate"] is not None
            and baseline[p]["rate"] is not None else None
        )
        results.append(res)

discovery = [
    r for r in results if r["period"] == "DISCOVERY"
]

# Selection fixee uniquement sur DISCOVERY.
selected_rules = [
    r for r in discovery
    if r["n"] >= 20
    and r["days"] >= 10
    and r["uplift_pp"] is not None
    and r["uplift_pp"] >= 8
]

selected_rules.sort(
    key=lambda r: (r["low95"], r["n"]),
    reverse=True
)
selected_rules = selected_rules[:5]

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "PRESSURE_CONTEXTS_ALL_PERIODS_V1.csv"

fields = [
    "context", "period", "n", "wins",
    "rate", "days", "low95", "uplift_pp"
]

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("=" * 76)
print("ETAPE 269 - CONTEXTES SCORE + PRESSION 80-87")
print("=" * 76)
print("RAW eligibles dans pool :", len(selected))
print("RAW avec pression utilisable :", len(data))
print("Exclusions techniques :", dict(quality))
print("Contextes examines :", len(RULES))

print("\n=== BASELINES RAW UNIQUEMENT ===")
for p in PERIODS:
    r = baseline[p]
    print(
        p, "|", r["wins"], "/", r["n"],
        "| taux =", f'{r["rate"]:.1%}' if r["rate"] is not None else "NA",
        "| jours =", r["days"]
    )

print("\n=== TOP 15 DECOUVERTE ===")
ranked = sorted(
    discovery,
    key=lambda r: (r["low95"] or 0, r["n"]),
    reverse=True
)

for r in ranked[:15]:
    print(
        f'{r["context"]:36s} '
        f'{r["wins"]:3d}/{r["n"]:3d} '
        f'={r["rate"]:.1%} '
        f'| jours={r["days"]:2d} '
        f'| uplift={r["uplift_pp"]:+.1f} pp '
        f'| borne95={r["low95"]:.1%}'
        if r["rate"] is not None else r["context"] + " : aucun match"
    )

print("\n=== REGLES PRESELECTIONNEES ===")
print("n>=20, jours>=10, uplift>=8pp ; classement borne95")
print("Nombre :", len(selected_rules))

for chosen in selected_rules:
    name = chosen["context"]
    print("\nREGLE :", name)
    for p in PERIODS:
        r = next(
            a for a in results
            if a["context"] == name and a["period"] == p
        )
        if r["rate"] is None:
            print(" ", p, "aucun match")
        else:
            print(
                f' {p:18s}: {r["wins"]}/{r["n"]}'
                f'={r["rate"]:.1%} '
                f'| uplift={r["uplift_pp"]:+.1f} pp'
                f' | jours={r["days"]}'
            )

print("\nCSV :", path)
print("RESEARCH_ONLY : aucune validation independante.")
print("Pression totale des deux equipes, pas par equipe.")
print("Source historique, temporalite non certifiee.")
print("Pas de TEST_FUTUR, pas d'API, pas d'entrainement.")
print("V3-C inchange.")
print("ETAPE 269 TERMINEE.")
