import csv
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/v4_research")
POOL = ROOT / "v41_context_pool_266/CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv"
RAW = Path("outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv")
OUT = ROOT / "v41_team_dominance_277"

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def integer(value):
    try:
        n = float(value)
        if math.isfinite(n) and n >= 0 and n.is_integer():
            return int(n)
    except (ValueError, TypeError):
        pass
    return None

def period(date):
    if date < "2026-08-01":
        return "DISCOVERY"
    if date < "2026-08-20":
        return "AUG01_19"
    if date < "2026-08-25":
        return "AUG20_24"
    return None

def wilson(w, n):
    if not n:
        return None
    z = 1.96
    p = w / n
    d = 1 + z*z/n
    return (
        p + z*z/(2*n)
        - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    ) / d

pool_rows = read(POOL)
if len(pool_rows) != 1131:
    raise SystemExit("Pool inattendu")

selected = {
    r["match_id"]: r for r in pool_rows
    if r.get("source_research") == "RAW_V2"
}
if len(selected) != 534:
    raise SystemExit("534 RAW attendus")

raw = {
    r["match_id"]: r for r in read(RAW)
    if r.get("match_id") in selected
}
if len(raw) != 534:
    raise SystemExit("RAW incomplet")

data = []
excluded = Counter()

for mid, p in selected.items():
    r = raw[mid]

    h = integer(p.get("score_87_home"))
    a = integer(p.get("score_87_away"))
    y = integer(p.get("goal_after_87"))

    rh = integer(r.get("score_87_home"))
    ra = integer(r.get("score_87_away"))
    ry = integer(r.get("goal_after_87"))

    if (h, a, y) != (rh, ra, ry):
        raise SystemExit("Incoherence RAW/POOL : " + mid)

    hs = integer(r.get("home_shots_87"))
    ass = integer(r.get("away_shots_87"))
    ht = integer(r.get("home_shots_on_target_87"))
    at = integer(r.get("away_shots_on_target_87"))

    if None in (h, a, y, hs, ass, ht, at):
        excluded["MISSING"] += 1
        continue

    if y not in (0, 1) or ht > hs or at > ass:
        excluded["INVALID"] += 1
        continue

    try:
        provenance = json.loads(
            r.get("_live_provenance_json") or "{}"
        )
    except (ValueError, TypeError):
        provenance = {}

    source = (
        provenance.get("shots_80_87", "UNKNOWN")
        if isinstance(provenance, dict) else "UNKNOWN"
    )

    # Pour commencer : source shotmap homogene.
    if source != "shotmap_80_87":
        excluded["NON_SHOTMAP"] += 1
        continue

    date = p.get("kickoff_utc", "")[:10]
    block = period(date)

    if block is None:
        raise SystemExit("Date hors periode : " + mid)

    if h == a:
        trailing = "NONE"
        ts = os = tt = ot = None
    elif h < a:
        trailing = "HOME"
        ts, os = hs, ass
        tt, ot = ht, at
    else:
        trailing = "AWAY"
        ts, os = ass, hs
        tt, ot = at, ht

    data.append({
        "id": mid,
        "date": date,
        "period": block,
        "competition": r.get("competition_id", ""),
        "h": h, "a": a, "y": y,
        "hs": hs, "as": ass, "ht": ht, "at": at,
        "trailing": trailing,
        "ts": ts, "os": os, "tt": tt, "ot": ot
    })

if len(data) != 503:
    raise SystemExit(
        "Population shotmap inattendue : " + str(len(data))
    )

# Criteres fixes, sans ajustement apres consultation des labels.
RULES = {
    "TRAIL_ANY": lambda x:
        x["trailing"] != "NONE",

    "TRAIL_MORE_SHOTS": lambda x:
        x["ts"] is not None and x["ts"] > x["os"],

    "TRAIL_MORE_SOT": lambda x:
        x["tt"] is not None and x["tt"] > x["ot"],

    "TRAIL_SOT_SHARE60": lambda x:
        x["tt"] is not None
        and x["tt"] + x["ot"] >= 3
        and x["tt"] / (x["tt"] + x["ot"]) >= 0.60,

    "TRAIL_MORE_SHOTS_AND_SOT": lambda x:
        x["ts"] is not None
        and x["ts"] > x["os"]
        and x["tt"] > x["ot"],

    "TRAIL_SOT_GE3": lambda x:
        x["tt"] is not None and x["tt"] >= 3,

    "TRAIL_SOT_GE5": lambda x:
        x["tt"] is not None and x["tt"] >= 5,

    "TRAIL_MARGIN1_MORE_SHOTS": lambda x:
        abs(x["h"] - x["a"]) == 1
        and x["ts"] is not None and x["ts"] > x["os"],

    "TRAIL_MARGIN1_MORE_SOT": lambda x:
        abs(x["h"] - x["a"]) == 1
        and x["tt"] is not None and x["tt"] > x["ot"],

    "TRAIL_MARGIN2PLUS_MORE_SOT": lambda x:
        abs(x["h"] - x["a"]) >= 2
        and x["tt"] is not None and x["tt"] > x["ot"],

    "HOME_TRAIL_MORE_SOT": lambda x:
        x["trailing"] == "HOME" and x["tt"] > x["ot"],

    "AWAY_TRAIL_MORE_SOT": lambda x:
        x["trailing"] == "AWAY" and x["tt"] > x["ot"],

    "HOME_TRAIL_MARGIN1_MORE_SOT": lambda x:
        x["trailing"] == "HOME"
        and x["a"] - x["h"] == 1
        and x["tt"] > x["ot"],

    "AWAY_TRAIL_MARGIN1_MORE_SOT": lambda x:
        x["trailing"] == "AWAY"
        and x["h"] - x["a"] == 1
        and x["tt"] > x["ot"],

    "DRAW_HOME_MORE_SOT": lambda x:
        x["h"] == x["a"] and x["ht"] > x["at"],

    "DRAW_AWAY_MORE_SOT": lambda x:
        x["h"] == x["a"] and x["at"] > x["ht"],
}

PERIODS = ["DISCOVERY", "AUG01_19", "AUG20_24"]

def stats(items):
    n = len(items)
    w = sum(x["y"] for x in items)
    return {
        "n": n,
        "wins": w,
        "rate": w/n if n else None,
        "days": len({x["date"] for x in items}),
        "competitions": len({x["competition"] for x in items}),
        "low95": wilson(w, n),
    }

baseline = {}
results = []

for block in PERIODS:
    baseline[block] = stats(
        [x for x in data if x["period"] == block]
    )

for name, rule in RULES.items():
    for block in PERIODS:
        sub = [
            x for x in data
            if x["period"] == block and rule(x)
        ]
        result = stats(sub)
        result.update({"context": name, "period": block})

        base = baseline[block]["rate"]
        result["uplift_pp"] = (
            100*(result["rate"] - base)
            if result["rate"] is not None else None
        )
        results.append(result)

discovery = [
    r for r in results if r["period"] == "DISCOVERY"
]

# Candidats exploratoires uniquement ; aucun seuil optimise.
qualified = [
    r for r in discovery
    if r["n"] >= 20
    and r["days"] >= 10
    and r["uplift_pp"] is not None
    and r["uplift_pp"] >= 8
]

qualified.sort(
    key=lambda r: (r["low95"], r["n"]),
    reverse=True
)

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "TEAM_DOMINANCE_CONTEXTS_V1.csv"

fields = [
    "context", "period", "n", "wins", "rate",
    "days", "competitions", "low95", "uplift_pp"
]

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("=" * 76)
print("ETAPE 277 - DOMINATION OFFENSIVE PAR EQUIPE A 87'")
print("=" * 76)
print("Matchs :", len(data))
print("Exclusions :", dict(excluded))
print("Contextes :", len(RULES))

print("\n=== POPULATION DE REFERENCE ===")
for block in PERIODS:
    b = baseline[block]
    print(
        f"{block:12s}: {b['wins']}/{b['n']}"
        f" = {b['rate']:.1%}"
        f" | jours={b['days']}"
        f" | competitions={b['competitions']}"
    )

print("\n=== CLASSEMENT DECOUVERTE ===")
ranked = sorted(
    discovery,
    key=lambda r: (r["low95"] or 0, r["n"]),
    reverse=True
)

for r in ranked:
    if r["rate"] is not None:
        print(
            f"{r['context']:34s} "
            f"{r['wins']:3d}/{r['n']:3d}"
            f" = {r['rate']:.1%}"
            f" | borne95={r['low95']:.1%}"
            f" | uplift={r['uplift_pp']:+.1f}pp"
            f" | jours={r['days']}"
        )

print("\n=== CANDIDATS A EXAMINER ===")
print("Criteres : n>=20, jours>=10, uplift>=8pp")
print("Nombre :", len(qualified))

for item in qualified[:5]:
    name = item["context"]
    print("\n", name)

    for block in PERIODS:
        r = next(
            z for z in results
            if z["context"] == name
            and z["period"] == block
        )
        if r["rate"] is None:
            print(" ", block, ": aucun match")
        else:
            print(
                f" {block:12s}: {r['wins']}/{r['n']}"
                f" = {r['rate']:.1%}"
                f" | competitions={r['competitions']}"
                f" | jours={r['days']}"
            )

print("\nCSV :", csv_path)
print("RESEARCH_ONLY : pas de validation independante.")
print("Tirs cumules par equipe, pas pression recente.")
print("Plusieurs contextes se chevauchent.")
print("Labels et temporalite non certifies.")
print("Aucun holdout ouvert, TEST_FUTUR non consulte.")
print("Aucun appel API, aucun entrainement.")
print("Scanner V3-C inchange.")
print("ETAPE 277 TERMINEE.")
