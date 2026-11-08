import csv
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/v4_research")
SOURCE = (
    ROOT / "v41_context_pool_266"
    / "CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv"
)
OUT = ROOT / "v41_score_contexts_268"
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def wilson(w, n, z=1.96):
    if not n:
        return 0.0, 1.0
    p = w / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt(
        (p*(1-p) + z*z/(4*n))/n
    ) / d
    return max(0, c-m), min(1, c+m)

def period(date):
    if date < "2026-08-01":
        return "DISCOVERY"
    if date < "2026-08-20":
        return "CHECK_AUG01_19"
    if date < "2026-08-25":
        return "CHECK_AUG20_24"
    return None

rows = read(SOURCE)

if len(rows) != 1131:
    raise SystemExit("Pool inattendu : 1131 attendus")

data = []
ids = set()

for r in rows:
    mid = r["match_id"]
    if mid in ids:
        raise SystemExit("Match duplique : " + mid)
    ids.add(mid)

    date = r["kickoff_utc"][:10]
    group = period(date)

    if group is None:
        raise SystemExit("Date hors population : " + mid)

    h = int(float(r["score_87_home"]))
    a = int(float(r["score_87_away"]))
    y = int(float(r["goal_after_87"]))

    if h < 0 or a < 0 or y not in (0, 1):
        raise SystemExit("Valeur invalide : " + mid)

    data.append({
        "match_id": mid,
        "date": date,
        "period": group,
        "home": h,
        "away": a,
        "label": y,
        "source": r["source_research"],
    })

if sum(x["label"] for x in data) != 315:
    raise SystemExit("315 labels positifs attendus")

# Contextes definis AVANT consultation des resultats.
RULES = {
    "MATCH_NUL": lambda h,a: h == a,
    "NUL_0_0": lambda h,a: h == 0 and a == 0,
    "NUL_1_1": lambda h,a: h == 1 and a == 1,
    "NUL_2_2_PLUS": lambda h,a: h == a and h >= 2,

    "DOMICILE_MENE": lambda h,a: h < a,
    "EXTERIEUR_MENE": lambda h,a: a < h,

    "DOMICILE_MENE_1": lambda h,a: a-h == 1,
    "EXTERIEUR_MENE_1": lambda h,a: h-a == 1,

    "DOMICILE_MENE_2_PLUS": lambda h,a: a-h >= 2,
    "EXTERIEUR_MENE_2_PLUS": lambda h,a: h-a >= 2,

    "DOMICILE_MENE_0_1": lambda h,a: h == 0 and a == 1,
    "EXTERIEUR_MENE_1_0": lambda h,a: h == 1 and a == 0,

    "DOMICILE_MENE_0_2_PLUS":
        lambda h,a: h == 0 and a >= 2,
    "EXTERIEUR_MENE_2_PLUS_0":
        lambda h,a: a == 0 and h >= 2,

    "UN_BUT_TOTAL": lambda h,a: h+a == 1,
    "DEUX_BUTS_TOTAL": lambda h,a: h+a == 2,
    "TROIS_BUTS_PLUS": lambda h,a: h+a >= 3,
    "QUATRE_BUTS_PLUS": lambda h,a: h+a >= 4,

    "ECART_1": lambda h,a: abs(h-a) == 1,
    "ECART_2_PLUS": lambda h,a: abs(h-a) >= 2,

    "DEUX_EQUIPES_ON_MARQUE":
        lambda h,a: h >= 1 and a >= 1,

    "NUL_AVEC_DEUX_EQUIPES_BUTEUSES":
        lambda h,a: h == a and h >= 1,

    "DOMICILE_MENE_1_ET_A_MARQUE":
        lambda h,a: a-h == 1 and h >= 1,

    "EXTERIEUR_MENE_1_ET_A_MARQUE":
        lambda h,a: h-a == 1 and a >= 1,
}

PERIODS = [
    "DISCOVERY",
    "CHECK_AUG01_19",
    "CHECK_AUG20_24",
]

def describe(subset):
    n = len(subset)
    w = sum(x["label"] for x in subset)
    lo, hi = wilson(w, n)

    return {
        "n": n,
        "wins": w,
        "rate": w/n if n else None,
        "ci95_low": lo if n else None,
        "ci95_high": hi if n else None,
        "days": len({x["date"] for x in subset}),
        "raw_n": sum(
            x["source"] == "RAW_V2" for x in subset
        ),
        "train_n": sum(
            x["source"] == "TRAIN638" for x in subset
        ),
    }

base = {
    p: describe([x for x in data if x["period"] == p])
    for p in PERIODS
}

results = []

for name, condition in RULES.items():
    for p in PERIODS:
        subset = [
            x for x in data
            if x["period"] == p
            and condition(x["home"], x["away"])
        ]
        result = describe(subset)
        result["context"] = name
        result["period"] = p

        b = base[p]["rate"]
        result["uplift_pp"] = (
            100 * (result["rate"] - b)
            if result["rate"] is not None else None
        )
        results.append(result)

discovery = [
    r for r in results
    if r["period"] == "DISCOVERY"
]

# Selection uniquement avec la periode DISCOVERY.
# Les deux autres periodes n'influencent pas cette selection.
selected = [
    r for r in discovery
    if r["n"] >= 25
    and r["days"] >= 10
    and r["uplift_pp"] is not None
    and r["uplift_pp"] >= 5
]

selected.sort(
    key=lambda r: (
        r["ci95_low"],
        r["n"]
    ),
    reverse=True
)

selected = selected[:5]
chosen = [r["context"] for r in selected]

OUT_FIELDS = [
    "context", "period", "n", "wins",
    "rate", "ci95_low", "ci95_high",
    "days", "uplift_pp", "raw_n", "train_n"
]

csv_path = OUT / "SCORE_CONTEXTS_ALL_PERIODS_V1.csv"

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    writer.writeheader()
    writer.writerows(results)

report = {
    "status": "RESEARCH_ONLY",
    "source": str(SOURCE),
    "match_count": len(data),
    "contexts_predefined": len(RULES),
    "baseline_by_period": base,
    "selected_from_discovery_only": chosen,
    "independent_validation": False,
    "test_futur_accessed": False,
}

json_path = OUT / "SCORE_CONTEXTS_SUMMARY_V1.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=" * 75)
print("ETAPE 268 - ETUDE DES CONTEXTES DISCRIMINANTS")
print("=" * 75)

print("\n=== POPULATIONS DE REFERENCE ===")
for p in PERIODS:
    b = base[p]
    print(
        f"{p:18s}: {b['wins']}/{b['n']} "
        f"= {b['rate']:.1%} | "
        f"jours={b['days']} | "
        f"RAW={b['raw_n']} TRAIN={b['train_n']}"
    )

print("\n=== TOP CONTEXTES EN DECOUVERTE ===")
ranked = sorted(
    discovery,
    key=lambda r: (
        r["ci95_low"],
        r["n"]
    ),
    reverse=True
)

for r in ranked[:15]:
    print(
        f"{r['context']:37s} "
        f"{r['wins']:3d}/{r['n']:3d} "
        f"= {r['rate']:.1%} "
        f"| IC95 bas={r['ci95_low']:.1%} "
        f"| jours={r['days']:3d} "
        f"| uplift={r['uplift_pp']:+.1f} pp"
    )

print("\n=== CONTEXTES PRESELECTIONNES ===")
print("Criteres : n>=25, jours>=10, uplift>=5 pp")
print("Classement : borne basse IC95")
print("Nombre selectionne :", len(chosen))

for name in chosen:
    print("\nCONTEXTE :", name)

    for p in PERIODS:
        r = next(
            x for x in results
            if x["context"] == name
            and x["period"] == p
        )

        if r["rate"] is None:
            print(p, ": aucun match")
        else:
            print(
                f"{p:18s}: "
                f"{r['wins']}/{r['n']} "
                f"= {r['rate']:.1%} "
                f"| jours={r['days']} "
                f"| uplift={r['uplift_pp']:+.1f} pp"
            )

print("\nCSV :", csv_path)
print("JSON :", json_path)

print("\nAVERTISSEMENT :")
print("- Recherche exploratoire, non validation independante")
print("- Risque de comparaisons multiples")
print("- Population et labels non certifies")
print("- Exclusions limitees aux sources deja auditees")
print("- Aucun entrainement, aucun appel API")
print("- TEST_FUTUR non consulte, V3-C inchange")
print("ETAPE 268 TERMINEE")
