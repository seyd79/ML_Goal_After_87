import csv
import math
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/v4_research")
RAW = Path(
    "outputs/historical_dataset_v2/"
    "Historical_Dataset_Goal_After_87_RAW_v2.csv"
)
TRAIN = ROOT / "V41_TRAIN638_B29_RESEARCH_V1.csv"
OUT = ROOT / "v41_context_pool_266"
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def integer(x):
    n = num(x)
    if n is not None and n >= 0 and n.is_integer():
        return int(n)
    return None

def binary(x):
    n = num(x)
    return int(n) if n in (0, 1) else None

def score(row):
    return (
        integer(row.get("score_87_home")),
        integer(row.get("score_87_away"))
    )

train = read(TRAIN)
raw = read(RAW)
t_by_id = {r["match_id"]: r for r in train}

if len(t_by_id) != len(train):
    raise SystemExit("Doublons internes TRAIN638")

bad = {}
new = {}
shared = 0

for r in raw:
    mid = r.get("match_id", "").strip()
    date = r.get("kickoff_utc", "")[:10]
    y = binary(r.get("goal_after_87"))
    h, a = score(r)

    if not mid or not date or date >= "2026-08-25":
        continue

    if (
        r.get("target_status") != "LABELED_V2"
        or r.get("target_confidence") != "HIGH"
        or y is None or h is None or a is None
    ):
        continue

    reasons = []
    fh = integer(r.get("final_score_home"))
    fa = integer(r.get("final_score_away"))

    if fh is None or fa is None:
        reasons.append("FINAL_INVALIDE")
    else:
        if fh < h or fa < a:
            reasons.append("SCORE_DIMINUE")

        delta = fh + fa - h - a

        if y == 1 and delta <= 0:
            reasons.append("POSITIF_SANS_BUT_NET")
        if y == 0 and delta > 0:
            reasons.append("NEGATIF_AVEC_BUT_NET")

    if mid in t_by_id:
        shared += 1
        t = t_by_id[mid]

        if binary(t.get("goal_after_87")) != y:
            reasons.append("LABEL_CONTRADICTOIRE")

        if score(t) != (h, a):
            reasons.append("SCORE_87_CONTRADICTOIRE")

    if reasons:
        bad[mid] = (
            date,
            "RAW_ET_TRAIN" if mid in t_by_id else "NOUVEAU_RAW",
            "|".join(reasons)
        )
    elif mid not in t_by_id:
        new[mid] = r

rows = []

for mid, t in t_by_id.items():
    if mid in bad:
        continue

    h, a = score(t)
    y = binary(t.get("goal_after_87"))

    if h is None or a is None or y is None:
        raise SystemExit("TRAIN638 invalide : " + mid)

    rows.append((
        mid, t.get("kickoff_utc", ""),
        y, h, a, "TRAIN638", "", "", "", ""
    ))

for mid, r in new.items():
    h, a = score(r)

    rows.append((
        mid, r.get("kickoff_utc", ""),
        binary(r.get("goal_after_87")),
        h, a, "RAW_V2",
        r.get("shots_80_87", ""),
        r.get("sot_80_87", ""),
        r.get("corners_80_87", ""),
        "RAW_ONLY_80_87"
    ))

rows.sort(key=lambda x: (x[1], x[0]))

if len({r[0] for r in rows}) != len(rows):
    raise SystemExit("Doublons apres fusion")

if (len(bad), len(new), shared, len(rows)) != (
    93, 534, 286, 1131
):
    raise SystemExit(
        "Effectifs inattendus : "
        f"quarantaine={len(bad)}, "
        f"nouveaux={len(new)}, "
        f"chevauchements={shared}, "
        f"pool={len(rows)}"
    )

pool_path = OUT / "CONTEXT_POOL1131_PROVISIONAL.csv"

columns = [
    "match_id", "kickoff_utc", "goal_after_87",
    "score_87_home", "score_87_away",
    "score_total_87", "score_diff_abs_87",
    "score_state_87", "trailing_side_87",
    "source_research",
    "raw_shots_80_87", "raw_sot_80_87",
    "raw_corners_80_87", "pressure_window",
    "admission_status"
]

with pool_path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for mid, dt, y, h, a, source, shots, sot, corners, window in rows:
        writer.writerow({
            "match_id": mid,
            "kickoff_utc": dt,
            "goal_after_87": y,
            "score_87_home": h,
            "score_87_away": a,
            "score_total_87": h + a,
            "score_diff_abs_87": abs(h - a),
            "score_state_87": (
                "DRAW" if h == a
                else "HOME_AHEAD" if h > a
                else "AWAY_AHEAD"
            ),
            "trailing_side_87": (
                "NONE" if h == a
                else "AWAY" if h > a
                else "HOME"
            ),
            "source_research": source,
            "raw_shots_80_87": shots,
            "raw_sot_80_87": sot,
            "raw_corners_80_87": corners,
            "pressure_window": window,
            "admission_status":
                "PROVISIONAL_NOT_PROTECTED_CLEARED"
        })

quarantine_path = OUT / "QUARANTINE93.csv"

with quarantine_path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "match_id", "kickoff_date", "origin", "reasons"
    ])
    for mid, values in sorted(bad.items()):
        writer.writerow([mid, *values])

print("=" * 72)
print("ETAPE 266 - BASE HISTORIQUE DES CONTEXTES")
print("=" * 72)
print("TRAIN638 conserves :", 638 - sum(
    mid in t_by_id for mid in bad
))
print("Nouveaux RAW conserves :", len(new))
print("POOL unique :", len(rows))
print("Positifs provisoires :", sum(r[2] for r in rows))
print("Matchs en quarantaine :", len(bad))
print("Origines :", dict(Counter(r[5] for r in rows)))

print("\nPOOL :", pool_path)
print("QUARANTAINE :", quarantine_path)

print("\nSTATUT : RESEARCH_ONLY / NON CERTIFIE")
print("Anciens holdouts et reserves non encore exclus.")
print("Ne pas entrainer ni evaluer sur ce pool en l'etat.")
print("Aucune source modifiee.")
print("Aucun appel API.")
print("TEST_FUTUR non consulte.")
print("Scanner V3-C inchange.")
print("ETAPE 266 TERMINEE.")
