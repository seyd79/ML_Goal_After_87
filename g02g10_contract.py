import csv, json, hashlib, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter

B = Path("outputs/v4_research")
G5 = B/"v42_g02g5_prematch_plan"
G6 = B/"v42_g02g6_prematch_pilot"
G7 = B/"v42_g02g7_blind12_prematch_coverage"
G8 = B/"v42_g02g8_blind12_prematch_coverage"

T = G5/"G02G5_TARGET_OBSERVATIONS_FROZEN.csv"
M = B/"v41_mass350/manifest.csv"
P6 = G6/"G02G6_PILOT6_TEAMS_FROZEN.csv"
P12 = G7/"G02G7_BLIND12_TEAMS_FROZEN.csv"
T12 = G7/"G02G7_BLIND12_TARGETS_FROZEN.csv"

EXPECTED = {
    T: "4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2",
    M: "61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e",
    P6: "a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0",
    P12: "d000987b8f342f5ce8f146bd79c42334848aff85b8350388958ce51a91c7cbbf",
    T12: "b0fbfc697c2f992f11914068dff0b6f85551a4e89224ecb975dc06f2ee3a0cf4",
}

def stop(message):
    raise SystemExit("STOP : " + message)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def utc(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        stop("Date sans fuseau : " + s)
    return d.astimezone(timezone.utc)

def numeric_goal(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
        if math.isfinite(x) and x >= 0 and x.is_integer():
            return int(x)
    except (TypeError, ValueError):
        pass
    return None

print("=" * 70)
print("G02-G10 — CONTRAT PREMATCH : DERNIERS 20 MATCHS")
print("=" * 70)

for p, expected in EXPECTED.items():
    if not p.exists() or sha(p) != expected:
        stop("Source absente ou modifiee : " + str(p))
    print("SHA OK :", p.name)

old = {r["team_id"] for r in load(P6)}
new = {r["team_id"] for r in load(P12)}

if len(old) != 6 or len(new) != 12 or old & new:
    stop("Equipes pilotes incorrectes")

master = load(T)
manifest_rows = load(M)
manifest = {r["match_id"]: r for r in manifest_rows}

if len(master) != 3000 or len(manifest) != 12415:
    stop("Effectifs incorrects")

obs = [r for r in master if r["team_id"] in old | new]

if len(obs) != 52:
    stop("52 observations attendues")

ref12 = {
    (r["match_id"], r["side"], r["team_id"])
    for r in load(T12)
}
actual12 = {
    (r["match_id"], r["side"], r["team_id"])
    for r in obs if r["team_id"] in new
}
if len(ref12) != 30 or ref12 != actual12:
    stop("Jointure G02-G7 incorrecte")

history = {}
sources = {}

for tid in sorted(old | new):
    folder = G6 if tid in old else G8
    path = folder/"raw_history"/(tid + ".json")

    if not path.is_file():
        stop("Cache absent : " + tid)

    wrapper = json.loads(path.read_text(encoding="utf-8"))
    response = wrapper.get("response", {})
    meta = response.get("meta", {})
    matches = response.get("data")

    if wrapper.get("team_id") != tid:
        stop("Identifiant incorrect : " + tid)

    if meta.get("page") != 1 or not isinstance(matches, list):
        stop("Pagination incorrecte : " + tid)

    dated = []

    for m in matches:
        if m.get("status") != "finished":
            stop("Match non termine : " + tid)

        home = m["home_team"]["id"] == tid
        away = m["away_team"]["id"] == tid

        if home == away:
            stop("Identite equipe incorrecte : " + tid)

        dated.append((utc(m["utc_date"]), m["id"], m))

    if len({x[1] for x in dated}) != len(dated):
        stop("Matchs dupliques : " + tid)

    if any(
        dated[i][0] < dated[i+1][0]
        for i in range(len(dated)-1)
    ):
        stop("Ordre chronologique incorrect : " + tid)

    history[tid] = dated
    sources[tid] = sha(path)

counts = Counter()
exceptions = []
ages = []

for r in sorted(
    obs,
    key=lambda x: (
        x["team_id"], x["safe_cutoff_utc"], x["match_id"]
    )
):
    tid = r["team_id"]
    mid = r["match_id"]
    side = r["side"]

    if mid not in manifest or manifest[mid][side+"_id"] != tid:
        stop("Jointure incorrecte : " + mid)

    cutoff = utc(r["safe_cutoff_utc"])
    kickoff = utc(manifest[mid]["kickoff_utc"])

    if kickoff - cutoff != timedelta(hours=4):
        stop("Cutoff incorrect : " + mid)

    prior = [
        (d, i, m)
        for d, i, m in history[tid]
        if d < cutoff and i != mid
    ]
    selected = prior[:20]

    values = []

    for d, historical_mid, m in selected:
        position = (
            "home" if m["home_team"]["id"] == tid
            else "away"
        )

        score = m.get("score") or {}
        reg = score.get("regulation")
        v = numeric_goal(
            reg.get(position) if isinstance(reg, dict)
            else None
        )

        if v is None:
            stop("Score invalide : " + historical_mid)

        values.append(v)

    n = len(values)

    if n >= 10:
        average = sum(values) / n
        status = "COMPLETE20" if n == 20 else "PARTIAL10_19"
    else:
        average = None
        status = "MISSING_LT10"

    counts[status] += 1
    counts["observations"] += 1
    counts["score_uses"] += n

    if selected:
        age = (cutoff - selected[0][0]).total_seconds()/3600
        ages.append(age)

    if n < 20:
        exceptions.append((tid, mid, n, status))

    print(
        tid, mid, side,
        "| n =", n,
        "| status =", status,
        "| moyenne =", round(average, 6)
        if average is not None else None
    )

print("\n=== BILAN G02-G10 ===")
print("Equipes :", len(history))
print("Observations :", counts["observations"])
print("Completes (20) :", counts["COMPLETE20"])
print("Partielles (10-19) :", counts["PARTIAL10_19"])
print("Insuffisantes (<10) :", counts["MISSING_LT10"])
print("Scores utilises :", counts["score_uses"])
print("Cas particuliers :", exceptions)
print("Age minimum, heures :", min(ages))
print("Age maximum, jours :", round(max(ages)/24, 2))

if (
    counts["observations"] != 52
    or counts["COMPLETE20"] != 51
    or counts["PARTIAL10_19"] != 1
    or counts["score_uses"] != 1039
):
    stop("Divergence avec G02-G9")

print("\nG02-G10 PLAN OK")
print("Methode candidate reproductible sur le pilote")
print("Sources non certifiees historiquement AS-OF")
print("AUCUNE API. AUCUNE ECRITURE.")
print("AUCUN LABEL. AUCUN ENTRAINEMENT.")
