import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path("outputs/v4_research/v42_g02g6_prematch_pilot")
RAW = ROOT / "raw_history"
PAGES = ROOT / "raw_history_pages"

TEAM = "tm_279303"
EXPECTED = {
    "tm_37445": ("c3900ea631f71c7f65405137c0b6f67d1f503f6774a6f6e29f5959671c8d7daf", 345, 4),
    "tm_087341": ("1693cd6bf92bf704c2317f90ac40079d217ed72cfc3bc95b0f50803dd9599c87", 238, 3),
    "tm_279303": ("40561207f351e74bd6e3dff22f77df04c86d5983c1ffc7bd7d68c3e5277e2c12", 101, 2),
    "tm_46516": ("29b25d63a5bc29cc510abb843aa97143d923855fc102ceaa909f18ee6d48d15d", 254, 3),
    "tm_19704": ("7f04081190aca47d1a817fa0ad11cd83602433b2cf6dee08d6b423b216289b73", 303, 4),
    "tm_6047": ("adef09c91a4f33bf8c846414dc7a5b6591610c2f39100e8f8d841efe6e595c67", 332, 4),
}

def stop(message):
    raise SystemExit("STOP : " + message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_page(response, expected_page, team):
    if not isinstance(response, dict):
        stop("reponse non JSON objet")

    meta = response.get("meta")
    data = response.get("data")

    if not isinstance(meta, dict) or not isinstance(data, list):
        stop("structure data/meta incorrecte")

    if int(meta.get("page", -1)) != expected_page:
        stop("numero de page incorrect")

    ids = []
    dates = []
    statuses = Counter()

    for match in data:
        if not isinstance(match, dict):
            stop("match non objet")

        mid = match.get("id")
        if not isinstance(mid, str) or not mid:
            stop("ID match manquant")

        home = match.get("home_team") or {}
        away = match.get("away_team") or {}

        if not isinstance(home, dict) or not isinstance(away, dict):
            stop("equipes mal formees")

        if team not in (home.get("id"), away.get("id")):
            stop("match non attribue a l'equipe : " + mid)

        ids.append(mid)
        dates.append(str(match.get("utc_date", "")))
        statuses[str(match.get("status", ""))] += 1

    if len(ids) != len(set(ids)):
        stop("doublons au sein de la page")

    print(
        "PAGE", expected_page,
        "| lignes =", len(ids),
        "| total =", meta.get("total"),
        "| total_pages =", meta.get("total_pages")
    )
    print("Statuts :", dict(statuses))
    print("Premieres dates dans l'ordre API :", dates[:3])
    print("Dernieres dates dans l'ordre API :", dates[-3:])

    if any(not d for d in dates):
        print("ALERTE : date de match manquante")

    if dates == sorted(dates):
        print("Ordre de la page : chronologique croissant")
    elif dates == sorted(dates, reverse=True):
        print("Ordre de la page : chronologique decroissant")
    else:
        print("Ordre de la page : non chronologique")

    return meta, ids

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, choices=[0, 1], default=0)
args = parser.parse_args()

print("=" * 76)
print("G02-G6C — VERIFICATION DE PAGINATION")
print("Budget API :", args.budget)
print("=" * 76)

cached = {}

for team, (expected_sha, total, n_pages) in EXPECTED.items():
    path = RAW / (team + ".json")

    if not path.is_file() or sha(path) != expected_sha:
        stop("cache page 1 modifie ou absent : " + team)

    obj = json.loads(path.read_text(encoding="utf-8"))

    if obj.get("team_id") != team:
        stop("team_id incoherent : " + team)

    params = obj.get("params", {})
    if (
        params.get("team_id") != team
        or params.get("status") != "finished"
        or params.get("per_page") != 100
    ):
        stop("parametres cache incorrects : " + team)

    meta, ids = get_page(obj["response"], 1, team)

    if (
        len(ids) != 100
        or int(meta.get("total", -1)) != total
        or int(meta.get("total_pages", -1)) != n_pages
    ):
        stop("metadonnees initiales divergentes : " + team)

    cached[team] = ids
    print("SHA OK :", team)
    print("-" * 50)

print("\nSIX PREMIERES PAGES VALIDEES")

destination = PAGES / (TEAM + "_page_2.json")

if destination.is_file():
    obj = json.loads(destination.read_text(encoding="utf-8"))
    if obj.get("team_id") != TEAM or obj.get("params", {}).get("page") != 2:
        stop("cache page 2 incorrect")
    response = obj["response"]
    print("\nPAGE 2 DEJA EN CACHE : aucune requete")
elif args.budget == 0:
    print("\nG02-G6C PLAN OK")
    print("1 page supplementaire proposee :", TEAM, "page 2")
    print("ZERO APPEL API")
    print("AUCUNE ECRITURE")
    raise SystemExit(0)
else:
    import rebuild_clean_prematch_v2 as pre
    from datetime import datetime, timezone

    lab = pre.load_lab()
    client = pre.build_rate_limited_client(
        lab, max_retries=1, min_interval=2.5
    )

    params = {
        "team_id": TEAM,
        "status": "finished",
        "per_page": 100,
        "page": 2,
    }

    print("\nAPPEL UNIQUE :", TEAM, "page 2")
    response = client.get("/football/matches", params=params)

    if not isinstance(response, dict) or response.get("error"):
        stop("reponse API absente ou en erreur")

    get_page(response, 2, TEAM)

    record = {
        "team_id": TEAM,
        "endpoint": "/football/matches",
        "params": params,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RAW_NOT_CERTIFIED",
        "response": response,
    }

    PAGES.mkdir(parents=True, exist_ok=True)
    temp = PAGES / (TEAM + "_page_2.json.tmp")

    if destination.exists() or temp.exists():
        stop("refus d'ecraser un fichier existant")

    with temp.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    temp.replace(destination)
    print("CACHE SAUVE :", destination)
    print("SHA256 :", sha(destination))

meta2, ids2 = get_page(response, 2, TEAM)
ids1 = cached[TEAM]

duplicates = set(ids1) & set(ids2)
union = set(ids1) | set(ids2)

print("\n=== CONTROLE INTER-PAGES ===")
print("Matchs page 1 :", len(ids1))
print("Matchs page 2 :", len(ids2))
print("Doublons entre pages :", len(duplicates))
print("Matchs distincts :", len(union))
print("Total API actuel :", meta2.get("total"))
print("Total pages API :", meta2.get("total_pages"))

if (
    not duplicates
    and len(union) == 101
    and int(meta2.get("total", -1)) == 101
    and int(meta2.get("total_pages", -1)) == 2
):
    print("\nPASS : pagination complete pour cette equipe")
else:
    print("\nREVIEW : pagination non certifiee")

print("AUCUNE FEATURE PREMATCH VALIDEE")
print("AUCUN ENTRAINEMENT")
print("G02-G6C TERMINE")
