#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETAPE 229 — V4.1 HISTORICAL DISCOVERY PILOT

Python 3.9
- Jusqu'a 2 appels API
- Aucun telechargement de timeline
- Aucun label utilise
- Exclusion des IDs historiques connus
- Aucune modification des modeles ou de V3-C
"""

import csv
import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
OUT = Path("outputs/v4_research")

SPLIT = JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
TRAIN = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
DEST = OUT / "V41_JULY_PILOT_MANIFEST_V1.csv"

START = datetime(2026, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 7, 8, tzinfo=timezone.utc)

MAX_PAGES = 2
PER_PAGE = 100

FIELDS = [
    "match_id",
    "home_team",
    "away_team",
    "home_id",
    "away_id",
    "kickoff_utc",
    "research_status",
    "label_used_for_selection",
]

print("=" * 72)
print("ETAPE 229 — DECOUVERTE HISTORIQUE V4.1")
print("=" * 72)

for path in (SPLIT, TRAIN):
    if not path.is_file():
        raise SystemExit(f"Fichier obligatoire absent : {path}")

if DEST.exists():
    raise SystemExit(
        f"Manifeste existant : {DEST}\n"
        "Aucun ecrasement autorise."
    )


def csv_rows(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


# Charger les IDs reserves SANS lire leurs labels.
split = csv_rows(SPLIT)

if not all(
    "match_id" in row and "split" in row
    for row in split
):
    raise SystemExit("Structure SPLIT incorrecte")

split_ids = {row["match_id"] for row in split}
test_ids = {
    row["match_id"]
    for row in split
    if row["split"] == "TEST_FUTUR"
}

if len(split) != 320 or len(split_ids) != 320 or len(test_ids) != 183:
    raise SystemExit("Population SPLIT inattendue")

train_rows = csv_rows(TRAIN)
train_ids = {
    row["match_id"] for row in train_rows
}

if len(train_rows) != 278 or len(train_ids) != 278:
    raise SystemExit("TRAIN278 inattendu")

# Exclure egalement les IDs connus dans les CSV
# historiques et de recherche existants.
existing = set(split_ids) | set(train_ids)
scanned = 0

for folder in (JOIN, OUT):
    for path in sorted(folder.rglob("*.csv")):
        if path == DEST:
            continue

        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                continue

            if "match_id" not in reader.fieldnames:
                continue

            scanned += 1

            for row in reader:
                mid = (row.get("match_id") or "").strip()

                if mid:
                    existing.add(mid)

print("\n=== PROTECTION DES POPULATIONS ===")
print("IDs du plan historique :", len(split_ids))
print("IDs TEST_FUTUR reserves :", len(test_ids))
print("IDs TRAIN278 :", len(train_ids))
print("CSV controles :", scanned)
print("IDs historiques connus :", len(existing))


# Charger le client API deja utilise a l'etape 228.
module_path = Path("diagnostic_live_api.py")

if not module_path.is_file():
    raise SystemExit("Client API introuvable")

spec = importlib.util.spec_from_file_location(
    "v41_discovery_api", module_path
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

if not callable(getattr(api, "api_get", None)):
    raise SystemExit("api_get indisponible")


def extract_matches(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "matches", "results", "items", "data"
        ):
            value = payload.get(key)
            found = extract_matches(value)

            if found is not None:
                return found

    return None


def parse_utc(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (ValueError, TypeError):
        return None


def team_info(value):
    if isinstance(value, dict):
        name = (
            value.get("name")
            or value.get("short_name")
            or ""
        )
        tid = value.get("id") or ""
        return str(name), str(tid)

    if isinstance(value, str):
        return value, ""

    return "", ""


seen_pages = set()
candidates = []
skipped_known = 0
page_sizes = []

print("\n=== RECHERCHE ===")
print("Periode : 2026-07-01 au 2026-07-07 UTC")
print("Maximum :", MAX_PAGES, "pages")
print("Budget :", MAX_PAGES, "appels API")

for page in range(1, MAX_PAGES + 1):

    payload = api.api_get(
        "/football/matches",
        {
            "status": "finished",
            "date_from": "2026-07-01",
            "date_to": "2026-07-07",
            "page": page,
            "per_page": PER_PAGE,
        },
    )

    matches = extract_matches(payload)

    if matches is None:
        raise SystemExit(
            f"Page {page} : structure non reconnue"
        )

    page_sizes.append(len(matches))

    print(
        f"\nPage {page} : {len(matches)} matchs"
    )

    if not matches:
        break

    page_ids = set()

    # Refuser tout schema incomplet plutot que
    # conserver des candidats incertains.
    for match in matches:
        if not isinstance(match, dict):
            raise SystemExit(
                f"Page {page} : match non structure"
            )

        mid = str(
            match.get("id")
            or match.get("match_id")
            or ""
        ).strip()

        if not re.fullmatch(r"mt_[0-9]+", mid):
            raise SystemExit(
                f"Page {page} : ID invalide : {mid!r}"
            )

        if mid in page_ids:
            raise SystemExit(
                f"Doublon dans la page {page} : {mid}"
            )

        if mid in seen_pages:
            raise SystemExit(
                "Chevauchement entre les pages : "
                f"{mid}. Pagination a examiner."
            )

        page_ids.add(mid)

        raw_date = (
            match.get("utc_date")
            or match.get("kickoff_utc")
            or match.get("start_time")
        )

        dt = parse_utc(raw_date)

        if dt is None or not (START <= dt < END):
            raise SystemExit(
                f"Date absente ou hors periode : "
                f"{mid} / {raw_date}"
            )

        if match.get("status") != "finished":
            raise SystemExit(
                f"Statut inattendu : {mid}"
            )

        if mid in existing:
            skipped_known += 1
            continue

        home, home_id = team_info(
            match.get("home_team")
        )
        away, away_id = team_info(
            match.get("away_team")
        )

        # Les noms sont necessaires a la
        # reconstruction historique.
        if not home or not away:
            raise SystemExit(
                f"Equipes non identifiees : {mid}"
            )

        candidates.append({
            "match_id": mid,
            "home_team": home,
            "away_team": away,
            "home_id": home_id,
            "away_id": away_id,
            "kickoff_utc": dt.isoformat(),
            "research_status": "DISCOVERED_ONLY",
            "label_used_for_selection": 0,
        })

    seen_pages.update(page_ids)

    print("IDs distincts cumules :", len(seen_pages))
    print("Nouveaux candidats :", len(candidates))

    if len(matches) < PER_PAGE:
        print("Page partielle : arret du pilote.")
        break


if set(c["match_id"] for c in candidates) & existing:
    raise SystemExit("Chevauchement historique detecte")

if len(candidates) != len(
    set(c["match_id"] for c in candidates)
):
    raise SystemExit("Doublons candidats")


# L'ecriture n'a lieu qu'apres tous les controles.
OUT.mkdir(parents=True, exist_ok=True)

with DEST.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=FIELDS
    )
    writer.writeheader()
    writer.writerows(candidates)


print("\n=== BILAN ===")
print("Pages interrogees :", len(page_sizes))
print("Matchs examines :", sum(page_sizes))
print("Matchs deja connus :", skipped_known)
print("Nouveaux candidats :", len(candidates))
print("Manifeste :", DEST)

print("\n=== PREMIERS CANDIDATS ===")

for row in candidates[:10]:
    print(
        row["match_id"],
        "|", row["kickoff_utc"],
        "|", row["home_team"],
        "vs", row["away_team"],
    )

print("\n=== LIMITES ===")
print("Ce manifeste contient des candidats non valides.")
print("La couverture des timelines n'est pas verifiee.")
print("Deux pages ne constituent pas un inventaire exhaustif.")
print("Aucun label ou score final utilise pour selectionner.")

print("\n=== GARANTIES ===")
print("Aucun cache telecharge.")
print("Aucun modele entraine.")
print("VALIDATION et TEST_FUTUR exclus par ID.")
print("Scanner V3-C inchange.")
print("ETAPE 229 TERMINEE.")
