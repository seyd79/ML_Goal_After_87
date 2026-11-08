import csv
from pathlib import Path
from dotenv import load_dotenv

import rebuild_clean_prematch_v2 as prematch

MANIFEST = Path(
    "outputs/v4_research/v41_pilot_plan_298/"
    "PILOT12_METADATA_ONLY.csv"
)

with MANIFEST.open(
    encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 12:
    raise SystemExit("STOP : 12 matchs attendus")

if len({r["match_id"] for r in rows}) != 12:
    raise SystemExit("STOP : IDs dupliques")

if any(
    r.get("collection_authorized") != "NO"
    for r in rows
):
    raise SystemExit("STOP : manifeste inattendu")

competitions = ["comp_8531", "comp_623699"]
sample = []

for comp in competitions:
    subset = sorted(
        [r for r in rows if r["competition_id"] == comp],
        key=lambda r: r["kickoff_utc"]
    )

    if len(subset) != 6:
        raise SystemExit(
            "STOP : 6 matchs attendus pour " + comp
        )

    sample.extend([subset[0], subset[-1]])

load_dotenv(Path.cwd() / ".env")

prematch.lab = prematch.load_lab()
client = prematch.build_rate_limited_client(
    prematch.lab
)

print("=" * 76)
print("ETAPE 300 - VERIFICATION DES MATCHS EXACTS")
print("=" * 76)
print("Nombre de matchs a verifier :", len(sample))

results = []

for expected in sample:
    mid = expected["match_id"]

    print("\nMATCH :", mid)

    try:
        detail = prematch.get_match_detail(
            prematch.lab,
            client,
            mid
        )

        if not detail:
            print("ERREUR : detail indisponible")
            results.append(False)
            continue

        actual_id = str(
            prematch.lab.extract_match_id(detail) or ""
        )
        actual_comp = str(
            prematch.lab.extract_competition_id(detail) or ""
        )
        actual_season = str(
            prematch.lab.extract_season_id(detail) or ""
        )

        kickoff = prematch.dt_utc_naive(
            prematch.lab.extract_match_date(detail)
        )

        actual_day = (
            kickoff.date().isoformat()
            if kickoff else ""
        )
        expected_day = expected["kickoff_utc"][:10]

        home = prematch.lab.extract_team_name(
            prematch.lab.extract_home_team(detail)
        )
        away = prematch.lab.extract_team_name(
            prematch.lab.extract_away_team(detail)
        )

        checks = {
            "match_id": actual_id == mid,
            "competition_id":
                actual_comp == expected["competition_id"],
            "season_id":
                actual_season == expected["season_id"],
            "kickoff_day":
                actual_day == expected_day,
        }

        print(
            "Competition :",
            actual_comp,
            "| attendue :",
            expected["competition_id"]
        )
        print(
            "Saison :",
            actual_season,
            "| attendue :",
            expected["season_id"]
        )
        print(
            "Date :",
            actual_day,
            "| attendue :",
            expected_day
        )
        print("Equipes :", home, "-", away)

        for name, valid in checks.items():
            print(
                name, ":",
                "OK" if valid else "ECHEC"
            )

        results.append(all(checks.values()))

    except Exception as exc:
        print(
            "ERREUR :",
            type(exc).__name__
        )
        results.append(False)

print("\n" + "=" * 76)
print("BILAN")
print("=" * 76)
print("Matchs verifies :", len(results))
print("Identites conformes :", sum(results))
print("Identites non conformes :", len(results)-sum(results))

if len(results) == 4 and all(results):
    print("PILOTE METADONNEES : CONFORME")
else:
    print("PILOTE METADONNEES : A INVESTIGUER")

print("\nAucun score ou label exploite.")
print("Aucune timeline telechargee par ce script.")
print("Aucun dataset ou manifeste modifie.")
print("Exclusions protegees toujours non certifiees.")
print("TEST_FUTUR non consulte.")
print("V3-C inchange.")
print("ETAPE 300 TERMINEE.")
