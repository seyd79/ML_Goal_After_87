import csv
import time
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch

OUT = Path(
    "outputs/v4_research/"
    "v41_chronological_inventory_296"
)

COMPETITIONS = [
    "comp_8531",
    "comp_623699",
]

WINDOWS = [
    ("2025-04-01", "2025-06-30"),
    ("2025-10-01", "2025-12-31"),
    ("2026-04-01", "2026-06-30"),
    ("2026-07-01", "2026-09-30"),
]

MAX_PAGES = 2
PER_PAGE = 100

if OUT.exists():
    raise SystemExit(
        "STOP : dossier inventaire deja existant. "
        "Aucun fichier ne sera ecrase."
    )

load_dotenv(Path.cwd() / ".env")

prematch.lab = prematch.load_lab()
client = prematch.build_rate_limited_client(
    prematch.lab
)

def extract_competition(match):
    value = collector.extract_competition_id_generic(match)
    if value:
        return str(value)

    try:
        value = prematch.lab.extract_competition_id(match)
    except Exception:
        return ""

    return str(value) if value is not None else ""

def extract_season(match):
    value = collector.extract_season_id_generic(match)
    return str(value) if value else ""

def extract_datetime(match):
    try:
        dt = prematch.match_datetime(
            prematch.lab, match
        )
        return dt.isoformat() if dt else ""
    except Exception:
        return ""

def extract_team(match, side):
    try:
        team = (
            prematch.lab.extract_home_team(match)
            if side == "home"
            else prematch.lab.extract_away_team(match)
        )
        return str(
            prematch.lab.extract_team_name(team) or ""
        )
    except Exception:
        return ""

records = []
seen_global = set()
summaries = []

print("=" * 78)
print("ETAPE 296 - INVENTAIRE CHRONOLOGIQUE V4.1")
print("=" * 78)
print("Competitions :", COMPETITIONS)
print("Fenetres par competition :", len(WINDOWS))
print("Pages max par fenetre :", MAX_PAGES)
print("Aucun filtre de saison pour ce diagnostic.")

for comp in COMPETITIONS:
    print("\n" + "=" * 78)
    print("COMPETITION :", comp)

    stop_competition = False

    for start, end in WINDOWS:
        if stop_competition:
            break

        seen_window = set()
        accepted = 0
        outside = 0
        wrong_comp = 0
        unknown_date = 0
        unknown_comp = 0
        duplicates = 0
        pages_received = 0
        error = ""
        cap_reached = False

        for page in range(1, MAX_PAGES + 1):
            params = {
                "competition_id": comp,
                "status": "finished",
                "date_from": start,
                "date_to": end,
                "page": page,
                "per_page": PER_PAGE,
            }

            try:
                response = client.get(
                    "/football/matches",
                    params=params,
                )
                matches = (
                    collector.as_list_from_payload(response)
                )
            except Exception as exc:
                error = type(exc).__name__
                print(
                    " ERREUR API :",
                    error,
                    "| fenetre", start, end,
                    "| page", page,
                )
                break

            if not matches:
                break

            pages_received += 1
            new_page_ids = 0

            for match in matches:
                if not isinstance(match, dict):
                    continue

                mid = str(
                    collector.extract_match_id_generic(
                        match
                    ) or ""
                )

                if not mid or mid in seen_window:
                    duplicates += 1
                    continue

                seen_window.add(mid)
                new_page_ids += 1

                actual_comp = extract_competition(match)
                actual_season = extract_season(match)
                kickoff = extract_datetime(match)
                day = kickoff[:10]

                if not actual_comp:
                    unknown_comp += 1
                    continue

                if actual_comp != comp:
                    wrong_comp += 1
                    continue

                if not day:
                    unknown_date += 1
                    continue

                if not (start <= day <= end):
                    outside += 1
                    continue

                key = (comp, mid)
                if key in seen_global:
                    duplicates += 1
                    continue

                seen_global.add(key)
                accepted += 1

                records.append({
                    "competition_id": comp,
                    "season_id": actual_season,
                    "match_id": mid,
                    "kickoff_utc": kickoff,
                    "home_team": extract_team(match, "home"),
                    "away_team": extract_team(match, "away"),
                    "query_start": start,
                    "query_end": end,
                    "metadata_status": "SCOPE_DATE_CHECKED",
                })

            if new_page_ids == 0:
                print(
                    " Pagination repetitive detectee :",
                    "arret de cette fenetre."
                )
                break

            if page == MAX_PAGES:
                cap_reached = True

            time.sleep(0.2)

        print(
            f"{start} -> {end}"
            f" | valides={accepted}"
            f" | pages={pages_received}"
            f" | hors_dates={outside}"
            f" | autre_comp={wrong_comp}"
            f" | comp_inconnue={unknown_comp}"
            f" | date_inconnue={unknown_date}"
            f" | doublons={duplicates}"
            f" | plafond_pages={int(cap_reached)}"
            f" | erreur={error or '-'}"
        )

        summaries.append({
            "competition_id": comp,
            "date_from": start,
            "date_to": end,
            "valid_matches": accepted,
            "pages_received": pages_received,
            "outside_date": outside,
            "wrong_competition": wrong_comp,
            "unknown_competition": unknown_comp,
            "unknown_date": unknown_date,
            "duplicates": duplicates,
            "page_cap_reached": int(cap_reached),
            "api_error": error,
        })

        # Si les filtres ne semblent pas respectes,
        # ne pas multiplier inutilement les requetes.
        if (
            pages_received
            and accepted == 0
            and (outside > 0 or wrong_comp > 0)
        ):
            print(
                "ALERTE : scope API non fiable "
                "pour cette fenetre."
            )
            print(
                "Arret des autres fenetres "
                "de cette competition."
            )
            stop_competition = True

        if error:
            stop_competition = True

        time.sleep(0.3)

OUT.mkdir(parents=True, exist_ok=False)

metadata_path = OUT / "MATCHES_METADATA_ONLY.csv"
summary_path = OUT / "WINDOW_COVERAGE.csv"

metadata_columns = [
    "competition_id", "season_id", "match_id",
    "kickoff_utc", "home_team", "away_team",
    "query_start", "query_end", "metadata_status",
]

summary_columns = list(summaries[0].keys())

with metadata_path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=metadata_columns
    )
    writer.writeheader()
    writer.writerows(
        sorted(
            records,
            key=lambda r: (
                r["competition_id"],
                r["kickoff_utc"],
                r["match_id"],
            ),
        )
    )

with summary_path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=summary_columns
    )
    writer.writeheader()
    writer.writerows(summaries)

print("\n" + "=" * 78)
print("BILAN CHRONOLOGIQUE")
print("=" * 78)

for comp in COMPETITIONS:
    subset = [
        r for r in records
        if r["competition_id"] == comp
    ]

    months = Counter(
        r["kickoff_utc"][:7] for r in subset
    )
    seasons = Counter(
        r["season_id"] or "UNKNOWN"
        for r in subset
    )

    print("\n", comp)
    print(" Matchs metadata :", len(subset))
    print(" Mois distincts :", len(months))
    print(" Repartition :", dict(sorted(months.items())))
    print(" Saisons retournees :", dict(seasons))

print("\nSORTIES")
print(metadata_path)
print(summary_path)

print("\nLIMITES")
print("- Inventaire partiel si le plafond de pages est atteint.")
print("- Aucun score final ou label conserve.")
print("- Pas de certification de la completude API.")
print("- Aucun match admis automatiquement au pool.")
print("- Aucun holdout historique ouvert.")
print("- TEST_FUTUR non consulte.")
print("- V3-C inchange.")
print("ETAPE 296 TERMINEE.")
