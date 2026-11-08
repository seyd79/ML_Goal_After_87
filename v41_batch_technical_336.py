import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch

from v41_prematch_safe_bridge_318 import ResearchPrematchProxy, ODDS_FIELDS
from v41_targeted_collector_328 import load_live87, make_stub

ROOT = Path("outputs")
BASE = ROOT / "v4_research"

PLAN = BASE / "v41_clearance_323/PLAN210_ID_CHECKED_NOT_AUTHORIZED.csv"
PILOT = BASE / "v41_clearance_323/PILOT10_ID_CHECKED_NOT_AUTHORIZED.csv"
DEST = BASE / "v41_technical_quarantine_336"

PROTECTED = [
    ROOT / "v3i_unseen_holdout_500/batch1_frozen_279/V3I_BATCH1_279_IDS.csv",
    ROOT / "v3i_unseen_holdout_500/batch2_frozen_221/V3I_BATCH2_221_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/reserve168_final_v3c_only/reserve168_predictions_v3c_final.csv",
    ROOT / "v3c_external_test/v3c_scored_snapshots_full.csv",
    ROOT / "v4_research/V4_CHRONO_VALIDATION_V1.csv",
    ROOT / "first_half_37/v3_draw_with_goals_historical_external_holdout/first_half_37_v3_historical_external_listing.csv",
]

def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def ids_only(path):
    if "test_futur" in str(path).lower():
        raise RuntimeError("TEST_FUTUR interdit")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        fields = [x.strip().lower() for x in next(reader)]
        choices = ("match_id", "fixture_id", "matchid", "id")
        positions = [fields.index(x) for x in choices if x in fields]

        if not positions:
            raise RuntimeError("Colonne ID absente : " + str(path))

        p = positions[0]
        return {
            row[p].strip()
            for row in reader
            if len(row) > p and row[p].strip()
        }

def available(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def save_csv(path, record):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

def select_batch(plan, pilot_ids):
    selected = []

    for comp in ("comp_8531", "comp_623699"):
        months = defaultdict(list)

        for row in plan:
            if (
                row["competition_id"] == comp
                and row["match_id"] not in pilot_ids
            ):
                month = row["kickoff_utc"][:7]
                months[month].append(row)

        for month in months:
            months[month].sort(
                key=lambda r: (r["kickoff_utc"], r["match_id"])
            )

        chosen = []
        while len(chosen) < 5:
            progress = False

            for month in sorted(months):
                if months[month] and len(chosen) < 5:
                    chosen.append(months[month].pop(0))
                    progress = True

            if not progress:
                raise RuntimeError(
                    "Pas assez de candidats pour " + comp
                )

        selected.extend(chosen)

    if len(selected) != 10:
        raise RuntimeError("Lot inattendu")

    if len({r["match_id"] for r in selected}) != 10:
        raise RuntimeError("Doublons dans le lot")

    return selected

def main():
    print("=" * 72)
    print("ETAPE 336 - COLLECTE TECHNIQUE V4.1 - LOT DE 10")
    print("=" * 72)

    if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
        raise RuntimeError("Autorisation ID-only manquante")

    if os.environ.get("V41_AUTORISE_QUARANTAINE_TECHNIQUE") != "OUI":
        raise RuntimeError("Autorisation technique manquante")

    if DEST.exists():
        raise RuntimeError(
            "STOP : dossier 336 deja existant, aucun ecrasement"
        )

    plan = read_manifest(PLAN)
    pilot = read_manifest(PILOT)

    assert len(plan) == 210
    assert len(pilot) == 10

    plan_ids = {r["match_id"] for r in plan}
    pilot_ids = {r["match_id"] for r in pilot}

    assert len(plan_ids) == 210
    assert len(pilot_ids) == 10
    assert pilot_ids <= plan_ids

    assert all(
        r.get("collection_authorized") == "NO"
        and r.get("protected_clearance")
            == "SEVEN_SOURCES_ID_ONLY_PASS"
        for r in plan
    )

    blocked = set()

    for path in PROTECTED:
        blocked.update(plan_ids & ids_only(path))

    if blocked:
        raise RuntimeError(
            "STOP : collision avec source protegee"
        )

    batch = select_batch(plan, pilot_ids)
    batch_ids = {r["match_id"] for r in batch}

    print("\n=== LOT SELECTIONNE ===")
    for i, row in enumerate(batch, 1):
        print(
            i,
            row["match_id"],
            row["competition_id"],
            row["kickoff_utc"][:10]
        )

    core = collector.load_core_features(None)
    features49 = [
        x for x in core
        if x not in {"odds_home", "odds_draw", "odds_away"}
    ]

    assert len(core) == 52
    assert len(features49) == 49

    # Initialiser seulement apres les controles.
    lab = prematch.load_lab()
    prematch.lab = lab

    client = prematch.build_rate_limited_client(lab)
    live87 = load_live87()
    proxy = ResearchPrematchProxy(prematch, lab)

    original_detail = prematch.get_match_detail

    summary = []
    missing_counter = Counter()

    for index, selected in enumerate(batch, 1):
        mid = selected["match_id"]

        print("\n" + "-" * 72)
        print(f"[{index}/10] {mid}")

        # Controle du match API avant reconstruction.
        def checked_detail(lab_obj, client_obj, requested_id):
            if requested_id != mid or requested_id not in batch_ids:
                raise RuntimeError("ID hors lot")

            detail = original_detail(
                lab_obj, client_obj, requested_id
            )

            if not detail:
                return detail

            actual_id = str(lab.extract_match_id(detail))
            actual_comp = str(lab.extract_competition_id(detail))
            actual_season = str(lab.extract_season_id(detail))

            dt = prematch.dt_utc_naive(
                lab.extract_match_date(detail)
            )

            if (
                actual_id != mid
                or actual_comp != selected["competition_id"]
                or actual_season != selected["season_id"]
                or dt is None
                or dt.date().isoformat()
                    != selected["kickoff_utc"][:10]
            ):
                raise RuntimeError(
                    "STOP : metadonnees API incoherentes"
                )

            return detail

        proxy.get_match_detail = checked_detail

        row, status = collector.collect_one_match(
            prematch=proxy,
            live87=live87,
            client=client,
            match_stub=make_stub(selected),
            core_features=core,
            xg_limit=0,
            with_league_second_half=False,
            preferred_bookmaker=None,
        )

        if status.get("match_id") != mid:
            raise RuntimeError("Statut avec mauvais ID")

        if row is None:
            raise RuntimeError(
                "Aucune ligne reconstruite pour " + mid
            )

        if row.get("match_id") != mid:
            raise RuntimeError("Ligne avec mauvais ID")

        if any(
            available(row.get(name))
            for name in ODDS_FIELDS
        ):
            raise RuntimeError("STOP : cote numerique detectee")

        missing = [
            name for name in features49
            if not available(row.get(name))
        ]

        missing_counter.update(missing)

        safe_row = {
            "match_id": mid,
            "competition_id": selected["competition_id"],
            "season_id": selected["season_id"],
            "kickoff_utc": selected["kickoff_utc"],
        }

        safe_row.update({
            name: row.get(name)
            for name in features49
        })

        diagnostic = {
            "match_id": mid,
            "technical_status": status.get("status", ""),
            "available49": 49 - len(missing),
            "training_authorized": "NO",
            "protected_certified": "NO",
        }

        # Une sauvegarde par match dans un dossier
        # de recherche isole.
        DEST.mkdir(parents=True, exist_ok=True)
        folder = DEST / f"match_{index:02d}"
        folder.mkdir(exist_ok=False)

        save_csv(
            folder / "TECHNICAL_FEATURES49_NO_LABEL.csv",
            safe_row
        )

        save_csv(
            folder / "TECHNICAL_STATUS_NO_LABEL.csv",
            diagnostic
        )

        summary.append(diagnostic)

        print("Statut :", diagnostic["technical_status"])
        print("CORE49 :", diagnostic["available49"], "/49")

    print("\n=== BILAN ===")
    print("Matchs collectes :", len(summary))
    print(
        "Matchs 49/49 :",
        sum(r["available49"] == 49 for r in summary)
    )
    print(
        "Disponibilite totale :",
        sum(r["available49"] for r in summary),
        "/ 490"
    )

    print("\nCaracteristiques manquantes :")
    for name, count in missing_counter.most_common():
        print(name, ":", count)

    print("\nSorties :", DEST)
    print("Quarantaine technique : OUI")
    print("Entrainement : INTERDIT")
    print("Partitions inconnues : NON CERTIFIEES")
    print("Aucun label ou score final enregistre.")
    print("TEST_FUTUR non ouvert par ce script.")
    print("V3-C non modifie.")
    print("ETAPE 336 TERMINEE.")

if __name__ == "__main__":
    main()
