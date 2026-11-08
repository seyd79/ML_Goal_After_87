import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import v41_batch_technical_336 as previous

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch

from v41_prematch_safe_bridge_318 import (
    ResearchPrematchProxy,
    ODDS_FIELDS,
)
from v41_targeted_collector_328 import (
    load_live87,
    make_stub,
)

BASE = Path("outputs/v4_research")
DEST = BASE / "v41_technical_quarantine_337"


def read_one(path):
    if not path.is_file():
        raise RuntimeError("Fichier absent : " + str(path))

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise RuntimeError(
            "Nombre de lignes inattendu : " + str(path)
        )

    return rows[0]


def already_collected(plan_ids):
    completed = set()

    paths = [
        BASE / "v41_technical_quarantine_331"
    ]

    paths += [
        BASE / "v41_technical_quarantine_332"
        / f"pilot_{i:02d}"
        for i in range(2, 11)
    ]

    paths += [
        BASE / "v41_technical_quarantine_336"
        / f"match_{i:02d}"
        for i in range(1, 11)
    ]

    for folder in paths:
        status = read_one(
            folder / "TECHNICAL_STATUS_NO_LABEL.csv"
        )
        features = read_one(
            folder / "TECHNICAL_FEATURES49_NO_LABEL.csv"
        )

        mid = status["match_id"]

        if (
            status["technical_status"] != "done"
            or features["match_id"] != mid
            or mid not in plan_ids
            or mid in completed
        ):
            raise RuntimeError(
                "Historique des collectes incoherent"
            )

        completed.add(mid)

    if len(completed) != 20:
        raise RuntimeError(
            "STOP : 20 collectes precedentes attendues"
        )

    return completed


def select_batch(plan, completed):
    selected = []

    for competition in ("comp_8531", "comp_623699"):
        months = defaultdict(list)

        for row in plan:
            if (
                row["competition_id"] == competition
                and row["match_id"] not in completed
            ):
                months[row["kickoff_utc"][:7]].append(row)

        for month in months:
            months[month].sort(
                key=lambda r: (
                    r["kickoff_utc"],
                    r["match_id"],
                )
            )

        chosen = []

        while len(chosen) < 10:
            progress = False

            for month in sorted(months):
                if len(chosen) >= 10:
                    break

                if months[month]:
                    chosen.append(months[month].pop(0))
                    progress = True

            if not progress:
                raise RuntimeError(
                    "Candidats insuffisants : " + competition
                )

        selected.extend(chosen)

    ids = [r["match_id"] for r in selected]

    if len(ids) != 20 or len(set(ids)) != 20:
        raise RuntimeError("Lot de 20 invalide")

    return selected


def check_detail(
    lab, detail, expected
):
    mid = expected["match_id"]

    dt = prematch.dt_utc_naive(
        lab.extract_match_date(detail)
    )

    if (
        str(lab.extract_match_id(detail)) != mid
        or str(lab.extract_competition_id(detail))
            != expected["competition_id"]
        or str(lab.extract_season_id(detail))
            != expected["season_id"]
        or dt is None
        or dt.date().isoformat()
            != expected["kickoff_utc"][:10]
    ):
        raise RuntimeError(
            "STOP : metadonnees API incompatibles : " + mid
        )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("ETAPE 337 - TROISIEME LOT TECHNIQUE V4.1")
    print("=" * 72)

    if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
        raise RuntimeError(
            "Autorisation de controle ID-only manquante"
        )

    plan = previous.read_manifest(previous.PLAN)
    pilot = previous.read_manifest(previous.PILOT)

    if len(plan) != 210 or len(pilot) != 10:
        raise RuntimeError("Manifestes inattendus")

    plan_ids = {r["match_id"] for r in plan}

    if len(plan_ids) != 210:
        raise RuntimeError("IDs du plan non uniques")

    if any(
        r.get("collection_authorized") != "NO"
        or r.get("protected_clearance")
            != "SEVEN_SOURCES_ID_ONLY_PASS"
        for r in plan
    ):
        raise RuntimeError(
            "Statut du plan inattendu"
        )

    blocked = set()

    for source in previous.PROTECTED:
        blocked.update(
            plan_ids & previous.ids_only(source)
        )

    if blocked:
        raise RuntimeError(
            "Collision avec les sources protegees connues"
        )

    completed = already_collected(plan_ids)
    batch = select_batch(plan, completed)

    print("\nCollectes deja terminees :", len(completed))
    print("Nouveau lot :", len(batch))

    for index, row in enumerate(batch, 1):
        print(
            f"{index:02d}",
            row["match_id"],
            row["competition_id"],
            row["kickoff_utc"][:10],
        )

    if args.dry_run:
        print("\nDRY RUN : OK")
        print("Aucun appel API.")
        print("Aucune nouvelle collecte.")
        print("TEST_FUTUR non consulte.")
        print("ETAPE 337 PREPARATION TERMINEE.")
        return

    if os.environ.get(
        "V41_AUTORISE_QUARANTAINE_TECHNIQUE"
    ) != "OUI":
        raise RuntimeError(
            "Autorisation technique manquante"
        )

    core = collector.load_core_features(None)

    features49 = [
        name for name in core
        if name not in {
            "odds_home", "odds_draw", "odds_away"
        }
    ]

    if len(core) != 52 or len(features49) != 49:
        raise RuntimeError("Schema incorrect")

    # Initialisation apres validation des identifiants.
    lab = prematch.load_lab()
    prematch.lab = lab

    client = prematch.build_rate_limited_client(lab)
    live87 = load_live87()
    proxy = ResearchPrematchProxy(prematch, lab)

    original_detail = prematch.get_match_detail

    results = []

    for index, selected in enumerate(batch, 1):
        mid = selected["match_id"]
        folder = DEST / f"match_{index:02d}"

        print("\n" + "-" * 72)
        print(f"[{index}/20] {mid}")

        if folder.exists():
            old_status = read_one(
                folder / "TECHNICAL_STATUS_NO_LABEL.csv"
            )
            old_features = read_one(
                folder / "TECHNICAL_FEATURES49_NO_LABEL.csv"
            )

            if (
                old_status.get("match_id") != mid
                or old_features.get("match_id") != mid
                or old_status.get("technical_status") != "done"
            ):
                raise RuntimeError(
                    "STOP : dossier existant incoherent"
                )

            n = sum(
                previous.available(old_features.get(name))
                for name in features49
            )

            results.append(n)
            print("Deja collecte :", n, "/49")
            continue

        def checked_detail(lab_obj, client_obj, requested_id):
            if requested_id != mid:
                raise RuntimeError("ID hors lot")

            detail = original_detail(
                lab_obj,
                client_obj,
                requested_id,
            )

            if detail:
                check_detail(lab, detail, selected)

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

        if (
            row is None
            or row.get("match_id") != mid
            or status.get("match_id") != mid
            or status.get("status") != "done"
        ):
            raise RuntimeError(
                "STOP : reconstruction non terminee : " + mid
            )

        if any(
            previous.available(row.get(name))
            for name in ODDS_FIELDS
        ):
            raise RuntimeError(
                "STOP : cote numerique detectee"
            )

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

        n = sum(
            previous.available(safe_row.get(name))
            for name in features49
        )

        diagnostic = {
            "match_id": mid,
            "technical_status": "done",
            "available49": n,
            "training_authorized": "NO",
            "protected_certified": "NO",
        }

        folder.mkdir(parents=True, exist_ok=False)

        previous.save_csv(
            folder / "TECHNICAL_FEATURES49_NO_LABEL.csv",
            safe_row,
        )

        previous.save_csv(
            folder / "TECHNICAL_STATUS_NO_LABEL.csv",
            diagnostic,
        )

        results.append(n)
        print("Collecte reussie :", n, "/49")

    print("\n=== BILAN LOT 337 ===")
    print("Matchs termines :", len(results))
    print("Matchs complets 49/49 :", results.count(49))
    print(
        "Completude :",
        f"{sum(results)}/{len(results) * 49}"
    )
    print("Sorties :", DEST)
    print("Entrainement : NON AUTORISE")
    print("Partitions inconnues : NON CERTIFIEES")
    print("Aucun label ou score final enregistre.")
    print("TEST_FUTUR non ouvert par ce script.")
    print("V3-C inchange.")
    print("ETAPE 337 TERMINEE.")


if __name__ == "__main__":
    main()
