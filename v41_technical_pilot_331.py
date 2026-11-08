import csv
import math
import os
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch

from v41_prematch_safe_bridge_318 import ResearchPrematchProxy
from v41_targeted_collector_328 import (
    load_manifest,
    preflight,
    load_live87,
    make_stub,
)

ROOT = Path("outputs")
DEST = ROOT / "v4_research/v41_technical_quarantine_331"

PROTECTED = [
    ROOT / "v3i_unseen_holdout_500/batch1_frozen_279/V3I_BATCH1_279_IDS.csv",
    ROOT / "v3i_unseen_holdout_500/batch2_frozen_221/V3I_BATCH2_221_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/reserve168_final_v3c_only/reserve168_predictions_v3c_final.csv",
    ROOT / "v3c_external_test/v3c_scored_snapshots_full.csv",
    ROOT / "v4_research/V4_CHRONO_VALIDATION_V1.csv",
    ROOT / "first_half_37/v3_draw_with_goals_historical_external_holdout/first_half_37_v3_historical_external_listing.csv",
]


def ids_only(path):
    if "test_futur" in str(path).lower():
        raise RuntimeError("TEST_FUTUR interdit")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [x.strip().lower() for x in next(reader)]
        keys = ("match_id", "fixture_id", "matchid", "id")
        positions = [
            header.index(k) for k in keys if k in header
        ]

        if not positions:
            raise RuntimeError("Colonne ID absente : " + str(path))

        p = positions[0]
        return {
            row[p].strip()
            for row in reader
            if len(row) > p and row[p].strip()
        }


def write_one(path, record):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)


def main():
    print("=" * 72)
    print("ETAPE 331 - PREMIER MATCH TECHNIQUE V4.1")
    print("=" * 72)

    if os.environ.get("V41_AUTORISE_QUARANTAINE_TECHNIQUE") != "OUI":
        raise RuntimeError(
            "STOP : autorisation explicite de l'essai technique manquante"
        )

    if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
        raise RuntimeError(
            "STOP : autorisation ID-only manquante"
        )

    if DEST.exists():
        raise RuntimeError(
            "STOP : dossier de sortie existant, aucun ecrasement"
        )

    rows = load_manifest()
    core = preflight(rows)

    if any(r.get("collection_authorized") != "NO" for r in rows):
        raise RuntimeError(
            "STOP : statut du manifeste inattendu"
        )

    all_ids = {r["match_id"].strip() for r in rows}
    blocked = set()

    for source in PROTECTED:
        blocked.update(all_ids & ids_only(source))

    if blocked:
        raise RuntimeError(
            "STOP : collisions avec les sources protegees connues"
        )

    selected = rows[0]
    mid = selected["match_id"].strip()

    print("\nMatch technique :", mid)
    print("Competition :", selected["competition_id"])
    print("Date :", selected["kickoff_utc"][:10])
    print("Sources ID-only controlees :", len(PROTECTED))
    print("Partitions inconnues : NON CERTIFIEES")
    print("Utilisation pour entrainement : INTERDITE")

    # Les verifications precedentes sont terminees
    # avant le premier appel API.
    lab = prematch.load_lab()
    prematch.lab = lab

    client = prematch.build_rate_limited_client(lab)
    live87 = load_live87()

    proxy = ResearchPrematchProxy(prematch, lab)

    # Controle les metadonnees du match retourne
    # avant la reconstruction des historiques.
    original_detail = prematch.get_match_detail

    def checked_detail(lab_obj, client_obj, match_id):
        if match_id != mid:
            raise RuntimeError("ID hors manifeste")

        detail = original_detail(lab_obj, client_obj, match_id)

        if not detail:
            return detail

        actual_id = str(lab.extract_match_id(detail))
        actual_comp = str(lab.extract_competition_id(detail))
        actual_season = str(lab.extract_season_id(detail))

        kickoff = prematch.dt_utc_naive(
            lab.extract_match_date(detail)
        )

        if (
            actual_id != mid
            or actual_comp != selected["competition_id"]
            or actual_season != selected["season_id"]
            or kickoff is None
            or kickoff.date().isoformat()
                != selected["kickoff_utc"][:10]
        ):
            raise RuntimeError(
                "STOP : metadonnees API non conformes au manifeste"
            )

        return detail

    proxy.get_match_detail = checked_detail

    print("\n=== COLLECTE TECHNIQUE ===")

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
        raise RuntimeError("Statut retourne pour un autre match")

    # Aucune sortie avec label ou score final.
    diagnostic = {
        "match_id": mid,
        "competition_id": selected["competition_id"],
        "kickoff_utc": selected["kickoff_utc"],
        "technical_status": status.get("status", ""),
        "technical_error": status.get("error", ""),
        "core_available_original52": status.get("core_available", ""),
        "training_authorized": "NO",
        "protected_certified": "NO",
    }

    features49 = [
        name for name in core
        if name not in {"odds_home", "odds_draw", "odds_away"}
    ]

    assert len(features49) == 49

    if row is not None:
        if row.get("match_id") != mid:
            raise RuntimeError("RAW retourne pour un autre match")

        for field in ("odds_home", "odds_draw", "odds_away"):
            value = row.get(field)

            if value is None:
                continue

            try:
                is_missing = math.isnan(float(value))
            except (TypeError, ValueError):
                is_missing = False

            if not is_missing:
                raise RuntimeError(
                    "STOP : cote reelle detectee : " + field
                )

        diagnostic["available49"] = collector.count_available(
            row, features49
        )

    DEST.mkdir(parents=True, exist_ok=False)

    write_one(
        DEST / "TECHNICAL_STATUS_NO_LABEL.csv",
        diagnostic,
    )

    if row is not None:
        safe_row = {
            "match_id": mid,
            "competition_id": selected["competition_id"],
            "season_id": selected["season_id"],
            "kickoff_utc": selected["kickoff_utc"],
        }
        safe_row.update({
            name: row.get(name) for name in features49
        })

        write_one(
            DEST / "TECHNICAL_FEATURES49_NO_LABEL.csv",
            safe_row,
        )

    print("\n=== RESULTAT ===")
    print("Statut technique :", diagnostic["technical_status"])
    print("Caracteristiques 49 disponibles :",
          diagnostic.get("available49", "aucune ligne"))
    print("Sorties :", DEST)
    print("Aucun label ou score final enregistre.")
    print("Collecte en quarantaine : OUI")
    print("Autorisation d'entrainement : NON")
    print("TEST_FUTUR non ouvert par ce script.")
    print("V3-C non modifie.")
    print("ETAPE 331 TERMINEE.")


if __name__ == "__main__":
    main()
