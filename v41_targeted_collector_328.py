import argparse
import ast
import csv
import importlib.util
import os
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch

from v41_prematch_safe_bridge_318 import (
    ResearchPrematchProxy,
)

MANIFEST = Path(
    "outputs/v4_research/v41_clearance_323/"
    "PILOT10_ID_CHECKED_NOT_AUTHORIZED.csv"
)

OUTPUT = Path(
    "outputs/v4_research/v41_pilot_collection_328"
)

EXPECTED_COMPETITIONS = {
    "comp_8531": 5,
    "comp_623699": 5,
}


def load_manifest():
    if not MANIFEST.is_file():
        raise RuntimeError(
            "STOP : manifeste pilote introuvable"
        )

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 10:
        raise RuntimeError(
            "STOP : le pilote doit contenir 10 matchs"
        )

    ids = [r["match_id"].strip() for r in rows]

    if len(set(ids)) != 10 or not all(ids):
        raise RuntimeError(
            "STOP : identifiants invalides"
        )

    counts = {
        comp: sum(
            r.get("competition_id") == comp
            for r in rows
        )
        for comp in EXPECTED_COMPETITIONS
    }

    if counts != EXPECTED_COMPETITIONS:
        raise RuntimeError(
            "STOP : repartition du pilote inattendue"
        )

    if any(
        r.get("protected_clearance")
        != "SEVEN_SOURCES_ID_ONLY_PASS"
        for r in rows
    ):
        raise RuntimeError(
            "STOP : controle ID-only absent"
        )

    return rows


def load_live87():
    path = Path(collector.LIVE87_MODULE_PATH)

    if not path.is_file():
        raise RuntimeError(
            "STOP : module LIVE87 absent : " + str(path)
        )

    spec = importlib.util.spec_from_file_location(
        "v41_live87_pilot_328",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "STOP : chargement LIVE87 impossible"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not callable(
        getattr(module, "extract_final_score", None)
    ):
        raise RuntimeError(
            "STOP : extract_final_score absent"
        )

    return module


def make_stub(row):
    match_id = row["match_id"].strip()

    return {
        "id": match_id,
        "match_id": match_id,
        "competition_id": row["competition_id"],
        "season_id": row["season_id"],
        "utc_date": row["kickoff_utc"],
    }


def preflight(rows):
    path = Path(collector.LIVE87_MODULE_PATH)

    if not path.is_file():
        raise RuntimeError(
            "STOP : fichier LIVE87 introuvable"
        )

    # Parse le code sans l'executer.
    tree = ast.parse(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        ),
        filename=str(path),
    )

    functions = {
        n.name
        for n in ast.walk(tree)
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef)
        )
    }

    if "extract_final_score" not in functions:
        raise RuntimeError(
            "STOP : fonction LIVE87 introuvable"
        )

    core = collector.load_core_features(None)

    if len(core) != 52:
        raise RuntimeError(
            "STOP : schema CORE inattendu"
        )

    for row in rows:
        stub = make_stub(row)

        actual_id = (
            collector.extract_match_id_generic(stub)
        )

        if actual_id != row["match_id"]:
            raise RuntimeError(
                "STOP : match_stub non reconnu : "
                + row["match_id"]
            )

    print("\n=== VERIFICATIONS ===")
    print("Module LIVE87 localise :", path)
    print("Fonction score final :", "OK")
    print("Schema CORE :", len(core))
    print("Identifiants reconnus :", len(rows))
    print("Filtre temporel V4.1 : disponible")
    print("Cotes V4.1 : neutralisation disponible")

    return core


def collect(rows, core):
    # Barriere explicite : ne jamais ignorer les
    # autorisations figurant dans le manifeste.
    if any(
        row.get("collection_authorized") != "YES"
        for row in rows
    ):
        raise RuntimeError(
            "STOP : collecte NON autorisee "
            "dans le manifeste"
        )

    if os.environ.get(
        "V41_AUTORISE_COLLECTE"
    ) != "OUI":
        raise RuntimeError(
            "STOP : autorisation explicite manquante"
        )

    # Les sept sources controlees ne constituent
    # pas a elles seules une certification exhaustive.
    if os.environ.get(
        "V41_PARTITIONS_CERTIFIEES"
    ) != "OUI":
        raise RuntimeError(
            "STOP : partitions protegees "
            "non certifiees"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            "STOP : dossier de sortie deja existant"
        )

    # Aucun appel API avant la validation
    # complete des trois barrieres precedentes.
    lab = prematch.load_lab()
    prematch.lab = lab

    client = prematch.build_rate_limited_client(
        lab
    )

    live87 = load_live87()

    proxy = ResearchPrematchProxy(
        prematch,
        lab,
    )

    records = []
    statuses = []

    for position, item in enumerate(rows, 1):
        mid = item["match_id"]

        print(
            f"[{position}/{len(rows)}] "
            f"Collecte ciblee : {mid}"
        )

        row, status = collector.collect_one_match(
            prematch=proxy,
            live87=live87,
            client=client,
            match_stub=make_stub(item),
            core_features=core,
            xg_limit=0,
            with_league_second_half=False,
            preferred_bookmaker=None,
        )

        if status.get("match_id") != mid:
            raise RuntimeError(
                "STOP : identifiant retourne inattendu"
            )

        if row is not None:
            if row.get("match_id") != mid:
                raise RuntimeError(
                    "STOP : ligne associee au mauvais ID"
                )

            if (
                str(row.get("competition_id"))
                != str(item["competition_id"])
                or str(row.get("season_id"))
                != str(item["season_id"])
                or str(row.get("kickoff_utc", ""))[:10]
                != str(item["kickoff_utc"])[:10]
            ):
                raise RuntimeError(
                    "STOP : incoherence de metadonnees"
                )

            if any(
                row.get(name) is not None
                for name in (
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                )
            ):
                raise RuntimeError(
                    "STOP : cotes non neutralisees"
                )

            records.append(row)

        statuses.append(status)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        if not data:
            return

        keys = list(dict.fromkeys(
            key
            for record in data
            for key in record
        ))

        with (OUTPUT / name).open(
            "x",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=keys,
            )
            writer.writeheader()
            writer.writerows(data)

    save("PILOT10_RAW_QUARANTINE.csv", records)
    save("PILOT10_STATUS.csv", statuses)

    print("\n=== COLLECTE TERMINEE ===")
    print("Lignes RAW :", len(records))
    print("Statuts :", len(statuses))
    print("Dossier :", OUTPUT)
    print(
        "Statut : QUARANTAINE TECHNIQUE, "
        "pas validation independante"
    )


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
    )
    mode.add_argument(
        "--collect",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 76)
    print("ETAPE 328 - COLLECTEUR CIBLE V4.1")
    print("=" * 76)

    rows = load_manifest()
    core = preflight(rows)

    print("\n=== PLAN DU PILOTE ===")

    for row in rows:
        print(
            row["match_id"],
            "|",
            row["competition_id"],
            "|",
            row["kickoff_utc"][:10],
            "| autorise =",
            row["collection_authorized"],
        )

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print("10 matchs verifies.")
        print("Aucun appel API.")
        print("Aucun label consulte.")
        print("Aucun dataset modifie.")
        print("Aucun fichier protege ouvert.")
        print("TEST_FUTUR non consulte.")
        print("V3-C inchange.")
        print("ETAPE 328 DRY RUN TERMINEE.")
        return

    collect(rows, core)


if __name__ == "__main__":
    main()
