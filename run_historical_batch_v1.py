import argparse
import csv
import re
import sys
from pathlib import Path

import safe_historical_pilot_v1 as collector

JOIN = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)
ROOT = Path("outputs/pressure_historical")
PLAN = JOIN / "HISTORICAL_COLLECTION_WAVE1_PLAN_V1.csv"

def read_rows(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(
        description="Lanceur de collecte historique sécurisée",
        add_help=False
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True
    )

    args, collector_args = parser.parse_known_args()

    tag = args.tag.upper()

    if not re.fullmatch(r"BATCH(?:0[3-9]|1[01])", tag):
        parser.error(
            "Tag autorisé : BATCH03 à BATCH11"
        )

    expected = JOIN / (
        f"HISTORICAL_WAVE1_{tag}_MANIFEST_V1.csv"
    )

    if args.manifest.resolve() != expected.resolve():
        parser.error(
            "Le manifeste doit correspondre au tag."
        )

    if not expected.is_file():
        raise FileNotFoundError(expected)

    batch = read_rows(expected)
    plan = read_rows(PLAN)

    plan_by_id = {
        row["match_id"]: row
        for row in plan
    }

    ids = [row["match_id"] for row in batch]

    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("20 IDs uniques attendus")

    for mid in ids:
        if mid not in plan_by_id:
            raise ValueError(
                f"Match absent du plan : {mid}"
            )

        if plan_by_id[mid]["target_split"] not in (
            "TRAIN", "VALIDATION"
        ):
            raise ValueError(
                f"Split interdit : {mid}"
            )

    # Réutiliser le collecteur sécurisé existant.
    collector.PLAN = expected

    # Journal distinct pour chaque lot.
    collector.LOG = (
        ROOT / f"{tag.lower()}_safe_fetch_log_v1.jsonl"
    )

    # Transmettre seulement les options de collecte
    # à son propre analyseur d'arguments.
    sys.argv = [sys.argv[0]] + collector_args

    print(
        f"\n=== COLLECTEUR UNIVERSEL {tag} ===",
        flush=True
    )

    collector.main()

if __name__ == "__main__":
    main()
