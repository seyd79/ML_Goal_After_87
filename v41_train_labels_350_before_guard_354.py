import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch
from v41_prematch_safe_bridge_318 import ResearchPrematchProxy
from v41_targeted_collector_328 import load_live87, make_stub
from v41_batch_technical_337 import check_detail

BASE = Path("outputs/v4_research")
DATA = BASE / "v41_dataset_preparation_343/V41_CORE49_197_QUARANTINE_NO_LABEL.csv"
FREEZE = BASE / "v41_dataset_freeze_344/V41_CORE49_197_QUARANTINE_FREEZE.json"
SPLIT = BASE / "v41_chronological_split_345/V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
LOCK = BASE / "v41_split_freeze_346/V41_SPLIT_FREEZE_PROVISIONAL.json"
PLAN = BASE / "v41_clearance_323/PLAN210_ID_CHECKED_NOT_AUTHORIZED.csv"
GATE = BASE / "v41_quality_gate_342/V41_TECHNICAL_ELIGIBILITY_210.csv"
DEST = BASE / "v41_train_labels_quarantine_350"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def binary(value):
    if isinstance(value, bool):
        return int(value)
    if str(value).strip() in ("0", "0.0"):
        return 0
    if str(value).strip() in ("1", "1.0"):
        return 1
    raise ValueError("Label non binaire ou absent")


def verify():
    with FREEZE.open(encoding="utf-8") as f:
        freeze = json.load(f)
    with LOCK.open(encoding="utf-8") as f:
        lock = json.load(f)

    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()

    if data_hash != freeze["sha256"]:
        raise RuntimeError("STOP : dataset modifie")

    if split_hash != lock["split_sha256"]:
        raise RuntimeError("STOP : split modifie")

    data = read_csv(DATA)
    splits = read_csv(SPLIT)
    plan = read_csv(PLAN)
    gate = read_csv(GATE)

    if any(len(x) != n for x, n in (
        (data, 197), (splits, 197),
        (plan, 210), (gate, 210)
    )):
        raise RuntimeError("STOP : manifestes incomplets")

    plan_by_id = {r["match_id"]: r for r in plan}
    gate_by_id = {r["match_id"]: r for r in gate}
    data_ids = {r["match_id"] for r in data}

    if len(plan_by_id) != 210 or len(gate_by_id) != 210:
        raise RuntimeError("STOP : doublons dans les manifestes")

    if len(data_ids) != 197:
        raise RuntimeError("STOP : doublons dans le dataset")

    if any(
        r.get("collection_authorized") != "NO"
        for r in plan
    ):
        raise RuntimeError("STOP : autorisation inattendue")

    train = [
        r for r in splits
        if r["split"] == "TRAIN_PROVISIONAL"
    ]

    if len(train) != 112:
        raise RuntimeError("STOP : 112 TRAIN attendus")

    for r in train:
        mid = r["match_id"]

        if (
            mid not in data_ids
            or mid not in plan_by_id
            or mid not in gate_by_id
            or gate_by_id[mid]["technical_eligibility"]
                != "TECHNICAL_CANDIDATE_NOT_CERTIFIED"
            or not r["kickoff_utc"].startswith("2025-")
        ):
            raise RuntimeError("STOP : candidat TRAIN incoherent")

        for k in ("competition_id", "kickoff_utc"):
            if r[k] != plan_by_id[mid][k]:
                raise RuntimeError("STOP : metadata incoherentes")

    return sorted(
        [plan_by_id[r["match_id"]] for r in train],
        key=lambda r: (r["kickoff_utc"], r["match_id"])
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("ETAPE 350 - LABELS TRAIN V4.1 EN QUARANTAINE")
    print("=" * 68)

    train = verify()
    print("TRAIN admissibles techniquement :", len(train))
    print("VALIDATION et HOLDOUT exclus de cette operation.")
    print("Gel du dataset et du split : OK")

    if args.dry_run:
        print("DRY-RUN OK : aucun appel API, aucun fichier cree.")
        return

    if os.environ.get("V41_AUTORISE_LABELS_TRAIN") != "OUI":
        raise RuntimeError(
            "STOP : autorisation explicite des labels TRAIN manquante"
        )

    if args.pilot:
        train = train[:1]

    core = collector.load_core_features(None)

    if len(core) != 52:
        raise RuntimeError("STOP : schema CORE inattendu")

    lab = prematch.load_lab()
    prematch.lab = lab
    client = prematch.build_rate_limited_client(lab)
    live87 = load_live87()
    proxy = ResearchPrematchProxy(prematch, lab)
    original_detail = prematch.get_match_detail

    count = 0

    for index, expected in enumerate(train, 1):
        mid = expected["match_id"]
        output = DEST / f"{mid}.json"

        if output.exists():
            with output.open(encoding="utf-8") as f:
                saved = json.load(f)

            if (
                saved.get("match_id") != mid
                or saved.get("split") != "TRAIN_PROVISIONAL"
                or saved.get("training_authorized") is not False
            ):
                raise RuntimeError("STOP : checkpoint incoherent")

            binary(saved.get("goal_after_87"))
            count += 1
            print(f"[{index}/{len(train)}] DEJA ETIQUETE : {mid}")
            continue

        def checked_detail(lab_obj, client_obj, requested_id):
            if requested_id != mid:
                raise RuntimeError("STOP : ID hors TRAIN")

            detail = original_detail(
                lab_obj, client_obj, requested_id
            )

            if detail:
                check_detail(lab, detail, expected)

            return detail

        proxy.get_match_detail = checked_detail

        row, status = collector.collect_one_match(
            prematch=proxy,
            live87=live87,
            client=client,
            match_stub=make_stub(expected),
            core_features=core,
            xg_limit=0,
            with_league_second_half=False,
            preferred_bookmaker=None,
        )

        if (
            row is None
            or status.get("status") != "done"
            or row.get("match_id") != mid
            or status.get("match_id") != mid
        ):
            raise RuntimeError(
                "STOP : reconstruction non complete : "
                + mid + " / " + str(status.get("error"))
            )

        a = binary(row.get("goal_after_87"))
        b = binary(status.get("goal_after_87"))

        if a != b:
            raise RuntimeError(
                "STOP : contradiction entre labels : " + mid
            )

        target_status = str(
            status.get("target_status") or ""
        ).strip()

        if target_status.lower() in (
            "", "unknown", "missing", "error",
            "failed", "partial", "unavailable", "none"
        ):
            raise RuntimeError(
                "STOP : provenance du label insuffisante : "
                + mid
            )

        record = {
            "match_id": mid,
            "split": "TRAIN_PROVISIONAL",
            "goal_after_87": a,
            "target_status": target_status,
            "label_origin": "historical_collector_recomputed",
            "training_authorized": False,
            "protected_certified": False,
            "verification_status": "PROVISIONAL_UNVERIFIED",
        }

        DEST.mkdir(parents=True, exist_ok=True)

        temporary = DEST / f"{mid}.json.tmp"
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if output.exists():
            raise RuntimeError("STOP : checkpoint concurrent")

        temporary.rename(output)
        count += 1

        print(
            f"[{index}/{len(train)}] LABEL PROVISOIRE : "
            f"{mid} | target_status={target_status}"
        )

    print("\nLabels TRAIN enregistres :", count)
    print("Sorties :", DEST)
    print("Validation / holdout : NON CONSULTES")
    print("Entrainement : NON AUTORISE")
    print("ETAPE 350 TERMINEE.")


if __name__ == "__main__":
    main()
