import argparse
import csv
import json
import re
import importlib.util
from pathlib import Path
from collections import Counter
from decimal import Decimal, InvalidOperation

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

PLAN = JOIN / "HISTORICAL_COLLECTION_WAVE1_PLAN_V1.csv"
MANIFEST = Path("historical_1710_manifest.csv")
BUILDER = Path("build_historical_ab_research_v3.py")


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "match_id" not in reader.fieldnames:
            raise ValueError(f"CSV invalide : {path}")
        rows = list(reader)

    result = {}
    for row in rows:
        mid = row["match_id"]
        if mid in result:
            raise ValueError(f"ID dupliqué : {mid}")
        result[mid] = row

    return result


def same(x, y):
    if str(x) == str(y):
        return True
    try:
        a = Decimal(str(x))
        b = Decimal(str(y))
        return a.is_finite() and b.is_finite() and a == b
    except (InvalidOperation, ValueError, TypeError):
        return False


def save_csv(path, rows):
    if not rows:
        raise ValueError("Rapport vide")

    fields = list(rows[0].keys())

    for row in rows:
        if set(row) != set(fields):
            raise ValueError(f"Colonnes incohérentes : {path}")

    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expect-train", type=int, default=10)
    parser.add_argument("--expect-validation", type=int, default=10)
    args = parser.parse_args()

    tag = args.tag.upper()

    if not re.fullmatch(r"[A-Z0-9_]{2,40}", tag):
        parser.error("Tag invalide")

    outputs = [
        JOIN / f"{tag}_A_RESEARCH_V1.csv",
        JOIN / f"{tag}_B_RESEARCH_V1.csv",
        JOIN / f"{tag}_QUALITY_AUDIT_V1.csv"
    ]

    for path in outputs:
        if path.exists():
            raise FileExistsError(path)

    manifest = load_csv(MANIFEST)
    plan = load_csv(PLAN)
    batch = load_csv(args.manifest)

    expected_total = args.expect_train + args.expect_validation

    if len(batch) != expected_total:
        raise ValueError(
            f"{expected_total} matchs attendus, "
            f"{len(batch)} trouvés"
        )

    spec = importlib.util.spec_from_file_location(
        "builder_v3", BUILDER
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    features_a = list(builder.A_FEATURES)
    features_b = features_a + list(builder.B_EXTRA)

    if len(features_a) != 23 or len(features_b) != 29:
        raise ValueError("Dimensions A/B incorrectes")

    rows_a = []
    rows_b = []
    audit = []
    stats = Counter()
    failures = []

    print(f"\n=== AUDIT HISTORIQUE {tag} ===")

    ordered_ids = sorted(
        batch,
        key=lambda mid: (batch[mid]["kickoff_utc"], mid)
    )

    for mid in ordered_ids:
        try:
            if mid not in manifest or mid not in plan:
                raise ValueError("MATCH_NOT_IN_PLAN")

            if (
                batch[mid]["kickoff_utc"]
                != manifest[mid]["kickoff_utc"]
            ):
                raise ValueError("KICKOFF_MISMATCH")

            group = plan[mid]["target_split"]

            if group not in ("TRAIN", "VALIDATION"):
                raise ValueError("FORBIDDEN_SPLIT")

            path = CACHE / f"{mid}.json"
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            data = payload.get("data")

            if not isinstance(data, dict):
                raise ValueError("INVALID_DATA")

            if str(data.get("match_id")) != mid:
                raise ValueError("MATCH_ID_MISMATCH")

            if data.get("coverage") != "full":
                raise ValueError("NOT_FULL")

            events = data.get("events")

            if not isinstance(events, list) or not events:
                raise ValueError("INVALID_EVENTS")

            full_a, full_b, quality = builder.build_match(
                mid, manifest[mid], data
            )

            if full_a is None or full_b is None:
                raise ValueError(f"FULL_BUILD_FAILED:{quality}")

            filtered = []
            at87 = after87 = other = 0

            for event in events:
                if not isinstance(event, dict):
                    other += 1
                    continue

                period = str(
                    event.get("period") or ""
                ).lower()

                if period not in ("first_half", "second_half"):
                    other += 1
                    continue

                try:
                    minute = builder.event_minute(event)
                except (ValueError, TypeError):
                    other += 1
                    continue

                if minute < 87:
                    filtered.append(event)
                elif minute == 87:
                    at87 += 1
                else:
                    after87 += 1

            asof_data = dict(data)
            asof_data["events"] = filtered

            asof_a, asof_b, asof_quality = (
                builder.build_match(
                    mid, manifest[mid], asof_data
                )
            )

            if asof_a is None or asof_b is None:
                raise ValueError(
                    f"ASOF_BUILD_FAILED:{asof_quality}"
                )

            checks = 0

            for name, full, asof, features in (
                ("A", full_a, asof_a, features_a),
                ("B", full_b, asof_b, features_b)
            ):
                for feature in features:
                    if feature not in full or feature not in asof:
                        raise ValueError(
                            f"MISSING_{name}:{feature}"
                        )

                    if not same(
                        full[feature], asof[feature]
                    ):
                        raise ValueError(
                            f"TEMPORAL_DIFFERENCE_{name}:{feature}"
                        )

                    checks += 1

                for side in ("home", "away"):
                    shots = Decimal(
                        str(full[f"{side}_shots_87"])
                    )
                    sot = Decimal(
                        str(full[f"{side}_shots_on_target_87"])
                    )

                    if not (
                        shots.is_finite()
                        and sot.is_finite()
                        and 0 <= sot <= shots
                    ):
                        raise ValueError(
                            f"INVALID_SHOTS_{name}:{side}"
                        )

            if checks != 52:
                raise ValueError("FEATURE_COUNT_MISMATCH")

            if not same(
                full_a["goal_after_87"],
                full_b["goal_after_87"]
            ):
                raise ValueError("LABEL_A_B_MISMATCH")

            label = int(float(full_a["goal_after_87"]))

            if label not in (0, 1):
                raise ValueError("INVALID_LABEL")

            metadata = {
                "research_split": group,
                "research_origin": tag,
                "research_quality_gate": "TECHNICAL_ONLY",
                "independent_verified": 0,
                "live_certified": 0,
                "training_certified": 0
            }

            rows_a.append({**full_a, **metadata})
            rows_b.append({**full_b, **metadata})

            audit.append({
                "match_id": mid,
                "kickoff_utc": manifest[mid]["kickoff_utc"],
                "split": group,
                "events_full": len(events),
                "events_asof87": len(filtered),
                "events_at87_excluded": at87,
                "events_after87_excluded": after87,
                "other_events_excluded": other,
                "label_from_full_timeline": label,
                "feature_checks": checks,
                "technical_gate": "PASSED_RESEARCH_ONLY",
                "independent_verified": 0,
                "live_certified": 0,
                "training_certified": 0
            })

            stats[group] += 1
            stats[f"{group}_POSITIVE"] += label
            stats["FEATURE_CHECKS"] += checks

            print(
                "PASSED", mid,
                "|", group,
                "| label:", label,
                "| events:", len(events),
                "| ASOF:", len(filtered)
            )

        except Exception as exc:
            failures.append((mid, str(exc)))
            print("FAILED", mid, "|", str(exc)[:160])

    print("\n=== BILAN ===")
    print("Réussites :", len(audit))
    print("Échecs :", len(failures))
    print("TRAIN :", stats["TRAIN"])
    print("VALIDATION :", stats["VALIDATION"])
    print("Positifs TRAIN :", stats["TRAIN_POSITIVE"])
    print("Positifs VALIDATION :", stats["VALIDATION_POSITIVE"])
    print("Comparaisons A/B :", stats["FEATURE_CHECKS"])

    if failures:
        print("\nDétails des échecs :")
        for failure in failures:
            print(failure)
        raise SystemExit(
            "AUDIT ÉCHOUÉ : aucun dataset créé"
        )

    if (
        len(audit) != expected_total
        or stats["TRAIN"] != args.expect_train
        or stats["VALIDATION"] != args.expect_validation
        or stats["FEATURE_CHECKS"] != expected_total * 52
    ):
        raise ValueError("BILAN_INCORRECT")

    save_csv(outputs[0], rows_a)
    save_csv(outputs[1], rows_b)
    save_csv(outputs[2], audit)

    print("\n=== FICHIERS CRÉÉS ===")
    for path in outputs:
        print(path)

    print("\nDonnées de recherche uniquement.")
    print("Aucun appel API.")
    print("Aucun entraînement.")
    print("Aucune certification LIVE.")


if __name__ == "__main__":
    main()
