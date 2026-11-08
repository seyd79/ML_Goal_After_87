import csv
import json
import math
import importlib
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
FROZEN = (
    ROOT / "outputs/v4_research"
    / "v41_threshold_challenger_389b"
    / "V41_A2_10_PREDICTIONS_FROZEN.csv"
)
OUTPUT = ROOT / "outputs/v4_research/v41_certification_400"
REPORT = OUTPUT / "V41_A2_10_LABELS_API_CANDIDATES.csv"
EVIDENCE = OUTPUT / "V41_A2_10_API_EVIDENCE.json"

EXPECTED_IDS = [
    "mt_021025189", "mt_518519825",
    "mt_021005111", "mt_642660235",
    "mt_376334684", "mt_376415356",
    "mt_466066020", "mt_460442012",
    "mt_205173127", "mt_734651725",
]

def clean(value):
    return "" if value is None else str(value)

def score_text(home, away):
    if home is None or away is None:
        return ""
    return f"{home}-{away}"

def event_minute(live87, event):
    try:
        return live87.extract_event_minute(event)
    except Exception:
        return None

def get_goals(collector, live87, events):
    result = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            if not collector.event_is_goal(live87, event):
                continue
            if not collector.event_is_regulation_time(event):
                continue
            minute = event_minute(live87, event)
            result.append({
                "minute": minute,
                "event": event,
            })
        except Exception:
            continue
    return result

def main():
    if REPORT.exists() or EVIDENCE.exists():
        raise SystemExit(
            "STOP : fichier de sortie deja existant. "
            "Aucun ecrasement autorise."
        )

    with FROZEN.open(encoding="utf-8-sig", newline="") as f:
        predictions = list(csv.DictReader(f))

    ids = [p["match_id"] for p in predictions]

    if ids != EXPECTED_IDS or len(set(ids)) != 10:
        raise SystemExit(
            "STOP : les 10 IDs figes ne correspondent pas."
        )

    for p in predictions:
        prob = float(p["probability"])
        if not math.isfinite(prob) or not 0.50 <= prob <= 1:
            raise SystemExit("STOP : probabilite invalide.")

    load_dotenv(ROOT / ".env")

    prematch = importlib.import_module("rebuild_clean_prematch_v2")
    live87 = importlib.import_module("v41_live87_own_goal_364")
    collector = importlib.import_module("collect_historical_dataset_v2")

    prematch.lab = prematch.load_lab()
    client = prematch.build_rate_limited_client(prematch.lab)

    results = []
    evidence = {}

    print("=" * 75)
    print("ETAPE 400D - VALIDATION API DES 10 PREDICTIONS")
    print("=" * 75)

    for i, p in enumerate(predictions, 1):
        mid = p["match_id"]
        print(f"\n[{i:02d}/10] {mid}", flush=True)

        row = {
            "match_id": mid,
            "kickoff_frozen": p["kickoff_utc"],
            "probability_frozen": p["probability"],
            "api_match_id": "",
            "identity_check": "UNKNOWN",
            "score_final": "",
            "timeline_count": 0,
            "shotmap_count": 0,
            "goal_minutes_regulation": "",
            "goal_minutes_after87": "",
            "label_candidate": "",
            "target_status": "",
            "target_resolution_class": "",
            "target_confidence": "",
            "target_warning": "",
            "certification_status": "REVIEW_REQUIRED",
            "error": "",
        }

        try:
            match = prematch.get_match_detail(
                prematch.lab, client, mid
            )

            if not match or not isinstance(match, dict):
                raise ValueError("match_detail_unavailable")

            try:
                returned_id = prematch.lab.extract_match_id(match)
            except Exception:
                returned_id = None

            row["api_match_id"] = clean(returned_id)

            if returned_id is None:
                row["identity_check"] = "UNVERIFIED"
            elif str(returned_id) != mid:
                row["identity_check"] = "MISMATCH"
                raise ValueError("match_id_mismatch")
            else:
                row["identity_check"] = "MATCH"

            final_home, final_away = live87.extract_final_score(
                match
            )
            row["score_final"] = score_text(
                final_home, final_away
            )

            timeline, shots = collector.fetch_timeline_and_shotmap(
                live87, client, mid
            )

            row["timeline_count"] = len(timeline)
            row["shotmap_count"] = len(shots)

            goals = get_goals(
                collector, live87, timeline
            )

            minutes = [
                g["minute"] for g in goals
                if g["minute"] is not None
            ]
            after = [m for m in minutes if m > 87]

            row["goal_minutes_regulation"] = json.dumps(minutes)
            row["goal_minutes_after87"] = json.dumps(after)

            target = collector.compute_target_after87(
                live87=live87,
                timeline_events=timeline,
                final_home=final_home,
                final_away=final_away,
            )

            label = target.get("goal_after_87")

            if label is not None and not (
                isinstance(label, float) and math.isnan(label)
            ):
                if str(label) in ("0", "0.0", "1", "1.0"):
                    row["label_candidate"] = int(float(label))

            for key in (
                "target_status",
                "target_resolution_class",
                "target_confidence",
                "target_warning",
            ):
                row[key] = clean(target.get(key))

            evidence[mid] = {
                "match_detail": match,
                "timeline_events": timeline,
                "shotmap_event_count": len(shots),
                "target_result": target,
            }

            print("Score final :", row["score_final"])
            print("Evenements :", len(timeline))
            print("Buts reglementaires :", minutes)
            print("Buts apres 87 :", after)
            print("Label candidat :", row["label_candidate"])
            print("Statut :", row["target_status"])

        except Exception as exc:
            row["error"] = (
                type(exc).__name__ + ": " + str(exc)
            )[:300]
            print("ERREUR :", row["error"])

        results.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with REPORT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    with EVIDENCE.open("x", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "protocol": "V41_A2_STEP400_API_REVIEW",
                "model_modified": False,
                "threshold_modified": False,
                "final_certification_performed": False,
                "matches": evidence,
            },
            f, ensure_ascii=False, indent=2, default=str
        )

    positive = sum(
        r["label_candidate"] == 1 for r in results
    )
    negative = sum(
        r["label_candidate"] == 0 for r in results
    )
    unknown = len(results) - positive - negative

    print("\n" + "=" * 75)
    print("BILAN ETAPE 400D")
    print("=" * 75)
    print("Labels candidats positifs :", positive)
    print("Labels candidats negatifs :", negative)
    print("Labels inconnus :", unknown)
    print("Labels definitivement certifies : 0")
    print("Rapport :", REPORT)
    print("Preuves API :", EVIDENCE)
    print("Modele modifie : NON")
    print("Predictions modifiees : NON")
    print("ETAPE 400D TERMINEE")

if __name__ == "__main__":
    main()
