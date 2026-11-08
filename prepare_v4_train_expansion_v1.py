from pathlib import Path
import json
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"
OUT = Path("outputs/v4_research")

print("=" * 72)
print("ÉTAPE 193 — PRÉPARATION DE L'EXTENSION TRAIN V4")
print("=" * 72)

manifest = pd.read_csv(
    "historical_1710_manifest.csv",
    dtype=str,
)

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype=str,
    usecols=["match_id", "split"],
)

if len(manifest) != 1710 or manifest.match_id.duplicated().any():
    raise SystemExit("Manifeste incorrect")

if len(plan) != 320 or plan.match_id.duplicated().any():
    raise SystemExit("Plan historique incorrect")

used = set(plan.match_id)
reserved = set(
    plan.loc[plan["split"] == "TEST_FUTUR", "match_id"]
)

batches = sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(batches) != 10:
    raise SystemExit("10 lots B attendus")

for path in batches:
    p = pd.read_csv(
        path,
        dtype=str,
        usecols=["match_id"],
    )

    ids = set(p.match_id)

    if len(p) != len(ids):
        raise SystemExit(f"Doublons dans {path.name}")

    if ids & used:
        raise SystemExit(f"Chevauchement dans {path.name}")

    used.update(ids)

if len(used) != 520 or len(reserved) != 183:
    raise SystemExit("Population de référence incorrecte")

manifest["kickoff"] = pd.to_datetime(
    manifest["kickoff_utc"],
    utc=True,
    errors="raise",
)

cutoff = pd.Timestamp("2026-08-25T00:00:00Z")

candidates = manifest[
    (manifest.kickoff < cutoff)
    & (~manifest.match_id.isin(used))
    & (~manifest.match_id.isin(reserved))
].copy()

candidates = candidates.sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

if len(candidates) != 95:
    raise SystemExit(
        f"95 candidats attendus, trouvé {len(candidates)}"
    )

def cache_path(mid):
    choices = [
        CACHE / f"{mid}.json",
        CACHE / f"mt_{mid}.json",
    ]
    return next((p for p in choices if p.is_file()), None)

candidates["cache_available"] = candidates.match_id.map(
    lambda mid: cache_path(mid) is not None
)

candidates["research_status"] = candidates[
    "cache_available"
].map({
    True: "CACHE_TO_AUDIT",
    False: "FETCH_REQUIRED",
})

candidates["training_certified"] = 0
candidates["live_certified"] = 0
candidates["label_used_for_selection"] = 0

print("\n=== PLAN DE COLLECTE ===")
print("Candidats :", len(candidates))
print(
    "Caches existants :",
    int(candidates.cache_available.sum())
)
print(
    "Sans cache :",
    int((~candidates.cache_available).sum())
)

audit_rows = []

for _, row in candidates[
    candidates.cache_available
].iterrows():
    mid = row.match_id
    path = cache_path(mid)

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
        payload = obj.get("data", obj)

        if not isinstance(payload, dict):
            raise ValueError("payload non dict")

        events = payload.get("events", [])

        if not isinstance(events, list):
            raise ValueError("events non liste")

        second_half = [
            e for e in events
            if isinstance(e, dict)
            and e.get("period") == "second_half"
        ]

        period_end_second = any(
            e.get("type") == "period_end"
            for e in second_half
        )

        coverage = str(
            payload.get("coverage", "UNKNOWN")
        )

        internal_id = str(
            payload.get("match_id", "")
        )

        status = (
            "BASIC_CHECK_PASS"
            if (
                coverage == "full"
                and internal_id == mid
                and len(events) > 0
                and len(second_half) > 0
                and period_end_second
            )
            else "REVIEW_REQUIRED"
        )

        audit_rows.append({
            "match_id": mid,
            "cache_file": str(path),
            "coverage": coverage,
            "internal_id_matches": internal_id == mid,
            "events_total": len(events),
            "events_second_half": len(second_half),
            "second_half_period_end": period_end_second,
            "status": status,
        })

    except Exception as exc:
        audit_rows.append({
            "match_id": mid,
            "cache_file": str(path),
            "coverage": "READ_ERROR",
            "internal_id_matches": False,
            "events_total": 0,
            "events_second_half": 0,
            "second_half_period_end": False,
            "status": f"ERROR: {str(exc)[:120]}",
        })

audit = pd.DataFrame(audit_rows)

print("\n=== AUDIT DES 12 TIMELINES ===")

if len(audit) != 12:
    raise SystemExit("12 caches attendus")

print(
    audit[
        [
            "match_id", "coverage",
            "events_total", "events_second_half",
            "second_half_period_end", "status"
        ]
    ].to_string(index=False)
)

print("\n=== SYNTHÈSE ===")
print(audit.status.value_counts().to_string())

OUT.mkdir(parents=True, exist_ok=True)

plan_path = OUT / "V4_TRAIN_EXPANSION_PLAN_V1.csv"
audit_path = OUT / "V4_CACHED_CANDIDATES_AUDIT_V1.csv"

if plan_path.exists() or audit_path.exists():
    raise SystemExit(
        "Fichier de sortie existant : aucun remplacement effectué"
    )

candidates.drop(columns=["kickoff"]).to_csv(
    plan_path, index=False
)
audit.to_csv(audit_path, index=False)

print("\n=== FICHIERS CRÉÉS ===")
print(plan_path)
print(audit_path)

print("\n=== GARANTIES ===")
print("Aucun résultat de match utilisé pour sélectionner.")
print("Aucun appel API.")
print("Aucun modèle entraîné.")
print("TRAIN original inchangé.")
print("VALIDATION et TEST_FUTUR préservés.")
print("Scanner V3-C inchangé.")
print("Nouveaux matchs : RESEARCH_ONLY.")
print("ÉTAPE 193 TERMINÉE.")
