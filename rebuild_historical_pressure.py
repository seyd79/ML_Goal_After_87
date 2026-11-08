#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3-C Étape 6 : reconstruire la pression offensive sur 5 minutes historiques.

Python 3.9+ / bibliothèque standard seulement.

  python rebuild_historical_pressure.py prepare --input historique.tsv --outdir outputs/pressure_historical
  python rebuild_historical_pressure.py fetch --manifest outputs/pressure_historical/manifest.csv --outdir outputs/pressure_historical --limit 5
  python rebuild_historical_pressure.py rebuild --manifest outputs/pressure_historical/manifest.csv --outdir outputs/pressure_historical

Aucune mise à jour de V3-C; sortie EXPLORATOIRE, non certifiée pour entraîner.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from v3c_pressure_features import FEATURES, build_features, event_category, event_minute, get_events, team_side

MINUTES = (85, 86, 87)
RAW_COLUMNS = ("match_id", "home_team", "away_team", "home_id", "away_id", "kickoff_utc")
META_COLUMNS = ("match_id", "minute", "home_team", "away_team", "kickoff_utc", "timeline_coverage",
                "timeline_last_updated", "timeline_events_total", "timeline_shot_events_total",
                "timeline_sot_events_total", "timeline_corner_events_total", "timeline_event_unknown_team_total",
                "timeline_event_invalid_minute_total", "timeline_duplicate_sequences", "raw_cache_path",
                "historical_reconstruction_status", "stats_crosscheck_status", "training_ready", "temporal_provenance")
FEATURE_COLUMNS = META_COLUMNS + (
    "events_timeline_count", "events_in_window", "events_unknown_team", "events_unknown_type",
    "events_future_filtered", "events_invalid_minute", "events_available_at_snapshot",
    *FEATURES,
)


def load_dot_env(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k in ("STATS_API_KEY", "STATS_API_BASE_URL") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def from_excel_serial(value):
    try:
        val = float(value)
        if 20000 < val < 90000:
            return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=val)).isoformat()
    except (ValueError, TypeError, OverflowError):
        pass
    return str(value or "")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fd:
        sample = fd.read(10000)
        fd.seek(0)
        delim = "\t" if path.suffix.lower() == ".tsv" or (sample.splitlines() and "\t" in sample.splitlines()[0]) else ","
        return list(csv.DictReader(fd, delimiter=delim))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    src = read_rows(args.input)
    if not src or not {"match_id", "home_team", "away_team"}.issubset(src[0]):
        raise ValueError("Entrée sans match_id/home_team/away_team. Fournir CSV ou TSV de matchs historiques.")
    seen = set()
    manifest, labels = [], []
    for entry in src:
        match_id = str(entry.get("match_id") or "").strip()
        home, away = str(entry.get("home_team") or "").strip(), str(entry.get("away_team") or "").strip()
        if not match_id or not home or not away or match_id in seen:
            continue
        seen.add(match_id)
        manifest.append({"match_id": match_id, "home_team": home, "away_team": away,
                         "home_id": entry.get("home_id") or "", "away_id": entry.get("away_id") or "",
                         "kickoff_utc": from_excel_serial(entry.get("kickoff_utc") or entry.get("date") or "")})
        labels.append({"match_id": match_id, "reference_goal_after_87": entry.get("goal_after_87") or "",
                       "reference_target_resolution_class": entry.get("target_resolution_class") or "",
                       "reference_target_confidence": entry.get("target_confidence") or "",
                       "reference_target_warning": entry.get("target_warning") or "",
                       "reference_target_status": entry.get("target_status") or ""})
    write_csv(args.outdir / "manifest.csv", RAW_COLUMNS, manifest)
    write_csv(args.outdir / "reference_labels_SEPARATE.csv",
              list(labels[0]) if labels else ["match_id"], labels)
    print(json.dumps({"input_rows": len(src), "unique_matches": len(manifest),
                      "manifest": str(args.outdir / "manifest.csv"),
                      "reference_labels": str(args.outdir / "reference_labels_SEPARATE.csv")}, indent=2))


def import_captures(args):
    """Réutilise les captures brutes; ne prétend JAMAIS qu'elles étaient live."""
    rows = {}
    if not args.captures.is_file():
        raise ValueError("Fichier de captures introuvable : %s" % args.captures)
    with args.captures.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                capture = json.loads(line)
                mid = safe_match_id(capture["match_id"])
                payload = capture["timeline_payload"]
                returned = payload.get("data", {}).get("match_id")
                if returned != mid:
                    print("IGNORE ligne", line_num, "ID discordant", file=sys.stderr)
                    continue
                home, away = capture.get("home_team"), capture.get("away_team")
                if not home or not away:
                    continue
                rows[mid] = (capture, payload)
            except (ValueError, KeyError, TypeError, AttributeError):
                print("IGNORE ligne", line_num, "capture malformée", file=sys.stderr)
                continue
    manifest = []
    for mid, (cap, payload) in rows.items():
        p = raw_path(args.outdir, mid)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest.append({"match_id": mid, "home_team": cap["home_team"],
                         "away_team": cap["away_team"],
                         "home_id": cap.get("home_id") or "", "away_id": cap.get("away_id") or "",
                         "kickoff_utc": ""})
    write_csv(args.outdir / "manifest.csv", RAW_COLUMNS, manifest)
    print(json.dumps({"imported_matches": len(manifest), "manifest": str(args.outdir / "manifest.csv"),
                      "provenance": "HISTORICAL_DIAGNOSTIC_ONLY", "api_requests": 0}, ensure_ascii=False, indent=2))


def safe_match_id(value):
    match_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,100}", match_id):
        raise ValueError("match_id invalide: %r" % match_id)
    return match_id


def raw_path(outdir, match_id):
    return outdir / "timeline_cache" / (safe_match_id(match_id) + ".json")


def api_get(match_id, api_key, base, timeout):
    if not base.startswith("https://"):
        raise ValueError("Base API non HTTPS, refusée.")
    url = base.rstrip("/") + "/football/matches/" + urllib.parse.quote(safe_match_id(match_id)) + "/timeline"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + api_key,
                                               "Accept": "application/json", "User-Agent": "v3c-historical-pressure/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def fetch(args):
    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("Manifest vide")
    load_dot_env(args.env)
    key = (os.environ.get("STATS_API_KEY") or "").strip()
    if key.lower().startswith("bearer "):
        key = key[7:]
    if not key and not args.dry_run:
        raise ValueError("STATS_API_KEY absente du .env ou des variables d'environnement")
    base = (os.environ.get("STATS_API_BASE_URL") or "https://api.thestatsapi.com/api").rstrip("/")
    done, skipped, failed, planned = 0, 0, 0, 0
    report_file = args.outdir / "api_fetch_log.jsonl"
    for r in rows:
        if done + failed >= args.limit:
            break
        match_id = safe_match_id(r["match_id"])
        cache = raw_path(args.outdir, match_id)
        if cache.exists() and not args.force:
            skipped += 1
            continue
        planned += 1
        if args.dry_run:
            print("WOULD_FETCH", match_id, r.get("home_team"), "-", r.get("away_team"))
            if planned >= args.limit:
                break
            continue
        try:
            payload = api_get(match_id, key, base, args.timeout)
            if not isinstance(payload, (dict, list)):
                raise ValueError("API JSON top-level non exploitable")
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache)
            done += 1
            print("OK", match_id, "(%s/%s nouveaux)" % (done, args.limit), flush=True)
            status = "OK"
        except urllib.error.HTTPError as e:
            failed += 1
            # No API response body in log: may contain sensitive details.
            status = "HTTP_%s" % e.code
            print("ECHEC", match_id, status, file=sys.stderr, flush=True)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as e:
            failed += 1
            status = type(e).__name__
            print("ECHEC", match_id, status, file=sys.stderr, flush=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"match_id": match_id, "status": status,
                                  "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        if args.delay and done + failed < args.limit:
            time.sleep(args.delay)
    print(json.dumps({"manifest_matches": len(rows), "fetched": done, "cached_skipped": skipped,
                      "failed": failed, "requests_planned": planned, "dry_run": args.dry_run,
                      "raw_directory": str(args.outdir / "timeline_cache")}, ensure_ascii=False, indent=2))


def structural_audit(payload, home_team, away_team, home_id=None, away_id=None):
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    cov = (data.get("coverage") or meta.get("coverage")) if isinstance(data, dict) and isinstance(meta, dict) else ""
    events = get_events(payload)
    if events is None:
        return {"coverage": cov, "meta_updated": meta.get("last_updated", "") if isinstance(meta, dict) else "",
                "events": None, "shots": 0, "sot": 0, "corners": 0,
                "unknown_team": 0, "invalid_minute": 0, "duplicate_sequences": 0}
    shots = sot = corners = unknown_team = invalid_minute = duplicate_sequences = 0
    seen_sequences = set()
    for e in events:
        seq = e.get("sequence")
        if seq is not None:
            if str(seq) in seen_sequences:
                duplicate_sequences += 1
                continue
            seen_sequences.add(str(seq))
        kind = event_category(e)
        if event_minute(e) is None:
            invalid_minute += 1
        if kind:
            if team_side(e, home_team, away_team, home_id, away_id) is None:
                unknown_team += 1
            if kind == "sot":
                sot += 1
                shots += 1
            elif kind == "shot":
                shots += 1
            elif kind == "corner":
                corners += 1
    return {"coverage": cov, "meta_updated": meta.get("last_updated", "") if isinstance(meta, dict) else "",
            "events": events, "shots": shots, "sot": sot, "corners": corners,
            "unknown_team": unknown_team, "invalid_minute": invalid_minute,
            "duplicate_sequences": duplicate_sequences}


def compare_stats(audit, official_stats):
    """Official full-time counts, from a separate source; no zero imputation."""
    if not official_stats:
        return "NOT_PROVIDED"
    needed = ("home_shots_ft", "away_shots_ft", "home_sot_ft", "away_sot_ft",
              "home_corners_ft", "away_corners_ft")
    try:
        stated = [int(official_stats[key]) for key in needed]
        if any(x < 0 for x in stated):
            return "REFERENCE_INVALID"
    except (KeyError, ValueError, TypeError):
        return "REFERENCE_INCOMPLETE"
    # This basic crosscheck requires a provider-specific team split.
    # It cannot be proven from aggregate totals; recomputed below using events.
    return "REFERENCE_PRESENT_NEEDS_SIDE_RECOUNT"


def reconstructed_rows(payload, manifest_row, cache_path="", official_stats=None):
    match_id = manifest_row["match_id"]
    home, away = manifest_row["home_team"], manifest_row["away_team"]
    h_id, a_id = manifest_row.get("home_id"), manifest_row.get("away_id")
    # Never silently attach timeline of another match.
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    returned_id = data.get("match_id") if isinstance(data, dict) else None
    match_id_ok = bool(returned_id) and str(returned_id) == match_id
    audit = structural_audit(payload, home, away, h_id, a_id)
    events = audit["events"]
    crosscheck = compare_stats(audit, official_stats)
    if crosscheck == "REFERENCE_PRESENT_NEEDS_SIDE_RECOUNT" and events is not None:
        counts = Counter()
        seq_seen = set()
        for event in events:
            seq = event.get("sequence")
            if seq is not None and str(seq) in seq_seen:
                continue
            if seq is not None:
                seq_seen.add(str(seq))
            kind = event_category(event)
            side = team_side(event, home, away, h_id, a_id)
            if kind and side:
                if kind == "sot":
                    counts["sot_" + side] += 1
                    counts["shots_" + side] += 1
                elif kind == "shot":
                    counts["shots_" + side] += 1
                elif kind == "corner":
                    counts["corners_" + side] += 1
        values = [counts["shots_home"], counts["shots_away"],
                  counts["sot_home"], counts["sot_away"],
                  counts["corners_home"], counts["corners_away"]]
        reference = [int(official_stats[k]) for k in
                     ("home_shots_ft", "away_shots_ft", "home_sot_ft", "away_sot_ft",
                      "home_corners_ft", "away_corners_ft")]
        crosscheck = "MATCH_6_FT_COUNTS" if values == reference else "MISMATCH_FT_COUNTS"
    output = []
    for minute in MINUTES:
        p = build_features(payload, match_id, minute, home, away,
                           source="HISTORICAL_RECONSTRUCTED",
                           home_id=h_id, away_id=a_id, schema_confirmed=False)
        state = "HISTORICAL_CANDIDATE_STATS_NOT_CROSSCHECKED"
        if not match_id_ok:
            state = "MISMATCH_OR_MISSING_RESPONSE_MATCH_ID"
        elif events is None:
            state = "MISSING_TIMELINE"
        elif str(audit["coverage"]).lower() != "full":
            state = "COVERAGE_NOT_FULL"
        elif audit["duplicate_sequences"]:
            state = "DUPLICATE_SEQUENCE_AUDIT_REQUIRED"
        elif audit["invalid_minute"] or audit["unknown_team"]:
            state = "BAD_EVENT_TIME_OR_TEAM"
        elif audit["shots"] + audit["corners"] == 0:
            state = "NO_OFFENSIVE_EVENT_TYPES_VERIFY_PROVIDER"
        elif crosscheck == "MISMATCH_FT_COUNTS":
            state = "OFFICIAL_STATS_MISMATCH"
        elif crosscheck in ("REFERENCE_INVALID", "REFERENCE_INCOMPLETE"):
            state = "OFFICIAL_STATS_BAD_REFERENCE"
        elif p["pressure_status"].startswith("INCOMPLETE"):
            state = "BAD_EVENTS_IN_WINDOW"
        fields = {
            "match_id": match_id, "minute": minute, "home_team": home, "away_team": away,
            "kickoff_utc": manifest_row.get("kickoff_utc", ""),
            "timeline_coverage": audit["coverage"], "timeline_last_updated": audit["meta_updated"],
            "timeline_events_total": len(events) if events is not None else "",
            "timeline_shot_events_total": audit["shots"], "timeline_sot_events_total": audit["sot"],
            "timeline_corner_events_total": audit["corners"],
            "timeline_event_unknown_team_total": audit["unknown_team"],
            "timeline_event_invalid_minute_total": audit["invalid_minute"],
            "timeline_duplicate_sequences": audit["duplicate_sequences"],
            "raw_cache_path": str(cache_path),
            "historical_reconstruction_status": state, "stats_crosscheck_status": crosscheck, "training_ready": False,
            "temporal_provenance": "HISTORICAL_HINDSIGHT_RECONSTRUCTION",
            "events_timeline_count": p["events_timeline_count"],
            "events_in_window": p["events_in_window"],
            "events_unknown_team": p["events_unknown_team"],
            "events_unknown_type": p["events_unknown_type"],
            "events_future_filtered": p["events_future_filtered"],
            "events_invalid_minute": p["events_invalid_minute"],
            "events_available_at_snapshot": False,
        }
        # Diagnostic features stay available only if minimally plausible.
        # No missing/partial timeline is converted to zeros.
        for k in FEATURES:
            fields[k] = p[k] if state == "HISTORICAL_CANDIDATE_STATS_NOT_CROSSCHECKED" else None
        output.append(fields)
    return output


def rebuild(args):
    manifest = read_rows(args.manifest)
    reconstructed = []
    official = {}
    if args.official_stats is not None:
        official = {r["match_id"]: r for r in read_rows(args.official_stats)}
    crosscheck_statuses = Counter()
    statuses = Counter()
    missing = []
    for entry in manifest:
        path = raw_path(args.outdir, entry["match_id"])
        if not path.is_file():
            missing.append(entry["match_id"])
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = reconstructed_rows(payload, entry, str(path), official.get(entry["match_id"]))
            reconstructed.extend(rows)
            crosscheck_statuses[rows[0]["stats_crosscheck_status"]] += 1
            statuses[rows[0]["historical_reconstruction_status"]] += 1
        except (ValueError, OSError, TypeError, KeyError) as e:
            statuses["CACHE_READ_OR_PARSE_ERROR"] += 1
            print("CACHE ERREUR", entry["match_id"], type(e).__name__, file=sys.stderr)
    write_csv(args.outdir / "historical_pressure_85_86_87.csv", FEATURE_COLUMNS, reconstructed)
    write_csv(args.outdir / "missing_timelines.csv", ["match_id"], ({"match_id": m} for m in missing))
    summary = {"manifest_matches": len(manifest), "cached_matches_rebuilt": sum(statuses.values()),
               "output_rows": len(reconstructed), "missing_cache": len(missing),
               "statuses_matches": dict(statuses), "stats_crosscheck_matches": dict(crosscheck_statuses),
               "training_ready_matches": 0,
               "note": "Exploratoire uniquement. Vérifier les tirs/corners vs statistiques indépendantes et les dates avant tout entraînement."}
    (args.outdir / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="Créer manifest et étiquettes de référence séparées")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--outdir", type=Path, default=Path("outputs/pressure_historical"))
    imp = sub.add_parser("import-captures", help="Importer un JSONL de captures existantes sans requête API")
    imp.add_argument("--captures", type=Path, required=True)
    imp.add_argument("--outdir", type=Path, default=Path("outputs/pressure_historical_demo"))
    fc = sub.add_parser("fetch", help="Télécharger par petits lots les timelines et conserver les JSON")
    fc.add_argument("--manifest", type=Path, required=True)
    fc.add_argument("--outdir", type=Path, default=Path("outputs/pressure_historical"))
    fc.add_argument("--env", type=Path, default=Path(".env"))
    fc.add_argument("--limit", type=int, default=5, help="Maximum de NOUVELLES requêtes (succès+échecs), défaut 5")
    fc.add_argument("--delay", type=float, default=1.0, help="Secondes entre deux requêtes")
    fc.add_argument("--timeout", type=int, default=20)
    fc.add_argument("--dry-run", action="store_true")
    fc.add_argument("--force", action="store_true", help="Réinterroger même si cache existant (coût API)")
    rb = sub.add_parser("rebuild", help="Recalculer les six features depuis le cache, sans API")
    rb.add_argument("--manifest", type=Path, required=True)
    rb.add_argument("--official-stats", type=Path, default=None,
                    help="CSV séparé de 6 comptes FT par équipe; doit provenir dune source indépendante")
    rb.add_argument("--outdir", type=Path, default=Path("outputs/pressure_historical"))
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "import-captures":
        import_captures(args)
    elif args.mode == "fetch":
        if args.limit <= 0 or args.delay < 0 or args.timeout <= 0:
            ap.error("limit>0, delay>=0 et timeout>0 requis")
        fetch(args)
    else:
        rebuild(args)


if __name__ == "__main__":
    main()
