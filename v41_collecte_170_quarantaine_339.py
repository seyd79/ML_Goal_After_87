#!/usr/bin/env python3
"""V4.1: targeted, restartable, LABEL-FREE technical quarantine for 170 remaining matches.

Requires project-local, previously validated V4.1 adapters and collectors.
Never modifies V3-C, the seven protected lists, or TEST_FUTUR.
WARNING: collector.collect_one_match may compute target internally; target is never persisted.
This is NOT certified training data; other protected partitions remain unverified.
"""

import argparse
import csv
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch
from v41_prematch_safe_bridge_318 import ResearchPrematchProxy, ODDS_FIELDS
from v41_targeted_collector_328 import load_live87, make_stub
import v41_batch_technical_336 as previous

BASE = Path("outputs/v4_research")
ROOT = Path("outputs")
DEST = BASE / "v41_technical_quarantine_339"
FROZEN = DEST / "SELECTION_170_IDS_ONLY.csv"
SCHEMA_FIELDS = ("match_id", "competition_id", "season_id", "kickoff_utc")
DONE_FIELDS = (*SCHEMA_FIELDS, "technical_status", "available49", "training_authorized", "protected_certified")
PARTIAL_FIELDS = (*SCHEMA_FIELDS, "technical_status", "error", "training_authorized", "protected_certified")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one(path):
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"STOP : fichier absent ou invalide : {path}")
    return rows[0]


def write_csv(path, records, fieldnames):
    # Never write arbitrary collector fields, especially goal/score/label.
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_missing_odds(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "na"}:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def checked_past_matches(plan_ids):
    old_folders = [BASE / "v41_technical_quarantine_331"]
    old_folders += [
        BASE / "v41_technical_quarantine_332" / f"pilot_{i:02d}"
        for i in range(2, 11)
    ]
    old_folders += [
        BASE / "v41_technical_quarantine_336" / f"match_{i:02d}"
        for i in range(1, 11)
    ]
    old_folders += [
        BASE / "v41_technical_quarantine_337" / f"match_{i:02d}"
        for i in range(1, 21)
    ]
    seen = set()
    partial_count = 0
    done_count = 0
    for folder in old_folders:
        status = read_one(folder / "TECHNICAL_STATUS_NO_LABEL.csv")
        match_id = status.get("match_id", "")
        if match_id not in plan_ids or match_id in seen:
            raise RuntimeError(f"STOP : ID incoherent/duplique dans {folder}")
        seen.add(match_id)
        kind = status.get("technical_status")
        feature_path = folder / "TECHNICAL_FEATURES49_NO_LABEL.csv"
        if kind == "done":
            features = read_one(feature_path)
            if features.get("match_id") != match_id:
                raise RuntimeError(f"STOP : ID des features incoherent dans {folder}")
            done_count += 1
        elif kind == "partial" and status.get("error") == "timeline_empty":
            if feature_path.exists():
                raise RuntimeError(f"STOP : features presentes pour un partial dans {folder}")
            partial_count += 1
        else:
            raise RuntimeError(f"STOP : statut precedent non reconnu : {folder} / {kind}")
    if len(seen) != 40 or done_count != 39 or partial_count != 1:
        raise RuntimeError(
            "STOP : historique attendu 40 tentatives = 39 done + 1 partial; "
            f"trouve {len(seen)} / {done_count} / {partial_count}"
        )
    return seen


def distribute_dates(rows):
    # Visit each available month before taking a second match in that month.
    by_comp = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_comp[row["competition_id"]][row["kickoff_utc"][:7]].append(row)
    each_comp = {}
    for comp, by_month in by_comp.items():
        buckets = {
            month: deque(sorted(matches, key=lambda r: (r["kickoff_utc"], r["match_id"])))
            for month, matches in by_month.items()
        }
        order = []
        while any(buckets.values()):
            for month in sorted(buckets):
                if buckets[month]:
                    order.append(buckets[month].popleft())
        each_comp[comp] = deque(order)
    ordered = []
    while any(each_comp.values()):
        for comp in sorted(each_comp):
            if each_comp[comp]:
                ordered.append(each_comp[comp].popleft())
    return ordered


def prepare_selection():
    plan = previous.read_manifest(previous.PLAN)
    pilot = previous.read_manifest(previous.PILOT)
    if len(plan) != 210 or len(pilot) != 10:
        raise RuntimeError("STOP : plan210/pilot10 absents ou comptes modifies")
    ids = {r.get("match_id") for r in plan}
    if len(ids) != 210 or None in ids:
        raise RuntimeError("STOP : identifiants de plan invalides")
    if {r.get("match_id") for r in pilot} - ids:
        raise RuntimeError("STOP : pilot10 absent du plan210")
    if any(
        r.get("collection_authorized") != "NO"
        or r.get("protected_clearance") != "SEVEN_SOURCES_ID_ONLY_PASS"
        for r in plan
    ):
        raise RuntimeError("STOP : autorisation ou clearance du plan modifiee")

    # ONLY the seven specifically authorized known protected ID sources.
    overlaps = set()
    for source in previous.PROTECTED:
        overlaps.update(ids & previous.ids_only(source))
    if overlaps:
        raise RuntimeError(f"STOP : {len(overlaps)} collision(s) protegees connues")

    old_ids = checked_past_matches(ids)
    new_rows = []
    for record in plan:
        if record["match_id"] in old_ids:
            continue
        new_rows.append({name: record[name].strip() for name in SCHEMA_FIELDS})
    if len(new_rows) != 170:
        raise RuntimeError(f"STOP : 170 candidats attendus, trouve {len(new_rows)}")
    if any(not all(row.values()) for row in new_rows):
        raise RuntimeError("STOP : metadonnees du plan incompletes")
    if len({r["match_id"] for r in new_rows}) != 170:
        raise RuntimeError("STOP : doublons dans les 170 matchs")
    return distribute_dates(new_rows)


def ensure_frozen(selected, create=False):
    if FROZEN.exists():
        existing = read_csv(FROZEN)
        if existing != selected:
            raise RuntimeError("STOP : selection figee differente de la selection actuelle")
    elif create:
        DEST.mkdir(parents=True, exist_ok=True)
        write_csv(FROZEN, selected, SCHEMA_FIELDS)
    elif DEST.exists():
        raise RuntimeError("STOP : dossier de sortie existant sans manifeste fige")


def match_folder(i):
    return DEST / f"match_{i:03d}"


def verified_saved(i, expected, features49):
    folder = match_folder(i)
    if not folder.exists():
        return None
    if not folder.is_dir():
        raise RuntimeError(f"STOP : dossier de match invalide : {folder}")
    status = read_one(folder / "TECHNICAL_STATUS_NO_LABEL.csv")
    if any(status.get(key) != expected[key] for key in SCHEMA_FIELDS):
        raise RuntimeError(f"STOP : checkpoint ne correspond pas au match {i}")
    if status.get("training_authorized") != "NO" or status.get("protected_certified") != "NO":
        raise RuntimeError(f"STOP : statut de quarantaine incoherent : {folder}")
    feature_path = folder / "TECHNICAL_FEATURES49_NO_LABEL.csv"
    kind = status.get("technical_status")
    if kind == "done":
        row = read_one(feature_path)
        if set(row) != set(SCHEMA_FIELDS) | set(features49):
            raise RuntimeError(f"STOP : colonnes du checkpoint invalides : {folder}")
        if any(row.get(key) != expected[key] for key in SCHEMA_FIELDS):
            raise RuntimeError(f"STOP : metadata features invalides : {folder}")
        n = sum(finite(row.get(name)) for name in features49)
        if str(n) != str(status.get("available49")):
            raise RuntimeError(f"STOP : couverture checkpoint invalide : {folder}")
        return ("done", n)
    if kind == "partial" and status.get("error") == "timeline_empty":
        if feature_path.exists():
            raise RuntimeError(f"STOP : features d'un partial presentes : {folder}")
        return ("partial", 0)
    raise RuntimeError(f"STOP : checkpoint non reconnu : {folder}, {kind}")


def save_checkpoint(i, expected, diagnostic, safe_features=None):
    if match_folder(i).exists():
        raise RuntimeError("STOP : refuse d'ecraser un checkpoint existant")
    # Stage a complete directory then publish it with one same-volume rename.
    staging = Path(tempfile.mkdtemp(prefix=".incomplet_", dir=DEST))
    try:
        if safe_features is not None:
            write_csv(
                staging / "TECHNICAL_FEATURES49_NO_LABEL.csv",
                [safe_features],
                tuple(safe_features),
            )
        write_csv(
            staging / "TECHNICAL_STATUS_NO_LABEL.csv",
            [diagnostic],
            tuple(diagnostic),
        )
        staging.rename(match_folder(i))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def check_api_detail(lab, detail, expected):
    utc = prematch.dt_utc_naive(lab.extract_match_date(detail))
    if (
        str(lab.extract_match_id(detail)) != expected["match_id"]
        or str(lab.extract_competition_id(detail)) != expected["competition_id"]
        or str(lab.extract_season_id(detail)) != expected["season_id"]
        or utc is None
        or utc.date().isoformat() != expected["kickoff_utc"][:10]
    ):
        raise RuntimeError(f"STOP : metadonnees API incorrectes pour {expected['match_id']}")


def main():
    parser = argparse.ArgumentParser(description="Collecte V4.1 technique, 170 matchs en quarantaine, reprenable")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="Affiche la selection, aucun appel API, aucune ecriture")
    action.add_argument("--collect", action="store_true", help="Lance les appels API et checkpoints sans labels")
    action.add_argument("--status", action="store_true", help="Bilan local des 170 sans appels API")
    parser.add_argument("--limit", type=int, default=20, help="Max nouveaux matchs par execution (defaut 20; 170 pour tous)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 170:
        parser.error("--limit doit etre entre 1 et 170")
    if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
        raise RuntimeError("STOP : V41_AUTHORISE_ID_ONLY=OUI manquant")

    print("=" * 74)
    print("ETAPE 339 - V4.1 - 170 MATCHS EN QUARANTAINE TECHNIQUE")
    print("=" * 74)
    selected = prepare_selection()
    if args.collect and os.environ.get("V41_AUTORISE_QUARANTAINE_TECHNIQUE") != "OUI":
        raise RuntimeError("STOP : autorisation explicite de quarantaine technique manquante")
    ensure_frozen(selected, create=args.collect)
    print("Anciennes tentatives preservees : 40 (39 done, 1 partial)")
    print("Nouveaux IDs exclusifs :", len(selected))
    print("Repartition championnats :", dict(Counter(r["competition_id"] for r in selected)))
    print("Periodes :", min(r["kickoff_utc"] for r in selected), "->", max(r["kickoff_utc"] for r in selected))
    print("Partitions supplementaires non certifiees ; ENTRAINEMENT INTERDIT")
    print("TEST_FUTUR non ouvert par ce script ; V3-C non modifie")

    if args.dry_run:
        print("\nPremiers 20 matchs de la selection :")
        for i, row in enumerate(selected[:20], 1):
            print(f"  {i:03d} | {row['match_id']} | {row['competition_id']} | {row['kickoff_utc']}")
        print("\nDRY RUN OK : aucun appel API ; aucune ecriture")
        return

    core = collector.load_core_features(None)
    odds3 = {"odds_home", "odds_draw", "odds_away"}
    features49 = [f for f in core if f not in odds3]
    if len(core) != 52 or len(features49) != 49 or not odds3.issubset(set(core)):
        raise RuntimeError("STOP : schema CORE52/49 inattendu")

    saved = {}
    for index, row in enumerate(selected, 1):
        old = verified_saved(index, row, features49)
        if old is not None:
            saved[index] = old
    done = sum(kind == "done" for kind, _ in saved.values())
    partial = sum(kind == "partial" for kind, _ in saved.values())
    print(f"Deja enregistres : {len(saved)}/170 (done={done}, partial={partial})")
    if args.status:
        print(f"Restants sans tentative : {170 - len(saved)}")
        print("Completude des done du lot339 :", sum(n for kind, n in saved.values() if kind == "done"), "/", done * 49)
        return

    pending = [(index, row) for index, row in enumerate(selected, 1) if index not in saved]
    pending = pending[:args.limit]
    if not pending:
        print("Aucun nouveau match a traiter. LOT339 DEJA TERMINE.")
        return
    print(f"Nouvelles tentatives prevues cette execution : {len(pending)}")

    # Only now touch the actual match API.
    lab = prematch.load_lab()
    prematch.lab = lab
    client = prematch.build_rate_limited_client(lab)
    live87 = load_live87()
    proxy = ResearchPrematchProxy(prematch, lab)
    original_detail = prematch.get_match_detail

    for index, expected in pending:
        match_id = expected["match_id"]
        print("\n" + "-" * 74)
        print(f"[{index:03d}/170] {match_id} | {expected['competition_id']} | {expected['kickoff_utc']}", flush=True)

        def checked_detail(lab_arg, client_arg, requested_id):
            if requested_id != match_id:
                raise RuntimeError("STOP : ID API hors selection")
            detail = original_detail(lab_arg, client_arg, requested_id)
            if detail:
                check_api_detail(lab, detail, expected)
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
        if status.get("match_id") != match_id:
            raise RuntimeError("STOP : ID status incorrect")
        kind = status.get("status")
        error = status.get("error")
        if kind == "partial" and error == "timeline_empty":
            if row is not None and row.get("match_id") != match_id:
                raise RuntimeError("STOP : ID ligne partial incorrect")
            diagnostic = {
                **expected,
                "technical_status": "partial",
                "error": "timeline_empty",
                "training_authorized": "NO",
                "protected_certified": "NO",
            }
            save_checkpoint(index, expected, diagnostic)
            print("PARTIAL : timeline_empty. Aucune feature ni aucun label sauvegarde.", flush=True)
            continue
        if kind != "done" or row is None or row.get("match_id") != match_id:
            print("STOP : reconstruction non terminee", flush=True)
            print("status =", repr(kind), "error =", repr(error), "row_present =", row is not None, flush=True)
            raise RuntimeError(f"STOP : echec non gere pour {match_id}; aucune ligne sauvegardee")
        for odd in ODDS_FIELDS:
            if not is_missing_odds(row.get(odd)):
                raise RuntimeError(f"STOP : cote non neutralisee ({odd}) pour {match_id}")

        safe_features = {**expected}
        safe_features.update({name: row.get(name) for name in features49})
        n = sum(finite(safe_features.get(name)) for name in features49)
        diagnostic = {
            **expected,
            "technical_status": "done",
            "available49": n,
            "training_authorized": "NO",
            "protected_certified": "NO",
        }
        save_checkpoint(index, expected, diagnostic, safe_features)
        print(f"DONE : CORE49 = {n}/49 ; checkpoint enregistre ; aucun label", flush=True)

    # No unverified totals: re-read saved checkpoints.
    results = [verified_saved(i, r, features49) for i, r in enumerate(selected, 1)]
    done = sum(x is not None and x[0] == "done" for x in results)
    partial = sum(x is not None and x[0] == "partial" for x in results)
    values = sum(x[1] for x in results if x is not None and x[0] == "done")
    print("\n=== BILAN LOT339 ===")
    print(f"Tentatives enregistrees : {done + partial}/170")
    print(f"DONE : {done} | PARTIAL timeline_empty : {partial} | non tentes : {170 - done - partial}")
    print(f"CORE49 disponibles sur DONE : {values}/{done * 49}")
    print("Chemin :", DEST)
    print("SEULEMENT QUARANTAINE TECHNIQUE ; entrainement NON autorise")
    print("TEST_FUTUR non consulte ; V3-C inchange")
    print("ETAPE 339 TERMINEE POUR CETTE EXECUTION")


if __name__ == "__main__":
    main()
