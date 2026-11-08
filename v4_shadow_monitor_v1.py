#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V4 SHADOW MONITOR V1
Surveillance au premier plan.
Compatible Python 3.9 / macOS.

- Lance périodiquement capture_v4_predict87_v1.py
- N'importe pas et ne modifie pas le scanner V3-C
- Interruption avec Ctrl+C
- Journal indépendant dans outputs/v4_research
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT / "capture_v4_predict87_v1.py"

OUT = ROOT / "outputs" / "v4_research"
LOG = OUT / "V4_SHADOW_MONITOR_LOG_V1.jsonl"
LOCK = OUT / "V4_SHADOW_MONITOR_V1.lock"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_log(record):
    with LOG.open(
        "a", encoding="utf-8"
    ) as f:
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )
        f.flush()


def monitor(interval, hours, max_errors):
    if not COLLECTOR.is_file():
        raise SystemExit(
            "Collecteur introuvable : "
            "capture_v4_predict87_v1.py"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    # Verrou système : empêche deux moniteurs simultanés.
    with LOCK.open("a+") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit(
                "Un autre moniteur V4 semble déjà actif."
            )

        print("=" * 72)
        print("V4 SHADOW MONITOR V1")
        print("=" * 72)
        print("Début UTC :", utc_now())
        print("Intervalle :", interval, "secondes")
        print("Durée maximale :", hours, "heures")
        print("Collecteur :", COLLECTOR.name)
        print("Journal :", LOG)
        print("Mode : RESEARCH_SHADOW")
        print("Arrêt manuel : Ctrl+C")
        print("=" * 72, flush=True)

        start = time.monotonic()
        deadline = start + hours * 3600

        cycles = 0
        errors = 0
        successful_calls = 0
        captures = 0

        try:
            while time.monotonic() < deadline:
                cycles += 1
                cycle_started = time.monotonic()

                print(
                    "\n" + "=" * 65,
                    flush=True,
                )
                print(
                    f"CYCLE {cycles} | {utc_now()}",
                    flush=True,
                )
                print("=" * 65, flush=True)

                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(COLLECTOR),
                        ],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=90,
                        check=False,
                    )

                    if result.stdout:
                        print(
                            result.stdout,
                            end="\n",
                            flush=True,
                        )

                    if result.stderr:
                        print(
                            result.stderr,
                            file=sys.stderr,
                            flush=True,
                        )

                    success = result.returncode == 0

                    # Diagnostic seulement : ne remplace pas
                    # le journal de prédiction du collecteur.
                    captured = (
                        "PRÉDICTION SHADOW ENREGISTRÉE"
                        in result.stdout
                    )

                    if success:
                        errors = 0
                        successful_calls += 1
                    else:
                        errors += 1

                    if captured:
                        captures += 1

                    write_log({
                        "timestamp_utc": utc_now(),
                        "cycle": cycles,
                        "returncode": result.returncode,
                        "prediction_recorded": captured,
                        "consecutive_errors": errors,
                    })

                except subprocess.TimeoutExpired:
                    errors += 1
                    print(
                        "ERREUR : délai maximal du "
                        "collecteur dépassé.",
                        flush=True,
                    )

                    write_log({
                        "timestamp_utc": utc_now(),
                        "cycle": cycles,
                        "error": "COLLECTOR_TIMEOUT",
                        "consecutive_errors": errors,
                    })

                except Exception as exc:
                    errors += 1
                    print(
                        "ERREUR :",
                        type(exc).__name__,
                        str(exc),
                        flush=True,
                    )

                    write_log({
                        "timestamp_utc": utc_now(),
                        "cycle": cycles,
                        "error": type(exc).__name__,
                        "details": str(exc),
                        "consecutive_errors": errors,
                    })

                if errors >= max_errors:
                    print(
                        "\nARRÊT DE SÉCURITÉ : "
                        "trop d'erreurs consécutives.",
                        flush=True,
                    )
                    break

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    break

                elapsed = time.monotonic() - cycle_started

                print(
                    "\nBILAN :",
                    f"cycles={cycles}",
                    f"captures={captures}",
                    f"erreurs_consécutives={errors}",
                    flush=True,
                )

                # Attendre au minimum l'intervalle demandé
                # entre les débuts de cycles.
                sleep_for = max(
                    0,
                    interval - elapsed,
                )

                # Éviter les interrogations rapprochées
                # si une requête a pris trop longtemps.
                if sleep_for == 0:
                    sleep_for = 10

                sleep_for = min(sleep_for, remaining)

                print(
                    f"Prochain contrôle dans "
                    f"{sleep_for:.0f} secondes.",
                    flush=True,
                )

                time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(
                "\nArrêt demandé par l'utilisateur.",
                flush=True,
            )

        finally:
            print("\n" + "=" * 72)
            print("BILAN FINAL V4 SHADOW")
            print("=" * 72)
            print("Cycles :", cycles)
            print("Cycles sans erreur :", successful_calls)
            print("Prédictions enregistrées :", captures)
            print("Fin UTC :", utc_now())
            print("Journal :", LOG)
            print("Modèles inchangés.")
            print("Scanner V3-C inchangé.")

            write_log({
                "timestamp_utc": utc_now(),
                "event": "MONITOR_STOPPED",
                "cycles": cycles,
                "successful_calls": successful_calls,
                "predictions_recorded": captures,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interval-sec",
        type=int,
        default=60,
        help="Intervalle entre cycles (minimum 30 s)",
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=3,
        help="Durée maximale de surveillance",
    )

    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Erreurs consécutives avant arrêt",
    )

    args = parser.parse_args()

    if args.interval_sec < 30:
        parser.error(
            "Intervalle minimum : 30 secondes"
        )

    if not 0 < args.hours <= 24:
        parser.error(
            "Durée autorisée : 0 à 24 heures"
        )

    if args.max_errors < 1:
        parser.error(
            "max-errors doit être positif"
        )

    monitor(
        interval=args.interval_sec,
        hours=args.hours,
        max_errors=args.max_errors,
    )
