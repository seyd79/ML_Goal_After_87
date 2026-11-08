#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_xg_fidelity_lab.py

AUDIT xG — DATASET HISTORIQUE ↔ PRE-MATCH LAB

Objectif
--------
Comprendre si `home_avg_xg` / `away_avg_xg` sont construits de façon compatible
avec le dataset historique, sans modifier le scanner de production.

Le script :
1) charge le module PRE-MATCH Fidelity LAB existant ;
2) récupère quelques matchs LIVE ;
3) récupère l'historique des équipes ;
4) enrichit les matchs historiques avec l'xG officiel via /stats ;
5) mesure combien de matchs xG sont réellement disponibles ;
6) calcule plusieurs résumés candidats :
   - moyenne de tous les xG disponibles,
   - moyenne des 10 derniers xG disponibles,
   - moyenne des 5 derniers xG disponibles,
   - médiane des xG disponibles ;
7) compare ces valeurs aux distributions historiques de
   ML_Dataset_PreMatch_Live87.csv si le CSV est présent ;
8) sauvegarde un CSV de diagnostic.

IMPORTANT
---------
- Aucun fichier de production n'est modifié.
- Aucun changement n'est appliqué au scanner.
- Le script est un outil d'audit uniquement.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

LAB_MODULE_CANDIDATES = [
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab_v2.py",
    BASE_DIR / "pre_match_feature_engineering_fidelity_lab_v1.py",
]

DATASET_CANDIDATES = [
    BASE_DIR / "ML_Dataset_PreMatch_Live87.csv",
    BASE_DIR / "data" / "ML_Dataset_PreMatch_Live87.csv",
]

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "fidelity_lab"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "xg_fidelity_audit.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "xg_fidelity_report.md"
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def now_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def resolve_lab_path() -> Path:
    for path in LAB_MODULE_CANDIDATES:
        if path.exists():
            return path

    names = "\n".join(
        f"  - {p.name}"
        for p in LAB_MODULE_CANDIDATES
    )

    raise FileNotFoundError(
        "Aucun module Fidelity LAB trouvé.\n"
        "Noms acceptés :\n"
        f"{names}"
    )


def load_lab():
    path = resolve_lab_path()

    print(
        f"Module LAB détecté : {path.name}"
    )

    spec = importlib.util.spec_from_file_location(
        "fidelity_lab_runtime",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Impossible de charger le module LAB."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def build_rate_limited_client(
    lab,
    max_retries: int = 5,
    min_interval: float = 0.25,
    base_wait: float = 2.0,
):
    class Client(
        lab.StatsAPIClient
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_request = 0.0

        def request(
            self,
            method,
            endpoint,
            params=None,
        ):
            url = (
                f"{self.base_url}/"
                f"{endpoint.lstrip('/')}"
            )

            for attempt in range(
                1,
                max_retries + 1,
            ):
                elapsed = (
                    time.monotonic()
                    - self._last_request
                )

                if elapsed < min_interval:
                    time.sleep(
                        min_interval
                        - elapsed
                    )

                try:
                    response = (
                        self.session.request(
                            method=method,
                            url=url,
                            params=params,
                            timeout=self.timeout,
                        )
                    )
                except Exception as exc:
                    if attempt >= max_retries:
                        print(
                            f"Erreur API finale : "
                            f"{exc}"
                        )
                        return None

                    wait = min(
                        base_wait
                        * (2 ** (attempt - 1)),
                        30.0,
                    )

                    print(
                        f"Erreur API temporaire "
                        f"| attente {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                self._last_request = (
                    time.monotonic()
                )

                if response.status_code == 429:
                    retry_after = (
                        response.headers.get(
                            "Retry-After"
                        )
                    )

                    try:
                        wait = float(
                            retry_after
                        )
                    except Exception:
                        wait = min(
                            base_wait
                            * (2 ** (attempt - 1)),
                            30.0,
                        )

                    print(
                        f"API 429 : "
                        f"{method} {endpoint} "
                        f"| tentative "
                        f"{attempt}/{max_retries} "
                        f"| attente {wait:.1f}s"
                    )

                    if attempt >= max_retries:
                        return None

                    time.sleep(
                        max(wait, 0.5)
                    )
                    continue

                if response.status_code >= 400:
                    print(
                        f"API "
                        f"{response.status_code} : "
                        f"{method} {endpoint}"
                    )
                    return None

                try:
                    return response.json()
                except ValueError:
                    return None

            return None

    return Client()


def match_info(
    lab,
    match: Mapping[str, Any],
) -> Dict[str, Any]:
    home = lab.extract_home_team(
        match
    )
    away = lab.extract_away_team(
        match
    )

    return {
        "match_id":
            lab.extract_match_id(match),
        "home_team_id":
            lab.extract_team_id(home),
        "away_team_id":
            lab.extract_team_id(away),
        "home_team":
            lab.extract_team_name(home)
            or "Home",
        "away_team":
            lab.extract_team_name(away)
            or "Away",
        "minute":
            lab.extract_live_minute(match),
    }


def choose_matches(
    lab,
    live_matches: Sequence[
        Mapping[str, Any]
    ],
    count: int,
) -> List[Mapping[str, Any]]:
    scored = []

    for match in live_matches:
        minute = (
            lab.extract_live_minute(
                match
            )
        )

        distance = (
            abs(87 - minute)
            if minute is not None
            else 999
        )

        scored.append(
            (
                distance,
                -(minute or -1),
                match,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return [
        item[2]
        for item in scored[:count]
    ]


def team_xg_audit(
    lab,
    client,
    team_id: str,
    team_name: str,
    history_limit: int,
    enrich_limit: int,
) -> Dict[str, Any]:

    history = lab.get_team_history(
        client,
        team_id,
        limit=history_limit,
    )

    raw_count = (
        len(history)
        if isinstance(history, list)
        else 0
    )

    enriched = (
        lab.enrich_history_with_xg(
            client,
            history,
            max_matches=enrich_limit,
        )
    )

    rows = (
        lab.team_history_rows(
            enriched,
            target_team_id=team_id,
            target_team_name=team_name,
        )
    )

    xg_rows: List[
        Tuple[Any, float]
    ] = []

    for row in rows:
        value = safe_float(
            row.get("xg_for")
        )

        if value is None:
            continue

        xg_rows.append(
            (
                row.get("date"),
                value,
            )
        )

    values = [
        value
        for _, value in xg_rows
    ]

    last10 = values[-10:]
    last5 = values[-5:]

    return {
        "history_rows":
            raw_count,
        "parsed_team_rows":
            len(rows),
        "xg_available":
            len(values),
        "xg_coverage_rate":
            (
                len(values) / len(rows)
                if rows
                else None
            ),
        "xg_mean_all":
            mean(values),
        "xg_mean_last10":
            mean(last10),
        "xg_mean_last5":
            mean(last5),
        "xg_median_all":
            median(values),
        "xg_min":
            min(values)
            if values
            else None,
        "xg_max":
            max(values)
            if values
            else None,
        "xg_values":
            "|".join(
                f"{v:.3f}"
                for v in values
            ),
    }


def historical_baseline() -> Dict[str, Any]:
    for path in DATASET_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(
            path
        )

        result: Dict[str, Any] = {
            "dataset_path": str(path),
        }

        for feature in (
            "home_avg_xg",
            "away_avg_xg",
        ):
            s = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            result[
                f"{feature}_mean"
            ] = float(s.mean())

            result[
                f"{feature}_median"
            ] = float(s.median())

            result[
                f"{feature}_q05"
            ] = float(
                s.quantile(0.05)
            )

            result[
                f"{feature}_q95"
            ] = float(
                s.quantile(0.95)
            )

            mode = s.mode(
                dropna=True
            )

            result[
                f"{feature}_mode"
            ] = (
                float(mode.iloc[0])
                if not mode.empty
                else None
            )

            mode_value = result[
                f"{feature}_mode"
            ]

            result[
                f"{feature}_mode_share"
            ] = (
                float(
                    (s == mode_value)
                    .mean()
                )
                if mode_value is not None
                else None
            )

        return result

    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Fidelity LAB des "
            "features home_avg_xg / away_avg_xg."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help=(
            "Nombre de matchs LIVE "
            "à auditer. Défaut: 3."
        ),
    )

    parser.add_argument(
        "--history-limit",
        type=int,
        default=50,
        help=(
            "Nombre de matchs historiques "
            "par équipe. Défaut: 50."
        ),
    )

    parser.add_argument(
        "--xg-enrich-limit",
        type=int,
        default=10,
        help=(
            "Nombre maximal de matchs récents "
            "à enrichir via /stats. Défaut: 10."
        ),
    )

    parser.add_argument(
        "--between-teams",
        type=float,
        default=4.0,
        help=(
            "Pause entre équipes pour limiter "
            "les 429. Défaut: 4 secondes."
        ),
    )

    args = parser.parse_args()

    lab = load_lab()

    if not clean(
        getattr(
            lab,
            "API_KEY",
            "",
        )
    ):
        print(
            "ERREUR : STATS_API_KEY "
            "absente."
        )
        return 2

    client = build_rate_limited_client(
        lab
    )

    print()
    print("=" * 100)
    print(
        "AUDIT xG FIDELITY LAB"
    )
    print("=" * 100)

    baseline = historical_baseline()

    if baseline:
        print()
        print(
            "BASELINE DATASET HISTORIQUE"
        )
        print("-" * 100)

        for feature in (
            "home_avg_xg",
            "away_avg_xg",
        ):
            print(
                f"{feature}: "
                f"mean="
                f"{baseline[f'{feature}_mean']:.4f} "
                f"| median="
                f"{baseline[f'{feature}_median']:.4f} "
                f"| q05="
                f"{baseline[f'{feature}_q05']:.4f} "
                f"| q95="
                f"{baseline[f'{feature}_q95']:.4f} "
                f"| mode="
                f"{baseline[f'{feature}_mode']:.4f} "
                f"| mode_share="
                f"{baseline[f'{feature}_mode_share']:.2%}"
            )

    live_matches = (
        lab.get_live_matches(
            client
        )
    )

    selected = choose_matches(
        lab,
        live_matches,
        max(
            1,
            args.count,
        ),
    )

    if not selected:
        print(
            "Aucun match LIVE disponible."
        )
        return 1

    rows: List[
        Dict[str, Any]
    ] = []

    team_counter = 0

    for match in selected:
        info = match_info(
            lab,
            match,
        )

        print()
        print("=" * 100)
        print(
            f"{info['home_team']} "
            f"vs "
            f"{info['away_team']} "
            f"| minute="
            f"{info['minute']}"
        )
        print("=" * 100)

        for side in (
            "home",
            "away",
        ):
            team_counter += 1

            if (
                team_counter > 1
                and args.between_teams > 0
            ):
                time.sleep(
                    args.between_teams
                )

            team_id = info[
                f"{side}_team_id"
            ]

            team_name = info[
                f"{side}_team"
            ]

            print()
            print(
                f"[{side.upper()}] "
                f"{team_name}"
            )

            result = team_xg_audit(
                lab=lab,
                client=client,
                team_id=team_id,
                team_name=team_name,
                history_limit=args.history_limit,
                enrich_limit=args.xg_enrich_limit,
            )

            print(
                f"Historique parsé : "
                f"{result['parsed_team_rows']}"
            )
            print(
                f"xG disponibles : "
                f"{result['xg_available']} "
                f"| coverage="
                f"{result['xg_coverage_rate']:.2%}"
                if result[
                    "xg_coverage_rate"
                ] is not None
                else "xG coverage : N/A"
            )
            print(
                f"mean_all="
                f"{result['xg_mean_all']} "
                f"| mean_last10="
                f"{result['xg_mean_last10']} "
                f"| mean_last5="
                f"{result['xg_mean_last5']} "
                f"| median="
                f"{result['xg_median_all']}"
            )

            rows.append({
                "captured_at_utc":
                    now_utc_iso(),
                "match_id":
                    info["match_id"],
                "minute":
                    info["minute"],
                "side":
                    side,
                "team_id":
                    team_id,
                "team_name":
                    team_name,
                **result,
            })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "captured_at_utc",
        "match_id",
        "minute",
        "side",
        "team_id",
        "team_name",
        "history_rows",
        "parsed_team_rows",
        "xg_available",
        "xg_coverage_rate",
        "xg_mean_all",
        "xg_mean_last10",
        "xg_mean_last5",
        "xg_median_all",
        "xg_min",
        "xg_max",
        "xg_values",
    ]

    file_exists = (
        OUTPUT_CSV.exists()
        and OUTPUT_CSV.stat().st_size > 0
    )

    with OUTPUT_CSV.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )

    available_values = [
        safe_float(
            row.get(
                "xg_mean_all"
            )
        )
        for row in rows
    ]

    available_values = [
        x
        for x in available_values
        if x is not None
    ]

    report_lines = [
        "# Audit xG Fidelity LAB",
        "",
        f"Équipes auditées : **{len(rows)}**",
        f"Équipes avec moyenne xG exploitable : **{len(available_values)}**",
        "",
    ]

    if baseline:
        report_lines.extend([
            "## Dataset historique",
            "",
            (
                "- home_avg_xg : "
                f"mean={baseline['home_avg_xg_mean']:.4f}, "
                f"median={baseline['home_avg_xg_median']:.4f}, "
                f"mode={baseline['home_avg_xg_mode']:.4f}, "
                f"mode_share={baseline['home_avg_xg_mode_share']:.2%}"
            ),
            (
                "- away_avg_xg : "
                f"mean={baseline['away_avg_xg_mean']:.4f}, "
                f"median={baseline['away_avg_xg_median']:.4f}, "
                f"mode={baseline['away_avg_xg_mode']:.4f}, "
                f"mode_share={baseline['away_avg_xg_mode_share']:.2%}"
            ),
            "",
        ])

    if available_values:
        report_lines.extend([
            "## Échantillon API actuel",
            "",
            f"- moyenne des `xg_mean_all` : **{mean(available_values):.4f}**",
            f"- médiane des `xg_mean_all` : **{median(available_values):.4f}**",
            "",
        ])

    report_lines.extend([
        "## Interprétation",
        "",
        "La comparaison sert à distinguer :",
        "",
        "1. une formule xG réellement incompatible ;",
        "2. une forte imputation historique ;",
        "3. une disponibilité API actuelle différente de celle du dataset historique.",
        "",
        "Aucune modification du scanner ou du builder LAB n'est appliquée automatiquement.",
    ])

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        f"CSV : {OUTPUT_CSV}"
    )
    print(
        f"Rapport : {REPORT_PATH}"
    )
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
