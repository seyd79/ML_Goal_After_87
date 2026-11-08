#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_historical_dataset_v2_hierarchical_football_only.py

LANCEUR FOOTBALL ONLY — DATASET HISTORIQUE V2
=============================================

OBJECTIF
--------
Continuer la collecte historique dans le MÊME dataset V2, mais en filtrant
automatiquement les compétitions/scopes/matchs qui ne sont pas du football.

Ce fichier est un lanceur séparé et sûr.

Il importe :

    collect_historical_dataset_v2_hierarchical.py

puis applique trois filtres avant la collecte :

    1) filtre compétition
    2) filtre scope compétition/saison
    3) filtre match

POURQUOI CETTE VERSION
----------------------
La version précédente filtrait bien les matchs dont le nom contenait clairement
"Håndbold", "Handbold", etc.

Mais dans ton test, deux matchs de handball sont quand même passés :

    HC Midtjylland vs Rækker Mølle Kfum
    Elitesport Vendsyssel vs Grindsted GIF

Ils appartenaient à la même compétition/saison où 12/14 matchs avaient déjà été
identifiés comme handball.

Cette version corrige donc le dernier point :

    Si un scope compétition/saison contient une majorité de matchs clairement
    non-football, on ignore le scope complet.

Cela évite de laisser passer des équipes de handball dont le nom ne contient pas
explicitement "Håndbold".

PRIORITÉ COMPÉTITIONS CORE_READY
--------------------------------
Cette version peut aussi sélectionner automatiquement les meilleures
compétitions à partir de :

    outputs/historical_dataset_v2/audit/audit_competition_quality_v2.csv

Objectif :
    atteindre plus vite 3000+ lignes CORE_READY.

Le fichier d'audit est lu au démarrage, puis les compétitions sont triées par :

    1) core_complete_rate le plus élevé
    2) core_complete le plus élevé
    3) labeled / target_resolved le plus élevé
    4) matches le plus élevé

Par défaut, ce mode est activé si le fichier d'audit existe.

Pour le désactiver :

    --priority-competitions-disable

Pour baisser le seuil à 80% :

    --min-core-complete-rate 0.80


SÉCURITÉ
--------
- Ne modifie PAS le scanner de production.
- Ne modifie PAS le modèle.
- Ne modifie PAS le fichier `collect_historical_dataset_v2_hierarchical.py`.
- Continue d'écrire dans les mêmes fichiers :
    outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_RAW_v2.csv
    outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv
    outputs/historical_dataset_v2/historical_matches_index_v2.csv

UTILISATION
-----------
Test court :

    caffeinate -dimsu python3 collect_historical_dataset_v2_hierarchical_football_only.py \
        --resume \
        --limit-new 50 \
        --max-competitions 10 \
        --max-seasons-per-competition 5 \
        --max-pages-per-season 10 \
        --print-filtered

Collecte plus large :

    caffeinate -dimsu python3 collect_historical_dataset_v2_hierarchical_football_only.py \
        --resume \
        --limit-new 1000 \
        --max-competitions 50 \
        --max-seasons-per-competition 6 \
        --max-pages-per-season 25

Ciblage manuel :

    caffeinate -dimsu python3 collect_historical_dataset_v2_hierarchical_football_only.py \
        --resume \
        --competition-ids comp_0406 comp_5840 comp_9799 \
        --limit-new 1000

OPTIONS AJOUTÉES PAR CE LANCEUR
--------------------------------
    --football-filter-disable
        Désactive le filtre football-only.

    --strict-football-metadata
        Accepte une compétition uniquement si ses métadonnées indiquent clairement
        football/soccer. À utiliser seulement si l'API fournit un champ sport fiable.

    --print-filtered
        Affiche les compétitions/matchs/scopes filtrés.

    --non-football-scope-skip-rate FLOAT
        Si une proportion >= FLOAT des matchs d'un scope est clairement
        non-football, le scope complet est ignoré.
        Défaut : 0.50.

    --non-football-scope-min-matches INT
        Nombre minimum de matchs dans un scope avant d'appliquer la règle
        de majorité non-football.
        Défaut : 5.

IMPORTANT
---------
Place ce fichier dans le même dossier que :

    collect_historical_dataset_v2_hierarchical.py
    collect_historical_dataset_v2.py
    rebuild_clean_prematch_v2.py
    audit_historical_live87_reconstructability.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent

HIERARCHICAL_COLLECTOR_PATH = (
    BASE_DIR
    / "collect_historical_dataset_v2_hierarchical.py"
)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass

    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "á": "a",
        "à": "a",
        "â": "a",
        "ä": "a",
        "ã": "a",
        "å": "a",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "ö": "o",
        "õ": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "ñ": "n",
        "ø": "o",
        "Ø": "o",
        "æ": "ae",
        "œ": "oe",
        "ß": "ss",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def stringify_item(item: Mapping[str, Any]) -> str:
    try:
        return normalize_text(
            json.dumps(
                dict(item),
                ensure_ascii=False,
                default=str,
            )
        )
    except Exception:
        return normalize_text(str(item))


def get_nested_values(
    item: Mapping[str, Any],
    keys: Sequence[str],
) -> List[str]:
    values: List[str] = []

    normalized_keys = {
        normalize_text(key).replace(" ", "_")
        for key in keys
    }

    def walk(obj: Any) -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                normalized_key = normalize_text(key).replace(" ", "_")

                if normalized_key in normalized_keys:
                    values.append(clean_text(value))

                walk(value)

        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(item)

    return values


# =============================================================================
# FOOTBALL FILTERS
# =============================================================================

FOOTBALL_POSITIVE_HINTS = {
    "football",
    "soccer",
    "futbol",
    "futebol",
    "fussball",
    "fußball",
    "calcio",
    "foot",
}

NON_FOOTBALL_STRONG_HINTS = {
    # Handball
    "handball",
    "handbold",
    "haandbold",
    "håndbold",
    "handebol",
    "handboll",
    "hbold",

    # Basketball
    "basketball",
    "basket",
    "euroleague",

    # Hockey
    "hockey",
    "ice hockey",

    # Volleyball
    "volleyball",
    "volley",

    # Tennis
    "tennis",

    # Baseball
    "baseball",

    # American football / rugby
    "american football",
    "rugby",

    # Cricket
    "cricket",

    # Other
    "water polo",
    "futsal",
    "esports",
    "e sports",
}

# Éviter de filtrer "nfl", "nba", "mlb" seulement sur payload complet :
# certaines compétitions football peuvent contenir de courts acronymes ambigus
# dans des URLs/IDs. Ces hints sont donc utilisés seulement dans les champs sport.
NON_FOOTBALL_SPORT_FIELD_HINTS = NON_FOOTBALL_STRONG_HINTS | {
    "nfl",
    "nba",
    "wnba",
    "nhl",
    "mlb",
    "atp",
    "wta",
}

NON_FOOTBALL_TEAM_HINTS = {
    "handbold",
    "haandbold",
    "håndbold",
    "handball",
    "basketball",
    "volleyball",
    "hockey",
    "rugby",
    "cricket",
}

# Patterns plus fins pour les clubs de handball qui n'ont pas toujours
# "handbold" dans le nom.
NON_FOOTBALL_TEAM_REGEXES = [
    r"\bhc\b",                       # HC Midtjylland
    r"\belitehandbold\b",
    r"\belitehaandbold\b",
    r"\belitehåndbold\b",
    r"\brakker molle\b",             # Rækker Mølle
    r"\braekker molle\b",
    r"\brækker mølle\b",
    r"\blemvig thyboron\b",          # Lemvig-Thyborøn
    r"\blemvig thyborøn\b",
    r"\btm tonder\b",                # TM Tønder
    r"\btm tønder\b",
    r"\bnorddjurs haandbold\b",
    r"\bnorddjurs handbold\b",
    r"\bnorddjurs håndbold\b",
    r"\bkoge handbold\b",
    r"\bkøge handbold\b",
    r"\bhøj elite\b",
    r"\bhoj elite\b",
    r"\bodder handbold\b",
    r"\bodder håndbold\b",
    r"\bstoholm handbold\b",
    r"\bstoholm håndbold\b",
]


def item_sport_text(item: Mapping[str, Any]) -> str:
    """
    Essaie d'extraire uniquement les champs qui ressemblent à des métadonnées de sport.
    """

    direct_keys = [
        "sport",
        "sport_name",
        "sport_slug",
        "sport_type",
        "category",
        "category_name",
        "type",
    ]

    values: List[str] = []

    for key in direct_keys:
        value = item.get(key)

        if isinstance(value, Mapping):
            values.extend(
                clean_text(value.get(child))
                for child in ("name", "slug", "type", "id")
                if value.get(child) is not None
            )

        elif value is not None:
            values.append(clean_text(value))

    nested_keys = [
        "sport",
        "sport_name",
        "sport_slug",
        "sport_type",
        "category",
        "category_name",
    ]

    values.extend(
        get_nested_values(
            item,
            nested_keys,
        )
    )

    return normalize_text(" ".join(values))


def item_name_text(item: Mapping[str, Any]) -> str:
    names = []

    for key in (
        "name",
        "title",
        "competition_name",
        "league_name",
        "season_name",
        "short_name",
        "slug",
    ):
        value = item.get(key)

        if value is not None:
            names.append(clean_text(value))

    return normalize_text(
        " ".join(names)
    )


def has_any_hint(text: str, hints: Iterable[str]) -> bool:
    text = normalize_text(text)

    return any(
        normalize_text(hint) in text
        for hint in hints
    )


def regex_any(text: str, patterns: Sequence[str]) -> bool:
    normalized = normalize_text(text)

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def is_clearly_non_football_competition(
    competition: Mapping[str, Any],
) -> bool:
    sport_text = item_sport_text(
        competition
    )

    name_text = item_name_text(
        competition
    )

    all_text = stringify_item(
        competition
    )

    # Si le champ sport dit clairement handball/basket/etc.
    if sport_text and has_any_hint(
        sport_text,
        NON_FOOTBALL_SPORT_FIELD_HINTS,
    ):
        return True

    # Si le nom de compétition contient un sport non-football.
    if name_text and has_any_hint(
        name_text,
        NON_FOOTBALL_STRONG_HINTS,
    ):
        return True

    # Fallback conservateur.
    if has_any_hint(
        all_text,
        {
            "handball",
            "handbold",
            "haandbold",
            "håndbold",
            "basketball",
            "volleyball",
            "ice hockey",
            "rugby",
            "cricket",
        },
    ):
        return True

    return False


def is_clearly_football_competition(
    competition: Mapping[str, Any],
) -> bool:
    sport_text = item_sport_text(
        competition
    )

    if sport_text:
        if has_any_hint(
            sport_text,
            FOOTBALL_POSITIVE_HINTS,
        ) and not has_any_hint(
            sport_text,
            NON_FOOTBALL_SPORT_FIELD_HINTS,
        ):
            return True

        if has_any_hint(
            sport_text,
            NON_FOOTBALL_SPORT_FIELD_HINTS,
        ):
            return False

    # Beaucoup de compétitions football n'ont pas "football" dans leur nom.
    # On accepte donc par défaut si rien n'indique fortement un autre sport.
    if not is_clearly_non_football_competition(
        competition
    ):
        return True

    return False


def extract_team_names_from_match(
    match: Mapping[str, Any],
) -> Tuple[str, str]:
    def side_name(side: str) -> str:
        team = match.get(f"{side}_team")

        if isinstance(team, Mapping):
            for key in ("name", "team_name", "short_name"):
                value = team.get(key)

                if value is not None:
                    return clean_text(value)

        teams = match.get("teams")

        if isinstance(teams, Mapping):
            team = teams.get(side)

            if isinstance(team, Mapping):
                for key in ("name", "team_name", "short_name"):
                    value = team.get(key)

                    if value is not None:
                        return clean_text(value)

        for key in (
            f"{side}_team_name",
            f"{side}_name",
        ):
            value = match.get(key)

            if value is not None:
                return clean_text(value)

        return ""

    return (
        side_name("home"),
        side_name("away"),
    )


def is_clearly_non_football_match(
    match: Mapping[str, Any],
) -> bool:
    sport_text = item_sport_text(
        match
    )

    if sport_text and has_any_hint(
        sport_text,
        NON_FOOTBALL_SPORT_FIELD_HINTS,
    ):
        return True

    home, away = extract_team_names_from_match(
        match
    )

    team_text = normalize_text(
        f"{home} {away}"
    )

    if has_any_hint(
        team_text,
        NON_FOOTBALL_TEAM_HINTS,
    ):
        return True

    if regex_any(
        team_text,
        NON_FOOTBALL_TEAM_REGEXES,
    ):
        return True

    # Protection additionnelle sur le payload complet, mais pas trop agressive.
    all_text = stringify_item(
        match
    )

    if has_any_hint(
        all_text,
        {
            "handbold",
            "haandbold",
            "håndbold",
            "handball",
        },
    ):
        return True

    return False


def filter_competitions(
    competitions: Sequence[Mapping[str, Any]],
    strict_metadata: bool,
    print_filtered: bool,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    removed = 0

    for competition in competitions:
        competition_dict = dict(competition)

        competition_id = clean_text(
            competition_dict.get("id")
            or competition_dict.get("competition_id")
        )

        competition_name = item_name_text(
            competition_dict
        )

        sport_text = item_sport_text(
            competition_dict
        )

        remove = False
        reason = ""

        if is_clearly_non_football_competition(
            competition_dict
        ):
            remove = True
            reason = "non_football_hint"

        elif strict_metadata and not has_any_hint(
            sport_text,
            FOOTBALL_POSITIVE_HINTS,
        ):
            remove = True
            reason = "strict_metadata_no_football_hint"

        if remove:
            removed += 1

            if print_filtered:
                print(
                    f"Filtre compétition : SKIP {competition_id} "
                    f"{competition_name or ''} "
                    f"reason={reason} sport={sport_text or '-'}"
                )

            continue

        kept.append(
            competition_dict
        )

    print(
        f"Filtre football compétitions : "
        f"{len(kept)} gardées / {len(competitions)} reçues "
        f"({removed} filtrées)"
    )

    return kept


def filter_matches_with_scope_guard(
    matches: Sequence[Mapping[str, Any]],
    print_filtered: bool,
    scope_skip_rate: float,
    scope_min_matches: int,
    competition_id: Optional[str],
    season_id: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Filtre match + garde de scope.

    Règle importante :
    si un scope contient majoritairement des matchs clairement non-football,
    on ignore tout le scope.

    Cela corrige les cas où certains clubs de handball ne contiennent pas
    explicitement "Handbold" dans leur nom.
    """

    total = len(matches)

    if total == 0:
        return []

    non_football_flags = [
        is_clearly_non_football_match(dict(match))
        for match in matches
    ]

    removed = int(sum(non_football_flags))
    rate = removed / total if total else 0.0

    if (
        total >= scope_min_matches
        and rate >= scope_skip_rate
    ):
        print(
            "Filtre football scope : SKIP scope complet "
            f"competition={competition_id or '-'} "
            f"season={season_id or 'ALL'} "
            f"non_football={removed}/{total} ({rate:.1%})"
        )

        if print_filtered:
            for match in matches:
                home, away = extract_team_names_from_match(
                    dict(match)
                )

                match_id = clean_text(
                    dict(match).get("id")
                    or dict(match).get("match_id")
                    or dict(match).get("event_id")
                )

                print(
                    f"  SKIP scope match {match_id} "
                    f"{home} vs {away}"
                )

        return []

    kept: List[Dict[str, Any]] = []

    for match, is_bad in zip(matches, non_football_flags):
        match_dict = dict(match)

        if is_bad:
            if print_filtered:
                home, away = extract_team_names_from_match(
                    match_dict
                )

                match_id = clean_text(
                    match_dict.get("id")
                    or match_dict.get("match_id")
                    or match_dict.get("event_id")
                )

                print(
                    f"Filtre match : SKIP {match_id} "
                    f"{home} vs {away}"
                )

            continue

        kept.append(match_dict)

    if removed:
        print(
            f"Filtre football matchs : "
            f"{len(kept)} gardés / {total} reçus "
            f"({removed} filtrés)"
        )

    return kept



# =============================================================================
# PRIORITY COMPETITIONS FROM AUDIT
# =============================================================================

def csv_number(value: Any, default: float = 0.0) -> float:
    """
    Convertit une valeur CSV en float.

    Supporte :
        0.95
        95
        95%
        **0.95**
        "0,95"
    """

    text = clean_text(value)

    if not text:
        return default

    text = (
        text
        .replace("*", "")
        .replace("%", "")
        .replace(",", ".")
        .strip()
    )

    try:
        number = float(text)
    except Exception:
        return default

    if not math.isfinite(number):
        return default

    return number


def csv_rate(value: Any, default: float = 0.0) -> float:
    number = csv_number(
        value,
        default,
    )

    # Si le CSV contient 95 au lieu de 0.95.
    if number > 1.0 and number <= 100.0:
        return number / 100.0

    return number


def csv_int(value: Any, default: int = 0) -> int:
    return int(
        csv_number(
            value,
            float(default),
        )
    )


def default_priority_file() -> Path:
    return (
        BASE_DIR
        / "outputs"
        / "historical_dataset_v2"
        / "audit"
        / "audit_competition_quality_v2.csv"
    )


def resolve_path_maybe(
    path_value: Optional[str],
    default_path: Path,
) -> Path:
    if path_value:
        path = Path(path_value).expanduser()

        if not path.is_absolute():
            path = BASE_DIR / path

        return path.resolve()

    return default_path.resolve()


def normalize_csv_key(key: Any) -> str:
    return (
        clean_text(key)
        .replace("\\_", "_")
        .replace("*", "")
        .replace(",", "")
        .replace(" ", "_")
        .strip()
        .lower()
    )


def normalize_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}

    for key, value in row.items():
        cleaned[normalize_csv_key(key)] = value

    return cleaned


def first_existing_value(
    row: Mapping[str, Any],
    candidates: Sequence[str],
    default: Any = "",
) -> Any:
    for candidate in candidates:
        key = normalize_csv_key(candidate)

        if key in row:
            value = row.get(key)

            if clean_text(value):
                return value

    return default


def load_priority_competitions_from_audit(
    audit_csv_path: Path,
    min_core_complete_rate: float,
    min_core_complete: int,
    min_matches: int,
    top_n: int,
    print_filtered: bool,
) -> List[str]:
    """
    Charge les compétitions les plus rentables depuis l'audit compétition.

    Colonnes attendues, selon les versions possibles :
        competition_id
        matches
        core_complete
        core_complete_rate
        labeled / target_resolved

    Le code est volontairement tolérant sur les noms de colonnes.
    """

    if not audit_csv_path.exists():
        print()
        print(
            "Priorité compétitions CORE_READY : fichier d'audit introuvable."
        )
        print(f"Chemin attendu : {audit_csv_path}")
        print("Le collecteur utilisera donc le parcours normal.")
        return []

    rows: List[Dict[str, Any]] = []

    try:
        with audit_csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as fh:
            reader = csv.DictReader(fh)

            for raw_row in reader:
                row = normalize_csv_row(raw_row)

                competition_id = clean_text(
                    first_existing_value(
                        row,
                        [
                            "competition_id",
                            "competition",
                            "comp_id",
                            "league_id",
                            "competition",
                        ],
                    )
                ).replace("\\_", "_")

                if not competition_id:
                    continue

                matches = csv_int(
                    first_existing_value(
                        row,
                        [
                            "matches",
                            "raw_rows",
                            "rows",
                            "total",
                            "total_rows",
                        ],
                        0,
                    ),
                    0,
                )

                core_complete = csv_int(
                    first_existing_value(
                        row,
                        [
                            "core_complete",
                            "core_ready",
                            "core_complet",
                            "complete_core",
                            "ready_rows",
                            "core_complete_rows",
                        ],
                        0,
                    ),
                    0,
                )

                labeled = csv_int(
                    first_existing_value(
                        row,
                        [
                            "labeled",
                            "target_resolved",
                            "resolved",
                            "targets_resolved",
                            "done",
                        ],
                        0,
                    ),
                    0,
                )

                raw_rate_value = first_existing_value(
                    row,
                    [
                        "core_complete_rate",
                        "core_ready_rate",
                        "complete_rate",
                        "core_rate",
                        "ready_rate",
                    ],
                    "",
                )

                if clean_text(raw_rate_value):
                    core_rate = csv_rate(
                        raw_rate_value,
                        0.0,
                    )
                elif matches > 0:
                    core_rate = core_complete / matches
                elif labeled > 0:
                    core_rate = core_complete / labeled
                else:
                    core_rate = 0.0

                target_rate = csv_rate(
                    first_existing_value(
                        row,
                        [
                            "target_rate",
                            "positive_rate",
                            "positives_rate",
                        ],
                        0,
                    ),
                    0.0,
                )

                if matches < min_matches:
                    continue

                if core_complete < min_core_complete:
                    continue

                if core_rate < min_core_complete_rate:
                    continue

                rows.append({
                    "competition_id": competition_id,
                    "matches": matches,
                    "core_complete": core_complete,
                    "core_complete_rate": core_rate,
                    "labeled": labeled,
                    "target_rate": target_rate,
                })

    except Exception as exc:
        print()
        print(
            "Priorité compétitions CORE_READY : lecture impossible "
            f"({type(exc).__name__}: {exc})"
        )
        print("Le collecteur utilisera donc le parcours normal.")
        return []

    rows.sort(
        key=lambda row: (
            row["core_complete_rate"],
            row["core_complete"],
            row["labeled"],
            row["matches"],
        ),
        reverse=True,
    )

    if top_n > 0:
        rows = rows[:top_n]

    competition_ids = [
        row["competition_id"]
        for row in rows
    ]

    print()
    print("Priorité compétitions CORE_READY activée.")
    print(f"Fichier audit : {audit_csv_path}")
    print(f"Seuil core_complete_rate : >= {min_core_complete_rate:.0%}")
    print(f"Minimum core_complete : >= {min_core_complete}")
    print(f"Minimum matches : >= {min_matches}")
    print(f"Compétitions prioritaires retenues : {len(competition_ids)}")

    if print_filtered:
        if rows:
            print("Top compétitions prioritaires :")

            for row in rows[:25]:
                print(
                    f"  {row['competition_id']} | "
                    f"core_rate={row['core_complete_rate']:.1%} | "
                    f"core={row['core_complete']} | "
                    f"matches={row['matches']} | "
                    f"target_rate={row['target_rate']:.1%}"
                )
        else:
            print("Aucune compétition n'a passé les seuils de priorité.")

    return competition_ids


def argv_has_option(
    argv: Sequence[str],
    option_name: str,
) -> bool:
    prefix = option_name + "="

    return any(
        item == option_name
        or item.startswith(prefix)
        for item in argv
    )


def remove_competition_ids_from_argv(
    argv: Sequence[str],
) -> List[str]:
    """
    Retire un éventuel bloc --competition-ids existant.

    Utile seulement avec --force-priority-competitions.
    """

    cleaned: List[str] = []

    i = 0

    while i < len(argv):
        item = argv[i]

        if item == "--competition-ids":
            i += 1

            while i < len(argv):
                nxt = argv[i]

                if nxt.startswith("--"):
                    break

                i += 1

            continue

        if item.startswith("--competition-ids="):
            i += 1
            continue

        cleaned.append(item)
        i += 1

    return cleaned


def inject_priority_competitions_if_needed(
    collector_argv: List[str],
    priority_competition_ids: Sequence[str],
    force_priority: bool,
) -> List[str]:
    """
    Injecte --competition-ids dans les arguments transmis au collecteur
    hiérarchique.

    Ainsi, le collecteur ne parcourt plus les compétitions au hasard :
    il commence par celles qui ont historiquement le meilleur taux CORE_READY.
    """

    if not priority_competition_ids:
        return collector_argv

    already_has_competitions = argv_has_option(
        collector_argv,
        "--competition-ids",
    )

    if already_has_competitions and not force_priority:
        print()
        print(
            "Priorité compétitions : non injectée car --competition-ids "
            "est déjà fourni manuellement."
        )
        return collector_argv

    if already_has_competitions and force_priority:
        collector_argv = remove_competition_ids_from_argv(
            collector_argv
        )

    print()
    print(
        "Priorité compétitions : injection de "
        f"{len(priority_competition_ids)} competition_id dans le collecteur."
    )

    return (
        list(collector_argv)
        + ["--competition-ids"]
        + list(priority_competition_ids)
    )



# =============================================================================
# SMART SCOPE FILTER — FIRST N MATCHES QUALITY TEST
# =============================================================================

def valid_target_value(value: Any) -> bool:
    text = clean_text(value)

    if not text:
        return False

    try:
        number = float(text)
    except Exception:
        return False

    return number in (0.0, 1.0)


def row_core_available(
    row: Mapping[str, Any],
    core_features: Sequence[str],
) -> int:
    count = 0

    for feature in core_features:
        value = row.get(feature)

        if value is None:
            continue

        text = clean_text(value)

        if not text:
            continue

        try:
            number = float(text)
        except Exception:
            continue

        if not math.isfinite(number):
            continue

        count += 1

    return count


def row_is_core_ready(
    row: Mapping[str, Any],
    core_features: Sequence[str],
) -> bool:
    if not valid_target_value(
        row.get("goal_after_87")
    ):
        return False

    return row_core_available(
        row,
        core_features,
    ) == len(core_features)


def scope_key_from_stub(
    stub: Mapping[str, Any],
) -> Tuple[str, str]:
    competition_id = clean_text(
        stub.get("competition_id")
        or stub.get("league_id")
        or stub.get("tournament_id")
    )

    competition = stub.get("competition")

    if not competition_id and isinstance(competition, Mapping):
        competition_id = clean_text(
            competition.get("id")
            or competition.get("competition_id")
        )

    season_id = clean_text(
        stub.get("season_id")
    )

    season = stub.get("season")

    if not season_id and isinstance(season, Mapping):
        season_id = clean_text(
            season.get("id")
            or season.get("season_id")
        )

    return (
        competition_id or "UNKNOWN_COMPETITION",
        season_id or "ALL",
    )


def scope_key_from_row(
    row: Mapping[str, Any],
) -> Tuple[str, str]:
    return (
        clean_text(row.get("competition_id")) or "UNKNOWN_COMPETITION",
        clean_text(row.get("season_id")) or "ALL",
    )


def existing_scope_quality(
    dataset_df: Any,
    scope_key: Tuple[str, str],
    core_features: Sequence[str],
    test_matches: int,
) -> Optional[Dict[str, Any]]:
    """
    Évalue la qualité d'un scope à partir des lignes déjà présentes.

    Cela évite de continuer une saison connue comme mauvaise, même si les
    premiers matchs testés ont déjà été collectés lors d'un run précédent.
    """

    if dataset_df is None:
        return None

    try:
        if dataset_df.empty:
            return None
    except Exception:
        return None

    if "competition_id" not in dataset_df.columns:
        return None

    competition_id, season_id = scope_key

    scope_df = dataset_df[
        dataset_df["competition_id"].astype(str) == str(competition_id)
    ].copy()

    if "season_id" in scope_df.columns:
        if season_id == "ALL":
            scope_df = scope_df[
                scope_df["season_id"].isna()
                | (scope_df["season_id"].astype(str).str.strip() == "")
            ].copy()
        else:
            scope_df = scope_df[
                scope_df["season_id"].astype(str) == str(season_id)
            ].copy()

    if scope_df.empty:
        return None

    if "goal_after_87" not in scope_df.columns:
        return None

    labeled_mask = scope_df["goal_after_87"].apply(
        valid_target_value
    )

    scope_df = scope_df[
        labeled_mask
    ].copy()

    if scope_df.empty:
        return None

    # Si possible, on prend les premiers matchs chronologiques du scope.
    sort_cols = [
        col
        for col in (
            "kickoff_utc",
            "utc_date",
            "collected_at_utc",
            "match_id",
        )
        if col in scope_df.columns
    ]

    if sort_cols:
        scope_df = scope_df.sort_values(
            by=sort_cols
        )

    sample = scope_df.head(
        max(1, test_matches)
    )

    if len(sample) < test_matches:
        return None

    core_ready_count = int(
        sample.apply(
            lambda row: row_is_core_ready(
                row,
                core_features,
            ),
            axis=1,
        ).sum()
    )

    rate = (
        core_ready_count / len(sample)
        if len(sample)
        else 0.0
    )

    return {
        "source": "existing_dataset",
        "tested": int(len(sample)),
        "core_ready": core_ready_count,
        "rate": rate,
    }


def should_skip_scope_by_quality(
    tested: int,
    core_ready: int,
    test_matches: int,
    min_core_ready: int,
    min_core_ready_rate: float,
) -> bool:
    if tested < test_matches:
        return False

    rate = (
        core_ready / tested
        if tested
        else 0.0
    )

    if core_ready < min_core_ready:
        return True

    if rate < min_core_ready_rate:
        return True

    return False


def install_smart_scope_filter(
    collector: Any,
    filter_args: argparse.Namespace,
) -> None:
    """
    Monkey-patch de collect_match_stub.

    Important :
    - Le scope est évalué uniquement avec les matchs qui auraient été collectés.
    - Si le scope échoue au test, les matchs suivants sont ignorés sans requête
      API de reconstruction PRE/LIVE87.
    - Les matchs ignorés ne sont pas marqués dans l'index. On pourra les
      recollecter plus tard si on change la politique.
    """

    if filter_args.scope_smart_filter_disable:
        print()
        print("Filtre scope intelligent : désactivé.")
        return

    original_collect_match_stub = collector.collect_match_stub

    scope_states: Dict[Tuple[str, str], Dict[str, Any]] = {}

    print()
    print("Filtre scope intelligent : activé.")
    print(
        "Règle : tester "
        f"{filter_args.scope_test_matches} match(s) par saison ; "
        f"continuer si au moins {filter_args.scope_min_core_ready} CORE complets "
        f"et taux >= {filter_args.scope_min_core_ready_rate:.0%}."
    )

    def patched_collect_match_stub(
        collector: Any,
        prematch: Any,
        live87: Any,
        client: Any,
        stub: Mapping[str, Any],
        core_features: Sequence[str],
        index_df: Any,
        dataset_df: Any,
        rejection_rows: List[Dict[str, Any]],
        output_dir: Path,
        force: bool,
        xg_limit: int,
        with_league_second_half: bool,
        preferred_bookmaker: Optional[str],
    ):
        scope_key = scope_key_from_stub(
            stub
        )

        state = scope_states.setdefault(
            scope_key,
            {
                "tested": 0,
                "core_ready": 0,
                "decision": "TESTING",
                "printed_skip": False,
                "existing_checked": False,
            },
        )

        competition_id, season_id = scope_key

        if state["decision"] == "SKIP":
            if not state["printed_skip"]:
                print(
                    "Filtre scope intelligent : SKIP reste du scope "
                    f"competition={competition_id} "
                    f"season={season_id} "
                    f"CORE_READY={state['core_ready']}/{state['tested']}"
                )
                state["printed_skip"] = True

            return (
                index_df,
                dataset_df,
                rejection_rows,
                False,
            )

        # Avant de faire des appels API, on peut utiliser les lignes déjà
        # collectées dans ce scope.
        if not state["existing_checked"]:
            state["existing_checked"] = True

            existing = existing_scope_quality(
                dataset_df=dataset_df,
                scope_key=scope_key,
                core_features=core_features,
                test_matches=filter_args.scope_test_matches,
            )

            if existing is not None:
                state["tested"] = existing["tested"]
                state["core_ready"] = existing["core_ready"]

                if should_skip_scope_by_quality(
                    tested=state["tested"],
                    core_ready=state["core_ready"],
                    test_matches=filter_args.scope_test_matches,
                    min_core_ready=filter_args.scope_min_core_ready,
                    min_core_ready_rate=filter_args.scope_min_core_ready_rate,
                ):
                    state["decision"] = "SKIP"

                    print(
                        "Filtre scope intelligent : SKIP scope selon dataset existant "
                        f"competition={competition_id} "
                        f"season={season_id} "
                        f"CORE_READY={state['core_ready']}/{state['tested']} "
                        f"({state['core_ready'] / state['tested']:.1%})"
                    )

                    state["printed_skip"] = True

                    return (
                        index_df,
                        dataset_df,
                        rejection_rows,
                        False,
                    )

                else:
                    state["decision"] = "KEEP"

                    print(
                        "Filtre scope intelligent : KEEP scope selon dataset existant "
                        f"competition={competition_id} "
                        f"season={season_id} "
                        f"CORE_READY={state['core_ready']}/{state['tested']} "
                        f"({state['core_ready'] / state['tested']:.1%})"
                    )

        # Si le scope est déjà validé, on ne ralentit plus la collecte.
        if state["decision"] == "KEEP":
            return original_collect_match_stub(
                collector=collector,
                prematch=prematch,
                live87=live87,
                client=client,
                stub=stub,
                core_features=core_features,
                index_df=index_df,
                dataset_df=dataset_df,
                rejection_rows=rejection_rows,
                output_dir=output_dir,
                force=force,
                xg_limit=xg_limit,
                with_league_second_half=with_league_second_half,
                preferred_bookmaker=preferred_bookmaker,
            )

        match_id = clean_text(
            stub.get("id")
            or stub.get("match_id")
            or stub.get("event_id")
        )

        (
            new_index_df,
            new_dataset_df,
            new_rejection_rows,
            was_processed,
        ) = original_collect_match_stub(
            collector=collector,
            prematch=prematch,
            live87=live87,
            client=client,
            stub=stub,
            core_features=core_features,
            index_df=index_df,
            dataset_df=dataset_df,
            rejection_rows=rejection_rows,
            output_dir=output_dir,
            force=force,
            xg_limit=xg_limit,
            with_league_second_half=with_league_second_half,
            preferred_bookmaker=preferred_bookmaker,
        )

        if was_processed:
            state["tested"] += 1

            current_row = None

            try:
                if (
                    match_id
                    and "match_id" in new_dataset_df.columns
                ):
                    rows = new_dataset_df[
                        new_dataset_df["match_id"].astype(str) == str(match_id)
                    ]

                    if not rows.empty:
                        current_row = rows.iloc[-1].to_dict()
            except Exception:
                current_row = None

            is_ready = (
                row_is_core_ready(
                    current_row,
                    core_features,
                )
                if current_row is not None
                else False
            )

            if is_ready:
                state["core_ready"] += 1

            print(
                "Filtre scope intelligent : test "
                f"{state['tested']}/{filter_args.scope_test_matches} "
                f"competition={competition_id} "
                f"season={season_id} "
                f"CORE_READY={state['core_ready']}/{state['tested']}"
            )

            if state["tested"] >= filter_args.scope_test_matches:
                rate = (
                    state["core_ready"] / state["tested"]
                    if state["tested"]
                    else 0.0
                )

                if should_skip_scope_by_quality(
                    tested=state["tested"],
                    core_ready=state["core_ready"],
                    test_matches=filter_args.scope_test_matches,
                    min_core_ready=filter_args.scope_min_core_ready,
                    min_core_ready_rate=filter_args.scope_min_core_ready_rate,
                ):
                    state["decision"] = "SKIP"

                    print(
                        "Filtre scope intelligent : ABANDON saison "
                        f"competition={competition_id} "
                        f"season={season_id} "
                        f"CORE_READY={state['core_ready']}/{state['tested']} "
                        f"({rate:.1%})"
                    )

                else:
                    state["decision"] = "KEEP"

                    print(
                        "Filtre scope intelligent : CONTINUE saison "
                        f"competition={competition_id} "
                        f"season={season_id} "
                        f"CORE_READY={state['core_ready']}/{state['tested']} "
                        f"({rate:.1%})"
                    )

        return (
            new_index_df,
            new_dataset_df,
            new_rejection_rows,
            was_processed,
        )

    collector.collect_match_stub = patched_collect_match_stub


# =============================================================================
# LAUNCHER
# =============================================================================

def parse_launcher_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Options du lanceur football-only.",
    )

    parser.add_argument(
        "--football-filter-disable",
        action="store_true",
        help="Désactive le filtre football-only.",
    )

    parser.add_argument(
        "--strict-football-metadata",
        action="store_true",
        help=(
            "N'accepte que les compétitions dont les métadonnées indiquent "
            "explicitement football/soccer."
        ),
    )

    parser.add_argument(
        "--print-filtered",
        action="store_true",
        help="Affiche les compétitions/matchs/scopes filtrés.",
    )

    parser.add_argument(
        "--non-football-scope-skip-rate",
        type=float,
        default=0.50,
        help=(
            "Si cette proportion des matchs d'un scope est clairement "
            "non-football, le scope entier est ignoré. Défaut: 0.50."
        ),
    )

    parser.add_argument(
        "--non-football-scope-min-matches",
        type=int,
        default=5,
        help=(
            "Minimum de matchs dans un scope avant d'appliquer la règle "
            "de majorité non-football. Défaut: 5."
        ),
    )


    parser.add_argument(
        "--priority-competitions-disable",
        action="store_true",
        help="Désactive la sélection intelligente des meilleures compétitions.",
    )

    parser.add_argument(
        "--priority-competition-file",
        default=None,
        help=(
            "Chemin vers audit_competition_quality_v2.csv. "
            "Défaut: outputs/historical_dataset_v2/audit/audit_competition_quality_v2.csv"
        ),
    )

    parser.add_argument(
        "--min-core-complete-rate",
        type=float,
        default=0.90,
        help="Taux minimum CORE_READY pour prioriser une compétition. Défaut: 0.90.",
    )

    parser.add_argument(
        "--min-core-complete",
        type=int,
        default=10,
        help="Minimum de lignes CORE_READY déjà observées. Défaut: 10.",
    )

    parser.add_argument(
        "--priority-min-matches",
        type=int,
        default=10,
        help="Minimum de matchs observés dans l'audit compétition. Défaut: 10.",
    )

    parser.add_argument(
        "--priority-top-n",
        type=int,
        default=50,
        help="Nombre maximum de compétitions prioritaires à utiliser. Défaut: 50.",
    )

    parser.add_argument(
        "--force-priority-competitions",
        action="store_true",
        help="Force les compétitions prioritaires même si --competition-ids est fourni.",
    )


    parser.add_argument(
        "--scope-smart-filter-disable",
        action="store_true",
        help=(
            "Désactive le filtre intelligent qui teste les premiers matchs "
            "d'une saison avant de continuer."
        ),
    )

    parser.add_argument(
        "--scope-test-matches",
        type=int,
        default=5,
        help="Nombre de matchs à tester par saison. Défaut: 5.",
    )

    parser.add_argument(
        "--scope-min-core-ready",
        type=int,
        default=3,
        help="Nombre minimum de matchs CORE complets requis. Défaut: 3.",
    )

    parser.add_argument(
        "--scope-min-core-ready-rate",
        type=float,
        default=0.60,
        help="Taux minimum CORE complet requis. Défaut: 0.60.",
    )

    args, remaining = parser.parse_known_args(
        list(argv)
    )

    # Bornes de sécurité.
    args.non_football_scope_skip_rate = max(
        0.10,
        min(
            1.0,
            float(args.non_football_scope_skip_rate),
        ),
    )

    args.non_football_scope_min_matches = max(
        1,
        int(args.non_football_scope_min_matches),
    )

    args.min_core_complete_rate = max(
        0.0,
        min(
            1.0,
            float(args.min_core_complete_rate),
        ),
    )

    args.min_core_complete = max(
        0,
        int(args.min_core_complete),
    )

    args.priority_min_matches = max(
        0,
        int(args.priority_min_matches),
    )

    args.priority_top_n = max(
        0,
        int(args.priority_top_n),
    )


    return args, remaining


def load_hierarchical_collector():
    if not HIERARCHICAL_COLLECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {HIERARCHICAL_COLLECTOR_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "collect_historical_dataset_v2_hierarchical_runtime",
        HIERARCHICAL_COLLECTOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Impossible de charger : {HIERARCHICAL_COLLECTOR_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "collect_historical_dataset_v2_hierarchical_runtime"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def main() -> int:
    filter_args, collector_argv = parse_launcher_args(
        sys.argv[1:]
    )

    collector = load_hierarchical_collector()

    priority_competition_ids: List[str] = []

    if not filter_args.priority_competitions_disable:
        priority_file = resolve_path_maybe(
            filter_args.priority_competition_file,
            default_priority_file(),
        )

        priority_competition_ids = load_priority_competitions_from_audit(
            audit_csv_path=priority_file,
            min_core_complete_rate=filter_args.min_core_complete_rate,
            min_core_complete=filter_args.min_core_complete,
            min_matches=filter_args.priority_min_matches,
            top_n=filter_args.priority_top_n,
            print_filtered=filter_args.print_filtered,
        )

        collector_argv = inject_priority_competitions_if_needed(
            collector_argv=list(collector_argv),
            priority_competition_ids=priority_competition_ids,
            force_priority=filter_args.force_priority_competitions,
        )

    else:
        print()
        print("Priorité compétitions CORE_READY désactivée.")

    if filter_args.football_filter_disable:
        print()
        print("Filtre football-only désactivé.")
    else:
        print()
        print("Filtre football-only activé.")
        print("Les compétitions/matchs clairement non-football seront ignorés.")
        print(
            "Garde scope : skip si "
            f">= {filter_args.non_football_scope_skip_rate:.0%} "
            "des matchs du scope sont non-football."
        )

        original_list_competitions = collector.list_competitions
        original_list_finished_matches_for_scope = (
            collector.list_finished_matches_for_scope
        )

        def patched_list_competitions(
            client: Any,
            limit: int,
            max_pages: int,
            page_sleep: float,
        ):
            # On récupère davantage de compétitions que demandé, car certaines
            # peuvent être filtrées.
            expanded_limit = (
                limit * 3
                if limit and limit > 0
                else limit
            )

            competitions = original_list_competitions(
                client=client,
                limit=expanded_limit,
                max_pages=max_pages,
                page_sleep=page_sleep,
            )

            filtered = filter_competitions(
                competitions=competitions,
                strict_metadata=filter_args.strict_football_metadata,
                print_filtered=filter_args.print_filtered,
            )

            if limit and limit > 0:
                return filtered[:limit]

            return filtered

        def patched_list_finished_matches_for_scope(
            client: Any,
            competition_id: Optional[str],
            season_id: Optional[str],
            date_from: Optional[str],
            date_to: Optional[str],
            max_pages: int,
            page_sleep: float,
        ):
            matches = original_list_finished_matches_for_scope(
                client=client,
                competition_id=competition_id,
                season_id=season_id,
                date_from=date_from,
                date_to=date_to,
                max_pages=max_pages,
                page_sleep=page_sleep,
            )

            return filter_matches_with_scope_guard(
                matches=matches,
                print_filtered=filter_args.print_filtered,
                scope_skip_rate=filter_args.non_football_scope_skip_rate,
                scope_min_matches=filter_args.non_football_scope_min_matches,
                competition_id=competition_id,
                season_id=season_id,
            )

        collector.list_competitions = patched_list_competitions
        collector.list_finished_matches_for_scope = (
            patched_list_finished_matches_for_scope
        )

    install_smart_scope_filter(
        collector=collector,
        filter_args=filter_args,
    )

    old_argv = sys.argv[:]

    sys.argv = [
        str(HIERARCHICAL_COLLECTOR_PATH),
    ] + collector_argv

    try:
        return int(
            collector.main()
            or 0
        )

    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
