#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

match_scheduler_v3c.py



MATCH SCHEDULER V3-C — GOAL AFTER 87'

====================================



Version alignée sur le scanner de production V3-C :

- le scheduler charge le schéma V3-C exact (38 features) utilisé par live_scanner_engine.py ;

- la readiness historique est dérivée du RAW avec CE schéma modèle ;

- si le CSV CORE_READY existe, il reste disponible comme fallback ;

- un dataset historique absent ne doit plus rejeter artificiellement 100 % des matchs ;

- les odds ne sont exigées en mode AUTO que si le modèle chargé utilise des features odds ;

- le scanner LIVE reste l'autorité finale pour la couverture et la prédiction ;

- zone prioritaire V3-C : 0.55 <= P(goal_after_87) < 0.65.



Le script ne modifie pas live_scanner_engine.py.

Il sert seulement à préparer les fenêtres où lancer le scanner.



Commandes :

    python3 match_scheduler_v3c.py

    python3 match_scheduler_v3c.py --date 2026-09-14 --show-rejected

    python3 match_scheduler_v3c.py --mode loose --show-rejected

    python3 match_scheduler_v3c.py --watch

    python3 match_scheduler_v3c.py --include-unknown

"""



from __future__ import annotations



import argparse

import csv

import json

import logging

import math

import os

import subprocess

import sys

import time

from dataclasses import dataclass, field, asdict

from datetime import date, datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple



import requests

from dotenv import load_dotenv



try:

    from zoneinfo import ZoneInfo

except ImportError:

    ZoneInfo = None





# =============================================================================

# PATHS / ENV

# =============================================================================



BASE_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)



OUTPUT_DIR = BASE_DIR / "outputs" / "scheduler_v3c"

CACHE_DIR = OUTPUT_DIR / "cache"

NOTIFICATION_STATE_PATH = CACHE_DIR / "notification_state_v3c.json"



HISTORICAL_DATASET_DIR = BASE_DIR / "outputs" / "historical_dataset_v2"



DEFAULT_RAW_DATASET_PATH = (

    HISTORICAL_DATASET_DIR / "Historical_Dataset_Goal_After_87_RAW_v2.csv"

)

DEFAULT_CORE_READY_DATASET_PATH = (

    HISTORICAL_DATASET_DIR / "Historical_Dataset_Goal_After_87_CORE_READY_v2.csv"

)

FIXED_CORE_READY_DATASET_PATH = (

    HISTORICAL_DATASET_DIR / "Historical_Dataset_Goal_After_87_CORE_READY_v2_FIXED.csv"

)



# IMPORTANT : le scanner LIVE actuel lit d'abord le schéma placé avec le modèle.

# L'ancien scheduler pointait seulement vers outputs/model_schema_v2/, ce qui

# pouvait désynchroniser le planning du modèle réellement chargé.

DEFAULT_MODEL_V3C_SCHEMA_PATH = (
    BASE_DIR / "outputs" / "model_v3_experiments" / "schemas" / "V3_C_AGGRESSIVE_schema.json"
)

LEGACY_MODEL_SCHEMA_V3C_PATH = (
    BASE_DIR / "outputs" / "model_v3_experiments" / "schemas" / "V3_C_AGGRESSIVE_schema.json"
)





def _env_path(name: str, default: Path) -> Path:

    raw = os.getenv(name, "").strip()

    if not raw:

        return default



    path = Path(raw).expanduser()

    if not path.is_absolute():

        path = BASE_DIR / path

    return path





RAW_DATASET_PATH = _env_path(

    "SCHEDULER_RAW_DATASET_PATH",

    DEFAULT_RAW_DATASET_PATH,

)



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR.mkdir(parents=True, exist_ok=True)





# =============================================================================

# LOGGING

# =============================================================================



logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger("match_scheduler_v3c")





# =============================================================================

# CONFIGURATION

# =============================================================================



STATS_API_KEY = os.getenv("STATS_API_KEY", "").strip()

if STATS_API_KEY.lower().startswith("bearer "):

    STATS_API_KEY = STATS_API_KEY[7:].strip()



STATS_API_BASE_URL = os.getenv(

    "STATS_API_BASE_URL",

    "https://api.thestatsapi.com/api",

).rstrip("/")



STATS_API_TIMEOUT = int(os.getenv("STATS_API_TIMEOUT", "20"))

SCHEDULER_PER_PAGE = int(os.getenv("SCHEDULER_PER_PAGE", "100"))

SCHEDULER_MAX_PAGES = int(os.getenv("SCHEDULER_MAX_PAGES", "20"))



SCHEDULER_TIMEZONE = os.getenv(

    "SCHEDULER_TIMEZONE",

    "America/Toronto",

).strip()



SCHEDULER_MATCH_STATUSES = [

    s.strip()

    for s in os.getenv(

        "SCHEDULER_MATCH_STATUSES",

        # "scheduled" est accepté par l'endpoint utilisé ici.

        # Les valeurs not_started/pending provoquent actuellement des HTTP 400

        # sur ce flux et n'apportent aucun match supplémentaire.

        "scheduled",

    ).split(",")

    if s.strip()

]



SCANNER_START_OFFSET_MINUTES = int(

    os.getenv("SCHEDULER_SCANNER_START_OFFSET_MINUTES", "90")

)

SCANNER_END_OFFSET_MINUTES = int(

    os.getenv("SCHEDULER_SCANNER_END_OFFSET_MINUTES", "112")

)

SESSION_MERGE_GAP_MINUTES = int(

    os.getenv("SCHEDULER_SESSION_MERGE_GAP_MINUTES", "10")

)

REMINDER_LEAD_MINUTES = int(

    os.getenv("SCHEDULER_REMINDER_LEAD_MINUTES", "5")

)



SCANNER_COMMAND = os.getenv(

    "SCHEDULER_SCANNER_COMMAND",

    "python3 live_scanner_engine.py",

).strip()



MODEL_ALERT_PROBABILITY = 0.55  # Début de la zone V3-C prioritaire [0.55, 0.65).
MODEL_ALERT_PROBABILITY_MAX = 0.65



# Même logique que le scanner LIVE actuel.

MIN_MODEL_COVERAGE = 38  # V3-C exige exactement 38/38; ignore les valeurs V2 du .env.



ODDS_FEATURES: Set[str] = {

    "odds_home",

    "odds_draw",

    "odds_away",

    "odds_over05",

    "odds_over15",

    "odds_over25",

    "odds_over35",

    "odds_under25",

    "odds_btts_yes",

    "odds_btts_no",

}



CRITICAL_MODEL_FEATURES: Set[str] = {

    "score_87_home",

    "score_87_away",

    "leading_team_87",

    "goal_diff_87",

    "is_draw_87",

    "home_shots_87",

    "away_shots_87",

    "home_shots_on_target_87",

    "away_shots_on_target_87",

    "shots_80_87",

    "sot_80_87",

    "last_shot_minute",

    "last_shot_on_target_minute",

    "minutes_since_last_shot",

    "minutes_since_last_shot_on_target",

}



# AUTO = n'exige odds_available que si le schéma du modèle contient au moins

# une feature de cote. "required" force le filtre ; "off" le désactive.

SCHEDULER_ODDS_POLICY = os.getenv(

    "SCHEDULER_ODDS_POLICY",

    "auto",

).strip().lower()



# Compatibilité avec l'ancien .env uniquement si la nouvelle variable n'est

# pas définie explicitement.

if "SCHEDULER_ODDS_POLICY" not in os.environ:

    legacy_odds_setting = os.getenv("SCHEDULER_REQUIRE_TARGET_ODDS")

    if legacy_odds_setting is not None:

        SCHEDULER_ODDS_POLICY = (

            "off"

            if legacy_odds_setting.strip().lower() in {"0", "false", "no", "off"}

            else "required"

        )



SCHEDULER_MIN_CORE_READY_RATE = float(

    os.getenv("SCHEDULER_MIN_CORE_READY_RATE", "0.35")

)

SCHEDULER_MIN_CORE_READY_MATCHES = int(

    os.getenv("SCHEDULER_MIN_CORE_READY_MATCHES", "5")

)

SCHEDULER_MIN_RAW_MATCHES = int(os.getenv("SCHEDULER_MIN_RAW_MATCHES", "8"))



INCLUDE_UNKNOWN_COMPETITIONS_DEFAULT = (

    os.getenv("SCHEDULER_INCLUDE_UNKNOWN_COMPETITIONS", "0")

    .strip()

    .lower()

    in {"1", "true", "yes", "on"}

)



TELEGRAM_NOTIFICATIONS_ENABLED = (

    os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "1")

    .strip()

    .lower()

    not in {"0", "false", "no", "off"}

)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TELEGRAM_API_TIMEOUT = int(os.getenv("TELEGRAM_API_TIMEOUT", "10"))



MACOS_NOTIFICATIONS_ENABLED = (

    os.getenv("MACOS_NOTIFICATIONS_ENABLED", "1")

    .strip()

    .lower()

    not in {"0", "false", "no", "off"}

)





# =============================================================================

# DATACLASSES

# =============================================================================



@dataclass

class Fixture:

    match_id: str

    competition_id: str

    season_id: str

    competition_name: str

    country: str

    home_team_id: str

    home_team: str

    away_team_id: str

    away_team: str

    kickoff_utc: datetime

    kickoff_local: datetime

    status: str

    odds_available: bool

    live_odds_available: bool

    xg_available: bool

    raw: Dict[str, Any] = field(default_factory=dict)





@dataclass

class CompetitionCapabilityV2:

    competition_id: str

    season_id: str

    competition_name: str

    country: str

    scope_used: str

    raw_rows_same_season: int

    core_ready_rows_same_season: int

    core_ready_rate_same_season: float

    raw_rows_all_seasons: int

    core_ready_rows_all_seasons: int

    core_ready_rate_all_seasons: float

    positives_all_seasons: int

    negatives_all_seasons: int

    target_rate_all_seasons: float

    grade: str

    eligible_by_history: bool

    notes: List[str] = field(default_factory=list)





@dataclass

class PlannedMatch:

    fixture: Fixture

    capability: CompetitionCapabilityV2

    launch_at: datetime

    expected_85_at: datetime

    expected_87_at: datetime

    stop_after: datetime

    eligible: bool

    rejection_reasons: List[str] = field(default_factory=list)





@dataclass

class Session:

    session_id: int

    start_at: datetime

    reminder_at: datetime

    end_at: datetime

    matches: List[PlannedMatch] = field(default_factory=list)





# =============================================================================

# HELPERS

# =============================================================================



def first_not_none(*values: Any) -> Any:

    for value in values:

        if value is not None:

            return value

    return None





def safe_int(value: Any) -> Optional[int]:

    try:

        if value is None or value == "":

            return None

        return int(float(value))

    except Exception:

        return None





def safe_float(value: Any) -> Optional[float]:

    try:

        if value is None or value == "":

            return None

        result = float(value)

        if math.isfinite(result):

            return result

    except Exception:

        return None

    return None





def safe_rate(numerator: float, denominator: float) -> float:

    if denominator <= 0:

        return 0.0

    return float(numerator / denominator)





def truthy(value: Any) -> bool:

    if isinstance(value, bool):

        return value

    if value is None:

        return False

    if isinstance(value, (int, float)):

        return bool(value)

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}





def as_list(payload: Any) -> List[Any]:

    if payload is None:

        return []

    if isinstance(payload, list):

        return payload

    if not isinstance(payload, Mapping):

        return []



    for key in ("matches", "results", "items", "events", "timeline", "standings", "table"):

        value = payload.get(key)

        if isinstance(value, list):

            return value



    data = payload.get("data")

    if isinstance(data, list):

        return data

    if isinstance(data, Mapping):

        for key in ("matches", "results", "items", "events", "timeline", "standings", "table"):

            value = data.get(key)

            if isinstance(value, list):

                return value

    return []





def parse_utc_datetime(value: Any) -> Optional[datetime]:

    if not value:

        return None

    text = str(value).strip()

    try:

        if text.endswith("Z"):

            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:

            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:

        return None





def load_json_file(path: Path, default: Any) -> Any:

    try:

        if path.exists():

            return json.loads(path.read_text(encoding="utf-8"))

    except Exception:

        pass

    return default





def save_json_file(path: Path, payload: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(

        json.dumps(payload, ensure_ascii=False, indent=2, default=str),

        encoding="utf-8",

    )

    tmp.replace(path)





def get_local_tz():

    if ZoneInfo is None:

        raise RuntimeError("zoneinfo indisponible. Python 3.9+ est requis.")

    try:

        return ZoneInfo(SCHEDULER_TIMEZONE)

    except Exception as exc:

        raise RuntimeError(f"Timezone invalide : {SCHEDULER_TIMEZONE}") from exc





LOCAL_TZ = get_local_tz()





def fmt_time(dt: datetime) -> str:

    return dt.strftime("%H:%M")





def print_header(title: str) -> None:

    bar = "=" * max(60, len(title) + 4)

    print()

    print(bar)

    print(f" {title}")

    print(bar)





# =============================================================================

# THESTATSAPI CLIENT

# =============================================================================



class StatsAPIClient:

    def __init__(self, api_key: str, base_url: str, timeout: int = 20) -> None:

        if not api_key:

            raise RuntimeError("STATS_API_KEY est absent du .env.")

        self.base_url = base_url.rstrip("/")

        self.timeout = timeout

        self.requests_total = 0

        self.session = requests.Session()

        self.session.headers.update({

            "Authorization": f"Bearer {api_key}",

            "Accept": "application/json",

        })



    def get(

        self,

        path: str,

        params: Optional[Mapping[str, Any]] = None,

        *,

        soft_fail: bool = True,

        retries: int = 3,

    ) -> Any:

        url = f"{self.base_url}/{path.lstrip('/')}"

        last_error: Optional[Exception] = None



        for attempt in range(1, retries + 1):

            try:

                self.requests_total += 1

                response = self.session.get(

                    url,

                    params=dict(params or {}),

                    timeout=self.timeout,

                )



                if response.status_code == 429:

                    retry_after = safe_int(response.headers.get("Retry-After"))

                    wait_seconds = retry_after or min(15, attempt * 3)

                    logger.warning("429 TheStatsAPI | attente %ss | %s", wait_seconds, path)

                    time.sleep(wait_seconds)

                    continue



                if 400 <= response.status_code < 500 and response.status_code != 429:

                    last_error = requests.HTTPError(

                        f"{response.status_code} Client Error for url: {response.url}",

                        response=response,

                    )

                    if soft_fail:

                        logger.warning("TheStatsAPI GET indisponible | %s | HTTP %s", path, response.status_code)

                        return None

                    raise last_error



                response.raise_for_status()

                return response.json()



            except requests.RequestException as exc:

                last_error = exc

                status_code = exc.response.status_code if getattr(exc, "response", None) is not None else None

                retryable = status_code is None or status_code >= 500 or status_code == 429

                if retryable and attempt < retries:

                    time.sleep(min(5, attempt * 1.5))

                    continue

                break

            except Exception as exc:

                last_error = exc

                break



        if soft_fail:

            logger.warning("TheStatsAPI GET indisponible | %s | %s", path, last_error)

            return None

        raise RuntimeError(f"TheStatsAPI GET failed: {path}: {last_error}")



    def paginated_matches(self, params: Mapping[str, Any]) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        seen_ids: Set[str] = set()



        for page in range(1, SCHEDULER_MAX_PAGES + 1):

            query = dict(params)

            query["page"] = page

            query["per_page"] = SCHEDULER_PER_PAGE



            payload = self.get("/football/matches", query, soft_fail=True)

            rows = as_list(payload)



            logger.info(

                "Fixtures page %d | status=%s | reçus=%d",

                page,

                query.get("status", ""),

                len(rows),

            )



            new_count = 0

            for row in rows:

                if not isinstance(row, Mapping):

                    continue

                match_id = str(first_not_none(row.get("id"), row.get("match_id"), ""))

                if not match_id or match_id in seen_ids:

                    continue

                seen_ids.add(match_id)

                results.append(dict(row))

                new_count += 1



            if len(rows) < SCHEDULER_PER_PAGE:

                break

            if new_count == 0:

                break



        return results





# =============================================================================

# FIXTURE EXTRACTION

# =============================================================================



def extract_team(raw: Mapping[str, Any], side: str) -> Tuple[str, str]:

    team = raw.get(f"{side}_team")

    if isinstance(team, Mapping):

        return (

            str(first_not_none(team.get("id"), team.get("team_id"), "")),

            str(first_not_none(team.get("name"), team.get("team_name"), "")),

        )

    return (

        str(first_not_none(raw.get(f"{side}_team_id"), "")),

        str(first_not_none(raw.get(f"{side}_team_name"), raw.get(f"{side}_name"), team, "")),

    )





def extract_competition_name(raw: Mapping[str, Any]) -> str:

    competition = raw.get("competition")

    if isinstance(competition, Mapping):

        return str(first_not_none(competition.get("name"), competition.get("competition_name"), ""))

    if isinstance(competition, str):

        return competition

    return str(first_not_none(raw.get("competition_name"), raw.get("league_name"), ""))





def extract_country(raw: Mapping[str, Any]) -> str:

    competition = raw.get("competition")

    if isinstance(competition, Mapping):

        country = competition.get("country")

        if isinstance(country, Mapping):

            return str(first_not_none(country.get("name"), country.get("country_name"), ""))

        if country:

            return str(country)

    country = raw.get("country")

    if isinstance(country, Mapping):

        return str(first_not_none(country.get("name"), ""))

    return str(country or "")





def fixture_from_raw(raw: Mapping[str, Any]) -> Optional[Fixture]:

    match_id = str(first_not_none(raw.get("id"), raw.get("match_id"), ""))

    competition_id = str(first_not_none(

        raw.get("competition_id"),

        raw.get("competition", {}).get("id") if isinstance(raw.get("competition"), Mapping) else None,

        "",

    ))

    season_id = str(first_not_none(

        raw.get("season_id"),

        raw.get("season", {}).get("id") if isinstance(raw.get("season"), Mapping) else None,

        "",

    ))

    kickoff_raw = first_not_none(

        raw.get("utc_date"),

        raw.get("kickoff_utc"),

        raw.get("date"),

        raw.get("start_time"),

        raw.get("start_at"),

    )

    kickoff_utc = parse_utc_datetime(kickoff_raw)



    if not (match_id and competition_id and season_id and kickoff_utc):

        return None



    home_id, home_name = extract_team(raw, "home")

    away_id, away_name = extract_team(raw, "away")



    return Fixture(

        match_id=match_id,

        competition_id=competition_id,

        season_id=season_id,

        competition_name=extract_competition_name(raw),

        country=extract_country(raw),

        home_team_id=home_id,

        home_team=home_name,

        away_team_id=away_id,

        away_team=away_name,

        kickoff_utc=kickoff_utc,

        kickoff_local=kickoff_utc.astimezone(LOCAL_TZ),

        status=str(raw.get("status") or ""),

        odds_available=truthy(raw.get("odds_available")),

        live_odds_available=truthy(raw.get("live_odds_available")),

        xg_available=truthy(raw.get("xg_available")),

        raw=dict(raw),

    )





def fetch_fixtures_for_local_date(client: StatsAPIClient, target_date: date) -> List[Fixture]:

    api_from = target_date - timedelta(days=1)

    api_to = target_date + timedelta(days=1)



    logger.info(

        "Récupération matchs | date locale=%s | plage API=%s -> %s",

        target_date.isoformat(),

        api_from.isoformat(),

        api_to.isoformat(),

    )



    all_rows: List[Dict[str, Any]] = []

    seen_ids: Set[str] = set()



    for status in SCHEDULER_MATCH_STATUSES:

        rows = client.paginated_matches({

            "date_from": api_from.isoformat(),

            "date_to": api_to.isoformat(),

            "status": status,

        })

        for row in rows:

            match_id = str(first_not_none(row.get("id"), row.get("match_id"), ""))

            if not match_id or match_id in seen_ids:

                continue

            seen_ids.add(match_id)

            all_rows.append(row)



    fixtures: List[Fixture] = []

    for raw in all_rows:

        fixture = fixture_from_raw(raw)

        if fixture is None:

            continue

        if fixture.kickoff_local.date() != target_date:

            continue

        fixtures.append(fixture)



    fixtures.sort(key=lambda x: x.kickoff_local)

    logger.info("Matchs programmés retenus pour %s : %d", target_date.isoformat(), len(fixtures))

    return fixtures





# =============================================================================

# DATASET V2 CAPABILITY

# =============================================================================



def read_csv_rows(path: Path) -> List[Dict[str, str]]:

    """

    Lecture CSV robuste.



    skipinitialspace=True est volontaire : certains anciens exports historiques

    écrivaient `, "{...}"`. Sans cette option, les virgules du JSON pouvaient

    être interprétées comme de nouvelles colonnes.

    """

    if not path.exists():

        logger.warning("Dataset introuvable : %s", path)

        return []



    rows: List[Dict[str, str]] = []

    malformed = 0



    with path.open("r", encoding="utf-8-sig", newline="") as handle:

        reader = csv.DictReader(handle, skipinitialspace=True)



        if not reader.fieldnames:

            logger.warning("Dataset sans header exploitable : %s", path)

            return []



        for line_number, row in enumerate(reader, start=2):

            # DictReader place les champs supplémentaires sous la clé None.

            extras = row.get(None)

            if extras:

                malformed += 1

                logger.warning(

                    "Ligne CSV ignorée | fichier=%s | ligne=%d | champs supplémentaires=%d",

                    path.name,

                    line_number,

                    len(extras),

                )

                continue



            normalized = {

                str(key).strip(): (

                    value.strip()

                    if isinstance(value, str)

                    else value

                )

                for key, value in row.items()

                if key is not None

            }

            rows.append(normalized)



    if malformed:

        logger.warning(

            "Dataset %s | lignes structurellement invalides ignorées=%d",

            path.name,

            malformed,

        )



    logger.info(

        "Dataset chargé : %s | lignes=%d | colonnes=%d",

        path,

        len(rows),

        len(reader.fieldnames or []),

    )

    return rows





def csv_value_available(value: Any) -> bool:

    if value is None:

        return False



    text = str(value).strip()

    if not text:

        return False



    return text.lower() not in {

        "nan",

        "none",

        "null",

        "na",

        "n/a",

        "<na>",

    }





def _extract_feature_names_from_schema(payload: Any) -> List[str]:

    if isinstance(payload, list):

        return [

            str(feature).strip()

            for feature in payload

            if str(feature).strip()

        ]



    if not isinstance(payload, Mapping):

        return []



    for key in (

        "features_used",

        "core_features",

        "features",

        "feature_names",

    ):

        value = payload.get(key)

        if isinstance(value, list):

            return [

                str(feature).strip()

                for feature in value

                if str(feature).strip()

            ]



    return []





def resolve_model_schema() -> Tuple[List[str], Optional[Path]]:

    """

    Charge le même schéma que le scanner LIVE autant que possible.



    Le chemin est fixe, identique au scanner V3-C :
    outputs/model_v3_experiments/schemas/V3_C_AGGRESSIVE_schema.json.
    Les chemins de schéma V2 définis dans .env sont ignorés.

    """

    candidates: List[Path] = []



    # V3-C ONLY: ignore MODEL_FEATURES_PATH from .env because it may point to V2.
    candidates = [DEFAULT_MODEL_V3C_SCHEMA_PATH]



    seen: Set[str] = set()



    for path in candidates:

        key = str(path)

        if key in seen:

            continue

        seen.add(key)



        if not path.exists():

            continue



        try:

            payload = json.loads(path.read_text(encoding="utf-8"))

        except Exception as exc:

            logger.warning("Schéma modèle illisible | %s | %s", path, exc)

            continue



        features = _extract_feature_names_from_schema(payload)

        if len(features) != 38:
            logger.warning(
                "Schéma ignoré : V3-C exige exactement 38 features | %s | trouvé=%d",
                path,
                len(features),
            )
            continue



        logger.info(

            "Schéma modèle scheduler : %s | features=%d",

            path,

            len(features),

        )

        return features, path



    logger.warning(

        "Aucun schéma V3-C valide (38 features) trouvé. "

        "Le scheduler ne fabriquera pas de CORE_READY artificiel."

    )

    return [], None





def resolve_core_ready_dataset_path() -> Optional[Path]:

    candidates: List[Path] = []



    explicit = os.getenv("SCHEDULER_CORE_READY_DATASET_PATH", "").strip()

    if explicit:

        explicit_path = Path(explicit).expanduser()

        if not explicit_path.is_absolute():

            explicit_path = BASE_DIR / explicit_path

        candidates.append(explicit_path)



    candidates.extend([

        DEFAULT_CORE_READY_DATASET_PATH,

        FIXED_CORE_READY_DATASET_PATH,

    ])



    seen: Set[str] = set()

    for path in candidates:

        key = str(path)

        if key in seen:

            continue

        seen.add(key)



        if path.exists():

            return path



    return None





def derive_model_ready_rows(

    raw_rows: Sequence[Mapping[str, Any]],

    model_features: Sequence[str],

) -> List[Dict[str, Any]]:

    """

    Reproduit la gate V3-C du scanner sur l'historique :

    - couverture >= MIN_MODEL_COVERAGE (plafonnée au nombre de features modèle)

    - aucune feature critique du modèle ne doit manquer.



    Avec V3-C, les 38 features du schéma sont obligatoires : 38/38 pour MODEL_READY.

    """

    if not raw_rows or not model_features:

        return []



    model_features = list(model_features)

    required_coverage = min(MIN_MODEL_COVERAGE, len(model_features))

    critical = set(model_features) & CRITICAL_MODEL_FEATURES



    ready_rows: List[Dict[str, Any]] = []



    for row in raw_rows:

        available = sum(

            csv_value_available(row.get(feature))

            for feature in model_features

        )



        critical_missing = [

            feature

            for feature in critical

            if not csv_value_available(row.get(feature))

        ]



        if available >= required_coverage and not critical_missing:

            ready_rows.append(dict(row))



    logger.info(

        "MODEL_READY dérivé du RAW | %d/%d | schéma=%d | couverture min=%d | critiques=%d",

        len(ready_rows),

        len(raw_rows),

        len(model_features),

        required_coverage,

        len(critical),

    )

    return ready_rows





def row_match_competition(

    row: Mapping[str, Any],

    competition_id: str,

    season_id: Optional[str] = None,

) -> bool:

    if str(row.get("competition_id") or "") != str(competition_id):

        return False

    if season_id is not None:

        return str(row.get("season_id") or "") == str(season_id)

    return True





def count_labeled_targets(

    rows: Sequence[Mapping[str, Any]],

) -> Tuple[int, int]:

    positives = 0

    negatives = 0

    for row in rows:

        target = safe_float(

            first_not_none(

                row.get("goal_after_87"),

                row.get("target"),

                row.get("y_true"),

            )

        )

        if target == 1:

            positives += 1

        elif target == 0:

            negatives += 1

    return positives, negatives





def grade_capability_v2(

    raw_rows: int,

    core_rows: int,

    core_rate: float,

    min_core_rate: float,

    min_core_rows: int,

) -> str:

    if core_rows <= 0 and raw_rows <= 0:

        return "UNKNOWN"

    if core_rows >= max(20, min_core_rows) and core_rate >= 0.75:

        return "EXCELLENT"

    if core_rows >= min_core_rows and core_rate >= 0.50:

        return "GOOD"

    if core_rows >= min_core_rows and core_rate >= min_core_rate:

        return "WATCH"

    if core_rows > 0:

        return "LOW_SAMPLE"

    return "LIMITED"





def build_competition_capability_v2(

    fixture: Fixture,

    raw_rows: Sequence[Mapping[str, Any]],

    core_rows: Sequence[Mapping[str, Any]],

    min_core_rate: float,

    min_core_rows: int,

    history_filter_available: bool,

) -> CompetitionCapabilityV2:

    # Si l'historique global ne peut pas être calculé, on ne doit surtout pas

    # transformer un problème de fichier en rejet de 100 % des matchs.

    if not history_filter_available:

        return CompetitionCapabilityV2(

            competition_id=fixture.competition_id,

            season_id=fixture.season_id,

            competition_name=fixture.competition_name,

            country=fixture.country,

            scope_used="scanner_live_authoritative",

            raw_rows_same_season=0,

            core_ready_rows_same_season=0,

            core_ready_rate_same_season=0.0,

            raw_rows_all_seasons=0,

            core_ready_rows_all_seasons=0,

            core_ready_rate_all_seasons=0.0,

            positives_all_seasons=0,

            negatives_all_seasons=0,

            target_rate_all_seasons=0.0,

            grade="UNVERIFIED",

            eligible_by_history=True,

            notes=[

                "historique MODEL_READY indisponible ; "

                "le scanner LIVE décidera selon sa couverture réelle"

            ],

        )



    same_raw = [

        row

        for row in raw_rows

        if row_match_competition(

            row,

            fixture.competition_id,

            fixture.season_id,

        )

    ]

    same_core = [

        row

        for row in core_rows

        if row_match_competition(

            row,

            fixture.competition_id,

            fixture.season_id,

        )

    ]

    all_raw = [

        row

        for row in raw_rows

        if row_match_competition(

            row,

            fixture.competition_id,

            None,

        )

    ]

    all_core = [

        row

        for row in core_rows

        if row_match_competition(

            row,

            fixture.competition_id,

            None,

        )

    ]



    same_raw_n = len(same_raw)

    same_core_n = len(same_core)

    all_raw_n = len(all_raw)

    all_core_n = len(all_core)



    same_rate = safe_rate(same_core_n, same_raw_n)

    all_rate = safe_rate(all_core_n, all_raw_n)



    if same_raw_n >= SCHEDULER_MIN_RAW_MATCHES:

        scope_used = "same_competition_same_season"

        ref_raw_n = same_raw_n

        ref_core_n = same_core_n

        ref_rate = same_rate

    else:

        scope_used = "same_competition_all_seasons"

        ref_raw_n = all_raw_n

        ref_core_n = all_core_n

        ref_rate = all_rate



    grade = grade_capability_v2(

        ref_raw_n,

        ref_core_n,

        ref_rate,

        min_core_rate,

        min_core_rows,

    )



    notes: List[str] = []

    if same_raw_n < SCHEDULER_MIN_RAW_MATCHES:

        notes.append("peu d'historique même saison ; fallback toutes saisons")

    if all_raw_n == 0:

        notes.append("compétition inconnue dans le dataset historique")

    if ref_core_n < min_core_rows:

        notes.append(f"MODEL_READY insuffisant : {ref_core_n} < {min_core_rows}")

    if ref_rate < min_core_rate and ref_raw_n > 0:

        notes.append(

            f"taux MODEL_READY faible : {ref_rate:.1%} < {min_core_rate:.1%}"

        )



    positives, negatives = count_labeled_targets(all_core)

    target_rate = safe_rate(positives, positives + negatives)



    return CompetitionCapabilityV2(

        competition_id=fixture.competition_id,

        season_id=fixture.season_id,

        competition_name=fixture.competition_name,

        country=fixture.country,

        scope_used=scope_used,

        raw_rows_same_season=same_raw_n,

        core_ready_rows_same_season=same_core_n,

        core_ready_rate_same_season=same_rate,

        raw_rows_all_seasons=all_raw_n,

        core_ready_rows_all_seasons=all_core_n,

        core_ready_rate_all_seasons=all_rate,

        positives_all_seasons=positives,

        negatives_all_seasons=negatives,

        target_rate_all_seasons=target_rate,

        grade=grade,

        eligible_by_history=grade in {"EXCELLENT", "GOOD", "WATCH"},

        notes=notes,

    )





def resolve_history_rows() -> Tuple[

    List[Dict[str, str]],

    List[Mapping[str, Any]],

    List[str],

    str,

    bool,

]:

    """

    Retourne RAW, MODEL_READY, features modèle, source affichable, disponibilité.



    Priorité :

    - RAW + schéma modèle => MODEL_READY recalculé selon le scanner actuel.

    - Sinon CSV CORE_READY/FIXED existant.

    - Sinon history_filter_available=False (fail-open contrôlé).

    """

    raw_rows = read_csv_rows(RAW_DATASET_PATH)

    model_features, schema_path = resolve_model_schema()



    if raw_rows and model_features:

        raw_columns = set(raw_rows[0].keys())

        missing_schema_columns = [

            feature

            for feature in model_features

            if feature not in raw_columns

        ]



        if not missing_schema_columns:

            model_ready_rows = derive_model_ready_rows(

                raw_rows,

                model_features,

            )

            source = (

                f"RAW dérivé avec schéma modèle "

                f"{schema_path.name if schema_path else '?'}"

            )

            return raw_rows, model_ready_rows, model_features, source, True



        logger.warning(

            "RAW incompatible avec le schéma modèle | colonnes modèle absentes=%d/%d | "

            "exemples=%s",

            len(missing_schema_columns),

            len(model_features),

            ", ".join(missing_schema_columns[:12]),

        )



    core_path = resolve_core_ready_dataset_path()

    if core_path is not None:

        core_rows = read_csv_rows(core_path)

        if core_rows:

            # Si RAW manque, le taux ne peut pas être estimé correctement.

            # On utilise CORE comme RAW de référence uniquement pour éviter

            # un faux 0/0 ; le grade sera alors fondé sur le volume disponible.

            reference_raw = raw_rows if raw_rows else list(core_rows)

            source = f"CSV fallback : {core_path.name}"

            logger.warning(

                "MODEL_READY dérivé du RAW indisponible ; fallback CSV : %s",

                core_path,

            )

            return reference_raw, core_rows, model_features, source, True



    logger.error(

        "Aucune source historique MODEL_READY exploitable. "

        "Le scheduler passe en mode scanner-authoritative et ne rejettera "

        "pas les matchs uniquement à cause de ce problème."

    )

    return raw_rows, [], model_features, "indisponible", False





def build_capabilities_for_fixtures(

    fixtures: Sequence[Fixture],

    min_core_rate: float,

    min_core_rows: int,

) -> Tuple[

    Dict[Tuple[str, str], CompetitionCapabilityV2],

    str,

    bool,

    List[str],

]:

    (

        raw_rows,

        core_rows,

        model_features,

        history_source,

        history_filter_available,

    ) = resolve_history_rows()



    logger.info(

        "Dataset V2 scheduler | RAW=%d | MODEL_READY=%d | source=%s",

        len(raw_rows),

        len(core_rows),

        history_source,

    )



    grouped: Dict[Tuple[str, str], Fixture] = {}

    for fixture in fixtures:

        grouped.setdefault(

            (fixture.competition_id, fixture.season_id),

            fixture,

        )



    capabilities: Dict[Tuple[str, str], CompetitionCapabilityV2] = {}



    for key, fixture in grouped.items():

        cap = build_competition_capability_v2(

            fixture=fixture,

            raw_rows=raw_rows,

            core_rows=core_rows,

            min_core_rate=min_core_rate,

            min_core_rows=min_core_rows,

            history_filter_available=history_filter_available,

        )

        capabilities[key] = cap



        logger.info(

            "Capability V2 | %s | %s | same=%d/%d %.1f%% | "

            "all=%d/%d %.1f%% | %s",

            cap.competition_id,

            cap.competition_name or "nom inconnu",

            cap.core_ready_rows_same_season,

            cap.raw_rows_same_season,

            cap.core_ready_rate_same_season * 100.0,

            cap.core_ready_rows_all_seasons,

            cap.raw_rows_all_seasons,

            cap.core_ready_rate_all_seasons * 100.0,

            cap.grade,

        )



    return (

        capabilities,

        history_source,

        history_filter_available,

        model_features,

    )





# =============================================================================

# PLANNING

# =============================================================================



def criteria_for_mode(mode: str) -> Tuple[float, int]:

    mode = mode.strip().lower()

    if mode == "strict":

        return 0.50, 10

    if mode == "loose":

        return 0.25, 3

    return SCHEDULER_MIN_CORE_READY_RATE, SCHEDULER_MIN_CORE_READY_MATCHES





def resolve_odds_requirement(model_features: Sequence[str]) -> bool:

    policy = SCHEDULER_ODDS_POLICY



    if policy in {"required", "require", "on", "1", "true", "yes"}:

        return True



    if policy in {"off", "0", "false", "no", "disabled"}:

        return False



    if policy != "auto":

        logger.warning(

            "SCHEDULER_ODDS_POLICY inconnue=%r ; AUTO utilisé.",

            policy,

        )



    odds_used = sorted(set(model_features) & ODDS_FEATURES)



    if odds_used:

        logger.info(

            "Odds scheduler AUTO : REQUIRED | features modèle=%s",

            ", ".join(odds_used),

        )

        return True



    logger.info(

        "Odds scheduler AUTO : NOT REQUIRED | "

        "aucune feature odds dans le schéma modèle chargé"

    )

    return False





def plan_match(

    fixture: Fixture,

    capability: CompetitionCapabilityV2,

    include_unknown: bool,

    require_target_odds: bool,

) -> PlannedMatch:

    launch_at = fixture.kickoff_local + timedelta(minutes=SCANNER_START_OFFSET_MINUTES)

    expected_85_at = fixture.kickoff_local + timedelta(minutes=102)

    expected_87_at = fixture.kickoff_local + timedelta(minutes=104)

    stop_after = fixture.kickoff_local + timedelta(minutes=SCANNER_END_OFFSET_MINUTES)



    reasons: List[str] = []



    if capability.grade == "UNKNOWN":

        if not include_unknown:

            reasons.append("compétition inconnue dans le dataset historique")

    elif not capability.eligible_by_history:

        reasons.append(f"historique insuffisant ({capability.grade})")



    if require_target_odds and not fixture.odds_available:

        reasons.append(

            "odds_available=false alors que le modèle/planning exige des odds"

        )



    return PlannedMatch(

        fixture=fixture,

        capability=capability,

        launch_at=launch_at,

        expected_85_at=expected_85_at,

        expected_87_at=expected_87_at,

        stop_after=stop_after,

        eligible=not reasons,

        rejection_reasons=reasons,

    )





def build_sessions(planned_matches: Sequence[PlannedMatch]) -> List[Session]:

    eligible = sorted((m for m in planned_matches if m.eligible), key=lambda x: x.launch_at)

    if not eligible:

        return []



    sessions: List[Session] = []

    current_matches = [eligible[0]]

    current_start = eligible[0].launch_at

    current_end = eligible[0].stop_after



    for match in eligible[1:]:

        merge_limit = current_end + timedelta(minutes=SESSION_MERGE_GAP_MINUTES)

        if match.launch_at <= merge_limit:

            current_matches.append(match)

            current_end = max(current_end, match.stop_after)

            continue



        sessions.append(Session(

            session_id=len(sessions) + 1,

            start_at=current_start,

            reminder_at=current_start - timedelta(minutes=REMINDER_LEAD_MINUTES),

            end_at=current_end,

            matches=current_matches,

        ))

        current_matches = [match]

        current_start = match.launch_at

        current_end = match.stop_after



    sessions.append(Session(

        session_id=len(sessions) + 1,

        start_at=current_start,

        reminder_at=current_start - timedelta(minutes=REMINDER_LEAD_MINUTES),

        end_at=current_end,

        matches=current_matches,

    ))



    return sessions





# =============================================================================

# OUTPUT

# =============================================================================



def print_planning(

    target_date: date,

    planned_matches: Sequence[PlannedMatch],

    sessions: Sequence[Session],

    show_rejected: bool,

    mode: str,

    min_core_rate: float,

    min_core_rows: int,

    include_unknown: bool,

    require_target_odds: bool,

    history_source: str,

    history_filter_available: bool,

    model_feature_count: int,

) -> None:

    print_header(f"MATCH SCHEDULER V3-C - {target_date.isoformat()} ({SCHEDULER_TIMEZONE})")



    eligible = [m for m in planned_matches if m.eligible]

    rejected = [m for m in planned_matches if not m.eligible]



    print(f"\nMatchs trouvés : {len(planned_matches)}")

    print(f"Matchs retenus : {len(eligible)}")

    print(f"Matchs rejetés : {len(rejected)}")

    print(f"Mode : {mode}")

    print(

        f"Critère historique : MODEL_READY rate >= {min_core_rate:.0%} | "

        f"MODEL_READY rows >= {min_core_rows}"

    )

    print(f"Schéma modèle détecté : {model_feature_count} feature(s)")

    print(f"Source historique : {history_source}")

    print(

        "Filtre historique : "

        + ("ACTIF" if history_filter_available else "INDISPONIBLE → scanner LIVE autoritaire")

    )

    print(f"Compétitions inconnues incluses : {'YES' if include_unknown else 'NO'}")

    print(f"Odds obligatoires : {'YES' if require_target_odds else 'NO'}")

    print(f"Zone V3-C prioritaire : {MODEL_ALERT_PROBABILITY * 100.0:.1f}% <= P < {MODEL_ALERT_PROBABILITY_MAX * 100.0:.1f}%")



    if sessions:

        print_header("SESSIONS RECOMMANDÉES")

        for session in sessions:

            print(f"\nSESSION {session.session_id}")

            print(f"  🔔 rappel       : {fmt_time(session.reminder_at)}")

            print(f"  🚀 lancer       : {fmt_time(session.start_at)}")

            print(f"  ⏹ fin indicative: {fmt_time(session.end_at)}")

            print(f"  Commande        : {SCANNER_COMMAND}")



            for match in session.matches:

                fixture = match.fixture

                cap = match.capability

                print(f"    {fmt_time(fixture.kickoff_local)} | {fixture.home_team} - {fixture.away_team}")

                print(

                    f"       {cap.competition_name or fixture.competition_name} | {cap.grade} | "

                    f"MODEL_READY same={cap.core_ready_rows_same_season}/{cap.raw_rows_same_season} "

                    f"({cap.core_ready_rate_same_season:.1%}) | "

                    f"all={cap.core_ready_rows_all_seasons}/{cap.raw_rows_all_seasons} "

                    f"({cap.core_ready_rate_all_seasons:.1%}) | "

                    f"odds={'YES' if fixture.odds_available else 'NO'}"

                )

    else:

        print("\nAucune session ne respecte les critères.")



    if show_rejected and rejected:

        print_header("MATCHS REJETÉS")

        for match in rejected:

            fixture = match.fixture

            cap = match.capability

            print(f"\n{fmt_time(fixture.kickoff_local)} | {fixture.home_team} - {fixture.away_team}")

            print(

                f"  {cap.competition_name or fixture.competition_name} | {cap.grade} | "

                f"same={cap.core_ready_rows_same_season}/{cap.raw_rows_same_season} "

                f"({cap.core_ready_rate_same_season:.1%}) | "

                f"all={cap.core_ready_rows_all_seasons}/{cap.raw_rows_all_seasons} "

                f"({cap.core_ready_rate_all_seasons:.1%})"

            )

            for reason in match.rejection_reasons:

                print(f"  ❌ {reason}")

            for note in cap.notes[:5]:

                print(f"  ⚠️ {note}")





def export_schedule_csv(target_date: date, planned_matches: Sequence[PlannedMatch]) -> Path:

    path = OUTPUT_DIR / f"schedule_v2_{target_date.isoformat()}.csv"

    columns = [

        "date_local", "kickoff_local", "launch_at", "expected_85_at", "expected_87_at", "stop_after",

        "match_id", "competition_id", "season_id", "competition_name", "country", "home_team", "away_team",

        "status", "odds_available", "live_odds_available", "xg_available",

        "capability_grade", "scope_used",

        "raw_rows_same_season", "core_ready_rows_same_season", "core_ready_rate_same_season",

        "raw_rows_all_seasons", "core_ready_rows_all_seasons", "core_ready_rate_all_seasons",

        "positives_all_seasons", "negatives_all_seasons", "target_rate_all_seasons",

        "eligible", "rejection_reasons", "notes",

    ]



    with path.open("w", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=columns)

        writer.writeheader()

        for planned in sorted(planned_matches, key=lambda x: x.fixture.kickoff_local):

            f = planned.fixture

            c = planned.capability

            writer.writerow({

                "date_local": f.kickoff_local.date().isoformat(),

                "kickoff_local": f.kickoff_local.isoformat(),

                "launch_at": planned.launch_at.isoformat(),

                "expected_85_at": planned.expected_85_at.isoformat(),

                "expected_87_at": planned.expected_87_at.isoformat(),

                "stop_after": planned.stop_after.isoformat(),

                "match_id": f.match_id,

                "competition_id": f.competition_id,

                "season_id": f.season_id,

                "competition_name": c.competition_name or f.competition_name,

                "country": c.country or f.country,

                "home_team": f.home_team,

                "away_team": f.away_team,

                "status": f.status,

                "odds_available": f.odds_available,

                "live_odds_available": f.live_odds_available,

                "xg_available": f.xg_available,

                "capability_grade": c.grade,

                "scope_used": c.scope_used,

                "raw_rows_same_season": c.raw_rows_same_season,

                "core_ready_rows_same_season": c.core_ready_rows_same_season,

                "core_ready_rate_same_season": c.core_ready_rate_same_season,

                "raw_rows_all_seasons": c.raw_rows_all_seasons,

                "core_ready_rows_all_seasons": c.core_ready_rows_all_seasons,

                "core_ready_rate_all_seasons": c.core_ready_rate_all_seasons,

                "positives_all_seasons": c.positives_all_seasons,

                "negatives_all_seasons": c.negatives_all_seasons,

                "target_rate_all_seasons": c.target_rate_all_seasons,

                "eligible": planned.eligible,

                "rejection_reasons": " | ".join(planned.rejection_reasons),

                "notes": " | ".join(c.notes),

            })

    return path





def export_sessions_json(target_date: date, sessions: Sequence[Session]) -> Path:

    path = OUTPUT_DIR / f"sessions_v2_{target_date.isoformat()}.json"

    payload = []

    for session in sessions:

        payload.append({

            "session_id": session.session_id,

            "reminder_at": session.reminder_at.isoformat(),

            "start_at": session.start_at.isoformat(),

            "end_at": session.end_at.isoformat(),

            "scanner_command": SCANNER_COMMAND,

            "alert_probability": MODEL_ALERT_PROBABILITY,

            "matches": [

                {

                    "match_id": m.fixture.match_id,

                    "kickoff_local": m.fixture.kickoff_local.isoformat(),

                    "expected_87_at": m.expected_87_at.isoformat(),

                    "competition_id": m.fixture.competition_id,

                    "competition": m.capability.competition_name,

                    "home_team": m.fixture.home_team,

                    "away_team": m.fixture.away_team,

                    "grade": m.capability.grade,

                    "core_ready_rows_all_seasons": m.capability.core_ready_rows_all_seasons,

                    "core_ready_rate_all_seasons": m.capability.core_ready_rate_all_seasons,

                    "odds_available": m.fixture.odds_available,

                }

                for m in session.matches

            ],

        })

    save_json_file(path, payload)

    return path





# =============================================================================

# NOTIFICATIONS

# =============================================================================



def telegram_configured() -> bool:

    return bool(TELEGRAM_NOTIFICATIONS_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)





def send_telegram(text: str) -> bool:

    if not telegram_configured():

        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:

        response = requests.post(

            url,

            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": "true"},

            timeout=TELEGRAM_API_TIMEOUT,

        )

        response.raise_for_status()

        payload = response.json()

        return bool(isinstance(payload, Mapping) and payload.get("ok") is True)

    except Exception as exc:

        logger.warning("Telegram reminder failed : %s", exc)

        return False





def send_macos_notification(title: str, message: str) -> bool:

    if not MACOS_NOTIFICATIONS_ENABLED or sys.platform != "darwin":

        return False

    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')

    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'

    try:

        subprocess.run(["osascript", "-e", script], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return True

    except Exception as exc:

        logger.warning("Notification macOS failed : %s", exc)

        return False





def session_match_lines(session: Session) -> List[str]:

    lines = []

    for m in session.matches:

        lines.append(

            f"• {fmt_time(m.fixture.kickoff_local)} {m.fixture.home_team} - {m.fixture.away_team} "

            f"({m.capability.grade}, CORE {m.capability.core_ready_rows_all_seasons}/{m.capability.raw_rows_all_seasons})"

        )

    return lines





def send_session_reminder(session: Session, kind: str) -> None:

    if kind == "prepare":

        title = "⚽ Goal After 87 V3-C - Prépare le scanner"

        text = "\n".join([

            "⚽ GOAL AFTER 87 V3-C", "",

            f"Scanner à lancer dans {REMINDER_LEAD_MINUTES} min.",

            f"Heure de lancement : {fmt_time(session.start_at)}",

            f"Zone V3-C : {MODEL_ALERT_PROBABILITY * 100.0:.1f}% <= P < {MODEL_ALERT_PROBABILITY_MAX * 100.0:.1f}%", "",

            *session_match_lines(session), "",

            f"Commande : {SCANNER_COMMAND}",

        ])

        mac_message = f"Scanner à lancer à {fmt_time(session.start_at)} ({len(session.matches)} match(s))"

    else:

        title = "🚨 Goal After 87 V3-C - LANCE LE SCANNER"

        text = "\n".join([

            "🚨 LANCE LE SCANNER MAINTENANT", "",

            f"Zone V3-C : {MODEL_ALERT_PROBABILITY * 100.0:.1f}% <= P < {MODEL_ALERT_PROBABILITY_MAX * 100.0:.1f}%", "",

            *session_match_lines(session), "",

            f"Commande : {SCANNER_COMMAND}", "",

            f"Fin indicative : {fmt_time(session.end_at)}",

        ])

        mac_message = f"Lance le scanner maintenant ({len(session.matches)} match(s))"

    send_telegram(text)

    send_macos_notification(title, mac_message)





def load_notification_state() -> Dict[str, Any]:

    payload = load_json_file(NOTIFICATION_STATE_PATH, {})

    return payload if isinstance(payload, dict) else {}





def notification_key(target_date: date, session_id: int, kind: str) -> str:

    return f"{target_date.isoformat()}|session_{session_id}|{kind}"





def watch_sessions(target_date: date, sessions: Sequence[Session]) -> None:

    if not sessions:

        logger.info("WATCH : aucune session.")

        return



    print_header("WATCH MODE V3-C")

    print("Aucun nouvel appel TheStatsAPI ne sera fait pendant l'attente.")

    print("Laisse ce terminal ouvert / Mac réveillé.")



    state = load_notification_state()

    final_end = max(session.end_at for session in sessions)



    while True:

        now = datetime.now(LOCAL_TZ)

        for session in sessions:

            prepare_key = notification_key(target_date, session.session_id, "prepare")

            launch_key = notification_key(target_date, session.session_id, "launch")



            if now >= session.reminder_at and prepare_key not in state:

                send_session_reminder(session, "prepare")

                state[prepare_key] = datetime.now(timezone.utc).isoformat()

                save_json_file(NOTIFICATION_STATE_PATH, state)

                logger.info("Rappel PREPARE envoyé | session %d", session.session_id)



            if now >= session.start_at and launch_key not in state:

                send_session_reminder(session, "launch")

                state[launch_key] = datetime.now(timezone.utc).isoformat()

                save_json_file(NOTIFICATION_STATE_PATH, state)

                logger.info("Rappel LAUNCH envoyé | session %d", session.session_id)



        if now > final_end + timedelta(minutes=5):

            logger.info("WATCH terminé pour %s.", target_date.isoformat())

            break



        time.sleep(15)





def send_daily_plan_summary(target_date: date, sessions: Sequence[Session]) -> bool:

    if not sessions:

        return False



    lines = [

        f"📅 GOAL AFTER 87 V3-C - {target_date.isoformat()}",

        "",

        f"{len(sessions)} session(s) prévue(s)",

        f"Zone V3-C : {MODEL_ALERT_PROBABILITY * 100.0:.1f}% <= P < {MODEL_ALERT_PROBABILITY_MAX * 100.0:.1f}%",

        "",

    ]



    for session in sessions:

        lines.extend([

            f"SESSION {session.session_id} — lancer {fmt_time(session.start_at)}",

            f"rappel {fmt_time(session.reminder_at)} | fin ≈ {fmt_time(session.end_at)}",

        ])

        lines.extend(session_match_lines(session))

        lines.append("")



    lines.extend([

        f"Commande : {SCANNER_COMMAND}",

        "",

        "Le scanner reste responsable de la validation finale.",

    ])

    return send_telegram("\n".join(lines))





# =============================================================================

# CLI / MAIN

# =============================================================================



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Planifie les fenêtres de lancement du scanner Goal After 87 V3-C.")

    parser.add_argument("--date", help="Date locale YYYY-MM-DD. Défaut : demain.")

    parser.add_argument("--mode", choices=["strict", "balanced", "loose"], default="balanced")

    parser.add_argument("--min-core-rate", type=float, default=None, help="Taux minimum MODEL_READY historique. Exemple: 0.35")

    parser.add_argument("--min-core-rows", type=int, default=None, help="Minimum de matchs MODEL_READY historiques.")

    parser.add_argument("--include-unknown", action="store_true", help="Inclut les compétitions inconnues dans le dataset historique.")

    parser.add_argument("--watch", action="store_true", help="Attend les sessions et envoie les rappels Telegram/macOS.")

    parser.add_argument("--show-rejected", action="store_true", help="Affiche les matchs rejetés.")

    parser.add_argument("--send-plan", action="store_true", help="Envoie immédiatement le planning sur Telegram.")

    return parser.parse_args()





def resolve_target_date(raw: Optional[str]) -> date:

    if raw:

        try:

            return date.fromisoformat(raw)

        except Exception as exc:

            raise SystemExit("--date doit être au format YYYY-MM-DD") from exc

    return datetime.now(LOCAL_TZ).date() + timedelta(days=1)





def main() -> int:

    args = parse_args()

    target_date = resolve_target_date(args.date)



    default_rate, default_rows = criteria_for_mode(args.mode)

    min_core_rate = float(args.min_core_rate) if args.min_core_rate is not None else default_rate

    min_core_rows = int(args.min_core_rows) if args.min_core_rows is not None else default_rows

    include_unknown = bool(args.include_unknown) or INCLUDE_UNKNOWN_COMPETITIONS_DEFAULT



    if not (0.0 <= min_core_rate <= 1.0):

        raise SystemExit("--min-core-rate doit être entre 0 et 1.")

    if min_core_rows < 0:

        raise SystemExit("--min-core-rows doit être >= 0.")



    print_header("MATCH SCHEDULER V3-C - GOAL AFTER 87'")

    logger.info(".env chargé : %s", ENV_PATH)

    logger.info("Timezone : %s", SCHEDULER_TIMEZONE)

    logger.info("Date ciblée : %s", target_date.isoformat())

    logger.info("Mode : %s", args.mode)

    logger.info(

        "Critère MODEL_READY : rate >= %.1f%% | rows >= %d",

        min_core_rate * 100.0,

        min_core_rows,

    )

    logger.info("Zone V3-C prioritaire : %.1f%% <= P < %.1f%%", MODEL_ALERT_PROBABILITY * 100.0, MODEL_ALERT_PROBABILITY_MAX * 100.0)

    logger.info("Scanner start offset : +%d min", SCANNER_START_OFFSET_MINUTES)

    logger.info("Telegram scheduler : %s", "ON" if telegram_configured() else "OFF / non configuré")



    client = StatsAPIClient(api_key=STATS_API_KEY, base_url=STATS_API_BASE_URL, timeout=STATS_API_TIMEOUT)

    fixtures = fetch_fixtures_for_local_date(client, target_date)



    if not fixtures:

        print(f"\nAucun match à venir trouvé pour {target_date}.")

        print(f"Requêtes TheStatsAPI utilisées : {client.requests_total}")

        return 0



    (

        capabilities,

        history_source,

        history_filter_available,

        model_features,

    ) = build_capabilities_for_fixtures(

        fixtures,

        min_core_rate,

        min_core_rows,

    )



    require_target_odds = resolve_odds_requirement(model_features)



    logger.info(

        "Schéma V3-C planning : %d feature(s) | couverture stricte attendue=38/38",

        len(model_features),

        min(MIN_MODEL_COVERAGE, len(model_features)) if model_features else 0,

    )

    logger.info(

        "Odds planning : %s | policy=%s",

        "REQUIRED" if require_target_odds else "NOT REQUIRED",

        SCHEDULER_ODDS_POLICY,

    )

    logger.info(

        "Historique planning : %s | filtre=%s",

        history_source,

        "ACTIF" if history_filter_available else "FAIL-OPEN",

    )



    planned_matches = []

    for fixture in fixtures:

        cap = capabilities[(fixture.competition_id, fixture.season_id)]

        planned_matches.append(

            plan_match(

                fixture,

                cap,

                include_unknown,

                require_target_odds,

            )

        )



    sessions = build_sessions(planned_matches)



    print_planning(

        target_date=target_date,

        planned_matches=planned_matches,

        sessions=sessions,

        show_rejected=args.show_rejected,

        mode=args.mode,

        min_core_rate=min_core_rate,

        min_core_rows=min_core_rows,

        include_unknown=include_unknown,

        require_target_odds=require_target_odds,

        history_source=history_source,

        history_filter_available=history_filter_available,

        model_feature_count=len(model_features),

    )



    csv_path = export_schedule_csv(target_date, planned_matches)

    json_path = export_sessions_json(target_date, sessions)



    print_header("FICHIERS")

    print(f"Planning CSV : {csv_path}")

    print(f"Sessions JSON: {json_path}")

    print(f"\nRequêtes TheStatsAPI utilisées cette exécution : {client.requests_total}")



    if args.send_plan:

        if send_daily_plan_summary(target_date, sessions):

            print("Planning Telegram envoyé.")

        else:

            print("Planning Telegram non envoyé (aucune session ou Telegram indisponible).")



    if args.watch:

        watch_sessions(target_date, sessions)



    return 0





if __name__ == "__main__":

    raise SystemExit(main())
