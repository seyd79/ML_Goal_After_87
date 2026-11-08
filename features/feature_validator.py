#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FEATURE VALIDATOR
=================

Validation stricte du vecteur de features utilisé par le modèle
Goal After 87.

Objectifs :
- vérifier que les 96 features attendues sont présentes ;
- détecter les features manquantes ;
- détecter les valeurs None / NaN / inf ;
- détecter les valeurs non numériques ;
- détecter certaines valeurs manifestement invalides ;
- mesurer la couverture PRE-MATCH / LIVE ;
- produire un rapport exploitable par le scanner ;
- NE JAMAIS remplacer automatiquement une donnée manquante.

Important :
Une feature manquante n'est PAS automatiquement égale à 0.

Le validator ne modifie jamais les données.
Il contrôle uniquement leur qualité.
"""

import math
from typing import Any, Dict, List, Optional


# =============================================================================
# FEATURES ATTENDUES PAR LE MODÈLE
# =============================================================================

PRE_MATCH_FEATURES = [
    "home_team_rank",
    "away_team_rank",

    "home_team_points_standing",
    "away_team_points_standing",

    "home_team_matches_played",
    "away_team_matches_played",

    "home_team_wins",
    "away_team_wins",

    "home_team_draws",
    "away_team_draws",

    "home_team_losses",
    "away_team_losses",

    "home_team_goals_for",
    "away_team_goals_for",

    "home_team_goals_against",
    "away_team_goals_against",

    "home_team_goal_difference",
    "away_team_goal_difference",

    "home_position",
    "away_position",

    "home_points",
    "away_points",

    "home_form5",
    "away_form5",

    "home_form10",
    "away_form10",

    "home_scored_freq",
    "away_scored_freq",

    "home_wins_freq",
    "away_wins_freq",

    "home_avg_goals_scored",
    "away_avg_goals_scored",

    "home_avg_goals_conceded",
    "away_avg_goals_conceded",

    "home_avg_xg",
    "away_avg_xg",

    "home_btts_freq",
    "away_btts_freq",

    "home_over25_freq",
    "away_over25_freq",

    "h2h5_over25_freq",
    "h2h5_btts_freq",

    "league_home_over25_freq",
    "league_away_over25_freq",

    "league_home_over05_freq",
    "league_away_over05_freq",

    "league_home_btts_freq",
    "league_away_btts_freq",

    "league_2nd_half_over15_freq",

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
]


LIVE_FEATURES = [
    "score_87_home",
    "score_87_away",
    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",

    "home_shots_87",
    "away_shots_87",

    "home_shots_on_target_87",
    "away_shots_on_target_87",

    "home_shots_off_target_87",
    "away_shots_off_target_87",

    "home_shots_blocked_87",
    "away_shots_blocked_87",

    "home_corners_87",
    "away_corners_87",

    "home_fouls_87",
    "away_fouls_87",

    "home_yellow_cards_87",
    "away_yellow_cards_87",

    "home_red_cards_87",
    "away_red_cards_87",

    "home_offsides_87",
    "away_offsides_87",

    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",

    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",

    "minutes_since_last_goal",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "minutes_since_last_corner",
]


ALL_FEATURES = PRE_MATCH_FEATURES + LIVE_FEATURES


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

EXPECTED_PRE_MATCH_COUNT = 59
EXPECTED_LIVE_COUNT = 37
EXPECTED_TOTAL_COUNT = 96


if len(PRE_MATCH_FEATURES) != EXPECTED_PRE_MATCH_COUNT:
    raise RuntimeError(
        f"ERREUR SCHEMA : PRE_MATCH_FEATURES contient "
        f"{len(PRE_MATCH_FEATURES)} features au lieu de "
        f"{EXPECTED_PRE_MATCH_COUNT}."
    )


if len(LIVE_FEATURES) != EXPECTED_LIVE_COUNT:
    raise RuntimeError(
        f"ERREUR SCHEMA : LIVE_FEATURES contient "
        f"{len(LIVE_FEATURES)} features au lieu de "
        f"{EXPECTED_LIVE_COUNT}."
    )


if len(ALL_FEATURES) != EXPECTED_TOTAL_COUNT:
    raise RuntimeError(
        f"ERREUR SCHEMA : ALL_FEATURES contient "
        f"{len(ALL_FEATURES)} features au lieu de "
        f"{EXPECTED_TOTAL_COUNT}."
    )


if len(set(ALL_FEATURES)) != len(ALL_FEATURES):
    raise RuntimeError(
        "ERREUR SCHEMA : doublon détecté dans les features."
    )


# =============================================================================
# UTILITAIRES
# =============================================================================

def is_missing(value: Any) -> bool:
    """
    Retourne True si la valeur est considérée comme manquante.

    Important :
    - None => manquant
    - NaN => manquant
    - inf/-inf => invalide/manquant
    """

    if value is None:
        return True

    if isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return (
        math.isnan(numeric_value)
        or math.isinf(numeric_value)
    )


def is_numeric(value: Any) -> bool:
    """
    Vérifie qu'une valeur est numérique et finie.
    """

    if value is None:
        return False

    if isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return (
        not math.isnan(numeric_value)
        and not math.isinf(numeric_value)
    )


def safe_float(value: Any) -> Optional[float]:
    """
    Conversion numérique sécurisée.
    """

    if not is_numeric(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# VALIDATION STRUCTURELLE
# =============================================================================

def validate_schema(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Vérifie la structure du dictionnaire.

    Contrôle :
    - features supplémentaires ;
    - features manquantes ;
    - nombre total de features ;
    - ordre non pertinent pour un dictionnaire.

    Retourne un rapport.
    """

    if not isinstance(features, dict):

        return {
            "valid": False,
            "reason": "features_not_dict",
            "expected_total": EXPECTED_TOTAL_COUNT,
            "actual_total": 0,
            "missing": ALL_FEATURES.copy(),
            "extra": [],
        }

    feature_keys = set(features.keys())
    expected_keys = set(ALL_FEATURES)

    missing = [
        feature
        for feature in ALL_FEATURES
        if feature not in feature_keys
    ]

    extra = [
        feature
        for feature in features.keys()
        if feature not in expected_keys
    ]

    valid = (
        len(missing) == 0
        and len(extra) == 0
        and len(features) == EXPECTED_TOTAL_COUNT
    )

    return {
        "valid": valid,
        "reason": None if valid else "schema_mismatch",
        "expected_total": EXPECTED_TOTAL_COUNT,
        "actual_total": len(features),
        "missing": missing,
        "extra": extra,
    }


# =============================================================================
# VALIDATION DES VALEURS
# =============================================================================

def validate_values(
    features: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Vérifie les valeurs des features.

    Catégories :
    - missing : None / NaN / inf
    - non_numeric : texte ou type incompatible
    - valid : valeur numérique finie
    """

    missing = []
    non_numeric = []
    valid = []

    for feature in feature_names:

        if feature not in features:

            missing.append(feature)
            continue

        value = features.get(feature)

        if is_missing(value):

            missing.append(feature)
            continue

        if not is_numeric(value):

            non_numeric.append(feature)
            continue

        valid.append(feature)

    return {
        "missing": missing,
        "non_numeric": non_numeric,
        "valid": valid,
        "missing_count": len(missing),
        "non_numeric_count": len(non_numeric),
        "valid_count": len(valid),
    }


# =============================================================================
# VALIDATION DES FRÉQUENCES
# =============================================================================

FREQUENCY_FEATURES = [
    "home_scored_freq",
    "away_scored_freq",

    "home_wins_freq",
    "away_wins_freq",

    "home_btts_freq",
    "away_btts_freq",

    "home_over25_freq",
    "away_over25_freq",

    "h2h5_over25_freq",
    "h2h5_btts_freq",

    "league_home_over25_freq",
    "league_away_over25_freq",

    "league_home_over05_freq",
    "league_away_over05_freq",

    "league_home_btts_freq",
    "league_away_btts_freq",

    "league_2nd_half_over15_freq",
]


def validate_frequencies(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Vérifie que les fréquences sont dans [0, 1].

    Les valeurs manquantes ne sont pas considérées
    comme une erreur de range : elles sont traitées
    séparément comme missing.
    """

    invalid = []

    for feature in FREQUENCY_FEATURES:

        if feature not in features:
            continue

        value = features.get(feature)

        if is_missing(value):
            continue

        numeric_value = safe_float(value)

        if numeric_value is None:
            continue

        if not 0.0 <= numeric_value <= 1.0:

            invalid.append({
                "feature": feature,
                "value": numeric_value,
                "reason": "frequency_out_of_range",
            })

    return {
        "valid": len(invalid) == 0,
        "invalid": invalid,
        "invalid_count": len(invalid),
    }


# =============================================================================
# VALIDATION DES ODDS
# =============================================================================

ODDS_FEATURES = [
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
]


def validate_odds(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Vérifie les cotes.

    Règles :
    - une cote disponible doit être > 1.0 ;
    - une cote manquante reste manquante ;
    - aucune valeur n'est remplacée.
    """

    invalid = []

    for feature in ODDS_FEATURES:

        if feature not in features:
            continue

        value = features.get(feature)

        if is_missing(value):
            continue

        numeric_value = safe_float(value)

        if numeric_value is None:
            continue

        if numeric_value <= 1.0:

            invalid.append({
                "feature": feature,
                "value": numeric_value,
                "reason": "odds_must_be_greater_than_1",
            })

    return {
        "valid": len(invalid) == 0,
        "invalid": invalid,
        "invalid_count": len(invalid),
    }


# =============================================================================
# VALIDATION DES COMPTEURS
# =============================================================================

NON_NEGATIVE_FEATURES = [
    "home_team_matches_played",
    "away_team_matches_played",

    "home_team_wins",
    "away_team_wins",

    "home_team_draws",
    "away_team_draws",

    "home_team_losses",
    "away_team_losses",

    "home_team_goals_for",
    "away_team_goals_for",

    "home_team_goals_against",
    "away_team_goals_against",

    "home_points",
    "away_points",

    "home_avg_goals_scored",
    "away_avg_goals_scored",

    "home_avg_goals_conceded",
    "away_avg_goals_conceded",

    "home_avg_xg",
    "away_avg_xg",

    "score_87_home",
    "score_87_away",

    "home_shots_87",
    "away_shots_87",

    "home_shots_on_target_87",
    "away_shots_on_target_87",

    "home_shots_off_target_87",
    "away_shots_off_target_87",

    "home_shots_blocked_87",
    "away_shots_blocked_87",

    "home_corners_87",
    "away_corners_87",

    "home_fouls_87",
    "away_fouls_87",

    "home_yellow_cards_87",
    "away_yellow_cards_87",

    "home_red_cards_87",
    "away_red_cards_87",

    "home_offsides_87",
    "away_offsides_87",

    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",
]


def validate_non_negative(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Vérifie les features qui ne peuvent normalement
    pas être négatives.
    """

    invalid = []

    for feature in NON_NEGATIVE_FEATURES:

        if feature not in features:
            continue

        value = features.get(feature)

        if is_missing(value):
            continue

        numeric_value = safe_float(value)

        if numeric_value is None:
            continue

        if numeric_value < 0:

            invalid.append({
                "feature": feature,
                "value": numeric_value,
                "reason": "negative_value",
            })

    return {
        "valid": len(invalid) == 0,
        "invalid": invalid,
        "invalid_count": len(invalid),
    }


# =============================================================================
# VALIDATION DES FEATURES LIVE
# =============================================================================

def validate_live_logic(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Vérifications de cohérence basiques des données LIVE.

    Ne force aucune valeur.

    Contrôles :
    - goal_diff_87 doit correspondre au score si les deux scores existent ;
    - is_draw_87 doit correspondre au score si les deux scores existent ;
    - leading_team_87 doit être cohérent avec le score.
    """

    errors = []

    home_score = safe_float(
        features.get("score_87_home")
    )

    away_score = safe_float(
        features.get("score_87_away")
    )

    goal_diff = safe_float(
        features.get("goal_diff_87")
    )

    is_draw = safe_float(
        features.get("is_draw_87")
    )

    leading_team = safe_float(
        features.get("leading_team_87")
    )

    # -------------------------------------------------------------------------
    # goal_diff_87
    # -------------------------------------------------------------------------

    if (
        home_score is not None
        and away_score is not None
        and goal_diff is not None
    ):

        # Convention du dataset / modèle de production :
        # différence de buts en valeur absolue.
        expected_goal_diff = abs(
            home_score - away_score
        )

        if goal_diff != expected_goal_diff:

            errors.append({
                "feature": "goal_diff_87",
                "value": goal_diff,
                "expected": expected_goal_diff,
                "reason": "score_goal_diff_inconsistent",
            })

    # -------------------------------------------------------------------------
    # is_draw_87
    # -------------------------------------------------------------------------

    if (
        home_score is not None
        and away_score is not None
        and is_draw is not None
    ):

        expected_is_draw = (
            1.0
            if home_score == away_score
            else 0.0
        )

        if is_draw != expected_is_draw:

            errors.append({
                "feature": "is_draw_87",
                "value": is_draw,
                "expected": expected_is_draw,
                "reason": "draw_flag_inconsistent",
            })

    # -------------------------------------------------------------------------
    # leading_team_87
    #
    # Convention du dataset / modèle de production :
    #   0 = home leads
    #   1 = away leads
    #   2 = draw
    # -------------------------------------------------------------------------

    if (
        home_score is not None
        and away_score is not None
        and leading_team is not None
    ):

        if home_score > away_score:
            expected_leading_team = 0.0

        elif away_score > home_score:
            expected_leading_team = 1.0

        else:
            expected_leading_team = 2.0

        if leading_team != expected_leading_team:

            errors.append({
                "feature": "leading_team_87",
                "value": leading_team,
                "expected": expected_leading_team,
                "reason": "leading_team_inconsistent",
            })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "error_count": len(errors),
    }


# =============================================================================
# VALIDATION COMPLÈTE
# =============================================================================

def validate_features(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validation complète des 96 features.

    Retourne un rapport détaillé.
    """

    schema = validate_schema(
        features
    )

    if not isinstance(features, dict):

        return {
            "valid": False,
            "ready_for_model": False,
            "schema": schema,
            "pre_match": None,
            "live": None,
            "frequencies": None,
            "odds": None,
            "non_negative": None,
            "live_logic": None,
            "coverage": {
                "total": EXPECTED_TOTAL_COUNT,
                "available": 0,
                "missing": EXPECTED_TOTAL_COUNT,
                "percentage": 0.0,
            },
        }

    pre_match = validate_values(
        features,
        PRE_MATCH_FEATURES,
    )

    live = validate_values(
        features,
        LIVE_FEATURES,
    )

    frequencies = validate_frequencies(
        features
    )

    odds = validate_odds(
        features
    )

    non_negative = validate_non_negative(
        features
    )

    live_logic = validate_live_logic(
        features
    )

    # -------------------------------------------------------------------------
    # Couverture
    # -------------------------------------------------------------------------

    available = 0

    missing = []

    for feature in ALL_FEATURES:

        if feature not in features:

            missing.append(feature)
            continue

        value = features.get(feature)

        if is_numeric(value):

            available += 1

        else:

            missing.append(feature)

    coverage = (
        available / EXPECTED_TOTAL_COUNT
        if EXPECTED_TOTAL_COUNT
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Erreurs
    # -------------------------------------------------------------------------

    has_errors = (
        not schema["valid"]
        or pre_match["non_numeric_count"] > 0
        or live["non_numeric_count"] > 0
        or not frequencies["valid"]
        or not odds["valid"]
        or not non_negative["valid"]
        or not live_logic["valid"]
    )

    # -------------------------------------------------------------------------
    # PRODUCTION GATE STRICT
    #
    # Le modèle de production ne doit être appelé que si les 96 features
    # sont réellement présentes, numériques, finies et cohérentes.
    # L'existence d'un SimpleImputer dans le pipeline d'entraînement ne
    # justifie pas l'invention ou l'acceptation de données absentes en LIVE.
    # -------------------------------------------------------------------------

    complete_coverage = (
        available == EXPECTED_TOTAL_COUNT
        and len(missing) == 0
        and pre_match["missing_count"] == 0
        and live["missing_count"] == 0
    )

    ready_for_model = (
        complete_coverage
        and not has_errors
    )

    return {
        "valid": (
            ready_for_model
        ),

        "status": (
            "OK"
            if ready_for_model
            else "FAIL"
        ),

        "ready_for_model": ready_for_model,

        # Champs plats de compatibilité avec le scanner.
        "available": available,
        "total": EXPECTED_TOTAL_COUNT,
        "missing": missing,

        "schema": schema,

        "pre_match": pre_match,

        "live": live,

        "frequencies": frequencies,

        "odds": odds,

        "non_negative": non_negative,

        "live_logic": live_logic,

        "coverage": {
            "total": EXPECTED_TOTAL_COUNT,
            "available": available,
            "missing": len(missing),
            "percentage": (
                coverage * 100.0
            ),
        },

        "missing_features": missing,
    }


# =============================================================================
# RAPPORT CONSOLE
# =============================================================================

def print_validation_report(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Affiche un rapport lisible dans la console.

    Retourne également le rapport complet.
    """

    report = validate_features(
        features
    )

    print()
    print("=" * 80)
    print("FEATURE VALIDATOR — GOAL AFTER 87")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    print()

    if report["valid"]:

        print("STATUS : OK")

    else:

        print("STATUS : FAIL")

    print(
        f"READY FOR MODEL : "
        f"{'YES' if report['ready_for_model'] else 'NO'}"
    )

    # -------------------------------------------------------------------------
    # COVERAGE
    # -------------------------------------------------------------------------

    coverage = report["coverage"]

    print()
    print("-" * 80)
    print("COVERAGE")
    print("-" * 80)

    print(
        f"Total      : {coverage['total']}"
    )

    print(
        f"Disponible : {coverage['available']}"
    )

    print(
        f"Manquantes : {coverage['missing']}"
    )

    print(
        f"Coverage   : {coverage['percentage']:.2f}%"
    )

    # -------------------------------------------------------------------------
    # PRE-MATCH
    # -------------------------------------------------------------------------

    pre_match = report["pre_match"]

    if pre_match:

        print()
        print("-" * 80)
        print("PRE-MATCH")
        print("-" * 80)

        print(
            f"Valides       : "
            f"{pre_match['valid_count']}/{EXPECTED_PRE_MATCH_COUNT}"
        )

        print(
            f"Manquantes    : "
            f"{pre_match['missing_count']}"
        )

        print(
            f"Non numériques: "
            f"{pre_match['non_numeric_count']}"
        )

        if pre_match["missing"]:

            print(
                "Features manquantes : "
                + ", ".join(
                    pre_match["missing"]
                )
            )

        if pre_match["non_numeric"]:

            print(
                "Features non numériques : "
                + ", ".join(
                    pre_match["non_numeric"]
                )
            )

    # -------------------------------------------------------------------------
    # LIVE
    # -------------------------------------------------------------------------

    live = report["live"]

    if live:

        print()
        print("-" * 80)
        print("LIVE")
        print("-" * 80)

        print(
            f"Valides       : "
            f"{live['valid_count']}/{EXPECTED_LIVE_COUNT}"
        )

        print(
            f"Manquantes    : "
            f"{live['missing_count']}"
        )

        print(
            f"Non numériques: "
            f"{live['non_numeric_count']}"
        )

        if live["missing"]:

            print(
                "Features manquantes : "
                + ", ".join(
                    live["missing"]
                )
            )

        if live["non_numeric"]:

            print(
                "Features non numériques : "
                + ", ".join(
                    live["non_numeric"]
                )
            )

    # -------------------------------------------------------------------------
    # ODDS
    # -------------------------------------------------------------------------

    odds = report["odds"]

    if odds:

        print()
        print("-" * 80)
        print("ODDS")
        print("-" * 80)

        print(
            f"Status : "
            f"{'OK' if odds['valid'] else 'FAIL'}"
        )

        if odds["invalid"]:

            for error in odds["invalid"]:

                print(
                    f"  - {error['feature']} = "
                    f"{error['value']} | "
                    f"{error['reason']}"
                )

    # -------------------------------------------------------------------------
    # FREQUENCIES
    # -------------------------------------------------------------------------

    frequencies = report["frequencies"]

    if frequencies:

        print()
        print("-" * 80)
        print("FREQUENCIES")
        print("-" * 80)

        print(
            f"Status : "
            f"{'OK' if frequencies['valid'] else 'FAIL'}"
        )

        if frequencies["invalid"]:

            for error in frequencies["invalid"]:

                print(
                    f"  - {error['feature']} = "
                    f"{error['value']} | "
                    f"{error['reason']}"
                )

    # -------------------------------------------------------------------------
    # NON NEGATIVE
    # -------------------------------------------------------------------------

    non_negative = report["non_negative"]

    if non_negative:

        print()
        print("-" * 80)
        print("NON-NEGATIVE CHECK")
        print("-" * 80)

        print(
            f"Status : "
            f"{'OK' if non_negative['valid'] else 'FAIL'}"
        )

        if non_negative["invalid"]:

            for error in non_negative["invalid"]:

                print(
                    f"  - {error['feature']} = "
                    f"{error['value']} | "
                    f"{error['reason']}"
                )

    # -------------------------------------------------------------------------
    # LIVE LOGIC
    # -------------------------------------------------------------------------

    live_logic = report["live_logic"]

    if live_logic:

        print()
        print("-" * 80)
        print("LIVE LOGIC")
        print("-" * 80)

        print(
            f"Status : "
            f"{'OK' if live_logic['valid'] else 'FAIL'}"
        )

        if live_logic["errors"]:

            for error in live_logic["errors"]:

                print(
                    f"  - {error['feature']} | "
                    f"valeur={error['value']} | "
                    f"attendu={error['expected']} | "
                    f"{error['reason']}"
                )

    # -------------------------------------------------------------------------
    # SCHEMA
    # -------------------------------------------------------------------------

    schema = report["schema"]

    print()
    print("-" * 80)
    print("SCHEMA")
    print("-" * 80)

    print(
        f"Expected : "
        f"{schema['expected_total']}"
    )

    print(
        f"Actual   : "
        f"{schema['actual_total']}"
    )

    print(
        f"Status   : "
        f"{'OK' if schema['valid'] else 'FAIL'}"
    )

    if schema["missing"]:

        print(
            "Schema missing : "
            + ", ".join(
                schema["missing"]
            )
        )

    if schema["extra"]:

        print(
            "Schema extra : "
            + ", ".join(
                schema["extra"]
            )
        )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)

    if report["ready_for_model"]:

        print(
            "RESULT : FEATURE VECTOR ACCEPTÉ"
        )

    else:

        print(
            "RESULT : FEATURE VECTOR À INVESTIGUER"
        )

    print("=" * 80)

    return report


# =============================================================================
# HELPERS
# =============================================================================

def get_missing_features(
    features: Dict[str, Any]
) -> List[str]:
    """
    Retourne uniquement les features manquantes.
    """

    report = validate_features(
        features
    )

    return report["missing_features"]


def get_coverage(
    features: Dict[str, Any]
) -> float:
    """
    Retourne la couverture en pourcentage.
    """

    report = validate_features(
        features
    )

    return report["coverage"]["percentage"]


def is_model_ready(
    features: Dict[str, Any]
) -> bool:
    """
    Retourne True uniquement si le vecteur
    passe toutes les validations critiques.
    """

    report = validate_features(
        features
    )

    return bool(
        report["ready_for_model"]
    )


# =============================================================================
# TEST LOCAL
# =============================================================================

def run_test():
    """
    Test minimal du validator.
    """

    print()
    print("=" * 80)
    print("TEST FEATURE VALIDATOR")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Création d'un vecteur vide
    # -------------------------------------------------------------------------

    empty_features = {
        feature: None
        for feature in ALL_FEATURES
    }

    print()
    print(
        "Test 1 : vecteur vide"
    )

    print_validation_report(
        empty_features
    )

    # -------------------------------------------------------------------------
    # Création d'un vecteur complet
    # -------------------------------------------------------------------------

    complete_features = {
        feature: 1.0
        for feature in ALL_FEATURES
    }

    # Valeurs LIVE cohérentes
    complete_features[
        "score_87_home"
    ] = 2.0

    complete_features[
        "score_87_away"
    ] = 1.0

    complete_features[
        "goal_diff_87"
    ] = 1.0

    complete_features[
        "is_draw_87"
    ] = 0.0

    complete_features[
        "leading_team_87"
    ] = 1.0

    # Fréquences
    for feature in FREQUENCY_FEATURES:

        complete_features[feature] = 0.5

    # Odds
    for feature in ODDS_FEATURES:

        complete_features[feature] = 2.0

    print()
    print(
        "Test 2 : vecteur complet"
    )

    print_validation_report(
        complete_features
    )

    # -------------------------------------------------------------------------
    # Test d'une erreur logique
    # -------------------------------------------------------------------------

    invalid_features = complete_features.copy()

    invalid_features[
        "goal_diff_87"
    ] = 5.0

    print()
    print(
        "Test 3 : incohérence goal_diff_87"
    )

    print_validation_report(
        invalid_features
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    run_test()