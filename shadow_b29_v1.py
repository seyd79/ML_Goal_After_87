from build_historical_ab_research_v3 import (
    A_FEATURES,
    B_EXTRA,
    build_match,
    event_minute,
)
from shadow_b_last5_v1 import calculate_b_last5

FEATURES = list(A_FEATURES) + list(B_EXTRA)

if len(FEATURES) != 29 or len(set(FEATURES)) != 29:
    raise RuntimeError("Schéma B invalide")


def calculate_b29(events, info, decision_minute=87):
    """
    Calculateur expérimental B.

    23 variables communes : constructeur historique.
    6 variables supplémentaires : SHADOW B last5.

    Aucun événement à la minute de décision
    ou après n'est transmis au constructeur.

    Utilisation de recherche seulement.
    """

    if decision_minute not in (85, 86, 87):
        raise ValueError("Minute de décision invalide")

    if not isinstance(events, list):
        raise ValueError("Liste d'événements attendue")

    mid = info.get("match_id")

    if not mid:
        raise ValueError("match_id manquant")

    filtered = []

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Événement non valide")

        try:
            minute = event_minute(event)
        except (ValueError, TypeError):
            continue

        if minute < decision_minute:
            filtered.append(event)

    payload = {
        "match_id": mid,
        "events": filtered,
    }

    row_a, row_b, quality = build_match(
        mid, info, payload
    )

    if row_a is None or row_b is None:
        raise ValueError(
            f"Reconstruction impossible : {mid}"
        )

    extra = calculate_b_last5(
        filtered,
        info,
        decision_minute=decision_minute,
    )

    # Le constructeur historique et le module
    # SHADOW doivent produire les mêmes 6 valeurs.
    for feature in B_EXTRA:
        if int(row_b[feature]) != int(extra[feature]):
            raise ValueError(
                f"Divergence B : {mid}, {feature}"
            )

    result = {}

    for feature in A_FEATURES:
        result[feature] = row_a[feature]

    for feature in B_EXTRA:
        result[feature] = extra[feature]

    if set(result) != set(FEATURES):
        raise ValueError("Schéma de sortie incorrect")

    return result
