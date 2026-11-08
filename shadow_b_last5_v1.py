from collections import Counter

from build_historical_ab_research_v3 import (
    SHOTS,
    SOT,
    match_side,
    event_minute,
)

B_EXTRA = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]


def calculate_b_last5(events, info, decision_minute=87):
    """
    Fenêtre fixe historique B : [82, 87).

    Toute donnée à la minute de décision ou après
    est exclue.

    Compatibilité historique uniquement.
    Non certifié pour le scanner LIVE.
    """

    if decision_minute not in (85, 86, 87):
        raise ValueError(
            "Décision autorisée : 85, 86 ou 87"
        )

    if not isinstance(events, list):
        raise ValueError("Liste d'événements attendue")

    counts = Counter()

    cutoff = min(decision_minute, 87)

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Événement invalide")

        try:
            minute = event_minute(event)
        except (ValueError, TypeError):
            continue

        if not (82 <= minute < cutoff):
            continue

        if event.get("period") != "second_half":
            continue

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        if (
            typ not in SHOTS
            and typ not in SOT
            and typ != "corner_kick"
        ):
            continue

        side = match_side(event, info)

        if side not in ("home", "away"):
            raise ValueError(
                f"Équipe non identifiable : {event}"
            )

        if typ in SHOTS:
            counts[f"shots_{side}_last5"] += 1

        if typ in SOT:
            counts[f"sot_{side}_last5"] += 1

        if typ == "corner_kick":
            # Convention historique B.
            opposite = (
                "away" if side == "home" else "home"
            )
            counts[f"corners_{opposite}_last5"] += 1

    return {
        feature: counts[feature]
        for feature in B_EXTRA
    }
