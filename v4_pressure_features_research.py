"""
V4 — Reconstruction des six caractéristiques B6.
Usage : tests historiques hors ligne uniquement.
Aucune certification de provenance LIVE.
"""

from build_historical_ab_research_v3 import SHOTS, SOT
from v3c_pressure_features import (
    get_events,
    event_minute,
    team_side,
)

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]


def compute_b6_historical(
    payload,
    home_team,
    away_team,
    home_id=None,
    away_id=None,
):
    events = get_events(payload)

    if events is None:
        raise ValueError("Timeline non reconnue")

    if not home_team or not away_team:
        raise ValueError("Identités des équipes manquantes")

    counts = {name: 0 for name in B6}

    for event in events:
        if not isinstance(event, dict):
            continue

        period = str(
            event.get("period", "")
        ).strip().lower()

        if period != "second_half":
            continue

        minute = event_minute(event)

        if minute is None or not (82 <= minute < 87):
            continue

        typ = str(
            event.get("type", "")
        ).strip().lower()

        if (
            typ not in SHOTS
            and typ not in SOT
            and typ != "corner_kick"
        ):
            continue

        side = team_side(
            event,
            home_team,
            away_team,
            home_id=home_id,
            away_id=away_id,
        )

        if side not in ("home", "away"):
            raise ValueError(
                f"Équipe inconnue : minute={minute}, "
                f"type={typ}"
            )

        if typ in SHOTS:
            counts[f"shots_{side}_last5"] += 1

        if typ in SOT:
            counts[f"sot_{side}_last5"] += 1

        if typ == "corner_kick":
            # Convention du constructeur historique V4.
            credited = (
                "away" if side == "home" else "home"
            )
            counts[f"corners_{credited}_last5"] += 1

    for side in ("home", "away"):
        if (
            counts[f"sot_{side}_last5"]
            > counts[f"shots_{side}_last5"]
        ):
            raise ValueError(
                f"SOT > tirs pour {side}"
            )

    return counts
