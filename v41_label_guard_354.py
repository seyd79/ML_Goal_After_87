import collect_historical_dataset_v2 as original

ORIGINAL_TARGET = original.compute_target_after87


def pending(reason):
    return {
        "goal_after_87": float("nan"),
        "target_status": "PENDING",
        "target_resolution_class": "PENDING",
        "target_confidence": "",
        "goal_minutes_after_87": "",
        "target_warning": reason,
    }


def compute_target_after87_guarded(
    live87,
    timeline_events,
    final_home,
    final_away,
):
    result = ORIGINAL_TARGET(
        live87=live87,
        timeline_events=timeline_events,
        final_home=final_home,
        final_away=final_away,
    )

    if result.get("target_status") != "LABELED_V2":
        return result

    home = original.safe_int(final_home)
    away = original.safe_int(final_away)

    if (
        home is None or away is None
        or home < 0 or away < 0
    ):
        return pending("final_score_unavailable")

    valid_goals = 0

    for event in timeline_events:
        if not original.event_is_goal(live87, event):
            continue

        if not original.event_is_regulation_time(event):
            continue

        minute = live87.extract_event_minute(event)

        if minute is None:
            return pending("goal_minute_missing")

        valid_goals += 1

    if valid_goals != home + away:
        return pending("timeline_final_score_mismatch")

    return result
