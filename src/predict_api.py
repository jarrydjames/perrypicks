from __future__ import annotations

from typing import Any, Dict, Union


def predict_game(game_input: str, **kwargs) -> Dict[str, Any]:
    """
    Streamlit-Cloud-safe entrypoint used by app.py.

    Key design choices:
    - NO sys.path mutations at import-time
    - Heavy imports happen INSIDE this function to avoid circular/partial import issues
    - Accepts **kwargs so older callers won't crash if they pass extra flags
    """
    # Lazy imports to prevent circular import / partial init issues on Streamlit Cloud
    from src.predict_from_gameid_v2 import (
        extract_game_id,
        fetch_box,
        fetch_pbp_df,
        first_half_score,
        behavior_counts_1h,
        team_totals_from_box_team,
        add_rate_features,
        load_models_and_intervals,
        predict_halftime,
        team_score_calibrate,
    )

    # If your betting module provides helper(s) used here, import lazily too.
    # If not present, we keep the app working anyway.
    try:
        from src.betting import normal_from_q10q90  # type: ignore
    except Exception:
        normal_from_q10q90 = None  # type: ignore

    # --- Core prediction pipeline (matches your existing v2 flow) ---
    gid = extract_game_id(game_input)

    box = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    h1_home, h1_away, status = first_half_score(box, pbp)

    # If game is final (or current score missing), we still return halftime-based forecast
    # behavior + rates from 1H
    beh = behavior_counts_1h(pbp)
    tt = team_totals_from_box_team(box)
    feats = add_rate_features({**beh, **tt})

    models = load_models_and_intervals()

    # halftime projection (2H total + 2H margin)
    pred2h = predict_halftime(
        models=models,
        feats=feats,
        h1_home=h1_home,
        h1_away=h1_away,
        status=status,
    )

    # team-score calibrated final projection + intervals
    out = team_score_calibrate(
        gid=gid,
        h1_home=h1_home,
        h1_away=h1_away,
        status=status,
        pred2h=pred2h,
        models=models,
    )

    # Optional: include Normal-approx intervals if helper exists
    # (Some parts of your app display both "normal" and "bands80".)
    if normal_from_q10q90 and "bands80" in out:
        try:
            b = out["bands80"]
            out["normal"] = {
                "h2_total": list(normal_from_q10q90(*b["h2_total"])),
                "h2_margin": list(normal_from_q10q90(*b["h2_margin"])),
                "final_total": list(normal_from_q10q90(*b["final_total"])),
                "final_margin": list(normal_from_q10q90(*b["final_margin"])),
            }
        except Exception:
            # Don't let optional normal calcs break the app
            pass

    return out
