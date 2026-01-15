from __future__ import annotations

from typing import Any, Dict


def predict_game(game_input: str, use_binned_intervals: bool = True) -> Dict[str, Any]:
    """
    Single public entrypoint used by app.py.

    Cloud-safe: avoids importing non-existent helpers from predict_from_gameid_v2.
    Delegates to src.predict_from_gameid_v2_ci.predict_from_game_id which returns
    the rich dict (status dict, bands80, normal, labels, text, etc).
    """
    from src.predict_from_gameid_v2_ci import predict_from_game_id

    return predict_from_game_id(game_input, use_binned_intervals=use_binned_intervals)
