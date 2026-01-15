import json
import os
import time
import random
import pandas as pd
from tqdm import tqdm

# Reuse the SAME proven-good parsing/feature code you already use in prediction
from predict_from_gameid_v2 import (
    fetch_box, fetch_pbp_df,
    first_half_score, behavior_counts_1h,
    team_totals_from_box_team, add_rate_features
)

SCHED_PATH = "data/processed/game_ids_2025.json"
OUT_PATH   = "data/processed/halftime_team_v2.parquet"

def final_score_from_box(game: dict):
    """Final score from box team statistics (reliable)."""
    home = game["homeTeam"]
    away = game["awayTeam"]
    hs = int((home.get("statistics", {}) or {}).get("points") or 0)
    a  = int((away.get("statistics", {}) or {}).get("points") or 0)
    if hs == 0 and a == 0:
        return None
    return hs, a

def main():
    with open(SCHED_PATH, "r") as f:
        sched = json.load(f)

    # Only completed games
    game_ids = [g["gameId"] for g in sched if int(g.get("gameStatus", 0)) == 3]
    game_ids = list(dict.fromkeys(game_ids))

    os.makedirs("data/processed", exist_ok=True)

    rows = []
    skipped = 0
    first_errors = 0

    for gid in tqdm(game_ids, desc="Building v2 dataset"):
        try:
            game = fetch_box(gid)
            pbp  = fetch_pbp_df(gid)

            # Use your proven halftime method (this is what fixed 0022500558)
            h1_home, h1_away = first_half_score(game)

            fin = final_score_from_box(game)
            if fin is None:
                skipped += 1
                continue
            fin_home, fin_away = fin

            # Targets
            h2_home = fin_home - h1_home
            h2_away = fin_away - h1_away

            h2_total  = h2_home + h2_away
            h2_margin = h2_home - h2_away

            final_total  = fin_home + fin_away
            final_margin = fin_home - fin_away

            # 1H behavior from pbp (your proven method)
            beh = behavior_counts_1h(pbp)

            # Rate features (as used by v2 predictor)
            home = game["homeTeam"]
            away = game["awayTeam"]
            ht = team_totals_from_box_team(home)
            at = team_totals_from_box_team(away)

            row = {
                "game_id": gid,
                "h1_home": int(h1_home),
                "h1_away": int(h1_away),
                "h1_total": int(h1_home + h1_away),
                "h1_margin": int(h1_home - h1_away),
            }
            row.update(beh)
            row.update(add_rate_features("home", ht, at))
            row.update(add_rate_features("away", at, ht))

            # Save targets
            row.update({
                "h2_total": float(h2_total),
                "h2_margin": float(h2_margin),
                "final_total": float(final_total),
                "final_margin": float(final_margin),
            })

            rows.append(row)

            time.sleep(0.01 + random.random() * 0.02)

        except Exception as e:
            skipped += 1
            # Print first few errors so you can see what’s happening if we still have trouble
            if first_errors < 5:
                first_errors += 1
                print(f"\n[SKIP] {gid} -> {repr(e)}\n")
            continue

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH}")
    print(f"Rows kept: {len(out)}")
    print(f"Rows skipped: {skipped}")

if __name__ == "__main__":
    main()
