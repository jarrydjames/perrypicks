import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_json(path: Path):
    return json.loads(path.read_text())

def load_pbp_actions(pbp_path: Path) -> pd.DataFrame:
    j = load_json(pbp_path)
    actions = (j.get("game") or {}).get("actions") or []
    df = pd.DataFrame(actions)
    return df

def extract_reg_scores_from_box(box_j: dict):
    """
    Returns:
      h1_home, h1_away, h2_home, h2_away, fin_home, fin_away
    Using period scores from boxscore, regulation only (periods 1-4).
    """
    game = box_j.get("game") or {}
    home = game.get("homeTeam") or {}
    away = game.get("awayTeam") or {}

    # Final scores (often present)
    fin_home = home.get("score")
    fin_away = away.get("score")

    # Period scores: usually a list under each team
    home_periods = home.get("periods") or []
    away_periods = away.get("periods") or []

    def per_scores(period_list):
        # period_list elements often like {"period":1,"score":28}
        d = {}
        for p in period_list:
            try:
                per = int(p.get("period"))
                sc = int(p.get("score"))
                d[per] = sc
            except Exception:
                continue
        return d

    hp = per_scores(home_periods)
    ap = per_scores(away_periods)

    # Need at least 1-4 for regulation split; if missing, fall back to None
    if not all(k in hp for k in (1,2,3,4)) or not all(k in ap for k in (1,2,3,4)):
        return None

    h1_home = hp[1] + hp[2]
    h1_away = ap[1] + ap[2]
    h2_home = hp[3] + hp[4]
    h2_away = ap[3] + ap[4]

    # If final not present, compute regulation final
    if fin_home is None or fin_away is None:
        fin_home = h1_home + h2_home
        fin_away = h1_away + h2_away

    return int(h1_home), int(h1_away), int(h2_home), int(h2_away), int(fin_home), int(fin_away)

def count_first_half_actions(pbp_df: pd.DataFrame) -> dict:
    if pbp_df.empty or "period" not in pbp_df.columns or "actionType" not in pbp_df.columns:
        return {
            "events": 0, "n_2pt": 0, "n_3pt": 0, "n_turnover": 0,
            "n_rebound": 0, "n_foul": 0, "n_timeout": 0, "n_sub": 0
        }

    fh = pbp_df[pbp_df["period"] <= 2].copy()
    at = fh["actionType"]

    def cnt(x): return int((at == x).sum())

    return {
        "events": int(len(fh)),
        "n_2pt": cnt("2pt"),
        "n_3pt": cnt("3pt"),
        "n_turnover": cnt("turnover"),
        "n_rebound": cnt("rebound"),
        "n_foul": cnt("foul"),
        "n_timeout": cnt("timeout"),
        "n_sub": cnt("substitution"),
    }

def main():
    pbp_dir = Path("data/raw/pbp")
    box_dir = Path("data/raw/box")

    pbp_files = sorted(pbp_dir.glob("*.json"))

    rows = []
    skipped = 0

    for pbp_path in tqdm(pbp_files, desc="Building dataset"):
        gid = pbp_path.stem
        box_path = box_dir / f"{gid}.json"
        if not box_path.exists():
            skipped += 1
            continue

        try:
            box_j = load_json(box_path)
            scores = extract_reg_scores_from_box(box_j)
            if scores is None:
                skipped += 1
                continue
            h1_home, h1_away, h2_home, h2_away, fin_home, fin_away = scores

            pbp_df = load_pbp_actions(pbp_path)
            feats = count_first_half_actions(pbp_df)

            rows.append({
                "game_id": gid,
                "h1_home": h1_home,
                "h1_away": h1_away,
                "h1_total": h1_home + h1_away,
                "h1_margin": h1_home - h1_away,

                "h1_events": feats["events"],
                "h1_n_2pt": feats["n_2pt"],
                "h1_n_3pt": feats["n_3pt"],
                "h1_n_turnover": feats["n_turnover"],
                "h1_n_rebound": feats["n_rebound"],
                "h1_n_foul": feats["n_foul"],
                "h1_n_timeout": feats["n_timeout"],
                "h1_n_sub": feats["n_sub"],

                # targets (regulation 2H)
                "h2_home": h2_home,
                "h2_away": h2_away,
                "h2_total": h2_home + h2_away,
                "h2_margin": h2_home - h2_away,

                # finals (may include OT if present in boxscore score fields)
                "final_total": int(fin_home) + int(fin_away),
                "final_margin": int(fin_home) - int(fin_away),
            })
        except Exception:
            skipped += 1
            continue

    out = pd.DataFrame(rows)
    out_path = Path("data/processed/halftime_team.parquet")
    out.to_parquet(out_path, index=False)

    print(f"Saved {out_path}")
    print(f"Rows kept: {len(out)}")
    print(f"Rows skipped: {skipped}")

if __name__ == "__main__":
    main()
