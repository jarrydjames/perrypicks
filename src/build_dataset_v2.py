import json
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

def fetch_box(gid: str) -> dict:
    return fetch_json(CDN_BOX.format(gid=gid))["game"]

def fetch_pbp_df(gid: str) -> pd.DataFrame:
    data = fetch_json(CDN_PBP.format(gid=gid))
    return pd.DataFrame(data["game"]["actions"])

def sum_first2(periods):
    s = 0
    for p in (periods or []):
        if int(p.get("period", 0)) in (1,2):
            for key in ("score","points","pts"):
                if key in p and p[key] is not None:
                    s += int(p[key]); break
    return s

def first_half_score(game):
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first2(home.get("periods")), sum_first2(away.get("periods"))

def behavior_counts_1h(pbp: pd.DataFrame) -> dict:
    fh = pbp[pbp["period"].astype(int) <= 2].copy()
    at = fh.get("actionType", pd.Series([""]*len(fh))).astype(str).fillna("")
    def c(prefix): return int(at.str.startswith(prefix).sum())
    return {
        "h1_events": int(len(fh)),
        "h1_n_2pt": c("2pt"),
        "h1_n_3pt": c("3pt"),
        "h1_n_turnover": c("turnover"),
        "h1_n_rebound": c("rebound"),
        "h1_n_foul": c("foul"),
        "h1_n_timeout": c("timeout"),
        "h1_n_sub": c("substitution"),
    }

def team_totals_from_box_team(team: dict) -> dict:
    """
    Uses game-total team stats from boxscore JSON.
    We'll later extend to first-half-only, but these still help as priors.
    """
    stats = (team.get("statistics") or {})
    def gi(k, default=0):
        v = stats.get(k, default)
        try: return int(v)
        except: return default
    def gf(k, default=0.0):
        v = stats.get(k, default)
        try: return float(v)
        except: return default

    return {
        "fga": gi("fieldGoalsAttempted"),
        "fgm": gi("fieldGoalsMade"),
        "tpa": gi("threePointersAttempted"),
        "tpm": gi("threePointersMade"),
        "fta": gi("freeThrowsAttempted"),
        "ftm": gi("freeThrowsMade"),
        "oreb": gi("reboundsOffensive"),
        "dreb": gi("reboundsDefensive"),
        "reb": gi("reboundsTotal"),
        "ast": gi("assists"),
        "stl": gi("steals"),
        "blk": gi("blocks"),
        "to": gi("turnovers"),
        "pf": gi("foulsPersonal"),
        "pts": gi("points"),
    }

def add_rate_features(prefix: str, t: dict, opp: dict) -> dict:
    # Possession-ish proxy
    poss = t["fga"] + 0.44*t["fta"] + t["to"] - t["oreb"]
    poss = max(poss, 1.0)

    efg = (t["fgm"] + 0.5*t["tpm"]) / max(t["fga"], 1)
    ftr = t["fta"] / max(t["fga"], 1)
    tpar = t["tpa"] / max(t["fga"], 1)
    tor = t["to"] / poss
    orbp = t["oreb"] / max(t["oreb"] + opp["dreb"], 1)

    return {
        f"{prefix}_efg": efg,
        f"{prefix}_ftr": ftr,
        f"{prefix}_tpar": tpar,
        f"{prefix}_tor": tor,
        f"{prefix}_orbp": orbp,
    }

def main():
    ids_path = Path("data/processed/game_ids_2025.json")
    games = json.loads(ids_path.read_text())

    # only completed games
    game_ids = [g["gameId"] for g in games if int(g.get("gameStatus", 0)) == 3]

    rows = []
    skipped = 0

    for gid in tqdm(game_ids, desc="Building v2 dataset"):
        try:
            game = fetch_box(gid)
            pbp = fetch_pbp_df(gid)

            home = game.get("homeTeam", {}) or {}
            away = game.get("awayTeam", {}) or {}

            h1_home, h1_away = first_half_score(game)
            if h1_home == 0 and h1_away == 0:
                raise ValueError("Missing period scoring for 1H.")

            # final
            fin_home = int(home.get("score", 0))
            fin_away = int(away.get("score", 0))
            if fin_home == 0 and fin_away == 0:
                raise ValueError("Missing final score.")

            beh = behavior_counts_1h(pbp)

            # team priors from box totals (game totals)
            ht = team_totals_from_box_team(home)
            at = team_totals_from_box_team(away)

            row = {
                "game_id": gid,
                "h1_home": h1_home,
                "h1_away": h1_away,
                "h1_total": h1_home + h1_away,
                "h1_margin": h1_home - h1_away,

                "h2_home": fin_home - h1_home,
                "h2_away": fin_away - h1_away,
                "h2_total": (fin_home + fin_away) - (h1_home + h1_away),
                "h2_margin": (fin_home - fin_away) - (h1_home - h1_away),

                "final_total": fin_home + fin_away,
                "final_margin": fin_home - fin_away,
            }

            row.update(beh)
            row.update(add_rate_features("home", ht, at))
            row.update(add_rate_features("away", at, ht))

            rows.append(row)

        except Exception:
            skipped += 1

    out = pd.DataFrame(rows)
    out_path = Path("data/processed/halftime_team_v2.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"Rows kept: {len(out)}")
    print(f"Rows skipped: {skipped}")

if __name__ == "__main__":
    main()
