import sys
import re
import requests
import pandas as pd
import joblib

CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

def extract_game_id(arg: str) -> str:
    m = re.search(r"(00\d{8,10})", arg)
    if not m:
        raise ValueError(f"Could not find a GAME_ID in: {arg}")
    return m.group(1)

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
        if int(p.get("period", 0)) in (1, 2):
            for key in ("score", "points", "pts"):
                if key in p and p[key] is not None:
                    s += int(p[key])
                    break
    return s

def first_half_score(game):
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first2(home.get("periods")), sum_first2(away.get("periods"))

def behavior_counts_1h(pbp: pd.DataFrame) -> dict:
    fh = pbp[pbp["period"].astype(int) <= 2].copy()
    at = fh.get("actionType", pd.Series([""] * len(fh))).astype(str).fillna("")
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
    stats = (team.get("statistics") or {})
    def gi(k, default=0):
        v = stats.get(k, default)
        try: return int(v)
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
    poss = t["fga"] + 0.44 * t["fta"] + t["to"] - t["oreb"]
    poss = max(poss, 1.0)
    efg = (t["fgm"] + 0.5 * t["tpm"]) / max(t["fga"], 1)
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

def load_model(path):
    obj = joblib.load(path)
    return obj["features"], obj["model"]

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/predict_from_gameid_v2.py <GAME_ID or NBA.com URL>")

    gid = extract_game_id(sys.argv[1])

    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    home_tri = home.get("teamTricode", "HOME")
    away_tri = away.get("teamTricode", "AWAY")

    h1_home, h1_away = first_half_score(game)
    if h1_home == 0 and h1_away == 0:
        raise ValueError("Missing 1H period scoring in boxscore JSON.")

    beh = behavior_counts_1h(pbp)
    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)

    row = {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_total": h1_home + h1_away,
        "h1_margin": h1_home - h1_away,
    }
    row.update(beh)
    row.update(add_rate_features("home", ht, at))
    row.update(add_rate_features("away", at, ht))

    X = pd.DataFrame([row])

    f_total, m_total = load_model("models/team_v2_2h_total.joblib")
    f_margin, m_margin = load_model("models/team_v2_2h_margin.joblib")

    pred_2h_total = float(m_total.predict(X[f_total])[0])
    pred_2h_margin = float(m_margin.predict(X[f_margin])[0])

    h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    final_home = h1_home + h2_home
    final_away = h1_away + h2_away

    print(f"GAME_ID: {gid}")
    print(f"Teams: {away_tri} @ {home_tri} (home={home_tri})")
    print(f"1H score: {home_tri} {h1_home} - {h1_away} {away_tri}")
    print("1H behavior:", beh)
    print("Rate features:", {k: round(row[k], 4) for k in row if k.startswith(("home_","away_"))})

    print("\nPrediction:")
    print(f"  pred_2h_total: {pred_2h_total:.2f}")
    print(f"  pred_2h_margin: {pred_2h_margin:.2f}")
    print(f"  pred_2h_home: {h2_home:.2f}")
    print(f"  pred_2h_away: {h2_away:.2f}")
    print(f"  pred_final_home: {final_home:.2f}")
    print(f"  pred_final_away: {final_away:.2f}")
    print(f"  pred_final_total: {final_home + final_away:.2f}")
    print(f"  pred_final_margin: {final_home - final_away:.2f}")

if __name__ == "__main__":
    main()
