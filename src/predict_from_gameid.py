import sys
import re
import requests
import pandas as pd
import joblib

CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

def extract_game_id(arg: str) -> str:
    # Accept either "0022500558" or a full NBA.com URL containing it
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

def load_model(path):
    obj = joblib.load(path)
    return obj["features"], obj["model"]

def first_half_score_from_box(game: dict):
    """
    Most reliable: sum points in periods 1 and 2 from boxscore JSON.
    """
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}

    # Many boxscore JSONs include per-period points under something like:
    # homeTeam["periods"] = [{"period":1,"score":...}, ...]
    hp = home.get("periods", []) or []
    ap = away.get("periods", []) or []

    def sum_first2(periods):
        s = 0
        for p in periods:
            if int(p.get("period", 0)) in (1, 2):
                # score key varies; try common ones
                for key in ("score", "points", "pts"):
                    if key in p and p[key] is not None:
                        s += int(p[key])
                        break
        return s

    h1_home = sum_first2(hp)
    h1_away = sum_first2(ap)

    # If periods missing, fallback to firstHalfScore fields if present
    if (h1_home == 0 and h1_away == 0):
        # Some schemas store totals differently
        for key in ("score", "points"):
            if key in home and key in away:
                # that's final score, not first half, so don't use
                pass

    if h1_home == 0 and h1_away == 0:
        raise ValueError("Could not compute 1H score from boxscore periods (period data missing).")

    return h1_home, h1_away

def compute_1h_behavior_from_pbp(pbp: pd.DataFrame):
    """
    Action counts from periods 1-2.
    """
    if "period" not in pbp.columns:
        raise ValueError(f"PBP missing 'period'. Columns: {list(pbp.columns)}")

    fh = pbp[pbp["period"].astype(int) <= 2].copy()
    at = fh.get("actionType", pd.Series([""] * len(fh))).astype(str).fillna("")

    def count(prefix):
        return int(at.str.startswith(prefix).sum())

    feats = {
        "h1_events": int(len(fh)),
        "h1_n_2pt": count("2pt"),
        "h1_n_3pt": count("3pt"),
        "h1_n_turnover": count("turnover"),
        "h1_n_rebound": count("rebound"),
        "h1_n_foul": count("foul"),
        "h1_n_timeout": count("timeout"),
        "h1_n_sub": count("substitution"),
    }
    return feats

def predict_from_halftime(h1_home: int, h1_away: int, beh: dict):
    features_total, m_total = load_model("models/team_2h_total.joblib")
    features_margin, m_margin = load_model("models/team_2h_margin.joblib")

    row = {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_total": h1_home + h1_away,
        "h1_margin": h1_home - h1_away,
        "h1_events": beh.get("h1_events", 0),
        "h1_n_2pt": beh.get("h1_n_2pt", 0),
        "h1_n_3pt": beh.get("h1_n_3pt", 0),
        "h1_n_turnover": beh.get("h1_n_turnover", 0),
        "h1_n_rebound": beh.get("h1_n_rebound", 0),
        "h1_n_foul": beh.get("h1_n_foul", 0),
        "h1_n_timeout": beh.get("h1_n_timeout", 0),
        "h1_n_sub": beh.get("h1_n_sub", 0),
    }

    X = pd.DataFrame([row])
    pred_2h_total = float(m_total.predict(X[features_total])[0])
    pred_2h_margin = float(m_margin.predict(X[features_margin])[0])

    h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    final_home = h1_home + h2_home
    final_away = h1_away + h2_away

    return row, {
        "pred_2h_total": pred_2h_total,
        "pred_2h_margin": pred_2h_margin,
        "pred_2h_home": h2_home,
        "pred_2h_away": h2_away,
        "pred_final_home": final_home,
        "pred_final_away": final_away,
        "pred_final_total": final_home + final_away,
        "pred_final_margin": final_home - final_away,
    }

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/predict_from_gameid.py <GAME_ID or NBA.com game URL>")

    gid = extract_game_id(sys.argv[1])

    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    home_tri = (game.get("homeTeam", {}) or {}).get("teamTricode", "HOME")
    away_tri = (game.get("awayTeam", {}) or {}).get("teamTricode", "AWAY")

    h1_home, h1_away = first_half_score_from_box(game)
    beh = compute_1h_behavior_from_pbp(pbp)

    _, pred = predict_from_halftime(h1_home, h1_away, beh)

    print(f"GAME_ID: {gid}")
    print(f"Teams: {away_tri} @ {home_tri}  (home={home_tri})")
    print(f"1H score: {home_tri} {h1_home} - {h1_away} {away_tri}")
    print("1H behavior:", beh)
    print("\nPrediction:")
    for k, v in pred.items():
        print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    main()
