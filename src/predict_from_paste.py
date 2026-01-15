import re
import sys
import joblib
import pandas as pd

def load_model(path):
    obj = joblib.load(path)
    return obj["features"], obj["model"]

def parse_totals_block(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # team names: first non-empty line of each block
    team_names = []
    for ln in lines:
        if ln.lower().startswith("click on any linked stat"):
            continue
        if ln.upper().startswith("PLAYER"):
            continue
        if ln.upper() == "TOTALS":
            continue
        if ln.lower().startswith("inactive"):
            continue
        if re.search(r"[A-Za-z]", ln) and not re.search(r"\d", ln):
            if ln not in team_names:
                team_names.append(ln)
        if len(team_names) >= 2:
            break

    totals_numeric_lines = []
    for i, ln in enumerate(lines):
        if ln.upper() == "TOTALS":
            j = i + 1
            while j < len(lines) and not re.search(r"\d", lines[j]):
                j += 1
            if j < len(lines):
                totals_numeric_lines.append(lines[j])

    if len(totals_numeric_lines) < 2:
        raise ValueError("Could not find TWO TOTALS numeric lines in the pasted text.")

    def parse_totals_numbers(numline: str):
        nums = re.findall(r"-?\d+\.\d+|-?\d+", numline)
        # Your paste is typically 19 numbers:
        # FGM FGA FG% 3PM 3PA 3P% FTM FTA FT% OREB DREB REB AST STL BLK TO PF PTS +/-
        # = 19 values (no extra column)
        if len(nums) == 19:
            vals = list(map(float, nums))
            return {
                "FGM": int(vals[0]),
                "FGA": int(vals[1]),
                "FGP": vals[2],
                "3PM": int(vals[3]),
                "3PA": int(vals[4]),
                "3PP": vals[5],
                "FTM": int(vals[6]),
                "FTA": int(vals[7]),
                "FTP": vals[8],
                "OREB": int(vals[9]),
                "DREB": int(vals[10]),
                "REB": int(vals[11]),
                "AST": int(vals[12]),
                "STL": int(vals[13]),
                "BLK": int(vals[14]),
                "TO": int(vals[15]),
                "PF": int(vals[16]),
                "PTS": int(vals[17]),
                "PLUSMINUS": int(vals[18]),
            }

        # Fallback: if some sources include an extra column, allow 20+ and map from the end
        if len(nums) >= 19:
            # Map from the end: ... TO PF PTS +/-
            vals = list(map(float, nums))
            plusminus = int(vals[-1])
            pts = int(vals[-2])
            pf = int(vals[-3])
            to = int(vals[-4])

            # Try to locate REB by scanning earlier known region:
            # We'll assume order near middle: OREB DREB REB AST STL BLK
            # Find a plausible REB as a value >= (OREB + DREB) and not huge.
            # We'll just take positions based on the common 19-layout if length is 20:
            if len(vals) >= 20:
                # assume same as 19-layout but with one extra somewhere; best effort:
                # take first 12 same, then AST/STL/BLK from near end-4 region
                fgm, fga, fgp = int(vals[0]), int(vals[1]), vals[2]
                tpm, tpa, tpp = int(vals[3]), int(vals[4]), vals[5]
                ftm, fta, ftp = int(vals[6]), int(vals[7]), vals[8]
                oreb, dreb, reb = int(vals[9]), int(vals[10]), int(vals[11])
                ast, stl, blk = int(vals[12]), int(vals[13]), int(vals[14])
                return {
                    "FGM": fgm, "FGA": fga, "FGP": fgp,
                    "3PM": tpm, "3PA": tpa, "3PP": tpp,
                    "FTM": ftm, "FTA": fta, "FTP": ftp,
                    "OREB": oreb, "DREB": dreb, "REB": reb,
                    "AST": ast, "STL": stl, "BLK": blk,
                    "TO": to, "PF": pf, "PTS": pts, "PLUSMINUS": plusminus
                }

        raise ValueError(f"Unrecognized TOTALS line format (found {len(nums)} nums): {numline}")

    teamA = parse_totals_numbers(totals_numeric_lines[0])
    teamB = parse_totals_numbers(totals_numeric_lines[1])
    teamA["TEAM_NAME"] = team_names[0] if len(team_names) >= 1 else "TeamA"
    teamB["TEAM_NAME"] = team_names[1] if len(team_names) >= 2 else "TeamB"
    return teamA, teamB

def build_model_features_from_totals(teamA: dict, teamB: dict):
    h1_home = teamA["PTS"]
    h1_away = teamB["PTS"]

    fga = teamA["FGA"] + teamB["FGA"]
    tpa = teamA["3PA"] + teamB["3PA"]
    to  = teamA["TO"]  + teamB["TO"]
    reb = teamA["REB"] + teamB["REB"]
    pf  = teamA["PF"]  + teamB["PF"]

    return {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_events": int(fga + to),
        "h1_n_3pt": int(tpa),
        "h1_n_2pt": int(max(0, fga - tpa)),
        "h1_n_turnover": int(to),
        "h1_n_rebound": int(reb),
        "h1_n_foul": int(pf),
        "h1_n_timeout": 0,
        "h1_n_sub": 0,
    }

def predict_from_features(feats: dict):
    features_total, m_total = load_model("models/team_2h_total.joblib")
    features_margin, m_margin = load_model("models/team_2h_margin.joblib")

    row = {
        "h1_home": feats["h1_home"],
        "h1_away": feats["h1_away"],
        "h1_total": feats["h1_home"] + feats["h1_away"],
        "h1_margin": feats["h1_home"] - feats["h1_away"],
        "h1_events": feats.get("h1_events", 0),
        "h1_n_2pt": feats.get("h1_n_2pt", 0),
        "h1_n_3pt": feats.get("h1_n_3pt", 0),
        "h1_n_turnover": feats.get("h1_n_turnover", 0),
        "h1_n_rebound": feats.get("h1_n_rebound", 0),
        "h1_n_foul": feats.get("h1_n_foul", 0),
        "h1_n_timeout": feats.get("h1_n_timeout", 0),
        "h1_n_sub": feats.get("h1_n_sub", 0),
    }

    X = pd.DataFrame([row])

    pred_2h_total = float(m_total.predict(X[features_total])[0])
    pred_2h_margin = float(m_margin.predict(X[features_margin])[0])

    h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    final_home = row["h1_home"] + h2_home
    final_away = row["h1_away"] + h2_away

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
    text = sys.stdin.read()
    teamA, teamB = parse_totals_block(text)
    feats = build_model_features_from_totals(teamA, teamB)
    _, pred = predict_from_features(feats)

    print(f"\nParsed teams: {teamA['TEAM_NAME']} vs {teamB['TEAM_NAME']}")
    print(f"Halftime score (ordered as parsed): {feats['h1_home']} - {feats['h1_away']}")
    print("\nPrediction:")
    for k, v in pred.items():
        print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    main()
