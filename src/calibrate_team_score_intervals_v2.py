import joblib
import numpy as np
import pandas as pd

PRED_PATH = "data/processed/backtest_v2_predictions.csv"
OUT_PATH  = "models/v2_team_score_intervals.joblib"

def q10_q90(x):
    return float(np.quantile(x, 0.10)), float(np.quantile(x, 0.90))

def main():
    df = pd.read_csv(PRED_PATH)

    # True 2H team scores derived from true 2H total+margin
    df["h2_home_true"] = (df["h2_total"] + df["h2_margin"]) / 2.0
    df["h2_away_true"] = (df["h2_total"] - df["h2_margin"]) / 2.0

    # Residuals (true - pred)
    df["h2_home_err"] = df["h2_home_true"] - df["pred_h2_home"]
    df["h2_away_err"] = df["h2_away_true"] - df["pred_h2_away"]
    df["final_home_err"] = df["final_home"] - df["pred_final_home"]
    df["final_away_err"] = df["final_away"] - df["pred_final_away"]

    out = {}
    out["h2_home_q10"], out["h2_home_q90"] = q10_q90(df["h2_home_err"])
    out["h2_away_q10"], out["h2_away_q90"] = q10_q90(df["h2_away_err"])
    out["final_home_q10"], out["final_home_q90"] = q10_q90(df["final_home_err"])
    out["final_away_q10"], out["final_away_q90"] = q10_q90(df["final_away_err"])

    joblib.dump(out, OUT_PATH)
    print(f"Saved {OUT_PATH}")
    for k in ["h2_home_q10","h2_home_q90","h2_away_q10","h2_away_q90","final_home_q10","final_home_q90","final_away_q10","final_away_q90"]:
        print(f"{k}: {out[k]:.3f}")

if __name__ == "__main__":
    main()
