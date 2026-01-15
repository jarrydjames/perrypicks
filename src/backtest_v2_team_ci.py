import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/processed/halftime_team_v2.parquet"

def coverage_and_width(y, lo, hi):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    cov = np.mean((y >= lo) & (y <= hi))
    width = np.mean(hi - lo)
    return float(cov), float(width)

def main():
    df = pd.read_parquet(DATA_PATH)

    total_obj = joblib.load("models/team_v2_2h_total.joblib")
    margin_obj = joblib.load("models/team_v2_2h_margin.joblib")
    tm_int = joblib.load("models/v2_intervals.joblib")
    team_int = joblib.load("models/v2_team_score_intervals.joblib")

    feat_total = total_obj["features"]
    feat_margin = margin_obj["features"]

    X = df

    pred_t = total_obj["model"].predict(X[feat_total]).astype(float)
    pred_m = margin_obj["model"].predict(X[feat_margin]).astype(float)

    # Total/margin conformal bands
    t_lo = pred_t + float(tm_int["resid_total_q10"])
    t_hi = pred_t + float(tm_int["resid_total_q90"])
    m_lo = pred_m + float(tm_int["resid_margin_q10"])
    m_hi = pred_m + float(tm_int["resid_margin_q90"])

    # Point team splits
    pred_h2_home = (pred_t + pred_m) / 2.0
    pred_h2_away = (pred_t - pred_m) / 2.0

    pred_final_home = df["h1_home"].to_numpy(dtype=float) + pred_h2_home
    pred_final_away = df["h1_away"].to_numpy(dtype=float) + pred_h2_away

    # True 2H team scores derived from true 2H total+margin
    true_h2_home = (df["h2_total"].to_numpy(dtype=float) + df["h2_margin"].to_numpy(dtype=float)) / 2.0
    true_h2_away = (df["h2_total"].to_numpy(dtype=float) - df["h2_margin"].to_numpy(dtype=float)) / 2.0

    # True final team scores derived from final_total + final_margin (home perspective)
    true_final_home = (df["final_total"].to_numpy(dtype=float) + df["final_margin"].to_numpy(dtype=float)) / 2.0
    true_final_away = (df["final_total"].to_numpy(dtype=float) - df["final_margin"].to_numpy(dtype=float)) / 2.0

    # Team-score calibrated bands
    h2_home_lo = pred_h2_home + float(team_int["h2_home_q10"])
    h2_home_hi = pred_h2_home + float(team_int["h2_home_q90"])
    h2_away_lo = pred_h2_away + float(team_int["h2_away_q10"])
    h2_away_hi = pred_h2_away + float(team_int["h2_away_q90"])

    final_home_lo = pred_final_home + float(team_int["final_home_q10"])
    final_home_hi = pred_final_home + float(team_int["final_home_q90"])
    final_away_lo = pred_final_away + float(team_int["final_away_q10"])
    final_away_hi = pred_final_away + float(team_int["final_away_q90"])

    print("\n=== Coverage + Avg Width (nominal 80%) ===")

    cov, w = coverage_and_width(df["h2_total"], t_lo, t_hi)
    print(f"2H Total  : coverage={cov:.3f}  avg_width={w:.2f}")

    cov, w = coverage_and_width(df["h2_margin"], m_lo, m_hi)
    print(f"2H Margin : coverage={cov:.3f}  avg_width={w:.2f}")

    cov, w = coverage_and_width(true_h2_home, h2_home_lo, h2_home_hi)
    print(f"2H Home   : coverage={cov:.3f}  avg_width={w:.2f}")

    cov, w = coverage_and_width(true_h2_away, h2_away_lo, h2_away_hi)
    print(f"2H Away   : coverage={cov:.3f}  avg_width={w:.2f}")

    cov, w = coverage_and_width(true_final_home, final_home_lo, final_home_hi)
    print(f"Final Home: coverage={cov:.3f}  avg_width={w:.2f}")

    cov, w = coverage_and_width(true_final_away, final_away_lo, final_away_hi)
    print(f"Final Away: coverage={cov:.3f}  avg_width={w:.2f}")

if __name__ == "__main__":
    main()
