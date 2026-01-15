import os
import joblib
import numpy as np
import pandas as pd

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))

def coverage(y, lo, hi):
    y = np.asarray(y); lo = np.asarray(lo); hi = np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))

def sharpness(lo, hi):
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))

def ensure_cols(df, cols, label=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required cols: {missing}")

def derive_team_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure final_home/final_away and h2_home/h2_away exist.
    Prefer existing columns; otherwise derive from totals+margins.
    """
    df = df.copy()

    # Derive final_home/final_away if missing but final_total/final_margin exist
    if ("final_home" not in df.columns) or ("final_away" not in df.columns):
        if ("final_total" in df.columns) and ("final_margin" in df.columns):
            df["final_home"] = (df["final_total"].astype(float) + df["final_margin"].astype(float)) / 2.0
            df["final_away"] = (df["final_total"].astype(float) - df["final_margin"].astype(float)) / 2.0

    # Derive h2_home/h2_away if missing but h2_total/h2_margin exist
    if ("h2_home" not in df.columns) or ("h2_away" not in df.columns):
        if ("h2_total" in df.columns) and ("h2_margin" in df.columns):
            df["h2_home"] = (df["h2_total"].astype(float) + df["h2_margin"].astype(float)) / 2.0
            df["h2_away"] = (df["h2_total"].astype(float) - df["h2_margin"].astype(float)) / 2.0

    return df

def main():
    data_path = "data/processed/halftime_team_v2.parquet"
    out_path = "data/processed/backtest_v2_predictions.csv"

    df = pd.read_parquet(data_path)
    df = derive_team_scores(df)

    # Minimum required truth for 2H evaluation + CI calibration
    required = ["game_id", "h2_total", "h2_margin", "h1_home", "h1_away", "final_total", "final_margin"]
    ensure_cols(df, required, label="Dataset")

    # Also require derived team scores now
    ensure_cols(df, ["final_home", "final_away"], label="Derived final team scores")

    total_obj = joblib.load("models/team_v2_2h_total.joblib")
    margin_obj = joblib.load("models/team_v2_2h_margin.joblib")
    intervals = joblib.load("models/v2_intervals.joblib")

    feat_total = total_obj["features"]
    feat_margin = margin_obj["features"]
    ensure_cols(df, list(set(feat_total + feat_margin)), label="Feature set")

    X_total = df[feat_total]
    X_margin = df[feat_margin]

    pred_t = total_obj["model"].predict(X_total).astype(float)
    pred_m = margin_obj["model"].predict(X_margin).astype(float)

    # Calibrated residual quantile bands (global)
    t_q10 = float(intervals["resid_total_q10"])
    t_q90 = float(intervals["resid_total_q90"])
    m_q10 = float(intervals["resid_margin_q10"])
    m_q90 = float(intervals["resid_margin_q90"])

    # 80% CI for 2H total/margin
    t_lo = pred_t + t_q10
    t_hi = pred_t + t_q90
    m_lo = pred_m + m_q10
    m_hi = pred_m + m_q90

    # Convert to team score point estimates
    h2_home_pred = (pred_t + pred_m) / 2.0
    h2_away_pred = (pred_t - pred_m) / 2.0

    final_home_pred = df["h1_home"].values.astype(float) + h2_home_pred
    final_away_pred = df["h1_away"].values.astype(float) + h2_away_pred
    final_total_pred = final_home_pred + final_away_pred
    final_margin_pred = final_home_pred - final_away_pred

    # Conservative team-score CI by evaluating 4 corners of (total, margin) rectangle
    corners = [(t_lo, m_lo), (t_lo, m_hi), (t_hi, m_lo), (t_hi, m_hi)]
    h2_home_vals = np.vstack([(t + m) / 2.0 for (t, m) in corners])
    h2_away_vals = np.vstack([(t - m) / 2.0 for (t, m) in corners])

    h2_home_lo, h2_home_hi = h2_home_vals.min(axis=0), h2_home_vals.max(axis=0)
    h2_away_lo, h2_away_hi = h2_away_vals.min(axis=0), h2_away_vals.max(axis=0)

    final_home_lo = df["h1_home"].values.astype(float) + h2_home_lo
    final_home_hi = df["h1_home"].values.astype(float) + h2_home_hi
    final_away_lo = df["h1_away"].values.astype(float) + h2_away_lo
    final_away_hi = df["h1_away"].values.astype(float) + h2_away_hi

    # Metrics on 2H targets
    y_t = df["h2_total"].values.astype(float)
    y_m = df["h2_margin"].values.astype(float)

    print("\n=== Backtest: 2H Targets ===")
    print(f"Rows: {len(df)}")
    print(f"2H Total  | MAE={mae(y_t, pred_t):.2f}  RMSE={rmse(y_t, pred_t):.2f}")
    print(f"2H Margin | MAE={mae(y_m, pred_m):.2f}  RMSE={rmse(y_m, pred_m):.2f}")

    cov_t = coverage(y_t, t_lo, t_hi)
    cov_m = coverage(y_m, m_lo, m_hi)
    print("\n=== Calibration (nominal 80%) ===")
    print(f"2H Total  | coverage={cov_t:.3f}  avg_width={sharpness(t_lo, t_hi):.2f}")
    print(f"2H Margin | coverage={cov_m:.3f}  avg_width={sharpness(m_lo, m_hi):.2f}")

    # Final team-score calibration (derived)
    y_fh = df["final_home"].values.astype(float)
    y_fa = df["final_away"].values.astype(float)
    cov_fh = coverage(y_fh, final_home_lo, final_home_hi)
    cov_fa = coverage(y_fa, final_away_lo, final_away_hi)

    print("\n=== Final Team Score Calibration (derived from 2H bands) ===")
    print(f"Final Home | coverage={cov_fh:.3f}  avg_width={sharpness(final_home_lo, final_home_hi):.2f}")
    print(f"Final Away | coverage={cov_fa:.3f}  avg_width={sharpness(final_away_lo, final_away_hi):.2f}")

    # Export predictions for deeper analysis
    out = df[["game_id","h1_home","h1_away","h2_total","h2_margin","final_total","final_margin","final_home","final_away"]].copy()
    out["pred_h2_total"] = pred_t
    out["pred_h2_margin"] = pred_m
    out["pred_h2_home"] = h2_home_pred
    out["pred_h2_away"] = h2_away_pred

    out["pred_final_home"] = final_home_pred
    out["pred_final_away"] = final_away_pred
    out["pred_final_total"] = final_total_pred
    out["pred_final_margin"] = final_margin_pred

    out["ci80_h2_total_lo"] = t_lo
    out["ci80_h2_total_hi"] = t_hi
    out["ci80_h2_margin_lo"] = m_lo
    out["ci80_h2_margin_hi"] = m_hi

    out["ci80_final_home_lo"] = final_home_lo
    out["ci80_final_home_hi"] = final_home_hi
    out["ci80_final_away_lo"] = final_away_lo
    out["ci80_final_away_hi"] = final_away_hi

    os.makedirs("data/processed", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions -> {out_path}\n")

if __name__ == "__main__":
    main()
