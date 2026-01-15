import joblib
import numpy as np
import pandas as pd

PRED_PATH = "data/processed/backtest_v2_predictions.csv"
OUT_PATH  = "models/v2_intervals_binned.joblib"

TOTAL_BINS = [-1e9, 110, 120, 1e9]   # low / mid / high pace
ABS_MARGIN_BINS = [-1e9, 5, 10, 1e9] # close / medium / large

def q10_q90(x):
    return float(np.quantile(x, 0.10)), float(np.quantile(x, 0.90))

def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}\nAvailable: {list(df.columns)}")

def main():
    df = pd.read_csv(PRED_PATH)

    # Auto-detect column names from your backtest output
    col_pred_total  = pick_col(df, ["pred_2h_total", "pred_h2_total", "pred_total_2h", "p_2h_total", "yhat_2h_total"], "predicted 2H total")
    col_pred_margin = pick_col(df, ["pred_2h_margin", "pred_h2_margin", "pred_margin_2h", "p_2h_margin", "yhat_2h_margin"], "predicted 2H margin")

    col_true_total  = pick_col(df, ["h2_total", "true_2h_total", "y_2h_total"], "true 2H total")
    col_true_margin = pick_col(df, ["h2_margin", "true_2h_margin", "y_2h_margin"], "true 2H margin")

    # residuals = true - pred
    df["resid_total"]  = df[col_true_total].astype(float)  - df[col_pred_total].astype(float)
    df["resid_margin"] = df[col_true_margin].astype(float) - df[col_pred_margin].astype(float)

    out = {
        "total_bins": TOTAL_BINS,
        "abs_margin_bins": ABS_MARGIN_BINS,
        "pred_cols": {"total": col_pred_total, "margin": col_pred_margin},
        "true_cols": {"total": col_true_total, "margin": col_true_margin},
        "total": {},
        "margin": {}
    }

    # Bin by predicted total
    for i in range(len(TOTAL_BINS)-1):
        lo, hi = TOTAL_BINS[i], TOTAL_BINS[i+1]
        sub = df[(df[col_pred_total] >= lo) & (df[col_pred_total] < hi)]
        if len(sub) < 30:
            continue
        qlo, qhi = q10_q90(sub["resid_total"])
        out["total"][i] = {"q10": qlo, "q90": qhi, "n": int(len(sub))}

    # Bin by abs(predicted margin)
    for i in range(len(ABS_MARGIN_BINS)-1):
        lo, hi = ABS_MARGIN_BINS[i], ABS_MARGIN_BINS[i+1]
        sub = df[(df[col_pred_margin].abs() >= lo) & (df[col_pred_margin].abs() < hi)]
        if len(sub) < 30:
            continue
        qlo, qhi = q10_q90(sub["resid_margin"])
        out["margin"][i] = {"q10": qlo, "q90": qhi, "n": int(len(sub))}

    joblib.dump(out, OUT_PATH)

    print(f"Saved {OUT_PATH}")
    print(f"Using columns: pred_total={col_pred_total}, pred_margin={col_pred_margin}, true_total={col_true_total}, true_margin={col_true_margin}\n")

    print("Total bins (by predicted 2H total):")
    for k in range(len(TOTAL_BINS)-1):
        d = out["total"].get(k)
        if not d:
            print(f"  bin {k}: (skipped, n<30)")
        else:
            print(f"  bin {k}: n={d['n']} q10={d['q10']:.2f} q90={d['q90']:.2f}")

    print("\nMargin bins (by abs(predicted 2H margin)):")
    for k in range(len(ABS_MARGIN_BINS)-1):
        d = out["margin"].get(k)
        if not d:
            print(f"  bin {k}: (skipped, n<30)")
        else:
            print(f"  bin {k}: n={d['n']} q10={d['q10']:.2f} q90={d['q90']:.2f}")

if __name__ == "__main__":
    main()
