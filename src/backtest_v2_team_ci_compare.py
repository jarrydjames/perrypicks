import joblib
import numpy as np
import pandas as pd

DATA = "data/processed/backtest_v2_predictions.csv"
GLOBAL = "models/v2_intervals.joblib"
BINNED = "models/v2_intervals_binned.joblib"

# must match src/predict_from_gameid_v2_ci.py
MARGIN_INFLATE = 1.03

def coverage(y, lo, hi):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))

def avg_width(lo, hi):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean(hi - lo))

def pick_bin(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2

def get_band(obj, kind, idx):
    d = (obj or {}).get(kind, {}).get(idx)
    if not d:
        return None
    return float(d["q10"]), float(d["q90"]), int(d.get("n", 0))

def inflate_band(q10, q90, factor):
    c = (q10 + q90) / 2.0
    h = (q90 - q10) / 2.0
    h2 = h * factor
    return c - h2, c + h2

def main():
    df = pd.read_csv(DATA)

    required = ["pred_h2_total","pred_h2_margin","h2_total","h2_margin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {DATA}: {missing}\nHave: {list(df.columns)}")

    g = joblib.load(GLOBAL)
    b = joblib.load(BINNED)

    pred_t = df["pred_h2_total"].to_numpy(dtype=float)
    pred_m = df["pred_h2_margin"].to_numpy(dtype=float)
    true_t = df["h2_total"].to_numpy(dtype=float)
    true_m = df["h2_margin"].to_numpy(dtype=float)

    # ---------- GLOBAL ----------
    g_t_lo = pred_t + float(g["resid_total_q10"])
    g_t_hi = pred_t + float(g["resid_total_q90"])
    g_m_lo = pred_m + float(g["resid_margin_q10"])
    g_m_hi = pred_m + float(g["resid_margin_q90"])

    # ---------- BINNED ----------
    t_edges = b.get("total_bins", [-1e9, 1e9])
    m_edges = b.get("abs_margin_bins", [-1e9, 1e9])

    b_t_lo = np.empty_like(pred_t)
    b_t_hi = np.empty_like(pred_t)
    b_m_lo = np.empty_like(pred_m)
    b_m_hi = np.empty_like(pred_m)

    used_t = {"binned": 0, "global_fallback": 0}
    used_m = {"binned": 0, "global_fallback": 0}

    for i in range(len(pred_t)):
        tb = pick_bin(pred_t[i], t_edges)
        mb = pick_bin(abs(pred_m[i]), m_edges)

        bt = get_band(b, "total", tb)
        bm = get_band(b, "margin", mb)

        # total band
        if bt and bt[2] >= 30:
            tq10, tq90 = bt[0], bt[1]
            used_t["binned"] += 1
        else:
            tq10, tq90 = float(g["resid_total_q10"]), float(g["resid_total_q90"])
            used_t["global_fallback"] += 1

        # margin band (+ inflation, to match predictor)
        if bm and bm[2] >= 30:
            mq10, mq90 = bm[0], bm[1]
            if MARGIN_INFLATE and MARGIN_INFLATE > 1.0:
                mq10, mq90 = inflate_band(mq10, mq90, MARGIN_INFLATE)
            used_m["binned"] += 1
        else:
            mq10, mq90 = float(g["resid_margin_q10"]), float(g["resid_margin_q90"])
            used_m["global_fallback"] += 1

        b_t_lo[i] = pred_t[i] + tq10
        b_t_hi[i] = pred_t[i] + tq90
        b_m_lo[i] = pred_m[i] + mq10
        b_m_hi[i] = pred_m[i] + mq90

    print("\n=== GLOBAL vs BINNED (nominal 80%) ===")
    print(f"Rows: {len(df)}")
    print(f"Total bands usage:  binned={used_t['binned']} fallback={used_t['global_fallback']}")
    print(f"Margin bands usage: binned={used_m['binned']} fallback={used_m['global_fallback']}  (inflate={MARGIN_INFLATE:.2f}x)")

    print("\n2H TOTAL")
    print(f"  Global | coverage={coverage(true_t, g_t_lo, g_t_hi):.3f} avg_width={avg_width(g_t_lo, g_t_hi):.2f}")
    print(f"  Binned | coverage={coverage(true_t, b_t_lo, b_t_hi):.3f} avg_width={avg_width(b_t_lo, b_t_hi):.2f}")

    print("\n2H MARGIN")
    print(f"  Global | coverage={coverage(true_m, g_m_lo, g_m_hi):.3f} avg_width={avg_width(g_m_lo, g_m_hi):.2f}")
    print(f"  Binned | coverage={coverage(true_m, b_m_lo, b_m_hi):.3f} avg_width={avg_width(b_m_lo, b_m_hi):.2f}")

if __name__ == "__main__":
    main()
