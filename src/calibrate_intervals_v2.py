import pandas as pd
import joblib
import numpy as np

def main():
    df = pd.read_parquet("data/processed/halftime_team_v2.parquet").dropna()

    obj_t = joblib.load("models/team_v2_2h_total.joblib")
    obj_m = joblib.load("models/team_v2_2h_margin.joblib")

    Xt = df[obj_t["features"]]
    Xm = df[obj_m["features"]]

    pred_t = obj_t["model"].predict(Xt)
    pred_m = obj_m["model"].predict(Xm)

    resid_t = df["h2_total"].values - pred_t
    resid_m = df["h2_margin"].values - pred_m

    q_t = np.quantile(resid_t, [0.1, 0.9])
    q_m = np.quantile(resid_m, [0.1, 0.9])

    joblib.dump(
        {"resid_total_q10": float(q_t[0]), "resid_total_q90": float(q_t[1]),
         "resid_margin_q10": float(q_m[0]), "resid_margin_q90": float(q_m[1])},
        "models/v2_intervals.joblib"
    )
    print("Saved models/v2_intervals.joblib")
    print("Total resid q10/q90:", q_t)
    print("Margin resid q10/q90:", q_m)

if __name__ == "__main__":
    main()
