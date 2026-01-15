import joblib
import pandas as pd

def load_model(path):
    obj = joblib.load(path)
    return obj["features"], obj["model"]

def predict(h1_home, h1_away,
            h1_events=0, h1_n_2pt=0, h1_n_3pt=0, h1_n_turnover=0,
            h1_n_rebound=0, h1_n_foul=0, h1_n_timeout=0, h1_n_sub=0):

    features_total, m_total = load_model("models/team_2h_total.joblib")
    features_margin, m_margin = load_model("models/team_2h_margin.joblib")

    row = {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_total": h1_home + h1_away,
        "h1_margin": h1_home - h1_away,
        "h1_events": h1_events,
        "h1_n_2pt": h1_n_2pt,
        "h1_n_3pt": h1_n_3pt,
        "h1_n_turnover": h1_n_turnover,
        "h1_n_rebound": h1_n_rebound,
        "h1_n_foul": h1_n_foul,
        "h1_n_timeout": h1_n_timeout,
        "h1_n_sub": h1_n_sub,
    }

    X = pd.DataFrame([row])

    # Ensure column order matches training
    X_total = X[features_total]
    X_margin = X[features_margin]

    pred_2h_total = float(m_total.predict(X_total)[0])
    pred_2h_margin = float(m_margin.predict(X_margin)[0])

    # Allocate predicted 2H points to teams based on predicted margin.
    # Let: (H2_home + H2_away) = T, (H2_home - H2_away) = M
    # => H2_home = (T+M)/2, H2_away = (T-M)/2
    h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    final_home = h1_home + h2_home
    final_away = h1_away + h2_away

    return {
        "pred_2h_total": pred_2h_total,
        "pred_2h_margin": pred_2h_margin,
        "pred_2h_home": h2_home,
        "pred_2h_away": h2_away,
        "pred_final_home": final_home,
        "pred_final_away": final_away,
        "pred_final_total": final_home + final_away,
        "pred_final_margin": final_home - final_away,
    }

if __name__ == "__main__":
    # Example call: halftime home 58 away 52
    out = predict(58, 52)
    for k, v in out.items():
        print(f"{k}: {v:.2f}")
