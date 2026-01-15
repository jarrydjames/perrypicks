import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

def train_one(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = HistGradientBoostingRegressor(random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    print(f"{name} MAE: {mae:.2f}")
    return m

def main():
    df = pd.read_parquet("data/processed/halftime_team.parquet")

    features = [
        "h1_home","h1_away","h1_total","h1_margin",
        "h1_events",
        "h1_n_2pt","h1_n_3pt","h1_n_turnover","h1_n_rebound","h1_n_foul",
        "h1_n_timeout","h1_n_sub",
    ]

    X = df[features]
    y_total = df["h2_total"]
    y_margin = df["h2_margin"]

    m_total = train_one(X, y_total, "2H total")
    m_margin = train_one(X, y_margin, "2H margin")

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"features": features, "model": m_total}, "models/team_2h_total.joblib")
    joblib.dump({"features": features, "model": m_margin}, "models/team_2h_margin.joblib")
    print("Saved: models/team_2h_total.joblib and models/team_2h_margin.joblib")

if __name__ == "__main__":
    main()
