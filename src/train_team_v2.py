import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def train_one(df, target, out_path):
    # Features: halftime score + behaviors + rate features
    feature_cols = [c for c in df.columns if c.startswith(("h1_", "home_", "away_"))]
    # remove targets from features
    feature_cols = [c for c in feature_cols if c not in ("h2_total","h2_margin")]

    X = df[feature_cols].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": feature_cols, "model": model}, out_path)
    print(f"Saved {out_path} | target={target} | test_R2={r2:.3f}")

def main():
    df = pd.read_parquet("data/processed/halftime_team_v2.parquet")
    df = df.dropna()

    train_one(df, "h2_total", "models/team_v2_2h_total.joblib")
    train_one(df, "h2_margin", "models/team_v2_2h_margin.joblib")

if __name__ == "__main__":
    main()
