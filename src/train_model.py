"""
train_model.py — Leakage-Free 7-Factor XGBoost Model Training
=============================================================
Trains the race-points prediction model (f1_2026_model.pkl) using all 7
essential factor groups built by build_features.py, strictly using
pre-race / rolling historical features (zero target leakage).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Strictly pre-race / historical rolling features across all 7 groups
FEATURE_COLS = [
    # Group 1: Championship Pace Rate (scale-invariant across all rounds)
    "pts_per_race_avg",
    # Group 2: Recent Driver Form
    "prev_position",
    "prev_points",
    "avg_pos_last3",
    "avg_pos_last5",
    "avg_pts_last3",
    "avg_pts_last5",
    "podium_rate_last5",
    "top5_rate_last5",
    "pts_std_last5",
    "form_momentum",
    "avg_quali_last3",
    # Group 3: Constructor & Car Performance
    "constructor_rolling10_pts",
    "constructor_rolling4_pts",
    "constructor_form_trend",
    "teammate_quali_pos",
    "teammate_race_pos_last3",
    "quali_gap_to_teammate",
    "tmgap_quali_last3",
    "tmgap_race_last3",
    # Group 4: Qualifying Performance
    "quali_position",
    "gap_to_pole_pct_last3",
    "q3_rate_last5",
    "pole_rate_last5",
    "grid_penalty",
    # Group 5: Race Pace & Strategy (rolling prior races)
    "pace_delta_last3",
    "positions_gained_last5",
    "pit_avg_ms_last5",
    # Group 6: Reliability & DNF Risk
    "rolling_5_dnf",
    "rolling_5_mech_dnf",
    "rolling_5_crash_dnf",
    "const_mech_dnf_rate",
    "circuit_historical_dnf_rate",
    # Group 7: Circuit Suitability
    "circuit_driver_avg_pos",
    "circuit_const_avg_pos",
    "circuit_overtake_diff",
    # Sprint
    "sprint_points",
]


def train_and_evaluate():
    print("Loading processed training data...")
    df = pd.read_csv(DATA_DIR / "train_data.csv")
    df = df.dropna(subset=["points", "positionOrder"]).copy()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Chronological validation on 2025 season (modern regulations)
    train_df = df[(df["year"] >= 2010) & (df["year"] < 2025)].copy()
    val_df = df[df["year"] == 2025].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["points"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["points"]

    print("\n============================================================")
    print("MODEL VALIDATION - 2025 Season Chronological Backtest")
    print("============================================================")

    # Model A: Baseline Quali Heuristic
    base = LinearRegression()
    base.fit(X_train[["quali_position"]], y_train)
    pred_a = np.maximum(0, base.predict(X_val[["quali_position"]]))
    mae_a = mean_absolute_error(y_val, pred_a)
    rmse_a = np.sqrt(mean_squared_error(y_val, pred_a))
    print(f"Model A (Baseline - Quali Heuristic) : MAE={mae_a:.3f} | RMSE={rmse_a:.3f}")

    # Model B: Unweighted 7-Factor XGBoost
    xgb_b = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_b.fit(X_train, y_train)
    pred_b = np.maximum(0, xgb_b.predict(X_val))
    mae_b = mean_absolute_error(y_val, pred_b)
    rmse_b = np.sqrt(mean_squared_error(y_val, pred_b))
    print(f"Model B (7-Factor Unweighted XGBoost): MAE={mae_b:.3f} | RMSE={rmse_b:.3f}")

    # Model C: Recency-Weighted 7-Factor XGBoost (Production)
    weights = np.exp((train_df["year"] - train_df["year"].max()) / 4.0)
    xgb_c = xgb.XGBRegressor(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
    )
    xgb_c.fit(X_train, y_train, sample_weight=weights)
    pred_c = np.maximum(0, xgb_c.predict(X_val))
    mae_c = mean_absolute_error(y_val, pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_val, pred_c))
    print(f"Model C (7-Factor Recency XGBoost)   : MAE={mae_c:.3f} | RMSE={rmse_c:.3f} <-- PRODUCTION")

    print("\n============================================================")
    print("TRAINING FINAL PRODUCTION MODEL (2010-2026 Adaptive)")
    print("============================================================")
    modern_df = df[df["year"] >= 2010].copy()
    X_all = modern_df[FEATURE_COLS]
    y_all = modern_df["points"]
    all_weights = np.exp((modern_df["year"] - modern_df["year"].max()) / 4.0)

    final_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
    )
    final_model.fit(X_all, y_all, sample_weight=all_weights)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "f1_2026_model.pkl"
    joblib.dump({"model": final_model, "features": FEATURE_COLS}, model_path)
    print(f"[OK] Production model saved --> {model_path}")

    imp = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
    print("\nTop 20 Feature Importances:")
    for feat, val in imp.head(20).items():
        bar = "=" * int(val * 60)
        print(f"  {feat:30s}: {val * 100:5.2f}%  {bar}")


if __name__ == "__main__":
    train_and_evaluate()
