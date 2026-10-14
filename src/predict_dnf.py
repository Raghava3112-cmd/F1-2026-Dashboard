"""
predict_dnf.py — Upgraded DNF Prediction Model (Power BI Compatible)
====================================================================
Key features:
  - Separates mechanical DNF vs crash/driver-fault DNF in model inputs
  - Incorporates constructor reliability, circuit attrition, pace delta, and pit consistency
  - Uses StandardScaler + LogisticRegression pipeline for clean convergence
  - Exports exact 12-column schema required by Power BI (next_race_dnf_predictions.csv)
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")


def _replace_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"\\N", np.nan, regex=True)


def _classify_dnf(status) -> int:
    if pd.isna(status):
        return 0
    s = str(status)
    if s == "Finished" or "Lap" in s:
        return 0
    crash_kw = ("accident", "collision", "spun off", "disqualified", "withdrew", "damage")
    s_low = s.lower()
    for kw in crash_kw:
        if kw in s_low:
            return 2
    return 1


DNF_FEATURES = [
    "driver_rolling_5_dnf",
    "driver_rolling_5_mech_dnf",
    "driver_season_dnf_rate",
    "constructor_rolling_dnf",
    "circuit_historical_dnf_rate",
    "grid_feat",
    "pace_delta_last3",
    "pit_avg_ms",
]


def build_dnf_dataset() -> pd.DataFrame:
    res = _replace_null(pd.read_csv(DATA_DIR / "results.csv"))
    status = pd.read_csv(DATA_DIR / "status.csv")
    races = _replace_null(pd.read_csv(DATA_DIR / "races.csv"))

    res = res.merge(status[["statusId", "status"]], on="statusId", how="left")
    res["dnf_type"] = res["status"].apply(_classify_dnf)
    res["is_dnf"] = (res["dnf_type"] > 0).astype(int)
    res["is_mech_dnf"] = (res["dnf_type"] == 1).astype(int)
    res["is_crash_dnf"] = (res["dnf_type"] == 2).astype(int)

    df = res.merge(
        races[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId",
        how="left",
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "round", "raceId", "driverId"]).reset_index(drop=True)

    g = df.groupby("driverId")
    df["driver_rolling_5_dnf"] = (
        g["is_dnf"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.18)
    )
    df["driver_rolling_5_mech_dnf"] = (
        g["is_mech_dnf"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.10)
    )
    df["driver_season_dnf_rate"] = (
        df.groupby(["driverId", "year"])["is_dnf"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .fillna(0.18)
    )
    df["constructor_rolling_dnf"] = (
        df.groupby("constructorId")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.18)
    )

    circ = df.groupby(["circuitId", "raceId", "date"])["is_dnf"].agg(["sum", "count"]).reset_index()
    circ = circ.sort_values(["circuitId", "date"]).reset_index(drop=True)
    circ["prior_sum"] = circ.groupby("circuitId")["sum"].cumsum() - circ["sum"]
    circ["prior_cnt"] = circ.groupby("circuitId")["count"].cumsum() - circ["count"]
    circ["circuit_historical_dnf_rate"] = np.where(
        circ["prior_cnt"] > 0, circ["prior_sum"] / circ["prior_cnt"], 0.18
    )
    df = df.merge(circ[["raceId", "circuit_historical_dnf_rate"]], on="raceId", how="left")

    df["grid_feat"] = pd.to_numeric(df["grid"], errors="coerce").fillna(11.0)

    try:
        td = pd.read_csv(
            PROCESSED_DIR / "train_data.csv",
            usecols=["raceId", "driverId", "pace_delta_last3", "pit_avg_ms_last5"],
        ).rename(columns={"pit_avg_ms_last5": "pit_avg_ms"})
        df = df.merge(td, on=["raceId", "driverId"], how="left")
    except Exception:
        df["pace_delta_last3"] = 0.0
        df["pit_avg_ms"] = 24_000.0

    df["pace_delta_last3"] = df["pace_delta_last3"].fillna(0.0)
    df["pit_avg_ms"] = df["pit_avg_ms"].fillna(24_000.0)
    return df


def train_and_predict_dnf():
    print("Building leakage-free DNF dataset...")
    df = build_dnf_dataset()

    drivers = pd.read_csv(DATA_DIR / "drivers.csv")
    constructors = pd.read_csv(DATA_DIR / "constructors.csv")
    races = _replace_null(pd.read_csv(DATA_DIR / "races.csv"))

    modern = df[df["year"] >= 2014].copy()
    print(f"Modern era training obs: {len(modern):,}")

    train_df = modern[modern["year"] < 2025]
    val_df = modern[modern["year"] == 2025]

    X_train = train_df[DNF_FEATURES]
    y_train = train_df["is_dnf"]
    X_val = val_df[DNF_FEATURES]
    y_val = val_df["is_dnf"]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, val_probs)
    auc = roc_auc_score(y_val, val_probs)
    ll = log_loss(y_val, val_probs)
    baseline = brier_score_loss(y_val, np.full(len(y_val), y_train.mean()))

    print("\n--- DNF Model Validation (2025 Chronological) ---")
    print(f"Brier Score : {brier:.4f}  (baseline: {baseline:.4f})")
    print(f"ROC-AUC     : {auc:.4f}")
    print(f"Log Loss    : {ll:.4f}")

    final_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])
    final_clf.fit(modern[DNF_FEATURES], modern["is_dnf"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_clf, "features": DNF_FEATURES}, MODEL_DIR / "dnf_model.pkl")
    print(f"[OK] DNF model saved --> {MODEL_DIR / 'dnf_model.pkl'}")

    races_2026 = races[races["year"] == 2026].sort_values("round").reset_index(drop=True)
    results_2026 = df[df["year"] == 2026]
    completed_rounds = sorted(results_2026["round"].unique())
    if not completed_rounds:
        return

    last_round = max(completed_rounds)
    remaining = races_2026[races_2026["round"] > last_round]
    if remaining.empty:
        return

    next_race = remaining.iloc[0]
    next_race_id = int(next_race["raceId"])
    next_round = int(next_race["round"])
    next_name = str(next_race["name"])
    next_circuit = int(next_race["circuitId"])

    last_res = results_2026[results_2026["round"] == last_round]
    active = last_res["driverId"].unique()

    circ_past = df[(df["circuitId"] == next_circuit) & (df["year"] < 2026)]
    circ_rate = float(circ_past["is_dnf"].mean()) if len(circ_past) > 0 else 0.18

    rows = []
    for d in active:
        d_hist = df[df["driverId"] == d].sort_values("date")
        d_roll5 = float(d_hist["is_dnf"].tail(5).mean()) if len(d_hist) >= 1 else 0.18
        d_roll5_mech = float(d_hist["is_mech_dnf"].tail(5).mean()) if len(d_hist) >= 1 else 0.10
        d_2026 = d_hist[d_hist["year"] == 2026]
        d_season_dnf = float(d_2026["is_dnf"].mean()) if len(d_2026) >= 1 else 0.18

        c_id = int(last_res[last_res["driverId"] == d]["constructorId"].iloc[0])
        c_hist = df[df["constructorId"] == c_id]
        c_dnf = float(c_hist["is_dnf"].tail(10).mean()) if len(c_hist) >= 1 else 0.18
        grid_e = float(d_2026["grid_feat"].mean()) if len(d_2026) >= 1 else 11.0

        pace_delta = float(d_2026["pace_delta_last3"].iloc[-1]) if len(d_2026) >= 1 else 0.0
        pit_ms = float(d_2026["pit_avg_ms"].iloc[-1]) if len(d_2026) >= 1 else 24_000.0

        feat = pd.DataFrame([{
            "driver_rolling_5_dnf": d_roll5,
            "driver_rolling_5_mech_dnf": d_roll5_mech,
            "driver_season_dnf_rate": d_season_dnf,
            "constructor_rolling_dnf": c_dnf,
            "circuit_historical_dnf_rate": circ_rate,
            "grid_feat": grid_e,
            "pace_delta_last3": pace_delta,
            "pit_avg_ms": pit_ms,
        }])
        dnf_prob = float(final_clf.predict_proba(feat[DNF_FEATURES])[0, 1])
        finish_prob = 1.0 - dnf_prob

        rows.append({
            "race_id": next_race_id,
            "round": next_round,
            "race_name": next_name,
            "driver_id": int(d),
            "constructor_id": int(c_id),
            "dnf_probability": dnf_prob,
            "finish_probability": finish_prob,
            "driver_season_dnf_rate": d_season_dnf,
            "circuit_historical_dnf_rate": circ_rate,
        })

    export = (
        pd.DataFrame(rows)
        .merge(drivers[["driverId", "code", "surname"]], left_on="driver_id", right_on="driverId", how="left")
        .merge(
            constructors[["constructorId", "name"]].rename(columns={"name": "team_name"}),
            left_on="constructor_id",
            right_on="constructorId",
            how="left",
        )
    )

    # Exact 12-column schema required by Power BI and test suite
    cols = [
        "race_id", "round", "race_name", "driver_id", "code", "surname",
        "constructor_id", "team_name", "dnf_probability", "finish_probability",
        "driver_season_dnf_rate", "circuit_historical_dnf_rate",
    ]
    export = export[cols].sort_values("dnf_probability", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUTPUT_DIR / "next_race_dnf_predictions.csv", index=False)
    print(f"[OK] DNF predictions saved ({len(export)} drivers, 12 columns)")


if __name__ == "__main__":
    train_and_predict_dnf()
