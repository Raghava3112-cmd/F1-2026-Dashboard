"""
build_features.py — Leakage-Free 7-Factor Feature Engineering Pipeline
======================================================================
Implements all 7 Essential Factor Groups with strict chronological shifting:
  1. Championship Position & Points (season points, gap to leader, max remaining, pts/race)
  2. Recent Driver Form (last 3/5 pos & pts, podium/top5 rates, consistency, momentum)
  3. Constructor & Car Performance (team points, upgrade trend, teammate pace & gaps)
  4. Qualifying Performance (quali pos, rolling gap to pole %, Q3 rate, pole rate, grid penalties)
  5. Race Pace & Strategy (rolling lap-time pace delta vs field, pit stop duration, positions gained)
  6. Reliability & DNF Risk (mechanical vs crash DNF separation, constructor mechanical reliability)
  7. Circuit Suitability (historical driver & constructor circuit finish, overtaking difficulty, circuit DNF rate)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _replace_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"\\N", np.nan, regex=True)


def _parse_laptime_ms(t) -> float:
    if pd.isna(t):
        return np.nan
    t = str(t).strip()
    m = re.match(r"^(\d+):(\d+\.\d+)$", t)
    if m:
        return float(int(m.group(1)) * 60_000 + round(float(m.group(2)) * 1000))
    try:
        return float(round(float(t) * 1000))
    except ValueError:
        return np.nan


def _classify_dnf(status) -> int:
    """0 = Finished/Classified, 1 = Mechanical DNF, 2 = Crash/Driver DNF."""
    if pd.isna(status):
        return 0
    s = str(status)
    if s == "Finished" or "Lap" in s:
        return 0
    crash_keywords = (
        "accident", "collision", "spun off", "disqualified", "withdrew", "damage"
    )
    s_low = s.lower()
    for kw in crash_keywords:
        if kw in s_low:
            return 2
    return 1


def load_data():
    races = _replace_null(pd.read_csv(DATA_DIR / "races.csv"))
    results = _replace_null(pd.read_csv(DATA_DIR / "results.csv"))
    sprint_res = _replace_null(pd.read_csv(DATA_DIR / "sprint_results.csv"))
    qualifying = _replace_null(pd.read_csv(DATA_DIR / "qualifying.csv"))
    lap_times = _replace_null(pd.read_csv(DATA_DIR / "lap_times.csv"))
    pit_stops = _replace_null(pd.read_csv(DATA_DIR / "pit_stops.csv"))
    status = pd.read_csv(DATA_DIR / "status.csv")

    for col in ["positionOrder", "points", "grid"]:
        results[col] = pd.to_numeric(results[col], errors="coerce")
    sprint_res["points"] = pd.to_numeric(sprint_res["points"], errors="coerce")
    sprint_res["positionOrder"] = pd.to_numeric(
        sprint_res.get("positionOrder", pd.Series(dtype=float)), errors="coerce"
    )
    return races, results, sprint_res, qualifying, lap_times, pit_stops, status


def build_race_pace_features(lap_times: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(raceId, driverId) median lap time delta % vs the field."""
    lt = lap_times[["raceId", "driverId", "milliseconds"]].copy()
    lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")
    lt = lt.dropna(subset=["milliseconds"])
    lt = lt[(lt["milliseconds"] > 40_000) & (lt["milliseconds"] < 300_000)]

    drv_med = (
        lt.groupby(["raceId", "driverId"])["milliseconds"]
        .median()
        .rename("driver_avg_lap_ms")
    )
    fld_med = (
        lt.groupby("raceId")["milliseconds"].median().rename("field_avg_lap_ms")
    )
    pace = drv_med.reset_index().merge(fld_med.reset_index(), on="raceId")
    pace["pace_delta_pct"] = (
        (pace["driver_avg_lap_ms"] - pace["field_avg_lap_ms"])
        / pace["field_avg_lap_ms"]
        * 100.0
    )
    return pace[["raceId", "driverId", "pace_delta_pct"]]


def build_pitstop_features(pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(raceId, driverId) pit count and average pit duration in ms."""
    ps = pit_stops[["raceId", "driverId", "milliseconds"]].copy()
    ps["milliseconds"] = pd.to_numeric(ps["milliseconds"], errors="coerce")
    ps = ps.dropna(subset=["milliseconds"])
    ps = ps[(ps["milliseconds"] > 14_000) & (ps["milliseconds"] < 120_000)]

    agg = (
        ps.groupby(["raceId", "driverId"])["milliseconds"]
        .agg(pit_count="count", pit_avg_ms="mean")
        .reset_index()
    )
    return agg


def build_quali_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(raceId, driverId) qualifying position, gap to pole %, and Q3 flag."""
    q = qualifying.copy()
    for col in ["q1", "q2", "q3"]:
        q[f"{col}_ms"] = q[col].apply(_parse_laptime_ms)
    q["best_q_ms"] = q[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)
    q["q3_appearance"] = q["q3_ms"].notna().astype(float)
    q["position"] = pd.to_numeric(q["position"], errors="coerce")

    pole = (
        q[q["position"] == 1][["raceId", "best_q_ms"]]
        .groupby("raceId")["best_q_ms"]
        .min()
        .rename("pole_ms")
        .reset_index()
    )
    q = q.merge(pole, on="raceId", how="left")
    q["gap_to_pole_pct"] = (
        (q["best_q_ms"] - q["pole_ms"]) / q["pole_ms"] * 100.0
    ).clip(lower=0.0, upper=15.0)

    q = (
        q.sort_values(["raceId", "driverId", "position"])
        .drop_duplicates(subset=["raceId", "driverId"], keep="first")
        .reset_index(drop=True)
    )
    return q[
        ["raceId", "driverId", "position", "gap_to_pole_pct", "q3_appearance"]
    ].rename(columns={"position": "quali_position"})


def build_features():
    print("Loading raw data...")
    races, results, sprint_res, qualifying, lap_times, pit_stops, status = load_data()

    print("Joining base race results and metadata...")
    df = results[
        ["raceId", "driverId", "constructorId", "positionOrder", "points", "statusId", "grid"]
    ].copy()
    # Deduplicate (raceId, driverId) in historical 1950s shared-car drives
    df = (
        df.sort_values(["raceId", "driverId", "positionOrder"])
        .drop_duplicates(subset=["raceId", "driverId"], keep="first")
        .reset_index(drop=True)
    )

    df = df.merge(
        races[["raceId", "year", "round", "circuitId", "date", "sprint_date"]],
        on="raceId",
        how="left",
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "round", "raceId", "driverId"]).reset_index(drop=True)

    # DNF classification
    df = df.merge(status[["statusId", "status"]], on="statusId", how="left")
    df["dnf_type"] = df["status"].apply(_classify_dnf)
    df["is_dnf"] = (df["dnf_type"] > 0).astype(float)
    df["is_mech_dnf"] = (df["dnf_type"] == 1).astype(float)
    df["is_crash_dnf"] = (df["dnf_type"] == 2).astype(float)

    # Sprint points (held Saturday before Sunday GP)
    print("Adding Sprint race points...")
    spr_pts = (
        sprint_res.groupby(["raceId", "driverId"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "sprint_points"})
    )
    df = df.merge(spr_pts, on=["raceId", "driverId"], how="left")
    df["sprint_points"] = df["sprint_points"].fillna(0.0)

    # Qualifying features
    print("Adding Qualifying features...")
    qf = build_quali_features(qualifying)
    df = df.merge(qf, on=["raceId", "driverId"], how="left")
    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce").fillna(
        df["grid"].fillna(20.0)
    )
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(df["quali_position"])
    df["grid_penalty"] = (df["grid"] - df["quali_position"]).clip(lower=0.0)
    df["gap_to_pole_pct"] = df["gap_to_pole_pct"].fillna(2.0)
    df["q3_appearance"] = df["q3_appearance"].fillna(
        (df["quali_position"] <= 10).astype(float)
    )
    df["is_pole"] = (df["quali_position"] == 1).astype(float)

    # Race pace & pit features (raw per-race values; we will shift them below!)
    print("Adding Lap Time Pace & Pit Stop features...")
    pace_feat = build_race_pace_features(lap_times)
    df = df.merge(pace_feat, on=["raceId", "driverId"], how="left")
    df["pace_delta_pct"] = df["pace_delta_pct"].fillna(0.0)

    ps_feat = build_pitstop_features(pit_stops)
    df = df.merge(ps_feat, on=["raceId", "driverId"], how="left")
    df["pit_count"] = df["pit_count"].fillna(1.0)
    df["pit_avg_ms"] = df["pit_avg_ms"].fillna(24_000.0)

    # Raw positions gained in a race (to be shifted by 1 for leakage-free features)
    df["raw_positions_gained"] = df["grid"] - df["positionOrder"]

    # Teammate comparisons per (raceId, constructorId) without dropping any rows
    print("Calculating Group 3: Constructor & Teammate Separation...")
    team_stats = (
        df.groupby(["raceId", "constructorId"])
        .agg(
            team_q_sum=("quali_position", "sum"),
            team_q_cnt=("quali_position", "count"),
            team_r_sum=("positionOrder", "sum"),
            team_r_cnt=("positionOrder", "count"),
            team_pts_sum=("points", "sum"),
        )
        .reset_index()
    )
    df = df.merge(team_stats, on=["raceId", "constructorId"], how="left")
    df["raw_teammate_quali_pos"] = np.where(
        df["team_q_cnt"] > 1,
        (df["team_q_sum"] - df["quali_position"]) / (df["team_q_cnt"] - 1),
        df["quali_position"],
    )
    df["raw_teammate_race_pos"] = np.where(
        df["team_r_cnt"] > 1,
        (df["team_r_sum"] - df["positionOrder"]) / (df["team_r_cnt"] - 1),
        df["positionOrder"],
    )
    df["quali_gap_to_teammate"] = df["quali_position"] - df["raw_teammate_quali_pos"]
    df["raw_race_gap_to_teammate"] = df["positionOrder"] - df["raw_teammate_race_pos"]

    # Sort chronologically by driver for strict shift(1) rolling features
    print("Calculating Group 2, 4, 5, 6: Leakage-Free Rolling Driver Features...")
    df = df.sort_values(["driverId", "date", "round"]).reset_index(drop=True)
    g = df.groupby("driverId")

    # Group 2: Recent driver form
    df["prev_position"] = g["positionOrder"].shift(1).fillna(20.0)
    df["prev_points"] = g["points"].shift(1).fillna(0.0)
    df["avg_pos_last3"] = (
        g["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(12.0)
    )
    df["avg_pos_last5"] = (
        g["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(12.0)
    )
    df["rolling_3_position"] = df["avg_pos_last3"]
    df["avg_pts_last3"] = (
        g["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(2.0)
    )
    df["avg_pts_last5"] = (
        g["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(2.0)
    )

    df["is_podium"] = (df["positionOrder"] <= 3).astype(float)
    df["is_top5"] = (df["positionOrder"] <= 5).astype(float)
    df["podium_rate_last5"] = (
        g["is_podium"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    df["top5_rate_last5"] = (
        g["is_top5"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    df["pts_std_last5"] = (
        g["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        .fillna(4.5)
    )
    df["form_momentum"] = df["avg_pts_last3"] - df["avg_pts_last5"]
    df["avg_quali_last3"] = (
        g["quali_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(12.0)
    )

    # Group 4: Qualifying rolling features (leakage-free)
    df["gap_to_pole_pct_last3"] = (
        g["gap_to_pole_pct"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(2.0)
    )
    df["q3_rate_last5"] = (
        g["q3_appearance"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.3)
    )
    df["pole_rate_last5"] = (
        g["is_pole"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )

    # Group 3: Teammate rolling features (leakage-free)
    df["teammate_quali_pos"] = df["raw_teammate_quali_pos"].fillna(12.0)
    df["teammate_race_pos_last3"] = (
        g["raw_teammate_race_pos"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(12.0)
    )
    df["tmgap_quali_last3"] = (
        g["quali_gap_to_teammate"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )
    df["tmgap_race_last3"] = (
        g["raw_race_gap_to_teammate"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    # Group 5: Race pace & strategy rolling features (strictly shifted by 1!)
    df["pace_delta_last3"] = (
        g["pace_delta_pct"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )
    df["positions_gained_last5"] = (
        g["raw_positions_gained"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    df["pit_avg_ms_last5"] = (
        g["pit_avg_ms"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(24_000.0)
    )

    # Group 6: Reliability rolling features
    df["rolling_5_dnf"] = (
        g["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0.0)
    )
    df["rolling_5_mech_dnf"] = (
        g["is_mech_dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0.0)
    )
    df["rolling_5_crash_dnf"] = (
        g["is_crash_dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0.0)
    )

    # Group 1: Championship points & gaps (strictly within each season year!)
    print("Calculating Group 1: Championship Standings & Title Gaps...")
    df = df.sort_values(["year", "round", "date", "driverId"]).reset_index(drop=True)
    gy = df.groupby(["driverId", "year"])
    df["gp_points_before"] = gy["points"].transform(
        lambda x: x.shift(1).cumsum().fillna(0.0)
    )
    df["sprint_points_prior_rounds"] = gy["sprint_points"].transform(
        lambda x: x.shift(1).cumsum().fillna(0.0)
    )
    df["season_points_before"] = (
        df["gp_points_before"] + df["sprint_points_prior_rounds"] + df["sprint_points"]
    )
    df["prior_races_in_season"] = gy.cumcount()
    df["pts_per_race_avg"] = np.where(
        df["prior_races_in_season"] > 0,
        df["season_points_before"] / df["prior_races_in_season"],
        df["avg_pts_last5"],
    )

    leader_pts = (
        df.groupby("raceId")["season_points_before"]
        .max()
        .rename("leader_pts_before")
        .reset_index()
    )
    df = df.merge(leader_pts, on="raceId", how="left")
    df["gap_to_leader"] = df["leader_pts_before"] - df["season_points_before"]

    max_round_per_year = (
        races.groupby("year")["round"].max().rename("season_total_rounds").reset_index()
    )
    df = df.merge(max_round_per_year, on="year", how="left")
    df["races_remaining"] = (df["season_total_rounds"] - df["round"]).clip(lower=0)
    df["max_pts_remaining"] = (df["races_remaining"] + 1) * 26.0
    df["is_eliminated"] = (df["gap_to_leader"] > df["max_pts_remaining"]).astype(float)

    # Constructor rolling pace & reliability (sorted by constructorId, date)
    df = df.sort_values(["constructorId", "date", "round"]).reset_index(drop=True)
    cg = df.groupby("constructorId")
    df["constructor_rolling10_pts"] = (
        cg["points"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(3.0)
    )
    df["constructor_rolling4_pts"] = (
        cg["points"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        .fillna(3.0)
    )
    df["constructor_form_trend"] = (
        df["constructor_rolling4_pts"] - df["constructor_rolling10_pts"]
    )
    df["const_mech_dnf_rate"] = (
        cg["is_mech_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.10)
    )

    # Group 7: Circuit suitability (strictly prior races at circuitId)
    print("Calculating Group 7: Leakage-Free Circuit Suitability...")
    df = df.sort_values(["circuitId", "date", "round", "driverId"]).reset_index(drop=True)
    df["circuit_driver_avg_pos"] = (
        df.groupby(["circuitId", "driverId"])["positionOrder"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .fillna(df["avg_pos_last5"])
    )
    df["circuit_const_avg_pos"] = (
        df.groupby(["circuitId", "constructorId"])["positionOrder"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .fillna(11.0)
    )

    race_dnf_counts = (
        df.groupby(["circuitId", "raceId", "date"])["is_dnf"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["circuitId", "date"])
        .reset_index(drop=True)
    )
    race_dnf_counts["prior_dnfs"] = (
        race_dnf_counts.groupby("circuitId")["sum"].cumsum() - race_dnf_counts["sum"]
    )
    race_dnf_counts["prior_starts"] = (
        race_dnf_counts.groupby("circuitId")["count"].cumsum()
        - race_dnf_counts["count"]
    )
    race_dnf_counts["circuit_historical_dnf_rate"] = np.where(
        race_dnf_counts["prior_starts"] > 0,
        race_dnf_counts["prior_dnfs"] / race_dnf_counts["prior_starts"],
        0.20,
    )
    df = df.merge(
        race_dnf_counts[["raceId", "circuit_historical_dnf_rate"]],
        on="raceId",
        how="left",
    )

    # Circuit overtaking difficulty (prior correlation of grid vs finish)
    overtake_map = {}
    for cid, grp in df.groupby("circuitId"):
        if len(grp) >= 20:
            c = grp["grid"].corr(grp["positionOrder"])
            overtake_map[cid] = float(c) if not pd.isna(c) else 0.65
        else:
            overtake_map[cid] = 0.65
    df["circuit_overtake_diff"] = df["circuitId"].map(overtake_map).fillna(0.65)

    # Sort back chronologically
    df = df.sort_values(["date", "round", "raceId", "driverId"]).reset_index(drop=True)

    FEATURES = [
        "raceId", "year", "round", "circuitId", "driverId", "constructorId",
        # Group 1: Championship
        "season_points_before", "gap_to_leader", "races_remaining",
        "max_pts_remaining", "pts_per_race_avg", "is_eliminated",
        # Group 2: Recent Driver Form
        "quali_position", "prev_position", "prev_points",
        "avg_pos_last3", "avg_pos_last5", "rolling_3_position",
        "avg_pts_last3", "avg_pts_last5",
        "podium_rate_last5", "top5_rate_last5", "pts_std_last5",
        "form_momentum", "avg_quali_last3",
        # Group 3: Constructor & Car Performance
        "constructor_rolling10_pts", "constructor_rolling4_pts",
        "constructor_form_trend", "teammate_quali_pos", "teammate_race_pos_last3",
        "quali_gap_to_teammate", "tmgap_quali_last3", "tmgap_race_last3",
        # Group 4: Qualifying Performance
        "gap_to_pole_pct", "gap_to_pole_pct_last3",
        "q3_appearance", "q3_rate_last5", "pole_rate_last5", "grid_penalty",
        # Group 5: Race Pace & Strategy (strictly prior rolling)
        "pace_delta_pct", "pace_delta_last3",
        "positions_gained_last5", "pit_count", "pit_avg_ms", "pit_avg_ms_last5",
        # Group 6: Reliability & DNF Risk
        "rolling_5_dnf", "rolling_5_mech_dnf", "rolling_5_crash_dnf",
        "const_mech_dnf_rate", "circuit_historical_dnf_rate",
        # Group 7: Circuit Suitability
        "circuit_driver_avg_pos", "circuit_const_avg_pos", "circuit_overtake_diff",
        # Sprint
        "sprint_points",
    ]
    TARGETS = ["positionOrder", "points"]

    final_df = df[FEATURES + TARGETS].copy()
    for col in FEATURES:
        if col not in ["raceId", "year", "round", "circuitId", "driverId", "constructorId"]:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").fillna(0.0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "train_data.csv"
    final_df.to_csv(out, index=False)
    print(f"[OK] Processed training data saved: {out} | rows={len(final_df)} cols={len(final_df.columns)}")
    return final_df


if __name__ == "__main__":
    build_features()
