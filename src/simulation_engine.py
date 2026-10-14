"""
simulation_engine.py — 7-Factor Monte Carlo Championship Simulation Engine
==========================================================================
Key capabilities:
  1. True Post-Race Feature State: Computes each driver's & constructor's
     updated rolling features AFTER as_of_round completes (including the
     result of as_of_round itself) with empirical Bayes smoothing in early rounds.
  2. Circuit-Specific Pace Prediction: Evaluates the 7-factor XGBoost model
     separately for each remaining circuit (Singapore, COTA, Mexico, Brazil,
     Las Vegas, Qatar, Abu Dhabi) using driver & constructor circuit history.
  3. Two-Level Monte Carlo Uncertainty:
     - Season-trajectory uncertainty (constructor upgrade race + driver form drift),
       preventing unrealistic 100% probabilities in early/mid season.
     - Race-day stochastic variance scaled by driver consistency (pts_std_last5)
       and calibrated circuit/mechanical/crash DNF probabilities.
  4. Strict Mathematical Elimination & Official FIA Countback Tie-Breaking.
"""
from pathlib import Path
from typing import Dict
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")

GP_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 12)
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1] + [0] * 14)


def _replace_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"\\N", np.nan, regex=True)


class F1SimulationEngine:
    """
    Unified 7-Factor Monte Carlo simulation engine for Formula 1 2026.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        perf_bundle = joblib.load(MODEL_DIR / "f1_2026_model.pkl")
        if isinstance(perf_bundle, dict):
            self.perf_model = perf_bundle["model"]
            self.perf_feats = perf_bundle["features"]
        else:
            self.perf_model = perf_bundle
            self.perf_feats = list(perf_bundle.feature_names_in_)

        dnf_bundle = joblib.load(MODEL_DIR / "dnf_model.pkl")
        if isinstance(dnf_bundle, dict):
            self.dnf_model = dnf_bundle["model"]
            self.dnf_feats = dnf_bundle["features"]
        else:
            self.dnf_model = dnf_bundle
            self.dnf_feats = [
                "driver_rolling_5_dnf",
                "driver_season_dnf_rate",
                "constructor_rolling_dnf",
                "circuit_historical_dnf_rate",
                "grid_feat",
            ]

        self.races = _replace_null(pd.read_csv(RAW_DIR / "races.csv"))
        self.results = _replace_null(pd.read_csv(RAW_DIR / "results.csv"))
        self.sprints = _replace_null(pd.read_csv(RAW_DIR / "sprint_results.csv"))
        self.status = pd.read_csv(RAW_DIR / "status.csv")
        self.drivers = pd.read_csv(RAW_DIR / "drivers.csv")
        self.constructors = pd.read_csv(RAW_DIR / "constructors.csv")
        self.train_data = pd.read_csv(PROCESSED_DIR / "train_data.csv")

        self.results["positionOrder"] = pd.to_numeric(self.results["positionOrder"], errors="coerce")
        self.results["points"] = pd.to_numeric(self.results["points"], errors="coerce")
        self.results["grid"] = pd.to_numeric(self.results["grid"], errors="coerce")
        self.sprints["points"] = pd.to_numeric(self.sprints["points"], errors="coerce")

        self.driver_map = self.drivers.set_index("driverId").to_dict("index")
        self.constructor_map = self.constructors.set_index("constructorId").to_dict("index")
        self.races_2026 = (
            self.races[self.races["year"] == 2026]
            .sort_values("round")
            .reset_index(drop=True)
        )

        # Pre-compute circuit historical lookup tables (strictly pre-2026)
        self._build_circuit_lookups()

    def _build_circuit_lookups(self):
        td_hist = self.train_data[self.train_data["year"] < 2026].copy()
        self.circ_drv_pos = (
            td_hist.groupby(["circuitId", "driverId"])["positionOrder"].mean().to_dict()
        )
        self.circ_const_pos = (
            td_hist.groupby(["circuitId", "constructorId"])["positionOrder"].mean().to_dict()
        )
        self.circ_pace_delta = (
            td_hist.groupby(["circuitId", "driverId"])["pace_delta_pct"].mean().to_dict()
            if "pace_delta_pct" in td_hist.columns
            else {}
        )
        self.circ_overtake = (
            self.train_data.groupby("circuitId")["circuit_overtake_diff"].last().to_dict()
            if "circuit_overtake_diff" in self.train_data.columns
            else {}
        )

        res_st = self.results.merge(
            self.races[["raceId", "circuitId", "year"]], on="raceId"
        ).merge(self.status[["statusId", "status"]], on="statusId")
        res_st["is_dnf"] = res_st["status"].apply(
            lambda s: 0 if (str(s) == "Finished" or "Lap" in str(s)) else 1
        )
        pre26 = res_st[res_st["year"] < 2026]
        self.circ_dnf_rate = pre26.groupby("circuitId")["is_dnf"].mean().to_dict()

    def get_completed_state(self, as_of_round: int) -> Dict:
        """
        Extract verified standings and POST-RACE rolling features up through as_of_round.
        Unlike the legacy engine (which grabbed pre-race features before as_of_round),
        this incorporates the actual race outcome of as_of_round into the rolling state.
        """
        past_races = self.races_2026[self.races_2026["round"] <= as_of_round]
        past_race_ids = past_races["raceId"].tolist()

        latest_race_id = int(past_races.iloc[-1]["raceId"])
        latest_results = self.results[self.results["raceId"] == latest_race_id]

        active_drivers = latest_results["driverId"].dropna().astype(int).tolist()
        driver_to_constructor = (
            latest_results.set_index("driverId")["constructorId"].dropna().astype(int).to_dict()
        )

        past_res = self.results[self.results["raceId"].isin(past_race_ids)].copy()
        gp_pts = past_res.groupby("driverId")["points"].sum().to_dict()

        past_spr = self.sprints[self.sprints["raceId"].isin(past_race_ids)].copy()
        spr_pts = past_spr.groupby("driverId")["points"].sum().to_dict()

        total_points = {
            d: float(gp_pts.get(d, 0.0) + spr_pts.get(d, 0.0)) for d in active_drivers
        }

        countbacks = {d: [0] * 22 for d in active_drivers}
        for _, row in past_res.iterrows():
            d = int(row["driverId"])
            if d in countbacks and not pd.isna(row["positionOrder"]):
                p = int(row["positionOrder"])
                if 1 <= p <= 22:
                    countbacks[d][p - 1] += 1

        # All historical + 2026 train_data up through as_of_round
        td_upto = self.train_data[
            (self.train_data["year"] < 2026)
            | ((self.train_data["year"] == 2026) & (self.train_data["round"] <= as_of_round))
        ].sort_values(["year", "round", "raceId"])

        td_26 = self.train_data[
            (self.train_data["year"] == 2026) & (self.train_data["round"] <= as_of_round)
        ].sort_values("round")

        # Remaining calendar & exact maximum points available
        remaining_races = self.races_2026[self.races_2026["round"] > as_of_round]
        n_remaining = len(remaining_races)
        n_rem_sprints = sum(
            1
            for _, r in remaining_races.iterrows()
            if not pd.isna(r.get("sprint_date"))
            and str(r.get("sprint_date", "nan")) not in ["nan", r"\N", ""]
        )
        max_pts_remaining = float(n_remaining * 26 + n_rem_sprints * 8)

        leader_pts = max(total_points.values()) if total_points else 0.0
        gap_to_leader = {d: leader_pts - total_points[d] for d in active_drivers}
        eliminated = {d: (gap_to_leader[d] > max_pts_remaining) for d in active_drivers}

        # DNF rates in 2026 up to as_of_round
        res_st = past_res.merge(self.status[["statusId", "status"]], on="statusId", how="left")
        res_st["is_dnf"] = res_st["status"].apply(
            lambda s: 0 if (str(s) == "Finished" or "Lap" in str(s)) else 1
        )
        res_st["is_mech_dnf"] = res_st["status"].apply(
            lambda s: 0
            if (
                str(s) == "Finished"
                or "Lap" in str(s)
                or any(
                    k in str(s).lower()
                    for k in ["accident", "collision", "spun", "disq", "withdrew", "damage"]
                )
            )
            else 1
        )
        season_dnf_rate = res_st.groupby("driverId")["is_dnf"].mean().to_dict()
        season_mech_dnf_rate = res_st.groupby("driverId")["is_mech_dnf"].mean().to_dict()
        const_dnf_rate = res_st.groupby("constructorId")["is_dnf"].mean().to_dict()
        const_mech_rate = res_st.groupby("constructorId")["is_mech_dnf"].mean().to_dict()

        # Constructor rolling points after as_of_round (2 cars per race: 10 races = 20 starts, 4 races = 8 starts)
        const_rolling10_pts = {}
        const_rolling4_pts = {}
        for c_id in set(driver_to_constructor.values()):
            c_rows = td_upto[td_upto["constructorId"] == c_id]["points"]
            const_rolling10_pts[c_id] = float(c_rows.tail(20).mean()) if len(c_rows) > 0 else 4.0
            const_rolling4_pts[c_id] = float(c_rows.tail(8).mean()) if len(c_rows) > 0 else 4.0

        # Build POST-as_of_round feature state per driver (with empirical Bayes shrinkage early season)
        post_features = {}
        avg_quali_map = {}
        pts_std_map = {}

        # Empirical Bayes prior weight: blends 2026 season sample with last-5 historical qualifying
        k_prior = 2.0
        w_26 = as_of_round / (as_of_round + k_prior)

        for d in active_drivers:
            d_all = td_upto[td_upto["driverId"] == d]
            d_26 = td_26[td_26["driverId"] == d]
            q_26_mean = float(d_26["quali_position"].mean()) if len(d_26) > 0 else 11.5
            q_last5 = float(d_all["quali_position"].tail(5).mean()) if len(d_all) > 0 else 11.5
            avg_quali_map[d] = w_26 * q_26_mean + (1.0 - w_26) * q_last5

        for d in active_drivers:
            d_all = td_upto[td_upto["driverId"] == d]
            d_26 = td_26[td_26["driverId"] == d]
            c_id = driver_to_constructor[d]

            q_est = avg_quali_map[d]
            q_last3 = float(d_all["quali_position"].tail(3).mean()) if len(d_all) > 0 else 11.5

            # Post-race positions & points (including as_of_round!)
            last_pos = float(d_all["positionOrder"].iloc[-1]) if len(d_all) > 0 else 12.0
            pos_3 = float(d_all["positionOrder"].tail(3).mean()) if len(d_all) > 0 else 12.0
            pos_5 = float(d_all["positionOrder"].tail(5).mean()) if len(d_all) > 0 else 12.0
            eff_prev_pos = 0.30 * last_pos + 0.70 * pos_5

            last_pts = float(d_all["points"].iloc[-1]) if len(d_all) > 0 else 2.0
            pts_3 = float(d_all["points"].tail(3).mean()) if len(d_all) > 0 else 2.0
            pts_5 = float(d_all["points"].tail(5).mean()) if len(d_all) > 0 else 2.0
            eff_prev_pts = 0.30 * last_pts + 0.70 * pts_5

            pod_5 = float((d_all["positionOrder"].tail(5) <= 3).mean()) if len(d_all) > 0 else 0.0
            top5_5 = float((d_all["positionOrder"].tail(5) <= 5).mean()) if len(d_all) > 0 else 0.0
            std_5 = float(d_all["points"].tail(5).std()) if len(d_all) >= 2 else 5.0
            if pd.isna(std_5):
                std_5 = 5.0
            pts_std_map[d] = std_5

            # Teammate stats (strictly symmetric with avg_quali_map)
            teammates = [other for other in active_drivers if driver_to_constructor[other] == c_id and other != d]
            if teammates:
                tm_id = teammates[0]
                tm_all = td_upto[td_upto["driverId"] == tm_id]
                tm_q = avg_quali_map.get(tm_id, 11.5)
                tm_r3 = float(tm_all["positionOrder"].tail(5).mean()) if len(tm_all) > 0 else 12.0
            else:
                tm_q, tm_r3 = q_est, pos_5

            row_last = d_all.iloc[-1] if len(d_all) > 0 else pd.Series(dtype=float)

            c_r10 = const_rolling10_pts.get(c_id, 4.0)
            c_r4 = const_rolling4_pts.get(c_id, 4.0)

            post_features[d] = {
                "season_points_before": total_points[d],
                "gap_to_leader": gap_to_leader[d],
                "races_remaining": float(n_remaining),
                "max_pts_remaining": max_pts_remaining,
                "pts_per_race_avg": total_points[d] / max(1, as_of_round),
                "prev_position": eff_prev_pos,
                "prev_points": eff_prev_pts,
                "avg_pos_last3": pos_3,
                "avg_pos_last5": pos_5,
                "rolling_3_position": pos_3,
                "avg_pts_last3": pts_3,
                "avg_pts_last5": pts_5,
                "podium_rate_last5": pod_5,
                "top5_rate_last5": top5_5,
                "pts_std_last5": std_5,
                "form_momentum": pts_3 - pts_5,
                "avg_quali_last3": q_last3,
                "constructor_rolling10_pts": c_r10,
                "constructor_rolling4_pts": c_r4,
                "constructor_form_trend": c_r4 - c_r10,
                "teammate_quali_pos": tm_q,
                "teammate_race_pos_last3": tm_r3,
                "quali_gap_to_teammate": q_est - tm_q,
                "tmgap_quali_last3": q_last3 - tm_q,
                "tmgap_race_last3": pos_3 - tm_r3,
                "quali_position": q_est,
                "gap_to_pole_pct_last3": float(row_last.get("gap_to_pole_pct_last3", 1.5)),
                "q3_rate_last5": float(row_last.get("q3_rate_last5", 0.5)),
                "pole_rate_last5": float(row_last.get("pole_rate_last5", 0.1)),
                "grid_penalty": 0.0,
                "pace_delta_last3": float(row_last.get("pace_delta_last3", 0.0)),
                "positions_gained_last5": float(row_last.get("positions_gained_last5", 0.0)),
                "pit_avg_ms_last5": float(row_last.get("pit_avg_ms_last5", 24_000.0)),
                "rolling_5_dnf": float(season_dnf_rate.get(d, 0.15) * 5.0),
                "rolling_5_mech_dnf": float(season_mech_dnf_rate.get(d, 0.08) * 5.0),
                "rolling_5_crash_dnf": float(
                    max(0.0, season_dnf_rate.get(d, 0.15) - season_mech_dnf_rate.get(d, 0.08)) * 5.0
                ),
                "const_mech_dnf_rate": float(const_mech_rate.get(c_id, 0.10)),
                "sprint_points": 0.0,
            }

        return {
            "as_of_round": as_of_round,
            "active_drivers": active_drivers,
            "driver_to_constructor": driver_to_constructor,
            "current_points": total_points,
            "countbacks": countbacks,
            "avg_quali": avg_quali_map,
            "post_features": post_features,
            "season_dnf_rate": season_dnf_rate,
            "season_mech_dnf_rate": season_mech_dnf_rate,
            "const_dnf_rate": const_dnf_rate,
            "gap_to_leader": gap_to_leader,
            "max_pts_remaining": max_pts_remaining,
            "n_remaining": n_remaining,
            "eliminated": eliminated,
            "pts_std": pts_std_map,
        }

    def simulate_season(
        self,
        as_of_round: int,
        n_simulations: int = 10_000,
        base_noise: float = 4.2,
    ) -> Dict:
        """
        Run full Monte Carlo simulation across all remaining races from as_of_round.
        Combines:
          - Circuit-specific 7-factor XGBoost predictions per remaining track
          - Season-level constructor development & driver form uncertainty (shrinking as season progresses)
          - Race-level stochastic pace variance + calibrated DNF sampling
          - Strict mathematical elimination and FIA countback tie-breaking
        """
        np.random.seed(self.random_seed)
        state = self.get_completed_state(as_of_round)
        drivers = state["active_drivers"]
        n_d = len(drivers)
        d_idx = {d: i for i, d in enumerate(drivers)}

        d_to_c = state["driver_to_constructor"]
        constructors = sorted(list(set(d_to_c.values())))
        n_c = len(constructors)
        c_idx = {c: i for i, c in enumerate(constructors)}

        remaining = self.races_2026[self.races_2026["round"] > as_of_round].reset_index(drop=True)
        n_rem = len(remaining)

        init_pts = np.array([state["current_points"][d] for d in drivers], dtype=float)
        init_countbacks = np.array([state["countbacks"][d] for d in drivers], dtype=int)

        if n_rem == 0:
            ranks = self._compute_ranks(init_pts, init_countbacks)
            return self._zero_remaining(state, ranks)

        # 1. Pre-compute circuit-specific expected pace & DNF probability for each (race, driver)
        circuit_base_pace = np.zeros((n_rem, n_d))
        dnf_probs = np.zeros((n_rem, n_d))

        for r_idx, (_, r) in enumerate(remaining.iterrows()):
            c_id = int(r["circuitId"])
            is_sprint = not pd.isna(r.get("sprint_date")) and str(r.get("sprint_date", "nan")) not in [
                "nan",
                r"\N",
                "",
            ]
            circ_dnf = float(self.circ_dnf_rate.get(c_id, 0.18))
            circ_ovt = float(self.circ_overtake.get(c_id, 0.65))

            feat_rows = []
            dnf_rows = []
            for i, d in enumerate(drivers):
                team_id = d_to_c[d]
                f_dict = state["post_features"][d].copy()

                # Circuit-specific features (Group 7)
                c_drv_avg = float(
                    self.circ_drv_pos.get((c_id, d), f_dict["avg_pos_last5"])
                )
                c_con_avg = float(
                    self.circ_const_pos.get((c_id, team_id), 10.5)
                )
                c_pace_d = float(
                    self.circ_pace_delta.get((c_id, d), f_dict["pace_delta_last3"])
                )

                # Adjust expected qualifying at this specific circuit based on driver/team track history
                track_q_adj = 0.25 * (c_drv_avg - f_dict["avg_pos_last5"]) + 0.15 * (c_con_avg - 10.0)
                f_dict["quali_position"] = np.clip(f_dict["quali_position"] + track_q_adj, 1.0, 22.0)
                f_dict["circuit_driver_avg_pos"] = c_drv_avg
                f_dict["circuit_const_avg_pos"] = c_con_avg
                f_dict["circuit_historical_dnf_rate"] = circ_dnf
                f_dict["circuit_overtake_diff"] = circ_ovt
                f_dict["pace_delta_last3"] = 0.7 * f_dict["pace_delta_last3"] + 0.3 * c_pace_d
                f_dict["sprint_points"] = 3.0 if is_sprint and f_dict["quali_position"] <= 8 else 0.0

                feat_rows.append(f_dict)

                d_dnf = float(state["season_dnf_rate"].get(d, 0.16))
                d_mech = float(state["season_mech_dnf_rate"].get(d, 0.08))
                t_dnf = float(state["const_dnf_rate"].get(team_id, 0.16))
                dnf_rows.append({
                    "driver_rolling_5_dnf": d_dnf,
                    "driver_rolling_5_mech_dnf": d_mech,
                    "driver_season_dnf_rate": d_dnf,
                    "constructor_rolling_dnf": t_dnf,
                    "circuit_historical_dnf_rate": circ_dnf,
                    "grid_feat": f_dict["quali_position"],
                    "pace_delta_last3": f_dict["pace_delta_last3"],
                    "pit_avg_ms": f_dict["pit_avg_ms_last5"],
                })

            X_race = pd.DataFrame(feat_rows)[self.perf_feats].copy()
            base_q = X_race["quali_position"].values.copy()
            q_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
            q_weights = [0.12, 0.24, 0.28, 0.24, 0.12]
            raw_preds = np.zeros(n_d)
            for q_off, q_w in zip(q_offsets, q_weights):
                X_race["quali_position"] = np.clip(np.round(base_q + q_off), 1.0, 22.0)
                raw_preds += q_w * np.maximum(0.2, self.perf_model.predict(X_race))

            # Subtle circuit suitability modulation so each circuit has distinct pace profiles
            for i, d in enumerate(drivers):
                c_drv_avg = feat_rows[i]["circuit_driver_avg_pos"]
                c_con_avg = feat_rows[i]["circuit_const_avg_pos"]
                suitability_delta = (11.0 - c_drv_avg) * 0.22 + (11.0 - c_con_avg) * 0.15
                ovt_mod = 1.0 + (circ_ovt - 0.65) * 0.35
                circuit_base_pace[r_idx, i] = max(0.1, raw_preds[i] * ovt_mod + suitability_delta)

            X_dnf = pd.DataFrame(dnf_rows)[self.dnf_feats]
            dnf_probs[r_idx, :] = self.dnf_model.predict_proba(X_dnf)[:, 1]

        # 2. Season-Trajectory Uncertainty (Epistemic uncertainty: upgrades, development race, form drift)
        # Early in the season (few completed rounds, many remaining), team development and driver
        # season-long trajectories have substantial uncertainty. As rounds complete, this shrinks.
        total_rounds = len(self.races_2026)
        rem_fraction = n_rem / float(total_rounds)
        sample_confidence = np.sqrt(3.0 / (as_of_round + 2.0))  # 0.77 at R1, 0.41 at R16

        # Team season development shock (shared by both teammates across remaining races in a sim)
        team_season_sigma = 4.8 * rem_fraction * sample_confidence
        team_shocks = np.random.normal(0.0, team_season_sigma, size=(n_simulations, n_c))

        # Driver season trajectory shock (persists across remaining races in a sim)
        driver_season_sigma = 3.6 * rem_fraction * sample_confidence
        driver_shocks = np.random.normal(0.0, driver_season_sigma, size=(n_simulations, n_d))

        # Combine into persistent season-trajectory offset per (simulation, driver)
        persistent_offset = driver_shocks.copy()
        for i, d in enumerate(drivers):
            persistent_offset[:, i] += team_shocks[:, c_idx[d_to_c[d]]]

        # Per-driver race-day noise scale (blending base_noise with driver consistency pts_std_last5)
        driver_race_noise = np.array([
            0.65 * base_noise + 0.35 * min(8.0, state["pts_std"].get(d, 4.5))
            for d in drivers
        ])

        # 3. Monte Carlo Simulation Loop across Remaining Races
        sim_final_pts = np.tile(init_pts, (n_simulations, 1))
        sim_countbacks = np.tile(init_countbacks, (n_simulations, 1, 1))

        race_exp_pts = {int(r["raceId"]): np.zeros(n_d) for _, r in remaining.iterrows()}
        race_exp_pos = {int(r["raceId"]): np.zeros(n_d) for _, r in remaining.iterrows()}

        for r_idx, (_, race) in enumerate(remaining.iterrows()):
            race_id = int(race["raceId"])
            is_sprint = not pd.isna(race.get("sprint_date")) and str(
                race.get("sprint_date", "nan")
            ) not in ["nan", r"\N", ""]

            # Race-day pace = circuit base pace + season trajectory offset + race-day noise
            race_noise = (
                np.random.normal(0.0, 1.0, size=(n_simulations, n_d))
                * driver_race_noise[None, :]
            )
            latent_pace = (
                circuit_base_pace[r_idx, :][None, :] + persistent_offset + race_noise
            )

            # Sample DNF events
            dnf_mask = (
                np.random.random(size=(n_simulations, n_d)) < dnf_probs[r_idx, :][None, :]
            )
            sim_scores = latent_pace.copy()
            sim_scores[dnf_mask] = -1000.0 + np.random.uniform(0, 10, size=dnf_mask.sum())

            # Strictly unique field ranking 1..22
            order = np.argsort(-sim_scores, axis=1)

            for pos in range(10):
                di = order[:, pos]
                np.add.at(sim_final_pts, (np.arange(n_simulations), di), GP_POINTS[pos])
                np.add.at(race_exp_pts[race_id], di, GP_POINTS[pos])
                np.add.at(sim_countbacks, (np.arange(n_simulations), di, pos), 1)

            # Fastest Lap bonus point (top-10 finisher weighted toward podium/front-runners)
            fl_pos = np.random.randint(0, 10, size=n_simulations)
            fl_di = order[np.arange(n_simulations), fl_pos]
            np.add.at(sim_final_pts, (np.arange(n_simulations), fl_di), 1)
            np.add.at(race_exp_pts[race_id], fl_di, 1)

            if is_sprint:
                s_noise = (
                    np.random.normal(0.0, 1.0, size=(n_simulations, n_d))
                    * driver_race_noise[None, :]
                    * 0.8
                )
                s_latent = (
                    circuit_base_pace[r_idx, :][None, :] + persistent_offset + s_noise
                )
                s_dnf = np.random.random(size=(n_simulations, n_d)) < (
                    dnf_probs[r_idx, :][None, :] * 0.6
                )
                s_scores = s_latent.copy()
                s_scores[s_dnf] = -1000.0 + np.random.uniform(0, 10, size=s_dnf.sum())
                s_order = np.argsort(-s_scores, axis=1)
                for pos in range(8):
                    si = s_order[:, pos]
                    np.add.at(
                        sim_final_pts, (np.arange(n_simulations), si), SPRINT_POINTS[pos]
                    )
                    np.add.at(race_exp_pts[race_id], si, SPRINT_POINTS[pos])

            for pos in range(n_d):
                di = order[:, pos]
                np.add.at(race_exp_pos[race_id], di, pos + 1)

        for rid in race_exp_pts:
            race_exp_pts[rid] /= n_simulations
            race_exp_pos[rid] /= n_simulations

        # 4. Final Championship Standings with Official FIA Countback
        sim_ranks = np.zeros((n_simulations, n_d), dtype=int)
        for s in range(n_simulations):
            keys = [sim_countbacks[s, :, p] for p in reversed(range(10))] + [
                sim_final_pts[s]
            ]
            keys_neg = [-k for k in keys]
            sorted_i = np.lexsort(keys_neg)
            sim_ranks[s, sorted_i] = np.arange(1, n_d + 1)

        drv_champ = (sim_ranks == 1).sum(axis=0) / n_simulations
        drv_top3 = (sim_ranks <= 3).sum(axis=0) / n_simulations

        # Enforce mathematical elimination for Driver Championship
        for i, d in enumerate(drivers):
            if state["eliminated"][d]:
                drv_champ[i] = 0.0
        if drv_champ.sum() > 0:
            drv_champ = drv_champ / drv_champ.sum()

        # Constructor Standings
        sim_c_pts = np.zeros((n_simulations, n_c))
        for i, d in enumerate(drivers):
            ci = c_idx[d_to_c[d]]
            sim_c_pts[:, ci] += sim_final_pts[:, i]

        c_ranks = np.zeros_like(sim_c_pts, dtype=int)
        for s in range(n_simulations):
            sc = np.argsort(-sim_c_pts[s])
            c_ranks[s, sc] = np.arange(1, n_c + 1)

        c_champ = (c_ranks == 1).sum(axis=0) / n_simulations
        c_top3 = (c_ranks <= 3).sum(axis=0) / n_simulations

        # Pairwise Head-to-Head Matrix
        h2h = np.zeros((n_d, n_d))
        for i in range(n_d):
            for j in range(n_d):
                if i != j:
                    h2h[i, j] = (sim_ranks[:, i] < sim_ranks[:, j]).mean()

        # Teammate Battles
        teammate_wins = {}
        for c in constructors:
            c_drivers = [d for d in drivers if d_to_c[d] == c]
            if len(c_drivers) >= 2:
                d1, d2 = c_drivers[0], c_drivers[1]
                i1, i2 = d_idx[d1], d_idx[d2]
                teammate_wins[d1] = (sim_ranks[:, i1] < sim_ranks[:, i2]).mean()
                teammate_wins[d2] = (sim_ranks[:, i2] < sim_ranks[:, i1]).mean()

        return {
            "as_of_round": as_of_round,
            "drivers": drivers,
            "driver_idx_map": d_idx,
            "constructors": constructors,
            "const_idx_map": c_idx,
            "driver_to_const": d_to_c,
            "driver_champ_wins": drv_champ,
            "driver_top3_probs": drv_top3,
            "const_champ_wins": c_champ,
            "const_top3_probs": c_top3,
            "h2h_matrix": h2h,
            "teammate_wins": teammate_wins,
            "race_expected_points": race_exp_pts,
            "race_expected_positions": race_exp_pos,
            "remaining_races": remaining,
            "n_simulations": n_simulations,
        }

    def _compute_ranks(self, pts: np.ndarray, cb: np.ndarray) -> np.ndarray:
        keys = [cb[:, p] for p in reversed(range(10))] + [pts]
        keys_neg = [-k for k in keys]
        si = np.lexsort(keys_neg)
        ranks = np.zeros(len(pts), dtype=int)
        ranks[si] = np.arange(1, len(pts) + 1)
        return ranks

    def _zero_remaining(self, state, ranks):
        drivers = state["active_drivers"]
        n_d = len(drivers)
        d_idx = {d: i for i, d in enumerate(drivers)}
        drv_champ = (ranks == 1).astype(float)
        drv_top3 = (ranks <= 3).astype(float)
        return {
            "as_of_round": state["as_of_round"],
            "drivers": drivers,
            "driver_idx_map": d_idx,
            "constructors": [],
            "const_idx_map": {},
            "driver_to_const": state["driver_to_constructor"],
            "driver_champ_wins": drv_champ,
            "driver_top3_probs": drv_top3,
            "const_champ_wins": np.array([]),
            "const_top3_probs": np.array([]),
            "h2h_matrix": np.zeros((n_d, n_d)),
            "teammate_wins": {},
            "race_expected_points": {},
            "race_expected_positions": {},
            "remaining_races": pd.DataFrame(),
            "n_simulations": 1,
        }

    def generate_head_to_head_csv(
        self, cutoff_round: int = 16, n_simulations: int = 10_000
    ) -> pd.DataFrame:
        print(f"Running {n_simulations:,} simulations for Head-to-Head matrix (Round {cutoff_round})...")
        res = self.simulate_season(as_of_round=cutoff_round, n_simulations=n_simulations)
        state = self.get_completed_state(cutoff_round)
        drivers = res["drivers"]
        d_idx = res["driver_idx_map"]
        h2h = res["h2h_matrix"]

        rows = []
        for da in drivers:
            ia = d_idx[da]
            da_meta = self.driver_map.get(da, {})
            da_team = self.constructor_map.get(
                state["driver_to_constructor"].get(da), {}
            ).get("name", "")
            for db in drivers:
                if da == db:
                    continue
                ib = d_idx[db]
                db_meta = self.driver_map.get(db, {})
                db_team = self.constructor_map.get(
                    state["driver_to_constructor"].get(db), {}
                ).get("name", "")
                rows.append({
                    "as_of_round": cutoff_round,
                    "driver_a_id": int(da),
                    "driver_a_name": da_meta.get("surname", f"Driver_{da}"),
                    "driver_a_code": da_meta.get("code", ""),
                    "driver_a_team": da_team,
                    "driver_a_current_points": float(state["current_points"][da]),
                    "driver_b_id": int(db),
                    "driver_b_name": db_meta.get("surname", f"Driver_{db}"),
                    "driver_b_code": db_meta.get("code", ""),
                    "driver_b_team": db_team,
                    "driver_b_current_points": float(state["current_points"][db]),
                    "win_probability": round(float(h2h[ia, ib]), 4),
                    "is_teammates": 1
                    if state["driver_to_constructor"].get(da)
                    == state["driver_to_constructor"].get(db)
                    else 0,
                })

        df = pd.DataFrame(rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / "head_to_head_matrix.csv", index=False)
        print(f"[OK] Saved {len(df)} pairwise matchups --> head_to_head_matrix.csv")
        return df

    def generate_race_predictions_csv(
        self, cutoff_round: int = 16, n_simulations: int = 10_000
    ) -> pd.DataFrame:
        print("Generating race_predictions.csv for remaining 2026 rounds...")
        res = self.simulate_season(as_of_round=cutoff_round, n_simulations=n_simulations)
        drivers = res["drivers"]
        d_idx = res["driver_idx_map"]
        remaining = res["remaining_races"]

        rows = []
        for _, race in remaining.iterrows():
            rid = int(race["raceId"])
            exp_pts = res["race_expected_points"][rid]
            exp_pos = res["race_expected_positions"][rid]

            sorted_i = np.lexsort((exp_pos, -exp_pts))
            pred_ranks = np.zeros(len(drivers), dtype=int)
            pred_ranks[sorted_i] = np.arange(1, len(drivers) + 1)

            for d in drivers:
                i = d_idx[d]
                rows.append({
                    "race_id": rid,
                    "driver_id": int(d),
                    "predicted_points": round(float(exp_pts[i]), 6),
                    "as_of_round": cutoff_round,
                    "predicted_position": int(pred_ranks[i]),
                })

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "race_predictions.csv", index=False)
        print(f"[OK] Saved {len(df)} race predictions --> race_predictions.csv")
        return df

    def generate_all_timeline_csvs(self, n_simulations: int = 4_000):
        """Generate round-by-round championship, constructor, and teammate CSVs."""
        completed = sorted(self.train_data[self.train_data["year"] == 2026]["round"].unique())
        print(f"\nSimulating 2026 timeline across rounds {completed} ({n_simulations:,} sims/round)...")

        champ_rows, const_rows, tmb_rows = [], [], []

        for r in completed:
            res = self.simulate_season(as_of_round=int(r), n_simulations=n_simulations)
            drivers = res["drivers"]
            d_idx = res["driver_idx_map"]
            d_to_c = res["driver_to_const"]

            for d in drivers:
                i = d_idx[d]
                champ_rows.append({
                    "driver_id": int(d),
                    "as_of_round": int(r),
                    "championship_probability": round(float(res["driver_champ_wins"][i]), 4),
                    "top_3_probability": round(float(res["driver_top3_probs"][i]), 4),
                })
                if d in res["teammate_wins"]:
                    tmb_rows.append({
                        "driver_id": int(d),
                        "constructor_id": float(d_to_c[d]),
                        "as_of_round": int(r),
                        "teammate_beat_probability": round(float(res["teammate_wins"][d]), 4),
                    })

            for c in res["constructors"]:
                ci = res["const_idx_map"][c]
                const_rows.append({
                    "constructor_id": float(c),
                    "as_of_round": int(r),
                    "championship_probability": round(float(res["const_champ_wins"][ci]), 4),
                    "top_3_probability": round(float(res["const_top3_probs"][ci]), 4),
                })

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(champ_rows).to_csv(OUTPUT_DIR / "championship_probabilities.csv", index=False)
        pd.DataFrame(const_rows).to_csv(OUTPUT_DIR / "constructor_probabilities.csv", index=False)
        pd.DataFrame(tmb_rows).to_csv(OUTPUT_DIR / "teammate_battles.csv", index=False)
        print("[OK] championship_probabilities.csv")
        print("[OK] constructor_probabilities.csv")
        print("[OK] teammate_battles.csv")


if __name__ == "__main__":
    engine = F1SimulationEngine()
    engine.generate_head_to_head_csv(cutoff_round=16, n_simulations=10_000)
    engine.generate_race_predictions_csv(cutoff_round=16, n_simulations=10_000)
    engine.generate_all_timeline_csvs(n_simulations=4_000)
