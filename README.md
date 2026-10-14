# 🏎️ Formula 1 2026 Season Analytics Dashboard & 7-Factor Prediction Engine

![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-7_Factor_Adaptive_ML-FF6600?style=for-the-badge)
![DAX](https://img.shields.io/badge/DAX-Advanced-blue?style=for-the-badge)
![Data Modeling](https://img.shields.io/badge/Data_Modeling-Star_Schema-success?style=for-the-badge)

## 📌 Project Overview
This project is an end-to-end data analytics and predictive intelligence system built for the **Formula 1 2026 season**. It combines historical and current-season telemetry (`lap_times.csv`, `pit_stops.csv`), qualifying session times (`q1`, `q2`, `q3`), sprint classifications, and race results with a **7-Factor Machine Learning and 10,000-run Monte Carlo Simulation Engine**.

The output feeds directly into an F1 TV-inspired dark-mode **Power BI** dashboard featuring dynamic driver head-to-head battle arenas, circuit-specific race previews, mechanical vs. crash reliability risk analysis, and full-season Driver and Constructor Championship probability trajectories.

---

## 🧠 7-Factor Feature Engineering Architecture (`src/build_features.py`)

The feature pipeline constructs **56 leakage-free columns** across **7 Essential Factor Groups** using strict chronological shifting (`.shift(1)`):

1. **Championship Position & Points**: Cumulative GP + Saturday Sprint points (`season_points_before`), points gap to leader (`gap_to_leader`), remaining races & maximum points still available (`max_pts_remaining`), mathematical elimination flags (`is_eliminated`), and scale-invariant scoring rate (`pts_per_race_avg`).
2. **Recent Driver Form**: Rolling 3- and 5-race finishing positions and points (`avg_pos_last3/5`, `avg_pts_last3/5`), podium & top-5 finish rates (`podium_rate_last5`, `top5_rate_last5`), consistency (`pts_std_last5`), and form momentum (`avg_pts_last3 - avg_pts_last5`).
3. **Constructor & Car Performance**: Separates car competitiveness from driver skill via team rolling 4- and 10-race points (`constructor_rolling4_pts`, `constructor_rolling10_pts`), car upgrade momentum (`constructor_form_trend`), and teammate-relative qualifying and race gaps (`quali_gap_to_teammate`, `tmgap_race_last3`).
4. **Qualifying Performance**: Starting qualifying classification (`quali_position`), millisecond-parsed percentage gap to pole position (`gap_to_pole_pct_last3`), Q3 appearance rate (`q3_rate_last5`), pole conversion rate (`pole_rate_last5`), and grid penalties (`grid - quali_position`).
5. **Race Pace & Strategy**: Median race lap-time delta % vs. the field computed from 24.8 MB of lap telemetry (`pace_delta_last3`), average positions gained on Sunday (`positions_gained_last5`), and pit-stop duration consistency (`pit_avg_ms_last5`).
6. **Reliability & DNF Risk**: Separates **Mechanical DNFs** (*Engine, Gearbox, Hydraulics, Brakes*) from **Crash/Driver DNFs** (*Accident, Collision, Spun off*) alongside constructor mechanical failure rates (`const_mech_dnf_rate`) and historical circuit attrition (`circuit_historical_dnf_rate`).
7. **Circuit Suitability**: Driver and constructor historical average finish at each specific `circuitId` (`circuit_driver_avg_pos`, `circuit_const_avg_pos`) and track overtaking difficulty (`circuit_overtake_diff`).

---

## 📈 Model Hierarchy & Chronological Validation (2025 Backtest)

### A. Race Performance Model (`models/f1_2026_model.pkl`)
Trained via `src/train_model.py` using scale-invariant pre-race features across all 7 factor groups:

| Model | Architecture | Validation MAE (pts) | Validation RMSE (pts) | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Model A** | Linear Quali Heuristic | 3.731 | 5.118 | Baseline mapping qualifying position to points. |
| **Model B** | 7-Factor Unweighted XGBoost | 2.790 | 4.121 | Standard gradient boosted trees across all 7 factor groups. |
| **Model C (Production)** | **7-Factor Recency-Weighted XGBoost** | **2.768** | **4.144** | **Exponential time decay ($\tau = 4$ yrs) with L1/L2 regularization.** |

*Top Feature Importances*: `quali_position` (30.90%), `pts_per_race_avg` (10.04%), `avg_pts_last5` (9.46%), `top5_rate_last5` (5.52%), `podium_rate_last5` (3.10%), `constructor_rolling10_pts` (2.71%), `quali_gap_to_teammate` (2.28%).

### B. Reliability & DNF Classifier (`models/dnf_model.pkl`)
Trained via `src/predict_dnf.py` on modern turbo-hybrid era observations (2014–2026, 5,457 records):
* **Architecture**: `StandardScaler` + Regularized `LogisticRegression` (`C=0.5`).
* **2025 Validation Metrics**: **Brier Score**: `0.1097` (beating baseline `0.1119`) | **ROC-AUC**: `0.5895` | **Log Loss**: `0.3762`.
* **Features**: Driver rolling 5 DNF rate, driver rolling 5 mechanical DNF rate, season DNF rate, constructor rolling 10 DNF rate, historical circuit DNF rate, grid estimate, rolling pace delta, and pit-stop duration.

---

## 🎲 Unified Monte Carlo Simulation Engine (`src/simulation_engine.py`)

Executes **10,000 full-season simulations** across remaining 2026 rounds with:
1. **True Post-Race State & Empirical Bayes Smoothing**: Updates rolling features immediately after each completed round and applies empirical Bayes shrinkage in early rounds.
2. **5-Point Qualifying Quadrature**: Integrates XGBoost predictions across a 5-point qualifying distribution around each driver's track-adjusted qualifying estimate to eliminate tree step-function cliffs.
3. **Two-Level Uncertainty Modeling**:
   * *Epistemic (Season-Trajectory) Uncertainty*: Persistent team upgrade shocks and driver trajectory shocks scaled by remaining races and sample size.
   * *Aleatory (Race-Day) Uncertainty*: Per-race Gaussian pace noise scaled by driver consistency (`pts_std_last5`) plus calibrated DNF sampling.
4. **Coherent Field Ranking & FIA Rules**: Strictly unique finishing positions (`1..22`), Grand Prix (`25..1` + Fastest Lap) and Sprint (`8..1`) scoring, mathematical elimination enforcement, and official FIA countback tie-breaking.

---

## 📊 Generated Power BI Outputs (`output/`)

| File | Rows | Schema & Description |
| :--- | :---: | :--- |
| `head_to_head_matrix.csv` | 462 | Pairwise championship finish probabilities for all $22 \times 21$ driver pairs ($P(A > B) + P(B > A) = 1.0$). |
| `next_race_dnf_predictions.csv` | 22 | Calibrated DNF & finish probabilities (12 columns) for Round 17 Singapore GP. |
| `race_predictions.csv` | 154 | Circuit-specific expected points and unique finishing positions (`1..22`) for Rounds 17–23. |
| `championship_probabilities.csv` | 354 | Round-by-round trajectory (Rounds 1–16) of Driver Championship win & Top-3 probabilities. |
| `constructor_probabilities.csv` | 178 | Round-by-round trajectory (Rounds 1–16) of Constructor Championship win & Top-3 probabilities. |
| `teammate_battles.csv` | 354 | Round-by-round probability of finishing ahead of one's teammate in the championship. |

---

## 🛠️ Execution & Unit Testing

```bash
# 1. Build 56-column leakage-free feature dataset
python src/build_features.py

# 2. Train 7-factor XGBoost pace model & calibrated DNF classifier
python src/train_model.py
python src/predict_dnf.py

# 3. Run Monte Carlo simulations and export all 6 Power BI CSVs
python src/simulation_engine.py

# 4. Run automated 8-test verification suite
python tests/test_prediction_engine.py
```

## 🤝 Contact & Credits
* **Author:** Raghavendra Yecham
* **LinkedIn:** [Raghavendra Yecham](https://www.linkedin.com/in/raghavendra-yecham-b0318b2b6/)
