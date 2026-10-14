import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")

def simulate():
    print("Loading model and data...")
    model = joblib.load(MODEL_DIR / "f1_2026_model.pkl")
    
    races = pd.read_csv(RAW_DIR / "races.csv")
    races_2026 = races[races['year'] == 2026].sort_values('round')
    
    train_data = pd.read_csv(DATA_DIR / "train_data.csv")
    train_2026 = train_data[train_data['year'] == 2026].copy()
    
    if len(train_2026) == 0:
        print("No 2026 data found!")
        return
        
    completed_rounds = sorted(train_2026['round'].unique())
    
    champ_probs = []
    
    # We increase the noise variance and add random DNFs to capture the true chaotic nature of F1
    # This prevents the model from assuming 100% certainty too early in the season.
    noise_scale = 5.0 
    N_SIMULATIONS = 300
    
    print(f"Running historical simulation across all {len(completed_rounds)} completed rounds...")
    
    features_cols = ['quali_position', 'prev_position', 'rolling_3_position', 'season_points_before', 'rolling_5_dnf']
    
    for current_round in completed_rounds:
        # Filter data as if we were exactly at this round in the past
        data_up_to_now = train_2026[train_2026['round'] <= current_round].copy()
        
        latest_race_data = data_up_to_now[data_up_to_now['round'] == current_round].copy()
        drivers = latest_race_data['driverId'].unique()
        
        base_features = latest_race_data.groupby('driverId').last().reset_index()
        
        avg_quali = data_up_to_now.groupby('driverId')['quali_position'].mean().reset_index()
        avg_quali.rename(columns={'quali_position': 'avg_quali'}, inplace=True)
        base_features = base_features.merge(avg_quali, on='driverId', how='left')
        base_features['quali_position'] = base_features['avg_quali']
        
        remaining_races = races_2026[races_2026['round'] > current_round]
        
        current_points = {}
        for _, row in latest_race_data.iterrows():
            current_points[row['driverId']] = row['season_points_before'] + row['points']
            
        simulation_results = {d: {'wins': 0, 'top_3': 0} for d in drivers}
        
        if len(remaining_races) == 0:
            sorted_standings = sorted(current_points.items(), key=lambda x: x[1], reverse=True)
            winner_id = sorted_standings[0][0]
            if winner_id in simulation_results:
                simulation_results[winner_id]['wins'] = N_SIMULATIONS
            for rank in range(min(3, len(sorted_standings))):
                if sorted_standings[rank][0] in simulation_results:
                    simulation_results[sorted_standings[rank][0]]['top_3'] = N_SIMULATIONS
        else:
            X = base_features[features_cols].copy().fillna(20)
            base_preds = model.predict(X)
            
            for i in range(N_SIMULATIONS):
                sim_points = current_points.copy()
                
                for _ in range(len(remaining_races)):
                    # Add variance
                    noise = np.random.normal(0, noise_scale, size=len(base_preds))
                    
                    # 10% chance of a DNF (0 points) for any driver in any race
                    dnf_mask = np.random.random(size=len(base_preds)) < 0.10
                    
                    race_pts = np.maximum(0, base_preds + noise)
                    race_pts[dnf_mask] = 0
                    
                    for idx, driver_id in enumerate(base_features['driverId']):
                        sim_points[driver_id] += race_pts[idx]
                        
                sorted_standings = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
                winner_id = sorted_standings[0][0]
                simulation_results[winner_id]['wins'] += 1
                for rank in range(min(3, len(sorted_standings))):
                    simulation_results[sorted_standings[rank][0]]['top_3'] += 1
                    
        for d in drivers:
            champ_probs.append({
                'driver_id': d,
                'as_of_round': current_round,
                'championship_probability': simulation_results[d]['wins'] / N_SIMULATIONS,
                'top_3_probability': simulation_results[d]['top_3'] / N_SIMULATIONS
            })
            
    champ_df = pd.DataFrame(champ_probs)
    champ_df.to_csv(OUTPUT_DIR / "championship_probabilities.csv", index=False)
    print(f"Generated {len(champ_df)} probability records across {len(completed_rounds)} rounds.")

if __name__ == "__main__":
    simulate()
