"""
Simulates remaining 2026 races and exports race_predictions.csv and championship probabilities
using the validated shared F1 Simulation Engine.
"""
from src.simulation_engine import F1SimulationEngine

def simulate():
    engine = F1SimulationEngine()
    engine.generate_race_predictions_csv(cutoff_round=16, n_simulations=10000)

if __name__ == "__main__":
    simulate()
