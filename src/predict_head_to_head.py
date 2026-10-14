"""
Generates head_to_head_matrix.csv using the validated shared F1 Simulation Engine.
"""
from src.simulation_engine import F1SimulationEngine

def generate_h2h_matrix():
    engine = F1SimulationEngine()
    engine.generate_head_to_head_csv(cutoff_round=16, n_simulations=10000)

if __name__ == "__main__":
    generate_h2h_matrix()
