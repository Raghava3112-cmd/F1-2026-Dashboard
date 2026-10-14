"""
Generates full timeline championship, constructor, and teammate probabilities
using the validated shared F1 Simulation Engine.
"""
from src.simulation_engine import F1SimulationEngine

def simulate():
    engine = F1SimulationEngine()
    engine.generate_all_timeline_csvs(n_simulations=1000)

if __name__ == "__main__":
    simulate()
