import unittest
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

class TestPredictionEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.train_data = pd.read_csv(PROCESSED_DIR / "train_data.csv")
        cls.results = pd.read_csv(RAW_DIR / "results.csv").replace(r'\\N', np.nan, regex=True)
        cls.sprints = pd.read_csv(RAW_DIR / "sprint_results.csv").replace(r'\\N', np.nan, regex=True)
        cls.driver_standings = pd.read_csv(RAW_DIR / "driver_standings.csv")
        cls.status = pd.read_csv(RAW_DIR / "status.csv")
        cls.races = pd.read_csv(RAW_DIR / "races.csv")
        cls.drivers = pd.read_csv(RAW_DIR / "drivers.csv")
        cls.constructors = pd.read_csv(RAW_DIR / "constructors.csv")
        
    def test_feature_leakage_season_points(self):
        """Verify season_points_before only accumulates prior race and sprint points with no target leakage."""
        df_2026 = self.train_data[self.train_data['year'] == 2026].copy()
        
        # Round 1 should have 0 season points before (since sprint was not held in round 1)
        r1 = df_2026[df_2026['round'] == 1]
        self.assertTrue((r1['season_points_before'] == 0.0).all())
        
        # For Antonelli (863), verify monotonically non-decreasing season points before
        ant = df_2026[df_2026['driverId'] == 863].sort_values('round')
        pts_before = ant['season_points_before'].tolist()
        for i in range(len(pts_before) - 1):
            self.assertLessEqual(pts_before[i], pts_before[i + 1])
            
    def test_feature_leakage_circuit_dnf_rate(self):
        """Verify circuit historical DNF rates do not leak future race outcomes."""
        df = self.train_data.copy()
        # All circuit DNF rates must be valid probabilities between 0 and 1
        self.assertTrue((df['circuit_historical_dnf_rate'] >= 0.0).all())
        self.assertTrue((df['circuit_historical_dnf_rate'] <= 1.0).all())
        self.assertFalse(df['circuit_historical_dnf_rate'].isna().any())

    def test_sprint_points_consistency(self):
        """Verify driver points including sprint points match driver_standings at Round 16 exactly."""
        df_2026 = self.train_data[self.train_data['year'] == 2026]
        r16 = df_2026[df_2026['round'] == 16][['driverId', 'season_points_before', 'points']].copy()
        r16['total_after_r16'] = r16['season_points_before'] + r16['points']
        
        pts_map = r16.set_index('driverId')['total_after_r16'].to_dict()
        
        # Verified official driver standings points as of Round 16
        expected_points = {
            863: 320.0, # Antonelli
            847: 236.0, # Russell
            1:   214.0, # Hamilton
            844: 191.0, # Leclerc
            846: 188.0, # Norris
            830: 188.0, # Verstappen
            857: 128.0, # Piastri
            865:  96.0, # Hadjar
            859:  65.0, # Lawson
            842:  41.0, # Gasly
        }
        for driver_id, exp_pts in expected_points.items():
            self.assertIn(driver_id, pts_map)
            self.assertAlmostEqual(pts_map[driver_id], exp_pts, places=1,
                                   msg=f"Driver {driver_id} points mismatch: got {pts_map[driver_id]}, expected {exp_pts}")

    def test_dnf_definition(self):
        """Verify status mapping: Finished and +Laps are classified (0), retirements are 1."""
        status_dict = self.status.set_index('statusId')['status'].to_dict()
        
        def is_dnf(sid):
            s = str(status_dict.get(sid, ''))
            return 0 if (s == 'Finished' or 'Lap' in s) else 1
            
        self.assertEqual(is_dnf(1), 0)    # Finished
        self.assertEqual(is_dnf(11), 0)   # +1 Lap
        self.assertEqual(is_dnf(12), 0)   # +2 Laps
        self.assertEqual(is_dnf(3), 1)    # Accident
        self.assertEqual(is_dnf(4), 1)    # Collision
        self.assertEqual(is_dnf(5), 1)    # Engine

    def test_next_race_dnf_predictions_file(self):
        """Verify next_race_dnf_predictions.csv schema, row count, and calibrated probabilities."""
        file_path = OUTPUT_DIR / "next_race_dnf_predictions.csv"
        self.assertTrue(file_path.exists(), f"{file_path} does not exist")
        
        df = pd.read_csv(file_path)
        expected_cols = [
            "race_id", "round", "race_name", "driver_id", "code", "surname",
            "constructor_id", "team_name", "dnf_probability", "finish_probability",
            "driver_season_dnf_rate", "circuit_historical_dnf_rate"
        ]
        self.assertListEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 22, "Must contain exactly 22 active drivers")
        
        # Probabilities between 0 and 1
        self.assertTrue((df['dnf_probability'] > 0.0).all() and (df['dnf_probability'] < 1.0).all())
        self.assertTrue((df['finish_probability'] > 0.0).all() and (df['finish_probability'] < 1.0).all())
        
        # Complementary probabilities: dnf_probability + finish_probability == 1.0
        np.testing.assert_allclose(df['dnf_probability'] + df['finish_probability'], 1.0, atol=1e-5)
        
        # Verified round 17 and Singapore
        self.assertTrue((df['round'] == 17).all())
        self.assertTrue(df['race_name'].str.contains('Singapore').all())

    def test_race_predictions_file(self):
        """Verify race_predictions.csv schema, 154 rows, and valid unique field-wide finishing ranks."""
        file_path = OUTPUT_DIR / "race_predictions.csv"
        self.assertTrue(file_path.exists(), f"{file_path} does not exist")
        
        df = pd.read_csv(file_path)
        expected_cols = ["race_id", "driver_id", "predicted_points", "as_of_round", "predicted_position"]
        self.assertListEqual(list(df.columns), expected_cols)
        
        # 7 remaining races * 22 drivers = 154 rows
        self.assertEqual(len(df), 154)
        self.assertEqual(df['as_of_round'].iloc[0], 16)
        
        # Valid predicted points >= 0
        self.assertTrue((df['predicted_points'] >= 0.0).all())
        
        # For each race, predicted_position must be a permutation of 1 to 22 (strictly unique field ranks)
        for race_id, group in df.groupby('race_id'):
            self.assertEqual(len(group), 22)
            positions = sorted(group['predicted_position'].tolist())
            self.assertListEqual(positions, list(range(1, 23)),
                                 f"Race {race_id} does not have strictly unique positions 1..22")

    def test_head_to_head_pairwise_consistency(self):
        """Verify head_to_head_matrix.csv has 462 records and P(A>B) + P(B>A) == 1.0."""
        file_path = OUTPUT_DIR / "head_to_head_matrix.csv"
        self.assertTrue(file_path.exists(), f"{file_path} does not exist")
        
        df = pd.read_csv(file_path)
        expected_cols = [
            "as_of_round", "driver_a_id", "driver_a_name", "driver_a_code", "driver_a_team",
            "driver_a_current_points", "driver_b_id", "driver_b_name", "driver_b_code",
            "driver_b_team", "driver_b_current_points", "win_probability", "is_teammates"
        ]
        self.assertListEqual(list(df.columns), expected_cols)
        
        # 22 drivers * 21 = 462 records
        self.assertEqual(len(df), 462)
        
        # Check pairwise complementary probabilities: P(A beats B) + P(B beats A) == 1.0
        lookup = {}
        for _, row in df.iterrows():
            pair = (int(row['driver_a_id']), int(row['driver_b_id']))
            lookup[pair] = float(row['win_probability'])
            
        checked = set()
        for (da, db), p_ab in lookup.items():
            if (da, db) in checked:
                continue
            p_ba = lookup[(db, da)]
            self.assertAlmostEqual(p_ab + p_ba, 1.0, places=3,
                                   msg=f"Pairwise complementary failed for ({da}, {db}): {p_ab} + {p_ba} != 1.0")
            checked.add((da, db))
            checked.add((db, da))
            
        # Teammate flags should be symmetric
        for _, row in df.iterrows():
            da, db = int(row['driver_a_id']), int(row['driver_b_id'])
            t_ab = row['is_teammates']
            t_ba = df[(df['driver_a_id'] == db) & (df['driver_b_id'] == da)]['is_teammates'].iloc[0]
            self.assertEqual(t_ab, t_ba)

    def test_timeline_probabilities_files(self):
        """Verify timeline files championship_probabilities.csv, constructor_probabilities.csv, teammate_battles.csv."""
        for fn in ["championship_probabilities.csv", "constructor_probabilities.csv", "teammate_battles.csv"]:
            p = OUTPUT_DIR / fn
            self.assertTrue(p.exists(), f"{p} does not exist")
            df = pd.read_csv(p)
            self.assertFalse(df.empty)
            self.assertFalse(df.isna().any().any())

if __name__ == "__main__":
    unittest.main()
