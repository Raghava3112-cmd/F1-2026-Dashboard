import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")

def test_outputs():
    print("==================================================")
    print("RUNNING VERIFICATION SUITE ON GENERATED CSV FILES")
    print("==================================================")
    
    # 1. Output 3: head_to_head_matrix.csv
    h2h = pd.read_csv(OUTPUT_DIR / "head_to_head_matrix.csv")
    print(f"\n[Test 1] head_to_head_matrix.csv:")
    print(f"  Rows: {len(h2h)} (Expected: 462 = 22 * 21)")
    assert len(h2h) == 462, f"Expected 462 rows, got {len(h2h)}"
    expected_cols = [
        'as_of_round', 'driver_a_id', 'driver_a_name', 'driver_a_code', 'driver_a_team',
        'driver_a_current_points', 'driver_b_id', 'driver_b_name', 'driver_b_code',
        'driver_b_team', 'driver_b_current_points', 'win_probability', 'is_teammates'
    ]
    assert list(h2h.columns) == expected_cols, "Column mismatch in head_to_head_matrix.csv"
    assert (h2h['win_probability'] >= 0.0).all() and (h2h['win_probability'] <= 1.0).all()
    assert h2h.isna().sum().sum() == 0, "Found unexpected nulls in head_to_head_matrix"
    
    # Check pairwise complementary symmetry
    pairs_tested = 0
    for _, row in h2h.iterrows():
        da = row['driver_a_id']
        db = row['driver_b_id']
        p_ab = row['win_probability']
        rev = h2h[(h2h['driver_a_id'] == db) & (h2h['driver_b_id'] == da)]
        if not rev.empty:
            p_ba = rev.iloc[0]['win_probability']
            assert abs(p_ab + p_ba - 1.0) < 1e-3, f"Symmetry broken for {da} vs {db}: {p_ab} + {p_ba} != 1"
            pairs_tested += 1
    print(f"  Pairwise complementary symmetry verified for all {pairs_tested} pairs: PASS")

    # 2. Output 4: next_race_dnf_predictions.csv
    dnf = pd.read_csv(OUTPUT_DIR / "next_race_dnf_predictions.csv")
    print(f"\n[Test 2] next_race_dnf_predictions.csv:")
    print(f"  Rows: {len(dnf)} (Expected: 22 active drivers)")
    assert len(dnf) == 22, f"Expected 22 rows, got {len(dnf)}"
    expected_dnf_cols = [
        'race_id', 'round', 'race_name', 'driver_id', 'code', 'surname',
        'constructor_id', 'team_name', 'dnf_probability', 'finish_probability',
        'driver_season_dnf_rate', 'circuit_historical_dnf_rate'
    ]
    assert list(dnf.columns) == expected_dnf_cols, "Column mismatch in next_race_dnf_predictions.csv"
    assert (dnf['dnf_probability'] >= 0.0).all() and (dnf['dnf_probability'] <= 1.0).all()
    assert (dnf['finish_probability'] >= 0.0).all() and (dnf['finish_probability'] <= 1.0).all()
    diff = np.abs((dnf['dnf_probability'] + dnf['finish_probability']) - 1.0)
    assert (diff < 1e-4).all(), "finish_probability != 1 - dnf_probability"
    print("  Exhaustive probability partition (dnf + finish == 1.0): PASS")

    # 3. Output 5: race_predictions.csv
    rp = pd.read_csv(OUTPUT_DIR / "race_predictions.csv")
    print(f"\n[Test 3] race_predictions.csv:")
    print(f"  Rows: {len(rp)} (Expected: 154 = 22 drivers * 7 remaining races)")
    assert len(rp) == 154, f"Expected 154 rows, got {len(rp)}"
    expected_rp_cols = ['race_id', 'driver_id', 'predicted_points', 'as_of_round', 'predicted_position']
    assert list(rp.columns) == expected_rp_cols, "Column mismatch in race_predictions.csv"
    
    # Verify each race has strictly distinct positions 1..22
    for r_id, group in rp.groupby('race_id'):
        positions = sorted(group['predicted_position'].tolist())
        assert positions == list(range(1, 23)), f"Invalid finish ranks for race {r_id}: {positions}"
    print("  Valid field-wide finish positions (1 to 22 unique per race): PASS")
    print(f"  Predicted points range: [{rp['predicted_points'].min():.2f}, {rp['predicted_points'].max():.2f}]")

    # 4. Outputs 1, 2, 6: Timeline tables
    champ = pd.read_csv(OUTPUT_DIR / "championship_probabilities.csv")
    const = pd.read_csv(OUTPUT_DIR / "constructor_probabilities.csv")
    team = pd.read_csv(OUTPUT_DIR / "teammate_battles.csv")
    
    print(f"\n[Test 4] Synchronized Timeline Outputs:")
    print(f"  championship_probabilities.csv rows: {len(champ)} (Expected: 352 = 22 * 16)")
    print(f"  constructor_probabilities.csv rows : {len(const)} (Expected: 176 = 11 * 16)")
    print(f"  teammate_battles.csv rows          : {len(team)} (Expected: 352 = 22 * 16)")
    assert len(champ) == 352
    assert len(const) == 176
    assert len(team) == 352
    print("  All row counts match: PASS")

    print("\n==================================================")
    print("ALL TESTS PASSED! PIPELINE AND OUTPUTS ARE VALID.")
    print("==================================================")

if __name__ == "__main__":
    test_outputs()
