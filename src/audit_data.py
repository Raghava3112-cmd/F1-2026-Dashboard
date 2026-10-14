import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = r"c:\Users\yecha\OneDrive\Desktop\DA_26\tableau\F1\data\raw"
OUTPUT_DIR = r"c:\Users\yecha\OneDrive\Desktop\DA_26\tableau\F1\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "audit_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

out_lines = []

def log(msg=""):
    print(msg)
    out_lines.append(str(msg))

def main():
    log("=" * 80)
    log("F1 DATA AUDIT REPORT")
    log("=" * 80)
    log(f"Data directory: {DATA_DIR}\n")

    # List all CSVs
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    log(f"Found {len(csv_files)} CSV files in {DATA_DIR}:")
    for f in csv_files:
        log(f"  - {os.path.basename(f)}")
    log()

    # Cache dataframes
    dfs = {}
    for f in csv_files:
        name = os.path.splitext(os.path.basename(f))[0]
        dfs[name] = pd.read_csv(f, dtype=str, keep_default_na=False)

    # 1. For each CSV in data/raw/
    log("=" * 80)
    log("TASK 1: PROFILE EACH CSV IN data/raw/")
    log("=" * 80)
    for name, df in dfs.items():
        log(f"--- File: {name}.csv ---")
        log(f"Total Rows: {len(df)}")
        log(f"Total Columns: {len(df.columns)}")
        log(f"Column Names: {list(df.columns)}")
        
        # Check duplicates
        dup_rows = df.duplicated().sum()
        log(f"Duplicate Rows Count: {dup_rows}")

        # Column stats
        log("Columns Breakdown:")
        for col in df.columns:
            n_slash_n = (df[col] == r"\N").sum()
            n_empty = (df[col] == "").sum()
            log(f"  - {col}: count='{len(df[col])}', '\\N' count={n_slash_n}, empty count={n_empty}")
        log()

    # 2. races.csv
    log("=" * 80)
    log("TASK 2: races.csv AUDIT")
    log("=" * 80)
    races = dfs.get("races")
    results = dfs.get("results")
    
    races['year_int'] = races['year'].astype(int)
    races['raceId_int'] = races['raceId'].astype(int)
    races['round_int'] = races['round'].astype(int)
    
    log(f"Total races: {len(races)}")
    log(f"Year range: {races['year_int'].min()} to {races['year_int'].max()}")
    for yr in [2024, 2025, 2026]:
        cnt = (races['year_int'] == yr).sum()
        log(f"Number of races in {yr}: {cnt}")
    
    races_2026 = races[races['year_int'] == 2026].sort_values('round_int')
    log(f"\n2026 Races ({len(races_2026)} total):")
    result_race_ids = set(results['raceId'].unique()) if results is not None else set()
    
    for _, r in races_2026.iterrows():
        has_results = r['raceId'] in result_race_ids
        log(f"  raceId={r['raceId']}, round={r['round']}, name='{r['name']}', date='{r['date']}', circuitId={r['circuitId']}, has_results={has_results}")
    
    races_2026_with_res = races_2026[races_2026['raceId'].isin(result_race_ids)]
    log(f"\n2026 races with results: {len(races_2026_with_res)}")
    for _, r in races_2026_with_res.iterrows():
        log(f"  Completed/Result Race: raceId={r['raceId']}, round={r['round']}, name='{r['name']}', date={r['date']}")
    log()

    # 3. results.csv
    log("=" * 80)
    log("TASK 3: results.csv AUDIT (Grain & Duplicates)")
    log("=" * 80)
    total_res = len(results)
    distinct_race_driver = len(results.drop_duplicates(subset=['raceId', 'driverId']))
    log(f"Total rows in results: {total_res}")
    log(f"Distinct (raceId, driverId): {distinct_race_driver}")
    dup_grain = results.duplicated(subset=['raceId', 'driverId']).sum()
    log(f"Duplicate (raceId, driverId) count: {dup_grain} (Grain is 1 row per driver per race: {dup_grain == 0})")
    
    # 2026 results details
    races_2026_ids = set(races_2026['raceId'])
    res_2026 = results[results['raceId'].isin(races_2026_ids)]
    log(f"\nTotal 2026 results rows: {len(res_2026)}")
    
    for rid in races_2026_with_res['raceId']:
        r_name = races[races['raceId'] == rid]['name'].values[0]
        r_round = races[races['raceId'] == rid]['round'].values[0]
        cnt_drivers = len(res_2026[res_2026['raceId'] == rid]['driverId'].unique())
        log(f"  raceId={rid} (Round {r_round}: {r_name}): distinct drivers = {cnt_drivers}")
    
    status_ids_2026 = sorted(res_2026['statusId'].unique(), key=lambda x: int(x))
    log(f"\nUnique statusId values appearing in 2026 results: {status_ids_2026}")
    log()

    # 4. status.csv
    log("=" * 80)
    log("TASK 4: status.csv AUDIT & CATEGORIZATION")
    log("=" * 80)
    status_df = dfs.get("status")
    
    accident_keywords = ['accident', 'collision', 'spun off', 'damage', 'crash']
    mech_keywords = [
        'engine', 'gearbox', 'transmission', 'clutch', 'hydraulics', 'electrical', 
        'suspension', 'brakes', 'differential', 'overheating', 'mechanical', 'tyre', 
        'puncture', 'oil pressure', 'water pressure', 'fuel pressure', 'wheel', 
        'throttle', 'steering', 'technical', 'exhaust', 'radiator', 'turbo', 
        'power unit', 'ers', 'battery', 'vibrations', 'oil leak', 'water leak', 
        'fuel pump', 'fire', 'alternator', 'power loss', 'brake duct', 'driveshaft',
        'ignition', 'chassis', 'spark plugs', 'fuel pipe', 'oil pipe', 'cv joint',
        'pneumatics', 'handling', 'rear wing', 'front wing', 'undertray'
    ]
    
    categorized = []
    for _, row in status_df.iterrows():
        sid = int(row['statusId'])
        st_text = row['status']
        st_lower = st_text.lower()
        
        if sid == 1 or 'lap' in st_lower:
            cat = "FINISHED"
        elif any(k in st_lower for k in accident_keywords):
            cat = "DNF_ACCIDENT"
        elif any(k in st_lower for k in mech_keywords):
            cat = "DNF_MECHANICAL"
        else:
            cat = "DNF_OTHER"
        categorized.append((sid, st_text, cat))
    
    categorized.sort(key=lambda x: x[0])
    log(f"All {len(categorized)} statusId Categorizations:")
    for sid, st_text, cat in categorized:
        log(f"  statusId={sid:3d} | status='{st_text}' | category={cat}")
    
    cat_counts = pd.Series([c[2] for c in categorized]).value_counts()
    log(f"\nStatus Category breakdown:")
    for cat_name, cnt in cat_counts.items():
        log(f"  {cat_name}: {cnt}")
    log()

    # 5. qualifying.csv
    log("=" * 80)
    log("TASK 5: qualifying.csv AUDIT")
    log("=" * 80)
    qual = dfs.get("qualifying")
    if qual is not None:
        qual_merged = qual.merge(races[['raceId', 'year', 'round']], on='raceId', how='left')
        seasons_qual = sorted(qual_merged['year'].dropna().unique(), key=lambda x: int(x))
        log(f"Qualifying data seasons ({len(seasons_qual)} seasons): {seasons_qual}")
        qual_2026 = qual_merged[qual_merged['year'] == '2026']
        races_2026_qual = qual_2026['raceId'].unique()
        log(f"Number of 2026 races with qualifying records: {len(races_2026_qual)} (raceIds: {list(races_2026_qual)})")
        log(f"Total rows of 2026 qualifying: {len(qual_2026)}")
        for col in ['q1', 'q2', 'q3']:
            slash_n_total = (qual[col] == r"\N").sum()
            slash_n_2026 = (qual_2026[col] == r"\N").sum() if len(qual_2026) > 0 else 0
            log(f"  {col}: total '\\N'={slash_n_total}, 2026 '\\N'={slash_n_2026}")
    else:
        log("qualifying.csv not found!")
    log()

    # 6. lap_times.csv
    log("=" * 80)
    log("TASK 6: lap_times.csv AUDIT")
    log("=" * 80)
    laps = dfs.get("lap_times")
    if laps is not None:
        laps_merged = laps.merge(races[['raceId', 'year']], on='raceId', how='left')
        seasons_laps = sorted(laps_merged['year'].dropna().unique(), key=lambda x: int(x))
        log(f"Lap times data seasons ({len(seasons_laps)} seasons): {seasons_laps}")
        laps_2026 = laps_merged[laps_merged['year'] == '2026']
        log(f"Any 2026 lap time records? Count = {len(laps_2026)}")
    else:
        log("lap_times.csv not found!")
    log()

    # 7. pit_stops.csv
    log("=" * 80)
    log("TASK 7: pit_stops.csv AUDIT")
    log("=" * 80)
    pits = dfs.get("pit_stops")
    if pits is not None:
        pits_merged = pits.merge(races[['raceId', 'year']], on='raceId', how='left')
        seasons_pits = sorted(pits_merged['year'].dropna().unique(), key=lambda x: int(x))
        log(f"Pit stops data seasons ({len(seasons_pits)} seasons): {seasons_pits}")
        pits_2026 = pits_merged[pits_merged['year'] == '2026']
        log(f"Any 2026 pit stop records? Count = {len(pits_2026)}")
    else:
        log("pit_stops.csv not found!")
    log()

    # 8. sprint_results.csv
    log("=" * 80)
    log("TASK 8: sprint_results.csv AUDIT")
    log("=" * 80)
    sprint = dfs.get("sprint_results")
    if sprint is not None:
        sprint_merged = sprint.merge(races[['raceId', 'year']], on='raceId', how='left')
        seasons_sprint = sorted(sprint_merged['year'].dropna().unique(), key=lambda x: int(x))
        log(f"Sprint results data seasons ({len(seasons_sprint)} seasons): {seasons_sprint}")
        sprint_2026 = sprint_merged[sprint_merged['year'] == '2026']
        log(f"Any 2026 sprint records? Count = {len(sprint_2026)}")
        if len(sprint_2026) > 0:
            log(f"2026 sprint raceIds: {list(sprint_2026['raceId'].unique())}")
    else:
        log("sprint_results.csv not found!")
    log()

    # Determine last completed 2026 round
    drivers = dfs.get("drivers")
    constructors = dfs.get("constructors")
    
    if len(races_2026_with_res) > 0:
        last_race_row = races_2026_with_res.sort_values('round_int').iloc[-1]
        last_race_id = last_race_row['raceId']
        last_round = last_race_row['round']
        last_name = last_race_row['name']
    else:
        last_race_id = None
        last_round = None
        last_name = None
    log(f"Last completed 2026 race: raceId={last_race_id}, round={last_round}, name='{last_name}'\n")

    # 9. Active 2026 driver roster
    log("=" * 80)
    log("TASK 9: ACTIVE 2026 DRIVER ROSTER (Last Completed 2026 Round)")
    log("=" * 80)
    if last_race_id is not None:
        last_race_res = res_2026[res_2026['raceId'] == last_race_id]
        c_sub = constructors[['constructorId', 'name']].rename(columns={'name': 'constructor_name'})
        roster = last_race_res.merge(drivers, on='driverId', how='left').merge(c_sub, on='constructorId', how='left')
        roster['position_order_int'] = roster['positionOrder'].astype(int)
        roster = roster.sort_values('position_order_int')
        log(f"Drivers in Round {last_round} ({last_name}) [Total: {len(roster)}]:")
        for _, row in roster.iterrows():
            log(f"  driverId={row['driverId']:3s} | code={row['code']:3s} | surname={row['surname']:15s} | constructorId={row['constructorId']:3s} ({row['constructor_name']}) | posOrder={row['positionOrder']}")
    log()

    # 10. 2026 scoring verification
    log("=" * 80)
    log("TASK 10: 2026 SCORING VERIFICATION (Last Completed 2026 Round)")
    log("=" * 80)
    if last_race_id is not None:
        standard_points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        log(f"Checking points for raceId={last_race_id} ({last_name}):")
        
        for _, row in roster.iterrows():
            pos_ord = int(row['positionOrder'])
            pts = float(row['points'])
            expected_base = standard_points.get(pos_ord, 0)
            fl_rank = row.get('rank', r'\N')
            log(f"  Pos={pos_ord:2d} | Driver={row['code']:3s} ({row['surname']:12s}) | Points={pts:4.1f} | Expected Base={expected_base:2d} | rank={fl_rank} | fastestLap={row.get('fastestLap', r'\N')}")
    log()

    # 11. driver_standings.csv grain check
    log("=" * 80)
    log("TASK 11: driver_standings.csv GRAIN & CUMULATIVE SNAPSHOT VERIFICATION")
    log("=" * 80)
    ds = dfs.get("driver_standings")
    if ds is not None and last_race_id is not None:
        ds_last = ds[ds['raceId'] == last_race_id].merge(drivers, on='driverId', how='left')
        ds_last['points_float'] = ds_last['points'].astype(float)
        ds_last['position_int'] = ds_last['position'].astype(int)
        ds_last = ds_last.sort_values('position_int')
        
        completed_2026_races = races_2026_with_res[races_2026_with_res['round_int'] <= int(last_round)]
        completed_race_ids = set(completed_2026_races['raceId'])
        log(f"Completed 2026 races up to round {last_round}: {list(completed_2026_races['name'])} (raceIds: {list(completed_race_ids)})")
        
        cum_res = results[results['raceId'].isin(completed_race_ids)].copy()
        cum_res['points_float'] = cum_res['points'].astype(float)
        cum_pts = cum_res.groupby('driverId')['points_float'].sum().reset_index()
        cum_pts.rename(columns={'points_float': 'cum_race_points'}, inplace=True)
        
        if sprint is not None:
            cum_sprint = sprint[sprint['raceId'].isin(completed_race_ids)].copy()
            if len(cum_sprint) > 0:
                cum_sprint['points_float'] = cum_sprint['points'].astype(float)
                sprint_pts = cum_sprint.groupby('driverId')['points_float'].sum().reset_index()
                sprint_pts.rename(columns={'points_float': 'cum_sprint_points'}, inplace=True)
                cum_pts = cum_pts.merge(sprint_pts, on='driverId', how='left').fillna(0)
                cum_pts['total_calc_points'] = cum_pts['cum_race_points'] + cum_pts['cum_sprint_points']
            else:
                cum_pts['total_calc_points'] = cum_pts['cum_race_points']
        else:
            cum_pts['total_calc_points'] = cum_pts['cum_race_points']
            
        merged_standings = ds_last.merge(cum_pts, on='driverId', how='left')
        merged_standings['total_calc_points'] = merged_standings['total_calc_points'].fillna(0.0)
        
        log(f"Driver Standings for Round {last_round} (raceId={last_race_id}):")
        for _, row in merged_standings.iterrows():
            match = abs(row['points_float'] - row['total_calc_points']) < 1e-4
            log(f"  Pos={row['position']:2s} | Driver={row['code']:3s} ({row['surname']:12s}) | Standings Pts={row['points_float']:5.1f} | Cumulative Calc Pts={row['total_calc_points']:5.1f} | Match={match}")
    log()

    # 12. constructor assignments in 2026
    log("=" * 80)
    log("TASK 12: CONSTRUCTOR ASSIGNMENTS IN 2026 & MID-SEASON SWITCHES")
    log("=" * 80)
    if len(res_2026) > 0:
        res_2026_m = res_2026.merge(drivers[['driverId', 'code', 'surname']], on='driverId', how='left') \
                             .merge(constructors[['constructorId', 'name']], on='constructorId', how='left')
        
        driver_constructors = res_2026_m.groupby(['driverId', 'code', 'surname'])['name'].unique().reset_index()
        log("2026 Driver-Constructor pairings:")
        switches = []
        for _, row in driver_constructors.iterrows():
            const_list = list(row['name'])
            log(f"  Driver: {row['code']:3s} ({row['surname']:12s}, driverId={row['driverId']}) -> Constructors: {const_list}")
            if len(const_list) > 1:
                switches.append((row['code'], row['surname'], const_list))
        
        if switches:
            log(f"\nMID-SEASON CONSTRUCTOR SWITCHES DETECTED ({len(switches)}):")
            for code, surname, clist in switches:
                log(f"  WARNING: {code} ({surname}) drove for: {clist}")
        else:
            log("\nNo drivers switched constructors mid-season in 2026 results.")
    log()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\nAudit completed. Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
