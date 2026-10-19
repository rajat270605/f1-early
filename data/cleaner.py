import pandas as pd

def clean_laps(laps_df):
    """
    Cleans FastF1 lap data by removing non-representative racing laps.
    """
    df = laps_df.copy()
    
    # Add StintNumber column (increments each time a driver pits, identified by PitOutTime)
    df = df.sort_values(['Driver', 'LapNumber'])
    # PitOutTime is not null when leaving the pit lane, which marks the start of a stint.
    # We cumsum this boolean per driver so it increments by 1 for each new stint.
    df['is_pit_out'] = df['PitOutTime'].notnull()
    df['StintNumber'] = df.groupby('Driver')['is_pit_out'].cumsum() + 1
    df = df.drop(columns=['is_pit_out'])

    initial_count = len(df)
    counts = {"initial": initial_count}
    
    # 1. Remove In-laps (lap started from pit -> out-lap effectively based on prompt description, but we check PitOutTime)
    rule1_mask = df['PitOutTime'].notnull()
    count_rule1 = rule1_mask.sum()
    df = df[~rule1_mask]
    counts["started_from_pit_removed"] = count_rule1
    
    # 2. Remove Out-laps (lap ended in pit -> in-lap based on prompt description, check PitInTime)
    rule2_mask = df['PitInTime'].notnull()
    count_rule2 = rule2_mask.sum()
    df = df[~rule2_mask]
    counts["ended_in_pit_removed"] = count_rule2
    
    # 3. Remove Safety car laps ('4' in TrackStatus)
    rule3_mask = df['TrackStatus'].astype(str).str.contains('4', na=False)
    count_rule3 = rule3_mask.sum()
    df = df[~rule3_mask]
    counts["safety_car_removed"] = count_rule3
    
    # 4. Remove VSC laps ('6' in TrackStatus)
    rule4_mask = df['TrackStatus'].astype(str).str.contains('6', na=False)
    count_rule4 = rule4_mask.sum()
    df = df[~rule4_mask]
    counts["vsc_removed"] = count_rule4
    
    # 5. Remove Red flag laps ('5' in TrackStatus)
    rule5_mask = df['TrackStatus'].astype(str).str.contains('5', na=False)
    count_rule5 = rule5_mask.sum()
    df = df[~rule5_mask]
    counts["red_flag_removed"] = count_rule5
    
    # 6. Any lap where TrackStatus is not '1' (green flag only)
    rule6_mask = df['TrackStatus'] != '1'
    count_rule6 = rule6_mask.sum()
    df = df[~rule6_mask]
    counts["non_green_flag_removed"] = count_rule6
    
    # 7. Laps where LapTime is null
    rule7_mask = df['LapTime'].isnull()
    count_rule7 = rule7_mask.sum()
    df = df[~rule7_mask]
    counts["null_laptime_removed"] = count_rule7
    
    # Print debug info
    print("\n[Cleaner] Lap Filtering Debug Info:")
    for k, v in counts.items():
        print(f" - {k}: {v}")
    print(f" - Final Count: {len(df)}\n")

    # IMPORTANT: Recalculate TyreLife per driver per stint.
    # FastF1 has a built-in 'Stint' column which identifies physical tyre sets (handles red flags properly).
    # "After a red flag restart, tyre age continues from where it left off on that set."
    # FastF1's built-in Stint column aligns with actual tyre sets. If a red flag happens
    # and they don't change tyres, FastF1 keeps Stint the same.
    # So we group by Driver and FastF1's Stint (if available, else StintNumber) to recalculate TyreLife
    
    group_col = 'Stint' if 'Stint' in df.columns else 'StintNumber'
    # Sort to ensure chronological order before cumcount
    df = df.sort_values(by=['Driver', 'LapNumber'])
    # Calculate a simple +1 tyre life for each lap within the set
    df['TyreLife'] = df.groupby(['Driver', group_col]).cumcount() + 1
    
    return df
