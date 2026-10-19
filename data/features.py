import pandas as pd
import numpy as np

def engineer_features(laps_df):
    """
    Engineers new physics-based and dataset-relative features for the F1 ML model.
    Takes a clean laps DataFrame and returns a new populated DataFrame.
    """
    # Do not modify the original DataFrame — work on a copy
    df = laps_df.copy()
    
    try:
        # Check prerequisite
        if 'StintNumber' not in df.columns:
            raise ValueError("Missing 'StintNumber' column! Run clean_laps() first.")
            
        initial_cols = len(df.columns)
        
        # 1. Convert LapTime to seconds
        if pd.api.types.is_timedelta64_dtype(df['LapTime']):
            df['laptime_seconds'] = df['LapTime'].dt.total_seconds()
        else:
            # Fallback if somehow already numeric
            df['laptime_seconds'] = df['LapTime']
            
        # FEATURE 1: compound_encoded
        # Mapping tyre compounds to discrete integers (ordinal roughly aligning with hardness)
        compound_map = {
            'SOFT': 1,
            'MEDIUM': 2,
            'HARD': 3,
            'INTERMEDIATE': 4,
            'WET': 5
        }
        # Convert to upper case safely and map, filling missing with -1
        if 'Compound' in df.columns:
            df['compound_encoded'] = df['Compound'].astype(str).str.upper().map(compound_map).fillna(-1).astype(int)
        else:
            df['compound_encoded'] = -1
            
        # FEATURE 3: track_evolution_median
        # Rubber buildup makes tracks faster as the race progresses.
        # We capture this by finding the median lap time per lap across all drivers.
        # Calculated BEFORE fuel correction to capture track conditions raw.
        track_evo = df.groupby('LapNumber')['laptime_seconds'].median().reset_index()
        track_evo.rename(columns={'laptime_seconds': 'track_evolution'}, inplace=True)
        df = df.merge(track_evo, on='LapNumber', how='left')
        
        # FEATURE 2: fuel_corrected_laptime
        # Mass physics: Heavier fuel load = slower lap time.
        # Approx: Cars start with 110kg, burn 1.7kg per lap, costing ~0.03 seconds per kg.
        df['fuel_mass'] = (110 - (1.7 * df['LapNumber'])).clip(lower=0)
        df['fuel_effect'] = df['fuel_mass'] * 0.03
        df['fuel_corrected_time'] = df['laptime_seconds'] - df['fuel_effect']
        
        # FEATURE 4: tyre_age_verified
        # Recalculate tyre age sequentially per stint, disregarding dirty API TyreLife
        # which can glitch during red flags.
        # It's an important index of degradation.
        # Sort to ensure chronological order before cumulative counting
        df = df.sort_values(by=['Driver', 'LapNumber'])
        df['tyre_age'] = df.groupby(['Driver', 'StintNumber']).cumcount() + 1
        
        # FEATURE 5: laptime_delta (ML Target Variable)
        # Represents the tyre degradation: how much slower is this lap vs the first lap of the stint?
        # A positive delta means slower than the starting lap.
        # Calculate using the fuel corrected lap times so fuel burn doesn't mask degradation.
        first_laps = df.groupby(['Driver', 'StintNumber'])['fuel_corrected_time'].transform('first')
        df['laptime_delta'] = df['fuel_corrected_time'] - first_laps
        
        # --- POST PROCESSING ---
        # Drop rows where laptime_delta is null
        df = df.dropna(subset=['laptime_delta'])
        
        # Drop rows where laptime_seconds < 60 (impossible lap time, data error)
        df = df[df['laptime_seconds'] >= 60]
        
        # Drop rows where laptime_seconds > 200 (almost certainly unfiltered slow lap/error)
        df = df[df['laptime_seconds'] <= 200]
        
        # Calculate summary
        cols_added = len(df.columns) - initial_cols
        mean_delta = df['laptime_delta'].mean()
        std_delta = df['laptime_delta'].std()
        
        print("\n[Feature Engineer] Pipeline Summary:")
        print(f" - Total rows resulting: {len(df)}")
        print(f" - Number of columns added: {cols_added}")
        print(f" - laptime_delta mean: {mean_delta:.4f}s")
        print(f" - laptime_delta standard dev: {std_delta:.4f}s\n")
        
    except Exception as e:
        print(f"\n[Feature Engineer] ERROR occurred during feature engineering: {e}")
        # Return what we have to prevent silent crashing, though it will lack some features
        return df
        
    return df
