import pandas as pd
from model.trainer import predict_lap_time

def run_undercut_simulation(laps_df, model, driver_a, driver_b, pit_loss_delta, start_lap=1, pit_compound='HARD'):
    """
    Runs the undercut simulation from Driver A's perspective.
    Returns a list of dictionaries with lap-by-lap simulation results.
    """
    # df = laps_df.copy() is already fine, or we can just use df = laps_df.copy()
    df = laps_df.copy()
    
    driver_a_df = df[df['Driver'] == driver_a].sort_values('LapNumber')
    driver_b_df = df[df['Driver'] == driver_b].sort_values('LapNumber')
    
    # Calculate driver_baseline for converting deltas to absolute times
    first_stint_a = driver_a_df[driver_a_df['StintNumber'] == 1]
    if not first_stint_a.empty:
        driver_baseline = first_stint_a['fuel_corrected_time'].iloc[0]
    else:
        driver_baseline = driver_a_df['fuel_corrected_time'].min()
        
    COMPOUND_ENCODING = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    pit_compound_encoded = COMPOUND_ENCODING.get(pit_compound.upper(), 3)
    
    max_laps = df['LapNumber'].max()
    
    results = []
    
    # Begin lap iteration from start_lap
    driver_a_iter_df = driver_a_df[driver_a_df['LapNumber'] >= start_lap]
    for idx, row in driver_a_iter_df.iterrows():
        L = int(row['LapNumber'])
        
        # Calculate gap to driver B
        gap_to_driver_b = None
        b_laps_up_to_L = driver_b_df[driver_b_df['LapNumber'] <= L]
        a_laps_up_to_L = driver_a_df[driver_a_df['LapNumber'] <= L]
        
        b_laps_set = set(b_laps_up_to_L['LapNumber'])
        a_laps_set = set(a_laps_up_to_L['LapNumber'])
        
        # Gap calculation: cumulative lap time difference
        # Use only common laps to avoid massive jumps when one driver has a missing lap (e.g. pitted)
        common_laps = a_laps_set.intersection(b_laps_set)
        
        if L in b_laps_set and common_laps:
            a_time = a_laps_up_to_L[a_laps_up_to_L['LapNumber'].isin(common_laps)]['fuel_corrected_time'].sum()
            b_time = b_laps_up_to_L[b_laps_up_to_L['LapNumber'].isin(common_laps)]['fuel_corrected_time'].sum()
            gap_to_driver_b = float(a_time - b_time)
            
        # Simulation
        remaining_laps = int(max_laps - L)
        # The simulation must not crash if a driver has fewer than 15 laps remaining 
        # — truncate the window to however many laps are available
        window_size = min(15, remaining_laps)
        
        stay_out_total = 0.0
        pit_now_total = 0.0
        crossover_lap = None
        undercut_window_open = False
        
        if window_size > 0:
            current_compound = row['compound_encoded']
            base_tyre_age = row['tyre_age']
            base_fuel_corrected_time = row['fuel_corrected_time']
            base_track_evolution = row['track_evolution']
            base_track_temp = row['track_temp']
            
            cum_stay_out = 0.0
            cum_pit_now = 0.0
            stay_out_curve = [0.0]
            pit_now_curve = [0.0]
            stay_out_lap_times = []
            pit_now_lap_times = []
            
            for i in range(1, window_size + 1):
                future_lap = L + i
                
                # Fetch track_evolution and track_temp for future lap, fallback to base values
                future_data = driver_a_df[driver_a_df['LapNumber'] == future_lap]
                if not future_data.empty:
                    track_ev = future_data.iloc[0]['track_evolution']
                    track_tmp = future_data.iloc[0]['track_temp']
                    if pd.isna(track_ev): track_ev = base_track_evolution
                    if pd.isna(track_tmp): track_tmp = base_track_temp
                else:
                    track_ev = base_track_evolution
                    track_tmp = base_track_temp
                    
                # Stay Out curve
                so_tyre_age = base_tyre_age + i
                so_fuel_time = base_fuel_corrected_time - (0.051 * i)
                so_delta = predict_lap_time(model, so_tyre_age, current_compound, so_fuel_time, track_ev, track_tmp)
                cum_stay_out += so_delta
                stay_out_curve.append(cum_stay_out)
                stay_out_lap_times.append(so_delta + driver_baseline)
                
                # Pit Now curve
                pn_tyre_age = i
                pn_fuel_time = base_fuel_corrected_time - (0.051 * i)
                pn_delta = predict_lap_time(model, pn_tyre_age, pit_compound_encoded, pn_fuel_time, track_ev, track_tmp)
                
                pn_abs_time = pn_delta + driver_baseline
                if i == 1:
                    pn_delta += pit_loss_delta
                    pn_abs_time += pit_loss_delta
                    
                cum_pit_now += pn_delta
                pit_now_curve.append(cum_pit_now)
                pit_now_lap_times.append(pn_abs_time)
                
                if cum_pit_now < cum_stay_out and crossover_lap is None:
                    crossover_lap = i
                    
            stay_out_total = cum_stay_out
            pit_now_total = cum_pit_now
            
            # undercut_window_open
            # crossover_lap <= 3 is the threshold — it reflects roughly the number of laps 
            # a following driver needs to build enough of a gap to pit reactively and still emerge ahead.
            if crossover_lap is not None and crossover_lap <= 3:
                undercut_window_open = True
                
        else:
            stay_out_curve = []
            pit_now_curve = []
            stay_out_lap_times = []
            pit_now_lap_times = []
            
        results.append({
            'lap': L,
            'stay_out_total': stay_out_total,
            'pit_now_total': pit_now_total,
            'stay_out_curve': stay_out_curve,
            'pit_now_curve': pit_now_curve,
            'stay_out_lap_times': stay_out_lap_times,
            'pit_now_lap_times': pit_now_lap_times,
            'crossover_lap': crossover_lap,
            'undercut_window_open': undercut_window_open,
            'gap_to_driver_b': gap_to_driver_b
        })
        
    return results
