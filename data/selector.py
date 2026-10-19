import fastf1
import pandas as pd

def get_available_races(year):
    """
    Uses fastf1 to get all valid rounds for a season.
    Returns list of EventNames.
    Does NOT exclude red-flagged races.
    """
    import logging
    # Suppress fastf1 logging for the schedule request
    logging.getLogger('fastf1.core').setLevel(logging.WARNING)
    
    schedule = fastf1.get_event_schedule(year)
    
    # Valid rounds only (RoundNumber > 0)
    schedule = schedule[schedule['RoundNumber'] > 0]
    
    # Exclude pre-season testing events
    schedule = schedule[schedule['EventFormat'] != 'testing']
    schedule = schedule[~schedule['EventName'].str.contains('Test', case=False, na=False)]
    
    return schedule['EventName'].tolist()

def get_pit_loss_delta(session):
    """
    Extracts pit stop timing data from FastF1 session.
    Calculates the median time lost in the pit lane across all pit stops in the session.
    Returns the pit loss delta in seconds as a float.
    Handles gracefully if pit stop data is unavailable — return a default of 22.0 seconds.
    """
    try:
        if not hasattr(session, 'pit_stops') or session.pit_stops is None or session.pit_stops.empty:
            return 22.0
            
        stops = session.pit_stops
        if 'Duration' in stops.columns:
            durations = stops['Duration']
            if pd.api.types.is_timedelta64_dtype(durations):
                median_delta = durations.dt.total_seconds().median()
            else:
                # Often loaded as numeric or string representation of float
                median_delta = pd.to_numeric(durations, errors='coerce').median()
            
            if pd.notna(median_delta) and median_delta > 0:
                return float(median_delta)
    except Exception:
        pass
        
    return 22.0
