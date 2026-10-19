import os
import fastf1
import streamlit as st

@st.cache_data
def get_session_laps(year, race_name, session_type='R'):
    """
    Loads a FastF1 session, configures cache, and returns laps and session.
    """
    # Point cache to the f1_strategy_engineer/cache directory (up one level from data/)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    fastf1.Cache.enable_cache(cache_dir)
    
    session = fastf1.get_session(year, race_name, session_type)
    session.load(laps=True, telemetry=False, weather=True)
    
    laps = session.laps
    try:
        # Must extract weather here before Streamlit cache serialization casts Laps back to raw DataFrame
        weather = laps.get_weather_data()
        laps['track_temp'] = weather['TrackTemp'].values
    except Exception:
        laps['track_temp'] = 30.0
        
    return laps, session
