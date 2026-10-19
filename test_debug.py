from data.selector import get_available_races
from data.fetcher import get_session_laps
from data.cleaner import clean_laps
from data.features import engineer_features
from ml.model import train_model

laps, session = get_session_laps(2026, 'Australian Grand Prix')

if hasattr(laps, 'get_weather_data'):
    try:
        weather = laps.get_weather_data()
        laps['track_temp'] = weather['TrackTemp'].values
    except Exception as e:
        print("EXCEPTION IN WEATHER:", e)
        laps['track_temp'] = 30.0 # safe fallback
else:
    laps['track_temp'] = 30.0

print("Fetched laps shape:", laps.shape)
cleaned = clean_laps(laps)
print("Cleaned laps shape:", cleaned.shape)
featured = engineer_features(cleaned)
print("Featured laps shape:", featured.shape)

try:
    model, X_test, y_test = train_model(featured)
    print("Model:", model)
except Exception as e:
    import traceback
    traceback.print_exc()
