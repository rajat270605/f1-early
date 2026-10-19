import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_degradation_model(laps_df):
    """
    Trains a Random Forest Regressor to predict lap_time_delta.
    The baseline is the minimum fuel_corrected_time in lap 1 of each driver's first stint.
    lap_time_delta is the deviation from that fresh-tyre baseline.
    """
    df = laps_df.copy()
    
    features = ['tyre_age', 'compound_encoded', 'fuel_corrected_time', 'track_evolution', 'track_temp']
    
    # "The baseline is the minimum fuel_corrected_time in lap 1 of each driver's first stint"
    # Find the very first lap of the first stint per driver
    first_stint_df = df[df['StintNumber'] == 1].sort_values(['Driver', 'LapNumber'])
    baselines = first_stint_df.groupby('Driver')['fuel_corrected_time'].first()
    
    df['baseline'] = df['Driver'].map(baselines)
    
    # lap_time_delta target variable — degradation differential vs a fresh tyre baseline
    df['lap_time_delta'] = df['fuel_corrected_time'] - df['baseline']
    
    # Drops any rows where any of the five features or the target are NaN
    df.dropna(subset=features + ['lap_time_delta'], inplace=True)
    
    if len(df) < 50:
        raise ValueError(
            f"Insufficient clean laps for training: {len(df)} rows after "
            f"filtering. Session data may be unavailable or incomplete."
        )
    
    X = df[features]
    y = df['lap_time_delta']
    
    # 80/20 train/test split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomForestRegressor with n_estimators=100, max_depth=10, random_state=42
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Prints MAE and R² on the test set to the console for debugging
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance dict mapping sorted descending
    importances = list(model.feature_importances_)
    feat_imp_dict = dict(zip(features, importances))
    feat_imp_dict = dict(sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True))
    
    return model, feat_imp_dict, mae

def predict_lap_time(model, tyre_age, compound_encoded, fuel_corrected_time, track_evolution, track_temp):
    """
    Takes a trained model and individual feature values.
    Returns a single predicted lap_time_delta as a float.
    Called statically one lap at a time inside the simulation loop.
    """
    feat_df = pd.DataFrame([{
        'tyre_age': tyre_age,
        'compound_encoded': compound_encoded,
        'fuel_corrected_time': fuel_corrected_time,
        'track_evolution': track_evolution,
        'track_temp': track_temp
    }])
    
    pred = model.predict(feat_df)[0]
    return float(pred)
