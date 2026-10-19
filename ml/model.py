import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(featured_df):
    """
    Trains a Random Forest Regressor to predict laptime_delta.
    Returns: model, X_test, y_test
    """
    try:
        # FastF1 lap weather fields is typically 'TrackTemp'
        track_temp_col = 'TrackTemp' if 'TrackTemp' in featured_df.columns else 'track_temp'
        
        features = [
            'tyre_age_verified',
            'compound_encoded',
            'fuel_corrected_laptime',
            'track_evolution_median',
            'track_temp'
        ]
        target = 'laptime_delta'
        
        # Check if columns exist
        missing = [c for c in features + [target] if c not in featured_df.columns]
        if missing:
            raise KeyError(f"Missing columns for model: {missing}")
            
        # Drop rows where any of the above are null
        df = featured_df.dropna(subset=features + [target])
        
        X = df[features]
        y = df[target]
        
        # Require enough data points to train
        if len(df) < 20:
            raise ValueError(f"Insufficient samples to train model (found {len(df)} after null drops).")
            
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate and print MAE and R² internally for logs
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"[ML Model] Trained successfully. MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return model, X_test, y_test
        
    except Exception as e:
        print(f"[ML Model] Error during training: {e}")
        return None, None, None

def get_feature_importance(model):
    """
    Returns pandas Series of feature importances sorted descending.
    """
    try:
        importances = model.feature_importances_
        feature_names = model.feature_names_in_
        serie = pd.Series(importances, index=feature_names)
        return serie.sort_values(ascending=False)
    except Exception as e:
        print(f"[ML Model] Could not get feature importances: {e}")
        return pd.Series()

def predict_future_laps(model, current_tyre_age, compound_encoded, current_lap, track_evolution_median_series, track_temp, n_laps=15):
    """
    Predict lap time deltas for the next n_laps laps.
    track_evolution_median_series should be indexed by LapNumber.
    """
    try:
        predicted_rows = []
        for i in range(n_laps):
            lap = current_lap + i
            tyre_age = current_tyre_age + i
            
            # Fuel mass estimate calculation
            fuel_mass = max(0, 110 - 1.7 * lap)
            fuel_effect = fuel_mass * 0.03
            estimated_laptime = 90.0 - fuel_effect
            
            # Lookup track evolution
            if hasattr(track_evolution_median_series, 'loc') and lap in track_evolution_median_series.index:
                track_evo = track_evolution_median_series.loc[lap]
            elif hasattr(track_evolution_median_series, 'empty') and not track_evolution_median_series.empty:
                # Use last known value if lap exceeds data
                track_evo = track_evolution_median_series.iloc[-1]
            else:
                track_evo = 90.0
                
            # The column named for track temp in training must be preserved here
            # I am checking what it was via feature_names_in_
            temp_col = 'TrackTemp'
            if hasattr(model, 'feature_names_in_'):
                if 'track_temp' in model.feature_names_in_:
                    temp_col = 'track_temp'
                    
            predicted_rows.append({
                'tyre_age_verified': tyre_age,
                'compound_encoded': compound_encoded,
                'fuel_corrected_laptime': estimated_laptime,
                'track_evolution_median': track_evo,
                temp_col: track_temp 
            })
            
        future_df = pd.DataFrame(predicted_rows)
        
        # Ensure ordered matching for model input
        if hasattr(model, 'feature_names_in_'):
            future_df = future_df[model.feature_names_in_]
            
        preds = model.predict(future_df)
        return preds.tolist()
    except Exception as e:
        print(f"[ML Model] Error predicting future laps: {e}")
        return []
