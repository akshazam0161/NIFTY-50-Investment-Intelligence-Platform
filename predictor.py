import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score

def train_directional_model(df: pd.DataFrame, target_symbol: str, save_dir: str = None):
    """
    Trains a time-series respected Random Forest model to predict 
    if a specific stock will go up or down the next day.
    
    Uses 5-fold TimeSeriesSplit cross-validation on the training window for
    a robust accuracy estimate, then evaluates on a held-out 20% test window.
    
    Args -> 
        df:            Full processed dataframe from data_pipeline.run_pipeline()
        target_symbol: Stock ticker to train on (e.g. "RELIANCE")
        save_dir:      Optional directory path to persist the model with joblib.
                       If None, model is not saved to disk.
    
    Returns:
        model, cv_accuracy, precision, importance_df
    """
    print(f"\n Initializing AI Predictor for: {target_symbol}")
    
    # Isolating the specific stock
    stock_df = df[df['Symbol'] == target_symbol].copy()
    
    # Defines the engineered features
    features = [
        'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 
        'BB_Width', 'RSI', 'MACD_Line', 'MACD_Hist', 
        'Volatility_20', 'ROC_10', 'Volume_Ratio'
    ]
    
    # NaN guard-> drops any row that still have NaN in features/target
    stock_df = stock_df.dropna(subset=features + ['Target_Dir']).reset_index(drop=True)
    stock_df = stock_df.sort_values("Date")
    
    X = stock_df[features]
    y = stock_df['Target_Dir'].astype(int)
    
    # Hard time-series split: 80% train, 20% holdout test (not allow model to look into the future, so no train_test_split)
    split_idx = int(len(stock_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Initialize the Random Forest
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,        # Kept shallow to prevent overfitting noise
        random_state=42,
        class_weight='balanced'
    )
    
    # 5-fold TimeSeriesSplit cross-validation on the training window
    # Each fold respects chronological order -> no future leakage.
    # This gives a mean ± std accuracy that is far more credible than a single split.
    print(" Running 5-fold TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"    -> CV Accuracy: {cv_accuracy:.2%} ± {cv_std:.2%} across 5 folds")
    
    # Final fit on the full training window
    print("[⚙️] Fitting final model on full training window...")
    model.fit(X_train, y_train)
    
    # Holdout test evaluation (the 20% the model has never seen)
    preds = model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, preds)
    # zero_division=0 prevents crash when model predicts a constant class on thin data
    precision = precision_score(y_test, preds, zero_division=0)
    
    print(f" Model Training Complete!")
    print(f"    -> Holdout Test Accuracy : {holdout_accuracy:.2%}")
    print(f"    -> CV Accuracy (reported): {cv_accuracy:.2%} ± {cv_std:.2%}")
    print(f"    -> Precision (UP calls)  : {precision:.2%}")
    
    # persist model to disk so app.py can reload without retraining
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{target_symbol}_rf_model.pkl")
        joblib.dump(model, model_path)
        print(f" Model saved → {model_path}")
    
    # Extract Feature Importances for Explainable AI
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Return cv_accuracy as the primary accuracy metric:
    # it is averaged over 5 time-ordered folds, making it more robust than a single holdout number.
    return model, cv_accuracy, precision, importance_df

if __name__ == "__main__":
    #  Dynamically imports the data pipeline function
    from data_pipeline import run_pipeline
    
    # xecute pipeline and saves it to the EXACT variable name needed
    processed_df = run_pipeline("data/NIFTY50_all.csv")
    
    # Passes the defined dataframe into your model training engine
    model, acc, prec, importance = train_directional_model(processed_df, "RELIANCE")
    
    # Prints the Explainable AI (XAI) output to confirm it works
    print("\n--- 🧠 Top 3 Market Drivers (Explainable AI) ---")
    print(importance.head(3))