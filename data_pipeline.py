import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    financial data cleaning layer. Handles datetime conflicts, 
    purges structural null entries, isolates outlier spikes from stock splits, 
    and filters out low-liquidity/short-timeline data anomalies.
    """
    df = pd.read_csv(file_path)
    
    # Standardizes timeline index
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Symbol', 'Close'])
    
    # Sort to keep timeline integrity intact
    df = df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)
    
    # Handles Corporate Actions / Outlier Spikes
    # Calculates raw daily returns temporarily to detect impossible 1-day jumps
    df['Temp_Return'] = df.groupby('Symbol')['Close'].pct_change()
    
    # If a stock drops more than 70% or gains more than 200% in a SINGLE day, 
    # it is almost always an unadjusted stock split or a data logging bug.
    outlier_condition = (df['Temp_Return'] < -0.7) | (df['Temp_Return'] > 2.0)
    
    # Clean anomalies by dropping those extreme corporate action days
    df = df[~outlier_condition].drop(columns=['Temp_Return'])
    
    # Filter out low-liquidity / Illiquid Assets
    # Ensure we only evaluate trading days where actual business happened (Volume > 0)
    df = df[df['Volume'] > 100] 
    
    return df

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Simple and Exponential Moving Averages."""
    # Simple Moving Averages
    df['SMA_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
    df['SMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean())
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
    
    # Exponential Moving Averages
    df['EMA_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    return df
def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Bollinger Bands (20-day SMA +/- 2 Standard Deviations)."""
    # Use transform to guarantee 1-to-1 row alignment
    df['BB_Middle'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean())
    df['BB_Std'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).std())
    
    # Calculates Upper and Lower bands
    df['BB_High'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Low']  = df['BB_Middle'] - (2 * df['BB_Std'])
    
    # Engineers the Band Width feature
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / (df['BB_Middle'] + 1e-9)
    
    # Drops the temporary Std column to keep the dataframe clean
    df = df.drop(columns=['BB_Std'])
    
    return df

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Moving Average Convergence Divergence (MACD)."""
    # Calculates EMAs safely using transform
    ema_12 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    
    # Calculates MACD Line
    df['MACD_Line'] = ema_12 - ema_26
    
    # Calculates Signal Line and Histogram
    df['MACD_Signal'] = df.groupby('Symbol')['MACD_Line'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculates the 14-day Relative Strength Index (RSI)."""
    def _rsi(series):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Wilder's smoothing: ewm with com=period-1 matches the original RSI spec
        # (smoothing factor α = 1/period, equivalent to com = period-1).
        # Simple rolling mean gives a different, non-standard RSI.
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        # Prevents division by zero
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = df.groupby('Symbol')['Close'].transform(_rsi)
    return df


def compute_volatility_and_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates returns, rolling volatility tracking, and historical volumes."""
    # Historical base returns
    df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
    
    # 20-day Rolling Volatility (annualized)
    df['Volatility_20'] = df.groupby('Symbol')['Daily_Return'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    
    # Momentum: Rate of Change (ROC - 10 day)
    df['ROC_10'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=10))
    
    # Volume metrics
    df['Volume_SMA_20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-9)
    
    return df

def generate_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates shift targets safely for supervised ML models.
    Shifts backward (-1) within symbols so models predict tomorrow's values.
    """
    # Compute the shifted return first — this preserves NaN on the last row per symbol
    df['Target_Return'] = df.groupby('Symbol')['Daily_Return'].shift(-1)
    
    # Vectorized classification: derive direction from Target_Return so NaN rows stay NaN
    df['Target_Dir'] = np.where(df['Target_Return'].notna(),
                                (df['Target_Return'] > 0).astype(int),
                                np.nan)
    
    return df

def run_pipeline(file_path: str) -> pd.DataFrame:
    """
    Executive orchestrator function to run data cleaning, advanced feature engineering, 
    and output a ready-to-train dataframe.
    """
    print("[+] Loading raw data and optimizing timelines...")
    df = load_raw_data(file_path)
    
    print("[+] Engineering basic and exponential moving averages...")
    df = compute_moving_averages(df)
    
    print("[+] Constructing Bollinger Bands parameters...")
    df = compute_bollinger_bands(df)
    
    print("[+] Engineering 14-day Relative Strength Index (RSI)...")
    df = compute_rsi(df)
    
    print("[+] Constructing MACD structures...")
    df = compute_macd(df)
    
    print("[+] Computing structural volatility and volume ratios...")
    df = compute_volatility_and_momentum(df)
    
    print("[+] Compiling future machine learning shift targets...")
    df = generate_ml_targets(df)
    
    # Clean up edge rows created by early rolling calculations and the final shift row
    print("[+] Purging structural NaNs from feature lookback margins...")
    initial_count = len(df)
    df = df.dropna(subset=[
        'SMA_50', 'BB_High', 'RSI', 'MACD_Line',
        'Volatility_20', 'Target_Return', 'Target_Dir'
    ])
    final_count = len(df)
    print(f"Pipeline complete. Cleaned from {initial_count} down to {final_count} high-fidelity records.")
    
    return df

if __name__ == "__main__":
    # Test path execution stub
    processed_df = run_pipeline("data/NIFTY50_all.csv")
    print(processed_df[['Date', 'Symbol', 'RSI', 'MACD_Hist', 'Target_Dir']].head())