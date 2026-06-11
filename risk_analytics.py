import pandas as pd
import numpy as np


def compute_asset_metrics(df: pd.DataFrame, risk_free_rate: float = 0.06) -> pd.DataFrame:
    """
    Scans the entire processed dataset and calculates historical risk/return
    metrics for every unique stock ticker dynamically.
    """
    asset_data = []
    symbols = df['Symbol'].unique()

    # Convert annual risk free rate to daily equivalent
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    for sym in symbols:
        stock_df = df[df['Symbol'] == sym]
        returns = stock_df['Daily_Return']

        if len(returns) < 50:
            continue

        # 1. Total Cumulative Return
        cum_return = (1 + returns).prod() - 1

        # 2. Annualized Volatility
        ann_vol = returns.std() * np.sqrt(252)

        # 3. Sharpe Ratio — denominator is excess_returns.std(), not returns.std()
        excess_returns = returns - daily_rf
        sharpe = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)

        # 4. Sortino Ratio — penalises only downside volatility, not upside.
        # Downside deviation = std of returns that fell below the risk-free threshold.
        # A stock that is volatile only on the upside is NOT penalised by Sortino.
        downside_returns = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / (downside_returns.std() + 1e-9)) * np.sqrt(252)

        # 5. Maximum Drawdown
        cum_prod = (1 + returns).cumprod()
        running_max = cum_prod.cummax()
        drawdowns = (cum_prod - running_max) / (running_max + 1e-9)
        max_dd = drawdowns.min()

        asset_data.append({
            'Symbol':               sym,
            'Cum_Return':           cum_return,
            'Annualized_Volatility': ann_vol,
            'Sharpe_Ratio':         sharpe,
            'Sortino_Ratio':        sortino,
            'Max_Drawdown':         max_dd,
        })

    return pd.DataFrame(asset_data)


def generate_profile_portfolio(metrics_df: pd.DataFrame, profile: str):
    """
    Dynamically constructs an investment portfolio with real-world quantitative
    justifications based on the user's selected risk tolerance profile.

    Allocation weights use inverse-volatility weighting:
        w_i = (1 / vol_i) / Σ(1 / vol_j)
    This gives proportionally higher weight to lower-volatility assets and
    normalises automatically to sum to 1 regardless of how many stocks are selected.
    """
    # Clean out any extreme infinite outliers
    metrics_df = metrics_df.dropna().replace([np.inf, -np.inf], 0)

    if profile == "Conservative":
        # Select 3 stocks with lowest historical volatility
        selected = metrics_df.nsmallest(3, 'Annualized_Volatility')
        justification = (
            "Prioritises capital preservation by selecting the three NIFTY-50 companies "
            "with the lowest historical price volatility. Inverse-volatility weighting "
            "allocates the most capital to the most stable anchor asset."
        )

    elif profile == "Balanced":
        # Select 3 stocks with highest risk-adjusted return (Sharpe)
        selected = metrics_df.nlargest(3, 'Sharpe_Ratio')
        justification = (
            "Optimised for risk-adjusted performance. Selects assets with the highest "
            "Sharpe Ratio, then applies inverse-volatility weighting so the least volatile "
            "of the three still receives proportionally higher allocation."
        )

    else:  # Aggressive
        # Select 3 stocks with highest total cumulative return
        selected = metrics_df.nlargest(3, 'Cum_Return')
        justification = (
            "Designed for maximum capital compounding. Isolates the three highest-return "
            "assets in the index, with inverse-volatility weighting providing a quantitative "
            "risk floor even within this high-momentum strategy."
        )

    # Inverse-volatility weighting:
    # Lower volatility → higher 1/vol → higher weight.
    # Normalising by the sum guarantees weights always sum exactly to 1,
    # and handles edge cases (< 3 stocks) without any separate guard logic.
    tickers = selected['Symbol'].tolist()
    vols    = selected['Annualized_Volatility'].values
    inv_vols = 1.0 / (vols + 1e-9)
    weights  = inv_vols / inv_vols.sum()

    allocations = {ticker: float(weights[i]) for i, ticker in enumerate(tickers)}

    return allocations, justification, selected
