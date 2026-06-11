import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_pipeline import run_pipeline
from predictor import train_directional_model
from risk_analytics import compute_asset_metrics, generate_profile_portfolio

st.set_page_config(page_title="NIFTY-50 Intelligence Engine", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; }
    .section-header { font-size:24px !important; font-weight: bold; margin-top: 20px; color: #0F766E; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 AI-Powered Investment Intelligence Platform")
st.subheader("Enterprise-Grade Financial Analytics on Indian Indices Market Data")
st.divider()

#  Cache Data Loading For Fast Execution 
@st.cache_data
def load_processed_market_data():
    return run_pipeline("data/NIFTY50_all.csv")

try:
    df = load_processed_market_data()
    all_symbols = sorted(df['Symbol'].unique())
    metrics_df = compute_asset_metrics(df)
except Exception as e:
    st.error(f"Failed to load dataset files. Please ensure your path is correct. Error: {e}")
    st.stop()

# Session State: persist model results across Streamlit reruns 
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'model_symbol' not in st.session_state:
    st.session_state.model_symbol = None

# Sidebar Controls
st.sidebar.header("🎯 System Controls")
selected_stock = st.sidebar.selectbox("Select Target Ticker Asset", all_symbols, index=0)
investor_profile = st.sidebar.radio("Target Risk Tolerance Profile", ["Conservative", "Balanced", "Aggressive"])

# Filter focused workspace data
stock_history = df[df['Symbol'] == selected_stock].sort_values('Date')

#  SECTION 1: Historical Market Performance & Indicators 
st.markdown(f"### 📈 Performance Visualizer & Advanced Indicators: {selected_stock}")
col1, col2 = st.columns([2, 1])

with col1:
    # Plot closing price and Bollinger Bands dynamically using Plotly
    fig = px.line(stock_history, x='Date', y=['Close', 'BB_High', 'BB_Low'],
                  labels={'value': 'Price (INR)', 'variable': 'Metrics'},
                  title=f"{selected_stock} Structural Valuation Timelines")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Core Risk Architecture Readings**")
    # Guard: compute_asset_metrics skips symbols with < 50 rows.
    # Without this check, .iloc[0] on an empty DataFrame crashes the app.
    stock_metrics_row = metrics_df[metrics_df['Symbol'] == selected_stock]

    if stock_metrics_row.empty:
        st.warning(f"Insufficient historical data to compute risk metrics for {selected_stock}.")
    else:
        stock_metrics = stock_metrics_row.iloc[0]
        st.metric("Total Compounded Return",          f"{stock_metrics['Cum_Return']:.2%}")
        st.metric("Annualized Asset Volatility",      f"{stock_metrics['Annualized_Volatility']:.2%}")
        st.metric("Sharpe Ratio",                     f"{stock_metrics['Sharpe_Ratio']:.2f}")
        st.metric("Sortino Ratio",                    f"{stock_metrics['Sortino_Ratio']:.2f}")
        st.metric("Maximum Peak-to-Trough Drawdown",  f"{stock_metrics['Max_Drawdown']:.2%}")

st.divider()

# SECTION 2: AI Predictive Modeling & Explainable Framework 
st.markdown("### 🔮 Machine Learning Predictive Engine & Explainability Framework")
col3, col4 = st.columns(2)

with col3:
    st.markdown(f"**Train Prediction Models for {selected_stock}**")
    st.write(
        "Constructs a Random Forest Classifier with 5-fold TimeSeriesSplit cross-validation "
        "to forecast next-day price movement direction. Each fold respects chronological order "
        "— no future data leakage."
    )

    if st.button("🚀 Execute Model Inference Run"):
        with st.spinner("Running 5-fold TimeSeriesSplit CV and fitting final model..."):
            model, cv_accuracy, precision, importance_df = train_directional_model(df, selected_stock)
            # Persist results in session state so they survive Streamlit's rerun cycle
            st.session_state.model_results = {
                'cv_accuracy':  cv_accuracy,
                'precision':    precision,
                'importance_df': importance_df,
            }
            st.session_state.model_symbol = selected_stock

    # Renders metrics only if a model exists for the currently selected stock
    if st.session_state.model_results and st.session_state.model_symbol == selected_stock:
        results = st.session_state.model_results
        st.success("Inference Engine Training Finalized!")
        st.metric("CV Accuracy (5-fold TimeSeriesSplit)", f"{results['cv_accuracy']:.2%}")
        st.metric("Model Precision Score",                f"{results['precision']:.2%}")

# col4 rendered independently — not nested inside col3.
with col4:
    st.markdown("**🧠 Explainable AI (XAI) Model Insights**")
    if st.session_state.model_results and st.session_state.model_symbol == selected_stock:
        results = st.session_state.model_results
        st.write(
            "Gini-importance structural scores highlighting exactly which technical "
            "features drove the directional forecast:"
        )
        feat_fig = px.bar(
            results['importance_df'].head(6),
            x='Importance', y='Feature',
            orientation='h',
            title="Top 6 Mathematical Feature Weights",
            color='Importance', color_continuous_scale='Viridis'
        )
        st.plotly_chart(feat_fig, use_container_width=True)
    else:
        st.info("Run the model on the left to see XAI feature importances here.")
        st.divider()

# SECTION 3: Dynamic Portfolio Management Optimization

st.markdown("### 💼 Automated Portfolio Construction Module")

allocations, justification, target_df = generate_profile_portfolio(metrics_df, investor_profile)

# Portfolio-level metrics
portfolio_return = (pd.Series(allocations) * target_df.set_index('Symbol')['Cum_Return']).sum()
portfolio_vol = (pd.Series(allocations) * target_df.set_index('Symbol')['Annualized_Volatility']).sum()
portfolio_sharpe = (pd.Series(allocations) * target_df.set_index('Symbol')['Sharpe_Ratio']).sum()

# Justification sits full-width at the top, clean and readable
st.info(f"**Strategy Rationale ({investor_profile}):** {justification}")

# Portfolio summary metrics row (full width, 3 cards)
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Weighted Cumulative Return", f"{portfolio_return:.2%}")
with m2:
    st.metric("Weighted Annual Volatility", f"{portfolio_vol:.2%}")
with m3:
    st.metric("Weighted Sharpe Ratio", f"{portfolio_sharpe:.2f}")

st.divider()

# Allocation table + pie chart side by side
col5, col6 = st.columns([1, 2])

with col5:
    st.write(f"**Asset Allocations — {investor_profile} Profile**")
    alloc_df = pd.DataFrame(list(allocations.items()), columns=['Asset Symbol', 'Weight Portfolio Allocation'])
    st.dataframe(alloc_df.style.format({'Weight Portfolio Allocation': '{:.0%}'}))

with col6:
    pie_fig = px.pie(
        alloc_df,
        values='Weight Portfolio Allocation',
        names='Asset Symbol',
        title=f"Dynamic Capital Weights for {investor_profile} Strategy",
        hole=0.4
    )
    st.plotly_chart(pie_fig, use_container_width=True)

# Correlation Heatmap (full width)
st.markdown("#### 🔗 Portfolio Return Correlation Matrix")
st.write(
    "Pairwise Pearson correlation of daily returns across selected portfolio stocks. "
    "Values close to 1 indicate the stocks move together — reducing diversification benefit."
)

portfolio_tickers = list(allocations.keys())
portfolio_returns = (
    df[df['Symbol'].isin(portfolio_tickers)][['Date', 'Symbol', 'Daily_Return']]
    .pivot_table(index='Date', columns='Symbol', values='Daily_Return')
    .dropna()
)

corr_matrix = portfolio_returns.corr()

corr_fig = px.imshow(
    corr_matrix,
    title=f"Correlation Matrix — {investor_profile} Portfolio",
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    text_auto='.2f'
)
st.plotly_chart(corr_fig, use_container_width=True)

st.divider()

# SECTION 4: Market Anomaly Detection
st.markdown("### 🚨 Market Anomaly Detection")
st.write(
    "Flags trading days where the stock's daily return exceeded ±3 standard deviations "
    "of its own historical mean — a widely used statistical threshold for detecting "
    "crash events, short squeezes, and data anomalies."
)

# Compute per-stock mean and std on non-NaN returns only
stock_ret = stock_history['Daily_Return'].dropna()
mean_ret  = stock_ret.mean()
std_ret   = stock_ret.std()

anomaly_mask = (stock_history['Daily_Return'] - mean_ret).abs() > 3 * std_ret
anomalies    = stock_history[anomaly_mask].dropna(subset=['Daily_Return'])

# Line chart of full price history with anomaly days overlaid as red dots
anom_fig = px.line(
    stock_history.dropna(subset=['Close']),
    x='Date', y='Close',
    title=f"{selected_stock} — Price History with Anomaly Days Highlighted (±3σ)"
)
anom_fig.add_scatter(
    x=anomalies['Date'],
    y=anomalies['Close'],
    mode='markers',
    marker=dict(color='red', size=8, symbol='circle'),
    name=f'Anomaly (±3σ return)'
)
st.plotly_chart(anom_fig, use_container_width=True)

st.markdown(f"**{len(anomalies)} anomaly day(s) detected for {selected_stock}**")
if not anomalies.empty:
    st.dataframe(
        anomalies[['Date', 'Close', 'Daily_Return']]
        .sort_values('Daily_Return')
        .reset_index(drop=True)
        .style.format({'Daily_Return': '{:.2%}', 'Close': '{:.2f}'})
    )

st.divider()

