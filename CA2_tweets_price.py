import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")

# Load data
df = pd.read_csv("stocks_price.csv")
df["date"] = pd.to_datetime(df["date"])

# Title 
st.title("Stock Price and Sentiment Analysis")
st.info("Analysis of 5 stocks (TSLA, AAPL, BA, DIS, AMZN) using Twitter sentiment "
        "to forecast prices. Models: ARIMAX, LSTM - January to December 2020.")

# Filters
col1, col2, col3a, col3b = st.columns(4)
with col1:
    ticker = st.selectbox("Ticker", sorted(df["ticker"].unique()))
with col2:
    metric_label = st.selectbox("Metric", ["Close Price", "Sentiment", "Volatility 7D", "Daily Return"])
    
	
# Month filter — cleaner for 12-month dataset
months = {
	"January": 1, "February": 2, "March": 3, "April": 4,"May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    
with col3a: start_month = st.selectbox("From month",  list(months.keys()), index=0)
with col3b: end_month = st.selectbox("To month",  list(months.keys()), index=0)

filtered = df[
    (df["ticker"] == ticker) &
    (df["date"].dt.month >= months[start_month]) &
    (df["date"].dt.month <= months[end_month])].sort_values("date")

# Metrics
METRIC_MAP = {"Close Price":"close","Sentiment":"avg_sentiment",
              "Volatility 7D":"volatility_7d","Daily Return":"daily_return"}
metric_col = METRIC_MAP[metric_label]


# All results using tabs
tab1, tab2, tab3, = st.tabs(["Overview", "Forecast Results", "Model Comparison",])

#Tab 1 - Overview
with tab1:
# Metrics
    a, b, c, d = st.columns(4)
    a.metric("Avg Close Price",   f"${filtered['close'].mean():.2f}",       border=True)
    b.metric("Avg Sentiment",     f"{filtered['avg_sentiment'].mean():.3f}", border=True)
    c.metric("Avg Daily Return",  f"{filtered['daily_return'].mean()*100:.2f}%", border=True)
    d.metric("Avg Volatility 7D", f"{filtered['volatility_7d'].mean():.4f}", border=True)

    st.divider()

    fig = px.line(filtered, x=filtered["date"], y=metric_col, 
                  title=f"Figure 1 - {metric_label} — {ticker}",
                  labels={"date": "Date", metric_col: metric_label},
                  template="plotly_white",
                  color_discrete_sequence=['#0000ff'])
    fig.update_traces(line=dict(width=2))

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Figure 1: {metric_label} for {ticker} from "
                   f"{start_month} to {end_month} 2020. "
                   f"Dashed line marks the train/test split (Nov 1).")
    with col_right:
        display_df = filtered[["date", metric_col]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        
        st.dataframe(display_df, use_container_width=True)
        st.caption("Table 1: observations for selected metric.")

    st.divider()

    # Sentiment vs Price
    st.subheader("Sentiment vs Close Price")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered["date"], y=filtered["close"], 
                   name="Close Price", yaxis="y1", line=dict(color="#1f77b4", width=2)))
    fig2.add_trace(go.Bar(x=filtered["date"], y=filtered["avg_sentiment"], 
                   name="Sentiment", yaxis="y2", marker_color=filtered["avg_sentiment"].apply
                   ( lambda x: "#2ca02c" if x >= 0 else "#d62728")))
    fig2.update_layout(
        template="plotly_white",
        yaxis=dict(title="Close Price ($)"),
        yaxis2=dict(title="Sentiment Score", overlaying="y", side="right", range=[-1,1]),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Figure 2: Daily sentiment score (bars) overlaid with close price (line). ")

#Tab 2 - Forecast Results

with tab2:
    st.subheader(f"Forecast Results — {ticker}")
    st.caption("Forecasts generated using ARIMAX and LSTM models trained on Jan–Oct 2020, "
               "evaluated on Nov–Dec 2020 test period.")

    # Load forecast CSVs if available, otherwise show placeholder
        df_arimax = pd.read_csv("forecast_arimax.csv")
        df_arimax["date"] = pd.to_datetime(df_arimax["date"])
        df_arimax_t = df_arimax[df_arimax["ticker"] == ticker]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_arimax_t["date"], y=df_arimax_t["actual"], 
                       name="Actual", line=dict(color="red", width=2)))
        fig3.add_trace(go.Scatter(x=df_arimax_t["date"], y=df_arimax_t["t1"], 
                       name="t+1", line=dict(color="green", width=1.5)))
        fig3.add_trace(go.Scatter(x=df_arimax_t["date"], y=df_arimax_t["t3"], 
                       name="t+3", line=dict(color="orange", width=1.5)))
        fig3.add_trace(go.Scatter(x=df_arimax_t["date"], y=df_arimax_t["t5"], 
                       name="t+5", line=dict(color="blue", width=1.5)))
        fig3.update_layout(title=f"Figure 3 — ARIMAX Forecast: {ticker}", 
                           template="plotly_white", xaxis_title="Date", yaxis_title="Close Price ($)")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Figure 3: ARIMAX rolling walk-forward forecast at t+1, t+3, t+5 horizons ")

        df_lstm = pd.read_csv("forecast_lstm.csv")
        df_lstm["date"] = pd.to_datetime(df_lstm["date"])
        df_lstm_t = df_lstm[df_lstm["ticker"] == ticker]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_lstm_t["date"], y=df_lstm_t["actual"],
                                   name="Actual", line=dict(color="red", width=2)))
        fig4.add_trace(go.Scatter(x=df_lstm_t["date"], y=df_lstm_t["t1"],
                                   name="t+1", line=dict(color="green", width=1.5)))
        fig4.add_trace(go.Scatter(x=df_lstm_t["date"], y=df_lstm_t["t3"],
                                   name="t+3", line=dict(color="orange", width=1.5)))
        fig4.add_trace(go.Scatter(x=df_lstm_t["date"], y=df_lstm_t["t5"],
                                   name="t+5", line=dict(color="blue", width=1.5)))
        fig4.update_layout(title=f"Figure 4 — LSTM Forecast: {ticker}",
                           template="plotly_white",
                           xaxis_title="Date", yaxis_title="Close Price ($)")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Figure 4: LSTM direct multi-step forecast at t+1, t+3, t+5 horizons. ")
