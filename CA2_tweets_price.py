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
st.info("Analysis of 5 stocks (TSLA, AAPL, BA, DIS, AMZN) using Twitter sentiment to forecast prices. Models: ARIMAX, LSTM - January to December 2020.")

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
                   f"{start_month} to {end_month} 2020")
    with col_right:
        display_df = filtered[["date", metric_col]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        
        st.dataframe(display_df, use_container_width=True)
  

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
   
#Tab 2 - Forecast Results

with tab2:
    st.subheader(f"Forecast Results — {ticker}")
 

    # Load forecast CSVs
    df_arimax = pd.read_csv("forecast_arimax.csv")
    df_arimax["date"] = pd.to_datetime(df_arimax["date"])
    df_arimax_t = df_arimax[df_arimax["ticker"] == ticker]
    df_arimax_t = df_arimax_t.fillna(0)

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
                       template="plotly_white", xaxis_title="Date", yaxis_title="Close Price ($)")
    st.plotly_chart(fig4, use_container_width=True)

# Tab 3 - Model Comparsion
with tab3:
    st.subheader("Model Performance — MAE / RMSE")
    
    results = pd.DataFrame({
        "Ticker": ["TSLA","TSLA","TSLA","AAPL","AAPL","AAPL","BA",  "BA",  "BA","DIS","DIS", "DIS", "AMZN","AMZN","AMZN"],
        "Model":  ["ARIMAX","LSTM", "LSTM Hyper. Tuning"] * 5,
        "MAE t+1":  [22.21, 36.96, 39.28,3.25, 3.69, 3.36,24.64, 21.79,18.02 ,21.17, 22.49, 15.69,2.81, 2.58,2.80],
        "RMSE t+1": [23.53, 43.67, 46.85,4.18, 4.74, 3.94,27.07, 25.28, 20.03,24.32, 25.65, 17.79,3.60, 3.39,3.67],
        "MAE t+3":  [35.40, 35.66, 37.01,3.77, 5.61, 5.79,11.30, 54.78, 15.32,25.85, 22.62, 33.53,5.63, 9.05,13.81],
        "RMSE t+3": [40.31, 40.91, 42.35,4.59, 6.49, 7.17,16.66, 57.00, 17.41,28.26, 24.59, 35.66,6.26, 6.68,14.19],
        "MAE t+5":  [78.08, 45.35, 31,88,4.21, 31.68, 6.21,39.13,54.46, 20.72,24.70, 31.11, 23.39,8.75, 38.48,8.65],
        "RMSE t+5": [83.73, 51.86, 34.70,5.00, 32.06, 7.54,41.78,56.51, 24.27,27.58, 33.81, 26.08,9.22, 38.58,9.16]})

    col_a, col_b = st.columns([1, 2])
    with col_a:
        selected = st.radio("Filter by ticker", ["All", "TSLA", "AAPL", "BA", "DIS", "AMZN"] )
    with col_b:
        show = results if selected == "All" else results[results["Ticker"] == selected]
        st.dataframe(show, use_container_width=True, hide_index=True)

    # Bar chart of MAE comparison t+1
    fig5 = px.bar(results, x="Ticker", y="MAE t+1", color="Model",
                  barmode="group", template="plotly_white",
                  title="Figure 5 — MAE t+1 by Model and Ticker",
                  color_discrete_sequence=["#1f77b4","#ff7f0e"])
    st.plotly_chart(fig5, use_container_width=True)
  
 
 # Bar chart of MAE comparison t+3
    fig6 = px.bar(results, x="Ticker", y="MAE t+3", color="Model",
                  barmode="group", template="plotly_white",
                  title="Figure 6 — MAE t+3 by Model and Ticker",
                  color_discrete_sequence=["#1f77b4","#ff7f0e"])
    st.plotly_chart(fig6, use_container_width=True)
   
 
 # Bar chart of MAE comparison t+5
    fig7 = px.bar(results, x="Ticker", y="MAE t+5", color="Model",
                  barmode="group", template="plotly_white",
                  title="Figure 7 — MAE t+3 by Model and Ticker",
                  color_discrete_sequence=["#1f77b4","#ff7f0e"])
    st.plotly_chart(fig7, use_container_width=True)
 