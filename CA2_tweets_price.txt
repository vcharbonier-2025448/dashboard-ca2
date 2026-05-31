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


