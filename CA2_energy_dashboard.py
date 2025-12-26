import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Energy Dashboard", layout="wide")

df= pd.read_csv("energy.csv")

st.title("ENERGY TRANSITION: IRELAND AND URUGUAY")
st.info("Uruguay shows consistently lower import dependency and higher renewable production, indicating a more advanced stage of energy transition compared to Ireland.")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", sorted(df.country.unique()))

with col2:
    year = st.slider("Year", int(df.year.min()), int(df.year.max()), 2023)

filtered = df[(df.country == country) & (df.year <= year)]

METRICS = {
    "Import Dependency": "import_dependency",
    "Domestic Supply": "domestic_supply",
    "% Renewable Production": "renew_prod_%",
    "% Non-Renewable Production": "no_renew_prod_%"
}
metric_label = st.selectbox("Select metric", list(METRICS.keys()))
metric_col = METRICS[metric_label]

row = df[(df["country"] == country) & (df["year"] == year)]

def get_value(col):
    if col not in row.columns or row.empty:
        return np.nan
    return row[col].mean()

def fmt(label, val):
    if pd.isna(val):
        return "N/A"
    else:
        return f"{val:,.2f*100}%"

a, b = st.columns(2)
c, d = st.columns(2)

slots = [a, b, c, d]
for (label, col), slot in zip(METRICS.items(), slots):
    val = get_value(col)
    slot.metric(label, fmt(label, val), delta=None, border=True)

summary = (
    filtered.groupby("year")[metric_col]
    .mean()
    .reset_index(name=metric_col)
)

chart = alt.Chart(summary).mark_line(point=True).encode(
    x="year:O",
    y=alt.Y(f"{metric_col}:Q", title=metric_col),
    tooltip=["year", metric_col]
)

col_left, col_right = st.columns([2,1])
with col_left: st.plotly_chart(fig, use_container_width=True)

with col_right: st.dataframe(summary, use_container_width=True)