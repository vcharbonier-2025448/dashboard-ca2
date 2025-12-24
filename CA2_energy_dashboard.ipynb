import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Energy Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ireland_energy.csv")

df = load_data()

st.title("Energy Domestic Supply")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", sorted(df.country.unique()))

with col2:
    year = st.slider("Year", int(df.year.min()), int(df.year.max()), 2023)

filtered = df[(df.country == country) & (df.year <= year)]

METRICS = {
    "Domestic Supply": "domestic_supply",
    "Import Dependency": "import_dependency",
    "% Renewable Production": "renew_prod_%"
}
metric_label = st.selectbox("Select metric", list(METRICS.keys()))
metric_col = METRICS[metric_label]

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

st.altair_chart(chart, use_container_width=True)
st.dataframe(summary, use_container_width=True)