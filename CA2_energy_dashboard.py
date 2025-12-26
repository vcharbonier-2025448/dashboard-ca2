import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 


# Importing standardscalar module 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scalar = StandardScaler()

st.set_page_config(page_title="Energy Dashboard", layout="wide")

df= pd.read_csv("energy.csv")
df_ie=pd.read_csv("ireland_energy.csv")

st.title("ENERGY TRANSITION: IRELAND AND URUGUAY")

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
        return f"{val:,.2f}*100%"

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

with col1:
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.dataframe(summary, use_container_width=True)


#Unnsupervised model

  
# fitting
scalar.fit(df_ie)
scaled_data = scalar.transform(df_ie)
  
# Importing PCA
from sklearn.decomposition import PCA
  
# components = 2
pca = PCA(n_components = 2, random_state = 42)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
  
# Instantiate the KMeans models

k_results = []

for k in range(2,8):
    km = KMeans(n_clusters = k, random_state=42,n_init=10)
    labels = km.fit_predict(x_pca)

    # Calculate Silhoutte Score
    score = silhouette_score(x_pca, labels, metric='euclidean')

    k_results.append({"k" : k,
                      "silhouette": score})


k_results_df = pd.DataFrame(k_results)
best_k = int(k_results_df.loc[k_results_df["silhouette"].idxmax(),"k"])
print(f' The best value of k is: {best_k}')

kmeans_final = KMeans (n_clusters = best_k, random_state = 42, n_init=10)
cluster_labels = kmeans_final.fit_predict(x_pca)

features = ["import_dependency","domestic_supply", "energy_balance", "renew_prod_%", "no_renew_prod_%"]
x_unsup = df_ie[features].dropna().reset_index(drop=True)

df_unsup = df_ie.loc[x_unsup.index].copy()

df_unsup["cluster"] = cluster_labels

df_unsup[["cluster"] + features].groupby("cluster").mean()

plt.figure(figsize = (8,4))

plt.scatter(x_pca[:,0], x_pca[:,1], c=df_unsup["cluster"], cmap = 'viridis' )

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("KMeans Clusters in PCA Space")

plt.show()