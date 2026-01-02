# app.py
# A Streamlit app to upload a CSV with Latitude/Longitude, run K-Means, and visualize clusters.

#pip install streamlit pandas numpy scikit-learn matplotlib pydeck


import streamlit as st
import pandas as pd
import numpy as np

# K-Means clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# For plotting
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Walk Clusters (K-Means)", layout="wide")

# -----------------------------
# Sidebar: App controls
# -----------------------------
st.sidebar.title("🧭 K-Means Clustering Map")
st.sidebar.write("Upload a CSV with **latitude** and **longitude** columns.")

uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Clustering settings")

k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=12, value=4, step=1)

# A random seed makes results reproducible
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

use_scaling = st.sidebar.checkbox("Scale coordinates before clustering (recommended)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Map settings")
map_style = st.sidebar.selectbox(
    "Map style (pydeck)",
    ["light", "dark", "road", "satellite"],
    index=0
)

# -----------------------------
# Helper: Find likely lat/lon columns
# -----------------------------
def guess_lat_lon_columns(columns):
    """
    Tries to guess latitude and longitude column names from CSV header.
    Returns (lat_col, lon_col) or (None, None).
    """
    cols_lower = [c.lower().strip() for c in columns]
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "lng", "long", "longitude", "x"]

    lat_col = None
    lon_col = None

    for i, c in enumerate(cols_lower):
        if c in lat_candidates:
            lat_col = columns[i]
        if c in lon_candidates:
            lon_col = columns[i]

    return lat_col, lon_col

# -----------------------------
# Main UI
# -----------------------------
st.title("🚶 Student Walking Coordinates → K-Means Clusters")
st.write(
    "Upload a CSV containing **latitude** and **longitude** points collected while students walked. "
    "This app will group points into **K clusters** and visualize them."
)

if uploaded is None:
    st.info("👈 Upload a CSV from the sidebar to begin.")
    st.stop()

# -----------------------------
# Load CSV
# -----------------------------
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV file. Error: {e}")
    st.stop()

st.subheader("1) Preview of your data")
st.write("Here are the first rows of your CSV:")
st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# Select lat/lon columns
# -----------------------------
st.subheader("2) Choose latitude & longitude columns")

guessed_lat, guessed_lon = guess_lat_lon_columns(df.columns)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    lat_col = st.selectbox(
        "Latitude column",
        options=list(df.columns),
        index=(list(df.columns).index(guessed_lat) if guessed_lat in df.columns else 0),
    )

with col2:
    lon_col = st.selectbox(
        "Longitude column",
        options=list(df.columns),
        index=(list(df.columns).index(guessed_lon) if guessed_lon in df.columns else min(1, len(df.columns) - 1)),
    )

with col3:
    st.write(
        "✅ Tip: Latitude values are usually around **-90 to 90**, longitude around **-180 to 180**.\n\n"
        "If your columns are named something else (like `Lat`/`Lng`), just pick them here."
    )

# -----------------------------
# Clean + validate
# -----------------------------
working = df.copy()

# Convert to numeric (turns bad values into NaN)
working[lat_col] = pd.to_numeric(working[lat_col], errors="coerce")
working[lon_col] = pd.to_numeric(working[lon_col], errors="coerce")

# Remove missing values
before = len(working)
working = working.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
after = len(working)

if after == 0:
    st.error("After cleaning, there are no valid latitude/longitude rows left. Check your chosen columns.")
    st.stop()

if after < before:
    st.warning(f"Removed {before - after} rows that had missing or non-numeric latitude/longitude values.")

# Optional: filter out unrealistic coordinate ranges
with st.expander("Optional: Filter out unrealistic coordinates"):
    filter_bad = st.checkbox("Remove points outside valid lat/lon ranges", value=True)
    if filter_bad:
        before2 = len(working)
        working = working[
            (working[lat_col].between(-90, 90)) &
            (working[lon_col].between(-180, 180))
        ].reset_index(drop=True)
        after2 = len(working)
        if after2 < before2:
            st.info(f"Filtered out {before2 - after2} points outside valid coordinate ranges.")

if len(working) < k:
    st.error(f"You have only {len(working)} valid points but K={k}. Reduce K or upload more data.")
    st.stop()

# -----------------------------
# Run K-Means
# -----------------------------
st.subheader("3) Run K-Means clustering")

coords = working[[lat_col, lon_col]].to_numpy()

# Scaling often helps K-Means behave better (especially if lat/lon range differs)
if use_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)
else:
    X = coords

kmeans = KMeans(n_clusters=k, random_state=int(seed), n_init="auto")
labels = kmeans.fit_predict(X)

working["cluster"] = labels

# Get cluster centers back in lat/lon for display
if use_scaling:
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
else:
    centers = kmeans.cluster_centers_

centers_df = pd.DataFrame(centers, columns=[lat_col, lon_col])
centers_df["cluster"] = np.arange(k)

st.success("✅ Clustering complete!")

# -----------------------------
# Show clustered data + filters
# -----------------------------
st.subheader("4) Explore clusters (filter + table)")

left, right = st.columns([1, 2])

with left:
    cluster_choice = st.multiselect(
        "Show only these clusters",
        options=sorted(working["cluster"].unique()),
        default=sorted(working["cluster"].unique())
    )

    show_centers = st.checkbox("Show cluster centers", value=True)

    st.write("Cluster sizes:")
    size_table = working.groupby("cluster").size().reset_index(name="points").sort_values("cluster")
    st.dataframe(size_table, use_container_width=True)

with right:
    filtered = working[working["cluster"].isin(cluster_choice)].copy()

    st.write(f"Showing **{len(filtered)}** points:")
    st.dataframe(filtered.head(200), use_container_width=True)

# Download clustered CSV
st.download_button(
    "⬇️ Download clustered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="clustered_coordinates.csv",
    mime="text/csv",
)

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("5) Visualizations")

viz1, viz2 = st.columns(2)

# --- Bar chart: points per cluster
with viz1:
    st.write("📊 Bar chart: number of points in each cluster (filtered)")
    bar_data = filtered.groupby("cluster").size().reindex(sorted(cluster_choice), fill_value=0)
    fig_bar = plt.figure()
    plt.bar(bar_data.index.astype(str), bar_data.values)
    plt.xlabel("Cluster")
    plt.ylabel("Number of points")
    st.pyplot(fig_bar)

# --- Pie chart: cluster proportions
with viz2:
    st.write("🥧 Pie chart: cluster proportions (filtered)")
    pie_data = filtered.groupby("cluster").size().reindex(sorted(cluster_choice), fill_value=0)
    fig_pie = plt.figure()
    plt.pie(pie_data.values, labels=[str(c) for c in pie_data.index], autopct="%1.1f%%")
    st.pyplot(fig_pie)

# -----------------------------
# Map view
# -----------------------------
st.subheader("6) Map of clusters")

# Streamlit has a simple map, but it doesn't color by cluster.
# We'll use pydeck for colored clusters.
import pydeck as pdk

# Create a stable color per cluster
def color_for_cluster(c):
    rng = np.random.default_rng(int(seed) + int(c) * 999)
    # RGB values 0..255
    return [int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255))]

map_df = filtered[[lat_col, lon_col, "cluster"]].copy()
map_df["color"] = map_df["cluster"].apply(color_for_cluster)

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=[lon_col, lat_col],
        get_color="color",
        get_radius=12,
        pickable=True,
        opacity=0.8,
    )
]

if show_centers:
    centers_filtered = centers_df[centers_df["cluster"].isin(cluster_choice)].copy()
    centers_filtered["color"] = centers_filtered["cluster"].apply(lambda c: [0, 0, 0])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=centers_filtered,
            get_position=[lon_col, lat_col],
            get_color="color",
            get_radius=40,
            pickable=True,
            opacity=0.9,
        )
    )

# Center the map around the average coordinate
view_state = pdk.ViewState(
    latitude=float(map_df[lat_col].mean()),
    longitude=float(map_df[lon_col].mean()),
    zoom=12,
)

style_lookup = {
    "light": "mapbox://styles/mapbox/light-v10",
    "dark": "mapbox://styles/mapbox/dark-v10",
    "road": "mapbox://styles/mapbox/streets-v11",
    "satellite": "mapbox://styles/mapbox/satellite-v9",
}

deck = pdk.Deck(
    map_style=style_lookup.get(map_style, style_lookup["light"]),
    initial_view_state=view_state,
    layers=layers,
    tooltip={"text": f"Cluster: {{cluster}}\nLat: {{{lat_col}}}\nLon: {{{lon_col}}}"},
)

st.pydeck_chart(deck, use_container_width=True)

# -----------------------------
# Cluster center table
# -----------------------------
st.subheader("7) Cluster centers (approximate group locations)")
st.write("These are the K-Means center points (in lat/lon).")
st.dataframe(centers_df.sort_values("cluster"), use_container_width=True)

st.caption(
    "Note: K-Means groups points by distance. On the Earth, lat/lon distance is only approximately Euclidean "
    "over small areas. For large regions, consider projecting coordinates or using haversine-based clustering."
)
