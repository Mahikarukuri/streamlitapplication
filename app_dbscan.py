# app.py
# Streamlit app to upload a CSV with Latitude/Longitude and run DBSCAN clustering.

import streamlit as st
import pandas as pd
import numpy as np

# DBSCAN clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# For plotting
import matplotlib.pyplot as plt

# For map visualization
import pydeck as pdk

st.set_page_config(page_title="Student Walk Clusters (DBSCAN)", layout="wide")

# -----------------------------
# Sidebar: App controls
# -----------------------------
st.sidebar.title("🧭 DBSCAN Clustering Map")
st.sidebar.write("Upload a CSV with **latitude** and **longitude** columns.")

uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("DBSCAN settings")

# eps in DBSCAN depends on whether we scale or not:
# - if scaling: eps ~ 0.1 to 2.0 is common
# - if not scaling (raw lat/lon): eps ~ 0.0001 to 0.05 might be common depending on area size
use_scaling = st.sidebar.checkbox("Scale coordinates before clustering (recommended)", value=True)

if use_scaling:
    eps = st.sidebar.slider("eps (neighborhood radius)", 0.05, 5.0, 0.5, 0.05)
else:
    eps = st.sidebar.slider("eps (in degrees lat/lon)", 0.0001, 0.1, 0.01, 0.0001)

min_samples = st.sidebar.slider("min_samples (points to form a dense area)", 2, 50, 8, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Map settings")
map_style = st.sidebar.selectbox(
    "Map style (pydeck)",
    ["light", "dark", "road", "satellite"],
    index=0
)

# -----------------------------
# Helper: Guess lat/lon columns
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
st.title("🚶 Student Walking Coordinates → DBSCAN Clusters")
st.write(
    "Upload a CSV containing **latitude** and **longitude** points collected while students walked. "
    "This app will use **DBSCAN** to find dense clusters and mark sparse points as **noise**."
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
st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# Select lat/lon columns
# -----------------------------
st.subheader("2) Choose latitude & longitude columns")

guessed_lat, guessed_lon = guess_lat_lon_columns(df.columns)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    lat_col = st.selectbox(
        "Latitude column",
        options=list(df.columns),
        index=(list(df.columns).index(guessed_lat) if guessed_lat in df.columns else 0),
    )

with c2:
    lon_col = st.selectbox(
        "Longitude column",
        options=list(df.columns),
        index=(list(df.columns).index(guessed_lon) if guessed_lon in df.columns else min(1, len(df.columns) - 1)),
    )

with c3:
    st.write(
        "✅ Tip: Latitude is typically **-90..90**, longitude **-180..180**.\n\n"
        "DBSCAN will label far-away / isolated points as **noise (-1)**."
    )

# -----------------------------
# Clean + validate
# -----------------------------
working = df.copy()

# Convert to numeric (bad values -> NaN)
working[lat_col] = pd.to_numeric(working[lat_col], errors="coerce")
working[lon_col] = pd.to_numeric(working[lon_col], errors="coerce")

before = len(working)
working = working.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
after = len(working)

if after == 0:
    st.error("After cleaning, there are no valid latitude/longitude rows left. Check your chosen columns.")
    st.stop()

if after < before:
    st.warning(f"Removed {before - after} rows that had missing or non-numeric latitude/longitude values.")

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

if len(working) < min_samples:
    st.error(f"You have only {len(working)} valid points, but min_samples={min_samples}. Upload more data or lower min_samples.")
    st.stop()

# -----------------------------
# Run DBSCAN
# -----------------------------
st.subheader("3) Run DBSCAN clustering")

coords = working[[lat_col, lon_col]].to_numpy()

# Scale if requested (often helps)
if use_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)
else:
    X = coords

db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
labels = db.fit_predict(X)

working["cluster"] = labels  # -1 is noise

n_noise = int((working["cluster"] == -1).sum())
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

st.success(f"✅ Done! Found **{n_clusters}** clusters and **{n_noise}** noise points (cluster = -1).")

# -----------------------------
# Explore clusters
# -----------------------------
st.subheader("4) Explore clusters (filter + table)")

left, right = st.columns([1, 2])

with left:
    all_clusters = sorted(working["cluster"].unique())

    # Default selection: all clusters including noise
    cluster_choice = st.multiselect(
        "Show only these labels (cluster numbers, -1 = noise)",
        options=all_clusters,
        default=all_clusters
    )

    st.write("Label sizes:")
    size_table = working.groupby("cluster").size().reset_index(name="points").sort_values("cluster")
    st.dataframe(size_table, use_container_width=True)

with right:
    filtered = working[working["cluster"].isin(cluster_choice)].copy()
    st.write(f"Showing **{len(filtered)}** points:")
    st.dataframe(filtered.head(200), use_container_width=True)

st.download_button(
    "⬇️ Download clustered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="dbscan_clustered_coordinates.csv",
    mime="text/csv",
)

# -----------------------------
# Charts
# -----------------------------
st.subheader("5) Visualizations")

viz1, viz2 = st.columns(2)

with viz1:
    st.write("📊 Bar chart: points per label (filtered)")
    bar_data = filtered.groupby("cluster").size().reindex(sorted(cluster_choice), fill_value=0)
    fig_bar = plt.figure()
    plt.bar([str(x) for x in bar_data.index], bar_data.values)
    plt.xlabel("Cluster label (-1 = noise)")
    plt.ylabel("Number of points")
    st.pyplot(fig_bar)

with viz2:
    st.write("🥧 Pie chart: label proportions (filtered)")
    pie_data = filtered.groupby("cluster").size().reindex(sorted(cluster_choice), fill_value=0)

    fig_pie = plt.figure()

    # If user includes noise, pie chart shows it too.
    plt.pie(pie_data.values, labels=[str(c) for c in pie_data.index], autopct="%1.1f%%")
    st.pyplot(fig_pie)

# -----------------------------
# Map visualization (pydeck)
# -----------------------------
st.subheader("6) Map of DBSCAN clusters")

def color_for_label(label):
    """
    Stable random color for each cluster label.
    Noise label (-1) is gray.
    """
    if int(label) == -1:
        return [140, 140, 140]
    rng = np.random.default_rng(int(label) * 999 + 12345)
    return [int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255))]

map_df = filtered[[lat_col, lon_col, "cluster"]].copy()
map_df["color"] = map_df["cluster"].apply(color_for_label)

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=[lon_col, lat_col],
        get_color="color",
        get_radius=12,
        pickable=True,
        opacity=0.85,
    )
]

view_state = pdk.ViewState(
    latitude=float(map_df[lat_col].mean()),
    longitude=float(map_df[lon_col].mean()),
    zoom=12
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
# Notes about DBSCAN + coordinates
# -----------------------------
st.subheader("7) Notes (important for lat/lon)")

st.write(
    """
**DBSCAN** needs a good value of **eps**:
- If you checked **Scale coordinates**, eps is in *standardized units* (often 0.2–1.5 is a good starting range).
- If you did **not** scale, eps is in *degrees* (for walking data, try very small values like 0.001–0.02).

Also: lat/lon are on a sphere. For **large areas**, a more accurate approach is to use a distance method like
**haversine** (great-circle distance). For short walking distances, this simple approach usually works fine.
"""
)
