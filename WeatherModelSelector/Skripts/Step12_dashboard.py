import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import sys
import time


from streamlit_autorefresh import st_autorefresh

# ============================================================
# WEATHER MODEL SELECTOR - STEP 12
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

PREDICTION_FILE = os.path.join(
    RESULTS_PATH,
    "live_predictions.csv"
)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "weather_model_selector_v2.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Model Selector",
    page_icon="🌦️",
    layout="wide"
)

st_autorefresh(
    interval=3600000,
    key="hourly_refresh"
)


# ============================================================
# AUTOMATIC WEATHER DATA UPDATE
# ============================================================

if "last_update" not in st.session_state:
    st.session_state.last_update = 0

current_time = time.time()

if current_time - st.session_state.last_update >= 3600:

    with st.spinner("Fetching latest weather forecasts..."):

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    BASE_PATH,
                    "scripts",
                    "Step11_live_prediction.py"
                )
            ],
            check=True
        )

    st.session_state.last_update = current_time

# ============================================================
# TITLE
# ============================================================

st.title("🌦️ Weather Model Selector")

st.write(
    "Machine Learning system for selecting the most suitable "
    "numerical weather prediction model for a location."
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_predictions():

    df = pd.read_csv(PREDICTION_FILE)

    return df


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    return model


try:

    predictions = load_predictions()
    model = load_model()

except Exception as error:

    st.error(f"Error loading project files: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Select Location")

locations = sorted(
    predictions["location"].unique()
)

selected_location = st.sidebar.selectbox(
    "Location",
    locations
)


# ============================================================
# SELECTED LOCATION DATA
# ============================================================

row = predictions[
    predictions["location"] == selected_location
].iloc[0]


recommended_model = row["recommended_model"]


# ============================================================
# MODEL NAME
# ============================================================

model_names = {

    "ecmwf_ifs": "ECMWF",

    "gfs_seamless": "GFS",

    "icon_seamless": "ICON",

    "gem_seamless": "GEM"
}


display_model = model_names.get(
    recommended_model,
    recommended_model
)


# ============================================================
# MAIN RESULT
# ============================================================

st.subheader(
    f"Weather Model Recommendation for {selected_location}"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Recommended Model",
        display_model
    )


with col2:

    st.metric(
        "Location",
        selected_location
    )


with col3:

    st.metric(
        "Prediction Status",
        "Successful"
    )


st.divider()


# ============================================================
# FORECAST INFORMATION
# ============================================================

st.subheader("Current Model Forecasts")

forecast_columns = {

    "ecmwf_temperature": "ECMWF",

    "gfs_temperature": "GFS",

    "icon_temperature": "ICON",

    "gem_temperature": "GEM"
}


available_forecasts = []

for column, name in forecast_columns.items():

    if column in row.index:

        try:

            temperature = float(row[column])

            available_forecasts.append(
                [name, temperature]
            )

        except:

            pass


if available_forecasts:

    cols = st.columns(len(available_forecasts))

    for col, (name, temperature) in zip(
        cols,
        available_forecasts
    ):

        with col:

            st.metric(
                name,
                f"{temperature:.1f} °C"
            )


# ============================================================
# RECOMMENDATION
# ============================================================

st.divider()

st.subheader("🏆 Recommended Weather Model")

st.success(
    f"### {display_model}\n\n"
    f"The ML model recommends **{display_model}** "
    f"for **{selected_location}**."
)


# ============================================================
# ALL LOCATION RESULTS
# ============================================================

st.divider()

st.subheader("All Location Predictions")

display_df = predictions[
    ["location", "recommended_model"]
].copy()

display_df["recommended_model"] = (
    display_df["recommended_model"]
    .map(model_names)
    .fillna(display_df["recommended_model"])
)

display_df.columns = [
    "Location",
    "Recommended Model"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL DISTRIBUTION
# ============================================================

st.divider()

st.subheader("Model Recommendation Distribution")

distribution = (
    predictions["recommended_model"]
    .map(model_names)
    .value_counts()
)

st.bar_chart(distribution)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("About the ML Model")

st.write(
    """
    The system uses a Random Forest classifier to select the
    most suitable weather forecast model.

    The candidate models are:

    • ECMWF
    • GFS
    • ICON
    • GEM

    Historical forecast performance was evaluated using ERA5
    reanalysis data as the reference dataset.

    The final model achieved 57.66% accuracy compared with
    a 39.22% baseline.
    """
)

st.caption(
    "Weather Model Selector | GCEK B.Tech IT Project"
)

# ============================================================
# AUTOMATIC HOURLY UPDATE
# ============================================================

UPDATE_INTERVAL = 3600  # 1 hour = 3600 seconds

if "last_update" not in st.session_state:
    st.session_state.last_update = 0

current_time = time.time()

if current_time - st.session_state.last_update >= UPDATE_INTERVAL:

    with st.spinner("Updating weather forecasts..."):

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    BASE_PATH,
                    "scripts",
                    "Step11_live_prediction.py"
                )
            ],
            check=True
        )

    st.session_state.last_update = current_time

    st.rerun()