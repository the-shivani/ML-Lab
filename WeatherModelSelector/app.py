import streamlit as st
import pandas as pd
import os

# ============================================================
# WEATHER MODEL SELECTOR DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Weather Model Selector",
    page_icon="🌦️",
    layout="wide"
)

BASE_PATH = r"C:\Projects\WeatherModelSelector"
RESULTS_PATH = os.path.join(BASE_PATH, "results")

LIVE_FILE = os.path.join(
    RESULTS_PATH,
    "live_improved_predictions.csv"
)

PRESSURE_FILE = os.path.join(
    RESULTS_PATH,
    "live_corrected_predictions.csv"
)

# ============================================================
# LOAD MAIN DATA
# ============================================================

if not os.path.exists(LIVE_FILE):
    st.error("Live prediction file not found.")
    st.stop()

df = pd.read_csv(LIVE_FILE)

# ============================================================
# LOAD CORRECTED PRESSURE DATA
# ============================================================

if os.path.exists(PRESSURE_FILE):

    pressure = pd.read_csv(PRESSURE_FILE)

    pressure_columns = [
        "location",
        "ecmwf_ifs_pressure",
        "gfs_seamless_pressure",
        "icon_seamless_pressure",
        "gem_seamless_pressure",
        "best_pressure_model"
    ]

    pressure = pressure[pressure_columns]

    # Remove old pressure columns if present
    old_pressure_columns = [
        c for c in pressure_columns
        if c != "location" and c in df.columns
    ]

    df = df.drop(
        columns=old_pressure_columns,
        errors="ignore"
    )

    df = df.merge(
        pressure,
        on="location",
        how="left"
    )

# ============================================================
# HEADER
# ============================================================

st.title("🌦️ Weather Model Selector")

st.markdown(
    "### AI-Based Weather Forecast Model Selection"
)

st.write(
    "Compare ECMWF, GFS, ICON and GEM and let the ML model "
    "select the best forecast model."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

# Location selector
locations = sorted(
    df["location"].dropna().unique()
)

selected_location = st.sidebar.selectbox(
    "📍 Select Location",
    locations
)

# Parameter selector
parameters = {
    "🌡️ Temperature": "temperature",
    "💧 Relative Humidity": "humidity",
    "🌧️ Precipitation": "precipitation",
    "💨 Wind Speed": "wind_speed",
    "📊 Pressure": "pressure"
}

selected_parameter_name = st.sidebar.selectbox(
    "🌦️ Select Parameter",
    list(parameters.keys())
)

selected_parameter = parameters[
    selected_parameter_name
]

# ============================================================
# GET SELECTED LOCATION
# ============================================================

location_data = df[
    df["location"] == selected_location
].copy()

if location_data.empty:
    st.error("No data available for this location.")
    st.stop()

row = location_data.iloc[0]

# ============================================================
# LOCATION INFORMATION
# ============================================================

st.header(f"📍 {selected_location}")

if "forecast_datetime" in row:

    st.caption(
        f"Forecast time: {row['forecast_datetime']}"
    )

st.divider()

# ============================================================
# PARAMETER CONFIGURATION
# ============================================================

parameter_config = {

    "temperature": {
        "title": "🌡️ Temperature",
        "unit": "°C",
        "columns": {
            "ECMWF": "ecmwf_ifs_temperature",
            "GFS": "gfs_seamless_temperature",
            "ICON": "icon_seamless_temperature",
            "GEM": "gem_seamless_temperature"
        },
        "best_column": "best_temperature_model"
    },

    "humidity": {
        "title": "💧 Relative Humidity",
        "unit": "%",
        "columns": {
            "ECMWF": "ecmwf_ifs_humidity",
            "GFS": "gfs_seamless_humidity",
            "ICON": "icon_seamless_humidity",
            "GEM": "gem_seamless_humidity"
        },
        "best_column": "best_humidity_model"
    },

    "precipitation": {
        "title": "🌧️ Precipitation",
        "unit": "mm",
        "columns": {
            "ECMWF": "ecmwf_ifs_precipitation",
            "GFS": "gfs_seamless_precipitation",
            "ICON": "icon_seamless_precipitation",
            "GEM": "gem_seamless_precipitation"
        },
        "best_column": "best_precipitation_model"
    },

    "wind_speed": {
        "title": "💨 Wind Speed",
        "unit": "km/h",
        "columns": {
            "ECMWF": "ecmwf_ifs_wind_speed",
            "GFS": "gfs_seamless_wind_speed",
            "ICON": "icon_seamless_wind_speed",
            "GEM": "gem_seamless_wind_speed"
        },
        "best_column": "best_wind_speed_model"
    },

    "pressure": {
        "title": "📊 Mean Sea Level Pressure",
        "unit": "hPa",
        "columns": {
            "ECMWF": "ecmwf_ifs_pressure",
            "GFS": "gfs_seamless_pressure",
            "ICON": "icon_seamless_pressure",
            "GEM": "gem_seamless_pressure"
        },
        "best_column": "best_pressure_model"
    }
}

config = parameter_config[selected_parameter]

# ============================================================
# PARAMETER TITLE
# ============================================================

st.subheader(config["title"])

# ============================================================
# MODEL VALUES
# ============================================================

model_columns = config["columns"]

cols = st.columns(4)

for col, (model_name, column_name) in zip(
    cols,
    model_columns.items()
):

    if column_name in row:

        value = row[column_name]

        if pd.notna(value):

            col.metric(
                model_name,
                f"{value:.2f} {config['unit']}"
            )

        else:

            col.metric(
                model_name,
                "N/A"
            )

# ============================================================
# ML RECOMMENDATION
# ============================================================

st.divider()

best_column = config["best_column"]

if best_column in row:

    best_model = row[best_column]

    model_display_names = {
        "ecmwf_ifs": "ECMWF",
        "gfs_seamless": "GFS",
        "icon_seamless": "ICON",
        "gem_seamless": "GEM"
    }

    best_display = model_display_names.get(
        best_model,
        best_model
    )

    st.success(
        f"🤖 ML Recommended Model: **{best_display}**"
    )

else:

    st.warning(
        "ML recommendation is not available."
    )

# ============================================================
# FORECAST COMPARISON TABLE
# ============================================================

st.divider()

st.subheader("📋 Model Comparison")

comparison_data = []

for model_name, column_name in model_columns.items():

    if column_name in row:

        value = row[column_name]

        comparison_data.append({
            "Model": model_name,
            "Forecast": (
                f"{value:.2f} {config['unit']}"
                if pd.notna(value)
                else "N/A"
            )
        })

comparison_df = pd.DataFrame(
    comparison_data
)

st.table(comparison_df)

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ About the Prediction")

st.write(
    "The Machine Learning model analyzes the current forecasts "
    "from ECMWF, GFS, ICON and GEM along with historical model "
    "error patterns to select the model expected to perform best "
    "for the selected weather parameter."
)

if selected_parameter == "pressure":

    st.info(
        "Pressure uses Mean Sea Level Pressure (MSLP), "
        "which is aligned with the IMD pressure observation."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weather Model Selector | ECMWF • GFS • ICON • GEM | "
    "Machine Learning"
)