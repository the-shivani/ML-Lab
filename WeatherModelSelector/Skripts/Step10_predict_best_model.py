import pandas as pd
import numpy as np
import os
import joblib


# ================================================================
# STEP 10 - REAL WEATHER MODEL PREDICTION
# ================================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")

CURRENT_FILE = os.path.join(
    DATA_PATH,
    "weather_data.csv"
)

HISTORICAL_FILE = os.path.join(
    DATA_PATH,
    "historical_forecast_data.csv"
)

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "weather_model_selector_v2.pkl"
)


print("=" * 70)
print("           WEATHER MODEL SELECTOR - STEP 10")
print("=" * 70)


# ================================================================
# 1. LOAD TRAINED MODEL
# ================================================================

print("\n[1/8] Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ================================================================
# 2. LOAD LOCATIONS
# ================================================================

print("\n[2/8] Loading locations...")

locations = pd.read_csv(LOCATIONS_FILE)

locations = locations.rename(
    columns={
        "City": "location",
        "Latitude": "latitude",
        "Longitude": "longitude"
    }
)

locations["location"] = locations["location"].astype(str)

print(
    "Locations loaded:",
    len(locations)
)


# ================================================================
# 3. ASK FOR LOCATION
# ================================================================

print("\nAvailable locations:")
print("-" * 40)

for location_name in locations["location"]:
    print(location_name)

print("-" * 40)

location_input = input(
    "\nEnter location name: "
).strip()


matches = locations[
    locations["location"].str.lower()
    == location_input.lower()
]

if len(matches) == 0:

    print("\nERROR: Location not found.")
    raise SystemExit


location_row = matches.iloc[0]

location = location_row["location"]

latitude = float(location_row["latitude"])

longitude = float(location_row["longitude"])


print("\nSelected location:", location)
print("Latitude:", latitude)
print("Longitude:", longitude)


# ================================================================
# 4. LOAD CURRENT FORECAST
# ================================================================

print("\n[3/8] Loading current forecast data...")

current = pd.read_csv(CURRENT_FILE)

current["datetime"] = pd.to_datetime(
    current["datetime"]
)

current_location = current[
    current["location"].str.lower()
    == location.lower()
].copy()


if len(current_location) == 0:

    print(
        "\nERROR: No current forecast data found."
    )

    raise SystemExit


prediction_time = current_location["datetime"].max()

current_at_time = current_location[
    current_location["datetime"]
    == prediction_time
].copy()


print(
    "Prediction time:",
    prediction_time
)


# ================================================================
# 5. GET CURRENT MODEL FORECASTS
# ================================================================

print("\n[4/8] Reading model forecasts...")


model_names = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


forecast_column_names = {
    "ecmwf_ifs": "ecmwf_temperature",
    "gfs_seamless": "gfs_temperature",
    "icon_seamless": "icon_temperature",
    "gem_seamless": "gem_temperature"
}


forecast_values = {}


for model_name in model_names:

    rows = current_at_time[
        current_at_time["model"]
        == model_name
    ]

    if len(rows) == 0:

        print(
            f"ERROR: No {model_name} forecast."
        )

        raise SystemExit

    temperature = float(
        rows.iloc[0]["temperature"]
    )

    column_name = forecast_column_names[
        model_name
    ]

    forecast_values[column_name] = temperature


print("\nCurrent forecasts:")

for column_name, value in forecast_values.items():

    print(
        f"{column_name:<25} {value:.2f} °C"
    )


# ================================================================
# 6. LOAD HISTORICAL DATA
# ================================================================

print("\n[5/8] Loading historical performance data...")

historical = pd.read_csv(
    HISTORICAL_FILE
)

print(
    "Historical rows loaded:",
    len(historical)
)

print(
    "Historical columns:",
    list(historical.columns)
)


historical["datetime"] = pd.to_datetime(
    historical["datetime"]
)


# ================================================================
# 7. FILTER LOCATION
# ================================================================

print(
    "\nFiltering historical data for:",
    location
)

historical_location = historical[
    historical["location"].astype(str).str.lower()
    == location.lower()
].copy()


print(
    "Historical rows for location:",
    len(historical_location)
)


if len(historical_location) == 0:

    print(
        "\nERROR: No historical data found for this location."
    )

    raise SystemExit


# ================================================================
# 8. SEPARATE REFERENCE DATA
# ================================================================

print("\n[6/8] Calculating historical model performance...")


reference = historical_location[
    historical_location["model"]
    == "ERA5_REFERENCE"
].copy()


forecast_history = historical_location[
    historical_location["model"]
    != "ERA5_REFERENCE"
].copy()


print(
    "Reference rows:",
    len(reference)
)

print(
    "Forecast rows:",
    len(forecast_history)
)


if len(reference) == 0:

    print(
        "\nERROR: ERA5 reference data not found."
    )

    raise SystemExit


# ================================================================
# 9. MATCH FORECAST WITH ERA5
# ================================================================

reference = reference[
    [
        "location",
        "datetime",
        "temperature"
    ]
].rename(
    columns={
        "temperature": "actual_temperature"
    }
)


performance = forecast_history.merge(
    reference,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


print(
    "Matched forecast/reference rows:",
    len(performance)
)


if len(performance) == 0:

    print(
        "\nERROR: Could not match forecasts with ERA5."
    )

    raise SystemExit


# ================================================================
# 10. CALCULATE ERROR
# ================================================================

performance["error"] = (
    performance["temperature"]
    - performance["actual_temperature"]
).abs()


performance = performance.sort_values(
    [
        "location",
        "model",
        "datetime"
    ]
).reset_index(drop=True)


# ================================================================
# 11. PREVIOUS ERROR
# ================================================================

performance["previous_error"] = (
    performance
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .shift(1)
)


# ================================================================
# 12. ROLLING ERRORS
# ================================================================

performance["rolling_error_3h"] = (
    performance
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .mean()
    )
)


performance["rolling_error_6h"] = (
    performance
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            6,
            min_periods=1
        )
        .mean()
    )
)


performance["rolling_error_12h"] = (
    performance
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            12,
            min_periods=1
        )
        .mean()
    )
)


# ================================================================
# 13. GET LATEST VALID PERFORMANCE
# ================================================================

print("\nFinding latest performance for each model...")


latest = {}


for model_name in model_names:

    rows = performance[
        performance["model"]
        == model_name
    ].copy()

    rows = rows.dropna(
        subset=[
            "previous_error",
            "rolling_error_3h",
            "rolling_error_6h",
            "rolling_error_12h"
        ]
    )

    if len(rows) == 0:

        print(
            f"ERROR: No valid performance history "
            f"for {model_name}"
        )

        raise SystemExit

    rows = rows.sort_values(
        "datetime"
    )

    latest[model_name] = rows.iloc[-1]


print("\nLatest historical performance:")

for model_name in model_names:

    row = latest[model_name]

    print(
        f"\n{model_name}"
    )

    print(
        f"  Previous error : "
        f"{row['previous_error']:.3f}"
    )

    print(
        f"  3-hour error   : "
        f"{row['rolling_error_3h']:.3f}"
    )

    print(
        f"  6-hour error   : "
        f"{row['rolling_error_6h']:.3f}"
    )

    print(
        f"  12-hour error  : "
        f"{row['rolling_error_12h']:.3f}"
    )


# ================================================================
# 14. FORECAST STATISTICS
# ================================================================

forecast_list = [
    forecast_values["ecmwf_temperature"],
    forecast_values["gfs_temperature"],
    forecast_values["icon_temperature"],
    forecast_values["gem_temperature"]
]


forecast_mean = np.mean(
    forecast_list
)

forecast_spread = (
    max(forecast_list)
    - min(forecast_list)
)


# ================================================================
# 15. TIME FEATURES
# ================================================================

hour = prediction_time.hour

day = prediction_time.day

day_of_week = prediction_time.dayofweek

month = prediction_time.month


# ================================================================
# 16. CREATE INPUT DATA
# ================================================================

print("\n[7/8] Creating prediction features...")


input_data = pd.DataFrame([{

    "latitude": latitude,
    "longitude": longitude,

    "ecmwf_temperature":
        forecast_values["ecmwf_temperature"],

    "gfs_temperature":
        forecast_values["gfs_temperature"],

    "icon_temperature":
        forecast_values["icon_temperature"],

    "gem_temperature":
        forecast_values["gem_temperature"],

    "forecast_mean":
        forecast_mean,

    "forecast_spread":
        forecast_spread,

    "hour":
        hour,

    "day":
        day,

    "day_of_week":
        day_of_week,

    "month":
        month,

    "ecmwf_previous_error":
        latest["ecmwf_ifs"]["previous_error"],

    "gfs_previous_error":
        latest["gfs_seamless"]["previous_error"],

    "icon_previous_error":
        latest["icon_seamless"]["previous_error"],

    "gem_previous_error":
        latest["gem_seamless"]["previous_error"],

    "ecmwf_rolling_error_3h":
        latest["ecmwf_ifs"]["rolling_error_3h"],

    "gfs_rolling_error_3h":
        latest["gfs_seamless"]["rolling_error_3h"],

    "icon_rolling_error_3h":
        latest["icon_seamless"]["rolling_error_3h"],

    "gem_rolling_error_3h":
        latest["gem_seamless"]["rolling_error_3h"],

    "ecmwf_rolling_error_6h":
        latest["ecmwf_ifs"]["rolling_error_6h"],

    "gfs_rolling_error_6h":
        latest["gfs_seamless"]["rolling_error_6h"],

    "icon_rolling_error_6h":
        latest["icon_seamless"]["rolling_error_6h"],

    "gem_rolling_error_6h":
        latest["gem_seamless"]["rolling_error_6h"],

    "ecmwf_rolling_error_12h":
        latest["ecmwf_ifs"]["rolling_error_12h"],

    "gfs_rolling_error_12h":
        latest["gfs_seamless"]["rolling_error_12h"],

    "icon_rolling_error_12h":
        latest["icon_seamless"]["rolling_error_12h"],

    "gem_rolling_error_12h":
        latest["gem_seamless"]["rolling_error_12h"]

}])


# ================================================================
# 17. EXACT FEATURE ORDER USED DURING TRAINING
# ================================================================

features = [

    "latitude",
    "longitude",

    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature",

    "forecast_mean",
    "forecast_spread",

    "hour",
    "day",
    "day_of_week",
    "month",

    "ecmwf_previous_error",
    "gfs_previous_error",
    "icon_previous_error",
    "gem_previous_error",

    "ecmwf_rolling_error_3h",
    "gfs_rolling_error_3h",
    "icon_rolling_error_3h",
    "gem_rolling_error_3h",

    "ecmwf_rolling_error_6h",
    "gfs_rolling_error_6h",
    "icon_rolling_error_6h",
    "gem_rolling_error_6h",

    "ecmwf_rolling_error_12h",
    "gfs_rolling_error_12h",
    "icon_rolling_error_12h",
    "gem_rolling_error_12h"
]


X = input_data[features]


# ================================================================
# 18. PREDICTION
# ================================================================

print("\n[8/8] Running ML prediction...")


prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

classes = model.classes_


results = sorted(
    zip(classes, probabilities),
    key=lambda x: x[1],
    reverse=True
)


# ================================================================
# FINAL RESULT
# ================================================================

print("\n")
print("=" * 70)
print("                    FINAL PREDICTION")
print("=" * 70)

print(
    f"\nLocation       : {location}"
)

print(
    f"Forecast time  : {prediction_time}"
)

print(
    "\nCurrent forecasts:"
)

for column_name, value in forecast_values.items():

    print(
        f"  {column_name:<23} {value:.2f} °C"
    )


print(
    "\nRecommended Weather Model:"
)

print(
    "\n       >>>",
    prediction.upper(),
    "<<<"
)


print(
    "\nPrediction probabilities:"
)

for model_name, probability in results:

    print(
        f"  {model_name:<18} "
        f"{probability * 100:6.2f}%"
    )


print("\n" + "=" * 70)
print("Prediction completed successfully.")
print("=" * 70)