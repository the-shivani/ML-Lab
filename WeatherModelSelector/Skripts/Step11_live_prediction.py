import pandas as pd
import numpy as np
import requests
import os
import joblib


# ================================================================
# STEP 11 - AUTOMATIC LIVE WEATHER MODEL PREDICTION
# ================================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

HISTORICAL_FILE = os.path.join(
    DATA_PATH,
    "historical_forecast_data.csv"
)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "weather_model_selector_v2.pkl"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_predictions.csv"
)


# ================================================================
# DISPLAY
# ================================================================

print("=" * 75)
print("             WEATHER MODEL SELECTOR - STEP 11")
print("=" * 75)


# ================================================================
# 1. LOAD MODEL
# ================================================================

print("\n[1/7] Loading trained ML model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ================================================================
# 2. LOAD LOCATIONS
# ================================================================

print("\n[2/7] Loading locations...")

locations = pd.read_csv(LOCATIONS_FILE)

locations = locations.rename(
    columns={
        "City": "location",
        "Latitude": "latitude",
        "Longitude": "longitude"
    }
)

print(
    "Total locations:",
    len(locations)
)


# ================================================================
# 3. LOAD HISTORICAL DATA
# ================================================================

print("\n[3/7] Loading historical performance...")

historical = pd.read_csv(
    HISTORICAL_FILE
)

historical["datetime"] = pd.to_datetime(
    historical["datetime"]
)

print(
    "Historical rows:",
    len(historical)
)


# ================================================================
# 4. CREATE HISTORICAL PERFORMANCE FEATURES
# ================================================================

print("\n[4/7] Creating historical performance features...")


reference = historical[
    historical["model"]
    == "ERA5_REFERENCE"
].copy()


reference = reference[
    [
        "location",
        "datetime",
        "temperature"
    ]
].rename(
    columns={
        "temperature":
        "actual_temperature"
    }
)


forecast_history = historical[
    historical["model"]
    != "ERA5_REFERENCE"
].copy()


performance = forecast_history.merge(
    reference,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


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
)


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
# 5. GET LATEST PERFORMANCE FOR EACH LOCATION
# ================================================================

model_names = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


latest_performance = {}


for location in locations["location"]:

    latest_performance[location] = {}

    location_data = performance[
        performance["location"].str.lower()
        == location.lower()
    ]

    for model_name in model_names:

        rows = location_data[
            location_data["model"]
            == model_name
        ].dropna(
            subset=[
                "previous_error",
                "rolling_error_3h",
                "rolling_error_6h",
                "rolling_error_12h"
            ]
        )

        if len(rows) == 0:
            continue

        latest_row = rows.sort_values(
            "datetime"
        ).iloc[-1]

        latest_performance[location][
            model_name
        ] = latest_row


# ================================================================
# 6. FETCH CURRENT FORECAST
# ================================================================

print("\n[5/7] Fetching latest weather forecasts...")


def get_forecast(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "models":
            "ecmwf_ifs,gfs_seamless,"
            "icon_seamless,gem_seamless",
        "forecast_days": 1,
        "timezone": "auto"
    }

    for attempt in range(1, 4):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as error:

            print(
                f"  API attempt {attempt}/3 failed: "
                f"{error}"
            )

            if attempt < 3:

                import time

                print(
                    "  Waiting 5 seconds before retry..."
                )

                time.sleep(5)

            else:

                raise error

# ================================================================
# 7. PREDICT FOR EVERY LOCATION
# ================================================================

results = []


for index, row in locations.iterrows():

    location = row["location"]

    latitude = float(
        row["latitude"]
    )

    longitude = float(
        row["longitude"]
    )

    print(
        f"\nProcessing {index + 1}/"
        f"{len(locations)}: {location}"
    )


    # ------------------------------------------------------------
    # Fetch forecast
    # ------------------------------------------------------------

    try:

        data = get_forecast(
            latitude,
            longitude
        )

    except Exception as error:

        print(
            "  Forecast error:",
            error
        )

        continue


    hourly = data.get(
        "hourly",
        {}
    )


    times = hourly.get(
        "time",
        []
    )


    if len(times) == 0:

        print(
            "  No forecast data."
        )

        continue


    # Use first available forecast hour
    prediction_time = pd.to_datetime(
        times[0]
    )


    # ------------------------------------------------------------
    # Get temperatures
    # ------------------------------------------------------------

    forecast = {}


    api_model_columns = {
        "ecmwf_ifs":
            "temperature_2m_ecmwf_ifs",

        "gfs_seamless":
            "temperature_2m_gfs_seamless",

        "icon_seamless":
            "temperature_2m_icon_seamless",

        "gem_seamless":
            "temperature_2m_gem_seamless"
    }


    missing_model = False


    for model_name in model_names:

        column = api_model_columns[
            model_name
        ]

        values = hourly.get(
            column,
            []
        )

        if len(values) == 0:

            print(
                f"  Missing {model_name}"
            )

            missing_model = True

            break

        forecast[model_name] = float(
            values[0]
        )


    if missing_model:

        continue


    # ------------------------------------------------------------
    # Historical performance
    # ------------------------------------------------------------

    location_performance = (
        latest_performance.get(
            location,
            {}
        )
    )


    if not all(
        model_name in location_performance
        for model_name in model_names
    ):

        print(
            "  Missing historical performance."
        )

        continue


    # ------------------------------------------------------------
    # Forecast statistics
    # ------------------------------------------------------------

    forecast_values = list(
        forecast.values()
    )


    forecast_mean = np.mean(
        forecast_values
    )


    forecast_spread = (
        max(forecast_values)
        -
        min(forecast_values)
    )


    # ------------------------------------------------------------
    # Create ML input
    # ------------------------------------------------------------

    input_data = pd.DataFrame([{

        "latitude":
            latitude,

        "longitude":
            longitude,

        "ecmwf_temperature":
            forecast["ecmwf_ifs"],

        "gfs_temperature":
            forecast["gfs_seamless"],

        "icon_temperature":
            forecast["icon_seamless"],

        "gem_temperature":
            forecast["gem_seamless"],

        "forecast_mean":
            forecast_mean,

        "forecast_spread":
            forecast_spread,

        "hour":
            prediction_time.hour,

        "day":
            prediction_time.day,

        "day_of_week":
            prediction_time.dayofweek,

        "month":
            prediction_time.month,

        "ecmwf_previous_error":
            location_performance[
                "ecmwf_ifs"
            ]["previous_error"],

        "gfs_previous_error":
            location_performance[
                "gfs_seamless"
            ]["previous_error"],

        "icon_previous_error":
            location_performance[
                "icon_seamless"
            ]["previous_error"],

        "gem_previous_error":
            location_performance[
                "gem_seamless"
            ]["previous_error"],

        "ecmwf_rolling_error_3h":
            location_performance[
                "ecmwf_ifs"
            ]["rolling_error_3h"],

        "gfs_rolling_error_3h":
            location_performance[
                "gfs_seamless"
            ]["rolling_error_3h"],

        "icon_rolling_error_3h":
            location_performance[
                "icon_seamless"
            ]["rolling_error_3h"],

        "gem_rolling_error_3h":
            location_performance[
                "gem_seamless"
            ]["rolling_error_3h"],

        "ecmwf_rolling_error_6h":
            location_performance[
                "ecmwf_ifs"
            ]["rolling_error_6h"],

        "gfs_rolling_error_6h":
            location_performance[
                "gfs_seamless"
            ]["rolling_error_6h"],

        "icon_rolling_error_6h":
            location_performance[
                "icon_seamless"
            ]["rolling_error_6h"],

        "gem_rolling_error_6h":
            location_performance[
                "gem_seamless"
            ]["rolling_error_6h"],

        "ecmwf_rolling_error_12h":
            location_performance[
                "ecmwf_ifs"
            ]["rolling_error_12h"],

        "gfs_rolling_error_12h":
            location_performance[
                "gfs_seamless"
            ]["rolling_error_12h"],

        "icon_rolling_error_12h":
            location_performance[
                "icon_seamless"
            ]["rolling_error_12h"],

        "gem_rolling_error_12h":
            location_performance[
                "gem_seamless"
            ]["rolling_error_12h"]

    }])


    # ------------------------------------------------------------
    # Exact training feature order
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_


    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )


    # ------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------

    results.append({

        "location":
            location,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "datetime":
            prediction_time,

        "ecmwf_temperature":
            forecast["ecmwf_ifs"],

        "gfs_temperature":
            forecast["gfs_seamless"],

        "icon_temperature":
            forecast["icon_seamless"],

        "gem_temperature":
            forecast["gem_seamless"],

        "recommended_model":
            prediction,

        "ecmwf_probability":
            probability_dict.get(
                "ecmwf_ifs",
                0
            ),

        "gfs_probability":
            probability_dict.get(
                "gfs_seamless",
                0
            ),

        "icon_probability":
            probability_dict.get(
                "icon_seamless",
                0
            ),

        "gem_probability":
            probability_dict.get(
                "gem_seamless",
                0
            )
    })


# ================================================================
# 8. SAVE RESULTS
# ================================================================

print("\n[6/7] Saving prediction results...")


results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ================================================================
# 9. DISPLAY SUMMARY
# ================================================================

print("\n[7/7] Prediction summary")

print("=" * 75)

print(
    f"\nSuccessful predictions: "
    f"{len(results_df)} / "
    f"{len(locations)}"
)


if len(results_df) > 0:

    print(
        "\nRecommended model distribution:"
    )

    print(
        results_df[
            "recommended_model"
        ].value_counts()
    )


    print("\nResults:")

    print(
        results_df[
            [
                "location",
                "recommended_model"
            ]
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 75)

print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 75)