import pandas as pd
import numpy as np
import os
import joblib
import requests
import time


print("=" * 70)
print("STEP 27 - LIVE PREDICTION USING IMPROVED ML MODELS")
print("=" * 70)


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_improved_predictions.csv"
)


# ---------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------

parameters = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


model_files = {
    "temperature":
        "temperature_selector_v2.pkl",

    "humidity":
        "humidity_selector_v2.pkl",

    "precipitation":
        "precipitation_selector_v2.pkl",

    "wind_speed":
        "wind_speed_selector_v2.pkl",

    "pressure":
        "pressure_selector_v2.pkl"
}


# ---------------------------------------------------------
# 3. LOAD ML MODELS
# ---------------------------------------------------------

print("\nLoading improved ML models...")

loaded_models = {}

for parameter in parameters:

    file_path = os.path.join(
        MODEL_PATH,
        model_files[parameter]
    )

    loaded_models[parameter] = joblib.load(
        file_path
    )

    print(
        parameter,
        "-> loaded"
    )


# ---------------------------------------------------------
# 4. LOAD LOCATIONS
# ---------------------------------------------------------

print("\nLoading locations...")

locations = pd.read_csv(
    LOCATIONS_FILE
)

print(
    "Total locations:",
    len(locations)
)


# ---------------------------------------------------------
# 5. OPEN-METEO API
# ---------------------------------------------------------

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ---------------------------------------------------------
# 6. PROCESS LOCATIONS
# ---------------------------------------------------------

all_results = []


for index, row in locations.iterrows():

    location = row["City"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]


    print("\n" + "-" * 70)

    print(
        f"{index + 1}/{len(locations)} "
        f"- {location}"
    )


    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "surface_pressure"
        ),

        "models": (
            "ecmwf_ifs,"
            "gfs_seamless,"
            "icon_seamless,"
            "gem_seamless"
        ),

        "forecast_days": 1,

        "timezone": "auto"
    }


    # -----------------------------------------------------
    # API request with retry
    # -----------------------------------------------------

    data = None


    for attempt in range(3):

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            break

        except Exception as e:

            print(
                f"API attempt {attempt + 1} failed:",
                e
            )

            time.sleep(2)


    if data is None:

        print(
            "Skipping:",
            location
        )

        continue


    # -----------------------------------------------------
    # Current forecast index
    # -----------------------------------------------------

    index_hour = 0


    # -----------------------------------------------------
    # Create result row
    # -----------------------------------------------------

    result = {

        "location": location,

        "latitude": latitude,

        "longitude": longitude,

        "forecast_datetime":
            data["hourly"]["time"][index_hour]
    }


    # -----------------------------------------------------
    # Store all model forecasts
    # -----------------------------------------------------

    for parameter in parameters:

        print(
            f"\n{parameter.upper()}"
        )


        feature_values = {}


        for model in models:

            # Open-Meteo model-specific columns
            if model == "ecmwf_ifs":
                prefix = "ecmwf_ifs"

            elif model == "gfs_seamless":
                prefix = "gfs_seamless"

            elif model == "icon_seamless":
                prefix = "icon_seamless"

            else:
                prefix = "gem_seamless"


            if parameter == "temperature":
                variable = "temperature_2m"

            elif parameter == "humidity":
                variable = "relative_humidity_2m"

            elif parameter == "precipitation":
                variable = "precipitation"

            elif parameter == "wind_speed":
                variable = "wind_speed_10m"

            else:
                variable = "surface_pressure"


            key = (
                f"{variable}_{prefix}"
            )


            if key in data["hourly"]:

                value = data[
                    "hourly"
                ][key][index_hour]

            else:

                value = np.nan


            feature_values[
                f"{model}_{parameter}"
            ] = value


            result[
                f"{model}_{parameter}"
            ] = value


        # -------------------------------------------------
        # Check missing values
        # -------------------------------------------------

        if any(
            pd.isna(value)
            for value in feature_values.values()
        ):

            print(
                "Missing forecast value - skipping parameter"
            )

            continue


        # -------------------------------------------------
        # Forecast statistics
        # -------------------------------------------------

        values = list(
            feature_values.values()
        )


        forecast_mean = np.mean(values)

        forecast_spread = (
            np.max(values)
            -
            np.min(values)
        )


        result[
            f"{parameter}_forecast_mean"
        ] = forecast_mean


        result[
            f"{parameter}_forecast_spread"
        ] = forecast_spread


        # -------------------------------------------------
        # Historical error features
        #
        # We do NOT have the actual current IMD value yet.
        #
        # Therefore use the latest historical error
        # information from the training dataset.
        # -------------------------------------------------

        history_file = os.path.join(
            DATA_PATH,
            "multivariable_training_with_history.csv"
        )


        history = pd.read_csv(
            history_file
        )


        history["datetime"] = pd.to_datetime(
            history["datetime"]
        )


        location_history = history[
            history["location"] == location
        ].sort_values(
            "datetime"
        )


        # Latest historical row

        if len(location_history) == 0:

            print(
                "No historical data for",
                location
            )

            continue


        latest_history = (
            location_history.iloc[-1]
        )


        for model in models:

            result[
                f"{model}_previous_{parameter}_error"
            ] = latest_history[
                f"{model}_previous_{parameter}_error"
            ]


            result[
                f"{model}_rolling_3h_{parameter}_error"
            ] = latest_history[
                f"{model}_rolling_3h_{parameter}_error"
            ]


            result[
                f"{model}_rolling_6h_{parameter}_error"
            ] = latest_history[
                f"{model}_rolling_6h_{parameter}_error"
            ]


            result[
                f"{model}_rolling_12h_{parameter}_error"
            ] = latest_history[
                f"{model}_rolling_12h_{parameter}_error"
            ]


        # -------------------------------------------------
        # Build feature row
        # -------------------------------------------------

        feature_row = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "hour":
                pd.to_datetime(
                    result["forecast_datetime"]
                ).hour,

            "day":
                pd.to_datetime(
                    result["forecast_datetime"]
                ).day,

            "day_of_week":
                pd.to_datetime(
                    result["forecast_datetime"]
                ).dayofweek,

            "month":
                pd.to_datetime(
                    result["forecast_datetime"]
                ).month,

            f"{models[0]}_{parameter}":
                result[
                    f"{models[0]}_{parameter}"
                ],

            f"{models[1]}_{parameter}":
                result[
                    f"{models[1]}_{parameter}"
                ],

            f"{models[2]}_{parameter}":
                result[
                    f"{models[2]}_{parameter}"
                ],

            f"{models[3]}_{parameter}":
                result[
                    f"{models[3]}_{parameter}"
                ],

            f"{parameter}_forecast_mean":
                forecast_mean,

            f"{parameter}_forecast_spread":
                forecast_spread
        }


        for model_name in models:

            for window in [
                "previous",
                "rolling_3h",
                "rolling_6h",
                "rolling_12h"
            ]:

                column = (
                    f"{model_name}_"
                    f"{window}_"
                    f"{parameter}_error"
                )


                feature_row[column] = result[
                    column
                ]


        # -------------------------------------------------
        # Predict best model
        # -------------------------------------------------

        package = loaded_models[
            parameter
        ]

        ml_model = package["model"]

        feature_names = package["features"]


        X = pd.DataFrame(
            [feature_row]
        )


        X = X[
            feature_names
        ]


        prediction = ml_model.predict(X)[0]


        # -------------------------------------------------
        # Store prediction
        # -------------------------------------------------

        result[
            f"best_{parameter}_model"
        ] = prediction


        print(
            "ML selected:",
            prediction
        )


    all_results.append(result)


# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 8. SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 27 COMPLETE")
print("=" * 70)

print(
    "Locations processed:",
    len(results_df)
)


for parameter in parameters:

    column = (
        f"best_{parameter}_model"
    )

    if column in results_df:

        print(
            f"\n{parameter.upper()} "
            "SELECTION DISTRIBUTION:"
        )

        print(
            results_df[
                column
            ].value_counts()
        )


print("\nSaved to:")

print(OUTPUT_FILE)