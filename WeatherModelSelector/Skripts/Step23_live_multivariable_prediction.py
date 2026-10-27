import pandas as pd
import requests
import joblib
import os
import time

print("=" * 70)
print("STEP 23 - LIVE MULTIVARIABLE MODEL PREDICTION")
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
    "live_multivariable_predictions.csv"
)


# ---------------------------------------------------------
# 2. LOAD LOCATIONS
# ---------------------------------------------------------

print("\nLoading locations...")

locations = pd.read_csv(
    LOCATIONS_FILE
)

print(
    "Locations found:",
    len(locations)
)


# ---------------------------------------------------------
# 3. LOAD FIVE ML MODELS
# ---------------------------------------------------------

print("\nLoading ML models...")

models = {}

parameters = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

for parameter in parameters:

    model_file = os.path.join(
        MODEL_PATH,
        f"{parameter}_selector.pkl"
    )

    models[parameter] = joblib.load(
        model_file
    )

    print(
        f"{parameter:15s} -> loaded"
    )


# ---------------------------------------------------------
# 4. MODEL NAMES
# ---------------------------------------------------------

weather_models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


# ---------------------------------------------------------
# 5. GET LIVE FORECAST
# ---------------------------------------------------------

def get_forecast(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "pressure_msl"
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
                f"    API attempt {attempt}/3 failed: "
                f"{error}"
            )

            if attempt < 3:

                time.sleep(5)

            else:

                raise


# ---------------------------------------------------------
# 6. CREATE ONE PREDICTION
# ---------------------------------------------------------

def create_prediction(
    latitude,
    longitude,
    forecast
):

    times = pd.to_datetime(
        forecast["hourly"]["time"]
    )

    # Use the first available forecast hour
    index = 0

    current_time = times[index]

    hour = current_time.hour
    day = current_time.day
    day_of_week = current_time.dayofweek
    month = current_time.month

    row = {
        "latitude": latitude,
        "longitude": longitude,
        "hour": hour,
        "day": day,
        "day_of_week": day_of_week,
        "month": month
    }


    # -----------------------------------------------------
    # GET FOUR MODEL VALUES
    # -----------------------------------------------------

    for parameter in parameters:

        for model_name in weather_models:

            if parameter == "temperature":

                key = (
                    f"temperature_2m_{model_name}"
                )

            elif parameter == "humidity":

                key = (
                    f"relative_humidity_2m_{model_name}"
                )

            elif parameter == "precipitation":

                key = (
                    f"precipitation_{model_name}"
                )

            elif parameter == "wind_speed":

                key = (
                    f"wind_speed_10m_{model_name}"
                )

            elif parameter == "pressure":

                key = (
                    f"pressure_msl_{model_name}"
                )

            value = forecast["hourly"][key][index]

            row[
                f"{model_name}_{parameter}"
            ] = value


        # -------------------------------------------------
        # FORECAST MEAN
        # -------------------------------------------------

        values = [
            row[f"{model}_{parameter}"]
            for model in weather_models
        ]

        row[
            f"{parameter}_forecast_mean"
        ] = sum(values) / len(values)

        row[
            f"{parameter}_forecast_spread"
        ] = max(values) - min(values)


    return row


# ---------------------------------------------------------
# 7. FEATURE ORDER
# ---------------------------------------------------------

features = [
    "latitude",
    "longitude",
    "hour",
    "day",
    "day_of_week",
    "month"
]

for parameter in parameters:

    for model_name in weather_models:

        features.append(
            f"{model_name}_{parameter}"
        )

    features.append(
        f"{parameter}_forecast_mean"
    )

    features.append(
        f"{parameter}_forecast_spread"
    )


# ---------------------------------------------------------
# 8. PROCESS LOCATIONS
# ---------------------------------------------------------

results = []

print("\nStarting live predictions...\n")


for number, location in locations.iterrows():

    city = location["City"]

    latitude = location["Latitude"]
    longitude = location["Longitude"]

    print(
        f"[{number + 1}/{len(locations)}] {city}"
    )

    try:

        forecast = get_forecast(
            latitude,
            longitude
        )

        prediction_row = create_prediction(
            latitude,
            longitude,
            forecast
        )

        prediction_row["location"] = city
        prediction_row["datetime"] = (
            forecast["hourly"]["time"][0]
        )


        # -------------------------------------------------
        # ML PREDICTION FOR EACH PARAMETER
        # -------------------------------------------------

        for parameter in parameters:

            X = pd.DataFrame(
                [prediction_row]
            )[features]

            prediction = models[
                parameter
            ].predict(X)[0]

            probabilities = models[
                parameter
            ].predict_proba(X)[0]

            classes = models[
                parameter
            ].classes_

            prediction_row[
                f"recommended_{parameter}_model"
            ] = prediction


            # Save probabilities
            for model_name in weather_models:

                probability = 0.0

                for class_name, value in zip(
                    classes,
                    probabilities
                ):

                    if class_name == model_name:

                        probability = value

                prediction_row[
                    f"{parameter}_{model_name}_probability"
                ] = probability


        results.append(
            prediction_row
        )

        print(
            "    Temperature :",
            prediction_row[
                "recommended_temperature_model"
            ]
        )

        print(
            "    Humidity    :",
            prediction_row[
                "recommended_humidity_model"
            ]
        )

        print(
            "    Precipitation:",
            prediction_row[
                "recommended_precipitation_model"
            ]
        )

        print(
            "    Wind Speed  :",
            prediction_row[
                "recommended_wind_speed_model"
            ]
        )

        print(
            "    Pressure    :",
            prediction_row[
                "recommended_pressure_model"
            ]
        )

    except Exception as error:

        print(
            "    ERROR:",
            error
        )


# ---------------------------------------------------------
# 9. SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 10. SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("LIVE MULTIVARIABLE PREDICTION COMPLETE")
print("=" * 70)

print(
    "Locations processed:",
    len(results_df)
)

print(
    "Total columns:",
    len(results_df.columns)
)

print("\nSaved to:")

print(OUTPUT_FILE)


# ---------------------------------------------------------
# 11. MODEL DISTRIBUTIONS
# ---------------------------------------------------------

for parameter in parameters:

    column = (
        f"recommended_{parameter}_model"
    )

    print(
        f"\n{parameter.upper()} RECOMMENDATIONS:"
    )

    print(
        results_df[column].value_counts()
    )


print("\n" + "=" * 70)