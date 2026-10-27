import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

BASE = r"C:\Projects\WeatherModelSelector"

DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

os.makedirs(RESULTS, exist_ok=True)

API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

PARAMETERS = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "precipitation": "precipitation",
    "wind_speed": "wind_speed_10m",
    "pressure": "pressure_msl"
}

OUTPUT_FILE = os.path.join(
    RESULTS, "step49B_live_31locations_predictions.csv"
)

ERROR_FILE = os.path.join(
    RESULTS, "step49B_fetch_errors.csv"
)

print("=" * 70)
print("STEP 49B - LIVE WEATHER MODEL SELECTION FOR 31 LOCATIONS")
print("=" * 70)

# Load exact locations from training dataset
location_file = os.path.join(
    DATA,
    "historical_multivariable_forecast_12months_previous_day1.csv"
)

locations = pd.read_csv(
    location_file,
    usecols=["location", "latitude", "longitude"]
)

locations = (
    locations
    .drop_duplicates(subset=["location"])
    .sort_values("location")
    .reset_index(drop=True)
)

print("Total locations:", len(locations))

# Load trained ML models once
saved_models = {}

for parameter in PARAMETERS:

    model_path = os.path.join(
        MODELS_DIR,
        f"{parameter}_forecast_only_selector_v5.pkl"
    )

    saved_models[parameter] = joblib.load(model_path)

    print("Loaded ML selector:", parameter)

session = requests.Session()

def fetch_forecast(model_name, latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(PARAMETERS.values()),
        "models": model_name,
        "timezone": "Asia/Kolkata",
        "forecast_days": 2,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    last_error = None

    for attempt in range(4):

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=60
            )

            if response.status_code == 429 or (
                500 <= response.status_code < 600
            ):
                response.raise_for_status()

            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                raise ValueError(
                    f"Missing hourly data: {data}"
                )

            hourly = pd.DataFrame(data["hourly"])

            hourly["datetime"] = pd.to_datetime(
                hourly["time"]
            )

            hourly = (
                hourly.drop(columns=["time"])
                .set_index("datetime")
            )

            return hourly

        except (
            requests.RequestException,
            ValueError
        ) as error:

            last_error = error

            # Retry temporary failures only
            retryable = (
                isinstance(error, requests.Timeout)
                or isinstance(error, requests.ConnectionError)
                or (
                    isinstance(error, requests.HTTPError)
                    and error.response is not None
                    and (
                        error.response.status_code == 429
                        or error.response.status_code >= 500
                    )
                )
            )

            if not retryable or attempt == 3:
                break

            delay = 3 * (2 ** attempt)

            print(
                f"Retrying {model_name} in {delay}s..."
            )

            time.sleep(delay)

    raise RuntimeError(str(last_error))


def predict_location(location, latitude, longitude):

    forecasts = {}

    for model_name in WEATHER_MODELS:

        forecasts[model_name] = fetch_forecast(
            model_name, latitude, longitude
        )

        print("  Fetched:", model_name)

        time.sleep(0.5)

    common_times = forecasts[
        WEATHER_MODELS[0]
    ].index

    for model_name in WEATHER_MODELS[1:]:

        common_times = common_times.intersection(
            forecasts[model_name].index
        )

    current_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).replace(tzinfo=None)

    common_times = common_times[
        common_times > pd.Timestamp(current_time)
    ].sort_values()

    if len(common_times) == 0:
        raise ValueError(
            "No common future forecast times found"
        )

    location_results = []

    for parameter, api_variable in PARAMETERS.items():

        saved = saved_models[parameter]

        feature_columns = saved["features"]
        regressors = saved["models"]
        threshold = saved["threshold"]

        frame = pd.DataFrame({
            "datetime": common_times
        })

        frame["location"] = location
        frame["latitude"] = latitude
        frame["longitude"] = longitude

        frame["hour"] = frame["datetime"].dt.hour

        frame["day_of_week"] = (
            frame["datetime"].dt.dayofweek
        )

        frame["month"] = frame["datetime"].dt.month

        model_columns = []

        for model_name in WEATHER_MODELS:

            column = f"{model_name}_{parameter}"

            values = forecasts[model_name][
                api_variable
            ].reindex(common_times)

            frame[column] = values.to_numpy()

            model_columns.append(column)

        frame["forecast_mean"] = frame[
            model_columns
        ].mean(axis=1)

        frame["forecast_spread"] = (
            frame[model_columns].max(axis=1)
            - frame[model_columns].min(axis=1)
        )

        frame = frame.dropna(
            subset=feature_columns
        ).copy()

        if frame.empty:
            print("  No complete data for:", parameter)
            continue

        predicted_errors = np.column_stack([
            regressors[model_name].predict(
                frame[feature_columns]
            )
            for model_name in WEATHER_MODELS
        ])

        selected_indices = np.argmin(
            predicted_errors,
            axis=1
        )

        ecmwf_index = WEATHER_MODELS.index(
            "ecmwf_ifs"
        )

        predicted_gain = (
            predicted_errors[:, ecmwf_index]
            - predicted_errors.min(axis=1)
        )

        selected_indices = np.where(
            predicted_gain > threshold,
            selected_indices,
            ecmwf_index
        )

        forecast_matrix = frame[
            model_columns
        ].to_numpy()

        selected_values = forecast_matrix[
            np.arange(len(frame)),
            selected_indices
        ]

        selected_errors = predicted_errors[
            np.arange(len(frame)),
            selected_indices
        ]

        selected_models = np.array(
            WEATHER_MODELS
        )[selected_indices]

        for i in range(len(frame)):

            record = {
                "location": location,
                "latitude": latitude,
                "longitude": longitude,
                "datetime": frame.iloc[i]["datetime"],
                "parameter": parameter,
                "selected_model": selected_models[i],
                "predicted_value": selected_values[i],
                "predicted_error": selected_errors[i],
                "ecmwf_value": forecast_matrix[
                    i, ecmwf_index
                ],
                "switch_threshold": threshold
            }

            # Save all four raw model values
            for j, model_name in enumerate(WEATHER_MODELS):
                record[f"{model_name}_value"] = (
                    forecast_matrix[i, j]
                )

            location_results.append(record)

        print(
            f"  {parameter}: {len(frame)} predictions"
        )

    return location_results


all_results = []
failures = []

for index, row in locations.iterrows():

    location = row["location"]
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    print("\n" + "=" * 70)
    print(
        f"LOCATION {index + 1}/{len(locations)}: {location}"
    )
    print("=" * 70)

    try:
        records = predict_location(
            location,
            latitude,
            longitude
        )

        all_results.extend(records)

        print(
            "Location completed:",
            location,
            "| Rows:", len(records)
        )

    except Exception as error:

        print("FAILED:", location, error)

        failures.append({
            "location": location,
            "error": str(error)
        })

    # Save completed locations after every iteration
    if all_results:

        pd.DataFrame(all_results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "Progress saved:",
            len(all_results),
            "rows"
        )

    time.sleep(1)


if failures:

    pd.DataFrame(failures).to_csv(
        ERROR_FILE,
        index=False
    )

elif os.path.exists(ERROR_FILE):

    os.remove(ERROR_FILE)


print("\n" + "=" * 70)
print("FINAL STEP 49B SUMMARY")
print("=" * 70)

if not all_results:
    print("No successful predictions were generated.")
    print("Failed locations:", len(failures))
    raise SystemExit(1)

result_df = pd.DataFrame(all_results)

print("Locations expected:", len(locations))

print(
    "Locations completed:",
    result_df["location"].nunique()
)

print("Locations failed:", len(failures))

print("Total prediction rows:", len(result_df))

print("\nPredictions by parameter:")
print(result_df["parameter"].value_counts())

print("\nSelected models:")
print(result_df["selected_model"].value_counts())

print("\nForecast time range:")
print(
    result_df["datetime"].min(),
    "to",
    result_df["datetime"].max()
)

print("\nMissing values:")
print(
    result_df[
        ["predicted_value", "predicted_error"]
    ].isna().sum()
)

print("\nSaved:", OUTPUT_FILE)

if failures:
    print("Errors saved:", ERROR_FILE)

print("\nSTEP 49B COMPLETE")
