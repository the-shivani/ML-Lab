import os
import time
import joblib
import requests
import numpy as np
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"
MODELS_DIR = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

os.makedirs(RESULTS, exist_ok=True)

LOCATION = {
    "location": "Karad",
    "latitude": 17.285,
    "longitude": 74.184
}

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

API_URL = "https://api.open-meteo.com/v1/forecast"

print("=" * 70)
print("STEP 49A - LIVE WEATHER MODEL SELECTION")
print("=" * 70)

def fetch_forecast(model_name):

    params = {
        "latitude": LOCATION["latitude"],
        "longitude": LOCATION["longitude"],
        "hourly": ",".join(PARAMETERS.values()),
        "models": model_name,
        "timezone": "Asia/Kolkata",
        "forecast_days": 2
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"No hourly data returned for {model_name}: {data}"
        )

    hourly = pd.DataFrame(data["hourly"])

    hourly["datetime"] = pd.to_datetime(
        hourly["time"]
    )

    hourly = hourly.drop(columns=["time"])

    return hourly.set_index("datetime")


forecasts = {}

for model_name in WEATHER_MODELS:

    print("Fetching:", model_name)

    try:
        forecasts[model_name] = fetch_forecast(model_name)
        print("Success:", len(forecasts[model_name]), "hours")

    except Exception as error:
        print("Failed:", model_name, error)

    time.sleep(1)


if len(forecasts) != 4:
    raise RuntimeError(
        "All four weather models are required. "
        "Stopping to avoid incomplete comparisons."
    )


# Find common timestamps across all models
common_times = forecasts[WEATHER_MODELS[0]].index

for model_name in WEATHER_MODELS[1:]:
    common_times = common_times.intersection(
        forecasts[model_name].index
    )

common_times = common_times.sort_values()

# Use future forecast-valid hours only
current_time = pd.Timestamp.now(
    tz="Asia/Kolkata"
).tz_localize(None)

common_times = common_times[
    common_times > current_time
]

if len(common_times) == 0:
    raise RuntimeError(
        "No future common forecast timestamps found."
    )

print("\nCommon future hours:", len(common_times))

results = []

for parameter, api_variable in PARAMETERS.items():

    print("\n" + "-" * 60)
    print("PARAMETER:", parameter.upper())
    print("-" * 60)

    model_file = os.path.join(
        MODELS_DIR,
        f"{parameter}_forecast_only_selector_v5.pkl"
    )

    saved = joblib.load(model_file)

    regressors = saved["models"]
    feature_columns = saved["features"]
    threshold = saved["threshold"]

    rows = []

    for timestamp in common_times:

        row = {
            "location": LOCATION["location"],
            "datetime": timestamp,
            "latitude": LOCATION["latitude"],
            "longitude": LOCATION["longitude"],
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month
        }

        values = []

        for model_name in WEATHER_MODELS:

            value = forecasts[model_name].loc[
                timestamp, api_variable
            ]

            row[f"{model_name}_{parameter}"] = value
            values.append(value)

        row["forecast_mean"] = np.nanmean(values)

        row["forecast_spread"] = (
            np.nanmax(values) - np.nanmin(values)
        )

        rows.append(row)

    frame = pd.DataFrame(rows)

    # Keep only hours with complete model forecasts
    frame = frame.dropna(
        subset=feature_columns
    ).copy()

    if frame.empty:
        print("No complete forecast hours available.")
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

    selected_models = np.array(
        WEATHER_MODELS
    )[selected_indices]

    forecast_matrix = frame[
        [
            f"{model_name}_{parameter}"
            for model_name in WEATHER_MODELS
        ]
    ].to_numpy()

    selected_values = forecast_matrix[
        np.arange(len(frame)),
        selected_indices
    ]

    for i in range(len(frame)):

        results.append({
            "location": LOCATION["location"],
            "latitude": LOCATION["latitude"],
            "longitude": LOCATION["longitude"],
            "datetime": frame.iloc[i]["datetime"],
            "parameter": parameter,
            "selected_model": selected_models[i],
            "predicted_value": selected_values[i],
            "predicted_error": predicted_errors[
                i, selected_indices[i]
            ],
            "ecmwf_value": forecast_matrix[
                i, ecmwf_index
            ],
            "switch_threshold": threshold
        })

    print("Forecasts generated:", len(frame))

    print(
        "Selected model distribution:",
        pd.Series(selected_models).value_counts().to_dict()
    )


result_df = pd.DataFrame(results)

output_file = os.path.join(
    RESULTS,
    "step49A_live_karad_predictions.csv"
)

result_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("LIVE PREDICTION SUMMARY")
print("=" * 70)

print("Total prediction rows:", len(result_df))

print("\nPredictions by parameter:")
print(result_df["parameter"].value_counts())

print("\nSample predictions:")
print(result_df.head(10).to_string(index=False))

print("\nSaved:", output_file)
print("STEP 49A COMPLETE")
