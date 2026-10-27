import requests
import pandas as pd
import os
import urllib3

# ============================================================
# STEP 18B - REAL-TIME WEATHER MODEL VS IMD COMPARISON
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

IMD_FILE = os.path.join(
    DATA_PATH,
    "imd_latest_observations.csv"
)

MAPPING_FILE = os.path.join(
    DATA_PATH,
    "imd_station_mapping.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "realtime_model_comparison.csv"
)

URL = "https://api.open-meteo.com/v1/forecast"

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


# ============================================================
# FUNCTION - GET OPEN-METEO FORECAST
# ============================================================

def get_forecast(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "pressure_msl"
        ),

        "models":
            "ecmwf_ifs,"
            "gfs_seamless,"
            "icon_seamless,"
            "gem_seamless",

        "forecast_days": 1,

        "timezone": "UTC"
    }

    for attempt in range(1, 4):

        try:

            response = requests.get(
                URL,
                params=params,
                timeout=60,
                verify=False
            )

            response.raise_for_status()

            return response.json()

        except Exception as error:

            print(
                f"  API attempt {attempt}/3 failed: {error}"
            )

            if attempt < 3:

                import time

                print(
                    "  Waiting 5 seconds..."
                )

                time.sleep(5)

            else:

                return None


# ============================================================
# FUNCTION - FIND NEAREST FORECAST TIME
# ============================================================

def get_nearest_index(times, target_time):

    differences = []

    for t in times:

        forecast_time = pd.to_datetime(
            t,
            utc=True
        )

        difference = abs(
            (
                forecast_time -
                target_time
            ).total_seconds()
        )

        differences.append(
            difference
        )

    return differences.index(
        min(differences)
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 18B - REAL-TIME MODEL VS IMD COMPARISON")
print("=" * 70)


# ============================================================
# LOAD FILES
# ============================================================

imd = pd.read_csv(
    IMD_FILE
)

mapping = pd.read_csv(
    MAPPING_FILE
)


print()
print(
    "IMD locations available:",
    len(imd)
)


# ============================================================
# PREPARE LOCATION COORDINATES
# ============================================================

mapping_small = mapping[
    [
        "location",
        "location_latitude",
        "location_longitude"
    ]
].drop_duplicates(
    subset=["location"]
)


mapping_small = mapping_small.rename(
    columns={
        "location_latitude": "latitude",
        "location_longitude": "longitude"
    }
)


# ============================================================
# MERGE
# ============================================================

df = imd.merge(
    mapping_small,
    on="location",
    how="left"
)


print(
    "Locations after coordinate merge:",
    len(df)
)


# ============================================================
# PROCESS LOCATIONS
# ============================================================

results = []


for _, row in df.iterrows():

    location = row["location"]

    latitude = row["latitude"]

    longitude = row["longitude"]


    print()
    print("-" * 70)

    print(
        "LOCATION:",
        location
    )

    print(
        "Coordinates:",
        latitude,
        longitude
    )


    # ========================================================
    # FIND IMD OBSERVATION TIME
    # ========================================================

    possible_times = []


    for column in [
        "temperature_time",
        "relative_humidity_time",
        "pressure_time",
        "wind_speed_time"
    ]:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                possible_times.append(
                    pd.to_datetime(
                        value,
                        utc=True
                    )
                )


    # Fallback time columns

    if not possible_times:

        for column in [
            "observation_time",
            "datetime",
            "phenomenonTime"
        ]:

            if column in row.index:

                value = row[column]

                if pd.notna(value):

                    possible_times.append(
                        pd.to_datetime(
                            value,
                            utc=True
                        )
                    )


    if not possible_times:

        print(
            "  No valid IMD observation time."
        )

        continue


    target_time = max(
        possible_times
    )


    print(
        "IMD observation time:",
        target_time
    )


    # ========================================================
    # GET OPEN-METEO FORECAST
    # ========================================================

    forecast = get_forecast(
        latitude,
        longitude
    )


    if forecast is None:

        print(
            "  Forecast unavailable."
        )

        continue


    hourly = forecast.get(
        "hourly"
    )


    if hourly is None:

        print(
            "  Hourly forecast missing."
        )

        continue


    times = hourly["time"]


    nearest_index = get_nearest_index(
        times,
        target_time
    )


    forecast_time = pd.to_datetime(
        times[nearest_index],
        utc=True
    )


    print(
        "Forecast time used:",
        forecast_time
    )


    # ========================================================
    # IMD OBSERVED VALUES
    # ========================================================

    imd_temperature = row.get(
        "temperature",
        None
    )

    imd_humidity = row.get(
        "relative_humidity",
        None
    )

    imd_wind = row.get(
        "wind_speed",
        None
    )

    imd_pressure = row.get(
        "pressure",
        None
    )


    # ========================================================
    # EACH MODEL
    # ========================================================

    for model in MODELS:

        # IMPORTANT:
        # Open-Meteo uses:
        #
        # temperature_2m_ecmwf_ifs
        #
        # NOT:
        #
        # ecmwf_ifs_temperature_2m


        temperature_key = (
            "temperature_2m_" +
            model
        )

        humidity_key = (
            "relative_humidity_2m_" +
            model
        )

        wind_key = (
            "wind_speed_10m_" +
            model
        )

        pressure_key = (
            "pressure_msl_" +
            model
        )


        # ====================================================
        # GET FORECAST VALUES
        # ====================================================

        temperature_values = hourly.get(
            temperature_key
        )

        humidity_values = hourly.get(
            humidity_key
        )

        wind_values = hourly.get(
            wind_key
        )

        pressure_values = hourly.get(
            pressure_key
        )


        temperature = None

        humidity = None

        wind = None

        pressure = None


        if temperature_values is not None:

            temperature = (
                temperature_values[
                    nearest_index
                ]
            )


        if humidity_values is not None:

            humidity = (
                humidity_values[
                    nearest_index
                ]
            )


        if wind_values is not None:

            wind = (
                wind_values[
                    nearest_index
                ]
            )


        if pressure_values is not None:

            pressure = (
                pressure_values[
                    nearest_index
                ]
            )


        # ====================================================
        # CALCULATE ERRORS
        # ====================================================

        temperature_error = None

        humidity_error = None

        wind_error = None

        pressure_error = None


        if (
            pd.notna(imd_temperature)
            and temperature is not None
        ):

            temperature_error = abs(
                float(temperature)
                -
                float(imd_temperature)
            )


        if (
            pd.notna(imd_humidity)
            and humidity is not None
        ):

            humidity_error = abs(
                float(humidity)
                -
                float(imd_humidity)
            )


        if (
            pd.notna(imd_wind)
            and wind is not None
        ):

            wind_error = abs(
                float(wind)
                -
                float(imd_wind)
            )


        if (
            pd.notna(imd_pressure)
            and pressure is not None
        ):

            pressure_error = abs(
                float(pressure)
                -
                float(imd_pressure)
            )


        # ====================================================
        # STORE RESULT
        # ====================================================

        results.append({

            "location":
                location,

            "imd_time":
                target_time,

            "forecast_time":
                forecast_time,

            "model":
                model,

            "imd_temperature":
                imd_temperature,

            "forecast_temperature":
                temperature,

            "temperature_error":
                temperature_error,

            "imd_humidity":
                imd_humidity,

            "forecast_humidity":
                humidity,

            "humidity_error":
                humidity_error,

            "imd_wind_speed":
                imd_wind,

            "forecast_wind_speed":
                wind,

            "wind_speed_error":
                wind_error,

            "imd_pressure":
                imd_pressure,

            "forecast_pressure":
                pressure,

            "pressure_error":
                pressure_error
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

if not results:

    print()
    print(
        "ERROR: No results created."
    )

    raise SystemExit


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("REAL-TIME COMPARISON COMPLETE")
print("=" * 70)

print()

print(
    "Rows created:",
    len(results_df)
)

print(
    "Locations compared:",
    results_df["location"].nunique()
)

print(
    "Models compared:",
    results_df["model"].nunique()
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# AVERAGE ERROR BY MODEL
# ============================================================

print()
print("=" * 70)
print("AVERAGE ERROR BY MODEL")
print("=" * 70)


error_columns = [
    "temperature_error",
    "humidity_error",
    "wind_speed_error",
    "pressure_error"
]


for column in error_columns:

    print()
    print(
        column.upper()
    )

    summary = (
        results_df
        .groupby("model")[column]
        .mean()
        .sort_values()
    )

    print(
        summary.to_string()
    )


# ============================================================
# BEST MODEL BY PARAMETER
# ============================================================

print()
print("=" * 70)
print("BEST MODEL BY PARAMETER")
print("=" * 70)


for column in error_columns:

    average_error = (
        results_df
        .groupby("model")[column]
        .mean()
        .dropna()
    )


    if len(average_error) == 0:

        print(
            column,
            "-> No valid data"
        )

        continue


    best_model = (
        average_error.idxmin()
    )

    best_error = (
        average_error.min()
    )


    parameter = column.replace(
        "_error",
        ""
    )


    print()

    print(
        parameter,
        "->",
        best_model,
        "| Average error:",
        round(
            best_error,
            3
        )
    )


print()
print("=" * 70)
print("STEP 18B FINISHED")
print("=" * 70)