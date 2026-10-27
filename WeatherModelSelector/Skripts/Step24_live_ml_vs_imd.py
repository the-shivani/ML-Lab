import pandas as pd
import numpy as np
import requests
import os
import time

print("=" * 70)
print("STEP 24 - LIVE ML VS IMD VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

RESULTS_PATH = os.path.join(
    BASE_PATH,
    "results"
)

LIVE_FILE = os.path.join(
    RESULTS_PATH,
    "live_multivariable_predictions.csv"
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
    RESULTS_PATH,
    "live_multivariable_ml_vs_imd.csv"
)


# ---------------------------------------------------------
# 2. LOAD FILES
# ---------------------------------------------------------

print("\nLoading live ML predictions...")

live = pd.read_csv(
    LIVE_FILE
)

print(
    "Live prediction rows:",
    len(live)
)


print("\nLoading IMD observations...")

imd = pd.read_csv(
    IMD_FILE
)

print(
    "IMD rows:",
    len(imd)
)


print("\nLoading station mapping...")

mapping = pd.read_csv(
    MAPPING_FILE
)

print(
    "Mappings:",
    len(mapping)
)


# ---------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_live = [
    "location",
    "latitude",
    "longitude",
    "recommended_temperature_model",
    "recommended_humidity_model",
    "recommended_wind_speed_model",
    "recommended_pressure_model"
]

required_imd = [
    "location",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "pressure"
]


for column in required_live:

    if column not in live.columns:

        raise ValueError(
            f"Missing column in live file: {column}"
        )


for column in required_imd:

    if column not in imd.columns:

        raise ValueError(
            f"Missing column in IMD file: {column}"
        )


# ---------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------

parameters = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]

weather_models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


# ---------------------------------------------------------
# 5. GET FORECAST
# ---------------------------------------------------------

def get_forecast(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
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
# 6. PROCESS EACH LOCATION
# ---------------------------------------------------------

results = []

print("\nStarting comparison...\n")


for number, ml_row in live.iterrows():

    city = ml_row["location"]

    print(
        f"[{number + 1}/{len(live)}] {city}"
    )


    # -----------------------------------------------------
    # FIND IMD OBSERVATION
    # -----------------------------------------------------

    imd_rows = imd[
        imd["location"] == city
    ]


    if len(imd_rows) == 0:

        print(
            "    No IMD observation available - skipped"
        )

        continue


    # Use the latest observation available
    imd_row = imd_rows.iloc[0]


    # -----------------------------------------------------
    # GET FORECAST
    # -----------------------------------------------------

    try:

        forecast = get_forecast(
            ml_row["latitude"],
            ml_row["longitude"]
        )

    except Exception as error:

        print(
            "    Forecast failed:",
            error
        )

        continue


    times = pd.to_datetime(
        forecast["hourly"]["time"]
    )


    # -----------------------------------------------------
    # FIND FORECAST HOUR CLOSEST TO IMD TIME
    # -----------------------------------------------------

    if "datetime" in imd_row:

        try:

            imd_time = pd.to_datetime(
                imd_row["datetime"],
                utc=True
            )

            if imd_time.tzinfo is not None:

                forecast_times = (
                    times
                    .tz_localize(
                        "UTC"
                    )
                )

            else:

                forecast_times = times

            differences = abs(
                forecast_times - imd_time
            )

            index = differences.argmin()

        except Exception:

            index = 0

    else:

        index = 0


    row = {

        "location": city,

        "imd_datetime": (
            imd_row.get(
                "datetime",
                ""
            )
        ),

        "forecast_datetime": str(
            times[index]
        ),

        "imd_temperature":
            imd_row["temperature"],

        "imd_humidity":
            imd_row["relative_humidity"],

        "imd_wind_speed":
            imd_row["wind_speed"],

        "imd_pressure":
            imd_row["pressure"],

        "ml_temperature_model":
            ml_row[
                "recommended_temperature_model"
            ],

        "ml_humidity_model":
            ml_row[
                "recommended_humidity_model"
            ],

        "ml_wind_speed_model":
            ml_row[
                "recommended_wind_speed_model"
            ],

        "ml_pressure_model":
            ml_row[
                "recommended_pressure_model"
            ]
    }


    # -----------------------------------------------------
    # 7. GET FORECAST VALUES
    # -----------------------------------------------------

    for model in weather_models:


        # Temperature

        temperature_key = (
            f"temperature_2m_{model}"
        )

        row[
            f"{model}_temperature"
        ] = forecast[
            "hourly"
        ][temperature_key][index]


        # Humidity

        humidity_key = (
            f"relative_humidity_2m_{model}"
        )

        row[
            f"{model}_humidity"
        ] = forecast[
            "hourly"
        ][humidity_key][index]


        # Wind

        wind_key = (
            f"wind_speed_10m_{model}"
        )

        row[
            f"{model}_wind_speed"
        ] = forecast[
            "hourly"
        ][wind_key][index]


        # Pressure

        pressure_key = (
            f"pressure_msl_{model}"
        )

        row[
            f"{model}_pressure"
        ] = forecast[
            "hourly"
        ][pressure_key][index]


    # -----------------------------------------------------
    # 8. CALCULATE ERRORS
    # -----------------------------------------------------

    for model in weather_models:

        row[
            f"{model}_temperature_error"
        ] = abs(
            row[
                f"{model}_temperature"
            ]
            -
            row["imd_temperature"]
        )


        row[
            f"{model}_humidity_error"
        ] = abs(
            row[
                f"{model}_humidity"
            ]
            -
            row["imd_humidity"]
        )


        row[
            f"{model}_wind_speed_error"
        ] = abs(
            row[
                f"{model}_wind_speed"
            ]
            -
            row["imd_wind_speed"]
        )


        row[
            f"{model}_pressure_error"
        ] = abs(
            row[
                f"{model}_pressure"
            ]
            -
            row["imd_pressure"]
        )


    # -----------------------------------------------------
    # 9. FIND ACTUAL BEST MODEL
    # -----------------------------------------------------

    for parameter in parameters:

        error_columns = [

            f"{model}_{parameter}_error"

            for model in weather_models

        ]


        best_column = min(
            error_columns,
            key=lambda column:
            row[column]
        )


        actual_model = best_column.replace(
            f"_{parameter}_error",
            ""
        )


        row[
            f"actual_best_{parameter}_model"
        ] = actual_model


    # -----------------------------------------------------
    # 10. CHECK ML SELECTION
    # -----------------------------------------------------

    for parameter in parameters:

        ml_column = (
            f"ml_{parameter}_model"
        )

        actual_column = (
            f"actual_best_{parameter}_model"
        )

        row[
            f"{parameter}_ml_correct"
        ] = (
            row[ml_column]
            ==
            row[actual_column]
        )


    results.append(row)


# ---------------------------------------------------------
# 11. CREATE DATAFRAME
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ---------------------------------------------------------
# 12. SAVE
# ---------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 13. FINAL RESULTS
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("REAL-TIME ML VALIDATION RESULTS")
print("=" * 70)

print(
    "\nLocations compared:",
    len(results_df)
)


for parameter in parameters:

    correct_column = (
        f"{parameter}_ml_correct"
    )

    if len(results_df) > 0:

        accuracy = (
            results_df[
                correct_column
            ].mean()
            * 100
        )

    else:

        accuracy = 0


    print(
        f"\n{parameter.upper()}"
    )

    print(
        "ML selection accuracy:",
        f"{accuracy:.2f}%"
    )


    print(
        "Actual best model:"
    )

    print(
        results_df[
            f"actual_best_{parameter}_model"
        ].value_counts()
    )


# ---------------------------------------------------------
# 14. AVERAGE ERRORS
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("AVERAGE ACTUAL ERROR BY MODEL")
print("=" * 70)


for parameter in parameters:

    print(
        f"\n{parameter.upper()}"
    )


    error_columns = [

        f"{model}_{parameter}_error"

        for model in weather_models

    ]


    averages = (
        results_df[
            error_columns
        ].mean()
    )


    for column, value in averages.items():

        print(
            f"{column:35s} "
            f"{value:.3f}"
        )


print("\n")
print("=" * 70)
print("SAVED RESULT")
print("=" * 70)

print(
    OUTPUT_FILE
)

print("=" * 70)