import requests
import pandas as pd
import os
import time

# ============================================================
# STEP 20 - COLLECT ERA5 MULTIVARIABLE REFERENCE DATA
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference.csv"
)

URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2026-09-01"
END_DATE = "2026-09-30"

HOURLY_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "precipitation,"
    "wind_speed_10m,"
    "surface_pressure"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 20 - ERA5 MULTIVARIABLE REFERENCE COLLECTION")
print("=" * 70)


# ============================================================
# LOAD LOCATIONS
# ============================================================

locations = pd.read_csv(
    LOCATIONS_FILE
)

print()
print(
    "Locations found:",
    len(locations)
)

print(
    "Columns:",
    locations.columns.tolist()
)


# ============================================================
# CHECK LOCATION COLUMNS
# ============================================================

required_columns = [
    "City",
    "Latitude",
    "Longitude"
]

for column in required_columns:

    if column not in locations.columns:

        print()
        print(
            "ERROR: Missing column:",
            column
        )

        raise SystemExit


# ============================================================
# COLLECT DATA
# ============================================================

all_data = []


for index, row in locations.iterrows():

    city = row["City"]

    latitude = row["Latitude"]

    longitude = row["Longitude"]


    print()
    print("-" * 70)

    print(
        f"{index + 1}/{len(locations)}",
        city
    )

    print(
        "Latitude :",
        latitude
    )

    print(
        "Longitude:",
        longitude
    )


    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            START_DATE,

        "end_date":
            END_DATE,

        "hourly":
            HOURLY_VARIABLES,

        "timezone":
            "UTC"
    }


    success = False


    for attempt in range(1, 4):

        try:

            response = requests.get(
                URL,
                params=params,
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            success = True

            break


        except Exception as error:

            print(
                f"  Attempt {attempt}/3 failed:"
            )

            print(
                " ",
                error
            )

            if attempt < 3:

                print(
                    "  Waiting 5 seconds..."
                )

                time.sleep(5)


    if not success:

        print(
            "  FAILED:",
            city
        )

        continue


    hourly = data.get(
        "hourly"
    )


    if hourly is None:

        print(
            "  No hourly data received."
        )

        continue


    times = hourly.get(
        "time",
        []
    )

    temperatures = hourly.get(
        "temperature_2m",
        []
    )

    humidities = hourly.get(
        "relative_humidity_2m",
        []
    )

    precipitation = hourly.get(
        "precipitation",
        []
    )

    wind_speed = hourly.get(
        "wind_speed_10m",
        []
    )

    pressure = hourly.get(
        "surface_pressure",
        []
    )


    row_count = len(times)


    print(
        "  Hours received:",
        row_count
    )


    # ========================================================
    # CREATE ROWS
    # ========================================================

    for i in range(row_count):

        all_data.append({

            "location":
                city,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "datetime":
                times[i],

            "reference_temperature":
                temperatures[i],

            "reference_humidity":
                humidities[i],

            "reference_precipitation":
                precipitation[i],

            "reference_wind_speed":
                wind_speed[i],

            "reference_pressure":
                pressure[i]
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_data
)


if len(df) == 0:

    print()
    print(
        "ERROR: No data collected."
    )

    raise SystemExit


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)


df = df.drop_duplicates(
    subset=[
        "location",
        "datetime"
    ]
)


after = len(df)


print()
print(
    "Duplicates removed:",
    before - after
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "location",
        "datetime"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ERA5 COLLECTION COMPLETE")
print("=" * 70)

print()

print(
    "Total rows:",
    len(df)
)

print(
    "Locations:",
    df["location"].nunique()
)

print(
    "Start:",
    df["datetime"].min()
)

print(
    "End:",
    df["datetime"].max()
)

print()

print(
    "Columns:"
)

print(
    df.columns.tolist()
)

print()

print(
    "Missing values:"
)

print(
    df.isna().sum().to_string()
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("STEP 20 FINISHED")
print("=" * 70)