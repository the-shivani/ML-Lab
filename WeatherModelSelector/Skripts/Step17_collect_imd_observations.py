import requests
import pandas as pd
import os
import urllib3
from datetime import datetime, timedelta, timezone
import math

# ============================================================
# STEP 17 - COLLECT REAL-TIME IMD OBSERVATIONS
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

MAPPING_FILE = os.path.join(
    DATA_PATH,
    "imd_station_mapping.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "imd_latest_observations.csv"
)

OBS_URL = (
    "https://wis2box.imd.gov.in/"
    "oapi/collections/"
    "urn:wmo:md:in-imd:"
    "surface-based-observations.synop/items"
)


# ============================================================
# HUMIDITY CALCULATION
# ============================================================

def calculate_relative_humidity(
    temperature,
    dewpoint
):

    if (
        pd.isna(temperature)
        or pd.isna(dewpoint)
    ):
        return None

    # Magnus formula
    a = 17.625
    b = 243.04

    gamma_dew = (
        (a * dewpoint)
        / (b + dewpoint)
    )

    gamma_temp = (
        (a * temperature)
        / (b + temperature)
    )

    rh = (
        100
        * math.exp(
            gamma_dew - gamma_temp
        )
    )

    return round(
        max(0, min(100, rh)),
        2
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 17 - REAL-TIME IMD OBSERVATION COLLECTOR")
print("=" * 70)


# ============================================================
# LOAD STATION MAPPING
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE
)

print()
print(
    "Mapped locations:",
    len(mapping)
)

station_ids = set(
    mapping["station_id"]
    .astype(str)
)

print(
    "Unique IMD stations:",
    len(station_ids)
)


# ============================================================
# TIME WINDOW
# ============================================================

now = datetime.now(
    timezone.utc
)

start_time = (
    now - timedelta(hours=6)
)

start = start_time.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

end = now.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

print()
print("IMD observation window:")
print("Start:", start)
print("End  :", end)


# ============================================================
# DOWNLOAD OBSERVATIONS
# ============================================================

print()
print("Downloading recent IMD observations...")

all_features = []

limit = 100
offset = 0

while True:

    print()
    print(
        f"Requesting records "
        f"{offset + 1} to {offset + limit}..."
    )

    params = {
        "f": "json",
        "limit": limit,
        "offset": offset,
        "datetime": f"{start}/{end}"
    }

    try:

        response = requests.get(
            OBS_URL,
            params=params,
            timeout=(15, 60),
            verify=False
        )

        print(
            "HTTP Status:",
            response.status_code
        )

        response.raise_for_status()

        data = response.json()

    except Exception as error:

        print()
        print(
            "Request failed:"
        )

        print(error)

        print()
        print(
            "Stopping collection."
        )

        break

    features = data.get(
        "features",
        []
    )

    print(
        "Received:",
        len(features)
    )

    all_features.extend(
        features
    )

    # --------------------------------------------------------
    # STOP CONDITIONS
    # --------------------------------------------------------

    if len(features) < limit:

        break

    offset += limit

    # Safety limit
    if offset >= 10000:

        print()
        print(
            "Safety limit reached."
        )

        break


print()
print(
    "Total observations downloaded:",
    len(all_features)
)


# ============================================================
# KEEP ONLY OUR 31 STATIONS
# ============================================================

print()
print(
    "Filtering observations "
    "for our 31 stations..."
)

records = []

for feature in all_features:

    properties = feature.get(
        "properties",
        {}
    )

    station_id = str(
        properties.get(
            "wigos_station_identifier"
        )
    )

    if station_id not in station_ids:

        continue

    name = properties.get(
        "name"
    )

    value = properties.get(
        "value"
    )

    units = properties.get(
        "units"
    )

    phenomenon_time = properties.get(
        "phenomenonTime"
    )

    records.append({

        "station_id":
            station_id,

        "parameter":
            name,

        "value":
            value,

        "units":
            units,

        "phenomenon_time":
            phenomenon_time
    })


raw_df = pd.DataFrame(
    records
)

print(
    "Matching observations:",
    len(raw_df)
)


# ============================================================
# CHECK IF DATA EXISTS
# ============================================================

if raw_df.empty:

    print()
    print("=" * 70)
    print("NO OBSERVATIONS FOUND FOR OUR STATIONS")
    print("=" * 70)

    print()
    print(
        "The IMD API returned recent data, "
        "but none matched our mapped stations."
    )

    raise SystemExit


# ============================================================
# DISPLAY PARAMETER COUNTS
# ============================================================

print()
print("Parameters found:")

print(
    raw_df["parameter"]
    .value_counts()
)


# ============================================================
# CONVERT TIME
# ============================================================

raw_df["time_string"] = (
    raw_df["phenomenon_time"]
    .astype(str)
)


# ============================================================
# CREATE STATION-PARAMETER LATEST DATA
# ============================================================

latest_records = []

for station_id in station_ids:

    station_data = raw_df[
        raw_df["station_id"]
        == station_id
    ]

    if station_data.empty:

        continue

    record = {
        "station_id":
            station_id
    }

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temp = station_data[
        station_data["parameter"]
        == "air_temperature"
    ]

    if not temp.empty:

        temp = temp.iloc[-1]

        record["temperature"] = (
            temp["value"]
        )

        record["temperature_time"] = (
            temp["phenomenon_time"]
        )

    else:

        record["temperature"] = None
        record["temperature_time"] = None


    # --------------------------------------------------------
    # DEW POINT
    # --------------------------------------------------------

    dew = station_data[
        station_data["parameter"]
        == "dewpoint_temperature"
    ]

    if not dew.empty:

        dew = dew.iloc[-1]

        record["dewpoint"] = (
            dew["value"]
        )

        record["dewpoint_time"] = (
            dew["phenomenon_time"]
        )

    else:

        record["dewpoint"] = None
        record["dewpoint_time"] = None


    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    pressure = station_data[
        station_data["parameter"]
        == "pressure_reduced_to_mean_sea_level"
    ]

    if not pressure.empty:

        pressure = pressure.iloc[-1]

        record["pressure"] = (
            pressure["value"]
        )

        record["pressure_time"] = (
            pressure["phenomenon_time"]
        )

    else:

        record["pressure"] = None
        record["pressure_time"] = None


    # --------------------------------------------------------
    # WIND SPEED
    # --------------------------------------------------------

    wind = station_data[
        station_data["parameter"]
        == "wind_speed"
    ]

    if not wind.empty:

        wind = wind.iloc[-1]

        record["wind_speed"] = (
            wind["value"]
        )

        record["wind_time"] = (
            wind["phenomenon_time"]
        )

    else:

        record["wind_speed"] = None
        record["wind_time"] = None


    # --------------------------------------------------------
    # CALCULATE RELATIVE HUMIDITY
    # --------------------------------------------------------

    record["relative_humidity"] = (
        calculate_relative_humidity(
            record["temperature"],
            record["dewpoint"]
        )
    )


    # --------------------------------------------------------
    # MAP CITY
    # --------------------------------------------------------

    station_mapping = mapping[
        mapping["station_id"].astype(str)
        == station_id
    ]

    if not station_mapping.empty:

        row = station_mapping.iloc[0]

        record["location"] = (
            row["location"]
        )

        record["station_name"] = (
            row["station_name"]
        )

        record["distance_km"] = (
            row["distance_km"]
        )


    latest_records.append(
        record
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    latest_records
)


# ============================================================
# REORDER COLUMNS
# ============================================================

columns = [

    "location",
    "station_id",
    "station_name",
    "distance_km",

    "temperature",
    "temperature_time",

    "dewpoint",
    "dewpoint_time",

    "relative_humidity",

    "pressure",
    "pressure_time",

    "wind_speed",
    "wind_time"
]

columns = [
    column
    for column in columns
    if column in result_df.columns
]

result_df = result_df[
    columns
]


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("IMD COLLECTION COMPLETE")
print("=" * 70)

print()
print(
    "Stations with observations:",
    len(result_df)
)

print(
    "Locations mapped:",
    len(mapping)
)

print()
print("Latest observations:")

print(
    result_df[
        [
            "location",
            "station_name",
            "temperature",
            "relative_humidity",
            "pressure",
            "wind_speed"
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved to:")

print(
    OUTPUT_FILE
)

print()
print("=" * 70)