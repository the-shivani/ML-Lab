import requests
import pandas as pd
import os
import math
import urllib3
import time

# ============================================================
# STEP 16A - FIND NEAREST IMD STATIONS
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

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
    "imd_station_mapping.csv"
)

STATIONS_URL = (
    "https://wis2box.imd.gov.in/"
    "oapi/collections/stations/items"
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(math.sqrt(a))
    )


# ============================================================
# LOAD LOCATIONS
# ============================================================

locations = pd.read_csv(
    LOCATIONS_FILE
)

print("=" * 70)
print("STEP 16A - FIND NEAREST IMD STATIONS")
print("=" * 70)

print()
print("Our locations:", len(locations))


# ============================================================
# DOWNLOAD ALL IMD STATIONS
# ============================================================

print()
print("Downloading complete IMD station list...")

all_features = []

offset = 0
limit = 100

while True:

    print(
        f"Downloading stations {offset + 1}"
        f" to {offset + limit}..."
    )

    params = {
        "f": "json",
        "limit": limit,
        "offset": offset
    }

    success = False

    for attempt in range(1, 4):

        try:

            response = requests.get(
                STATIONS_URL,
                params=params,
                timeout=(15, 45),
                verify=False
            )

            print(
                f"  HTTP Status: "
                f"{response.status_code}"
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
                f"  {error}"
            )

            if attempt < 3:

                print(
                    "  Waiting 5 seconds..."
                )

                time.sleep(5)

    if not success:

        print()
        print("=" * 70)
        print("FAILED TO DOWNLOAD STATIONS")
        print("=" * 70)

        raise SystemExit


    features = data.get(
        "features",
        []
    )

    print(
        "  Received:",
        len(features)
    )

    all_features.extend(
        features
    )

    # --------------------------------------------------------
    # STOP WHEN LAST PAGE IS REACHED
    # --------------------------------------------------------

    if len(features) < limit:

        break

    offset += limit


print()
print(
    "Total station records received:",
    len(all_features)
)


# ============================================================
# EXTRACT STATIONS
# ============================================================

stations = []

for feature in all_features:

    geometry = feature.get(
        "geometry",
        {}
    )

    coordinates = geometry.get(
        "coordinates"
    )

    properties = feature.get(
        "properties",
        {}
    )

    if not coordinates:
        continue

    if len(coordinates) < 2:
        continue

    longitude = coordinates[0]
    latitude = coordinates[1]

    # Only use fixed land stations
    facility_type = properties.get(
        "facility_type"
    )

    if facility_type != "landFixed":
        continue

    station_id = (
        feature.get("id")
        or properties.get("id")
    )

    station_name = properties.get(
        "name"
    )

    status = properties.get(
        "status"
    )

    stations.append({

        "station_id":
            station_id,

        "station_name":
            station_name,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "status":
            status
    })


stations_df = pd.DataFrame(
    stations
)


print(
    "Usable fixed land stations:",
    len(stations_df)
)


# ============================================================
# FIND NEAREST STATION
# ============================================================

results = []

for _, location in locations.iterrows():

    city = location["City"]

    city_lat = location["Latitude"]
    city_lon = location["Longitude"]

    distances = []

    for _, station in stations_df.iterrows():

        distance = distance_km(
            city_lat,
            city_lon,
            station["latitude"],
            station["longitude"]
        )

        distances.append(
            distance
        )

    stations_df["distance_km"] = distances

    nearest = stations_df.sort_values(
        "distance_km"
    ).iloc[0]

    results.append({

        "location":
            city,

        "location_latitude":
            city_lat,

        "location_longitude":
            city_lon,

        "station_id":
            nearest["station_id"],

        "station_name":
            nearest["station_name"],

        "station_latitude":
            nearest["latitude"],

        "station_longitude":
            nearest["longitude"],

        "distance_km":
            round(
                nearest["distance_km"],
                2
            ),

        "station_status":
            nearest["status"]
    })

    print(
        f"{city:35s}"
        f" -> "
        f"{str(nearest['station_name']):30s}"
        f" ({nearest['distance_km']:.1f} km)"
    )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("STATION MAPPING COMPLETE")
print("=" * 70)

print()
print(
    "Mappings:",
    len(result_df)
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