import requests
import csv
import os


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

LOCATIONS_FILE = os.path.join(
    BASE_PATH,
    "Locations.csv"
)

WEATHER_FILE = os.path.join(
    BASE_PATH,
    "weather_data.csv"
)


# --------------------------------------------------
# GET TIMEZONE FOR LOCATION
# --------------------------------------------------

def get_timezone(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Could not get timezone.")
        return "auto"

    data = response.json()

    return data.get("timezone", "auto")


# --------------------------------------------------
# GET WEATHER FROM ONE MODEL
# --------------------------------------------------

def get_weather_model(latitude, longitude, timezone, model_name):

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

        "forecast_days": 1,

        "timezone": timezone,

        "models": model_name
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:

        print(f"{model_name}: ERROR")
        print(response.text)

        return None

    return response.json()


# --------------------------------------------------
# SAVE WEATHER DATA
# --------------------------------------------------

def save_to_csv(location, model_name, data):

    file_exists = os.path.exists(WEATHER_FILE)

    hourly = data["hourly"]

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    precipitation = hourly["precipitation"]
    wind_speed = hourly["wind_speed_10m"]
    pressure = hourly["pressure_msl"]

    with open(
        WEATHER_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Create header
        if not file_exists:

            writer.writerow([
                "location",
                "latitude",
                "longitude",
                "datetime",
                "model",
                "temperature",
                "humidity",
                "precipitation",
                "wind_speed",
                "pressure"
            ])

        # Save every hour
        for i in range(len(times)):

            writer.writerow([
                location["City"],
                location["Latitude"],
                location["Longitude"],
                times[i],
                model_name,
                temperatures[i],
                humidity[i],
                precipitation[i],
                wind_speed[i],
                pressure[i]
            ])


# --------------------------------------------------
# READ LOCATIONS CSV
# --------------------------------------------------

def read_locations():

    locations = []

    with open(
        LOCATIONS_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        print("\nCSV COLUMNS FOUND:")
        print(reader.fieldnames)

        for row in reader:
            locations.append(row)

    return locations


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 70)
    print("       OPEN-METEO MULTI-LOCATION WEATHER COLLECTOR")
    print("=" * 70)

    # Read locations from CSV
    locations = read_locations()

    print(f"\nTotal locations found: {len(locations)}")

    # Weather models
    models = [
        "ecmwf_ifs",
        "gfs_seamless",
        "icon_seamless",
        "gem_seamless"
    ]

    # --------------------------------------------------
    # LOOP THROUGH LOCATIONS
    # --------------------------------------------------

    for location in locations:

       name = location["City"]
       latitude = float(location["Latitude"])
       longitude = float(location["Longitude"])

    print("\n")
    print("=" * 70)
    print(f"LOCATION: {name}")
    print(f"Latitude : {latitude}")
    print(f"Longitude: {longitude}")
    print("=" * 70)

        # Get timezone
    timezone = get_timezone(
            latitude,
            longitude
        )

    print(f"Timezone : {timezone}")

        # --------------------------------------------------
        # LOOP THROUGH WEATHER MODELS
        # --------------------------------------------------

    for model in models:

            print(f"\nFetching {model} for {name}...")

            data = get_weather_model(
                latitude,
                longitude,
                timezone,
                model
            )

            if data is not None:

                save_to_csv(
                    location,
                    model,
                    data
                )

                print(
                    f"✓ {model} data saved"
                )

    print("\n")
    print("=" * 70)
    print("DATA COLLECTION COMPLETED")
    print("=" * 70)

    print("\nWeather data saved at:")
    print(WEATHER_FILE)


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()