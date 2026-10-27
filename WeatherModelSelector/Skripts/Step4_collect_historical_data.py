import requests
import csv
import os
import time

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")

LOCATIONS_FILE = os.path.join(DATA_PATH, "Locations.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "historical_forecast_data.csv")START_DATE = "2026-09-01"
END_DATE = "2026-09-30"

FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


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


def request_with_retry(
    url,
    params,
    name,
    attempts=5
):

    for attempt in range(1, attempts + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=120
            )

            if response.status_code == 200:

                return response.json()

            print(
                f"    {name}: HTTP {response.status_code}"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"    {name}: connection error"
            )

            print(
                f"    Attempt {attempt}/{attempts}"
            )

        if attempt < attempts:

            print(
                "    Waiting 10 seconds before retry..."
            )

            time.sleep(10)

    print(
        f"    {name}: FAILED"
    )

    return None


def get_previous_run_temperature(
    latitude,
    longitude,
    model
):

    params = {

        "latitude": latitude,
        "longitude": longitude,

        "hourly":
            "temperature_2m_previous_day1",

        "start_date": START_DATE,
        "end_date": END_DATE,

        "timezone": "GMT",

        "models": model,

        "temperature_unit": "celsius"
    }

    return request_with_retry(
        FORECAST_URL,
        params,
        model
    )


def get_era5_temperature(
    latitude,
    longitude
):

    params = {

        "latitude": latitude,
        "longitude": longitude,

        "hourly": "temperature_2m",

        "start_date": START_DATE,
        "end_date": END_DATE,

        "timezone": "GMT",

        "temperature_unit": "celsius"
    }

    return request_with_retry(
        ARCHIVE_URL,
        params,
        "ERA5"
    )


def save_forecast_data(
    location,
    model,
    data,
    writer
):

    if data is None:
        return 0

    hourly = data.get(
        "hourly",
        {}
    )

    times = hourly.get(
        "time",
        []
    )

    temperatures = hourly.get(
        "temperature_2m_previous_day1",
        []
    )

    if not temperatures:

        print(
            f"    No temperature data returned for {model}"
        )

        return 0

    count = 0

    for i in range(len(times)):

        temperature = temperatures[i]

        if temperature is None:
            continue

        writer.writerow([
            location["City"],
            location["Latitude"],
            location["Longitude"],
            times[i],
            model,
            temperature
        ])

        count += 1

    return count


def save_era5_data(
    location,
    data,
    writer
):

    if data is None:
        return 0

    hourly = data.get(
        "hourly",
        {}
    )

    times = hourly.get(
        "time",
        []
    )

    temperatures = hourly.get(
        "temperature_2m",
        []
    )

    count = 0

    for i in range(len(times)):

        temperature = temperatures[i]

        if temperature is None:
            continue

        writer.writerow([
            location["City"],
            location["Latitude"],
            location["Longitude"],
            times[i],
            "ERA5_REFERENCE",
            temperature
        ])

        count += 1

    return count


def main():

    print("=" * 70)

    print(
        "     WEATHER MODEL HISTORICAL DATA COLLECTOR"
    )

    print(
        "     FIXED 1-DAY FORECAST LEAD TIME"
    )

    print("=" * 70)

    locations = read_locations()

    print(
        f"\nTotal locations: {len(locations)}"
    )

    print(
        f"Date range: {START_DATE} to {END_DATE}"
    )

    print("\nForecast variable:")
    print(
        "temperature_2m_previous_day1"
    )

    print("\nReference:")
    print("ERA5 temperature_2m")

    print("\nOutput:")
    print(
        "historical_forecast_data.csv"
    )

    if os.path.exists(OUTPUT_FILE):

        print(
            "\nExisting historical_forecast_data.csv found."
        )

        print(
            "It will be replaced."
        )

        os.remove(OUTPUT_FILE)

    total_rows = 0

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "location",
            "latitude",
            "longitude",
            "datetime",
            "model",
            "temperature"
        ])

        for location_number, location in enumerate(
            locations,
            start=1
        ):

            city = location["City"]

            latitude = float(
                location["Latitude"]
            )

            longitude = float(
                location["Longitude"]
            )

            print("\n")
            print("=" * 70)

            print(
                f"LOCATION {location_number}/{len(locations)}: {city}"
            )

            print(
                f"Latitude : {latitude}"
            )

            print(
                f"Longitude: {longitude}"
            )

            print("=" * 70)

            for model in MODELS:

                print(
                    f"\nFetching {model}..."
                )

                data = get_previous_run_temperature(
                    latitude,
                    longitude,
                    model
                )

                rows = save_forecast_data(
                    location,
                    model,
                    data,
                    writer
                )

                print(
                    f"    Saved {rows} rows"
                )

                total_rows += rows

                time.sleep(2)

            print(
                "\nFetching ERA5 reference..."
            )

            era5_data = get_era5_temperature(
                latitude,
                longitude
            )

            rows = save_era5_data(
                location,
                era5_data,
                writer
            )

            print(
                f"    Saved {rows} ERA5 rows"
            )

            total_rows += rows

            time.sleep(2)

    print("\n")
    print("=" * 70)

    print(
        "DATA COLLECTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nTotal rows collected: {total_rows}"
    )

    print("\nFile saved at:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()