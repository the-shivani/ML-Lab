import requests
import csv
import os


# --------------------------------------------------
# GET LOCATION COORDINATES
# --------------------------------------------------

def get_coordinates(location):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error finding location.")
        return None

    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        print("Location not found.")
        return None

    result = data["results"][0]

    return {
        "name": result["name"],
        "country": result.get("country", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "auto")
    }


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
# DISPLAY MODEL DATA
# --------------------------------------------------

def display_model_data(model_name, data):

    print("\n")
    print("=" * 90)
    print(f"MODEL: {model_name}")
    print("=" * 90)

    hourly = data["hourly"]

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    precipitation = hourly["precipitation"]
    wind_speed = hourly["wind_speed_10m"]
    pressure = hourly["pressure_msl"]

    print(
        f"{'Time':<20}"
        f"{'Temp':<10}"
        f"{'Humidity':<12}"
        f"{'Rain':<10}"
        f"{'Wind':<12}"
        f"{'Pressure':<12}"
    )

    print("-" * 90)

    for i in range(len(times)):

        print(
            f"{times[i]:<20}"
            f"{temperatures[i]:<10}"
            f"{humidity[i]:<12}"
            f"{precipitation[i]:<10}"
            f"{wind_speed[i]:<12}"
            f"{pressure[i]:<12}"
        )


# --------------------------------------------------
# SAVE DATA TO CSV
# --------------------------------------------------

def save_to_csv(location, model_name, data):

    filename  = r"C:\Projects\WeatherModelSelector\weather_data.csv"

    hourly = data["hourly"]

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    precipitation = hourly["precipitation"]
    wind_speed = hourly["wind_speed_10m"]
    pressure = hourly["pressure_msl"]

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

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

        for i in range(len(times)):

            writer.writerow([
                location["name"],
                location["latitude"],
                location["longitude"],
                times[i],
                model_name,
                temperatures[i],
                humidity[i],
                precipitation[i],
                wind_speed[i],
                pressure[i]
            ])


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("       MULTI-MODEL OPEN-METEO WEATHER")
    print("=" * 60)

    location_name = input("\nEnter location: ")

    location = get_coordinates(location_name)

    if location is None:
        return

    print("\nLocation:")
    print(f"Name      : {location['name']}")
    print(f"Country   : {location['country']}")
    print(f"Latitude  : {location['latitude']}")
    print(f"Longitude : {location['longitude']}")

    # Models for our first experiment
    models = [
        "ecmwf_ifs",
        "gfs_seamless",
        "icon_seamless",
        "gem_seamless"
    ]

    print("\nFetching weather data...")

    for model in models:

        print(f"\nFetching {model}...")

        data = get_weather_model(
            location["latitude"],
            location["longitude"],
            location["timezone"],
            model
        )

        if data is not None:

            display_model_data(model, data)

            save_to_csv(
                location,
                model,
                data
            )

            print(f"{model} data saved to CSV.")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()