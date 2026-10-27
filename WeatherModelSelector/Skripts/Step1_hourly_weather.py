import requests


# --------------------------------------------------
# STEP 1: Get latitude and longitude from location
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
        print("Error while finding location.")
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
# STEP 2: Get hourly weather from Open-Meteo
# --------------------------------------------------

def get_hourly_weather(latitude, longitude, timezone):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "pressure_msl"
        ]),

        "forecast_days": 1,
        "timezone": timezone
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error while getting weather data.")
        print(response.text)
        return None

    return response.json()


# --------------------------------------------------
# STEP 3: Display hourly weather
# --------------------------------------------------

def display_weather(data, location):

    hourly = data["hourly"]

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    precipitation = hourly["precipitation"]
    wind_speed = hourly["wind_speed_10m"]
    pressure = hourly["pressure_msl"]

    print("\n" + "=" * 80)
    print(f"HOURLY WEATHER - {location['name']}, {location['country']}")
    print("=" * 80)

    print(
        f"{'Time':<20}"
        f"{'Temp (°C)':<12}"
        f"{'Humidity (%)':<15}"
        f"{'Rain (mm)':<12}"
        f"{'Wind (km/h)':<15}"
        f"{'Pressure':<12}"
    )

    print("-" * 80)

    for i in range(len(times)):

        print(
            f"{times[i]:<20}"
            f"{temperatures[i]:<12}"
            f"{humidity[i]:<15}"
            f"{precipitation[i]:<12}"
            f"{wind_speed[i]:<15}"
            f"{pressure[i]:<12}"
        )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 50)
    print("     OPEN-METEO HOURLY WEATHER")
    print("=" * 50)

    location_name = input("\nEnter location: ")

    # Find coordinates
    location = get_coordinates(location_name)

    if location is None:
        return

    print("\nLocation found!")
    print(f"Name      : {location['name']}")
    print(f"Country   : {location['country']}")
    print(f"Latitude  : {location['latitude']}")
    print(f"Longitude : {location['longitude']}")
    print(f"Timezone  : {location['timezone']}")

    # Get weather
    weather_data = get_hourly_weather(
        location["latitude"],
        location["longitude"],
        location["timezone"]
    )

    if weather_data is None:
        return

    # Display weather
    display_weather(weather_data, location)


if __name__ == "__main__":
    main()