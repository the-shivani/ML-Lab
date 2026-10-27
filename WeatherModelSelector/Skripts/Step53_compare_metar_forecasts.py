import os
import numpy as np
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"
RESULTS = os.path.join(BASE, "results")
OBS = os.path.join(BASE, "data", "observations")

forecast = pd.read_csv(
    os.path.join(RESULTS, "step49B_live_31locations_predictions.csv"),
    parse_dates=["datetime"]
)

observations = pd.read_csv(
    os.path.join(OBS, "metar_observations.csv")
)

# The original forecast datetimes are in IST.
forecast["valid_utc"] = (
    pd.to_datetime(forecast["datetime"])
    .dt.tz_localize("Asia/Kolkata")
    .dt.tz_convert("UTC")
)

observations["obs_utc"] = pd.to_datetime(
    observations["observation_time_utc"],
    utc=True
)

CITY_STATIONS = {
    "Mumbai": "VABB",
    "Delhi": "VIDP",
    "Bengaluru": "VOBL"
}

PARAMETERS = {
    "temperature": "temperature",
    "wind_speed": "wind_speed_kmh"
}

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    a1, a2 = np.radians([lat1, lat2])
    dlat = a2 - a1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(a1) * np.cos(a2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


comparisons = []

for city, station in CITY_STATIONS.items():

    print("\n" + "=" * 60)
    print("CITY:", city, "| STATION:", station)
    print("=" * 60)

    city_forecasts = forecast[
        forecast["location"] == city
    ].copy()

    station_obs = observations[
        observations["station"] == station
    ].copy()

    if city_forecasts.empty or station_obs.empty:
        print("No usable forecast or station observations.")
        continue

    # Check physical separation between forecast location
    # and airport observation station.
    station_sample = station_obs.dropna(
        subset=["latitude", "longitude"]
    )

    if station_sample.empty:
        print("Station coordinates missing; skipping.")
        continue

    f0 = city_forecasts.iloc[0]
    o0 = station_sample.iloc[0]

    distance = distance_km(
        f0["latitude"], f0["longitude"],
        float(o0["latitude"]), float(o0["longitude"])
    )

    print("Station distance (km):", round(distance, 2))

    if distance > 40:
        print("Skipping: station is over 40 km away.")
        continue

    for parameter, observed_column in PARAMETERS.items():

        f = city_forecasts[
            city_forecasts["parameter"] == parameter
        ].copy()

        o = station_obs.dropna(
            subset=[observed_column]
        ).copy()

        if f.empty or o.empty:
            print(parameter, ": no observations")
            continue

        # Use observations as the left-hand table to avoid
        # giving multiple forecast hours the same observation.
        # One observation may still be reported more than once
        # by the source; remove duplicate station timestamps.
        o = (
            o.sort_values("obs_utc")
            .drop_duplicates(subset=["obs_utc"])
        )

        # Keep forecast-valid hours distinct.
        f = (
            f.sort_values("valid_utc")
            .drop_duplicates(subset=["valid_utc"])
        )

        matched = pd.merge_asof(
            o,
            f,
            left_on="obs_utc",
            right_on="valid_utc",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=30),
            suffixes=("_station", "_forecast")
        )

        matched = matched.dropna(
            subset=["valid_utc", "predicted_value"]
        )

        if matched.empty:
            print(parameter, ": no matching forecast hours")
            continue

        # Keep the nearest observation per forecast-valid hour.
        matched["time_gap_minutes"] = (
            (matched["obs_utc"] - matched["valid_utc"])
            .abs()
            .dt.total_seconds() / 60
        )

        matched = (
            matched.sort_values("time_gap_minutes")
            .drop_duplicates(subset=["valid_utc"])
        )

        for _, row in matched.iterrows():

            observed = float(row[observed_column])

            record = {
                "location": city,
                "station": station,
                "parameter": parameter,
                "station_distance_km": round(distance, 2),
                "observation_time_utc": row["obs_utc"],
                "forecast_valid_utc": row["valid_utc"],
                "time_gap_minutes": row["time_gap_minutes"],
                "observed_value": observed,
                "selected_model": row["selected_model"],
                "selected_forecast": row["predicted_value"],
                "selected_abs_error": abs(
                    row["predicted_value"] - observed
                )
            }

            for model in MODELS:
                value = row[f"{model}_value"]

                record[f"{model}_abs_error"] = abs(
                    value - observed
                ) if pd.notna(value) else np.nan

            comparisons.append(record)

        print(
            parameter,
            "matched hours:",
            len(matched)
        )


if not comparisons:
    print("\nNo matched observations found.")
    raise SystemExit(0)

result = pd.DataFrame(comparisons)

output = os.path.join(
    RESULTS, "step53_metar_actual_errors.csv"
)

result.to_csv(output, index=False)

print("\n" + "=" * 65)
print("STEP 53 - MATCHED OBSERVATION SUMMARY")
print("=" * 65)

print("Total matched records:", len(result))

print("\nRecords by city and parameter:")
print(
    result.groupby(
        ["location", "parameter"]
    ).size().to_string()
)

error_columns = (
    ["selected_abs_error"] +
    [f"{m}_abs_error" for m in MODELS]
)

print("\nInitial MAE (exploratory only):")

summary = (
    result.groupby(["location", "parameter"])[error_columns]
    .agg(["mean", "count"])
)

print(summary.round(3).to_string())

print("\nIMPORTANT:")
print(
    "These are preliminary station comparisons. "
    "Do not claim verified forecast skill until "
    "forecast issue time, spatial matching, "
    "measurement definitions and adequate sample "
    "size have been validated."
)

print("\nSaved:", output)
print("STEP 53 COMPLETE")
