# ================================================================
# STEP 28B: REAL-TIME IMPROVED ML VS IMD COMPARISON
# ================================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

ML_FILE = os.path.join(
    RESULTS_PATH,
    "live_improved_predictions.csv"
)

IMD_FILE = os.path.join(
    DATA_PATH,
    "imd_latest_observations.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_improved_ml_vs_imd.csv"
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("=" * 70)
print("STEP 28B: IMPROVED ML VS IMD")
print("=" * 70)

ml = pd.read_csv(ML_FILE)
imd = pd.read_csv(IMD_FILE)

print("\nML rows :", len(ml))
print("IMD rows:", len(imd))

# ------------------------------------------------
# COMMON LOCATIONS
# ------------------------------------------------

common = sorted(
    set(ml["location"]) &
    set(imd["location"])
)

print("\nCommon locations:", len(common))
print(common)

# ------------------------------------------------
# MERGE
# ------------------------------------------------

df = ml.merge(
    imd,
    on="location",
    how="inner",
    suffixes=("_ml", "_imd")
)

print("\nMatched rows:", len(df))

# ------------------------------------------------
# MODEL NAMES
# ------------------------------------------------

models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

# ------------------------------------------------
# CALCULATE ERRORS
# ------------------------------------------------

parameters = {
    "temperature": "temperature",
    "humidity": "relative_humidity",
    "wind_speed": "wind_speed",
    "pressure": "pressure"
}

results = []

# ------------------------------------------------
# PROCESS EACH LOCATION
# ------------------------------------------------

for _, row in df.iterrows():

    location = row["location"]

    print("\n" + "-" * 70)
    print("LOCATION:", location)

    result = {
        "location": location
    }

    # ============================================================
    # TEMPERATURE
    # ============================================================

    actual_temp = row["temperature"]

    temp_errors = {}

    for model in models:

        col = model + "_temperature"

        forecast = row[col]

        if pd.notna(forecast) and pd.notna(actual_temp):
            temp_errors[model] = abs(forecast - actual_temp)

    if temp_errors:

        actual_best = min(
            temp_errors,
            key=temp_errors.get
        )

        ml_selected = row["best_temperature_model"]

        correct = ml_selected == actual_best

        result["temperature_ml_model"] = ml_selected
        result["temperature_actual_best"] = actual_best
        result["temperature_correct"] = correct

        for model in models:
            result[f"temperature_error_{model}"] = temp_errors.get(
                model, np.nan
            )

        print("\nTemperature")
        print("IMD:", actual_temp)
        print("ML selected:", ml_selected)
        print("Actual best:", actual_best)
        print("Errors:", temp_errors)
        print("Correct:", correct)

    # ============================================================
    # HUMIDITY
    # ============================================================

    actual_humidity = row["relative_humidity"]

    humidity_errors = {}

    for model in models:

        col = model + "_humidity"

        forecast = row[col]

        if pd.notna(forecast) and pd.notna(actual_humidity):
            humidity_errors[model] = abs(
                forecast - actual_humidity
            )

    if humidity_errors:

        actual_best = min(
            humidity_errors,
            key=humidity_errors.get
        )

        ml_selected = row["best_humidity_model"]

        correct = ml_selected == actual_best

        result["humidity_ml_model"] = ml_selected
        result["humidity_actual_best"] = actual_best
        result["humidity_correct"] = correct

        for model in models:
            result[f"humidity_error_{model}"] = humidity_errors.get(
                model, np.nan
            )

        print("\nHumidity")
        print("IMD:", actual_humidity)
        print("ML selected:", ml_selected)
        print("Actual best:", actual_best)
        print("Errors:", humidity_errors)
        print("Correct:", correct)

    # ============================================================
    # WIND SPEED
    # ============================================================

    actual_wind = row["wind_speed"]

    wind_errors = {}

    for model in models:

        col = model + "_wind_speed"

        forecast = row[col]

        if pd.notna(forecast) and pd.notna(actual_wind):
            wind_errors[model] = abs(
                forecast - actual_wind
            )

    if wind_errors:

        actual_best = min(
            wind_errors,
            key=wind_errors.get
        )

        ml_selected = row["best_wind_speed_model"]

        correct = ml_selected == actual_best

        result["wind_speed_ml_model"] = ml_selected
        result["wind_speed_actual_best"] = actual_best
        result["wind_speed_correct"] = correct

        for model in models:
            result[f"wind_speed_error_{model}"] = wind_errors.get(
                model, np.nan
            )

        print("\nWind Speed")
        print("IMD:", actual_wind)
        print("ML selected:", ml_selected)
        print("Actual best:", actual_best)
        print("Errors:", wind_errors)
        print("Correct:", correct)

    # ============================================================
    # PRESSURE
    # ============================================================

    actual_pressure = row["pressure"]

    pressure_errors = {}

    for model in models:

        col = model + "_pressure"

        forecast = row[col]

        if pd.notna(forecast) and pd.notna(actual_pressure):
            pressure_errors[model] = abs(
                forecast - actual_pressure
            )

    if pressure_errors:

        actual_best = min(
            pressure_errors,
            key=pressure_errors.get
        )

        ml_selected = row["best_pressure_model"]

        correct = ml_selected == actual_best

        result["pressure_ml_model"] = ml_selected
        result["pressure_actual_best"] = actual_best
        result["pressure_correct"] = correct

        for model in models:
            result[f"pressure_error_{model}"] = pressure_errors.get(
                model, np.nan
            )

        print("\nPressure")
        print("IMD:", actual_pressure)
        print("ML selected:", ml_selected)
        print("Actual best:", actual_best)
        print("Errors:", pressure_errors)
        print("Correct:", correct)

    results.append(result)

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL REAL-TIME VALIDATION")
print("=" * 70)

for parameter in [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]:

    correct_col = parameter + "_correct"

    if correct_col in results_df.columns:

        valid = results_df[correct_col].dropna()

        if len(valid) > 0:

            accuracy = valid.mean() * 100

            print(
                f"\n{parameter.upper()}"
            )

            print(
                f"ML Selection Accuracy: "
                f"{accuracy:.2f}%"
            )

            print(
                f"Correct: "
                f"{int(valid.sum())}/{len(valid)}"
            )

# ------------------------------------------------
# AVERAGE MODEL ERRORS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("AVERAGE ERROR OF EACH MODEL")
print("=" * 70)

for parameter in [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]:

    print(f"\n{parameter.upper()}")

    for model in models:

        col = f"{parameter}_error_{model}"

        if col in results_df.columns:

            value = results_df[col].mean()

            print(
                f"{model:15s}: {value:.3f}"
            )

# ------------------------------------------------
# MODEL ACTUAL BEST COUNTS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("ACTUAL BEST MODEL COUNTS")
print("=" * 70)

for parameter in [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]:

    col = parameter + "_actual_best"

    if col in results_df.columns:

        print("\n" + parameter.upper())

        print(
            results_df[col]
            .value_counts()
            .to_string()
        )

print("\n")
print("=" * 70)
print("STEP 28 COMPLETED")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)