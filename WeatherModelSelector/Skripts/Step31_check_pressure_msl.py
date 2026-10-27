# ================================================================
# STEP 31: CHECK PRESSURE MSL AGAINST IMD
# ================================================================

import os
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

ML_FILE = os.path.join(
    RESULTS_PATH,
    "live_pressure_msl_forecasts.csv"
)

IMD_FILE = os.path.join(
    DATA_PATH,
    "imd_latest_observations.csv"
)

print("=" * 70)
print("STEP 31: PRESSURE MSL VS IMD CHECK")
print("=" * 70)

# ------------------------------------------------
# LOAD FILES
# ------------------------------------------------

ml = pd.read_csv(ML_FILE)
imd = pd.read_csv(IMD_FILE)

# ------------------------------------------------
# CONVERT TIMES
# ------------------------------------------------

ml["forecast_datetime"] = pd.to_datetime(
    ml["forecast_datetime"],
    utc=True
)

imd["pressure_time"] = pd.to_datetime(
    imd["pressure_time"],
    utc=True,
    errors="coerce"
)

# ------------------------------------------------
# SHOW PRESSURE VALUES
# ------------------------------------------------

print("\nOPEN-METEO PRESSURE MSL")
print("-" * 70)

print(
    ml[
        [
            "location",
            "model",
            "forecast_datetime",
            "pressure_msl"
        ]
    ].to_string(index=False)
)

print("\nIMD PRESSURE")
print("-" * 70)

print(
    imd[
        [
            "location",
            "pressure",
            "pressure_time"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------
# MERGE COMMON LOCATIONS
# ------------------------------------------------

merged = ml.merge(
    imd[
        [
            "location",
            "pressure",
            "pressure_time"
        ]
    ],
    on="location",
    how="inner"
)

# ------------------------------------------------
# TIME DIFFERENCE
# ------------------------------------------------

merged["time_difference_hours"] = (
    (
        merged["forecast_datetime"]
        - merged["pressure_time"]
    )
    .abs()
    .dt.total_seconds()
    / 3600
)

# ------------------------------------------------
# PRESSURE ERROR
# ------------------------------------------------

merged["pressure_error"] = (
    merged["pressure_msl"]
    - merged["pressure"]
)

merged["absolute_pressure_error"] = (
    merged["pressure_error"].abs()
)

# ------------------------------------------------
# SHOW COMPARISON
# ------------------------------------------------

print("\n")
print("=" * 70)
print("PRESSURE COMPARISON")
print("=" * 70)

print(
    merged[
        [
            "location",
            "model",
            "forecast_datetime",
            "pressure_time",
            "time_difference_hours",
            "pressure_msl",
            "pressure",
            "absolute_pressure_error"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------
# STATISTICS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("PRESSURE ERROR STATISTICS")
print("=" * 70)

print(
    f"\nMean Absolute Error: "
    f"{merged['absolute_pressure_error'].mean():.3f} hPa"
)

print(
    f"Minimum Error: "
    f"{merged['absolute_pressure_error'].min():.3f} hPa"
)

print(
    f"Maximum Error: "
    f"{merged['absolute_pressure_error'].max():.3f} hPa"
)

# ------------------------------------------------
# TIME STATISTICS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("TIME ALIGNMENT STATISTICS")
print("=" * 70)

print(
    f"\nCommon locations: "
    f"{merged['location'].nunique()}"
)

print(
    f"Rows compared: "
    f"{len(merged)}"
)

print(
    f"Mean time difference: "
    f"{merged['time_difference_hours'].mean():.2f} hours"
)

print(
    f"Maximum time difference: "
    f"{merged['time_difference_hours'].max():.2f} hours"
)

print("\nTime difference distribution:")

print(
    merged[
        "time_difference_hours"
    ].value_counts().sort_index().to_string()
)

# ------------------------------------------------
# BEST PRESSURE MODEL
# ------------------------------------------------

print("\n")
print("=" * 70)
print("BEST MODEL BY CURRENT PRESSURE ERROR")
print("=" * 70)

valid = merged.dropna(
    subset=[
        "pressure",
        "pressure_msl"
    ]
)

if len(valid) > 0:

    model_errors = (
        valid
        .groupby("model")["absolute_pressure_error"]
        .mean()
        .sort_values()
    )

    print(model_errors.to_string())

# ------------------------------------------------
# CONCLUSION
# ------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 31 COMPLETED")
print("=" * 70)

print("\nSend me the complete output.")