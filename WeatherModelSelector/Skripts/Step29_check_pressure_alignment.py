# ================================================================
# STEP 29: CHECK PRESSURE ALIGNMENT
# ================================================================

import os
import pandas as pd

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

print("=" * 70)
print("STEP 29: PRESSURE ALIGNMENT CHECK")
print("=" * 70)

# ------------------------------------------------
# LOAD FILES
# ------------------------------------------------

ml = pd.read_csv(ML_FILE)
imd = pd.read_csv(IMD_FILE)

# ------------------------------------------------
# SHOW PRESSURE VALUES
# ------------------------------------------------

pressure_columns = [
    "ecmwf_ifs_pressure",
    "gfs_seamless_pressure",
    "icon_seamless_pressure",
    "gem_seamless_pressure"
]

print("\nOPEN-METEO PRESSURE VALUES")
print("-" * 70)

print(
    ml[
        ["location"] + pressure_columns
    ].head(17).to_string(index=False)
)

print("\nIMD PRESSURE VALUES")
print("-" * 70)

print(
    imd[
        ["location", "pressure", "pressure_time"]
    ].to_string(index=False)
)

# ------------------------------------------------
# STATISTICS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("PRESSURE STATISTICS")
print("=" * 70)

print("\nOpen-Meteo model pressures:")

for col in pressure_columns:

    print(
        f"{col:25s} "
        f"min={ml[col].min():.2f} "
        f"max={ml[col].max():.2f} "
        f"mean={ml[col].mean():.2f}"
    )

print("\nIMD pressure:")

print(
    f"IMD pressure "
    f"min={imd['pressure'].min():.2f} "
    f"max={imd['pressure'].max():.2f} "
    f"mean={imd['pressure'].mean():.2f}"
)

# ------------------------------------------------
# TIME CHECK
# ------------------------------------------------

print("\n")
print("=" * 70)
print("TIME CHECK")
print("=" * 70)

print("\nML forecast times:")

print(
    ml[
        ["location", "forecast_datetime"]
    ].head(17).to_string(index=False)
)

print("\nIMD pressure observation times:")

print(
    imd[
        ["location", "pressure_time"]
    ].to_string(index=False)
)

# ------------------------------------------------
# CONCLUSION
# ------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 29 COMPLETED")
print("=" * 70)

print("\nPlease send me this complete output.")