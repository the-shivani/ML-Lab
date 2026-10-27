import pandas as pd
import os

# ============================================================
# STEP 14 - CREATE MULTI-PARAMETER TRAINING DATA
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_data.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

PARAMETERS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 14 - MULTI-PARAMETER TRAINING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

print()
print("Raw rows:", len(df))

# ============================================================
# LOAD ERA5 REFERENCE
# ============================================================

print()
print("Loading ERA5 reference data...")

reference_file = os.path.join(
    DATA_PATH,
    "historical_forecast_data.csv"
)

reference = pd.read_csv(
    reference_file
)

reference["datetime"] = pd.to_datetime(
    reference["datetime"]
)

# Keep only ERA5 reference rows
reference = reference[
    reference["model"] == "ERA5_REFERENCE"
].copy()

print(
    "ERA5 reference rows:",
    len(reference)
)

# ============================================================
# IMPORTANT
# ============================================================
# The current ERA5 file contains temperature reference.
#
# For the first version, temperature can be evaluated
# against ERA5 directly.
#
# For the other parameters, we will use the same
# historical forecast structure and later connect
# official observations for real verification.
# ============================================================

# ============================================================
# TEMPERATURE TRAINING DATA
# ============================================================

print()
print("Creating temperature training data...")

temperature = df[
    [
        "location",
        "latitude",
        "longitude",
        "datetime",
        "model",
        "temperature"
    ]
].copy()

temperature = temperature.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    columns="model",
    values="temperature"
).reset_index()

temperature.columns.name = None

temperature = temperature.rename(
    columns={
        "ecmwf_ifs": "ecmwf_temperature",
        "gfs_seamless": "gfs_temperature",
        "icon_seamless": "icon_temperature",
        "gem_seamless": "gem_temperature"
    }
)

# ============================================================
# MERGE ERA5 TEMPERATURE
# ============================================================

era5_temp = reference[
    [
        "location",
        "datetime",
        "temperature"
    ]
].copy()

era5_temp = era5_temp.rename(
    columns={
        "temperature": "reference_temperature"
    }
)

temperature = temperature.merge(
    era5_temp,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)

# ============================================================
# CALCULATE ERRORS
# ============================================================

for model in MODELS:

    short = model.split("_")[0]

    temperature[
        f"{short}_temperature_error"
    ] = (
        temperature[
            f"{short}_temperature"
        ]
        - temperature["reference_temperature"]
    ).abs()

# ============================================================
# FIND BEST TEMPERATURE MODEL
# ============================================================

error_columns = [
    "ecmwf_temperature_error",
    "gfs_temperature_error",
    "icon_temperature_error",
    "gem_temperature_error"
]

model_names = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

temperature["best_temperature_model"] = (
    temperature[error_columns]
    .idxmin(axis=1)
    .map(
        dict(
            zip(
                error_columns,
                model_names
            )
        )
    )
)

# ============================================================
# TIME FEATURES
# ============================================================

temperature["hour"] = (
    temperature["datetime"].dt.hour
)

temperature["day"] = (
    temperature["datetime"].dt.day
)

temperature["day_of_week"] = (
    temperature["datetime"].dt.dayofweek
)

temperature["month"] = (
    temperature["datetime"].dt.month
)

# ============================================================
# FORECAST STATISTICS
# ============================================================

forecast_columns = [
    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature"
]

temperature["forecast_mean"] = (
    temperature[forecast_columns].mean(axis=1)
)

temperature["forecast_spread"] = (
    temperature[forecast_columns].max(axis=1)
    -
    temperature[forecast_columns].min(axis=1)
)

# ============================================================
# SAVE
# ============================================================

temperature.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("TEMPERATURE TRAINING DATA CREATED")
print("=" * 70)

print()
print("Rows:", len(temperature))

print()
print("Columns:")
print(
    list(temperature.columns)
)

print()
print("Best model distribution:")

print(
    temperature[
        "best_temperature_model"
    ].value_counts()
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 70)