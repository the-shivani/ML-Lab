import pandas as pd
import numpy as np
import os


# ================================================================
# STEP 7 - CREATE IMPROVED ML TRAINING DATA
# ================================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_forecast_data.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "training_data_v2.csv"
)
print("=" * 70)
print("     STEP 7 - CREATE IMPROVED ML TRAINING DATA")
print("=" * 70)


# ================================================================
# 1. LOAD HISTORICAL DATA
# ================================================================

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded:", len(df))


# ================================================================
# 2. CONVERT DATETIME
# ================================================================

df["datetime"] = pd.to_datetime(
    df["datetime"]
)


# ================================================================
# 3. SEPARATE ERA5 REFERENCE
# ================================================================

reference = df[
    df["model"] == "ERA5_REFERENCE"
][
    [
        "location",
        "datetime",
        "temperature"
    ]
].copy()


reference = reference.rename(
    columns={
        "temperature": "actual_temperature"
    }
)


print(
    "ERA5 reference rows:",
    len(reference)
)


# ================================================================
# 4. SELECT FORECAST DATA
# ================================================================

forecasts = df[
    df["model"] != "ERA5_REFERENCE"
].copy()


print(
    "Forecast rows:",
    len(forecasts)
)


# ================================================================
# 5. MERGE FORECASTS WITH ERA5
# ================================================================

merged = forecasts.merge(
    reference,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


print(
    "Matched forecast rows:",
    len(merged)
)


# ================================================================
# 6. CALCULATE FORECAST ERROR
# ================================================================

merged["error"] = (
    merged["temperature"]
    - merged["actual_temperature"]
).abs()


# ================================================================
# 7. SORT CHRONOLOGICALLY
# ================================================================

merged = merged.sort_values(
    [
        "location",
        "datetime"
    ]
).reset_index(drop=True)


# ================================================================
# 8. CREATE PREVIOUS-HOUR ERROR
# ================================================================

merged["previous_error"] = (
    merged
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .shift(1)
)


# ================================================================
# 9. CREATE 3-HOUR ROLLING ERROR
# ================================================================

merged["rolling_error_3h"] = (
    merged
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )
)


# ================================================================
# 10. CREATE 6-HOUR ROLLING ERROR
# ================================================================

merged["rolling_error_6h"] = (
    merged
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=6,
            min_periods=1
        )
        .mean()
    )
)


# ================================================================
# 11. CREATE 12-HOUR ROLLING ERROR
# ================================================================

merged["rolling_error_12h"] = (
    merged
    .groupby(
        [
            "location",
            "model"
        ]
    )["error"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=12,
            min_periods=1
        )
        .mean()
    )
)


# ================================================================
# 12. CREATE FORECAST TABLE
# ================================================================

forecast_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="temperature"
).reset_index()


forecast_table = forecast_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_temperature",

        "gfs_seamless":
            "gfs_temperature",

        "icon_seamless":
            "icon_temperature",

        "gem_seamless":
            "gem_temperature"
    }
)


# ================================================================
# 13. CREATE ERROR TABLE
# ================================================================

error_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="error"
).reset_index()


error_table = error_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_error",

        "gfs_seamless":
            "gfs_error",

        "icon_seamless":
            "icon_error",

        "gem_seamless":
            "gem_error"
    }
)


# ================================================================
# 14. CREATE PREVIOUS ERROR TABLE
# ================================================================

previous_error_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="previous_error"
).reset_index()


previous_error_table = previous_error_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_previous_error",

        "gfs_seamless":
            "gfs_previous_error",

        "icon_seamless":
            "icon_previous_error",

        "gem_seamless":
            "gem_previous_error"
    }
)


# ================================================================
# 15. CREATE 3-HOUR ROLLING ERROR TABLE
# ================================================================

rolling3_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="rolling_error_3h"
).reset_index()


rolling3_table = rolling3_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_rolling_error_3h",

        "gfs_seamless":
            "gfs_rolling_error_3h",

        "icon_seamless":
            "icon_rolling_error_3h",

        "gem_seamless":
            "gem_rolling_error_3h"
    }
)


# ================================================================
# 16. CREATE 6-HOUR ROLLING ERROR TABLE
# ================================================================

rolling6_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="rolling_error_6h"
).reset_index()


rolling6_table = rolling6_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_rolling_error_6h",

        "gfs_seamless":
            "gfs_rolling_error_6h",

        "icon_seamless":
            "icon_rolling_error_6h",

        "gem_seamless":
            "gem_rolling_error_6h"
    }
)


# ================================================================
# 17. CREATE 12-HOUR ROLLING ERROR TABLE
# ================================================================

rolling12_table = merged.pivot_table(
    index=[
        "location",
        "datetime"
    ],
    columns="model",
    values="rolling_error_12h"
).reset_index()


rolling12_table = rolling12_table.rename(
    columns={
        "ecmwf_ifs":
            "ecmwf_rolling_error_12h",

        "gfs_seamless":
            "gfs_rolling_error_12h",

        "icon_seamless":
            "icon_rolling_error_12h",

        "gem_seamless":
            "gem_rolling_error_12h"
    }
)


# ================================================================
# 18. CREATE BEST MODEL TARGET
# ================================================================

error_columns = [
    "ecmwf_error",
    "gfs_error",
    "icon_error",
    "gem_error"
]


model_names = {
    "ecmwf_error":
        "ecmwf_ifs",

    "gfs_error":
        "gfs_seamless",

    "icon_error":
        "icon_seamless",

    "gem_error":
        "gem_seamless"
}


target_table = error_table.copy()


target_table["best_model"] = (
    target_table[
        error_columns
    ]
    .idxmin(axis=1)
    .map(model_names)
)


# ================================================================
# 19. MERGE ALL TABLES
# ================================================================

training = forecast_table.merge(
    previous_error_table,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


training = training.merge(
    rolling3_table,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


training = training.merge(
    rolling6_table,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


training = training.merge(
    rolling12_table,
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


training = training.merge(
    target_table[
        [
            "location",
            "datetime",
            "best_model"
        ]
    ],
    on=[
        "location",
        "datetime"
    ],
    how="inner"
)


# ================================================================
# 20. CREATE TIME FEATURES
# ================================================================

training["datetime"] = pd.to_datetime(
    training["datetime"]
)


training["hour"] = (
    training["datetime"].dt.hour
)


training["day"] = (
    training["datetime"].dt.day
)


training["day_of_week"] = (
    training["datetime"].dt.dayofweek
)


training["month"] = (
    training["datetime"].dt.month
)


# ================================================================
# 21. CREATE FORECAST AGREEMENT FEATURES
# ================================================================

temperature_columns = [
    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature"
]


training["forecast_mean"] = (
    training[
        temperature_columns
    ].mean(axis=1)
)


training["forecast_max"] = (
    training[
        temperature_columns
    ].max(axis=1)
)


training["forecast_min"] = (
    training[
        temperature_columns
    ].min(axis=1)
)


training["forecast_spread"] = (
    training["forecast_max"]
    - training["forecast_min"]
)


# ================================================================
# 22. ADD LATITUDE AND LONGITUDE
# ================================================================

locations = pd.read_csv(
    os.path.join(
        DATA_PATH,
        "Locations.csv"
    )
)

locations = pd.read_csv(
    locations_file
)


locations.columns = [
    column.strip().lower()
    for column in locations.columns
]


locations = locations.rename(
    columns={
        "city": "location"
    }
)


training = training.merge(
    locations[
        [
            "location",
            "latitude",
            "longitude"
        ]
    ],
    on="location",
    how="left"
)


# ================================================================
# 23. SORT FINAL DATA
# ================================================================

training = training.sort_values(
    [
        "datetime",
        "location"
    ]
).reset_index(drop=True)


# ================================================================
# 24. REMOVE FIRST ROWS WITH NO PAST INFORMATION
# ================================================================

past_features = [
    "ecmwf_previous_error",
    "gfs_previous_error",
    "icon_previous_error",
    "gem_previous_error"
]


before = len(training)


training = training.dropna(
    subset=past_features
).reset_index(drop=True)


after = len(training)


print(
    "\nRows removed because previous-hour "
    "performance was unavailable:",
    before - after
)


# ================================================================
# 25. CHECK DATA
# ================================================================

print(
    "\nFinal training rows:",
    len(training)
)


print(
    "\nFinal columns:"
)


print(
    list(training.columns)
)


print(
    "\nBest model distribution:"
)


print(
    training["best_model"]
    .value_counts()
)


print(
    "\nMissing values in past-performance features:"
)


print(
    training[
        [
            "ecmwf_previous_error",
            "gfs_previous_error",
            "icon_previous_error",
            "gem_previous_error",
            "ecmwf_rolling_error_3h",
            "gfs_rolling_error_3h",
            "icon_rolling_error_3h",
            "gem_rolling_error_3h",
            "ecmwf_rolling_error_6h",
            "gfs_rolling_error_6h",
            "icon_rolling_error_6h",
            "gem_rolling_error_6h",
            "ecmwf_rolling_error_12h",
            "gfs_rolling_error_12h",
            "icon_rolling_error_12h",
            "gem_rolling_error_12h"
        ]
    ]
    .isnull()
    .sum()
)


# ================================================================
# 26. SAVE
# ================================================================

training.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nTraining data v2 created successfully."
)


print(
    "\nSaved file:"
)


print(
    OUTPUT_FILE
)


print("\n")


print("=" * 70)

print("STEP 7 COMPLETED")

print("=" * 70)