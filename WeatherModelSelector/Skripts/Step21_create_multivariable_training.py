import pandas as pd
import numpy as np
import os

print("=" * 70)
print("STEP 21 - CREATE MULTIVARIABLE TRAINING DATA")
print("=" * 70)

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

FORECAST_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast.csv"
)

REFERENCE_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_all_parameters.csv"
)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("\nLoading historical forecast data...")

forecast = pd.read_csv(FORECAST_FILE)

print("Forecast rows:", len(forecast))

print("\nLoading ERA5 reference data...")

reference = pd.read_csv(REFERENCE_FILE)

print("Reference rows:", len(reference))


# ---------------------------------------------------------
# 2. CONVERT DATETIME
# ---------------------------------------------------------

forecast["datetime"] = pd.to_datetime(
    forecast["datetime"]
)

reference["datetime"] = pd.to_datetime(
    reference["datetime"]
)


# ---------------------------------------------------------
# 3. PIVOT FOUR WEATHER MODELS
# ---------------------------------------------------------

print("\nCreating model forecast columns...")

parameters = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]


pivoted = forecast.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    columns="model",
    values=parameters
)

pivoted.columns = [
    f"{model}_{parameter}"
    for parameter, model in pivoted.columns
]

pivoted = pivoted.reset_index()


# ---------------------------------------------------------
# 4. MERGE WITH ERA5
# ---------------------------------------------------------

print("\nMerging with ERA5 reference...")

merged = pivoted.merge(
    reference,
    on=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    how="inner"
)

print("Merged rows:", len(merged))


# ---------------------------------------------------------
# 5. CALCULATE ERRORS
# ---------------------------------------------------------

print("\nCalculating model errors...")

for parameter in parameters:

    reference_column = (
        f"reference_{parameter}"
    )

    for model in models:

        forecast_column = (
            f"{model}_{parameter}"
        )

        error_column = (
            f"{model}_{parameter}_error"
        )

        merged[error_column] = (
            merged[forecast_column]
            - merged[reference_column]
        ).abs()


# ---------------------------------------------------------
# 6. FIND BEST MODEL FOR EACH PARAMETER
# ---------------------------------------------------------

print("\nFinding best model for each parameter...")

for parameter in parameters:

    error_columns = [
        f"{model}_{parameter}_error"
        for model in models
    ]

    model_names = [
        "ecmwf_ifs",
        "gfs_seamless",
        "icon_seamless",
        "gem_seamless"
    ]

    merged[
        f"best_{parameter}_model"
    ] = (
        merged[error_columns]
        .idxmin(axis=1)
        .str.replace(
            f"_{parameter}_error",
            "",
            regex=False
        )
    )


# ---------------------------------------------------------
# 7. TIME FEATURES
# ---------------------------------------------------------

print("\nCreating time features...")

merged["hour"] = merged["datetime"].dt.hour

merged["day"] = merged["datetime"].dt.day

merged["day_of_week"] = (
    merged["datetime"].dt.dayofweek
)

merged["month"] = (
    merged["datetime"].dt.month
)


# ---------------------------------------------------------
# 8. FORECAST STATISTICS
# ---------------------------------------------------------

for parameter in parameters:

    model_columns = [
        f"{model}_{parameter}"
        for model in models
    ]

    merged[
        f"{parameter}_forecast_mean"
    ] = merged[model_columns].mean(axis=1)

    merged[
        f"{parameter}_forecast_spread"
    ] = merged[model_columns].max(axis=1) - \
        merged[model_columns].min(axis=1)


# ---------------------------------------------------------
# 9. CHECK DATA
# ---------------------------------------------------------

print("\nChecking missing values...")

print(
    "Total missing values:",
    merged.isna().sum().sum()
)

print(
    "Total rows:",
    len(merged)
)

print(
    "Total locations:",
    merged["location"].nunique()
)


# ---------------------------------------------------------
# 10. SHOW BEST MODEL DISTRIBUTIONS
# ---------------------------------------------------------

for parameter in parameters:

    print(
        f"\n{parameter.upper()} BEST MODEL DISTRIBUTION:"
    )

    print(
        merged[
            f"best_{parameter}_model"
        ].value_counts()
    )


# ---------------------------------------------------------
# 11. SAVE
# ---------------------------------------------------------

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("TRAINING DATA CREATED SUCCESSFULLY")
print("=" * 70)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nFinal rows:", len(merged))

print("Final columns:", len(merged.columns))