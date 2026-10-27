import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 45 - CREATE 12-MONTH ERROR FEATURES
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

FORECAST_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast_12months.csv"
)

REFERENCE_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference_12months.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_12months_with_history.csv"
)

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

PARAMETERS = {
    "temperature": "reference_temperature",
    "humidity": "reference_humidity",
    "precipitation": "reference_precipitation",
    "wind_speed": "reference_wind_speed",
    "pressure": "reference_pressure"
}

print("=" * 70)
print("STEP 45 - CREATE 12-MONTH ERROR FEATURES")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading forecast data...")

forecast = pd.read_csv(FORECAST_FILE)
forecast["datetime"] = pd.to_datetime(forecast["datetime"])

print("Forecast rows:", len(forecast))

print("\nLoading ERA5 reference data...")

reference = pd.read_csv(REFERENCE_FILE)
reference["datetime"] = pd.to_datetime(reference["datetime"])

print("Reference rows:", len(reference))

# ============================================================
# 2. MERGE
# ============================================================

print("\nMerging forecast and reference data...")

merged = forecast.merge(
    reference,
    on=["location", "datetime"],
    how="inner",
    suffixes=("", "_reference"),
    validate="many_to_one"
)

print("Merged rows:", len(merged))

# ============================================================
# 3. CALCULATE ABSOLUTE ERRORS
# ============================================================

print("\nCalculating model errors...")

for parameter, reference_column in PARAMETERS.items():

    merged[f"{parameter}_error"] = (
        merged[parameter] - merged[reference_column]
    ).abs()

# ============================================================
# 4. SORT DATA
# ============================================================

merged = merged.sort_values(
    ["location", "datetime", "model"]
).reset_index(drop=True)

# ============================================================
# 5. HISTORICAL ERROR FEATURES
# ============================================================

print("\nCreating historical error features...")

for parameter in PARAMETERS.keys():

    error_column = f"{parameter}_error"

    merged[f"previous_{parameter}_error"] = (
        merged.groupby(
            ["location", "model"]
        )[error_column].shift(1)
    )

    merged[f"rolling_3h_{parameter}_error"] = (
        merged.groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x: x.shift(1).rolling(
                3, min_periods=3
            ).mean()
        )
    )

    merged[f"rolling_6h_{parameter}_error"] = (
        merged.groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x: x.shift(1).rolling(
                6, min_periods=6
            ).mean()
        )
    )

    merged[f"rolling_12h_{parameter}_error"] = (
        merged.groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x: x.shift(1).rolling(
                12, min_periods=12
            ).mean()
        )
    )

# ============================================================
# 6. TIME FEATURES
# ============================================================

merged["hour"] = merged["datetime"].dt.hour
merged["day"] = merged["datetime"].dt.day
merged["day_of_week"] = merged["datetime"].dt.dayofweek
merged["month"] = merged["datetime"].dt.month

merged["hour_sin"] = np.sin(
    2 * np.pi * merged["hour"] / 24
)

merged["hour_cos"] = np.cos(
    2 * np.pi * merged["hour"] / 24
)

merged["month_sin"] = np.sin(
    2 * np.pi * merged["month"] / 12
)

merged["month_cos"] = np.cos(
    2 * np.pi * merged["month"] / 12
)

# ============================================================
# 7. CURRENT FORECAST FEATURES
# ============================================================

print("\nCreating wide forecast features...")

forecast_values = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

wide_forecast = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values=forecast_values,
    aggfunc="first"
)

wide_forecast.columns = [
    f"{model}_{parameter}"
    for parameter, model in wide_forecast.columns
]

wide_forecast = wide_forecast.reset_index()

# ============================================================
# 8. FORECAST MEAN AND SPREAD
# ============================================================

for parameter in PARAMETERS.keys():

    columns = [
        f"{model}_{parameter}"
        for model in MODELS
    ]

    wide_forecast[
        f"{parameter}_forecast_mean"
    ] = wide_forecast[columns].mean(axis=1)

    wide_forecast[
        f"{parameter}_forecast_spread"
    ] = (
        wide_forecast[columns].max(axis=1)
        -
        wide_forecast[columns].min(axis=1)
    )

# ============================================================
# 9. HISTORICAL ERROR FEATURES - WIDE FORMAT
# ============================================================

print("\nCreating wide historical error features...")

history_columns = [
    "previous_",
    "rolling_3h_",
    "rolling_6h_",
    "rolling_12h_"
]

history_value_columns = [
    column
    for column in merged.columns
    if any(
        column.startswith(prefix)
        for prefix in history_columns
    )
]

wide_history = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values=history_value_columns,
    aggfunc="first"
)

wide_history.columns = [
    f"{model}_{feature}"
    for feature, model in wide_history.columns
]

wide_history = wide_history.reset_index()

# ============================================================
# 10. COMBINE FEATURES
# ============================================================

print("\nCombining features...")

training = wide_forecast.merge(
    wide_history,
    on=["location", "datetime"],
    how="left",
    validate="one_to_one"
)

# Add latitude and longitude
location_info = forecast[
    ["location", "latitude", "longitude"]
].drop_duplicates(
    subset=["location"]
)

training = training.merge(
    location_info,
    on="location",
    how="left",
    validate="many_to_one"
)

# Add time features
time_features = merged[
    [
        "location",
        "datetime",
        "hour",
        "day",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
    ]
].drop_duplicates(
    subset=["location", "datetime"]
)

training = training.merge(
    time_features,
    on=["location", "datetime"],
    how="left",
    validate="one_to_one"
)

# ============================================================
# 11. FIND ACTUAL BEST MODEL
# ============================================================

print("\nFinding actual best model for each parameter...")

for parameter in PARAMETERS.keys():

    print(f"  Calculating best {parameter} model...")

    error_table = merged.pivot_table(
        index=["location", "datetime"],
        columns="model",
        values=f"{parameter}_error",
        aggfunc="first"
    )

    # Force exact model order
    error_table = error_table.reindex(
        columns=MODELS
    )

    # Find model with minimum absolute error
    best_model = error_table.idxmin(
        axis=1
    )

    target = best_model.rename(
        f"best_{parameter}_model"
    ).reset_index()

    training = training.merge(
        target,
        on=["location", "datetime"],
        how="left",
        validate="one_to_one"
    )

# ============================================================
# 12. REMOVE ROWS WITHOUT HISTORICAL INFORMATION
# ============================================================

history_feature_columns = [
    column
    for column in training.columns
    if (
        "previous_" in column
        or "rolling_3h_" in column
        or "rolling_6h_" in column
        or "rolling_12h_" in column
    )
]

rows_before = len(training)

training = training.dropna(
    subset=history_feature_columns
).reset_index(drop=True)

rows_after = len(training)

print("\nRows before history filtering:", rows_before)
print("Rows removed:", rows_before - rows_after)
print("Final rows:", rows_after)

# ============================================================
# 13. CHECK MISSING VALUES
# ============================================================

missing_total = int(
    training.isna().sum().sum()
)

print("\nTotal missing values:", missing_total)

# ============================================================
# 14. TARGET DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTIONS")
print("=" * 70)

for parameter in PARAMETERS.keys():

    target_column = f"best_{parameter}_model"

    print(f"\n{parameter.upper()}:")

    print(
        training[target_column].value_counts()
    )

# ============================================================
# 15. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET INFORMATION")
print("=" * 70)

print("Rows:", len(training))
print("Columns:", len(training.columns))
print("Locations:", training["location"].nunique())
print("Start:", training["datetime"].min())
print("End:", training["datetime"].max())

# ============================================================
# 16. CHECK TARGET VALIDITY
# ============================================================

print("\nTarget validity check:")

for parameter in PARAMETERS.keys():

    target_column = f"best_{parameter}_model"

    invalid = ~training[target_column].isin(MODELS)

    print(
        f"{parameter}: invalid targets = {invalid.sum()}"
    )

# ============================================================
# 17. SAVE
# ============================================================

training.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 45 COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nDONE")
print("=" * 70)
