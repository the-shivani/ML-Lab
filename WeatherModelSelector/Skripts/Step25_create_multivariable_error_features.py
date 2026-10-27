import pandas as pd
import numpy as np
import os

print("=" * 70)
print("STEP 25 - CREATE MULTIVARIABLE HISTORICAL ERROR FEATURES")
print("=" * 70)


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

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
    "multivariable_training_with_history.csv"
)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("\nLoading historical forecast data...")

forecast = pd.read_csv(
    FORECAST_FILE
)

print(
    "Forecast rows:",
    len(forecast)
)


print("\nLoading ERA5 reference data...")

reference = pd.read_csv(
    REFERENCE_FILE
)

print(
    "Reference rows:",
    len(reference)
)


# ---------------------------------------------------------
# 3. DATETIME
# ---------------------------------------------------------

forecast["datetime"] = pd.to_datetime(
    forecast["datetime"]
)

reference["datetime"] = pd.to_datetime(
    reference["datetime"]
)


# ---------------------------------------------------------
# 4. MERGE FORECAST + ERA5
# ---------------------------------------------------------

print("\nMerging forecast and ERA5 data...")

merged = forecast.merge(
    reference,
    on=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    how="inner"
)

print(
    "Matched rows:",
    len(merged)
)


# ---------------------------------------------------------
# 5. PARAMETERS AND MODELS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 6. CALCULATE HISTORICAL ERRORS
# ---------------------------------------------------------

print("\nCalculating historical errors...")

for parameter in parameters:

    reference_column = (
        f"reference_{parameter}"
    )

    merged[
        f"{parameter}_error"
    ] = (
        merged[parameter]
        -
        merged[reference_column]
    ).abs()


# ---------------------------------------------------------
# 7. CREATE PAST ERROR FEATURES
# ---------------------------------------------------------

print("\nCreating leakage-safe historical features...")

merged = merged.sort_values(
    [
        "location",
        "model",
        "datetime"
    ]
).reset_index(drop=True)


for parameter in parameters:

    print(
        f"  Processing {parameter}..."
    )

    error_column = (
        f"{parameter}_error"
    )


    # Previous hour error

    merged[
        f"previous_{parameter}_error"
    ] = (
        merged
        .groupby(
            ["location", "model"]
        )[error_column]
        .shift(1)
    )


    # Previous 3-hour average

    merged[
        f"rolling_3h_{parameter}_error"
    ] = (
        merged
        .groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .mean()
        )
    )


    # Previous 6-hour average

    merged[
        f"rolling_6h_{parameter}_error"
    ] = (
        merged
        .groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                6,
                min_periods=1
            )
            .mean()
        )
    )


    # Previous 12-hour average

    merged[
        f"rolling_12h_{parameter}_error"
    ] = (
        merged
        .groupby(
            ["location", "model"]
        )[error_column]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                12,
                min_periods=1
            )
            .mean()
        )
    )


# ---------------------------------------------------------
# 8. PIVOT CURRENT FORECASTS
# ---------------------------------------------------------

print("\nCreating model forecast columns...")

forecast_columns = [
    "location",
    "latitude",
    "longitude",
    "datetime"
]

current_forecasts = merged[
    forecast_columns +
    parameters +
    ["model"]
].copy()


pivot_forecasts = current_forecasts.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    columns="model",
    values=parameters
)

pivot_forecasts.columns = [
    f"{model}_{parameter}"
    for parameter, model
    in pivot_forecasts.columns
]

pivot_forecasts = (
    pivot_forecasts
    .reset_index()
)


# ---------------------------------------------------------
# 9. PIVOT HISTORICAL ERRORS
# ---------------------------------------------------------

history_columns = [
    "location",
    "latitude",
    "longitude",
    "datetime",
    "model"
]

history_values = []

for parameter in parameters:

    history_values.extend([
        f"previous_{parameter}_error",
        f"rolling_3h_{parameter}_error",
        f"rolling_6h_{parameter}_error",
        f"rolling_12h_{parameter}_error"
    ])


history_data = merged[
    history_columns +
    history_values
].copy()


history_pivot = history_data.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    columns="model",
    values=history_values
)

history_pivot.columns = [
    f"{model}_{feature}"
    for feature, model
    in history_pivot.columns
]

history_pivot = (
    history_pivot
    .reset_index()
)


# ---------------------------------------------------------
# 10. CREATE TARGETS
# ---------------------------------------------------------

print("\nCreating best-model targets...")

target_data = merged[
    [
        "location",
        "latitude",
        "longitude",
        "datetime",
        "model"
    ] +
    [
        f"{parameter}_error"
        for parameter in parameters
    ]
].copy()


target_pivot = target_data.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    columns="model",
    values=[
        f"{parameter}_error"
        for parameter in parameters
    ]
)

target_pivot.columns = [
    f"{model}_{error}"
    for error, model
    in target_pivot.columns
]

target_pivot = (
    target_pivot
    .reset_index()
)


for parameter in parameters:

    error_columns = [
        f"{model}_{parameter}_error"
        for model in models
    ]

    target_pivot[
        f"best_{parameter}_model"
    ] = (
        target_pivot[
            error_columns
        ]
        .idxmin(axis=1)
        .str.replace(
            f"_{parameter}_error",
            "",
            regex=False
        )
    )


# ---------------------------------------------------------
# 11. MERGE EVERYTHING
# ---------------------------------------------------------

print("\nCombining training data...")

training = pivot_forecasts.merge(
    history_pivot,
    on=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    how="inner"
)

training = training.merge(
    target_pivot[
        [
            "location",
            "latitude",
            "longitude",
            "datetime"
        ] +
        [
            f"best_{parameter}_model"
            for parameter in parameters
        ]
    ],
    on=[
        "location",
        "latitude",
        "longitude",
        "datetime"
    ],
    how="inner"
)


# ---------------------------------------------------------
# 12. TIME FEATURES
# ---------------------------------------------------------

print("\nCreating time features...")

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


# ---------------------------------------------------------
# 13. FORECAST STATISTICS
# ---------------------------------------------------------

for parameter in parameters:

    model_columns = [
        f"{model}_{parameter}"
        for model in models
    ]

    training[
        f"{parameter}_forecast_mean"
    ] = training[
        model_columns
    ].mean(axis=1)

    training[
        f"{parameter}_forecast_spread"
    ] = (
        training[model_columns].max(axis=1)
        -
        training[model_columns].min(axis=1)
    )


# ---------------------------------------------------------
# 14. REMOVE ROWS WITHOUT HISTORY
# ---------------------------------------------------------

history_feature_columns = []

for parameter in parameters:

    for model in models:

        history_feature_columns.extend([
            f"{model}_previous_{parameter}_error",
            f"{model}_rolling_3h_{parameter}_error",
            f"{model}_rolling_6h_{parameter}_error",
            f"{model}_rolling_12h_{parameter}_error"
        ])


before = len(training)

training = training.dropna(
    subset=history_feature_columns
)

after = len(training)

print(
    "\nRows removed because historical "
    "information was unavailable:",
    before - after
)


# ---------------------------------------------------------
# 15. CHECK
# ---------------------------------------------------------

print("\nFinal checks:")

print(
    "Rows:",
    len(training)
)

print(
    "Locations:",
    training["location"].nunique()
)

print(
    "Columns:",
    len(training.columns)
)

print(
    "Missing values:",
    training.isna().sum().sum()
)


# ---------------------------------------------------------
# 16. BEST MODEL DISTRIBUTIONS
# ---------------------------------------------------------

for parameter in parameters:

    print(
        f"\n{parameter.upper()} BEST MODEL:"
    )

    print(
        training[
            f"best_{parameter}_model"
        ].value_counts()
    )


# ---------------------------------------------------------
# 17. SAVE
# ---------------------------------------------------------

training.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)

print("\nSaved to:")

print(OUTPUT_FILE)