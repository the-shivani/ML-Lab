import os
import numpy as np
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"
DATA = os.path.join(BASE, "data")

FORECAST = os.path.join(
    DATA, "historical_multivariable_forecast_12months_previous_day1.csv"
)
REFERENCE = os.path.join(
    DATA, "era5_multivariable_reference_12months.csv"
)

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

PARAMETERS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

print("=" * 65)
print("STEP 45B - CORRECTED 12-MONTH FEATURE ENGINEERING")
print("=" * 65)

print("\nLoading forecast data...")
forecast = pd.read_csv(FORECAST, parse_dates=["datetime"])

print("Loading ERA5 reference...")
reference = pd.read_csv(REFERENCE, parse_dates=["datetime"])

reference = reference.set_index(["location", "datetime"])

summary = []

for parameter in PARAMETERS:

    print("\n" + "=" * 65)
    print("PROCESSING:", parameter.upper())
    print("=" * 65)

    # One row per location and valid forecast hour
    wide = forecast.pivot(
        index=["location", "datetime"],
        columns="model",
        values=parameter
    ).reindex(columns=MODELS)

    # Match reference values
    actual = reference["reference_" + parameter].reindex(wide.index)

    # Absolute error of each forecasting model
    errors = wide.sub(actual, axis=0).abs()

    # Historical error features must be available before
    # the 24-hour-ahead forecast was issued.
    # A 48-hour lag provides a conservative availability buffer.
    historical = pd.DataFrame(index=wide.index)

    for model in MODELS:

        error_series = errors[model]

        # Operate separately for every location.
        # Shift 48 hourly rows into the past.
        grouped = error_series.groupby(level="location")

        lagged = grouped.shift(48)

        historical[f"{model}_previous_error"] = lagged

        for window in [3, 6, 12]:

            rolling = (
                lagged.groupby(level="location")
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=window
                    ).mean()
                )
            )

            historical[
                f"{model}_rolling_{window}h_error"
            ] = rolling

    # Construct training features
    features = wide.copy()
    features.columns = [
        f"{model}_{parameter}" for model in MODELS
    ]

    features[f"{parameter}_forecast_mean"] = wide.mean(axis=1)
    features[f"{parameter}_forecast_spread"] = (
        wide.max(axis=1) - wide.min(axis=1)
    )

    features = features.join(historical)

    # Identify complete four-model comparisons
    complete = errors.notna().all(axis=1)

    minimum_error = errors.min(axis=1)

    winner_count = errors.eq(
        minimum_error, axis=0
    ).sum(axis=1)

    unique_winner = winner_count == 1

    eligible = complete & unique_winner

    print("Total location-hour records:", len(wide))
    print("Complete comparisons:", int(complete.sum()))
    print(
        "Tied complete comparisons:",
        int((complete & ~unique_winner).sum())
    )
    print("Unique-winner comparisons:", int(eligible.sum()))

    # Only calculate winner labels for eligible records
    eligible_errors = errors.loc[eligible]

    labels = eligible_errors.idxmin(axis=1)

    # Create training dataset
    training = features.loc[eligible].copy()

    training["best_model"] = labels

    training = training.reset_index()

    training["latitude"] = training["location"].map(
        forecast.drop_duplicates("location")
        .set_index("location")["latitude"]
    )

    training["longitude"] = training["location"].map(
        forecast.drop_duplicates("location")
        .set_index("location")["longitude"]
    )

    training["hour"] = training["datetime"].dt.hour
    training["day_of_week"] = training["datetime"].dt.dayofweek
    training["month"] = training["datetime"].dt.month

    # Remove records without sufficient historical features
    before = len(training)

    training = training.dropna().copy()

    removed = before - len(training)

    print("Rows removed due to unavailable features:", removed)
    print("Final training rows:", len(training))

    print("\nBest-model distribution:")
    print(training["best_model"].value_counts())

    print("\nRemaining missing values:")
    print(int(training.isna().sum().sum()))

    output = os.path.join(
        DATA,
        f"training_12months_{parameter}_45B.csv"
    )

    training.to_csv(output, index=False)

    print("\nSaved:", output)

    summary.append({
        "parameter": parameter,
        "complete_comparisons": int(complete.sum()),
        "unique_winner_comparisons": int(eligible.sum()),
        "final_training_rows": len(training),
        "removed_missing_features": removed
    })

summary_df = pd.DataFrame(summary)

summary_file = os.path.join(
    DATA, "training_12months_45B_summary.csv"
)

summary_df.to_csv(summary_file, index=False)

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(summary_df.to_string(index=False))
print("\nSaved summary:", summary_file)
print("\nSTEP 45B COMPLETED")
