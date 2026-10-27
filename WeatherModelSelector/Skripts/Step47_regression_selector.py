import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

BASE = r"C:\Projects\WeatherModelSelector"
DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

PARAMETERS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

WEATHER_MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

FORECAST_FILE = os.path.join(
    DATA,
    "historical_multivariable_forecast_12months_previous_day1.csv"
)

REFERENCE_FILE = os.path.join(
    DATA,
    "era5_multivariable_reference_12months.csv"
)

print("=" * 70)
print("STEP 47 - REGRESSION-BASED WEATHER MODEL SELECTOR")
print("=" * 70)

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["datetime"]
)

reference = pd.read_csv(
    REFERENCE_FILE,
    parse_dates=["datetime"]
)

reference = reference.set_index(
    ["location", "datetime"]
)

location_info = (
    forecast.drop_duplicates("location")
    .set_index("location")[["latitude", "longitude"]]
)

summary = []

for parameter in PARAMETERS:

    print("\n" + "=" * 70)
    print("PARAMETER:", parameter.upper())
    print("=" * 70)

    wide = forecast.pivot(
        index=["location", "datetime"],
        columns="model",
        values=parameter
    ).reindex(columns=WEATHER_MODELS)

    actual = reference[
        "reference_" + parameter
    ].reindex(wide.index)

    errors = wide.sub(actual, axis=0).abs()

    features = wide.copy()
    features.columns = [
        f"{model}_{parameter}"
        for model in WEATHER_MODELS
    ]

    features["forecast_mean"] = wide.mean(axis=1)
    features["forecast_spread"] = (
        wide.max(axis=1) - wide.min(axis=1)
    )

    # Lag historical errors by 48 hours.
    # No current/future ERA5 values are used as features.
    for model in WEATHER_MODELS:

        lagged = (
            errors[model]
            .groupby(level="location")
            .shift(48)
        )

        features[f"{model}_error_lag48"] = lagged

        for window in [3, 6, 12]:

            features[
                f"{model}_error_rolling{window}"
            ] = (
                lagged.groupby(level="location")
                .transform(
                    lambda s: s.rolling(
                        window,
                        min_periods=window
                    ).mean()
                )
            )

    features = features.reset_index()

    features["latitude"] = features["location"].map(
        location_info["latitude"]
    )

    features["longitude"] = features["location"].map(
        location_info["longitude"]
    )

    features["hour"] = features["datetime"].dt.hour
    features["day_of_week"] = (
        features["datetime"].dt.dayofweek
    )
    features["month"] = features["datetime"].dt.month

    error_df = errors.reset_index()

    for model in WEATHER_MODELS:
        features[f"target_error_{model}"] = (
            error_df[model].to_numpy()
        )

    target_columns = [
        f"target_error_{model}"
        for model in WEATHER_MODELS
    ]

    # Keep records where all four models can be compared.
    features = features.dropna().copy()

    excluded = (
        ["location", "datetime"] + target_columns
    )

    feature_columns = [
        col for col in features.columns
        if col not in excluded
    ]

    train = features[
        features["datetime"] < "2026-07-01"
    ]

    validation = features[
        (features["datetime"] >= "2026-07-01") &
        (features["datetime"] < "2026-08-01")
    ]

    test = features[
        features["datetime"] >= "2026-08-01"
    ]

    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    regressors = {}

    for model_name in WEATHER_MODELS:

        print("Training error regressor:", model_name)

        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )

        regressor.fit(
            train[feature_columns],
            train[f"target_error_{model_name}"]
        )

        regressors[model_name] = regressor

    def predict_errors(frame):
        return np.column_stack([
            regressors[m].predict(
                frame[feature_columns]
            )
            for m in WEATHER_MODELS
        ])

    validation_predictions = predict_errors(validation)

    # Tune how much improvement is required before
    # switching away from ECMWF.
    thresholds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

    def evaluate(frame, predicted_errors, threshold):

        selected_indices = np.argmin(
            predicted_errors,
            axis=1
        )

        ecmwf_index = WEATHER_MODELS.index(
            "ecmwf_ifs"
        )

        best_predicted_error = (
            predicted_errors.min(axis=1)
        )

        ecmwf_predicted_error = (
            predicted_errors[:, ecmwf_index]
        )

        improvement = (
            ecmwf_predicted_error -
            best_predicted_error
        )

        selected_indices = np.where(
            improvement > threshold,
            selected_indices,
            ecmwf_index
        )

        actual_errors = frame[
            target_columns
        ].to_numpy()

        selected_actual_errors = actual_errors[
            np.arange(len(frame)),
            selected_indices
        ]

        selected_mae = np.mean(
            selected_actual_errors
        )

        baseline_mae = np.mean(
            actual_errors[:, ecmwf_index]
        )

        return selected_mae, baseline_mae

    best_threshold = None
    best_validation_mae = float("inf")

    for threshold in thresholds:

        selected_mae, baseline_mae = evaluate(
            validation,
            validation_predictions,
            threshold
        )

        print(
            "Threshold:", threshold,
            "Validation ML MAE:", round(selected_mae, 4),
            "ECMWF MAE:", round(baseline_mae, 4)
        )

        if selected_mae < best_validation_mae:
            best_validation_mae = selected_mae
            best_threshold = threshold

    print("\nBest validation threshold:", best_threshold)

    test_predictions = predict_errors(test)

    ml_mae, ecmwf_mae = evaluate(
        test,
        test_predictions,
        best_threshold
    )

    improvement_percent = (
        (ecmwf_mae - ml_mae) / ecmwf_mae * 100
        if ecmwf_mae else 0
    )

    print("\nFINAL TEST RESULTS")
    print("ML selected MAE:", round(ml_mae, 4))
    print("ECMWF baseline MAE:", round(ecmwf_mae, 4))
    print("Improvement:", round(improvement_percent, 2), "%")

    model_file = os.path.join(
        MODELS_DIR,
        f"{parameter}_regression_selector_12months_v4.pkl"
    )

    joblib.dump(
        {
            "models": regressors,
            "features": feature_columns,
            "parameter": parameter,
            "threshold": best_threshold,
            "weather_models": WEATHER_MODELS
        },
        model_file
    )

    summary.append({
        "parameter": parameter,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": best_threshold,
        "ml_mae": round(ml_mae, 4),
        "ecmwf_mae": round(ecmwf_mae, 4),
        "improvement_percent": round(
            improvement_percent, 2
        )
    })

summary_df = pd.DataFrame(summary)

output_file = os.path.join(
    RESULTS,
    "step47_regression_results.csv"
)

summary_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("FINAL STEP 47 SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))
print("\nSaved:", output_file)
print("STEP 47 COMPLETE")
