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

forecast = pd.read_csv(
    os.path.join(
        DATA,
        "historical_multivariable_forecast_12months_previous_day1.csv"
    ),
    parse_dates=["datetime"]
)

reference = pd.read_csv(
    os.path.join(
        DATA,
        "era5_multivariable_reference_12months.csv"
    ),
    parse_dates=["datetime"]
).set_index(["location", "datetime"])

location_info = (
    forecast.drop_duplicates("location")
    .set_index("location")[["latitude", "longitude"]]
)

summary = []

print("=" * 70)
print("STEP 48 - FORECAST-ONLY REGRESSION SELECTOR")
print("=" * 70)

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

    # Features available from forecast data alone
    features = wide.copy()

    features.columns = [
        f"{model}_{parameter}"
        for model in WEATHER_MODELS
    ]

    features["forecast_mean"] = wide.mean(axis=1)

    features["forecast_spread"] = (
        wide.max(axis=1) - wide.min(axis=1)
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

    features = features.dropna().copy()

    excluded = [
        "location",
        "datetime"
    ] + target_columns

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded
    ]

    train = features[
        features["datetime"] < "2026-07-01"
    ].copy()

    validation = features[
        (features["datetime"] >= "2026-07-01") &
        (features["datetime"] < "2026-08-01")
    ].copy()

    test = features[
        features["datetime"] >= "2026-08-01"
    ].copy()

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Testing records:", len(test))
    print("Feature count:", len(feature_columns))

    regressors = {}

    for model_name in WEATHER_MODELS:

        print("Training:", model_name, flush=True)

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
            regressors[model].predict(
                frame[feature_columns]
            )
            for model in WEATHER_MODELS
        ])

    def evaluate(frame, predictions, threshold):

        ecmwf_index = WEATHER_MODELS.index("ecmwf_ifs")

        selected = np.argmin(predictions, axis=1)

        predicted_gain = (
            predictions[:, ecmwf_index]
            - predictions.min(axis=1)
        )

        selected = np.where(
            predicted_gain > threshold,
            selected,
            ecmwf_index
        )

        actual_errors = frame[
            target_columns
        ].to_numpy()

        selected_mae = actual_errors[
            np.arange(len(frame)),
            selected
        ].mean()

        baseline_mae = actual_errors[
            :, ecmwf_index
        ].mean()

        return selected_mae, baseline_mae

    validation_predictions = predict_errors(validation)

    thresholds = [
        0.0, 0.05, 0.10, 0.20, 0.30, 0.50
    ]

    best_threshold = 0.0
    best_validation_mae = float("inf")

    for threshold in thresholds:

        ml_mae, baseline_mae = evaluate(
            validation,
            validation_predictions,
            threshold
        )

        print(
            f"Threshold {threshold}: "
            f"ML MAE={ml_mae:.4f}, "
            f"ECMWF MAE={baseline_mae:.4f}"
        )

        if ml_mae < best_validation_mae:
            best_validation_mae = ml_mae
            best_threshold = threshold

    test_predictions = predict_errors(test)

    ml_mae, ecmwf_mae = evaluate(
        test,
        test_predictions,
        best_threshold
    )

    improvement = (
        (ecmwf_mae - ml_mae) / ecmwf_mae * 100
        if ecmwf_mae > 0 else 0
    )

    print("\nFINAL TEST RESULTS")
    print("Best threshold:", best_threshold)
    print("ML MAE:", round(ml_mae, 4))
    print("ECMWF MAE:", round(ecmwf_mae, 4))
    print("Improvement:", round(improvement, 2), "%")

    model_file = os.path.join(
        MODELS_DIR,
        f"{parameter}_forecast_only_selector_v5.pkl"
    )

    joblib.dump(
        {
            "models": regressors,
            "features": feature_columns,
            "parameter": parameter,
            "threshold": best_threshold,
            "weather_models": WEATHER_MODELS,
            "model_type": "forecast_only_regression"
        },
        model_file
    )

    print("Saved:", model_file)

    summary.append({
        "parameter": parameter,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": best_threshold,
        "ml_mae": round(ml_mae, 4),
        "ecmwf_mae": round(ecmwf_mae, 4),
        "improvement_percent": round(improvement, 2)
    })

summary_df = pd.DataFrame(summary)

output_file = os.path.join(
    RESULTS,
    "step48_forecast_only_results.csv"
)

summary_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("FINAL STEP 48 SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print("\nSaved:", output_file)
print("STEP 48 COMPLETE")
