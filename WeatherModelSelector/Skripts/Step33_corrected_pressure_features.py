import os
import pandas as pd
import joblib

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

LIVE_FILE = os.path.join(
    RESULTS_PATH,
    "live_pressure_msl_forecasts.csv"
)

HISTORY_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_with_history.csv"
)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "pressure_selector_v2.pkl"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_corrected_predictions.csv"
)

print("=" * 70)
print("STEP 40: CORRECTED PRESSURE ML PREDICTION")
print("=" * 70)

# ============================================================
# 1. LOAD LIVE PRESSURE FORECASTS
# ============================================================

print("\nLoading live pressure forecasts...")

live = pd.read_csv(LIVE_FILE)

live["forecast_datetime"] = pd.to_datetime(
    live["forecast_datetime"]
)

print("Live rows:", len(live))
print("Live locations:", live["location"].nunique())

# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

live["hour"] = live["forecast_datetime"].dt.hour
live["day"] = live["forecast_datetime"].dt.day
live["day_of_week"] = live["forecast_datetime"].dt.dayofweek
live["month"] = live["forecast_datetime"].dt.month

# ============================================================
# 3. PIVOT CURRENT PRESSURE FORECASTS
# ============================================================

print("\nCreating current pressure features...")

pressure_pivot = live.pivot_table(
    index=[
        "location",
        "latitude",
        "longitude",
        "forecast_datetime",
        "hour",
        "day",
        "day_of_week",
        "month"
    ],
    columns="model",
    values="pressure_msl"
).reset_index()

pressure_pivot.columns.name = None

rename_map = {
    "ecmwf_ifs": "ecmwf_ifs_pressure",
    "gfs_seamless": "gfs_seamless_pressure",
    "icon_seamless": "icon_seamless_pressure",
    "gem_seamless": "gem_seamless_pressure"
}

pressure_pivot = pressure_pivot.rename(
    columns=rename_map
)

required_current = [
    "ecmwf_ifs_pressure",
    "gfs_seamless_pressure",
    "icon_seamless_pressure",
    "gem_seamless_pressure"
]

missing_current = [
    c for c in required_current
    if c not in pressure_pivot.columns
]

if missing_current:

    print("\nERROR: Missing current pressure columns:")

    for c in missing_current:
        print("  ", c)

    raise ValueError(
        "Current pressure forecast columns are missing."
    )

# ============================================================
# 4. PRESSURE MEAN AND SPREAD
# ============================================================

pressure_pivot["pressure_forecast_mean"] = (
    pressure_pivot[required_current].mean(axis=1)
)

pressure_pivot["pressure_forecast_spread"] = (
    pressure_pivot[required_current].max(axis=1)
    - pressure_pivot[required_current].min(axis=1)
)

print(
    "Current pressure feature rows:",
    len(pressure_pivot)
)

# ============================================================
# 5. LOAD HISTORICAL TRAINING DATA
# ============================================================

print("\nLoading historical training data...")

history = pd.read_csv(HISTORY_FILE)

print("Historical rows:", len(history))

history["datetime"] = pd.to_datetime(
    history["datetime"]
)

# ============================================================
# 6. PRESSURE HISTORY FEATURE NAMES
# ============================================================

print("\nGetting latest historical pressure error features...")

pressure_history_columns = [

    "ecmwf_ifs_previous_pressure_error",
    "ecmwf_ifs_rolling_3h_pressure_error",
    "ecmwf_ifs_rolling_6h_pressure_error",
    "ecmwf_ifs_rolling_12h_pressure_error",

    "gfs_seamless_previous_pressure_error",
    "gfs_seamless_rolling_3h_pressure_error",
    "gfs_seamless_rolling_6h_pressure_error",
    "gfs_seamless_rolling_12h_pressure_error",

    "icon_seamless_previous_pressure_error",
    "icon_seamless_rolling_3h_pressure_error",
    "icon_seamless_rolling_6h_pressure_error",
    "icon_seamless_rolling_12h_pressure_error",

    "gem_seamless_previous_pressure_error",
    "gem_seamless_rolling_3h_pressure_error",
    "gem_seamless_rolling_6h_pressure_error",
    "gem_seamless_rolling_12h_pressure_error"
]

missing_history = [
    c
    for c in pressure_history_columns
    if c not in history.columns
]

if missing_history:

    print("\nERROR: Missing pressure history columns:")

    for c in missing_history:
        print("  ", c)

    raise ValueError(
        "Pressure history features are missing."
    )

# ============================================================
# 7. GET LATEST HISTORY FOR EACH LOCATION
# ============================================================

history_latest = (
    history
    .sort_values("datetime")
    .groupby("location")
    .tail(1)
)

print(
    "Locations with historical pressure features:",
    len(history_latest)
)

history_features = history_latest[
    ["location"] + pressure_history_columns
].copy()

# ============================================================
# 8. MERGE LIVE + HISTORICAL FEATURES
# ============================================================

print("\nMerging live and historical features...")

features = pressure_pivot.merge(
    history_features,
    on="location",
    how="left"
)

# ============================================================
# 9. LOAD MODEL PACKAGE
# ============================================================

print("\nLoading pressure ML model...")

model_package = joblib.load(MODEL_FILE)

print("Model package loaded successfully.")

print(
    "Parameter:",
    model_package.get("parameter")
)

model = model_package["model"]

trained_features = model_package["features"]

print(
    "Training features:",
    len(trained_features)
)

# ============================================================
# 10. CHECK MODEL FEATURES
# ============================================================

print("\nChecking required training features...")

missing_model_features = [
    c
    for c in trained_features
    if c not in features.columns
]

if missing_model_features:

    print(
        "\nERROR: Required model features are missing:"
    )

    for c in missing_model_features:
        print("  ", c)

    print(
        "\nAvailable feature columns:"
    )

    for c in features.columns:
        print("  ", c)

    raise ValueError(
        "Some features required by the trained model are missing."
    )

print("All trained features are available.")

# ============================================================
# 11. CHECK MISSING VALUES
# ============================================================

missing_values = (
    features[trained_features]
    .isna()
    .sum()
)

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:

    print("\nERROR: Missing feature values:")

    print(missing_values)

    raise ValueError(
        "Some model features contain missing values."
    )

print(
    "\nAll",
    len(trained_features),
    "model features are available."
)

# ============================================================
# 12. PREDICT BEST PRESSURE MODEL
# ============================================================

print("\nPredicting best pressure model...")

X = features[trained_features]

prediction = model.predict(X)

features["best_pressure_model"] = prediction

# ============================================================
# 13. SAVE RESULTS
# ============================================================

output_columns = [
    "location",
    "latitude",
    "longitude",
    "forecast_datetime",

    "ecmwf_ifs_pressure",
    "gfs_seamless_pressure",
    "icon_seamless_pressure",
    "gem_seamless_pressure",

    "pressure_forecast_mean",
    "pressure_forecast_spread",

    "best_pressure_model"
]

output = features[output_columns].copy()

output.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PRESSURE ML PREDICTIONS")
print("=" * 70)

print(
    output[
        [
            "location",
            "forecast_datetime",
            "ecmwf_ifs_pressure",
            "gfs_seamless_pressure",
            "icon_seamless_pressure",
            "gem_seamless_pressure",
            "best_pressure_model"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("PRESSURE MODEL SELECTION DISTRIBUTION")
print("=" * 70)

print(
    output["best_pressure_model"]
    .value_counts()
)

print("\nSaved successfully:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STEP 40 COMPLETE")
print("=" * 70)