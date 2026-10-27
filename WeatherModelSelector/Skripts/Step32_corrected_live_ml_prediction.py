# ================================================================
# STEP 32: CORRECTED LIVE ML PREDICTION WITH PRESSURE MSL
# ================================================================

import os
import joblib
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"

RESULTS_PATH = os.path.join(BASE_PATH, "results")
MODEL_PATH = os.path.join(BASE_PATH, "models")

OLD_FILE = os.path.join(
    RESULTS_PATH,
    "live_improved_predictions.csv"
)

PRESSURE_FILE = os.path.join(
    RESULTS_PATH,
    "live_pressure_msl_forecasts.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_corrected_predictions.csv"
)

PRESSURE_MODEL = os.path.join(
    MODEL_PATH,
    "pressure_selector_v2.pkl"
)

print("=" * 70)
print("STEP 32: CORRECTED LIVE ML PREDICTION")
print("=" * 70)

# ------------------------------------------------
# LOAD FILES
# ------------------------------------------------

print("\nLoading previous live predictions...")

ml = pd.read_csv(OLD_FILE)

print(f"Rows: {len(ml)}")
print(f"Locations: {ml['location'].nunique()}")

print("\nLoading corrected pressure MSL forecasts...")

pressure = pd.read_csv(PRESSURE_FILE)

print(f"Rows: {len(pressure)}")
print(f"Locations: {pressure['location'].nunique()}")

# ------------------------------------------------
# LOAD PRESSURE MODEL
# ------------------------------------------------

print("\nLoading pressure ML model...")

model = joblib.load(PRESSURE_MODEL)

print("Pressure model loaded.")

# ------------------------------------------------
# MERGE PRESSURE MSL
# ------------------------------------------------

pressure_wide = pressure.pivot(
    index=["location", "latitude", "longitude", "forecast_datetime"],
    columns="model",
    values="pressure_msl"
).reset_index()

pressure_wide = pressure_wide.rename(columns={
    "ecmwf_ifs": "ecmwf_ifs_pressure",
    "gfs_seamless": "gfs_seamless_pressure",
    "icon_seamless": "icon_seamless_pressure",
    "gem_seamless": "gem_seamless_pressure"
})

# ------------------------------------------------
# MERGE WITH LIVE ML DATA
# ------------------------------------------------

ml = ml.merge(
    pressure_wide[
        [
            "location",
            "ecmwf_ifs_pressure",
            "gfs_seamless_pressure",
            "icon_seamless_pressure",
            "gem_seamless_pressure"
        ]
    ],
    on="location",
    how="left",
    suffixes=("", "_new")
)

# ------------------------------------------------
# REPLACE OLD PRESSURE VALUES
# ------------------------------------------------

pressure_columns = [
    "ecmwf_ifs_pressure",
    "gfs_seamless_pressure",
    "icon_seamless_pressure",
    "gem_seamless_pressure"
]

for col in pressure_columns:

    new_col = col + "_new"

    if new_col in ml.columns:
        ml[col] = ml[new_col]

        ml.drop(
            columns=[new_col],
            inplace=True
        )

# ------------------------------------------------
# RECALCULATE PRESSURE FEATURES
# ------------------------------------------------

ml["pressure_forecast_mean"] = ml[
    pressure_columns
].mean(axis=1)

ml["pressure_forecast_spread"] = ml[
    pressure_columns
].max(axis=1) - ml[
    pressure_columns
].min(axis=1)

# ------------------------------------------------
# PRESSURE MODEL FEATURES
# ------------------------------------------------

pressure_features = [
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "month",

    "ecmwf_ifs_pressure",
    "gfs_seamless_pressure",
    "icon_seamless_pressure",
    "gem_seamless_pressure",

    "pressure_forecast_mean",
    "pressure_forecast_spread",

    "ecmwf_ifs_pressure_prev_error",
    "ecmwf_ifs_pressure_rolling_3h_error",
    "ecmwf_ifs_pressure_rolling_6h_error",
    "ecmwf_ifs_pressure_rolling_12h_error",

    "gfs_seamless_pressure_prev_error",
    "gfs_seamless_pressure_rolling_3h_error",
    "gfs_seamless_pressure_rolling_6h_error",
    "gfs_seamless_pressure_rolling_12h_error",

    "icon_seamless_pressure_prev_error",
    "icon_seamless_pressure_rolling_3h_error",
    "icon_seamless_pressure_rolling_6h_error",
    "icon_seamless_pressure_rolling_12h_error",

    "gem_seamless_pressure_prev_error",
    "gem_seamless_pressure_rolling_3h_error",
    "gem_seamless_pressure_rolling_6h_error",
    "gem_seamless_pressure_rolling_12h_error"
]

print("\nChecking pressure features...")

missing_features = [
    col for col in pressure_features
    if col not in ml.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    for col in missing_features:
        print("  ", col)

    raise ValueError(
        "Required pressure features are missing."
    )

# ------------------------------------------------
# PREDICT
# ------------------------------------------------

X_pressure = ml[pressure_features]

print(
    f"\nPressure feature count: "
    f"{len(pressure_features)}"
)

ml["best_pressure_model"] = model.predict(
    X_pressure
)

# ------------------------------------------------
# SAVE
# ------------------------------------------------

ml.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------
# SHOW RESULTS
# ------------------------------------------------

print("\n")
print("=" * 70)
print("CORRECTED PRESSURE ML SELECTION")
print("=" * 70)

print(
    ml[
        [
            "location",
            "ecmwf_ifs_pressure",
            "gfs_seamless_pressure",
            "icon_seamless_pressure",
            "gem_seamless_pressure",
            "best_pressure_model"
        ]
    ].to_string(index=False)
)

print("\nPressure selection distribution:")

print(
    ml[
        "best_pressure_model"
    ].value_counts().to_string()
)

print("\nSaved file:")
print(OUTPUT_FILE)

print("\n")
print("=" * 70)
print("STEP 32 COMPLETED")
print("=" * 70)