import pandas as pd
import os

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")

INPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_forecast_data.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "training_data.csv"
)
OUTPUT_FILE = os.path.join(
    BASE_PATH,
    "training_data.csv"
)

print("=" * 70)
print("        STEP 5 - CREATE ML TRAINING DATA")
print("=" * 70)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded:", len(df))

# --------------------------------------------------
# 2. Separate ERA5 reference
# --------------------------------------------------

reference = df[
    df["model"] == "ERA5_REFERENCE"
][
    ["location", "datetime", "temperature"]
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

# --------------------------------------------------
# 3. Separate forecasts
# --------------------------------------------------

forecasts = df[
    df["model"] != "ERA5_REFERENCE"
].copy()

print(
    "Forecast rows:",
    len(forecasts)
)

# --------------------------------------------------
# 4. Merge forecasts with ERA5
# --------------------------------------------------

merged = forecasts.merge(
    reference,
    on=["location", "datetime"],
    how="inner"
)

print(
    "Matched forecast rows:",
    len(merged)
)

# --------------------------------------------------
# 5. Calculate absolute error
# --------------------------------------------------

merged["error"] = (
    merged["temperature"]
    - merged["actual_temperature"]
).abs()

# --------------------------------------------------
# 6. Create error table
# --------------------------------------------------

error_table = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="error"
).reset_index()

error_table = error_table.rename(
    columns={
        "ecmwf_ifs": "ecmwf_error",
        "gfs_seamless": "gfs_error",
        "icon_seamless": "icon_error",
        "gem_seamless": "gem_error"
    }
)

# --------------------------------------------------
# 7. Create forecast temperature table
# --------------------------------------------------

forecast_table = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="temperature"
).reset_index()

forecast_table = forecast_table.rename(
    columns={
        "ecmwf_ifs": "ecmwf_temperature",
        "gfs_seamless": "gfs_temperature",
        "icon_seamless": "icon_temperature",
        "gem_seamless": "gem_temperature"
    }
)

# --------------------------------------------------
# 8. Merge forecast and error tables
# --------------------------------------------------

training = forecast_table.merge(
    error_table,
    on=["location", "datetime"],
    how="inner"
)

# --------------------------------------------------
# 9. Determine best model
# --------------------------------------------------

error_columns = [
    "ecmwf_error",
    "gfs_error",
    "icon_error",
    "gem_error"
]

model_names = {
    "ecmwf_error": "ecmwf_ifs",
    "gfs_error": "gfs_seamless",
    "icon_error": "icon_seamless",
    "gem_error": "gem_seamless"
}

training["best_model"] = (
    training[error_columns]
    .idxmin(axis=1)
    .map(model_names)
)

# --------------------------------------------------
# 10. Save training data
# --------------------------------------------------

training.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTraining data created successfully.")

print(
    "Training rows:",
    len(training)
)

print(
    "\nBest model distribution:"
)

print(
    training["best_model"]
    .value_counts()
)

print(
    "\nSaved file:"
)

print(OUTPUT_FILE)

print("\n")
print("=" * 70)
print("STEP 5 COMPLETED")
print("=" * 70)