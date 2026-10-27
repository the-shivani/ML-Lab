import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

# ============================================================
# STEP 9: FINAL LEAKAGE-SAFE VALIDATION
# ============================================================

BASE = r"C:\Projects\WeatherModelSelector"

DATA_PATH = BASE + r"\data"

RAW_FILE = DATA_PATH + r"\historical_forecast_data.csv"
LOCATIONS_FILE = DATA_PATH + r"\Locations.csv"

print("=" * 70)
print("FINAL LEAKAGE-SAFE VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load raw historical data
# ------------------------------------------------------------

df = pd.read_csv(RAW_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("\nRaw rows:", len(df))

# Separate ERA5 reference
reference = df[df["model"] == "ERA5_REFERENCE"][
    ["location", "datetime", "temperature"]
].copy()

reference = reference.rename(
    columns={"temperature": "actual_temperature"}
)

# Forecast data
forecast = df[df["model"] != "ERA5_REFERENCE"].copy()

print("Forecast rows:", len(forecast))
print("Reference rows:", len(reference))

# ------------------------------------------------------------
# 2. Match forecasts with ERA5
# ------------------------------------------------------------

merged = forecast.merge(
    reference,
    on=["location", "datetime"],
    how="inner"
)

print("Matched rows:", len(merged))

# Calculate error
merged["error"] = (
    merged["temperature"] - merged["actual_temperature"]
).abs()

# ------------------------------------------------------------
# 3. Sort chronologically
# ------------------------------------------------------------

merged = merged.sort_values(
    ["location", "model", "datetime"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Create ONLY past-performance features
# ------------------------------------------------------------

group = merged.groupby(["location", "model"])["error"]

merged["previous_error"] = group.shift(1)

# Previous hour error
merged["previous_error"] = (
    merged.groupby(["location", "model"])["error"].shift(1)
)

# Rolling errors using only previous observations
merged["rolling_error_3h"] = (
    merged.groupby(["location", "model"])["error"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

merged["rolling_error_6h"] = (
    merged.groupby(["location", "model"])["error"]
    .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
)

merged["rolling_error_12h"] = (
    merged.groupby(["location", "model"])["error"]
    .transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
)

# ------------------------------------------------------------
# 5. Create BEST MODEL target
# ------------------------------------------------------------

best = (
    merged.loc[
        merged.groupby(["location", "datetime"])["error"].idxmin()
    ][["location", "datetime", "model"]]
    .rename(columns={"model": "best_model"})
)

# ------------------------------------------------------------
# 6. Pivot current forecast temperatures
# ------------------------------------------------------------

temperature_pivot = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="temperature"
).reset_index()

temperature_pivot.columns.name = None

temperature_pivot = temperature_pivot.rename(
    columns={
        "ecmwf_ifs": "ecmwf_temperature",
        "gfs_seamless": "gfs_temperature",
        "icon_seamless": "icon_temperature",
        "gem_seamless": "gem_temperature"
    }
)

# ------------------------------------------------------------
# 7. Pivot previous-error features
# ------------------------------------------------------------

previous = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="previous_error"
).reset_index()

previous.columns.name = None

previous = previous.rename(
    columns={
        "ecmwf_ifs": "ecmwf_previous_error",
        "gfs_seamless": "gfs_previous_error",
        "icon_seamless": "icon_previous_error",
        "gem_seamless": "gem_previous_error"
    }
)

# ------------------------------------------------------------
# 8. Pivot rolling 3h
# ------------------------------------------------------------

rolling3 = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="rolling_error_3h"
).reset_index()

rolling3.columns.name = None

rolling3 = rolling3.rename(
    columns={
        "ecmwf_ifs": "ecmwf_rolling_error_3h",
        "gfs_seamless": "gfs_rolling_error_3h",
        "icon_seamless": "icon_rolling_error_3h",
        "gem_seamless": "gem_rolling_error_3h"
    }
)

# ------------------------------------------------------------
# 9. Pivot rolling 6h
# ------------------------------------------------------------

rolling6 = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="rolling_error_6h"
).reset_index()

rolling6.columns.name = None

rolling6 = rolling6.rename(
    columns={
        "ecmwf_ifs": "ecmwf_rolling_error_6h",
        "gfs_seamless": "gfs_rolling_error_6h",
        "icon_seamless": "icon_rolling_error_6h",
        "gem_seamless": "gem_rolling_error_6h"
    }
)

# ------------------------------------------------------------
# 10. Pivot rolling 12h
# ------------------------------------------------------------

rolling12 = merged.pivot_table(
    index=["location", "datetime"],
    columns="model",
    values="rolling_error_12h"
).reset_index()

rolling12.columns.name = None

rolling12 = rolling12.rename(
    columns={
        "ecmwf_ifs": "ecmwf_rolling_error_12h",
        "gfs_seamless": "gfs_rolling_error_12h",
        "icon_seamless": "icon_rolling_error_12h",
        "gem_seamless": "gem_rolling_error_12h"
    }
)

# ------------------------------------------------------------
# 11. Combine everything
# ------------------------------------------------------------

data = temperature_pivot.copy()

for extra in [previous, rolling3, rolling6, rolling12]:
    data = data.merge(
        extra,
        on=["location", "datetime"],
        how="inner"
    )

data = data.merge(
    best,
    on=["location", "datetime"],
    how="inner"
)

# ------------------------------------------------------------
# 12. Time features
# ------------------------------------------------------------

data["hour"] = data["datetime"].dt.hour
data["day"] = data["datetime"].dt.day
data["day_of_week"] = data["datetime"].dt.dayofweek
data["month"] = data["datetime"].dt.month

# Forecast statistics
forecast_columns = [
    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature"
]

data["forecast_mean"] = data[forecast_columns].mean(axis=1)
data["forecast_max"] = data[forecast_columns].max(axis=1)
data["forecast_min"] = data[forecast_columns].min(axis=1)
data["forecast_spread"] = (
    data["forecast_max"] - data["forecast_min"]
)

# ------------------------------------------------------------
# 13. Add latitude and longitude
# ------------------------------------------------------------

locations = pd.read_csv(LOCATIONS_FILE)

locations = locations.rename(
    columns={
        "City": "location",
        "Latitude": "latitude",
        "Longitude": "longitude"
    }
)

data = data.merge(
    locations[["location", "latitude", "longitude"]],
    on="location",
    how="left"
)

# ------------------------------------------------------------
# 14. Remove rows without past information
# ------------------------------------------------------------

past_features = [
    "ecmwf_previous_error",
    "gfs_previous_error",
    "icon_previous_error",
    "gem_previous_error"
]

before = len(data)

data = data.dropna(
    subset=past_features
).copy()

print("\nRows removed due to missing past information:",
      before - len(data))

# ------------------------------------------------------------
# 15. STRICT DATE SPLIT
# ------------------------------------------------------------

train = data[
    data["datetime"].dt.date <= pd.Timestamp("2026-09-24").date()
].copy()

test = data[
    data["datetime"].dt.date >= pd.Timestamp("2026-09-25").date()
].copy()

print("\nSTRICT DATE SPLIT")
print("-" * 50)

print(
    "Training:",
    len(train),
    "|",
    train["datetime"].min(),
    "to",
    train["datetime"].max()
)

print(
    "Testing:",
    len(test),
    "|",
    test["datetime"].min(),
    "to",
    test["datetime"].max()
)

# ------------------------------------------------------------
# 16. Features
# ------------------------------------------------------------

features = [
    "latitude",
    "longitude",

    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature",

    "forecast_mean",
    "forecast_spread",

    "hour",
    "day",
    "day_of_week",
    "month",

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

X_train = train[features]
y_train = train["best_model"]

X_test = test[features]
y_test = test["best_model"]

# ------------------------------------------------------------
# 17. Train final validation model
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 18. Evaluate
# ------------------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

labels = [
    "ecmwf_ifs",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless"
]

print("Confusion Matrix:")
print(labels)

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )
)

# ------------------------------------------------------------
# 19. Baseline
# ------------------------------------------------------------

baseline_prediction = (
    train["best_model"]
    .value_counts()
    .idxmax()
)

baseline_accuracy = accuracy_score(
    y_test,
    [baseline_prediction] * len(y_test)
)

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    f"\nBaseline accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Final ML accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Improvement: "
    f"{(accuracy - baseline_accuracy) * 100:.2f} percentage points"
)

# ------------------------------------------------------------
# 20. Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

print("\nFinal validation completed successfully.")
