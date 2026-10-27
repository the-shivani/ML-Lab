import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import joblib


# ================================================================
# STEP 6 - TRAIN WEATHER MODEL SELECTOR
# ================================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

INPUT_FILE = os.path.join(
    BASE_PATH,
    "training_data.csv"
)

LOCATION_FILE = os.path.join(
    BASE_PATH,
    "Locations.csv"
)

MODEL_FILE = os.path.join(
    BASE_PATH,
    "weather_model_selector.pkl"
)


print("=" * 70)
print("        STEP 6 - TRAIN WEATHER MODEL SELECTOR")
print("=" * 70)


# ================================================================
# 1. LOAD TRAINING DATA
# ================================================================

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded:", len(df))


# ================================================================
# 2. LOAD LOCATION DATA
# ================================================================

locations = pd.read_csv(LOCATION_FILE)

print("Locations loaded:", len(locations))


# ================================================================
# 3. CHECK LOCATION COLUMN NAMES
# ================================================================

print("\nLocation file columns:")

print(list(locations.columns))


# ================================================================
# 4. STANDARDIZE LOCATION COLUMN NAMES
# ================================================================

locations.columns = [
    column.strip().lower()
    for column in locations.columns
]

df.columns = [
    column.strip().lower()
    for column in df.columns
]


# ================================================================
# 5. MERGE LATITUDE AND LONGITUDE
# ================================================================

location_columns = [
    "city",
    "latitude",
    "longitude"
]

locations_small = locations[location_columns].copy()

locations_small = locations_small.rename(
    columns={
        "city": "location"
    }
)

df = df.merge(
    locations_small,
    on="location",
    how="left"
)


print("\nRows after location merge:", len(df))


# ================================================================
# 6. CHECK LOCATION MATCHING
# ================================================================

missing_coordinates = df[
    ["latitude", "longitude"]
].isnull().any(axis=1).sum()

print(
    "Rows with missing coordinates:",
    missing_coordinates
)

if missing_coordinates > 0:

    print("\nLocations with missing coordinates:")

    print(
        df.loc[
            df["latitude"].isnull(),
            "location"
        ].unique()
    )

    raise ValueError(
        "Some locations could not be matched with latitude/longitude."
    )


# ================================================================
# 7. CONVERT DATETIME
# ================================================================

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ================================================================
# 8. CREATE TIME FEATURES
# ================================================================

df["hour"] = df["datetime"].dt.hour

df["day"] = df["datetime"].dt.day

df["day_of_week"] = (
    df["datetime"].dt.dayofweek
)

df["month"] = (
    df["datetime"].dt.month
)


# ================================================================
# 9. CREATE FORECAST FEATURES
# ================================================================

temperature_columns = [
    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature"
]


df["forecast_mean"] = (
    df[temperature_columns].mean(axis=1)
)


df["forecast_max"] = (
    df[temperature_columns].max(axis=1)
)


df["forecast_min"] = (
    df[temperature_columns].min(axis=1)
)


df["forecast_spread"] = (
    df["forecast_max"]
    - df["forecast_min"]
)


# ================================================================
# 10. SELECT FEATURES
# ================================================================

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
    "month"
]

target = "best_model"


print("\nFeatures used by ML model:")

for feature in features:

    print("-", feature)


# ================================================================
# 11. CHECK MISSING VALUES
# ================================================================

print("\nMissing values:")

print(
    df[
        features + [target]
    ].isnull().sum()
)


# ================================================================
# 12. REMOVE ANY INCOMPLETE ROWS
# ================================================================

before = len(df)

df = df.dropna(
    subset=features + [target]
).reset_index(drop=True)

after = len(df)

print(
    "\nRows removed because of missing values:",
    before - after
)


# ================================================================
# 13. PREPARE X AND Y
# ================================================================

X = df[features]

y = df[target]


print("\nTarget distribution:")

print(
    y.value_counts()
)


# ================================================================
# 14. CHRONOLOGICAL TRAIN / TEST SPLIT
# ================================================================

split_index = int(
    len(df) * 0.80
)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print(
    "\nTrain rows:",
    len(X_train)
)

print(
    "Test rows:",
    len(X_test)
)


print("\nTraining period:")

print(
    df["datetime"].iloc[0],
    "to",
    df["datetime"].iloc[split_index - 1]
)


print("\nTesting period:")

print(
    df["datetime"].iloc[split_index],
    "to",
    df["datetime"].iloc[-1]
)


# ================================================================
# 15. TRAIN RANDOM FOREST
# ================================================================

print("\nTraining Random Forest...")


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ================================================================
# 16. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(
    X_test
)


# ================================================================
# 17. MODEL ACCURACY
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")

print("=" * 70)

print("MODEL PERFORMANCE")

print("=" * 70)


print(
    "\nAccuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ================================================================
# 18. CLASSIFICATION REPORT
# ================================================================

print("\nClassification Report:")


print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ================================================================
# 19. CONFUSION MATRIX
# ================================================================

labels = sorted(
    y.unique()
)


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\nConfusion Matrix:")

print(
    "Labels:"
)

print(
    labels
)

print()

print(
    cm
)


# ================================================================
# 20. BASELINE MODEL
# ================================================================

print("\n")

print("=" * 70)

print("BASELINE COMPARISON")

print("=" * 70)


baseline = DummyClassifier(
    strategy="most_frequent"
)


baseline.fit(
    X_train,
    y_train
)


baseline_pred = baseline.predict(
    X_test
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)


print(
    "\nBaseline accuracy:",
    round(
        baseline_accuracy * 100,
        2
    ),
    "%"
)


print(
    "ML model accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ================================================================
# 21. FEATURE IMPORTANCE
# ================================================================

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n")

print("=" * 70)

print("FEATURE IMPORTANCE")

print("=" * 70)


print(
    importance.to_string(
        index=False
    )
)


# ================================================================
# 22. SAVE MODEL
# ================================================================

joblib.dump(
    model,
    MODEL_FILE
)


print("\n")

print("=" * 70)

print("MODEL SAVED SUCCESSFULLY")

print("=" * 70)


print(
    "\nModel file:"
)


print(
    MODEL_FILE
)


print("\n")

print("=" * 70)

print("STEP 6 COMPLETED")

print("=" * 70)