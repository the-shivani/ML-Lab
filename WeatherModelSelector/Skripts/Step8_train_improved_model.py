import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.dummy import DummyClassifier
import joblib


# ================================================================
# STEP 8 - TRAIN IMPROVED WEATHER MODEL SELECTOR
# ================================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

MODELS_PATH = os.path.join(
    BASE_PATH,
    "models"
)

INPUT_FILE = os.path.join(
    DATA_PATH,
    "training_data_v2.csv"
)

MODEL_FILE = os.path.join(
    MODELS_PATH,
    "weather_model_selector_v2.pkl"
)

print("=" * 70)
print("       STEP 8 - TRAIN IMPROVED WEATHER MODEL SELECTOR")
print("=" * 70)


# ================================================================
# 1. LOAD DATA
# ================================================================

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded:", len(df))


# ================================================================
# 2. CONVERT DATETIME
# ================================================================

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ================================================================
# 3. DEFINE FEATURES
# ================================================================

features = [

    # Location
    "latitude",
    "longitude",

    # Current model forecasts
    "ecmwf_temperature",
    "gfs_temperature",
    "icon_temperature",
    "gem_temperature",

    # Forecast agreement
    "forecast_mean",
    "forecast_spread",

    # Time
    "hour",
    "day",
    "day_of_week",
    "month",

    # Previous-hour model performance
    "ecmwf_previous_error",
    "gfs_previous_error",
    "icon_previous_error",
    "gem_previous_error",

    # 3-hour performance
    "ecmwf_rolling_error_3h",
    "gfs_rolling_error_3h",
    "icon_rolling_error_3h",
    "gem_rolling_error_3h",

    # 6-hour performance
    "ecmwf_rolling_error_6h",
    "gfs_rolling_error_6h",
    "icon_rolling_error_6h",
    "gem_rolling_error_6h",

    # 12-hour performance
    "ecmwf_rolling_error_12h",
    "gfs_rolling_error_12h",
    "icon_rolling_error_12h",
    "gem_rolling_error_12h"
]


target = "best_model"


print("\nNumber of features:", len(features))

print("\nFeatures used:")

for feature in features:
    print("-", feature)


# ================================================================
# 4. CHECK MISSING VALUES
# ================================================================

missing = df[
    features + [target]
].isnull().sum()


print("\nMissing values:")

print(missing)


if missing.sum() > 0:

    raise ValueError(
        "Missing values found in training data."
    )


# ================================================================
# 5. PREPARE X AND Y
# ================================================================

X = df[features]

y = df[target]


print("\nTarget distribution:")

print(
    y.value_counts()
)


# ================================================================
# 6. CHRONOLOGICAL TRAIN / TEST SPLIT
# ================================================================

split_index = int(
    len(df) * 0.80
)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\nTrain rows:", len(X_train))

print("Test rows:", len(X_test))


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
# 7. TRAIN RANDOM FOREST
# ================================================================

print("\nTraining improved Random Forest...")


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
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
# 8. PREDICT
# ================================================================

y_pred = model.predict(
    X_test
)


# ================================================================
# 9. ACCURACY
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n")

print("=" * 70)

print("IMPROVED MODEL PERFORMANCE")

print("=" * 70)


print(
    "\nAccuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Macro F1:",
    round(
        macro_f1,
        4
    )
)


# ================================================================
# 10. CLASSIFICATION REPORT
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
# 11. CONFUSION MATRIX
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
# 12. BASELINE
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


baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    average="macro"
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
    "Baseline macro F1:",
    round(
        baseline_f1,
        4
    )
)


print(
    "\nImproved ML accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Improved ML macro F1:",
    round(
        macro_f1,
        4
    )
)


print(
    "\nImprovement over baseline:",
    round(
        (accuracy - baseline_accuracy) * 100,
        2
    ),
    "percentage points"
)


# ================================================================
# 13. FEATURE IMPORTANCE
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

print("TOP FEATURE IMPORTANCE")

print("=" * 70)


print(
    importance.head(15).to_string(
        index=False
    )
)


# ================================================================
# 14. SAVE MODEL
# ================================================================

joblib.dump(
    model,
    MODEL_FILE
)


print("\n")

print("=" * 70)

print("IMPROVED MODEL SAVED SUCCESSFULLY")

print("=" * 70)


print("\nModel file:")

print(
    MODEL_FILE
)


print("\n")

print("=" * 70)

print("STEP 8 COMPLETED")

print("=" * 70)