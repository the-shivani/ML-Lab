import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("=" * 70)
print("STEP 22 - TRAIN ALL PARAMETER MODEL SELECTORS")
print("=" * 70)

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")

INPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_all_parameters.csv"
)

os.makedirs(MODEL_PATH, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("\nLoading training data...")

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])

print("Rows:", len(df))
print("Locations:", df["location"].nunique())


# ---------------------------------------------------------
# 2. TRAIN / TEST DATE SPLIT
# ---------------------------------------------------------

train = df[
    df["datetime"] < "2026-09-25"
].copy()

test = df[
    df["datetime"] >= "2026-09-25"
].copy()

print("\nSTRICT DATE SPLIT")

print(
    "Training:",
    len(train),
    "|",
    train["datetime"].min(),
    "to",
    train["datetime"].max()
)

print(
    "Testing :",
    len(test),
    "|",
    test["datetime"].min(),
    "to",
    test["datetime"].max()
)


# ---------------------------------------------------------
# 3. FEATURES
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


features = [
    "latitude",
    "longitude",
    "hour",
    "day",
    "day_of_week",
    "month"
]


# Add forecast values and statistics

for parameter in parameters:

    for model in models:

        features.append(
            f"{model}_{parameter}"
        )

    features.append(
        f"{parameter}_forecast_mean"
    )

    features.append(
        f"{parameter}_forecast_spread"
    )


print("\nTotal features:", len(features))


# ---------------------------------------------------------
# 4. TRAIN FIVE MODELS
# ---------------------------------------------------------

results = {}

for parameter in parameters:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {parameter.upper()} MODEL"
    )
    print("=" * 70)

    target = f"best_{parameter}_model"

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    print("\nTraining Random Forest...")

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

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

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

    print(
        f"\n{parameter.upper()} RESULTS"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(
        ascending=False
    )

    print("\nTop 10 Features:")

    print(
        importance.head(10)
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_file = os.path.join(
        MODEL_PATH,
        f"{parameter}_selector.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        "\nModel saved:"
    )

    print(model_file)

    results[parameter] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "model_file": model_file
    }


# ---------------------------------------------------------
# 5. FINAL SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for parameter, result in results.items():

    print(
        f"{parameter:15s} "
        f"Accuracy: {result['accuracy'] * 100:.2f}% "
        f"Macro F1: {result['macro_f1']:.4f}"
    )

print("\nAll 5 models trained successfully.")

print("\nModels saved in:")

print(MODEL_PATH)

print("=" * 70)