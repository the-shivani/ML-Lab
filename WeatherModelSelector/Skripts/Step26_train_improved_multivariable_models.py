import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


print("=" * 70)
print("STEP 26 - TRAIN IMPROVED MULTIVARIABLE ML MODELS")
print("=" * 70)


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")


INPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_with_history.csv"
)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("\nLoading training data...")

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])

print("Rows:", len(df))
print("Locations:", df["location"].nunique())
print("Columns:", len(df.columns))


# ---------------------------------------------------------
# 3. MODELS AND PARAMETERS
# ---------------------------------------------------------

models = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

parameters = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]


# ---------------------------------------------------------
# 4. TIME FEATURES
# ---------------------------------------------------------

base_features = [
    "latitude",
    "longitude",
    "hour",
    "day",
    "day_of_week",
    "month"
]


# ---------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------

train_mask = (
    (df["datetime"] >= "2026-09-01") &
    (df["datetime"] < "2026-09-25")
)

test_mask = (
    (df["datetime"] >= "2026-09-25") &
    (df["datetime"] < "2026-10-01")
)

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()


print("\n" + "=" * 70)
print("STRICT DATE SPLIT")
print("=" * 70)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)

print(
    "Training:",
    train_df["datetime"].min(),
    "to",
    train_df["datetime"].max()
)

print(
    "Testing:",
    test_df["datetime"].min(),
    "to",
    test_df["datetime"].max()
)


# ---------------------------------------------------------
# 6. TRAIN EACH PARAMETER MODEL
# ---------------------------------------------------------

results = {}


for parameter in parameters:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {parameter.upper()} SELECTOR"
    )
    print("=" * 70)


    # -----------------------------------------------------
    # Current forecast features
    # -----------------------------------------------------

    current_features = [
        f"{model}_{parameter}"
        for model in models
    ]


    # -----------------------------------------------------
    # Forecast statistics
    # -----------------------------------------------------

    statistics_features = [
        f"{parameter}_forecast_mean",
        f"{parameter}_forecast_spread"
    ]


    # -----------------------------------------------------
    # Historical error features
    # -----------------------------------------------------

    historical_features = []

    for model in models:

        historical_features.extend([
            f"{model}_previous_{parameter}_error",
            f"{model}_rolling_3h_{parameter}_error",
            f"{model}_rolling_6h_{parameter}_error",
            f"{model}_rolling_12h_{parameter}_error"
        ])


    # -----------------------------------------------------
    # Final feature list
    # -----------------------------------------------------

    features = (
        base_features
        +
        current_features
        +
        statistics_features
        +
        historical_features
    )


    target = f"best_{parameter}_model"


    print(
        "\nNumber of features:",
        len(features)
    )


    print("\nFeatures:")

    for feature in features:
        print(" -", feature)


    # -----------------------------------------------------
    # Prepare X and y
    # -----------------------------------------------------

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

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
    # Prediction
    # -----------------------------------------------------

    y_pred = model.predict(X_test)


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )


    print("\nRESULTS")

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_file = os.path.join(
        MODEL_PATH,
        f"{parameter}_selector_v2.pkl"
    )


    joblib.dump(
        {
            "model": model,
            "features": features,
            "parameter": parameter
        },
        model_file
    )


    print(
        "Model saved:",
        model_file
    )


    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    results[parameter] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# ---------------------------------------------------------
# 7. FINAL SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 26 - FINAL SUMMARY")
print("=" * 70)

print(
    f"{'Parameter':<20}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
)

print("-" * 50)


for parameter in parameters:

    accuracy = results[
        parameter
    ]["accuracy"]

    macro_f1 = results[
        parameter
    ]["macro_f1"]


    print(
        f"{parameter:<20}"
        f"{accuracy * 100:>8.2f}%      "
        f"{macro_f1:>8.4f}"
    )


print("\n")
print("=" * 70)
print("STEP 26 COMPLETE")
print("=" * 70)