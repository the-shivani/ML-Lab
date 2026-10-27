import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# ============================================================
# STEP 15 - TRAIN TEMPERATURE MODEL SELECTOR
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models"
)

INPUT_FILE = os.path.join(
    DATA_PATH,
    "multivariable_training_data.csv"
)

OUTPUT_FILE = os.path.join(
    MODEL_PATH,
    "temperature_selector.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 15 - TEMPERATURE MODEL SELECTOR")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

print()
print("Total rows:", len(df))

# ============================================================
# SORT BY TIME
# ============================================================

df = df.sort_values(
    "datetime"
).reset_index(drop=True)

# ============================================================
# FEATURES
# ============================================================

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

target = "best_temperature_model"

X = df[features]

y = df[target]

# ============================================================
# STRICT DATE SPLIT
# ============================================================

split_date = pd.Timestamp(
    "2026-09-25 00:00:00"
)

train = df[
    df["datetime"] < split_date
].copy()

test = df[
    df["datetime"] >= split_date
].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print()
print("STRICT DATE SPLIT")

print(
    "Training rows:",
    len(train)
)

print(
    "Testing rows:",
    len(test)
)

print(
    "Training period:",
    train["datetime"].min(),
    "to",
    train["datetime"].max()
)

print(
    "Testing period:",
    test["datetime"].min(),
    "to",
    test["datetime"].max()
)

# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("Training Random Forest...")

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

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print()
print("=" * 70)
print("TEMPERATURE MODEL RESULTS")
print("=" * 70)

print()
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("Top Features:")

print(
    importance.head(10).to_string(
        index=False
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

joblib.dump(
    model,
    OUTPUT_FILE
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print()
print(OUTPUT_FILE)

print()
print("=" * 70)