import pandas as pd
import os

# ============================================================
# STEP 19 - ML TEMPERATURE MODEL VS IMD ACTUAL BEST MODEL
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

RESULTS_PATH = os.path.join(
    BASE_PATH,
    "results"
)

ML_FILE = os.path.join(
    RESULTS_PATH,
    "live_predictions.csv"
)

REALTIME_FILE = os.path.join(
    DATA_PATH,
    "realtime_model_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "temperature_ml_vs_imd.csv"
)


print("=" * 70)
print("STEP 19 - ML TEMPERATURE MODEL VS IMD")
print("=" * 70)


# ============================================================
# LOAD FILES
# ============================================================

ml = pd.read_csv(
    ML_FILE
)

realtime = pd.read_csv(
    REALTIME_FILE
)


print()
print("ML prediction rows:", len(ml))

print(
    "Real-time comparison rows:",
    len(realtime)
)


# ============================================================
# FIND ML MODEL COLUMN
# ============================================================

print()
print("ML columns:")

print(
    ml.columns.tolist()
)


# The live prediction file should contain
# a recommended model column.

possible_columns = [
    "recommended_model",
    "best_model",
    "predicted_model",
    "model"
]


ml_model_column = None


for column in possible_columns:

    if column in ml.columns:

        ml_model_column = column

        break


if ml_model_column is None:

    print()
    print(
        "ERROR: Could not find ML recommendation column."
    )

    raise SystemExit


print()
print(
    "ML recommendation column:",
    ml_model_column
)


# ============================================================
# KEEP TEMPERATURE DATA ONLY
# ============================================================

temperature = realtime[
    [
        "location",
        "model",
        "temperature_error"
    ]
].copy()


# ============================================================
# FIND ACTUAL BEST MODEL
# ============================================================

best_temperature = (
    temperature
    .dropna(
        subset=["temperature_error"]
    )
    .loc[
        temperature
        .dropna(
            subset=["temperature_error"]
        )
        .groupby("location")[
            "temperature_error"
        ]
        .idxmin()
    ]
)


best_temperature = best_temperature[
    [
        "location",
        "model",
        "temperature_error"
    ]
]


best_temperature = best_temperature.rename(
    columns={
        "model":
            "actual_best_temperature_model",

        "temperature_error":
            "actual_best_temperature_error"
    }
)


# ============================================================
# PREPARE ML PREDICTION
# ============================================================

ml_small = ml[
    [
        "location",
        ml_model_column
    ]
].copy()


ml_small = ml_small.rename(
    columns={
        ml_model_column:
            "ml_selected_temperature_model"
    }
)


# ============================================================
# MERGE
# ============================================================

comparison = ml_small.merge(
    best_temperature,
    on="location",
    how="inner"
)


# ============================================================
# CHECK CORRECTNESS
# ============================================================

comparison[
    "ml_selection_correct"
] = (
    comparison[
        "ml_selected_temperature_model"
    ]
    ==
    comparison[
        "actual_best_temperature_model"
    ]
)


# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

total = len(
    comparison
)

correct = int(
    comparison[
        "ml_selection_correct"
    ].sum()
)


accuracy = (
    correct / total * 100
    if total > 0
    else 0
)


print()
print("=" * 70)
print("TEMPERATURE ML VS IMD RESULTS")
print("=" * 70)

print()

print(
    "Locations compared:",
    total
)

print(
    "Correct ML selections:",
    correct
)

print(
    "Incorrect ML selections:",
    total - correct
)

print()

print(
    "REAL-TIME ML SELECTION ACCURACY:",
    round(
        accuracy,
        2
    ),
    "%"
)


# ============================================================
# SHOW COMPARISON
# ============================================================

print()
print("=" * 70)
print("LOCATION-WISE COMPARISON")
print("=" * 70)

print()

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE LOCATION-WISE RESULT
# ============================================================

print()
print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)


print()
print("=" * 70)
print("STEP 19 FINISHED")
print("=" * 70)