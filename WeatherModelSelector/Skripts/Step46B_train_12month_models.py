import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE = r"C:\Projects\WeatherModelSelector"

DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

PARAMETERS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure"
]

WEATHER_MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

REFERENCE_FILE = os.path.join(
    DATA, "era5_multivariable_reference_12months.csv"
)

reference = pd.read_csv(
    REFERENCE_FILE,
    parse_dates=["datetime"]
)

reference = reference.set_index(
    ["location", "datetime"]
)

summary = []

print("=" * 70)
print("STEP 46B - 12-MONTH RANDOM FOREST TRAINING")
print("=" * 70)

for parameter in PARAMETERS:

    print("\n" + "=" * 70)
    print("TRAINING:", parameter.upper())
    print("=" * 70)

    file = os.path.join(
        DATA,
        f"training_12months_{parameter}_45B.csv"
    )

    df = pd.read_csv(
        file,
        parse_dates=["datetime"]
    )

    # Chronological train-test split
    train = df[
        df["datetime"] < "2026-08-01"
    ].copy()

    test = df[
        df["datetime"] >= "2026-08-01"
    ].copy()

    excluded = [
        "location",
        "datetime",
        "best_model"
    ]

    features = [
        col for col in df.columns
        if col not in excluded
    ]

    X_train = train[features]
    y_train = train["best_model"]

    X_test = test[features]
    y_test = test["best_model"]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Feature count:", len(features))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nClassification accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))

    # Evaluate actual forecast error against ERA5
    actual = reference[
        "reference_" + parameter
    ].reindex(
        pd.MultiIndex.from_frame(
            test[["location", "datetime"]]
        )
    ).to_numpy()

    selected_values = np.array([
        test.iloc[i][f"{predictions[i]}_{parameter}"]
        for i in range(len(test))
    ])

    ecmwf_values = test[
        f"ecmwf_ifs_{parameter}"
    ].to_numpy()

    selected_mae = np.mean(
        np.abs(selected_values - actual)
    )

    ecmwf_mae = np.mean(
        np.abs(ecmwf_values - actual)
    )

    improvement = (
        (ecmwf_mae - selected_mae) / ecmwf_mae * 100
        if ecmwf_mae != 0 else 0
    )

    print("\nML-selected forecast MAE:", round(selected_mae, 4))
    print("ECMWF baseline MAE:", round(ecmwf_mae, 4))
    print("MAE improvement:", round(improvement, 2), "%")

    model_file = os.path.join(
        MODELS_DIR,
        f"{parameter}_selector_12months_v3.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "parameter": parameter,
            "training_end": "2026-07-31",
            "test_start": "2026-08-01"
        },
        model_file
    )

    print("\nSaved model:", model_file)

    summary.append({
        "parameter": parameter,
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy_percent": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "ml_selected_mae": round(selected_mae, 4),
        "ecmwf_mae": round(ecmwf_mae, 4),
        "mae_improvement_percent": round(improvement, 2)
    })

summary_df = pd.DataFrame(summary)

summary_file = os.path.join(
    RESULTS,
    "step46B_12month_training_summary.csv"
)

summary_df.to_csv(summary_file, index=False)

print("\n" + "=" * 70)
print("FINAL 12-MONTH TRAINING SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print("\nSaved summary:", summary_file)
print("\nSTEP 46B COMPLETE")
