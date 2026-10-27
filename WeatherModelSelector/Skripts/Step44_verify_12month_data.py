import pandas as pd
import os

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

FORECAST_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast_12months.csv"
)

REFERENCE_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference_12months.csv"
)

print("=" * 70)
print("STEP 44 - VERIFY 12-MONTH DATA")
print("=" * 70)

print("\nReading forecast data...")
forecast = pd.read_csv(FORECAST_FILE)

print("Forecast rows:", len(forecast))
print("Forecast locations:", forecast["location"].nunique())
print("Forecast models:", forecast["model"].unique())

print("\nReading ERA5 reference data...")
reference = pd.read_csv(REFERENCE_FILE)

print("Reference rows:", len(reference))
print("Reference locations:", reference["location"].nunique())

# Convert datetime
forecast["datetime"] = pd.to_datetime(forecast["datetime"])
reference["datetime"] = pd.to_datetime(reference["datetime"])

print("\n" + "=" * 70)
print("DATE RANGE CHECK")
print("=" * 70)

print("Forecast start:", forecast["datetime"].min())
print("Forecast end  :", forecast["datetime"].max())

print("Reference start:", reference["datetime"].min())
print("Reference end  :", reference["datetime"].max())

# Unique key checks
forecast_keys = forecast[
    ["location", "datetime", "model"]
].drop_duplicates()

reference_keys = reference[
    ["location", "datetime"]
].drop_duplicates()

print("\n" + "=" * 70)
print("UNIQUE KEY CHECK")
print("=" * 70)

print(
    "Forecast unique (location, datetime, model):",
    len(forecast_keys)
)

print(
    "Reference unique (location, datetime):",
    len(reference_keys)
)

# Location comparison
forecast_locations = set(forecast["location"].unique())
reference_locations = set(reference["location"].unique())

print("\n" + "=" * 70)
print("LOCATION CHECK")
print("=" * 70)

print("Locations only in forecast:")
print(sorted(forecast_locations - reference_locations))

print("\nLocations only in reference:")
print(sorted(reference_locations - forecast_locations))

# Merge
merged = forecast.merge(
    reference,
    on=["location", "datetime"],
    how="left",
    validate="many_to_one"
)

print("\n" + "=" * 70)
print("MERGE CHECK")
print("=" * 70)

print("Merged rows:", len(merged))

expected_rows = 31 * 8760 * 4

print("Expected rows:", expected_rows)

# Check missing reference values
reference_columns = [
    "reference_temperature",
    "reference_humidity",
    "reference_precipitation",
    "reference_wind_speed",
    "reference_pressure"
]

print("\nMissing reference values after merge:")

for col in reference_columns:
    print(
        f"{col}:",
        merged[col].isna().sum()
    )

# Rows per location
print("\n" + "=" * 70)
print("FORECAST ROWS PER LOCATION")
print("=" * 70)

print(
    forecast.groupby("location").size().value_counts().sort_index()
)

print("\n" + "=" * 70)
print("REFERENCE ROWS PER LOCATION")
print("=" * 70)

print(
    reference.groupby("location").size().value_counts().sort_index()
)

# Model counts
print("\n" + "=" * 70)
print("FORECAST MODEL COUNTS")
print("=" * 70)

print(
    forecast["model"].value_counts()
)

# Final status
print("\n" + "=" * 70)
print("VERIFICATION RESULT")
print("=" * 70)

success = (
    len(forecast) == 1086240
    and len(reference) == 271560
    and len(forecast_keys) == 1086240
    and len(reference_keys) == 271560
    and len(merged) == 1086240
    and len(forecast_locations) == 31
    and len(reference_locations) == 31
    and merged[reference_columns].isna().sum().sum() == 0
)

if success:
    print("✅ ALL CHECKS PASSED")
    print("The 12-month datasets are ready for ML feature creation.")
else:
    print("❌ SOME CHECKS FAILED")
    print("Do NOT train the model yet.")

print("=" * 70)
