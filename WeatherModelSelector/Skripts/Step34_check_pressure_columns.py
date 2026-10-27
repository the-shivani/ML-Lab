import os
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

FILE = os.path.join(DATA_PATH, "multivariable_training_with_history.csv")

print("=" * 70)
print("STEP 34: CHECK ACTUAL PRESSURE COLUMNS")
print("=" * 70)

print("\nLoading historical training data...")

df = pd.read_csv(FILE)

print("Rows:", len(df))
print("Columns:", len(df))

print("\n" + "=" * 70)
print("ALL PRESSURE-RELATED COLUMNS")
print("=" * 70)

pressure_cols = [
    c for c in df.columns
    if "pressure" in c.lower()
]

if pressure_cols:
    for c in pressure_cols:
        print(c)
else:
    print("NO PRESSURE COLUMNS FOUND")

print("\n" + "=" * 70)
print("PRESSURE COLUMN COUNT:", len(pressure_cols))
print("=" * 70)