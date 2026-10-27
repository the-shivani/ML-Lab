import os
import sys
import subprocess
from datetime import datetime, timezone
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"

SCRIPT = os.path.join(
    BASE, "scripts", "Step49B_live_31locations.py"
)

SOURCE = os.path.join(
    BASE, "results", "step49B_live_31locations_predictions.csv"
)

ARCHIVE = os.path.join(
    BASE, "data", "forecast_archive"
)

os.makedirs(ARCHIVE, exist_ok=True)

print("=" * 65)
print("STEP 54 - FORECAST SNAPSHOT ARCHIVING")
print("=" * 65)

started = datetime.now(timezone.utc)

print("Collection started UTC:", started.isoformat())

result = subprocess.run(
    [sys.executable, "-u", SCRIPT],
    cwd=BASE
)

if result.returncode != 0:
    raise SystemExit(
        "Forecast collection failed. Snapshot not archived."
    )

finished = datetime.now(timezone.utc)

if not os.path.exists(SOURCE):
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE)

if df.empty:
    raise ValueError("Forecast CSV is empty.")

df["collection_started_utc"] = started.isoformat()
df["collection_finished_utc"] = finished.isoformat()

# For verification, only use observations after collection
# was completed, so the comparison is genuinely prospective.
df["forecast_valid_utc"] = (
    pd.to_datetime(df["datetime"])
    .dt.tz_localize("Asia/Kolkata")
    .dt.tz_convert("UTC")
)

cutoff = pd.Timestamp(finished)

df["prospective_eligible"] = (
    df["forecast_valid_utc"] > cutoff
)

df["lead_hours_since_collection"] = (
    df["forecast_valid_utc"] - cutoff
).dt.total_seconds() / 3600

stamp = finished.strftime("%Y%m%d_%H%M%S_%f")

filename = f"forecast_snapshot_{stamp}.csv"

destination = os.path.join(ARCHIVE, filename)

df.to_csv(destination, index=False)

print("\nSNAPSHOT SUMMARY")
print("Total rows:", len(df))
print("Locations:", df["location"].nunique())
print("Parameters:", df["parameter"].nunique())
print(
    "Prospective-eligible rows:",
    int(df["prospective_eligible"].sum())
)
print("Saved:", destination)
print("\nSTEP 54 COMPLETE")
