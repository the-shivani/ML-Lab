import os
import joblib

BASE_PATH = r"C:\Projects\WeatherModelSelector"
MODEL_FILE = os.path.join(
    BASE_PATH,
    "models",
    "pressure_selector_v2.pkl"
)

print("=" * 70)
print("STEP 36: CHECK PRESSURE MODEL FILE")
print("=" * 70)

print("\nLoading model...")

obj = joblib.load(MODEL_FILE)

print("\nObject type:")
print(type(obj))

print("\nDictionary keys:")

if isinstance(obj, dict):
    for key in obj.keys():
        print("  ", key)

        value = obj[key]

        print("     Type:", type(value))

else:
    print("Model is not a dictionary.")

print("\n" + "=" * 70)
print("STEP 36 COMPLETE")
print("=" * 70)