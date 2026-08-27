import requests
import pandas as pd
import os
import time

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

TOTAL_RECORDS = 500_000
BATCH_SIZE = 50_000

OUTPUT_FILE = "data/raw/chicago_crimes_500k.csv"

# --------------------------------------------------
# Download data in batches
# --------------------------------------------------

all_data = []

print("Starting Chicago Crime Dataset download...")
print(f"Target records: {TOTAL_RECORDS:,}")
print(f"Batch size: {BATCH_SIZE:,}")
print()

for offset in range(0, TOTAL_RECORDS, BATCH_SIZE):

    print(
        f"Downloading records "
        f"{offset + 1:,} - {min(offset + BATCH_SIZE, TOTAL_RECORDS):,}..."
    )

    params = {
        "$limit": BATCH_SIZE,
        "$offset": offset,
        "$order": "date DESC, id DESC"
    }

    response = requests.get(API_URL, params=params, timeout=120)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text[:500])
        raise SystemExit

    batch = response.json()

    if not batch:
        print("No more records available.")
        break

    all_data.extend(batch)

    print(f"Downloaded: {len(batch):,} records")
    print(f"Total downloaded: {len(all_data):,}")
    print()

    time.sleep(1)

# --------------------------------------------------
# Save as CSV
# --------------------------------------------------

df = pd.DataFrame(all_data)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("DOWNLOAD COMPLETED")
print("=" * 60)
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)