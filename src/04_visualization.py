import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------------

INPUT_FILE = "data/cleaned/chicago_crimes_cleaned.csv"
OUTPUT_DIR = "reports/figures"

df = pd.read_csv(INPUT_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("VISUALIZATION STARTED")
print("=" * 60)

# --------------------------------------------------
# 1. Top 10 Crime Types
# --------------------------------------------------

crime_counts = df["primary_type"].value_counts().head(10)

plt.figure(figsize=(10, 6))
crime_counts.sort_values().plot(kind="barh")
plt.title("Top 10 Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/top_10_crime_types.png")
plt.close()

print("Created: top_10_crime_types.png")

# --------------------------------------------------
# 2. Year-wise Crime Trend
# --------------------------------------------------

year_counts = df["year"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
year_counts.plot(kind="line", marker="o")
plt.title("Year-wise Crime Trend")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/year_wise_crime_trend.png")
plt.close()

print("Created: year_wise_crime_trend.png")

# --------------------------------------------------
# 3. Top 10 Crime Locations
# --------------------------------------------------

location_counts = df["location_description"].value_counts().head(10)

plt.figure(figsize=(10, 6))
location_counts.sort_values().plot(kind="barh")
plt.title("Top 10 Crime Locations")
plt.xlabel("Number of Crimes")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/top_10_crime_locations.png")
plt.close()

print("Created: top_10_crime_locations.png")

# --------------------------------------------------
# Completed
# --------------------------------------------------

print("=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)
print(f"Charts saved to: {OUTPUT_DIR}")