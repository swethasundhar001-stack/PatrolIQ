import pandas as pd
import os

# --------------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------------

INPUT_FILE = "data/cleaned/chicago_crimes_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("EDA STARTED")
print("=" * 60)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# --------------------------------------------------
# Basic Information
# --------------------------------------------------

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# --------------------------------------------------
# Crime Type Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 CRIME TYPES")
print("=" * 60)

crime_counts = df["primary_type"].value_counts()

print(crime_counts.head(10))

# --------------------------------------------------
# Year-wise Crime Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("YEAR-WISE CRIME COUNT")
print("=" * 60)

year_counts = df["year"].value_counts().sort_index()

print(year_counts)

# --------------------------------------------------
# Location Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 CRIME LOCATIONS")
print("=" * 60)

location_counts = df["location_description"].value_counts()

print(location_counts.head(10))

# --------------------------------------------------
# Save EDA Results
# --------------------------------------------------

OUTPUT_FILE = "reports/eda_summary.txt"

os.makedirs("reports", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("PATROLIQ - EDA SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("TOP 10 CRIME TYPES\n")
    f.write("-" * 40 + "\n")
    f.write(crime_counts.head(10).to_string())

    f.write("\n\nYEAR-WISE CRIME COUNT\n")
    f.write("-" * 40 + "\n")
    f.write(year_counts.to_string())

    f.write("\n\nTOP 10 CRIME LOCATIONS\n")
    f.write("-" * 40 + "\n")
    f.write(location_counts.head(10).to_string())

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"Results saved to: {OUTPUT_FILE}")