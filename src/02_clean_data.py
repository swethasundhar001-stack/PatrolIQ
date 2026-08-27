import pandas as pd
import os

# --------------------------------------------------
# Load raw dataset
# --------------------------------------------------

INPUT_FILE = "data/raw/chicago_crimes_500k.csv"
OUTPUT_FILE = "data/cleaned/chicago_crimes_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATA CLEANING STARTED")
print("=" * 60)

print(f"Original rows    : {len(df):,}")
print(f"Original columns : {len(df.columns)}")

# --------------------------------------------------
# Remove duplicate records
# --------------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicates:,}")

df = df.drop_duplicates()

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------

# Remove rows where important crime information is missing
important_columns = [
    "date",
    "primary_type",
    "description"
]

df = df.dropna(subset=important_columns)

# Fill missing text values
text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].fillna("UNKNOWN")

# --------------------------------------------------
# Convert date column
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove invalid dates
df = df.dropna(subset=["date"])

# --------------------------------------------------
# Clean text columns
# --------------------------------------------------

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# --------------------------------------------------
# Create useful date features
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_name()

# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print(f"Final rows    : {len(df):,}")
print(f"Final columns : {len(df.columns)}")
print(f"Saved to      : {OUTPUT_FILE}")

print("=" * 60)