import pandas as pd
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/cleaned/chicago_crimes_cleaned.csv"
OUTPUT_FILE = "data/processed/crime_features.csv"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("=" * 60)
print("FEATURE ENGINEERING STARTED")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Original rows    : {len(df):,}")
print(f"Original columns : {len(df.columns)}")


# --------------------------------------------------
# Date Features
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["crime_year"] = df["date"].dt.year
df["crime_month"] = df["date"].dt.month
df["crime_day"] = df["date"].dt.day
df["crime_hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek

# Weekend feature
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# --------------------------------------------------
# Time of Day Feature
# --------------------------------------------------

def get_time_period(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


df["time_period"] = df["crime_hour"].apply(get_time_period)


# --------------------------------------------------
# Crime Severity Feature
# --------------------------------------------------

high_severity_crimes = [
    "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT",
    "ROBBERY",
    "KIDNAPPING",
    "ARSON",
    "ASSAULT",
    "BATTERY"
]

df["crime_severity"] = df["primary_type"].isin(
    high_severity_crimes
).astype(int)


# --------------------------------------------------
# Location Features
# --------------------------------------------------

df["has_location"] = (
    df["latitude"].notna() &
    df["longitude"].notna()
).astype(int)


# --------------------------------------------------
# Select Useful Features
# --------------------------------------------------

feature_columns = [
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "crime_year",
    "crime_month",
    "crime_day",
    "crime_hour",
    "day_of_week",
    "is_weekend",
    "time_period",
    "crime_severity",
    "has_location",
    "latitude",
    "longitude"
]

# Keep only columns that exist
feature_columns = [
    col for col in feature_columns
    if col in df.columns
]

df_features = df[feature_columns].copy()


# --------------------------------------------------
# Handle Missing Values
# --------------------------------------------------

numeric_columns = df_features.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_columns:
    df_features[col] = df_features[col].fillna(0)


categorical_columns = df_features.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    df_features[col] = df_features[col].fillna("UNKNOWN")


# --------------------------------------------------
# Save Processed Data
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Final Information
# --------------------------------------------------

print()
print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"Final rows    : {len(df_features):,}")
print(f"Final columns : {len(df_features.columns)}")
print(f"Saved to      : {OUTPUT_FILE}")

print()
print("Created Features:")

new_features = [
    "crime_year",
    "crime_month",
    "crime_day",
    "crime_hour",
    "day_of_week",
    "is_weekend",
    "time_period",
    "crime_severity",
    "has_location"
]

for feature in new_features:
    if feature in df_features.columns:
        print(f"  ✓ {feature}")

print("=" * 60)