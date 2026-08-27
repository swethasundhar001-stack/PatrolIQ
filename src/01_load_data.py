import pandas as pd

# Dataset path
DATA_PATH = "data/raw/chicago_crimes_500k.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("=" * 60)