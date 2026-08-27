import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/processed/crime_features.csv"
MODEL_FILE = "models/crime_prediction_model.pkl"
ENCODER_FILE = "models/crime_label_encoder.pkl"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# --------------------------------------------------
# Target Variable
# --------------------------------------------------

target_column = "primary_type"

df = df.dropna(subset=[target_column])

print(f"\nTarget column: {target_column}")
print(f"Crime types  : {df[target_column].nunique()}")


# --------------------------------------------------
# Select Features
# --------------------------------------------------

feature_columns = [
    "crime_year",
    "crime_month",
    "crime_day",
    "crime_hour",
    "day_of_week",
    "is_weekend",
    "crime_severity",
    "has_location",
    "latitude",
    "longitude",
    "beat",
    "district",
    "ward",
    "community_area",
    "arrest",
    "domestic"
]

feature_columns = [
    col for col in feature_columns
    if col in df.columns
]

X = df[feature_columns].copy()
y = df[target_column].copy()


# --------------------------------------------------
# Convert Boolean Columns
# --------------------------------------------------

for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)


# --------------------------------------------------
# Handle Missing Values
# --------------------------------------------------

X = X.fillna(0)


# --------------------------------------------------
# Encode Target
# --------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print(f"\nFeatures used: {len(feature_columns)}")
print(feature_columns)


# --------------------------------------------------
# Train / Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")


# --------------------------------------------------
# Train Random Forest Model
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_FILE)
joblib.dump(label_encoder, ENCODER_FILE)


# --------------------------------------------------
# Save Feature Information
# --------------------------------------------------

feature_info = {
    "features": feature_columns,
    "target": target_column
}

joblib.dump(
    feature_info,
    "models/feature_info.pkl"
)


# --------------------------------------------------
# Final Information
# --------------------------------------------------

print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"Model saved   : {MODEL_FILE}")
print(f"Encoder saved : {ENCODER_FILE}")
print("Feature info  : models/feature_info.pkl")

print("=" * 60)