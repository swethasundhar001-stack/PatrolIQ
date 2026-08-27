import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# File paths
# --------------------------------------------------

MODEL_FILE = "models/crime_prediction_model.pkl"
ENCODER_FILE = "models/crime_label_encoder.pkl"
FEATURE_FILE = "models/feature_info.pkl"
DATA_FILE = "data/processed/crime_features.csv"


# --------------------------------------------------
# Load saved files
# --------------------------------------------------

print("=" * 60)
print("MODEL TESTING STARTED")
print("=" * 60)

model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
feature_info = joblib.load(FEATURE_FILE)

df = pd.read_csv(DATA_FILE)

print("Model loaded successfully")
print("Encoder loaded successfully")
print("Feature information loaded successfully")

print(f"Dataset rows    : {len(df):,}")
print(f"Dataset columns : {len(df.columns)}")


# --------------------------------------------------
# Display feature information
# --------------------------------------------------

print("\nFeatures used by model:")

if isinstance(feature_info, dict):
    features = feature_info.get("features", feature_info.get("feature_columns", []))
else:
    features = feature_info

print(features)


print("\n" + "=" * 60)
print("MODEL TESTING SETUP COMPLETED")
print("=" * 60)

# --------------------------------------------------
# Model Information
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(f"Model type : {type(model).__name__}")

# Check whether model has feature information
if hasattr(model, "n_features_in_"):
    print(f"Expected features : {model.n_features_in_}")

# Check model classes
if hasattr(model, "classes_"):
    print(f"Number of classes : {len(model.classes_)}")
    print(f"Classes : {model.classes_}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\n" + "=" * 60)
print("MODEL INFORMATION CHECK COMPLETED")
print("=" * 60)

# --------------------------------------------------
# Actual Model Prediction Test
# --------------------------------------------------

print("\n" + "=" * 60)
print("ACTUAL MODEL PREDICTION TEST")
print("=" * 60)

# Features used during training
X = df[features].copy()

# Target column
y = df["primary_type"]

print(f"Test rows : {len(X):,}")
print(f"Input features : {X.shape[1]}")

# Make predictions
predictions = model.predict(X)

# Convert encoded predictions back to crime names
predicted_labels = encoder.inverse_transform(predictions.astype(int))

# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y, predicted_labels)

print(f"\nModel Accuracy : {accuracy:.4f}")
print(f"Model Accuracy : {accuracy * 100:.2f}%")

# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample = pd.DataFrame({
    "Actual Crime": y.iloc[:20].values,
    "Predicted Crime": predicted_labels[:20]
})

print(sample.to_string(index=False))

print("\n" + "=" * 60)
print("MODEL PREDICTION TEST COMPLETED")
print("=" * 60)