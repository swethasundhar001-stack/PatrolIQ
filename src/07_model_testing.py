import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# File paths
# --------------------------------------------------

MODEL_FILE = "models/crime_prediction_model.pkl"
ENCODER_FILE = "models/crime_label_encoder.pkl"
FEATURE_FILE = "models/feature_info.pkl"
DATA_FILE = "data/processed/crime_features.csv"


# --------------------------------------------------
# Start
# --------------------------------------------------

print("=" * 60)
print("MODEL TESTING STARTED")
print("=" * 60)


# --------------------------------------------------
# Load saved files
# --------------------------------------------------

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
# Get feature information
# --------------------------------------------------

if isinstance(feature_info, dict):
    features = feature_info.get(
        "features",
        feature_info.get("feature_columns", [])
    )
else:
    features = feature_info


print("\nFeatures used by model:")
print(features)


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

df = df.dropna(subset=["primary_type"])

X = df[features].copy()
y = df["primary_type"].copy()


# --------------------------------------------------
# Convert boolean columns
# --------------------------------------------------

for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)


# --------------------------------------------------
# Handle missing values
# --------------------------------------------------

X = X.fillna(0)


# --------------------------------------------------
# Encode target
# --------------------------------------------------

y_encoded = encoder.transform(y)


# --------------------------------------------------
# Same Train/Test Split used during training
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
# Model Prediction
# --------------------------------------------------

print("\nRunning predictions...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)


# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

actual_labels = encoder.inverse_transform(y_test)
predicted_labels = encoder.inverse_transform(y_pred)

sample = pd.DataFrame({
    "Actual Crime": actual_labels[:20],
    "Predicted Crime": predicted_labels[:20]
})

print(sample.to_string(index=False))


# --------------------------------------------------
# Completed
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TESTING COMPLETED")
print("=" * 60)
