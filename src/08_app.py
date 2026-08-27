import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PatrolIQ - Crime Prediction",
    page_icon="🚔",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_FILE = "models/crime_prediction_model.pkl"
ENCODER_FILE = "models/crime_label_encoder.pkl"

model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚔 PatrolIQ")
st.subheader("AI-Powered Crime Prediction System")

st.write(
    "Enter the crime details below to predict the most likely crime type."
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.header("📋 Crime Information")

col1, col2 = st.columns(2)


with col1:

    crime_year = st.number_input(
        "Crime Year",
        min_value=2001,
        max_value=2030,
        value=2025
    )

    crime_month = st.number_input(
        "Crime Month",
        min_value=1,
        max_value=12,
        value=1
    )

    crime_day = st.number_input(
        "Crime Day",
        min_value=1,
        max_value=31,
        value=1
    )

    crime_hour = st.number_input(
        "Crime Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=0
    )

    is_weekend = st.selectbox(
        "Weekend?",
        [0, 1]
    )

    crime_severity = st.selectbox(
        "Crime Severity",
        [0, 1]
    )

    has_location = st.selectbox(
        "Location Available?",
        [0, 1]
    )


with col2:

    latitude = st.number_input(
        "Latitude",
        value=41.8781,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        value=-87.6298,
        format="%.6f"
    )

    beat = st.number_input(
        "Beat",
        min_value=0,
        value=100
    )

    district = st.number_input(
        "District",
        min_value=0,
        value=1
    )

    ward = st.number_input(
        "Ward",
        min_value=0,
        value=1
    )

    community_area = st.number_input(
        "Community Area",
        min_value=0,
        value=1
    )

    arrest = st.selectbox(
        "Arrest Made?",
        [0, 1]
    )

    domestic = st.selectbox(
        "Domestic Case?",
        [0, 1]
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Crime", use_container_width=True):

    input_data = pd.DataFrame([{
        "crime_year": crime_year,
        "crime_month": crime_month,
        "crime_day": crime_day,
        "crime_hour": crime_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "crime_severity": crime_severity,
        "has_location": has_location,
        "latitude": latitude,
        "longitude": longitude,
        "beat": beat,
        "district": district,
        "ward": ward,
        "community_area": community_area,
        "arrest": arrest,
        "domestic": domestic
    }])


    # Prediction

    prediction = model.predict(input_data)

    predicted_class = prediction[0]

    try:
        crime_name = encoder.inverse_transform(
            [predicted_class]
        )[0]
    except:
        crime_name = str(predicted_class)


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.success(
        f"🚨 Predicted Crime Type: **{crime_name}**"
    )

    st.dataframe(
        input_data,
        width="stretch"
    )