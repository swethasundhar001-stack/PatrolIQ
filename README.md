# 🚔 PatrolIQ

## AI-Powered Crime Prediction System

PatrolIQ is a machine learning-based crime prediction system that analyzes historical crime data and predicts the most likely crime type based on time, location, and crime-related features.

## 🎯 Project Objective

The main objective of PatrolIQ is to identify patterns in historical crime data and build a machine learning classification model that predicts the most likely crime category for given crime-related inputs.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Visualization
5. Feature Engineering
6. Machine Learning Model Training
7. Model Testing
8. Streamlit Web Application

## 🤖 Machine Learning

The system uses a trained machine learning classification model to predict the most likely crime type.

### Features Used

- Crime Year
- Crime Month
- Crime Day
- Crime Hour
- Day of Week
- Weekend Indicator
- Crime Severity
- Location Availability
- Latitude
- Longitude
- Beat
- District
- Ward
- Community Area
- Arrest Indicator
- Domestic Case Indicator

## 🚀 Streamlit Application

PatrolIQ provides an interactive Streamlit web application where users can enter crime-related information and receive a predicted crime type.

### Application Features

- User-friendly input interface
- Crime prediction using the trained ML model
- Crime label decoding
- Prediction result display
- Input data preview

## 📁 Project Structure

```text
PatrolIQ/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── models/
│   ├── crime_label_encoder.pkl
│   ├── crime_prediction_model.pkl
│   └── feature_info.pkl
│
├── reports/
│   └── figures/
│
├── src/
│   ├── download_data.py
│   ├── 01_load_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   ├── 04_visualization.py
│   ├── 05_feature_engineering.py
│   ├── 06_model_training.py
│   ├── 07_model_testing.py
│   └── 08_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
