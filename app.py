import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load models
model_names = [
    "AdaBoost",
    "CatBoost",
    "Extra_Trees",
    "Gaussian_Naive_Bayes",
    "Gradient_Boosting",
    "K-Nearest_Neighbors",
    "LightGBM",
    "Logistic_Regression",
    "Random_Forest",
    "Support_Vector_Classifier",
    "XGBoost"
]

# Create a mapping of model names to file paths
models = {}
for model_name in model_names:
    model_path = f'models/{model_name}.joblib'
    models[model_name] = joblib.load(model_path)

# Function to preprocess input data
def preprocess_input(data):
    # Create a copy to avoid modifying the original DataFrame
    data_processed = data.copy()
    
    # Example of label encoding for categorical features
    # You'll need to define label encoders based on your training data
    label_encoders = {
        "gender": LabelEncoder().fit(["Male", "Female"]),
        "Partner": LabelEncoder().fit(["Yes", "No"]),
        "Dependents": LabelEncoder().fit(["Yes", "No"]),
        "PhoneService": LabelEncoder().fit(["Yes", "No"]),
        "MultipleLines": LabelEncoder().fit(["Yes", "No", "No phone service"]),
        "InternetService": LabelEncoder().fit(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "OnlineBackup": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "DeviceProtection": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "TechSupport": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "StreamingTV": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "StreamingMovies": LabelEncoder().fit(["Yes", "No", "No internet service"]),
        "Contract": LabelEncoder().fit(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": LabelEncoder().fit(["Yes", "No"]),
        "PaymentMethod": LabelEncoder().fit(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    }

    # Encode each feature using the appropriate LabelEncoder
    for column in label_encoders:
        data_processed[column] = label_encoders[column].transform(data_processed[column])
    
    return data_processed

# Streamlit app
st.title("Customer Churn Prediction")

# Create a dropdown for model selection
selected_model_name = st.selectbox("Select Model", model_names)

# Input fields for features
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])  
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Collect user inputs into a DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# When the button is pressed
if st.button("Predict"):
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Get the selected model
    selected_model = models[selected_model_name]
    
    # Make predictions
    prediction = selected_model.predict(processed_input)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")
