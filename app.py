import streamlit as st
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data
from src.model import predict_churn, model, label_encoders, scaler

# Load model and preprocessing encoders/scalers
# In this example, the model and preprocessors are pre-trained and loaded directly from memory.

# Streamlit app layout
st.title("Customer Churn Prediction App")
st.write("""
This application predicts whether a customer is likely to churn based on input data.
""")

# Collect input data from the user
def user_input_features():
    st.sidebar.header('Input Customer Data')
    
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.slider('Monthly Charges', 0, 120, 50)
    total_charges = st.sidebar.slider('Total Charges', 0, 6000, 500)
    
    # Convert input to DataFrame
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Preprocess the input data
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

input_df = scaler.transform(input_df)

# Predict churn using the trained model
if st.button('Predict'):
    prediction = predict_churn(pd.DataFrame(input_df))
    st.subheader('Prediction Result:')
    st.write("This customer is likely to churn." if prediction == 1 else "This customer is not likely to churn.")

    # Add some basic retention strategy information if they are likely to churn
    if prediction == 1:
        st.write("""
        ### Suggested Retention Strategies:
        - Offer a discounted rate for the next few months.
        - Consider offering a long-term contract to lock in the customer.
        - Provide a personalized support representative to address issues.
        """)

st.write("Adjust the sliders and dropdowns on the left to update customer details and predict churn.")
