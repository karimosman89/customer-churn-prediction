import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def preprocess_data(df):
    """
    Preprocess the data for training.

    Parameters:
        df (pd.DataFrame): The raw dataset.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data split into training and testing sets.
    """
    # Drop customerID column
    df = df.drop(columns=['customerID'])
    
    # Handle 'TotalCharges' conversion
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)  # Drop any rows with missing values after conversion

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

