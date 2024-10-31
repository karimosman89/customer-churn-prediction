
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Drop unnecessary columns, if any
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Convert categorical columns to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.fillna(df.mean())  # Fill missing values in TotalCharges
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

