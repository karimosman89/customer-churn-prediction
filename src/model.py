import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocess import preprocess_data
from utils import load_data

# Load and preprocess data
data_path = '../data/customer_churn.csv'
df = load_data(data_path)
X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(df)

# Choose a model - Logistic Regression for simplicity
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model Evaluation:\n Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1 Score: {f1}")

def predict_churn(new_data):
    """
    Predict churn for a new customer record.

    Parameters:
        new_data (pd.DataFrame): New customer data as a DataFrame.

    Returns:
        int: Churn prediction (0 for 'No', 1 for 'Yes').
    """
    # Encode and scale new data
    for col, le in label_encoders.items():
        if col in new_data.columns:
            new_data[col] = le.transform(new_data[col])
    new_data = scaler.transform(new_data)

    return model.predict(new_data)
