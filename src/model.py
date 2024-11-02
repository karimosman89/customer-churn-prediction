import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocess import preprocess_data
from src.utils import load_data


# Define paths
DATA_PATH = 'data/customer_churn.csv'
MODELS_DIR = 'models'


def load_model(model_name: str):
    """
    Load a saved model by name.

    Parameters:
        model_name (str): Name of the model to load (e.g., 'Logistic_Regression').

    Returns:
        model: Loaded model object.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model '{model_name}' not found in the models directory.")


def load_label_encoders_and_scaler():
    # Load the label encoders (assuming you saved them in a similar way)
    label_encoders = {}
    for col in categorical_cols:  # Make sure categorical_cols is defined
        label_encoders[col] = load_model(f"{col}_label_encoder")

    # Load the scaler
    scaler_path = 'scaler.joblib'  # Specify the path where you saved the scaler
    scaler = joblib.load(scaler_path)

    return label_encoders, scaler

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using common classification metrics.

    Parameters:
        model: Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    
    return metrics


def main(model_name="Logistic_Regression"):
    """
    Main function to load data, preprocess, load model, and evaluate.

    Parameters:
        model_name (str): Name of the saved model to load and evaluate.
    """
    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(df)

    # Load the chosen model
    model = load_model(model_name)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nModel Evaluation for {model_name}:\n", "\n".join([f"{k.capitalize()}: {v:.4f}" for k, v in metrics.items()]))

    return model, label_encoders, scaler


def predict_churn(new_data, model, label_encoders, scaler):
    """
    Predict churn for a new customer record.
    
    Parameters:
        new_data (pd.DataFrame): New customer data as a DataFrame.
        model: Trained model.
        label_encoders (dict): Label encoders for categorical features.
        scaler (StandardScaler): Scaler used during training.

    Returns:
        int: Churn prediction (0 for 'No', 1 for 'Yes').
    """
    # Specify the exact feature order
    feature_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    # Ensure new_data has columns in the correct order and all expected columns
    new_data = new_data[feature_order]

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in new_data.columns:
            new_data[col] = le.transform(new_data[col])

    # Convert 'TotalCharges' to numeric
    new_data['TotalCharges'] = pd.to_numeric(new_data['TotalCharges'], errors='coerce')
    new_data.fillna(new_data.mean(), inplace=True)  # Handle any NaN values

    # Convert new_data to DataFrame with correct column names before scaling
    new_data = pd.DataFrame(scaler.transform(new_data), columns=feature_order)

    # Make and return the prediction
    return model.predict(new_data)


if __name__ == "__main__":
    # Run main function and evaluate the specified model
    selected_model_name = "Random Forest"  # Change model_name as needed
    model = load_model()
    label_encoders, scaler = load_label_encoders_and_scaler()
    # Example new data (replace with actual data for real use)
    new_customer_data = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [24],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [70.35],
        'TotalCharges': ['1683.6']
    })

    # Predict churn for new customer data
    prediction_results = predict_churn(new_customer_data, selected_model, label_encoders, scaler)
    print("\nNew Customer Prediction:\n", prediction_results)


