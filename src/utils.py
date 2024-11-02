import pandas as pd
import joblib
import os

def load_data(filepath: str):
    """
    Load the dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def load_model(model_name: str):
    """
    Load the trained model, label encoders, and scaler from the specified path.

    Parameters:
        model_name (str): The name of the model to load.

    Returns:
        tuple: Loaded model, label encoders, and scaler.
    """
    # Define the path to the models directory
    models_dir = os.path.join('models', model_name.replace(" ", "_").lower())  # Adjust the naming convention as necessary

    # Load the model
    model_path = os.path.join(models_dir, f'{model_name.replace(" ", "_").lower()}.joblib')
    model = joblib.load(model_path)

    # Load the label encoders
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))

    # Load the scaler
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))

    return model, label_encoders, scaler
