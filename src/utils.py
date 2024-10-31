

import pandas as pd

def load_data(filepath: str):
    """
    Load the dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(filepath)
