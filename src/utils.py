

import pandas as pd

def load_data(X, y):
    """
    Combine features and labels into a single DataFrame.

    Parameters:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target labels.

    Returns:
        pd.DataFrame: Combined DataFrame of features and target labels.
    """
    return pd.concat([X, pd.Series(y, name='Churn')], axis=1)
