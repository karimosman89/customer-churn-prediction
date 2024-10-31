

import pickle

def save_model(model, file_path):
    """Save the trained model to a file."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    """Load a saved model from a file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

