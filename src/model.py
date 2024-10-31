
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from src.preprocess import load_data, preprocess_data, split_data
from src.utils import save_model

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy and classification report."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

def main():
    # Load and preprocess data
    df = load_data('../data/churn_data.csv')  # Path to the data file
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model, '../models/churn_model.pkl')

if __name__ == "__main__":
    main()

