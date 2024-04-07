import joblib
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

if __name__ == "__main__":
    model = joblib.load('titanic_model.pkl')
    # Assuming X_test and y_test are loaded elsewhere for this standalone execution
    # evaluate_model(model, X_test, y_test)
