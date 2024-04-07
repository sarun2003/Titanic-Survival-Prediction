from data_preprocessing import load_data, preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model
import pandas as pd

def main():
    # Load and preprocess the data
    raw_data = load_data('Titanic-Dataset.csv')
    processed_data = preprocess_data(raw_data)

    # Save the processed data for future use
    processed_data.to_csv('processed_titanic.csv', index=False)

    # Feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'

    # Train the model
    model, X_train, X_test, y_train, y_test = train_model(processed_data, features, target)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
