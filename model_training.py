import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data, features, target):
    """Train the machine learning model."""
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv('processed_titanic.csv')
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    model, X_train, X_test, y_train, y_test = train_model(data, features, 'Survived')
    joblib.dump(model, 'titanic_model.pkl')  # Save the model
