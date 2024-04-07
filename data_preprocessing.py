import pandas as pd

def load_data(filename):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filename)

def preprocess_data(data):
    """Preprocess the dataset (handle missing values, convert categorical data, etc.)."""
    data.ffill(inplace=True)  # Using ffill to handle missing values
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    return data

if __name__ == "__main__":
    dataset = load_data('Titanic-Dataset.csv')
    processed_data = preprocess_data(dataset)
    processed_data.to_csv('processed_titanic.csv', index=False)
