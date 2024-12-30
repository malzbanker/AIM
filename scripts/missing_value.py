import pandas as pd
import numpy as np

def handle_missing_values(data):
    """
    Fill missing values in the dataset.
    - Categorical columns: Fill with mode
    - Numerical columns: Fill with mean
    """
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(exclude=np.number).columns

    # Fill missing values
    for col in numerical_cols:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)

    for col in categorical_cols:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

    return data

# Example Usage
if __name__ == '__main__':
    # Create a sample dataset
    sample_data = pd.DataFrame({
        'NumericalCol': [1, 2, np.nan, 4],
        'CategoricalCol': ['A', np.nan, 'B', 'B']
    })

    print("Before Handling Missing Values:")
    print(sample_data)

    # Handle missing values
    processed_data = handle_missing_values(sample_data)

    print("\nAfter Handling Missing Values:")
    print(processed_data)
