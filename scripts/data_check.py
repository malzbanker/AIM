import pandas as pd

def check_missing_value(data):
    """
    Function to check for missing values in a DataFrame.

    Args:
        data (DataFrame): The input data to check for missing values.

    Returns:
        Series: A Series containing the count of missing values for each column.
    """
    # Check for missing values
    missing_values = data.isnull().sum()
    
    # Filter and display columns with missing values
    missing_values_with_data = missing_values[missing_values > 0]
    print(missing_values_with_data)
    
    return missing_values_with_data