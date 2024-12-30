import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import shap
import matplotlib.pyplot as plt



def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column, categorical_columns):
    """
    Preprocess the data: handle missing values, encode categorical data, and split into train/test sets.
    """
    # Handle missing values (drop rows with missing target or features)
    data = data.dropna(subset=[target_column] + categorical_columns)

    # Create additional features
    data['Margin'] = data['TotalPremium'] - data['TotalClaims']
    data['RiskRatio'] = data['TotalClaims'] / (data['TotalPremium'] + 1e-9)  # Avoid division by zero

    # Feature and target separation
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical data
    encoder = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), categorical_columns)],
        remainder='passthrough'
    )

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(encoder.fit_transform(X_train))
    X_test = scaler.transform(encoder.transform(X_test))

    return X_train, X_test, y_train, y_test, encoder
