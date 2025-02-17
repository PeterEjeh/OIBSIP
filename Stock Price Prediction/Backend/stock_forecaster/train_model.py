import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from pathlib import Path


def prepare_stock_data(data):
    """Prepare stock data for modeling"""
    # Convert 'date' to datetime and create ordinal feature
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])
    data['date_ordinal'] = data['date'].apply(lambda x: x.toordinal())

    # Feature Engineering
    data['daily_return'] = data['last_value'].pct_change()
    data['volatility'] = data['daily_return'].rolling(window=5).std()

    # Add moving averages
    data['ma5'] = data['last_value'].rolling(window=5).mean()
    data['ma20'] = data['last_value'].rolling(window=20).mean()

    # Add price momentum indicators
    data['momentum'] = data['last_value'] - data['last_value'].shift(5)
    data['price_range'] = data['high_value'] - data['low_value']

    # Fill missing values with mean
    data.fillna(data.mean(numeric_only=True), inplace=True)

    return data


def train_and_save_model():
    # Load the data
    file_path = "CBX Stock Trading.csv"  # Update this path
    data = pd.read_csv(file_path)

    # Prepare data
    prepared_data = prepare_stock_data(data)

    # Feature selection
    features = [
        'open_value', 'high_value', 'low_value', 'turnover',
        'change_prev_close_percentage', 'volatility', 'ma5', 'ma20',
        'momentum', 'price_range'
    ]

    X = prepared_data[features]
    y = prepared_data['last_value']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Train model
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 50, 100, 500]}
    ridge_search = GridSearchCV(Ridge(), param_grid, scoring='r2', cv=5)
    ridge_search.fit(X_train_poly, y_train)

    # Create saved_models directory
    save_dir = Path('forecaster/saved_models')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models and preprocessors
    joblib.dump(ridge_search.best_estimator_, save_dir / 'ridge_model.joblib')
    joblib.dump(scaler, save_dir / 'scaler.joblib')
    joblib.dump(poly, save_dir / 'poly_features.joblib')

    # Evaluate model
    y_pred = ridge_search.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Training Results:")
    print(f"Best Alpha: {ridge_search.best_params_['alpha']}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"\nModels saved in: {save_dir.absolute()}")


if __name__ == "__main__":
    train_and_save_model()