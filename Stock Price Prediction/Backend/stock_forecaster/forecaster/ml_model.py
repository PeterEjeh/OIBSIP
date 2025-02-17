import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
from pathlib import Path


class StockPredictor:
    def __init__(self):
        model_path = Path(__file__).parent / 'saved_models'
        self.model = joblib.load(model_path / 'ridge_model.joblib')
        self.scaler = joblib.load(model_path / 'scaler.joblib')
        self.poly = joblib.load(model_path / 'poly_features.joblib')

    def prepare_features(self, data):
        df = pd.DataFrame([data])

        # Calculate additional features
        df['volatility'] = 0  # For real-time prediction, we might need historical data
        df['ma5'] = df['last_value']  # Same as above
        df['ma20'] = df['last_value']
        df['momentum'] = 0
        df['price_range'] = df['high_value'] - df['low_value']

        feature_cols = [
            'open_value', 'high_value', 'low_value', 'turnover',
            'change_prev_close_percentage', 'volatility', 'ma5', 'ma20',
            'momentum', 'price_range'
        ]

        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)

        return X_poly

    def predict(self, data):
        features = self.prepare_features(data)
        prediction = self.model.predict(features)[0]
        return float(prediction)