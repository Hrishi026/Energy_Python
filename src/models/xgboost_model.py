import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, Tuple, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostForecaster:
    def __init__(self):
        """Initialize XGBoost forecaster with model parameters."""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.model = None
        self.params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load and preprocess data."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features for XGBoost model."""
        try:
            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['day_of_month'] = df['timestamp'].dt.day
            
            # Create lag features
            for lag in [1, 2, 3, 6, 12, 24]:  # Various time lags
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
            
            # Create rolling mean features
            for window in [6, 12, 24]:  # Various window sizes
                df[f'rolling_mean_{window}'] = df['price'].rolling(window=window).mean()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Separate features and target
            features = ['hour', 'day_of_week', 'month', 'day_of_month'] + \
                      [col for col in df.columns if col.startswith(('price_lag_', 'rolling_mean_'))]
            
            X = df[features]
            y = df['price']
            
            return X, y
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split data into training and testing sets."""
        try:
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit XGBoost model to training data."""
        try:
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_train, y_train)
            logger.info("Model fitting completed successfully")
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_feature_importance(self) -> None:
        """Plot feature importance."""
        try:
            importance = self.model.feature_importances_
            feature_names = self.model.get_booster().feature_names
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importance)), importance)
            plt.xticks(range(len(importance)), feature_names, rotation=45)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'plots', 'xgboost_feature_importance.png'))
            plt.close()
            logger.info("Feature importance plot saved successfully")
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise

    def plot_results(self, y_train: pd.Series, y_test: pd.Series, y_pred: np.ndarray,
                    train_dates: pd.Series, test_dates: pd.Series) -> None:
        """Plot actual vs predicted values."""
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(train_dates, y_train, label='Training Data')
            plt.plot(test_dates, y_test, label='Actual Test Data')
            plt.plot(test_dates, y_pred, label='Predictions', linestyle='--')
            plt.title('Electricity Price Forecasting Results')
            plt.xlabel('Timestamp')
            plt.ylabel('Price (INR/kWh)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'plots', 'xgboost_forecast.png'))
            plt.close()
            logger.info("Results plot saved successfully")
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

    def save_metrics(self, metrics: Dict) -> None:
        """Save performance metrics to file."""
        try:
            metrics_file = os.path.join(self.data_dir, 'xgboost_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

def main():
    try:
        forecaster = XGBoostForecaster()
        
        # Get the most recent data file
        data_files = os.listdir(forecaster.data_dir)
        price_files = [f for f in data_files if f.startswith('electricity_prices_')]
        if not price_files:
            raise FileNotFoundError("No electricity price data files found")
        latest_price_file = max(price_files)
        
        # Load and prepare data
        df = forecaster.load_data(latest_price_file)
        X, y = forecaster.prepare_features(df)
        X_train, X_test, y_train, y_test = forecaster.train_test_split(X, y)
        
        # Train model
        forecaster.fit(X_train, y_train)
        
        # Generate predictions
        predictions = forecaster.predict(X_test)
        
        # Evaluate model
        metrics = forecaster.evaluate(y_test, predictions)
        logger.info(f"Model Performance Metrics: {metrics}")
        
        # Plot and save results
        forecaster.plot_feature_importance()
        forecaster.plot_results(y_train, y_test, predictions,
                              df['timestamp'].iloc[:len(y_train)],
                              df['timestamp'].iloc[-len(y_test):])
        forecaster.save_metrics(metrics)
        
        logger.info("Forecasting process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main forecasting process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 