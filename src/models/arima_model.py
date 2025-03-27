import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

class ARIMAForecaster:
    def __init__(self):
        """Initialize ARIMA forecaster with model parameters."""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.model = None
        self.order = (1, 1, 1)  # Default ARIMA order (p, d, q)
        self.seasonal_order = (1, 1, 1, 24)  # Default seasonal order (P, D, Q, s)
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load and preprocess data."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        try:
            split_idx = int(len(df) * (1 - test_size))
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
            return train, test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit ARIMA model to training data."""
        try:
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = self.model.fit(disp=False)
            logger.info("Model fitting completed successfully")
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def predict(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """Generate predictions for specified date range."""
        try:
            predictions = self.model.predict(start=start_date, end=end_date)
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate performance metrics."""
        try:
            metrics = {
                'mse': mean_squared_error(actual, predicted),
                'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                'mae': mean_absolute_error(actual, predicted),
                'r2': r2_score(actual, predicted)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_results(self, train: pd.DataFrame, test: pd.DataFrame, predictions: pd.Series) -> None:
        """Plot actual vs predicted values."""
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(train.index, train['price'], label='Training Data')
            plt.plot(test.index, test['price'], label='Actual Test Data')
            plt.plot(predictions.index, predictions, label='Predictions', linestyle='--')
            plt.title('Electricity Price Forecasting Results')
            plt.xlabel('Timestamp')
            plt.ylabel('Price (INR/kWh)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'plots', 'arima_forecast.png'))
            plt.close()
            logger.info("Results plot saved successfully")
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

    def save_metrics(self, metrics: Dict) -> None:
        """Save performance metrics to file."""
        try:
            metrics_file = os.path.join(self.data_dir, 'arima_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

def main():
    try:
        forecaster = ARIMAForecaster()
        
        # Get the most recent data file
        data_files = os.listdir(forecaster.data_dir)
        price_files = [f for f in data_files if f.startswith('electricity_prices_')]
        if not price_files:
            raise FileNotFoundError("No electricity price data files found")
        latest_price_file = max(price_files)
        
        # Load and prepare data
        df = forecaster.load_data(latest_price_file)
        train_data, test_data = forecaster.train_test_split(df)
        
        # Train model
        forecaster.fit(train_data)
        
        # Generate predictions
        predictions = forecaster.predict(test_data.index[0], test_data.index[-1])
        
        # Evaluate model
        metrics = forecaster.evaluate(test_data['price'], predictions)
        logger.info(f"Model Performance Metrics: {metrics}")
        
        # Plot and save results
        forecaster.plot_results(train_data, test_data, predictions)
        forecaster.save_metrics(metrics)
        
        logger.info("Forecasting process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main forecasting process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 