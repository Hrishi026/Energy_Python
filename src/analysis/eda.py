import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EDA:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.plots_dir = os.path.join(self.data_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Successfully loaded data from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise

    def analyze_price_trends(self, prices_df: pd.DataFrame, 
                           save_plot: bool = True) -> Dict:
        """
        Analyze electricity price trends
        """
        try:
            # Convert timestamp to datetime if needed
            if 'timestamp' in prices_df.columns:
                prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
            
            # Calculate basic statistics
            stats = {
                'mean_price': prices_df['price'].mean(),
                'std_price': prices_df['price'].std(),
                'min_price': prices_df['price'].min(),
                'max_price': prices_df['price'].max(),
                'price_range': prices_df['price'].max() - prices_df['price'].min()
            }
            
            # Create trend plot
            plt.figure(figsize=(15, 8))
            plt.plot(prices_df['timestamp'], prices_df['price'])
            plt.title('Electricity Prices Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Price (INR/kWh)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(os.path.join(self.plots_dir, 'price_trends.png'))
            plt.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing price trends: {str(e)}")
            raise

    def analyze_seasonality(self, prices_df: pd.DataFrame,
                          save_plot: bool = True) -> Dict:
        """
        Analyze daily and weekly seasonality
        """
        try:
            # Ensure timestamp is datetime
            if 'timestamp' in prices_df.columns:
                prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
            
            # Extract time components
            prices_df['hour'] = prices_df['timestamp'].dt.hour
            prices_df['day_of_week'] = prices_df['timestamp'].dt.dayofweek
            
            # Calculate average prices by hour and day
            hourly_avg = prices_df.groupby('hour')['price'].mean()
            daily_avg = prices_df.groupby('day_of_week')['price'].mean()
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Hourly pattern
            ax1.plot(hourly_avg.index, hourly_avg.values)
            ax1.set_title('Average Price by Hour')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Average Price (INR/kWh)')
            ax1.grid(True)
            
            # Daily pattern
            ax2.plot(daily_avg.index, daily_avg.values)
            ax2.set_title('Average Price by Day of Week')
            ax2.set_xlabel('Day of Week (0=Monday)')
            ax2.set_ylabel('Average Price (INR/kWh)')
            ax2.grid(True)
            
            if save_plot:
                plt.savefig(os.path.join(self.plots_dir, 'seasonality.png'))
            plt.close()
            
            return {
                'hourly_pattern': hourly_avg.to_dict(),
                'daily_pattern': daily_avg.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            raise

    def analyze_correlations(self, prices_df: pd.DataFrame, 
                           weather_df: pd.DataFrame,
                           demand_df: pd.DataFrame,
                           save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze correlations between prices, weather, and demand
        """
        try:
            # Merge datasets
            merged_df = pd.merge(prices_df, weather_df, on='timestamp', how='inner')
            merged_df = pd.merge(merged_df, demand_df, on='timestamp', how='inner')
            
            # Calculate correlation matrix
            corr_matrix = merged_df[['price', 'temperature', 'humidity', 'demand']].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            
            if save_plot:
                plt.savefig(os.path.join(self.plots_dir, 'correlation_matrix.png'))
            plt.close()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            raise

    def analyze_price_distribution(self, prices_df: pd.DataFrame,
                                  save_plot: bool = True) -> None:
        """
        Analyze price distribution
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=prices_df, x='price', bins=50)
            plt.title('Price Distribution')
            plt.xlabel('Price (INR/kWh)')
            plt.ylabel('Count')
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(os.path.join(self.plots_dir, 'price_distribution.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error analyzing price distribution: {str(e)}")
            raise

    def detect_outliers(self, prices_df: pd.DataFrame,
                       save_plot: bool = True) -> Dict:
        """
        Detect and analyze outliers in price data
        """
        try:
            # Calculate IQR
            Q1 = prices_df['price'].quantile(0.25)
            Q3 = prices_df['price'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = prices_df[
                (prices_df['price'] < lower_bound) | 
                (prices_df['price'] > upper_bound)
            ]
            
            # Create box plot
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=prices_df['price'])
            plt.title('Price Distribution with Outliers')
            
            if save_plot:
                plt.savefig(os.path.join(self.plots_dir, 'outliers.png'))
            plt.close()
            
            return {
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(prices_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': outliers
            }
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise

def main():
    try:
        eda = EDA()
        
        # Get the most recent data files
        data_files = os.listdir(eda.data_dir)
        price_files = [f for f in data_files if f.startswith('electricity_prices_')]
        weather_files = [f for f in data_files if f.startswith('weather_')]
        demand_files = [f for f in data_files if f.startswith('demand_')]
        
        if not price_files:
            raise FileNotFoundError("No electricity price data files found")
            
        # Use the most recent file
        latest_price_file = max(price_files)
        latest_weather_file = max(weather_files)
        latest_demand_file = max(demand_files)
        
        # Load data
        prices_df = eda.load_data(latest_price_file)
        weather_df = eda.load_data(latest_weather_file)
        demand_df = eda.load_data(latest_demand_file)
        
        # Run analysis
        price_stats = eda.analyze_price_trends(prices_df)
        seasonality = eda.analyze_seasonality(prices_df)
        correlations = eda.analyze_correlations(prices_df, weather_df, demand_df)
        outliers = eda.detect_outliers(prices_df)
        eda.analyze_price_distribution(prices_df)
        
        # Log results
        logger.info(f"Price Statistics: {price_stats}")
        logger.info(f"Seasonality Analysis: {seasonality}")
        logger.info(f"Outlier Analysis: {outliers}")
        
        logger.info("EDA completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main EDA process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 