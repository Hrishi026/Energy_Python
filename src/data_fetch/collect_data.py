import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json
from ..utils.api_utils import APIClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        """Initialize data collector with API clients."""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # API Keys
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.power_market_api_key = os.getenv('POWER_MARKET_API_KEY')
        
        # Initialize API clients
        self.weather_client = APIClient(
            base_url="https://api.weatherapi.com/v1",
            api_key=self.weather_api_key,
            rate_limit=30  # Weather API rate limit
        )
        
        self.power_market_client = APIClient(
            base_url="https://api.power-market.com/v1",
            api_key=self.power_market_api_key,
            rate_limit=60  # Power market API rate limit
        )
        
        # Cities for data collection
        self.cities = json.loads(os.getenv('CITIES', '["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]'))
        
        # Date range
        self.start_date = datetime.strptime(os.getenv('START_DATE', '2024-01-01'), '%Y-%m-%d')
        self.end_date = datetime.strptime(os.getenv('END_DATE', '2024-01-31'), '%Y-%m-%d')

    def fetch_electricity_prices(self) -> pd.DataFrame:
        """Fetch electricity prices from power market API."""
        try:
            logger.info("Fetching electricity prices...")
            
            # Prepare request parameters
            params = {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'region': 'IN'  # India region
            }
            
            # Make API request
            data = self.power_market_client.get('prices', params=params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['prices'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save to CSV
            filename = f"electricity_prices_{self.start_date.strftime('%Y-%m-%d')}_{self.end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(os.path.join(self.data_dir, filename), index=False)
            
            logger.info(f"Successfully saved electricity prices to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching electricity prices: {str(e)}")
            raise

    def fetch_weather_data(self, city: str) -> pd.DataFrame:
        """Fetch weather data for a specific city."""
        try:
            logger.info(f"Fetching weather data for {city}...")
            
            # Prepare request parameters
            params = {
                'key': self.weather_api_key,
                'q': city,
                'dt': self.start_date.strftime('%Y-%m-%d'),
                'end_dt': self.end_date.strftime('%Y-%m-%d'),
                'aqi': 'yes'
            }
            
            # Make API request
            data = self.weather_client.get('history.json', params=params)
            
            # Process response
            weather_data = []
            for day in data['forecast']['forecastday']:
                for hour in day['hour']:
                    weather_data.append({
                        'timestamp': pd.to_datetime(hour['time']),
                        'temperature': hour['temp_c'],
                        'humidity': hour['humidity'],
                        'wind_speed': hour['wind_kph'],
                        'precipitation': hour['precip_mm']
                    })
            
            df = pd.DataFrame(weather_data)
            
            # Save to CSV
            filename = f"weather_{city}_{self.start_date.strftime('%Y-%m-%d')}_{self.end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(os.path.join(self.data_dir, filename), index=False)
            
            logger.info(f"Successfully saved weather data for {city} to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {city}: {str(e)}")
            raise

    def fetch_demand_data(self, city: str) -> pd.DataFrame:
        """Fetch electricity demand data for a specific city."""
        try:
            logger.info(f"Fetching demand data for {city}...")
            
            # Prepare request parameters
            params = {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'region': city
            }
            
            # Make API request
            data = self.power_market_client.get('demand', params=params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['demand'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save to CSV
            filename = f"demand_{city}_{self.start_date.strftime('%Y-%m-%d')}_{self.end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(os.path.join(self.data_dir, filename), index=False)
            
            logger.info(f"Successfully saved demand data for {city} to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching demand data for {city}: {str(e)}")
            raise

    def collect_all_data(self) -> None:
        """Collect all required data."""
        try:
            # Fetch electricity prices
            self.fetch_electricity_prices()
            
            # Fetch weather and demand data for each city
            for city in self.cities:
                self.fetch_weather_data(city)
                self.fetch_demand_data(city)
            
            logger.info("Data collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data collection process: {str(e)}")
            raise
        finally:
            # Close API clients
            self.weather_client.close()
            self.power_market_client.close()

def main():
    try:
        collector = DataCollector()
        collector.collect_all_data()
    except Exception as e:
        logger.error(f"Error in main data collection process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 