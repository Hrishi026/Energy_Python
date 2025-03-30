import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherDataCollector:
    def __init__(self):
        self.api_key = os.getenv('WEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def verify_api_key(self) -> bool:
        """Verify if the API key is valid by making a test call."""
        # Test with London coordinates
        test_params = {
            'lat': 51.5074,
            'lon': -0.1278,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=test_params)
            if response.status_code == 401:
                print("Error: Invalid API key. Please check your WEATHER_API_KEY in .env file")
                print("Note: New API keys may take a few hours to activate after registration.")
                return False
            elif response.status_code == 429:
                print("Error: API key has reached its usage limit. Please try again later.")
                return False
            response.raise_for_status()
            print("API key verified successfully!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error verifying API key: {e}")
            return False
        
    def get_weather_data(self, lat: float, lon: float, lang: str = 'en') -> Dict[str, Any]:
        """Fetch weather data using latitude and longitude."""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric',  # Use metric units for temperature
            'lang': lang,       # Language for weather descriptions
            'mode': 'json'      # Response format
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 401:
                print("Error: Invalid API key. Please check your WEATHER_API_KEY in .env file")
                return None
            elif response.status_code == 404:
                print(f"Error: Location not found for coordinates lat={lat}, lon={lon}")
                return None
            elif response.status_code == 429:
                print("Error: API key has reached its usage limit. Please try again later.")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for coordinates lat={lat}, lon={lon}: {e}")
            return None

    def process_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw weather data into desired format."""
        if not data:
            return None
            
        # Extract all available weather data
        weather_data = {
            'timestamp': datetime.fromtimestamp(data['dt']),
            'temperature': data['main']['temp'],
            'feels_like': data['main'].get('feels_like'),
            'temp_min': data['main'].get('temp_min'),
            'temp_max': data['main'].get('temp_max'),
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'sea_level': data['main'].get('sea_level'),
            'grnd_level': data['main'].get('grnd_level'),
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg'),
            'wind_gust': data['wind'].get('gust'),
            'weather_id': data['weather'][0]['id'],
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description'],
            'weather_icon': data['weather'][0]['icon'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
            'rain_1h': data.get('rain', {}).get('1h', 0),
            'snow_1h': data.get('snow', {}).get('1h', 0),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']) if 'sunrise' in data['sys'] else None,
            'sunset': datetime.fromtimestamp(data['sys']['sunset']) if 'sunset' in data['sys'] else None,
            'timezone': data['timezone'],
            'city_id': data['id'],
            'city_name': data['name'],
            'country': data['sys']['country'],
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon']
        }
        
        return weather_data

    def collect_historical_data(self, lat: float, lon: float, 
                              start_date: datetime, end_date: datetime,
                              interval_hours: int = 1, lang: str = 'en') -> pd.DataFrame:
        """Collect historical weather data for a location over a date range."""
        # Verify API key before starting collection
        if not self.verify_api_key():
            return pd.DataFrame()
            
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Note: Free API only provides current weather
            # For historical data, you would need to use a different endpoint or service
            data = self.get_weather_data(lat, lon, lang)
            if data:
                processed_data = self.process_weather_data(data)
                if processed_data:
                    all_data.append(processed_data)
                    print(f"Successfully collected data for {current_date}")
            
            # Wait to respect API rate limits (60 calls/minute for free tier)
            time.sleep(1)
            current_date += timedelta(hours=interval_hours)
        
        if not all_data:
            print("No data was collected. Please check your API key and try again.")
            return pd.DataFrame()
            
        return pd.DataFrame(all_data)

    def collect_weather_data(self, lat: float, lon: float, lang: str = 'en') -> Dict[str, Any]:
        """Collect current weather data for a location."""
        data = self.get_weather_data(lat, lon, lang)
        if data:
            return self.process_weather_data(data)
        return None

    def save_to_csv(self, df: pd.DataFrame, city: str, date: datetime):
        """Save weather data to CSV file."""
        if df.empty:
            print(f"No data to save for {city}.")
            return
            
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Format date for filename
        date_str = date.strftime('%Y-%m-%d')
        filename = f'data/weather_{city}_{date_str}.csv'
        
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

def test_single_call():
    """Test a single API call to verify the setup."""
    try:
        collector = WeatherDataCollector()
        if collector.verify_api_key():
            # Test with London coordinates
            data = collector.get_weather_data(51.5074, -0.1278)
            if data:
                processed_data = collector.process_weather_data(data)
                print("\nTest successful! Sample data:")
                print(processed_data)
                return True
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    try:
        # First test a single API call
        print("Testing API connection...")
        if not test_single_call():
            print("API test failed. Please check your API key and try again.")
            return
            
        # Initialize the collector
        collector = WeatherDataCollector()
        
        # Define cities and their coordinates
        cities = {
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Delhi': (28.6139, 77.2090),
            'Kolkata': (22.5726, 88.3639),
            'Mumbai': (19.0760, 72.8777)
        }
        
        # Set the date to today
        current_date = datetime.now()
        
        # Collect data for each city
        for city, (lat, lon) in cities.items():
            print(f"\nCollecting data for {city}...")
            data = collector.collect_weather_data(lat, lon)
            
            if data:
                # Convert single data point to DataFrame
                df = pd.DataFrame([data])
                collector.save_to_csv(df, city, current_date)
            else:
                print(f"Failed to collect data for {city}")
            
            # Wait to respect API rate limits
            time.sleep(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 