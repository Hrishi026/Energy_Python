import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate sample electricity prices with some patterns
    np.random.seed(42)
    base_price = 5.0
    seasonal_pattern = np.sin(2 * np.pi * np.arange(len(timestamps)) / 24) * 2  # Daily seasonality
    weekly_pattern = np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 7)) * 1.5  # Weekly seasonality
    noise = np.random.normal(0, 0.5, len(timestamps))
    prices = base_price + seasonal_pattern + weekly_pattern + noise
    
    # Create electricity prices DataFrame
    prices_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices
    })
    
    # Generate sample weather data
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
    for city in cities:
        # Temperature with daily and weekly patterns
        base_temp = 25 + np.random.normal(0, 5)  # Different base temperature for each city
        temp_pattern = np.sin(2 * np.pi * np.arange(len(timestamps)) / 24) * 5  # Daily temperature variation
        temp_noise = np.random.normal(0, 1, len(timestamps))
        temperatures = base_temp + temp_pattern + temp_noise
        
        # Humidity with inverse relationship to temperature
        humidity = 80 - (temperatures - base_temp) + np.random.normal(0, 5, len(timestamps))
        humidity = np.clip(humidity, 0, 100)
        
        weather_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
            'humidity': humidity
        })
        
        # Save weather data
        weather_df.to_csv(os.path.join(data_dir, f'weather_{city}_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.csv'), index=False)
    
    # Generate sample demand data
    for city in cities:
        # Demand with daily and weekly patterns
        base_demand = 1000 + np.random.normal(0, 100)  # Different base demand for each city
        demand_pattern = np.sin(2 * np.pi * np.arange(len(timestamps)) / 24) * 200  # Daily demand variation
        weekly_pattern = np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 7)) * 150  # Weekly demand variation
        demand_noise = np.random.normal(0, 50, len(timestamps))
        demand = base_demand + demand_pattern + weekly_pattern + demand_noise
        
        demand_df = pd.DataFrame({
            'timestamp': timestamps,
            'demand': demand
        })
        
        # Save demand data
        demand_df.to_csv(os.path.join(data_dir, f'demand_{city}_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.csv'), index=False)
    
    # Save electricity prices
    prices_df.to_csv(os.path.join(data_dir, f'electricity_prices_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.csv'), index=False)
    
    print("Sample data generated successfully!")

if __name__ == "__main__":
    generate_sample_data() 