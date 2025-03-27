import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self):
        """Initialize dashboard with data directory path."""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.plots_dir = os.path.join(self.data_dir, 'plots')
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise

    def load_metrics(self, model_name: str) -> dict:
        """Load model metrics from JSON file."""
        try:
            metrics_file = os.path.join(self.data_dir, f'{model_name}_metrics.json')
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics for {model_name}: {str(e)}")
            return {}

    def plot_price_trends(self, df: pd.DataFrame) -> None:
        """Plot electricity price trends."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Price'
        ))
        fig.update_layout(
            title='Electricity Price Trends',
            xaxis_title='Time',
            yaxis_title='Price (INR/kWh)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_daily_pattern(self, df: pd.DataFrame) -> None:
        """Plot average daily price pattern."""
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['price'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['price'],
            mode='lines+markers',
            name='Average Price'
        ))
        fig.update_layout(
            title='Average Daily Price Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Average Price (INR/kWh)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_weekly_pattern(self, df: pd.DataFrame) -> None:
        """Plot average weekly price pattern."""
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        daily_avg = df.groupby('day_of_week')['price'].mean().reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg['day_name'] = daily_avg['day_of_week'].map(lambda x: days[x])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_avg['day_name'],
            y=daily_avg['price'],
            name='Average Price'
        ))
        fig.update_layout(
            title='Average Weekly Price Pattern',
            xaxis_title='Day of Week',
            yaxis_title='Average Price (INR/kWh)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_price_distribution(self, df: pd.DataFrame) -> None:
        """Plot price distribution."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['price'],
            nbinsx=50,
            name='Price Distribution'
        ))
        fig.update_layout(
            title='Price Distribution',
            xaxis_title='Price (INR/kWh)',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_metrics(self, arima_metrics: dict, xgboost_metrics: dict) -> None:
        """Display model performance metrics."""
        st.subheader('Model Performance Metrics')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('ARIMA Model')
            for metric, value in arima_metrics.items():
                st.metric(label=metric.upper(), value=f"{value:.4f}")
        
        with col2:
            st.write('XGBoost Model')
            for metric, value in xgboost_metrics.items():
                st.metric(label=metric.upper(), value=f"{value:.4f}")

    def run(self):
        """Run the Streamlit dashboard."""
        try:
            st.set_page_config(
                page_title='Electricity Price Forecasting Dashboard',
                page_icon='⚡',
                layout='wide'
            )
            
            st.title('⚡ Electricity Price Forecasting Dashboard')
            
            # Get the most recent data file
            data_files = os.listdir(self.data_dir)
            price_files = [f for f in data_files if f.startswith('electricity_prices_')]
            if not price_files:
                st.error("No electricity price data files found")
                return
            latest_price_file = max(price_files)
            
            # Load data and metrics
            df = self.load_data(latest_price_file)
            arima_metrics = self.load_metrics('arima')
            xgboost_metrics = self.load_metrics('xgboost')
            
            # Date range selector
            st.sidebar.header('Date Range Selection')
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            start_date = st.sidebar.date_input('Start Date', min_date)
            end_date = st.sidebar.date_input('End Date', max_date)
            
            # Filter data based on date range
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df[mask]
            
            if filtered_df.empty:
                st.warning('No data available for the selected date range')
                return
            
            # Display metrics
            self.display_metrics(arima_metrics, xgboost_metrics)
            
            # Display plots
            st.header('Price Analysis')
            
            tab1, tab2, tab3, tab4 = st.tabs(['Price Trends', 'Daily Pattern', 'Weekly Pattern', 'Distribution'])
            
            with tab1:
                self.plot_price_trends(filtered_df)
            
            with tab2:
                self.plot_daily_pattern(filtered_df)
            
            with tab3:
                self.plot_weekly_pattern(filtered_df)
            
            with tab4:
                self.plot_price_distribution(filtered_df)
            
            # Display raw data
            st.header('Raw Data')
            st.dataframe(filtered_df)
            
            logger.info("Dashboard rendered successfully")
            
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

def main():
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Error in main dashboard process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 