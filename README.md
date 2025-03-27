# Indian Electricity Price Forecasting

This project implements a comprehensive system for forecasting and backtesting electricity prices in India's power market. It includes data collection, analysis, modeling, and visualization components.

## Features

- Historical electricity price data collection from Indian power markets
- Weather and demand data integration
- Exploratory Data Analysis (EDA)
- Multiple forecasting models (ARIMA and XGBoost)
- Backtesting engine
- Performance analysis and metrics
- Interactive dashboard using Streamlit

## Project Structure

```
├── data/                   # Data storage directory
├── src/                    # Source code
│   ├── data/              # Data collection modules
│   ├── analysis/          # EDA and analysis modules
│   ├── models/            # Forecasting models
│   ├── backtesting/       # Backtesting engine
│   └── visualization/     # Visualization and dashboard
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/indian-electricity-forecasting.git
cd indian-electricity-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your API keys in a `.env` file:
```
WEATHER_API_KEY=your_weather_api_key
POWER_MARKET_API_KEY=your_power_market_api_key
```

2. Run the data collection script:
```bash
python src/data/collect_data.py
```

3. Launch the dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

## Models

The project implements two main forecasting models:

1. ARIMA (Autoregressive Integrated Moving Average)
   - Suitable for capturing linear patterns and seasonality
   - Incorporates weather and demand features

2. XGBoost
   - Handles non-linear relationships
   - Can capture complex patterns in the data

## Performance Metrics

The models are evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 