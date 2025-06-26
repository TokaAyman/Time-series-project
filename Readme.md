# 📈 Time Series Sales Forecasting Project

A comprehensive machine learning project that compares 8 different forecasting approaches and deploys the best model through an interactive Streamlit web application.

## 🎯 Project Overview

This project develops and compares multiple time series forecasting models to predict retail sales, focusing on Department 24. The analysis includes traditional statistical methods, modern machine learning approaches, and deep learning techniques, culminating in an interactive web application for real-time forecasting.

## 🏆 Key Results

- **Best Model**: Random Forest Regressor
- **Performance**: MAE ~2,000, RMSE ~2,500
- **Application**: Interactive Streamlit dashboard
- **Forecasting Range**: 1-30 days ahead

## 📊 Dataset Description

- **Source**: Retail sales dataset with weekly granularity
- **Focus**: Department 24 analysis
- **Time Period**: Weekly sales data (Friday-ending weeks)
- **Features**: Historical sales, holidays, economic indicators, weather data
- **Target**: Weekly sales predictions

### Features Used:
- **Sales History**: 5 lagged sales values (Sales_Lag1 to Sales_Lag5)
- **Holiday Indicator**: Binary flag for holiday weeks
- **Economic Factors**: Fuel price, CPI, unemployment rate
- **Weather**: Temperature data
- **Department**: Department identifier

## 🔄 Data Preprocessing Pipeline

1. **Data Filtering**: Extracted Department 24 specific data
2. **Date Processing**: Converted to datetime index with weekly frequency
3. **Stationarity Testing**: Performed ADF test (confirmed stationary)
4. **Missing Values**: Handled missing values in external regressors
5. **Feature Engineering**: Created lagged sales features
6. **Train-Test Split**: Last 30 weeks reserved for testing

## 🤖 Models Implemented & Compared

### Traditional Time Series Models
1. **ARIMA**: Auto-regressive Integrated Moving Average
2. **SARIMA**: Seasonal ARIMA (52-week seasonality)
3. **SARIMAX**: SARIMA with external variables
4. **Holt-Winters**: Exponential smoothing with trend and seasonality

### Machine Learning Models
5. **Random Forest**: Ensemble method with lagged features ⭐ **WINNER**
6. **Prophet**: Facebook's time series forecasting tool

### Deep Learning Models
7. **LSTM**: Long Short-Term Memory neural networks
8. **CNN**: Convolutional Neural Networks for time series

## 📈 Model Performance Results

| Rank | Model | MAE | RMSE | Notes |
|------|-------|-----|------|-------|
| 1 | Random Forest | ~2,000 | ~2,500 | Best overall performance |
| 2 | SARIMAX | Higher | Higher | Good with external variables |
| 3 | Prophet | Higher | Higher | Good for trend capture |
| 4 | LSTM | Higher | Higher | Deep learning approach |
| 5 | CNN | Higher | Higher | Convolutional approach |
| 6 | SARIMA | Higher | Higher | Seasonal patterns |
| 7 | Holt-Winters | Higher | Higher | Traditional smoothing |
| 8 | ARIMA | Highest | Highest | Basic time series |

## 🚀 Streamlit Application

### Application Features
- **Interactive Dashboard**: User-friendly web interface
- **Flexible Forecasting**: 1-30 day predictions
- **Holiday Configuration**: Multiple holiday setting options
- **Real-time Visualization**: Combined historical and forecast charts
- **Data Export**: CSV download functionality

### Input Parameters

#### Historical Sales Data (Required)
- **Sales_Lag1**: Most recent day's sales
- **Sales_Lag2**: Sales from 2 days ago
- **Sales_Lag3**: Sales from 3 days ago
- **Sales_Lag4**: Sales from 4 days ago  
- **Sales_Lag5**: Sales from 5 days ago

#### External Factors
- **Department**: Department number (default: 1)
- **Temperature**: Average temperature in °F (default: 70.0)
- **Fuel Price**: Current fuel price in $ (default: 3.50)
- **CPI**: Consumer Price Index (default: 220.0)
- **Unemployment**: Unemployment rate % (default: 5.0)

#### Holiday Configuration Options
1. **No holidays**: Standard business days
2. **Specify holiday days**: Enter specific days (e.g., 1,3,7)
3. **All days are holidays**: Every day treated as holiday

### Output Components

#### Summary Metrics
- **Total Forecast**: Sum of all predicted sales
- **Daily Average**: Mean daily sales prediction
- **Highest Day**: Maximum predicted sales day
- **Lowest Day**: Minimum predicted sales day

#### Visualizations
- **Combined Chart**: Historical vs. forecast line chart
- **Holiday Markers**: Special indicators for holiday days
- **Interactive Features**: Hover details, zoom capabilities

#### Detailed Results
- **Forecast Table**: Day-by-day predictions with dates
- **Holiday Indicators**: Clear marking of holiday periods
- **CSV Export**: Downloadable results for further analysis

## 🛠️ Technical Stack

### Core Libraries
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, pmdarima
- **Deep Learning**: TensorFlow/Keras
- **Time Series**: statsmodels, Prophet
- **Visualization**: matplotlib, plotly
- **Web App**: Streamlit

### Model Deployment
- **Serialization**: joblib for model persistence
- **Caching**: Streamlit resource caching
- **Error Handling**: Comprehensive exception management

## 📁 Project Structure

```
├── Time_series_final_project.ipynb    # Main analysis notebook
├── streamlit_app.py                   # Web application code
├── sales_forecasting_final_model.pkl  # Trained Random Forest model
├── project_dataset.csv               # Dataset (not included)
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### Running the Application
1. Ensure model file is in the same directory
2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
3. Open browser to localhost:8501

### Using the Jupyter Notebook
1. Install required packages
2. Load your dataset as 'project_dataset.csv'
3. Run cells sequentially for full analysis

## 🔍 Key Insights & Findings

### Model Performance Insights
- **Random Forest superiority**: Outperformed both traditional and deep learning methods
- **Feature importance**: Lagged sales features are most predictive
- **External factors impact**: Economic and weather variables improve accuracy
- **Deep learning limitations**: Complex models didn't provide advantage for this dataset size

### Business Insights
- **Seasonal patterns**: Strong 52-week seasonal cycles detected
- **Holiday impact**: Significant sales variations during holiday periods
- **Economic sensitivity**: Sales correlate with fuel prices and unemployment
- **Weather influence**: Temperature affects consumer purchasing behavior

### Technical Insights
- **Stationarity**: Data preprocessing confirmed statistical requirements
- **Feature engineering**: Lagged variables crucial for ML model success
- **Model selection**: Ensemble methods handle retail data complexity well
- **Deployment**: Streamlit provides effective model operationalization

## 📊 Business Applications

### Immediate Use Cases
- **Inventory Planning**: Optimize stock levels based on demand forecasts
- **Staff Scheduling**: Align workforce with predicted sales volumes
- **Marketing Timing**: Schedule promotions during high-sales periods
- **Budget Planning**: Revenue forecasting for financial planning

## 🔮 Future Enhancements

### Model Improvements
- **Multi-department Support**: Expand to all retail departments
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Confidence Intervals**: Add uncertainty quantification
- **Online Learning**: Implement continuous model updates

### Application Features
- **Real-time Data**: Integration with live data sources
- **Advanced Visualization**: More interactive charts and dashboards
- **A/B Testing**: Framework for model comparison
- **Alert System**: Automated notifications for significant changes


**Note**: This project demonstrates the complete machine learning pipeline from data preprocessing through model deployment, showcasing both theoretical understanding and practical implementation skills in time series forecasting.
