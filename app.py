import streamlit as st
import pickle
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go

# Load the model
@st.cache_resource
def load_model():
    try:
        return load('sales_forecasting_modell.pkl')
    except FileNotFoundError:
        st.error("Model file 'sales_forecasting_modell.pkl' not found. Please upload the model file.")
        return None

model = load_model()

# Title and description
st.title("📈 Time Series Sales Forecasting App")
st.write("Predict future sales for multiple days using historical data and external factors")

if model is not None:
    # Sidebar for configuration
    st.sidebar.header("Forecasting Configuration")
    forecast_days = st.sidebar.slider("Number of days to forecast", 1, 30, 7)
    
    # Main input section
    st.header("Historical Sales Data (Last 5 Days)")
    st.write("Enter the sales data for the most recent 5 days:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sales_lag_1 = st.number_input('Sales (1 day ago)', value=1000.0, help="Most recent day's sales")
        sales_lag_2 = st.number_input('Sales (2 days ago)', value=950.0)
        sales_lag_3 = st.number_input('Sales (3 days ago)', value=1100.0)
    
    with col2:
        sales_lag_4 = st.number_input('Sales (4 days ago)', value=980.0)
        sales_lag_5 = st.number_input('Sales (5 days ago)', value=1050.0)
    
    sales_lags = [sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_4, sales_lag_5]
    
    # External factors section
    st.header("External Factors")
    st.write("These factors will be applied to the forecasting period:")
    
    col3, col4 = st.columns(2)
    
    with col3:
        dept = st.number_input("Department", min_value=1, value=1, step=1)
        temperature = st.number_input("Average Temperature (°F)", value=70.0)
        fuel_price = st.number_input("Fuel Price ($)", value=3.50)
    
    with col4:
        cpi = st.number_input("Consumer Price Index", value=220.0)
        unemployment = st.number_input("Unemployment Rate (%)", value=5.0)
    
    # Holiday configuration
    st.header("Holiday Information")
    holiday_option = st.radio(
        "Holiday configuration for forecast period:",
        ["No holidays", "Specify holiday days", "All days are holidays"]
    )
    
    holiday_days = []
    if holiday_option == "Specify holiday days":
        holiday_input = st.text_input(
            "Enter holiday day numbers (comma-separated, e.g., 1,3,7):",
            help="Day 1 = tomorrow, Day 2 = day after tomorrow, etc."
        )
        if holiday_input:
            try:
                holiday_days = [int(x.strip()) for x in holiday_input.split(',')]
                holiday_days = [d for d in holiday_days if 1 <= d <= forecast_days]
            except ValueError:
                st.error("Please enter valid day numbers separated by commas")
    
    # Prediction section
    if st.button("🔮 Generate Forecast", type="primary"):
        try:
            predictions = []
            dates = pd.date_range(start=pd.Timestamp.now().date() + pd.Timedelta(days=1), 
                                periods=forecast_days, freq='D')
            
            # Use current sales lags for rolling predictions
            current_lags = sales_lags.copy()
            
            for day in range(forecast_days):
                # Determine if current day is holiday
                if holiday_option == "All days are holidays":
                    is_holiday = 1
                elif holiday_option == "Specify holiday days":
                    is_holiday = 1 if (day + 1) in holiday_days else 0
                else:
                    is_holiday = 0
                
                # Prepare input data
                input_data = [[*current_lags, is_holiday, dept, temperature, fuel_price, cpi, unemployment]]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                predictions.append(prediction)
                
                # Update lags for next prediction (rolling window)
                current_lags = [prediction] + current_lags[:-1]
            
            # Display results
            st.success(f"✅ Successfully generated {forecast_days}-day forecast!")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Date': dates,
                'Day': [f"Day {i+1}" for i in range(forecast_days)],
                'Predicted_Sales': predictions,
                'Is_Holiday': [1 if (holiday_option == "All days are holidays" or 
                                   (holiday_option == "Specify holiday days" and (i+1) in holiday_days)) 
                              else 0 for i in range(forecast_days)]
            })
            
            # Summary statistics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Total Forecast", f"${sum(predictions):,.2f}")
            with col6:
                st.metric("Daily Average", f"${np.mean(predictions):,.2f}")
            with col7:
                st.metric("Highest Day", f"${max(predictions):,.2f}")
            with col8:
                st.metric("Lowest Day", f"${min(predictions):,.2f}")
            
            # Visualization
            st.header("📊 Forecast Visualization")
            
            # Create combined historical + forecast chart
            historical_dates = pd.date_range(end=pd.Timestamp.now().date(), periods=5, freq='D')
            historical_df = pd.DataFrame({
                'Date': historical_dates,
                'Sales': sales_lags[::-1],  # Reverse to get chronological order
                'Type': 'Historical'
            })
            
            forecast_df = pd.DataFrame({
                'Date': dates,
                'Sales': predictions,
                'Type': 'Forecast'
            })
            
            # Combine data
            combined_df = pd.concat([
                historical_df[['Date', 'Sales', 'Type']],
                forecast_df[['Date', 'Sales', 'Type']]
            ])
            
            # Create plot
            fig = px.line(combined_df, x='Date', y='Sales', color='Type',
                         title='Sales Forecast: Historical vs Predicted',
                         markers=True)
            
            # Highlight holidays in forecast
            if holiday_days and holiday_option == "Specify holiday days":
                holiday_dates = [dates[i-1] for i in holiday_days]
                holiday_sales = [predictions[i-1] for i in holiday_days]
                fig.add_trace(go.Scatter(
                    x=holiday_dates, y=holiday_sales,
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='star'),
                    name='Holidays'
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.header("📋 Detailed Forecast")
            results_display = results_df.copy()
            results_display['Predicted_Sales'] = results_display['Predicted_Sales'].apply(lambda x: f"${x:,.2f}")
            results_display['Is_Holiday'] = results_display['Is_Holiday'].apply(lambda x: "Yes" if x else "No")
            
            st.dataframe(
                results_display,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Day": "Forecast Day",
                    "Predicted_Sales": "Predicted Sales",
                    "Is_Holiday": "Holiday"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"sales_forecast_{forecast_days}_days.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            st.write("Please check your model file and input data.")

else:
    st.warning("Please ensure the model file 'sales_forecasting_final_model.pkl' is in the same directory as this script.")