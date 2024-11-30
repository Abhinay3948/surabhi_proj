import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="Crude Oil Price Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🛢️ Crude Oil Price Real-Time Prediction Dashboard")
st.markdown("""
This dashboard provides real-time crude oil price analysis and predictions using machine learning.
Data is updated every minute to simulate real-time market conditions.
""")

# Function to simulate real-time data by fetching from Yahoo Finance
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_real_time_data():
    try:
        # Fetch WTI Crude Oil data
        oil_data = yf.download("CL=F", period="1y", interval="1d")
        df = oil_data.reset_index()
        
        # Prepare features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
        df['Price_MA5'] = df['Close'].rolling(window=5).mean()
        df['Price_MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to train/load ML model
@st.cache_resource
def get_model(df):
    try:
        # Prepare features for training
        features = ['Hour', 'DayOfWeek', 'Month', 'Volume_MA', 
                   'Price_MA5', 'Price_MA20', 'Volatility']
        X = df[features]
        y = df['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# Main dashboard
def main():
    # Fetch data
    df = fetch_real_time_data()
    
    if df is not None:
        # Train/load model
        model, scaler, features = get_model(df)
        
        if model is not None:
            # Dashboard layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Real-Time Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                                       mode='lines',
                                       name='Actual Price'))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Current Statistics")
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
                
                st.metric("Current Price", f"${current_price:.2f}",
                         f"{price_change:.2f} ({price_change_pct:.2f}%)")
                
                st.metric("30-Day Volatility", 
                         f"{df['Volatility'].iloc[-1]*100:.2f}%")
                
                st.metric("Volume (MA5)", 
                         f"{df['Volume_MA'].iloc[-1]:,.0f}")
            
            # Predictions section
            st.subheader("Price Predictions")
            
            # Prepare latest data for prediction
            latest_features = df[features].iloc[-1:].copy()
            latest_features_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Next Price", 
                         f"${prediction:.2f}",
                         f"{prediction - current_price:.2f}")
            
            with col2:
                confidence = model.score(
                    scaler.transform(df[features]), df['Close'])
                st.metric("Model Confidence", f"{confidence:.2%}")
            
            # Technical Analysis
            st.subheader("Technical Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                                   mode='lines',
                                   name='Price'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Price_MA5'],
                                   mode='lines',
                                   name='MA5'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Price_MA20'],
                                   mode='lines',
                                   name='MA20'))
            fig.update_layout(
                title="Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical Data Table
            st.subheader("Historical Data")
            st.dataframe(
                df[['Date', 'Close', 'Volume', 'Price_MA5', 'Price_MA20', 'Volatility']]
                .tail(10)
                .style.format({
                    'Close': '${:.2f}',
                    'Volume': '{:,.0f}',
                    'Price_MA5': '${:.2f}',
                    'Price_MA20': '${:.2f}',
                    'Volatility': '{:.2%}'
                })
            )
            
            # Add auto-refresh
            st.empty()
            time.sleep(60)  # Refresh every 60 seconds
            st.experimental_rerun()

if __name__ == "__main__":
    main()