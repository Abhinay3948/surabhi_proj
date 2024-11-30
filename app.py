import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# OilPriceAPI key
API_KEY = '2712e3c889f666a399bbd6e230d7f391'

# Function to fetch the latest oil price
def fetch_latest_oil_price(api_key):
    url = "https://api.oilpriceapi.com/v1/prices/latest"
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data:
            return data['data']['price']
        else:
            st.error("Error: 'data' field not found in the response.")
            return None
    else:
        st.error(f"Error fetching latest oil price: {response.status_code}")
        return None

# Load historical data from a CSV file
@st.cache
def load_historical_data():
    # Replace with a valid dataset URL or local path
    data_path = 'https://www.kaggleusercontent.com/sc231997/crude-oil-price'  # Update if needed
    try:
        df = pd.read_csv(data_path)
        df = df[['date', 'price']]  # Ensure required columns exist
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

historical_data = load_historical_data()

# Display the latest oil price
latest_price = fetch_latest_oil_price(API_KEY)
if latest_price:
    st.write(f"Latest Oil Price: ${latest_price:.2f}")

# Display historical data
if not historical_data.empty:
    st.write("Historical Data:")
    st.dataframe(historical_data)

# Function to train a model and predict future prices
def predict_prices(data, hours):
    data['timestamp'] = data['date'].astype(int) / 10**9  # Convert datetime to Unix timestamp
    X = data['timestamp'].values.reshape(-1, 1)
    y = data['price'].values

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate future timestamps for hourly predictions
    future_timestamps = np.array(
        [(data['timestamp'].max() + i * 3600) for i in range(1, hours + 1)]
    ).reshape(-1, 1)

    # Predict prices for future timestamps
    predicted_prices = model.predict(future_timestamps)
    predicted_dates = [pd.to_datetime(ts * 10**9) for ts in future_timestamps.flatten()]

    return predicted_dates, predicted_prices

# Input for hours to forecast
hours_to_forecast = st.number_input("Enter number of hours to forecast", min_value=1, step=1, value=24)

if hours_to_forecast and not historical_data.empty:
    # Predict future prices
    predicted_dates, predicted_prices = predict_prices(historical_data, hours_to_forecast)

    # Create DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Time': predicted_dates,
        'Predicted Price (USD)': predicted_prices
    })

    # Display predictions
    st.write(f"Predicted Prices for the Next {hours_to_forecast} Hours:")
    st.dataframe(predictions_df)

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions_df['Time'], predictions_df['Predicted Price (USD)'], marker='o', color='b', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.set_title(f'Predicted Crude Oil Prices for the Next {hours_to_forecast} Hours')
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Display percentage change and price change
    price_change = predicted_prices[-1] - predicted_prices[0]
    percentage_change = (price_change / predicted_prices[0]) * 100
    st.write(f"Price Change: ${price_change:.2f}")
    st.write(f"Percentage Change: {percentage_change:.2f}%")
else:
    st.warning("Please load valid historical data or adjust forecast parameters.")
            time.sleep(60)  # Refresh every 60 seconds
            st.experimental_rerun()

if __name__ == "__main__":
    main()
