import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import ta
import matplotlib.pyplot as plt
import streamlit as st

# Define the stocks
stocks = ['6742.KL', '1155.KL', '5398.KL', '1818.KL', '0083.KL']

# Define the date range: start from 1 year ago until today
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Fetch stock data
def fetch_stock_data(stock):
    df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    df['Date'] = df.index.date
    df = df[['Date', 'Close', 'Volume']].copy()
    df.columns = ['Date', 'Close Price', 'Volume']
    return df

# Process stock data and generate predictions
def process_stock_data(stock):
    df = fetch_stock_data(stock)
    today_close = df.iloc[-1]['Close Price']

    # Define indicators globally
    indicators = ['CCI', 'OBV', 'BollingerBands']
    predicted_close_all = []

    for indicator in indicators:
        if indicator == 'CCI':
            df[indicator] = ta.trend.cci(df['Close Price'], df['Volume'], df['Close Price'], window=20)
        elif indicator == 'OBV':
            df[indicator] = ta.volume.on_balance_volume(df['Close Price'], df['Volume'])
        elif indicator == 'BollingerBands':
            bb_bands = ta.volatility.BollingerBands(df['Close Price'])
            df['BB_upper'] = bb_bands.bollinger_hband()
            df['BB_middle'] = bb_bands.bollinger_mavg()
            df['BB_lower'] = bb_bands.bollinger_lband()
            df[indicator] = df['BB_middle']

        # Create lag features
        for i in range(1, 4):
            df[f'close_lag_{i}day'] = df['Close Price'].shift(i)
            df[f'{indicator}_lag_{i}day'] = df[indicator].shift(i)
        
        # Create lead features
        for i in range(1, 4):
            df[f'close_next_{i}day'] = df['Close Price'].shift(-i)

        # Drop rows with missing values
        df.dropna(inplace=True)

        if len(df) > 0:
            features = df[['close_lag_3day', 'close_lag_2day', 'close_lag_1day', indicator]].values
            targets = df[['close_next_1day', 'close_next_2day', 'close_next_3day']].values

            # Feature Selection using RFE
            rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
            rfe.fit(features[:-1], targets[:-1])

            # Linear Regression Prediction
            model = LinearRegression()
            model.fit(features[:-1], targets[:-1])
            predicted_close = model.predict(features[-1].reshape(1, -1))[0]
            predicted_close_all.append(predicted_close)
        else:
            st.warning(f"Not enough data for {indicator} to make predictions.")
    
    # Check if we have predictions for all indicators
    if len(predicted_close_all) == len(indicators):
        plt.figure(figsize=(10, 6))
        days = ['Today', 'Day 1', 'Day 2', 'Day 3']
        for i, indicator in enumerate(indicators):
            plt.plot(days, [today_close] + list(predicted_close_all[i]), marker='o', label=f'{indicator} Prediction')
        plt.title(f"{stock} - 3-Day Close Price Predictions (3 Indicators)")
        plt.xlabel("Days")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        return df, predicted_close_all, plt
    else:
        st.error("Prediction failed for one or more indicators.")
        return df, None, None

# Streamlit UI
st.title("Stock Prediction App")

# Stock selection dropdown
selected_stock = st.selectbox('Select a stock:', stocks)

# Button to fetch and process stock data
if st.button('Run Prediction'):
    # Fetch and process stock data
    df, predicted_close_all, plt_fig = process_stock_data(selected_stock)
    
    # Display stock data
    if df is not None:
        st.subheader(f"{selected_stock} - Stock Data")
        st.dataframe(df.tail())  # Display the last few rows of the DataFrame

    # Display prediction chart
    if plt_fig is not None:
        st.subheader(f"{selected_stock} - Predicted Close Price (Next 3 Days)")
        st.pyplot(plt_fig)

    # Trade Profit and Stop-Loss Table
    if predicted_close_all is not None:
        st.subheader(f"{selected_stock} - Trade Analysis")
        today_close = df.iloc[-1]['Close Price']
        indicators = ['CCI', 'OBV', 'BollingerBands']  # Define indicators again to avoid error
        for indicator, predicted_close in zip(indicators, predicted_close_all):
            day_trade_profit = ((predicted_close[0] - today_close) / today_close) * 100
            week_trade_profit = ((predicted_close[2] - today_close) / today_close) * 100
            stop_loss = today_close * 0.95
            sell_recommendation = predicted_close[0] * 1.05

            trade_row = pd.DataFrame({
                'Indicator': [indicator],
                'Day Trade Profit (%)': [day_trade_profit],
                'Week Trade Profit (%)': [week_trade_profit],
                'Stop-Loss': [stop_loss],
                'Sell Recommendation': [sell_recommendation]
            })
            st.dataframe(trade_row)
