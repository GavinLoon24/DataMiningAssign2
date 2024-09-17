import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
import matplotlib.pyplot as plt
import streamlit as st

# Define the stocks
stocks = ['6742.KL', '1155.KL', '5398.KL', '1818.KL', '0083.KL']

# Define the date range: start from 1 year ago until today
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Fetch and process stock data
def fetch_stock_data(stock):
    df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    # Use the last available date if today is not a trading day
    if df.index[-1].date() < end_date.date():
        print(f"Latest market data for {stock} is from {df.index[-1].date()}, using that as the current day.")
    
    df['Date'] = df.index.date
    df = df[['Date', 'Close', 'Volume']].copy()
    df.columns = ['Date', 'Close Price', 'Volume']
    return df

def process_stock_data(stock):
    df = fetch_stock_data(stock)
    today_close = df.iloc[-1]['Close Price']

    # Prepare the tables and chart data for the 3 indicators
    indicators = ['CCI', 'OBV', 'BollingerBands']
    predicted_close_all = []
    table_rows = []

    for indicator in indicators:
        # Add different technical indicators
        if indicator == 'CCI':
            # Calculate CCI with high and low prices
            df[indicator] = ta.trend.cci(df['Close Price'], df['Close Price'], df['Close Price'], window=20)
        elif indicator == 'OBV':
            df[indicator] = ta.volume.on_balance_volume(df['Close Price'], df['Volume'])
        elif indicator == 'BollingerBands':
            bb_bands = ta.volatility.BollingerBands(df['Close Price'])
            df['BB_upper'] = bb_bands.bollinger_hband()
            df['BB_middle'] = bb_bands.bollinger_mavg()
            df['BB_lower'] = bb_bands.bollinger_lband()
            df[indicator] = df['BB_middle']  # Use middle Bollinger band as the indicator

        # Create lag features (3 days) for the stock price and indicator
        for i in range(1, 4):
            df[f'close_lag_{i}day'] = df['Close Price'].shift(i)
            df[f'{indicator}_lag_{i}day'] = df[indicator].shift(i)
        
        # Create lead features (for next 3 days)
        for i in range(1, 4):
            df[f'close_next_{i}day'] = df['Close Price'].shift(-i)

        # Drop rows with missing values due to lags and leads
        df.dropna(inplace=True)

        # Prepare features and targets
        features = df[['close_lag_3day', 'close_lag_2day', 'close_lag_1day', indicator]].values
        targets = df[['close_next_1day', 'close_next_2day', 'close_next_3day']].values

        # Feature Selection using RFE
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
        rfe.fit(features[:-1], targets[:-1])

        # Classification Algorithms
        df['classification_target'] = (df['close_next_1day'] > df['Close Price']).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(features[:-1], df['classification_target'][:-1], test_size=0.3, random_state=42)

        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)

        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)

        # Regression Model
        model = LinearRegression()
        model.fit(features[:-1], targets[:-1])

        # Predict the next 3 days' close prices using the latest row
        predicted_close = model.predict(features[-1].reshape(1, -1))[0]
        predicted_close_all.append(predicted_close)

        # Create final table row for each indicator
        table_row = pd.DataFrame({
            'Stock': [stock],
            'Indicator': [indicator],
            'close_lag_3day': [df.iloc[-1]['close_lag_3day']],
            'close_lag_2day': [df.iloc[-1]['close_lag_2day']],
            'close_lag_1day': [df.iloc[-1]['close_lag_1day']],
            f'{indicator}_lag_3day': [df.iloc[-1][f'{indicator}_lag_3day']],
            f'{indicator}_lag_2day': [df.iloc[-1][f'{indicator}_lag_2day']],
            f'{indicator}_lag_1day': [df.iloc[-1][f'{indicator}_lag_1day']],
            'today_close': [today_close],  # Ensure today_close is the same across indicators
            'close_next_1day': [predicted_close[0]],
            'close_next_2day': [predicted_close[1]],
            'close_next_3day': [predicted_close[2]],
            'PREDICTION': ['Upward' if predicted_close[0] > today_close else 'Downward'],
            'DECISION': ['Buy' if predicted_close[0] > today_close else 'Sell']
        })

        table_rows.append(table_row)

    return table_rows, df

# Streamlit UI integration
st.title("Stock Prediction App")

# Stock selection dropdown
selected_stock = st.selectbox('Select a stock:', stocks)

# Button to fetch and process stock data
if st.button('Run Prediction'):
    # Fetch and process stock data
    table_rows, df = process_stock_data(selected_stock)

    # Display stock data
    st.subheader(f"{selected_stock} - Stock Data")
    st.dataframe(df.tail())  # Display the last few rows of the DataFrame

    # Display the tables for each indicator
    for row in table_rows:
        st.subheader(f"{selected_stock} - Indicator: {row['Indicator'][0]}")
        st.dataframe(row)

    # Plot predictions for the next 3 days
    plt.figure(figsize=(10, 6))
    days = ['Today', 'Day 1', 'Day 2', 'Day 3']
    for i in range(len(table_rows)):
        predicted_close = table_rows[i]['close_next_1day'].values[0], table_rows[i]['close_next_2day'].values[0], table_rows[i]['close_next_3day'].values[0]
        plt.plot(days, [today_close] + list(predicted_close), marker='o', label=f"{row['Indicator'][0]} Prediction")

    plt.title(f"{selected_stock} - 3-Day Close Price Predictions")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Optional: Fetch and display stock news
