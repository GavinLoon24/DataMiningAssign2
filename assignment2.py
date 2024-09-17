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
import requests

# Define the stocks
stocks = ['6742.KL', '1155.KL', '5398.KL', '1818.KL', '0083.KL']

# Define the date range: start from 1 year ago until today
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Fetch and process stock data
def fetch_stock_data(stock):
    df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    # Use the last available date if today is not a trading day
    last_available_date = df.index[-1].date()
    if last_available_date < end_date.date():
        print(f"Latest market data for {stock} is from {last_available_date}, using that as the current day.")
    
    df['Date'] = df.index.date
    df = df[['Date', 'Close', 'Volume']].copy()
    df.columns = ['Date', 'Close Price', 'Volume']
    return df, last_available_date  # Return the last available date as well

# Process stock data and generate predictions
def process_stock_data(stock):
    df, last_available_date = fetch_stock_data(stock)
    today_close = df.iloc[-1]['Close Price']

    # Prepare indicators and model predictions
    indicators = ['CCI', 'OBV', 'BollingerBands']
    predicted_close_all = []
    table_rows = []
    trade_analysis_rows = []

    for indicator in indicators:
        # Add different technical indicators
        if indicator == 'CCI':
            df[indicator] = ta.trend.cci(df['Close Price'], df['Close Price'], df['Close Price'], window=20)
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

        # Create buy/sell decisions based on price movement
        decisions = ['Buy' if pred > today_close else 'Sell' for pred in predicted_close]
        predictions = ['Upward' if pred > today_close else 'Downward' for pred in predicted_close]

        # Create the final formatted table row for each indicator
        table_row = pd.DataFrame({
            'Stock': stock,
            'Indicator': indicator,
            'close_lag_3day': [df.iloc[-1]['close_lag_3day']],
            'close_lag_2day': [df.iloc[-1]['close_lag_2day']],
            'close_lag_1day': [df.iloc[-1]['close_lag_1day']],
            f'{indicator}_lag_3day': [df.iloc[-1][f'{indicator}_lag_3day']],
            f'{indicator}_lag_2day': [df.iloc[-1][f'{indicator}_lag_2day']],
            f'{indicator}_lag_1day': [df.iloc[-1][f'{indicator}_lag_1day']],
            'today_close': [today_close],
            'close_next_1day': [predicted_close[0]],
            'close_next_2day': [predicted_close[1]],
            'close_next_3day': [predicted_close[2]],
            'PREDICTION': [predictions[0]],
            'DECISION': [decisions[0]]
        })

        table_rows.append(table_row)

        # Calculate Trade Profit and Stop-Loss
        day_trade_profit = ((predicted_close[0] - today_close) / today_close) * 100
        week_trade_profit = ((predicted_close[2] - today_close) / today_close) * 100
        stop_loss = today_close * 0.95
        sell_recommendation = predicted_close[0] * 1.05

        trade_row = pd.DataFrame({
            'Stock': stock,
            'Indicator': indicator,
            'Day Trade Profit (%)': [day_trade_profit],
            'Week Trade Profit (%)': [week_trade_profit],
            'Trade Suggestion': ['Day Trade' if day_trade_profit > week_trade_profit else 'Week Trade'],
            'Stop-Loss': [stop_loss],
            'Sell Recommendation': [sell_recommendation]
        })

        trade_analysis_rows.append(trade_row)

    return table_rows, trade_analysis_rows, predicted_close_all, today_close, last_available_date

# Fetch Stock News using NewsAPI
def fetch_stock_news():
    api_key = "473567c8d990486db8d201a392933b7b"  # Your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q=malaysia stock&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    
    if news_data['status'] == 'ok':
        articles = news_data['articles']
        news_list = []
        for article in articles[:5]:  # Display top 5 articles
            news_list.append({
                "Title": article['title'],
                "Source": article['source']['name'],
                "Published At": article['publishedAt'],
                "URL": article['url']
            })
        return news_list
    else:
        return []

# Streamlit UI
st.title("Stock Prediction App")

# Stock selection dropdown
selected_stock = st.selectbox('Select a stock:', stocks)

# Button to fetch and process stock data
if st.button('Run Prediction'):
    # Fetch and process stock data
    table_rows, trade_analysis_rows, predicted_close_all, today_close, last_available_date = process_stock_data(selected_stock)

    # Display current date being used
    st.subheader(f"Using data as of: {last_available_date}")

    # Display stock data
    for i, indicator in enumerate(['CCI', 'OBV', 'BollingerBands']):
        st.subheader(f"{selected_stock} - Indicator: {indicator}")
        st.dataframe(table_rows[i])

        st.subheader(f"Trade Analysis for {indicator}")
        st.dataframe(trade_analysis_rows[i])

    # Plot predictions for the next 3 days
    plt.figure(figsize=(10, 6))
    days = ['Today', 'Day 1', 'Day 2', 'Day 3']
    for i, indicator in enumerate(['CCI', 'OBV', 'BollingerBands']):
        predicted_close = predicted_close_all[i]
        plt.plot(days, [today_close] + list(predicted_close), marker='o', label=f'{indicator} Prediction')

    plt.title(f"{selected_stock} - 3-Day Close Price Predictions (3 Indicators)")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Fetch and display stock news
    st.subheader("Latest Stock News")
    news_list = fetch_stock_news()
    if news_list:
        for news in news_list:
            st.write(f"**Title**: [{news['Title']}]({news['URL']})")
            st.write(f"Source: {news['Source']}")
            st.write(f"Published At: {news['Published At']}")
            st.write("\n")
    else:
        st.write("No news found.")




