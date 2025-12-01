import yfinance as yf
import pandas as pd
import datetime as dt
import os

# Function to fetch historical stock data for a given period
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    all_data = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if not data.empty:
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            all_data.append(data)
        else:
            print(f"No data found for {ticker}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])

# Function to save stock data to CSV inside Data/Raw/
def save_to_csv(df, filename='stock_data.csv'):
    """Save the DataFrame to Data/Raw/."""
    os.makedirs("Data/Raw", exist_ok=True)
    filepath = os.path.join("Data/Raw", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved stock data to {filepath}")

# Example usage
if __name__ == "__main__":
    tickers = [
        'RELIANCE.NS', 'TATASTEEL.NS', 'INFY.NS', 'HDFCBANK.NS', '^NSEI',
        'ICICIBANK.NS', 'WIPRO.NS', 'ADANIENT.NS', 'MARUTI.NS', 'LT.NS'
    ]

    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    start_date = (dt.datetime.now() - dt.timedelta(days=16*365)).strftime('%Y-%m-%d')

    stock_df = fetch_stock_data(tickers, start_date, end_date)
    save_to_csv(stock_df, filename="stock_data.csv")

    print(f"Saved stock data for {len(tickers)} stocks.")
