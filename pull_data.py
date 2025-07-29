import yfinance as yf
import os
import argparse
from datetime import datetime

def download_stock_data(tickers, start_date="1990-01-01", end_date=None, data_folder="data"):
    """Downloads historical stock data for the given tickers and saves them as CSV files."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_folder)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    for ticker_symbol in tickers:
        try:
            print(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}...")
            # Note: auto_adjust=True adjusts ohlc data for stock splits and dividends
            ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if ticker_data.empty:
                print(f"No data found for {ticker_symbol}. Skipping.")
                continue

            file_path = os.path.join(data_path, f"{ticker_symbol.upper()}.csv")
            ticker_data.to_csv(file_path)
            print(f"Data for {ticker_symbol} saved to {file_path}")
        except Exception as e:
            print(f"Could not download data for {ticker_symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("tickers", metavar="TICKER", type=str, nargs="+",
                        help="One or more stock ticker symbols (e.g., AAPL MSFT GOOG).")
    parser.add_argument("--start", type=str, default="1990-01-01",
                        help="Start date for historical data in YYYY-MM-DD format (default: 1990-01-01).")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for historical data in YYYY-MM-DD format (default: today).")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory to save CSV files (default: data, relative to script location).")

    args = parser.parse_args()

    download_stock_data(args.tickers, args.start, args.end, args.data_dir)

    print("Data download process finished.")
