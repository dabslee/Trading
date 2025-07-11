import yfinance as yf
import os
import argparse
from datetime import datetime

def download_stock_data(tickers, start_date="1990-01-01", end_date=None, data_folder="data"):
    """
    Downloads historical stock data for the given tickers and saves them as CSV files.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        start_date (str): Start date for historical data in "YYYY-MM-DD" format.
        end_date (str): End date for historical data in "YYYY-MM-DD" format.
                        Defaults to today if None.
        data_folder (str): The name of the folder to save CSV files.
                           It will be created if it doesn't exist, relative
                           to the script's parent directory.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Ensure the data folder exists (relative to the script's parent directory)
    # Assuming pull_data.py is in stock_trading_env/
    # and data folder should be stock_trading_env/data/
    script_dir = os.path.dirname(os.path.abspath(__file__)) # stock_trading_env/
    # Construct path to data folder, assuming it's a sibling to the script or defined relative to project root
    # For this project, data_folder is "data" which means "stock_trading_env/data"

    # Correct path for data_folder assuming it's inside the project directory (stock_trading_env)
    project_root_dir = os.path.dirname(script_dir) # This would be one level above if script is in a subdir
                                                # If script is at stock_trading_env/pull_data.py, script_dir is stock_trading_env/
                                                # then project_root_dir is the parent of stock_trading_env/. This is not what we want.

    # Let's assume data_folder is relative to where the script is run from, or an absolute path.
    # For this project structure, data_folder is "stock_trading_env/data" when running from repo root.
    # The script itself is at "stock_trading_env/pull_data.py".
    # So, the data_folder argument should be "data" if we want it inside "stock_trading_env".

    # Let's define data_path relative to the script's location.
    # If data_folder is "data", it means stock_trading_env/data
    data_path = os.path.join(script_dir, data_folder)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    for ticker_symbol in tickers:
        try:
            print(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}...")
            ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
            # print(f"For {ticker_symbol}, DataFrame columns: {ticker_data.columns}") # DEBUG
            # print(f"For {ticker_symbol}, DataFrame index name: {ticker_data.index.name}") # DEBUG
            if ticker_data.empty:
                print(f"No data found for {ticker_symbol}. Skipping.")
                continue

            # Ensure 'Date' is a column and not the index for consistency, or keep as index
            # yf.download typically returns with Date as index.
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

    # Adjust data_dir to be relative to the script's location as designed in the function
    # The download_stock_data function expects data_folder to be the name of the folder
    # and constructs the path relative to its own location.
    download_stock_data(args.tickers, args.start, args.end, args.data_dir)

    # Example: python pull_data.py AAPL --start 2020-01-01 --data_dir data
    # This will save to stock_trading_env/data/AAPL.csv
    # if pull_data.py is in stock_trading_env/
    # and you run it from stock_trading_env/
    # If you run from parent of stock_trading_env: python stock_trading_env/pull_data.py AAPL ...
    # then script_dir is stock_trading_env/, data_dir is "data", so path is stock_trading_env/data. This is correct.

    # Let's verify the structure
    # Project root
    # |- stock_trading_env
    #    |- pull_data.py
    #    |- data/  <-- This is where CSVs should go
    #
    # If I run `python stock_trading_env/pull_data.py AAPL` from project root:
    #   os.path.abspath(__file__) is /path/to/project_root/stock_trading_env/pull_data.py
    #   script_dir is /path/to/project_root/stock_trading_env
    #   args.data_dir is "data" by default
    #   data_path = os.path.join(script_dir, args.data_dir)
    #             = /path/to/project_root/stock_trading_env/data
    # This seems correct.

    print("Data download process finished.")
