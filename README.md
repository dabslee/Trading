# Stock Trading Environment

This project provides a simple stock trading environment using OpenAI's Gymnasium.

## Project Structure

-   `data/`: Stores downloaded CSV files of stock historical data.
-   `pull_data.py`: Script to download historical stock data using Yahoo Finance.
-   `trading_env.py`: Implements the `TradingEnv` Gymnasium environment for stock trading.
-   `requirements.txt`: Lists Python dependencies.
-   `example.py`: Example script to demonstrate usage of the components.
-   `compare_policies.py`: A script to compare the performance of different trading policies.
-   `strategies/`: Contains different trading policy implementations.

## Setup and Installation

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Scripts

### 1. Download Stock Data

Use `pull_data.py` to download data for specific stock tickers.

**Syntax:**
```bash
python pull_data.py <TICKER1> [TICKER2 ...] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
```

**Example:**
```bash
python pull_data.py AAPL MSFT --start 2020-01-01 --end 2023-12-31
```

### 2. Running the Example

The `example.py` script demonstrates a typical workflow:

**Example:**
```bash
python example.py --ticker GOOG --policy sma_crossover
```

### 3. Comparing Policies

The `compare_policies.py` script runs simulations for multiple policies and generates a comparison report.

**Example:**
```bash
python compare_policies.py --ticker GOOG
```