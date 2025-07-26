import argparse
import pandas as pd
import matplotlib.pyplot as plt
from main import run_example_session

def generate_performance_chart(results_dict, output_filename="policy_comparison_chart.png"):
    """
    Generates and saves a chart comparing the portfolio value over time for each policy.
    """
    plt.figure(figsize=(14, 8))

    # Plot markers with a single label for all buys/sells
    plt.scatter([], [], color='green', marker='^', s=100, label='Buy Action')
    plt.scatter([], [], color='red', marker='v', s=100, label='Sell Action')

    for policy_name, results in results_dict.items():
        history = results['portfolio_value_history']
        line, = plt.plot(history, label=f"{policy_name.replace('_', ' ').title()} Policy")

        trade_history = results.get('trade_history', [])
        buys_x, buys_y, sells_x, sells_y = [], [], [], []

        for trade in trade_history:
            day_idx = trade['date_idx']
            if day_idx < len(history):
                if trade['type'].startswith('BUY'):
                    buys_x.append(day_idx)
                    buys_y.append(history[day_idx])
                elif trade['type'].startswith('SELL'):
                    sells_x.append(day_idx)
                    sells_y.append(history[day_idx])

        plt.scatter(buys_x, buys_y, color=line.get_color(), marker='^', s=50, zorder=5, alpha=0.7)
        plt.scatter(sells_x, sells_y, color=line.get_color(), marker='v', s=50, zorder=5, alpha=0.7)

    plt.title("Portfolio Value Over Time by Policy")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_filename)
    print(f"\nPerformance chart saved to '{output_filename}'")
    plt.close() # Close the figure to free memory

def generate_xlsx_report(results_dict, table_df, output_filename="policy_comparison_report.xlsx"):
    """
    Generates an XLSX file summarizing policy performance, including detailed trade logs.
    """
    best_policy_name = max(results_dict, key=lambda p: results_dict[p]['net_gain_loss'])
    best_policy_results = results_dict[best_policy_name]

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # --- Summary Sheet ---
        summary_df = table_df.copy()
        summary_df.loc['--- ANALYSIS ---'] = ''
        summary_df.loc['Best Performing Policy'] = best_policy_name.replace('_', ' ').title()
        summary_df.loc['Best Policy Net Gain'] = f"${best_policy_results['net_gain_loss']:,.2f}"

        summary_df.to_excel(writer, sheet_name='Summary', index=True)

        # --- Detailed Trade History Sheets ---
        for policy_name, results in results_dict.items():
            trade_history = results.get('trade_history', [])
            if trade_history:
                trade_df = pd.DataFrame(trade_history)
                # Format for readability
                trade_df['price'] = trade_df['price'].apply(lambda x: f"${x:,.2f}")
                trade_df['shares'] = trade_df['shares'].apply(lambda x: f"{x:,.2f}")
                trade_df.rename(columns={'date_idx': 'Day', 'type': 'Action', 'price': 'Price', 'shares': 'Shares'}, inplace=True)
                trade_df.to_excel(writer, sheet_name=f"{policy_name[:25]}_Trades", index=False)

    print(f"XLSX report saved to '{output_filename}'")

def generate_comparison_table(results_dict):
    """
    Creates a pandas DataFrame from the results dictionary for display.
    """
    # Reformat the dictionary for DataFrame constructor
    data_for_df = {
        'Policy': list(results_dict.keys()),
        'Final Portfolio Value': [f"${v['final_portfolio_value']:,.2f}" for v in results_dict.values()],
        'Net Gain/Loss': [f"${v['net_gain_loss']:,.2f}" for v in results_dict.values()],
        'Total Trades': [v['total_trades'] for v in results_dict.values()]
    }

    df = pd.DataFrame(data_for_df)
    df.set_index('Policy', inplace=True)
    return df

def compare_policies(ticker, start_date, end_date, env_start, env_days):
    """
    Runs simulations for a list of policies and collects their results.
    """
    policies_to_compare = ['no_action', 'buy_and_hold', 'sma_crossover']
    all_results = {}

    print(f"Comparing policies for ticker: {ticker} over {env_days} days...")

    for policy_name in policies_to_compare:
        print(f"  Running simulation for policy: {policy_name}...")

        # Run the simulation from main.py, but without the verbose output
        results = run_example_session(
            ticker=ticker,
            start_date_data=start_date,
            end_date_data=end_date,
            env_start_date=env_start,
            env_horizon_days=env_days,
            policy_name=policy_name,
            render_mode='ansi', # Use ansi for no GUI pop-up
            verbose=False # Keep the output clean
        )

        if results:
            all_results[policy_name] = results
            print(f"  ...'{policy_name}' completed.")
        else:
            print(f"  ...'{policy_name}' failed to run.")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trading policies.")
    parser.add_argument("--ticker", default="GOOG", help="Stock ticker symbol.")
    parser.add_argument("--data_start", default="2019-01-01", help="Start date for data download.")
    parser.add_argument("--data_end", default="2023-12-31", help="End date for data download.")
    parser.add_argument("--env_start", default="2021-01-01", help="Start date for simulation.")
    parser.add_argument("--env_days", type=int, default=252, help="Number of trading days for simulation.")

    args = parser.parse_args()

    # Run the comparison
    comparison_results = compare_policies(
        ticker=args.ticker,
        start_date=args.data_start,
        end_date=args.data_end,
        env_start=args.env_start,
        env_days=args.env_days
    )

    if not comparison_results:
        print("\nNo policies were successfully simulated. Exiting.")
    else:
        print("\n--- All simulations complete ---")

        # --- 1. Generate and Display Comparison Table ---
        print("\n--- Performance Comparison Table ---")
        comparison_table = generate_comparison_table(comparison_results)
        print(comparison_table)

        # --- 2. Generate and Save Performance Chart ---
        generate_performance_chart(comparison_results)

        # --- 3. Generate and Save XLSX Report ---
        generate_xlsx_report(comparison_results, comparison_table)
