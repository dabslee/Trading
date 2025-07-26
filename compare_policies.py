import argparse
import pandas as pd
import numpy as np
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

def _calculate_metrics(daily_history, risk_free_rate=0.02):
    """Calculates advanced performance metrics."""
    if not daily_history:
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }

    portfolio_values = pd.Series([day['portfolio_value'] for day in daily_history])
    daily_returns = portfolio_values.pct_change().dropna()

    # Sharpe Ratio
    # Assuming 252 trading days in a year
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0

    # Max Drawdown
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }

def generate_xlsx_report(results_dict, output_filename="policy_comparison_report.xlsx"):
    """
    Generates a professionally formatted XLSX file summarizing policy performance.
    """
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        workbook = writer.book

        # --- Define Formats ---
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DDEBF7', 'border': 1, 'align': 'center'})
        currency_format = workbook.add_format({'num_format': '$#,##0.00', 'border': 1})
        percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
        float_format = workbook.add_format({'num_format': '0.00', 'border': 1})
        default_format = workbook.add_format({'border': 1})

        # --- Summary Sheet ---
        summary_data = []
        for policy, results in results_dict.items():
            metrics = _calculate_metrics(results.get('daily_history', []))
            summary_data.append({
                'Policy': policy.replace('_', ' ').title(),
                'Final Portfolio Value': results['final_portfolio_value'],
                'Total Cash Injected': results['total_cash_injected'],
                'Net Gain/Loss': results['net_gain_loss'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Total Trades': results['total_trades'],
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', startrow=1, header=False, index=False)
        worksheet = writer.sheets['Summary']

        # Write headers with formatting
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Apply formatting to columns
        worksheet.set_column('A:A', 20, default_format) # Policy Name
        worksheet.set_column('B:D', 22, currency_format) # Currency columns
        worksheet.set_column('E:E', 15, float_format) # Sharpe Ratio
        worksheet.set_column('F:F', 15, percent_format) # Max Drawdown
        worksheet.set_column('G:G', 15, default_format) # Total Trades

        # --- Daily History Sheets ---
        for policy_name, results in results_dict.items():
            daily_history = results.get('daily_history', [])
            if not daily_history:
                continue

            sheet_name = f"{policy_name.replace('_', ' ').title()[:24]}_Log"

            # Create a clean DataFrame from the history
            history_df = pd.DataFrame(daily_history)
            history_df = history_df[['current_date', 'current_price', 'shares_held', 'current_cash', 'portfolio_value', 'action_taken']]
            history_df.rename(columns={
                'current_date': 'Date', 'current_price': 'Price', 'shares_held': 'Shares Held',
                'current_cash': 'Cash', 'portfolio_value': 'Portfolio Value', 'action_taken': 'Action (Shares)'
            }, inplace=True)

            history_df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
            worksheet = writer.sheets[sheet_name]

            # Write formatted headers
            for col_num, value in enumerate(history_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Set column formats and widths
            worksheet.set_column('A:A', 12, default_format) # Date
            worksheet.set_column('B:B', 12, currency_format) # Price
            worksheet.set_column('C:C', 12, float_format) # Shares Held
            worksheet.set_column('D:E', 18, currency_format) # Cash, Portfolio Value
            worksheet.set_column('F:F', 15, float_format) # Action

    print(f"XLSX report saved to '{output_filename}'")


def compare_policies(ticker, start_date, end_date, env_start, env_days):
    """
    Runs simulations for a list of policies and collects their results.
    """
    policies_to_compare = ['no_action', 'buy_and_hold', 'sma_crossover']
    all_results = {}

    print(f"Comparing policies for ticker: {ticker} over {env_days} days...")

    for policy_name in policies_to_compare:
        print(f"  Running simulation for policy: {policy_name}...")

        results = run_example_session(
            ticker=ticker,
            start_date_data=start_date,
            end_date_data=end_date,
            env_start_date=env_start,
            env_horizon_days=env_days,
            policy_name=policy_name,
            render_mode='ansi',
            verbose=False
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

        # This table is now just for console output, the XLSX is the main report
        console_table_data = {
            'Policy': list(comparison_results.keys()),
            'Final Value': [f"${v['final_portfolio_value']:,.2f}" for v in comparison_results.values()],
            'Net Gain': [f"${v['net_gain_loss']:,.2f}" for v in comparison_results.values()]
        }
        console_df = pd.DataFrame(console_table_data).set_index('Policy')
        print("\n--- Quick Summary ---")
        print(console_df)

        generate_performance_chart(comparison_results)
        generate_xlsx_report(comparison_results)
