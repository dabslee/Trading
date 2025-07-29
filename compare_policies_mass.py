import argparse
import pandas as pd
import numpy as np
from compare_policies import compare_policies, generate_xlsx_report
import os
from collections import defaultdict
import time

def run_mass_comparison(ticker, start_date, end_date, env_days, num_trials):
    """Runs the policy comparison multiple times with random start dates and aggregates results."""
    all_runs_results = defaultdict(list)

    print(f"Starting mass comparison for {num_trials} trials...")
    start_time = time.time()

    # Calculate the valid range for random start dates
    data_start_dt = pd.to_datetime(start_date)
    data_end_dt = pd.to_datetime(end_date)

    # Ensure the environment duration doesn't exceed the data range
    if (data_end_dt - data_start_dt).days < env_days:
        raise ValueError("The total data range is shorter than the simulation period ('env_days').")

    # The latest possible start day for a simulation
    latest_start_day = data_end_dt - pd.Timedelta(days=env_days)

    # The total number of possible start dates
    total_possible_start_days = (latest_start_day - data_start_dt).days

    if total_possible_start_days <= 0:
        raise ValueError("Cannot determine a valid start date range. Check data and simulation day parameters.")

    for i in range(num_trials):
        print(f"\n--- Starting Trial {i+1}/{num_trials} ---")

        # Generate a random start date for the simulation
        random_offset = np.random.randint(0, total_possible_start_days)
        sim_start_date = data_start_dt + pd.Timedelta(days=random_offset)
        sim_start_date_str = sim_start_date.strftime('%Y-%m-%d')

        print(f"  Randomized Simulation Start Date: {sim_start_date_str}")

        trial_results = compare_policies(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            env_start=sim_start_date_str,
            env_days=env_days
        )

        if not trial_results:
            print(f"Trial {i+1} failed to produce results. Skipping.")
            continue

        for policy_name, results in trial_results.items():
            all_runs_results[policy_name].append(results)

        print(f"--- Trial {i+1} complete ---")

    end_time = time.time()
    print(f"\nMass comparison finished in {end_time - start_time:.2f} seconds.")

    return all_runs_results

def generate_mass_summary_report(mass_results, output_filename):
    """Generates an XLSX report summarizing the performance across all trials."""
    summary_data = []

    for policy_name, runs in mass_results.items():
        final_values = [run['final_portfolio_value'] for run in runs]
        net_gains = [run['net_gain_loss'] for run in runs]

        summary_data.append({
            'Policy': policy_name.replace('_', ' ').title(),
            'Trials': len(runs),
            'Avg Final Value': np.mean(final_values),
            'Std Dev Final Value': np.std(final_values),
            'Min Final Value': np.min(final_values),
            'Max Final Value': np.max(final_values),
            'Avg Net Gain': np.mean(net_gains),
            'Std Dev Net Gain': np.std(net_gains),
        })

    summary_df = pd.DataFrame(summary_data)

    # Generate a simple console output
    print("\n--- Mass Comparison Summary ---")
    print(summary_df.to_string(index=False))

    # Generate XLSX report
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Mass Summary', index=False)
        worksheet = writer.sheets['Mass Summary']

        # Auto-adjust column widths
        for i, col in enumerate(summary_df.columns):
            worksheet.set_column(i, i, len(col) + 2)

    print(f"\nMass summary report saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mass comparison of trading policies.")
    parser.add_argument("--ticker", default="GOOG", help="Stock ticker symbol.")
    parser.add_argument("--data_start", default="2019-01-01", help="Start date for data download.")
    parser.add_argument("--data_end", default="2023-12-31", help="End date for data download.")
    parser.add_argument("--env_days", type=int, default=252, help="Number of trading days for simulation.")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of comparison trials to run.")

    args = parser.parse_args()

    try:
        mass_results = run_mass_comparison(
            ticker=args.ticker,
            start_date=args.data_start,
            end_date=args.data_end,
            env_days=args.env_days,
            num_trials=args.num_trials
        )
    except ValueError as e:
        print(f"Error: {e}")
        mass_results = None

    if not mass_results:
        print("\nNo results from any trial. Exiting.")
    else:
        # Create reports directory if it doesn't exist
        if not os.path.exists("reports"):
            os.makedirs("reports")

        generate_mass_summary_report(mass_results, "reports/mass_comparison_summary.xlsx")
