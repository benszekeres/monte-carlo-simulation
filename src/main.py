"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Append the project root directory to sys.path to import from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import plots


def main(args):
    # Obtain the absolute path to the current script (main.py)
    script_dir = Path(__file__).resolve().parent

    # Load data using a relative path to the data file
    data_path = script_dir / '..' / 'data' / 'ASML.csv'
    df = pd.read_csv(data_path)

    # Obtain date column for plotting
    dates = pd.to_datetime(df['Date'].values)

    # Use the adjusted close price to compute log returns
    adj_close = df['Adj Close'].values
    log_returns = np.log(adj_close[1:] / adj_close[:-1])

    # Compute mean and standard deviation
    mean = np.mean(log_returns)
    sigma = np.std(log_returns)
    
    # Set up any other simulation parameters and variables
    T = args.days  # number of future days to simulate
    N = args.iterations  # number of paths to simulate
    last_adj_close = adj_close[-1]  # last known adjusted close at the time of simulation
    price_paths = np.zeros((T+1, N))  # pre-allocate numpy array to store simulated paths
    price_paths[0] = last_adj_close

    # Perform simulation
    for t in range(1, T+1):
        random_shocks = np.random.normal(mean, sigma, N)  # one random shock per path
        price_paths[t] = price_paths[t-1] * np.exp(random_shocks)

    # Compute summary statistics
    mean_prices = np.mean(price_paths, axis=1)  # has shape T+1 i.e. mean price per day
    pct_10 = np.percentile(price_paths, q=10, axis=1)
    pct_25 = np.percentile(price_paths, q=25, axis=1)
    pct_75 = np.percentile(price_paths, q=75, axis=1)
    pct_90 = np.percentile(price_paths, q=90, axis=1)

    # Compute necessary date variables for plotting
    days = np.arange(T+1)  # x-axis
    max_history = min(len(adj_close), (T+1)*3)  # avoid too much historical data
    dates_axis = dates[-max_history:]
    simulation_dates = pd.date_range(start=dates_axis[-1] + pd.Timedelta(days=1), periods=T+1, freq='D')
    combined_dates = np.concatenate((dates_axis, simulation_dates))  # combine historical and simulation horizon dates

    # Plot simulated price paths including an 80% confidence interval
    plots.plot_price_paths(days, pct_10, pct_25, mean_prices, pct_75, pct_90, base_dir=script_dir)

    # Plot both historical share price and simulated price paths
    plots.plot_price_paths_with_history(combined_dates, max_history, adj_close, pct_10, pct_25, mean_prices, pct_75, pct_90, base_dir=script_dir)
    
    # Plot histogram of final prices
    plots.plot_histogram(price_paths, N, base_dir=script_dir)

    # Add box plot of prices at given five evenly spaced time points
    plots.plot_box(price_paths, simulation_dates, T, base_dir=script_dir)


if __name__ == '__main__':
    # Instantiate the parser
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')

    # TODO: add any other command line arguments (i.e. share price history to consider)
    parser.add_argument('--days', '-d', type=int,
                         help='Number of future days to simulate', default=365)
    parser.add_argument('--iterations', '-i', type=int,
                         help='Number of simulation paths', default=1000)
    args = parser.parse_args()
    print(vars(args))
    
    # Run main with the arguments passed
    main(args)
