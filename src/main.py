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


class MonteCarlo:
    def __init__(self, T, N, ticker):
        """Docstring to follow.
        """
        self.T = T  # number of future days to simulate
        self.N = N  # number of paths to simulate
        self.ticker = ticker  # Stock ticker symbol of the stock to be simulated

        # Obtain the absolute path to the current script (main.py)
        self.script_dir = Path(__file__).resolve().parent

        # Load data
        self.load_data()

    def load_data(self):
        """Docstring to follow.
        """
        # Load data using a relative path to the data file
        data_path = self.script_dir / '..' / 'data' / f'{self.ticker}.csv'
        self.df = pd.read_csv(data_path)

    def simulate(self):
        """Docstring to follow.
        """
        # Use the adjusted close price to compute log returns
        self.adj_close = self.df['Adj Close'].values
        log_returns = np.log(self.adj_close[1:] / self.adj_close[:-1])

        # Compute mean and standard deviation of log returns
        self.mean = np.mean(log_returns)
        self.sigma = np.std(log_returns)

        # Set up any other simulation parameters and variables
        last_adj_close = self.adj_close[-1]  # last known adjusted close at the time of simulation
        self.price_paths = np.zeros((self.T+1, self.N))  # pre-allocate numpy array to store simulated paths
        self.price_paths[0] = last_adj_close

        # Compute simulated price paths
        for t in range(1, self.T+1):
            random_shocks = np.random.normal(self.mean, self.sigma, self.N)  # one random shock per path
            self.price_paths[t] = self.price_paths[t-1] * np.exp(random_shocks)

        # Compute summary statistics
        self.mean_prices = np.mean(self.price_paths, axis=1)  # has shape T+1 i.e. mean price per day
        self.pct_10 = np.percentile(self.price_paths, q=10, axis=1)
        self.pct_25 = np.percentile(self.price_paths, q=25, axis=1)
        self.pct_75 = np.percentile(self.price_paths, q=75, axis=1)
        self.pct_90 = np.percentile(self.price_paths, q=90, axis=1)

    def plot(self):
        """Docstring to follow.
        """
        # Compute necessary date variables for plotting
        dates = pd.to_datetime(self.df['Date'].values)
        days = np.arange(self.T+1)  # x-axis
        max_history = min(len(self.adj_close), (self.T+1)*3)  # avoid too much historical data
        dates_axis = dates[-max_history:]
        simulation_dates = pd.date_range(start=dates_axis[-1] + pd.Timedelta(days=1), periods=self.T+1, freq='D')
        combined_dates = np.concatenate((dates_axis, simulation_dates))  # combine historical and simulation horizon dates

        # Plot simulated price paths including an 80% confidence interval
        plots.plot_price_paths(days, self.pct_10, self.pct_25, self.mean_prices, self.pct_75, self.pct_90, base_dir=self.script_dir)

        # Plot both historical share price and simulated price paths
        plots.plot_price_paths_with_history(combined_dates, max_history, self.adj_close, self.pct_10, self.pct_25, self.mean_prices, self.pct_75, self.pct_90, base_dir=self.script_dir)
        
        # Plot histogram of final prices
        plots.plot_histogram(self.price_paths, self.N, base_dir=self.script_dir)

        # Add box plot of prices at given five evenly spaced time points
        plots.plot_box(self.price_paths, simulation_dates, self.T, base_dir=self.script_dir)


def main(args):
    monte_carlo = MonteCarlo(T=args.days, N=args.iterations)
    monte_carlo.simulate()
    monte_carlo.plot()


if __name__ == '__main__':
    # Instantiate the parser
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')

    # TODO: add any other command line arguments (i.e. share price history to consider)
    parser.add_argument('--days', '-d', type=int,
                         help='Number of future days to simulate', default=365)
    parser.add_argument('--iterations', '-i', type=int,
                         help='Number of simulation paths', default=1000)
    parser.add_argument('--ticker', '-t', type=str,
                         help='Stock ticker symbol of the stock to be simulated', default='ASML')
    args = parser.parse_args()
    print(vars(args))
    
    # Run main with the arguments passed
    main(args)
