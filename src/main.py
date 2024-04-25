"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Append the project root directory to sys.path to import from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import plots


class MonteCarlo:
    def __init__(self, T: int, N: int, ticker: str) -> None:
        """Docstring to follow.
        """
        self.T = T  # number of future trading days to simulate
        self.N = N  # number of paths to simulate
        self.ticker = ticker  # Stock ticker symbol of the stock to be simulated

        # Obtain the absolute path to the current script (main.py)
        self.script_dir = Path(__file__).resolve().parent

        # Load data
        self.load_data()

    def load_data(self) -> None:
        """Docstring to follow.
        """
        # Load data using a relative path to the data file
        data_path = self.script_dir / '..' / 'data' / f'{self.ticker}.csv'
        self.df = pd.read_csv(data_path)

    def simulate(self) -> None:
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

        # Compute simulated returns
        self.simulated_returns = self.price_paths[-1] / self.price_paths[0] - 1

        # Compute VaR
        self.compute_var_and_cvar()

        # Compute summary statistics
        self.compute_summary_statistics()

    def compute_var_and_cvar(self) -> None:
        """Docstring to follow.
        """
        # Compute VaR and CVar at 95% and 99% confidence thresholds
        self.var = {}
        self.cvar = {}
        self.confidence_thresh = [0.95, 0.99]
        sorted_returns = np.sort(self.simulated_returns)  # ascending, uses Timsort O(nlogn)

        for thresh in self.confidence_thresh:
            # Compute VaR
            var_idx = int((1 - thresh) * len(sorted_returns))
            self.var[thresh] = sorted_returns[var_idx]

            # Compute CVaR
            losses = sorted_returns[:var_idx]
            self.cvar[thresh] = np.mean(losses)

    def compute_summary_statistics(self) -> None:
        """Docstring to follow.
        """
        # Compute basic summary statistics
        self.mean_prices = np.mean(self.price_paths, axis=1)  # has shape T+1 i.e. mean price per day
        self.min_price = np.min(self.price_paths[-1])
        self.max_price = np.max(self.price_paths[-1])
        self.pct_10 = np.percentile(self.price_paths, q=10, axis=1)
        self.pct_25 = np.percentile(self.price_paths, q=25, axis=1)
        self.pct_75 = np.percentile(self.price_paths, q=75, axis=1)
        self.pct_90 = np.percentile(self.price_paths, q=90, axis=1)

        # Create summary statistics table
        data = []

        # Simulation parameters
        data.append({'Metric': 'Number of Simulated Paths', 'Value': self.N})
        data.append({'Metric': 'Simulation Time Horizon', 'Value': f'{self.T} trading days'})

        # Mean, min, max
        data.append({'Metric': 'Mean Final Price', 'Value': f'{self.mean_prices[-1]:.0f}'})
        data.append({'Metric': 'Min Final Price', 'Value': f'{self.min_price:.0f}'})
        data.append({'Metric': 'Max Final Price', 'Value': f'{self.max_price:.0f}'})

        # VaR and CVaR
        for thresh in self.confidence_thresh:
            data.append({'Metric': f'VaR {int(thresh*100)}%', 'Value': f'{self.var[thresh]:.1%}'})
            data.append({'Metric': f'CVaR {int(thresh*100)}%', 'Value': f'{self.cvar[thresh]:.1%}'})
        
        # Concatenate into a class member DataFrame
        self.summary_stats = pd.concat([pd.DataFrame(data)], ignore_index=True)

    def plot(self) -> None:
        """Docstring to follow.
        """
        # Compute necessary date variables for plotting
        dates = pd.to_datetime(self.df['Date'].values)
        days = np.arange(self.T+1)  # x-axis (+1 to consider the latest existing data point)
        max_history = min(len(self.adj_close), (self.T+1)*3)  # avoid displaying too much historical data
        dates_axis = dates[-max_history:]
        simulation_dates = pd.date_range(start=dates_axis[-1] + pd.Timedelta(days=1), periods=self.T+1, freq='B')
        combined_dates = np.concatenate((dates_axis, simulation_dates))  # combine historical and simulation horizon dates

        # Plot simulated price paths including an 80% confidence interval
        plots.plot_price_paths(days, self.pct_10, self.pct_25, self.mean_prices, self.pct_75, self.pct_90, base_dir=self.script_dir, ticker=self.ticker)
        
        # Plot both historical share price and simulated price paths
        plots.plot_price_paths_with_history(combined_dates, max_history, self.adj_close, self.pct_10, self.pct_25, self.mean_prices, self.pct_75, self.pct_90, base_dir=self.script_dir, ticker=self.ticker)
        
        # Plot histogram of simulated returns
        plots.plot_histogram(self.simulated_returns, self.N, base_dir=self.script_dir, ticker=self.ticker)

        # Add box plot of prices at given five evenly spaced time points
        plots.plot_box(self.price_paths, simulation_dates, self.T, base_dir=self.script_dir, ticker=self.ticker)

        # Save table of summary statistics as an image
        plots.plot_summary_statistics(self.summary_stats, self.ticker)


def main(args: argparse.Namespace) -> None:
    monte_carlo = MonteCarlo(T=args.days, N=args.iterations, ticker=args.ticker)
    monte_carlo.simulate()
    monte_carlo.plot()


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')

    parser.add_argument('--days', '-d', type=int,
                         help='Number of future trading days to simulate. Defaults to one 252 reflecting one year.', default=252)
    parser.add_argument('--iterations', '-i', type=int,
                         help='Number of simulation paths', default=1000)
    parser.add_argument('--ticker', '-t', type=str,
                         help='Stock ticker symbol of the stock to be simulated', default='ASML')
    args = parser.parse_args()
    print(vars(args))
    
    # Run main with the arguments passed
    main(args)
