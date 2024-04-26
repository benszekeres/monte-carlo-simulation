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
    """A class for running Monte Carlo simulations on share price data.

    Attributes:
        T (int): Number of future trading days to simulate.
        N (int): Number of paths to simulate.
        ticker (str): Stock ticker symbol of the stock to be simulated.
        script_dir (Path): The absolute path to the current script directory.
        df (pd.DataFrame): DataFrame holding the share price data.
        adj_close (np.ndarray): Adjusted close prices extracted from `df`.
        price_paths (np.ndarray): Simulated price paths for the stock.
        simulated_returns (np.ndarray): Simulated returns from all price paths.
        var (dict[float, float]): Value at Risk (VaR) values for the specified confidence levels.
        cvar (dict[float, float]): Conditional Value at Risk (CVaR) values for the specified confidence levels.
        summary_stats (pd.DataFrame): Summary statistics for the simulation results.
        confidence_thresh (list[float]): Confidence thresholds used for VaR and CVaR calculations.
        mean_prices (np.ndarray): Mean prices calculated per day over the simulation.
        min_price (float): Minimum price on the last day of simulation.
        max_price (float): Maximum price on the last day of simulation.
        pct_10 (np.ndarray): 10th percentile prices calculated per day over the simulation.
        pct_25 (np.ndarray): 25th percentile prices calculated per day over the simulation.
        pct_75 (np.ndarray): 75th percentile prices calculated per day over the simulation.
        pct_90 (np.ndarray): 90th percentile prices calculated per day over the simulation.
    """

    def __init__(self, T: int, N: int, ticker: str) -> None:
        """Initialise MonteCarlo.

        Arguments:
            T (int): Number of future trading days to simulate.
            N (int): Number of paths to simulate.
            ticker (str): Stock ticker symbol of the stock to be simulated.
        """
        self.T = T
        self.N = N
        self.ticker = ticker

        # Obtain the absolute path to the current script (main.py)
        self.script_dir = Path(__file__).resolve().parent

        # Load data
        self.load_data()

    def load_data(self) -> None:
        """Loads share price data from a CSV file into the DataFrame `self.df`.

        The CSV file is expected to be named after a stock ticker symbol (`self.ticker`), 
        located in the `data` directory relative to the parent directory of the script.
        The CSV file should contain historical share prices with an 'Adj Close' header.
        """
        data_path = self.script_dir / '..' / 'data' / f'{self.ticker}.csv'
        self.df = pd.read_csv(data_path)

    def simulate(self) -> None:
        """Performs the MonteCarlo simulation.

        Log returns are computed along with their mean and standard deviation,
        which are used to simulate `self.N` number of price paths over `self.T` days.
        Simulated returns are computed from said price paths, and two class member
        functions are called to compute VaR, CVaR, and summary statistics.
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
        """Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR).

        Computation is done using two confidence thresholds: 0.95 and 0.99. 
        """
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
        """Computes statistics to summarise the simulation outcomes.

        The statistics computed are stored in a DataFrame `self.summary_stats`, 
        and include the following metrics:
            mean_price: Mean prices on the last day of simulation.
            min_price: Minimum price on the last day of simulation.
            max_price: Maximum price on the last day of simulation.
            pct_10, pct_25, pct_75, pct_90: The 10th, 25th, 75th and 90th percentile
            prices calculated per day over the simulation.
            var: Value at Risk (VaR) values for the specified confidence levels.
            cvar: Conditional Value at Risk (CVaR) values for the specified confidence levels.
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
        """Plots various figures to visualise the simulation outcomes.

        The figures plotted are:
            1) Line chart with simulated price paths.
            2) Line chart with historical & simulated price paths.
            3) Histogram showing the distribution of simulated returns.
            4) Box plot showing the distribution of simulated prices across
               four evenly spaced points in time over the simulation.
            5) Table displaying the contents of `self.summary_stats`.
        Some date-related variables are computed in this function, while
        the for actual plotting calls are made to appropriate methods in
        the `plots.py` utility file. 
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
                         help='Number of paths to simulate', default=1000)
    parser.add_argument('--ticker', '-t', type=str,
                         help='Stock ticker symbol of the stock to be simulated', default='ASML')
    args = parser.parse_args()
    print(vars(args))
    
    # Run main with the arguments passed
    main(args)
