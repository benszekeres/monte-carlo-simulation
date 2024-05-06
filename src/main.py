"""This module implements a simple Monte Carlo simulation.

Example use:
    python main.py --ticker ASML.AS --iterations 1000 --days 252
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Append the project root directory to sys.path to import from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import plots
from utils.helpers import positive_int, valid_ticker

# Set up and configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class MonteCarlo:
    """A class for running Monte Carlo simulations on share price data.

    Attributes:
        T (int): Number of future trading days to simulate.
        N (int): Number of paths to simulate.
        ticker (str): Stock ticker symbol of the stock to be simulated.
        script_dir (Path): The absolute path to the current script directory.
        df (pd.DataFrame): DataFrame holding the share price data.
        adj_close (np.ndarray): Adjusted close prices extracted from `df`.
        dates (pd.Series): The dates pertaining to closing prices, extracted from `df`.
        mean (float): Mean of the historical logarithmic returns computed from `adj_close`.
        sigma (float): Standard deviation of the historical logarithmic returns computed from `adj_close`.
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
        max_history (int): Maximum share price history to display during plotting.
        simulation_dates (pd.Series): The future dates encompassed by the simulation time horizon.
        combined_dates (np.ndarray): The combined range of historical and simulated dates to plot. 
        summary_stats (pd.DataFrame): A variety of simulation-related summary metrics. 
    """

    def __init__(self, T: int, N: int, ticker: str) -> None:
        """Initialise MonteCarlo.

        Arguments:
            T: Number of future trading days to simulate.
            N: Number of paths to simulate.
            ticker: Stock ticker symbol of the stock to be simulated.
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
        The CSV file should contain historical share prices under an 'Adj Close' header, 
        as well as corresponding dates under a 'Date' header. 

        Missing values in the 'Adj Close' series are populated via linear interpolation, 
        while missing values in the 'Date' series are populated using a helper function.

        Raises:
            FileNotFoundError: If the CSV file cannot be loaded.
            KeyError: If the CSV has no columns called 'Adj Close' and 'Date'.
        """
        # Try loading the CSV
        try:
            data_path = self.script_dir / '..' / 'data' / f'{self.ticker}.csv'
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            logging.error(f'File {self.ticker}.csv was not found.')
            raise FileNotFoundError(f'File {self.ticker}.csv was not found.')
        
        # Define potential spreadsheet-related error codes that could exist in the CSV file
        error_codes = ['#N/A', '#VALUE!', '#REF!', '#NAME?', '#DIV/0!', '#NULL!', '#NUM!']

        # Try accessing and cleaning the 'Adj Close' column, fill missing values with interpolation
        try:
            self.df['Adj Close'] = self.df['Adj Close'].replace(error_codes, np.nan)
            self.df['Adj Close'] = self.df['Adj Close'].interpolate()  # fill in NaNs
            self.adj_close = pd.to_numeric(self.df['Adj Close']).values
        except KeyError:
            logging.error(f'Column "Adj Close" not found in {self.ticker}.csv.')
            raise KeyError(f'Column "Adj Close" not found in {self.ticker}.csv.')
        
        # Try accessing and cleaning the 'Date' column, fill in missing values if there are any
        try:
            self.df['Date'] = self.df['Date'].replace(error_codes, np.nan)
            self.df['Date'] = pd.to_datetime(self.df['Date'].values)
            self.dates = self.fill_dates(self.df['Date'])
        except KeyError:
            logging.error(f'Column "Date" not found in {self.ticker}.csv.')
            raise KeyError(f'Column "Date" not found in {self.ticker}.csv.')
        
    @staticmethod
    def fill_dates(dates: pd.Series) -> pd.Series:
        """Fills in the missing dates in a series, if there are any.

        If the input series does not contain missing values, returns the original series unchanged.
        If any missing values are present, returns a new series with all gaps filled, assuming
        that each missing date is sequential by at least a single day.

        Arguments:
            dates: The range of dates that may include missing values.

        Returns:
            filled_dates: The range of dates without any gaps.
        """
        # If no missing dates exist, return `dates` unchanged
        if not dates.isna().any():
            return dates

        # List to store the completed range of dates
        filled_dates = []

        dates = pd.to_datetime(dates)  # convert to DateTime if needed

        # Iterate through the `dates` Series and fill in gaps
        for i in range(len(dates) - 1):
            current_date = dates.iat[i]

            # If there is no gap, simply update `filled_dates` and continue
            if not pd.isna(current_date):
                filled_dates.append(current_date)
                continue

            # If there is a gap, increment `current_date` by one and update `filled_dates`
            current_date += pd.Timedelta(days=1)
            filled_dates.append(current_date)

        # Append the last date
        filled_dates.append(dates.iat[-1])

        return pd.Series(filled_dates, dtype='datetime64[ns]')
        
    def simulate(self) -> None:
        """Performs the MonteCarlo simulation.

        Log returns are computed along with their mean and standard deviation,
        which are used to simulate `self.N` number of price paths over `self.T` days.
        Simulated returns are computed from said price paths, and two class member
        functions are called to compute VaR, CVaR, and summary statistics.
        """
        # Use the adjusted close price to compute log returns
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
        """Computes statistics to summarise the simulation parameters and its outcomes.

        The statistics are grouped into four categories: 
            - 'Simulation Overview': Path counts, simulation dates, and time horizons.
            - 'Share Prices': Starting, mean, minimum, and maximum final prices.
            - 'Return Metrics': Mean, minimum, and maximum returns.
            - 'Risk Metrics': VaR and CVaR at specified confidence levels of 95% and 99%.

        The summary statistics are stored as a class member DataFrame to facilitate
        visualisation in `self.plot`.
        """
        # Compute prices and return metrics
        self.mean_prices = np.mean(self.price_paths, axis=1)  # has shape T+1 i.e. mean price per day
        self.min_price = np.min(self.price_paths[-1])
        self.max_price = np.max(self.price_paths[-1])
        returns = (self.price_paths[-1] - self.adj_close[-1]) / self.adj_close[-1]
        mean_return, min_return, max_return = np.mean(returns), np.min(returns), np.max(returns)
        self.pct_10 = np.percentile(self.price_paths, q=10, axis=1)
        self.pct_25 = np.percentile(self.price_paths, q=25, axis=1)
        self.pct_75 = np.percentile(self.price_paths, q=75, axis=1)
        self.pct_90 = np.percentile(self.price_paths, q=90, axis=1)

        # Compute date-related variables
        self.max_history = min(len(self.adj_close), (self.T+1)*3)  # avoid displaying too much historical data
        dates_axis = self.dates[-self.max_history:]
        self.simulation_dates = pd.date_range(
            start=dates_axis.iat[-1] + pd.Timedelta(days=1), periods=self.T+1, freq='B'
            )
        # Combine historical and simulation horizon dates
        self.combined_dates = np.concatenate((dates_axis, self.simulation_dates))

        # Create summary statistics table with sections
        data = {
            'Simulation Overview': [
                {'Metric': 'Number of Simulated Paths', 'Value': self.N},
                {'Metric': 'Simulation Time Horizon', 'Value': f'{self.T} trading days'},
                {'Metric': 'Simulation Start Date', 'Value': f'{self.simulation_dates[0].date()}'},
                {'Metric': 'Simulation End Date', 'Value': f'{self.simulation_dates[-1].date()}'}
            ],
            'Share Prices': [
                {'Metric': 'Starting Price', 'Value': f'{self.adj_close[-1]:.0f}'},
                {'Metric': 'Mean Final Price', 'Value': f'{self.mean_prices[-1]:.0f}'},
                {'Metric': 'Min Final Price', 'Value': f'{self.min_price:.0f}'},
                {'Metric': 'Max Final Price', 'Value': f'{self.max_price:.0f}'}
            ],
            'Return Metrics': [
                {'Metric': f'Mean Return', 'Value': f'{mean_return:.1%}'},
                {'Metric': f'Min Return', 'Value': f'{min_return:.1%}'},
                {'Metric': f'Max Return', 'Value': f'{max_return:.1%}'}
            ],
            'Risk Metrics': [
                {'Metric': f'VaR 95%', 'Value': f'{self.var[0.95]:.1%}'},
                {'Metric': f'CVaR 95%', 'Value': f'{self.cvar[0.95]:.1%}'},
                {'Metric': f'VaR 99%', 'Value': f'{self.var[0.99]:.1%}'},
                {'Metric': f'CVaR 99%', 'Value': f'{self.cvar[0.99]:.1%}'},
            ]
        }

        # Convert into DataFrame
        rows = []
        for section, entries in data.items():
            for entry in entries:
                rows.append({**entry, 'Section': section})  # unpack dict and add section
        self.summary_stats = pd.DataFrame(rows)
        
    def plot(self) -> None:
        """Plots various figures to visualise the simulation outcomes.

        The figures plotted are:
            - Line chart with simulated price paths.
            - Line chart with historical & simulated price paths.
            - Histogram showing the distribution of simulated returns.
            - Box plot showing the distribution of simulated prices across
               four evenly spaced points in time over the simulation.
            - Table displaying the contents of `self.summary_stats`.
        """
        # Plot simulated price paths including an 80% confidence interval
        days = np.arange(self.T+1)  # x-axis
        plots.plot_price_paths(days, self.pct_10, self.pct_25, self.mean_prices, self.pct_75, self.pct_90,
                               base_dir=self.script_dir, ticker=self.ticker)
        
        # Plot both historical share price and simulated price paths
        plots.plot_price_paths_with_history(
            self.combined_dates, self.max_history, self.adj_close, self.pct_10, self.pct_25, self.mean_prices,
            self.pct_75, self.pct_90, base_dir=self.script_dir, ticker=self.ticker
            )
        
        # Plot histogram of simulated returns
        plots.plot_histogram(self.simulated_returns, self.N, base_dir=self.script_dir, ticker=self.ticker)

        # Add box plot of prices at given five evenly spaced time points
        plots.plot_box(self.price_paths, self.simulation_dates, self.T, base_dir=self.script_dir, ticker=self.ticker)

        # Save table of summary statistics as an image
        plots.plot_summary_statistics(self.summary_stats, self.ticker, base_dir=self.script_dir)


def main(args: argparse.Namespace) -> None:
    """Executes the Monte Carlo simulation for share price forecasting.

    This function initializes the MonteCarlo class with user-provided arguments,
    runs the simulation, and then plots the results. It handles any exceptions that
    occur during the simulation process by logging the error and exiting the program.

    Args:
        args: Command line arguments passed to the script.
            - days (int): The number of future trading days to simulate.
            - iterations (int): The number of simulation paths to generate.
            - ticker (str): The stock ticker symbol to simulate.

    Raises:
        Exception: Catches and logs any exceptions that occur during the simulation,
                   then exits the program with a status code of 1.
    """
    try:
        monte_carlo = MonteCarlo(T=args.days, N=args.iterations, ticker=args.ticker)
        monte_carlo.simulate()
        monte_carlo.plot()
    except Exception as e:
        logging.error(f'An error has occcured: {e}')
        sys.exit(1)


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')

    parser.add_argument('--days', '-d', type=positive_int, default=252,
                         help='Number of future trading days to simulate. Defaults to one 252 reflecting one year.')
    parser.add_argument('--iterations', '-i', type=positive_int, default=1000,
                         help='Number of paths to simulate.')
    parser.add_argument('--ticker', '-t', type=valid_ticker, default='ASML.AS',
                         help='Stock ticker symbol of the stock to be simulated. Must be alphanumeric.')
    args = parser.parse_args()
    logging.info(vars(args))
    
    # Run main with the arguments passed
    main(args)
