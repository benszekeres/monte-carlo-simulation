"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define and apply global constants for the sizes of plots
FIG_SIZE = (8, 4.5)  # downsized 16:9 aspect ratio specified as inches
plt.rcParams['figure.figsize'] = FIG_SIZE

def main(args):
    # Obtain the absolute path to the current script (main.py)
    script_dir = Path(__file__).resolve().parent

    # Load data using a relative path to the data file
    data_path = script_dir / '..' / 'data' / 'ASML.csv'
    df = pd.read_csv(data_path)

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

    # Visualise summary statistics
    days = np.arange(T+1)  # x-axis 

    plt.plot(days, pct_75, linewidth=1.5, alpha=1, label='75th percentile')
    plt.plot(days, mean_prices, linewidth=1.5, alpha=1, label='Mean')
    plt.plot(days, pct_25, linewidth=1.5, alpha=1, label='25th percentile')
    plt.fill_between(days, pct_10, pct_90, color='gray', alpha=0.2, label='80% Confidence Interval')

    # Configure axes' limits
    plt.xlim(left=days[0], right=days[-1])
    plt.ylim(bottom=0, top=pct_90[-1])

    # Add secondary axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())

    # Set labels and legend
    plt.title('ASML Simulated Share Price Paths')
    ax1.set_xlabel('Days into the Future')
    ax1.set_ylabel('Share Price')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = script_dir / '..' / 'price_paths_shaded.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()
    

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
