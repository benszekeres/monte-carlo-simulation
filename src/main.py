"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    # Visualise each price path over time
    days = np.arange(T+1)  # x-axis 
    for path_idx in range(price_paths.shape[1]):  # price_paths has shape (T+1, N)
        plt.plot(days, price_paths[:, path_idx], linewidth=0.5, alpha=0.2)
    
    # Format plot
    plt.title('Monte Carlo Simulation of ASML Share Price')
    plt.xlabel('Days into the Future')
    plt.ylabel('Share Price')
    plt.xlim(left=days[0], right=days[-1])
    
    # Save plot in the repository's home directory
    fig_savepath = script_dir / '..' / 'price_paths.png'
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
