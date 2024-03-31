"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

import numpy as np
import pandas as pd


def main(args):
    # Load data
    df = pd.read_csv('../data/ASML.csv')
    adj_close = df['Adj Close'].values

    # Compute stock price returns
    log_returns = np.log(adj_close[1:] / adj_close[:-1])

    # Compute mean and standard deviation
    mean = np.mean(log_returns)
    sigma = np.std(log_returns)
    
    # set up any other simulation parameters
    # construct simulation paths
    # analysis and visualisation
    pass


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
