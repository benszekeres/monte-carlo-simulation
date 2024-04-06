"""This module implements the Monte Carlo simulation for the stock price of ASML.

Example use:
    python3 main.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# Define and apply global constants for the sizes of plots
FIG_SIZE = (8, 4.5)  # downsized 16:9 aspect ratio specified as inches
plt.rcParams['figure.figsize'] = FIG_SIZE

def plot_price_paths(days, pct_10, pct_25, mean, pct_75, pct_90, base_dir):
    plt.plot(days, pct_75, linewidth=1.5, alpha=1, color='#2ca02c', label='75th percentile')
    plt.plot(days, mean, linewidth=1.5, alpha=1, color='#ff7f0e', label='Mean')
    plt.plot(days, pct_25, linewidth=1.5, alpha=1, color='#d62728', label='25th percentile')
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
    fig_savepath = base_dir / '..' / 'price_paths_shaded.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_price_paths_with_history(combined_dates, max_history, adj_close, pct_10, pct_25, mean, pct_75, pct_90, base_dir):
    plt.plot(combined_dates[:max_history], adj_close[-max_history:], alpha=1, color='#1f77b4', label='Historical Share Price')
    plt.plot(combined_dates[max_history:], pct_75, linewidth=1.5, alpha=1, color='#2ca02c', label='75th percentile')
    plt.plot(combined_dates[max_history:], mean, linewidth=1.5, alpha=1, color='#ff7f0e', label='Mean')
    plt.plot(combined_dates[max_history:], pct_25, linewidth=1.5, alpha=1, color='#d62728', label='25th percentile')
    plt.fill_between(combined_dates[max_history:], pct_10, pct_90, color='gray', alpha=0.2, label='80% Confidence Interval')

    # Configure axes' limits
    plt.xlim(left=combined_dates[0], right=combined_dates[-1])
    plt.ylim(bottom=0, top=pct_90[-1])

    # Add secondary axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())

    # Set labels and legend
    plt.title('ASML Share Prices: Historical & Simulated')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Share Price')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / 'price_paths_with_history.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_histogram(price_paths, N, base_dir):
    num_bins = int(N / 20)  # to maintain bin density regardless of number of paths
    plt.hist(price_paths[-1], bins=num_bins, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Distribution of Simulated Share Prices on Final Day')
    plt.xlabel('Share Price')
    plt.ylabel('Frequency')

    # Adjust x-axis tick frequency
    ax = plt.gca()
    ticker_frequency = max(price_paths[-1]) / 10  # ensure ten ticks regardless of values
    ticker_frequency_rounded = round(ticker_frequency, -int(np.floor(np.log10(ticker_frequency)))) # Rounds to nearest power of 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_frequency_rounded))

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / 'histogram_final_prices.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_box(tp_prices, month_ends, base_dir):
    sns.boxplot(data=tp_prices)
    plt.title('Box Plot of Simulated Share Prices at Selected Time Points')
    plt.xlabel('Nearest Month End')
    plt.ylabel('Share Price')
    plt.xticks(ticks=range(4), labels=month_ends)

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / 'box_plot.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()


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

    days = np.arange(T+1)  # x-axis

    # Plot simulated price paths including an 80% confidence interval
    plot_price_paths(days, pct_10, pct_25, mean_prices, pct_75, pct_90, base_dir=script_dir)

    # Add plot also showing historical share price
    max_history = min(len(adj_close), (T+1)*3)  # avoid too much historical data
    dates_axis = dates[-max_history:]
    # Combine historical and simulation horizon dates
    simulation_dates = pd.date_range(start=dates_axis[-1] + pd.Timedelta(days=1), periods=T+1, freq='D')
    combined_dates = np.concatenate((dates_axis, simulation_dates))

    # Plot both historical share price and simulated price paths
    plot_price_paths_with_history(combined_dates, max_history, adj_close, pct_10, pct_25, mean_prices, pct_75, pct_90, base_dir=script_dir)
    
    # Plot histogram of final prices
    plot_histogram(price_paths, N, base_dir=script_dir)

    tp_prices = [price_paths[i] for i in [T//4, 2*T//4, 3*T//4, -1]]
    tp_dates = [simulation_dates[i] for i in [T//4, 2*T//4, 3*T//4, -1]]

    # Convert dates to nearest month-end
    month_ends = [tp_date.to_period('M').to_timestamp(how='end').date().strftime('%Y/%m/%d')
                   for tp_date in tp_dates]  # nearest month ends
    month_ends[-1] = f'{month_ends[-1]} (Final)'

    # Add box plot of prices at given five evenly spaced time points
    plot_box(tp_prices, month_ends, base_dir=script_dir)


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
