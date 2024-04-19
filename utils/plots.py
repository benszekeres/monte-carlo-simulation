import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


# Define and apply global constants for the sizes of plots
FIG_SIZE = (8, 4.5)  # downsized 16:9 aspect ratio specified as inches
plt.rcParams['figure.figsize'] = FIG_SIZE

def plot_price_paths(days, pct_10, pct_25, mean, pct_75, pct_90, base_dir, ticker):
    """Docstring to follow.
    """
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
    plt.title(f'{ticker.upper()} Simulated Share Price Paths')
    ax1.set_xlabel('Days into the Future')
    ax1.set_ylabel('Share Price')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_price_paths_shaded.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_price_paths_with_history(combined_dates, max_history, adj_close, pct_10, pct_25, mean, pct_75, pct_90, base_dir, ticker):
    """Docstring to follow.
    """
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
    plt.title(f'{ticker.upper()} Share Prices: Historical & Simulated')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Share Price')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_price_paths_with_history.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_histogram(returns, N, base_dir, ticker):
    """Docstring to follow.
    """
    num_bins = int(N / 20)  # to maintain bin density regardless of number of paths

    plt.hist(returns, bins=num_bins, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title(f'Distribution of {ticker.upper()} Simulated Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')

    # Format x-axis as a percentage
    formatter = mticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)

    # Adjust x-axis tick frequency
    ticker_frequency = max(returns) / 10  # ensure ten ticks regardless of values
    ticker_frequency_rounded = round(ticker_frequency, -int(np.floor(np.log10(ticker_frequency)))) # Rounds to nearest power of 10
    ax.xaxis.set_major_locator(mticker.MultipleLocator(ticker_frequency_rounded))

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_histogram_returns.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_box(price_paths, simulation_dates, T, base_dir, ticker):
    """Docstring to follow.
    """
    # Compute time point prices and dates
    tp_prices = [price_paths[i] for i in [T//4, 2*T//4, 3*T//4, -1]]
    tp_dates = [simulation_dates[i] for i in [T//4, 2*T//4, 3*T//4, -1]]

    # Convert dates to nearest month-end
    month_ends = [tp_date.to_period('M').to_timestamp(how='end').date().strftime('%Y/%m/%d')
                   for tp_date in tp_dates]  # nearest month ends
    month_ends[-1] = f'{month_ends[-1]} (Final)'
    
    sns.boxplot(data=tp_prices)
    plt.title(f'Box Plot of {ticker.upper()} Simulated Share Prices at Selected Time Points')
    plt.xlabel('Nearest Month End')
    plt.ylabel('Share Price')
    plt.xticks(ticks=range(4), labels=month_ends)

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_box_plot.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()
