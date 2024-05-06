import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


# Define and apply global constants for the sizes of plots
CHART_SIZE = (8, 4.5)  # downsized 16:9 aspect ratio specified as inches
TABLE_SIZE = (8, 10)  # custom table size specified as inches
plt.rcParams['figure.figsize'] = CHART_SIZE

# Define a dictionary that maps the exchange on which a given stock is listed to its share price currency
CURRENCIES = {
    'AS': 'EUR',  # Euronext Amsterdam
    'CO': 'DKK',  # Copenhagen Stock Exchange
    'L': 'GBp',   # London Stock Exchange (GBp stands for Great British Pence)
    'PA': 'EUR',  # Euronext Paris
    'SW': 'CHF'   # SIX Swiss Exchange
}

def plot_price_paths(days: np.ndarray,
                     pct_10: np.ndarray,
                     pct_25: np.ndarray,
                     mean: np.ndarray,
                     pct_75: np.ndarray,
                     pct_90: np.ndarray,
                     base_dir: Path,
                     ticker: str) -> None:
    """Plot simulated share price paths.

    Args:
        days: Array of trading days.
        pct_10: Array of the 10th percentile prices at each trading day.
        pct_25: Array of the 25th percentile prices at each trading day.
        mean: Array of the mean prices at each trading day.
        pct_75: Array of the 75th percentile prices at each trading day.
        pct_90: Array of the 90th percentile prices at each trading day.
        base_dir: The base directory where the plot image will be saved.
        ticker: The stock ticker symbol.
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
    ax1.set_xlabel('Trading days into the Future')
    currency = CURRENCIES[ticker.split('.')[-1]]
    ax1.set_ylabel(f'Share Price ({currency})')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '../results' / f'{ticker}_price_paths_shaded.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.close()

def plot_price_paths_with_history(combined_dates: pd.DatetimeIndex,
                                  max_history: int,
                                  adj_close: np.ndarray,
                                  pct_10: np.ndarray,
                                  pct_25: np.ndarray,
                                  mean: np.ndarray,
                                  pct_75: np.ndarray,
                                  pct_90: np.ndarray,
                                  base_dir: Path,
                                  ticker: str) -> None:
    """Plot both historical and simulated share price paths.

    Args:
        combined_dates: Combined array of historical and simulated dates.
        max_history: Number of days to include historical data for.
        adj_close: Array of adjusted closing prices for the historical days.
        pct_10: Array of the 10th percentile simulated prices.
        pct_25: Array of the 25th percentile simulated prices.
        mean: Array of the mean simulated prices.
        pct_75: Array of the 75th percentile simulated prices.
        pct_90: Array of the 90th percentile simulated prices.
        base_dir: The base directory where the plot image will be saved.
        ticker: The stock ticker symbol.
    """
    plt.plot(combined_dates[:max_history], adj_close[-max_history:], alpha=1, color='#1f77b4',
             label='Historical Share Price')
    plt.plot(combined_dates[max_history:], pct_75, linewidth=1.5, alpha=1, color='#2ca02c',
             label='75th percentile')
    plt.plot(combined_dates[max_history:], mean, linewidth=1.5, alpha=1, color='#ff7f0e',
             label='Mean')
    plt.plot(combined_dates[max_history:], pct_25, linewidth=1.5, alpha=1, color='#d62728',
             label='25th percentile')
    plt.fill_between(combined_dates[max_history:], pct_10, pct_90, color='gray', alpha=0.2,
                     label='80% Confidence Interval')

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
    currency = CURRENCIES[ticker.split('.')[-1]]
    ax1.set_ylabel(f'Share Price ({currency})')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '../results' / f'{ticker}_price_paths_with_history.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.close()

def plot_histogram(returns: np.ndarray, N: int, base_dir: Path, ticker: str) -> None:
    """Plot a histogram of simulated returns.

    The returns are computed on the last day of each price path, relative
    to the starting share price. 

    Args:
        returns: Array of simulated return values computed on the last day.
        N: Number of simulation paths used to determine the number of bins.
        base_dir: The base directory where the histogram will be saved.
        ticker: The stock ticker symbol.
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

    # Set x-axis limits to the nearest full percentage
    min_return = np.floor(min(returns))
    max_return = np.ceil(max(returns))
    ax.set_xlim(left=min_return, right=max_return)

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '../results' / f'{ticker}_histogram_returns.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.close()

def plot_box(price_paths: np.ndarray, simulation_dates: pd.DatetimeIndex,
             T: int, base_dir: Path, ticker: str) -> None:
    """Plot a box plot of simulated share prices at selected time point.

    Args:
        price_paths: Array of simulated prices at each chosen time point.
        simulation_dates: Dates for each chosen time point.
        T: The total number of simulated days.
        base_dir: The base directory where the box plot will be saved.
        ticker: The stock ticker symbol.
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
    currency = CURRENCIES[ticker.split('.')[-1]]
    plt.ylabel(f'Share Price ({currency})')
    plt.xticks(ticks=range(4), labels=month_ends)

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '../results' / f'{ticker}_box_plot.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.close()

def plot_summary_statistics(statistics_df: pd.DataFrame, ticker: str, base_dir: Path) -> None:
    """Plot a table of summary statistics for simulated data.

    The table plotted consists of four sub-tables pertaining to sections:
        - 'Simulation Overview': Path counts, simulation dates, and time horizons.
        - 'Price Statistics': Starting, mean, minimum, and maximum final prices.
        - 'Return Metrics': Mean, minimum, and maximum returns.
        - 'Risk Metrics': VaR and CVaR at specified confidence levels of 95% and 99%.

    Args:
        statistics_df: DataFrame containing calculated statistics.
        ticker: The stock ticker symbol.
        base_dir: The base directory where the box plot will be saved.
    """
    # Define colours
    edge_colour = 'white'
    header_cell_colour = '#141866'  # dark blue/navy
    header_text_colour = 'white'
    row_colors = ['white', 'lightgrey']  # every other row will be shaded

    # Initialise the figure and axis which will accommodate the entire table
    fig, ax = plt.subplots(figsize=TABLE_SIZE)
    ax.axis('off')

    # Calculate the number of rows needed
    total_rows = sum(1 + len(group) + 1
                      for _, group in statistics_df.groupby('Section'))  # +1 for header, +1 for empty rows
    row_height = 1.0 / total_rows  # height per row

    # Initialise variable to keep track of the current position within the overall table
    current_position = 0

    # Define the order of the sections (in reverse, since first list item will be the bottom one)
    section_order = ['Risk Metrics', 'Return Metrics', 'Share Prices', 'Simulation Overview']

    # Convert the 'Section' column to a categorical type with the defined order
    statistics_df['Section'] = pd.Categorical(statistics_df['Section'],
                                               categories=section_order,
                                               ordered=True)

    # Iterate through each section and plot a table for it
    for section, group in statistics_df.groupby('Section', observed=True):
        num_rows = len(group) + 1  # +1 for header
        section_height = num_rows * row_height

        # Plot a table for the given section
        table = ax.table(cellText=group[['Metric', 'Value']].values,
                         colLabels=[section, ''],
                         bbox=[0, current_position, 1, section_height],
                         cellLoc='center',
                         loc='bottom')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.2)  # increase row heights

        # Obtain the currency for the given stock to add to the share prices appearing in the table
        currency = CURRENCIES[ticker.split('.')[-1]]

        # Format the table
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor(edge_colour)
            if row_idx == 0:
              cell.set_facecolor(header_cell_colour)
              cell.get_text().set_color(header_text_colour)
              cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor(row_colors[row_idx % 2])  # shade every other row
                # If the given entry is a share price then add its currency
                if section == 'Share Prices':
                    cell.get_text().set_text(f'{cell.get_text().get_text()} ({currency})')

        # Update current position in the overall plot
        current_position += section_height

        # Add empty row after each section (except after the last one)
        if current_position < total_rows:
            current_position += row_height

    # Save figure in the repository's home directory
    plt.tight_layout()
    fig_savepath = base_dir / '../results/' / f'{ticker}_summary_statistics.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.close()
