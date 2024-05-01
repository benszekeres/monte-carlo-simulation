import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


# Define and apply global constants for the sizes of plots
FIG_SIZE = (8, 4.5)  # downsized 16:9 aspect ratio specified as inches
plt.rcParams['figure.figsize'] = FIG_SIZE

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
        days: An array of trading days.
        pct_10: An array of the 10th percentile prices at each trading day.
        pct_25: An array of the 25th percentile prices at each trading day.
        mean: An array of the mean prices at each trading day.
        pct_75: An array of the 75th percentile prices at each trading day.
        pct_90: An array of the 90th percentile prices at each trading day.
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
    ax1.set_ylabel('Share Price')
    ax1.legend(loc='upper left')

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_price_paths_shaded.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

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

def plot_histogram(returns: np.ndarray, N: int, base_dir: Path, ticker: str) -> None:
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

    # Set x-axis limits to the nearest full percentage
    min_return = np.floor(min(returns))
    max_return = np.ceil(max(returns))
    ax.set_xlim(left=min_return, right=max_return)

    # Save plot in the repository's home directory
    fig_savepath = base_dir / '..' / f'{ticker}_histogram_returns.png'
    plt.savefig(fig_savepath)
    plt.show()
    plt.clf()

def plot_box(price_paths: np.ndarray, simulation_dates: pd.DatetimeIndex,
             T: int, base_dir: Path, ticker: str) -> None:
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

def plot_summary_statistics(statistics_df: pd.DataFrame, ticker: str) -> None:
    """Docstring to follow.
    """
    # Define colours
    edge_colour = 'white'
    header_cell_colour = '#141866'  # dark blue/navy
    header_text_colour = 'white'
    row_colors = ['white', 'lightgrey']  # every other row will be shaded

    # Set a figure size that can accommodate the full table
    _, ax = plt.subplots(figsize=FIG_SIZE)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=statistics_df.values,
                     colLabels=statistics_df.columns,
                     bbox=[0, 0, 1, 1],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)

    # Make the column headers bold
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor(edge_colour)
        cell.set_height(0.1)  # adjust row height for all rows
        if row_idx == 0:  # i.e. first row
            cell.get_text().set_weight('bold')
            cell.get_text().set_color(header_text_colour)
            cell.set_facecolor(header_cell_colour)
            cell.set_fontsize(13)
        else:
            is_shaded = row_idx % len(row_colors)
            cell.set_facecolor(row_colors[is_shaded])  # 'lightgrey' if True
            cell.set_fontsize(12)

    plt.tight_layout()

    # Save figure in the repository's home directory
    plt.savefig(f'{ticker}_summary_statistics.png')
    plt.show()
    plt.clf()
