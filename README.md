# Monte Carlo Simulation for Share Price Forecasting

## Introduction
This project implements a simple Monte Carlo simulation to forecast future share prices using historical data.

## Features
- Simulate the share price of a stock using a Monte Carlo simulation.
- Specify the number of future trading days over which to simulate, and the number of simulation paths to use.
- Output statistical summaries and visualisations of potential future share prices.

## Data Sources

### Share Price Data
The Monte Carlo simulations in this project are based on historical share price data from **[Yahoo Finance](https://uk.finance.yahoo.com/)**. The data includes daily openings, closings, intraday highs and lows, and volume information.

### Stocks Included
- **ASML Holding N.V. (```ASML.AS```)** - Historical data ranges from 20/07/1998 to 03/05/2024, share prices are in EUR.
- **LVMH Moët Hennessy Louis Vuitton S.E. (```MC.PA```)** - Historical data ranges from 03/01/2000 to 03/05/2024, share prices are in EUR.
- **Nestlé S.A. (```NESN.SW```)** - Historical data ranges 03/01/1990 to 03/05/2024, share prices are in CHF.
- **Novo Nordisk A/S (```NOVO-B.CO```)** - Historical data ranges from 06/01/2001 to 03/05/2024, share prices are in DKK.
- **Shell plc (```SHEL.L```)** - Historical data ranges from 29/08/1996 to 03/05/2024, share prices are in GBp.

For more details on the data used in this project, please see the README in the [data directory](data/README.md).

## Simulation Methodology

### Log-Normal Returns
The Monte Carlo simulation implemented in this project assumes that share price returns follow a log-normal distribution.

### Mathematical Model
The simulation uses historical price data to:
- **Compute Logarithmic Returns**: Calculate the log returns from the historical adjusted close prices.
- **Apply Geometric Brownian Motion (GBM)**: Simulate future stock prices assuming that price movements are modeled by a GBM, which is characterised by continuous compounding returns influenced by stochastic drift and volatility factors.
- **Simulate Price Paths**: Generate multiple price paths to forecast future price behavior and assess potential financial outcomes.

### Risk Metrics
Included in the simulation are calculations of risk metrics such as:
- **Value at Risk (VaR)**: Estimates the maximum loss expected over a set time period at confidence levels of 95% and 99%.
- **Conditional Value at Risk (CVaR)**: Computes the mean loss assuming that losses exceed the VaR thresholds, providing insight into potential tail risks.

## Getting Started
### Prerequisites
- Python 3.8+

### Installation
1. Clone the repo:
```shell
git clone https://github.com/benszekeres/monte-carlo-simulation
cd monte-carlo-simulation
```

2. Set up a virtual environment
 - On macOS and Linux:
 ```shell
 python3 -m venv env
 source env/bin/activate
 ```
 - On Windows:
 ```shell
 python3 -m venv env
 .\env\Scripts\activate
 ```

3. Install required packages:
```shell
pip install -r requirements.txt
```

### Usage
Run the simulation:
```shell
python src/main.py --ticker <stock_ticker> --iterations <num_price_paths> --days <num_trading_days>
```

Example run:
```shell
python src/main.py --ticker ASML.AS --iterations 1000 --days 252
```

## Results
The following are example results from running the simulation as follows:
```shell
python src/main.py --ticker ASML.AS --iterations 1000 --days 252
```

**Figure 1: Summary of a Monte Carlo simulation for ASML.AS over 252 trading days and 1000 price paths**

<img src="results/ASML.AS_summary_statistics.png" alt="Summary Statistics" width="640" height="800">

**Figure 2: Future price paths predicted by the Monte Carlo simulation**

<img src="results/ASML.AS_price_paths_shaded.png" alt="Price Paths" width="711" height="400">

**Figure 3: Historical and future price paths predicted by the Monte Carlo simulation**

<img src="results/ASML.AS_price_paths_with_history.png" alt="Price Paths with History" width="711" height="400">

**Figure 4: Distribution of returns from the Monte Carlo simulation**

<img src="results/ASML.AS_histogram_returns.png" alt="Histogram of Returns" width="711" height="400">

**Figure 5: Box plot of the Monte Carlo simulation at four chosen dates**

<img src="results/ASML.AS_box_plot.png" alt="Box Plot" width="711" height="400">

## Contact
Benjamin Szekeres – [benjamin.a.szekeres@gmail.com](mailto:benjamin.a.szekeres@gmail.com)
