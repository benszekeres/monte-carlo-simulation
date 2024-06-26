import argparse
import logging

# Set up and configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# Define a validator that checks whether the integer-converted representation of a string is positive
def positive_int(value: str) -> int:
    """Converts a string to an integer and checks whether it is positive.

    Args:
        value: The string to be converted to an integer.

    Returns:
        ivalue: The converted integer, if it is positive.
    
    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        logging.exception(f'{value} is an invalid positive int value.')
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value.')
    return ivalue

# Define a validator that checks whether a ticker is an appropriate string
def valid_ticker(ticker: str) -> str:
    """Checks whether a ticker is a valid string. 

    A ticker is a valid string if, once replacing any `.` and `-` characters, it is alphanumeric.
    The reason that dots are allowed is because the stock tickers contain one, used as a
    separator between the stock ticker symbol and the exchange suffix. While a dash `-` is
    arguably less common in stock tickers, some such as `NOVO-B.CO` still contain one. 

    Args:
        ticker: The ticker of the stock to be used in the simulation.

    Returns:
        ticker: The ticker if it is a valid alphanumeric string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid stock ticker.
    """
    ticker_to_check = ticker.replace('.', '').replace('-', '')
    if not ticker_to_check.isalnum():
        logging.exception(f'Invalid stock ticker. Stock tickers should be alphanumeric.')
        raise argparse.ArgumentTypeError(f'Invalid stock ticker. Stock tickers should be alphanumeric.')
    return ticker
