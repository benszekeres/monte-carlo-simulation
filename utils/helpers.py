import argparse
import logging

# Set up and configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# Define a validator function for command line arguments
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
    """Checks whether a ticker is a valid string, i.e. is alphanumeric.

    Args:
        ticker: The ticker of the stock to be used in the simulation.

    Returns:
        ticker: The ticker if it is a valid alphanumeric string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid stock ticker.
    """
    if not ticker.isalnum():
        logging.exception(f"Invalid stock ticker. Stock tickers should be alphanumeric.")
        raise argparse.ArgumentTypeError(f"Invalid stock ticker. Stock tickers should be alphanumeric.")
    return ticker
