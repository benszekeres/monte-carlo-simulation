import argparse


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
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
    return ivalue
