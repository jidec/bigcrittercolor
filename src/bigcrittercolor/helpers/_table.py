import numpy as np
from collections import Counter


def _table(array):
    """
    Mimics the R table function for 2D arrays in Python.
    Each row in the array is considered an observation.

    Parameters:
    array (np.array): The input 2D numpy array.

    Returns:
    dict: A dictionary with tuples representing unique rows as keys and their counts as values.
    """
    # Ensure the input is a numpy array
    array = np.asarray(array)

    # Check if the array is indeed 2D
    if array.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Convert each row to a tuple and count occurrences
    row_tuples = [tuple(row) for row in array]
    counts = Counter(row_tuples)

    return counts


# Example usage
#array = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]])
#print(_table(array))