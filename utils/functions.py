import numpy as np
from typing import Iterable

def logistic(x, scale=1.0):
    return 1.0 / (1.0 + np.exp(-scale * x))

def logistic_derivative(x, scale=1.0):
    return scale * logistic(x, scale=scale) * (1 - logistic(x, scale=scale))

def get_variance(a: Iterable, p: Iterable) -> float:
    """
    Get the variance of a discrete random variable.

    Args:
        a (Iterable): The set of possible values. Shaped (n,).
        p (Iterable): The probabilities of each value. Shaped (n,).

    Returns:
        a_var (float): The variance of the random variable.
    """
    assert np.isclose(sum(p), 1, rtol=1e-6), "Probabilities must sum to 1."
    assert len(a) == len(p), "a and p must have the same length."
    a_mean = np.dot(p, a) # sum of element-wise multiplication
    a_var = np.dot(p, (a - a_mean) ** 2)
    return a_var

def moving_average_smoothing(data: np.ndarray, window_size: int) -> np.ndarray:
    assert type(data) == np.ndarray, "data must be a numpy array."
    assert data.ndim == 1, "data must be a 1-dimensional array."
    weights = np.ones(window_size) / window_size
    smoothed_data_moving_average = np.convolve(data, weights, 'same')
    return smoothed_data_moving_average
