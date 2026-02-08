"""
Test module for R-Python bridge verification.
This module provides simple functions to verify reticulate can call Python.
"""

import numpy as np


def get_test_weights(n_assets: int) -> np.ndarray:
    """
    Return dummy equal weights for testing the R-Python bridge.

    Args:
        n_assets: Number of assets in portfolio

    Returns:
        NumPy array of equal weights summing to 1
    """
    return np.ones(n_assets) / n_assets


def get_random_weights(n_assets: int, seed: int = 42) -> np.ndarray:
    """
    Return random weights for testing.

    Args:
        n_assets: Number of assets in portfolio
        seed: Random seed for reproducibility

    Returns:
        NumPy array of random weights summing to 1
    """
    np.random.seed(seed)
    weights = np.random.random(n_assets)
    return weights / weights.sum()


def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two arrays element-wise.

    Args:
        a: First array
        b: Second array

    Returns:
        Element-wise sum
    """
    return np.array(a) + np.array(b)


def bridge_info() -> dict:
    """
    Return information about the Python environment for debugging.

    Returns:
        Dictionary with Python version and numpy version
    """
    import sys
    return {
        "python_version": sys.version,
        "numpy_version": np.__version__
    }


if __name__ == "__main__":
    # Test functions locally
    print("Testing bridge functions...")

    weights = get_test_weights(5)
    print(f"Equal weights (5 assets): {weights}")
    print(f"Sum: {weights.sum()}")

    random_weights = get_random_weights(5)
    print(f"Random weights: {random_weights}")
    print(f"Sum: {random_weights.sum()}")

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"Array addition: {a} + {b} = {add_arrays(a, b)}")

    print(f"Bridge info: {bridge_info()}")
