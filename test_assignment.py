import numpy as np
import pandas as pd
import assignment

def test_create_1d_array():
    arr = assignment.create_1d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 1, "Must be a 1D array"
    assert np.array_equal(arr, np.array([1, 2, 3, 4, 5])), "Array must be [1, 2, 3, 4, 5]"

def test_create_2d_array():
    arr = assignment.create_2d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 2, "Must be a 2D array"
    assert arr.shape == (3, 3), "Array must have shape (3,3)"

def test_array_operations():
    arr = np.array([1, 2, 3, 4, 5])
    result = assignment.array_operations(arr)
    assert len(result) == 3, "Must return tuple of 3 values"
    assert np.isclose(result[0], 3), "Mean should be 3"
    assert np.isclose(result[1], 1.4142, atol=0.1), "Standard deviation check"
    assert result[2] == 5, "Max value should be 5"
