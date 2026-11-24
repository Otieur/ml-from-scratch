"""Linear regression by least squares.

Model: y is approximated by w_0 + w_1 x_1 + ... + w_D x_D. With a leading column of ones
in the data matrix X (N, D + 1), the weights minimize ||X w - y||^2.
"""
import numpy as np


def add_bias(X):
    """Prepend a column of ones to X (N, D), giving (N, D + 1)."""
    X = np.asarray(X, dtype=np.float64)
    return np.hstack((np.ones((X.shape[0], 1)), X))


def fit_least_squares(X, y):
    """Weights that minimize the sum of squared errors.

    Args:
    - X (ndarray (Shape: (N, D))): one row per sample.
    - y (ndarray (Shape: (N,) or (N, T))): targets.

    Output:
    - w (ndarray (Shape: (D + 1,) or (D + 1, T))): the bias first, then one weight per feature.

    The system is solved by a least-squares routine instead of forming (X^T X)^-1, which
    would square the condition number of X. When X^T X is singular (for example when a
    feature is duplicated) the minimum-norm solution is returned, and its predictions are
    still the best fit.
    """
    w, _, _, _ = np.linalg.lstsq(add_bias(X), np.asarray(y, dtype=np.float64), rcond=None)
    return w


def predict(w, X):
    """Predictions for X (N, D) with weights w as returned by fit_least_squares."""
    return add_bias(X) @ w


def mean_squared_error(y_true, y_pred):
    """Mean of the squared differences.

    The two arrays must have the same shape: a (N,) vector against a (N, 1) column would
    broadcast to an N x N matrix of differences and give a wrong value without an error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape, f"Shapes differ: {y_true.shape} and {y_pred.shape}"
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true, y_pred):
    """Share of the variance of y_true explained by y_pred (1 is a perfect fit)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    return 1.0 - mean_squared_error(y_true, y_pred) / float(np.var(y_true))
