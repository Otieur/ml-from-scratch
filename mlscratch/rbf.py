"""Regression with radial basis functions.

Model, for an input x in R^d and K basis functions with centers c_k and widths s_k:

    f(x) = w_0 + sum_k w_k b_k(x),      b_k(x) = exp(-||x - c_k||^2 / (2 s_k^2))

The weights are the regularized least-squares solution. With B the (N, K + 1) matrix whose
first column is ones and whose other columns are the basis functions at the N inputs,

    W = (B^T B + l2 * R)^-1 B^T Y

where R is the identity with a zero in the position of the bias, so that the penalty
shrinks the bumps toward zero without pulling the overall level of the fit toward zero.
"""
import numpy as np


class RBFRegression:
    def __init__(self, centers, width):
        """
        Args:
        - centers (ndarray (Shape: (K, d))): the centers of the K basis functions.
        - width (float or ndarray (Shape: (K,))): their widths (the same for all if a float).
        """
        self.centers = np.asarray(centers, dtype=np.float64)
        self.widths = np.broadcast_to(np.asarray(width, dtype=np.float64), (self.centers.shape[0],))
        assert np.all(self.widths > 0), "Widths must be positive."
        self.weights = None

    def design_matrix(self, X):
        """The bias column and every basis function evaluated at X (N, d): a (N, K + 1) matrix."""
        X = np.asarray(X, dtype=np.float64)
        sq_dist = np.sum((X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]) ** 2, axis=2)
        return np.hstack((np.ones((X.shape[0], 1)), np.exp(-sq_dist / (2.0 * self.widths ** 2))))

    def fit(self, X, Y, l2=0.0):
        """Fit the weights.

        Args:
        - X (ndarray (Shape: (N, d))): inputs.
        - Y (ndarray (Shape: (N,) or (N, T))): targets; T outputs are fitted at once and share B.
        - l2 (float): regularization constant.
        """
        assert l2 >= 0, f"The regularization constant cannot be negative. Got: {l2}"
        B = self.design_matrix(X)
        penalty = l2 * np.eye(B.shape[1])
        penalty[0, 0] = 0.0
        A = B.T @ B + penalty
        rhs = B.T @ np.asarray(Y, dtype=np.float64)
        try:
            self.weights = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Singular without regularization (fewer distinct inputs than weights):
            # take the minimum-norm least-squares solution.
            self.weights = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return self

    def predict(self, X):
        """Model output at X (N, d): (N,) or (N, T), matching the targets given to fit."""
        return self.design_matrix(X) @ self.weights
