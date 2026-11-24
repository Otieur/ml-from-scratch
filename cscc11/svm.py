"""Support vector machine for two classes, trained in the dual by sequential minimal optimization.

Primal:    minimize (1/2) ||w||^2 + C sum_i xi_i
           subject to y_i (w . phi(x_i) + b) >= 1 - xi_i and xi_i >= 0, with y_i in {-1, +1}
Dual:      minimize (1/2) a^T Q a - sum_i a_i,   Q_ij = y_i y_j K(x_i, x_j)
           subject to 0 <= a_i <= C and sum_i a_i y_i = 0
Decision:  f(x) = sum_i a_i y_i K(x_i, x) + b; the points with a_i > 0 are the support vectors

Only two multipliers change per step, so the equality constraint can be kept exactly:
a_i moves by y_i t and a_j by -y_j t. The pair is the one that violates the optimality
conditions the most, and t is the minimizer along that direction, clipped to the box.
"""
import numpy as np


def linear_kernel(A, B):
    """K(a, b) = a . b for every pair of rows."""
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T


def rbf_kernel(A, B, gamma):
    """K(a, b) = exp(-gamma ||a - b||^2) for every pair of rows."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    squared = np.sum(A ** 2, axis=1)[:, None] - 2.0 * (A @ B.T) + np.sum(B ** 2, axis=1)[None, :]
    return np.exp(-gamma * np.maximum(squared, 0.0))


class SVM:
    """Soft-margin kernel SVM.

    Args:
    - C (float): penalty on margin violations.
    - kernel (str): "linear" or "rbf".
    - gamma (float): width parameter of the RBF kernel.
    - tol (float): stop when the largest violation of the optimality conditions is below tol.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, tol=1e-3, max_iter=1_000_000):
        assert kernel in ("linear", "rbf")
        self.C, self.kernel, self.gamma, self.tol, self.max_iter = float(C), kernel, gamma, tol, max_iter

    def _kernel(self, A, B):
        return linear_kernel(A, B) if self.kernel == "linear" else rbf_kernel(A, B, self.gamma)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, index = np.unique(np.asarray(y), return_inverse=True)
        assert len(self.classes_) == 2, "SVM separates two classes; see OneVsRestSVM for more"
        signs = np.where(index == 1, 1.0, -1.0)
        C, K = self.C, self._kernel(X, X)
        alpha = np.zeros(len(signs))
        grad = -np.ones(len(signs))                 # gradient of the dual objective, Q a - 1

        for iteration in range(self.max_iter):
            # A multiplier can still move "up" (in the direction y_i) or "down" (against it).
            score = -signs * grad
            up = ((alpha < C) & (signs > 0)) | ((alpha > 0) & (signs < 0))
            down = ((alpha < C) & (signs < 0)) | ((alpha > 0) & (signs > 0))
            i = np.flatnonzero(up)[np.argmax(score[up])]
            j = np.flatnonzero(down)[np.argmin(score[down])]
            violation = score[i] - score[j]
            if violation < self.tol:
                break
            curvature = max(K[i, i] + K[j, j] - 2.0 * K[i, j], 1e-12)
            room_i = C - alpha[i] if signs[i] > 0 else alpha[i]
            room_j = alpha[j] if signs[j] > 0 else C - alpha[j]
            step = min(violation / curvature, room_i, room_j)
            # A multiplier that reaches a bound is set to it exactly, not to a rounded neighbour.
            alpha[i] = (C if signs[i] > 0 else 0.0) if step == room_i else alpha[i] + signs[i] * step
            alpha[j] = (0.0 if signs[j] > 0 else C) if step == room_j else alpha[j] - signs[j] * step
            grad += step * signs * (K[:, i] - K[:, j])
        self.n_iter_ = iteration

        # On a free support vector y f(x) = 1, which gives b = -y_i grad_i; average over all of them.
        score = -signs * grad
        free = (alpha > 0) & (alpha < C)
        if free.any():
            self.intercept_ = float(np.mean(score[free]))
        else:
            up = ((alpha < C) & (signs > 0)) | ((alpha > 0) & (signs < 0))
            down = ((alpha < C) & (signs < 0)) | ((alpha > 0) & (signs > 0))
            self.intercept_ = float((score[up].max() + score[down].min()) / 2.0)

        self.alpha_ = alpha
        self.support_ = np.flatnonzero(alpha > 0)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * signs[self.support_]
        self.dual_objective_ = float(0.5 * (alpha * signs) @ K @ (alpha * signs) - alpha.sum())
        return self

    def decision_function(self, X):
        """f(x) for every row; positive on the side of the second class."""
        return self._kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class OneVsRestSVM:
    """One SVM per class against all the others; the class with the largest f(x) wins."""

    def __init__(self, **svm_arguments):
        self.svm_arguments = svm_arguments

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.machines_ = [SVM(**self.svm_arguments).fit(X, (y == label).astype(int)) for label in self.classes_]
        return self

    def decision_function(self, X):
        return np.column_stack([machine.decision_function(X) for machine in self.machines_])

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
