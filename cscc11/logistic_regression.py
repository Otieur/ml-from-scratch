"""Binary logistic regression fitted by gradient descent.

Model:      P(y = 1 | x) = g(w . x + b),   g(a) = 1 / (1 + exp(-a))
Objective:  L(w, b) = - sum_i [ y_i ln p_i + (1 - y_i) ln(1 - p_i) ]  +  (l2 / 2) ||w||^2
Gradient:   dL/dw = sum_i (p_i - y_i) x_i + l2 w,      dL/db = sum_i (p_i - y_i)
Update:     parameters <- parameters - step * gradient

The parameter vector is [w_1, ..., w_D, b]: the bias comes last and is not penalized.
"""
import numpy as np


def sigmoid(a):
    """g(a) = 1 / (1 + exp(-a)), computed without overflow for large |a|."""
    a = np.asarray(a, dtype=np.float64)
    out = np.empty_like(a)
    positive = a >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-a[positive]))
    exp_a = np.exp(a[~positive])
    out[~positive] = exp_a / (1.0 + exp_a)
    return out


def augment(X):
    """Append a column of ones, so that the last parameter acts as the bias."""
    X = np.asarray(X, dtype=np.float64)
    return np.hstack((X, np.ones((X.shape[0], 1))))


def predict_proba(params, X):
    """P(y = 1 | x) for every row of X (N, D), with params of length D + 1 (bias last)."""
    return sigmoid(augment(X) @ np.asarray(params, dtype=np.float64))


def predict(params, X):
    """Class 1 when P(y = 1 | x) >= 0.5, that is on the positive side of the decision boundary."""
    return (predict_proba(params, X) >= 0.5).astype(int)


def accuracy(params, X, y):
    return float(np.mean(predict(params, X) == np.asarray(y).astype(int)))


def negative_log_likelihood(params, X, y, l2=0.0):
    """The objective L, summed over the training points."""
    params = np.asarray(params, dtype=np.float64)
    a = augment(X) @ params
    # -[y ln g(a) + (1 - y) ln(1 - g(a))] = ln(1 + exp(a)) - y a, stable for large |a|.
    return float(np.sum(np.logaddexp(0.0, a) - np.asarray(y) * a) + 0.5 * l2 * np.sum(params[:-1] ** 2))


def gradient(params, X, y, l2=0.0):
    """Gradient of the objective, a vector of length D + 1."""
    params = np.asarray(params, dtype=np.float64)
    grad = augment(X).T @ (predict_proba(params, X) - np.asarray(y, dtype=np.float64))
    grad[:-1] += l2 * params[:-1]
    return grad


def gradient_descent(X, y, initial, step_size, iterations, l2=0.0):
    """Run a fixed number of gradient descent updates.

    Output:
    - list of iterations + 1 parameter vectors: the initial one, then one per update.
    """
    params = np.array(initial, dtype=np.float64)  # a copy: the caller's array is left untouched
    path = [params.copy()]
    for _ in range(iterations):
        params = params - step_size * gradient(params, X, y, l2)
        path.append(params.copy())
    return path


def fit(X, y, l2=0.0, max_iters=100_000, tol=1e-8):
    """Fit by gradient descent until the gradient is small.

    The step is 1 / L, where L = lambda_max(A^T A) / 4 + l2 bounds the curvature of the
    objective (A is X with the column of ones appended); with that step every update
    lowers the objective.

    Output:
    - params (ndarray (Shape: (D + 1,))), and the number of updates used.
    """
    A = augment(X)
    step = 1.0 / (np.linalg.eigvalsh(A.T @ A)[-1] / 4.0 + l2)
    params = np.zeros(A.shape[1])
    for iteration in range(max_iters):
        grad = gradient(params, X, y, l2)
        if np.linalg.norm(grad) < tol:
            return params, iteration
        params = params - step * grad
    return params, max_iters
