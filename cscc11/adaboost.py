"""AdaBoost with decision stumps for two classes.

A stump looks at one feature: h(x) = s if x_d > threshold, otherwise -s, with s in {-1, +1}.
Round t, with weights w that sum to one and labels y in {-1, +1}:
    h_t      the stump with the smallest weighted error  e_t = sum_i w_i [h_t(x_i) != y_i]
    alpha_t  = (1/2) ln((1 - e_t) / e_t)
    w_i      <- w_i exp(-alpha_t y_i h_t(x_i)) / Z_t,    Z_t = 2 sqrt(e_t (1 - e_t))
Prediction:  sign(F(x)),  F(x) = sum_t alpha_t h_t(x)

The mean of exp(-y F(x)) over the training set equals the product of the Z_t, and it
bounds the training error from above.
"""
import numpy as np


def stump_predict(X, feature, threshold, polarity):
    """polarity where X[:, feature] > threshold, otherwise -polarity."""
    return np.where(np.asarray(X)[:, feature] > threshold, polarity, -polarity)


def best_stump(X, signs, weights, order=None):
    """The stump with the smallest weighted error.

    Args:
    - X (ndarray (Shape: (N, D))), signs (ndarray (Shape: (N,))): labels in {-1, +1}.
    - weights (ndarray (Shape: (N,))): non-negative, any total.
    - order (ndarray (Shape: (N, D))): np.argsort(X, axis=0), if already computed.

    Output:
    - feature (int), threshold (float), polarity (+1 or -1), weighted error (float).
      Thresholds lie halfway between neighbouring distinct values; -inf stands for the
      stump that predicts the same class everywhere.
    """
    X = np.asarray(X, dtype=np.float64)
    if order is None:
        order = np.argsort(X, axis=0, kind="stable")
    columns = np.arange(X.shape[1])
    values = X[order, columns]
    positive = (weights * (signs > 0))[order]                 # weight of the positive points, sorted per feature
    negative = (weights * (signs < 0))[order]
    total_positive, total_negative = positive[:, 0].sum(), negative[:, 0].sum()

    # Threshold after position k with polarity +1: rows 0..k are predicted -1, the rest +1.
    errors = np.cumsum(positive, axis=0) + (total_negative - np.cumsum(negative, axis=0))
    usable = np.vstack((values[1:] > values[:-1], np.zeros((1, X.shape[1]), dtype=bool)))
    plus = np.where(usable, errors, np.inf)                   # polarity +1
    minus = np.where(usable, (total_positive + total_negative) - errors, np.inf)   # polarity -1

    candidates = [(total_negative, 0, -np.inf, 1), (total_positive, 0, -np.inf, -1)]
    for table, polarity in ((plus, 1), (minus, -1)):
        k, d = np.unravel_index(np.argmin(table.T), table.T.shape)[::-1]
        if np.isfinite(table[k, d]):
            candidates.append((table[k, d], int(d), 0.5 * (values[k, d] + values[k + 1, d]), polarity))
    error, feature, threshold, polarity = min(candidates, key=lambda c: c[0])
    return feature, float(threshold), polarity, float(error)


class AdaBoost:
    """AdaBoost over decision stumps.

    After fit: feature_, threshold_, polarity_ and alpha_ describe the stumps in the order
    they were added; error_ holds each stump's weighted error when it was chosen.
    """

    def __init__(self, n_rounds=100):
        self.n_rounds = n_rounds

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, index = np.unique(np.asarray(y), return_inverse=True)
        assert len(self.classes_) == 2, "this AdaBoost separates two classes"
        signs = np.where(index == 1, 1.0, -1.0)
        order = np.argsort(X, axis=0, kind="stable")
        weights = np.full(len(signs), 1.0 / len(signs))
        self.feature_, self.threshold_, self.polarity_, self.alpha_, self.error_ = [], [], [], [], []
        for _ in range(self.n_rounds):
            feature, threshold, polarity, error = best_stump(X, signs, weights, order)
            if error >= 0.5:                                  # no stump is better than chance: stop
                break
            alpha = 0.5 * np.log((1.0 - max(error, 1e-12)) / max(error, 1e-12))
            for store, value in ((self.feature_, feature), (self.threshold_, threshold),
                                 (self.polarity_, polarity), (self.alpha_, alpha), (self.error_, error)):
                store.append(value)
            if error == 0.0:                                  # a perfect stump: nothing left to reweight
                break
            weights = weights * np.exp(-alpha * signs * stump_predict(X, feature, threshold, polarity))
            weights /= weights.sum()
        self.weights_ = weights
        return self

    def staged_decision_function(self, X):
        """F(x) after 1, 2, ..., T rounds: an array of shape (T, N)."""
        votes = np.array([alpha * stump_predict(X, feature, threshold, polarity)
                          for feature, threshold, polarity, alpha
                          in zip(self.feature_, self.threshold_, self.polarity_, self.alpha_)])
        return np.cumsum(votes, axis=0)

    def decision_function(self, X):
        return self.staged_decision_function(X)[-1]

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
