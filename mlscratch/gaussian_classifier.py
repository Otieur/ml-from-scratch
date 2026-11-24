"""Gaussian class-conditional classifier with diagonal covariances.

One Gaussian per class, fitted by maximum likelihood:

    priors     P(y = c)  = N_c / N
    means      mu_c      = mean of the class-c feature vectors                       (D,)
    variances  var_c     = mean of (x - mu_c)^2 over the class-c vectors, per feature (D,)

A full D x D covariance cannot be estimated when there are more features than samples per
class (it would be singular), so the features are treated as independent given the class.

A feature that is constant within a class (for term counts: a term that never occurs in
it) has maximum-likelihood variance exactly zero. A floor is therefore added to every
variance:

    floor = var_smoothing * (largest per-feature variance over the whole training set)

For sparse counts the value of var_smoothing matters a great deal; choose it by
cross-validation.
"""
import numpy as np


class GaussianClassConditional:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """
        Args:
        - X (ndarray (Shape: (N, D))): term frequencies.
        - y (ndarray (Shape: (N,))): class index of each document.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C, D = len(self.classes_), X.shape[1]
        self.log_prior_ = np.empty(C)
        self.mean_ = np.empty((C, D))
        self.var_ = np.empty((C, D))
        for index, label in enumerate(self.classes_):
            members = X[y == label]
            self.log_prior_[index] = np.log(len(members) / len(y))
            self.mean_[index] = members.mean(axis=0)
            self.var_[index] = members.var(axis=0)      # maximum likelihood: divide by N_c
        self.floor_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ += self.floor_
        return self

    def joint_log_likelihood(self, X):
        """log P(y = c) + log N(x | mu_c, diag(var_c)) for every document and class: (N, C)."""
        X = np.asarray(X, dtype=np.float64)
        joint = np.empty((X.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]))
            # sum_j (x_j - mu_j)^2 / var_j, expanded so that no (N, D) temporary per class is squared
            inv = 1.0 / self.var_[c]
            quad = (X * X) @ inv - 2.0 * X @ (self.mean_[c] * inv) + np.sum(self.mean_[c] ** 2 * inv)
            joint[:, c] = self.log_prior_[c] + log_norm - 0.5 * quad
        return joint

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]

    def score(self, X, y):
        """Share of documents whose predicted class equals the given label."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
