"""Naive Bayes with binary features (for text: a term is present in a document or not).

For C classes and D terms the model has

    priors      P(y = c)            = N_c / N
    likelihoods P(x_j = 1 | y = c)  = (number of class-c documents containing term j + alpha) / (N_c + 2 alpha)

The smoothing constant alpha keeps a term that was never seen in a class from giving that
class probability zero. The features are assumed independent given the class.
"""
import numpy as np


class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        """alpha (float): smoothing; alpha = 1 adds one imaginary document with and one without each term."""
        self.alpha = alpha

    def fit(self, X, y):
        """
        Args:
        - X (ndarray (Shape: (N, D))): 1 where document n contains term j, else 0.
        - y (ndarray (Shape: (N,))): class index of each document.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C, D = len(self.classes_), X.shape[1]
        self.log_prior_ = np.empty(C)
        self.theta_ = np.empty((C, D))          # theta_[c, j] = P(x_j = 1 | y = c)
        for index, label in enumerate(self.classes_):
            members = X[y == label]
            self.log_prior_[index] = np.log(len(members) / len(y))
            self.theta_[index] = (members.sum(axis=0, dtype=np.float64) + self.alpha) / (len(members) + 2.0 * self.alpha)
        return self

    def joint_log_likelihood(self, X):
        """log P(y = c) + sum_j log P(x_j | y = c) for every document and class: a (N, C) matrix."""
        X = np.asarray(X, dtype=np.float64)
        log_present = np.log(self.theta_)
        log_absent = np.log1p(-self.theta_)
        # sum_j [x_j log theta_j + (1 - x_j) log(1 - theta_j)]
        #   = sum_j log(1 - theta_j) + x . (log theta - log(1 - theta))
        return self.log_prior_ + log_absent.sum(axis=1) + X @ (log_present - log_absent).T

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        """Posterior P(y = c | x), normalized in log space."""
        joint = self.joint_log_likelihood(X)
        joint -= joint.max(axis=1, keepdims=True)
        posterior = np.exp(joint)
        return posterior / posterior.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """Share of documents whose predicted class equals the given label."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def most_indicative_terms(self, top=10):
        """For each class, the terms whose presence raises its log-odds the most
        against the average of the other classes. Output: list of index arrays."""
        log_theta = np.log(self.theta_)
        indicative = []
        for c in range(len(self.classes_)):
            others = np.delete(log_theta, c, axis=0).mean(axis=0)
            indicative.append(np.argsort(log_theta[c] - others)[::-1][:top])
        return indicative
