"""Data splitting, feature scaling and cross-validation.

Everything that is learned from data (the feature scaling, the PCA directions, the stored
neighbours) is fitted on the training part of each fold only, so the held-out fold never
influences the model that is scored on it.
"""
import numpy as np

from cscc11.knn import neighbour_order, vote
from cscc11.pca import PCA


def stratified_split(y, test_size, seed):
    """Random split that keeps the class proportions.

    Output:
    - train_index, test_index (ndarrays of row indices, each sorted).
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    test = []
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        test.extend(members[:int(round(test_size * len(members)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


def k_fold_indices(y, n_folds, seed):
    """Split the rows into n_folds disjoint folds with similar class proportions.

    Output:
    - list of n_folds index arrays; together they contain every row exactly once.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        for position, row in enumerate(members):
            folds[position % n_folds].append(row)
    return [np.sort(np.array(fold)) for fold in folds]


class Standardizer:
    """Zero mean and unit variance per feature, with the statistics of the data it was fitted on."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0   # a constant feature stays constant instead of dividing by zero
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


def prepare(X_train, X_other, standardize=True, pca_variance=None):
    """Fit the preprocessing on X_train and apply it to both sets.

    Output:
    - Z_train, Z_other, and the number of features after preprocessing.
    """
    if standardize:
        scaler = Standardizer().fit(X_train)
        X_train, X_other = scaler.transform(X_train), scaler.transform(X_other)
    if pca_variance is not None:
        pca = PCA(variance_threshold=pca_variance).fit(X_train)
        X_train, X_other = pca.transform(X_train), pca.transform(X_other)
    return X_train, X_other, X_train.shape[1]


def cross_validate_knn(X, y, k_values, n_folds=5, seed=0, standardize=True, pca_variance=None):
    """Mean validation accuracy of k-NN for every candidate k.

    Args:
    - X (ndarray (Shape: (N, D))), y (ndarray (Shape: (N,))): the training set only.
    - k_values (list of int): candidates for k.
    - n_folds (int): number of folds.

    Output:
    - mean_accuracy (dict k -> float), fold_accuracy (dict k -> list of n_folds floats).
    """
    X = np.asarray(X, dtype=np.float64)
    classes, y_index = np.unique(np.asarray(y), return_inverse=True)
    folds = k_fold_indices(y_index, n_folds, seed)
    fold_accuracy = {k: [] for k in k_values}
    for held_out in folds:
        rest = np.setdiff1d(np.arange(len(y_index)), held_out)
        Z_rest, Z_held, _ = prepare(X[rest], X[held_out], standardize, pca_variance)
        # Sort the neighbours once per fold; every k reads a prefix of the same ordering.
        order = neighbour_order(Z_held, Z_rest)
        for k in k_values:
            predicted = vote(y_index[rest][order[:, :k]], len(classes))
            fold_accuracy[k].append(float(np.mean(predicted == y_index[held_out])))
    mean_accuracy = {k: float(np.mean(scores)) for k, scores in fold_accuracy.items()}
    return mean_accuracy, fold_accuracy


def best_k(mean_accuracy):
    """The k with the highest mean accuracy; the smallest such k when several tie."""
    top = max(mean_accuracy.values())
    return min(k for k, accuracy in mean_accuracy.items() if accuracy >= top - 1e-12)
