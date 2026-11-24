"""k-nearest-neighbours classifier."""
import numpy as np


def squared_distances(A, B):
    """Squared Euclidean distance between every row of A (M, D) and every row of B (N, D): (M, N)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d2 = (np.sum(A * A, axis=1)[:, np.newaxis]
          - 2.0 * (A @ B.T)
          + np.sum(B * B, axis=1)[np.newaxis, :])
    return np.maximum(d2, 0.0)  # rounding can leave tiny negative values


def neighbour_order(X_query, X_train):
    """For every query point, the indices of the training points from nearest to farthest: (M, N).

    The sort is stable, so points at exactly the same distance keep their training order.
    """
    return np.argsort(squared_distances(X_query, X_train), axis=1, kind="stable")


def nearest_neighbours(X_query, X_train, k):
    """For every query point, the indices of its k nearest training points, in no particular order: (M, k).

    Only the k smallest distances are needed for a vote, and finding them with a partial
    sort costs time proportional to N instead of N log N for a full sort.
    """
    d2 = squared_distances(X_query, X_train)
    if k >= d2.shape[1]:
        return np.tile(np.arange(d2.shape[1]), (d2.shape[0], 1))
    return np.argpartition(d2, k - 1, axis=1)[:, :k]


def vote(neighbour_labels, n_classes):
    """Majority vote over the label of each query's neighbours.

    Args:
    - neighbour_labels (ndarray (Shape: (M, k))): integer class indices in 0 .. n_classes - 1.

    Output:
    - ndarray (Shape: (M,)): the most frequent class per row; a tie goes to the smaller index.
    """
    counts = np.zeros((neighbour_labels.shape[0], n_classes), dtype=np.int64)
    for c in range(n_classes):
        counts[:, c] = np.sum(neighbour_labels == c, axis=1)
    return np.argmax(counts, axis=1)


class KNNClassifier:
    def __init__(self, k=3):
        assert k >= 1, f"k must be at least 1. Got: {k}"
        self.k = k

    def fit(self, X, y):
        """k-NN has no training step: the classifier stores the data."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.classes_, self.y_index = np.unique(np.asarray(y), return_inverse=True)
        assert self.k <= len(self.X_train), f"k = {self.k} exceeds the {len(self.X_train)} training points."
        return self

    def predict(self, X):
        """Label each query point with the majority class among its k nearest training points."""
        nearest = nearest_neighbours(X, self.X_train, self.k)
        return self.classes_[vote(self.y_index[nearest], len(self.classes_))]

    def score(self, X, y):
        """Share of the points whose predicted class equals the given label."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
