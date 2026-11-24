"""K-means clustering (Lloyd's algorithm).

Alternate two steps until the labels stop changing: give every point the label of its nearest
centroid, then move every centroid to the mean of its points. Each step can only lower the
within-cluster sum of squares.
"""
import numpy as np


def _squared_distances(X, centroids):
    """Squared Euclidean distance from every row of X (N, D) to every centroid (K, D): a (N, K) matrix."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 avoids building a (N, K, D) array.
    d2 = (np.sum(X * X, axis=1, keepdims=True)
          - 2.0 * (X @ centroids.T)
          + np.sum(centroids * centroids, axis=1)[np.newaxis, :])
    return np.maximum(d2, 0.0)  # rounding can leave tiny negative values


def _lloyd(X, centroids, max_iters):
    """Run Lloyd's algorithm from the given centroids. X must already be float64."""
    N, n_clusters = X.shape[0], centroids.shape[0]
    labels = np.full(N, -1)
    history = []
    for _ in range(max_iters):
        # Calculate the distance from each point to each centroid
        d2 = _squared_distances(X, centroids)

        # Assign each point to the nearest cluster
        new_labels = np.argmin(d2, axis=1)
        nearest = d2[np.arange(N), new_labels]
        # Within-cluster sum of squares for these labels; it never increases.
        history.append(float(nearest.sum()))

        # Recompute each centroid as the mean of its points
        counts = np.bincount(new_labels, minlength=n_clusters)
        new_centroids = np.empty_like(centroids)
        for dim in range(X.shape[1]):
            new_centroids[:, dim] = np.bincount(new_labels, weights=X[:, dim], minlength=n_clusters)
        filled = counts > 0
        new_centroids[filled] /= counts[filled, np.newaxis]

        # A cluster can end up empty (for instance when two starting centroids are the same
        # colour). Its mean is undefined, so restart it at the point that is currently
        # farthest from its own centroid.
        for k in np.flatnonzero(~filled):
            far = int(np.argmax(nearest))
            new_centroids[k] = X[far]
            nearest[far] = 0.0

        # Check for convergence: no label changed and no cluster was restarted
        converged = np.array_equal(new_labels, labels) and bool(filled.all())
        centroids, labels = new_centroids, new_labels
        if converged:
            break

    # The labels returned must be the ones that match the centroids returned.
    d2 = _squared_distances(X, centroids)
    labels = np.argmin(d2, axis=1)
    return centroids, labels, history, float(d2[np.arange(N), labels].sum())


def kmeans(X, n_clusters, max_iters=100, *, seed=None, init=None, n_init=1, return_history=False):
    """K-means clustering with Lloyd's algorithm.

    Args:
    - X (ndarray (Shape: (N, D))): The N points to cluster (for an image, D = 3 and one row per pixel).
    - n_clusters (int): The number of clusters K.
    - max_iters (int): The largest number of assign/update rounds.
    - seed (int or None): Seed for the random choice of the starting centroids.
    - init (ndarray (Shape: (K, D)) or None): Starting centroids to use instead of a random choice.
    - n_init (int): Number of random starts. The run with the smallest within-cluster sum of
                    squares is returned, because one random start can stop in a poor local optimum.
    - return_history (bool): Also return the within-cluster sum of squares after each round of labelling.

    Output:
    - centroids (ndarray (Shape: (K, D))): The cluster centers.
    - labels (ndarray (Shape: (N,))): The index of the cluster each point belongs to.
    - history (list of float): Only when return_history is True.
    """
    # Image pixels arrive as uint8. Subtracting uint8 arrays wraps around (10 - 20 = 246),
    # which corrupts the distances, so all the arithmetic is done in float64.
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    assert 1 <= n_clusters <= N, f"n_clusters must be between 1 and {N}. Got: {n_clusters}"
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(1 if init is not None else max(1, n_init)):
        # Initialize cluster centroids randomly
        if init is None:
            start = X[rng.choice(N, n_clusters, replace=False)].copy()
        else:
            start = np.array(init, dtype=np.float64)
            assert start.shape == (n_clusters, X.shape[1])
        run = _lloyd(X, start, max_iters)
        if best is None or run[3] < best[3]:
            best = run

    centroids, labels, history, _ = best
    if return_history:
        return centroids, labels, history
    return centroids, labels
