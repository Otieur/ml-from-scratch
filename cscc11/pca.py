"""Principal component analysis.

Steps, for training data X (N, D):
1. mean (D,): the average of every feature; subtract it from the data.
2. covariance (D, D): C = Xc^T Xc / (N - 1) for the centered data Xc.
3. eigen-decomposition of C: the eigenvalues are the variances along the eigenvectors.
4. sort the eigenvalues from largest to smallest, with their eigenvectors.
5. keep the first k: either a given k, or the smallest k whose eigenvalues add up to
   at least a given share of the total variance.
6. project: Z = (X - mean) W, with W (D, k) holding the kept eigenvectors as columns.
"""
import numpy as np


class PCA:
    def __init__(self, n_components=None, variance_threshold=None):
        """Give exactly one of:
        - n_components (int): the number of directions to keep;
        - variance_threshold (float in (0, 1]): keep the fewest directions that explain
          at least this share of the variance.
        """
        assert (n_components is None) != (variance_threshold is None), \
            "Give either n_components or variance_threshold."
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        self.covariance_ = centered.T @ centered / (N - 1)

        # eigh is the routine for symmetric matrices: real eigenvalues, orthonormal eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)   # clear tiny negative rounding errors
        eigenvectors = eigenvectors[:, order]
        # An eigenvector is defined up to its sign. Make the largest entry of each positive,
        # so that the same data always gives the same projection.
        signs = np.sign(eigenvectors[np.argmax(np.abs(eigenvectors), axis=0), np.arange(D)])
        signs[signs == 0] = 1.0
        eigenvectors = eigenvectors * signs

        self.all_explained_variance_ratio_ = eigenvalues / eigenvalues.sum()
        if self.n_components is not None:
            k = int(self.n_components)
        else:
            cumulative = np.cumsum(self.all_explained_variance_ratio_)
            # First position where the cumulative share reaches the threshold (small slack for rounding).
            k = int(np.searchsorted(cumulative, self.variance_threshold - 1e-12) + 1)
        assert 1 <= k <= D, f"The number of components must be between 1 and {D}. Got: {k}"

        self.n_components_ = k
        self.components_ = eigenvectors[:, :k].T             # (k, D), one direction per row
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = self.all_explained_variance_ratio_[:k]
        return self

    def transform(self, X):
        """Coordinates of X (M, D) in the kept directions: (M, k)."""
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Back to the original D features; exact only when all directions were kept."""
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_
