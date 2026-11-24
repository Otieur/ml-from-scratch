"""Gaussian mixture model fitted by expectation-maximization.

Model, for a point x in R^D and K components:

    p(x) = sum_k pi_k N(x | mu_k, Sigma_k)

with mixing weights pi (K,), means mu (K, D) and full covariances Sigma (K, D, D).
"""
import numpy as np

from cscc11.kmeans import kmeans


def log_gaussian(X, mean, cov):
    """Log density of N(mean, cov) at every row of X.

    Args:
    - X (ndarray (Shape: (N, D))), mean (ndarray (Shape: (D,))), cov (ndarray (Shape: (D, D))).

    Output:
    - ndarray (Shape: (N,)): log N(x_n | mean, cov).
    """
    D = X.shape[1]
    # cov = L L^T. Solving L z = (x - mean) gives the Mahalanobis distance as ||z||^2
    # and log|cov| as twice the sum of the logs of the diagonal of L.
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    mahalanobis = np.sum(z * z, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2.0 * np.pi) + log_det + mahalanobis)


def e_step(X, weights, means, covs):
    """Responsibilities of every component for every point.

    Output:
    - resp (ndarray (Shape: (N, K))): resp[n, k] = pi_k N(x_n | mu_k, Sigma_k) / sum_j pi_j N(x_n | mu_j, Sigma_j).
      Every row sums to 1.
    - log_likelihood (float): sum_n log p(x_n).
    """
    K = len(weights)
    log_joint = np.empty((X.shape[0], K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + log_gaussian(X, means[k], covs[k])
    # Log-sum-exp over the components, computed without underflow.
    top = log_joint.max(axis=1, keepdims=True)
    log_px = top + np.log(np.sum(np.exp(log_joint - top), axis=1, keepdims=True))
    return np.exp(log_joint - log_px), float(log_px.sum())


def m_step(X, resp, reg_covar):
    """Maximum-likelihood update of the parameters given the responsibilities.

    Output: weights (K,), means (K, D), covs (K, D, D).
    """
    N, D = X.shape
    Nk = resp.sum(axis=0)                                # effective number of points per component
    Nk = np.maximum(Nk, 10 * np.finfo(np.float64).eps)   # a component with no points must not divide by zero
    weights = Nk / N
    means = (resp.T @ X) / Nk[:, np.newaxis]
    covs = np.empty((len(Nk), D, D))
    for k in range(len(Nk)):
        diff = X - means[k]
        covs[k] = (resp[:, k, np.newaxis] * diff).T @ diff / Nk[k]
        # A small ridge keeps the covariance positive definite when a component
        # collapses onto a few identical pixels.
        covs[k] += reg_covar * np.eye(D)
    return weights, means, covs


def fit_gmm(X, n_components, max_iters=100, tol=1e-3, reg_covar=1e-6, *, seed=None, init=None, return_history=False):
    """Fit a Gaussian mixture by expectation-maximization.

    Args:
    - X (ndarray (Shape: (N, D))): The data.
    - n_components (int): The number of Gaussians K.
    - max_iters (int): The largest number of EM rounds.
    - tol (float): Stop when the average log-likelihood per point improves by less than this.
    - reg_covar (float): Value added to the diagonal of every covariance.
    - seed (int or None): Seed for the K-means run that provides the starting point.
    - init (tuple or None): (weights, means, covs) to start from instead.
    - return_history (bool): Also return the log-likelihood before each update.

    Output:
    - weights (K,), means (K, D), covs (K, D, D), resp (N, K), and the history when asked.
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    if init is None:
        # Start from a K-means partition: its centroids, the share of points in each
        # cluster and the scatter of each cluster.
        centroids, labels = kmeans(X, n_components, max_iters=10, seed=seed)
        hard = np.zeros((N, n_components))
        hard[np.arange(N), labels] = 1.0
        weights, means, covs = m_step(X, hard, reg_covar)
    else:
        weights, means, covs = (np.array(part, dtype=np.float64) for part in init)

    history = []
    previous = -np.inf
    for _ in range(max_iters):
        resp, log_likelihood = e_step(X, weights, means, covs)
        history.append(log_likelihood)
        if (log_likelihood - previous) / N < tol:
            break
        previous = log_likelihood
        weights, means, covs = m_step(X, resp, reg_covar)

    resp, _ = e_step(X, weights, means, covs)
    if return_history:
        return weights, means, covs, resp, history
    return weights, means, covs, resp


def gmm_cluster(X, n_components, *, seed=0, max_iters=100, tol=1e-3, reg_covar=1e-2):
    """Hard clustering with a mixture model.

    Each point is assigned to the component with the largest responsibility.

    Output:
    - means (ndarray (Shape: (K, D))): the mean of each component.
    - labels (ndarray (Shape: (N,))): the component index of each point.
    """
    _, means, _, resp = fit_gmm(X, n_components, max_iters=max_iters, tol=tol, reg_covar=reg_covar, seed=seed)
    return means, np.argmax(resp, axis=1)
