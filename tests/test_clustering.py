"""K-means and the Gaussian mixture model against scikit-learn."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from mlscratch.gmm import e_step, fit_gmm, gmm_cluster, log_gaussian
from mlscratch.imaging import apply_palette, indexed_png_size, load_sample, png_size, psnr
from mlscratch.kmeans import kmeans


def three_blobs(seed=0, n=300):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 10.0, 10.0]])  # distinct coordinate sums, so they sort unambiguously
    return np.vstack([c + rng.normal(size=(n, 3)) for c in centers]), centers


# ---------------------------------------------------------------- K-means

def test_kmeans_finds_well_separated_clusters():
    X, centers = three_blobs()
    centroids, labels = kmeans(X, 3, seed=0, n_init=10)
    assert centroids.shape == (3, 3) and labels.shape == (900,)
    found = centroids[np.argsort(centroids.sum(axis=1))]
    np.testing.assert_allclose(found, centers[np.argsort(centers.sum(axis=1))], atol=0.3)
    assert sorted(np.bincount(labels)) == [300, 300, 300]


def test_kmeans_matches_scikit_learn_from_the_same_start():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 3)) * [3.0, 1.0, 2.0]
    init = X[rng.choice(500, 6, replace=False)]
    centroids, labels = kmeans(X, 6, max_iters=300, init=init)
    reference = KMeans(n_clusters=6, init=init, n_init=1, algorithm="lloyd", max_iter=300, tol=0).fit(X)
    np.testing.assert_allclose(centroids, reference.cluster_centers_, atol=1e-8)
    np.testing.assert_array_equal(labels, reference.labels_)


def test_restarts_never_give_a_worse_objective_than_one_start():
    X, _ = three_blobs(seed=12)

    def objective(centroids, labels):
        return float(np.sum((X - centroids[labels]) ** 2))

    single = objective(*kmeans(X, 3, seed=0, n_init=1))
    best_of_ten = objective(*kmeans(X, 3, seed=0, n_init=10))
    assert best_of_ten <= single + 1e-9


def test_kmeans_objective_never_increases():
    X, _ = three_blobs(seed=2)
    _, _, history = kmeans(X, 5, seed=1, return_history=True)
    assert len(history) >= 2
    assert all(later <= earlier + 1e-9 for earlier, later in zip(history, history[1:]))


def test_kmeans_is_correct_on_uint8_pixels():
    # uint8 subtraction wraps around (10 - 20 = 246), so clustering image pixels as they
    # come would corrupt the distances. The result must equal the one for the same values as floats.
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(400, 3), dtype=np.uint8)
    init = pixels[:5].astype(np.float64)
    c_uint8, l_uint8 = kmeans(pixels, 5, init=init)
    c_float, l_float = kmeans(pixels.astype(np.float64), 5, init=init)
    np.testing.assert_array_equal(c_uint8, c_float)
    np.testing.assert_array_equal(l_uint8, l_float)


def test_kmeans_survives_an_empty_cluster():
    # Two identical starting centroids leave one cluster empty after the first round.
    X, _ = three_blobs(seed=3, n=50)
    init = np.vstack((X[0], X[0], X[60]))
    centroids, labels = kmeans(X, 3, init=init)
    assert np.isfinite(centroids).all()
    assert len(np.unique(labels)) == 3


def test_kmeans_is_reproducible_with_a_seed():
    X, _ = three_blobs(seed=5)
    first = kmeans(X, 4, seed=11)
    second = kmeans(X, 4, seed=11)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_palette_image_uses_rounded_centroids():
    centroids = np.array([[10.6, 20.4, 30.5], [200.2, 100.7, 0.0]])
    labels = np.array([0, 1, 1, 0, 0, 1])
    image = apply_palette(labels, centroids, 2, 3)
    assert image.shape == (2, 3, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], [11, 20, 30])   # numpy rounds 30.5 to the even 30
    np.testing.assert_array_equal(image[0, 1], [200, 101, 0])


def test_colour_quantization_end_to_end(tmp_path):
    image = load_sample()
    assert image.shape == (427, 640, 3) and image.dtype == np.uint8
    image = np.ascontiguousarray(image[::4, ::4])     # a quarter of the size keeps the test fast
    height, width, _ = image.shape
    pixels = image.reshape(-1, 3)
    results = {}
    for k in (4, 16):
        centroids, labels = kmeans(pixels, k, seed=0)
        compressed = apply_palette(labels, centroids, height, width)
        assert len(np.unique(compressed.reshape(-1, 3), axis=0)) <= k
        results[k] = (psnr(image, compressed),
                      png_size(compressed, tmp_path / f"k{k}.png"),
                      indexed_png_size(labels, centroids, height, width, tmp_path / f"k{k}_indexed.png"))
    original_size = png_size(image, tmp_path / "original.png")
    # More clusters: closer to the original, and a larger file.
    assert results[16][0] > results[4][0]
    assert results[4][1] < results[16][1] < original_size
    # Storing one index per pixel is smaller than storing three bytes per pixel.
    assert results[16][2] < results[16][1]


# ---------------------------------------------------------------- Gaussian mixture

def test_log_gaussian_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    cov = A @ A.T + np.eye(3)
    mean = rng.normal(size=3)
    X = rng.normal(size=(40, 3))
    np.testing.assert_allclose(log_gaussian(X, mean, cov), multivariate_normal(mean, cov).logpdf(X), rtol=1e-10)


def test_responsibilities_sum_to_one():
    X, centers = three_blobs(seed=1, n=40)
    covs = np.stack([np.eye(3)] * 3)
    resp, log_likelihood = e_step(X, np.full(3, 1 / 3), centers, covs)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(log_likelihood)


def test_em_never_decreases_the_log_likelihood():
    rng = np.random.default_rng(6)
    X = np.vstack((rng.normal(size=(200, 3)), 3 + 2 * rng.normal(size=(200, 3))))
    *_, history = fit_gmm(X, 4, max_iters=60, tol=0.0, seed=0, return_history=True)
    assert len(history) > 5
    assert all(later >= earlier - 1e-7 for earlier, later in zip(history, history[1:]))


def test_gmm_recovers_the_parameters_of_separated_blobs():
    X, centers = three_blobs(seed=8)
    weights, means, covs, resp = fit_gmm(X, 3, seed=0)
    order = np.argsort(means.sum(axis=1))
    np.testing.assert_allclose(means[order], centers[np.argsort(centers.sum(axis=1))], atol=0.3)
    np.testing.assert_allclose(weights, 1 / 3, atol=0.02)
    for cov in covs:
        np.testing.assert_allclose(cov, np.eye(3), atol=0.35)
    assert resp.shape == (900, 3)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gmm_matches_scikit_learn_from_the_same_start():
    rng = np.random.default_rng(9)
    X = np.vstack((rng.normal(size=(250, 3)) * [1.0, 2.0, 0.5], [4.0, 0.0, 3.0] + rng.normal(size=(250, 3))))
    means0 = X[[0, 300]]
    weights0 = np.array([0.5, 0.5])
    covs0 = np.stack([np.eye(3)] * 2)
    iterations = 25
    weights, means, covs, _ = fit_gmm(X, 2, max_iters=iterations + 1, tol=-np.inf, reg_covar=1e-6,
                                      init=(weights0, means0, covs0))
    reference = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6, tol=0.0,
                                max_iter=iterations, weights_init=weights0, means_init=means0,
                                precisions_init=np.linalg.inv(covs0)).fit(X)
    np.testing.assert_allclose(weights, reference.weights_, atol=1e-6)
    np.testing.assert_allclose(means, reference.means_, atol=1e-6)
    np.testing.assert_allclose(covs, reference.covariances_, atol=1e-6)


def test_gmm_clustering_gives_each_point_to_its_most_responsible_component():
    X, _ = three_blobs(seed=10, n=100)
    means, labels = gmm_cluster(X, 3, seed=0)
    assert means.shape == (3, 3) and labels.shape == (300,)
    assert sorted(np.bincount(labels, minlength=3)) == [100, 100, 100]
