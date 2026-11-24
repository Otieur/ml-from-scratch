"""RBF regression and the inpainting built on it."""
import numpy as np
from sklearn.linear_model import Ridge

from cscc11.inpainting import colour_mask, inpaint
from cscc11.rbf import RBFRegression


def random_problem(seed=0, n=200, grid=4):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 10, grid)
    centers = np.array([(a, b) for a in xs for b in xs])
    X = rng.uniform(0, 10, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.1 * X[:, 1] + 0.05 * rng.normal(size=n)
    return RBFRegression(centers, 2.0), X, y


# ---------------------------------------------------------------- regression

def test_design_matrix_matches_the_definition():
    model, X, _ = random_problem()
    B = model.design_matrix(X)
    assert B.shape == (len(X), len(model.centers) + 1)
    np.testing.assert_array_equal(B[:, 0], 1.0)
    for k, center in enumerate(model.centers):
        direct = np.exp(-np.sum((X - center) ** 2, axis=1) / (2 * 2.0 ** 2))
        np.testing.assert_allclose(B[:, k + 1], direct, rtol=1e-12)


def test_recovers_a_function_that_lies_in_the_model():
    model, X, _ = random_problem(seed=1)
    true_weights = np.random.default_rng(2).normal(size=len(model.centers) + 1)
    y = model.design_matrix(X) @ true_weights
    model.fit(X, y, l2=0.0)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)


def test_matches_ridge_regression_on_the_same_basis_functions():
    model, X, y = random_problem()
    for l2 in (1e-3, 0.5, 10.0):
        model.fit(X, y, l2)
        # scikit-learn's ridge with an intercept also leaves the intercept out of the penalty.
        reference = Ridge(alpha=l2, fit_intercept=True, solver="cholesky").fit(model.design_matrix(X)[:, 1:], y)
        np.testing.assert_allclose(model.weights[0], reference.intercept_, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(model.weights[1:], reference.coef_, rtol=1e-6, atol=1e-8)


def test_gradient_of_the_regularized_loss_is_zero_at_the_solution():
    model, X, y = random_problem(seed=3)
    l2 = 0.7
    model.fit(X, y, l2)
    B = model.design_matrix(X)
    penalized = model.weights.copy()
    penalized[0] = 0.0
    np.testing.assert_allclose(B.T @ (B @ model.weights - y) + l2 * penalized, 0.0, atol=1e-8)


def test_more_regularization_gives_smaller_bump_weights():
    model, X, y = random_problem(seed=5)
    norms = []
    for l2 in (0.0, 0.1, 1.0, 10.0, 100.0):
        model.fit(X, y, l2)
        norms.append(np.linalg.norm(model.weights[1:]))
    assert all(a >= b for a, b in zip(norms, norms[1:]))


def test_heavy_regularization_leaves_the_mean():
    # The bias is not penalized, so the fit falls back to the average of the targets.
    model, X, y = random_problem(seed=6)
    model.fit(X, y + 5.0, l2=1e9)
    np.testing.assert_allclose(model.predict(X), np.mean(y + 5.0), atol=1e-3)


def test_several_outputs_share_one_fit():
    model, X, y = random_problem(seed=7)
    Y = np.column_stack((y, 2 * y + 1, -y))
    together = RBFRegression(model.centers, 2.0).fit(X, Y, l2=0.3)
    assert together.predict(X).shape == (len(X), 3)
    for column in range(3):
        alone = RBFRegression(model.centers, 2.0).fit(X, Y[:, column], l2=0.3)
        np.testing.assert_allclose(together.weights[:, column], alone.weights, atol=1e-9)


# ---------------------------------------------------------------- inpainting

RED = np.array([1.0, 0.0, 0.0])


def gradient_image(size=80, slope=0.3):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.stack((0.2 + slope * xx / size, 0.3 + slope * yy / size, 0.6 - 0.2 * xx / size), axis=2)


def test_colour_mask_finds_the_painted_pixels():
    image = gradient_image()
    image[10:14, 20:30] = RED
    mask = colour_mask(image, RED)
    assert mask.sum() == 4 * 10 and mask[10:14, 20:30].all()


def test_inpainting_recovers_a_smooth_image():
    clean = gradient_image()
    corrupted = clean.copy()
    corrupted[38:42, 10:50] = RED
    mask = colour_mask(corrupted, RED)
    restored = inpaint(corrupted, mask, spacing=3, width=2, l2=0.01)
    assert np.abs(restored - clean)[mask].max() < 0.05
    # Known pixels are left alone, and the input array is not modified.
    np.testing.assert_array_equal(restored[~mask], clean[~mask])
    assert colour_mask(corrupted, RED).sum() == 4 * 40


def test_inpainting_reaches_the_image_border():
    clean = gradient_image()
    corrupted = clean.copy()
    corrupted[38:42, :] = RED           # a bar across the whole width, touching both edges
    corrupted[:, 76:80] = RED           # and one along the right edge
    corrupted[0:3, 0:10] = RED          # and a corner
    mask = colour_mask(corrupted, RED)
    restored = inpaint(corrupted, mask, spacing=3, width=2, l2=0.01)
    assert not colour_mask(restored, RED).any()
    assert np.abs(restored - clean)[mask].max() < 0.15


def test_each_prediction_lands_on_its_own_pixel():
    # A steep ramp in both directions under the mask: if the predicted values were written
    # to the masked pixels in any other order than they were predicted in, neighbouring
    # pixels would receive each other's values and the error would be several times larger.
    size = 80
    yy, xx = np.mgrid[0:size, 0:size]
    clean = np.stack((xx / size, yy / size, 0.5 * (xx + yy) / size), axis=2)
    mask = np.zeros((size, size), dtype=bool)
    mask[30:36, 12:40] = True
    corrupted = clean.copy()
    corrupted[mask] = 0.0
    error = np.abs(inpaint(corrupted, mask, spacing=4, width=4, l2=1e-3) - clean)[mask]
    assert error.mean() < 0.01 and error.max() < 0.03


def test_a_fully_masked_patch_borrows_from_its_surroundings():
    clean = gradient_image(size=100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True           # larger than a 25-pixel patch
    corrupted = clean.copy()
    corrupted[mask] = 0.0
    restored = inpaint(corrupted, mask, spacing=3, width=3, l2=0.01, patch_size=25)
    assert np.isfinite(restored).all()
    assert np.abs(restored - clean)[mask].mean() < 0.1
