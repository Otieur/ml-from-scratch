"""Least-squares regression against independent references."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from cscc11.datasets import load_admissions
from cscc11.linear_regression import add_bias, fit_least_squares, mean_squared_error, predict, r_squared


def test_admissions_data_has_the_expected_shape():
    X, y, names = load_admissions()
    assert X.shape == (400, 7) and y.shape == (400,)
    assert names == ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    assert 0.0 < y.min() and y.max() <= 1.0


def test_recovers_known_weights_exactly():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    true_w = np.array([2.0, 1.0, -3.0, 0.5, 4.0])  # bias first
    y = add_bias(X) @ true_w
    w = fit_least_squares(X, y)
    assert w.shape == (5,)
    np.testing.assert_allclose(w, true_w, atol=1e-10)
    np.testing.assert_allclose(predict(w, X), y, atol=1e-10)


def test_matches_scikit_learn_on_the_admissions_data():
    X, y, _ = load_admissions()
    w = fit_least_squares(X, y)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(w[0], reference.intercept_, rtol=1e-8)
    np.testing.assert_allclose(w[1:], reference.coef_, rtol=1e-8)
    np.testing.assert_allclose(predict(w, X), reference.predict(X), rtol=1e-10)


def test_residuals_are_orthogonal_to_every_column():
    # The normal equations say X^T (y - Xw) = 0 at the optimum.
    X, y, _ = load_admissions()
    w = fit_least_squares(X, y)
    np.testing.assert_allclose(add_bias(X).T @ (y - predict(w, X)), 0.0, atol=1e-7)


def test_duplicated_feature_still_gives_the_best_fit():
    # X^T X is singular here; the predictions must still be the least-squares ones.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = 1.0 + X @ np.array([2.0, -1.0]) + 0.1 * rng.normal(size=30)
    doubled = np.hstack((X, X[:, [0]]))
    np.testing.assert_allclose(predict(fit_least_squares(doubled, y), doubled),
                               predict(fit_least_squares(X, y), X), atol=1e-9)


def test_several_targets_are_fitted_independently():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    Y = rng.normal(size=(40, 2))
    together = fit_least_squares(X, Y)
    assert together.shape == (4, 2)
    for column in range(2):
        np.testing.assert_allclose(together[:, column], fit_least_squares(X, Y[:, column]), atol=1e-10)


def test_mse_value_and_shape_check():
    assert mean_squared_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(4.0 / 3.0)
    with pytest.raises(AssertionError):
        mean_squared_error(np.zeros(3), np.zeros((3, 1)))


def test_recorded_errors_on_the_admissions_split():
    # The numbers reported in the README, for a 70/30 split with random_state=42.
    X, y, _ = load_admissions()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    w = fit_least_squares(X_train, y_train)
    assert mean_squared_error(y_train, predict(w, X_train)) == pytest.approx(0.00378313283560819, rel=1e-9)
    test_mse = mean_squared_error(y_test, predict(w, X_test))
    assert test_mse == pytest.approx(0.004652821846466748, rel=1e-9)
    assert r_squared(y_test, predict(w, X_test)) == pytest.approx(1 - test_mse / np.var(y_test))
    # Far better than predicting the training mean for everyone.
    assert test_mse < 0.25 * mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
