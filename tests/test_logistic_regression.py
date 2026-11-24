"""Logistic regression by gradient descent, and the majority-vote error rate."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlscratch import logistic_regression as lr
from mlscratch.voting import ensemble_error, simulate_ensemble_error

# A seven-point problem small enough to work through by hand: two features, binary label.
TOY_X = np.array([[4, 4], [6, 4], [6, 5], [6, 8], [6, 10], [8, 8], [8, 10]], dtype=np.float64)
TOY_Y = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.float64)
TOY_START = np.array([0.3, -0.2, 0.7])     # two weights, then the bias
TOY_STEP = 0.01


def overlapping_classes(seed=3, n=200):
    """Classes that overlap, so the maximum-likelihood solution is finite."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X @ np.array([1.5, -1.0]) + 0.3 + rng.logistic(size=n) > 0).astype(float)
    return X, y


# ---------------------------------------------------------------- logistic regression

def test_sigmoid_is_stable_and_symmetric():
    values = lr.sigmoid(np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0]))
    np.testing.assert_allclose(values, [0.0, 1 / (1 + np.e ** 2), 0.5, 1 / (1 + np.e ** -2), 1.0], atol=1e-12)
    np.testing.assert_allclose(lr.sigmoid(np.array([-3.0])) + lr.sigmoid(np.array([3.0])), 1.0)


def test_probabilities_by_hand():
    # Scores 0.3 * x1 - 0.2 * x2 + 0.7, worked out for each of the seven points.
    scores = np.array([1.1, 1.7, 1.5, 0.9, 0.5, 1.5, 1.1])
    np.testing.assert_allclose(lr.predict_proba(TOY_START, TOY_X), 1 / (1 + np.exp(-scores)), atol=1e-12)
    # Every point is predicted positive at the start, and 4 of the 7 are positive.
    assert lr.accuracy(TOY_START, TOY_X, TOY_Y) == pytest.approx(4 / 7)


@pytest.mark.parametrize("l2", [0.0, 0.7])
def test_gradient_matches_finite_differences(l2):
    rng = np.random.default_rng(0)
    for _ in range(5):
        params = rng.normal(size=3)
        numeric = np.empty(3)
        for j in range(3):
            step = np.zeros(3)
            step[j] = 1e-6
            numeric[j] = (lr.negative_log_likelihood(params + step, TOY_X, TOY_Y, l2)
                          - lr.negative_log_likelihood(params - step, TOY_X, TOY_Y, l2)) / 2e-6
        np.testing.assert_allclose(lr.gradient(params, TOY_X, TOY_Y, l2), numeric, rtol=1e-6, atol=1e-7)


def test_first_update_by_hand():
    # Residuals p - y at the start, then the three sums that make up the gradient.
    p = 1 / (1 + np.exp(-np.array([1.1, 1.7, 1.5, 0.9, 0.5, 1.5, 1.1])))
    residual = p - TOY_Y
    by_hand = np.array([residual @ TOY_X[:, 0], residual @ TOY_X[:, 1], residual.sum()])
    np.testing.assert_allclose(by_hand, [9.5233, 15.4274, 1.3147], atol=2e-3)
    first = lr.gradient_descent(TOY_X, TOY_Y, TOY_START, TOY_STEP, 1)[1]
    np.testing.assert_allclose(first, TOY_START - TOY_STEP * by_hand, atol=1e-12)
    np.testing.assert_allclose(first, [0.2048, -0.3543, 0.6869], atol=1e-4)


def test_updates_lower_the_objective_and_keep_the_start_untouched():
    start = TOY_START.copy()
    path = lr.gradient_descent(TOY_X, TOY_Y, TOY_START, TOY_STEP, 3)
    assert len(path) == 4
    losses = [lr.negative_log_likelihood(params, TOY_X, TOY_Y) for params in path]
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    np.testing.assert_array_equal(TOY_START, start)
    # Two updates are enough to classify all seven points correctly.
    assert [lr.accuracy(params, TOY_X, TOY_Y) for params in path] == pytest.approx([4 / 7, 6 / 7, 1.0, 1.0])


def test_fit_reaches_the_maximum_likelihood_solution():
    X, y = overlapping_classes()
    params, iterations = lr.fit(X, y)
    assert iterations < 100_000
    reference = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(params[:2], reference.coef_[0], atol=1e-5)
    np.testing.assert_allclose(params[2], reference.intercept_[0], atol=1e-5)
    np.testing.assert_allclose(lr.gradient(params, X, y), 0.0, atol=1e-7)


def test_fit_with_a_penalty_matches_scikit_learn():
    X, y = overlapping_classes(seed=4)
    l2 = 2.0
    params, _ = lr.fit(X, y, l2=l2)
    # scikit-learn minimizes ||w||^2 / 2 + C * (sum of losses): the same minimizer for C = 1 / l2.
    reference = LogisticRegression(C=1.0 / l2, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(params[:2], reference.coef_[0], atol=1e-5)
    np.testing.assert_allclose(params[2], reference.intercept_[0], atol=1e-5)


def test_every_step_of_size_one_over_the_curvature_bound_lowers_the_objective():
    X, y = overlapping_classes(seed=5)
    A = lr.augment(X)
    step = 1.0 / (np.linalg.eigvalsh(A.T @ A)[-1] / 4.0)
    path = lr.gradient_descent(X, y, np.zeros(3), step, 50)
    losses = [lr.negative_log_likelihood(params, X, y) for params in path]
    assert all(later <= earlier for earlier, later in zip(losses, losses[1:]))


# ---------------------------------------------------------------- majority vote

def test_one_classifier_has_its_own_error_rate():
    assert ensemble_error(0.3, 1) == pytest.approx(0.3)


def test_three_classifiers_by_hand():
    # Wrong when 2 or 3 of 3 are wrong: 3 e^2 (1 - e) + e^3.
    e = 0.2
    assert ensemble_error(e, 3) == pytest.approx(3 * e ** 2 * (1 - e) + e ** 3)


def test_ensemble_error_matches_simulation():
    for epsilon, n in ((0.3, 5), (0.45, 11), (0.6, 7)):
        assert ensemble_error(epsilon, n) == pytest.approx(simulate_ensemble_error(epsilon, n), abs=4e-3)


def test_voting_helps_only_when_the_classifiers_beat_chance():
    assert ensemble_error(0.3, 21) < ensemble_error(0.3, 5) < 0.3
    assert ensemble_error(0.5, 21) == pytest.approx(0.5)
    assert ensemble_error(0.7, 21) > ensemble_error(0.7, 5) > 0.7


def test_an_even_number_of_classifiers_is_rejected():
    with pytest.raises(AssertionError):
        ensemble_error(0.3, 4)
