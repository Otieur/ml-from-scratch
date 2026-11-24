"""The support vector machine against hand-worked cases, its optimality conditions and scikit-learn."""
import numpy as np
import pytest
from sklearn.svm import SVC

from mlscratch.svm import SVM, OneVsRestSVM, linear_kernel, rbf_kernel


def two_blobs(seed=0, n=120, gap=1.0):
    """Two overlapping Gaussian classes in the plane."""
    rng = np.random.default_rng(seed)
    X = np.vstack((rng.normal(-gap, 1.0, size=(n // 2, 2)), rng.normal(gap, 1.0, size=(n // 2, 2))))
    y = np.repeat([0, 1], n // 2)
    return X, y


def rings(seed=1, n=160):
    """An inner disc and a ring around it: not separable by a line."""
    rng = np.random.default_rng(seed)
    radius = np.concatenate((rng.uniform(0, 1, n // 2), rng.uniform(1.5, 2.5, n // 2)))
    angle = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    return X, np.repeat([0, 1], n // 2)


def test_kernels_match_their_definitions():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    for i in range(5):
        for j in range(4):
            assert linear_kernel(A, B)[i, j] == pytest.approx(A[i] @ B[j])
            assert rbf_kernel(A, B, 0.7)[i, j] == pytest.approx(np.exp(-0.7 * np.sum((A[i] - B[j]) ** 2)))


def test_two_points_by_hand():
    # x = -1 and x = +1 with a linear kernel: the margin is the whole gap, so w = 1, b = 0,
    # and from w = sum a_i y_i x_i with a_1 = a_2 both multipliers are 1/2.
    model = SVM(C=10.0, kernel="linear", tol=1e-10).fit(np.array([[-1.0], [1.0]]), np.array([0, 1]))
    np.testing.assert_allclose(model.alpha_, [0.5, 0.5], atol=1e-9)
    assert model.intercept_ == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(model.decision_function(np.array([[-1.0], [0.0], [3.0]])), [-1.0, 0.0, 3.0], atol=1e-9)


def test_three_points_by_hand():
    # (0,0) and (1,0) are negative, (3,0) is positive. Only (1,0) and (3,0) touch the margin:
    # the boundary is x = 2, w = (1, 0), b = -2, and both active multipliers equal 1/2.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    model = SVM(C=100.0, kernel="linear", tol=1e-10).fit(X, np.array([0, 0, 1]))
    np.testing.assert_allclose(model.alpha_, [0.0, 0.5, 0.5], atol=1e-8)
    assert model.intercept_ == pytest.approx(-2.0, abs=1e-8)
    assert model.support_.tolist() == [1, 2]


@pytest.mark.parametrize("kernel, C, gamma", [("linear", 1.0, 1.0), ("rbf", 1.0, 0.5), ("rbf", 10.0, 2.0)])
def test_optimality_conditions_hold_at_the_solution(kernel, C, gamma):
    X, y = two_blobs() if kernel == "linear" else rings()
    model = SVM(C=C, kernel=kernel, gamma=gamma, tol=1e-6).fit(X, y)
    signs = np.where(y == 1, 1.0, -1.0)
    margins = signs * model.decision_function(X)
    alpha = model.alpha_
    assert alpha.min() >= 0.0 and alpha.max() <= C
    assert alpha @ signs == pytest.approx(0.0, abs=1e-9)
    assert np.all(margins[alpha == 0] >= 1 - 1e-4)                       # outside the margin
    np.testing.assert_allclose(margins[(alpha > 0) & (alpha < C)], 1.0, atol=1e-4)   # on it
    assert np.all(margins[alpha == C] <= 1 + 1e-4)                       # inside or misclassified


@pytest.mark.parametrize("kernel, C, gamma", [("linear", 0.5, 1.0), ("rbf", 1.0, 0.5), ("rbf", 20.0, 1.5)])
def test_matches_scikit_learn(kernel, C, gamma):
    X, y = two_blobs(seed=2) if kernel == "linear" else rings(seed=3)
    model = SVM(C=C, kernel=kernel, gamma=gamma, tol=1e-7).fit(X, y)
    reference = SVC(C=C, kernel=kernel, gamma=gamma, tol=1e-7).fit(X, y)
    probe = np.random.default_rng(9).uniform(-3, 3, size=(200, 2))
    np.testing.assert_allclose(model.decision_function(probe), reference.decision_function(probe), atol=1e-4)
    assert model.intercept_ == pytest.approx(reference.intercept_[0], abs=1e-4)
    assert model.support_.tolist() == sorted(reference.support_.tolist())
    np.testing.assert_allclose(model.dual_coef_, reference.dual_coef_[0][np.argsort(reference.support_)], atol=1e-4)
    # The dual objective of scikit-learn's solution, from its own multipliers.
    K = linear_kernel(X, X) if kernel == "linear" else rbf_kernel(X, X, gamma)
    coefficients = np.zeros(len(y))
    coefficients[reference.support_] = reference.dual_coef_[0]
    assert model.dual_objective_ == pytest.approx(
        0.5 * coefficients @ K @ coefficients - np.abs(coefficients).sum(), abs=1e-5)


def test_rbf_kernel_separates_what_a_line_cannot():
    X, y = rings(seed=4, n=300)
    assert SVM(C=1.0, kernel="linear").fit(X, y).score(X, y) < 0.75
    assert SVM(C=1.0, kernel="rbf", gamma=1.0).fit(X, y).score(X, y) == 1.0


def test_a_larger_penalty_leaves_fewer_margin_violations():
    X, y = two_blobs(seed=5, gap=0.8)
    signs = np.where(y == 1, 1.0, -1.0)
    slack = []
    for C in (0.01, 0.1, 1.0, 10.0):
        model = SVM(C=C, kernel="linear", tol=1e-6).fit(X, y)
        slack.append(np.maximum(0.0, 1 - signs * model.decision_function(X)).sum())
    assert all(later <= earlier + 1e-6 for earlier, later in zip(slack, slack[1:]))


def test_labels_keep_their_original_values():
    X, y = two_blobs(seed=6)
    names = np.where(y == 1, "spam", "ham")
    model = SVM(kernel="linear").fit(X, names)
    assert set(model.predict(X)) == {"ham", "spam"}
    assert model.score(X, names) > 0.85


def test_one_vs_rest_matches_scikit_learn_on_three_classes():
    from sklearn.multiclass import OneVsRestClassifier

    rng = np.random.default_rng(7)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    X = np.vstack([rng.normal(center, 0.9, size=(50, 2)) for center in centers])
    y = np.repeat([0, 1, 2], 50)
    model = OneVsRestSVM(C=1.0, kernel="rbf", gamma=0.5, tol=1e-6).fit(X, y)
    reference = OneVsRestClassifier(SVC(C=1.0, kernel="rbf", gamma=0.5, tol=1e-6)).fit(X, y)
    probe = rng.uniform(-2, 5, size=(300, 2))
    np.testing.assert_allclose(model.decision_function(probe), reference.decision_function(probe), atol=1e-3)
    np.testing.assert_array_equal(model.predict(probe), reference.predict(probe))
