"""Naive Bayes and the Gaussian classifier against scikit-learn and against hand computation."""
import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from cscc11.gaussian_classifier import GaussianClassConditional
from cscc11.naive_bayes import BernoulliNaiveBayes


def count_data(seed=0, n=300, d=40, classes=4):
    """Sparse non-negative counts whose rates depend on the class, like term frequencies."""
    rng = np.random.default_rng(seed)
    rates = rng.gamma(0.3, 1.0, size=(classes, d))
    y = rng.integers(0, classes, size=n)
    return rng.poisson(rates[y]).astype(np.float64), y


# ---------------------------------------------------------------- Bernoulli Naive Bayes

def test_naive_bayes_parameters_by_hand():
    # Five documents, three terms.
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
    y = np.array([0, 0, 0, 1, 1])
    model = BernoulliNaiveBayes().fit(X, y)
    np.testing.assert_allclose(np.exp(model.log_prior_), [3 / 5, 2 / 5])
    # Class 0 has 3 documents: counts (2, 1, 2) -> (2 + 1) / (3 + 2), ...
    np.testing.assert_allclose(model.theta_[0], [3 / 5, 2 / 5, 3 / 5])
    # Class 1 has 2 documents: counts (0, 2, 1) -> (0 + 1) / (2 + 2), ...
    np.testing.assert_allclose(model.theta_[1], [1 / 4, 3 / 4, 2 / 4])

    # Posterior of the document (1, 0, 0), multiplied out directly.
    class0 = 3 / 5 * (3 / 5) * (1 - 2 / 5) * (1 - 3 / 5)
    class1 = 2 / 5 * (1 / 4) * (1 - 3 / 4) * (1 - 2 / 4)
    np.testing.assert_allclose(model.predict_proba(np.array([[1, 0, 0]]))[0],
                               [class0 / (class0 + class1), class1 / (class0 + class1)])
    assert model.predict(np.array([[1, 0, 0]]))[0] == 0


def test_unseen_term_does_not_zero_out_a_class():
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y = np.array([0, 0, 1, 1])
    model = BernoulliNaiveBayes().fit(X, y)
    assert np.all(model.theta_ > 0) and np.all(model.theta_ < 1)
    assert np.isfinite(model.joint_log_likelihood(np.array([[1, 1]]))).all()


def test_naive_bayes_matches_scikit_learn():
    counts, y = count_data()
    X = (counts > 0).astype(np.float64)
    train, test = slice(0, 200), slice(200, None)
    model = BernoulliNaiveBayes().fit(X[train], y[train])
    reference = BernoulliNB(alpha=1.0, binarize=None).fit(X[train], y[train])
    np.testing.assert_allclose(model.log_prior_, reference.class_log_prior_)
    np.testing.assert_allclose(np.log(model.theta_), reference.feature_log_prob_)
    np.testing.assert_allclose(model.joint_log_likelihood(X[test]),
                               reference.predict_joint_log_proba(X[test]), rtol=1e-10)
    np.testing.assert_array_equal(model.predict(X[test]), reference.predict(X[test]))
    assert model.score(X[test], y[test]) == pytest.approx(reference.score(X[test], y[test]))


def test_most_indicative_terms_point_at_the_right_class():
    # Term j is common in class j and rare elsewhere.
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(3), 100)
    X = (rng.random((300, 6)) < 0.05).astype(int)
    for c in range(3):
        X[y == c, c] = (rng.random(100) < 0.9).astype(int)
    model = BernoulliNaiveBayes().fit(X, y)
    assert [int(terms[0]) for terms in model.most_indicative_terms(top=1)] == [0, 1, 2]


# ---------------------------------------------------------------- Gaussian class conditionals

def test_gaussian_parameters_are_the_maximum_likelihood_estimates():
    X, y = count_data(seed=2)
    model = GaussianClassConditional(var_smoothing=0.0).fit(X, y)
    for index, label in enumerate(model.classes_):
        members = X[y == label]
        np.testing.assert_allclose(model.mean_[index], members.mean(axis=0))
        np.testing.assert_allclose(model.var_[index], ((members - members.mean(axis=0)) ** 2).mean(axis=0))
        assert np.exp(model.log_prior_[index]) == pytest.approx(len(members) / len(y))


def test_gaussian_log_likelihood_matches_the_density_formula():
    X, y = count_data(seed=3, n=120, d=5, classes=2)
    model = GaussianClassConditional(var_smoothing=1e-3).fit(X, y)
    x = X[:7]
    for c in range(2):
        direct = (model.log_prior_[c]
                  - 0.5 * np.sum(np.log(2 * np.pi * model.var_[c]))
                  - 0.5 * np.sum((x - model.mean_[c]) ** 2 / model.var_[c], axis=1))
        np.testing.assert_allclose(model.joint_log_likelihood(x)[:, c], direct, rtol=1e-10)


@pytest.mark.parametrize("var_smoothing", [1e-9, 1e-3, 1e-1])
def test_gaussian_classifier_matches_scikit_learn(var_smoothing):
    X, y = count_data(seed=4)
    train, test = slice(0, 200), slice(200, None)
    model = GaussianClassConditional(var_smoothing=var_smoothing).fit(X[train], y[train])
    reference = GaussianNB(var_smoothing=var_smoothing).fit(X[train], y[train])
    np.testing.assert_allclose(model.mean_, reference.theta_)
    np.testing.assert_allclose(model.var_, reference.var_)
    np.testing.assert_array_equal(model.predict(X[test]), reference.predict(X[test]))


def test_zero_variance_features_do_not_break_the_classifier():
    # A feature that is always 0 in one class has maximum-likelihood variance 0.
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [5.0, 1.0], [6.0, 2.0], [7.0, 3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = GaussianClassConditional().fit(X, y)
    assert np.all(model.var_ > 0)
    assert np.isfinite(model.joint_log_likelihood(X)).all()
    np.testing.assert_array_equal(model.predict(X), y)
