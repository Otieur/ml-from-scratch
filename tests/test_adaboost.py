"""AdaBoost with decision stumps: the stump search, a hand-worked run and the boosting loop."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier

from cscc11.adaboost import AdaBoost, best_stump, stump_predict

# Ten points on a line that no single threshold separates.
LINE_X = np.arange(10, dtype=np.float64).reshape(-1, 1)
LINE_Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])


def noisy_classes(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.7 * X[:, 1] ** 2 - 0.5 * X[:, 2] + 0.5 * rng.normal(size=n) > 0.6).astype(int)
    return X, y


def brute_force_stump(X, signs, weights):
    """Every feature, every threshold between neighbouring values, both polarities."""
    best = (np.inf, None)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        thresholds = np.concatenate(([-np.inf], (values[:-1] + values[1:]) / 2))
        for threshold in thresholds:
            for polarity in (1, -1):
                error = weights[stump_predict(X, feature, threshold, polarity) != signs].sum()
                if error < best[0] - 1e-12:
                    best = (error, (feature, threshold, polarity))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stump_search_finds_the_smallest_weighted_error(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, size=(40, 4)).astype(float)        # repeated values on purpose
    signs = rng.choice([-1.0, 1.0], size=40)
    weights = rng.dirichlet(np.ones(40))
    feature, threshold, polarity, error = best_stump(X, signs, weights)
    expected_error, _ = brute_force_stump(X, signs, weights)
    assert error == pytest.approx(expected_error, abs=1e-12)
    # The reported error is the error of the reported stump.
    assert weights[stump_predict(X, feature, threshold, polarity) != signs].sum() == pytest.approx(error, abs=1e-12)


def test_a_threshold_never_falls_between_equal_values():
    X = np.array([[1.0], [1.0], [1.0], [2.0]])
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    feature, threshold, polarity, error = best_stump(X, signs, np.full(4, 0.25))
    assert threshold in (1.5, -np.inf)
    assert error == pytest.approx(0.25)


def test_three_rounds_by_hand():
    model = AdaBoost(n_rounds=3).fit(LINE_X, LINE_Y)
    # Round 1: the split at 2.5 gets points 6, 7 and 8 wrong, each with weight 1/10.
    assert model.threshold_[0] == 2.5 and model.polarity_[0] == -1
    assert model.error_[0] == pytest.approx(0.3)
    assert model.alpha_[0] == pytest.approx(0.5 * np.log(0.7 / 0.3))
    # The three mistakes now weigh 1/6 each and the seven correct points 1/14 each.
    # Round 2: the split at 8.5 gets points 3, 4 and 5 wrong: 3/14.
    assert model.threshold_[1] == 8.5 and model.polarity_[1] == -1
    assert model.error_[1] == pytest.approx(3 / 14)
    assert model.alpha_[1] == pytest.approx(0.5 * np.log(11 / 3))
    # Round 3: the split at 5.5 with the opposite polarity; weighted error 0.1818.
    assert model.threshold_[2] == 5.5 and model.polarity_[2] == 1
    assert model.error_[2] == pytest.approx(0.1818, abs=1e-4)
    assert model.alpha_[2] == pytest.approx(0.7520, abs=1e-4)
    # No single stump is right on more than 7 of the 10 points; three together are right on all.
    assert AdaBoost(n_rounds=1).fit(LINE_X, LINE_Y).score(LINE_X, LINE_Y) == pytest.approx(0.7)
    assert model.score(LINE_X, LINE_Y) == 1.0


def test_the_stump_just_added_is_no_better_than_chance_under_the_new_weights():
    X, y = noisy_classes()
    signs = np.where(y == 1, 1.0, -1.0)
    for rounds in (1, 2, 5, 20):
        model = AdaBoost(n_rounds=rounds).fit(X, y)
        last = stump_predict(X, model.feature_[-1], model.threshold_[-1], model.polarity_[-1])
        assert model.weights_.sum() == pytest.approx(1.0)
        assert model.weights_[last != signs].sum() == pytest.approx(0.5, abs=1e-9)


def test_exponential_loss_equals_the_product_of_the_normalizers_and_bounds_the_error():
    X, y = noisy_classes(seed=1)
    signs = np.where(y == 1, 1.0, -1.0)
    model = AdaBoost(n_rounds=40).fit(X, y)
    errors = np.array(model.error_)
    assert np.all(errors < 0.5)
    bound = np.cumprod(2 * np.sqrt(errors * (1 - errors)))
    scores = model.staged_decision_function(X)
    np.testing.assert_allclose(np.mean(np.exp(-signs * scores), axis=1), bound, rtol=1e-9)
    training_error = np.mean(np.sign(scores) != signs, axis=1)
    assert np.all(training_error <= bound + 1e-12)
    assert training_error[-1] < training_error[0]


class Stump(ClassifierMixin, BaseEstimator):
    """The stump search wrapped as a scikit-learn estimator, so that scikit-learn runs the boosting loop."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        signs = np.where(y == self.classes_[1], 1.0, -1.0)
        weights = np.full(len(y), 1.0 / len(y)) if sample_weight is None else sample_weight
        self.feature_, self.threshold_, self.polarity_, _ = best_stump(X, signs, weights)
        return self

    def predict(self, X):
        return self.classes_[(stump_predict(X, self.feature_, self.threshold_, self.polarity_) > 0).astype(int)]


def test_boosting_loop_matches_scikit_learn_given_the_same_weak_learner():
    X, y = noisy_classes(seed=2)
    held_out, _ = noisy_classes(seed=3)
    rounds = 30
    model = AdaBoost(n_rounds=rounds).fit(X, y)
    reference = AdaBoostClassifier(estimator=Stump(), n_estimators=rounds).fit(X, y)
    np.testing.assert_allclose(reference.estimator_errors_[:rounds], model.error_, atol=1e-9)
    # scikit-learn's vote is ln((1 - e) / e): twice alpha, which leaves the sign of the sum unchanged.
    np.testing.assert_allclose(reference.estimator_weights_[:rounds], 2 * np.array(model.alpha_), atol=1e-8)
    np.testing.assert_array_equal(model.predict(held_out), reference.predict(held_out))


def test_stops_after_a_perfect_stump_and_keeps_label_values():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["low", "low", "high", "high"])
    model = AdaBoost(n_rounds=10).fit(X, y)
    assert len(model.alpha_) == 1 and model.threshold_[0] == 1.5
    assert model.predict(np.array([[-5.0], [9.0]])).tolist() == ["low", "high"]
