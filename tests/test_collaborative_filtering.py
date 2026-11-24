"""Matrix factorization by alternating least squares."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from cscc11.collaborative_filtering import MatrixFactorization, rmse


def low_rank_ratings(seed=0, n_users=60, n_items=40, rank=3, observed=0.6, noise=0.0):
    """Ratings that follow the model exactly (plus optional noise), with a random part observed."""
    rng = np.random.default_rng(seed)
    full = (3.0 + rng.normal(0, 0.5, n_users)[:, None] + rng.normal(0, 0.5, n_items)[None, :]
            + rng.normal(size=(n_users, rank)) @ rng.normal(size=(n_items, rank)).T / np.sqrt(rank))
    full += noise * rng.normal(size=full.shape)
    users, items = np.nonzero(rng.random(full.shape) < observed)
    return users, items, full[users, items], full


def test_rmse():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(np.sqrt(4 / 3))


def test_offsets_alone_fit_a_table_of_row_and_column_effects():
    # A fully observed table that is exactly "row effect + column effect": with a tiny
    # penalty the fitted predictions reproduce it.
    rows, columns = np.array([0.0, 1.0, -1.0, 2.0]), np.array([3.0, 3.5, 2.0])
    table = rows[:, None] + columns[None, :]
    users, items = np.nonzero(np.ones_like(table, dtype=bool))
    model = MatrixFactorization(n_factors=0, l2=1e-9, n_sweeps=50).fit(users, items, table[users, items])
    np.testing.assert_allclose(model.predict(users, items), table[users, items], atol=1e-6)


def test_one_user_update_is_a_ridge_regression():
    users, items, ratings, _ = low_rank_ratings(seed=1)
    model = MatrixFactorization(n_factors=4, l2=2.5, n_sweeps=3).fit(users, items, ratings)
    # Redo the user half-sweep with the fitted item side, then compare one user with scikit-learn.
    from cscc11.collaborative_filtering import _groups, _solve_side

    by_user = _groups(users, users.max() + 1)
    offsets, factors = _solve_side(ratings - model.mean_ - model.item_offset_[items],
                                   model.item_factors_, items, by_user, model.l2)
    for user in (0, 7, 31):
        rated = by_user[user]
        inputs = np.hstack((np.ones((len(rated), 1)), model.item_factors_[items[rated]]))
        targets = ratings[rated] - model.mean_ - model.item_offset_[items[rated]]
        reference = Ridge(alpha=2.5, fit_intercept=False, solver="cholesky").fit(inputs, targets)
        np.testing.assert_allclose(np.concatenate(([offsets[user]], factors[user])), reference.coef_, atol=1e-9)


def test_objective_never_increases():
    users, items, ratings, _ = low_rank_ratings(seed=2, noise=0.3)
    for n_factors in (0, 2, 8):
        history = MatrixFactorization(n_factors=n_factors, l2=1.0, n_sweeps=25).fit(users, items, ratings).history_
        assert all(later <= earlier + 1e-9 for earlier, later in zip(history, history[1:]))


def test_item_side_is_stationary_after_the_last_update():
    users, items, ratings, _ = low_rank_ratings(seed=3, noise=0.2)
    model = MatrixFactorization(n_factors=3, l2=0.8, n_sweeps=5).fit(users, items, ratings)
    residual = ratings - model.predict(users, items)
    # d objective / d q_i = -2 sum_u residual_ui p_u + 2 l2 q_i, and the same with 1 in place of p_u for b_i.
    factor_gradient = 2 * model.l2 * model.item_factors_
    np.subtract.at(factor_gradient, items, 2 * residual[:, None] * model.user_factors_[users])
    offset_gradient = 2 * model.l2 * model.item_offset_ - 2 * np.bincount(items, residual, len(model.item_offset_))
    np.testing.assert_allclose(factor_gradient, 0.0, atol=1e-8)
    np.testing.assert_allclose(offset_gradient, 0.0, atol=1e-8)


def test_recovers_hidden_entries_of_a_low_rank_table():
    users, items, ratings, full = low_rank_ratings(seed=4, n_users=120, n_items=80, observed=0.5)
    hidden = np.ones_like(full, dtype=bool)
    hidden[users, items] = False
    hidden_users, hidden_items = np.nonzero(hidden)
    factorized = MatrixFactorization(n_factors=3, l2=0.01, n_sweeps=60).fit(users, items, ratings, 120, 80)
    offsets = MatrixFactorization(n_factors=0, l2=0.01, n_sweeps=60).fit(users, items, ratings, 120, 80)
    error = rmse(factorized.predict(hidden_users, hidden_items), full[hidden])
    assert error < 0.1
    assert error < 0.2 * rmse(offsets.predict(hidden_users, hidden_items), full[hidden])


def test_an_item_without_ratings_gets_the_mean_plus_the_user_offset():
    users, items, ratings, _ = low_rank_ratings(seed=5)
    model = MatrixFactorization(n_factors=3, l2=1.0, n_sweeps=5).fit(users, items, ratings, n_items=41)
    assert model.item_offset_[40] == 0.0 and not model.item_factors_[40].any()
    assert model.predict([2], [40])[0] == pytest.approx(model.mean_ + model.user_offset_[2])


def test_recommendations_skip_rated_items_and_follow_the_predictions():
    users, items, ratings, _ = low_rank_ratings(seed=6)
    model = MatrixFactorization(n_factors=3, l2=0.5, n_sweeps=10).fit(users, items, ratings)
    seen = set(items[users == 0].tolist())
    top = model.recommend(0, seen, n=5)
    assert not seen.intersection(top.tolist())
    scores = model.predict(np.zeros(5, dtype=int), top)
    assert np.all(np.diff(scores) <= 1e-12)
    unseen = [i for i in range(40) if i not in seen]
    assert scores[0] == pytest.approx(model.predict(np.zeros(len(unseen), dtype=int), unseen).max())


def test_clipping_keeps_predictions_on_the_rating_scale():
    users, items, ratings, _ = low_rank_ratings(seed=7)
    model = MatrixFactorization(n_factors=3, l2=0.5, n_sweeps=5).fit(users, items, ratings)
    clipped = model.predict(users, items, clip=(2.5, 3.5))
    assert clipped.min() >= 2.5 and clipped.max() <= 3.5
