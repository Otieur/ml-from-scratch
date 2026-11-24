"""Collaborative filtering by matrix factorization, fitted with alternating least squares.

Model:      r_ui  ~  mu + b_u + b_i + p_u . q_i
            mu is the mean rating, b_u and b_i are user and item offsets, and p_u, q_i are
            vectors of n_factors numbers. With n_factors = 0 only the offsets remain.
Objective:  sum over the known ratings of (r_ui - prediction)^2
            + l2 * (sum_u (b_u^2 + ||p_u||^2) + sum_i (b_i^2 + ||q_i||^2))

With the item side fixed, the objective separates over users, and each user's (b_u, p_u)
is the solution of a small ridge regression on that user's ratings: the inputs are
[1, q_i] and the targets r_ui - mu - b_i. The same holds for items with the user side
fixed. Each half-sweep solves its side exactly, so the objective never increases.
"""
import numpy as np


def rmse(predicted, ratings):
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(ratings)) ** 2)))


def _groups(keys, count):
    """For each key value 0..count-1, the positions where it occurs."""
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(count + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(count)]


def _solve_side(targets, other_factors, other_index, groups, l2):
    """Ridge solution [offset, factors] for every row of one side, the other side being fixed."""
    width = other_factors.shape[1] + 1
    solution = np.zeros((len(groups), width))
    penalty = l2 * np.eye(width)
    for row, positions in enumerate(groups):
        if len(positions) == 0:
            continue                                   # no ratings: the penalty alone gives zero
        A = np.hstack((np.ones((len(positions), 1)), other_factors[other_index[positions]]))
        solution[row] = np.linalg.solve(A.T @ A + penalty, A.T @ targets[positions])
    return solution[:, 0], solution[:, 1:]


class MatrixFactorization:
    """Rating prediction from user and item offsets plus a low-rank interaction.

    Args:
    - n_factors (int): length of the user and item vectors; 0 fits the offsets only.
    - l2 (float): penalty on the offsets and the vectors; must be positive.
    - n_sweeps (int): number of (users, then items) updates.
    """

    def __init__(self, n_factors=10, l2=10.0, n_sweeps=20, seed=0):
        assert l2 > 0, "without a penalty a user or item with few ratings has no unique solution"
        self.n_factors, self.l2, self.n_sweeps, self.seed = n_factors, float(l2), n_sweeps, seed

    def fit(self, users, items, ratings, n_users=None, n_items=None):
        """users and items are integer indices starting at 0, one entry per known rating."""
        users, items = np.asarray(users), np.asarray(items)
        ratings = np.asarray(ratings, dtype=np.float64)
        n_users = int(users.max()) + 1 if n_users is None else n_users
        n_items = int(items.max()) + 1 if n_items is None else n_items
        rng = np.random.default_rng(self.seed)
        self.mean_ = float(ratings.mean())
        self.user_offset_, self.item_offset_ = np.zeros(n_users), np.zeros(n_items)
        self.user_factors_ = np.zeros((n_users, self.n_factors))
        self.item_factors_ = 0.1 * rng.standard_normal((n_items, self.n_factors))
        by_user, by_item = _groups(users, n_users), _groups(items, n_items)

        self.history_ = []
        for _ in range(self.n_sweeps):
            self.user_offset_, self.user_factors_ = _solve_side(
                ratings - self.mean_ - self.item_offset_[items], self.item_factors_, items, by_user, self.l2)
            self.item_offset_, self.item_factors_ = _solve_side(
                ratings - self.mean_ - self.user_offset_[users], self.user_factors_, users, by_item, self.l2)
            self.history_.append(self.objective(users, items, ratings))
        return self

    def predict(self, users, items, clip=None):
        """Predicted ratings; clip=(low, high) keeps them inside the rating scale."""
        users, items = np.asarray(users), np.asarray(items)
        predicted = (self.mean_ + self.user_offset_[users] + self.item_offset_[items]
                     + np.sum(self.user_factors_[users] * self.item_factors_[items], axis=1))
        return predicted if clip is None else np.clip(predicted, *clip)

    def objective(self, users, items, ratings):
        residual = np.asarray(ratings, dtype=np.float64) - self.predict(users, items)
        penalty = (np.sum(self.user_offset_ ** 2) + np.sum(self.item_offset_ ** 2)
                   + np.sum(self.user_factors_ ** 2) + np.sum(self.item_factors_ ** 2))
        return float(residual @ residual + self.l2 * penalty)

    def recommend(self, user, exclude, n=10):
        """The n items with the highest predicted rating for one user, skipping those in exclude."""
        scores = self.predict(np.full(len(self.item_offset_), user), np.arange(len(self.item_offset_)))
        scores[np.asarray(list(exclude), dtype=int)] = -np.inf
        return np.argsort(-scores, kind="stable")[:n]
