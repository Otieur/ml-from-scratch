"""k-NN, cross-validation and PCA against scikit-learn."""
import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.neighbors import KNeighborsClassifier

from mlscratch.model_selection import (
    Standardizer, best_k, cross_validate_knn, k_fold_indices, prepare, stratified_split,
)
from mlscratch.knn import KNNClassifier, nearest_neighbours, neighbour_order, squared_distances, vote
from mlscratch.pca import PCA


def two_classes(seed=0, n=240, d=12):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, d)) * np.linspace(0.5, 3.0, d)
    X[y == 1, :3] += 1.5
    order = rng.permutation(n)
    return X[order], y[order]


# ---------------------------------------------------------------- k-NN

def test_squared_distances_match_the_definition():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
    direct = np.array([[np.sum((a - b) ** 2) for b in B] for a in A])
    np.testing.assert_allclose(squared_distances(A, B), direct, atol=1e-10)


def test_vote_counts_and_breaks_ties_toward_the_smaller_class():
    labels = np.array([[0, 1, 1], [2, 2, 0], [0, 1, 2], [1, 0, 0]])
    np.testing.assert_array_equal(vote(labels, 3), [1, 2, 0, 0])


def test_partial_sort_returns_the_same_neighbours_as_a_full_sort():
    rng = np.random.default_rng(5)
    A, B = rng.normal(size=(30, 6)), rng.normal(size=(80, 6))
    full = neighbour_order(A, B)
    for k in (1, 4, 17, 80):
        partial = nearest_neighbours(A, B, k)
        assert partial.shape == (30, k)
        np.testing.assert_array_equal(np.sort(partial, axis=1), np.sort(full[:, :k], axis=1))


def test_one_neighbour_on_the_training_set_is_always_right():
    X, y = two_classes()
    assert KNNClassifier(k=1).fit(X, y).score(X, y) == 1.0


@pytest.mark.parametrize("k", [1, 3, 5, 9, 15])
def test_knn_matches_scikit_learn(k):
    X, y = two_classes(seed=2)
    train, test = slice(0, 170), slice(170, None)
    mine = KNNClassifier(k=k).fit(X[train], y[train]).predict(X[test])
    reference = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X[train], y[train]).predict(X[test])
    np.testing.assert_array_equal(mine, reference)


def test_knn_keeps_the_original_labels():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array(["calm", "calm", "angry", "angry"])
    np.testing.assert_array_equal(KNNClassifier(k=1).fit(X, y).predict(np.array([[0.2], [4.9]])), ["calm", "angry"])


# ---------------------------------------------------------------- splitting and cross-validation

def test_stratified_split_is_disjoint_complete_and_balanced():
    y = np.array([0] * 70 + [1] * 30)
    train, test = stratified_split(y, test_size=0.3, seed=0)
    assert len(np.intersect1d(train, test)) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate((train, test))), np.arange(100))
    assert (y[test] == 0).sum() == 21 and (y[test] == 1).sum() == 9


def test_folds_partition_the_rows_with_similar_class_shares():
    y = np.array([0] * 53 + [1] * 47)
    folds = k_fold_indices(y, n_folds=5, seed=1)
    assert len(folds) == 5
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(100))
    for fold in folds:
        assert 19 <= len(fold) <= 21
        assert 9 <= (y[fold] == 0).sum() <= 12


def test_standardizer_uses_training_statistics_only():
    rng = np.random.default_rng(2)
    train, other = rng.normal(3, 2, size=(50, 4)), rng.normal(-1, 5, size=(20, 4))
    train[:, 3] = 7.0  # a constant feature
    scaler = Standardizer().fit(train)
    z = scaler.transform(train)
    np.testing.assert_allclose(z[:, :3].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z[:, :3].std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(z[:, 3], 0.0)
    np.testing.assert_allclose(scaler.transform(other), (other - scaler.mean_) / scaler.scale_)


def test_cross_validation_equals_a_manual_loop_with_library_knn():
    X, y = two_classes(seed=3)
    k_values = [1, 3, 7]
    mean_accuracy, fold_accuracy = cross_validate_knn(X, y, k_values, n_folds=4, seed=5, standardize=True)
    folds = k_fold_indices(y, 4, 5)
    for k in k_values:
        manual = []
        for held_out in folds:
            rest = np.setdiff1d(np.arange(len(y)), held_out)
            Z_rest, Z_held, _ = prepare(X[rest], X[held_out], standardize=True)
            reference = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Z_rest, y[rest])
            manual.append(reference.score(Z_held, y[held_out]))
        np.testing.assert_allclose(fold_accuracy[k], manual)
        assert mean_accuracy[k] == pytest.approx(np.mean(manual))


def test_training_accuracy_would_always_choose_one_neighbour():
    # Scored on the training set itself, k = 1 is 100% accurate by construction and would
    # always be chosen. Cross-validation must not behave like that on noisy data.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] + 1.5 * rng.normal(size=300) > 0).astype(int)   # heavily overlapping classes
    mean_accuracy, _ = cross_validate_knn(X, y, [1, 5, 15, 31], n_folds=5, seed=0)
    assert mean_accuracy[1] < 0.9
    assert best_k(mean_accuracy) > 1


def test_best_k_prefers_the_smaller_k_on_a_tie():
    assert best_k({3: 0.8, 5: 0.8, 7: 0.75}) == 3


# ---------------------------------------------------------------- PCA

def test_pca_matches_scikit_learn():
    X, _ = two_classes(seed=6)
    mine = PCA(n_components=5).fit(X)
    reference = SklearnPCA(n_components=5).fit(X)
    np.testing.assert_allclose(mine.explained_variance_, reference.explained_variance_, rtol=1e-10)
    np.testing.assert_allclose(mine.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-10)
    # Directions agree up to their sign.
    np.testing.assert_allclose(np.abs(mine.components_ @ reference.components_.T), np.eye(5), atol=1e-8)
    np.testing.assert_allclose(np.abs(mine.transform(X)), np.abs(reference.transform(X)), atol=1e-8)


def test_pca_components_are_orthonormal_and_sorted():
    X, _ = two_classes(seed=7)
    pca = PCA(n_components=8).fit(X)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(8), atol=1e-10)
    assert np.all(np.diff(pca.explained_variance_) <= 1e-12)
    # The variance of the projected data along each direction is its eigenvalue.
    np.testing.assert_allclose(pca.transform(X).var(axis=0, ddof=1), pca.explained_variance_, rtol=1e-10)


def test_variance_threshold_keeps_the_fewest_components_that_reach_it():
    X, _ = two_classes(seed=8)
    for threshold in (0.5, 0.9, 0.99, 1.0):
        pca = PCA(variance_threshold=threshold).fit(X)
        cumulative = np.cumsum(pca.all_explained_variance_ratio_)
        k = pca.n_components_
        assert cumulative[k - 1] >= threshold - 1e-12
        assert k == 1 or cumulative[k - 2] < threshold - 1e-12


def test_all_components_reconstruct_the_data_exactly():
    X, _ = two_classes(seed=9)
    pca = PCA(n_components=X.shape[1]).fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-9)


def test_projection_uses_the_training_mean():
    X, _ = two_classes(seed=10)
    pca = PCA(n_components=3).fit(X[:150])
    np.testing.assert_allclose(pca.transform(X[150:]), (X[150:] - X[:150].mean(axis=0)) @ pca.components_.T)


def test_preprocessing_fitted_on_training_part_only():
    X, _ = two_classes(seed=11)
    Z_train, Z_other, d = prepare(X[:150], X[150:], standardize=True, pca_variance=0.9)
    assert Z_train.shape[1] == Z_other.shape[1] == d < X.shape[1]
    np.testing.assert_allclose(Z_train.mean(axis=0), 0.0, atol=1e-10)
