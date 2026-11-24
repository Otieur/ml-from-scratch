"""The neural network: backpropagation against finite differences, and training against known answers."""
import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression

from mlscratch.neural_network import NeuralNetwork, softmax


def small_problem(seed=0, n=30, d=5, classes=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, classes, size=n)


def test_softmax_rows_sum_to_one_and_large_scores_do_not_overflow():
    scores = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0]])
    p = softmax(scores)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[0], np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum())
    np.testing.assert_allclose(p[1], 1 / 3)
    np.testing.assert_allclose(p[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_initial_loss_is_close_to_the_log_of_the_number_of_classes():
    X, y = small_problem(n=400, classes=4)
    network = NeuralNetwork((5, 4))
    network.weights[0] *= 0.01
    assert network.loss(X, y) == pytest.approx(np.log(4), abs=0.01)


def test_single_layer_gradient_matches_its_formula():
    # Without hidden layers: dW = X^T (P - Y) / N + l2 W and db = column means of (P - Y).
    X, y = small_problem(seed=1)
    network = NeuralNetwork((5, 3), l2=0.3, seed=1)
    weight_gradients, bias_gradients = network.gradients(X, y)
    difference = network.predict_proba(X) - np.eye(3)[y]
    np.testing.assert_allclose(weight_gradients[0], X.T @ difference / len(y) + 0.3 * network.weights[0], atol=1e-12)
    np.testing.assert_allclose(bias_gradients[0], difference.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("layer_sizes, l2", [((5, 3), 0.0), ((5, 7, 3), 0.0), ((5, 8, 6, 3), 0.2)])
def test_backpropagation_matches_finite_differences(layer_sizes, l2):
    X, y = small_problem(seed=2)
    network = NeuralNetwork(layer_sizes, l2=l2, seed=2)
    for bias in network.biases:                       # non-zero biases, so their gradients are exercised
        bias += np.random.default_rng(3).normal(0, 0.1, size=bias.shape)
    weight_gradients, bias_gradients = network.gradients(X, y)
    step = 1e-6
    for parameters, gradients in ((network.weights, weight_gradients), (network.biases, bias_gradients)):
        for parameter, gradient in zip(parameters, gradients):
            numeric = np.empty_like(parameter)
            for position in np.ndindex(parameter.shape):
                original = parameter[position]
                parameter[position] = original + step
                above = network.loss(X, y)
                parameter[position] = original - step
                below = network.loss(X, y)
                parameter[position] = original
                numeric[position] = (above - below) / (2 * step)
            np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-8)


def test_without_hidden_layers_training_reaches_the_softmax_regression_optimum():
    data = load_iris()
    X = (data.data - data.data.mean(axis=0)) / data.data.std(axis=0)
    y, l2 = data.target, 0.05
    network = NeuralNetwork((4, 3), l2=l2).fit(X, y, epochs=3000, batch_size=len(y), learning_rate=0.5, momentum=0.9)
    weight_gradients, bias_gradients = network.gradients(X, y)
    assert max(np.abs(weight_gradients[0]).max(), np.abs(bias_gradients[0]).max()) < 1e-8
    # scikit-learn minimizes ||W||^2 / 2 + C * (sum of losses): the same problem for C = 1 / (l2 N).
    reference = LogisticRegression(C=1.0 / (l2 * len(y)), tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(network.predict_proba(X), reference.predict_proba(X), atol=1e-5)


def test_full_batch_updates_with_a_small_step_lower_the_loss():
    X, y = small_problem(seed=4, n=80)
    network = NeuralNetwork((5, 10, 3), seed=4).fit(X, y, epochs=50, batch_size=80, learning_rate=0.05, momentum=0.0)
    losses = [record["loss"] for record in network.history_]
    assert all(later <= earlier for earlier, later in zip(losses, losses[1:]))


def test_a_hidden_layer_learns_xor_and_a_single_layer_cannot():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    hidden = NeuralNetwork((2, 8, 2), seed=1).fit(X, y, epochs=500, batch_size=4, learning_rate=0.1)
    assert hidden.score(X, y) == 1.0
    linear = NeuralNetwork((2, 2), seed=1).fit(X, y, epochs=500, batch_size=4, learning_rate=0.1)
    assert linear.score(X, y) <= 0.75


def test_training_is_reproducible_and_the_monitor_set_is_recorded():
    X, y = small_problem(seed=5, n=60)
    first = NeuralNetwork((5, 6, 3), seed=7).fit(X, y, epochs=5, batch_size=16, seed=3, monitor=(X, y))
    second = NeuralNetwork((5, 6, 3), seed=7).fit(X, y, epochs=5, batch_size=16, seed=3, monitor=(X, y))
    for a, b in zip(first.weights, second.weights):
        np.testing.assert_array_equal(a, b)
    assert len(first.history_) == 5 and first.history_[-1]["accuracy"] == first.score(X, y)


def test_reads_handwritten_digits():
    data = load_digits()
    X, y = data.data / 16.0, data.target
    rows = np.random.default_rng(0).permutation(len(y))
    train, test = rows[:1300], rows[1300:]
    network = NeuralNetwork((64, 32, 10), l2=1e-4).fit(X[train], y[train], epochs=30, batch_size=32, learning_rate=0.05)
    assert network.score(X[test], y[test]) > 0.95
