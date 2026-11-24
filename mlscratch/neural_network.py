"""A fully connected neural network for classification, trained by backpropagation.

Forward:   h_0 = x,   a_l = h_{l-1} W_l + b_l,   h_l = max(a_l, 0) on the hidden layers,
           p = softmax(a_L) on the last one
Loss:      mean over the batch of -ln p[y]  +  (l2 / 2) sum_l ||W_l||^2
Backward:  d_L = (p - onehot(y)) / N
           dW_l = h_{l-1}^T d_l + l2 W_l,   db_l = column sums of d_l
           d_{l-1} = (d_l W_l^T) * [a_{l-1} > 0]
Update:    v <- momentum * v - learning_rate * gradient,   parameter <- parameter + v

Without hidden layers the model is softmax (multinomial logistic) regression.
"""
import numpy as np


def softmax(scores):
    """Row-wise softmax, computed after subtracting the row maximum so that exp cannot overflow."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """Multi-layer perceptron with ReLU hidden layers and a softmax output.

    Args:
    - layer_sizes (sequence of int): inputs, hidden layers..., classes; (64, 32, 10) has one
      hidden layer of 32 units. Labels are the integers 0 .. classes - 1.
    - l2 (float): weight penalty (the biases are not penalized).
    """

    def __init__(self, layer_sizes, l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.layer_sizes, self.l2 = tuple(layer_sizes), float(l2)
        # He initialization keeps the scale of the activations steady through ReLU layers.
        self.weights = [rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
                        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.biases = [np.zeros(n_out) for n_out in self.layer_sizes[1:]]

    def _forward(self, X):
        """Activations of every layer: [x, h_1, ..., h_{L-1}, p]."""
        activations = [np.asarray(X, dtype=np.float64)]
        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            scores = activations[-1] @ W + b
            last = layer == len(self.weights) - 1
            activations.append(softmax(scores) if last else np.maximum(scores, 0.0))
        return activations

    def predict_proba(self, X):
        return self._forward(X)[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def loss(self, X, y):
        y = np.asarray(y)
        probabilities = self.predict_proba(X)
        cross_entropy = -np.mean(np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-300)))
        return float(cross_entropy + 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.weights))

    def gradients(self, X, y):
        """Gradient of the loss with respect to every weight matrix and bias vector.

        Output:
        - weight_gradients, bias_gradients (lists with the shapes of self.weights and self.biases).
        """
        y = np.asarray(y)
        activations = self._forward(X)
        delta = activations[-1].copy()
        delta[np.arange(len(y)), y] -= 1.0
        delta /= len(y)
        weight_gradients, bias_gradients = [None] * len(self.weights), [None] * len(self.weights)
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = activations[layer].T @ delta + self.l2 * self.weights[layer]
            bias_gradients[layer] = delta.sum(axis=0)
            if layer > 0:
                # h_l > 0 exactly where a_l > 0, so the stored activation tells which units were on.
                delta = (delta @ self.weights[layer].T) * (activations[layer] > 0)
        return weight_gradients, bias_gradients

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.1, momentum=0.9, seed=0, monitor=None):
        """Mini-batch gradient descent with momentum.

        Args:
        - monitor (tuple (X_other, y_other) or None): a set whose accuracy is recorded after each epoch.

        After fit: history_ is a list with one dict per epoch ("loss", and "accuracy" when
        monitor is given).
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        rng = np.random.default_rng(seed)
        weight_velocity = [np.zeros_like(W) for W in self.weights]
        bias_velocity = [np.zeros_like(b) for b in self.biases]
        self.history_ = []
        for _ in range(epochs):
            rows = rng.permutation(len(y)) if batch_size < len(y) else np.arange(len(y))
            for start in range(0, len(y), batch_size):
                batch = rows[start:start + batch_size]
                weight_gradients, bias_gradients = self.gradients(X[batch], y[batch])
                for layer in range(len(self.weights)):
                    weight_velocity[layer] = momentum * weight_velocity[layer] - learning_rate * weight_gradients[layer]
                    bias_velocity[layer] = momentum * bias_velocity[layer] - learning_rate * bias_gradients[layer]
                    self.weights[layer] += weight_velocity[layer]
                    self.biases[layer] += bias_velocity[layer]
            record = {"loss": self.loss(X, y)}
            if monitor is not None:
                record["accuracy"] = self.score(*monitor)
            self.history_.append(record)
        return self
