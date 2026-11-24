"""A neural network on handwritten digits: python -m experiments.neural_network

The data ships with scikit-learn (1,797 images of 8 x 8 pixels). The settings are fixed in
advance; the test set is only watched during training, never used to choose anything.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

from experiments.common import OUT, save_metrics
from mlscratch.model_selection import stratified_split
from mlscratch.neural_network import NeuralNetwork

TRAINING = dict(epochs=60, batch_size=32, learning_rate=0.05, momentum=0.9)
L2 = 1e-4


def main():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    train, test = stratified_split(y, test_size=0.3, seed=0)

    models = {
        "Softmax regression (no hidden layer)": NeuralNetwork((64, 10), l2=L2),
        "One hidden layer of 64 units": NeuralNetwork((64, 64, 10), l2=L2),
    }
    for model in models.values():
        model.fit(X[train], y[train], monitor=(X[test], y[test]), **TRAINING)
    network = models["One hidden layer of 64 units"]
    reference = MLPClassifier(hidden_layer_sizes=(64,), alpha=L2, max_iter=2000, random_state=0).fit(X[train], y[train])

    predicted = network.predict(X[test])
    wrong = np.flatnonzero(predicted != y[test])
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y[test], predicted), 1)

    fig = plt.figure(figsize=(11.5, 4.3))
    left = fig.add_subplot(1, 2, 1)
    for name, model in models.items():
        left.plot(range(1, TRAINING["epochs"] + 1), [1 - record["accuracy"] for record in model.history_],
                  label=f"{name}: {model.score(X[test], y[test]):.1%}")
    left.set(xlabel="Epoch", ylabel="Test error", ylim=(0, 0.12), title=f"Test set of {len(test)} digits")
    left.legend(fontsize=8)
    shown = wrong[:10]
    for position, row in enumerate(shown):
        ax = fig.add_subplot(2, 10, 6 + position if position < 5 else 11 + position)
        ax.imshow(X[test][row].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"{y[test][row]} read as {predicted[row]}", fontsize=8)
        ax.axis("off")
    fig.text(0.75, 0.94, f"The network's {len(wrong)} mistakes" + (" (first 10)" if len(wrong) > 10 else ""), ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "neural_network.png", dpi=140)
    plt.close(fig)

    save_metrics("neural_network", {
        "samples": int(len(y)), "n_train": int(len(train)), "n_test": int(len(test)), "l2": L2, **TRAINING,
        "softmax_regression": {"test_accuracy": models["Softmax regression (no hidden layer)"].score(X[test], y[test]),
                               "train_accuracy": models["Softmax regression (no hidden layer)"].score(X[train], y[train])},
        "hidden_64": {"test_accuracy": network.score(X[test], y[test]), "train_accuracy": network.score(X[train], y[train]),
                      "final_loss": network.history_[-1]["loss"], "mistakes": int(len(wrong)),
                      "most_confused": [f"{t} read as {p}: {confusion[t, p]}" for t, p in
                                        sorted(((t, p) for t in range(10) for p in range(10) if t != p),
                                               key=lambda pair: -confusion[pair])[:3]]},
        "scikit_learn_mlp_test_accuracy": float(reference.score(X[test], y[test])),
    })
    print(f"neural network: softmax regression {models['Softmax regression (no hidden layer)'].score(X[test], y[test]):.4f}, "
          f"hidden layer {network.score(X[test], y[test]):.4f} ({len(wrong)} mistakes), "
          f"scikit-learn MLP {reference.score(X[test], y[test]):.4f}")


if __name__ == "__main__":
    main()
