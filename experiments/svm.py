"""Support vector machines on two interleaved half-moons and on handwritten digits:
python -m experiments.svm

Both data sets come from scikit-learn. The settings are fixed in advance, not tuned on
the test sets.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from experiments.common import OUT, save_metrics
from cscc11.model_selection import stratified_split
from cscc11.svm import SVM, OneVsRestSVM

MOONS = {"linear": dict(C=1.0, kernel="linear"), "rbf": dict(C=1.0, kernel="rbf", gamma=1.0)}


def main():
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    X_test, y_test = make_moons(n_samples=2000, noise=0.25, random_state=1)
    xx, yy = np.meshgrid(np.linspace(-2.0, 3.0, 300), np.linspace(-1.6, 2.1, 300))
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    moons = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (name, settings) in zip(axes, MOONS.items()):
        model = SVM(tol=1e-5, **settings).fit(X, y)
        reference = SVC(tol=1e-5, **settings).fit(X, y)
        surface = model.decision_function(grid).reshape(xx.shape)
        moons[name] = {
            "test_accuracy": model.score(X_test, y_test), "train_accuracy": model.score(X, y),
            "support_vectors": int(len(model.support_)), "updates": int(model.n_iter_),
            "largest_difference_from_scikit_learn": float(np.abs(
                model.decision_function(X_test) - reference.decision_function(X_test)).max()),
        }
        ax.contourf(xx, yy, surface, levels=[-1e9, 0, 1e9], colors=["#dbe9f6", "#fde5d3"])
        ax.contour(xx, yy, surface, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[0.8, 1.4, 0.8])
        ax.scatter(*X[y == 0].T, s=12, color="#2b6cb0")
        ax.scatter(*X[y == 1].T, s=12, color="#c05621")
        ax.scatter(*X[model.support_].T, s=46, facecolors="none", edgecolors="k", linewidths=0.6)
        kernel = "Linear kernel" if name == "linear" else "RBF kernel"
        ax.set(title=f"{kernel}: {moons[name]['test_accuracy']:.1%} on 2,000 new points, "
                     f"{len(model.support_)} support vectors", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(OUT / "svm.png", dpi=140)
    plt.close(fig)

    digits = load_digits()
    D, labels = digits.data / 16.0, digits.target
    train, test = stratified_split(labels, test_size=0.3, seed=0)
    settings = dict(C=10.0, kernel="rbf", gamma=float(1.0 / (D.shape[1] * D[train].var())))
    model = OneVsRestSVM(**settings).fit(D[train], labels[train])
    reference = OneVsRestClassifier(SVC(**settings)).fit(D[train], labels[train])
    linear = OneVsRestSVM(C=10.0, kernel="linear").fit(D[train], labels[train])

    save_metrics("svm", {
        "moons": {"n_train": 400, "n_test": 2000, **moons},
        "digits": {
            "samples": int(len(labels)), "n_train": int(len(train)), "n_test": int(len(test)),
            "C": settings["C"], "gamma": settings["gamma"],
            "test_accuracy": model.score(D[test], labels[test]),
            "linear_kernel_test_accuracy": linear.score(D[test], labels[test]),
            "scikit_learn_test_accuracy": float(reference.score(D[test], labels[test])),
            "predictions_equal_to_scikit_learn": bool(np.array_equal(model.predict(D[test]), reference.predict(D[test]))),
            "support_vectors_per_machine": [int(len(machine.support_)) for machine in model.machines_],
        },
    })
    print(f"svm: moons linear {moons['linear']['test_accuracy']:.4f}, rbf {moons['rbf']['test_accuracy']:.4f} "
          f"(largest difference from scikit-learn {moons['rbf']['largest_difference_from_scikit_learn']:.1e}); "
          f"digits rbf {model.score(D[test], labels[test]):.4f}, linear {linear.score(D[test], labels[test]):.4f}, "
          f"scikit-learn {reference.score(D[test], labels[test]):.4f}")


if __name__ == "__main__":
    main()
