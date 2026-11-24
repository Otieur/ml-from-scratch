"""Logistic regression by gradient descent on the Wisconsin breast-cancer data:
python -m experiments.logistic_regression

The data ships with scikit-learn (569 tumours, 30 measurements, benign or malignant).
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from experiments.common import OUT, save_metrics
from mlscratch import logistic_regression as lr
from mlscratch.model_selection import Standardizer, stratified_split

L2 = 1.0


def roc_curve(y_true, scores):
    """False-positive and true-positive rates for every threshold, and the area under the curve."""
    order = np.argsort(-scores, kind="stable")
    hits = y_true[order] == 1
    tpr = np.concatenate(([0.0], np.cumsum(hits) / hits.sum()))
    fpr = np.concatenate(([0.0], np.cumsum(~hits) / (~hits).sum()))
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def main():
    data = load_breast_cancer()
    X, y = data.data, (data.target == 0).astype(float)        # 1 = malignant
    train, test = stratified_split(y, test_size=0.3, seed=0)
    scaler = Standardizer().fit(X[train])
    Z_train, Z_test = scaler.transform(X[train]), scaler.transform(X[test])

    params, iterations = lr.fit(Z_train, y[train], l2=L2)
    reference = LogisticRegression(C=1.0 / L2, tol=1e-10, max_iter=10000).fit(Z_train, y[train])
    largest_difference = float(max(np.abs(params[:-1] - reference.coef_[0]).max(),
                                   abs(params[-1] - reference.intercept_[0])))

    probabilities = lr.predict_proba(params, Z_test)
    predicted = (probabilities >= 0.5).astype(int)
    fpr, tpr, auc = roc_curve(y[test], probabilities)
    tp = int(np.sum((predicted == 1) & (y[test] == 1)))
    fp = int(np.sum((predicted == 1) & (y[test] == 0)))
    fn = int(np.sum((predicted == 0) & (y[test] == 1)))
    tn = int(np.sum((predicted == 0) & (y[test] == 0)))

    # The objective along the way, for the convergence plot.
    A = lr.augment(Z_train)
    step = 1.0 / (np.linalg.eigvalsh(A.T @ A)[-1] / 4.0 + L2)
    path = lr.gradient_descent(Z_train, y[train], np.zeros(A.shape[1]), step, 2000, l2=L2)
    optimum = lr.negative_log_likelihood(params, Z_train, y[train], L2)
    gaps = [lr.negative_log_likelihood(p, Z_train, y[train], L2) - optimum for p in path]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].semilogy(np.maximum(gaps, 1e-16))
    axes[0].set(xlabel="Gradient descent update", ylabel="Objective minus its minimum",
                title="Convergence with step 1 / (curvature bound)")
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title=f"Test set ({len(test)} tumours): accuracy {np.mean(predicted == y[test]):.1%}, AUC {auc:.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "logistic_regression.png", dpi=140)
    plt.close(fig)

    save_metrics("logistic_regression", {
        "samples": int(len(y)), "features": int(X.shape[1]), "n_train": int(len(train)), "n_test": int(len(test)),
        "l2": L2, "iterations": int(iterations),
        "train_accuracy": lr.accuracy(params, Z_train, y[train]),
        "test_accuracy": float(np.mean(predicted == y[test])), "test_auc": auc,
        "confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "largest_difference_from_scikit_learn": largest_difference,
        "scikit_learn_test_accuracy": float(reference.score(Z_test, y[test])),
    })
    print(f"logistic regression: {iterations} updates, test accuracy {np.mean(predicted == y[test]):.4f}, AUC {auc:.4f}, "
          f"missed malignant {fn}, false alarms {fp}; largest weight difference from scikit-learn {largest_difference:.1e}")


if __name__ == "__main__":
    main()
