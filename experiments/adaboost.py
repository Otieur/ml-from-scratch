"""AdaBoost with decision stumps on the Wisconsin breast-cancer data:
python -m experiments.adaboost

The data ships with scikit-learn (569 tumours, 30 measurements, benign or malignant).
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier

from experiments.common import OUT, save_metrics
from cscc11.adaboost import AdaBoost
from cscc11.model_selection import stratified_split

ROUNDS = 300
CHECKPOINTS = (1, 10, 50, 100, 300)


def main():
    data = load_breast_cancer()
    X, y = data.data, (data.target == 0).astype(int)          # 1 = malignant
    signs = np.where(y == 1, 1.0, -1.0)
    train, test = stratified_split(y, test_size=0.3, seed=0)

    model = AdaBoost(n_rounds=ROUNDS).fit(X[train], y[train])
    train_error = np.mean(np.sign(model.staged_decision_function(X[train])) != signs[train], axis=1)
    test_error = np.mean(np.sign(model.staged_decision_function(X[test])) != signs[test], axis=1)
    errors = np.array(model.error_)
    bound = np.cumprod(2 * np.sqrt(errors * (1 - errors)))
    reference = AdaBoostClassifier(n_estimators=ROUNDS, random_state=0).fit(X[train], y[train])

    # How much of the total vote each measurement carries.
    votes = np.bincount(model.feature_, weights=model.alpha_, minlength=X.shape[1])
    top = np.argsort(-votes)[:8]

    repeated = []
    for seed in range(1, 21):
        a, b = stratified_split(y, test_size=0.3, seed=seed)
        repeated.append(AdaBoost(n_rounds=100).fit(X[a], y[a]).score(X[b], y[b]))

    rounds = np.arange(1, len(errors) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].semilogx(rounds, bound, color="gray", linestyle=":", label="Bound on the training error")
    axes[0].semilogx(rounds, train_error, label="Training error")
    axes[0].semilogx(rounds, test_error, label="Test error")
    axes[0].set(xlabel="Number of stumps", ylabel="Error rate", ylim=(0, 0.2),
                title=f"One stump {1 - test_error[0]:.1%}, {ROUNDS} stumps {1 - test_error[-1]:.1%} on the test set")
    axes[0].legend(fontsize=8)
    axes[1].barh([data.feature_names[d] for d in top][::-1], (votes[top] / votes.sum())[::-1], color="#6c8ebf")
    axes[1].set(xlabel="Share of the total vote", title="Measurements the stumps rely on")
    fig.tight_layout()
    fig.savefig(OUT / "adaboost.png", dpi=140)
    plt.close(fig)

    save_metrics("adaboost", {
        "samples": int(len(y)), "n_train": int(len(train)), "n_test": int(len(test)), "rounds": int(len(errors)),
        "test_accuracy_by_rounds": {str(t): float(1 - test_error[t - 1]) for t in CHECKPOINTS},
        "train_accuracy_by_rounds": {str(t): float(1 - train_error[t - 1]) for t in CHECKPOINTS},
        "first_round_with_no_training_error": int(np.argmax(train_error == 0) + 1) if (train_error == 0).any() else None,
        "scikit_learn_test_accuracy": float(reference.score(X[test], y[test])),
        "repeated_splits_100_rounds": {"n": len(repeated), "mean_test_accuracy": float(np.mean(repeated)),
                                       "std_test_accuracy": float(np.std(repeated))},
        "vote_share": {str(data.feature_names[d]): float(votes[d] / votes.sum()) for d in top},
    })
    print("adaboost: test accuracy " + ", ".join(f"{t} stumps {1 - test_error[t - 1]:.4f}" for t in CHECKPOINTS)
          + f"; scikit-learn {reference.score(X[test], y[test]):.4f}; "
          f"20 splits at 100 stumps {np.mean(repeated):.4f} +/- {np.std(repeated):.4f}")


if __name__ == "__main__":
    main()
