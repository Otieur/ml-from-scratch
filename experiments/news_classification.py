"""Classify BBC news articles with two Bayesian classifiers: python -m experiments.news_classification"""
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import OUT, save_metrics
from mlscratch.datasets import load_bbc
from mlscratch.gaussian_classifier import GaussianClassConditional
from mlscratch.model_selection import k_fold_indices, stratified_split
from mlscratch.naive_bayes import BernoulliNaiveBayes

VAR_SMOOTHING_GRID = (1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1)


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def main():
    counts, labels, class_names, terms, _ = load_bbc()
    binary = (counts > 0).astype(np.float32)
    train, test = stratified_split(labels, test_size=0.3, seed=0)

    # Naive Bayes on whether each term occurs.
    naive = BernoulliNaiveBayes().fit(binary[train], labels[train])
    naive_pred = naive.predict(binary[test])

    # Gaussian class conditionals on how often each term occurs. The variance floor is chosen
    # by 5-fold cross-validation inside the training set; the test set is not used for it.
    folds = k_fold_indices(labels[train], n_folds=5, seed=0)
    cv_accuracy = {}
    for smoothing in VAR_SMOOTHING_GRID:
        scores = []
        for held_out in folds:
            rest = np.setdiff1d(np.arange(len(train)), held_out)
            model = GaussianClassConditional(smoothing).fit(counts[train][rest], labels[train][rest])
            scores.append(model.score(counts[train][held_out], labels[train][held_out]))
        cv_accuracy[smoothing] = float(np.mean(scores))
    chosen = max(cv_accuracy, key=cv_accuracy.get)
    gaussian = GaussianClassConditional(chosen).fit(counts[train], labels[train])
    gaussian_pred = gaussian.predict(counts[test])
    minimal = GaussianClassConditional(VAR_SMOOTHING_GRID[0]).fit(counts[train], labels[train])

    # How much do the test accuracies move with the split?
    repeated = {"naive_bayes": [], "gaussian": []}
    for seed in range(20):
        tr, te = stratified_split(labels, test_size=0.3, seed=seed)
        repeated["naive_bayes"].append(BernoulliNaiveBayes().fit(binary[tr], labels[tr]).score(binary[te], labels[te]))
        repeated["gaussian"].append(GaussianClassConditional(chosen).fit(counts[tr], labels[tr]).score(counts[te], labels[te]))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for ax, predicted, title in ((axes[0], naive_pred, "Naive Bayes, term present or absent"),
                                 (axes[1], gaussian_pred, "Gaussian class conditionals, term counts")):
        matrix = confusion_matrix(labels[test], predicted, len(class_names))
        ax.imshow(matrix, cmap="Blues")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, matrix[i, j], ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), yticklabels=class_names,
               xlabel="Predicted", ylabel="Actual",
               title=f"{title}\ntest accuracy {np.mean(predicted == labels[test]):.1%} ({len(test)} articles)")
        ax.set_xticklabels(class_names, rotation=30, ha="right")
    axes[2].semilogx(list(cv_accuracy), list(cv_accuracy.values()), marker="o")
    axes[2].axvline(chosen, color="gray", linewidth=1)
    axes[2].set(xlabel="Variance floor (share of the largest term variance)", ylabel="5-fold cross-validation accuracy",
                title="Choosing the variance floor")
    fig.savefig(OUT / "news_classification.png", dpi=130)
    plt.close(fig)

    indicative = {class_names[c]: [terms[j] for j in indices]
                  for c, indices in enumerate(naive.most_indicative_terms(top=8))}
    save_metrics("news_classification", {
        "documents": int(len(labels)), "terms": int(counts.shape[1]),
        "class_sizes": dict(zip(class_names, (int(v) for v in np.bincount(labels)))),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "naive_bayes": {"train_accuracy": naive.score(binary[train], labels[train]),
                        "test_accuracy": float(np.mean(naive_pred == labels[test])),
                        "most_indicative_terms": indicative},
        "gaussian": {"var_smoothing": chosen, "cv_accuracy_by_var_smoothing": {str(k): v for k, v in cv_accuracy.items()},
                     "train_accuracy": gaussian.score(counts[train], labels[train]),
                     "test_accuracy": float(np.mean(gaussian_pred == labels[test])),
                     "minimal_floor": {"var_smoothing": VAR_SMOOTHING_GRID[0],
                                       "train_accuracy": minimal.score(counts[train], labels[train]),
                                       "test_accuracy": minimal.score(counts[test], labels[test])}},
        "repeated_splits": {name: {"n": len(values), "mean_test_accuracy": float(np.mean(values)),
                                   "std_test_accuracy": float(np.std(values)), "min": float(np.min(values)),
                                   "max": float(np.max(values))} for name, values in repeated.items()},
    })
    print(f"news: Naive Bayes  train {naive.score(binary[train], labels[train]):.4f}  test {np.mean(naive_pred == labels[test]):.4f}")
    print(f"news: Gaussian     train {gaussian.score(counts[train], labels[train]):.4f}  test {np.mean(gaussian_pred == labels[test]):.4f}"
          f"  (variance floor {chosen:g} from cross-validation; with {VAR_SMOOTHING_GRID[0]:g}: test {minimal.score(counts[test], labels[test]):.4f})")
    print(f"news: over 20 splits, Naive Bayes {np.mean(repeated['naive_bayes']):.4f} +/- {np.std(repeated['naive_bayes']):.4f}, "
          f"Gaussian {np.mean(repeated['gaussian']):.4f} +/- {np.std(repeated['gaussian']):.4f}")


if __name__ == "__main__":
    main()
