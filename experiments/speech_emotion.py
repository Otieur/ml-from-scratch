"""Positive or negative emotion in speech with k-NN, cross-validation and PCA:
python -m experiments.speech_emotion
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import OUT, save_metrics
from mlscratch.datasets import load_speech_emotion, log_mel
from mlscratch.knn import KNNClassifier
from mlscratch.model_selection import best_k, cross_validate_knn, prepare, stratified_split
from mlscratch.pca import PCA

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]
PCA_VARIANCE = 0.95
N_SPLITS = 20

# name -> (description, feature transform, standardize, share of variance kept by PCA or None)
VARIANTS = {
    "raw_unscaled": ("Mel energies, not scaled", lambda X: X, False, None),
    "raw": ("Mel energies, standardized", lambda X: X, True, None),
    "raw_pca": (f"Mel energies, PCA keeping {PCA_VARIANCE:.0%}", lambda X: X, True, PCA_VARIANCE),
    "log": ("Log-Mel energies, standardized", log_mel, True, None),
    "log_pca": (f"Log-Mel energies, PCA keeping {PCA_VARIANCE:.0%}", log_mel, True, PCA_VARIANCE),
}
SHORT = {"raw_unscaled": "Mel, not scaled", "raw": "Mel, standardized", "raw_pca": "Mel, PCA",
         "log": "Log-Mel, standardized", "log_pca": "Log-Mel, PCA"}


def run_pipeline(F, y, seed, standardize, pca_variance):
    """One complete run: split, choose k on the training set by cross-validation, score on the test set."""
    train, test = stratified_split(y, test_size=0.3, seed=seed)
    mean_accuracy, fold_accuracy = cross_validate_knn(F[train], y[train], K_VALUES, n_folds=5, seed=0,
                                                      standardize=standardize, pca_variance=pca_variance)
    k = best_k(mean_accuracy)
    Z_train, Z_test, dimensions = prepare(F[train], F[test], standardize, pca_variance)
    predicted = KNNClassifier(k).fit(Z_train, y[train]).predict(Z_test)
    return {"k": int(k), "dimensions": int(dimensions), "cv_accuracy": mean_accuracy,
            "cv_fold_std": {key: float(np.std(value)) for key, value in fold_accuracy.items()},
            "test_accuracy": float(np.mean(predicted == y[test])),
            "Z_train": Z_train, "y_train": y[train], "Z_test": Z_test,
            "n_train": int(len(train)), "n_test": int(len(test))}


def time_predictions(models, repeats=300):
    """Median time in milliseconds to classify the test set, per model.

    The models are timed in turn within every repeat, so a change in machine load during
    the measurement affects all of them alike.
    """
    durations = {name: [] for name in models}
    for _ in range(repeats):
        for name, (model, Z_test) in models.items():
            start = time.perf_counter()
            model.predict(Z_test)
            durations[name].append(time.perf_counter() - start)
    return {name: float(np.median(values) * 1000) for name, values in durations.items()}


def main():
    X, y, info = load_speech_emotion()

    # The split with seed 0 is reported in detail; 19 more splits show how much the numbers move.
    results, first, models = {}, {}, {}
    for name, (label, transform, standardize, pca_variance) in VARIANTS.items():
        F = transform(X)
        runs = [run_pipeline(F, y, seed, standardize, pca_variance) for seed in range(N_SPLITS)]
        first[name] = runs[0]
        models[name] = (KNNClassifier(runs[0]["k"]).fit(runs[0]["Z_train"], runs[0]["y_train"]), runs[0]["Z_test"])
        accuracies = [run["test_accuracy"] for run in runs]
        results[name] = {
            "label": label, "dimensions": runs[0]["dimensions"], "best_k": runs[0]["k"],
            "cv_accuracy": {str(k): v for k, v in runs[0]["cv_accuracy"].items()},
            "cv_fold_std": {str(k): v for k, v in runs[0]["cv_fold_std"].items()},
            "test_accuracy": runs[0]["test_accuracy"],
            "repeated_splits": {"n": N_SPLITS, "mean_test_accuracy": float(np.mean(accuracies)),
                                "std_test_accuracy": float(np.std(accuracies)),
                                "k_chosen": [run["k"] for run in runs]},
        }
    for name, milliseconds in time_predictions(models).items():
        results[name]["prediction_ms"] = milliseconds

    # In the splits above the same actor speaks in the training and the test set. Leaving whole
    # actors out measures the classifier on voices it has never heard: 6 folds of 4 actors each.
    F = log_mel(X)
    actors = info["actor"]
    unseen = []
    for fold in range(6):
        held = np.isin(actors, np.arange(1, 25)[fold::6])
        k = best_k(cross_validate_knn(F[~held], y[~held], K_VALUES, n_folds=5, seed=0)[0])
        Z_rest, Z_held, _ = prepare(F[~held], F[held], standardize=True)
        unseen.append(KNNClassifier(k).fit(Z_rest, y[~held]).score(Z_held, y[held]))

    # How many principal components does each share of the variance need?
    train, _ = stratified_split(y, test_size=0.3, seed=0)
    spectra, kept = {}, {}
    for name, transform in (("raw", lambda X: X), ("log", log_mel)):
        scaled, _, _ = prepare(transform(X)[train], transform(X)[train], standardize=True)
        cumulative = np.cumsum(PCA(n_components=X.shape[1]).fit(scaled).all_explained_variance_ratio_)
        spectra[name] = cumulative
        kept[name] = {str(t): int(np.searchsorted(cumulative, t - 1e-12) + 1) for t in (0.8, 0.9, 0.95, 0.99)}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for name, marker in (("raw", "o"), ("raw_pca", "s"), ("log", "^"), ("log_pca", "D")):
        r = results[name]
        axes[0].errorbar(K_VALUES, [r["cv_accuracy"][str(k)] for k in K_VALUES],
                         yerr=[r["cv_fold_std"][str(k)] for k in K_VALUES], marker=marker, capsize=2,
                         label=f"{r['label']} (best k = {r['best_k']})")
    axes[0].set(xlabel="k", ylabel="5-fold cross-validation accuracy on the training set", ylim=(0.56, 0.84),
                title="Choosing k (bars: spread over the folds)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    for name, label in (("raw", "Mel energies"), ("log", "Log-Mel energies")):
        axes[1].plot(np.arange(1, len(spectra[name]) + 1), spectra[name],
                     label=f"{label}: {kept[name][str(PCA_VARIANCE)]} components for {PCA_VARIANCE:.0%}")
    axes[1].axhline(PCA_VARIANCE, color="gray", linewidth=1)
    axes[1].set(xlabel="Number of principal components", ylabel="Share of the variance explained",
                title="Variance explained by the leading components")
    axes[1].legend(fontsize=8, loc="lower right")
    names = list(VARIANTS)
    means = [results[n]["repeated_splits"]["mean_test_accuracy"] for n in names]
    stds = [results[n]["repeated_splits"]["std_test_accuracy"] for n in names]
    axes[2].barh(range(len(names)), means, xerr=stds, capsize=3,
                 color=["tab:gray", "tab:blue", "tab:blue", "tab:green", "tab:green"])
    axes[2].set_yticks(range(len(names)), [f"{SHORT[n]}\n{results[n]['dimensions']} dimensions" for n in names], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].axvline(0.5, color="black", linewidth=1, linestyle=":")
    axes[2].set(xlim=(0.4, 0.9), xlabel=f"Test accuracy (mean and spread over {N_SPLITS} splits)",
                title="Test accuracy; dotted line: guessing")
    fig.savefig(OUT / "speech_emotion.png", dpi=130)
    plt.close(fig)

    save_metrics("speech_emotion", {
        "clips": int(len(y)), "features": int(X.shape[1]),
        "n_train": first["raw"]["n_train"], "n_test": first["raw"]["n_test"],
        "class_sizes": {"positive": int((y == 0).sum()), "negative": int((y == 1).sum())},
        "stereo_clips": int(info["stereo"].sum()), "k_values": K_VALUES, "pca_variance": PCA_VARIANCE,
        "variants": results, "components_kept_by_variance_threshold": kept,
        "unseen_actors": {"fold_accuracy": unseen, "mean_accuracy": float(np.mean(unseen)),
                          "std_accuracy": float(np.std(unseen))},
    })
    for name in names:
        r = results[name]
        print(f"speech: {r['label']:36s} {r['dimensions']:3d} dims  k {r['best_k']:2d}  test {r['test_accuracy']:.4f}  "
              f"{N_SPLITS} splits {r['repeated_splits']['mean_test_accuracy']:.4f} +/- "
              f"{r['repeated_splits']['std_test_accuracy']:.4f}  predict {r['prediction_ms']:.2f} ms")
    print(f"speech: unseen actors (log-Mel): {np.mean(unseen):.4f} +/- {np.std(unseen):.4f}")


if __name__ == "__main__":
    main()
