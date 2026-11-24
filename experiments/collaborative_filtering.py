"""Rating prediction on MovieLens 100K by matrix factorization:
python -m experiments.collaborative_filtering

The number of factors and the penalty are chosen on a validation part cut from the
training ratings of the first split; the five predefined test sets are used once each.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import OUT, save_metrics
from mlscratch.collaborative_filtering import MatrixFactorization, rmse
from mlscratch.datasets import load_movielens

N_USERS, N_ITEMS = 943, 1682
FACTORS = (0, 5, 10, 20, 40)
PENALTIES = (2.0, 5.0, 10.0, 15.0, 25.0)
SWEEPS = 15
SCALE = (1.0, 5.0)


def fit(n_factors, l2, triple):
    return MatrixFactorization(n_factors, l2, n_sweeps=SWEEPS).fit(*triple, N_USERS, N_ITEMS)


def main():
    (users, items, ratings), _ = load_movielens(1)
    held = np.zeros(len(ratings), dtype=bool)
    held[np.random.default_rng(0).choice(len(ratings), size=len(ratings) // 10, replace=False)] = True
    fitting = (users[~held], items[~held], ratings[~held])
    validation = {}
    for n_factors in FACTORS:
        for l2 in PENALTIES:
            model = fit(n_factors, l2, fitting)
            validation[(n_factors, l2)] = rmse(model.predict(users[held], items[held], clip=SCALE), ratings[held])
    best_factors, best_l2 = min((key for key in validation if key[0] > 0), key=validation.get)
    offsets_l2 = min((key for key in validation if key[0] == 0), key=validation.get)[1]

    test = {"mean rating": [], "offsets": [], "factorization": []}
    seconds = []
    for split in range(1, 6):
        train, (test_users, test_items, test_ratings) = load_movielens(split)
        test["mean rating"].append(rmse(np.full(len(test_ratings), train[2].mean()), test_ratings))
        test["offsets"].append(rmse(fit(0, offsets_l2, train).predict(test_users, test_items, clip=SCALE), test_ratings))
        start = time.perf_counter()
        model = fit(best_factors, best_l2, train)
        seconds.append(time.perf_counter() - start)
        test["factorization"].append(rmse(model.predict(test_users, test_items, clip=SCALE), test_ratings))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for n_factors in FACTORS:
        label = "offsets only" if n_factors == 0 else f"{n_factors} factors"
        axes[0].plot(PENALTIES, [validation[(n_factors, l2)] for l2 in PENALTIES], marker="o", label=label)
    axes[0].set(xlabel="Penalty", ylabel="Validation RMSE", title="Choosing the factors and the penalty (first split)")
    axes[0].legend(fontsize=8)
    names = list(test)
    means = [np.mean(test[name]) for name in names]
    bars = axes[1].bar(names, means, yerr=[np.std(test[name]) for name in names], capsize=4,
                       color=["#9aa0a6", "#6c8ebf", "#2f6f4f"])
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
    axes[1].set(ylabel="Test RMSE (stars)", ylim=(0.85, 1.2),
                title=f"Five test sets of 20,000 ratings ({best_factors} factors, penalty {best_l2:g})")
    fig.tight_layout()
    fig.savefig(OUT / "collaborative_filtering.png", dpi=140)
    plt.close(fig)

    save_metrics("collaborative_filtering", {
        "ratings": 100000, "users": N_USERS, "items": N_ITEMS, "sweeps": SWEEPS,
        "validation_rmse": {f"{k} factors, penalty {l2:g}": value for (k, l2), value in validation.items()},
        "chosen": {"n_factors": best_factors, "l2": best_l2, "offsets_l2": offsets_l2},
        "test_rmse": {name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "splits": values}
                      for name, values in test.items()},
        "seconds_per_fit": float(np.mean(seconds)),
    })
    print(f"collaborative filtering: chose {best_factors} factors, penalty {best_l2:g}; test RMSE "
          + ", ".join(f"{name} {np.mean(values):.4f} +/- {np.std(values):.4f}" for name, values in test.items()))


if __name__ == "__main__":
    main()
