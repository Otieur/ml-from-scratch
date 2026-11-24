"""Linear regression on the graduate admissions data: python -m experiments.admissions_regression"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from experiments.common import OUT, save_metrics
from cscc11.datasets import load_admissions
from cscc11.linear_regression import fit_least_squares, mean_squared_error, predict, r_squared


def main():
    X, y, names = load_admissions()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    w = fit_least_squares(X_train, y_train)
    pred_train, pred_test = predict(w, X_train), predict(w, X_test)
    test_mse = mean_squared_error(y_test, pred_test)
    # Predicting the training mean for everyone is the error a model must beat.
    baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))

    # One split is one draw. Repeat the 70/30 split to see how much the test error moves.
    repeats = []
    for seed in range(200):
        a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        repeats.append(mean_squared_error(b_test, predict(fit_least_squares(a_train, b_train), a_test)))
    repeats = np.array(repeats)

    # Weights on standardized features are comparable across features.
    w_std = fit_least_squares((X - X.mean(axis=0)) / X.std(axis=0), y)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(y_test, pred_test, s=18, alpha=0.7)
    axes[0].plot([0.3, 1.0], [0.3, 1.0], color="black", linewidth=1)
    axes[0].set(xlim=(0.3, 1.0), ylim=(0.3, 1.0), xlabel="Actual chance of admission",
                ylabel="Predicted chance of admission",
                title=f"Test set ({len(y_test)} applicants), MSE = {test_mse:.5f}")
    order = np.argsort(np.abs(w_std))
    axes[1].barh(np.array(names)[order], w_std[order])
    axes[1].set(xlabel="Weight per standard deviation of the feature", title="Which features move the prediction")
    fig.tight_layout()
    fig.savefig(OUT / "admissions_regression.png", dpi=150)
    plt.close(fig)

    save_metrics("admissions_regression", {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "train_mse": mean_squared_error(y_train, pred_train), "test_mse": test_mse,
        "test_rmse": float(np.sqrt(test_mse)), "test_r2": r_squared(y_test, pred_test),
        "baseline_test_mse": baseline_mse, "bias": float(w[0]),
        "weights": dict(zip(names, (float(v) for v in w[1:]))),
        "standardized_weights": dict(zip(names, (float(v) for v in w_std))),
        "repeated_splits": {"n": int(len(repeats)), "mean_test_mse": float(repeats.mean()),
                            "std_test_mse": float(repeats.std()), "min": float(repeats.min()),
                            "max": float(repeats.max())},
    })
    print(f"admissions: test MSE {test_mse:.6f} (R^2 {r_squared(y_test, pred_test):.3f}, mean predictor "
          f"{baseline_mse:.6f}); over 200 splits {repeats.mean():.6f} +/- {repeats.std():.6f}")


if __name__ == "__main__":
    main()
