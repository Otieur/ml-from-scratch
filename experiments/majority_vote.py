"""Error rate of a majority vote of independent classifiers: python -m experiments.majority_vote"""
import matplotlib.pyplot as plt

from experiments.common import OUT, save_metrics
from cscc11.voting import ensemble_error, simulate_ensemble_error


def main():
    sizes = list(range(1, 42, 2))
    rates = (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    table = {}
    for epsilon in rates:
        errors = [ensemble_error(epsilon, n) for n in sizes]
        table[str(epsilon)] = {str(n): e for n, e in zip(sizes, errors) if n in (1, 3, 5, 11, 21, 41)}
        ax.plot(sizes, errors, label=f"error rate of each classifier = {epsilon}")
        checks = [5, 21]
        ax.scatter(checks, [simulate_ensemble_error(epsilon, n, trials=100_000, seed=n) for n in checks],
                   color="black", s=14, zorder=3)
    ax.set(xlabel="Number of independent classifiers N", ylabel="Error rate of the majority vote",
           title="Lines: formula. Dots: simulation with 100,000 votes.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "majority_vote.png", dpi=140)
    plt.close(fig)
    save_metrics("majority_vote", {"ensemble_error": table})
    print(f"majority vote: error rate 0.3 -> {ensemble_error(0.3, 5):.3f} with 5 classifiers, "
          f"{ensemble_error(0.3, 21):.4f} with 21")


if __name__ == "__main__":
    main()
