"""Error rate of a majority vote of independent classifiers."""
from math import comb

import numpy as np


def ensemble_error(epsilon, n_classifiers):
    """Probability that the majority vote of N independent classifiers is wrong.

    Each classifier is wrong with probability epsilon, independently of the others.
    With N odd the vote is wrong exactly when at least (N + 1) / 2 classifiers are wrong,
    so the error rate is the upper tail of a Binomial(N, epsilon) distribution:

        sum_{k = (N + 1) / 2}^{N}  C(N, k) epsilon^k (1 - epsilon)^(N - k)
    """
    assert n_classifiers % 2 == 1, "N must be odd so that the vote cannot tie."
    assert 0.0 <= epsilon <= 1.0
    first_losing = (n_classifiers + 1) // 2
    return float(sum(comb(n_classifiers, k) * epsilon ** k * (1.0 - epsilon) ** (n_classifiers - k)
                     for k in range(first_losing, n_classifiers + 1)))


def simulate_ensemble_error(epsilon, n_classifiers, trials=200_000, seed=0):
    """Estimate the same probability by drawing the classifiers' mistakes at random."""
    rng = np.random.default_rng(seed)
    wrong = rng.random((trials, n_classifiers)) < epsilon
    return float(np.mean(wrong.sum(axis=1) > n_classifiers / 2))
