"""Marginal compliance probabilities with Gaussian latent confounding."""

from functools import lru_cache

import numpy as np
from scipy.special import expit, logit, ndtr


@lru_cache(maxsize=1)
def _quadrature():
    return np.polynomial.legendre.leggauss(64)


def marginal_logistic_probability(mean, latent_scale, lower=0.0, upper=1.0):
    """E[clip(sigmoid(mean + latent_scale * U), lower, upper)], U ~ N(0,1).

    Integrate the smooth part by Gauss-Legendre quadrature and the clipped
    tails analytically. Truncation at |U| = 10 has probability below 2e-23.
    This deterministic calculation does not consume simulation randomness.
    """
    if latent_scale <= 0 or not 0 <= lower < upper <= 1:
        raise ValueError("Require positive latent scale and 0 <= lower < upper <= 1.")
    mean = np.asarray(mean, dtype=np.float64)
    a = np.clip((logit(lower) - mean) / latent_scale, -10.0, 10.0)
    b = np.clip((logit(upper) - mean) / latent_scale, -10.0, 10.0)
    nodes, weights = _quadrature()
    half = (b - a) / 2
    u = (a + b)[..., None] / 2 + half[..., None] * nodes
    density = np.exp(-u**2 / 2) / np.sqrt(2 * np.pi)
    middle = half * np.sum(weights * density * expit(mean[..., None] + latent_scale * u), axis=-1)
    return lower * ndtr(a) + middle + upper * ndtr(-b)
