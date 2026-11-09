"""
Synthetic binary-IV active-design benchmarks (Appendix B.3.1 of the paper).

Four target-aligned scenarios share one data-generating family

    x ~ N(0, I_d),  z ~ Bernoulli(0.5),  u, eps ~ N(0, 1)
    t ~ Bernoulli(sigmoid(l_s(x, z, u))),   y = tau_s(x) t + h_s(u, x) + sigma_s eps

and differ in the compliance logit ``l_s``, the structural CATE ``tau_s``, the
additive residual ``h_s`` and the noise level ``sigma_s``:

* ``sharp``        -- Localized Heterogeneity (canonical target-aligned benchmark)
* ``frontier``     -- Frontier Heterogeneity (informative region sits on the
                      compliance frontier)
* ``confounding``  -- Confounding-Dominant (stronger t-u interaction in h)
* ``multiscale``   -- Multiscale Heterogeneity (coarse + fine local structure)

The candidate pool is always drawn from the source distribution N(0, I_d).
The *target / evaluation* population is controlled by ``target_shift`` in
[0, 1]: 0 means the target equals the source distribution (the "standard"
setting in the paper), 0.5 is the paper's "target-shift" setting, and 1
concentrates the target fully on the pocket
x_1 ~ N(0, 0.32^2), x_2 ~ N(1.0, 0.55^2).

Scenario strings accepted everywhere: ``<name>`` or ``<name>_tshift_<pct>``
e.g. ``sharp``, ``sharp_tshift_050``, ``multiscale_tshift_100``.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

from src.library.datasets.data_class import TestDataSet
from src.library.datasets.propensity import marginal_logistic_probability

DEFAULT_D_COVARIATE = 5
BASE_SCENARIOS = ("sharp", "frontier", "confounding", "multiscale")

_SCENARIO_RE = re.compile(r"^(sharp|frontier|confounding|multiscale)(?:_tshift_(\d{3}))?$")


# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------
def parse_scenario(scenario: str) -> Tuple[str, float]:
    """Return ``(base_name, target_shift)`` for a scenario string."""
    m = _SCENARIO_RE.match(str(scenario))
    if m is None:
        raise ValueError(
            f"Unknown synthetic scenario '{scenario}'. Expected one of {BASE_SCENARIOS} "
            "optionally suffixed with '_tshift_<pct>' (e.g. 'sharp_tshift_050')."
        )
    base = m.group(1)
    shift = 0.0 if m.group(2) is None else float(m.group(2)) / 100.0
    return base, float(np.clip(shift, 0.0, 1.0))


def make_scenario(base: str, target_shift: float) -> str:
    if base not in BASE_SCENARIOS:
        raise ValueError(f"Unknown base scenario '{base}'. Choose from {BASE_SCENARIOS}.")
    shift = float(np.clip(target_shift, 0.0, 1.0))
    if shift <= 0.0:
        return base
    return f"{base}_tshift_{int(round(shift * 100)):03d}"


def is_binary_scenario(scenario: str) -> bool:
    return _SCENARIO_RE.match(str(scenario)) is not None


def get_default_d_covariate(scenario: str) -> int:
    return DEFAULT_D_COVARIATE


# ---------------------------------------------------------------------------
# Structural pieces
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _local_gate(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x[:, 0:1] ** 2) / 0.12) * _sigmoid(1.2 * (x[:, 1:2] - 0.2))


def _ridge_gate(x: np.ndarray) -> np.ndarray:
    ridge = np.exp(-(x[:, 0:1] ** 2) / (2.0 * 0.45 ** 2))
    gate = _sigmoid(1.0 * (x[:, 1:2] - 0.1))
    return ridge * gate


def _sharp_local_cate(x: np.ndarray) -> np.ndarray:
    return (
        4.2 * np.sin(2.4 * x[:, 4:5])
        + 2.8 * x[:, 1:2] * x[:, 4:5]
        + 2.2 * np.cos(3.0 * x[:, 0:1])
        + 1.5 * np.tanh(x[:, 2:3])
    )


def cate_function(scenario: str, x: np.ndarray) -> np.ndarray:
    """Structural CATE tau_s(x)."""
    base, _ = parse_scenario(scenario)
    offset = 0.7 + 0.2 * x[:, 1:2] + 0.15 * x[:, 2:3] * x[:, 3:4]
    gate = _local_gate(x)
    if base == "sharp":
        local = _sharp_local_cate(x)
    elif base == "frontier":
        local = (
            3.8 * np.sin(2.0 * x[:, 4:5])
            + 2.2 * x[:, 1:2] * x[:, 4:5]
            + 2.0 * np.cos(2.7 * x[:, 0:1])
            + 1.1 * np.tanh(x[:, 2:3])
        )
    elif base == "confounding":
        local = (
            4.0 * np.sin(2.3 * x[:, 4:5])
            + 2.5 * x[:, 1:2] * x[:, 4:5]
            + 2.0 * np.cos(2.8 * x[:, 0:1])
            + 1.3 * np.tanh(x[:, 2:3])
        )
    else:  # multiscale
        local = (
            2.7 * np.sin(1.4 * x[:, 4:5])
            + 1.9 * np.sin(3.1 * x[:, 4:5])
            + 1.8 * x[:, 1:2] * x[:, 4:5]
            + 1.8 * np.cos(2.6 * x[:, 0:1])
            + 1.0 * np.sin(1.7 * x[:, 2:3] * x[:, 3:4])
        )
    return (offset + gate * local).astype(np.float32)


def iv_strength(scenario: str, x: np.ndarray) -> np.ndarray:
    """Local instrument strength s(x) (first-stage coefficient on z)."""
    base, _ = parse_scenario(scenario)
    rg = _ridge_gate(x)
    if base == "sharp":
        strength = 0.25 + 1.8 * rg
    elif base == "frontier":
        frontier = 4.0 * rg * (1.0 - rg)
        strength = 0.2 + 1.5 * rg + 0.35 * frontier
    elif base == "confounding":
        strength = 0.25 + 1.75 * rg
    else:  # multiscale
        strength = 0.25 + 1.7 * rg
    return strength.astype(np.float32)


def compliance_logit(scenario: str, z: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compliance logit l_s(x, z, u); t ~ Bernoulli(sigmoid(l_s))."""
    base, _ = parse_scenario(scenario)
    iv_term = iv_strength(scenario, x) * z
    if base == "sharp":
        logit = iv_term + 0.9 * u + 0.12 * x[:, 0:1] + 0.10 * x[:, 1:2]
    elif base == "frontier":
        logit = iv_term + 1.0 * u + 0.10 * x[:, 0:1] + 0.08 * x[:, 1:2] - 0.35 * _local_gate(x)
    elif base == "confounding":
        logit = iv_term + 0.9 * u + 0.12 * x[:, 0:1] + 0.10 * x[:, 1:2]
    else:  # multiscale
        logit = iv_term + 0.92 * u + 0.12 * x[:, 0:1] + 0.10 * x[:, 1:2]
    return logit.astype(np.float32)


def confounding_term(scenario: str, t: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Additive residual h_s(u, x) (does not depend on t in the main benchmarks)."""
    base, _ = parse_scenario(scenario)
    g = _local_gate(x)
    if base == "sharp":
        term = 0.3 * u + 0.05 * u * g + 0.02 * (u ** 2) * g
    elif base == "frontier":
        term = 0.28 * u + 0.04 * u * g + 0.015 * (u ** 2) * g
    elif base == "confounding":
        term = 0.3 * u + 0.11 * u * g + 0.05 * (u ** 2) * g + 0.05 * x[:, 0:1] * u * g
    else:  # multiscale
        term = 0.28 * u + 0.05 * u * g + 0.02 * (u ** 2) * g
    return term.astype(np.float32)


def marginal_propensity(scenario: str, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(T=1 | X=x, Z=z), integrating out the latent standard-normal U."""
    base, _ = parse_scenario(scenario)
    scale = {"sharp": 0.9, "frontier": 1.0, "confounding": 0.9, "multiscale": 0.92}[base]
    mean = compliance_logit(scenario, z, np.zeros((len(x), 1), dtype=np.float32), x)
    return marginal_logistic_probability(mean, scale).astype(np.float32)


def noise_scale(scenario: str) -> float:
    base, _ = parse_scenario(scenario)
    return 0.24 if base == "multiscale" else 0.22


def sample_target_covariates(
    scenario: str,
    n_samples: int,
    d_covariate: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Target / evaluation covariates, interpolated from N(0, I) toward the pocket."""
    _, shift = parse_scenario(scenario)
    x = rng.randn(n_samples, d_covariate).astype(np.float32)
    if shift > 0.0:
        x[:, 0:1] = rng.normal(
            loc=0.0,
            scale=(1.0 - shift) * 1.0 + shift * 0.32,
            size=(n_samples, 1),
        ).astype(np.float32)
        x[:, 1:2] = rng.normal(
            loc=shift * 1.0,
            scale=(1.0 - shift) * 1.0 + shift * 0.55,
            size=(n_samples, 1),
        ).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# Pool generation for the active-design environment
# ---------------------------------------------------------------------------
def _sample_test_data(scenario: str, d_covariate: int, n_test: int, seed: int) -> TestDataSet:
    # Draw order mirrors the original experiment code so that seeds reproduce
    # the paper's target reference sets exactly.
    rng = np.random.RandomState(seed)
    z = rng.binomial(1, 0.5, (1, 1)).astype(np.float32)
    u = rng.randn(1, 1).astype(np.float32)
    x = rng.randn(1, d_covariate).astype(np.float32)
    _ = rng.randn(1, 1).astype(np.float32)
    prob_t = _sigmoid(compliance_logit(scenario, z, u, x))
    _ = rng.binomial(1, prob_t)

    x_test = sample_target_covariates(scenario, n_test, d_covariate, rng)
    true_cate = cate_function(scenario, x_test)
    return TestDataSet(
        treatment=np.zeros((n_test, 1), dtype=np.float32),
        covariate=x_test,
        structural=true_cate,
    )


def generate_binary_pool(
    scenario: str,
    n_pool: int = 1000,
    d_covariate: int | None = None,
    n_test: int = 1000,
    seed: int | None = None,
):
    """
    Pre-generate a candidate pool together with its potential feedback under
    z = 0 and z = 1.

    Returns ``(x_pool, z0_outcomes, z1_outcomes, test_data)`` where each
    ``z*_outcomes`` dict holds ``T`` (realised treatment), ``Y`` (outcome) and
    ``P`` (marginal compliance probability given x and z) for every pool unit.
    """
    parse_scenario(scenario)
    if d_covariate is None:
        d_covariate = DEFAULT_D_COVARIATE
    seed = 0 if seed is None else int(seed)
    rng = np.random.RandomState(seed)

    u = rng.randn(n_pool, 1).astype(np.float32)
    x = rng.randn(n_pool, d_covariate).astype(np.float32)
    eps_y = rng.randn(n_pool, 1).astype(np.float32)

    tau_x = cate_function(scenario, x)
    sigma = noise_scale(scenario)

    def _responses(z_val: float) -> Dict[str, np.ndarray]:
        z = np.full((n_pool, 1), z_val, dtype=np.float32)
        prob_t = _sigmoid(compliance_logit(scenario, z, u, x))
        t = rng.binomial(1, prob_t).astype(np.float32).reshape(-1, 1)
        h = confounding_term(scenario, t, u, x)
        y = tau_x * t + h + sigma * eps_y
        return {"T": t, "Y": y.astype(np.float32), "P": marginal_propensity(scenario, z, x)}

    z0_outcomes = _responses(0.0)
    z1_outcomes = _responses(1.0)
    test_data = _sample_test_data(scenario, d_covariate, n_test, seed + 10000)
    return x, z0_outcomes, z1_outcomes, test_data
