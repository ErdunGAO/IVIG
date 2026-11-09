"""
TripAdvisor-inspired binary-IV simulation benchmark.

Covariates are generated in a TripAdvisor-inspired feature space
(six day-count features, two binary features, a three-level categorical
feature and a clipped log-normal revenue feature); instrument, realised
treatment, outcome and ground-truth CATE come from a controlled binary-IV
mechanism (Appendix B.3.2 of the paper).  Two scenarios are exposed:

* `tripadvisor_standard`: source, pool, and test covariates are sampled from
  the same source distribution.
* `tripadvisor_shift`: source and pool covariates come from the source
  distribution, while test covariates come from a shifted target
  distribution.

Both scenarios use the same smooth branch-pocket mechanism.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.library.datasets.data_class import TestDataSet
from src.library.datasets.propensity import marginal_logistic_probability


SCENARIO_NAME_STANDARD = "tripadvisor_standard"
SCENARIO_NAME_SHIFT = "tripadvisor_shift"
BINARY_DGP_SCENARIOS = [SCENARIO_NAME_STANDARD, SCENARIO_NAME_SHIFT]

DEFAULT_D_COVARIATE = 10

_DAY_MAX = 28
_SCALE_DENOM = np.array(
    [28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 1.0, 1.0, 2.0, 5000.0],
    dtype=np.float32,
)


def is_binary_scenario(scenario: str) -> bool:
    return scenario in BINARY_DGP_SCENARIOS


def get_default_d_covariate(scenario: str) -> int:
    return DEFAULT_D_COVARIATE


def _check_scenario(scenario: str) -> None:
    if not is_binary_scenario(scenario):
        raise ValueError(
            f"Unknown semi-synthetic scenario '{scenario}'. Expected one of {BINARY_DGP_SCENARIOS}."
        )


def _validate_d_covariate(d_covariate: int | None) -> int:
    if d_covariate is None:
        return DEFAULT_D_COVARIATE
    if int(d_covariate) != DEFAULT_D_COVARIATE:
        raise ValueError(
            f"TripAdvisor semi-synthetic scenarios use exactly {DEFAULT_D_COVARIATE} covariates; "
            f"got d_covariate={d_covariate}."
        )
    return DEFAULT_D_COVARIATE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _fixed_normalize_features(x_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim == 1:
        return (x / _SCALE_DENOM).astype(np.float32)
    return (x / _SCALE_DENOM.reshape(1, -1)).astype(np.float32)


def _sample_source_covariates(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.empty((n, DEFAULT_D_COVARIATE), dtype=np.float32)
    x[:, 0:6] = rng.randint(0, _DAY_MAX + 1, size=(n, 6)).astype(np.float32)
    x[:, 6] = rng.binomial(1, 0.5, size=n).astype(np.float32)
    x[:, 7] = rng.binomial(1, 0.5, size=n).astype(np.float32)
    x[:, 8] = rng.choice([0.0, 1.0, 2.0], size=n).astype(np.float32)
    revenue = np.clip(
        np.round(rng.lognormal(mean=0.0, sigma=3.0, size=n), 2),
        0.0,
        5000.0,
    )
    x[:, 9] = revenue.astype(np.float32)
    return x


def _sample_target_covariates(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.empty((n, DEFAULT_D_COVARIATE), dtype=np.float32)
    x[:, 0] = np.clip(
        np.round(rng.normal(loc=15.0, scale=5.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 1] = np.clip(
        np.round(rng.normal(loc=13.0, scale=5.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 2] = np.clip(
        np.round(rng.normal(loc=10.0, scale=4.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 3] = np.clip(
        np.round(rng.normal(loc=8.0, scale=4.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 4] = np.clip(
        np.round(rng.normal(loc=9.0, scale=4.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 5] = np.clip(
        np.round(rng.normal(loc=7.0, scale=4.0, size=n)),
        0,
        _DAY_MAX,
    ).astype(np.float32)
    x[:, 6] = rng.binomial(1, 0.30, size=n).astype(np.float32)
    x[:, 7] = rng.binomial(1, 0.60, size=n).astype(np.float32)
    x[:, 8] = rng.choice([0.0, 1.0, 2.0], p=[0.25, 0.55, 0.20], size=n).astype(
        np.float32
    )
    revenue = np.clip(
        np.round(rng.lognormal(mean=0.1, sigma=2.8, size=n), 2),
        0.0,
        5000.0,
    )
    x[:, 9] = revenue.astype(np.float32)
    return x


def _sample_test_covariates(
    scenario: str,
    n: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if scenario == SCENARIO_NAME_SHIFT:
        return _sample_target_covariates(n, rng)
    return _sample_source_covariates(n, rng)


def _left_focus(x_raw: np.ndarray) -> np.ndarray:
    x = _fixed_normalize_features(np.asarray(x_raw, dtype=np.float32))
    if x.ndim == 1:
        x = x[None, :]
    return _sigmoid(-4.0 * (x[:, 1] - 0.43))


def _reward_pocket(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    return np.exp(
        -0.5 * ((x_raw[:, 0] - 16.0) / 5.1) ** 2
        -0.5 * ((x_raw[:, 1] - 12.0) / 5.1) ** 2
    ).astype(np.float32)


def _micro_pocket(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    return np.exp(
        -0.5 * ((x_raw[:, 0] - 16.0) / 2.6) ** 2
        -0.5 * ((x_raw[:, 1] - 12.0) / 2.8) ** 2
    ).astype(np.float32)


def _search_wave(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    x = _fixed_normalize_features(x_raw)
    wave = (
        np.sin(2.1 * np.pi * x[:, 2])
        + 0.55 * np.cos(1.7 * np.pi * x[:, 3])
        - 0.35 * np.sin(1.5 * np.pi * x[:, 4])
    )
    return np.tanh(wave).astype(np.float32)


def _mu0(x_raw: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = _fixed_normalize_features(x_raw)
    if x.ndim == 1:
        x = x[None, :]
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    reward_pocket = _reward_pocket(x_raw)
    return (
        0.18
        + 0.22 * x[:, 0]
        - 0.10 * x[:, 3]
        + 0.08 * np.asarray(x_raw, dtype=np.float32).reshape(-1, DEFAULT_D_COVARIATE)[:, 7]
        + 0.11 * u
        + 0.10 * reward_pocket * np.cos(np.pi * x[:, 4])
    ).astype(np.float32)


def _true_cate_base(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    single = x_raw.ndim == 1
    if single:
        x_raw = x_raw[None, :]
    x = _fixed_normalize_features(x_raw)
    reward_pocket = _reward_pocket(x_raw)
    micro_pocket = _micro_pocket(x_raw)
    left_focus = _left_focus(x_raw)
    right_focus = (1.0 - left_focus).astype(np.float32)
    local_wave = (
        1.25 * np.sin(2.6 * np.pi * x[:, 2])
        + 0.85 * np.cos(2.2 * np.pi * x[:, 3])
        + 0.65 * np.sin(1.9 * np.pi * x[:, 4] + 0.4 * x[:, 5])
    )
    local_wave = np.tanh(local_wave).astype(np.float32)
    tau = (
        0.36
        + 0.04 * np.sin(np.pi * x[:, 1])
        - 0.06 * x_raw[:, 6]
        + 1.65 * reward_pocket * left_focus * (0.55 + 0.95 * local_wave)
        - 0.58 * reward_pocket * right_focus * np.cos(2.0 * np.pi * x[:, 4])
        + 0.42 * micro_pocket * left_focus * np.sin(3.0 * np.pi * x[:, 2])
    ).astype(np.float32)
    return tau[0] if single else tau


def _true_cate(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    single = x_raw.ndim == 1
    if single:
        x_raw = x_raw[None, :]
    x = _fixed_normalize_features(x_raw)
    baseline = (
        0.36
        + 0.04 * np.sin(np.pi * x[:, 1])
        - 0.06 * x_raw[:, 6]
    ).astype(np.float32)
    local = (_true_cate_base(x_raw).astype(np.float32) - baseline).astype(np.float32)
    tau = (baseline + 1.75 * local).astype(np.float32)
    return tau[0] if single else tau


def true_cate_raw(
    x_raw: np.ndarray,
    scenario: str = SCENARIO_NAME_STANDARD,
) -> np.ndarray:
    _check_scenario(scenario)
    return _true_cate(x_raw)


def _compliance_logits(x_raw: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    x = _fixed_normalize_features(x_raw)
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    reward_pocket = _reward_pocket(x_raw)
    micro_pocket = _micro_pocket(x_raw)
    left_focus = _left_focus(x_raw)
    branch_wave = _search_wave(x_raw)

    mid_logit = (
        -1.28
        + 0.16 * (x[:, 0] - 0.5)
        + 0.09 * u
        + 0.50 * reward_pocket * left_focus
    )
    separation = 1.28 * (
        0.52
        + 1.45 * reward_pocket * left_focus
        + 0.48 * micro_pocket * (1.0 + branch_wave)
    )
    logit_p0 = mid_logit - separation
    logit_p1 = mid_logit + separation
    return logit_p0, logit_p1


def _compliance_probabilities(x_raw: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logit_p0, logit_p1 = _compliance_logits(x_raw, u)
    p0 = np.clip(_sigmoid(logit_p0), 0.04, 0.88)
    p1 = np.clip(_sigmoid(logit_p1), 0.04, 0.88)
    return p0.astype(np.float32), p1.astype(np.float32)


def marginal_propensities(x_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """P(T=1 | X=x, Z=z) for both instruments, integrating out U."""
    mean0, mean1 = _compliance_logits(x_raw, np.zeros(1, dtype=np.float32))
    return tuple(
        marginal_logistic_probability(mean, 0.09, lower=0.04, upper=0.88).astype(np.float32)
        for mean in (mean0, mean1)
    )


def _generate_pool_outcomes(
    x_raw: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n = x_raw.shape[0]
    u = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    p0, p1 = _compliance_probabilities(x_raw, u)

    t0 = rng.binomial(1, p0).astype(np.float32)
    t1 = rng.binomial(1, p1).astype(np.float32)

    tau = _true_cate(x_raw).astype(np.float32)
    mu0 = _mu0(x_raw, u)
    eps0 = rng.normal(0.0, 0.20, size=n).astype(np.float32)
    eps1 = rng.normal(0.0, 0.16, size=n).astype(np.float32)
    y0_potential = mu0 + eps0
    y1_potential = mu0 + tau + eps1

    z0_y = np.where(t0 > 0.5, y1_potential, y0_potential).astype(np.float32)
    z1_y = np.where(t1 > 0.5, y1_potential, y0_potential).astype(np.float32)
    marginal_p0, marginal_p1 = marginal_propensities(x_raw)

    z0 = {
        "T": t0.reshape(-1, 1),
        "Y": z0_y.reshape(-1, 1),
        "P": marginal_p0.reshape(-1, 1),
    }
    z1 = {
        "T": t1.reshape(-1, 1),
        "Y": z1_y.reshape(-1, 1),
        "P": marginal_p1.reshape(-1, 1),
    }
    return z0, z1


def generate_binary_pool(
    scenario: str,
    n_pool: int = 800,
    d_covariate: int = DEFAULT_D_COVARIATE,
    n_test: int = 1000,
    seed: int | None = None,
):
    _check_scenario(scenario)
    _validate_d_covariate(d_covariate)

    rng = np.random.RandomState(seed)
    x_pool_raw = _sample_source_covariates(n_pool, rng)
    x_pool = _fixed_normalize_features(x_pool_raw)
    z0_outcomes, z1_outcomes = _generate_pool_outcomes(x_pool_raw, rng)

    x_test_raw = _sample_test_covariates(scenario, n_test, rng)
    x_test = _fixed_normalize_features(x_test_raw)
    true_cate = true_cate_raw(x_test_raw, scenario=scenario).reshape(-1, 1).astype(np.float32)
    test_data = TestDataSet(
        treatment=np.zeros((n_test, 1), dtype=np.float32),
        covariate=x_test,
        structural=true_cate,
    )
    return x_pool.astype(np.float32), z0_outcomes, z1_outcomes, test_data
