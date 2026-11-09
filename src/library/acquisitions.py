"""Acquisition rules for active CATE estimation under noncompliance.

Every acquisition has the signature

    fn(trainer, pool_x, n_acquire, target_covariate=None, **kwargs)
        -> (local_idx, z_assign)

where ``trainer`` is the most recently fitted estimator (``trainer.mdl`` is a
``GPIVModel``; ``None`` before the first fit), ``pool_x`` holds the covariates
of the units still in the pool, ``target_covariate`` is the unlabeled target
reference set used to approximate expectations over p_tar, and the returned
``local_idx`` indexes ``pool_x``.  ``z_assign`` is the binary instrument
assigned to each selected unit.  The registry ``FUNCTIONS`` maps names to
callables.

Three families live here:

* **Passive / local-uncertainty baselines** -- ``random``, ``mu_bald``
  (observable-outcome BALD at the plug-in propensity feature), ``tau_bald``
  (posterior CATE variance), ``mu_rho_bald`` (causal-BALD style combination)
  and ``gamma`` (local decision uncertainty, inspired by Sundin et al.).
  These score each remaining x under both z = 0 and z = 1, keep the better
  instrument per unit, and take the top-scoring batch.

* **IVIG** -- ``ivig`` is the empirical-Bayes instrumental-variable
  information gain of the paper: for every candidate (x, z) the score
  is the expected reduction in posterior entropy of tau(x*) over the target
  reference set, averaged over the two treatment-realisation branches with
  plug-in branch probabilities and branch-specific fantasy updates.

* **IVIG diagnostics** -- ``y_ivig`` (outcome-only comparator that integrates
  the realised treatment out; Theorem 1), ``no_stage1_ivig`` (treats the
  assigned instrument as if it were the treatment), ``oracle_ivig`` (uses the
  marginal compliance probabilities inside the score) and ``krr_mc_ivig``
  (score averaging over ``mc_particles`` first-stage working-posterior
  particles; one particle recovers ``ivig`` exactly).

The ``indpolicy`` baseline (learned instrument policy) is not a scoring rule
and is handled directly by the active-learning workflow.
"""

import numpy as np
from scipy import stats

_EPS = 1e-7
_Z_GRID = np.asarray([0.0, 1.0], dtype=float)


# =============================================================================
# Helpers
# =============================================================================
def _fitted_model(trainer):
    if trainer is None or getattr(trainer, "mdl", None) is None:
        return None
    return trainer.mdl


def random_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Uniform unit selection with Bernoulli(0.5) instruments (passive design)."""
    n_pool = pool_x.shape[0]
    idx = np.random.choice(n_pool, size=n_acquire, replace=False)
    z = np.random.choice(_Z_GRID, size=n_acquire, replace=True)
    return idx, z


def _top_k(scores, n_select):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_select = min(int(n_select), scores.shape[0])
    if n_select <= 0:
        return np.array([], dtype=int)
    top = np.argpartition(-scores, n_select - 1)[:n_select]
    return top[np.argsort(-scores[top])].astype(int)


def _best_instrument(score_z0, score_z1):
    """Per-unit best instrument; ties are broken in favour of z = 0."""
    score_z0 = np.asarray(score_z0, dtype=float).reshape(-1)
    score_z1 = np.asarray(score_z1, dtype=float).reshape(-1)
    best_z = (score_z1 > score_z0).astype(float)
    best_score = np.where(best_z > 0.5, score_z1, score_z0)
    return best_score, best_z


def _select(score_z0, score_z1, n_acquire):
    best_score, best_z = _best_instrument(score_z0, score_z1)
    top_idx = _top_k(best_score, n_acquire)
    return top_idx, best_z[top_idx]


def _draw_target_subsets(pool_x, target_covariate, target_mc=100, target_repeats=1):
    """Monte Carlo subsets of the target reference set used for the p_tar expectation."""
    base = pool_x if target_covariate is None else np.asarray(target_covariate)
    n_available = base.shape[0]
    n_mc = min(int(target_mc), n_available)
    repeats = max(int(target_repeats or 1), 1)
    if n_available == 0 or n_mc <= 0:
        return []
    if n_mc >= n_available:
        return [np.asarray(base, dtype=float) for _ in range(repeats)]
    return [
        np.asarray(base[np.random.choice(n_available, size=n_mc, replace=False)], dtype=float)
        for _ in range(repeats)
    ]


def _oracle_branch_probabilities(kwargs):
    """True P(T=1 | x, z) for the remaining pool, if the environment is available."""
    env = kwargs.get("env")
    pool_global_idx = kwargs.get("pool_global_idx")
    if env is None or pool_global_idx is None:
        return None, None
    idx = np.asarray(pool_global_idx, dtype=int).reshape(-1)
    p0 = env.get_oracle_propensity(idx, np.zeros(idx.shape[0])).reshape(-1)
    p1 = env.get_oracle_propensity(idx, np.ones(idx.shape[0])).reshape(-1)
    return p0, p1


def _mu_bald_by_z(model, pool_x):
    return np.vstack([
        np.asarray(model.sanitized_bald_scores(pool_x, float(z)), dtype=float).reshape(-1)
        for z in _Z_GRID
    ])


# =============================================================================
# Local-uncertainty baselines
# =============================================================================
def mu_bald_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """mu-BALD: observable-outcome information gain at the plug-in propensity feature."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    score_by_z = _mu_bald_by_z(model, pool_x)
    return _select(score_by_z[0], score_by_z[1], n_acquire)


def tau_bald_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """tau-BALD: rank x by posterior CATE variance; assign z by the mu-BALD rule."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    _, cate_var = model.predict_cate_with_variance(pool_x)
    top_idx = _top_k(np.asarray(cate_var, dtype=float).reshape(-1), n_acquire)
    score_by_z = _mu_bald_by_z(model, pool_x)
    _, best_z = _best_instrument(score_by_z[0, top_idx], score_by_z[1, top_idx])
    return top_idx, best_z


def mu_rho_bald_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """mu-rho-BALD: 0.5*log Var[tau(x)] + mu-BALD(x, z) - mu-BALD(x, 1-z)."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    _, cate_var = model.predict_cate_with_variance(pool_x)
    tau_score = 0.5 * np.log(np.clip(np.asarray(cate_var, dtype=float).reshape(-1), _EPS, None))
    mu_by_z = _mu_bald_by_z(model, pool_x)
    score_z0 = tau_score + mu_by_z[0] - mu_by_z[1]
    score_z1 = tau_score + mu_by_z[1] - mu_by_z[0]
    return _select(score_z0, score_z1, n_acquire)


def gamma_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Local decision-uncertainty heuristic inspired by Sundin et al."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    cate_mean, cate_var = model.predict_cate_with_variance(pool_x)
    mean = np.asarray(cate_mean, dtype=float).reshape(-1)
    std = np.sqrt(np.clip(np.asarray(cate_var, dtype=float).reshape(-1), _EPS, None))
    n_mc = max(int(kwargs.get("gamma_mc_samples", 128)), 8)
    tau_samples = mean[None, :] + std[None, :] * np.random.randn(n_mc, mean.shape[0])
    gammas = np.clip(stats.norm().cdf(-np.abs(tau_samples) / np.sqrt(2.0)), _EPS, 1.0 - _EPS)
    gamma = np.clip(gammas.mean(0), _EPS, 1.0 - _EPS)
    score = stats.bernoulli(gamma).entropy() - stats.bernoulli(gammas).entropy().mean(0)
    top_idx = _top_k(np.maximum(score, 0.0), n_acquire)
    score_by_z = _mu_bald_by_z(model, pool_x)
    _, best_z = _best_instrument(score_by_z[0, top_idx], score_by_z[1, top_idx])
    return top_idx, best_z


# =============================================================================
# IVIG
# =============================================================================
def _ivig_core(trainer, pool_x, n_acquire, target_covariate=None, *,
               branch_p0=None, branch_p1=None, mc_particles=None, **kwargs):
    """
    Shared EB-IVIG scorer.  For every target draw and every z in {0, 1} it
    calls ``GPIVModel.phantom_ivig_scores`` (or the particle variant), averages
    over target draws, keeps the best instrument per unit and returns the
    top-scoring batch.  ``branch_p0``/``branch_p1`` optionally override the
    plug-in first-stage branch probabilities (oracle diagnostics).
    """
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)

    target_draws = _draw_target_subsets(
        pool_x, target_covariate,
        target_mc=kwargs.get("target_mc", 100),
        target_repeats=kwargs.get("target_repeats", 1),
    )
    if not target_draws:
        raise ValueError("IVIG requires at least one target covariate draw.")
    chunk_size = int(kwargs.get("chunk_size", 128))

    score_by_z = np.zeros((2, pool_x.shape[0]), dtype=float)
    for target_x in target_draws:
        _, prior_var = model.predict_cate_with_variance(target_x)
        for z_pos, z_val in enumerate(_Z_GRID):
            branch_prob = branch_p0 if z_pos == 0 else branch_p1
            if mc_particles is not None and int(mc_particles) > 1:
                score_by_z[z_pos] += model.krr_particle_ivig_scores(
                    pool_x, float(z_val), target_x,
                    prior_cate_var=prior_var, branch_prob=branch_prob,
                    n_particles=int(mc_particles), chunk_size=chunk_size,
                )
            else:
                score_by_z[z_pos] += model.phantom_ivig_scores(
                    pool_x, float(z_val), target_x,
                    prior_cate_var=prior_var, branch_prob=branch_prob,
                    chunk_size=chunk_size,
                )
    score_by_z /= float(len(target_draws))
    return _select(score_by_z[0], score_by_z[1], n_acquire)


def ivig_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """EB-IVIG: target-averaged, branch-aware information gain about tau."""
    kwargs.pop("mc_particles", None)
    return _ivig_core(trainer, pool_x, n_acquire, target_covariate, mc_particles=None, **kwargs)


def krr_mc_ivig_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Diagnostic averaging IVIG scores over first-stage KRR particles."""
    particles = int(kwargs.pop("mc_particles", 4))
    return _ivig_core(trainer, pool_x, n_acquire, target_covariate, mc_particles=particles, **kwargs)


def oracle_ivig_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Diagnostic: IVIG scored with the true compliance probabilities."""
    kwargs.pop("mc_particles", None)
    p0, p1 = _oracle_branch_probabilities(kwargs)
    return _ivig_core(trainer, pool_x, n_acquire, target_covariate,
                      branch_p0=p0, branch_p1=p1, mc_particles=None, **kwargs)


def no_stage1_ivig_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Diagnostic: information gain that treats the assigned instrument as the treatment."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    target_draws = _draw_target_subsets(
        pool_x, target_covariate,
        target_mc=kwargs.get("target_mc", 100),
        target_repeats=kwargs.get("target_repeats", 1),
    )
    score_by_z = np.zeros((2, pool_x.shape[0]), dtype=float)
    for target_x in target_draws:
        _, prior_var = model.predict_cate_with_variance(target_x)
        for z_pos, z_val in enumerate(_Z_GRID):
            score_by_z[z_pos] += model.direct_tau_ig_scores(
                pool_x, float(z_val), target_x, prior_cate_var=prior_var,
            )
    score_by_z /= float(len(target_draws))
    return _select(score_by_z[0], score_by_z[1], n_acquire)


def y_ivig_acquisition(trainer, pool_x, n_acquire, target_covariate=None, **kwargs):
    """Outcome-only comparator E_{x*}[ I(y; tau(x*) | x, z, H) ] (realised t integrated out)."""
    model = _fitted_model(trainer)
    if model is None:
        return random_acquisition(trainer, pool_x, n_acquire)
    target_draws = _draw_target_subsets(
        pool_x, target_covariate,
        target_mc=kwargs.get("target_mc", 100),
        target_repeats=kwargs.get("target_repeats", 1),
    )
    score_by_z = np.zeros((2, pool_x.shape[0]), dtype=float)
    for target_x in target_draws:
        for z_pos, z_val in enumerate(_Z_GRID):
            score_by_z[z_pos] += model.y_ivig_scores(
                pool_x, float(z_val), target_x,
                chunk_size=int(kwargs.get("y_ivig_chunk_size", 64)),
                y_quad_nodes=int(kwargs.get("y_quad_nodes", 5)),
                tau_quad_nodes=int(kwargs.get("tau_quad_nodes", 9)),
            )
    score_by_z /= float(len(target_draws))
    return _select(score_by_z[0], score_by_z[1], n_acquire)


# =============================================================================
# Registry
# =============================================================================
FUNCTIONS = {
    "random": random_acquisition,
    "mu_bald": mu_bald_acquisition,
    "tau_bald": tau_bald_acquisition,
    "mu_rho_bald": mu_rho_bald_acquisition,
    "gamma": gamma_acquisition,
    "ivig": ivig_acquisition,
    "y_ivig": y_ivig_acquisition,
    "no_stage1_ivig": no_stage1_ivig_acquisition,
    "oracle_ivig": oracle_ivig_acquisition,
    "krr_mc_ivig": krr_mc_ivig_acquisition,
}

# Handled by the workflow rather than a scoring function.
POLICY_ACQUISITIONS = {"indpolicy"}
