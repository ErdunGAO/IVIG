"""Sequential indirect experimental design loop (Algorithm 1 in the paper)."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.library import acquisitions, datasets
from src.library.metrics import metric_record
from src.application.workflows import utils

logger = logging.getLogger(__name__)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def active_learner(config, experiment_dir, trial):
    """
    Run one active-design trial.

    ``experiment_dir`` / ``<acquisition>`` / ``trial-XXX`` receives
    ``config.json``, ``active_results.json`` (per-step sqrt-PEHE and timings)
    and ``acquisition_trace.json`` (the queried units, instruments and feedback).
    """
    seed = int(config["seed"]) + int(trial)
    config = dict(config)
    config["seed_with_trial"] = seed
    utils.set_seed(seed)

    acq_name = config["acquisition_function"]
    uses_policy = acq_name in acquisitions.POLICY_ACQUISITIONS
    if not uses_policy and acq_name not in acquisitions.FUNCTIONS:
        raise ValueError(
            f"Unknown acquisition '{acq_name}'. Choose from "
            f"{sorted(acquisitions.FUNCTIONS) + sorted(acquisitions.POLICY_ACQUISITIONS)}."
        )
    acq_fn = None if uses_policy else acquisitions.FUNCTIONS[acq_name]

    # Environment / pool / target reference set
    sim = datasets.get_simulation_module(config["dataset_name"])
    simulation_kwargs = {}
    if config["dataset_name"] == "ihdp":
        # Pass the resolved path explicitly, including to reused joblib workers.
        config["ihdp_csv"] = str(sim.resolve_data_path(config.get("ihdp_csv")))
        simulation_kwargs["csv_path"] = config["ihdp_csv"]
    env = datasets.IVEnvironment(
        sim,
        config["scenario"],
        n_pool=config["n_pool"],
        d_covariate=config.get("d_covariate"),
        n_test=config["n_test"],
        seed=seed,
        **simulation_kwargs,
    )
    pool = datasets.ActiveIVDataset(env.x_pool)
    test_data = env.get_test_data()

    trial_dir = Path(experiment_dir) / acq_name / f"trial-{trial:03d}"
    if (trial_dir / "active_results.json").exists():
        raise FileExistsError(f"Results already exist in {trial_dir}; choose a new --job-dir.")
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "config.json").open("w") as fp:
        json.dump(config, fp, indent=2, default=str)
    with (trial_dir / "dataset_diagnostics.json").open("w") as fp:
        json.dump(env.diagnostics(), fp, indent=2)

    warm_start_size = int(config["warm_start_size"])
    step_size = int(config["step_size"])
    max_acquisitions = int(config["max_acquisitions"])
    acq_kwargs = {
        "target_mc": int(config.get("target_mc", 100)),
        "target_repeats": int(config.get("target_repeats", 1)),
        "mc_particles": int(config.get("mc_particles", 4)),
    }

    # A dedicated RNG stream for acquisition keeps batch selection
    # reproducible regardless of how much randomness model training consumes.
    acquisition_rng = np.random.RandomState(seed)

    def _with_acquisition_rng(fn):
        prev = np.random.get_state()
        np.random.set_state(acquisition_rng.get_state())
        try:
            return fn()
        finally:
            acquisition_rng.set_state(np.random.get_state())
            np.random.set_state(prev)

    trainer = None
    policy_trainer = None
    results, trace = [], []

    for step in range(max_acquisitions):
        n_to_acquire = warm_start_size if step == 0 else step_size
        pool_x = pool.get_pool_x()
        if pool_x.shape[0] < n_to_acquire:
            logger.warning("Pool exhausted after %d steps.", step)
            break

        # ---- 1. choose (x, z) queries
        _sync_cuda()
        t0 = time.perf_counter()
        if step == 0:
            # Every method starts with the same units and instrument draws.
            local_idx, z_assign = _with_acquisition_rng(lambda: acquisitions.random_acquisition(
                None, pool_x, n_to_acquire,
            ))
        elif uses_policy:
            def _policy_acquire():
                local_idx = np.random.choice(pool_x.shape[0], size=n_to_acquire, replace=False)
                z = policy_trainer.assign_instruments(pool_x[local_idx])
                return local_idx, z
            local_idx, z_assign = _with_acquisition_rng(_policy_acquire)
        else:
            local_idx, z_assign = _with_acquisition_rng(lambda: acq_fn(
                trainer, pool_x, n_to_acquire,
                target_covariate=test_data.covariate,
                env=env, pool_global_idx=pool.pool_indices,
                **acq_kwargs,
            ))
        _sync_cuda()
        acquisition_sec = time.perf_counter() - t0

        local_idx = np.asarray(local_idx, dtype=int).reshape(-1)
        z_assign = np.asarray(z_assign, dtype=float).reshape(-1)
        if local_idx.shape[0] != z_assign.shape[0]:
            raise ValueError("Acquisition returned mismatched indices and instrument assignments.")

        # ---- 2. run the indirect experiment and observe (t, y)
        global_idx = pool.pool_indices[local_idx]
        t_obs, y_obs = env.get_feedback(global_idx, z_assign)
        pool.acquire(local_idx, z_assign, t_obs, y_obs)

        # ---- 3. refit the estimator from scratch on all acquired data
        train_data = pool.get_train_data()
        _sync_cuda()
        t0 = time.perf_counter()
        utils.set_seed(seed)
        trainer = utils.train_gpiv(train_data, test_data, config, seed)
        _sync_cuda()
        retrain_sec = time.perf_counter() - t0

        policy_sec = 0.0
        if uses_policy:
            t0 = time.perf_counter()
            utils.set_seed(seed)
            policy_trainer = utils.train_indpolicy(train_data, test_data, config, seed)
            _sync_cuda()
            policy_sec = time.perf_counter() - t0

        # ---- 4. evaluate sqrt-PEHE on the target reference set
        pehe = float(trainer.evaluate_cate(test_data))
        logger.info("step %3d | n=%4d | sqrt-PEHE=%.4f", step, pool.n_acquired, pehe)

        results.append({
            "step": step,
            "n_samples": int(pool.n_acquired),
            **metric_record(pehe),
            "seed": seed,
            "acquisition_sec": float(acquisition_sec),
            "retrain_sec": float(retrain_sec),
            "policy_sec": float(policy_sec),
        })
        trace.append({
            "step": step,
            "n_samples": int(pool.n_acquired),
            "global_idx": global_idx.astype(int).tolist(),
            "z": z_assign.tolist(),
            "t": np.asarray(t_obs).reshape(-1).astype(int).tolist(),
            "y": np.asarray(y_obs).reshape(-1).astype(float).tolist(),
        })
        with (trial_dir / "active_results.json").open("w") as fp:
            json.dump(results, fp, indent=2, allow_nan=False)
        with (trial_dir / "acquisition_trace.json").open("w") as fp:
            json.dump(trace, fp)

    return results
