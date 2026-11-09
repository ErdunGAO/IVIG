"""Environment and bookkeeping for sequential indirect experimental design."""

from __future__ import annotations

import numpy as np

from src.library.datasets.data_class import TrainDataSet


class IVEnvironment:
    """
    Simulated encouragement-design environment.

    A pool of units (x_i, u_i, eps_i) is generated once; for every unit the
    realised treatment and outcome under z = 0 and z = 1 are pre-computed.
    The learner picks pool indices and assigns instruments, and the
    environment reveals the corresponding (t, y).  The true compliance
    probabilities are exposed only for oracle diagnostics.
    """

    def __init__(self, sim_module, scenario: str, n_pool: int = 1000,
                 d_covariate: int | None = None, n_test: int = 1000, seed: int = 0,
                 **simulation_kwargs):
        if not sim_module.is_binary_scenario(scenario):
            raise ValueError(f"Unknown scenario '{scenario}' for dataset module {sim_module.__name__}.")
        self.sim = sim_module
        self.scenario = scenario
        self.n_pool = int(n_pool)
        self.seed = int(seed)
        self.d_covariate = int(d_covariate) if d_covariate is not None else int(sim_module.get_default_d_covariate(scenario))
        self.n_test = int(n_test)

        x_pool, z0, z1, test_data = sim_module.generate_binary_pool(
            scenario,
            n_pool=self.n_pool,
            d_covariate=self.d_covariate,
            n_test=self.n_test,
            seed=self.seed,
            **simulation_kwargs,
        )
        self.x_pool = np.asarray(x_pool, dtype=np.float32)
        self.n_pool = int(self.x_pool.shape[0])  # semi-synthetic pools may be capped by the data
        self.z0_outcomes = z0
        self.z1_outcomes = z1
        self.test_data = test_data

    def get_feedback(self, pool_indices, z_assignments):
        """Reveal (T, Y) for the selected pool units under the assigned instruments."""
        idx = np.asarray(pool_indices, dtype=int).reshape(-1)
        z = np.asarray(z_assignments, dtype=float).reshape(-1, 1)
        t = np.where(z < 0.5, self.z0_outcomes["T"][idx], self.z1_outcomes["T"][idx])
        y = np.where(z < 0.5, self.z0_outcomes["Y"][idx], self.z1_outcomes["Y"][idx])
        return t, y

    def get_oracle_propensity(self, pool_indices, z_assignments):
        """True P(T = 1 | x_i, z) for the selected pool units (diagnostics only)."""
        idx = np.asarray(pool_indices, dtype=int).reshape(-1)
        z = np.asarray(z_assignments, dtype=float).reshape(-1, 1)
        return np.where(z < 0.5, self.z0_outcomes["P"][idx], self.z1_outcomes["P"][idx])

    def get_test_data(self):
        return self.test_data

    def diagnostics(self) -> dict:
        p0 = np.asarray(self.z0_outcomes["P"], dtype=float).reshape(-1)
        p1 = np.asarray(self.z1_outcomes["P"], dtype=float).reshape(-1)
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "n_pool": int(self.n_pool),
            "d_covariate": int(self.d_covariate),
            "n_test": int(np.asarray(self.test_data.covariate).shape[0]),
            "mean_p0": float(np.mean(p0)),
            "mean_p1": float(np.mean(p1)),
            "mean_compliance_gap": float(np.mean(p1 - p0)),
            "treated_rate_z0": float(np.mean(self.z0_outcomes["T"])),
            "treated_rate_z1": float(np.mean(self.z1_outcomes["T"])),
        }


class ActiveIVDataset:
    """Tracks which pool units have been queried and stores the revealed data."""

    def __init__(self, x_pool: np.ndarray):
        self.x_pool = np.asarray(x_pool)
        self.n_pool = self.x_pool.shape[0]
        self.dim_x = self.x_pool.shape[1]
        self.acquired_mask = np.zeros(self.n_pool, dtype=bool)
        self.x_train = np.empty((0, self.dim_x))
        self.z_train = np.empty((0, 1))
        self.t_train = np.empty((0, 1))
        self.y_train = np.empty((0, 1))

    @property
    def n_acquired(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def pool_indices(self) -> np.ndarray:
        """Global indices of the units still available in the pool."""
        return np.where(~self.acquired_mask)[0]

    def get_pool_x(self) -> np.ndarray:
        return self.x_pool[self.pool_indices]

    def acquire(self, local_idx, z, t, y):
        """Move units (indexed in the *current* pool) into the training set."""
        local_idx = np.asarray(local_idx, dtype=int).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        t = np.asarray(t, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(np.unique(local_idx)) != local_idx.shape[0]:
            raise ValueError("Attempted to acquire the same pool unit twice in one step.")
        if not (local_idx.shape[0] == z.shape[0] == t.shape[0] == y.shape[0]):
            raise ValueError("local_idx, z, t and y must have the same length.")
        if (local_idx < 0).any() or (local_idx >= self.pool_indices.size).any():
            raise ValueError("Acquisition indices must refer to remaining pool units.")
        if not np.isin(z, [0.0, 1.0]).all() or not np.isin(t, [0.0, 1.0]).all() or not np.isfinite(y).all():
            raise ValueError("Instruments/treatments must be binary and outcomes finite.")
        global_idx = self.pool_indices[local_idx]
        self.acquired_mask[global_idx] = True
        self.x_train = np.vstack([self.x_train, self.x_pool[global_idx]])
        self.z_train = np.vstack([self.z_train, z.reshape(-1, 1)])
        self.t_train = np.vstack([self.t_train, t.reshape(-1, 1)])
        self.y_train = np.vstack([self.y_train, y.reshape(-1, 1)])
        return global_idx

    def get_train_data(self) -> TrainDataSet | None:
        if self.n_acquired == 0:
            return None
        return TrainDataSet(
            treatment=self.t_train,
            instrumental=self.z_train,
            covariate=self.x_train,
            outcome=self.y_train,
            structural=np.zeros_like(self.y_train),
        )
