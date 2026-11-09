"""Small generated fixtures; no external benchmark data are bundled."""

import numpy as np
import pytest


@pytest.fixture
def ihdp_csv_factory(tmp_path):
    def make(name="ihdp", seed=0, effect_offset=0.0):
        rng = np.random.RandomState(seed)
        raw = rng.normal(size=(32, 30))
        raw[:, 0] = rng.binomial(1, 0.5, size=32)
        raw[:, 4] = raw[:, 3] + effect_offset + 2.0 + rng.normal(size=32)
        raw[:, 18] = rng.choice([1, 2], size=32)
        path = tmp_path / name / "ihdp_npci_1.csv"
        path.parent.mkdir(parents=True)
        np.savetxt(path, raw, delimiter=",")
        return path
    return make
