"""Plain-numpy containers for acquired data and the target reference set."""

from typing import NamedTuple, Optional
import numpy as np


class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray


class TestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: Optional[np.ndarray]
    structural: np.ndarray
    sample_weight: Optional[np.ndarray] = None


from src.library.metrics import weighted_pehe_np, weighted_sqrt_pehe_np
