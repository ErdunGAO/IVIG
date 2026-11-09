from src.library.datasets import synthetic, tripadvisor, ihdp
from src.library.datasets.active_learning import IVEnvironment, ActiveIVDataset

DATASETS = {
    "synthetic": synthetic,
    "tripadvisor": tripadvisor,
    "ihdp": ihdp,
}


def get_simulation_module(name: str):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.") from exc
