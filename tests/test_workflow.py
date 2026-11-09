import json
from types import SimpleNamespace

import numpy as np
from click.testing import CliRunner

from src.application import main
from src.application.workflows import active_learning
from src.library import acquisitions
from src.library.datasets import ihdp


def test_all_methods_share_warm_start(tmp_path, monkeypatch):
    def train(*args):
        np.random.normal(size=17)  # Model RNG consumption cannot affect selection.
        return SimpleNamespace(evaluate_cate=lambda _: 2.0)
    monkeypatch.setattr(active_learning.utils, "train_gpiv", train)
    monkeypatch.setattr(active_learning.utils, "train_indpolicy", train)
    config = dict(seed=3, dataset_name="synthetic", scenario="sharp", n_pool=40,
                  n_test=5, warm_start_size=20, step_size=2, max_acquisitions=1, gpu=False)
    traces = []
    for name in sorted(set(acquisitions.FUNCTIONS) | set(acquisitions.POLICY_ACQUISITIONS)):
        active_learning.active_learner(dict(config, acquisition_function=name), tmp_path, 0)
        traces.append(json.loads((tmp_path / name / "trial-000/acquisition_trace.json").read_text()))
    assert all(trace == traces[0] for trace in traces)
    expected_rng = np.random.RandomState(3)
    assert traces[0][0]["global_idx"] == expected_rng.choice(40, size=20, replace=False).tolist()
    assert traces[0][0]["z"] == expected_rng.choice([0.0, 1.0], size=20).tolist()


def test_ihdp_cli_path_precedence_and_late_resolution(tmp_path, ihdp_csv_factory, monkeypatch):
    first = ihdp_csv_factory("explicit")
    second = ihdp_csv_factory("environment", effect_offset=5)
    monkeypatch.setenv("IVIG_IHDP_CSV", str(second))
    seen = []
    def capture(config):
        path = config["ihdp_csv"]
        seen.append((path, ihdp._load_ihdp_base(path)["tau"].copy()))
    monkeypatch.setattr(main, "_run_trials", capture)
    command = ["active-learning", "--job-dir", str(tmp_path / "runs"), "--cpu", "ihdp"]
    for args in (["--root", str(first.parent)], []):
        result = CliRunner().invoke(main.cli, command + args + ["gpiv", "--n-epoch", "1"])
        assert result.exit_code == 0, result.output
    assert [item[0] for item in seen] == [str(first), str(second)]
    np.testing.assert_allclose(seen[1][1] - seen[0][1], 5, atol=2e-6)


def test_cli_parallel_workers_are_forwarded(tmp_path, monkeypatch):
    seen = {}
    class CaptureParallel:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def __call__(self, tasks):
            seen["tasks"] = len(list(tasks))
    monkeypatch.setattr(main, "Parallel", CaptureParallel)
    result = CliRunner().invoke(main.cli, [
        "active-learning", "--job-dir", str(tmp_path), "--num-trials", "3",
        "--mode", "parallel", "--n-jobs", "2", "--cpu", "synthetic", "gpiv",
    ])
    assert result.exit_code == 0, result.output
    assert seen == dict(n_jobs=2, backend="loky", tasks=3)
