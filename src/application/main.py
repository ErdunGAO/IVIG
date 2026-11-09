"""
Command-line entry point.  Commands are chained:

    python src/application/main.py active-learning [...] <dataset> [...] gpiv [...]
    python src/application/main.py evaluate --experiment-dir ... pehe
    python src/application/main.py evaluate --experiment-dir ... plot-convergence -m ivig -m random
"""

import logging
import sys
from pathlib import Path

import click
import torch
from joblib import Parallel, delayed

# Allow `python src/application/main.py ...` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application import workflows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {}


def _run_trials(config):
    experiment_dir = config["experiment_dir"]
    num_trials = int(config["num_trials"])
    if config.get("mode", "sequential") == "parallel":
        return Parallel(n_jobs=min(int(config.get("n_jobs", 2)), num_trials), backend="loky")(
            delayed(workflows.active_learning.active_learner)(
                config=config, experiment_dir=experiment_dir, trial=trial,
            )
            for trial in range(num_trials)
        )
    return [
        workflows.active_learning.active_learner(config=config, experiment_dir=experiment_dir, trial=trial)
        for trial in range(num_trials)
    ]


# ---------------------------------------------------------------------------
# active-learning protocol
# ---------------------------------------------------------------------------
@cli.command("active-learning")
@click.option("--job-dir", type=str, required=True, help="root directory for results")
@click.option("--num-trials", default=1, type=int, help="number of independent trials (seeds), default=1")
@click.option("--seed", default=0, type=int, help="base random seed; trial k uses seed+k, default=0")
@click.option("--warm-start-size", default=20, type=int, help="random warm-start queries, default=20")
@click.option("--step-size", default=20, type=int, help="queries acquired per round, default=20")
@click.option("--max-acquisitions", default=30, type=int, help="number of rounds incl. warm start, default=30")
@click.option("--acquisition-function", default="ivig", type=str,
              help="random | mu_bald | tau_bald | mu_rho_bald | gamma | indpolicy | ivig | "
                   "y_ivig | no_stage1_ivig | oracle_ivig | krr_mc_ivig   (default=ivig)")
@click.option("--n-pool", default=1000, type=int, help="candidate pool size, default=1000")
@click.option("--n-test", default=1000, type=int, help="target reference set size, default=1000")
@click.option("--target-mc", default=100, type=int, help="target covariates sampled per acquisition round, default=100")
@click.option("--mc-particles", default=4, type=int, help="first-stage particles for krr_mc_ivig, default=4")
@click.option("--mode", default="sequential", type=click.Choice(["sequential", "parallel"]),
              help="run trials sequentially or in parallel processes (CPU only)")
@click.option("--n-jobs", default=2, type=click.IntRange(min=1), help="parallel CPU workers, default=2")
@click.option("--gpu/--cpu", default=True, help="use CUDA when available (default: --gpu)")
@click.pass_context
def active_learning(context, job_dir, num_trials, seed, warm_start_size, step_size, max_acquisitions,
                    acquisition_function, n_pool, n_test, target_mc, mc_particles, mode, n_jobs, gpu):
    if mode == "parallel" and gpu and torch.cuda.is_available():
        raise click.ClickException("Parallel trials require --cpu; use sequential mode for CUDA.")
    if warm_start_size < 2:
        raise click.ClickException("GPIV requires --warm-start-size >= 2.")
    if min(num_trials, warm_start_size, step_size, max_acquisitions, n_pool, n_test, target_mc, mc_particles) < 1:
        raise click.ClickException("Trial counts, sample sizes and MC sizes must be positive.")
    context.obj.update({
        "job_dir": str(Path(job_dir)),
        "num_trials": num_trials,
        "seed": seed,
        "warm_start_size": warm_start_size,
        "step_size": step_size,
        "max_acquisitions": max_acquisitions,
        "acquisition_function": acquisition_function,
        "n_pool": n_pool,
        "n_test": n_test,
        "target_mc": target_mc,
        "mc_particles": mc_particles,
        "mode": mode,
        "n_jobs": n_jobs,
        "gpu": bool(gpu and torch.cuda.is_available()),
    })


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def _set_dataset(context, dataset_name, scenario, experiment_name, d_covariate=None):
    job_dir = Path(context.obj["job_dir"])
    context.obj.update({
        "dataset_name": dataset_name,
        "scenario": scenario,
        "d_covariate": d_covariate,
        "experiment_dir": str(job_dir / experiment_name),
    })


@cli.command("synthetic")
@click.option("--scenario", default="sharp", type=click.Choice(["sharp", "frontier", "confounding", "multiscale"]),
              help="Localized (sharp) / Frontier / Confounding-dominant / Multiscale heterogeneity, default=sharp")
@click.option("--target-shift", default=0.0, type=float,
              help="0 = target equals the source distribution (standard); 0.5 = paper's target-shift setting")
@click.option("--dim-covariate", default=5, type=int, help="covariate dimension, default=5")
@click.pass_context
def synthetic(context, scenario, target_shift, dim_covariate):
    from src.library.datasets import synthetic as sim
    name = sim.make_scenario(scenario, target_shift)
    _set_dataset(context, "synthetic", name, f"synthetic_{name}", dim_covariate)


@cli.command("tripadvisor")
@click.option("--setting", default="standard", type=click.Choice(["standard", "shift"]),
              help="standard: pool and target share the covariate distribution; shift: shifted target")
@click.pass_context
def tripadvisor(context, setting):
    _set_dataset(context, "tripadvisor", f"tripadvisor_{setting}", f"tripadvisor_{setting}")


@cli.command("ihdp")
@click.option("--root", default=None, type=click.Path(path_type=Path),
              help="IHDP directory; defaults to IVIG_IHDP_CSV or repository data/ihdp")
@click.pass_context
def ihdp(context, root):
    from src.library.datasets.ihdp import resolve_data_path
    csv = resolve_data_path(root / "ihdp_npci_1.csv" if root is not None else None)
    if not csv.is_file():
        raise click.ClickException(
            f"{csv} not found. Supply --root or set IVIG_IHDP_CSV (see data/README.md)."
        )
    context.obj["ihdp_csv"] = str(csv)
    _set_dataset(context, "ihdp", "ihdp_iv_semisynth", "ihdp")


# ---------------------------------------------------------------------------
# estimator (launches the run)
# ---------------------------------------------------------------------------
@cli.command("gpiv")
@click.option("--n-epoch", default=200, type=int, help="joint MLL optimisation steps, default=200")
@click.option("--lr", default=0.05, type=float, help="Adam learning rate, default=0.05")
@click.option("--lengthscale-init", default=None, type=str,
              help="initial RBF lengthscale: a float, 'median', 'sqrt_dim' (default: GPyTorch default)")
@click.pass_context
def gpiv(context, n_epoch, lr, lengthscale_init):
    if "dataset_name" not in context.obj:
        raise click.ClickException("Specify a dataset command (synthetic / tripadvisor / ihdp) before gpiv.")
    if n_epoch < 1 or lr <= 0:
        raise click.ClickException("--n-epoch and --lr must be positive.")
    context.obj["gpiv"] = {"n_epoch": n_epoch, "lr": lr, "lengthscale_init": lengthscale_init}
    context.obj["model_name"] = "gpiv"
    _run_trials(context.obj)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@cli.command("evaluate")
@click.option("--experiment-dir", type=str, required=True,
              help="for `pehe`: <job-dir>/<experiment>/<acquisition>; for `plot-convergence`: the results dir")
@click.option("--output-dir", type=str, default=None, help="where to write aggregated results (default: experiment-dir)")
@click.pass_context
def evaluate(context, experiment_dir, output_dir):
    context.obj.update({
        "experiment_dir": experiment_dir,
        "output_dir": experiment_dir if output_dir is None else output_dir,
    })


@cli.command("pehe")
@click.pass_context
def pehe(context):
    workflows.evaluation.pehe(
        experiment_dir=Path(context.obj["experiment_dir"]),
        output_dir=Path(context.obj["output_dir"]),
    )


@cli.command("plot-convergence")
@click.option("--methods", "-m", multiple=True, type=str, help="acquisition names to plot")
@click.option("--prefix", default="convergence", type=str, help="output file prefix")
@click.option("--y-max", default=None, type=float, help="upper y-axis limit")
@click.option("--error-band", default="std", type=click.Choice(["std", "sem"]),
              help="standard deviation (default) or standard error across trials")
@click.pass_context
def plot_convergence(context, methods, prefix, y_max, error_band):
    workflows.evaluation.plot_convergence(
        results_dir=Path(context.obj["experiment_dir"]),
        methods=methods,
        prefix=prefix,
        y_max=y_max,
        error_band=error_band,
    )


if __name__ == "__main__":
    cli()
