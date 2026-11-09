"""Aggregate per-trial results and plot sqrt-PEHE convergence curves."""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from src.library.metrics import read_sqrt_pehe, METRIC_SCHEMA_VERSION


_LABELS = {
    "random": "Random",
    "mu_bald": r"$\mu$-BALD",
    "tau_bald": r"$\tau$-BALD",
    "mu_rho_bald": r"$\mu\rho$-BALD",
    "gamma": r"$\gamma$",
    "indpolicy": "IndPolicy",
    "ivig": "IVIG",
    "y_ivig": "Y-IVIG",
    "no_stage1_ivig": "No-Stage-I IVIG",
    "oracle_ivig": "Oracle IVIG",
    "krr_mc_ivig": "IVIG (MC stage I)",
}

_STYLES = {
    "random": ("tab:gray", "--", "X"),
    "mu_bald": ("tab:cyan", "--", "o"),
    "tau_bald": ("tab:olive", "--", "s"),
    "mu_rho_bald": ("tab:purple", "--", "^"),
    "gamma": ("tab:brown", "--", "v"),
    "indpolicy": ("tab:green", "-.", "D"),
    "ivig": ("tab:red", "-", "o"),
    "y_ivig": ("tab:orange", "-", "P"),
    "no_stage1_ivig": ("tab:pink", "-", "*"),
    "oracle_ivig": ("black", ":", "o"),
    "krr_mc_ivig": ("tab:blue", "-", "o"),
}


def _trial_dirs(experiment_dir: Path):
    return sorted(p for p in Path(experiment_dir).iterdir() if p.is_dir() and p.name.startswith("trial-"))


def pehe(experiment_dir: Path, output_dir: Path):
    """
    Collect ``active_results.json`` from every ``trial-XXX`` under
    ``experiment_dir`` (one acquisition rule) into
    ``output_dir/<acquisition>_pehe.json`` and print a summary.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = _trial_dirs(experiment_dir)
    if not trials:
        raise FileNotFoundError(f"No trial-XXX directories found under {experiment_dir}")

    acquisition = None
    per_trial = {}
    for trial_dir in trials:
        rows = json.loads((trial_dir / "active_results.json").read_text())
        if not rows:
            raise ValueError(f"Empty trajectory: {trial_dir}")
        cfg = json.loads((trial_dir / "config.json").read_text())
        name = cfg.get("acquisition_function", experiment_dir.name)
        if acquisition is not None and acquisition != name:
            raise ValueError("Cannot aggregate different acquisition functions.")
        acquisition = name
        grid = [int(r["n_samples"]) for r in rows]
        if any(n <= 0 for n in grid) or any(b <= a for a, b in zip(grid, grid[1:])):
            raise ValueError(f"Sample sizes must strictly increase: {trial_dir}")
        per_trial[trial_dir.name] = {
            "n_samples": grid,
            "sqrt_pehe": [read_sqrt_pehe(r) for r in rows],
            "acquisition_sec": [float(r.get("acquisition_sec", 0.0)) for r in rows],
            "retrain_sec": [float(r.get("retrain_sec", 0.0)) for r in rows],
        }

    # Align on actual sample counts, rather than slicing unrelated positions.
    shared = set.intersection(*(set(v["n_samples"]) for v in per_trial.values()))
    if not shared:
        raise ValueError("Trials have no shared sample sizes to aggregate.")
    n_samples = sorted(shared)
    curves = np.asarray([
        [dict(zip(v["n_samples"], v["sqrt_pehe"]))[n] for n in n_samples]
        for v in per_trial.values()
    ], dtype=float)
    mean = curves.mean(0)
    sem = stats.sem(curves, axis=0) if curves.shape[0] > 1 else np.zeros_like(mean)
    summary = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "metric": "sqrt_pehe",
        "acquisition_function": acquisition,
        "num_trials": int(curves.shape[0]),
        "n_samples": n_samples,
        "sqrt_pehe_mean": mean.tolist(),
        "sqrt_pehe_sem": sem.tolist(),
        "sqrt_pehe_std": curves.std(0, ddof=1).tolist() if curves.shape[0] > 1 else [0.0] * len(n_samples),
        "trajectory_average_sqrt_pehe": float(mean.mean()),
        "final_sqrt_pehe": float(mean[-1]),
        "trials": per_trial,
    }
    out = output_dir / f"{acquisition}_pehe.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"[{acquisition}] {curves.shape[0]} trials")
    print("  n_samples :", summary["n_samples"])
    print("  sqrt-PEHE :", [round(v, 4) for v in mean])
    print(f"  trajectory average = {summary['trajectory_average_sqrt_pehe']:.4f}   final = {summary['final_sqrt_pehe']:.4f}")
    print(f"  written to {out}")
    return summary


def plot_convergence(results_dir: Path, methods, prefix="convergence", y_max=None, error_band="std"):
    """Plot mean sqrt-PEHE with trial SD (default), or SEM if requested."""
    if error_band not in ("std", "sem"):
        raise ValueError("error_band must be 'std' or 'sem'.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    plotted = 0
    for method in methods:
        path = results_dir / f"{method}_pehe.json"
        if not path.exists():
            print(f"skipping {method}: {path} not found")
            continue
        s = json.loads(path.read_text())
        x = np.asarray(s["n_samples"])
        version = s.get("metric_schema_version", 1)
        if version == METRIC_SCHEMA_VERSION and s.get("metric") == "sqrt_pehe":
            metric_prefix = "sqrt_pehe"
        elif version == 1 and s.get("metric") in (None, "sqrt_pehe"):
            # Old release summaries contain RMSE despite the pehe_mean name.
            metric_prefix = "pehe"
        else:
            raise ValueError(f"Unsupported summary metric schema: {path}")
        m = np.asarray(s[f"{metric_prefix}_mean"], dtype=float)
        band_key = f"{metric_prefix}_{error_band}"
        if band_key in s:
            e = np.asarray(s[band_key], dtype=float)
        elif error_band == "std" and s.get("num_trials", 0) > 0:
            e = np.asarray(s[f"{metric_prefix}_sem"], dtype=float) * np.sqrt(s["num_trials"])
        else:
            raise ValueError(f"Cannot recover {error_band} from summary: {path}. Re-aggregate the trial results.")
        if x.ndim != 1 or m.shape != x.shape or e.shape != x.shape or x.size == 0:
            raise ValueError(f"Invalid summary curve shape: {path}")
        if not np.isfinite(m).all() or not np.isfinite(e).all() or (m < 0).any() or (e < 0).any():
            raise ValueError(f"Invalid summary metric values: {path}")
        plotted += 1
        color, ls, marker = _STYLES.get(method, ("C0", "-", "o"))
        ax.plot(x, m, label=_LABELS.get(method, method), color=color, linestyle=ls, marker=marker, markersize=4)
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.2)
    if not plotted:
        plt.close(fig)
        raise FileNotFoundError("No requested convergence summaries found.")
    ax.set_xlabel("Number of acquired units")
    ax.set_ylabel(r"$\sqrt{\mathrm{PEHE}}$")
    ax.grid(True, linestyle="--", alpha=0.5)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.legend()
    fig.tight_layout()
    out = results_dir / f"{prefix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(results_dir / f"{prefix}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
