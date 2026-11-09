# IVIG: Bayesian Causal Experimental Design for CATE Estimation under Noncompliance

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)](https://pytorch.org/)

Official implementation of **"Bayesian Causal Experimental Design for CATE Estimation under Noncompliance"**.

Many experiments cannot assign the treatment of interest directly; they assign an *encouragement*
(an instrument `z`) and the treatment `t` is realised through stochastic compliance.
**Instrumental Variable Information Gain (IVIG)** scores a candidate unit–instrument query `(x, z)`
by the expected information that its joint treatment-realisation and outcome feedback `(t, y)`
carries about the treatment-level CATE `τ(x*)` over a target population.
IVIG is implemented on top of a Gaussian-process IV estimator (GPIV: a kernel-ridge first stage and
an expected-treatment-state-kernel GP second stage) with treatment-realisation-specific fantasy updates.

## Installation

```bash
git clone https://github.com/ErdunGAO/IVIG.git
cd IVIG
bash setup_env.sh          # portable conda env "ivig", Python 3.11 / PyTorch 2.3.1
conda activate ivig
```

or, inside an existing Python 3.11 environment, `pip install -r requirements.txt`.
For Linux CUDA 12.1, use `pip install -r requirements-cuda.txt` or `DEVICE=cuda bash setup_env.sh`.
A GPU is optional; pass `--cpu` to run on CPU.

## Data

* **Synthetic** benchmarks and **TripAdvisor-inspired simulations** are generated in code — no downloads.
* **IHDP** needs the standard NPCI file `ihdp_npci_1.csv` at `data/ihdp/` (see [data/README.md](data/README.md)).
  We do not redistribute it.

## Quick start

```bash
bash run_example.sh
```

runs IVIG and Random on the synthetic *Localized Heterogeneity* benchmark under the target-shift
setting (2 trials, 160 acquired units each), aggregates the per-step √PEHE and writes a convergence
plot to `experiments/example/results/example_sharp_shift.pdf`.

## Usage

The entry point `src/application/main.py` uses chained commands:
`active-learning <protocol>  →  <dataset>  →  gpiv`, and `evaluate → pehe | plot-convergence`.

### 1. Active learning

```bash
python src/application/main.py \
    active-learning \
        --job-dir experiments/main \
        --num-trials 10 \
        --seed 0 \
        --warm-start-size 20 \
        --step-size 20 \
        --max-acquisitions 30 \
        --acquisition-function ivig \
        --n-pool 1000 \
        --n-test 1000 \
        --target-mc 100 \
    synthetic \
        --scenario sharp \
        --target-shift 0.5 \
    gpiv \
        --n-epoch 200 \
        --lr 0.05
```

Results are written to `experiments/main/<experiment>/<acquisition>/trial-XXX/`
(`<experiment>` is `synthetic_<scenario>`, `tripadvisor_<setting>` or `ihdp`)
(`active_results.json` with per-step √PEHE and timings, `acquisition_trace.json` with the
queried units, instruments and observed feedback).

**Acquisition functions** (`--acquisition-function`)

| name | description |
|---|---|
| `ivig` | **IVIG** (proposed): target-averaged, treatment-realisation-aware information gain about the CATE |
| `random` | uniform unit selection, Bernoulli(0.5) instruments |
| `mu_bald` | BALD on the observable outcome at the plug-in propensity feature |
| `tau_bald` | posterior CATE variance |
| `mu_rho_bald` | causal-BALD style combination of CATE variance and outcome-arm information |
| `gamma` | local decision-uncertainty score inspired by Sundin et al. |
| `indpolicy` | learned instrument-assignment policy (Chandak et al.) with random unit selection |
| `y_ivig` | outcome-only IVIG (realised treatment integrated out; Theorem 1 diagnostic) |
| `no_stage1_ivig` | information gain treating the assigned instrument as the treatment |
| `oracle_ivig` | IVIG scored with the true compliance probabilities |
| `krr_mc_ivig` | first-stage particle score averaging diagnostic (`--mc-particles`) |

**Datasets**

| command | options | paper setting |
|---|---|---|
| `synthetic` | `--scenario {sharp,frontier,confounding,multiscale}`, `--target-shift s` | `s=0`: standard; `s=0.5`: target shift (warm 20 / batch 20 / 30 rounds / pool 1000) |
| `tripadvisor` | `--setting {standard,shift}` | warm 50 / batch 50 / 14 rounds / pool 1000 |
| `ihdp` | `--root data/ihdp` | warm 50 / batch 50 / 12 rounds / pool 747 |

`--mode parallel --cpu --n-jobs 2` runs trials in separate CPU processes.
`--n-jobs` defaults to 2; `--gpu/--cpu` selects the device.

### 2. Evaluation

```bash
python src/application/main.py \
    evaluate \
        --experiment-dir experiments/main/synthetic_sharp_tshift_050/ivig \
        --output-dir experiments/main/results \
    pehe
```

aggregates all trials of one acquisition rule into `<output-dir>/ivig_pehe.json`
(mean, standard deviation, standard error and per-trial √PEHE curves, trajectory average, final value).

### 3. Plotting

```bash
python src/application/main.py \
    evaluate \
        --experiment-dir experiments/main/results \
    plot-convergence \
        --prefix sharp_shift \
        -m random -m mu_bald -m tau_bald -m mu_rho_bald -m gamma -m indpolicy -m ivig
```

## Project structure

```
src/application/main.py                 # click CLI (chained commands)
src/application/workflows/
    active_learning.py                  # sequential indirect design loop (Algorithm 1)
    evaluation.py                       # aggregation and convergence plots
    utils.py                            # seeding, estimator constructors, default hyper-parameters
src/library/acquisitions.py             # IVIG, its diagnostics and the baseline acquisition rules
src/library/models/gpiv.py              # GPIV estimator and the posterior quantities used by IVIG
src/library/models/indpolicy.py         # IndPolicy baseline
src/library/modules/gaussian_process.py # first-stage KRR, expected treatment-state kernel, exact GP
src/library/datasets/                   # synthetic, TripAdvisor and IHDP benchmarks; pool environment
```

## Metrics

PEHE is mean squared CATE error; sqrt-PEHE is its square root and uses the
same units as the treatment effect. We compute sqrt-PEHE for each trial before
averaging across trials. Plots show mean ± standard deviation across trials
by default; use `plot-convergence --error-band sem` for the standard error of
the mean. A single trial is plotted with a zero-width band.

Trial results record `sqrt_pehe` (rooted PEHE) and `pehe` (squared PEHE).
Aggregate results report `sqrt_pehe_mean`, `sqrt_pehe_sem`, and
`sqrt_pehe_std`.

## Citation

```bibtex
@inproceedings{gao2026ivig,
  title     = {Bayesian Causal Experimental Design for {CATE} Estimation under Noncompliance},
  author    = {Gao, Erdun and Wang, Yuanyuan and Zhang, Liang and Liu, Yuhang and Li, Haoxuan and Gong, Mingming and Sejdinovic, Dino},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
