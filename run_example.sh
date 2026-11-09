#!/usr/bin/env bash
# Minimal end-to-end example: IVIG vs. Random on the synthetic "Localized
# Heterogeneity" benchmark under the paper's target-shift setting, followed by
# aggregation and a convergence plot.  Runs in a few minutes on one GPU
# (longer on CPU).  Outputs go to experiments/example/.
set -euo pipefail

JOB_DIR="${JOB_DIR:-experiments/example}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MAX_ACQ="${MAX_ACQ:-8}"        # 20 warm-start + 7 x 20 = 160 acquired units
mkdir -p "$JOB_DIR"

for ACQ in random ivig; do
    echo "===== active learning: $ACQ ====="
    python src/application/main.py \
        active-learning \
            --job-dir "$JOB_DIR" \
            --num-trials "$NUM_TRIALS" \
            --seed 0 \
            --warm-start-size 20 \
            --step-size 20 \
            --max-acquisitions "$MAX_ACQ" \
            --acquisition-function "$ACQ" \
            --n-pool 1000 \
            --n-test 1000 \
            --target-mc 100 \
        synthetic \
            --scenario sharp \
            --target-shift 0.5 \
        gpiv \
            --n-epoch 200 \
            --lr 0.05

    echo "===== aggregate: $ACQ ====="
    python src/application/main.py \
        evaluate \
            --experiment-dir "$JOB_DIR/synthetic_sharp_tshift_050/$ACQ" \
            --output-dir "$JOB_DIR/results" \
        pehe
done

echo "===== plot ====="
python src/application/main.py \
    evaluate \
        --experiment-dir "$JOB_DIR/results" \
    plot-convergence \
        --prefix example_sharp_shift \
        -m random \
        -m ivig

echo "Done. Curves in $JOB_DIR/results/example_sharp_shift.pdf"
