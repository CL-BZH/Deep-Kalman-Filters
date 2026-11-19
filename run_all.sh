#!/bin/bash
# Reproduces the experiments of the report.

# Sine waves: vanilla DKF and AR DKF, 4 repetitions x 2 frequencies
for rep in 1 2 3 4; do
  for f in 1 2; do
    python main.py --dataset sine --sine_frequency $f \
      --models DKF_1 AR_DKF_1 || exit 1
  done
done

# Lorenz attractor: 4 repetitions x 3 datasets.
# Train, then evaluate the ELBO before the next run overwrites the checkpoints.
for rep in 1 2 3 4; do
  for r in "4500 6500" "8600 10500" "11000 13000"; do
    python main.py --dataset lorenz --lorenz_range $r \
      --train_seq_len 100 --test_seq_len 100 \
      --models AR_DKF_1 FAR_DKF_1 BAR_DKF_1 || exit 1
    python evaluate_elbo.py --lorenz_range $r \
      --models AR_DKF_1 FAR_DKF_1 BAR_DKF_1 --K 1 10 100 || exit 1
  done
done
