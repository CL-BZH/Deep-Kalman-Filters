#!/bin/bash
# 3 repetitions x 3 Lorenz datasets: train, then evaluate before the next run
for rep in 1 2 3; do
  for r in "4500 6500" "8600 10500" "11000 13000"; do
    python main.py --dataset lorenz --lorenz_range $r \
      --train_seq_len 100 --test_seq_len 100 \
      --models AR_DKF_1 FAR_DKF_1 BAR_DKF_1 || exit 1
    python evaluate_elbo.py --lorenz_range $r \
      --models AR_DKF_1 FAR_DKF_1 BAR_DKF_1 --K 1 10 100 || exit 1
  done
done
