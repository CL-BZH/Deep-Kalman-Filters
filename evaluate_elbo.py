"""
Evaluate trained DKF-family models on the Lorenz test sequences with the
quantities the theory talks about, rather than with forecasting MSE:

  * ELBO   -- the single-trajectory estimator used for training
              (reconstruction - analytic KL), averaged over many trajectories
              so that its Monte Carlo noise is negligible;
  * L_K    -- the importance-weighted (IWAE) bound with K trajectories,
              log (1/K) sum_k p(x, z_k) / q(z_k | x).  L_1 = ELBO, and L_K
              increases towards log p(x) as K grows.

Both are reported per time step (divided by the sequence length), averaged
over the test sequences, with a 95% confidence half-width over sequences.

The test sequences and their noise are rebuilt exactly as in main.py; the
noise is drawn once, with a fixed seed, so that all models are evaluated on
the same noisy sequences.

Usage (from the repository root, after training with main.py on the same
lorenz_range, and before the checkpoints are overwritten by another run):

  python evaluate_elbo.py --lorenz_range 11000 13000 \
      --models AR_DKF_1 FAR_DKF_1 BAR_DKF_1 --K 1 10 100

Results are printed and appended to elbo_results.csv.
"""

import argparse
import csv
import math
import os
from functools import partial

import torch

from Utils.tools import (generate_lorenz_data, normalize_data,
                         split_into_sequences, get_model_pth)
from Utils.model_fcts import model_build

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_density(v, mean, logvar):
    """Sum over the feature dimension of log N(v; mean, exp(logvar))."""
    return -0.5 * (LOG_2PI + logvar + (v - mean) ** 2 / torch.exp(logvar)).sum(-1)


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """Sum over the feature dimension of KL(N(mu_q, var_q) || N(mu_p, var_p))."""
    return 0.5 * (logvar_p - logvar_q
                  + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
                  - 1.0).sum(-1)


@torch.no_grad()
def per_trajectory_terms(model, x):
    """
    One ancestral trajectory per sequence of x, shape (seq_len, batch, x_dim).
    Returns, per sequence (shape (batch,)):
      elbo_rb -- reconstruction minus analytic KL (the training estimator)
      log_w   -- log p(x, z) - log q(z | x) at the sampled trajectory
    """
    (_, x_hat, x_hat_logvar, z_mean, z_logvar,
     z_tr_mean, z_tr_logvar) = model(x)
    z = model.sampled_z

    log_px_z = gaussian_log_density(x, x_hat, x_hat_logvar).sum(0)
    log_pz = gaussian_log_density(z, z_tr_mean, z_tr_logvar).sum(0)
    log_qz = gaussian_log_density(z, z_mean, z_logvar).sum(0)
    kl = gaussian_kl(z_mean, z_logvar, z_tr_mean, z_tr_logvar).sum(0)

    return log_px_z - kl, log_px_z + log_pz - log_qz


@torch.no_grad()
def evaluate(model, x, Ks, n_elbo_samples, chunk):
    """
    x: (seq_len, n_seq, x_dim). Returns {name: (mean, 95% half-width)} with
    values per time step, over the n_seq sequences.
    """
    seq_len, n_seq, _ = x.shape
    results = {}

    # ELBO: average of the training estimator over many trajectories
    elbo = torch.zeros(n_seq)
    for _ in range(n_elbo_samples):
        for s in range(0, n_seq, chunk):
            e, _ = per_trajectory_terms(model, x[:, s:s + chunk])
            elbo[s:s + chunk] += e
    elbo /= n_elbo_samples
    results["ELBO"] = elbo

    # IWAE bounds: K trajectories per sequence, log-mean-exp of the weights
    for K in Ks:
        lk = torch.zeros(n_seq)
        per_chunk = max(1, chunk // K)
        for s in range(0, n_seq, per_chunk):
            xs = x[:, s:s + per_chunk]
            b = xs.shape[1]
            xr = xs.repeat_interleave(K, dim=1)          # (seq_len, b*K, x_dim)
            _, log_w = per_trajectory_terms(model, xr)
            log_w = log_w.view(b, K)
            lk[s:s + b] = torch.logsumexp(log_w, dim=1) - math.log(K)
        results[f"L_{K}"] = lk

    out = {}
    for name, v in results.items():
        v = v / seq_len                                   # per time step
        half = 1.96 * v.std(unbiased=True).item() / math.sqrt(n_seq)
        out[name] = (v.mean().item(), half)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lorenz_range", type=int, nargs=2, required=True)
    p.add_argument("--models", nargs="+", default=["AR_DKF_1", "FAR_DKF_1", "BAR_DKF_1"])
    p.add_argument("--K", type=int, nargs="+", default=[1, 10, 100])
    p.add_argument("--elbo_samples", type=int, default=20,
                   help="Trajectories averaged for the ELBO estimate.")
    p.add_argument("--train_seq_len", type=int, default=100)
    p.add_argument("--test_seq_len", type=int, default=100)
    p.add_argument("--noise_var", type=float, default=0.006)
    p.add_argument("--lorenz_points", type=int, default=20000)
    p.add_argument("--lorenz_initial_point", type=float, nargs=3, default=[2, 1, 1])
    p.add_argument("--checkpoint_dir", type=str, default="Checkpoints")
    p.add_argument("--chunk", type=int, default=2000,
                   help="Max sequences per forward pass (lower it if memory is short).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="elbo_results.csv")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ---- rebuild the test sequences exactly as main.py does -------------
    normalizer = partial(normalize_data, normalization_type="standardize",
                         per_feature=False, per_sequence=False, range_=None)
    lorenz = generate_lorenz_data(args.lorenz_points,
                                  initial_point=args.lorenz_initial_point)
    data = normalizer(lorenz[args.lorenz_range[0]:args.lorenz_range[1]])
    n = len(data)
    l_train, l_val = int(n * (3 / 5)), int(n * (1 / 20))
    test_data = data[l_train + l_val:]
    test_seq = split_into_sequences(test_data,
                                    seq_length=args.train_seq_len + args.test_seq_len,
                                    stride=1)
    # observed part only, with noise drawn once for all models
    x = test_seq[:, :args.train_seq_len, :].float()
    x = x + torch.randn_like(x) * math.sqrt(args.noise_var)
    x = x.permute(1, 0, 2).contiguous()                  # (seq_len, n_seq, x_dim)
    print(f"Lorenz range {args.lorenz_range}: {x.shape[1]} test sequences "
          f"of length {x.shape[0]}")

    # ---- evaluate every model --------------------------------------------
    rows = []
    for name in args.models:
        model, _ = model_build(name, {"x_dim": 3})
        model.load_model(get_model_pth("Lorenz attractor", name,
                                       directory=args.checkpoint_dir))
        model.eval()
        res = evaluate(model, x, args.K, args.elbo_samples, args.chunk)
        for q, (m, h) in res.items():
            rows.append([args.lorenz_range[0], args.lorenz_range[1], name, q, m, h])

    # ---- print and save ---------------------------------------------------
    quantities = ["ELBO"] + [f"L_{K}" for K in args.K]
    print("\nPer-time-step values (higher is better), mean ± 95% half-width")
    print(f"{'model':<12}" + "".join(f"{q:>20}" for q in quantities))
    for name in args.models:
        cells = {r[3]: (r[4], r[5]) for r in rows if r[2] == name}
        print(f"{name:<12}" + "".join(f"{cells[q][0]:>12.4f} ± {cells[q][1]:<5.4f}"
                                      for q in quantities))

    # L_K approaches log p(x) as K grows, so L_Kmax - ELBO estimates the gap
    # KL(q || posterior) per time step: small means the inference network
    # approximates the posterior well (the quantity Proposition 2 is about).
    k_max = f"L_{max(args.K)}"
    print(f"\nEstimated ELBO gap per time step ({k_max} - ELBO; lower bound on "
          f"KL(q || posterior)):")
    for name in args.models:
        cells = {r[3]: r[4] for r in rows if r[2] == name}
        print(f"{name:<12}{cells[k_max] - cells['ELBO']:>12.4f}")

    new_file = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["range_start", "range_end", "model", "quantity",
                        "mean_per_step", "ci95_half_width"])
        w.writerows(rows)
    print(f"\nAppended to {args.output}")


if __name__ == "__main__":
    main()
