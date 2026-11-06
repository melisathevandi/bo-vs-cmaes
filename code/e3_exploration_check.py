"""Can the BO implementation explore at all on the decoy function?  UCB with a large weight should look like space filling.
Sanity check, not an experiment: 6 seeds (method seeds 1000-1005), w = 0.1. Console output saved by hand in
results/summary/e3_exploration_check.txt (about 17 minutes on one core)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(1)
from methods import bo  # noqa: E402
from problems import F_DECOY, decoy_basin  # noqa: E402

w = 0.1
for beta in (4.0, 100.0, 1e4):
    succ, near, nn = [], [], []
    for seed in range(6):
        p = decoy_basin(2, w, seed)
        out = bo.run(p, 200, 1000 + seed, acq="ucb", beta=beta, refit_every=10)
        X, F = np.array(out["X"]), np.array(out["F"])
        d_b = np.linalg.norm(X[4:] - p.centre_broad, axis=1)
        succ.append(F.min() < F_DECOY)
        near.append(np.mean(d_b < 0.1))
        nn.append(np.median([np.min(np.linalg.norm(X[:i] - X[i], axis=1)) for i in range(4, len(X))]))
    print(f"UCB beta={beta:g} (weight sqrt(beta)={np.sqrt(beta):g}): success {np.mean(succ):.2f}, "
          f"post-init fraction within 0.1 of decoy centre {np.mean(near):.2f}, median nearest-neighbour distance {np.median(nn):.4f}", flush=True)
