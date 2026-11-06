"""E3 sanity check: is 'BO no better than Sobol at 200 evaluations' an implementation artefact?  (read-only)

    uv run python code/e3_sanity.py   -> results/summary/e3_sanity.csv
Per width and method (medians over the 30 seeds unless stated): share of the post-initial-design evaluations that lie on the
broad branch of f and within 0.1 / 0.02 of the decoy centre, median nearest-earlier-point distance, share of runs that
reached the decoy minimum (gap < 1e-3), median distance to the narrow centre, success, and how many successes came from
the 4 initial points. Also whether UCB and LogEI trajectories differ."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
from problems import F_DECOY, decoy_basin  # noqa: E402

rows = []
for f in sorted((ROOT / "results/raw/e3_d2").glob("*.json")):
    r = json.loads(f.read_text())
    s = r["spec"]
    m = s["method"]["name"]
    if m not in ("bo_logei", "bo_ucb", "sobol"):
        continue
    w = s["problem"]["width"]
    prob = decoy_basin(2, w, s["instance_seed"])
    X, F = np.array(r["X"]), np.array(r["F"])
    cb, cn = prob.centre_broad, prob.u_star
    post = slice(4, None)                                  # after the 4-point initial design (BO) / same slice for Sobol
    d_b = np.linalg.norm(X - cb, axis=1)
    d_n = np.linalg.norm(X - cn, axis=1)
    broad = np.array([F_DECOY + 4.0 * float(np.sum((x - cb) ** 2)) for x in X]) <= (d_n ** 2) / w ** 2  # f on the broad branch
    # distance of each post-init point to the nearest earlier point (clustering)
    nn = [np.min(np.linalg.norm(X[:i] - X[i], axis=1)) for i in range(4, len(X))]
    first_success = int(np.argmax(F < F_DECOY)) + 1 if (F < F_DECOY).any() else 0
    rows.append({"method": m, "width": w, "seed": s["seed"],
                 "frac_broad_branch_post": float(broad[post].mean()),
                 "frac_within_0.1_of_decoy_post": float((d_b[post] < 0.1).mean()),
                 "frac_within_0.02_of_decoy_post": float((d_b[post] < 0.02).mean()),
                 "median_nn_dist_post": float(np.median(nn)),
                 "best_minus_decoy_min": float(F.min() - F_DECOY),
                 "min_dist_to_narrow_centre": float(d_n.min()),
                 "success": bool(F.min() < F_DECOY), "first_success_eval": first_success,
                 "n_evals": len(F),
                 "acq": r["meta"].get("acq"), "n_init": r["meta"].get("n_init")})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
g = df.groupby(["width", "method"]).agg(
    frac_broad=("frac_broad_branch_post", "median"), frac_in_0p1=("frac_within_0.1_of_decoy_post", "median"),
    frac_in_0p02=("frac_within_0.02_of_decoy_post", "median"), nn_dist=("median_nn_dist_post", "median"),
    converged_to_decoy_min=("best_minus_decoy_min", lambda x: float(np.mean(np.asarray(x) < 1e-3))),
    min_dist_narrow=("min_dist_to_narrow_centre", "median"), success=("success", "mean"),
    succ_in_init=("first_success_eval", lambda x: int(np.sum((np.asarray(x) > 0) & (np.asarray(x) <= 4)))),
    succ_total=("success", "sum"))
g.round(4).to_csv(ROOT / "results" / "summary" / "e3_sanity.csv")
print(g.round(3).to_string())
print("acquisition / n_init / n_evals recorded:", {m: (sorted(set(x.acq.dropna())), sorted(set(x.n_init.dropna())), sorted(set(x.n_evals))) for m, x in df.groupby("method")})

# UCB and LogEI really are different trajectories (same seed, same initial design)
same_init, same_post = [], []
for w in sorted(df.width.unique()):
    for sd in range(30):
        a = df[(df.method == "bo_logei") & (df.width == w) & (df.seed == sd)]
        b = df[(df.method == "bo_ucb") & (df.width == w) & (df.seed == sd)]
        same_post.append(float(a.iloc[0].median_nn_dist_post != b.iloc[0].median_nn_dist_post))
print("fraction of (width, seed) pairs where UCB and LogEI trajectories differ:", np.mean(same_post).round(3))
