"""H5: expected scaling exponent of CMA-ES's learning time, from its learning rates alone.

The covariance matrix moves towards its target at a rate of about c1 + c_mu per
generation, so learning takes about 1 / (c1 + c_mu) generations, i.e. about
lambda / (c1 + c_mu) evaluations, up to a constant that does not depend on d. Fitting
log of that quantity against log d gives the exponent alpha that the learning rates
alone predict. No optimisation is run.

Two sources for the rates are compared:
  tutorial  c1 = 2 / ((d + 1.3)^2 + mu_eff)
            c_mu = min(1 - c1, 2 (mu_eff - 2 + 1/mu_eff) / ((d + 2)^2 + mu_eff))
  pycma     the values pycma 4.5.0 actually uses (es.sp.c1, es.sp.cmu)

Writes results/summary/h5_expected_alpha.csv.
"""

from pathlib import Path

import cma
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "h5_expected_alpha.csv"
DIMS = [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80]


def tutorial_rates(d, mueff):
    c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
    return c1, cmu


def slope(dims, values):
    return float(np.polyfit(np.log(dims), np.log(values), 1)[0])


if __name__ == "__main__":
    rows = []
    for d in DIMS:
        es = cma.CMAEvolutionStrategy(np.full(d, 0.5), 0.3, {"verbose": -9})
        lam, mueff = es.popsize, es.sp.weights.mueff
        t_c1, t_cmu = tutorial_rates(d, mueff)
        rows.append({"d": d, "lambda": lam, "mu_eff": mueff,
                     "c1_pycma": es.sp.c1, "cmu_pycma": es.sp.cmu,
                     "c1_tutorial": t_c1, "cmu_tutorial": t_cmu,
                     "evals_pycma": lam / (es.sp.c1 + es.sp.cmu),
                     "evals_tutorial": lam / (t_c1 + t_cmu)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(df.round(5).to_string(index=False))
    big = df[df.d >= 10]
    for src in ("pycma", "tutorial"):
        print(f"{src:8s} expected alpha: all d = {slope(df.d, df[f'evals_{src}']):.3f}, "
              f"d >= 10 = {slope(big.d, big[f'evals_{src}']):.3f}")
    # How much of the exponent comes from lambda / (numerator of c1 + c_mu)?
    num = 2 + 2 * (df.mu_eff - 2 + 1 / df.mu_eff)
    print(f"log-term correction (slope of lambda / (2 mu_eff - 2 + 2/mu_eff)): "
          f"all d = {slope(df.d, df['lambda'] / num):.3f}, "
          f"d >= 10 = {slope(big.d, big['lambda'] / num[df.d >= 10]):.3f}")
