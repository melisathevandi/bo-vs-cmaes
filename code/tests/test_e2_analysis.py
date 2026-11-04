import numpy as np
import pandas as pd

from analyze_e2 import ALIGNED, REPLICATE, ROTATED, TOLERANCE, interactions, noise_floor


def table(offsets, rot_offsets, n=30, seed=0):
    """Synthetic wide table: per method a shift of the replicate / rotated gap relative to aligned."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([[ALIGNED, ROTATED, REPLICATE], range(n)], names=["problem", "seed"])
    t = pd.DataFrame(index=idx, columns=list(offsets), dtype=float)
    for m in offsets:
        base = rng.normal(size=n)
        t.loc[ALIGNED, m] = base
        t.loc[REPLICATE, m] = base + offsets[m] + 0.02 * rng.normal(size=n)
        t.loc[ROTATED, m] = base + rot_offsets[m] + 0.02 * rng.normal(size=n)
    return t


def test_noise_floor_passes_when_zero_and_flags_a_large_shift():
    t = table({"a": 0.0, "b": 0.5}, {"a": 0.0, "b": 0.0})
    verdicts = {r["method"]: r["verdict"] for r in noise_floor(t, ["a", "b"])}
    assert verdicts == {"a": "pass", "b": "FLAG"}


def test_noise_floor_tolerates_a_significant_but_small_shift():
    t = table({"a": 0.5 * TOLERANCE}, {"a": 0.0})
    row = noise_floor(t, ["a"])[0]
    assert not row["ci_contains_0"] and row["verdict"] == "pass"


def test_noise_floor_uses_a_99_percent_interval():
    t = table({"a": 0.0}, {"a": 0.0})
    row = noise_floor(t, ["a"])[0]
    assert row["level"] == 0.99


def test_interaction_is_difference_of_rotation_penalties():
    t = table({"x": 0.0, "y": 0.0}, {"x": 0.8, "y": 0.1})
    row = interactions(t, {"x_minus_y": ("x", "y")})[0]
    assert abs(row["median"] - 0.7) < 0.05 and row["supports_h2"]
