"""Figures and LaTeX numbers for the report, built only from files in results/.

    uv run python code/make_figures.py

Writes report/figures/*.pdf (git-ignored, regenerated) and report/generated/numbers.tex (tracked),
a list of \\newcommand macros so that every number in the report text comes from a results file.
A figure or number whose input has not been run is skipped and listed at the end.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import bootstrap_ci, gap_table, load_results, log_gap
from diagnostics import relevance_ratio
from runner import RAW, make_problem

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"
SUMMARY = ROOT / "results" / "summary"
COLORS = {"bo_ard": "#1f77b4", "bo_iso": "#17becf", "cma": "#d62728", "sep_cma": "#ff7f0e",
          "turbo": "#2ca02c", "sobol": "#7f7f7f"}
LABELS = {"bo_ard": "ARD-BO", "bo_iso": "iso-BO", "cma": "CMA-ES", "sep_cma": "sep-CMA-ES",
          "turbo": "TuRBO", "sobol": "Sobol"}
NUMBERS, SKIPPED = {}, []


def have(experiment):
    return (RAW / experiment).exists() and any((RAW / experiment).glob("*.json"))


def number(name, value, fmt="{:.2f}"):
    """Register \\newcommand{\\name}{value}; names may contain letters only."""
    NUMBERS[name] = fmt.format(value) if not isinstance(value, str) else value


def ci_str(med, lo, hi):
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def fig1_e2_primary():
    """E2 primary (d=10, kappa=1e4 only): median final gap and rotation penalty, Sobol as reference."""
    if not have("e2_primary"):
        SKIPPED.append("fig1 (e2_primary not run)")
        return
    df = load_results("e2_primary")
    n = int(df["budget"].max())
    t = gap_table(df, n)
    methods = [m for m in COLORS if m in t.columns and m != "sobol"]
    probs = {"aligned": "ell_d10_k1e4_aligned", "rotated": "ell_d10_k1e4_rotated",
             "replicate": "ell_d10_k1e4_aligned_rep"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for i, (lab, mark) in enumerate((("aligned", "o"), ("rotated", "s"))):
        for j, m in enumerate(methods):
            x = t.loc[probs[lab]][m].dropna().to_numpy()
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            xp = j + (i - 0.5) * 0.3
            ax[0].plot([xp, xp], [q1, q3], "-", color=COLORS[m], lw=1.2)
            ax[0].plot(xp, med, mark, color=COLORS[m], mfc=COLORS[m] if i == 0 else "white", ms=5)
            number(f"gapmed{m.replace('_', '')}{lab}", med)
    for lab, ls in (("aligned", "--"), ("rotated", ":")):
        sob = np.median(t.loc[probs[lab]]["sobol"].dropna())
        ax[0].axhline(sob, color=COLORS["sobol"], ls=ls, lw=1, label=f"Sobol {lab}")
        number(f"gapmedsobol{lab}", sob)
    ax[0].set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30, fontsize=7)
    ax[0].set_ylabel(f"median log$_{{10}}$ gap at n={n} (IQR)")
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) final gap: filled = aligned, open = rotated", fontsize=8)

    allm = methods + ["sobol"]
    for j, m in enumerate(allm):
        for k, other in enumerate(("rotated", "replicate")):
            d = (t.loc[probs[other]][m] - t.loc[probs["aligned"]][m]).dropna().to_numpy()
            med, lo, hi = bootstrap_ci(d)
            ax[1].errorbar(j + (k - 0.5) * 0.25, med, yerr=[[med - lo], [hi - med]],
                           fmt="o" if k == 0 else "s", color=COLORS[m],
                           mfc=COLORS[m] if k == 0 else "white", capsize=2, ms=4)
            number(f"drot{m.replace('_', '')}{other}", ci_str(med, lo, hi), "{}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(range(len(allm)), [LABELS[m] for m in allm], rotation=30, fontsize=7)
    ax[1].set_ylabel(r"$\Delta$ vs aligned (log$_{10}$ gap)")
    ax[1].set_title("(b) circle: rotated; square: aligned replicate (noise floor)", fontsize=8)
    for name, (a_, b_) in {"interactionBO": ("bo_ard", "bo_iso"), "interactionCMA": ("sep_cma", "cma")}.items():
        d = ((t.loc[probs["rotated"]][a_] - t.loc[probs["aligned"]][a_])
             - (t.loc[probs["rotated"]][b_] - t.loc[probs["aligned"]][b_])).dropna().to_numpy()
        number(name, ci_str(*bootstrap_ci(d)), "{}")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_e2_primary.pdf")
    plt.close(fig)


def _checkpoint_table(df, method):
    rows = []
    for r in df[df.method == method].itertuples():
        d = r.meta.get("diagnostics")
        if not d:
            continue
        spec = json.loads((RAW / "e2_lowdim" / f"{r.job_id}.json").read_text())["spec"]
        prob = make_problem(spec["problem"], spec["instance_seed"])
        for i, n in enumerate(d["checkpoint_n"]):
            rows.append({"problem": r.problem, "seed": r.seed, "n": n, "rank_corr": d["rank_corr"][i],
                         "rho": relevance_ratio(d["lengthscales"][i], prob.P)})
    return pd.DataFrame(rows)


def fig2_e2_lowdim():
    """Low effective dimension: convergence, relevance ratio and GP rank correlation."""
    if not have("e2_lowdim"):
        SKIPPED.append("fig2 (e2_lowdim not run)")
        return
    df = load_results("e2_lowdim")
    n_final = int(df["budget"].max())
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.6))
    style = {"lowdim_d20_deff2_aligned": "-", "lowdim_d20_deff2_rotated": "--"}
    ns = np.unique(np.linspace(10, n_final, 40).astype(int))
    for m in ("bo_ard", "bo_iso", "cma", "sobol", "turbo"):
        for prob, ls in style.items():
            sub = df[(df.method == m) & (df.problem == prob)]
            if sub.empty:
                continue
            med = [np.median([log_gap(F, fs, n) for F, fs in zip(sub.F, sub.f_star)]) for n in ns]
            ax[0].plot(ns, med, ls, color=COLORS[m], label=f"{LABELS[m]} {'aligned' if ls == '-' else 'rotated'}")
    ax[0].set_xlabel("evaluations")
    ax[0].set_ylabel("median log$_{10}$ gap")
    ax[0].legend(fontsize=5)
    ax[0].set_title("(a)", fontsize=8)
    ard = _checkpoint_table(df, "bo_ard")
    iso = _checkpoint_table(df, "bo_iso")
    for prob, ls in style.items():
        for tab, key, a in ((ard, "rho", ax[1]), (ard, "rank_corr", ax[2])):
            s = tab[tab.problem == prob].groupby("n")[key].median()
            a.plot(s.index, s.values, ls, color=COLORS["bo_ard"], marker="o", ms=3)
        s = iso[iso.problem == prob].groupby("n")["rank_corr"].median()
        ax[2].plot(s.index, s.values, ls, color=COLORS["bo_iso"], marker="o", ms=3)
    ax[1].axhline(2 / 20, color="k", lw=0.6, ls=":")
    ax[1].set_title("(b) relevance ratio $\\rho$ (dotted: $d_{eff}/d$)", fontsize=8)
    ax[2].set_title("(c) GP rank correlation", fontsize=8)
    for a in ax[1:]:
        a.set_xlabel("n")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_e2_lowdim.pdf")
    plt.close(fig)
    if not ard.empty:
        for prob, tag in (("lowdim_d20_deff2_aligned", "aligned"), ("lowdim_d20_deff2_rotated", "rotated")):
            s = ard[(ard.problem == prob) & (ard.n == 200)]
            if len(s):
                number(f"rhoNtwohundred{tag}", float(s.rho.median()))
                number(f"rankcorrNtwohundred{tag}", float(s.rank_corr.median()))


def fig3_e1_h5():
    """Regime map (BO - CMA-ES per seed) and CMA-ES learning time against d."""
    exps = [e for e in ("e1_d2", "e1_d3", "e1_d5", "e1_d8", "e1_d10", "e1_d20") if have(e)]
    h5 = SUMMARY / "h5_learning_time.csv"
    if not exps and not h5.exists():
        SKIPPED.append("fig3 (e1 and h5 not run)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for e in exps:
        df = load_results(e)
        d = int(e.split("_d")[1])
        N = int(df["budget"].max())
        grid = np.unique(np.linspace(2 * d, N, 25).astype(int))
        for prob, ls in (("ell_k100_rotated", "-"), ("rosenbrock", "--")):
            sub = df[df.problem == prob]
            diffs = []
            for n in grid:
                t = gap_table(sub, n)
                diffs.append(np.median((t["bo_ard"] - t["cma"]).dropna()))
            ax[0].plot(grid / d, diffs, ls, label=f"d={d}" if ls == "-" else None)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("evaluations / d")
    ax[0].set_ylabel("median (BO $-$ CMA-ES) log$_{10}$ gap")
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) regime map (solid ellipsoid, dashed Rosenbrock)", fontsize=8)
    if h5.exists():
        t = pd.read_csv(h5)
        t = t[(t.tau == 10) & t["T"].notna()]
        g = t.groupby("d")["T"]
        ax[1].errorbar(g.median().index, g.median().values,
                       yerr=[g.median() - g.quantile(0.25), g.quantile(0.75) - g.median()], fmt="o", ms=3)
        a = pd.read_csv(SUMMARY / "h5_alpha.csv")
        for _, r in a[a.tau == 10].iterrows():
            tag = "all" if r.fit == "all d" else "dge10"
            number(f"alpha{tag}", ci_str(r.alpha, r.ci_lo, r.ci_hi), "{}")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("d")
        ax[1].set_ylabel("evaluations to $\\kappa_t<10$")
        ax[1].set_title("(c) CMA-ES learning time", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_e1_h5.pdf")
    plt.close(fig)


def write_numbers():
    GEN.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by code/make_figures.py from results/ ; do not edit."]
    lines += [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in sorted(NUMBERS.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_e2_primary()
    fig2_e2_lowdim()
    fig3_e1_h5()
    write_numbers()
    print(f"{len(NUMBERS)} numbers written to {GEN / 'numbers.tex'}")
    for s in SKIPPED:
        print("skipped:", s)
