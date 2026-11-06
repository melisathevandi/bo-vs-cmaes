"""Figures and LaTeX numbers for the report, built only from files in results/.

    uv run python code/make_figures.py

Writes report/figures/*.pdf (git-ignored, regenerated) and report/generated/numbers.tex (tracked),
a list of \\newcommand macros so that every number in the report text comes from a results file.
A figure or number whose input has not been run is skipped and listed at the end.
Macro names contain letters only (TeX), so d = 2 is written "two" and so on.
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
WORD = {2: "two", 3: "three", 5: "five", 8: "eight", 10: "ten", 20: "twenty"}
PROB_WORD = {"ell_k100_rotated": "ellipsoid", "rosenbrock": "rosenbrock"}
NUMBERS, SKIPPED = {}, []


def have(experiment):
    return (RAW / experiment).exists() and any((RAW / experiment).glob("*.json"))


def number(name, value, fmt="{:.2f}"):
    """Register \\newcommand{\\num<name>}{value}; names may contain letters only."""
    assert name.isalpha(), name
    NUMBERS[name] = fmt.format(value) if not isinstance(value, str) else value


def ci_str(med, lo, hi):
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def csv(name):
    p = SUMMARY / name
    return pd.read_csv(p) if p.exists() else None


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
    fig.savefig(FIG / "fig1_e2_primary.pdf", metadata={"CreationDate": None})
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
    ax[0].legend(fontsize=4.5, ncol=2, loc="upper right")
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
    fig.savefig(FIG / "fig2_e2_lowdim.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    if not ard.empty:
        for prob, tag in (("lowdim_d20_deff2_aligned", "aligned"), ("lowdim_d20_deff2_rotated", "rotated")):
            s = ard[(ard.problem == prob) & (ard.n == 200)]
            if len(s):
                number(f"rhoNtwohundred{tag}".replace("N", "n"), float(s.rho.median()))
                number(f"rankcorrntwohundred{tag}", float(s.rank_corr.median()))


def fig3_e1_h5():
    """(a) regime map, (b) the H1 statistic D(d) at both kappa, (c) CMA-ES learning time against d."""
    exps = [e for e in ("e1_d2", "e1_d3", "e1_d5", "e1_d8", "e1_d10") if have(e)]
    h5 = csv("h5_learning_time.csv")
    if not exps and h5 is None:
        SKIPPED.append("fig3 (e1 and h5 not run)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.7))
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(exps):
        df = load_results(e)
        d = int(e.split("_d")[1])
        N = int(df["budget"].max())
        grid = np.unique(np.linspace(2 * d, N, 25).astype(int))
        for prob, ls in (("ell_k100_rotated", "-"), ("rosenbrock", "--")):
            sub = df[df.problem == prob]
            diffs = [np.median((gap_table(sub, n)["bo_ard"] - gap_table(sub, n)["cma"]).dropna()) for n in grid]
            ax[0].plot(grid / d, diffs, ls, color=cmap(i / max(len(exps) - 1, 1)), lw=1,
                       label=f"d={d}" if ls == "-" else None)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("evaluations / d")
    ax[0].set_ylabel("median (BO $-$ CMA-ES) log$_{10}$ gap", fontsize=7)
    ax[0].legend(fontsize=5)
    ax[0].set_title("(a) regime map (solid ellipsoid, dashed Rosenbrock)", fontsize=7)

    h1, k4 = csv("e1_h1.csv"), csv("e1_kappa1e4_h1.csv")
    if h1 is not None:
        series = []
        for prob, lab, col, mark in (("ell_k100_rotated", "ellipsoid $\\kappa$=100", "#1f77b4", "o"),
                                     ("rosenbrock", "Rosenbrock", "#ff7f0e", "s")):
            r = h1[(h1.problem == prob) & h1.statistic.str.startswith("D(d=")]
            series.append((lab, col, mark, r.statistic.str.extract(r"d=(\d+)")[0].astype(int).to_numpy(),
                           r.estimate.to_numpy(), r.ci_lo.to_numpy(), r.ci_hi.to_numpy()))
            s_row = h1[(h1.problem == prob) & (h1.statistic == "S=D(10)-D(2)")].iloc[0]
            number("HoneS" + PROB_WORD[prob], ci_str(s_row.estimate, s_row.ci_lo, s_row.ci_hi), "{}")
            for _, rr in r.iterrows():
                dd = int(rr.statistic.split("d=")[1].rstrip(")"))
                number(f"Hone{PROB_WORD[prob]}D{WORD[dd]}", ci_str(rr.estimate, rr.ci_lo, rr.ci_hi), "{}")
        if k4 is not None:
            r = k4[k4.statistic.str.contains("at 20 evaluations per dimension") & k4.statistic.str.startswith("D(")]
            series.append(("ellipsoid $\\kappa$=1e4", "#2ca02c", "^", r.d.astype(int).to_numpy(), r.estimate.to_numpy(),
                           r.ci_lo.to_numpy(), r.ci_hi.to_numpy()))
            s_row = k4[k4.statistic.str.startswith("S=")].iloc[0]
            number("HoneSkfour", ci_str(s_row.estimate, s_row.ci_lo, s_row.ci_hi), "{}")
            for _, rr in r.iterrows():
                number(f"HoneKfourD{WORD[int(rr.d)]}", ci_str(rr.estimate, rr.ci_lo, rr.ci_hi), "{}")
        for k, (lab, col, mark, x, y, lo, hi) in enumerate(series):
            ax[1].errorbar(x + (k - 1) * 0.12, y, yerr=[y - lo, hi - y], fmt=mark + "-", color=col, ms=3, lw=0.8,
                           capsize=2, label=lab)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("d")
        ax[1].set_ylabel("D(d), 20 evaluations per dimension", fontsize=7)
        ax[1].legend(fontsize=5)
        ax[1].set_title("(b) H1 statistic (negative = BO ahead)", fontsize=7)
    h4 = csv("e1_h4.csv")
    if h4 is not None:
        for _, r in h4[h4.source.str.startswith("own")].iterrows():
            number("Hfourspearman" + PROB_WORD[r.problem], float(r.spearman_c_star_vs_N))
    ct = csv("cost_table.csv")
    if ct is not None:
        for _, r in ct[ct.target == "own"].iterrows():
            d = int(r.experiment.split("_d")[1])
            number(f"cstar{PROB_WORD[r.problem]}{WORD[d]}", ci_str(r.median_c_star_s, r.q1_c_star_s, r.q3_c_star_s).replace("[", "[IQR ", 1), "{}")
            number(f"cstardefined{PROB_WORD[r.problem]}{WORD[d]}", f"{int(r.c_star_defined)}", "{}")
    rm = csv("e1_regime_map.csv")
    if rm is not None:
        for _, r in rm[(rm.comparison == "bo_ard-cma") & rm.final_budget].iterrows():
            number(f"Honefinal{PROB_WORD[r.problem]}{WORD[int(r.d)]}", ci_str(r.median_diff, r.ci_lo, r.ci_hi), "{}")
    oh = csv("e1_overhead.csv")
    if oh is not None:
        for _, r in oh[oh.method == "bo_ard"].iterrows():
            number(f"overheadboard{PROB_WORD[r.problem]}{WORD[int(r.d)]}", float(r.median_overhead_s_per_eval))
    if h5 is not None:
        t = h5[(h5.tau == 10) & h5["T"].notna()]
        g = t.groupby("d")["T"]
        ax[2].errorbar(g.median().index, g.median().values,
                       yerr=[g.median() - g.quantile(0.25), g.quantile(0.75) - g.median()], fmt="o", ms=3)
        a = csv("h5_alpha.csv")
        for _, r in a[a.tau == 10].iterrows():
            tag = "all" if r.fit == "all d" else "dge10"
            number(f"alpha{tag}".replace("dge10", "dgeten"), ci_str(r.alpha, r.ci_lo, r.ci_hi), "{}")
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        ax[2].minorticks_off()
        ax[2].set_xticks([2, 5, 10, 20, 40, 80], ["2", "5", "10", "20", "40", "80"], fontsize=6)
        ax[2].set_xlabel("d")
        ax[2].set_ylabel("evaluations to $\\kappa_t<10$", fontsize=7)
        ax[2].set_title("(c) CMA-ES learning time", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_e1_h5.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig4_e3():
    """E3 at d = 2: P(success) against basin width, Wilson intervals, Sobol as reference."""
    s = csv("e3_success.csv")
    if s is None:
        SKIPPED.append("fig4 (e3 not analysed)")
        return
    col = {"bo_logei": "#1f77b4", "bo_ucb": "#9467bd", "cma": "#d62728", "cma_ipop": "#8c564b", "sobol": "#7f7f7f"}
    lab = {"bo_logei": "ARD-BO LogEI", "bo_ucb": "ARD-BO UCB", "cma": "CMA-ES", "cma_ipop": "IPOP-CMA-ES", "sobol": "Sobol"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    for a, src, title in ((ax[0], "short200", "(a) 200 evaluations"), (ax[1], "long2000", "(b) 2000 evaluations")):
        for m in col:
            r = s[(s.source == src) & (s.method == m)].sort_values("width")
            if r.empty:
                continue
            a.errorbar(r.width * (1 + 0.04 * (list(col).index(m) - 2)), r.p_success,
                       yerr=[r.p_success - r.wilson_lo, r.wilson_hi - r.p_success], fmt="o-", ms=3, lw=0.8,
                       capsize=1.5, color=col[m], label=lab[m])
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks([0.0125, 0.025, 0.05, 0.1, 0.2], ["0.0125", "0.025", "0.05", "0.1", "0.2"], fontsize=6)
        a.set_xlabel("basin width $w$")
        a.set_title(title, fontsize=8)
    ax[0].set_ylabel("P(success)")
    ax[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_e3.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    t = csv("e3_tests.csv")
    for src, nn in (("short200", "twohundred"), ("long2000", "twothousand")):
        for w, wn in ((0.2, "wide"), (0.0125, "narrow")):
            for m, mn in (("bo_logei", "logei"), ("bo_ucb", "ucb"), ("cma", "cma"), ("cma_ipop", "ipop"), ("sobol", "sobol")):
                r = s[(s.source == src) & (s.method == m) & (np.isclose(s.width, w))]
                if len(r):
                    number(f"eprob{nn}{wn}{mn}", float(r.p_success.iloc[0]))
    for _, r in t.iterrows():
        if r.test.startswith("IPOP - CMA-ES pooled over widths, n=2000"):
            number("ethreeipoppooled", ci_str(r.estimate, r.ci_lo, r.ci_hi), "{}")
        if r.test.startswith("UCB - LogEI, w=0.2,"):
            number("ethreeucbminuslogeiwide", ci_str(r.estimate, r.ci_lo, r.ci_hi), "{}")
        if r.test.startswith("UCB - LogEI, w=0.0125"):
            number("ethreeucbminuslogeinarrow", ci_str(r.estimate, r.ci_lo, r.ci_hi), "{}")


def fig5_ablation():
    """TuRBO rotation-penalty ablation: gaps by problem and Delta_rot with its noise floor."""
    t = csv("turbo_ablation.csv")
    if t is None:
        SKIPPED.append("fig5 (ablation not analysed)")
        return
    order = ["turbo (default)", "turbo_cubic_ard", "turbo_iso", "bo_ard (E2)", "bo_iso (E2)", "sobol (E2)"]
    nice = {"turbo (default)": "TuRBO\n(ARD box)", "turbo_cubic_ard": "TuRBO\nARD, cube", "turbo_iso": "TuRBO\niso, cube",
            "bo_ard (E2)": "ARD-BO", "bo_iso (E2)": "iso-BO", "sobol (E2)": "Sobol"}
    tt = t.set_index("method").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    x = np.arange(len(order))
    for off, lab, mark, fill in ((-0.15, "aligned", "o", True), (0.15, "rotated", "s", False)):
        med = tt[f"median_{lab}"].to_numpy()
        ax[0].errorbar(x + off, med, yerr=[med - tt[f"q1_{lab}"].to_numpy(), tt[f"q3_{lab}"].to_numpy() - med],
                       fmt=mark, color="#1f77b4" if fill else "#d62728", mfc=None if fill else "white", ms=4, capsize=2,
                       label=lab)
    ax[0].set_xticks(x, [nice[o] for o in order], fontsize=6)
    ax[0].set_ylabel("median log$_{10}$ gap at n=500 (IQR)")
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) final gap", fontsize=8)
    for off, key, lab, mark in ((-0.15, "delta_rot", "$\\Delta_{rot}$", "o"), (0.15, "noise_floor", "noise floor", "s")):
        y = tt[key].to_numpy()
        ax[1].errorbar(x + off, y, yerr=[y - tt[f"{key}_lo"].to_numpy(), tt[f"{key}_hi"].to_numpy() - y], fmt=mark,
                       mfc=None if mark == "o" else "white", ms=4, capsize=2, label=lab, color="#2ca02c" if mark == "o" else "k")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x, [nice[o] for o in order], fontsize=6)
    ax[1].set_ylabel("log$_{10}$ gap difference")
    ax[1].legend(fontsize=6)
    ax[1].set_title("(b) rotation penalty and aligned-replicate floor", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_ablation.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    for name, tag in (("turbo (default)", "default"), ("turbo_cubic_ard", "cubic"), ("turbo_iso", "iso")):
        r = tt.loc[name]
        number(f"ablationdrot{tag}", ci_str(r.delta_rot, r.delta_rot_lo, r.delta_rot_hi), "{}")
        number(f"ablationfloor{tag}", ci_str(r.noise_floor, r.noise_floor_lo, r.noise_floor_hi), "{}")
        number(f"ablationaligned{tag}", float(r.median_aligned))
        number(f"ablationrotated{tag}", float(r.median_rotated))
    v = csv("turbo_ablation_verdict.csv")
    r = v[v["item"].str.startswith("cubic ARD - default")].iloc[0]
    number("ablationcontrastcubicdefault", ci_str(r.value, r.ci_lo, r.ci_hi), "{}")


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
    fig4_e3()
    fig5_ablation()
    write_numbers()
    print(f"{len(NUMBERS)} numbers written to {GEN / 'numbers.tex'}")
    for s in SKIPPED:
        print("skipped:", s)
