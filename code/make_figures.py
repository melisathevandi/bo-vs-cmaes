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
W = 6.5                                  # figure width in inches = the report's text width, so sizes below are print sizes
W_COL = 3.3                              # single-column figure width in inches (two-column layout)
RC_COL = {"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
          "legend.fontsize": 7}         # print sizes for single-column figures
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "lines.linewidth": 1.1})


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
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.8))
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
    ax[0].set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30, fontsize=6.5)
    ax[0].set_ylabel(f"median log$_{{10}}$ gap at n={n} (IQR)")
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) final gap: filled = aligned, open = rotated")

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
    ax[1].set_xticks(range(len(allm)), [LABELS[m] for m in allm], rotation=30, fontsize=6.5)
    ax[1].set_ylabel(r"$\Delta$ vs aligned (log$_{10}$ gap)")
    ax[1].set_title("(b) circle: rotated; square: aligned replicate")
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


def _band(ax, tab, prob, key, color, ls, label, marker="o"):
    """Median across seeds with an IQR band (25th to 75th percentile over seeds) at each checkpoint."""
    g = tab[tab.problem == prob].groupby("n")[key]
    q1, med, q3 = g.quantile(0.25), g.median(), g.quantile(0.75)
    ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.18, lw=0)
    ax.plot(med.index, med.values, ls, color=color, marker=marker, ms=2.5, label=label)


def fig2_e2_lowdim():
    """Low effective dimension: convergence, relevance ratio and GP rank correlation (bands: IQR over 20 seeds)."""
    if not have("e2_lowdim"):
        SKIPPED.append("fig2 (e2_lowdim not run)")
        return
    df = load_results("e2_lowdim")
    n_final = int(df["budget"].max())
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5))
    probs = {"lowdim_d20_deff2_aligned": ("-", "aligned"), "lowdim_d20_deff2_rotated": ("--", "rotated")}
    ns = np.unique(np.linspace(10, n_final, 40).astype(int))
    for m in ("bo_ard", "bo_iso", "cma", "sobol", "turbo"):
        for prob, (ls, lab) in probs.items():
            sub = df[(df.method == m) & (df.problem == prob)]
            if sub.empty:
                continue
            med = [np.median([log_gap(F, fs, n) for F, fs in zip(sub.F, sub.f_star)]) for n in ns]
            ax[0].plot(ns, med, ls, color=COLORS[m], label=f"{LABELS[m]} {lab}")
    ax[0].set_xlabel("evaluations")
    ax[0].set_ylabel("median log$_{10}$ gap")
    ax[0].set_title("(a) convergence")
    ard = _checkpoint_table(df, "bo_ard")
    iso = _checkpoint_table(df, "bo_iso")
    for prob, (ls, lab) in probs.items():
        _band(ax[1], ard, prob, "rho", COLORS["bo_ard"], ls, f"ARD-BO, {lab}")
        _band(ax[2], ard, prob, "rank_corr", COLORS["bo_ard"], ls, f"ARD-BO, {lab}")
        _band(ax[2], iso, prob, "rank_corr", COLORS["bo_iso"], ls, f"iso-BO, {lab}")
    ax[1].axhline(2 / 20, color="k", lw=0.7, ls=":", label="isotropic value $d_{eff}/d$")
    ax[1].set_ylabel(r"relevance ratio $\rho$")
    ax[1].set_title("(b) relevance ratio $\\rho$")
    ax[2].set_ylabel("Spearman rank correlation")
    ax[2].set_title("(c) GP rank correlation")
    for a_ in ax[1:]:
        a_.set_xlabel("evaluations n")
    # one shared legend above the panels: colour = method, line style = instance, dotted = isotropic value in (b)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[m], lw=1.5) for m in ("bo_ard", "bo_iso", "cma", "turbo", "sobol")]
    names = [LABELS[m] for m in ("bo_ard", "bo_iso", "cma", "turbo", "sobol")]
    handles += [Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--"), Line2D([], [], color="k", ls=":", lw=0.7)]
    names += ["aligned", "rotated", "isotropic value $d_{eff}/d$ (b)"]
    from matplotlib.patches import Patch
    handles.append(Patch(color="0.5", alpha=0.3, lw=0))
    names.append("IQR over seeds (b, c)")
    order_ = [0, 5, 1, 6, 2, 7, 3, 8, 4, None]            # matplotlib fills columns first: methods on row 1, line styles on row 2
    handles = [handles[i] if i is not None else Line2D([], [], alpha=0) for i in order_]
    names = [names[i] if i is not None else "" for i in order_]
    fig.legend(handles, names, loc="upper center", ncol=5, frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig2_e2_lowdim.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    if not ard.empty:
        for prob, tag in (("lowdim_d20_deff2_aligned", "aligned"), ("lowdim_d20_deff2_rotated", "rotated")):
            s_ = ard[(ard.problem == prob) & (ard.n == 200)]
            if len(s_):
                number(f"rhontwohundred{tag}", float(s_.rho.median()))
                number(f"rankcorrntwohundred{tag}", float(s_.rank_corr.median()))


def _regime_panel(ax, exps, prob, title):
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(exps):
        df = load_results(e)
        d = int(e.split("_d")[1])
        N = int(df["budget"].max())
        grid = np.unique(np.linspace(2 * d, N, 25).astype(int))
        sub = df[df.problem == prob]
        diffs = [np.median((gap_table(sub, n)["bo_ard"] - gap_table(sub, n)["cma"]).dropna()) for n in grid]
        ax.plot(grid / d, diffs, color=cmap(i / max(len(exps) - 1, 1)), label=f"d={d}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("evaluations / d")
    ax.set_ylabel("median (BO $-$ CMA-ES) log$_{10}$ gap")
    ax.set_title(title)


def fig3_e1_h5():
    """(a, b) regime maps, (c) the H1 statistic D(d) at both kappa, (d) CMA-ES learning time against d."""
    exps = [e for e in ("e1_d2", "e1_d3", "e1_d5", "e1_d8", "e1_d10") if have(e)]
    h5 = csv("h5_learning_time.csv")
    if not exps and h5 is None:
        SKIPPED.append("fig3 (e1 and h5 not run)")
        return
    fig, ax = plt.subplots(2, 2, figsize=(W, 5.2))
    _regime_panel(ax[0, 0], exps, "ell_k100_rotated", "(a) regime map: ellipsoid, $\\kappa=100$")
    ax[0, 0].legend(loc="lower right", ncol=2)
    _regime_panel(ax[0, 1], exps, "rosenbrock", "(b) regime map: Rosenbrock")

    h1, k4 = csv("e1_h1.csv"), csv("e1_kappa1e4_h1.csv")
    axc = ax[1, 0]
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
            axc.errorbar(x + (k - 1) * 0.12, y, yerr=[y - lo, hi - y], fmt=mark + "-", color=col, ms=3, lw=0.8,
                         capsize=2, label=lab)
        axc.axhline(0, color="k", lw=0.6)
        axc.set_xticks([2, 3, 5, 8, 10], ["2", "3", "5", "8", "10"])
        axc.set_xlabel("$d$")
        axc.set_ylabel("D(d) at 20 evaluations per dimension")
        axc.legend(loc="lower right")
        axc.set_title("(c) H1 statistic D(d) (BO $-$ CMA-ES, 95% CI)")
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
        _h5_panel(ax[1, 1], h5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_e1_h5.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _h5_panel(ax, h5):
    """CMA-ES learning time T (evaluations to kappa_t < 10) against d with the fitted and the predicted slopes."""
    t10 = h5[h5.tau == 10]
    t = t10[t10["T"].notna()]
    g = t.groupby("d")["T"]
    med = g.median()
    ax.errorbar(med.index, med.values, yerr=[med - g.quantile(0.25), g.quantile(0.75) - med], fmt="o", ms=3,
                color="k", capsize=1.5, label="median, IQR (runs with $T$)", zorder=3)
    a = csv("h5_alpha.csv")
    centre = {}
    for dmin, ls, col, name, tag in ((0, "-", "#1f77b4", "fit, all $d$", "all"), (10, "--", "#ff7f0e", "fit, $d\\geq10$", "dgeten")):
        sub = t[t.d >= dmin]
        x, y = np.log(sub.d.to_numpy(float)), np.log(sub["T"].to_numpy(float))
        slope, icpt = np.polyfit(x, y, 1)
        csv_alpha = a[(a.tau == 10) & (a.fit == ("all d" if dmin == 0 else "d >= 10"))].alpha.iloc[0]
        assert abs(slope - csv_alpha) < 1e-6, (slope, csv_alpha)          # the plotted line is the fit in h5_alpha.csv
        dd = np.array([2.0, 40.0]) if dmin == 0 else np.array([10.0, 40.0])     # each fit is drawn over the d it used
        ax.plot(dd, np.exp(icpt) * dd ** slope, ls, color=col, lw=1.4, label=f"{name}: $\\hat\\alpha$={slope:.2f}", zorder=2)
        centre[tag] = (x.mean(), y.mean())                                       # centroid of the runs used by this fit
        row = a[(a.tau == 10) & (a.fit == ("all d" if dmin == 0 else "d >= 10"))].iloc[0]
        number(f"alpha{tag}", ci_str(row.alpha, row.ci_lo, row.ci_hi), "{}")
    # 1.71 is the prediction for the all-d fit (anchored at the all-d centroid); 1.87 and 2 are for d >= 10 (d >= 10 centroid)
    for slope, tag, col in ((1.71, "all", "#7f7f7f"), (1.87, "dgeten", "#2ca02c"), (2.0, "dgeten", "#d62728")):
        xm, ym = centre[tag]
        dd = np.array([2.0, 40.0]) if tag == "all" else np.array([10.0, 40.0])
        ax.plot(dd, np.exp(ym) * (dd / np.exp(xm)) ** slope, ":", color=col, lw=1.3,
                label=f"predicted {slope:g}, " + ("all-$d$ centroid" if tag == "all" else "$d\\geq10$ centroid"), zorder=1)
    n_tot = t10.groupby("d").size()
    n_T = g.size().reindex(n_tot.index, fill_value=0)
    miss = n_tot[n_T < n_tot].index
    lines = [f"$d$={d}: $T$ observed in {int(n_T[d])}/{int(n_tot[d])} runs" for d in miss]
    ax.text(0.98, 0.03, "no $T$ (machine precision first):\n" + "\n".join(lines), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax.set_xlim(1.7, 55)
    ax.set_xlabel("$d$")
    ax.set_ylabel("evaluations until $\\kappa_t<10$")
    ax.set_title("(d) CMA-ES learning time, $\\tau=10$")
    ax.set_ylim(top=ax.get_ylim()[1] * 12)                        # headroom: the legend must not cover the data
    ax.legend(loc="upper left", fontsize=5.5, ncol=2, columnspacing=0.8, handlelength=1.6)


def fig4_e3():
    """E3 at d = 2: P(success) against basin width, Wilson intervals, Sobol as reference."""
    s = csv("e3_success.csv")
    if s is None:
        SKIPPED.append("fig4 (e3 not analysed)")
        return
    col = {"bo_logei": "#1f77b4", "bo_ucb": "#9467bd", "cma": "#d62728", "cma_ipop": "#8c564b", "sobol": "#7f7f7f"}
    lab = {"bo_logei": "ARD-BO LogEI", "bo_ucb": "ARD-BO UCB", "cma": "CMA-ES", "cma_ipop": "IPOP-CMA-ES", "sobol": "Sobol"}
    # single-column figure (W_COL wide): panels stacked, 7-7.5 pt text at print size
    with plt.rc_context(RC_COL):
        fig, ax = plt.subplots(2, 1, figsize=(W_COL, 3.9), sharex=True, sharey=True)
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
            a.set_xticks([0.0125, 0.025, 0.05, 0.1, 0.2], ["0.0125", "0.025", "0.05", "0.1", "0.2"])
            a.set_ylabel("P(success)")
            a.set_title(title)
        ax[1].set_xlabel("basin width $w$")
        ax[0].legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.5)
        ax[0].set_ylim(-0.05, 1.45)                         # headroom for the legend above the curves
        ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1])            # probabilities only (no ticks in the headroom)
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
    nice = {"turbo (default)": "TuRBO (ARD-scaled region)", "turbo_cubic_ard": "TuRBO (ARD kernel, cubic region)",
            "turbo_iso": "TuRBO (iso kernel, cubic region)", "bo_ard (E2)": "ARD-BO", "bo_iso (E2)": "iso-BO", "sobol (E2)": "Sobol"}
    tt = t.set_index("method").loc[order]
    x = np.arange(len(order))
    from matplotlib.lines import Line2D
    mcol = {"turbo (default)": COLORS["turbo"], "turbo_cubic_ard": COLORS["turbo"], "turbo_iso": COLORS["turbo"],
            "bo_ard (E2)": COLORS["bo_ard"], "bo_iso (E2)": COLORS["bo_iso"], "sobol (E2)": COLORS["sobol"]}   # as in Figs 1, 2
    wrap = {"turbo (default)": "TuRBO (ARD-\nscaled region)", "turbo_cubic_ard": "TuRBO (ARD kernel,\ncubic region)",
            "turbo_iso": "TuRBO (iso kernel,\ncubic region)"}                  # the names of `nice`, on two lines
    # single-column figure (W_COL wide): panels stacked, 7-7.5 pt text at print size
    with plt.rc_context(RC_COL):
        fig, ax = plt.subplots(2, 1, figsize=(W_COL, 4.6), sharex=True)
        for off, lab, mark, fill in ((-0.15, "aligned", "o", True), (0.15, "rotated", "s", False)):
            for i, o in enumerate(order):
                r = tt.loc[o]
                med = r[f"median_{lab}"]
                ax[0].errorbar(i + off, med, yerr=[[med - r[f"q1_{lab}"]], [r[f"q3_{lab}"] - med]], fmt=mark,
                               color=mcol[o], mfc=mcol[o] if fill else "white", ms=4, capsize=2)
        ax[0].legend([Line2D([], [], marker="o", color="k", ls=""), Line2D([], [], marker="s", color="k", mfc="white", ls="")],
                     ["aligned (filled)", "rotated (open)"], loc="upper left")
        ax[0].set_ylabel("median log$_{10}$ gap, n=500")
        ax[0].set_title("(a) final gap at n=500 (median, IQR)")
        for off, key, mark in ((-0.15, "delta_rot", "o"), (0.15, "noise_floor", "s")):
            for i, o in enumerate(order):
                r = tt.loc[o]
                ax[1].errorbar(i + off, r[key], yerr=[[r[key] - r[f"{key}_lo"]], [r[f"{key}_hi"] - r[key]]], fmt=mark,
                               ms=4, capsize=2, color=mcol[o], mfc=mcol[o] if mark == "o" else "white")
        ax[1].legend([Line2D([], [], marker="o", color="k", ls=""), Line2D([], [], marker="s", color="k", mfc="white", ls="")],
                     ["$\\Delta_{rot}$ (filled)", "noise floor (open)"], loc="upper right")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xticks(x, [wrap.get(o, nice[o]) for o in order], rotation=35, ha="right", rotation_mode="anchor")
        ax[1].set_ylabel("log$_{10}$ gap difference")
        ax[1].set_title("(b) rotation penalty and noise floor")
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
