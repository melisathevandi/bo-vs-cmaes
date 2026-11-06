"""LaTeX tables for the report, built only from results/summary (no number is typed by hand).

    uv run python code/make_tables.py

Writes report/generated/tab_*.tex (tabular environments; the report wraps them in table floats):
  tab_predictions   hypotheses H1-H5 and the follow-up: statistic, estimate [95% CI], outcome by the pre-registered rule
  tab_e2            E2 primary: Delta_rot per method, noise floor, interactions
  tab_lowdim        E2 low effective dimension: relevance ratio and GP rank correlation at n = 200
  tab_cstar         E1: break-even cost c* by dimension and problem
  tab_e3            E3: P(success) by width and method
  tab_ablation      TuRBO rotation-penalty ablation
  tab_h5            H5: fitted exponent alpha
The outcome column applies the rules written in predictions.md and the addenda; the rule is evaluated here, not by hand.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "summary"
GEN = ROOT / "report" / "generated"
ABL = {"turbo (default)": "TuRBO (ARD-scaled region)", "turbo_cubic_ard": "TuRBO (ARD kernel, cubic region)",
       "turbo_iso": "TuRBO (iso kernel, cubic region)", "bo_ard (E2)": "ARD-BO", "bo_iso (E2)": "iso-BO", "sobol (E2)": "Sobol"}
LABEL = {"bo_ard": "ARD-BO", "bo_iso": "iso-BO", "turbo": "TuRBO", "cma": "CMA-ES", "sep_cma": "sep-CMA-ES",
         "sobol": "Sobol", "cma_ipop": "IPOP-CMA-ES", "bo_logei": "ARD-BO (LogEI)", "bo_ucb": "ARD-BO (UCB)"}


def csv(name):
    return pd.read_csv(SUMMARY / name)


def ci(m, lo, hi, dec=2):
    return f"{m:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"


def tex(df, cols, header, align):
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(name, text):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text(text, encoding="utf-8")


def tab_e2():
    t = csv("e2_primary.csv")
    rows = []
    for m in ("bo_ard", "bo_iso", "turbo", "cma", "sep_cma", "sobol"):
        d = t[(t.quantity == "delta_rot") & (t.method == m)].iloc[0]
        f = t[(t.quantity == "noise_floor") & (t.method == m)].iloc[0]
        rows.append({"m": LABEL[m], "d": ci(d["median"], d.ci_lo, d.ci_hi), "f": ci(f["median"], f.ci_lo, f.ci_hi)})
    for key, lab in (("ard_minus_iso", "ARD-BO $-$ iso-BO"), ("sepcma_minus_cma", "sep-CMA-ES $-$ CMA-ES")):
        i = t[(t.quantity == "interaction") & (t.method == key)].iloc[0]
        rows.append({"m": "Interaction: " + lab, "d": ci(i["median"], i.ci_lo, i.ci_hi), "f": ""})
    write("tab_e2", tex(pd.DataFrame(rows), ["m", "d", "f"],
                        ["Method", "$\\Delta_{rot}$ [95\\% CI]", "Noise floor [99\\% CI]"], "lcc"))


def tab_lowdim():
    t = csv("e2_lowdim_rho.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append({"i": r.instance, "rho": f"{r.rho_median:.3f} [{r.rho_q1:.3f}, {r.rho_q3:.3f}]",
                     "thr": r.threshold.replace(">=", "$\\geq$").replace("<=", "$\\leq$").replace("rho", "$\\rho$"),
                     "frac": f"{int(round(r.frac_runs_meeting * r.n_runs))}/{int(r.n_runs)}",
                     "ard": f"{r.rank_corr_bo_ard_median:.2f} [{r.rank_corr_bo_ard_q1:.2f}, {r.rank_corr_bo_ard_q3:.2f}]",
                     "iso": f"{r.rank_corr_bo_iso_median:.2f} [{r.rank_corr_bo_iso_q1:.2f}, {r.rank_corr_bo_iso_q3:.2f}]"})
    write("tab_lowdim", tex(pd.DataFrame(rows), ["i", "rho", "thr", "frac", "ard", "iso"],
                            ["Instance", "$\\rho$ median [IQR]", "Threshold", "Runs meeting", "Rank corr. ARD-BO",
                             "Rank corr. iso-BO"], "lccccc"))


def tab_cstar():
    t = csv("cost_table.csv")
    t = t[t.target == "own"].copy()
    t["d"] = t.experiment.str.extract(r"e1_d(\d+)")[0].astype(int)
    rows = []
    for d in sorted(t.d.unique()):
        row = {"d": d}
        for key, p in (("e", "ell_k100_rotated"), ("r", "rosenbrock")):
            r = t[(t.d == d) & (t.problem == p)].iloc[0]
            row[key] = f"{r.median_c_star_s:.2f} [{r.q1_c_star_s:.2f}, {r.q3_c_star_s:.2f}]"
            row[key + "n"] = f"{int(r.c_star_defined)}/{int(r.n_seeds)}"
        rows.append(row)
    write("tab_cstar", tex(pd.DataFrame(rows), ["d", "e", "en", "r", "rn"],
                           ["$d$", "Ellipsoid $c^*$ (s) median [IQR]", "Seeds", "Rosenbrock $c^*$ (s) median [IQR]",
                            "Seeds"], "rcccc"))


def tab_e3():
    s = csv("e3_success.csv")
    rows = []
    for src, n in (("short200", 200), ("long2000", 2000)):
        sub = s[s.source == src]
        for w in sorted(sub.width.unique(), reverse=True):
            row = {"n": n, "w": f"{w:g}"}
            for m in ("bo_logei", "bo_ucb", "cma", "cma_ipop", "sobol"):
                r = sub[(sub.method == m) & np.isclose(sub.width, w)]
                row[m] = f"{r.p_success.iloc[0]:.2f}" if len(r) else "--"
            rows.append(row)
    write("tab_e3", tex(pd.DataFrame(rows), ["n", "w", "bo_logei", "bo_ucb", "cma", "cma_ipop", "sobol"],
                        ["$n$", "$w$", "LogEI-BO", "UCB-BO", "CMA-ES", "IPOP", "Sobol"], "rcccccc"))


def tab_ablation():
    t = csv("turbo_ablation.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append({"m": ABL.get(r.method, r.method), "a": f"{r.median_aligned:.2f}", "r": f"{r.median_rotated:.2f}",
                     "d": ci(r.delta_rot, r.delta_rot_lo, r.delta_rot_hi), "f": ci(r.noise_floor, r.noise_floor_lo, r.noise_floor_hi)})
    write("tab_ablation", tex(pd.DataFrame(rows), ["m", "a", "r", "d", "f"],
                              ["Method", "Aligned", "Rotated", "$\\Delta_{rot}$ [95\\% CI]", "Noise floor [95\\% CI]"], "lcccc"))


def tab_h5():
    t = csv("h5_alpha.csv")
    rows = [{"tau": int(r.tau), "fit": r.fit.replace(">=", "$\\geq$"), "a": ci(r.alpha, r.ci_lo, r.ci_hi, 3),
             "n": int(r.n_runs), "c187": "yes" if r["contains_1.87"] else "no", "c2": "yes" if r["contains_2"] else "no"}
            for _, r in t.iterrows()]
    write("tab_h5", tex(pd.DataFrame(rows), ["tau", "fit", "a", "n", "c187", "c2"],
                        ["$\\tau$", "Fit", "$\\hat\\alpha$ [95\\% CI]", "Runs", "Contains 1.87", "Contains 2"], "clcccc"))


def sign_outcome(est, lo, hi, positive_means="supported"):
    if lo > 0:
        return positive_means
    if hi < 0:
        return "opposite sign"
    return "not resolved"


def tab_predictions():
    rows = []

    def add(h, stat, est, outcome):
        for old, new in (("0<S<0.5", "$0<S<0.5$"), ("S>=0.5", r"$S\geq0.5$"), ("S<=0", r"$S\leq0$")):
            outcome = outcome.replace(old, new)
        rows.append({"h": h, "s": stat, "e": est, "o": outcome})

    h1, k4 = csv("e1_h1.csv"), csv("e1_kappa1e4_h1.csv")
    for p, lab in (("ell_k100_rotated", "ellipsoid $\\kappa=100$"), ("rosenbrock", "Rosenbrock")):
        r = h1[(h1.problem == p) & (h1.statistic == "S=D(10)-D(2)")].iloc[0]
        add("H1", f"$S=D(10)-D(2)$, {lab}", ci(r.estimate, r.ci_lo, r.ci_hi), r.verdict_rule)
    r = k4[k4.statistic.str.startswith("S=")].iloc[0]
    add("H1", "$S=D(10)-D(2)$, ellipsoid $\\kappa=10^4$ (supplement)", ci(r.estimate, r.ci_lo, r.ci_hi), r.verdict_rule)
    e2 = csv("e2_primary.csv")
    for key, lab in (("ard_minus_iso", "ARD-BO $-$ iso-BO"), ("sepcma_minus_cma", "sep-CMA-ES $-$ CMA-ES")):
        r = e2[(e2.quantity == "interaction") & (e2.method == key)].iloc[0]
        add("H2", f"$\\Delta_{{rot}}$ interaction, {lab}", ci(r["median"], r.ci_lo, r.ci_hi),
            "supported (CI excludes 0)" if r.ci_lo > 0 else "not supported")
    for _, r in csv("e2_lowdim_rho.csv").iterrows():
        add("H2", f"$\\rho$ at $n=200$, {r.instance} ({r.threshold.replace('>=', '$\\geq$').replace('<=', '$\\leq$').replace('rho', '$\\rho$')})",
            f"{r.rho_median:.3f} [{r.rho_q1:.3f}, {r.rho_q3:.3f}] (IQR)", "threshold met" if r.median_meets else "threshold not met")
    t = csv("e3_tests.csv")
    wd = t[t.test.str.startswith("width dependence")]
    add("H3", "$P(w{=}0.2)-P(w{=}0.0125)$, every method", f"{wd.estimate.min():.2f} to {wd.estimate.max():.2f}",
        "supported (all CIs exclude 0)" if (wd.ci_lo > 0).all() else "not for every method")
    r = t[t.test.str.startswith("UCB - LogEI, w=0.0125")].iloc[0]
    add("H3", "UCB $-$ LogEI success, $w=0.0125$", ci(r.estimate, r.ci_lo, r.ci_hi),
        "supported (UCB better)" if r.ci_lo > 0 else ("refuted (LogEI better)" if r.ci_hi < 0 else "not resolved"))
    r = t[t.test.str.startswith("IPOP - CMA-ES pooled over widths, n=2000")].iloc[0]
    add("H3", "IPOP $-$ CMA-ES success, pooled, $n=2000$", ci(r.estimate, r.ci_lo, r.ci_hi), sign_outcome(r.estimate, r.ci_lo, r.ci_hi))
    for _, r in csv("e1_h4.csv").iterrows():
        if r.source.startswith("own"):
            add("H4", f"Spearman($c^*$, budget), {'ellipsoid' if r.problem == 'ell_k100_rotated' else 'Rosenbrock'}",
                f"{r.spearman_c_star_vs_N:.2f}", f"$c^*$ {r.sign} with budget")
    a = csv("h5_alpha.csv")
    r = a[(a.tau == 10) & (a.fit == "d >= 10")].iloc[0]
    add("H5", "$\\hat\\alpha$, $d\\geq10$, $\\tau=10$", ci(r.alpha, r.ci_lo, r.ci_hi, 3),
        "supported (CI contains 1.87)" if r["contains_1.87"] else "refuted (CI excludes 1.87)")
    v = csv("turbo_ablation_verdict.csv")
    for key, lab in (("cubic ARD prediction", "cubic ARD $\\Delta_{rot}$"), ("isotropic: |delta_rot|", "isotropic $\\Delta_{rot}$")):
        r = v[v["item"].str.startswith(key)].iloc[0]
        tt = csv("turbo_ablation.csv").set_index("method").loc["turbo_cubic_ard" if "cubic" in key else "turbo_iso"]
        add("Follow-up", lab, ci(tt.delta_rot, tt.delta_rot_lo, tt.delta_rot_hi), "prediction met" if r.met else "prediction not met")
    write("tab_predictions", tex(pd.DataFrame(rows), ["h", "s", "e", "o"],
                                 ["", "Statistic", "Estimate [95\\% CI]", "Outcome (rule in \\texttt{predictions.md})"], "llcl"))


if __name__ == "__main__":
    tab_e2()
    tab_lowdim()
    tab_cstar()
    tab_e3()
    tab_ablation()
    tab_h5()
    tab_predictions()
    print("wrote", ", ".join(sorted(p.name for p in GEN.glob("tab_*.tex"))))
