# Assignment guidelines

Source: https://alexxthiery.github.io/teaching/guidelines/guidelines.html (page captured 2026-10-28; transcribed from the PDF for agent use, so the original page is authoritative). The page links an Overleaf LaTeX template; the link was not captured here.

## What I expect

Read relevant papers and documentation, and explore beyond the starter references. Compare methods or design choices using justified metrics, appropriate baselines, and comparable experimental conditions. Investigate unexpected results, sensitivity to assumptions, and unsuccessful attempts. Explain what the experiments establish and what remains uncertain. A carefully investigated negative result can earn high marks.

## Report and submission

Make **one submission per group on Canvas**, containing:

- A **PDF report of at most 10 pages**, including figures, references, and any appendix, with all group members' names. There is an Overleaf LaTeX template to get started.
- The **GitHub repository URL** matching the report. It should be clearly stated at the start of the report.
- All group members' **names and student numbers** clearly indicated at the start of the report. **Each group can have at most 4 members.**

Describe your methods and experimental setup, interpret the results, and cite external sources. **Do not include code in the main text.** Larger figures and tables may be placed in an appendix.

## Repository and reproducibility

The repository may be **public or private**; if private, **invite GitHub user `alexxthiery`** and provide access for marking.

At the repository root, include `readme.md`, `authors.md`, and `report/`. An example layout is:

```
...
├── readme.md         # Setup, data, experiment, how to reproduce results
├── authors.md        # Name and student ID of every group member
├── report/
│   ├── report.tex    # Or report.md: complete machine-readable report source
│   ...
├── code/             # Experiment and plotting code
└── results/          # Saved numerical results
```

The names `code/`, `configs/`, and `results/` are suggestions. The `readme.md` must explain the environment and dependencies, data preparation, and how to reproduce the main results. The `report/` folder must contain the submitted report's complete **LaTeX or Markdown source**, with its bibliography and figures. And an agent should be able to check the report to verify that the experiments described in the report can be reproduced from the repository.

## Marking

You should demonstrate that you have developed a clear understanding of the problem, designed and conducted experiments rigorously, and drawn evidence-based conclusions. Importantly, you should demonstrate that you have genuinely engaged with the material and contributed your own insights (i.e. not merely let an agentic AI system generate your submission).

| Category | Weight | What is assessed |
|---|---|---|
| Understanding and problem formulation | 25% | Accurate explanations, engagement with sources, and justified choices of methods. |
| Experimental design and exploration | 35% | Breadth and depth, controlled comparisons, and follow-up on findings. |
| Analysis and report quality | 20% | Evidence-based conclusions, discussion of limitations, and clear presentation. |
| Implementation and reproducibility | 20% | Correct, readable code; reproducible results; agreement between the report and repository. |

## AI assistance and academic integrity

AI assistance for ideas, coding, debugging, and writing is allowed. Read the cited sources yourself and verify the code, results, and references. You must understand everything you submit. Discussion with other groups is encouraged, but each group must produce its own submission.

I may use AI-assisted tools to check that the report agrees with the repository's code, configurations, and saved results.
