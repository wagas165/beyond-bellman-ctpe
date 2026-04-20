# beyond-bellman-ctpe

Code for the paper **“Beyond Bellman: High-Order Generator Regression for Continuous-Time Policy Evaluation.”**

This repository contains the experiment code used to study finite-horizon continuous-time policy evaluation from discretely observed, time-inhomogeneous closed-loop trajectories. The central comparison is between the matched one-step Bellman baseline and higher-order generator-regression estimators built from multi-step moment matching. In the paper, the Bellman baseline is first-order in the decision interval, while Gen2 and Gen3 target higher-order local accuracy through explicit generator surrogates.

The public repository is intentionally **source-only**. Generated figures, CSV summaries, tables, logs, PDFs, and dashboards are not versioned. Users are expected to regenerate numerical artifacts locally from the committed code.

## What is in scope

The codebase is organized around the paper’s empirical program:

- calibration of the discretization-order argument;
- the main benchmark suite comparing the Bellman baseline and Gen2 across multiple task scales;
- model-based anchor comparisons;
- sensitivity analyses for feature restriction, start-up control, runtime scaling, temporal pooling, and gain mismatch;
- regime-diagnostic experiments used to study when higher-order gains are visible.

## Repository structure

```text
beyond-bellman-ctpe/
├── .github/workflows/          # lightweight CI smoke checks
├── README.md
├── pyproject.toml              # package metadata and dependencies
├── requirements.txt            # minimal runtime dependencies
├── .gitignore
├── scripts/                    # thin CLI wrappers
│   ├── reproduce_stage3.py
│   ├── reproduce_stage4_local.py
│   ├── reproduce_stage5.py
│   ├── reproduce_all.py
│   └── ...
├── src/
│   └── ctpe/                   # importable experiment package
│       ├── run_stage2_extended_suite.py
│       ├── run_stage3_heavy_benchmark.py
│       ├── run_stage3_model_based_suite.py
│       ├── run_stage3_sensitivity_suite.py
│       ├── run_stage4_regime_refinement.py
│       ├── run_stage4_near_offpolicy_suite.py
│       ├── run_stage5_clean_regime_suite.py
│       ├── run_stage5_operating_regime_diagnostic.py
│       ├── run_journal_benchmark.py
│       ├── stage3_heavy_common.py
│       ├── stage5_common.py
│       └── ...
└── tests/                      # import-level smoke tests
```

## Installation

Use an isolated Python environment and install the repository in editable mode.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you prefer a requirements-based install:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

Run from the repository root.

A lightweight local check:

```bash
python scripts/reproduce_stage3.py --profile smoke
python scripts/reproduce_stage4_local.py --profile smoke
python scripts/reproduce_stage5.py --profile smoke
```

A fuller experimental pass:

```bash
python scripts/reproduce_stage3.py --profile main
python scripts/reproduce_stage4_local.py --profile main
python scripts/reproduce_stage5.py --profile main
```

End-to-end reproduction:

```bash
python scripts/reproduce_all.py --profile main
```

The journal-style benchmark can also be run directly:

```bash
python scripts/run_journal_benchmark.py
```

## Outputs

The scripts create generated artifacts locally, typically under directories such as:

- `results/` for CSV outputs and selected summaries;
- `figures/` for static figures and interactive dashboards;
- `tables/` for LaTeX tables;
- `logs/` for reproduction logs.

These paths are excluded from version control by default.

## Development notes

- The package name is `ctpe`.
- The repository name is `beyond-bellman-ctpe`.
- The `scripts/` directory contains thin wrappers only; reusable code lives in `src/ctpe/`.
- The current repository is focused on reproducibility of the paper’s experiments rather than on providing a polished general-purpose library API.

## Citation

If you use this repository in academic work, please cite the companion manuscript:

**Yaowei Zheng, Richong Zhang, Shenxi Wu, Shirui Bian, Haosong Zhang, Li Zeng, Xingjian Ma, and Yichi Zhang.**
**Beyond Bellman: High-Order Generator Regression for Continuous-Time Policy Evaluation.**
