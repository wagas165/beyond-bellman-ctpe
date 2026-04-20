from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common
from ctpe import stage5_common as s5


PROFILES = {
    "smoke": {
        "small": (60, 1),
        "medium": (80, 1),
        "large": (100, 1),
        "xlarge": (120, 1),
        "bw_time_grid": [0.0, 0.12, 0.24, 0.36],
    },
    "main": {
        "small": (560, 10),
        "medium": (720, 10),
        "large": (960, 6),
        "xlarge": (1120, 4),
        "bw_time_grid": [0.0, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64, 0.96],
    },
    "heavy": {
        "small": (720, 16),
        "medium": (840, 16),
        "large": (1080, 10),
        "xlarge": (1280, 8),
        "bw_time_grid": [0.0, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64, 0.96, 1.28],
    },
}


def _run_task(task, seed: int, bw_time_grid: Sequence[float]) -> list[dict]:
    rng = np.random.default_rng(seed)
    states, rewards = base.rollout_episodes(task, task.n_episodes, rng, policy_kind="behavior")
    splits = base.split_episodes(states, rewards)
    train_m = base.precompute_moments(task, *splits["train"], max_order=3)
    val_m = base.precompute_moments(task, *splits["val"], max_order=3)
    eval_bundle = common.safe_prepare_eval_bundle(task, splits["test"][0], seed + 202, eval_indices=task.eval_indices)
    bw_steps_grid = s5.bw_time_grid_to_steps(task.dt, bw_time_grid)
    rows = []
    for order, method in [(None, "BE"), (2, "Gen2"), (3, "Gen3")]:
        best = None
        best_score = float("inf")
        for bw in bw_steps_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=int(bw))
            row = {
                "method": method,
                "bandwidth_steps": int(bw),
                "bandwidth_time": float(bw * task.dt),
                "validation_score": float(payload["validation_score"]),
                "integrated_rmse": float(payload["integrated_rmse"]),
                "boundary_hit": int(int(bw) == max(bw_steps_grid)),
            }
            if row["validation_score"] < best_score:
                best = row
                best_score = row["validation_score"]
        assert best is not None
        rows.append(best)
    return rows


def plot_diagnostic(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    labels = ["Small", "Medium", "Large", "XLarge"]
    methods = ["BE", "Gen2", "Gen3"]
    for method in methods:
        sub = df[df["method"] == method]
        means = [float(sub[sub["label"] == lab]["integrated_rmse"].mean()) for lab in labels]
        axes[0].plot(labels, means, marker="o", label=method)
        hits = [float(sub[sub["label"] == lab]["boundary_hit"].mean()) for lab in labels]
        axes[1].plot(labels, hits, marker="o", label=method)
    axes[0].set_ylabel("integrated RMSE")
    axes[1].set_ylabel("boundary-hit rate")
    axes[0].set_title("Operating-regime diagnostic")
    axes[1].set_title("Selected bandwidth saturation")
    for ax in axes:
        ax.grid(alpha=0.18)
    axes[1].legend(frameon=False)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_bandwidth(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = ["Small", "Medium", "Large", "XLarge"]
    x = np.arange(len(labels))
    width = 0.24
    for j, method in enumerate(["BE", "Gen2", "Gen3"]):
        sub = df[df["method"] == method]
        means = [float(sub[sub["label"] == lab]["bandwidth_time"].mean()) for lab in labels]
        ax.bar(x + (j - 1) * width, means, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("selected bandwidth time")
    ax.set_title("Selected physical-time bandwidths")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-5 operating-regime diagnostic.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="heavy")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    if args.profile == "smoke":
        tasks = [
            (common.make_small_heavy_task(n_episodes=cfg["small"][0], mc_rollouts=12), cfg["small"][1]),
            (common.make_medium_heavy_task(n_episodes=cfg["medium"][0], mc_rollouts=12), cfg["medium"][1]),
            (common.make_large_heavy_task(n_episodes=cfg["large"][0], mc_rollouts=10), cfg["large"][1]),
            (common.make_xlarge_heavy_task(n_episodes=cfg["xlarge"][0], mc_rollouts=8), cfg["xlarge"][1]),
        ]
    else:
        tasks = [
            (common.make_small_heavy_task(n_episodes=cfg["small"][0], mc_rollouts=56), cfg["small"][1]),
            (common.make_medium_heavy_task(n_episodes=cfg["medium"][0], mc_rollouts=56), cfg["medium"][1]),
            (common.make_large_heavy_task(n_episodes=cfg["large"][0], mc_rollouts=48), cfg["large"][1]),
            (common.make_xlarge_heavy_task(n_episodes=cfg["xlarge"][0], mc_rollouts=40), cfg["xlarge"][1]),
        ]

    rows: List[dict] = []
    for task, n_seeds in tasks:
        for i in range(n_seeds):
            for row in _run_task(task, 20306000 + 1000 * ["Small", "Medium", "Large", "XLarge"].index(task.label) + i, cfg["bw_time_grid"]):
                rows.append({
                    "task": task.name,
                    "label": task.label,
                    "seed": i,
                    **row,
                })

    df = pd.DataFrame(rows)
    root = Path(__file__).resolve().parents[2]
    paths = common.ensure_dirs(root)
    results = paths["results"]
    figures = paths["figures"]

    out_csv = results / f"stage5_operating_regime_diagnostic_{args.profile}.csv"
    df.to_csv(out_csv, index=False)
    plot_diagnostic(df, figures / f"stage5_operating_regime_diagnostic_{args.profile}.pdf")
    plot_bandwidth(df, figures / f"stage5_operating_regime_selected_bandwidth_{args.profile}.pdf")

    print(out_csv)


if __name__ == "__main__":
    main()
