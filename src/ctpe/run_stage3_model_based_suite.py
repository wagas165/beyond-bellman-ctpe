from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common


PROFILES = {
    "smoke": {"small": (120, 2), "medium": (140, 2), "large": (180, 1)},
    "main": {"small": (320, 8), "medium": (420, 8), "large": (560, 6)},
    "heavy": {"small": (480, 12), "medium": (620, 12), "large": (820, 8)},
}


def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def run_one(task, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior")
    splits = common.split_with_actions(states, actions, rewards)
    train_states, train_actions, train_rewards = splits["train"]
    val_states, val_actions, val_rewards = splits["val"]
    test_states, _, _ = splits["test"]
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 1111, eval_indices=task.eval_indices)
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=2)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=2)

    rows: list[dict] = []
    for order in [None, 2]:
        best_score = float("inf")
        best = None
        for bw in task.bandwidth_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "task": task.label,
                "seed": seed,
                "method": ("BE" if order is None else "Gen2"),
                "bandwidth_steps": bw,
                "validation_score": payload["validation_score"],
                "t0_rmse": payload["t0_rmse"],
                "integrated_rmse": payload["integrated_rmse"],
                "runtime_sec": payload["runtime_sec"],
            }
            if payload["validation_score"] < best_score:
                best_score = payload["validation_score"]
                best = row
        assert best is not None
        rows.append(best)

    for feature_kind in ["linear", "quadratic"]:
        best_score = float("inf")
        best = None
        for bw in task.bandwidth_grid:
            tic = time.perf_counter()
            model = common.fit_model_based_baseline(task, train_states, train_actions, train_rewards, bw, feature_kind=feature_kind)
            runtime = time.perf_counter() - tic
            val_score = common.model_validation_score(task, model, val_states, val_actions, val_rewards)
            t0_rmse, integrated_rmse = common.evaluate_model_based_value(task, model, eval_bundle, seed + 2222 + bw)
            row = {
                "task": task.label,
                "seed": seed,
                "method": f"MB{feature_kind.capitalize()}",
                "bandwidth_steps": bw,
                "validation_score": val_score,
                "t0_rmse": t0_rmse,
                "integrated_rmse": integrated_rmse,
                "runtime_sec": runtime,
            }
            if val_score < best_score:
                best_score = val_score
                best = row
        assert best is not None
        rows.append(best)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ["Small", "Medium", "Large"]:
        row = {"Scale": task}
        for method in ["BE", "Gen2", "MBLinear", "MBQuadratic"]:
            sub = df[(df["task"] == task) & (df["method"] == method)]
            m, ci = mean_ci(sub["integrated_rmse"])
            rt, rtci = mean_ci(sub["runtime_sec"])
            row[f"{method} RMSE"] = m
            row[f"{method} CI"] = ci
            row[f"{method} runtime"] = rt
            row[f"{method} runtime CI"] = rtci
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_pdf: Path) -> None:
    methods = ["BE", "Gen2", "MBLinear", "MBQuadratic"]
    x = np.arange(len(summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for j, method in enumerate(methods):
        ax.bar(x + (j - 1.5) * width, summary[f"{method} RMSE"], width=width, yerr=summary[f"{method} CI"], capsize=4, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Scale"])
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Broader baseline comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime(summary: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for _, r in summary.iterrows():
        for method in ["BE", "Gen2", "MBLinear", "MBQuadratic"]:
            ax.scatter(r[f"{method} runtime"], r[f"{method} RMSE"], s=85)
            ax.annotate(f"{r['Scale']}-{method}", (r[f"{method} runtime"], r[f"{method} RMSE"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Model-based versus generator baselines")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_plotly(summary: pd.DataFrame, out_html: Path) -> None:
    fig = go.Figure()
    for method in ["BE", "Gen2", "MBLinear", "MBQuadratic"]:
        fig.add_bar(name=method, x=summary["Scale"], y=summary[f"{method} RMSE"], error_y=dict(type="data", array=summary[f"{method} CI"]))
    fig.update_layout(barmode="group", template="plotly_white", title="Model-based baseline suite", yaxis_title="Integrated RMSE")
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="main")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    tasks = [
        (common.make_small_heavy_task(n_episodes=cfg["small"][0], mc_rollouts=56), cfg["small"][1], 20273000),
        (common.make_medium_heavy_task(n_episodes=cfg["medium"][0], mc_rollouts=56), cfg["medium"][1], 20274000),
        (common.make_large_heavy_task(n_episodes=cfg["large"][0], mc_rollouts=48), cfg["large"][1], 20275000),
    ]

    rows: List[dict] = []
    for task, n_seeds, base_seed in tasks:
        for i in range(n_seeds):
            rows.extend(run_one(task, base_seed + i))

    df = pd.DataFrame(rows)
    summary = summarize(df)

    root = Path(__file__).resolve().parents[2]
    paths = common.ensure_dirs(root)
    results = paths["results"]
    figures = paths["figures"]
    interactive = paths["interactive"]
    tables = paths["tables"]

    df.to_csv(results / f"stage3_model_based_suite_{args.profile}.csv", index=False)
    summary.to_csv(results / f"stage3_model_based_summary_{args.profile}.csv", index=False)
    plot_summary(summary, figures / f"stage3_model_based_summary_{args.profile}.pdf")
    plot_runtime(summary, figures / f"stage3_model_based_pareto_{args.profile}.pdf")
    write_plotly(summary, interactive / f"stage3_model_based_summary_{args.profile}.html")

    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Broader baseline comparison adding simple action-conditioned model-based baselines. Entries are mean integrated RMSE $\pm$ 95\% confidence interval across seeds.}",
        r"\label{tab:stage3_model_based}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Scale & BE & Gen2 & MBLinear & MBQuadratic \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        table_lines.append(
            f"{r['Scale']} & {r['BE RMSE']:.3f} $\\pm$ {r['BE CI']:.3f} & {r['Gen2 RMSE']:.3f} $\\pm$ {r['Gen2 CI']:.3f} & {r['MBLinear RMSE']:.3f} $\\pm$ {r['MBLinear CI']:.3f} & {r['MBQuadratic RMSE']:.3f} $\\pm$ {r['MBQuadratic CI']:.3f} \\")
    table_lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    (tables / f"stage3_model_based_{args.profile}.tex").write_text("\n".join(table_lines), encoding="utf-8")

    print(results / f"stage3_model_based_suite_{args.profile}.csv")
    print(results / f"stage3_model_based_summary_{args.profile}.csv")
    print(figures / f"stage3_model_based_summary_{args.profile}.pdf")
    print(figures / f"stage3_model_based_pareto_{args.profile}.pdf")


if __name__ == "__main__":
    main()
