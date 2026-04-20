from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common


PROFILES = {
    "smoke": {
        "small": (120, 2),
        "medium": (140, 2),
        "large": (180, 1),
        "xlarge": (220, 1),
    },
    "main": {
        "small": (480, 16),
        "medium": (560, 16),
        "large": (720, 10),
        "xlarge": (880, 8),
    },
    "heavy": {
        "small": (720, 24),
        "medium": (840, 24),
        "large": (1080, 14),
        "xlarge": (1280, 10),
    },
}


def seed_list(start: int, count: int) -> list[int]:
    return [start + i for i in range(count)]


def make_tasks(profile: str):
    cfg = PROFILES[profile]
    if profile == "smoke":
        return [
            (common.make_small_heavy_task(n_episodes=cfg["small"][0], horizon_steps=20, dt=0.08, mc_rollouts=8), seed_list(20261101, cfg["small"][1])),
            (common.make_medium_heavy_task(n_episodes=cfg["medium"][0], horizon_steps=20, dt=0.08, mc_rollouts=8), seed_list(20261201, cfg["medium"][1])),
            (common.make_large_heavy_task(n_episodes=cfg["large"][0], horizon_steps=16, dt=0.08, mc_rollouts=6, feature_family="linear"), seed_list(20261301, cfg["large"][1])),
            (common.make_xlarge_heavy_task(n_episodes=cfg["xlarge"][0], horizon_steps=16, dt=0.08, mc_rollouts=4, feature_family="linear"), seed_list(20261401, cfg["xlarge"][1])),
        ]
    return [
        (common.make_small_heavy_task(n_episodes=cfg["small"][0]), seed_list(20261101, cfg["small"][1])),
        (common.make_medium_heavy_task(n_episodes=cfg["medium"][0]), seed_list(20261201, cfg["medium"][1])),
        (common.make_large_heavy_task(n_episodes=cfg["large"][0]), seed_list(20261301, cfg["large"][1])),
        (common.make_xlarge_heavy_task(n_episodes=cfg["xlarge"][0]), seed_list(20261401, cfg["xlarge"][1])),
    ]


def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def fmt_pm(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def eval_model_over_time(task, model, test_states, seed, time_grid: Sequence[int]) -> List[dict]:
    rows = []
    for idx in time_grid:
        bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 5000 + idx, eval_indices=[idx])
        rmse0, irmse = common.evaluate_model_based_value(task, model, bundle, seed + 7000 + idx)
        rows.append({"time_index": int(idx), "time": task.dt * idx, "rmse": irmse})
    return rows


def fit_mb_selected(task, train_states, train_actions, train_rewards, val_states, val_actions, val_rewards, eval_bundle, bandwidth_grid: Sequence[int], feature_kind: str = "linear"):
    best_score = float("inf")
    best_row = None
    best_model = None
    candidate_rows = []
    for bw in bandwidth_grid:
        tic = time.perf_counter()
        model = common.fit_model_based_baseline(task, train_states, train_actions, train_rewards, bandwidth_steps=bw, feature_kind=feature_kind)
        runtime = time.perf_counter() - tic
        val_score = common.model_validation_score(task, model, val_states, val_actions, val_rewards)
        t0_rmse, integrated_rmse = common.evaluate_model_based_value(task, model, eval_bundle, seed=91000 + 17 * bw)
        row = {
            "method": f"MB{feature_kind.capitalize()}",
            "order": -1,
            "bandwidth_steps": int(bw),
            "validation_score": val_score,
            "t0_rmse": t0_rmse,
            "integrated_rmse": integrated_rmse,
            "runtime_sec": runtime,
            "feature_kind": feature_kind,
        }
        candidate_rows.append(row)
        if val_score < best_score:
            best_score = val_score
            best_row = row
            best_model = model
    assert best_row is not None and best_model is not None
    return candidate_rows, best_row, best_model


def run_seed(task, seed: int) -> tuple[List[dict], List[dict], List[dict]]:
    rng = np.random.default_rng(seed)
    states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior")
    splits = common.split_with_actions(states, actions, rewards)
    train_states, train_actions, train_rewards = splits["train"]
    val_states, val_actions, val_rewards = splits["val"]
    test_states, _, _ = splits["test"]

    max_order = 2
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=max_order)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=max_order)
    time_grid = sorted({0, max(1, task.horizon_steps // 6), max(1, task.horizon_steps // 3), max(1, task.horizon_steps // 2), max(1, 2 * task.horizon_steps // 3), max(1, 5 * task.horizon_steps // 6), task.horizon_steps - 2})
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 999, eval_indices=time_grid)

    candidate_rows: List[dict] = []
    selected_rows: List[dict] = []
    over_time_rows: List[dict] = []

    for order in [None, 2]:
        best_score = float("inf")
        best_coeffs = None
        best_row = None
        for bw in task.bandwidth_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "task": task.name,
                "label": task.label,
                "seed": seed,
                "method": ("BE" if order is None else "Gen2"),
                "order": -1 if order is None else order,
                "bandwidth_steps": int(bw),
                "validation_score": payload["validation_score"],
                "t0_rmse": payload["t0_rmse"],
                "integrated_rmse": payload["integrated_rmse"],
                "runtime_sec": payload["runtime_sec"],
            }
            candidate_rows.append(row)
            if payload["validation_score"] < best_score:
                best_score = payload["validation_score"]
                best_coeffs = payload["coeffs"]
                best_row = row
        assert best_coeffs is not None and best_row is not None
        selected_rows.append(best_row)
        for ot in base.evaluate_over_time(task, best_coeffs, test_states, seed + 314, time_grid):
            over_time_rows.append(
                {
                    "task": task.name,
                    "label": task.label,
                    "seed": seed,
                    "method": best_row["method"],
                    "time_index": ot["time_index"],
                    "time": ot["time"],
                    "rmse": ot["rmse"],
                }
            )

    mb_candidates, mb_row, mb_model = fit_mb_selected(
        task,
        train_states,
        train_actions,
        train_rewards,
        val_states,
        val_actions,
        val_rewards,
        eval_bundle,
        task.bandwidth_grid,
        feature_kind="linear",
    )
    for row in mb_candidates:
        candidate_rows.append({"task": task.name, "label": task.label, "seed": seed, **row})
    selected_rows.append({"task": task.name, "label": task.label, "seed": seed, **mb_row})
    for ot in eval_model_over_time(task, mb_model, test_states, seed + 777, time_grid):
        over_time_rows.append(
            {
                "task": task.name,
                "label": task.label,
                "seed": seed,
                "method": mb_row["method"],
                "time_index": ot["time_index"],
                "time": ot["time"],
                "rmse": ot["rmse"],
            }
        )
    return candidate_rows, selected_rows, over_time_rows


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["Small", "Medium", "Large", "XLarge"]:
        row = {"Scale": label, "Seeds": int((selected["label"] == label).sum() // 3)}
        for method in ["BE", "Gen2", "MBLinear"]:
            sub = selected[(selected["label"] == label) & (selected["method"] == method)]
            m, ci = mean_ci(sub["integrated_rmse"])
            t0, t0ci = mean_ci(sub["t0_rmse"])
            rt, rtci = mean_ci(sub["runtime_sec"])
            row[f"{method} RMSE"] = m
            row[f"{method} CI"] = ci
            row[f"{method} t0"] = t0
            row[f"{method} t0 CI"] = t0ci
            row[f"{method} runtime"] = rt
            row[f"{method} runtime CI"] = rtci
            row[f"{method} mode h"] = int(sub["bandwidth_steps"].mode().iloc[0]) if not sub.empty else -1
        row["Gen2 vs BE gain"] = (row["BE RMSE"] - row["Gen2 RMSE"]) / max(row["BE RMSE"], 1e-12)
        row["Gen2 vs MB gain"] = (row["MBLinear RMSE"] - row["Gen2 RMSE"]) / max(row["MBLinear RMSE"], 1e-12)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_pdf: Path) -> None:
    methods = ["BE", "Gen2", "MBLinear"]
    labels = summary["Scale"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for j, method in enumerate(methods):
        means = summary[f"{method} RMSE"].to_numpy()
        cis = summary[f"{method} CI"].to_numpy()
        ax.bar(x + (j - 1) * width, means, width=width, yerr=cis, capsize=4, label=method)
    for idx, r in summary.iterrows():
        ax.text(idx, max(r["BE RMSE"], r["Gen2 RMSE"], r["MBLinear RMSE"]) + 0.02, f"Gen2 vs BE: {100*r['Gen2 vs BE gain']:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Stage-3 heavy benchmark summary")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_over_time(over_time: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large", "XLarge"]
    methods = ["BE", "Gen2", "MBLinear"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharey=True)
    for ax, task in zip(axes.flat, tasks):
        for method in methods:
            sub = over_time[(over_time["label"] == task) & (over_time["method"] == method)]
            times = sorted(sub["time"].unique())
            means, cis = [], []
            for t in times:
                vals = sub.loc[sub["time"] == t, "rmse"]
                m, ci = mean_ci(vals)
                means.append(m)
                cis.append(ci)
            means = np.asarray(means)
            cis = np.asarray(cis)
            ax.plot(times, means, marker="o", label=method)
            ax.fill_between(times, np.maximum(means - cis, 0.0), means + cis, alpha=0.12)
        ax.set_title(task)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Time")
    axes[0, 0].set_ylabel("RMSE")
    axes[1, 0].set_ylabel("RMSE")
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_pareto(summary: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, r in summary.iterrows():
        for method in ["BE", "Gen2", "MBLinear"]:
            ax.scatter(r[f"{method} runtime"], r[f"{method} RMSE"], s=85)
            ax.annotate(f"{r['Scale']}-{method}", (r[f"{method} runtime"], r[f"{method} RMSE"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Accuracy-runtime Pareto view")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_seed_distribution(selected: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large", "XLarge"]
    methods = ["BE", "Gen2", "MBLinear"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharey=True)
    for ax, task in zip(axes.flat, tasks):
        data = [selected[(selected["label"] == task) & (selected["method"] == m)]["integrated_rmse"].to_numpy() for m in methods]
        ax.boxplot(data, tick_labels=methods, widths=0.65)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Integrated RMSE")
    axes[1, 0].set_ylabel("Integrated RMSE")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_bandwidth_selection(selected: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large", "XLarge"]
    methods = ["BE", "Gen2", "MBLinear"]
    bw_values = sorted({int(v) for v in selected["bandwidth_steps"].unique()})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharey=True)
    for ax, task in zip(axes.flat, tasks):
        x = np.arange(len(bw_values))
        width = 0.24
        for j, method in enumerate(methods):
            counts = []
            sub = selected[(selected["label"] == task) & (selected["method"] == method)]
            for bw in bw_values:
                counts.append(int((sub["bandwidth_steps"] == bw).sum()))
            ax.bar(x + (j - 1) * width, counts, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(bw_values)
        ax.set_title(task)
        ax.set_xlabel("Selected pooling width")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_plotly_summary(summary: pd.DataFrame, out_html: Path) -> None:
    fig = go.Figure()
    for method in ["BE", "Gen2", "MBLinear"]:
        fig.add_bar(name=method, x=summary["Scale"], y=summary[f"{method} RMSE"], error_y=dict(type="data", array=summary[f"{method} CI"]))
    fig.update_layout(barmode="group", template="plotly_white", title="Stage-3 heavy benchmark summary", yaxis_title="Integrated RMSE")
    fig.write_html(out_html, include_plotlyjs="cdn")


def write_plotly_over_time(over_time: pd.DataFrame, out_html: Path) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Small", "Medium", "Large", "XLarge"], shared_yaxes=True)
    loc = {"Small": (1, 1), "Medium": (1, 2), "Large": (2, 1), "XLarge": (2, 2)}
    for task in loc:
        r, c = loc[task]
        for method in ["BE", "Gen2", "MBLinear"]:
            sub = over_time[(over_time["label"] == task) & (over_time["method"] == method)]
            grouped = sub.groupby("time")["rmse"].agg(["mean", "std", "count"]).reset_index()
            grouped["ci"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
            fig.add_trace(
                go.Scatter(
                    x=grouped["time"],
                    y=grouped["mean"],
                    mode="lines+markers",
                    name=f"{task}-{method}",
                    error_y=dict(type="data", array=grouped["ci"], visible=True),
                    showlegend=(task == "Small"),
                ),
                row=r,
                col=c,
            )
    fig.update_layout(template="plotly_white", title="Over-time RMSE profiles", yaxis_title="RMSE")
    fig.write_html(out_html, include_plotlyjs="cdn")


def make_latex_tables(summary: pd.DataFrame, out_main: Path, out_detail: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stage-3 heavy benchmark summary. Entries are mean integrated RMSE $\pm$ 95\% confidence interval across seeds.}",
        r"\label{tab:stage3_heavy_main}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Scale & BE & Gen2 & MBLinear & Gen2 vs BE gain \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(f"{r['Scale']} & {fmt_pm(r['BE RMSE'], r['BE CI'])} & {fmt_pm(r['Gen2 RMSE'], r['Gen2 CI'])} & {fmt_pm(r['MBLinear RMSE'], r['MBLinear CI'])} & {100*r['Gen2 vs BE gain']:.1f}\\% \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_main.write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detailed Stage-3 heavy benchmark summary including runtime and selected pooling width.}",
        r"\label{tab:stage3_heavy_detail}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Scale--Method & Integrated RMSE & $t=0$ RMSE & Runtime (s) & Mode $h$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        for method in ["BE", "Gen2", "MBLinear"]:
            lines.append(
                f"{r['Scale']}--{method} & {fmt_pm(r[f'{method} RMSE'], r[f'{method} CI'])} & {fmt_pm(r[f'{method} t0'], r[f'{method} t0 CI'])} & {fmt_pm(r[f'{method} runtime'], r[f'{method} runtime CI'], 4)} & {int(r[f'{method} mode h'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_detail.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="main")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = common.ensure_dirs(root)

    candidate_rows: List[dict] = []
    selected_rows: List[dict] = []
    over_time_rows: List[dict] = []

    for task, seeds in make_tasks(args.profile):
        for seed in seeds:
            cands, selected, over = run_seed(task, seed)
            candidate_rows.extend(cands)
            selected_rows.extend(selected)
            over_time_rows.extend(over)

    candidate_df = pd.DataFrame(candidate_rows)
    selected_df = pd.DataFrame(selected_rows)
    over_time_df = pd.DataFrame(over_time_rows)
    summary = summarize(selected_df)

    results = paths["results"]
    figures = paths["figures"]
    interactive = paths["interactive"]
    tables = paths["tables"]

    candidate_csv = results / f"stage3_heavy_candidates_{args.profile}.csv"
    selected_csv = results / f"stage3_heavy_selected_{args.profile}.csv"
    over_time_csv = results / f"stage3_heavy_over_time_{args.profile}.csv"
    summary_csv = results / f"stage3_heavy_summary_{args.profile}.csv"
    candidate_df.to_csv(candidate_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    over_time_df.to_csv(over_time_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plot_summary(summary, figures / f"stage3_heavy_summary_{args.profile}.pdf")
    plot_over_time(over_time_df, figures / f"stage3_heavy_over_time_{args.profile}.pdf")
    plot_runtime_pareto(summary, figures / f"stage3_heavy_pareto_{args.profile}.pdf")
    plot_seed_distribution(selected_df, figures / f"stage3_heavy_seed_distribution_{args.profile}.pdf")
    plot_bandwidth_selection(selected_df, figures / f"stage3_heavy_selected_bandwidths_{args.profile}.pdf")

    write_plotly_summary(summary, interactive / f"stage3_heavy_summary_{args.profile}.html")
    write_plotly_over_time(over_time_df, interactive / f"stage3_heavy_over_time_{args.profile}.html")
    make_latex_tables(summary, tables / f"stage3_heavy_main_{args.profile}.tex", tables / f"stage3_heavy_detail_{args.profile}.tex")

    print(candidate_csv)
    print(selected_csv)
    print(over_time_csv)
    print(summary_csv)
    print(figures / f"stage3_heavy_summary_{args.profile}.pdf")
    print(figures / f"stage3_heavy_over_time_{args.profile}.pdf")
    print(figures / f"stage3_heavy_pareto_{args.profile}.pdf")
    print(figures / f"stage3_heavy_seed_distribution_{args.profile}.pdf")
    print(figures / f"stage3_heavy_selected_bandwidths_{args.profile}.pdf")


if __name__ == "__main__":
    main()
