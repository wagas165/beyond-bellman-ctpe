from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ctpe import run_stage2_extended_suite as base


# -----------------------------
# Configuration
# -----------------------------
SMALL_SEEDS = [20260401 + i for i in range(10)]
MEDIUM_SEEDS = [20260501 + i for i in range(10)]
LARGE_SEEDS = [20260601 + i for i in range(5)]


def task_specs() -> list[tuple[base.TaskSpec, list[int]]]:
    return [
        (base.make_small_task(n_episodes=220, nonstat_scale=1.0, dt=0.08, mc_rollouts=64), SMALL_SEEDS),
        (base.make_medium_task(n_episodes=240, nonstat_scale=1.0, dt=0.08, mc_rollouts=64), MEDIUM_SEEDS),
        (base.make_large_task(n_episodes=280, nonstat_scale=1.0, dt=0.08, mc_rollouts=48), LARGE_SEEDS),
    ]


# -----------------------------
# Helpers
# -----------------------------

def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def fmt_pm(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def selected_bandwidth_mode(sub: pd.DataFrame) -> int:
    mode = sub["bandwidth_steps"].mode()
    return int(mode.iloc[0]) if not mode.empty else -1


def summarize_main(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label in ["Small", "Medium", "Large"]:
        be = selected[(selected["label"] == label) & (selected["method"] == "BE")]
        gen = selected[(selected["label"] == label) & (selected["method"] == "Gen2")]
        be_rmse, be_ci = mean_ci(be["integrated_rmse"])
        gen_rmse, gen_ci = mean_ci(gen["integrated_rmse"])
        be_t0, be_t0_ci = mean_ci(be["t0_rmse"])
        gen_t0, gen_t0_ci = mean_ci(gen["t0_rmse"])
        be_rt, be_rt_ci = mean_ci(be["runtime_sec"])
        gen_rt, gen_rt_ci = mean_ci(gen["runtime_sec"])
        rows.append(
            {
                "Scale": label,
                "Seeds": len(be),
                "BE integrated RMSE": be_rmse,
                "BE integrated CI": be_ci,
                "Gen2 integrated RMSE": gen_rmse,
                "Gen2 integrated CI": gen_ci,
                "Relative improvement": (be_rmse - gen_rmse) / max(be_rmse, 1e-12),
                "BE t0 RMSE": be_t0,
                "BE t0 CI": be_t0_ci,
                "Gen2 t0 RMSE": gen_t0,
                "Gen2 t0 CI": gen_t0_ci,
                "BE runtime": be_rt,
                "BE runtime CI": be_rt_ci,
                "Gen2 runtime": gen_rt,
                "Gen2 runtime CI": gen_rt_ci,
                "BE mode bandwidth": selected_bandwidth_mode(be),
                "Gen2 mode bandwidth": selected_bandwidth_mode(gen),
            }
        )
    return pd.DataFrame(rows)


def summarize_ablation(order_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label in ["Small", "Medium"]:
        for method in ["BE", "Gen1", "Gen2", "Gen3"]:
            sub = order_rows[(order_rows["label"] == label) & (order_rows["method"] == method)]
            m, ci = mean_ci(sub["integrated_rmse"])
            t0, t0ci = mean_ci(sub["t0_rmse"])
            rows.append(
                {
                    "Scale": label,
                    "Method": method,
                    "Integrated RMSE": m,
                    "Integrated CI": ci,
                    "t0 RMSE": t0,
                    "t0 CI": t0ci,
                    "Mode bandwidth": selected_bandwidth_mode(sub),
                }
            )
    return pd.DataFrame(rows)


def summarize_pooling(selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label in ["Small", "Medium", "Large"]:
        for method in ["BE", "Gen2"]:
            sel = selected[(selected["label"] == label) & (selected["method"] == method)]
            nopool = candidates[(candidates["label"] == label) & (candidates["method"] == method) & (candidates["bandwidth_steps"] == 0)]
            sel_m, sel_ci = mean_ci(sel["integrated_rmse"])
            no_m, no_ci = mean_ci(nopool["integrated_rmse"])
            rows.append(
                {
                    "Scale": label,
                    "Method": method,
                    "No pooling": no_m,
                    "No pooling CI": no_ci,
                    "Validation-selected pooling": sel_m,
                    "Validation-selected pooling CI": sel_ci,
                    "Improvement from pooling": (no_m - sel_m) / max(no_m, 1e-12),
                }
            )
    return pd.DataFrame(rows)



def _latex_row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def make_latex_main_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main benchmark results for the controlled small-, medium-, and large-scale study. Entries are mean integrated RMSE $\pm$ 95\% confidence interval across seeds. Relative improvement is computed against the backward Bellman baseline.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        _latex_row("Scale", "Bellman baseline", "Generator (order 2)", "Relative improvement"),
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            _latex_row(
                str(r["Scale"]),
                fmt_pm(r["BE integrated RMSE"], r["BE integrated CI"]),
                fmt_pm(r["Gen2 integrated RMSE"], r["Gen2 integrated CI"]),
                f"{100*r['Relative improvement']:.1f}\\%",
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_latex_detailed_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detailed benchmark summary. The table reports mean integrated RMSE, mean initial-time RMSE, mean runtime, and the modal selected pooling window.}",
        r"\label{tab:detailed_results}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        _latex_row("Scale", "Method", "Integrated RMSE", "$t=0$ RMSE", "Runtime (s)", "Mode $h$", "Seeds"),
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        for method in ["BE", "Gen2"]:
            prefix = "BE" if method == "BE" else "Gen2"
            lines.append(
                _latex_row(
                    str(r["Scale"]),
                    method,
                    fmt_pm(r[f"{prefix} integrated RMSE"], r[f"{prefix} integrated CI"]),
                    fmt_pm(r[f"{prefix} t0 RMSE"], r[f"{prefix} t0 CI"]),
                    fmt_pm(r[f"{prefix} runtime"], r[f"{prefix} runtime CI"], 4),
                    str(int(r[f"{prefix} mode bandwidth"])),
                    str(int(r["Seeds"])),
                )
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_latex_ablation_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Order ablation on the small and medium tasks. Entries are mean integrated RMSE $\pm$ 95\% confidence interval across 10 seeds.}",
        r"\label{tab:order_ablation}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        _latex_row("Scale", "BE", "Gen1", "Gen2", "Gen3"),
        r"\midrule",
    ]
    for label in ["Small", "Medium"]:
        vals = []
        for method in ["BE", "Gen1", "Gen2", "Gen3"]:
            r = summary[(summary["Scale"] == label) & (summary["Method"] == method)].iloc[0]
            vals.append(fmt_pm(r["Integrated RMSE"], r["Integrated CI"]))
        lines.append(_latex_row(label, *vals))
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_latex_pooling_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pooling ablation comparing no temporal pooling to validation-selected pooling windows.}",
        r"\label{tab:pooling_ablation}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        _latex_row("Scale--Method", "No pooling", "Selected pooling", "Relative gain"),
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            _latex_row(
                f"{r['Scale']}--{r['Method']}",
                fmt_pm(r["No pooling"], r["No pooling CI"]),
                fmt_pm(r["Validation-selected pooling"], r["Validation-selected pooling CI"]),
                f"{100*r['Improvement from pooling']:.1f}\\%",
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_latex_suite_table(out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled benchmark suite used in the empirical study. Each task is simulated under a fixed target policy and evaluated from logged trajectories generated by the same controller with additive exploration noise and train/validation/test splits.}",
        r"\label{tab:benchmark_suite}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.12\textwidth} >{\raggedright\arraybackslash}p{0.18\textwidth} >{\raggedright\arraybackslash}p{0.12\textwidth} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{0.16\textwidth}}",
        r"\toprule",
        _latex_row("Scale", "Task family", "State dim.", "Time variation", "Seed budget"),
        r"\midrule",
        _latex_row("Small", "Nonlinear pendulum stabilization", "2", "Gravity, damping, and actuator gain vary smoothly over time", "10 seeds"),
        _latex_row("Medium", "Coupled cart-pole-like regulator", "4", "Drift, damping, and control gain vary over the horizon", "10 seeds"),
        _latex_row("Large", "Time-varying networked LQ regulator", "12", "Drift matrix, diffusion, and quadratic costs vary over time", "5 seeds"),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\normalsize",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")

def plot_main_summary(summary: pd.DataFrame, out_pdf: Path) -> None:
    x = np.arange(len(summary))
    width = 0.34
    be_means = summary["BE integrated RMSE"].to_numpy()
    be_ci = summary["BE integrated CI"].to_numpy()
    gen_means = summary["Gen2 integrated RMSE"].to_numpy()
    gen_ci = summary["Gen2 integrated CI"].to_numpy()
    labels = summary["Scale"].tolist()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width/2, be_means, width, yerr=be_ci, capsize=4, label="Bellman baseline")
    ax.bar(x + width/2, gen_means, width, yerr=gen_ci, capsize=4, label="Generator (order 2)")
    for idx, r in summary.iterrows():
        ax.text(idx, max(r["BE integrated RMSE"], r["Gen2 integrated RMSE"]) + 0.02, f"{100*r['Relative improvement']:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Benchmark summary across three problem scales")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_over_time(over_time: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True)
    for ax, task in zip(axes, tasks):
        for method in methods:
            sub = over_time[(over_time["label"] == task) & (over_time["method"] == method)]
            times = sorted(sub["time"].unique())
            means, cis = [], []
            for t in times:
                vals = sub.loc[sub["time"] == t, "rmse"]
                m, ci = mean_ci(vals)
                means.append(m)
                cis.append(ci)
            means = np.array(means)
            cis = np.array(cis)
            ax.plot(times, means, marker="o", label=("Bellman" if method == "BE" else "Generator"))
            ax.fill_between(times, np.maximum(means - cis, 0.0), means + cis, alpha=0.15)
        ax.set_title(task)
        ax.set_xlabel("Time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_order_ablation(ablation: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium"]
    methods = ["BE", "Gen1", "Gen2", "Gen3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, task in zip(axes, tasks):
        sub = ablation[ablation["Scale"] == task]
        means = [float(sub[sub["Method"] == m]["Integrated RMSE"].iloc[0]) for m in methods]
        cis = [float(sub[sub["Method"] == m]["Integrated CI"].iloc[0]) for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=cis, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Integrated RMSE")
    fig.suptitle("Order ablation")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_nonstationarity_heatmap(heat_rows: pd.DataFrame, out_pdf: Path) -> None:
    episode_grid = sorted({int(v) for v in heat_rows["episodes"].unique()})
    amp_grid = sorted({float(v) for v in heat_rows["nonstat_scale"].unique()})
    data = np.zeros((len(amp_grid), len(episode_grid)))
    for i, amp in enumerate(amp_grid):
        for j, ep in enumerate(episode_grid):
            sub = heat_rows[(heat_rows["nonstat_scale"] == amp) & (heat_rows["episodes"] == ep)]
            data[i, j] = float(sub["relative_improvement"].mean())
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(episode_grid)))
    ax.set_xticklabels(episode_grid)
    ax.set_yticks(np.arange(len(amp_grid)))
    ax.set_yticklabels([f"{a:.1f}" for a in amp_grid])
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Nonstationarity scale")
    ax.set_title("Generator improvement over Bellman baseline")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative integrated-RMSE gain")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_pareto(summary: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for _, r in summary.iterrows():
        ax.scatter(r["BE runtime"], r["BE integrated RMSE"], s=90)
        ax.annotate(f"{r['Scale']}-BE", (r["BE runtime"], r["BE integrated RMSE"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.scatter(r["Gen2 runtime"], r["Gen2 integrated RMSE"], s=90)
        ax.annotate(f"{r['Scale']}-Gen2", (r["Gen2 runtime"], r["Gen2 integrated RMSE"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Integrated RMSE")
    ax.set_title("Accuracy-runtime Pareto view")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_seed_distribution(selected: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    for ax, task in zip(axes, tasks):
        data = [selected[(selected["label"] == task) & (selected["method"] == m)]["integrated_rmse"].to_numpy() for m in methods]
        ax.boxplot(data, tick_labels=["BE", "Gen2"], widths=0.6)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Integrated RMSE")
    fig.suptitle("Seed-wise benchmark distribution")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_bandwidth_curves(candidates: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True)
    for col, task in enumerate(tasks):
        for row, metric in enumerate(["validation_score", "integrated_rmse"]):
            ax = axes[row, col]
            for method in methods:
                sub = candidates[(candidates["label"] == task) & (candidates["method"] == method)]
                xs = sorted(sub["bandwidth_steps"].unique())
                means, cis = [], []
                for x in xs:
                    vals = sub.loc[sub["bandwidth_steps"] == x, metric]
                    m, ci = mean_ci(vals)
                    means.append(m)
                    cis.append(ci)
                means = np.array(means)
                cis = np.array(cis)
                ax.plot(xs, means, marker="o", label=method)
                ax.fill_between(xs, np.maximum(means-cis,0.0), means+cis, alpha=0.15)
            ax.set_title(f"{task} -- {'validation residual' if metric=='validation_score' else 'integrated RMSE'}")
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("Bandwidth steps")
            if col == 0:
                ax.set_ylabel("Validation residual" if metric == "validation_score" else "Integrated RMSE")
    axes[0, -1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_selected_bandwidths(selected: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    bw_values = sorted(selected["bandwidth_steps"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    for ax, task in zip(axes, tasks):
        x = np.arange(len(bw_values))
        width = 0.35
        for j, method in enumerate(methods):
            counts = []
            sub = selected[(selected["label"] == task) & (selected["method"] == method)]
            for bw in bw_values:
                counts.append(int((sub["bandwidth_steps"] == bw).sum()))
            ax.bar(x + (j - 0.5) * width, counts, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(bw_values)
        ax.set_xlabel("Selected bandwidth steps")
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Selection count")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_plotly_main(summary: pd.DataFrame, out_html: Path) -> None:
    fig = go.Figure()
    fig.add_bar(name="Bellman baseline", x=summary["Scale"], y=summary["BE integrated RMSE"], error_y=dict(type="data", array=summary["BE integrated CI"]))
    fig.add_bar(name="Generator (order 2)", x=summary["Scale"], y=summary["Gen2 integrated RMSE"], error_y=dict(type="data", array=summary["Gen2 integrated CI"]))
    fig.update_layout(barmode="group", title="Benchmark summary", yaxis_title="Integrated RMSE", template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="cdn")


def write_plotly_over_time(over_time: pd.DataFrame, out_html: Path) -> None:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Small", "Medium", "Large"], shared_yaxes=True)
    col_map = {"Small": 1, "Medium": 2, "Large": 3}
    for task in ["Small", "Medium", "Large"]:
        for method in ["BE", "Gen2"]:
            sub = over_time[(over_time["label"] == task) & (over_time["method"] == method)]
            grouped = sub.groupby("time")["rmse"].agg(["mean", "std", "count"]).reset_index()
            grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
            fig.add_trace(go.Scatter(x=grouped["time"], y=grouped["mean"], mode="lines+markers", name=f"{task}-{method}", error_y=dict(type="data", array=grouped["ci"], visible=True), showlegend=(task == "Small")), row=1, col=col_map[task])
    fig.update_layout(title="Over-time RMSE profiles", yaxis_title="RMSE", template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="cdn")


def write_plotly_heatmap(heat: pd.DataFrame, out_html: Path) -> None:
    episode_grid = sorted({int(v) for v in heat["episodes"].unique()})
    amp_grid = sorted({float(v) for v in heat["nonstat_scale"].unique()})
    z = []
    for amp in amp_grid:
        row = []
        for ep in episode_grid:
            row.append(float(heat[(heat["nonstat_scale"] == amp) & (heat["episodes"] == ep)]["relative_improvement"].mean()))
        z.append(row)
    fig = go.Figure(data=go.Heatmap(x=episode_grid, y=[f"{a:.1f}" for a in amp_grid], z=z, colorbar_title="Relative gain"))
    fig.update_layout(title="Nonstationarity-versus-data heat map", xaxis_title="Number of episodes", yaxis_title="Nonstationarity scale", template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="cdn")


def write_plotly_pareto(summary: pd.DataFrame, out_html: Path) -> None:
    x, y, txt = [], [], []
    for _, r in summary.iterrows():
        x.extend([r["BE runtime"], r["Gen2 runtime"]])
        y.extend([r["BE integrated RMSE"], r["Gen2 integrated RMSE"]])
        txt.extend([f"{r['Scale']}-BE", f"{r['Scale']}-Gen2"])
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers+text", text=txt, textposition="top center"))
    fig.update_xaxes(type="log", title_text="Runtime (seconds)")
    fig.update_yaxes(title_text="Integrated RMSE")
    fig.update_layout(title="Accuracy-runtime Pareto view", template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    figures = root / "figures"
    results = root / "results"
    tables = root / "tables"
    interactive = figures / "interactive"
    for p in [figures, results, tables, interactive]:
        p.mkdir(parents=True, exist_ok=True)

    # Main benchmark: BE vs Gen2 with full seed budget.
    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []
    over_time_rows: list[dict] = []

    for task, seeds in task_specs():
        time_grid = sorted(set([0, max(1, task.horizon_steps // 6), max(1, task.horizon_steps // 3), max(1, task.horizon_steps // 2), int(0.75 * task.horizon_steps), task.horizon_steps - 2]))
        for seed in seeds:
            task_rows, selected, test_states = base.run_task(task, seed, method_orders=[None, 2])
            candidate_rows.extend(task_rows)
            for method in ["BE", "Gen2"]:
                selected_rows.append({k: v for k, v in selected[method].items() if k not in {"coeffs", "test_states"}})
                for ot in base.evaluate_over_time(task, selected[method]["coeffs"], test_states, seed + 314, time_grid):
                    over_time_rows.append(
                        {
                            "task": task.name,
                            "label": task.label,
                            "seed": seed,
                            "method": method,
                            "time_index": ot["time_index"],
                            "time": ot["time"],
                            "rmse": ot["rmse"],
                        }
                    )

    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    over_time = pd.DataFrame(over_time_rows)

    # Order ablation on small and medium tasks.
    order_rows: list[dict] = []
    for task, seeds in [
        (base.make_small_task(n_episodes=220, nonstat_scale=1.0, dt=0.08, mc_rollouts=64), SMALL_SEEDS),
        (base.make_medium_task(n_episodes=240, nonstat_scale=1.0, dt=0.08, mc_rollouts=64), MEDIUM_SEEDS),
    ]:
        for seed in seeds:
            _, selected_ab, _ = base.run_task(task, seed + 700, method_orders=[None, 1, 2, 3])
            for method in ["BE", "Gen1", "Gen2", "Gen3"]:
                order_rows.append({k: v for k, v in selected_ab[method].items() if k not in {"coeffs", "test_states"}})
    order_df = pd.DataFrame(order_rows)

    # Heat map sweep on the small task.
    heat_rows: list[dict] = []
    amp_grid = [0.0, 0.5, 1.0, 1.5]
    episode_grid = [80, 140, 220, 320]
    for amp in amp_grid:
        for episodes in episode_grid:
            task = base.make_small_task(n_episodes=episodes, nonstat_scale=amp, dt=0.08, mc_rollouts=48)
            _, selected_heat, _ = base.run_task(task, 20260700 + int(100 * amp) + episodes, method_orders=[None, 2])
            be = selected_heat["BE"]["integrated_rmse"]
            gen = selected_heat["Gen2"]["integrated_rmse"]
            heat_rows.append(
                {
                    "nonstat_scale": amp,
                    "episodes": episodes,
                    "be_integrated_rmse": be,
                    "gen2_integrated_rmse": gen,
                    "relative_improvement": (be - gen) / max(be, 1e-12),
                }
            )
    heat_df = pd.DataFrame(heat_rows)

    summary = summarize_main(selected)
    ablation_summary = summarize_ablation(order_df)
    pooling_summary = summarize_pooling(selected, candidates)

    # Save raw and summary results.
    write_df(candidates, results / "journal_benchmark_candidates.csv")
    write_df(selected, results / "journal_benchmark_selected.csv")
    write_df(over_time, results / "journal_benchmark_over_time.csv")
    write_df(order_df, results / "journal_order_ablation.csv")
    write_df(heat_df, results / "journal_nonstationarity_heatmap.csv")
    write_df(summary, results / "journal_benchmark_summary.csv")
    write_df(ablation_summary, results / "journal_order_ablation_summary.csv")
    write_df(pooling_summary, results / "journal_pooling_ablation_summary.csv")

    # Tables.
    make_latex_suite_table(tables / "benchmark_suite.tex")
    make_latex_main_table(summary, tables / "main_results.tex")
    make_latex_detailed_table(summary, tables / "detailed_results.tex")
    make_latex_ablation_table(ablation_summary, tables / "order_ablation.tex")
    make_latex_pooling_table(pooling_summary, tables / "pooling_ablation.tex")

    # Static figures.
    plot_main_summary(summary, figures / "journal_benchmark_summary.pdf")
    plot_over_time(over_time, figures / "journal_over_time_rmse.pdf")
    plot_order_ablation(ablation_summary, figures / "journal_order_ablation.pdf")
    plot_nonstationarity_heatmap(heat_df, figures / "journal_nonstationarity_heatmap.pdf")
    plot_runtime_pareto(summary, figures / "journal_runtime_pareto.pdf")
    plot_seed_distribution(selected, figures / "journal_seed_distribution.pdf")
    plot_bandwidth_curves(candidates, figures / "journal_bandwidth_curves.pdf")
    plot_selected_bandwidths(selected, figures / "journal_selected_bandwidths.pdf")

    # Interactive Plotly dashboards.
    write_plotly_main(summary, interactive / "journal_benchmark_summary.html")
    write_plotly_over_time(over_time, interactive / "journal_over_time_rmse.html")
    write_plotly_heatmap(heat_df, interactive / "journal_nonstationarity_heatmap.html")
    write_plotly_pareto(summary, interactive / "journal_runtime_pareto.html")

    print("Wrote journal benchmark results, tables, figures, and interactive dashboards.")
    print(results / "journal_benchmark_summary.csv")
    print(tables / "main_results.tex")
    print(figures / "journal_benchmark_summary.pdf")
    print(interactive / "journal_benchmark_summary.html")


if __name__ == "__main__":
    main()
