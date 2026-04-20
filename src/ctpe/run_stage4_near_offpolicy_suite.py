from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common


PROFILES = {
    "smoke": {
        "small": (140, 2),
        "medium": (180, 2),
        "large": (240, 1),
        "mismatch_grid": [0.0, 0.15],
    },
    "main": {
        "small": (560, 12),
        "medium": (720, 12),
        "large": (920, 8),
        "mismatch_grid": [0.0, 0.1, 0.2, 0.35, 0.5],
    },
    "heavy": {
        "small": (800, 20),
        "medium": (960, 20),
        "large": (1280, 14),
        "mismatch_grid": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7],
    },
}

MISMATCH_TYPES = ("gain", "covariance", "time_shift")


def rollout_with_mismatch(task, n_episodes: int, rng: np.random.Generator, mismatch_type: str, level: float):
    states = np.zeros((n_episodes, task.horizon_steps + 1, task.state_dim))
    actions = np.zeros((n_episodes, task.horizon_steps, task.action_dim))
    rewards = np.zeros((n_episodes, task.horizon_steps))
    mse_terms = []
    states[:, 0, :] = task.sample_initial_states(n_episodes, rng)
    prev_x = states[:, 0, :].copy()
    for n in range(task.horizon_steps):
        t = n * task.dt
        x = states[:, n, :]
        u_tar = task.target_policy(x, t)
        u_beh = task.behavior_policy(x, t, rng)
        if mismatch_type == "gain":
            u = np.clip((1.0 + level) * u_beh, -6.0, 6.0)
        elif mismatch_type == "covariance":
            extra = (0.35 + 0.05 * task.action_dim) * level * rng.normal(size=u_beh.shape)
            u = np.clip(u_beh + extra, -6.0, 6.0)
        elif mismatch_type == "time_shift":
            shifted_t = max(0.0, t - level)
            u = task.target_policy(x, shifted_t)
            u = np.clip(u + 0.25 * rng.normal(size=u.shape), -6.0, 6.0)
        else:
            raise ValueError(f"Unknown mismatch_type: {mismatch_type}")
        mse_terms.append(np.mean((u - u_tar) ** 2))
        actions[:, n, :] = u
        rewards[:, n] = task.reward_fn(x, t, u)
        states[:, n + 1, :] = task.step_fn(x, t, u, task.dt, rng)
        prev_x = x
    mismatch_mse = float(np.mean(mse_terms))
    return states, actions, rewards, mismatch_mse


def split_with_actions(states, actions, rewards):
    return common.split_with_actions(states, actions, rewards)


def _run_state_methods_from_rollout(task, states, rewards, seed: int, method_orders: Sequence[int | None]):
    splits = base.split_episodes(states, rewards)
    train_states, train_rewards = splits["train"]
    val_states, val_rewards = splits["val"]
    test_states, _ = splits["test"]
    max_order = max([o for o in method_orders if o is not None], default=1)
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=max_order)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=max_order)
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 1234, eval_indices=task.eval_indices)

    rows = []
    for order in method_orders:
        best_score = float("inf")
        best_row = None
        for bw in task.bandwidth_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "method": ("BE" if order is None else f"Gen{order}"),
                "order": -1 if order is None else order,
                "bandwidth_steps": int(bw),
                "validation_score": float(payload["validation_score"]),
                "t0_rmse": float(payload["t0_rmse"]),
                "integrated_rmse": float(payload["integrated_rmse"]),
                "runtime_sec": float(payload["runtime_sec"]),
            }
            if row["validation_score"] < best_score:
                best_score = row["validation_score"]
                best_row = row
        assert best_row is not None
        rows.append(best_row)
    return rows


# A light neighboring CT baseline proxy. This is not a literal implementation of
# An auxiliary continuous-time proxy baseline retained for local diagnostics.
# It minimizes a discrete residual and serves as a local extension point for users
# who want to plug in an alternative comparator.
def fit_cttd_proxy_from_moments(task, moments: dict[str, np.ndarray], bandwidth_steps: int) -> np.ndarray:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    coeffs[N] = base.solve_reg(moments["term_G"], moments["term_y"], task.ridge)
    for n in range(N - 1, -1, -1):
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        C = math.exp(task.beta * task.dt) * base.pooled_tensor(moments["P"], n, bandwidth_steps)
        b = base.pooled_tensor(moments["b_gen_orders"][1], n, bandwidth_steps)
        lhs = (task.beta + 1.0 / task.dt) * G
        rhs = b + (1.0 / task.dt) * C @ coeffs[n + 1]
        coeffs[n] = base.solve_reg(lhs, rhs, task.ridge)
    return coeffs


def validation_score_cttd_proxy(task, coeffs: np.ndarray, moments: dict[str, np.ndarray], bandwidth_steps: int) -> float:
    N = moments["N"]
    vals = []
    for n in range(N - 1, -1, -1):
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        C = math.exp(task.beta * task.dt) * base.pooled_tensor(moments["P"], n, bandwidth_steps)
        b = base.pooled_tensor(moments["b_gen_orders"][1], n, bandwidth_steps)
        res = (task.beta + 1.0 / task.dt) * G @ coeffs[n] - b - (1.0 / task.dt) * C @ coeffs[n + 1]
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


def run_cttd_proxy(task, states, rewards, seed: int) -> dict:
    splits = base.split_episodes(states, rewards)
    train_states, train_rewards = splits["train"]
    val_states, val_rewards = splits["val"]
    test_states, _ = splits["test"]
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=1)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=1)
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 2345, eval_indices=task.eval_indices)
    best_score = float("inf")
    best = None
    for bw in task.bandwidth_grid:
        tic = time.perf_counter()
        coeffs = fit_cttd_proxy_from_moments(task, train_m, bandwidth_steps=bw)
        runtime = time.perf_counter() - tic
        val = validation_score_cttd_proxy(task, coeffs, val_m, bandwidth_steps=bw)
        t0, integrated = base.evaluate_coeffs(task, coeffs, eval_bundle)
        row = {
            "method": "CTTDProxy",
            "order": -1,
            "bandwidth_steps": int(bw),
            "validation_score": float(val),
            "t0_rmse": float(t0),
            "integrated_rmse": float(integrated),
            "runtime_sec": float(runtime),
        }
        if val < best_score:
            best_score = val
            best = row
    assert best is not None
    return best


def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def plot_main(df: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    levels = sorted(df["mismatch_level"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True)
    for col, task in enumerate(tasks):
        sub = df[(df["task_label"] == task) & (df["mismatch_type"] == "gain") & (df["method"].isin(["BE", "Gen2", "CTTDProxy"]))]
        for method in ["BE", "Gen2", "CTTDProxy"]:
            sm = sub[sub["method"] == method]
            y_i, y_t0 = [], []
            for lvl in levels:
                sl = sm[sm["mismatch_level"] == lvl]
                y_i.append(float(sl["integrated_rmse"].mean()))
                y_t0.append(float(sl["t0_rmse"].mean()))
            axes[0, col].plot(levels, y_i, marker="o", label=method)
            axes[1, col].plot(levels, y_t0, marker="o", label=method)
        axes[0, col].set_title(task)
        axes[0, col].grid(alpha=0.15)
        axes[1, col].grid(alpha=0.15)
        axes[1, col].set_xlabel("gain-shift mismatch level")
    axes[0, 0].set_ylabel("integrated RMSE")
    axes[1, 0].set_ylabel(r"$t=0$ RMSE")
    axes[0, 2].legend(loc="upper left")
    fig.suptitle("Stage-4 near-off-policy stress test (gain-shift view)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_pooling_heatmap(df: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    levels = sorted(df[df["mismatch_type"] == "gain"]["mismatch_level"].unique())
    bw_grid = sorted(df[df["method"] == "Gen2"]["bandwidth_steps"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        sub = df[(df["task_label"] == task) & (df["mismatch_type"] == "gain") & (df["method"] == "Gen2")]
        mat = np.zeros((len(bw_grid), len(levels)))
        for i, bw in enumerate(bw_grid):
            for j, lvl in enumerate(levels):
                sel = sub[sub["mismatch_level"] == lvl]
                if len(sel) == 0:
                    continue
                mat[i, j] = np.mean(sel["bandwidth_steps"].to_numpy() == bw)
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        ax.set_title(task)
        ax.set_xticks(np.arange(len(levels)), labels=[f"{x:.2f}" for x in levels])
        ax.set_yticks(np.arange(len(bw_grid)), labels=[str(int(bw)) for bw in bw_grid])
        ax.set_xlabel("gain-shift mismatch level")
        if task == "Small":
            ax.set_ylabel("selected Gen2 bandwidth")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("selection frequency")
    fig.savefig(out_pdf)
    plt.close(fig)


def write_dashboard(df: pd.DataFrame, out_html: Path) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Integrated RMSE", "T0 RMSE"], horizontal_spacing=0.12)
    sub = df[(df["mismatch_type"] == "gain") & (df["task_label"].isin(["Small", "Medium", "Large"])) & (df["method"].isin(["BE", "Gen2", "CTTDProxy"]))]
    for task in ["Small", "Medium", "Large"]:
        for method in ["BE", "Gen2", "CTTDProxy"]:
            ss = sub[(sub["task_label"] == task) & (sub["method"] == method)]
            xs, ys1, ys2 = [], [], []
            for lvl in sorted(ss["mismatch_level"].unique()):
                sl = ss[ss["mismatch_level"] == lvl]
                xs.append(float(lvl))
                ys1.append(float(sl["integrated_rmse"].mean()))
                ys2.append(float(sl["t0_rmse"].mean()))
            fig.add_trace(go.Scatter(x=xs, y=ys1, mode="lines+markers", name=f"{task}-{method}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=ys2, mode="lines+markers", name=f"{task}-{method}", showlegend=False), row=1, col=2)
    fig.update_layout(template="plotly_white", title="Stage-4 near-off-policy dashboard")
    fig.update_xaxes(title_text="gain-shift mismatch level", row=1, col=1)
    fig.update_xaxes(title_text="gain-shift mismatch level", row=1, col=2)
    fig.update_yaxes(title_text="integrated RMSE", row=1, col=1)
    fig.update_yaxes(title_text="t0 RMSE", row=1, col=2)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-4 near-off-policy stress suite with multiple mismatch axes.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--include-cttd-proxy", action="store_true", help="Also fit the neighboring CTTD proxy baseline.")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    figures = root / "figures"
    interactive = figures / "interactive"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    interactive.mkdir(parents=True, exist_ok=True)

    task_specs = [
        common.make_small_heavy_task(n_episodes=cfg["small"][0], mc_rollouts=56),
        common.make_medium_heavy_task(n_episodes=cfg["medium"][0], mc_rollouts=56),
        common.make_large_heavy_task(n_episodes=cfg["large"][0], mc_rollouts=48),
    ]
    seeds_by_label = {"Small": cfg["small"][1], "Medium": cfg["medium"][1], "Large": cfg["large"][1]}

    rows: List[dict] = []
    for mismatch_type in MISMATCH_TYPES:
        for task in task_specs:
            for level in cfg["mismatch_grid"]:
                for seed in range(seeds_by_label[task.label]):
                    rng = np.random.default_rng(20270000 + 100000 * MISMATCH_TYPES.index(mismatch_type) + 1000 * (task.label == "Medium") + 2000 * (task.label == "Large") + int(100 * level) + seed)
                    states, actions, rewards, mse = rollout_with_mismatch(task, task.n_episodes, rng, mismatch_type, level)
                    selected = _run_state_methods_from_rollout(task, states, rewards, seed + 11, [None, 2])
                    if args.include_cttd_proxy:
                        selected.append(run_cttd_proxy(task, states, rewards, seed + 29))
                    for row in selected:
                        rows.append({
                            "task": task.name,
                            "task_label": task.label,
                            "seed": seed,
                            "mismatch_type": mismatch_type,
                            "mismatch_level": float(level),
                            "mismatch_metric": mse,
                            **row,
                        })

    df = pd.DataFrame(rows)
    out_csv = results / f"stage4_near_offpolicy_{args.profile}.csv"
    df.to_csv(out_csv, index=False)
    plot_main(df, figures / f"stage4_near_offpolicy_main_{args.profile}.pdf")
    plot_pooling_heatmap(df, figures / f"stage4_near_offpolicy_pooling_{args.profile}.pdf")
    write_dashboard(df, interactive / f"stage4_near_offpolicy_{args.profile}.html")
    print(out_csv)
    print(figures / f"stage4_near_offpolicy_main_{args.profile}.pdf")
    print(figures / f"stage4_near_offpolicy_pooling_{args.profile}.pdf")


if __name__ == "__main__":
    main()
