from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common

Array = np.ndarray
TaskSpec = base.TaskSpec


# -----------------------------------------------------------------------------
# Feature selection helpers
# -----------------------------------------------------------------------------

def feature_fn_from_name(name: str) -> Callable[[Array], Array]:
    if name == "linear":
        return common.linear_features
    if name == "quadratic":
        return lambda x: base.quadratic_features(x, include_linear=True)
    if name == "rich":
        return base.pendulum_features
    if name == "reduced":
        return common.reduced_pendulum_features
    raise ValueError(f"Unknown feature family: {name}")


def task_with_feature(task: TaskSpec, feature_name: str) -> TaskSpec:
    feat = feature_fn_from_name(feature_name)
    ridge = task.ridge
    if feature_name == "linear":
        ridge = max(ridge, 4e-4)
    elif feature_name == "quadratic":
        ridge = max(ridge, 6e-4)
    return replace(task, feature_fn=feat, ridge=ridge)


# -----------------------------------------------------------------------------
# Bandwidth handling
# -----------------------------------------------------------------------------

def bw_time_grid_to_steps(dt: float, bw_time_grid: Sequence[float]) -> list[int]:
    out = []
    for bw in bw_time_grid:
        steps = int(round(float(bw) / float(dt)))
        out.append(max(0, steps))
    return sorted(set(out))


def boundary_hit_rate(selected_bandwidths: Sequence[int], grid: Sequence[int]) -> float:
    arr = np.asarray(list(selected_bandwidths), dtype=int)
    if arr.size == 0:
        return float("nan")
    max_bw = int(max(grid))
    return float(np.mean(arr == max_bw))


# -----------------------------------------------------------------------------
# Continuous-time neighboring baselines
# -----------------------------------------------------------------------------

def fit_dtd_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> Array:
    """Differential-TD style adapter.

    This is the in-project finite-horizon closed-loop adapter used for stage-5.
    It is still an adapter to the current setting, but it is more explicit than the
    stage-4 proxy and is tuned/evaluated under the same split as the other methods.
    """
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    coeffs[N] = base.solve_reg(moments["term_G"], moments["term_y"], task.ridge)
    for n in range(N - 1, -1, -1):
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        # Un-discount the next-feature moment because P already contains exp(-beta dt).
        C = math.exp(task.beta * task.dt) * base.pooled_tensor(moments["P"], n, bandwidth_steps)
        b = base.pooled_tensor(moments["b_gen_orders"][1], n, bandwidth_steps)
        lhs = (task.beta + 1.0 / task.dt) * G
        rhs = b + (1.0 / task.dt) * C @ coeffs[n + 1]
        coeffs[n] = base.solve_reg(lhs, rhs, task.ridge)
    return coeffs


def validation_score_dtd(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int) -> float:
    N = moments["N"]
    vals = []
    for n in range(N - 1, -1, -1):
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        C = math.exp(task.beta * task.dt) * base.pooled_tensor(moments["P"], n, bandwidth_steps)
        b = base.pooled_tensor(moments["b_gen_orders"][1], n, bandwidth_steps)
        res = (task.beta + 1.0 / task.dt) * G @ coeffs[n] - b - (1.0 / task.dt) * C @ coeffs[n + 1]
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


def precompute_jzmo_moments(task: TaskSpec, states: Array, rewards: Array, trace_steps: int) -> dict[str, Array]:
    """Multi-step martingale-orthogonality adapter.

    The moments correspond to the orthogonality condition
        E[phi_n (sum_{k < m} e^{-beta k dt} r_{n+k} dt + e^{-beta m dt} V_{n+m} - V_n)] = 0.
    This is a finite-horizon closed-loop adapter of the martingale view, not vendor code.
    """
    Phi = task.feature_fn(states)
    M, Np1, p = Phi.shape
    N = Np1 - 1
    G = np.zeros((N, p, p))
    C = np.zeros((N, p, p))
    B = np.zeros((N, p))
    disc0 = math.exp(-task.beta * task.dt)
    for n in range(N):
        span = min(int(trace_steps), N - n)
        phi_n = Phi[:, n, :]
        G[n] = (phi_n.T @ phi_n) / M
        disc = np.ones(M)
        ret = np.zeros(M)
        for k in range(span):
            ret += disc * rewards[:, n + k] * task.dt
            disc *= disc0
        B[n] = (phi_n.T @ ret) / M
        C[n] = (phi_n.T @ (disc[:, None] * Phi[:, n + span, :])) / M
    term_G = (Phi[:, N, :].T @ Phi[:, N, :]) / M
    term_y = (Phi[:, N, :].T @ task.terminal_fn(states[:, N, :])) / M
    return {"G": G, "C": C, "B": B, "term_G": term_G, "term_y": term_y, "N": N, "p": p, "trace_steps": int(trace_steps)}


def fit_jzmo_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> Array:
    N = moments["N"]
    p = moments["p"]
    trace_steps = int(moments["trace_steps"])
    coeffs = np.zeros((N + 1, p))
    coeffs[N] = base.solve_reg(moments["term_G"], moments["term_y"], task.ridge)
    for n in range(N - 1, -1, -1):
        span = min(trace_steps, N - n)
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        C = base.pooled_tensor(moments["C"], n, bandwidth_steps)
        B = base.pooled_tensor(moments["B"], n, bandwidth_steps)
        coeffs[n] = base.solve_reg(G, B + C @ coeffs[n + span], task.ridge)
    return coeffs


def validation_score_jzmo(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int) -> float:
    N = moments["N"]
    trace_steps = int(moments["trace_steps"])
    vals = []
    for n in range(N - 1, -1, -1):
        span = min(trace_steps, N - n)
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        C = base.pooled_tensor(moments["C"], n, bandwidth_steps)
        B = base.pooled_tensor(moments["B"], n, bandwidth_steps)
        res = G @ coeffs[n] - B - C @ coeffs[n + span]
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


# -----------------------------------------------------------------------------
# Model-based helpers
# -----------------------------------------------------------------------------

def evaluate_model_based_over_time(task: TaskSpec, model: dict, test_states: Array, seed: int, time_grid: Sequence[int]) -> list[dict]:
    bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 9090, eval_indices=time_grid)
    rows = []
    for idx in time_grid:
        mask = bundle["indices"] == idx
        t0_rmse, integrated_rmse = common.evaluate_model_based_value(
            task,
            model,
            {"states": bundle["states"][mask], "truth": bundle["truth"][mask], "indices": bundle["indices"][mask]},
            seed + 1212 + int(idx),
        )
        rows.append({"time_index": int(idx), "time": task.dt * int(idx), "rmse": float(integrated_rmse)})
    return rows


# -----------------------------------------------------------------------------
# Plotting and table utilities
# -----------------------------------------------------------------------------

def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def save_latex_table(lines: list[str], out_path: Path) -> None:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_basic_bar_plot(summary: pd.DataFrame, methods: Sequence[str], metric_col: str, ci_suffix: str, title: str, ylabel: str, out_pdf: Path) -> None:
    x = np.arange(len(summary))
    width = 0.78 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for j, method in enumerate(methods):
        y = summary[f"{method} {metric_col}"]
        yerr = summary[f"{method} {ci_suffix}"]
        ax.bar(x + (j - (len(methods) - 1) / 2.0) * width, y, width=width, yerr=yerr, capsize=4, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Scale"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_basic_plotly_dashboard(df: pd.DataFrame, out_html: Path, title: str, x_col: str, y_col: str, facet_col: str | None = None) -> None:
    fig = go.Figure()
    if facet_col is None:
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode="lines+markers", name=str(method)))
    else:
        for key in sorted(df[facet_col].unique()):
            subk = df[df[facet_col] == key]
            for method in sorted(subk["method"].unique()):
                sub = subk[subk["method"] == method]
                fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode="lines+markers", name=f"{key}-{method}"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.write_html(out_html, include_plotlyjs="cdn")


# -----------------------------------------------------------------------------
# Clean-regime task factories
# -----------------------------------------------------------------------------

def make_ou1d_task(*, n_episodes: int, dt: float, horizon_steps: int, kappa: float, mc_rollouts: int = 120) -> TaskSpec:
    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        return rng.normal(scale=0.85, size=(n, 1))

    def target_policy(x: Array, t: float) -> Array:
        gain = 0.95 + 0.20 * kappa * math.sin(0.35 * t)
        return np.clip(-gain * x[:, [0]], -3.0, 3.0)

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        return np.clip(target_policy(x, t) + 0.25 * rng.normal(size=(len(x), 1)), -3.0, 3.0)

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        drift = -(0.55 + 0.40 * kappa * math.cos(0.7 * t)) * x[:, 0] + (0.70 + 0.15 * kappa * math.sin(0.5 * t)) * u[:, 0]
        sigma = 0.16 + 0.18 * kappa * (1.0 + math.sin(0.6 * t + 0.2)) / 2.0
        x_next = x[:, 0] + dt_ * drift + sigma * math.sqrt(dt_) * rng.normal(size=len(x))
        return x_next[:, None]

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        return -(1.0 + 0.10 * kappa * math.cos(0.5 * t)) * x[:, 0] ** 2 - 0.04 * u[:, 0] ** 2

    def terminal_fn(x: Array) -> Array:
        return -(1.6 * x[:, 0] ** 2)

    return TaskSpec(
        name=f"ou1d_k{kappa:.2f}_dt{dt:.3f}",
        label="OU-1D",
        state_dim=1,
        action_dim=1,
        dt=dt,
        horizon_steps=horizon_steps,
        beta=0.05,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, max(1, horizon_steps // 3), max(1, 2 * horizon_steps // 3), horizon_steps - 2],
        n_eval_states=24,
        mc_rollouts=mc_rollouts,
        ridge=2e-4,
        feature_fn=common.linear_features,
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_tvlqr2d_task(*, n_episodes: int, dt: float, horizon_steps: int, kappa: float, mc_rollouts: int = 120) -> TaskSpec:
    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        return rng.normal(scale=np.array([0.8, 0.55]), size=(n, 2))

    def target_policy(x: Array, t: float) -> Array:
        k1 = 1.10 + 0.15 * kappa * math.sin(0.40 * t)
        k2 = 0.85 + 0.12 * kappa * math.cos(0.30 * t)
        u = -(k1 * x[:, 0] + k2 * x[:, 1])
        return np.clip(u[:, None], -4.0, 4.0)

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        return np.clip(target_policy(x, t) + 0.28 * rng.normal(size=(len(x), 1)), -4.0, 4.0)

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        A = np.array(
            [
                [0.0, 1.0],
                [-(0.95 + 0.22 * kappa * math.sin(0.55 * t)), -(0.42 + 0.16 * kappa * math.cos(0.45 * t))],
            ]
        )
        B = np.array([[0.0], [1.0 + 0.18 * kappa * math.sin(0.35 * t + 0.2)]])
        drift = x @ A.T + u @ B.T
        noise = np.zeros_like(x)
        noise[:, 1] = (0.08 + 0.10 * kappa * (1.0 + math.cos(0.45 * t)) / 2.0) * math.sqrt(dt_) * rng.normal(size=len(x))
        return x + dt_ * drift + noise

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        q = np.array([1.0 + 0.12 * kappa * math.sin(0.4 * t), 0.30 + 0.05 * kappa * math.cos(0.35 * t)])
        return -(np.sum(q[None, :] * x ** 2, axis=1) + 0.06 * u[:, 0] ** 2)

    def terminal_fn(x: Array) -> Array:
        return -(1.4 * x[:, 0] ** 2 + 0.55 * x[:, 1] ** 2)

    return TaskSpec(
        name=f"tvlqr2d_k{kappa:.2f}_dt{dt:.3f}",
        label="TV-LQR-2D",
        state_dim=2,
        action_dim=1,
        dt=dt,
        horizon_steps=horizon_steps,
        beta=0.05,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, max(1, horizon_steps // 3), max(1, 2 * horizon_steps // 3), horizon_steps - 2],
        n_eval_states=20,
        mc_rollouts=mc_rollouts,
        ridge=3e-4,
        feature_fn=lambda x: base.quadratic_features(x, include_linear=True),
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )
