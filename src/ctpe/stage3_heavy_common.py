from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ctpe import run_stage2_extended_suite as base

Array = np.ndarray
TaskSpec = base.TaskSpec


# -----------------------------------------------------------------------------
# Feature helpers
# -----------------------------------------------------------------------------

def linear_features(x: Array) -> Array:
    x = np.asarray(x)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    out = np.concatenate([np.ones((flat.shape[0], 1)), flat], axis=1)
    return out.reshape(*orig_shape, out.shape[-1])


def reduced_pendulum_features(x: Array) -> Array:
    x = np.asarray(x)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    theta = flat[:, 0]
    omega = flat[:, 1]
    feats = np.stack(
        [
            np.ones_like(theta),
            np.cos(theta),
            np.sin(theta),
            omega,
        ],
        axis=1,
    )
    return feats.reshape(*orig_shape, feats.shape[-1])


def action_state_linear_features(x: Array, a: Array) -> Array:
    x = np.asarray(x)
    a = np.asarray(a)
    x_flat = x.reshape(-1, x.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    out = np.concatenate([np.ones((x_flat.shape[0], 1)), x_flat, a_flat], axis=1)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def action_state_quadratic_features(x: Array, a: Array, max_interactions: int = 12) -> Array:
    x = np.asarray(x)
    a = np.asarray(a)
    orig_shape = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    af = a.reshape(-1, a.shape[-1])
    cols = [np.ones((xf.shape[0], 1)), xf, af]

    # Limit the number of quadratic terms to keep the baseline affordable.
    qcols = []
    d = xf.shape[1]
    count = 0
    for i in range(d):
        for j in range(i, d):
            qcols.append((xf[:, i] * xf[:, j])[:, None])
            count += 1
            if count >= max_interactions:
                break
        if count >= max_interactions:
            break
    if qcols:
        cols.append(np.concatenate(qcols, axis=1))
    ax_cols = []
    for i in range(min(xf.shape[1], max_interactions // max(1, af.shape[1]))):
        for j in range(af.shape[1]):
            ax_cols.append((xf[:, i] * af[:, j])[:, None])
    if ax_cols:
        cols.append(np.concatenate(ax_cols, axis=1))
    cols.append(af ** 2)
    out = np.concatenate(cols, axis=1)
    return out.reshape(*orig_shape, out.shape[-1])


# -----------------------------------------------------------------------------
# Task factories
# -----------------------------------------------------------------------------

def make_small_heavy_task(
    *,
    n_episodes: int = 480,
    nonstat_scale: float = 1.0,
    dt: float = 0.06,
    mc_rollouts: int = 96,
    horizon_steps: int = 56,
    feature_family: str = "rich",
) -> TaskSpec:
    task = base.make_small_task(n_episodes=n_episodes, nonstat_scale=nonstat_scale, dt=dt, mc_rollouts=mc_rollouts)
    if feature_family == "rich":
        feat = base.pendulum_features
    elif feature_family == "quadratic":
        feat = lambda x: base.quadratic_features(x, include_linear=True)
    elif feature_family == "reduced":
        feat = reduced_pendulum_features
    elif feature_family == "linear":
        feat = linear_features
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")
    eval_idx = sorted({0, max(1, horizon_steps // 5), max(1, 2 * horizon_steps // 5), max(1, 3 * horizon_steps // 5), max(1, 4 * horizon_steps // 5), horizon_steps - 2})
    return replace(
        task,
        name=f"small_pendulum_{feature_family}",
        label="Small",
        n_episodes=n_episodes,
        dt=dt,
        horizon_steps=horizon_steps,
        mc_rollouts=mc_rollouts,
        bandwidth_grid=[0, 1, 2, 4, 6, 8],
        eval_indices=eval_idx,
        n_eval_states=20,
        ridge=2e-4,
        feature_fn=feat,
    )


def make_medium_heavy_task(
    *,
    n_episodes: int = 560,
    nonstat_scale: float = 1.0,
    dt: float = 0.06,
    mc_rollouts: int = 96,
    horizon_steps: int = 56,
    feature_family: str = "quadratic",
) -> TaskSpec:
    task = base.make_medium_task(n_episodes=n_episodes, nonstat_scale=nonstat_scale, dt=dt, mc_rollouts=mc_rollouts)
    if feature_family == "quadratic":
        feat = lambda x: base.quadratic_features(x, include_linear=True)
    elif feature_family == "linear":
        feat = linear_features
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")
    eval_idx = sorted({0, max(1, horizon_steps // 5), max(1, 2 * horizon_steps // 5), max(1, 3 * horizon_steps // 5), max(1, 4 * horizon_steps // 5), horizon_steps - 2})
    return replace(
        task,
        name=f"medium_cartpole_like_{feature_family}",
        label="Medium",
        n_episodes=n_episodes,
        dt=dt,
        horizon_steps=horizon_steps,
        mc_rollouts=mc_rollouts,
        bandwidth_grid=[0, 1, 2, 4, 6, 8],
        eval_indices=eval_idx,
        n_eval_states=18,
        ridge=3e-4,
        feature_fn=feat,
    )


def make_networked_lq_task(
    *,
    state_dim: int,
    action_dim: int | None = None,
    label: str = "Large",
    n_episodes: int = 720,
    nonstat_scale: float = 1.0,
    dt: float = 0.06,
    mc_rollouts: int = 72,
    horizon_steps: int = 48,
    feature_family: str = "quadratic",
) -> TaskSpec:
    if state_dim % 2 != 0:
        raise ValueError("state_dim must be even.")
    n_blocks = state_dim // 2
    if action_dim is None:
        action_dim = max(1, n_blocks // 2)

    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        scales = np.linspace(0.55, 0.85, state_dim)
        return rng.normal(scale=scales, size=(n, state_dim))

    # Build a sparse control map from states to actions.
    K = np.zeros((action_dim, state_dim))
    for a_idx in range(action_dim):
        start = (2 * a_idx) % state_dim
        idxs = [(start + j) % state_dim for j in range(4)]
        vals = np.array([1.15, 0.65, 0.85, 0.45]) / (1.0 + 0.05 * a_idx)
        K[a_idx, idxs] = vals

    def target_policy(x: Array, t: float) -> Array:
        u = -(x @ K.T)
        clip = 3.2 + 0.04 * action_dim
        return np.clip(u, -clip, clip)

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        u = target_policy(x, t)
        noise_scale = 0.30 + 0.02 * min(action_dim, 6)
        return np.clip(u + noise_scale * rng.normal(size=u.shape), -4.0, 4.0)

    def matrices(t: float) -> tuple[Array, Array, Array]:
        A = np.zeros((state_dim, state_dim))
        for k in range(n_blocks):
            i = 2 * k
            omega = 0.90 + nonstat_scale * 0.18 * math.sin(0.22 * t + 0.13 * k)
            damp = 0.36 + nonstat_scale * 0.07 * math.cos(0.16 * t + 0.07 * k)
            A[i, i + 1] = 1.0
            A[i + 1, i] = -(omega ** 2)
            A[i + 1, i + 1] = -damp
            if k < n_blocks - 1:
                A[i + 1, i + 2] = nonstat_scale * 0.08 * math.sin(0.37 * t + 0.11 * k)
            if k > 0:
                A[i + 1, i - 2] = nonstat_scale * 0.05 * math.cos(0.29 * t + 0.09 * k)
        B = np.zeros((state_dim, action_dim))
        for j in range(action_dim):
            base_idx = min(state_dim - 1, 2 * j)
            vel_idx = min(state_dim - 1, base_idx + 1)
            B[vel_idx, j] = 1.0 + nonstat_scale * 0.08 * math.cos(0.33 * t + j)
            if vel_idx + 2 < state_dim:
                B[vel_idx + 2, j] = 0.55 + nonstat_scale * 0.05 * math.sin(0.41 * t + j)
        q_diag = np.linspace(1.0, 1.45, state_dim)
        q_diag *= 1.0 + nonstat_scale * 0.09 * math.sin(0.20 * t)
        return A, B, q_diag

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        A, B, _ = matrices(t)
        drift = x @ A.T + u @ B.T
        noise = np.zeros_like(x)
        vel_idx = list(range(1, state_dim, 2))
        noise[:, vel_idx] = 0.05 * math.sqrt(dt_) * rng.normal(size=(len(x), len(vel_idx)))
        return x + dt_ * drift + noise

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        _, _, q_diag = matrices(t)
        state_cost = np.sum(q_diag[None, :] * x ** 2, axis=1)
        ctrl_weights = np.linspace(0.04, 0.07, action_dim)
        control_cost = np.sum(ctrl_weights[None, :] * u ** 2, axis=1)
        return -(state_cost + control_cost)

    def terminal_fn(x: Array) -> Array:
        qf = np.linspace(1.25, 1.75, state_dim)
        return -np.sum(qf[None, :] * x ** 2, axis=1)

    if feature_family == "quadratic":
        feat = lambda x: base.quadratic_features(x, include_linear=True)
        ridge = 6e-4 if state_dim <= 16 else 1.0e-3
    elif feature_family == "linear":
        feat = linear_features
        ridge = 5e-4 if state_dim <= 16 else 8.0e-4
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")

    eval_idx = sorted({0, max(1, horizon_steps // 4), max(1, horizon_steps // 2), max(1, 3 * horizon_steps // 4), horizon_steps - 2})
    return TaskSpec(
        name=f"networked_lq_{state_dim}d_{feature_family}",
        label=label,
        state_dim=state_dim,
        action_dim=action_dim,
        dt=dt,
        horizon_steps=horizon_steps,
        beta=0.04,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4, 6, 8],
        eval_indices=eval_idx,
        n_eval_states=14 if state_dim <= 24 else 10,
        mc_rollouts=mc_rollouts,
        ridge=ridge,
        feature_fn=feat,
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_large_heavy_task(**kwargs: Any) -> TaskSpec:
    defaults = dict(state_dim=12, action_dim=3, label="Large", n_episodes=720, mc_rollouts=72, horizon_steps=48)
    defaults.update(kwargs)
    return make_networked_lq_task(**defaults)


def make_xlarge_heavy_task(**kwargs: Any) -> TaskSpec:
    defaults = dict(state_dim=24, action_dim=6, label="XLarge", n_episodes=880, mc_rollouts=56, horizon_steps=48)
    defaults.update(kwargs)
    return make_networked_lq_task(**defaults)


def heavy_main_task_specs() -> list[tuple[TaskSpec, list[int]]]:
    return [
        (make_small_heavy_task(), [20260701 + i for i in range(16)]),
        (make_medium_heavy_task(), [20260801 + i for i in range(16)]),
        (make_large_heavy_task(), [20260901 + i for i in range(10)]),
        (make_xlarge_heavy_task(), [20261001 + i for i in range(8)]),
    ]


# -----------------------------------------------------------------------------
# Data generation with actions
# -----------------------------------------------------------------------------

def rollout_with_actions(task: TaskSpec, n_episodes: int, rng: np.random.Generator, policy_kind: str = "behavior", gain_bias: float = 0.0) -> tuple[Array, Array, Array]:
    states = np.zeros((n_episodes, task.horizon_steps + 1, task.state_dim))
    actions = np.zeros((n_episodes, task.horizon_steps, task.action_dim))
    rewards = np.zeros((n_episodes, task.horizon_steps))
    states[:, 0, :] = task.sample_initial_states(n_episodes, rng)
    for n in range(task.horizon_steps):
        t = n * task.dt
        x = states[:, n, :]
        if policy_kind == "target":
            u = task.target_policy(x, t)
        else:
            u = task.behavior_policy(x, t, rng)
        if gain_bias != 0.0:
            u = np.clip((1.0 + gain_bias) * u, -6.0, 6.0)
        actions[:, n, :] = u
        rewards[:, n] = task.reward_fn(x, t, u)
        states[:, n + 1, :] = task.step_fn(x, t, u, task.dt, rng)
    return states, actions, rewards


def split_with_actions(states: Array, actions: Array, rewards: Array) -> dict[str, tuple[Array, Array, Array]]:
    M = states.shape[0]
    n_train = int(0.6 * M)
    n_val = int(0.2 * M)
    return {
        "train": (states[:n_train], actions[:n_train], rewards[:n_train]),
        "val": (states[n_train : n_train + n_val], actions[n_train : n_train + n_val], rewards[n_train : n_train + n_val]),
        "test": (states[n_train + n_val :], actions[n_train + n_val :], rewards[n_train + n_val :]),
    }


# -----------------------------------------------------------------------------
# Generator start-up variants and diagnostics
# -----------------------------------------------------------------------------

def fit_generator_with_startup(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int, order: int, startup_mode: str = "be") -> Array:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    if startup_mode == "be":
        be_coeffs = base.fit_be_from_moments(task, moments, bandwidth_steps)
        coeffs[N - order + 1 :] = be_coeffs[N - order + 1 :]
    elif startup_mode == "zero":
        coeffs[N - order + 1 :] = 0.0
    else:
        raise ValueError(f"Unknown startup_mode: {startup_mode}")
    a = base.a_coeffs(order)
    G_short = moments["G"][: N - order + 1]
    A = moments["A_orders"][order]
    b = moments["b_gen_orders"][order]
    for n in range(N - order, -1, -1):
        G = base.pooled_tensor(G_short, n, bandwidth_steps)
        A_n = base.pooled_tensor(A, n, bandwidth_steps)
        b_n = base.pooled_tensor(b, n, bandwidth_steps)
        Mmat = (task.beta - a[0] / task.dt) * G - A_n
        future = np.zeros(p)
        for j in range(1, order + 1):
            future += a[j] * coeffs[n + j]
        rhs = b_n + (1.0 / task.dt) * G @ future
        coeffs[n] = base.solve_reg(Mmat, rhs, task.ridge)
    return coeffs


def gram_diagnostics(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    N = int(moments["N"])
    for n in range(N):
        G = base.pooled_tensor(moments["G"], n, bandwidth_steps)
        evals = np.linalg.eigvalsh(0.5 * (G + G.T))
        rows.append(
            {
                "time_index": n,
                "time": n * task.dt,
                "bandwidth_steps": int(bandwidth_steps),
                "min_eig": float(evals.min()),
                "max_eig": float(evals.max()),
                "cond": float(evals.max() / max(evals.min(), 1e-10)),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Model-based baseline
# -----------------------------------------------------------------------------

def _ridge_fit(Z: Array, Y: Array, lam: float) -> Array:
    p = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ Y)


def _pooled_regression_data(states: Array, actions: Array, rewards: Array, center: int, bandwidth_steps: int, feature_kind: str) -> tuple[Array, Array, Array]:
    N = rewards.shape[1]
    idx = np.arange(max(0, center - bandwidth_steps), min(N - 1, center + bandwidth_steps) + 1)
    w = base.triangular_kernel_weights(center, idx, bandwidth_steps)
    Z_parts, Yx_parts, Yr_parts = [], [], []
    for weight, n in zip(w, idx):
        x = states[:, n, :]
        a = actions[:, n, :]
        if feature_kind == "linear":
            Z = action_state_linear_features(x, a)
        elif feature_kind == "quadratic":
            Z = action_state_quadratic_features(x, a)
        else:
            raise ValueError(f"Unknown feature_kind: {feature_kind}")
        sw = math.sqrt(float(weight))
        Z_parts.append(sw * Z)
        Yx_parts.append(sw * states[:, n + 1, :])
        Yr_parts.append(sw * rewards[:, n, None])
    return np.concatenate(Z_parts, axis=0), np.concatenate(Yx_parts, axis=0), np.concatenate(Yr_parts, axis=0)


def fit_model_based_baseline(task: TaskSpec, states: Array, actions: Array, rewards: Array, bandwidth_steps: int, feature_kind: str = "linear") -> dict[str, list[Array]]:
    N = rewards.shape[1]
    dyn_weights: list[Array] = []
    rew_weights: list[Array] = []
    noise_scales: list[Array] = []
    feature_dim = None
    lam = task.ridge
    for n in range(N):
        Z, Yx, Yr = _pooled_regression_data(states, actions, rewards, n, bandwidth_steps, feature_kind)
        Wx = _ridge_fit(Z, Yx, lam)
        Wr = _ridge_fit(Z, Yr, lam).reshape(-1)
        resid = Yx - Z @ Wx
        noise = resid.std(axis=0, ddof=1)
        noise = np.maximum(noise, 1e-3)
        dyn_weights.append(Wx)
        rew_weights.append(Wr)
        noise_scales.append(noise)
        feature_dim = Z.shape[1]
    return {
        "dyn_weights": dyn_weights,
        "rew_weights": rew_weights,
        "noise_scales": noise_scales,
        "feature_kind": feature_kind,
        "feature_dim": int(feature_dim or 0),
    }


def model_validation_score(task: TaskSpec, model: dict[str, list[Array]], states: Array, actions: Array, rewards: Array) -> float:
    errs: list[float] = []
    N = rewards.shape[1]
    for n in range(N):
        x = states[:, n, :]
        a = actions[:, n, :]
        if model["feature_kind"] == "linear":
            Z = action_state_linear_features(x, a)
        else:
            Z = action_state_quadratic_features(x, a)
        pred_x = Z @ model["dyn_weights"][n]
        pred_r = Z @ model["rew_weights"][n]
        err_x = np.mean((pred_x - states[:, n + 1, :]) ** 2)
        err_r = np.mean((pred_r - rewards[:, n]) ** 2)
        errs.append(float(err_x + err_r))
    return float(np.mean(errs))


def evaluate_model_based_value(task: TaskSpec, model: dict[str, list[Array]], eval_bundle: dict[str, Array], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    preds = np.zeros_like(eval_bundle["truth"])
    disc0 = math.exp(-task.beta * task.dt)
    unique_idx = np.unique(eval_bundle["indices"])
    for idx in unique_idx:
        mask = eval_bundle["indices"] == idx
        start_states = eval_bundle["states"][mask]
        K = start_states.shape[0]
        values = np.zeros(K)
        for _ in range(task.mc_rollouts):
            x = start_states.copy()
            disc = np.ones(K)
            ret = np.zeros(K)
            for n in range(int(idx), task.horizon_steps):
                t = n * task.dt
                a = task.target_policy(x, t)
                if model["feature_kind"] == "linear":
                    Z = action_state_linear_features(x, a)
                else:
                    Z = action_state_quadratic_features(x, a)
                mean_next = Z @ model["dyn_weights"][n]
                reward_pred = Z @ model["rew_weights"][n]
                noise = rng.normal(size=mean_next.shape) * model["noise_scales"][n][None, :]
                x = mean_next + noise
                ret += disc * reward_pred * task.dt
                disc *= disc0
            ret += disc * task.terminal_fn(x)
            values += ret
        preds[mask] = values / task.mc_rollouts
    err = preds - eval_bundle["truth"]
    integrated_rmse = math.sqrt(float(np.mean(err ** 2)))
    mask0 = eval_bundle["indices"] == 0
    if np.any(mask0):
        t0_rmse = math.sqrt(float(np.mean(err[mask0] ** 2)))
    else:
        t0_rmse = float("nan")
    return t0_rmse, integrated_rmse


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dirs(root: Path) -> dict[str, Path]:
    figures = root / "figures"
    results = root / "results"
    tables = root / "tables"
    interactive = figures / "interactive"
    external = root / "external_results"
    for p in [figures, results, tables, interactive, external]:
        p.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, "results": results, "tables": tables, "interactive": interactive, "external": external}


def safe_prepare_eval_bundle(task: TaskSpec, test_states: Array, seed: int, eval_indices: Sequence[int] | None = None) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    if eval_indices is None:
        eval_indices = task.eval_indices
    n_take = min(task.n_eval_states, test_states.shape[0])
    state_list = []
    truth_list = []
    index_list = []
    for idx in eval_indices:
        episode_idx = rng.choice(test_states.shape[0], size=n_take, replace=False)
        s = test_states[episode_idx, idx, :]
        gt = base.estimate_mc_value(task, s, idx, rng)
        state_list.append(s)
        truth_list.append(gt)
        index_list.append(np.full(n_take, idx))
    return {
        "states": np.concatenate(state_list, axis=0),
        "truth": np.concatenate(truth_list, axis=0),
        "indices": np.concatenate(index_list, axis=0),
    }
