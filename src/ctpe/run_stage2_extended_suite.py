"""Expanded stage-2 benchmark suite for the offline continuous-time RL revision.

This runner extends the lightweight pilot by adding:
  * repeated-seed summary statistics,
  * mean validation-based bandwidth curves,
  * over-time RMSE profiles,
  * order ablation on the small task,
  * a nonstationarity-vs-data heat map,
  * and an accuracy-runtime comparison.

Run from the project root:
    python scripts/run_stage2_extended_suite.py
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


def a_coeffs(order: int) -> Array:
    j = np.arange(order + 1)
    A = np.vander(j, N=order + 1, increasing=True).T
    b = np.zeros(order + 1)
    b[1] = 1.0
    return np.linalg.solve(A, b)


def wrap_angle(x: Array) -> Array:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def triangular_kernel_weights(center: int, indices: Array, bandwidth_steps: int) -> Array:
    if bandwidth_steps <= 0:
        return np.ones(len(indices), dtype=float) / max(1, len(indices))
    w = 1.0 - np.abs(indices - center) / (bandwidth_steps + 1.0)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.ones(len(indices), dtype=float) / max(1, len(indices))
    return w / s


def quadratic_features(x: Array, include_linear: bool = True) -> Array:
    x = np.asarray(x)
    orig_shape = x.shape[:-1]
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    cols = [np.ones((flat.shape[0], 1))]
    if include_linear:
        cols.append(flat)
    quad_terms = []
    for i in range(d):
        for j in range(i, d):
            quad_terms.append((flat[:, i] * flat[:, j])[:, None])
    cols.append(np.concatenate(quad_terms, axis=1))
    out = np.concatenate(cols, axis=1)
    return out.reshape(*orig_shape, out.shape[-1])


def pendulum_features(x: Array) -> Array:
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
            omega ** 2,
            np.sin(theta) * omega,
            np.cos(theta) * omega,
            theta ** 2,
        ],
        axis=1,
    )
    return feats.reshape(*orig_shape, feats.shape[-1])


@dataclass
class TaskSpec:
    name: str
    label: str
    state_dim: int
    action_dim: int
    dt: float
    horizon_steps: int
    beta: float
    n_episodes: int
    bandwidth_grid: List[int]
    eval_indices: List[int]
    n_eval_states: int
    mc_rollouts: int
    ridge: float
    feature_fn: Callable[[Array], Array]
    sample_initial_states: Callable[[int, np.random.Generator], Array]
    target_policy: Callable[[Array, float], Array]
    behavior_policy: Callable[[Array, float, np.random.Generator], Array]
    step_fn: Callable[[Array, float, Array, float, np.random.Generator], Array]
    reward_fn: Callable[[Array, float, Array], Array]
    terminal_fn: Callable[[Array], Array]


# -----------------------------------------------------------------------------
# Task factories
# -----------------------------------------------------------------------------

def make_small_task(n_episodes: int = 180, nonstat_scale: float = 1.0, dt: float = 0.08, mc_rollouts: int = 48) -> TaskSpec:
    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        theta = rng.uniform(-1.4, 1.4, size=n)
        omega = rng.normal(scale=0.8, size=n)
        return np.stack([theta, omega], axis=1)

    def target_policy(x: Array, t: float) -> Array:
        theta = wrap_angle(x[:, 0])
        omega = x[:, 1]
        u = -2.4 * np.sin(theta) - 0.9 * omega
        u = np.clip(u, -2.5, 2.5)
        return u[:, None]

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        u = target_policy(x, t)
        return np.clip(u + 0.45 * rng.normal(size=u.shape), -2.5, 2.5)

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        theta = wrap_angle(x[:, 0])
        omega = x[:, 1]
        gravity = 9.81 * (1.0 + nonstat_scale * 0.12 * math.sin(1.4 * t))
        damping = 0.14 + nonstat_scale * 0.05 * (1.0 + math.sin(0.7 * t + 0.3)) / 2.0
        gain = 1.0 + nonstat_scale * 0.08 * math.cos(0.5 * t)
        noise = 0.12 * math.sqrt(dt_) * rng.normal(size=len(x))
        omega_next = omega + dt_ * (-gravity * np.sin(theta) - damping * omega + gain * u[:, 0]) + noise
        theta_next = wrap_angle(theta + dt_ * omega_next)
        return np.stack([theta_next, omega_next], axis=1)

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        theta = wrap_angle(x[:, 0])
        omega = x[:, 1]
        return -(theta ** 2 + 0.15 * omega ** 2 + 0.02 * u[:, 0] ** 2)

    def terminal_fn(x: Array) -> Array:
        theta = wrap_angle(x[:, 0])
        omega = x[:, 1]
        return -(2.0 * theta ** 2 + 0.3 * omega ** 2)

    return TaskSpec(
        name="small_pendulum",
        label="Small",
        state_dim=2,
        action_dim=1,
        dt=dt,
        horizon_steps=40,
        beta=0.04,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 16, 30],
        n_eval_states=12,
        mc_rollouts=mc_rollouts,
        ridge=2e-4,
        feature_fn=pendulum_features,
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_medium_task(n_episodes: int = 200, nonstat_scale: float = 1.0, dt: float = 0.08, mc_rollouts: int = 48) -> TaskSpec:
    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        return rng.normal(scale=np.array([0.7, 0.7, 0.35, 0.6]), size=(n, 4))

    def target_policy(x: Array, t: float) -> Array:
        k = np.array([1.3, 0.9, 2.4, 1.1])
        u = -(x @ k)
        u = np.clip(u, -4.0, 4.0)
        return u[:, None]

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        u = target_policy(x, t)
        return np.clip(u + 0.6 * rng.normal(size=u.shape), -4.0, 4.0)

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        pos = x[:, 0]
        vel = x[:, 1]
        ang = x[:, 2]
        ang_vel = x[:, 3]
        a1 = 1.1 + nonstat_scale * 0.25 * math.sin(0.6 * t)
        a2 = 0.35 + nonstat_scale * 0.08 * math.cos(0.4 * t)
        a3 = 0.75 + nonstat_scale * 0.12 * math.sin(0.9 * t + 0.2)
        b1 = 0.55 + nonstat_scale * 0.15 * math.cos(0.5 * t)
        b2 = 1.55 + nonstat_scale * 0.18 * math.sin(0.7 * t)
        b3 = 0.42 + nonstat_scale * 0.08 * math.sin(0.3 * t)
        gain = 1.0 + nonstat_scale * 0.12 * math.cos(0.45 * t)
        noise = 0.08 * math.sqrt(dt_) * rng.normal(size=(len(x), 2))
        vel_next = vel + dt_ * (-a1 * pos - a2 * vel + a3 * ang + 0.45 * gain * u[:, 0]) + noise[:, 0]
        ang_vel_next = ang_vel + dt_ * (b1 * pos - b2 * ang - b3 * ang_vel + 1.0 * gain * u[:, 0] - 0.12 * np.sin(ang)) + noise[:, 1]
        pos_next = pos + dt_ * vel_next
        ang_next = ang + dt_ * ang_vel_next
        return np.stack([pos_next, vel_next, ang_next, ang_vel_next], axis=1)

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        q = np.array([1.0, 0.25, 2.2, 0.35])
        return -(np.sum(q * x ** 2, axis=1) + 0.03 * u[:, 0] ** 2)

    def terminal_fn(x: Array) -> Array:
        qf = np.array([1.5, 0.4, 3.0, 0.6])
        return -np.sum(qf * x ** 2, axis=1)

    return TaskSpec(
        name="medium_cartpole_like",
        label="Medium",
        state_dim=4,
        action_dim=1,
        dt=dt,
        horizon_steps=40,
        beta=0.04,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 16, 30],
        n_eval_states=12,
        mc_rollouts=mc_rollouts,
        ridge=3e-4,
        feature_fn=lambda x: quadratic_features(x, include_linear=True),
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_large_task(n_episodes: int = 240, nonstat_scale: float = 1.0, dt: float = 0.08, mc_rollouts: int = 40) -> TaskSpec:
    def sample_initial_states(n: int, rng: np.random.Generator) -> Array:
        return rng.normal(scale=0.7, size=(n, 12))

    def target_policy(x: Array, t: float) -> Array:
        u0 = -(1.15 * x[:, 0] + 0.65 * x[:, 1] + 0.8 * x[:, 2] + 0.45 * x[:, 3])
        u1 = -(1.10 * x[:, 4] + 0.60 * x[:, 5] + 0.85 * x[:, 6] + 0.45 * x[:, 7])
        u2 = -(1.05 * x[:, 8] + 0.60 * x[:, 9] + 0.80 * x[:, 10] + 0.40 * x[:, 11])
        U = np.stack([u0, u1, u2], axis=1)
        return np.clip(U, -3.5, 3.5)

    def behavior_policy(x: Array, t: float, rng: np.random.Generator) -> Array:
        u = target_policy(x, t)
        return np.clip(u + 0.35 * rng.normal(size=u.shape), -3.5, 3.5)

    def matrices(t: float) -> tuple[Array, Array, Array]:
        A = np.zeros((12, 12))
        for k in range(6):
            i = 2 * k
            omega = 0.95 + nonstat_scale * 0.16 * math.sin(0.25 * t + 0.25 * k)
            damp = 0.42 + nonstat_scale * 0.05 * math.cos(0.18 * t + 0.15 * k)
            A[i, i + 1] = 1.0
            A[i + 1, i] = -(omega ** 2)
            A[i + 1, i + 1] = -damp
            if k < 5:
                A[i + 1, i + 2] = nonstat_scale * 0.06 * math.sin(0.4 * t + 0.1 * k)
        B = np.zeros((12, 3))
        for j in range(3):
            B[4 * j + 1, j] = 1.0 + nonstat_scale * 0.08 * math.cos(0.35 * t + j)
            B[4 * j + 3, j] = 0.75 + nonstat_scale * 0.06 * math.sin(0.45 * t + j)
        q_diag = np.array([1.2, 0.3, 1.0, 0.25, 1.3, 0.3, 1.1, 0.25, 1.15, 0.28, 1.05, 0.25], dtype=float)
        q_diag *= 1.0 + nonstat_scale * 0.08 * math.sin(0.22 * t)
        return A, B, q_diag

    def step_fn(x: Array, t: float, u: Array, dt_: float, rng: np.random.Generator) -> Array:
        A, B, _ = matrices(t)
        drift = x @ A.T + u @ B.T
        noise = np.zeros_like(x)
        vel_idx = [1, 3, 5, 7, 9, 11]
        noise[:, vel_idx] = 0.06 * math.sqrt(dt_) * rng.normal(size=(len(x), len(vel_idx)))
        return x + dt_ * drift + noise

    def reward_fn(x: Array, t: float, u: Array) -> Array:
        _, _, q_diag = matrices(t)
        state_cost = np.sum(q_diag[None, :] * x ** 2, axis=1)
        control_cost = np.sum(np.array([0.06, 0.05, 0.05])[None, :] * u ** 2, axis=1)
        return -(state_cost + control_cost)

    def terminal_fn(x: Array) -> Array:
        qf = np.array([1.6, 0.35, 1.3, 0.30, 1.7, 0.35, 1.4, 0.30, 1.5, 0.32, 1.35, 0.30])
        return -np.sum(qf[None, :] * x ** 2, axis=1)

    return TaskSpec(
        name="large_lq",
        label="Large",
        state_dim=12,
        action_dim=3,
        dt=dt,
        horizon_steps=32,
        beta=0.04,
        n_episodes=n_episodes,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 12, 24],
        n_eval_states=10,
        mc_rollouts=mc_rollouts,
        ridge=6e-4,
        feature_fn=lambda x: quadratic_features(x, include_linear=True),
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


# -----------------------------------------------------------------------------
# Data generation and linear-algebra utilities
# -----------------------------------------------------------------------------

def rollout_episodes(task: TaskSpec, n_episodes: int, rng: np.random.Generator, policy_kind: str) -> tuple[Array, Array]:
    states = np.zeros((n_episodes, task.horizon_steps + 1, task.state_dim))
    rewards = np.zeros((n_episodes, task.horizon_steps))
    states[:, 0, :] = task.sample_initial_states(n_episodes, rng)
    for n in range(task.horizon_steps):
        t = n * task.dt
        x = states[:, n, :]
        if policy_kind == "target":
            u = task.target_policy(x, t)
        else:
            u = task.behavior_policy(x, t, rng)
        rewards[:, n] = task.reward_fn(x, t, u)
        states[:, n + 1, :] = task.step_fn(x, t, u, task.dt, rng)
    return states, rewards


def split_episodes(states: Array, rewards: Array) -> dict[str, tuple[Array, Array]]:
    M = states.shape[0]
    n_train = int(0.6 * M)
    n_val = int(0.2 * M)
    return {
        "train": (states[:n_train], rewards[:n_train]),
        "val": (states[n_train : n_train + n_val], rewards[n_train : n_train + n_val]),
        "test": (states[n_train + n_val :], rewards[n_train + n_val :]),
    }


def solve_reg(mat: Array, rhs: Array, ridge: float) -> Array:
    p = mat.shape[0]
    lam = ridge * (1.0 + np.trace(mat) / max(1, p))
    return np.linalg.solve(mat + lam * np.eye(p), rhs)


def pooled_tensor(arr: Array, center: int, bandwidth_steps: int) -> Array:
    max_index = arr.shape[0] - 1
    idx = np.arange(max(0, center - bandwidth_steps), min(max_index, center + bandwidth_steps) + 1)
    w = triangular_kernel_weights(center, idx, bandwidth_steps)
    return np.tensordot(w, arr[idx], axes=(0, 0))


def precompute_moments(task: TaskSpec, states: Array, rewards: Array, max_order: int = 3) -> dict[str, Array]:
    Phi = task.feature_fn(states)
    M, Np1, p = Phi.shape
    N = Np1 - 1
    G = np.zeros((N, p, p))
    P = np.zeros((N, p, p))
    b_be = np.zeros((N, p))
    A_orders: Dict[int, Array] = {}
    b_gen_orders: Dict[int, Array] = {}

    for n in range(N):
        phi_n = Phi[:, n, :]
        G[n] = (phi_n.T @ phi_n) / M
        b_be[n] = (phi_n.T @ rewards[:, n]) / M * task.dt
        P[n] = math.exp(-task.beta * task.dt) * (phi_n.T @ Phi[:, n + 1, :]) / M

    for order in range(1, max_order + 1):
        a = a_coeffs(order)
        A = np.zeros((N - order + 1, p, p))
        b = np.zeros((N - order + 1, p))
        for n in range(N - order + 1):
            phi_n = Phi[:, n, :]
            delta_phi = np.zeros_like(phi_n)
            for j in range(order + 1):
                delta_phi += a[j] * Phi[:, n + j, :]
            A[n] = (phi_n.T @ (delta_phi / task.dt)) / M
            b[n] = (phi_n.T @ rewards[:, n]) / M
        A_orders[order] = A
        b_gen_orders[order] = b

    term_G = (Phi[:, N, :].T @ Phi[:, N, :]) / M
    term_y = (Phi[:, N, :].T @ task.terminal_fn(states[:, N, :])) / M
    return {
        "Phi": Phi,
        "G": G,
        "P": P,
        "b_be": b_be,
        "A_orders": A_orders,
        "b_gen_orders": b_gen_orders,
        "term_G": term_G,
        "term_y": term_y,
        "N": N,
        "p": p,
    }


def fit_be_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> Array:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    coeffs[N] = solve_reg(moments["term_G"], moments["term_y"], task.ridge)
    for n in range(N - 1, -1, -1):
        G = pooled_tensor(moments["G"], n, bandwidth_steps)
        P = pooled_tensor(moments["P"], n, bandwidth_steps)
        b = pooled_tensor(moments["b_be"], n, bandwidth_steps)
        coeffs[n] = solve_reg(G, b + P @ coeffs[n + 1], task.ridge)
    return coeffs


def fit_generator_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int, order: int) -> Array:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    be_coeffs = fit_be_from_moments(task, moments, bandwidth_steps)
    coeffs[N - order + 1 :] = be_coeffs[N - order + 1 :]
    a = a_coeffs(order)
    G_short = moments["G"][: N - order + 1]
    A = moments["A_orders"][order]
    b = moments["b_gen_orders"][order]
    for n in range(N - order, -1, -1):
        G = pooled_tensor(G_short, n, bandwidth_steps)
        A_n = pooled_tensor(A, n, bandwidth_steps)
        b_n = pooled_tensor(b, n, bandwidth_steps)
        Mmat = (task.beta - a[0] / task.dt) * G - A_n
        future = np.zeros(p)
        for j in range(1, order + 1):
            future += a[j] * coeffs[n + j]
        rhs = b_n + (1.0 / task.dt) * G @ future
        coeffs[n] = solve_reg(Mmat, rhs, task.ridge)
    return coeffs


def validation_score_be(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int) -> float:
    N = moments["N"]
    vals = []
    for n in range(N - 1, -1, -1):
        G = pooled_tensor(moments["G"], n, bandwidth_steps)
        P = pooled_tensor(moments["P"], n, bandwidth_steps)
        b = pooled_tensor(moments["b_be"], n, bandwidth_steps)
        res = G @ coeffs[n] - b - P @ coeffs[n + 1]
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


def validation_score_generator(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int, order: int) -> float:
    N = moments["N"]
    a = a_coeffs(order)
    vals = []
    G_short = moments["G"][: N - order + 1]
    A = moments["A_orders"][order]
    b = moments["b_gen_orders"][order]
    for n in range(N - order, -1, -1):
        G = pooled_tensor(G_short, n, bandwidth_steps)
        A_n = pooled_tensor(A, n, bandwidth_steps)
        b_n = pooled_tensor(b, n, bandwidth_steps)
        future = np.zeros_like(coeffs[n])
        for j in range(1, order + 1):
            future += a[j] * coeffs[n + j]
        res = ((task.beta - a[0] / task.dt) * G - A_n) @ coeffs[n] - b_n - (1.0 / task.dt) * G @ future
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


def predict_values(task: TaskSpec, coeffs: Array, states: Array, index: int) -> Array:
    feats = task.feature_fn(states)
    return feats @ coeffs[index]


def estimate_mc_value(task: TaskSpec, states: Array, start_index: int, rng: np.random.Generator) -> Array:
    states = np.asarray(states)
    K = states.shape[0]
    values = np.zeros(K)
    disc0 = np.exp(-task.beta * task.dt)
    for _ in range(task.mc_rollouts):
        x = states.copy()
        disc = np.ones(K)
        ret = np.zeros(K)
        for n in range(start_index, task.horizon_steps):
            t = n * task.dt
            u = task.target_policy(x, t)
            ret += disc * task.reward_fn(x, t, u) * task.dt
            x = task.step_fn(x, t, u, task.dt, rng)
            disc *= disc0
        ret += disc * task.terminal_fn(x)
        values += ret
    values /= task.mc_rollouts
    return values


def prepare_eval_bundle(task: TaskSpec, test_states: Array, seed: int, eval_indices: Sequence[int] | None = None) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    if eval_indices is None:
        eval_indices = task.eval_indices
    state_list = []
    truth_list = []
    index_list = []
    for idx in eval_indices:
        episode_idx = rng.choice(test_states.shape[0], size=task.n_eval_states, replace=False)
        s = test_states[episode_idx, idx, :]
        gt = estimate_mc_value(task, s, idx, rng)
        state_list.append(s)
        truth_list.append(gt)
        index_list.append(np.full(task.n_eval_states, idx))
    return {
        "states": np.concatenate(state_list, axis=0),
        "truth": np.concatenate(truth_list, axis=0),
        "indices": np.concatenate(index_list, axis=0),
    }


def evaluate_coeffs(task: TaskSpec, coeffs: Array, eval_bundle: dict[str, Array]) -> tuple[float, float]:
    preds = np.zeros_like(eval_bundle["truth"])
    for idx in np.unique(eval_bundle["indices"]):
        mask = eval_bundle["indices"] == idx
        preds[mask] = predict_values(task, coeffs, eval_bundle["states"][mask], int(idx))
    err = preds - eval_bundle["truth"]
    integrated_rmse = math.sqrt(float(np.mean(err ** 2)))
    mask0 = eval_bundle["indices"] == 0
    t0_rmse = math.sqrt(float(np.mean(err[mask0] ** 2)))
    return t0_rmse, integrated_rmse


def evaluate_over_time(task: TaskSpec, coeffs: Array, test_states: Array, seed: int, time_grid: Sequence[int]) -> List[dict]:
    bundle = prepare_eval_bundle(task, test_states, seed + 777, eval_indices=time_grid)
    rows: List[dict] = []
    for idx in time_grid:
        mask = bundle["indices"] == idx
        preds = predict_values(task, coeffs, bundle["states"][mask], int(idx))
        err = preds - bundle["truth"][mask]
        rows.append({"time_index": int(idx), "time": task.dt * idx, "rmse": math.sqrt(float(np.mean(err ** 2)))})
    return rows


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------

def method_label(order: int | None) -> str:
    if order is None:
        return "BE"
    return f"Gen{order}"


def fit_method(task: TaskSpec, train_m: dict[str, Array], val_m: dict[str, Array], eval_bundle: dict[str, Array], order: int | None, bandwidth_steps: int) -> dict:
    tic = time.perf_counter()
    if order is None:
        coeffs = fit_be_from_moments(task, train_m, bandwidth_steps)
        runtime = time.perf_counter() - tic
        val_score = validation_score_be(task, coeffs, val_m, bandwidth_steps)
    else:
        coeffs = fit_generator_from_moments(task, train_m, bandwidth_steps, order=order)
        runtime = time.perf_counter() - tic
        val_score = validation_score_generator(task, coeffs, val_m, bandwidth_steps, order=order)
    t0_rmse, integrated_rmse = evaluate_coeffs(task, coeffs, eval_bundle)
    return {
        "coeffs": coeffs,
        "validation_score": val_score,
        "runtime_sec": runtime,
        "t0_rmse": t0_rmse,
        "integrated_rmse": integrated_rmse,
    }


def run_task(task: TaskSpec, seed: int, method_orders: Sequence[int | None]) -> tuple[List[dict], dict[str, dict], Array]:
    rng = np.random.default_rng(seed)
    states, rewards = rollout_episodes(task, task.n_episodes, rng, policy_kind="behavior")
    splits = split_episodes(states, rewards)
    train_states, train_rewards = splits["train"]
    val_states, val_rewards = splits["val"]
    test_states, _ = splits["test"]
    train_m = precompute_moments(task, train_states, train_rewards, max_order=max(o for o in method_orders if o is not None) if any(o is not None for o in method_orders) else 1)
    val_m = precompute_moments(task, val_states, val_rewards, max_order=max(o for o in method_orders if o is not None) if any(o is not None for o in method_orders) else 1)
    eval_bundle = prepare_eval_bundle(task, test_states, seed + 999)

    all_rows: List[dict] = []
    selected: dict[str, dict] = {}
    for order in method_orders:
        mname = method_label(order)
        cand_rows = []
        best_payload = None
        best_score = float("inf")
        for bw in task.bandwidth_grid:
            payload = fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "task": task.name,
                "label": task.label,
                "seed": seed,
                "method": mname,
                "order": -1 if order is None else order,
                "bandwidth_steps": bw,
                "validation_score": payload["validation_score"],
                "t0_rmse": payload["t0_rmse"],
                "integrated_rmse": payload["integrated_rmse"],
                "runtime_sec": payload["runtime_sec"],
            }
            all_rows.append(row)
            cand_rows.append((row, payload))
            if payload["validation_score"] < best_score:
                best_score = payload["validation_score"]
                best_payload = payload
        assert best_payload is not None
        best_row = min((r for r, _ in cand_rows), key=lambda r: r["validation_score"])
        selected[mname] = {**best_row, "coeffs": best_payload["coeffs"], "test_states": test_states}
    return all_rows, selected, test_states


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def mean_and_ci(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def plot_bandwidth_validation(rows: List[dict], out_path: Path) -> None:
    rows_small = [r for r in rows if r["task"] == "small_pendulum" and r["method"] in {"BE", "Gen2"}]
    methods = ["BE", "Gen2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        rr = sorted([r for r in rows_small if r["method"] == method], key=lambda r: (r["bandwidth_steps"], r["seed"]))
        xs = sorted({r["bandwidth_steps"] for r in rr})
        val_means, test_means = [], []
        for bw in xs:
            subset = [r for r in rr if r["bandwidth_steps"] == bw]
            val_means.append(np.mean([r["validation_score"] for r in subset]))
            test_means.append(np.mean([r["integrated_rmse"] for r in subset]))
        axes[0].plot(xs, val_means, marker="o", label=method)
        axes[1].plot(xs, test_means, marker="o", label=method)
    axes[0].set_xlabel("bandwidth steps")
    axes[0].set_ylabel("validation residual")
    axes[0].set_title("Small task: validation selection")
    axes[1].set_xlabel("bandwidth steps")
    axes[1].set_ylabel("held-out integrated RMSE")
    axes[1].set_title("Small task: test performance")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary(selected_rows: List[dict], out_path: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    xpos = np.arange(len(tasks))
    width = 0.34
    plt.figure(figsize=(8.2, 4.6))
    for j, method in enumerate(methods):
        means, cis = [], []
        for task_label in tasks:
            subset = [r for r in selected_rows if r["label"] == task_label and r["method"] == method]
            mean, ci = mean_and_ci([r["integrated_rmse"] for r in subset])
            means.append(mean)
            cis.append(ci)
        plt.bar(xpos + (j - 0.5) * width, means, width=width, label=method, yerr=cis, capsize=4)
    plt.xticks(xpos, tasks)
    plt.ylabel("integrated RMSE")
    plt.title("Expanded stage-2 pilot benchmark summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_over_time(rows: List[dict], out_path: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, task_label in zip(axes, tasks):
        for method in methods:
            subset = [r for r in rows if r["label"] == task_label and r["method"] == method]
            times = sorted({r["time"] for r in subset})
            means = []
            for t in times:
                vals = [r["rmse"] for r in subset if abs(r["time"] - t) < 1e-12]
                means.append(float(np.mean(vals)))
            ax.plot(times, means, marker="o", label=method)
        ax.set_title(task_label)
        ax.set_xlabel("time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_order_ablation(rows: List[dict], out_path: Path) -> None:
    methods = ["BE", "Gen1", "Gen2", "Gen3"]
    means, cis = [], []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        mean, ci = mean_and_ci([r["integrated_rmse"] for r in subset])
        means.append(mean)
        cis.append(ci)
    x = np.arange(len(methods))
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(x, means, yerr=cis, capsize=4)
    plt.xticks(x, methods)
    plt.ylabel("integrated RMSE")
    plt.title("Small-task order ablation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_nonstationarity_heatmap(rows: List[dict], out_path: Path) -> None:
    episode_grid = sorted({int(r["episodes"]) for r in rows})
    amp_grid = sorted({float(r["nonstat_scale"]) for r in rows})
    data = np.zeros((len(amp_grid), len(episode_grid)))
    for i, amp in enumerate(amp_grid):
        for j, ep in enumerate(episode_grid):
            subset = [r for r in rows if int(r["episodes"]) == ep and abs(float(r["nonstat_scale"]) - amp) < 1e-12]
            data[i, j] = float(np.mean([r["relative_improvement"] for r in subset]))
    plt.figure(figsize=(6.8, 4.6))
    im = plt.imshow(data, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(episode_grid)), episode_grid)
    plt.yticks(np.arange(len(amp_grid)), [f"{a:.1f}" for a in amp_grid])
    plt.xlabel("number of episodes")
    plt.ylabel("nonstationarity scale")
    plt.title("Generator improvement over Bellman baseline")
    cbar = plt.colorbar(im)
    cbar.set_label("relative RMSE improvement")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_runtime_pareto(rows: List[dict], out_path: Path) -> None:
    plt.figure(figsize=(6.8, 4.4))
    for row in rows:
        plt.scatter(row["runtime_sec"], row["integrated_rmse"], s=70)
        plt.annotate(f"{row['label']}-{row['method']}", (row["runtime_sec"], row["integrated_rmse"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("runtime (seconds)")
    plt.ylabel("integrated RMSE")
    plt.title("Accuracy-runtime comparison")
    plt.xscale("log")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "figures"
    results_dir = project_root / "results"
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    seeds = [20260326, 20260337, 20260348]
    tasks = [make_small_task(), make_medium_task(), make_large_task()]

    candidate_rows: List[dict] = []
    selected_rows: List[dict] = []
    over_time_rows: List[dict] = []

    # Main repeated-seed benchmark summary for BE vs Gen2.
    for task in tasks:
        time_grid = sorted(set([0, max(1, task.horizon_steps // 6), max(1, task.horizon_steps // 3), max(1, task.horizon_steps // 2), int(0.75 * task.horizon_steps), task.horizon_steps - 2]))
        for seed in seeds:
            task_rows, selected, test_states = run_task(task, seed, method_orders=[None, 2])
            candidate_rows.extend(task_rows)
            for method in ["BE", "Gen2"]:
                row = {k: v for k, v in selected[method].items() if k not in {"coeffs", "test_states"}}
                selected_rows.append(row)
                for ot in evaluate_over_time(task, selected[method]["coeffs"], test_states, seed + 314, time_grid):
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

    # Order ablation on the small task.
    order_rows: List[dict] = []
    for seed in seeds:
        task_rows, selected, _ = run_task(make_small_task(), seed + 500, method_orders=[None, 1, 2, 3])
        for method in ["BE", "Gen1", "Gen2", "Gen3"]:
            row = {k: v for k, v in selected[method].items() if k not in {"coeffs", "test_states"}}
            order_rows.append(row)

    # Nonstationarity-vs-data heatmap on the small task.
    heat_rows: List[dict] = []
    amp_grid = [0.0, 0.5, 1.0, 1.5]
    episode_grid = [60, 120, 180, 300]
    for amp in amp_grid:
        for episodes in episode_grid:
            task = make_small_task(n_episodes=episodes, nonstat_scale=amp, mc_rollouts=32)
            _, selected, _ = run_task(task, 20260500 + int(100 * amp) + episodes, method_orders=[None, 2])
            be = selected["BE"]["integrated_rmse"]
            gen = selected["Gen2"]["integrated_rmse"]
            heat_rows.append(
                {
                    "nonstat_scale": amp,
                    "episodes": episodes,
                    "be_integrated_rmse": be,
                    "gen2_integrated_rmse": gen,
                    "relative_improvement": (be - gen) / max(be, 1e-12),
                }
            )

    # Write CSVs.
    write_csv(
        results_dir / "stage2_extended_candidates.csv",
        candidate_rows,
        ["task", "label", "seed", "method", "order", "bandwidth_steps", "validation_score", "t0_rmse", "integrated_rmse", "runtime_sec"],
    )
    write_csv(
        results_dir / "stage2_extended_selected.csv",
        selected_rows,
        ["task", "label", "seed", "method", "order", "bandwidth_steps", "validation_score", "t0_rmse", "integrated_rmse", "runtime_sec"],
    )
    write_csv(
        results_dir / "stage2_extended_over_time.csv",
        over_time_rows,
        ["task", "label", "seed", "method", "time_index", "time", "rmse"],
    )
    write_csv(
        results_dir / "stage2_order_ablation.csv",
        order_rows,
        ["task", "label", "seed", "method", "order", "bandwidth_steps", "validation_score", "t0_rmse", "integrated_rmse", "runtime_sec"],
    )
    write_csv(
        results_dir / "stage2_nonstationarity_heatmap.csv",
        heat_rows,
        ["nonstat_scale", "episodes", "be_integrated_rmse", "gen2_integrated_rmse", "relative_improvement"],
    )

    # Figures.
    plot_bandwidth_validation(candidate_rows, figures_dir / "stage2_bandwidth_validation.pdf")
    plot_summary(selected_rows, figures_dir / "stage2_extended_summary.pdf")
    plot_over_time(over_time_rows, figures_dir / "stage2_over_time_rmse.pdf")
    plot_order_ablation(order_rows, figures_dir / "stage2_order_ablation.pdf")
    plot_nonstationarity_heatmap(heat_rows, figures_dir / "stage2_nonstationarity_heatmap.pdf")

    pareto_points: List[dict] = []
    for task_label in ["Small", "Medium", "Large"]:
        for method in ["BE", "Gen2"]:
            subset = [r for r in selected_rows if r["label"] == task_label and r["method"] == method]
            pareto_points.append(
                {
                    "label": task_label,
                    "method": method,
                    "runtime_sec": float(np.mean([r["runtime_sec"] for r in subset])),
                    "integrated_rmse": float(np.mean([r["integrated_rmse"] for r in subset])),
                }
            )
    plot_runtime_pareto(pareto_points, figures_dir / "stage2_runtime_pareto.pdf")

    print("Wrote expanded stage-2 results and figures.")
    print(results_dir / "stage2_extended_selected.csv")
    print(results_dir / "stage2_extended_over_time.csv")
    print(results_dir / "stage2_order_ablation.csv")
    print(results_dir / "stage2_nonstationarity_heatmap.csv")
    print(figures_dir / "stage2_bandwidth_validation.pdf")
    print(figures_dir / "stage2_extended_summary.pdf")
    print(figures_dir / "stage2_over_time_rmse.pdf")
    print(figures_dir / "stage2_order_ablation.pdf")
    print(figures_dir / "stage2_nonstationarity_heatmap.pdf")
    print(figures_dir / "stage2_runtime_pareto.pdf")


if __name__ == "__main__":
    main()
