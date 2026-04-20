
"""Stage-2 pilot benchmark suite for the time-inhomogeneous policy-evaluation revision.

This runner instantiates three self-contained control tasks of increasing scale,
splits episodes into train/validation/test sets, selects temporal-pooling
bandwidth on the validation set, evaluates on held-out states with Monte Carlo
ground truth, and writes the pilot figures used in the week-1/week-2 package.

Run from the project root:
    python scripts/run_stage2_benchmark_suite.py
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


def a_coeffs(i: int) -> Array:
    j = np.arange(i + 1)
    A = np.vander(j, N=i + 1, increasing=True).T
    b = np.zeros(i + 1)
    b[1] = 1.0
    return np.linalg.solve(A, b)


def wrap_angle(x: Array) -> Array:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def triangular_kernel_weights(center: int, indices: Array, bandwidth_steps: int) -> Array:
    if bandwidth_steps <= 0:
        return np.ones(len(indices), dtype=float)
    w = 1.0 - np.abs(indices - center) / (bandwidth_steps + 1.0)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        w = np.ones(len(indices), dtype=float)
        s = float(w.sum())
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


def make_small_task() -> TaskSpec:
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

    def step_fn(x: Array, t: float, u: Array, dt: float, rng: np.random.Generator) -> Array:
        theta = wrap_angle(x[:, 0])
        omega = x[:, 1]
        gravity = 9.81 * (1.0 + 0.12 * math.sin(1.4 * t))
        damping = 0.14 + 0.05 * (1.0 + math.sin(0.7 * t + 0.3)) / 2.0
        gain = 1.0 + 0.08 * math.cos(0.5 * t)
        noise = 0.12 * math.sqrt(dt) * rng.normal(size=len(x))
        omega_next = omega + dt * (-gravity * np.sin(theta) - damping * omega + gain * u[:, 0]) + noise
        theta_next = wrap_angle(theta + dt * omega_next)
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
        dt=0.08,
        horizon_steps=40,
        beta=0.04,
        n_episodes=180,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 16, 30],
        n_eval_states=12,
        mc_rollouts=64,
        ridge=2e-4,
        feature_fn=pendulum_features,
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_medium_task() -> TaskSpec:
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

    def step_fn(x: Array, t: float, u: Array, dt: float, rng: np.random.Generator) -> Array:
        pos = x[:, 0]
        vel = x[:, 1]
        ang = x[:, 2]
        ang_vel = x[:, 3]
        a1 = 1.1 + 0.25 * math.sin(0.6 * t)
        a2 = 0.35 + 0.08 * math.cos(0.4 * t)
        a3 = 0.75 + 0.12 * math.sin(0.9 * t + 0.2)
        b1 = 0.55 + 0.15 * math.cos(0.5 * t)
        b2 = 1.55 + 0.18 * math.sin(0.7 * t)
        b3 = 0.42 + 0.08 * math.sin(0.3 * t)
        gain = 1.0 + 0.12 * math.cos(0.45 * t)
        noise = 0.08 * math.sqrt(dt) * rng.normal(size=(len(x), 2))
        vel_next = vel + dt * (-a1 * pos - a2 * vel + a3 * ang + 0.45 * gain * u[:, 0]) + noise[:, 0]
        ang_vel_next = ang_vel + dt * (b1 * pos - b2 * ang - b3 * ang_vel + 1.0 * gain * u[:, 0] - 0.12 * np.sin(ang)) + noise[:, 1]
        pos_next = pos + dt * vel_next
        ang_next = ang + dt * ang_vel_next
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
        dt=0.08,
        horizon_steps=40,
        beta=0.04,
        n_episodes=200,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 16, 30],
        n_eval_states=12,
        mc_rollouts=64,
        ridge=3e-4,
        feature_fn=lambda x: quadratic_features(x, include_linear=True),
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


def make_large_task() -> TaskSpec:
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
            omega = 0.95 + 0.16 * math.sin(0.25 * t + 0.25 * k)
            damp = 0.42 + 0.05 * math.cos(0.18 * t + 0.15 * k)
            A[i, i + 1] = 1.0
            A[i + 1, i] = -(omega ** 2)
            A[i + 1, i + 1] = -damp
            if k < 5:
                A[i + 1, i + 2] = 0.06 * math.sin(0.4 * t + 0.1 * k)
        B = np.zeros((12, 3))
        for j in range(3):
            B[4 * j + 1, j] = 1.0 + 0.08 * math.cos(0.35 * t + j)
            B[4 * j + 3, j] = 0.75 + 0.06 * math.sin(0.45 * t + j)
        q_diag = np.array(
            [1.2, 0.3, 1.0, 0.25, 1.3, 0.3, 1.1, 0.25, 1.15, 0.28, 1.05, 0.25],
            dtype=float,
        ) * (1.0 + 0.08 * math.sin(0.22 * t))
        return A, B, q_diag

    def step_fn(x: Array, t: float, u: Array, dt: float, rng: np.random.Generator) -> Array:
        A, B, _ = matrices(t)
        drift = x @ A.T + u @ B.T
        noise = np.zeros_like(x)
        vel_idx = [1, 3, 5, 7, 9, 11]
        noise[:, vel_idx] = 0.06 * math.sqrt(dt) * rng.normal(size=(len(x), len(vel_idx)))
        return x + dt * drift + noise

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
        dt=0.08,
        horizon_steps=32,
        beta=0.04,
        n_episodes=240,
        bandwidth_grid=[0, 1, 2, 4],
        eval_indices=[0, 12, 24],
        n_eval_states=10,
        mc_rollouts=48,
        ridge=6e-4,
        feature_fn=lambda x: quadratic_features(x, include_linear=True),
        sample_initial_states=sample_initial_states,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        step_fn=step_fn,
        reward_fn=reward_fn,
        terminal_fn=terminal_fn,
    )


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


def precompute_moments(task: TaskSpec, states: Array, rewards: Array) -> dict[str, Array]:
    Phi = task.feature_fn(states)
    M, Np1, p = Phi.shape
    N = Np1 - 1
    G = np.zeros((N, p, p))
    P = np.zeros((N, p, p))
    b_be = np.zeros((N, p))
    A = np.zeros((N - 1, p, p))
    b_gen = np.zeros((N - 1, p))
    for n in range(N):
        phi_n = Phi[:, n, :]
        G[n] = (phi_n.T @ phi_n) / M
        b_be[n] = (phi_n.T @ rewards[:, n]) / M * task.dt
        P[n] = math.exp(-task.beta * task.dt) * (phi_n.T @ Phi[:, n + 1, :]) / M
        if n <= N - 2:
            a = np.array([-1.5, 2.0, -0.5])
            delta_phi = a[0] * Phi[:, n, :] + a[1] * Phi[:, n + 1, :] + a[2] * Phi[:, n + 2, :]
            A[n] = (phi_n.T @ (delta_phi / task.dt)) / M
            b_gen[n] = (phi_n.T @ rewards[:, n]) / M
    term_G = (Phi[:, N, :].T @ Phi[:, N, :]) / M
    term_y = (Phi[:, N, :].T @ task.terminal_fn(states[:, N, :])) / M
    return {
        "Phi": Phi,
        "G": G,
        "P": P,
        "b_be": b_be,
        "A": A,
        "b_gen": b_gen,
        "term_G": term_G,
        "term_y": term_y,
        "N": N,
        "p": p,
    }


def pooled_tensor(arr: Array, center: int, bandwidth_steps: int, max_index: int) -> Array:
    idx = np.arange(max(0, center - bandwidth_steps), min(max_index, center + bandwidth_steps) + 1)
    w = triangular_kernel_weights(center, idx, bandwidth_steps)
    return np.tensordot(w, arr[idx], axes=(0, 0))


def fit_be_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> Array:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    coeffs[N] = solve_reg(moments["term_G"], moments["term_y"], task.ridge)
    for n in range(N - 1, -1, -1):
        G = pooled_tensor(moments["G"], n, bandwidth_steps, N - 1)
        P = pooled_tensor(moments["P"], n, bandwidth_steps, N - 1)
        b = pooled_tensor(moments["b_be"], n, bandwidth_steps, N - 1)
        coeffs[n] = solve_reg(G, b + P @ coeffs[n + 1], task.ridge)
    return coeffs


def fit_generator_from_moments(task: TaskSpec, moments: dict[str, Array], bandwidth_steps: int) -> Array:
    N = moments["N"]
    p = moments["p"]
    coeffs = np.zeros((N + 1, p))
    be_coeffs = fit_be_from_moments(task, moments, bandwidth_steps)
    coeffs[N] = be_coeffs[N]
    coeffs[N - 1] = be_coeffs[N - 1]
    a = np.array([-1.5, 2.0, -0.5])
    for n in range(N - 2, -1, -1):
        G = pooled_tensor(moments["G"][: N - 1], n, bandwidth_steps, N - 2)
        A = pooled_tensor(moments["A"], n, bandwidth_steps, N - 2)
        b = pooled_tensor(moments["b_gen"], n, bandwidth_steps, N - 2)
        Mmat = (task.beta - a[0] / task.dt) * G - A
        rhs = b + (1.0 / task.dt) * G @ (a[1] * coeffs[n + 1] + a[2] * coeffs[n + 2])
        coeffs[n] = solve_reg(Mmat, rhs, task.ridge)
    return coeffs


def validation_score_be(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int) -> float:
    N = moments["N"]
    vals = []
    for n in range(N - 1, -1, -1):
        G = pooled_tensor(moments["G"], n, bandwidth_steps, N - 1)
        P = pooled_tensor(moments["P"], n, bandwidth_steps, N - 1)
        b = pooled_tensor(moments["b_be"], n, bandwidth_steps, N - 1)
        res = G @ coeffs[n] - b - P @ coeffs[n + 1]
        vals.append(float(np.linalg.norm(res)))
    return float(np.mean(vals))


def validation_score_generator(task: TaskSpec, coeffs: Array, moments: dict[str, Array], bandwidth_steps: int) -> float:
    N = moments["N"]
    a = np.array([-1.5, 2.0, -0.5])
    vals = []
    for n in range(N - 2, -1, -1):
        G = pooled_tensor(moments["G"][: N - 1], n, bandwidth_steps, N - 2)
        A = pooled_tensor(moments["A"], n, bandwidth_steps, N - 2)
        b = pooled_tensor(moments["b_gen"], n, bandwidth_steps, N - 2)
        res = ((task.beta - a[0] / task.dt) * G - A) @ coeffs[n] - b - (1.0 / task.dt) * G @ (
            a[1] * coeffs[n + 1] + a[2] * coeffs[n + 2]
        )
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
    for r in range(task.mc_rollouts):
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


def prepare_eval_bundle(task: TaskSpec, test_states: Array, seed: int) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    state_list = []
    truth_list = []
    index_list = []
    for idx in task.eval_indices:
        episode_idx = rng.choice(test_states.shape[0], size=task.n_eval_states, replace=False)
        s = test_states[episode_idx, idx, :]
        gt = estimate_mc_value(task, s, idx, rng)
        state_list.append(s)
        truth_list.append(gt)
        index_list.append(np.full(task.n_eval_states, idx))
    states_eval = np.concatenate(state_list, axis=0)
    truth_eval = np.concatenate(truth_list, axis=0)
    indices_eval = np.concatenate(index_list, axis=0)
    return {"states": states_eval, "truth": truth_eval, "indices": indices_eval}


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


def run_task(task: TaskSpec, seed: int) -> tuple[List[dict], dict]:
    rng = np.random.default_rng(seed)
    states, rewards = rollout_episodes(task, task.n_episodes, rng, policy_kind="behavior")
    splits = split_episodes(states, rewards)
    train_states, train_rewards = splits["train"]
    val_states, val_rewards = splits["val"]
    test_states, test_rewards = splits["test"]
    train_m = precompute_moments(task, train_states, train_rewards)
    val_m = precompute_moments(task, val_states, val_rewards)
    eval_bundle = prepare_eval_bundle(task, test_states, seed + 999)

    records: List[dict] = []
    methods = ["BE", "Generator"]
    for method in methods:
        for bw in task.bandwidth_grid:
            tic = time.perf_counter()
            if method == "BE":
                coeffs = fit_be_from_moments(task, train_m, bw)
                runtime = time.perf_counter() - tic
                val_score = validation_score_be(task, coeffs, val_m, bw)
            else:
                coeffs = fit_generator_from_moments(task, train_m, bw)
                runtime = time.perf_counter() - tic
                val_score = validation_score_generator(task, coeffs, val_m, bw)
            t0_rmse, integrated_rmse = evaluate_coeffs(task, coeffs, eval_bundle)
            records.append(
                {
                    "task": task.name,
                    "label": task.label,
                    "method": method,
                    "bandwidth_steps": bw,
                    "validation_score": val_score,
                    "t0_rmse": t0_rmse,
                    "integrated_rmse": integrated_rmse,
                    "runtime_sec": runtime,
                }
            )

    # select by validation score
    selected = {}
    for method in methods:
        cand = [r for r in records if r["method"] == method]
        best = min(cand, key=lambda r: r["validation_score"])
        selected[method] = best
    return records, selected


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_bandwidth_figure(rows: List[dict], out_path: Path) -> None:
    rows_small = [r for r in rows if r["task"] == "small_pendulum"]
    methods = ["BE", "Generator"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        rr = sorted([r for r in rows_small if r["method"] == method], key=lambda r: r["bandwidth_steps"])
        x = [r["bandwidth_steps"] for r in rr]
        axes[0].plot(x, [r["validation_score"] for r in rr], marker="o", label=method)
        axes[1].plot(x, [r["integrated_rmse"] for r in rr], marker="o", label=method)
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


def make_summary_figure(selected_rows: List[dict], out_path: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Generator"]
    xpos = np.arange(len(tasks))
    width = 0.34
    plt.figure(figsize=(8, 4.2))
    for j, method in enumerate(methods):
        vals = []
        for task_label in tasks:
            rec = next(r for r in selected_rows if r["label"] == task_label and r["method"] == method)
            vals.append(rec["integrated_rmse"])
        plt.bar(xpos + (j - 0.5) * width, vals, width=width, label=method)
    plt.xticks(xpos, tasks)
    plt.ylabel("integrated RMSE")
    plt.title("Stage-2 pilot benchmark summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "figures"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    tasks = [make_small_task(), make_medium_task(), make_large_task()]
    all_rows: List[dict] = []
    selected_rows: List[dict] = []

    seed = 20260325
    for i, task in enumerate(tasks):
        task_rows, selected = run_task(task, seed + 97 * i)
        all_rows.extend(task_rows)
        selected_rows.extend([selected["BE"], selected["Generator"]])

    write_csv(
        results_dir / "stage2_bandwidth_validation.csv",
        [r for r in all_rows if r["task"] == "small_pendulum"],
        ["task", "label", "method", "bandwidth_steps", "validation_score", "t0_rmse", "integrated_rmse", "runtime_sec"],
    )
    write_csv(
        results_dir / "stage2_pilot_summary.csv",
        selected_rows,
        ["task", "label", "method", "bandwidth_steps", "validation_score", "t0_rmse", "integrated_rmse", "runtime_sec"],
    )

    make_bandwidth_figure(all_rows, figures_dir / "bandwidth_validation.pdf")
    make_summary_figure(selected_rows, figures_dir / "stage2_suite_summary.pdf")

    print("Wrote:")
    print(results_dir / "stage2_bandwidth_validation.csv")
    print(results_dir / "stage2_pilot_summary.csv")
    print(figures_dir / "bandwidth_validation.pdf")
    print(figures_dir / "stage2_suite_summary.pdf")


if __name__ == "__main__":
    main()
