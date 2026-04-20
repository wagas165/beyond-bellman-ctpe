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
from plotly.subplots import make_subplots

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common


PROFILES = {
    "smoke": {
        "seeds_small": 1,
        "seeds_medium": 1,
        "seeds_large": 1,
        "episodes_small": 80,
        "episodes_medium": 100,
        "episodes_large": 120,
    },
    "main": {
        "seeds_small": 8,
        "seeds_medium": 8,
        "seeds_large": 6,
        "episodes_small": 320,
        "episodes_medium": 420,
        "episodes_large": 560,
    },
    "heavy": {
        "seeds_small": 12,
        "seeds_medium": 12,
        "seeds_large": 8,
        "episodes_small": 480,
        "episodes_medium": 620,
        "episodes_large": 820,
    },
}


def mean_ci(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def run_state_only_methods(task, seed: int, method_orders: Sequence[int | None], gain_bias: float = 0.0):
    rng = np.random.default_rng(seed)
    states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior", gain_bias=gain_bias)
    splits = common.split_with_actions(states, actions, rewards)
    train_states, train_actions, train_rewards = splits["train"]
    val_states, val_actions, val_rewards = splits["val"]
    test_states, test_actions, test_rewards = splits["test"]
    max_order = max([o for o in method_orders if o is not None], default=1)
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=max_order)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=max_order)
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 1234, eval_indices=task.eval_indices)

    rows = []
    extra = {
        "train": (train_states, train_actions, train_rewards),
        "val": (val_states, val_actions, val_rewards),
        "test_states": test_states,
        "train_m": train_m,
    }
    selected_payloads = {}
    for order in method_orders:
        best_score = float("inf")
        best_row = None
        best_payload = None
        for bw in task.bandwidth_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "task": task.name,
                "label": task.label,
                "seed": seed,
                "method": ("BE" if order is None else f"Gen{order}"),
                "order": -1 if order is None else order,
                "bandwidth_steps": int(bw),
                "validation_score": payload["validation_score"],
                "t0_rmse": payload["t0_rmse"],
                "integrated_rmse": payload["integrated_rmse"],
                "runtime_sec": payload["runtime_sec"],
                "gain_bias": gain_bias,
            }
            if payload["validation_score"] < best_score:
                best_score = payload["validation_score"]
                best_row = row
                best_payload = payload
        assert best_row is not None and best_payload is not None
        rows.append(best_row)
        selected_payloads[best_row["method"]] = best_payload
    return rows, selected_payloads, extra


def run_mb_linear(task, seed: int, gain_bias: float = 0.0):
    rng = np.random.default_rng(seed)
    states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior", gain_bias=gain_bias)
    splits = common.split_with_actions(states, actions, rewards)
    train_states, train_actions, train_rewards = splits["train"]
    val_states, val_actions, val_rewards = splits["val"]
    test_states, _, _ = splits["test"]
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 4321, eval_indices=task.eval_indices)
    best_score = float("inf")
    best_row = None
    for bw in task.bandwidth_grid:
        tic = time.perf_counter()
        model = common.fit_model_based_baseline(task, train_states, train_actions, train_rewards, bw, feature_kind="linear")
        runtime = time.perf_counter() - tic
        val_score = common.model_validation_score(task, model, val_states, val_actions, val_rewards)
        t0_rmse, integrated_rmse = common.evaluate_model_based_value(task, model, eval_bundle, seed + 8765 + bw)
        row = {
            "task": task.name,
            "label": task.label,
            "seed": seed,
            "method": "MBLinear",
            "order": -1,
            "bandwidth_steps": int(bw),
            "validation_score": val_score,
            "t0_rmse": t0_rmse,
            "integrated_rmse": integrated_rmse,
            "runtime_sec": runtime,
            "gain_bias": gain_bias,
        }
        if val_score < best_score:
            best_score = val_score
            best_row = row
    assert best_row is not None
    return best_row


def dt_sweep(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    T = 3.36
    dt_grid = [0.12, 0.08, 0.04, 0.02] if profile_cfg["seeds_small"] >= 8 else ([0.12, 0.06] if profile_cfg["seeds_small"] <= 1 else [0.12, 0.08, 0.04])
    for family, seed_count in [("Small", profile_cfg["seeds_small"]), ("Medium", profile_cfg["seeds_medium"])]:
        for dt in dt_grid:
            horizon = max(24, int(round(T / dt)))
            if family == "Small":
                task = common.make_small_heavy_task(n_episodes=profile_cfg["episodes_small"], dt=dt, horizon_steps=horizon, mc_rollouts=64)
            else:
                task = common.make_medium_heavy_task(n_episodes=profile_cfg["episodes_medium"], dt=dt, horizon_steps=horizon, mc_rollouts=64)
            for seed in range(seed_count):
                chosen, _, _ = run_state_only_methods(task, 20262000 + 1000 * (family == "Medium") + 10 * int(100 * dt) + seed, [None, 1, 2, 3])
                for row in chosen:
                    rows.append({**row, "family": family, "dt": dt, "horizon_steps": horizon})
    return pd.DataFrame(rows)


def budget_nonstat_sweep(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    if profile_cfg["seeds_small"] <= 1:
        episode_grid = [80, 160]
        amp_grid = [0.0, 1.0]
    else:
        episode_grid = [120, 240, 480, 960] if profile_cfg["seeds_small"] >= 8 else [120, 240, 480]
        amp_grid = [0.0, 0.5, 1.0, 1.5, 2.0] if profile_cfg["seeds_small"] >= 8 else [0.0, 0.5, 1.0, 1.5]
    for episodes in episode_grid:
        for amp in amp_grid:
            for seed in range(profile_cfg["seeds_small"]):
                task = common.make_small_heavy_task(n_episodes=episodes, nonstat_scale=amp, mc_rollouts=48)
                selected, _, _ = run_state_only_methods(task, 20263000 + episodes + int(100 * amp) + seed, [None, 2])
                be = next(r for r in selected if r["method"] == "BE")
                gen = next(r for r in selected if r["method"] == "Gen2")
                rows.append(
                    {
                        "episodes": episodes,
                        "nonstat_scale": amp,
                        "seed": seed,
                        "be_integrated_rmse": be["integrated_rmse"],
                        "gen2_integrated_rmse": gen["integrated_rmse"],
                        "relative_gain": (be["integrated_rmse"] - gen["integrated_rmse"]) / max(be["integrated_rmse"], 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def behavior_mismatch_sweep(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    gain_grid = [0.0, 0.15, 0.35] if profile_cfg["seeds_small"] <= 1 else [0.0, 0.1, 0.2, 0.35, 0.5]
    task_specs = [
        common.make_small_heavy_task(n_episodes=profile_cfg["episodes_small"], mc_rollouts=56),
        common.make_medium_heavy_task(n_episodes=profile_cfg["episodes_medium"], mc_rollouts=56),
        common.make_large_heavy_task(n_episodes=profile_cfg["episodes_large"], mc_rollouts=48),
    ]
    seeds_by_label = {
        "Small": profile_cfg["seeds_small"],
        "Medium": profile_cfg["seeds_medium"],
        "Large": profile_cfg["seeds_large"],
    }
    for task in task_specs:
        for gain in gain_grid:
            for seed in range(seeds_by_label[task.label]):
                selected, _, _ = run_state_only_methods(task, 20264000 + 1000 * (task.label == "Medium") + 2000 * (task.label == "Large") + int(100 * gain) + seed, [None, 2], gain_bias=gain)
                mb = run_mb_linear(task, 20265000 + 1000 * (task.label == "Medium") + 2000 * (task.label == "Large") + int(100 * gain) + seed, gain_bias=gain)
                for row in selected + [mb]:
                    rows.append({**row, "behavior_gain_bias": gain})
    return pd.DataFrame(rows)


def feature_ablation(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    if profile_cfg["seeds_small"] <= 1:
        small_families = ["rich", "linear"]
        med_families = ["quadratic", "linear"]
        lg_families = ["quadratic"]
    else:
        small_families = ["rich", "quadratic", "reduced", "linear"]
        med_families = ["quadratic", "linear"]
        lg_families = ["quadratic", "linear"]
    for family in small_families:
        task = common.make_small_heavy_task(n_episodes=profile_cfg["episodes_small"], feature_family=family, mc_rollouts=56)
        for seed in range(profile_cfg["seeds_small"]):
            selected, _, _ = run_state_only_methods(task, 20266000 + 17 * seed + len(family), [None, 2])
            for row in selected:
                rows.append({**row, "feature_family": family, "task_family": "Small"})
    for family in med_families:
        task = common.make_medium_heavy_task(n_episodes=profile_cfg["episodes_medium"], feature_family=family, mc_rollouts=56)
        for seed in range(profile_cfg["seeds_medium"]):
            selected, _, _ = run_state_only_methods(task, 20267000 + 17 * seed + len(family), [None, 2])
            for row in selected:
                rows.append({**row, "feature_family": family, "task_family": "Medium"})
    for family in lg_families:
        task = common.make_large_heavy_task(n_episodes=profile_cfg["episodes_large"], feature_family=family, mc_rollouts=48)
        for seed in range(profile_cfg["seeds_large"]):
            selected, _, _ = run_state_only_methods(task, 20268000 + 17 * seed + len(family), [None, 2])
            for row in selected:
                rows.append({**row, "feature_family": family, "task_family": "Large"})
    return pd.DataFrame(rows)


def startup_ablation(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    tasks = [
        common.make_small_heavy_task(n_episodes=profile_cfg["episodes_small"], mc_rollouts=56),
        common.make_medium_heavy_task(n_episodes=profile_cfg["episodes_medium"], mc_rollouts=56),
    ]
    for task in tasks:
        seeds = profile_cfg["seeds_small"] if task.label == "Small" else profile_cfg["seeds_medium"]
        for order in [2, 3]:
            for startup_mode in ["be", "zero"]:
                for seed in range(seeds):
                    rng = np.random.default_rng(20269000 + 100 * order + 10 * seed + (startup_mode == "zero"))
                    states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior")
                    splits = common.split_with_actions(states, actions, rewards)
                    train_states, train_actions, train_rewards = splits["train"]
                    val_states, val_actions, val_rewards = splits["val"]
                    test_states, _, _ = splits["test"]
                    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=order)
                    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=order)
                    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, 20269500 + seed, eval_indices=task.eval_indices)
                    best = None
                    best_val = float("inf")
                    for bw in task.bandwidth_grid:
                        tic = time.perf_counter()
                        coeffs = common.fit_generator_with_startup(task, train_m, bw, order=order, startup_mode=startup_mode)
                        runtime = time.perf_counter() - tic
                        val_score = base.validation_score_generator(task, coeffs, val_m, bw, order)
                        t0_rmse, irmse = base.evaluate_coeffs(task, coeffs, eval_bundle)
                        row = {
                            "task": task.label,
                            "seed": seed,
                            "order": order,
                            "startup_mode": startup_mode,
                            "bandwidth_steps": bw,
                            "validation_score": val_score,
                            "t0_rmse": t0_rmse,
                            "integrated_rmse": irmse,
                            "runtime_sec": runtime,
                        }
                        if val_score < best_val:
                            best_val = val_score
                            best = row
                    assert best is not None
                    rows.append(best)
    return pd.DataFrame(rows)


def stability_diagnostics(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    tasks = [
        common.make_small_heavy_task(n_episodes=profile_cfg["episodes_small"]),
        common.make_medium_heavy_task(n_episodes=profile_cfg["episodes_medium"]),
        common.make_large_heavy_task(n_episodes=profile_cfg["episodes_large"]),
    ]
    for task in tasks:
        for seed in range(min(4, profile_cfg["seeds_small"])):
            rng = np.random.default_rng(20270000 + 1000 * (task.label == "Medium") + 2000 * (task.label == "Large") + seed)
            states, actions, rewards = common.rollout_with_actions(task, task.n_episodes, rng, policy_kind="behavior")
            train_states, train_actions, train_rewards = common.split_with_actions(states, actions, rewards)["train"]
            moments = base.precompute_moments(task, train_states, train_rewards, max_order=2)
            for bw in task.bandwidth_grid:
                for row in common.gram_diagnostics(task, moments, bw):
                    rows.append({**row, "task": task.label, "seed": seed})
    return pd.DataFrame(rows)


def runtime_scaling(profile_cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    dims = [4, 12, 24] if profile_cfg["seeds_small"] <= 1 else [4, 8, 12, 24, 48]
    horizons = [24, 48] if profile_cfg["seeds_small"] <= 1 else ([32, 48, 64] if profile_cfg["seeds_large"] >= 6 else [32, 48])
    for dim in dims:
        action_dim = max(1, dim // 4)
        task = common.make_networked_lq_task(
            state_dim=dim,
            action_dim=action_dim,
            label=f"D{dim}",
            n_episodes=profile_cfg["episodes_large"],
            horizon_steps=48,
            mc_rollouts=32,
            feature_family="linear" if dim >= 24 else "quadratic",
        )
        for seed in range(min(5, profile_cfg["seeds_large"])):
            selected, _, _ = run_state_only_methods(task, 20271000 + dim * 10 + seed, [None, 2])
            for row in selected:
                rows.append({**row, "dimension": dim, "horizon_mode": 48})
    for H in horizons:
        task = common.make_large_heavy_task(n_episodes=profile_cfg["episodes_large"], horizon_steps=H, mc_rollouts=32)
        for seed in range(min(5, profile_cfg["seeds_large"])):
            selected, _, _ = run_state_only_methods(task, 20272000 + H * 10 + seed, [None, 2])
            for row in selected:
                rows.append({**row, "dimension": task.state_dim, "horizon_mode": H})
    return pd.DataFrame(rows)


def plot_dt_sweep(df: pd.DataFrame, out_pdf: Path) -> None:
    families = ["Small", "Medium"]
    methods = ["BE", "Gen1", "Gen2", "Gen3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, fam in zip(axes, families):
        sub = df[df["family"] == fam]
        for method in methods:
            ss = sub[sub["method"] == method]
            xs = sorted(ss["dt"].unique())
            means = [ss[ss["dt"] == x]["integrated_rmse"].mean() for x in xs]
            ax.plot(xs, means, marker="o", label=method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(fam)
        ax.set_xlabel("dt")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Integrated RMSE")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_budget_nonstat(df: pd.DataFrame, out_pdf: Path) -> None:
    episode_grid = sorted(df["episodes"].unique())
    amp_grid = sorted(df["nonstat_scale"].unique())
    z = np.zeros((len(amp_grid), len(episode_grid)))
    for i, amp in enumerate(amp_grid):
        for j, ep in enumerate(episode_grid):
            z[i, j] = df[(df["episodes"] == ep) & (df["nonstat_scale"] == amp)]["relative_gain"].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(episode_grid)))
    ax.set_xticklabels(episode_grid)
    ax.set_yticks(np.arange(len(amp_grid)))
    ax.set_yticklabels([f"{a:.1f}" for a in amp_grid])
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Nonstationarity scale")
    ax.set_title("Gen2 gain over BE")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative gain")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_behavior_mismatch(df: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    methods = ["BE", "Gen2", "MBLinear"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    for ax, task in zip(axes, tasks):
        for method in methods:
            sub = df[(df["label"] == task) & (df["method"] == method)]
            xs = sorted(sub["behavior_gain_bias"].unique())
            means = [sub[sub["behavior_gain_bias"] == x]["integrated_rmse"].mean() for x in xs]
            ax.plot(xs, means, marker="o", label=method)
        ax.set_title(task)
        ax.set_xlabel("Behavior gain bias")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Integrated RMSE")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_feature_ablation(df: pd.DataFrame, out_pdf: Path) -> None:
    tasks = ["Small", "Medium", "Large"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for ax, task in zip(axes, tasks):
        sub = df[df["task_family"] == task]
        families = list(dict.fromkeys(sub["feature_family"].tolist()))
        x = np.arange(len(families))
        width = 0.34
        for j, method in enumerate(["BE", "Gen2"]):
            means = [sub[(sub["feature_family"] == fam) & (sub["method"] == method)]["integrated_rmse"].mean() for fam in families]
            ax.bar(x + (j - 0.5) * width, means, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=20)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Integrated RMSE")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_startup_ablation(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, task in zip(axes, ["Small", "Medium"]):
        sub = df[df["task"] == task]
        labels, means = [], []
        for order in [2, 3]:
            for startup in ["be", "zero"]:
                labels.append(f"Gen{order}-{startup}")
                means.append(sub[(sub["order"] == order) & (sub["startup_mode"] == startup)]["integrated_rmse"].mean())
        ax.bar(np.arange(len(labels)), means)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Integrated RMSE")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_stability(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric in zip(axes, ["min_eig", "cond"]):
        for task in ["Small", "Medium", "Large"]:
            sub = df[df["task"] == task].groupby("bandwidth_steps")[metric].mean().reset_index()
            ax.plot(sub["bandwidth_steps"], sub[metric], marker="o", label=task)
        ax.set_xlabel("Bandwidth steps")
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average value")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_scaling(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dim_sub = df[df["horizon_mode"] == 48]
    for method in ["BE", "Gen2"]:
        ss = dim_sub[dim_sub["method"] == method]
        axes[0].plot(sorted(ss["dimension"].unique()), [ss[ss["dimension"] == d]["runtime_sec"].mean() for d in sorted(ss["dimension"].unique())], marker="o", label=method)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("State dimension")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Dimension scaling")
    axes[0].grid(alpha=0.25)
    hor_sub = df[df["dimension"] == 12]
    for method in ["BE", "Gen2"]:
        ss = hor_sub[hor_sub["method"] == method]
        axes[1].plot(sorted(ss["horizon_mode"].unique()), [ss[ss["horizon_mode"] == h]["runtime_sec"].mean() for h in sorted(ss["horizon_mode"].unique())], marker="o", label=method)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Horizon steps")
    axes[1].set_title("Horizon scaling")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_plotly_dashboard(df_behavior: pd.DataFrame, df_feature: pd.DataFrame, out_html: Path) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Behavior mismatch", "Feature ablation"])
    for method in ["BE", "Gen2", "MBLinear"]:
        sub = df_behavior[(df_behavior["label"] == "Medium") & (df_behavior["method"] == method)]
        grouped = sub.groupby("behavior_gain_bias")["integrated_rmse"].mean().reset_index()
        fig.add_trace(go.Scatter(x=grouped["behavior_gain_bias"], y=grouped["integrated_rmse"], mode="lines+markers", name=f"Medium-{method}"), row=1, col=1)
    feat_mid = df_feature[df_feature["task_family"] == "Small"]
    for method in ["BE", "Gen2"]:
        grouped = feat_mid[feat_mid["method"] == method].groupby("feature_family")["integrated_rmse"].mean().reset_index()
        fig.add_trace(go.Bar(x=grouped["feature_family"], y=grouped["integrated_rmse"], name=f"Small-{method}"), row=1, col=2)
    fig.update_layout(template="plotly_white", title="Stage-3 sensitivity dashboard")
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="main")
    parser.add_argument("--sections", default="all", help="Comma-separated subset of {dt,budget,mismatch,feature,startup,stability,scaling}")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = common.ensure_dirs(root)
    results = paths["results"]
    figures = paths["figures"]
    interactive = paths["interactive"]

    cfg = PROFILES[args.profile]
    if args.sections == "all":
        sections = ["dt", "budget", "mismatch", "feature", "startup", "stability", "scaling"]
    else:
        sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    mismatch_df = None
    feat_df = None
    if "dt" in sections:
        dt_df = dt_sweep(cfg)
        dt_df.to_csv(results / f"stage3_dt_sweep_{args.profile}.csv", index=False)
        plot_dt_sweep(dt_df, figures / f"stage3_dt_sweep_{args.profile}.pdf")
        print(results / f"stage3_dt_sweep_{args.profile}.csv")
        print(figures / f"stage3_dt_sweep_{args.profile}.pdf")
    if "budget" in sections:
        bn_df = budget_nonstat_sweep(cfg)
        bn_df.to_csv(results / f"stage3_budget_nonstat_{args.profile}.csv", index=False)
        plot_budget_nonstat(bn_df, figures / f"stage3_budget_nonstat_{args.profile}.pdf")
        print(results / f"stage3_budget_nonstat_{args.profile}.csv")
        print(figures / f"stage3_budget_nonstat_{args.profile}.pdf")
    if "mismatch" in sections:
        mismatch_df = behavior_mismatch_sweep(cfg)
        mismatch_df.to_csv(results / f"stage3_behavior_mismatch_{args.profile}.csv", index=False)
        plot_behavior_mismatch(mismatch_df, figures / f"stage3_behavior_mismatch_{args.profile}.pdf")
        print(results / f"stage3_behavior_mismatch_{args.profile}.csv")
        print(figures / f"stage3_behavior_mismatch_{args.profile}.pdf")
    if "feature" in sections:
        feat_df = feature_ablation(cfg)
        feat_df.to_csv(results / f"stage3_feature_ablation_{args.profile}.csv", index=False)
        plot_feature_ablation(feat_df, figures / f"stage3_feature_ablation_{args.profile}.pdf")
        print(results / f"stage3_feature_ablation_{args.profile}.csv")
        print(figures / f"stage3_feature_ablation_{args.profile}.pdf")
    if "startup" in sections:
        startup_df = startup_ablation(cfg)
        startup_df.to_csv(results / f"stage3_startup_ablation_{args.profile}.csv", index=False)
        plot_startup_ablation(startup_df, figures / f"stage3_startup_ablation_{args.profile}.pdf")
        print(results / f"stage3_startup_ablation_{args.profile}.csv")
        print(figures / f"stage3_startup_ablation_{args.profile}.pdf")
    if "stability" in sections:
        stability_df = stability_diagnostics(cfg)
        stability_df.to_csv(results / f"stage3_stability_{args.profile}.csv", index=False)
        plot_stability(stability_df, figures / f"stage3_stability_{args.profile}.pdf")
        print(results / f"stage3_stability_{args.profile}.csv")
        print(figures / f"stage3_stability_{args.profile}.pdf")
    if "scaling" in sections:
        scaling_df = runtime_scaling(cfg)
        scaling_df.to_csv(results / f"stage3_runtime_scaling_{args.profile}.csv", index=False)
        plot_runtime_scaling(scaling_df, figures / f"stage3_runtime_scaling_{args.profile}.pdf")
        print(results / f"stage3_runtime_scaling_{args.profile}.csv")
        print(figures / f"stage3_runtime_scaling_{args.profile}.pdf")

    if mismatch_df is not None and feat_df is not None:
        write_plotly_dashboard(mismatch_df, feat_df, interactive / f"stage3_sensitivity_dashboard_{args.profile}.html")


if __name__ == "__main__":
    main()
