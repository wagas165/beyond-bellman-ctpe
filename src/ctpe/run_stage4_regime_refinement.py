from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common
from ctpe import run_stage3_sensitivity_suite as sens


PROFILES = {
    "smoke": {"episodes_small": 160, "episodes_medium": 220, "seeds_small": 2, "seeds_medium": 2},
    "main": {"episodes_small": 640, "episodes_medium": 760, "seeds_small": 12, "seeds_medium": 12},
    "heavy": {"episodes_small": 960, "episodes_medium": 1120, "seeds_small": 20, "seeds_medium": 20},
}
EXTENDED_BW = [0, 1, 2, 4, 6, 8, 12, 16, 24]


def _run_task(task, seed: int, orders=(None, 1, 2, 3)) -> list[dict]:
    rng = np.random.default_rng(seed)
    states, rewards = base.rollout_episodes(task, task.n_episodes, rng, policy_kind="behavior")
    splits = base.split_episodes(states, rewards)
    train_m = base.precompute_moments(task, *splits["train"], max_order=3)
    val_m = base.precompute_moments(task, *splits["val"], max_order=3)
    eval_bundle = common.safe_prepare_eval_bundle(task, splits["test"][0], seed + 777, eval_indices=task.eval_indices)
    rows = []
    for order in orders:
        best = None
        best_score = float("inf")
        for bw in task.bandwidth_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=bw)
            row = {
                "method": "BE" if order is None else f"Gen{order}",
                "order": -1 if order is None else order,
                "bandwidth_steps": int(bw),
                "validation_score": float(payload["validation_score"]),
                "t0_rmse": float(payload["t0_rmse"]),
                "integrated_rmse": float(payload["integrated_rmse"]),
                "runtime_sec": float(payload["runtime_sec"]),
            }
            if row["validation_score"] < best_score:
                best_score = row["validation_score"]
                best = row
        assert best is not None
        rows.append(best)
    return rows


def dt_refinement(cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    T = 3.36
    dt_grid = [0.12, 0.06] if cfg["seeds_small"] <= 2 else [0.12, 0.08, 0.06, 0.04, 0.03, 0.02]
    for family in ["Small", "Medium"]:
        for dt in dt_grid:
            horizon = max(32, int(round(T / dt)))
            if family == "Small":
                task = common.make_small_heavy_task(n_episodes=cfg["episodes_small"], dt=dt, horizon_steps=horizon, mc_rollouts=72)
            else:
                task = common.make_medium_heavy_task(n_episodes=cfg["episodes_medium"], dt=dt, horizon_steps=horizon, mc_rollouts=72)
            task = common.replace(task, bandwidth_grid=EXTENDED_BW) if hasattr(common, 'replace') else task
            task = task.__class__(**{**task.__dict__, "bandwidth_grid": EXTENDED_BW})
            seed_count = cfg["seeds_small"] if family == "Small" else cfg["seeds_medium"]
            for seed in range(seed_count):
                selected = _run_task(task, 20271000 + 1000 * (family == "Medium") + int(100 * dt) + seed)
                for row in selected:
                    rows.append({**row, "family": family, "dt": dt, "horizon_steps": horizon})
    return pd.DataFrame(rows)


def budget_nonstat_refinement(cfg: dict) -> pd.DataFrame:
    rows: List[dict] = []
    episode_grid = [160, 320] if cfg["seeds_small"] <= 2 else [160, 320, 640, 960, 1440]
    amp_grid = [0.0, 1.0, 2.0] if cfg["seeds_small"] <= 2 else [0.0, 0.35, 0.7, 1.0, 1.5, 2.0, 2.5]
    for episodes in episode_grid:
        for amp in amp_grid:
            task = common.make_small_heavy_task(n_episodes=episodes, nonstat_scale=amp, mc_rollouts=64)
            task = task.__class__(**{**task.__dict__, "bandwidth_grid": EXTENDED_BW})
            for seed in range(cfg["seeds_small"]):
                selected = _run_task(task, 20272000 + episodes + int(100 * amp) + seed, orders=(None, 2))
                be = next(r for r in selected if r["method"] == "BE")
                gen = next(r for r in selected if r["method"] == "Gen2")
                rows.append({
                    "episodes": episodes,
                    "nonstat_scale": amp,
                    "seed": seed,
                    "be_integrated_rmse": be["integrated_rmse"],
                    "gen2_integrated_rmse": gen["integrated_rmse"],
                    "relative_gain": (be["integrated_rmse"] - gen["integrated_rmse"]) / max(be["integrated_rmse"], 1e-12),
                    "be_bandwidth": int(be["bandwidth_steps"]),
                    "gen2_bandwidth": int(gen["bandwidth_steps"]),
                })
    return pd.DataFrame(rows)


def plot_dt(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, family in zip(axes, ["Small", "Medium"]):
        sub = df[df["family"] == family]
        for method in ["BE", "Gen1", "Gen2", "Gen3"]:
            sm = sub[sub["method"] == method]
            xs = sorted(sm["dt"].unique())
            means = [float(sm[sm["dt"] == x]["integrated_rmse"].mean()) for x in xs]
            ax.plot(xs, means, marker="o", label=method)
        ax.set_title(family)
        ax.set_xlabel("dt")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("integrated RMSE")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_budget(df: pd.DataFrame, out_pdf: Path) -> None:
    grouped = df.groupby(["episodes", "nonstat_scale"], as_index=False)["relative_gain"].mean()
    episodes = sorted(grouped["episodes"].unique())
    amps = sorted(grouped["nonstat_scale"].unique())
    heat = np.zeros((len(amps), len(episodes)))
    for i, a in enumerate(amps):
        for j, e in enumerate(episodes):
            heat[i, j] = float(grouped[(grouped["episodes"] == e) & (grouped["nonstat_scale"] == a)]["relative_gain"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(heat, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(episodes)), labels=[str(e) for e in episodes])
    ax.set_yticks(np.arange(len(amps)), labels=[f"{a:.2f}" for a in amps])
    ax.set_xlabel("logged episodes")
    ax.set_ylabel("nonstationarity scale")
    ax.set_title("Extended regime-refinement heat map")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("relative RMSE gain")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-4 regime refinement with expanded bandwidth grids.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    figures = root / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    dt_df = dt_refinement(cfg)
    dt_csv = results / f"stage4_regime_dt_refinement_{args.profile}.csv"
    dt_df.to_csv(dt_csv, index=False)
    plot_dt(dt_df, figures / f"stage4_regime_dt_refinement_{args.profile}.pdf")

    bn_df = budget_nonstat_refinement(cfg)
    bn_csv = results / f"stage4_regime_budget_nonstat_{args.profile}.csv"
    bn_df.to_csv(bn_csv, index=False)
    plot_budget(bn_df, figures / f"stage4_regime_budget_nonstat_{args.profile}.pdf")

    print(dt_csv)
    print(bn_csv)


if __name__ == "__main__":
    main()
