from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ctpe import run_stage2_extended_suite as base
from ctpe import stage3_heavy_common as common
from ctpe import stage5_common as s5


PROFILES = {
    "smoke": {
        "dt_grid": [0.12, 0.08],
        "kappa_grid": [0.10, 0.20],
        "mult_first": [0.5, 1.0],
        "mult_second": [0.5, 1.0],
        "seeds_ou": 1,
        "seeds_lqr": 1,
        "bw_time_grid": [0.0, 0.08, 0.16, 0.24],
    },
    "main": {
        "dt_grid": [0.12, 0.10, 0.08, 0.06],
        "kappa_grid": [0.05, 0.10, 0.20, 0.40],
        "mult_first": [0.25, 0.5, 1.0, 2.0],
        "mult_second": [0.25, 0.5, 1.0, 2.0],
        "seeds_ou": 10,
        "seeds_lqr": 16,
        "bw_time_grid": [0.0, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64, 0.96],
    },
    "heavy": {
        "dt_grid": [0.12, 0.10, 0.08, 0.06],
        "kappa_grid": [0.05, 0.10, 0.20, 0.40],
        "mult_first": [0.25, 0.5, 1.0, 2.0, 4.0],
        "mult_second": [0.25, 0.5, 1.0, 2.0],
        "seeds_ou": 40,
        "seeds_lqr": 30,
        "bw_time_grid": [0.0, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64, 0.96, 1.28],
    },
}

METHODS = [(None, "BE"), (2, "Gen2"), (3, "Gen3")]
HORIZON = 3.36
# Normalization constants used to target manageable but informative budgets.
C_FIRST = 8.0
C_SECOND = 2e-4


def _episodes_from_boundary(kappa: float, dt: float, multiplier: float, power: int) -> int:
    const = C_FIRST if power == 2 else C_SECOND
    val = multiplier * const * kappa / (dt ** power)
    return int(np.clip(round(val), 80, 2400))


def _family_task(family: str, n_episodes: int, dt: float, kappa: float) -> base.TaskSpec:
    horizon_steps = max(24, int(round(HORIZON / dt)))
    if family == "OU-1D":
        return s5.make_ou1d_task(n_episodes=n_episodes, dt=dt, horizon_steps=horizon_steps, kappa=kappa, mc_rollouts=96)
    if family == "TV-LQR-2D":
        return s5.make_tvlqr2d_task(n_episodes=n_episodes, dt=dt, horizon_steps=horizon_steps, kappa=kappa, mc_rollouts=96)
    raise ValueError(f"Unknown family: {family}")


def _run_setting(task: base.TaskSpec, seed: int, bw_time_grid: Sequence[float]) -> list[dict]:
    rng = np.random.default_rng(seed)
    states, rewards = base.rollout_episodes(task, task.n_episodes, rng, policy_kind="behavior")
    train_states, train_rewards = base.split_episodes(states, rewards)["train"]
    val_states, val_rewards = base.split_episodes(states, rewards)["val"]
    test_states, _ = base.split_episodes(states, rewards)["test"]
    train_m = base.precompute_moments(task, train_states, train_rewards, max_order=3)
    val_m = base.precompute_moments(task, val_states, val_rewards, max_order=3)
    eval_bundle = common.safe_prepare_eval_bundle(task, test_states, seed + 909, eval_indices=task.eval_indices)
    bw_steps_grid = s5.bw_time_grid_to_steps(task.dt, bw_time_grid)

    rows = []
    for order, method in METHODS:
        best = None
        best_score = float("inf")
        best_bw = None
        for bw in bw_steps_grid:
            payload = base.fit_method(task, train_m, val_m, eval_bundle, order=order, bandwidth_steps=int(bw))
            row = {
                "method": method,
                "order": -1 if order is None else int(order),
                "bandwidth_steps": int(bw),
                "bandwidth_time": float(bw * task.dt),
                "validation_score": float(payload["validation_score"]),
                "t0_rmse": float(payload["t0_rmse"]),
                "integrated_rmse": float(payload["integrated_rmse"]),
            }
            if row["validation_score"] < best_score:
                best = row
                best_score = row["validation_score"]
                best_bw = int(bw)
        assert best is not None
        best = dict(best)
        best["boundary_hit"] = int(best_bw == max(bw_steps_grid))
        rows.append(best)
    return rows


def _aggregate_hits(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["suite", "family", "dt", "kappa", "multiplier", "boundary_power", "method"]
    return (
        df.groupby(grp_cols, as_index=False)
        .agg(
            boundary_hit_rate=("boundary_hit", "mean"),
            mean_bandwidth_time=("bandwidth_time", "mean"),
            mean_rmse=("integrated_rmse", "mean"),
        )
    )


def _winner_map(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["suite", "family", "dt", "kappa", "multiplier", "boundary_power"]
    rows = []
    for keys, sub in df.groupby(grp_cols):
        means = sub.groupby("method", as_index=False)["integrated_rmse"].mean()
        winner = means.loc[means["integrated_rmse"].idxmin(), "method"]
        hit = sub.groupby("method", as_index=False)["boundary_hit"].mean()
        rows.append({
            **dict(zip(grp_cols, keys)),
            "winner": winner,
            "be_mean": float(means.loc[means["method"] == "BE", "integrated_rmse"].iloc[0]),
            "gen2_mean": float(means.loc[means["method"] == "Gen2", "integrated_rmse"].iloc[0]),
            "gen3_mean": float(means.loc[means["method"] == "Gen3", "integrated_rmse"].iloc[0]),
            "gen2_hit_rate": float(hit.loc[hit["method"] == "Gen2", "boundary_hit"].iloc[0]),
            "gen3_hit_rate": float(hit.loc[hit["method"] == "Gen3", "boundary_hit"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _plot_suite(df: pd.DataFrame, suite: str, out_pdf: Path) -> None:
    families = ["OU-1D", "TV-LQR-2D"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    sub = df[df["suite"] == suite]
    for ax, family in zip(axes, families):
        ss = sub[(sub["family"] == family) & (sub["method"] == "Gen2")]
        xvals = sorted(ss["multiplier"].unique())
        yvals = sorted(ss["dt"].unique()) if suite == "dtM" else sorted(ss["kappa"].unique())
        mat = np.zeros((len(yvals), len(xvals)))
        for i, y in enumerate(yvals):
            for j, x in enumerate(xvals):
                if suite == "dtM":
                    cell = ss[(np.isclose(ss["dt"], y)) & (np.isclose(ss["multiplier"], x))]
                else:
                    cell = ss[(np.isclose(ss["kappa"], y)) & (np.isclose(ss["multiplier"], x))]
                if len(cell):
                    mat[i, j] = float((cell["be_rmse_ref"] - cell["integrated_rmse"]).mean()) if "be_rmse_ref" in cell else float(cell["integrated_rmse"].mean())
        im = ax.imshow(mat, origin="lower", aspect="auto")
        ax.set_title(family)
        ax.set_xticks(np.arange(len(xvals)), labels=[f"{x:.2f}" for x in xvals])
        ax.set_xlabel("boundary multiplier")
        if suite == "dtM":
            ax.set_yticks(np.arange(len(yvals)), labels=[f"{y:.2f}" for y in yvals])
            ax.set_ylabel("dt")
        else:
            ax.set_yticks(np.arange(len(yvals)), labels=[f"{y:.2f}" for y in yvals])
            ax.set_ylabel("kappa")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Gen2 mean RMSE")
    fig.suptitle(f"Stage-5 clean regime {suite} sweep", y=1.02)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_winners(winner_df: pd.DataFrame, out_pdf: Path) -> None:
    families = ["OU-1D", "TV-LQR-2D"]
    suite_names = ["dtM", "kappaM"]
    method_to_code = {"BE": 0, "Gen2": 1, "Gen3": 2}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    for row, suite in enumerate(suite_names):
        for col, family in enumerate(families):
            sub = winner_df[(winner_df["suite"] == suite) & (winner_df["family"] == family)]
            xvals = sorted(sub["multiplier"].unique())
            yvals = sorted(sub["dt"].unique()) if suite == "dtM" else sorted(sub["kappa"].unique())
            mat = np.zeros((len(yvals), len(xvals)))
            for i, y in enumerate(yvals):
                for j, x in enumerate(xvals):
                    if suite == "dtM":
                        cell = sub[(np.isclose(sub["dt"], y)) & (np.isclose(sub["multiplier"], x))]
                    else:
                        cell = sub[(np.isclose(sub["kappa"], y)) & (np.isclose(sub["multiplier"], x))]
                    if len(cell):
                        mat[i, j] = method_to_code[str(cell["winner"].iloc[0])]
            ax = axes[row, col]
            im = ax.imshow(mat, origin="lower", aspect="auto", vmin=0, vmax=2)
            ax.set_title(f"{suite} / {family}")
            ax.set_xticks(np.arange(len(xvals)), labels=[f"{x:.2f}" for x in xvals])
            ax.set_xlabel("boundary multiplier")
            ax.set_yticks(np.arange(len(yvals)), labels=[f"{y:.2f}" for y in yvals])
            ax.set_ylabel("dt" if suite == "dtM" else "kappa")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1, 2], labels=["BE", "Gen2", "Gen3"])
    cbar.set_label("winner")
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_hits(hits_df: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for method in ["BE", "Gen2", "Gen3"]:
        sub = hits_df[hits_df["method"] == method]
        ax.plot(np.arange(len(sub)), sub["boundary_hit_rate"], marker="o", label=method)
    ax.set_xlabel("setting index")
    ax.set_ylabel("boundary-hit rate")
    ax.set_title("Stage-5 clean regime bandwidth boundary hits")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_dashboard(dt_df: pd.DataFrame, kappa_df: pd.DataFrame, out_html: Path) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["dtM suite", "kappaM suite"], horizontal_spacing=0.12)
    for method in ["BE", "Gen2", "Gen3"]:
        sub = dt_df[dt_df["method"] == method]
        grp = sub.groupby("multiplier", as_index=False)["integrated_rmse"].mean()
        fig.add_trace(go.Scatter(x=grp["multiplier"], y=grp["integrated_rmse"], mode="lines+markers", name=f"dtM-{method}"), row=1, col=1)
        sub = kappa_df[kappa_df["method"] == method]
        grp = sub.groupby("multiplier", as_index=False)["integrated_rmse"].mean()
        fig.add_trace(go.Scatter(x=grp["multiplier"], y=grp["integrated_rmse"], mode="lines+markers", name=f"kappaM-{method}", showlegend=False), row=1, col=2)
    fig.update_layout(template="plotly_white", title="Stage-5 clean regime dashboard")
    fig.update_xaxes(title_text="boundary multiplier", row=1, col=1)
    fig.update_xaxes(title_text="boundary multiplier", row=1, col=2)
    fig.update_yaxes(title_text="integrated RMSE", row=1, col=1)
    fig.update_yaxes(title_text="integrated RMSE", row=1, col=2)
    fig.write_html(out_html, include_plotlyjs="cdn")


def make_table(winner_df: pd.DataFrame, out_path: Path) -> None:
    clean = winner_df[winner_df["gen2_hit_rate"] <= 0.10]
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stage-5 clean-regime winner map summary. Cells with Gen2 boundary-hit rate at most 0.10 are reported.}",
        r"\label{tab:stage5_clean_regime}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Suite & Family & Multiplier & Winner & Gen2 hit rate & Gen3 hit rate \\",
        r"\midrule",
    ]
    for _, r in clean.head(24).iterrows():
        rows.append(f"{r['suite']} & {r['family']} & {r['multiplier']:.2f} & {r['winner']} & {r['gen2_hit_rate']:.2f} & {r['gen3_hit_rate']:.2f} \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    s5.save_latex_table(rows, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-5 clean asymptotic regime suite.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="heavy")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    dt_rows: List[dict] = []
    kappa_rows: List[dict] = []

    # dt--M sweep at fixed kappa.
    fixed_kappa = 0.20
    for family, n_seeds in [("OU-1D", cfg["seeds_ou"]), ("TV-LQR-2D", cfg["seeds_lqr"] )]:
        for dt in cfg["dt_grid"]:
            for power, multipliers in [(2, cfg["mult_first"]), (5, cfg["mult_second"] )]:
                for mult in multipliers:
                    episodes = _episodes_from_boundary(fixed_kappa, float(dt), float(mult), power)
                    task = _family_task(family, episodes, float(dt), fixed_kappa)
                    task = replace(task, bandwidth_grid=s5.bw_time_grid_to_steps(task.dt, cfg["bw_time_grid"]))
                    for i in range(n_seeds):
                        rows = _run_setting(task, 20286000 + (0 if family == "OU-1D" else 10000) + int(1000 * dt) + int(100 * mult) + power * 100000 + i, cfg["bw_time_grid"])
                        for row in rows:
                            dt_rows.append({
                                "suite": "dtM",
                                "family": family,
                                "seed": i,
                                "dt": float(dt),
                                "kappa": fixed_kappa,
                                "multiplier": float(mult),
                                "boundary_power": int(power),
                                "n_episodes": int(episodes),
                                "rho_first": float(episodes * (dt ** 2) / fixed_kappa),
                                "rho_second": float(episodes * (dt ** 5) / fixed_kappa),
                                **row,
                            })

    # kappa--M sweep at fixed dt.
    fixed_dt = 0.08
    for family, n_seeds in [("OU-1D", cfg["seeds_ou"]), ("TV-LQR-2D", cfg["seeds_lqr"] )]:
        for kappa in cfg["kappa_grid"]:
            for power, multipliers in [(2, cfg["mult_first"]), (5, cfg["mult_second"] )]:
                for mult in multipliers:
                    episodes = _episodes_from_boundary(float(kappa), fixed_dt, float(mult), power)
                    task = _family_task(family, episodes, fixed_dt, float(kappa))
                    task = replace(task, bandwidth_grid=s5.bw_time_grid_to_steps(task.dt, cfg["bw_time_grid"]))
                    for i in range(n_seeds):
                        rows = _run_setting(task, 20296000 + (0 if family == "OU-1D" else 10000) + int(1000 * kappa) + int(100 * mult) + power * 100000 + i, cfg["bw_time_grid"])
                        for row in rows:
                            kappa_rows.append({
                                "suite": "kappaM",
                                "family": family,
                                "seed": i,
                                "dt": fixed_dt,
                                "kappa": float(kappa),
                                "multiplier": float(mult),
                                "boundary_power": int(power),
                                "n_episodes": int(episodes),
                                "rho_first": float(episodes * (fixed_dt ** 2) / float(kappa)),
                                "rho_second": float(episodes * (fixed_dt ** 5) / float(kappa)),
                                **row,
                            })

    dt_df = pd.DataFrame(dt_rows)
    kappa_df = pd.DataFrame(kappa_rows)
    all_df = pd.concat([dt_df, kappa_df], ignore_index=True)
    hits_df = _aggregate_hits(all_df)
    winner_df = _winner_map(all_df)

    root = Path(__file__).resolve().parents[2]
    paths = common.ensure_dirs(root)
    results = paths["results"]
    figures = paths["figures"]
    interactive = paths["interactive"]
    tables = paths["tables"]

    dt_df.to_csv(results / f"stage5_clean_regime_dtM_{args.profile}.csv", index=False)
    kappa_df.to_csv(results / f"stage5_clean_regime_kappaM_{args.profile}.csv", index=False)
    winner_df.to_csv(results / f"stage5_clean_regime_winner_map_{args.profile}.csv", index=False)
    hits_df.to_csv(results / f"stage5_clean_regime_bandwidth_hits_{args.profile}.csv", index=False)

    _plot_suite(dt_df, "dtM", figures / f"stage5_clean_regime_dtM_{args.profile}.pdf")
    _plot_suite(kappa_df, "kappaM", figures / f"stage5_clean_regime_kappaM_{args.profile}.pdf")
    _plot_winners(winner_df, figures / f"stage5_clean_regime_winner_map_{args.profile}.pdf")
    _plot_hits(hits_df, figures / f"stage5_clean_regime_bandwidth_hits_{args.profile}.pdf")
    write_dashboard(dt_df, kappa_df, interactive / f"stage5_clean_regime_dashboard_{args.profile}.html")
    make_table(winner_df, tables / f"stage5_clean_regime_main_{args.profile}.tex")

    print(results / f"stage5_clean_regime_dtM_{args.profile}.csv")
    print(results / f"stage5_clean_regime_kappaM_{args.profile}.csv")


if __name__ == "__main__":
    main()
