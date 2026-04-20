from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INTERACTIVE = FIGURES / "interactive"


def _ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    INTERACTIVE.mkdir(parents=True, exist_ok=True)


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, 1.96 * float(se)


def regime_map_and_heatmap() -> None:
    df = pd.read_csv(RESULTS / "stage3_budget_nonstat_heavy.csv")
    grouped = (
        df.groupby(["episodes", "nonstat_scale"], as_index=False)["relative_gain"]
        .mean()
        .sort_values(["nonstat_scale", "episodes"])
    )
    episodes = sorted(grouped["episodes"].unique())
    amps = sorted(grouped["nonstat_scale"].unique())
    heat = np.zeros((len(amps), len(episodes)))
    for i, amp in enumerate(amps):
        for j, ep in enumerate(episodes):
            heat[i, j] = float(grouped[(grouped["episodes"] == ep) & (grouped["nonstat_scale"] == amp)]["relative_gain"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)

    # Left panel: schematic regime map.
    ax = axes[0]
    x = np.logspace(-2.0, -0.7, 400)
    y1 = x
    y2 = x ** 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.fill_between(x, y1, 5e-1, alpha=0.12)
    ax.fill_between(x, y2, y1, alpha=0.18)
    ax.fill_between(x, 1e-4, y2, alpha=0.10)
    ax.plot(x, y1, linestyle="--", linewidth=1.6)
    ax.plot(x, y2, linestyle=":", linewidth=1.8)
    ax.text(0.018, 0.14, "nonstationarity floor dominates\nvisible order gain is weak", fontsize=9)
    ax.text(0.030, 0.010, "Gen2 sweet spot:\nBE still first-order,\nGen3 not reliably visible", fontsize=9)
    ax.text(0.055, 6e-4, "Gen3 can appear if\nmultistep variance stays controlled", fontsize=9)
    ax.set_xlim(1e-2, 2e-1)
    ax.set_ylim(1e-4, 3e-1)
    ax.set_xlabel(r"decision interval $\Delta t$")
    ax.set_ylabel(r"optimized nonstationarity floor $F_{\mathrm{ns}}$")
    ax.set_title("Theoretical regime map")
    ax.text(0.012, 0.19, r"$F_{\mathrm{ns}} \approx \Delta t$", fontsize=8)
    ax.text(0.011, 1.8e-3, r"$F_{\mathrm{ns}} \approx \Delta t^2$", fontsize=8)
    ax.grid(alpha=0.15, which="both")

    # Right panel: empirical heavy-suite heatmap.
    ax = axes[1]
    im = ax.imshow(heat, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(episodes)), labels=[str(ep) for ep in episodes])
    ax.set_yticks(np.arange(len(amps)), labels=[f"{amp:.1f}" for amp in amps])
    ax.set_xlabel("logged episodes")
    ax.set_ylabel("nonstationarity scale")
    ax.set_title("Empirical Gen2 gain over BE")
    for i, amp in enumerate(amps):
        for j, ep in enumerate(episodes):
            ax.text(j, i, f"{100*heat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("relative RMSE gain")

    out_pdf = FIGURES / "stage4_regime_map.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)

    # Interactive version.
    figi = make_subplots(rows=1, cols=2, subplot_titles=["Theoretical regime map", "Empirical Gen2 gain over BE"], horizontal_spacing=0.12)
    figi.add_trace(go.Scatter(x=x, y=y1, mode="lines", name="F_ns = Dt", line=dict(dash="dash")), row=1, col=1)
    figi.add_trace(go.Scatter(x=x, y=y2, mode="lines", name="F_ns = Dt^2", line=dict(dash="dot")), row=1, col=1)
    figi.update_xaxes(type="log", title_text="decision interval Dt", row=1, col=1)
    figi.update_yaxes(type="log", title_text="optimized floor F_ns", row=1, col=1)
    figi.add_trace(
        go.Heatmap(
            z=heat,
            x=[str(ep) for ep in episodes],
            y=[f"{amp:.1f}" for amp in amps],
            colorbar=dict(title="relative gain"),
            name="gain",
        ),
        row=1,
        col=2,
    )
    figi.update_xaxes(title_text="logged episodes", row=1, col=2)
    figi.update_yaxes(title_text="nonstationarity scale", row=1, col=2)
    figi.update_layout(template="plotly_white", title="Stage-4 regime-map view")
    figi.write_html(INTERACTIVE / "stage4_regime_map.html", include_plotlyjs="cdn")



def behavior_mismatch_panels() -> None:
    df = pd.read_csv(RESULTS / "stage3_behavior_mismatch_heavy.csv")
    df = df[df["method"].isin(["BE", "Gen2"])]
    tasks = ["Small", "Medium", "Large"]
    gains = sorted(df["behavior_gain_bias"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.7), sharex=True)
    for col, task in enumerate(tasks):
        ss = df[df["label"] == task]
        for method in ["BE", "Gen2"]:
            sm = ss[ss["method"] == method]
            mean_int, ci_int, mean_t0, ci_t0 = [], [], [], []
            for g in gains:
                gg = sm[sm["behavior_gain_bias"] == g]
                m, c = _mean_ci(gg["integrated_rmse"].to_numpy())
                mean_int.append(m)
                ci_int.append(c)
                m0, c0 = _mean_ci(gg["t0_rmse"].to_numpy())
                mean_t0.append(m0)
                ci_t0.append(c0)
            axes[0, col].plot(gains, mean_int, marker="o", label=method)
            axes[0, col].fill_between(gains, np.array(mean_int) - np.array(ci_int), np.array(mean_int) + np.array(ci_int), alpha=0.15)
            axes[1, col].plot(gains, mean_t0, marker="o", label=method)
            axes[1, col].fill_between(gains, np.array(mean_t0) - np.array(ci_t0), np.array(mean_t0) + np.array(ci_t0), alpha=0.15)
        axes[0, col].set_title(task)
        axes[0, col].grid(alpha=0.15)
        axes[1, col].grid(alpha=0.15)
        axes[1, col].set_xlabel("control-gain mismatch level")
    axes[0, 0].set_ylabel("integrated RMSE")
    axes[1, 0].set_ylabel(r"$t=0$ RMSE")
    axes[0, 2].legend(loc="upper left")
    fig.suptitle("Near-off-policy stress test under gain-shift mismatch", y=0.99)
    out_pdf = FIGURES / "stage4_behavior_mismatch_main.pdf"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_pdf)
    plt.close(fig)

    # Gap-vs-variability diagnostic.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, task in zip(axes, tasks):
        sub = df[df["label"] == task]
        gaps, stds = [], []
        for g in gains:
            be = sub[(sub["method"] == "BE") & (sub["behavior_gain_bias"] == g)]["integrated_rmse"].to_numpy()
            gen = sub[(sub["method"] == "Gen2") & (sub["behavior_gain_bias"] == g)]["integrated_rmse"].to_numpy()
            gap = np.mean(be - gen)
            std = np.std(be - gen, ddof=1) if len(be) > 1 else 0.0
            gaps.append(gap)
            stds.append(std)
        ax.plot(gains, gaps, marker="o")
        ax.fill_between(gains, np.array(gaps) - np.array(stds), np.array(gaps) + np.array(stds), alpha=0.15)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(task)
        ax.set_xlabel("control-gain mismatch level")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("mean error gap (BE - Gen2)")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4_behavior_mismatch_gapstd.pdf")
    plt.close(fig)

    # Pooling-sensitivity heatmap: selected Gen2 bandwidth frequencies.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), constrained_layout=True)
    bw_grid = sorted(df[df["method"] == "Gen2"]["bandwidth_steps"].unique())
    for ax, task in zip(axes, tasks):
        sub = df[(df["label"] == task) & (df["method"] == "Gen2")]
        mat = np.zeros((len(bw_grid), len(gains)))
        for i, bw in enumerate(bw_grid):
            for j, g in enumerate(gains):
                sg = sub[sub["behavior_gain_bias"] == g]
                if len(sg) == 0:
                    continue
                mat[i, j] = np.mean(sg["bandwidth_steps"].to_numpy() == bw)
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(gains)), labels=[f"{g:.2f}" for g in gains])
        ax.set_yticks(np.arange(len(bw_grid)), labels=[str(int(bw)) for bw in bw_grid])
        ax.set_title(task)
        ax.set_xlabel("mismatch level")
        if task == "Small":
            ax.set_ylabel("selected Gen2 bandwidth")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("selection frequency")
    fig.savefig(FIGURES / "stage4_behavior_mismatch_pooling.pdf")
    plt.close(fig)

    # Interactive dashboard.
    dash = make_subplots(rows=1, cols=2, subplot_titles=["Integrated RMSE", "T0 RMSE"], horizontal_spacing=0.12)
    for task in tasks:
        for method in ["BE", "Gen2"]:
            sub = df[(df["label"] == task) & (df["method"] == method)]
            xs, ys_i, ys_t0 = [], [], []
            for g in gains:
                sg = sub[sub["behavior_gain_bias"] == g]
                xs.append(g)
                ys_i.append(float(sg["integrated_rmse"].mean()))
                ys_t0.append(float(sg["t0_rmse"].mean()))
            dash.add_trace(go.Scatter(x=xs, y=ys_i, mode="lines+markers", name=f"{task}-{method}"), row=1, col=1)
            dash.add_trace(go.Scatter(x=xs, y=ys_t0, mode="lines+markers", name=f"{task}-{method}", showlegend=False), row=1, col=2)
    dash.update_xaxes(title_text="control-gain mismatch level", row=1, col=1)
    dash.update_xaxes(title_text="control-gain mismatch level", row=1, col=2)
    dash.update_yaxes(title_text="integrated RMSE", row=1, col=1)
    dash.update_yaxes(title_text="t0 RMSE", row=1, col=2)
    dash.update_layout(template="plotly_white", title="Stage-4 near-off-policy stress diagnostics")
    dash.write_html(INTERACTIVE / "stage4_behavior_mismatch_dashboard.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    _ensure_dirs()
    regime_map_and_heatmap()
    behavior_mismatch_panels()
    print(FIGURES / "stage4_regime_map.pdf")
    print(FIGURES / "stage4_behavior_mismatch_main.pdf")
    print(FIGURES / "stage4_behavior_mismatch_gapstd.pdf")
    print(FIGURES / "stage4_behavior_mismatch_pooling.pdf")
