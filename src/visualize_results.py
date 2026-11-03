"""Create presentation-ready plots from the saved real-instance model results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT, PROC, scenario, DEMAND_SCENARIOS
from geo import to_xy
import params

COLORS = {"low": "#4C78A8", "base": "#F58518", "high": "#54A24B"}


def _xy(frame):
    x, y = to_xy(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    return x / 1000, y / 1000


def plot_site_maps(ins, result_dir: Path, figures_dir: Path, scenarios: list[str]) -> Path:
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.1 * len(scenarios), 8), squeeze=False)
    tr_x, tr_y = _xy(ins.tracts)
    c_x, c_y = _xy(ins.cands)
    lk_x, lk_y = _xy(ins.luckin)
    cp_x, cp_y = _xy(ins.comps)
    for ax, name in zip(axes[0], scenarios):
        ax.scatter(c_x, c_y, s=5, c="#D9DDE3", alpha=0.45, linewidths=0, label="Vacant candidate sites")
        ax.scatter(cp_x, cp_y, s=9, c="#D55E5E", alpha=0.42, linewidths=0, label="Coffee/tea competitors")
        ax.scatter(lk_x, lk_y, s=38, marker="*", c="#2463A6", edgecolors="white", linewidths=0.35,
                   label="Luckin records (unverified status)", zorder=4)
        plan = pd.read_csv(result_dir / f"plan_{name}.csv")
        if not plan.empty:
            p_x, p_y = _xy(plan)
            ax.scatter(p_x, p_y, s=90, marker="P", c=COLORS[name], edgecolors="black", linewidths=0.65,
                       label=f"Selected sites ({name})", zorder=5)
        ax.set_title(f"{name.title()} demand scenario")
        ax.set_xlabel("East / west from Manhattan center (km)")
        ax.set_ylabel("North / south from Manhattan center (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Candidate locations and recommended openings\nManhattan (schematic coordinates; no street basemap)", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = figures_dir / "site_selection_maps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scenario_dashboard(report: dict, out_dir: Path, scenarios: list[str]) -> Path:
    names = [name for name in scenarios if name in report]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    positions = np.arange(len(names))
    colors = [COLORS[name] for name in names]
    npv = [report[name]["Z"] / 1e6 for name in names]
    axes[0].bar(positions, npv, color=colors)
    axes[0].set_xticks(positions, [n.title() for n in names])
    axes[0].set_ylabel("10-year incremental NPV ($ millions)")
    axes[0].set_title("Objective value")
    for i, value in enumerate(npv):
        axes[0].text(i, value, f"${value:.1f}M", ha="center", va="bottom", fontsize=9)

    cannibalized = np.array([report[name]["lost_incumbent_orders"] for name in names])
    external = np.array([report[name]["from_competitors"] + report[name]["from_no_purchase"] for name in names])
    axes[1].bar(positions, external, color="#59A14F", label="From competitors + no-purchase")
    axes[1].bar(positions, cannibalized, bottom=external, color="#E15759", label="Cannibalized incumbent orders")
    axes[1].set_xticks(positions, [n.title() for n in names])
    axes[1].set_ylabel("Orders per day (model estimates)")
    axes[1].set_title("Source of new-store demand")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(0.02, 0.98, "Gross new-store orders shown separately in site table",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=8)

    counts = [report[name]["n_open"] for name in names]
    axes[2].bar(positions, counts, color=colors)
    axes[2].set_xticks(positions, [n.title() for n in names])
    axes[2].set_ylabel("Number of selected candidate sites")
    axes[2].set_title("Network expansion")
    axes[2].set_ylim(0, max(10, max(counts, default=0) + 1))
    for i, name in enumerate(names):
        retention = report[name]["min_retention"]
        axes[2].text(i, counts[i] + 0.15, f"min. retention {retention:.1%}",
                     ha="center", va="bottom", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Real-data scenario comparison\nNPV is conditional on model assumptions and provisional store records", y=1.02)
    fig.tight_layout()
    path = out_dir / "scenario_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_demand_index(ins, out_dir: Path) -> Path:
    x, y = _xy(ins.tracts)
    fig, ax = plt.subplots(figsize=(8, 9))
    points = ax.scatter(x, y, c=ins.h, cmap="viridis", s=23, edgecolors="white", linewidths=0.25)
    colorbar = fig.colorbar(points, ax=ax, shrink=0.74)
    colorbar.set_label("Composite demand-potential index (relative quintile score)")
    ax.set_title("Manhattan census-tract demand proxy\nEqual-weight average of population, jobs, and subway-entry tiers")
    ax.set_xlabel("East / west from Manhattan center (km)")
    ax.set_ylabel("North / south from Manhattan center (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    path = out_dir / "demand_potential.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real"])
    args = parser.parse_args()
    out_dir = OUT / args.data
    result_path = out_dir / "summary.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Run the real-data optimization first: {result_path}")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    scenarios = [name for name in DEMAND_SCENARIOS if name in report]
    if not scenarios:
        raise ValueError(f"No recognized demand scenarios in {result_path}")
    ins = params.build(PROC, scenario("base"))
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_site_maps(ins, out_dir, figures_dir, scenarios),
        plot_scenario_dashboard(report, figures_dir, scenarios),
        plot_demand_index(ins, figures_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
