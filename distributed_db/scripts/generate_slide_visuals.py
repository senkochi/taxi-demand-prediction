#!/usr/bin/env python3
"""Generate slide-ready visuals for the distributed database module.

Outputs:
- distributed_db/results/slide_01_sharding_topology.png
- distributed_db/results/slide_02_chunk_allocation.png
- distributed_db/results/slide_03_mongos_routing.png
- distributed_db/results/slide_04_distributed_aggregation.png
- distributed_db/results/slide_05_distributed_transparency.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SHARD_VISUAL = RESULTS_DIR / "shard_distribution_visualization.txt"
BENCHMARK_FILES = {
    "centralized": RESULTS_DIR / "benchmark_centralized_200k.txt",
    "sharded_pre": RESULTS_DIR / "benchmark_sharded_200k.txt",
    "sharded_post": RESULTS_DIR / "benchmark_sharded_after_split_200k.txt",
}


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"Unable to read {path}")


def parse_shard_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = re.search(r"^(shard\d+RS)\s*\|.*\|\s*(\d+)\s*$", line.strip())
        if match:
            counts[match.group(1)] = int(match.group(2))
    if counts:
        return counts

    matches = re.findall(r"shardName: '([^']+)'.*?docs: (\d+)", text, flags=re.S)
    if matches:
        return {name: int(value) for name, value in matches}

    raise RuntimeError("Could not parse shard counts from shard visualization")


def parse_benchmark(path: Path) -> dict[str, int]:
    text = read_text(path)
    values: dict[str, int] = {}
    for key in ("total_docs", "demand_agg_ms", "od_agg_ms"):
        match = re.search(rf"{key}=(\d+)", text)
        if not match:
            raise RuntimeError(f"Missing {key} in {path}")
        values[key] = int(match.group(1))
    return values


def base_canvas(title: str, figsize=(12, 6)):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=18, weight="bold")
    return fig, ax


def add_box(ax, xy, wh, text, facecolor, edgecolor="#2c3e50", textcolor="#1f2d3d", fontsize=11, weight="normal"):
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        weight=weight,
        wrap=True,
    )


def add_arrow(ax, start, end, color="#34495e", style="-|>", mutation_scale=18, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=1.8,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def save(fig, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / filename
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sharding_topology() -> Path:
    fig, ax = base_canvas("Sharding Architecture")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, (0.03, 0.38), (0.16, 0.22), "Client / App\nSingle query endpoint", "#D6EAF8", fontsize=12, weight="bold")
    add_box(ax, (0.29, 0.38), (0.16, 0.22), "mongos\nRouter", "#FCF3CF", fontsize=12, weight="bold")
    add_box(ax, (0.52, 0.72), (0.18, 0.16), "Config Server\n(cfgRS)", "#E8DAEF", fontsize=11, weight="bold")
    add_box(ax, (0.74, 0.72), (0.2, 0.16), "Shard 1\n(shard1RS)", "#D5F5E3", fontsize=11, weight="bold")
    add_box(ax, (0.74, 0.44), (0.2, 0.16), "Shard 2\n(shard2RS)", "#FADBD8", fontsize=11, weight="bold")
    add_box(ax, (0.74, 0.16), (0.2, 0.16), "Shard 3\n(shard3RS)", "#D6EAF8", fontsize=11, weight="bold")

    add_arrow(ax, (0.19, 0.49), (0.29, 0.49))
    add_arrow(ax, (0.45, 0.49), (0.69, 0.80))
    add_arrow(ax, (0.45, 0.49), (0.69, 0.52))
    add_arrow(ax, (0.45, 0.49), (0.69, 0.24))
    add_arrow(ax, (0.56, 0.72), (0.71, 0.58), color="#7D3C98")

    ax.text(
        0.5,
        0.06,
        "The application talks only to mongos; shard routing and metadata coordination stay inside the cluster.",
        ha="center",
        va="center",
        fontsize=11,
        color="#2c3e50",
    )
    return save(fig, "slide_01_sharding_topology.png")


def plot_chunk_allocation(shard_counts: dict[str, int]) -> Path:
    labels = list(shard_counts.keys())
    values = list(shard_counts.values())
    total = sum(values)
    mean = total / len(values)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2E86AB", "#58D68D", "#F39C12"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], width=0.58)
    ax.axhline(mean, color="#7D3C98", linestyle="--", linewidth=1.8, label=f"Mean = {mean:,.0f}")
    ax.set_title("Chunk Allocation Across Shards", fontsize=16, weight="bold")
    ax.set_ylabel("Owned Documents")
    ax.set_xlabel("Shard")
    ax.legend(frameon=False)

    for bar, value in zip(bars, values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.012,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.text(
        0.5,
        -0.18,
        "Post-split distribution is roughly balanced, but not perfectly equal because chunks follow the shard key ranges.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#566573",
    )
    fig.tight_layout()
    return save(fig, "slide_02_chunk_allocation.png")


def plot_mongos_routing() -> Path:
    fig, ax = base_canvas("Mongos Routing Path")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, (0.05, 0.4), (0.16, 0.2), "Query\n(e.g. aggregation)", "#D6EAF8", fontsize=12, weight="bold")
    add_box(ax, (0.31, 0.4), (0.17, 0.2), "mongos\nRoute planner", "#FCF3CF", fontsize=12, weight="bold")
    add_box(ax, (0.58, 0.7), (0.14, 0.14), "Shard 1", "#D5F5E3", fontsize=11, weight="bold")
    add_box(ax, (0.58, 0.48), (0.14, 0.14), "Shard 2", "#FADBD8", fontsize=11, weight="bold")
    add_box(ax, (0.58, 0.26), (0.14, 0.14), "Shard 3", "#D6EAF8", fontsize=11, weight="bold")
    add_box(ax, (0.8, 0.4), (0.15, 0.2), "Merged\nresult", "#E8DAEF", fontsize=12, weight="bold")

    add_arrow(ax, (0.21, 0.5), (0.31, 0.5))
    add_arrow(ax, (0.48, 0.5), (0.58, 0.77))
    add_arrow(ax, (0.48, 0.5), (0.58, 0.55))
    add_arrow(ax, (0.48, 0.5), (0.58, 0.33))
    add_arrow(ax, (0.72, 0.77), (0.8, 0.5), color="#7F8C8D")
    add_arrow(ax, (0.72, 0.55), (0.8, 0.5), color="#7F8C8D")
    add_arrow(ax, (0.72, 0.33), (0.8, 0.5), color="#7F8C8D")

    ax.text(0.5, 0.1, "mongos decides target shards from shard key + query shape, then merges partial results.", ha="center", va="center", fontsize=11)
    return save(fig, "slide_03_mongos_routing.png")


def plot_distributed_aggregation() -> Path:
    values = {
        "Centralized": {"demand": 1226, "od": 503},
        "Sharded\n(pre-split)": {"demand": 1403, "od": 662},
        "Sharded\n(post-split)": {"demand": 1290, "od": 478},
    }

    labels = list(values.keys())
    demand = [v["demand"] for v in values.values()]
    od = [v["od"] for v in values.values()]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#5DADE2", "#F5B041", "#58D68D"]

    axes[0].bar(labels, demand, color=colors, width=0.58)
    axes[0].set_title("Demand Aggregation Time")
    axes[0].set_ylabel("Milliseconds")
    for idx, value in enumerate(demand):
        axes[0].text(idx, value + max(demand) * 0.02, f"{value}", ha="center", va="bottom", fontsize=10)

    axes[1].bar(labels, od, color=colors, width=0.58)
    axes[1].set_title("OD Aggregation Time")
    for idx, value in enumerate(od):
        axes[1].text(idx, value + max(od) * 0.02, f"{value}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Distributed Aggregation Comparison (200k docs)", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "slide_04_distributed_aggregation.png")


def plot_distributed_transparency() -> Path:
    fig, ax = base_canvas("Distributed Transparency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, (0.05, 0.38), (0.18, 0.22), "Application\nSame query interface", "#D6EAF8", fontsize=12, weight="bold")
    add_box(ax, (0.31, 0.38), (0.18, 0.22), "mongos\nHides shard layout", "#FCF3CF", fontsize=12, weight="bold")
    add_box(ax, (0.58, 0.68), (0.16, 0.16), "Shard 1", "#D5F5E3", fontsize=11, weight="bold")
    add_box(ax, (0.58, 0.44), (0.16, 0.16), "Shard 2", "#FADBD8", fontsize=11, weight="bold")
    add_box(ax, (0.58, 0.20), (0.16, 0.16), "Shard 3", "#D6EAF8", fontsize=11, weight="bold")
    add_box(ax, (0.8, 0.38), (0.15, 0.22), "One logical\ncollection view", "#E8DAEF", fontsize=12, weight="bold")

    add_arrow(ax, (0.23, 0.49), (0.31, 0.49))
    add_arrow(ax, (0.49, 0.49), (0.58, 0.76))
    add_arrow(ax, (0.49, 0.49), (0.58, 0.52))
    add_arrow(ax, (0.49, 0.49), (0.58, 0.28))
    add_arrow(ax, (0.74, 0.49), (0.8, 0.49), color="#7F8C8D")

    ax.text(
        0.5,
        0.1,
        "The client queries taxi_db.trips normally; shard distribution stays hidden behind mongos.",
        ha="center",
        va="center",
        fontsize=11,
    )
    return save(fig, "slide_05_distributed_transparency.png")


def main() -> int:
    if not SHARD_VISUAL.exists():
        raise FileNotFoundError(f"Missing shard visualization file: {SHARD_VISUAL}")

    shard_counts = parse_shard_counts(read_text(SHARD_VISUAL))
    outputs = [
        plot_sharding_topology(),
        plot_chunk_allocation(shard_counts),
        plot_mongos_routing(),
        plot_distributed_aggregation(),
        plot_distributed_transparency(),
    ]

    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())