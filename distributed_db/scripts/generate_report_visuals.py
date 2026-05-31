#!/usr/bin/env python3
"""Generate report-ready visuals for the distributed database module.

Outputs:
- distributed_db/results/shard_distribution_report.png
- distributed_db/results/benchmark_comparison_report.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SHARD_RAW = RESULTS_DIR / "shard_distribution_raw.txt"
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
    # Parse lines like: shard1RS  | ███ | 68070
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = re.search(r"^(shard\d+RS)\s*\|.*\|\s*(\d+)\s*$", line.strip())
        if match:
            counts[match.group(1)] = int(match.group(2))
    if counts:
        return counts

    # Fallback: parse raw mongosh output.
    matches = re.findall(r"shardName: '([^']+)'.*?docs: (\d+)", text, flags=re.S)
    if matches:
        return {name: int(value) for name, value in matches}

    raise RuntimeError("Could not parse shard counts from raw output")


def parse_benchmark(path: Path) -> dict[str, int]:
    text = read_text(path)
    values: dict[str, int] = {}
    for key in ("total_docs", "demand_agg_ms", "od_agg_ms"):
        match = re.search(rf"{key}=(\d+)", text)
        if not match:
            raise RuntimeError(f"Missing {key} in {path}")
        values[key] = int(match.group(1))
    return values


def plot_shard_distribution(counts: dict[str, int]) -> Path:
    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    mean = total / len(values)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values, color=["#2E86AB", "#4CAF50", "#F39C12"][: len(labels)])
    ax.axhline(mean, color="#7D3C98", linestyle="--", linewidth=1.8, label=f"Mean = {mean:,.0f}")
    ax.set_title("Shard Distribution After Sharding")
    ax.set_ylabel("Owned Documents")
    ax.set_xlabel("Shard")
    ax.legend(frameon=False)

    for bar, value in zip(bars, values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.01,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output = RESULTS_DIR / "shard_distribution_report.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_benchmark_comparison(values: dict[str, dict[str, int]]) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    labels = ["Centralized", "Sharded\n(pre-split)", "Sharded\n(post-split)"]
    demand = [values["centralized"]["demand_agg_ms"], values["sharded_pre"]["demand_agg_ms"], values["sharded_post"]["demand_agg_ms"]]
    od = [values["centralized"]["od_agg_ms"], values["sharded_pre"]["od_agg_ms"], values["sharded_post"]["od_agg_ms"]]

    colors = ["#5DADE2", "#F5B041", "#58D68D"]
    axes[0].bar(labels, demand, color=colors)
    axes[0].set_title("Demand Aggregation Time")
    axes[0].set_ylabel("Milliseconds")
    for idx, value in enumerate(demand):
        axes[0].text(idx, value + max(demand) * 0.02, f"{value}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, od, color=colors)
    axes[1].set_title("OD Aggregation Time")
    for idx, value in enumerate(od):
        axes[1].text(idx, value + max(od) * 0.02, f"{value}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Benchmark Comparison (200k documents)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = RESULTS_DIR / "benchmark_comparison_report.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    shard_source = SHARD_VISUAL if SHARD_VISUAL.exists() else SHARD_RAW
    if not shard_source.exists():
        raise FileNotFoundError(f"Missing shard distribution file: {SHARD_VISUAL} or {SHARD_RAW}")

    shard_counts = parse_shard_counts(read_text(shard_source))
    benchmark_values = {name: parse_benchmark(path) for name, path in BENCHMARK_FILES.items()}

    shard_png = plot_shard_distribution(shard_counts)
    benchmark_png = plot_benchmark_comparison(benchmark_values)

    print(f"Wrote {shard_png}")
    print(f"Wrote {benchmark_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())