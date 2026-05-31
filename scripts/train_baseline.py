#!/usr/bin/env python3
"""Baseline SSTZIP-GNN training wrapper.

This keeps the stable trainer unchanged, generates a baseline cluster artifact
if it does not already exist, then trains only the baseline method.
Baseline cluster strategy: each zone is assigned to its own cluster, which
produces an identity adjacency structure for the baseline graph.
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path


def load_stable_module():
    trainer_path = Path(__file__).parent / "04_train_model_stable.py"
    spec = importlib.util.spec_from_file_location("stable_trainer", str(trainer_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_baseline_clusters(duckdb_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    import duckdb

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        zone_ids = [row[0] for row in conn.execute("SELECT DISTINCT zone_id FROM baseline_features ORDER BY zone_id").fetchall()]
    finally:
        conn.close()

    if not zone_ids:
        raise RuntimeError("No zones found in baseline_features; cannot create baseline clusters.")

    zone_to_cluster = {int(zone_id): int(zone_id) for zone_id in zone_ids}
    artifact = {
        "zone_to_cluster": zone_to_cluster,
        "silhouette_score": 0.0,
        "optimal_k": len(zone_ids),
        "cluster_centers": [[float(zone_id)] for zone_id in zone_ids],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(artifact, handle)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SSTZIP-GNN baseline method")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    stable = load_stable_module()
    method = "baseline"

    stable.DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
    stable.DEFAULT_RESULTS_DIR = Path("reports")
    stable.DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stable.DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = stable.load_config(args.config)
    device = stable.torch.device("cuda" if stable.torch.cuda.is_available() else "cpu")

    cluster_path = ensure_baseline_clusters(Path("data/processed/taxi_features.duckdb"), Path("data/models/baseline_clusters.pkl"))

    stable.print_header(f"SINGLE-METHOD TRAINER: {method}")
    print(f"[Env] Device: {device}")
    print(f"[Env] Cluster artifact: {cluster_path}")
    stable.summarize_data_source(stable.DEFAULT_DUCKDB_PATH)
    stable.summarize_cluster_artifacts([method])

    stable.run_smoke_check(config, method, device)
    if args.smoke_only:
        print("[Smoke] done")
        return 0

    stable.train_method(config, method, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
