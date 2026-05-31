#!/usr/bin/env python3
"""Baseline SSTZIP-GNN training wrapper.

This delegates baseline training to the stable trainer so baseline runs through
the same smoke-check and training flow as the other methods.
Baseline clustering is handled by the shared data loader, which creates the
identity clustering artifact on demand when baseline is selected.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_stable_module():
    trainer_path = Path(__file__).parent / "04_train_model_stable.py"
    spec = importlib.util.spec_from_file_location("stable_trainer", str(trainer_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    stable.print_header(f"SINGLE-METHOD TRAINER: {method}")
    print(f"[Env] Device: {device}")
    print("[Env] Cluster artifact: data/models/baseline_clusters.pkl (auto-generated on demand)")
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
