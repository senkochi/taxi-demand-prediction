#!/usr/bin/env python3
"""Run training for a single method (method1) using the existing stable trainer module.
This wrapper imports the stable trainer file as a module, overrides its output dirs,
and runs smoke checks then full training. It does NOT modify the original file.
"""
from pathlib import Path
import importlib.util
import argparse
import sys
import os


def load_stable_module():
    trainer_path = Path(__file__).parent / "04_train_model_stable.py"
    spec = importlib.util.spec_from_file_location("stable_trainer", str(trainer_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    stable = load_stable_module()

    method = "method1"

    # isolate outputs per-method
    stable.DEFAULT_CHECKPOINT_DIR = Path(f"checkpoints/{method}")
    stable.DEFAULT_RESULTS_DIR = Path(f"reports/{method}")
    stable.DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stable.DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = stable.load_config(args.config)

    device = stable.torch.device("cuda" if stable.torch.cuda.is_available() else "cpu")

    # preflight summaries
    stable.print_header(f"SINGLE-METHOD TRAINER: {method}")
    stable.print(f"[Env] Device: {device}")
    stable.summarize_data_source(stable.DEFAULT_DUCKDB_PATH)
    stable.summarize_cluster_artifacts([method])

    # smoke check
    stable.run_smoke_check(config, method, device)

    if args.smoke_only:
        print("[Smoke] done")
        return 0

    # train
    stable.train_method(config, method, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
