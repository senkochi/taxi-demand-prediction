"""
Stable SSTZIP-GNN training entrypoint.

This script replaces the older brittle training flow with:
- explicit preflight inspection of data, cluster artifacts, and feature columns
- a one-batch smoke check before any full training run
- safer runtime defaults for CPU and Colab GPU sessions
- per-method training and evaluation with saved metrics
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import TaxiDemandDataModule
from src.evaluation.metrics import Metrics
from src.models.sstzip_gnn import SSTZIPGNNModel
from src.training.trainer import SSTZIPGNNLightning


DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_DUCKDB_PATH = Path("data/processed/taxi_features.duckdb")
DEFAULT_RESULTS_DIR = Path("reports")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pick_methods(config: Dict[str, Any], requested_methods: Iterable[str] | None) -> List[str]:
    configured = config.get("clustering", {}).get("methods", ["method1", "method2", "method3"])
    if requested_methods:
        requested = [method for method in requested_methods if method in configured]
        return requested if requested else list(configured)
    return list(configured)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_cluster_artifacts(methods: Iterable[str]) -> None:
    print_header("CLUSTER ARTIFACTS")
    for method in methods:
        cluster_path = Path(f"data/models/{method}_clusters.pkl")
        if not cluster_path.exists():
            print(f"[MISSING] {method}: {cluster_path}")
            continue

        with open(cluster_path, "rb") as handle:
            artifact = pickle.load(handle)

        zone_to_cluster = artifact.get("zone_to_cluster") if isinstance(artifact, dict) else None
        cluster_counts: Dict[int, int] = {}
        if isinstance(zone_to_cluster, dict):
            for cluster_id in zone_to_cluster.values():
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        print(f"[OK] {method}: {cluster_path.name}")
        if isinstance(artifact, dict):
            print(f"  keys: {sorted(artifact.keys())}")
            if cluster_counts:
                print(f"  zone_count: {len(zone_to_cluster)}")
                print(f"  cluster_counts: {cluster_counts}")


def summarize_data_source(duckdb_path: Path) -> None:
    print_header("DATA SOURCE SUMMARY")
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {duckdb_path}")

    import duckdb

    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        columns = connection.execute("PRAGMA table_info(baseline_features)").fetchall()
        print("[Schema]")
        for column in columns:
            print(f"  {column[1]}\t{column[2]}")

        summary = connection.execute(
            """
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT zone_id) AS zones,
              SUM(CASE WHEN stddev_fare IS NULL THEN 1 ELSE 0 END) AS stddev_fare_nulls,
              SUM(CASE WHEN avg_fare IS NULL THEN 1 ELSE 0 END) AS avg_fare_nulls,
              SUM(CASE WHEN avg_distance IS NULL THEN 1 ELSE 0 END) AS avg_distance_nulls,
              SUM(CASE WHEN avg_passenger IS NULL THEN 1 ELSE 0 END) AS avg_passenger_nulls,
              SUM(CASE WHEN demand_count IS NULL THEN 1 ELSE 0 END) AS demand_nulls
            FROM baseline_features
            """
        ).fetchdf().iloc[0].to_dict()

        print("\n[Null Summary]")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        ranges = connection.execute(
            """
            SELECT
              MIN(date_str) AS min_date,
              MAX(date_str) AS max_date,
              MIN(time_bucket) AS min_bucket,
              MAX(time_bucket) AS max_bucket,
              MIN(demand_count) AS min_demand,
              MAX(demand_count) AS max_demand,
              AVG(demand_count) AS mean_demand
            FROM baseline_features
            """
        ).fetchdf().iloc[0].to_dict()

        print("\n[Range Summary]")
        for key, value in ranges.items():
            print(f"  {key}: {value}")
    finally:
        connection.close()


def build_data_module(config: Dict[str, Any], method: str, batch_size_override: int | None = None, num_workers_override: int | None = None) -> TaxiDemandDataModule:
    training_config = config.get("model", {}).get("training", {})
    split_config = config.get("train_val_test", {})

    batch_size = batch_size_override or training_config.get("batch_size", 128)
    configured_workers = training_config.get("num_workers", 0)
    num_workers = num_workers_override if num_workers_override is not None else min(configured_workers, 2)

    return TaxiDemandDataModule(
        duckdb_path=str(DEFAULT_DUCKDB_PATH),
        clustering_method=method,
        sequence_length=config.get("model", {}).get("sequence_length", 96),
        forecast_horizon=1,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=split_config.get("train_ratio", 0.85),
        val_ratio=split_config.get("val_ratio", 0.08),
        test_ratio=split_config.get("test_ratio", 0.07),
    )


def build_model(config: Dict[str, Any], feature_dim: int, num_zones: int) -> SSTZIPGNNModel:
    model_config = config.get("model", {})
    return SSTZIPGNNModel(
        num_zones=num_zones,
        feature_dim=feature_dim,
        spatial_dim=model_config.get("spatial_dim", 32),
        temporal_dim=model_config.get("temporal_dim", 32),
        num_spatial_layers=model_config.get("num_spatial_layers", 1),
        num_spatial_hops=model_config.get("num_spatial_hops", 2),
        num_temporal_layers=model_config.get("num_temporal_layers", 2),
        hidden_dim_zip=model_config.get("hidden_dim_zip", 64),
        dropout=model_config.get("dropout", 0.1),
    )


def apply_fast_mode_overrides(config: Dict[str, Any], device: torch.device) -> None:
    """Apply conservative speed-up defaults for Colab runs."""
    training_config = config.setdefault("model", {}).setdefault("training", {})

    training_config["epochs"] = min(int(training_config.get("epochs", 10)), 3)
    training_config["early_stopping_patience"] = min(int(training_config.get("early_stopping_patience", 5)), 2)

    if device.type == "cuda":
        training_config["precision"] = "16-mixed"
        training_config["num_workers"] = max(2, min(int(training_config.get("num_workers", 0)), max((os.cpu_count() or 2) - 1, 2)))
    else:
        training_config["precision"] = "32-true"


def run_smoke_check(config: Dict[str, Any], method: str, device: torch.device) -> tuple[TaxiDemandDataModule, SSTZIPGNNLightning, torch.Tensor, torch.Tensor]:
    print_header(f"SMOKE CHECK: {method}")
    data_module = build_data_module(config, method, batch_size_override=8, num_workers_override=0)
    data_module.setup()

    loader = data_module.train_dataloader()
    batch = next(iter(loader))
    x_batch, y_batch = batch

    print(f"[Smoke] x shape: {tuple(x_batch.shape)}")
    print(f"[Smoke] x finite: {torch.isfinite(x_batch).all().item()}")
    print(f"[Smoke] y shape: {tuple(y_batch.shape)}")
    print(f"[Smoke] y finite: {torch.isfinite(y_batch).all().item()}")

    if not torch.isfinite(x_batch).all():
        raise ValueError(
            "[Smoke] Non-finite values detected in the input batch. "
            "Restart the runtime and re-run the setup cells if you are in Colab."
        )

    model = build_model(config, feature_dim=x_batch.shape[-1], num_zones=data_module.adjacency_matrix.shape[0]).to(device)
    lightning_module = SSTZIPGNNLightning(
        model=model,
        learning_rate=config.get("model", {}).get("training", {}).get("learning_rate", 0.0003),
        weight_decay=1e-5,
        patience=config.get("model", {}).get("training", {}).get("early_stopping_patience", 5),
        accumulation_steps=1,
    ).to(device)

    lightning_module.eval()
    model.eval()

    with torch.no_grad():
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_proj = lightning_module.feature_projection(x_batch).transpose(1, 2)
        x_temporal = lightning_module.model.temporal_encoder(x_proj)
        x_agg = lightning_module.model.temporal_pooling(x_temporal)
        pi, lambda_param = lightning_module.model.zip_head(x_agg)
        smoke_loss = lightning_module.model.loss_fn((pi.squeeze(-1), lambda_param.squeeze(-1)), y_batch)

    print(f"[Smoke] pi finite: {torch.isfinite(pi).all().item()}")
    print(f"[Smoke] lambda finite: {torch.isfinite(lambda_param).all().item()}")
    print(f"[Smoke] loss finite: {torch.isfinite(smoke_loss).item()}")
    print(f"[Smoke] loss: {float(smoke_loss.item()):.6f}")

    return data_module, lightning_module, x_batch, y_batch


def evaluate_on_test(lightning_module: SSTZIPGNNLightning, test_loader, device: torch.device) -> Dict[str, float]:
    lightning_module.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_proj = lightning_module.feature_projection(x_batch).transpose(1, 2)
            x_temporal = lightning_module.model.temporal_encoder(x_proj)
            x_agg = lightning_module.model.temporal_pooling(x_temporal)
            pi, lambda_param = lightning_module.model.zip_head(x_agg)
            y_pred = (1 - pi.squeeze(-1)) * lambda_param.squeeze(-1)

            predictions.append(y_pred.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0).flatten()
    y_true = np.concatenate(targets, axis=0).flatten()

    return {
        "MAE": float(Metrics.mean_absolute_error(y_pred, y_true)),
        "RMSE": float(Metrics.root_mean_squared_error(y_pred, y_true)),
        "MAPE": float(Metrics.mean_absolute_percentage_error(y_pred, y_true)),
    }


def train_method(config: Dict[str, Any], method: str, device: torch.device) -> Dict[str, Any]:
    print_header(f"TRAINING: {method}")

    data_module = build_data_module(config, method)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    sample_batch = next(iter(train_loader))
    sample_x, sample_y = sample_batch

    print(f"[Config] Sequence length: {data_module.sequence_length}")
    print(f"[Config] Batch size: {data_module.batch_size}")
    print(f"[Config] Training samples: {len(data_module.train_dataset)}")
    print(f"[Config] Validation samples: {len(data_module.val_dataset)}")
    print(f"[Config] Test samples: {len(data_module.test_dataset)}")
    print(f"[Config] Feature columns: {data_module.train_dataset.feature_cols}")

    model = build_model(config, feature_dim=sample_x.shape[-1], num_zones=data_module.adjacency_matrix.shape[0]).to(device)
    lightning_module = SSTZIPGNNLightning(
        model=model,
        learning_rate=config.get("model", {}).get("training", {}).get("learning_rate", 0.0003),
        weight_decay=1e-5,
        patience=config.get("model", {}).get("training", {}).get("early_stopping_patience", 5),
        accumulation_steps=1,
    )

    checkpoint_dir = DEFAULT_CHECKPOINT_DIR / method
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=config.get("model", {}).get("training", {}).get("epochs", 10),
        accelerator="auto",
        devices=1,
        precision=config.get("model", {}).get("training", {}).get("precision", "32-true"),
        gradient_clip_val=config.get("model", {}).get("training", {}).get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config.get("model", {}).get("training", {}).get("accumulate_grad_batches", 1),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=config.get("model", {}).get("training", {}).get("early_stopping_patience", 5),
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="epoch-{epoch:02d}-val_loss-{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print("[Train] Starting fit...")
    fit_start = datetime.now()
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = (datetime.now() - fit_start).total_seconds()
    print(f"[OK] Training completed in {training_time:.1f} seconds")

    metrics = evaluate_on_test(lightning_module, test_loader, device)
    metrics.update(
        {
            "Training_Time_Sec": training_time,
            "Num_Parameters": sum(p.numel() for p in model.parameters()),
            "Method": method,
            "Feature_Count": sample_x.shape[-1],
            "Train_Batches": len(train_loader),
            "Val_Batches": len(val_loader),
            "Test_Batches": len(test_loader),
        }
    )

    print("[Metrics]")
    for key, value in metrics.items():
        if key != "Method":
            print(f"  {key}: {value}")

    results_dir = checkpoint_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"method": method, "metrics": metrics, "timestamp": datetime.now().isoformat()}, handle, indent=2)

    torch.save(model.state_dict(), results_dir / "model.pt")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable SSTZIP-GNN training script")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config/config.yaml")
    parser.add_argument("--smoke-only", action="store_true", help="Run the smoke check and exit")
    parser.add_argument("--method", help="Run a single clustering method (method1, method2, or method3)")
    parser.add_argument("--methods", nargs="*", help="Optional subset of clustering methods to run")
    parser.add_argument("--fast", action="store_true", help="Apply speed-up defaults for Colab runs")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[WARN] Ignoring extra arguments: {' '.join(unknown_args)}")

    config = load_config(args.config)
    requested_methods = [args.method] if args.method else args.methods
    methods = pick_methods(config, requested_methods)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fast:
        apply_fast_mode_overrides(config, device)

    print_header("STABLE SSTZIP-GNN TRAINING")
    print(f"[Env] Device: {device}")
    print(f"[Env] CUDA available: {torch.cuda.is_available()}")
    print(f"[Env] Methods: {methods}")

    summarize_data_source(DEFAULT_DUCKDB_PATH)
    summarize_cluster_artifacts(methods)

    if not methods:
        raise RuntimeError("No clustering methods selected")

    # Smoke check first, using the first selected method.
    smoke_method = methods[0]
    smoke_data_module, smoke_lightning_module, _, _ = run_smoke_check(config, smoke_method, device)

    if args.smoke_only:
        print("[Smoke] Smoke-only mode complete; exiting before full training.")
        return 0

    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        try:
            metrics = train_method(config, method, device)
            results[method] = metrics
        except Exception as exc:
            print(f"[ERROR] {method}: {exc}")
            traceback.print_exc()

    print_header("TRAINING SUMMARY")
    if results:
        summary_df = pd.DataFrame(results).T
        print(summary_df.to_string())
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = DEFAULT_RESULTS_DIR / "training_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"[OK] Summary saved to {summary_path}")
    else:
        print("[WARN] No methods completed successfully.")

    print("\n[Done] Stable training script finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())