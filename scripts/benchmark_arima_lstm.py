#!/usr/bin/env python3
"""Benchmark ARIMA and LSTM across clustering methods.

For each method, the script builds a cluster-level demand series from the
shared baseline feature table, then evaluates:
- ARIMA(1,1,1)
- LSTM regressor over a rolling window of past demand values

The benchmark uses the mean demand per cluster at each time bucket and then
averages across clusters, which keeps the comparison method-specific while
staying fast enough to run on the full dataset.

Outputs are written to reports/time_series_ml/ as JSON and CSV summaries.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASELINE_FEATURES = Path("data/processed/baseline_features/baseline_features.parquet")
BASELINE_CLUSTERS = Path("data/models/baseline_clusters.pkl")
METHOD_CLUSTER_FILES = {
    "baseline": BASELINE_CLUSTERS,
    "method1": Path("data/models/method1_clusters.pkl"),
    "method2": Path("data/models/method2_clusters.pkl"),
    "method3": Path("data/models/method3_clusters.pkl"),
}
DEFAULT_OUTPUT_DIR = Path("reports/time_series_ml")


@dataclass
class ModelResult:
    method: str
    model: str
    train_points: int
    test_points: int
    train_seconds: float
    predict_seconds: float
    mae: float
    rmse: float
    r2: float
    mape: float


def read_baseline_features(limit: int | None = None) -> pd.DataFrame:
    query = f"SELECT date_str, time_bucket, zone_id, demand_count FROM read_parquet('{BASELINE_FEATURES.as_posix()}')"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"
    connection = duckdb.connect()
    try:
        df = connection.execute(query).fetchdf()
    finally:
        connection.close()
    return df.sort_values(["time_bucket", "zone_id"]).reset_index(drop=True)


def load_cluster_mapping(method: str) -> Dict[int, int]:
    cluster_path = METHOD_CLUSTER_FILES[method]
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing cluster artifact: {cluster_path}")

    with open(cluster_path, "rb") as handle:
        artifact = pickle.load(handle)

    zone_to_cluster = artifact.get("zone_to_cluster") if isinstance(artifact, dict) else None
    if not isinstance(zone_to_cluster, dict):
        raise ValueError(f"Invalid cluster artifact: {cluster_path}")
    return {int(zone): int(cluster) for zone, cluster in zone_to_cluster.items()}


def build_cluster_series(method: str, max_rows: int | None) -> pd.Series:
    df = read_baseline_features(limit=max_rows)
    mapping = load_cluster_mapping(method)
    df["cluster_id"] = df["zone_id"].map(mapping)
    if df["cluster_id"].isna().any():
        missing = int(df["cluster_id"].isna().sum())
        raise ValueError(f"Cluster mapping failed for {missing} rows in {method}")

    cluster_mean = (
        df.groupby(["time_bucket", "cluster_id"], as_index=False)["demand_count"]
        .mean()
        .rename(columns={"demand_count": "cluster_mean_demand"})
    )
    series = cluster_mean.groupby("time_bucket")["cluster_mean_demand"].mean().sort_index()
    return series.astype(float)


def build_zone_series(method: str, max_rows: int | None) -> Dict[int, pd.Series]:
    """Return a mapping of zone_id -> time series of demand_count for the method's zones.

    This preserves zone-level granularity so we can run per-zone forecasts and
    average metrics across zones (comparable to zone-level GNN predictions).
    """
    df = read_baseline_features(limit=max_rows)
    mapping = load_cluster_mapping(method)
    df["cluster_id"] = df["zone_id"].map(mapping)
    if df["cluster_id"].isna().any():
        missing = int(df["cluster_id"].isna().sum())
        raise ValueError(f"Cluster mapping failed for {missing} rows in {method}")

    # Pivot to time_bucket × zone_id and return series per zone
    pivot = (
        df.groupby(["time_bucket", "zone_id"], as_index=False)["demand_count"].mean()
        .pivot(index="time_bucket", columns="zone_id", values="demand_count")
        .sort_index()
    )
    zone_series: Dict[int, pd.Series] = {}
    for zone in pivot.columns:
        zone_series[int(zone)] = pivot[zone].astype(float).dropna()
    return zone_series


def split_series(series: pd.Series, test_size: float) -> Tuple[pd.Series, pd.Series]:
    split_index = max(2, int(len(series) * (1.0 - test_size)))
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]
    if len(test) == 0:
        raise ValueError("Test split is empty; reduce train size or provide more data")
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0),
    }


def fit_arima(train: pd.Series, test: pd.Series) -> Tuple[np.ndarray, float, float]:
    from statsmodels.tsa.arima.model import ARIMA

    start = time.perf_counter()
    model = ARIMA(train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit()
    train_seconds = time.perf_counter() - start

    start_predict = time.perf_counter()
    forecast = fitted.forecast(steps=len(test))
    predict_seconds = time.perf_counter() - start_predict
    return np.clip(np.asarray(forecast, dtype=float), 0.0, None), train_seconds, predict_seconds


class DemandLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def make_windows(values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    inputs: List[np.ndarray] = []
    targets: List[float] = []
    for end_idx in range(seq_len, len(values)):
        inputs.append(values[end_idx - seq_len:end_idx])
        targets.append(values[end_idx])
    if not inputs:
        raise ValueError("Series too short for LSTM windows")
    x = np.asarray(inputs, dtype=np.float32)[:, :, None]
    y = np.asarray(targets, dtype=np.float32)
    return x, y


def fit_lstm(train: pd.Series, test: pd.Series, seq_len: int, random_state: int) -> Tuple[np.ndarray, float, float]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.to_numpy().reshape(-1, 1)).astype(np.float32).reshape(-1)
    full_scaled = scaler.transform(pd.concat([train, test]).to_numpy().reshape(-1, 1)).astype(np.float32).reshape(-1)

    try:
        x_train, y_train = make_windows(train_scaled, seq_len)
    except ValueError:
        # Series too short for LSTM windows: return empty forecasts
        return np.array([]), 0.0, 0.0
    split_idx = max(1, int(len(x_train) * 0.9))
    x_fit, y_fit = x_train[:split_idx], y_train[:split_idx]
    x_val, y_val = x_train[split_idx:], y_train[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=512,
        shuffle=False,
    )

    model = DemandLSTM(hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    start_train = time.perf_counter()
    best_state = None
    best_val = float("inf")
    patience = 1
    stale = 0

    for _epoch in range(5):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_losses.append(loss_fn(pred, batch_y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_seconds = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions: List[float] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(train_scaled), len(full_scaled)):
            window = full_scaled[idx - seq_len:idx]
            x = torch.from_numpy(window.reshape(1, seq_len, 1))
            pred = float(model(x).item())
            predictions.append(pred)
    predict_seconds = time.perf_counter() - start_predict

    forecast = scaler.inverse_transform(np.asarray(predictions, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    return np.clip(forecast, 0.0, None), train_seconds, predict_seconds


def run_benchmark(methods: Iterable[str], max_rows: int | None, test_size: float, random_state: int) -> List[ModelResult]:
    results: List[ModelResult] = []
    for method in methods:
        if getattr(run_benchmark, "granularity", "cluster") == "cluster":
            series = build_cluster_series(method, max_rows=max_rows)
            train, test = split_series(series, test_size=test_size)

            print(f"[Series] {method} (cluster-level): train={len(train):,}, test={len(test):,}", flush=True)

            start = time.perf_counter()
            arima_pred, arima_train_s, arima_pred_s = fit_arima(train, test)
            arima_total = time.perf_counter() - start
            arima_metrics = regression_metrics(test.to_numpy(), arima_pred)
            print(f"[Done] {method} / arima_111: MAE={arima_metrics['mae']:.4f}, RMSE={arima_metrics['rmse']:.4f}", flush=True)
            results.append(
                ModelResult(
                    method=method,
                    model="arima_111",
                    train_points=len(train),
                    test_points=len(test),
                    train_seconds=arima_train_s,
                    predict_seconds=arima_pred_s,
                    **arima_metrics,
                )
            )

            start = time.perf_counter()
            lstm_pred, lstm_train_s, lstm_pred_s = fit_lstm(train, test, seq_len=24, random_state=random_state)
            lstm_total = time.perf_counter() - start
            lstm_metrics = regression_metrics(test.to_numpy(), lstm_pred)
            print(f"[Done] {method} / lstm: MAE={lstm_metrics['mae']:.4f}, RMSE={lstm_metrics['rmse']:.4f}", flush=True)
            results.append(
                ModelResult(
                    method=method,
                    model="lstm_seq48",
                    train_points=len(train),
                    test_points=len(test),
                    train_seconds=lstm_train_s,
                    predict_seconds=lstm_pred_s,
                    **lstm_metrics,
                )
            )

            print(f"[Time] {method}: arima_total={arima_total:.1f}s, lstm_total={lstm_total:.1f}s", flush=True)
        else:
            # zone-level: run per-zone forecasts then average metrics across zones
            zone_series = build_zone_series(method, max_rows=max_rows)
            zone_results: List[Dict[str, float]] = []
            print(f"[Series] {method} (zone-level): zones={len(zone_series)}", flush=True)

            for zone_id, series in zone_series.items():
                # require minimal length for ARIMA/LSTM: at least 10 points
                if len(series) < 10:
                    continue
                try:
                    train, test = split_series(series, test_size=test_size)
                except ValueError:
                    # skip zones with too-short series after split
                    continue

                arima_pred, arima_train_s, arima_pred_s = fit_arima(train, test)
                arima_metrics = regression_metrics(test.to_numpy(), arima_pred)

                lstm_pred, lstm_train_s, lstm_pred_s = fit_lstm(train, test, seq_len=24, random_state=random_state)
                if lstm_pred.size == 0:
                    # LSTM couldn't run for this zone (too short); mark as NaN and skip in averaging
                    lstm_metrics = {"mae": np.nan, "rmse": np.nan}
                else:
                    lstm_metrics = regression_metrics(test.to_numpy(), lstm_pred)

                zone_results.append({
                    "zone_id": int(zone_id),
                    "arima_mae": arima_metrics["mae"],
                    "arima_rmse": arima_metrics["rmse"],
                    "lstm_mae": lstm_metrics["mae"],
                    "lstm_rmse": lstm_metrics["rmse"],
                })

            # Average across zones
            if not zone_results:
                print(f"No valid zone series for {method}; skipping.", flush=True)
                continue

            arima_mae = float(np.mean([z["arima_mae"] for z in zone_results]))
            arima_rmse = float(np.mean([z["arima_rmse"] for z in zone_results]))
            lstm_mae = float(np.mean([z["lstm_mae"] for z in zone_results]))
            lstm_rmse = float(np.mean([z["lstm_rmse"] for z in zone_results]))

            results.append(
                ModelResult(
                    method=method,
                    model="arima_111",
                    train_points=int(np.mean([len(build_zone_series(method, max_rows=max_rows)[z]) * (1 - test_size) for z in build_zone_series(method, max_rows=max_rows)])),
                    test_points=int(np.mean([len(build_zone_series(method, max_rows=max_rows)[z]) * test_size for z in build_zone_series(method, max_rows=max_rows)])),
                    train_seconds=0.0,
                    predict_seconds=0.0,
                    mae=arima_mae,
                    rmse=arima_rmse,
                    r2=0.0,
                    mape=0.0,
                )
            )

            results.append(
                ModelResult(
                    method=method,
                    model="lstm_seq48",
                    train_points=int(np.mean([len(build_zone_series(method, max_rows=max_rows)[z]) * (1 - test_size) for z in build_zone_series(method, max_rows=max_rows)])),
                    test_points=int(np.mean([len(build_zone_series(method, max_rows=max_rows)[z]) * test_size for z in build_zone_series(method, max_rows=max_rows)])),
                    train_seconds=0.0,
                    predict_seconds=0.0,
                    mae=lstm_mae,
                    rmse=lstm_rmse,
                    r2=0.0,
                    mape=0.0,
                )
            )

    return results


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(result) for result in results])
    return df.sort_values(["method", "mae"], ascending=[True, True]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ARIMA and LSTM across clustering methods")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["baseline", "method1", "method2", "method3"],
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to load from the baseline feature table; use 0 for full data",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--granularity", type=str, choices=["cluster", "zone"], default="cluster", help="Granularity to benchmark: 'cluster' or 'zone' (zone-level averages per-zone)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [method for method in args.methods if method in METHOD_CLUSTER_FILES]
    if not methods:
        raise ValueError("No valid methods selected")

    print("=" * 80)
    print("ARIMA + LSTM BENCHMARK")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Max rows: {args.max_rows if args.max_rows > 0 else 'full'}")
    print(f"Test size: {args.test_size}")

    # Attach granularity to function for easy access
    setattr(run_benchmark, "granularity", args.granularity)
    results = run_benchmark(methods, max_rows=args.max_rows, test_size=args.test_size, random_state=args.random_state)
    summary = summarize_results(results)

    json_path = args.output_dir / "arima_lstm_benchmark.json"
    csv_path = args.output_dir / "arima_lstm_benchmark.csv"

    payload = {
        "methods": methods,
        "max_rows": args.max_rows,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "results": [asdict(result) for result in results],
        "best_by_method": summary.groupby("method", as_index=False).first().to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    summary.to_csv(csv_path, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())