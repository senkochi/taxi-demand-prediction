#!/usr/bin/env python3
"""Fast classical-ML benchmark for each clustering method.

This script trains two lightweight regressors per method on the shared
baseline feature table augmented with the cluster assignment from each method:
- Ridge
- SGDRegressor

The benchmark predicts `demand_count`, which keeps the target consistent across
methods while still measuring the effect of the spatial partitioning.

Outputs are written to reports/classical_ml/ as JSON and CSV summaries.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASELINE_FEATURES = Path("data/processed/baseline_features/baseline_features.parquet")
METHOD_FEATURES = {
    "baseline": Path("data/processed/baseline_features/baseline_features.parquet"),
    "method1": Path("data/processed/method1_features/method1_demand_features.parquet"),
    "method2": Path("data/processed/method2_features/method2_mobility_features.parquet"),
    "method3": Path("data/processed/method3_features/method3_od_features.parquet"),
}
BASELINE_CLUSTERS = Path("data/models/baseline_clusters.pkl")
DEFAULT_OUTPUT_DIR = Path("reports/classical_ml")


@dataclass
class ModelResult:
    method: str
    model: str
    train_rows: int
    test_rows: int
    num_features: int
    train_seconds: float
    predict_seconds: float
    mae: float
    rmse: float
    r2: float
    mape: float


def read_parquet(path: Path, limit: int | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM read_parquet('{path.as_posix()}')"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"
    connection = duckdb.connect()
    try:
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def load_cluster_mapping(method: str) -> pd.Series:
    if method == "baseline":
        cluster_path = BASELINE_CLUSTERS
    else:
        cluster_path = Path(f"data/models/{method}_clusters.pkl")

    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing cluster artifact: {cluster_path}")

    with open(cluster_path, "rb") as handle:
        artifact = pickle.load(handle)

    zone_to_cluster = artifact.get("zone_to_cluster") if isinstance(artifact, dict) else None
    if not isinstance(zone_to_cluster, dict):
        raise ValueError(f"Invalid cluster artifact: {cluster_path}")

    return pd.Series(zone_to_cluster, name="cluster_id")


def load_method_frame(method: str, max_rows: int | None) -> Tuple[pd.DataFrame, pd.Series]:
    df = read_parquet(BASELINE_FEATURES, limit=max_rows)
    df = df.sort_values(["time_bucket", "zone_id"]).reset_index(drop=True)

    cluster_map = load_cluster_mapping(method)
    df = df.merge(cluster_map.rename("cluster_id"), left_on="zone_id", right_index=True, how="left")
    if df["cluster_id"].isna().any():
        missing = int(df["cluster_id"].isna().sum())
        raise ValueError(f"Cluster mapping failed for {missing} rows in {method}")

    y = df["demand_count"].astype(float)
    return df, y


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "demand_count",
        "cluster_id",
        "zone_id",
        "date_str",
        "time_bucket",
        "window_start",
        "window_end",
    }
    numeric = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [col for col in numeric.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("No numeric features available for benchmark")
    features = df[feature_cols].copy()
    features["cluster_id"] = df["cluster_id"].astype(float)
    return features


def make_models(random_state: int) -> Dict[str, Pipeline]:
    ridge = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    sgd = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=200,
                    tol=1e-2,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=3,
                    average=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "ridge_regressor": ridge,
        "sgd_regressor": sgd,
    }


def evaluate_model(method: str, model_name: str, model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> ModelResult:
    print(f"[Run] {method} / {model_name}: fitting on {len(X_train):,} rows", flush=True)
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start_train

    print(f"[Run] {method} / {model_name}: predicting on {len(X_test):,} rows", flush=True)
    start_predict = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - start_predict

    print(
        f"[Done] {method} / {model_name}: MAE={mean_absolute_error(y_test, y_pred):.4f}, "
        f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}",
        flush=True,
    )

    return ModelResult(
        method=method,
        model=model_name,
        train_rows=len(X_train),
        test_rows=len(X_test),
        num_features=X_train.shape[1],
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        r2=float(r2_score(y_test, y_pred)),
        mape=float(np.mean(np.abs((y_test.to_numpy() - y_pred) / np.maximum(np.abs(y_test.to_numpy()), 1e-8))) * 100.0),
    )


def split_by_time(df: pd.DataFrame, test_size: float) -> Tuple[pd.Index, pd.Index]:
    unique_times = pd.Index(df["time_bucket"].drop_duplicates().sort_values())
    split_index = max(1, int(len(unique_times) * (1.0 - test_size)))
    train_times = unique_times[:split_index]
    test_times = unique_times[split_index:]
    return train_times, test_times


def run_benchmark(methods: Iterable[str], max_rows: int | None, test_size: float, random_state: int) -> List[ModelResult]:
    results: List[ModelResult] = []
    models = make_models(random_state)

    for method in methods:
        df, y = load_method_frame(method, max_rows=max_rows)
        X = build_feature_matrix(df)

        train_times, test_times = split_by_time(df, test_size=test_size)
        train_mask = df["time_bucket"].isin(train_times)
        test_mask = df["time_bucket"].isin(test_times)

        X_train = X.loc[train_mask].reset_index(drop=True)
        X_test = X.loc[test_mask].reset_index(drop=True)
        y_train = y.loc[train_mask].reset_index(drop=True)
        y_test = y.loc[test_mask].reset_index(drop=True)

        for model_name, model in models.items():
            results.append(
                evaluate_model(
                    method=method,
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
            )

    return results


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(result) for result in results])
    df = df.sort_values(["method", "mae"], ascending=[True, True]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast classical ML benchmark for taxi demand methods")
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
        help="Maximum rows to load from the shared baseline table for each method; use 0 for full data",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [method for method in args.methods if method in METHOD_FEATURES]
    if not methods:
        raise ValueError("No valid methods selected")

    print("=" * 80)
    print("FAST CLASSICAL ML BENCHMARK")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Max rows: {args.max_rows if args.max_rows > 0 else 'full'}")
    print(f"Test size: {args.test_size}")

    results = run_benchmark(methods, max_rows=args.max_rows, test_size=args.test_size, random_state=args.random_state)
    summary = summarize_results(results)

    json_path = args.output_dir / "classical_ml_benchmark.json"
    csv_path = args.output_dir / "classical_ml_benchmark.csv"

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