#!/usr/bin/env python3
"""Generate predictions analysis plot using a classical ML baseline.

Trains a Ridge or SGD regressor on a subset of the baseline features,
produces test-set predictions, and calls the visualization module to
create `predictions_analysis.png` from real experiment outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization.visualization import ModelVisualization


BASELINE_FEATURES = Path("data/processed/baseline_features/baseline_features.parquet")
BASELINE_CLUSTERS = Path("data/models/baseline_clusters.pkl")


def read_parquet(path: Path, limit: int | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM read_parquet('{path.as_posix()}')"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"
    conn = duckdb.connect()
    try:
        return conn.execute(query).fetchdf()
    finally:
        conn.close()


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
    features = df[feature_cols].copy()
    return features


def make_model(name: str):
    if name == "ridge_regressor":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "sgd_regressor":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SGDRegressor(max_iter=200))])
    raise ValueError(f"Unknown model: {name}")


def split_by_time(df: pd.DataFrame, test_size: float):
    unique_times = pd.Index(df["time_bucket"].drop_duplicates().sort_values())
    split_index = max(1, int(len(unique_times) * (1.0 - test_size)))
    train_times = unique_times[:split_index]
    test_times = unique_times[split_index:]
    return train_times, test_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ridge_regressor")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("reports/figures/predictions_analysis.png"))
    args = parser.parse_args()

    df = read_parquet(BASELINE_FEATURES, limit=args.max_rows)
    df = df.sort_values(["time_bucket", "zone_id"]).reset_index(drop=True)

    # For this quick plot we use the baseline demand_count target
    y = df["demand_count"].astype(float)
    X = build_feature_matrix(df)

    train_times, test_times = split_by_time(df, test_size=args.test_size)
    train_mask = df["time_bucket"].isin(train_times)
    test_mask = df["time_bucket"].isin(test_times)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    model = make_model(args.model)
    print(f"Training {args.model} on {len(X_train)} rows; testing on {len(X_test)} rows")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    viz = ModelVisualization()
    # For classical models, y_pred_mean is prediction; we set y_pred_zero to zeros and lambda equal to prediction
    y_pred_mean = np.asarray(y_pred)
    y_pred_zero = np.zeros_like(y_pred_mean)
    y_pred_lambda = y_pred_mean.copy()

    viz.plot_predictions_analysis(y_true=y_test.to_numpy(), y_pred_mean=y_pred_mean, y_pred_zero=y_pred_zero, y_pred_lambda=y_pred_lambda, save_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
