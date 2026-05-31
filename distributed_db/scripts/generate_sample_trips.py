#!/usr/bin/env python3
"""Generate a synthetic NYC taxi sample dataset for sharded-cluster testing."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def build_rows(row_count: int, start: datetime, end: datetime):
    total_seconds = int((end - start).total_seconds())

    for _ in range(row_count):
        pickup = start + timedelta(seconds=random.randint(0, total_seconds))
        trip_minutes = random.randint(3, 75)
        dropoff = pickup + timedelta(minutes=trip_minutes)

        yield {
            "pickup_datetime": pickup.strftime("%Y-%m-%dT%H:%M:%S"),
            "dropoff_datetime": dropoff.strftime("%Y-%m-%dT%H:%M:%S"),
            "PULocationID": random.randint(1, 263),
            "DOLocationID": random.randint(1, 263),
            "passenger_count": random.randint(1, 6),
            "trip_distance": round(random.uniform(0.3, 22.0), 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample taxi trip data")
    parser.add_argument("--rows", type=int, default=100000, help="Number of rows to generate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("distributed_db/data/taxi_sample.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 12, 31, 23, 59, 59)

    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "pickup_datetime",
                "dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "passenger_count",
                "trip_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(build_rows(args.rows, start, end))

    print(f"[OK] Generated {args.rows} rows at: {args.output}")


if __name__ == "__main__":
    main()
