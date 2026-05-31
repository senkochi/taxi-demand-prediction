# Module: Distributed Database - MongoDB Atlas Sharding for Taxi Time-Series

IMPORTANT: This project has two tracks: Big Data and Distributed Database.

Owner: Person A (with Person B support)
Duration: Weeks 1-2 setup, Weeks 3-5 analysis
Deliverables: MongoDB sharded deployment, write/read pipelines, distributed systems analysis, performance report

---

## 1. Scope and Correct Architecture

This module is implemented as a separate data platform track, not inside the main model training loop.

### 1.1 Separation of Responsibilities

- Distributed Database module:
  - Ingests aggregated features into MongoDB.
  - Serves analytics and distributed systems experiments.
  - Benchmarks replication, failover, and shard balancing.
- Main training pipeline:
  - Trains models from curated offline datasets (DuckDB/Parquet).
  - Does not depend on live MongoDB reads during each training epoch.

This separation keeps training reproducible and keeps database experiments independent.

### 1.2 Why Sharding (Not Replica Set Only)

Replica sets solve high availability, but not horizontal write scale.
For this module, we need both:

- Replica sets: durability and failover.
- Sharding: horizontal distribution for large write/read volume across nodes.

Target architecture in Atlas:

- 1 mongos router layer (managed by Atlas)
- 1 config server replica set (managed by Atlas)
- N shards, each shard is a replica set

---

## 2. Atlas Setup for Sharding

### 2.1 Cluster Tier Requirement

Use an Atlas dedicated tier that supports sharded clusters (typically M30+; verify current Atlas limits at setup time).
Shared free tiers are not suitable for production-like sharding experiments.

### 2.2 Setup Steps

1. Create Atlas project and dedicated cluster.
2. Choose sharded cluster topology.
3. Configure database user and network access (avoid open IP ranges outside dev testing).
4. Store connection string in environment:

```bash
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/taxi_db?retryWrites=true&w=majority
```

5. Validate connection with a short ping script before ingestion.

---

## 3. Data Model and Shard Key Strategy

### 3.1 Collection Design

Use one primary collection for demand features:

- Collection: taxi_demand
- Document shape:
  - timestamp (event time)
  - hour_bucket (timestamp truncated to hour for partitioning and queries)
  - metadata.method
  - metadata.time_bucket
  - metadata.cluster_id
  - metadata.zone_id
  - measurements.{demand_count, avg_fare, avg_distance, ...}
  - created_at

### 3.2 Recommended Shard Key

Recommended balanced shard key:

- { metadata.method: 1, metadata.time_bucket: 1, hour_bucket: 1, metadata.zone_id: "hashed" }

Rationale:

- method + time_bucket + hour_bucket keeps query routing selective for time-window analytics.
- hashed zone_id prevents single-shard hot spotting under write bursts.

If your dominant queries are only broad time ranges without zone filters, benchmark an alternative shard key before locking in.

### 3.3 Indexes

Create supporting indexes for common filters and sort order:

- { metadata.method: 1, hour_bucket: -1 }
- { metadata.cluster_id: 1, hour_bucket: -1 }
- { metadata.zone_id: 1, hour_bucket: -1 }

Note: Keep index count controlled. Every extra index increases write overhead.

---

## 4. Sharding Initialization

Run once from mongosh with admin privileges:

```javascript
use taxi_db

sh.enableSharding("taxi_db")

sh.shardCollection("taxi_db.taxi_demand", {
  "metadata.method": 1,
  "metadata.time_bucket": 1,
  "hour_bucket": 1,
  "metadata.zone_id": "hashed"
})
```

Confirm distribution:

```javascript
sh.status()
db.taxi_demand.getShardDistribution()
```

---

## 5. Write Pipeline (Separate from Model Training)

### 5.1 Pipeline Contract

Input from feature jobs:

- df_features: zone_id, trip_count, avg_fare, avg_distance, ...
- df_clusters: zone_id, cluster_id
- method_name and time_bucket

Output:

- Upserted hourly documents in MongoDB taxi_demand collection.

### 5.2 Correct Python Pattern

Use managed client lifecycle and idempotent upserts:

```python
from datetime import datetime
import os
from pymongo import MongoClient, UpdateOne


def write_features_to_mongodb(df_features, df_clusters, method_name="method2", time_bucket=60):
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is not set")

    merged = df_features.merge(df_clusters[["zone_id", "cluster_id"]], on="zone_id", how="left")

    operations = []
    now = datetime.utcnow()

    for _, row in merged.iterrows():
        event_ts = row.get("timestamp", now)
        hour_bucket = event_ts.replace(minute=0, second=0, microsecond=0)

        doc = {
            "timestamp": event_ts,
            "hour_bucket": hour_bucket,
            "metadata": {
                "method": method_name,
                "time_bucket": int(time_bucket),
                "cluster_id": int(row["cluster_id"]),
                "zone_id": int(row["zone_id"]),
            },
            "measurements": {
                "demand_count": int(row["trip_count"]),
                "avg_fare": float(row["avg_fare"]),
                "avg_distance": float(row["avg_distance"]),
                "avg_passenger_count": float(row["avg_passenger_count"]),
                "avg_trip_duration": int(row["avg_trip_duration"]),
                "sum_fare": float(row["sum_fare"]),
            },
            "created_at": now,
        }

        identity = {
            "metadata.method": method_name,
            "metadata.time_bucket": int(time_bucket),
            "metadata.zone_id": int(row["zone_id"]),
            "hour_bucket": hour_bucket,
        }

        operations.append(UpdateOne(identity, {"$set": doc}, upsert=True))

    if not operations:
        return {"records_written": 0}

    with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
        result = client["taxi_db"]["taxi_demand"].bulk_write(operations, ordered=False)

    return {
        "records_written": result.upserted_count + result.modified_count,
        "upserted": result.upserted_count,
        "modified": result.modified_count,
    }
```

---

## 6. Read Pipeline for Analytics and Validation

This pipeline is for analytics and reporting, not for per-batch model training reads.

```python
from datetime import datetime
import os
from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference


def read_taxi_demand_by_timerange(method_name, time_bucket, cluster_id, start_time, end_time):
    uri = os.getenv("MONGODB_URI")
    with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
        collection = client.get_database("taxi_db").get_collection(
            "taxi_demand", read_preference=ReadPreference.SECONDARY_PREFERRED
        )

        query = {
            "hour_bucket": {"$gte": start_time, "$lte": end_time},
            "metadata.method": method_name,
            "metadata.time_bucket": int(time_bucket),
            "metadata.cluster_id": int(cluster_id),
        }

        return list(collection.find(query).sort("hour_bucket", -1))
```

---

## 7. Distributed Systems Experiments (Sharding + Replication)

### 7.1 Required Experiments

1. Shard distribution test:
   - Insert high-volume synthetic traffic.
   - Verify reasonably even chunk distribution.
2. Hot shard test:
   - Simulate skewed zone_id writes.
   - Measure imbalance and adjust shard key if needed.
3. Failover test:
   - Trigger primary failover of one shard replica set.
   - Measure write outage and recovery behavior.
4. Read preference test:
   - Compare primary vs secondaryPreferred latency and staleness.

### 7.2 Correct Failover Read Preference Example

```python
from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference


with MongoClient(os.getenv("MONGODB_URI")) as client:
    primary_coll = client.get_database("taxi_db").get_collection(
        "taxi_demand", read_preference=ReadPreference.PRIMARY
    )
    secondary_coll = client.get_database("taxi_db").get_collection(
        "taxi_demand", read_preference=ReadPreference.SECONDARY_PREFERRED
    )

    _ = primary_coll.find_one({"metadata.method": "method2"})
    _ = secondary_coll.find_one({"metadata.method": "method2"})
```

---

## 8. CAP Theorem Positioning (Accurate Framing)

Avoid absolute claims like "MongoDB is always CP".
More accurate for this project:

- With majority writes and primary reads, behavior prioritizes consistency under partitions.
- With secondaryPreferred reads, availability can improve at the cost of potentially stale reads.
- Trade-off depends on read preference and write concern configuration.

Document your chosen settings in the experiment report.

---

## 9. Integration Contract with Main Training Pipeline

Use the following contract so this module remains separate:

1. Feature jobs write hourly aggregates to MongoDB.
2. Optional export job materializes training snapshots to Parquet/DuckDB.
3. Model training consumes snapshots, not online MongoDB queries.
4. Training metadata references snapshot version and export timestamp.

This gives reproducibility and clear ownership boundaries.

---

## 10. Benchmarking Guidance

Measure and report at least:

- Write throughput (documents per second)
- P95 read latency per query pattern
- Chunk balance over time
- Recovery time after shard primary failover
- Impact of index count on write speed

Do not run cleanup queries that can wipe unrelated production-like data. Restrict test cleanup to tagged test records only.

---

## 11. Quick Commands

```bash
# Connectivity check
python scripts/test_mongodb_connection.py

# Initialize sharding (if scripted)
python scripts/01_create_mongodb_schema.py

# Write feature batch into MongoDB
python scripts/04_load_features_mongodb.py

# Run failover and read preference checks
python scripts/test_mongodb_failover.py

# Run benchmark suite
python scripts/benchmark_mongodb.py
```

---

## 12. Final Recommendation

Use MongoDB Atlas sharding for the distributed database module, and keep it decoupled from the main training loop.

This satisfies:

- Distributed systems learning goals (sharding, replication, failover)
- Scalable write/read behavior
- Reproducible ML training via offline snapshots

End of Distributed Database Module
