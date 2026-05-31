# Distributed Database Module - MongoDB Sharded Cluster

This module is isolated from the existing clustering and forecasting pipeline.
It adds a separate distributed database layer for course requirements.

## Deliverable 1: Docker Compose Project Structure

Structure:

- distributed_db/docker-compose.yml
- distributed_db/docker-compose.centralized.yml
- distributed_db/scripts/init-cluster.sh
- distributed_db/scripts/init-cluster.ps1
- distributed_db/scripts/start-cluster.sh
- distributed_db/scripts/generate_sample_trips.py
- distributed_db/scripts/import-data.sh
- distributed_db/scripts/import-data.ps1
- distributed_db/scripts/queries.js
- distributed_db/scripts/run-distributed-queries.sh
- distributed_db/scripts/benchmark-methodology.js

Cluster topology in `docker-compose.yml`:

- 1 Config Server Replica Set (`cfgRS`)
- 3 Shards (`shard1RS`, `shard2RS`, `shard3RS`)
- 1 Mongos Router (`mongo-mongos`)

### Startup Instructions

PowerShell (Windows):

```powershell
cd distributed_db
docker compose up -d
./scripts/init-cluster.ps1
```

Bash:

```bash
cd distributed_db
docker compose up -d
./scripts/init-cluster.sh
```

## Deliverable 2: Cluster Setup Guide

The exact cluster setup commands are executed by `scripts/init-cluster.sh` and `scripts/init-cluster.ps1`.
Core commands applied:

1. Initiate config server replica set:

```javascript
rs.initiate({ _id: "cfgRS", configsvr: true, members: [{ _id: 0, host: "configsvr:27019" }] })
```

2. Initiate each shard replica set:

```javascript
rs.initiate({ _id: "shard1RS", members: [{ _id: 0, host: "shard1:27018" }] })
rs.initiate({ _id: "shard2RS", members: [{ _id: 0, host: "shard2:27018" }] })
rs.initiate({ _id: "shard3RS", members: [{ _id: 0, host: "shard3:27018" }] })
```

3. Add shards through mongos:

```javascript
sh.addShard("shard1RS/shard1:27018")
sh.addShard("shard2RS/shard2:27018")
sh.addShard("shard3RS/shard3:27018")
```

4. Enable sharding and shard collection:

```javascript
sh.enableSharding("taxi_db")
sh.shardCollection("taxi_db.trips", { PULocationID: 1 })
```

## Deliverable 3: Data Import Guide

### A. Generate sample dataset (100k-1M)

```powershell
python distributed_db/scripts/generate_sample_trips.py --rows 100000
python distributed_db/scripts/generate_sample_trips.py --rows 1000000
```

### B. Import into sharded cluster

PowerShell:

```powershell
./distributed_db/scripts/import-data.ps1 -CsvPath distributed_db/data/taxi_sample.csv
```

Bash:

```bash
./distributed_db/scripts/import-data.sh distributed_db/data/taxi_sample.csv
```

Equivalent raw `mongoimport` command:

```bash
docker cp distributed_db/data/taxi_sample.csv mongo-mongos:/tmp/taxi_sample.csv
docker exec mongo-mongos mongoimport --db taxi_db --collection trips --type csv --headerline --file /tmp/taxi_sample.csv
```

### C. Validation queries

```bash
docker exec mongo-mongos mongosh --quiet --eval 'db.getSiblingDB("taxi_db").trips.countDocuments()'
docker exec mongo-mongos mongosh --quiet --eval 'printjson(db.getSiblingDB("taxi_db").trips.findOne({}, {_id:0}))'
```

## Deliverable 4: Distributed Query Examples

Run examples:

```bash
docker cp distributed_db/scripts/queries.js mongo-mongos:/tmp/queries.js
docker exec mongo-mongos mongosh --quiet /tmp/queries.js
```

### 1) Demand Statistics (PULocationID, Hour)

```javascript
[
  {
    $addFields: {
      pickup_hour: {
        $dateToString: {
          format: "%Y-%m-%dT%H:00:00",
          date: { $toDate: "$pickup_datetime" }
        }
      }
    }
  },
  {
    $group: {
      _id: { PULocationID: "$PULocationID", pickup_hour: "$pickup_hour" },
      trip_count: { $sum: 1 },
      avg_trip_distance: { $avg: "$trip_distance" },
      avg_passenger_count: { $avg: "$passenger_count" }
    }
  }
]
```

### 2) OD Statistics (PULocationID, DOLocationID)

```javascript
[
  {
    $group: {
      _id: { PULocationID: "$PULocationID", DOLocationID: "$DOLocationID" },
      trip_count: { $sum: 1 },
      avg_trip_distance: { $avg: "$trip_distance" }
    }
  },
  { $sort: { trip_count: -1 } }
]
```

## Deliverable 5: Performance Evaluation Plan (Methodology Only)

No benchmark results are pre-filled.

### Goal

Compare:

- Centralized MongoDB (`docker-compose.centralized.yml`)
- MongoDB sharded cluster (`docker-compose.yml`)

### Workload

- Same dataset size per round (100k, 500k, 1M)
- Same indexes in both deployments
- Same aggregation pipelines
- 3 repeated runs per query

### Metrics

- Query execution time (ms)
- Aggregation time (ms)
- Explain plan execution stats
- Scalability discussion under increasing data volume

### Procedure

1. Start centralized MongoDB:

```bash
docker compose -f distributed_db/docker-compose.centralized.yml up -d
```

2. Import same dataset to centralized and run timing script:

```bash
docker cp distributed_db/data/taxi_sample.csv mongo-centralized:/tmp/taxi_sample.csv
docker exec mongo-centralized mongoimport --db taxi_db --collection trips --type csv --headerline --file /tmp/taxi_sample.csv
docker cp distributed_db/scripts/benchmark-methodology.js mongo-centralized:/tmp/benchmark.js
docker exec mongo-centralized mongosh --quiet /tmp/benchmark.js
```

3. Start sharded cluster and import same dataset.
4. Run the same benchmark script through mongos:

```bash
docker cp distributed_db/scripts/benchmark-methodology.js mongo-mongos:/tmp/benchmark.js
docker exec mongo-mongos mongosh --quiet /tmp/benchmark.js
```

5. Save raw outputs in `distributed_db/results/` and compare trends.

## Deliverable 6: Course Mapping

This module maps to distributed database concepts as follows:

1. Horizontal Fragmentation
- Collection `taxi_db.trips` is sharded by `{ PULocationID: 1 }`.
- Data is partitioned across shards by location-based key ranges.

2. Data Allocation
- MongoDB places chunks across `shard1`, `shard2`, and `shard3`.
- Allocation and balancing are managed by the cluster metadata/config server.

3. Distributed Query Processing
- Queries are submitted to `mongos`.
- Router targets one or multiple shards based on shard key and query shape.
- Aggregation is executed in distributed fashion and merged at router/coordinator stages.

4. Distributed Database Architecture
- Demonstrates config server, shard replica sets, and mongos routing layer.
- Includes failover behavior at replica set level.

5. NoSQL Distributed Database
- Uses MongoDB document model and distributed sharding mechanisms.
- Supports scalable ingestion and analytical aggregation workloads.

## Shard Key Rationale: `{ PULocationID: 1 }`

Why chosen:

- Common access pattern starts from pickup zone analysis.
- Easy to explain and directly tied to taxi demand hotspots.
- Supports locality-based distributed grouping.

Benefits:

- Straightforward horizontal fragmentation for coursework.
- Efficient routing for predicates filtering on `PULocationID`.
- Simple debugging and explain-plan interpretation.

Limitations:

- Potential skew if some pickup zones dominate traffic.
- Time-based queries without `PULocationID` filter may scatter across shards.
- For production, a compound shard key with time component may be more balanced.

## Notes

- Existing clustering and forecasting modules are untouched.
- This module is fully separate under `distributed_db/`.
