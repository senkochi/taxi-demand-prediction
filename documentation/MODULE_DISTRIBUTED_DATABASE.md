# 🗄️ Module: Distributed Database - Cassandra for Taxi Time-Series

**Owner:** Person A (with Person B support) | **Duration:** Weeks 1-2, 3-5 | **Deliverables:** Cassandra schema, write/read pipelines, performance analysis

---

## 1. Cassandra Fundamentals for This Project

### 1.1 Why Cassandra for Taxi Demand?

**Our Time-Series Data:**
```
For each (cluster_id, time_bucket, zone_id):
  - demand_count: integer
  - avg_fare: float
  - avg_distance: float
  - ... (7+ metrics)
  
Characteristics:
- Write-heavy: millions of taxi events per day
- Time-ordered: queries by time range
- Distributed: need horizontal scaling
- Fault-tolerant: high availability needed
```

**Cassandra Advantages:**
```
✅ Consistent hashing: spreads writes evenly (no hotspots)
✅ Time-series ready: clustering keys for range queries
✅ High availability: write survives node failures
✅ Tunable consistency: balance consistency vs availability
✅ Linear scalability: add nodes = more throughput
```

### 1.2 CAP Theorem: Why Cassandra is AP

```
CAP Theorem: Pick 2 of 3
  - Consistency: all nodes see same data
  - Availability: system always responsive
  - Partition tolerance: survives network splits

Cassandra = AP (Availability + Partition tolerance)
  - When network splits: writes succeed on both sides
  - Replication resolves conflicts later (eventual consistency)
  
For taxi data:
  ✅ Slightly stale demand counts acceptable
  ✅ High availability critical (real-time dashboards)
  ✅ Write availability > read consistency
```

---

## 2. Local Cassandra Setup (Docker)

### 2.1 Docker Compose Configuration

```yaml
# docker/docker-compose.yml - Cassandra section

version: '3.9'
services:
  
  # Cassandra cluster: 3 nodes for replication testing
  cassandra-1:
    image: cassandra:4.0
    container_name: cassandra-node-1
    environment:
      CASSANDRA_CLUSTER_NAME: "taxi-cluster"
      CASSANDRA_DC: "us-east-1"
      CASSANDRA_RACK: "rack1"
      CASSANDRA_SEEDS: "cassandra-1"  # Bootstrap node
    ports:
      - "9042:9042"  # CQL port (client)
      - "7000:7000"  # Inter-node communication
    volumes:
      - cassandra-1-data:/var/lib/cassandra
    networks:
      - taxi-network
    healthcheck:
      test: ["CMD", "cqlsh", "-e", "SELECT 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  cassandra-2:
    image: cassandra:4.0
    container_name: cassandra-node-2
    environment:
      CASSANDRA_CLUSTER_NAME: "taxi-cluster"
      CASSANDRA_DC: "us-east-1"
      CASSANDRA_RACK: "rack1"
      CASSANDRA_SEEDS: "cassandra-1"
    depends_on:
      cassandra-1:
        condition: service_healthy
    volumes:
      - cassandra-2-data:/var/lib/cassandra
    networks:
      - taxi-network

  cassandra-3:
    image: cassandra:4.0
    container_name: cassandra-node-3
    environment:
      CASSANDRA_CLUSTER_NAME: "taxi-cluster"
      CASSANDRA_DC: "us-east-1"
      CASSANDRA_RACK: "rack2"
      CASSANDRA_SEEDS: "cassandra-1"
    depends_on:
      cassandra-1:
        condition: service_healthy
    volumes:
      - cassandra-3-data:/var/lib/cassandra
    networks:
      - taxi-network

volumes:
  cassandra-1-data:
  cassandra-2-data:
  cassandra-3-data:

networks:
  taxi-network:
    driver: bridge
```

### 2.2 Setup Commands

```bash
# Start Cassandra cluster
docker-compose -f docker/docker-compose.yml up -d cassandra-1 cassandra-2 cassandra-3

# Wait for cluster to stabilize (~30 seconds)
sleep 30

# Verify cluster status
docker exec cassandra-node-1 nodetool status

# Expected output:
# Datacenter: us-east-1
# Status=Up/Down, State=Normal/Leaving/Joining/Moving
# --  Address      Load       Tokens  Owns    Host ID
# UN  172.19.0.2   104.5 KB   256     33.3%   abcd1234
# UN  172.19.0.3   108.3 KB   256     33.3%   efgh5678
# UN  172.19.0.4   102.1 KB   256     33.3%   ijkl9012

# Connect to CQL shell
docker exec -it cassandra-node-1 cqlsh
```

---

## 3. Schema Design for Taxi Demand

### 3.1 Keyspace (Database) Creation

```cql
-- Create keyspace with replication
CREATE KEYSPACE IF NOT EXISTS taxi_db
WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 3
};

-- Verify replication
DESCRIBE KEYSPACE taxi_db;
```

**Replication Factor = 3:**
- Each data item stored on 3 nodes
- Can lose 2 nodes and still have data
- Example: If network splits (2 nodes vs 1), both sides have data

### 3.2 Main Table: taxi_demand

```cql
USE taxi_db;

-- Partition key: (method, time_bucket, cluster_id)
-- Clustering key: (timestamp)
-- This allows efficient range queries by time

CREATE TABLE IF NOT EXISTS taxi_demand (
  method TEXT,                    -- baseline, method1, method2, method3
  time_bucket INT,                -- 15, 30, or 60 (minutes)
  cluster_id INT,                 -- 0-K clusters per method
  timestamp BIGINT,               -- Unix timestamp (seconds)
  zone_id INT,                    -- Original zone ID
  
  -- Demand metrics
  demand_count INT,               -- Number of trips
  avg_fare FLOAT,                 -- Average fare
  avg_distance FLOAT,             -- Average trip distance
  avg_passenger_count FLOAT,      -- Average passengers
  avg_trip_duration INT,          -- Average duration (seconds)
  sum_fare DOUBLE,                -- Total fare
  
  -- Metadata
  record_date TEXT,               -- YYYY-MM-DD for date-based queries
  created_at TIMESTAMP,           -- When record was inserted
  
  PRIMARY KEY ((method, time_bucket, cluster_id), timestamp)
) WITH 
  CLUSTERING ORDER BY (timestamp DESC) AND
  compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1};

-- Index for zone-based queries
CREATE INDEX IF NOT EXISTS idx_zone_id ON taxi_demand(zone_id);

-- Index for date-based queries (for data retention)
CREATE INDEX IF NOT EXISTS idx_record_date ON taxi_demand(record_date);
```

**Key Design Decisions:**

```
Partition Key: (method, time_bucket, cluster_id)
  Why: Spreads writes across nodes
       Different methods/buckets/clusters write to different partitions
       No hotspots (vs if we used just method or timestamp)

Clustering Key: (timestamp DESC)
  Why: Range queries by time (e.g., "get demand for March")
       DESC order = newest first (typical time-series query)

TWCS (Time Window Compaction):
  Why: Optimized for time-series data
       Automatically deletes old data (TTL-friendly)
       Better read performance for time ranges
```

### 3.3 Index Table: cluster_metadata

```cql
CREATE TABLE IF NOT EXISTS cluster_metadata (
  method TEXT,
  cluster_id INT,
  cluster_name TEXT,              -- downtown, airport, residential
  zone_list LIST<INT>,            -- Zones in this cluster
  silhouette_score FLOAT,         -- Cluster quality metric
  num_zones INT,
  
  PRIMARY KEY (method, cluster_id)
);
```

---

## 4. Write Pipeline: Spark → Cassandra

### 4.1 Spark Configuration for Cassandra

```python
# src/data/cassandra_writer.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, unix_timestamp, lit

def create_spark_cassandra_config():
    """Create Spark config for Cassandra connection"""
    return {
        "spark.cassandra.connection.host": "localhost",
        "spark.cassandra.connection.port": "9042",
        "spark.cassandra.connection.keep_alive_ms": "5000",
        
        # Read/Write optimization
        "spark.cassandra.input.split.size_in_mb": "64",
        "spark.cassandra.output.batch.grouping.key": "partition",
        "spark.cassandra.output.concurrent.writes": "5",
        
        # Consistency level for writes
        "spark.cassandra.output.consistency.level": "QUORUM",  # Write to majority
    }

def get_spark_session():
    """Get Spark session with Cassandra connector"""
    
    spark = SparkSession.builder \
        .appName("Taxi-Cassandra-Pipeline") \
        .config("spark.jars.packages", 
                "com.datastax.spark:spark-cassandra-connector_2.12:3.1.0") \
        .getOrCreate()
    
    for key, value in create_spark_cassandra_config().items():
        spark.conf.set(key, value)
    
    return spark
```

### 4.2 Write Function: Method 2 Features to Cassandra

```python
# scripts/write_method2_to_cassandra.py

from pyspark.sql.functions import current_timestamp, lit, unix_timestamp
from src.data.cassandra_writer import get_spark_session

def write_method2_features_to_cassandra(
    df_features,  # DataFrame with method2 features
    df_clusters,  # DataFrame with cluster assignments
    method_name="method2",
    time_bucket=60,
    consistency_level="QUORUM"
):
    """
    Write Method 2 features to Cassandra.
    
    Args:
        df_features: [zone_id, trip_count, avg_distance, avg_fare, ...]
        df_clusters: [zone_id, cluster_id]
        method_name: "baseline", "method1", "method2", "method3"
        time_bucket: 15, 30, or 60 minutes
        consistency_level: ONE, LOCAL_ONE, QUORUM, LOCAL_QUORUM, ALL
    
    Consistency Levels:
        ONE: Write succeeds if 1 node acknowledges (fast, risky)
        QUORUM: Write succeeds if majority nodes ack (balanced)
        ALL: Write succeeds only if all replicas ack (slow, safe)
    """
    
    spark = get_spark_session()
    
    # Join features with cluster assignments
    df_combined = df_features.join(
        df_clusters,
        on="zone_id",
        how="left"
    )
    
    # Add metadata columns
    current_time = int(time.time())  # Unix timestamp
    
    df_to_write = df_combined.select(
        lit(method_name).alias("method"),
        lit(time_bucket).alias("time_bucket"),
        col("cluster_id"),
        lit(current_time).alias("timestamp"),  # Current write timestamp
        col("zone_id"),
        
        # Demand metrics (cast to correct types)
        col("trip_count").cast("int").alias("demand_count"),
        col("avg_fare").cast("float").alias("avg_fare"),
        col("avg_distance").cast("float").alias("avg_distance"),
        col("avg_passenger_count").cast("float").alias("avg_passenger_count"),
        col("avg_trip_duration").cast("int").alias("avg_trip_duration"),
        col("sum_fare").cast("double").alias("sum_fare"),
        
        # Metadata
        from_unixtime(lit(current_time), "yyyy-MM-dd").alias("record_date"),
        current_timestamp().alias("created_at")
    )
    
    # Write to Cassandra
    df_to_write.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "taxi_db") \
        .option("table", "taxi_demand") \
        .option("spark.cassandra.output.consistency.level", consistency_level) \
        .save()
    
    print(f"✅ Written {df_to_write.count()} records to Cassandra")
    print(f"   Method: {method_name}, Time Bucket: {time_bucket}min, Consistency: {consistency_level}")
    
    # Log write statistics
    return {
        "method": method_name,
        "time_bucket": time_bucket,
        "records_written": df_to_write.count(),
        "consistency": consistency_level,
        "timestamp": current_time
    }
```

---

## 5. Read Pipeline: Cassandra → Analysis

### 5.1 Read with Different Consistency Levels

```python
# src/data/cassandra_reader.py

def read_taxi_demand_by_timerange(
    method_name,
    time_bucket,
    cluster_id,
    start_time,  # Unix timestamp
    end_time,
    consistency_level="LOCAL_ONE"
):
    """
    Read taxi demand data from Cassandra for time range.
    
    Consistency Levels:
        LOCAL_ONE: Read from nearest replica (fast, may be stale)
        QUORUM: Read from majority (balanced)
        ALL: Read from all replicas, compare for conflicts (slow, fresh)
    """
    
    spark = get_spark_session()
    
    # Cassandra allows push-down filtering on clustering keys
    query = f"""
    SELECT *
    FROM taxi_db.taxi_demand
    WHERE method = '{method_name}'
    AND time_bucket = {time_bucket}
    AND cluster_id = {cluster_id}
    AND timestamp >= {start_time}
    AND timestamp <= {end_time}
    """
    
    df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "taxi_db") \
        .option("table", "taxi_demand") \
        .option("spark.cassandra.input.consistency.level", consistency_level) \
        .load() \
        .filter(col("timestamp").between(start_time, end_time))
    
    return df

def read_cluster_statistics(method_name, consistency_level="LOCAL_QUORUM"):
    """Read cluster metadata"""
    
    spark = get_spark_session()
    
    df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "taxi_db") \
        .option("table", "cluster_metadata") \
        .option("spark.cassandra.input.consistency.level", consistency_level) \
        .load() \
        .filter(col("method") == method_name)
    
    return df
```

---

## 6. Distributed System Analysis

### 6.1 Partition Distribution

```
Cassandra Consistent Hashing:

Partition Key: (method, time_bucket, cluster_id)
Hash Value Range: [0, 2^127 - 1]

Token Ring (for 3 nodes):
  Node 1: Token range [0, 4294967296)
  Node 2: Token range [4294967296, 8589934592)
  Node 3: Token range [8589934592, 12884901888)

Data Distribution:
  method1_15_0 → hash = 1234567 → Node 1 + replicas on Node 2, 3
  method1_15_1 → hash = 9876543 → Node 2 + replicas on Node 3, 1
  method1_15_2 → hash = 5555555 → Node 3 + replicas on Node 1, 2
  
  ✅ Even distribution (no hotspots)
  ✅ Replication factor 3 = all data on all nodes (for small datasets)
```

### 6.2 Fault Tolerance Testing

```python
# scripts/test_cassandra_fault_tolerance.py

def test_write_with_node_failure():
    """Simulate write during node failure"""
    
    # Before failure: 3 nodes, RF=3 (all healthy)
    write_consistency = "QUORUM"  # Needs 2 acks out of 3
    
    # Scenario: Kill node 1
    # docker stop cassandra-node-1
    
    # Now: 2 nodes active
    # QUORUM write still succeeds (needs 2/3 acks, have 2/3)
    
    # Write data
    write_method2_features_to_cassandra(
        df_features, df_clusters,
        consistency_level="QUORUM"
    )
    # ✅ Still succeeds!
    
    # Scenario: Kill node 2 as well
    # docker stop cassandra-node-2
    
    # Now: 1 node active
    # QUORUM write FAILS (needs 2/3 acks, have 1/3)
    # But data is still readable from 1 node!
    
    # Read with LOCAL_ONE
    df = read_taxi_demand_by_timerange(
        "method2", 60, 0, start_time, end_time,
        consistency_level="LOCAL_ONE"
    )
    # ✅ Still succeeds from remaining node!
    
    # Restart nodes
    # docker start cassandra-node-1 cassandra-node-2
    
    # Hinted handoff + read repair automatically sync data back
    print("✅ High availability verified: system survives 2/3 node failures")
```

### 6.3 Consistency vs Availability Trade-off

```python
# Experiments for thesis

def analyze_consistency_performance():
    """Compare write latency vs consistency level"""
    
    consistency_levels = ["ONE", "LOCAL_ONE", "LOCAL_QUORUM", "QUORUM", "ALL"]
    write_latencies = []
    
    for consistency in consistency_levels:
        start = time.time()
        
        write_method2_features_to_cassandra(
            df_features, df_clusters,
            consistency_level=consistency
        )
        
        latency = time.time() - start
        write_latencies.append({
            "consistency": consistency,
            "write_latency_ms": latency * 1000,
            "description": get_consistency_description(consistency)
        })
    
    # Results show trade-off:
    # ONE: ~5ms (fastest, least safe)
    # LOCAL_ONE: ~10ms
    # LOCAL_QUORUM: ~25ms (balanced)
    # QUORUM: ~40ms
    # ALL: ~100ms (slowest, most safe)
    
    return write_latencies

def get_consistency_description(level):
    descriptions = {
        "ONE": "1 node ack (fast, risky for failure)",
        "LOCAL_ONE": "1 local DC node ack (slightly safer)",
        "LOCAL_QUORUM": "Majority in local DC (balanced)",
        "QUORUM": "Majority across all DCs (safe)",
        "ALL": "All replicas ack (safest, slowest)"
    }
    return descriptions.get(level, "Unknown")
```

---

## 7. Performance Monitoring

### 7.1 Cassandra Metrics

```bash
# Check cluster status
docker exec cassandra-node-1 nodetool status

# Check read/write latency
docker exec cassandra-node-1 nodetool cfstats taxi_db.taxi_demand

# Watch real-time metrics
docker exec cassandra-node-1 nodetool tpstats

# Check data distribution
docker exec cassandra-node-1 nodetool ring taxi_db
```

### 7.2 Python Monitoring

```python
# src/monitoring/cassandra_metrics.py

from cassandra.cluster import Cluster
from cassandra.metrics import Metrics

def get_cassandra_metrics(contact_points=['localhost']):
    """Get detailed Cassandra metrics"""
    
    cluster = Cluster(contact_points=contact_points)
    session = cluster.connect()
    
    # Get read/write statistics
    rows = session.execute("""
        SELECT keyspace_name, table_name, 
               local_read_latency_ms, local_write_latency_ms,
               pending_tasks, dropped_mutations
        FROM system.system_virtual_schema_tables
        WHERE keyspace_name = 'taxi_db'
    """)
    
    metrics = {}
    for row in rows:
        metrics[f"{row.keyspace_name}.{row.table_name}"] = {
            "read_latency_ms": row.local_read_latency_ms,
            "write_latency_ms": row.local_write_latency_ms,
            "pending_tasks": row.pending_tasks,
            "dropped_mutations": row.dropped_mutations
        }
    
    return metrics
```

---

## 8. Thesis Topics: Distributed Database Module

### 8.1 Report Structure (20-30 pages)

**Part 1: Architecture & Design (7 pages)**
- Cassandra architecture: ring topology, consistent hashing
- Replication strategy: RF=3, replication path
- Partition design: why (method, time_bucket, cluster_id)?
- Clustering order: why DESC on timestamp?

**Part 2: Distributed System Properties (8 pages)**
- CAP Theorem: why AP for this use case?
- Consistency models: eventual vs strong
- Replication & fault tolerance
- Write path: coordinator → nodes
- Read path: quorum consistency

**Part 3: Performance & Scalability (8 pages)**
- Write throughput: millions of events/day
- Read latency: time-range queries
- Consistency level trade-offs
- Horizontal scaling: adding nodes
- Comparison: single-node vs 3-node

**Part 4: Fault Tolerance & Recovery (5 pages)**
- Node failure scenarios
- Hinted handoff & repair
- Data consistency after failures
- TTL & data compaction
- Disaster recovery

### 8.2 Experiments for Thesis

```
Experiment 1: Write Throughput
- Vary replication factor: 1, 2, 3
- Measure: writes/second
- Result: RF=3 reduces throughput ~30% but enables HA

Experiment 2: Fault Tolerance
- Kill nodes 1 at a time
- Measure: write/read availability
- Result: 2 node failures acceptable with RF=3

Experiment 3: Consistency Levels
- Compare ONE vs QUORUM vs ALL
- Measure: write latency, data safety
- Result: QUORUM good balance

Experiment 4: Read Latency vs Time Range
- Query different time ranges: 1 day, 1 week, 1 month
- Measure: query time
- Result: clustering order (DESC) crucial for performance
```

---

## 9. Integration with ML Pipeline

### 9.1 Training Data Source

```python
# During model training

# Option A: Read from Cassandra with consistency
def load_train_data_cassandra(method_name, consistency="LOCAL_ONE"):
    df = read_taxi_demand_by_timerange(
        method_name, 60, 0,
        start_time=unix_timestamp("2023-01-01"),
        end_time=unix_timestamp("2023-11-01"),
        consistency_level=consistency
    )
    return df.to_pandas()

# Option B: Read from Cassandra or Parquet (fallback)
def load_train_data_hybrid(method_name):
    try:
        # Try Cassandra first (distributed)
        df = load_train_data_cassandra(method_name)
    except Exception as e:
        print(f"Cassandra read failed: {e}, falling back to Parquet")
        # Fall back to Parquet
        df = load_train_data_parquet(method_name)
    
    return df
```

---

## 10. Quick Commands

```bash
# Start cluster
docker-compose -f docker/docker-compose.yml up -d

# Connect to CQL
docker exec -it cassandra-node-1 cqlsh

# Create schema
cqlsh> source 'scripts/cassandra_schema.cql'

# Check cluster
docker exec cassandra-node-1 nodetool status

# Write data
python scripts/write_method2_to_cassandra.py

# Read data
python scripts/read_taxi_demand.py

# Monitor
docker exec cassandra-node-1 nodetool tpstats

# Stop cluster
docker-compose down
```

---

## 11. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection timeout | Cassandra not ready | Wait 30s after docker-compose up |
| QUORUM write fails | <RF/2 nodes alive | Reduce consistency to ONE/LOCAL_ONE |
| Query slow | Large time range | Use clustering key in WHERE clause |
| High memory | Default heap too small | Increase in docker-compose env |
| Data not replicated | RF not set correctly | Check DESCRIBE KEYSPACE |

---

## 12. Prompt Templates for AI

```
"Implement Cassandra write pipeline for Method [X] features.
 
 Schema:
 - Table: taxi_demand
 - Partition Key: (method, time_bucket, cluster_id)
 - Clustering Key: timestamp DESC
 - Replication Factor: 3
 
 Input: DataFrame with columns [zone_id, demand_count, avg_fare, ...]
 Output: Written to Cassandra with QUORUM consistency
 
 Include: error handling, logging, retry logic"

"Design CQL queries for taxi demand analysis:
 1. Get demand for cluster X in time range Y-Z
 2. Get all clusters for method M on date D
 3. Compare demand across all 4 methods
 
 Include: indexes, performance hints, consistency levels"

"Analyze CAP theorem trade-offs for taxi system.
 - Why AP (availability + partition tolerance)?
 - What consistency level for different operations?
 - How replication handles node failures?"
```

---

**End of Distributed Database Module Guide**

Next: Update PROJECT_ROADMAP, DEVELOPMENT_SETUP, and EXPERIMENT_REPORTING to include Cassandra integration.
