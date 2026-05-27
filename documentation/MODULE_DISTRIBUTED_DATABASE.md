# 🗄️ Module: Cloud Database - MongoDB Atlas for Taxi Time-Series
# 🗄️ Module: Distributed Database - Cassandra for Taxi Time-Series

**⚠️ IMPORTANT:** This is a project for 2 subjects: **Big Data** and **Distributed Database**

**Owner:** Person A (with Person B support) | **Duration:** Weeks 1-2 (FAST!), 3-5 | **Deliverables:** MongoDB Atlas setup, write/read pipelines, performance analysis

---

## 1. MongoDB Atlas vs Self-Hosted: Why We Choose Atlas (SPEED!)

### 1.1 For Time-Series Taxi Data (Quick Setup)

**Our Time-Series Data:**
```
For each (method, time_bucket, cluster_id):
  - demand_count: integer
  - avg_fare: float
  - avg_distance: float
  - ... (7+ metrics)
  
Characteristics:
- Write-heavy: millions of taxi events per day
- Time-ordered: queries by time range
- Distributed: need automatic failover
- Fault-tolerant: high availability needed
- FAST SETUP: One-click provisioning needed (1 week deadline!)
```

**MongoDB Atlas Advantages:**
```
✅ No infrastructure: Fully managed, 5-minute setup
✅ Time-series ready: Native time-series collections  
✅ High availability: Automatic multi-region failover
✅ Tunable consistency: Choice of read preferences
✅ Auto-scaling: Handles growth automatically
✅ Free tier: Perfect for development/thesis work
```

### 1.2 CAP Theorem: MongoDB for Distributed Systems Thesis

**Replica Sets (Not Sharding):**
```
CAP Theorem: Pick 2 of 3
  - Consistency: all replicas see same data
  - Availability: system always responsive
  - Partition tolerance: survives network splits

MongoDB (Replica Set) = CP + Partition Tolerance
  - Strong consistency by default (PRIMARY writes)
  - Automatic failover: new PRIMARY elected instantly
  - Multi-region deployment: test partition scenarios
  
For taxi data + thesis:
  ✅ Consistency important: demand predictions need accuracy
  ✅ High availability critical: real-time dashboards
  ✅ Automatic failover saves week's worth of setup time
  ✅ Multi-region tests distributed system concepts
```

---

## 2. MongoDB Atlas Setup (5 Minutes!)

### 2.1 Quick Start

**Step 1: Create Free Cluster (2 min)**
```
1. Go to https://www.mongodb.com/cloud/atlas
2. Click "Try Free"
3. Create Account (use Google/GitHub for SPEED)
4. Create Organization → Create Project  
5. Build Database → Choose Shared Tier (FREE)
6. Select Cloud Provider: AWS
7. Region: us-east-1
8. Create Cluster → DONE (5-10 min provisioning)
```

**Step 2: Network Access (1 min)**
```
In Atlas Dashboard:
  Security → Network Access
  Add IP: 0.0.0.0/0  (dev only)
  OR: Your specific IP
```

**Step 3: Get Connection String (1 min)**
```
Cluster → Connect → Connect your application
Copy: mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/taxi_db
```

**Step 4: Store in .env (30 sec)**
```bash
# .env
MONGODB_URI=mongodb+srv://your_user:your_pass@your_cluster.mongodb.net/taxi_db?retryWrites=true&w=majority
```

✅ **TOTAL TIME: 5 minutes. Start coding immediately.**

---

## 3. Database Schema for Taxi Demand

### 3.1 Create Collections (1 minute)

```python
# scripts/01_create_mongodb_schema.py

from pymongo import MongoClient
from datetime import datetime
import os

def create_taxi_demand_schema():
    """Create MongoDB collections and indexes"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['taxi_db']
    
    # Create time-series collection
    try:
        db.create_collection(
            "taxi_demand",
            timeseries={
                "timeField": "timestamp",
                "metaField": "metadata",
                "granularity": "hours"
            }
        )
        print("✓ Time-series collection created")
    except:
        print("✓ Collection already exists")
    
    # Create metadata collection
    try:
        db.create_collection("cluster_metadata")
    except:
        pass
    
    # Create indexes for speed
    db.taxi_demand.create_index([("metadata.method", 1), ("timestamp", -1)])
    db.taxi_demand.create_index([("metadata.cluster_id", 1)])
    db.taxi_demand.create_index([("timestamp", 1)])  # For TTL
    
    db.cluster_metadata.create_index([("method", 1), ("cluster_id", 1)], unique=True)
    
    print("✓ All indexes created")
    client.close()
```

### 3.2 Document Schema

**Time-Series Document:**
```javascript
{
  "timestamp": ISODate("2024-03-15T14:30:00Z"),
  "metadata": {
    "method": "method2",
    "time_bucket": 60,
    "cluster_id": 5,
    "zone_id": 42
  },
  "measurements": {
    "demand_count": 1250,
    "avg_fare": 14.85,
    "avg_distance": 2.3,
    "avg_passenger_count": 1.8,
    "avg_trip_duration": 850,
    "sum_fare": 18562.50
  },
  "created_at": ISODate("2024-03-15T14:35:00Z")
}
```

---

## 4. Write Pipeline: Pandas/PySpark → MongoDB

### 4.1 Write Function (Super Simple!)

```python
# scripts/write_method2_to_mongodb.py

from pymongo import MongoClient
from datetime import datetime
import os

def write_method2_features_to_mongodb(
    df_features,    # [zone_id, trip_count, avg_fare, ...]
    df_clusters,    # [zone_id, cluster_id]
    method_name="method2",
    time_bucket=60
):
    """Write to MongoDB - faster than Cassandra setup!"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['taxi_demand']
    
    # Join data
    df_combined = df_features.merge(
        df_clusters[['zone_id', 'cluster_id']],
        on='zone_id',
        how='left'
    )
    
    # Build documents
    current_time = datetime.utcnow()
    documents = []
    
    for _, row in df_combined.iterrows():
        doc = {
            "timestamp": current_time,
            "metadata": {
                "method": method_name,
                "time_bucket": time_bucket,
                "cluster_id": int(row['cluster_id']),
                "zone_id": int(row['zone_id'])
            },
            "measurements": {
                "demand_count": int(row['trip_count']),
                "avg_fare": float(row['avg_fare']),
                "avg_distance": float(row['avg_distance']),
                "avg_passenger_count": float(row['avg_passenger_count']),
                "avg_trip_duration": int(row['avg_trip_duration']),
                "sum_fare": float(row['sum_fare'])
            },
            "created_at": current_time
        }
        documents.append(doc)
    
    # Bulk insert
    if documents:
        result = collection.insert_many(documents)
        print(f"✅ Inserted {len(result.inserted_ids)} records to MongoDB")
        return {"records_written": len(result.inserted_ids), "timestamp": current_time.isoformat()}
    
    client.close()
```

---

## 5. Read Pipeline: MongoDB → Analysis

### 5.1 Read Functions

```python
# src/data/mongodb_reader.py

from pymongo import MongoClient
from datetime import datetime
import os

def read_taxi_demand_by_timerange(
    method_name,
    time_bucket,
    cluster_id,
    start_time,
    end_time,
    read_preference="primary"
):
    """Read from MongoDB"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['taxi_demand']
    
    query = {
        "timestamp": {"$gte": start_time, "$lte": end_time},
        "metadata.method": method_name,
        "metadata.time_bucket": time_bucket,
        "metadata.cluster_id": cluster_id
    }
    
    results = list(collection.find(query).sort("timestamp", -1))
    client.close()
    return results


def read_cluster_metadata(method_name):
    """Read cluster info"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['cluster_metadata']
    
    results = list(collection.find({"method": method_name}))
    client.close()
    return results
```

### 5.2 Aggregation Pipeline (Fast Queries)

```python
def aggregate_demand_by_hour(method_name, start_date, end_date):
    """Aggregate using MongoDB pipeline (server-side)"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['taxi_demand']
    
    pipeline = [
        {"$match": {
            "metadata.method": method_name,
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }},
        {"$group": {
            "_id": {
                "year": {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"},
                "day": {"$dayOfMonth": "$timestamp"},
                "hour": {"$hour": "$timestamp"}
            },
            "total_demand": {"$sum": "$measurements.demand_count"},
            "avg_fare": {"$avg": "$measurements.avg_fare"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": -1}}
    ]
    
    results = list(collection.aggregate(pipeline))
    client.close()
    return results
```

---

## 6. Distributed Systems Analysis

### 6.1 Multi-Region Failover Testing

```python
# scripts/test_mongodb_failover.py

from pymongo import MongoClient
from datetime import datetime
import os

def test_replica_set_failover():
    """Test automatic failover in MongoDB replica set"""
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['taxi_demand']
    
    print("Testing MongoDB Replica Set Failover...")
    print("1. Insert test document...")
    
    test_doc = {
        "timestamp": datetime.utcnow(),
        "test": "failover_test",
        "metadata": {"method": "test"}
    }
    
    result = collection.insert_one(test_doc)
    print(f"   ✓ Document inserted")
    
    print("\n2. Reading with different preferences...")
    
    # PRIMARY: always fresh data
    client.read_preference = "primary"
    doc = collection.find_one({"_id": result.inserted_id})
    print(f"   ✓ PRIMARY read succeeded")
    
    # SECONDARY: may be stale, available even if PRIMARY is down
    client.read_preference = "secondary"
    doc = collection.find_one({"_id": result.inserted_id})
    print(f"   ✓ SECONDARY read succeeded (may be 1-5s stale)")
    
    # Clean up
    collection.delete_one({"_id": result.inserted_id})
    client.close()
    
    print("\n✅ Failover test complete")
```

### 6.2 Performance Benchmarks

```python
def benchmark_mongodb(num_documents=10000):
    """Benchmark write/read performance"""
    
    import time
    client = MongoClient(os.getenv('MONGODB_URI'))
    collection = client['taxi_db']['taxi_demand']
    
    # Write benchmark
    docs = [{"timestamp": datetime.utcnow(), "value": i} for i in range(num_documents)]
    
    start = time.time()
    collection.insert_many(docs)
    write_time = time.time() - start
    
    print(f"Write Performance: {num_documents / write_time:.0f} docs/sec")
    
    # Read benchmark
    start = time.time()
    list(collection.find({"timestamp": {"$gte": datetime.utcnow()}}).limit(1000))
    read_time = time.time() - start
    
    print(f"Read Performance: {1000 / read_time:.0f} docs/sec")
    
    collection.delete_many({"timestamp": {"$exists": True}})
    client.close()
```

---

## 7. Quick Comparison: Cassandra vs MongoDB Atlas

| Aspect | Cassandra (Docker) | MongoDB Atlas | ⭐ Winner |
|--------|-------------------|---------------|----------|
| **Setup Time** | 45-60 min | 5 min | 🏃 **Atlas** |
| **Infrastructure** | Docker + 3 nodes | Fully managed | 🏃 **Atlas** |
| **High Availability** | Manual config | Automatic | 🏃 **Atlas** |
| **Time-Series Support** | Schema tricks | Native | ⭐ **Atlas** |
| **Cost** | Free (Docker) | Free tier or $57/mo | ⭐ **Tie** |
| **Learning Curve** | Steep | Shallow | 🏃 **Atlas** |
| **For 1-week deadline** | ❌ Too much setup | ✅✅ **Ready NOW** | 🏃 **Atlas** |

---

## 8. Thesis Topics: Cloud Database Module

**Focus:** Distributed systems concepts without infrastructure overhead

### 8.1 Replica Set Architecture (Thesis Material)

```
MongoDB Replica Set (3 nodes):

  PRIMARY → replicates to → SECONDARY 1
           ├──────────────→ SECONDARY 2
  
Write Flow:
  1. Client writes to PRIMARY
  2. PRIMARY acknowledges to client
  3. PRIMARY replicates to SECONDARYs in background
  4. SECONDARYs acknowledge replication

Read Flow (read_preference = primary):
  1. Client reads from PRIMARY → always fresh
  
Read Flow (read_preference = secondary):
  1. Client reads from SECONDARY → faster, ~1-5s stale

Failover Flow:
  1. If PRIMARY dies → Heartbeat fails
  2. SECONDARYs detect failure (10 seconds)
  3. SECONDARYs elect new PRIMARY via voting
  4. New PRIMARY starts accepting writes
  ✅ Automatic, no manual intervention!
```

### 8.2 CAP Theorem in Practice

```
Your Experiments:

1. Consistency Experiment
   - Write to PRIMARY with w:"majority"
   - Kill SECONDARY nodes
   - Verify write still succeeds (quorum)
   
2. Availability Experiment  
   - Kill PRIMARY node
   - Measure failover time (~10 seconds)
   - Verify reads available on SECONDARY
   
3. Partition Tolerance Experiment
   - Network partition: PRIMARY vs SECONDARY networks
   - PRIMARY can't reach SECONDARY
   - Monitor: write succeeds on PRIMARY (CP behavior)
   - Monitor: read fails on SECONDARY side

Result: MongoDB is CP (Consistency + Partition Tolerance)
  - Not fully Available during partitions
  - But automatic failover is fast
```

---

## 9. Integration with ML Training

```python
# Simple data loading

def load_train_data(method_name):
    """Load data for model training"""
    
    results = read_taxi_demand_by_timerange(
        method_name, 60, 0,
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 11, 1),
        read_preference="primary"  # Always fresh for training
    )
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "cluster_id": r["metadata"]["cluster_id"],
            "timestamp": r["timestamp"],
            **r["measurements"]
        }
        for r in results
    ])
    
    return df
```

---

## 10. Quick Commands

```bash
# Verify connection
python scripts/test_mongodb_connection.py

# Create schema
python scripts/01_create_mongodb_schema.py

# Write data
python scripts/write_method2_to_mongodb.py

# Test failover
python scripts/test_mongodb_failover.py

# Benchmark
python scripts/benchmark_mongodb.py
```

---

## 11. Why MongoDB Atlas Saves Your Week

```
Cassandra Approach (NOT HAPPENING):
  Day 1: Docker setup (1.5 days)
  Day 2-3: Schema design, CQL
  Day 4: Debug clustering, replication  
  Day 5: Fix Spark connector issues
  = 5 days of infrastructure = NO TIME FOR MODEL TRAINING

MongoDB Atlas Approach (THIS WEEK):
  Day 1: Create account + cluster (5 min)
  Day 1: Write schema script (10 min)
  Day 1: Start training (still Day 1!)
  Days 2-7: Focus on MODEL QUALITY, not infrastructure
  = 1 week focused on ML = RESULTS!
```

✅ **RECOMMENDATION: Use MongoDB Atlas for this project. Ship fast.**

---

**End of Distributed Database Module**
