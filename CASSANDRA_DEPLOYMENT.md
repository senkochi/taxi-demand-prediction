# 🚀 Deployment Guide: Pandas + Cassandra

**Status:** Ready to deploy  
**Date:** 2026-05-16

---

## 📋 **Prerequisite**

- Docker Desktop installed
- Python 3.9+
- Required packages:
  ```bash
  pip install cassandra-driver pandas pyarrow
  ```

---

## 🎯 **Execution Steps**

### **Step 1: Start Cassandra 3-Node Cluster**

```bash
# Navigate to docker directory
cd docker

# Start cluster
docker-compose up -d

# Check status
docker-compose ps
docker logs cassandra-1
```

**Wait for all 3 nodes to be healthy (2-3 minutes):**
```bash
docker exec cassandra-1 cqlsh -u cassandra -p cassandra -e "DESCRIBE CLUSTER"
```

Expected output: All 3 nodes showing in cluster.

---

### **Step 2: Create Cassandra Schema**

```bash
# Install cassandra-driver
pip install cassandra-driver

# Run setup script
python scripts/00_cassandra_setup.py
```

**Expected output:**
```
✅ Keyspace 'taxi_db' created
✅ Table 'baseline_features' created
✅ Table 'method1_clusters' created
✅ Table 'method2_clusters' created
✅ Table 'method3_clusters' created
✅ Table 'cluster_metadata' created
✅ CASSANDRA SETUP COMPLETE
```

---

### **Step 3: Load Baseline Features to Cassandra**

```bash
# Load data from Parquet
python scripts/04_load_features_cassandra.py
```

**Expected output:**
```
💾 Inserting features into Cassandra...
   ✓ Inserted 100,000 rows
   ✓ Inserted 200,000 rows
   ...
✅ Inserted 3,513,833 rows total
✅ Data verification complete
   Total rows in Cassandra: 3,513,833
   Unique zones: 261
   Time range: 2019-01-01 00:00:00 → 2020-06-30 00:15:00
```

---

## 🧪 **Verify Installation**

### **Query via CQL Shell**

```bash
# Connect to cassandra-1
docker exec -it cassandra-1 cqlsh -u cassandra -p cassandra

# In CQL shell
cqlsh> USE taxi_db;
cqlsh> SELECT COUNT(*) FROM baseline_features;
cqlsh> SELECT * FROM baseline_features WHERE zone_id = 1 LIMIT 5;
cqlsh> SELECT DISTINCT date_day, COUNT(*) FROM baseline_features GROUP BY date_day;
```

### **Query via Python (Test Consistency Levels)**

```python
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

cluster = Cluster(['127.0.0.1', '127.0.0.2', '127.0.0.3'])
session = cluster.connect('taxi_db')

# Query with different consistency levels
for consistency in [ConsistencyLevel.ONE, ConsistencyLevel.LOCAL_QUORUM, ConsistencyLevel.QUORUM]:
    result = session.execute(
        "SELECT * FROM baseline_features WHERE zone_id = 1",
        consistency_level=consistency
    )
    print(f"Consistency {consistency}: {len(result)} rows")

session.shutdown()
cluster.shutdown()
```

---

## 📊 **Current Status**

| Component | Status | Deliverable |
|-----------|--------|-------------|
| **Week 2.1: Baseline Features (Pandas)** | ✅ DONE | 3.5M records, 85.83 MB |
| **Week 1.1: Cassandra Cluster (Docker)** | ✅ READY | 3-node cluster setup |
| **Task 1.5: Schema Creation** | ✅ READY | 5 tables, indexes |
| **Task 4.4: Load to Cassandra** | ✅ READY | ~3.5M records to load |

---

## 🔄 **Next Steps (Week 3+)**

1. **Week 3: Method 1 Features**
   - Extract demand patterns
   - K-Means clustering
   - Load to `method1_clusters` table

2. **Week 4: Method 2 Features**
   - Extract mobility patterns
   - Normalization & PCA
   - K-Means++ clustering
   - Load to `method2_clusters` table

3. **Week 5: Method 3 Features**
   - Extract OD-flow patterns
   - Combine with Method 2 features
   - K-Means clustering
   - Load to `method3_clusters` table

4. **Week 6: Cassandra Analysis**
   - Fault tolerance testing
   - Consistency level benchmarking
   - Query performance analysis
   - CAP theorem analysis

---

## ⚠️ **Troubleshooting**

### **Problem: Can't connect to Cassandra**
```bash
# Check if containers are running
docker ps

# Check logs
docker logs cassandra-1

# Restart if needed
docker-compose restart
```

### **Problem: Nodes not joining cluster**
```bash
# Remove old volumes and restart
docker-compose down -v
docker-compose up -d
# Wait 3 minutes for cluster to form
```

### **Problem: Python cassandra-driver install fails**
```bash
# For Windows, may need Visual C++ tools
# Or use conda
conda install cassandra-driver
```

---

## 📝 **Configuration Notes**

**Cassandra 4.1.3**
- Replication Factor: 3
- Consistency Level: ONE (writes), QUORUM (critical reads)
- Partitioner: Murmur3 (default)
- Partition Key: zone_id
- Clustering Column: window_start (DESC)

**Data Load Strategy**
- Batch size: 10,000 rows/batch
- Total: 3,513,833 rows (~85 MB)
- Insert consistency: ONE (fast)
- Read consistency: QUORUM (safe)

---

## 📚 **Related Files**

- Docker Compose: `docker/docker-compose.yml`
- Setup Script: `scripts/00_cassandra_setup.py`
- Load Script: `scripts/04_load_features_cassandra.py`
- Baseline Features: `data/processed/baseline_features/baseline_features.parquet`
- Setup Log: `logs/00_cassandra_setup.log`
- Load Log: `logs/04_load_features_cassandra.log`

---

**Ready to deploy? ✅ All systems go!**
