# 📦 Module 1: Big Data Engineering - Scalable Feature Engineering & Distributed Processing

**Owner:** Person A | **Duration:** Weeks 1-6 | **Deliverables:** Feature datasets, ETL scripts, performance reports

---

## 1. Spark ETL Implementation (Weeks 1-2)

### 1.1 Data Ingestion Pipeline
```python
# scripts/01_data_ingestion.py
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Taxi-Ingestion") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", 200) \
    .getOrCreate()

# Load CSV
df = spark.read.csv("data/raw/taxi_trips.csv", header=True, inferSchema=True)

# Repartition for efficiency
df = df.repartition(100, "PULocationID")

# Save as Parquet (columnar format, faster for analytics)
df.write.mode("overwrite").parquet("data/processed/ingested_data.parquet")
```

**Key Metrics to Log:**
- Record count: X million rows
- File size: Y GB
- Repartition parallelism: 100 partitions
- Ingestion time: Z minutes

### 1.2 Data Quality & Outlier Detection
```python
# scripts/02_data_validation.py
from pyspark.sql.functions import col, abs, mean, stddev

def detect_outliers(df, column, threshold=3.0):
    """Remove records with |z-score| > threshold"""
    stats = df.agg(mean(column), stddev(column)).collect()[0]
    mean_val, std_val = stats[0], stats[1]
    
    df_filtered = df.filter(
        (abs(col(column) - mean_val) / std_val) <= threshold
    )
    
    removed_count = df.count() - df_filtered.count()
    print(f"Removed {removed_count} outliers from {column}")
    return df_filtered

# Apply to multiple columns
for col_name in ['fare_amount', 'trip_distance', 'passenger_count']:
    df = detect_outliers(df, col_name)
```

**Validation Checklist:**
- [ ] Schema validated (all columns present)
- [ ] Null handling (< 5% nulls)
- [ ] Outliers removed (retain 95%+ records)
- [ ] Date range checked [2022-01-01, 2024-01-01]
- [ ] LocationID valid [1-263]

---

## 2. Feature Extraction by Method (Weeks 3-5)

### 2.1 Baseline Method: Zone Aggregation
```python
# scripts/03_method0_baseline.py
from pyspark.sql.functions import window, count, avg, sum as spark_sum

# Aggregate by zone & 15-min time window
baseline_features = df.groupBy(
    window(col("tpep_pickup_datetime"), "15 minutes"),
    col("PULocationID").alias("zone_id")
).agg(
    count("*").alias("demand_count"),
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    avg("passenger_count").alias("avg_passenger"),
    spark_sum("fare_amount").alias("total_revenue")
).withColumnRenamed("window", "time_bucket")

baseline_features.write.parquet("data/processed/baseline_features/")
```

**Output Schema:**
```json
{
  "time_bucket": "timestamp",
  "zone_id": "integer",
  "demand_count": "integer",
  "avg_fare": "float",
  "avg_distance": "float",
  "avg_passenger": "float",
  "total_revenue": "float"
}
```

**Acceptance Criteria:**
- ✅ Covers all 263 zones
- ✅ No null values in features
- ✅ Demand count > 0 for at least 80% of zone-time pairs

### 2.2 Method 1: Demand-Based Clustering Features
```python
# scripts/04_method1_clustering.py

# Extract hourly demand patterns per zone
hourly_demand = df.groupBy(
    "PULocationID",
    F.hour("tpep_pickup_datetime").alias("hour")
).agg(
    count("*").alias("demand")
)

# Pivot to get 24-hour demand profile per zone
demand_profile = hourly_demand.pivot("hour").agg(F.first("demand"))

# Compute statistics
demand_features = demand_profile.select(
    col("PULocationID").alias("zone_id"),
    F.mean(F.struct([F.col(str(h)) for h in range(24)])).alias("demand_mean"),
    F.stddev(F.struct([F.col(str(h)) for h in range(24)])).alias("demand_std"),
    F.max(F.struct([F.col(str(h)) for h in range(24)])).alias("demand_peak"),
    ...
)

demand_features.write.parquet("data/processed/method1_features/")
```

**Feature List:**
- `demand_mean`: Average hourly demand
- `demand_std`: Variability in demand
- `demand_peak_hour`: Busiest hour
- `night_demand_ratio`: Night (0-6) vs total
- `morning_peak`: Max(7-10 demand)
- `evening_peak`: Max(17-20 demand)

### 2.3 Method 2: Mobility Pattern Clustering Features ⭐
```python
# scripts/05_method2_clustering.py

# Aggregate per PULocationID (zone)
mobility_features = df.groupBy("PULocationID").agg(
    count("*").alias("trip_count"),
    avg("trip_distance").alias("avg_trip_distance"),
    avg("fare_amount").alias("avg_fare"),
    avg("passenger_count").alias("avg_passenger_count"),
    # Trip duration: dropoff - pickup
    avg(F.col("tpep_dropoff_datetime") - F.col("tpep_pickup_datetime")).alias("avg_trip_duration")
).withColumnRenamed("PULocationID", "zone_id")

# Normalize features (zero mean, unit variance)
from pyspark.ml.feature import StandardScaler, VectorAssembler

assembler = VectorAssembler(
    inputCols=["trip_count", "avg_trip_distance", "avg_fare", "avg_passenger_count", "avg_trip_duration"],
    outputCol="features"
)

mobility_normalized = assembler.transform(mobility_features)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
mobility_scaled = scaler.fit(mobility_normalized).transform(mobility_normalized)

mobility_scaled.write.parquet("data/processed/method2_features/")
```

**Feature Output:**
```
zone_id | trip_count | avg_distance | avg_fare | avg_passenger | avg_duration | scaled_features
1       | 50000      | 3.2          | 14.5     | 1.5           | 720          | [...]
2       | 1200       | 15.8         | 45.3     | 2.1           | 1450         | [...]
...
```

### 2.4 Method 3: OD-Flow Based Clustering Features
```python
# scripts/06_method3_clustering.py

# Compute Origin-Destination flows
od_matrix = df.groupBy("PULocationID", "DOLocationID").count().withColumnRenamed("count", "flow")

# Inflow & Outflow per zone
inflow = od_matrix.groupBy("DOLocationID").agg(F.sum("flow").alias("inflow"))
outflow = od_matrix.groupBy("PULocationID").agg(F.sum("flow").alias("outflow"))

# Get top destinations per origin zone
top_destinations = df.groupBy("PULocationID").agg(
    F.collect_list(F.col("DOLocationID")).alias("destinations")
)

# Combine with Method 2 features
method3_features = mobility_scaled.join(inflow, on=F.col("zone_id") == inflow["DOLocationID"], how="left") \
    .join(outflow, on=F.col("zone_id") == outflow["PULocationID"], how="left") \
    .drop(outflow["PULocationID"]).drop(inflow["DOLocationID"])

method3_features.write.parquet("data/processed/method3_features/")
```

---

## 3. K-Means Clustering Implementation

### 3.1 Optimal K Selection via Silhouette Analysis
```python
# scripts/clustering_utils.py
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def find_optimal_k(features_df, k_range=[3, 4, 5, 6, 7, 8]):
    """Find optimal number of clusters using silhouette score"""
    
    results = {}
    for k in k_range:
        kmeans = KMeans(k=k, seed=42, maxIter=100)
        model = kmeans.fit(features_df)
        
        predictions = model.transform(features_df)
        
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="scaled_features",
            metricName="silhouette"
        )
        silhouette = evaluator.evaluate(predictions)
        
        results[k] = silhouette
        print(f"K={k}: Silhouette Score = {silhouette:.4f}")
    
    optimal_k = max(results, key=results.get)
    return optimal_k, results

# Run for Method 2
optimal_k, scores = find_optimal_k(mobility_scaled)
print(f"✅ Optimal K for Method 2: {optimal_k}")
```

**Performance Benchmarking:**
- Document execution time for each K
- Document silhouette scores
- Choose K with silhouette > 0.5

### 3.2 Final Clustering Model
```python
# Train final model with optimal K
final_kmeans = KMeans(k=optimal_k, seed=42, maxIter=100)
final_model = final_kmeans.fit(mobility_scaled)

# Get cluster assignments
clustered_data = final_model.transform(mobility_scaled)

# Save model & assignments
final_model.save("data/models/method2_kmeans_model")

# Export cluster assignments as lookup table
cluster_lookup = clustered_data.select("zone_id", "prediction").rdd.collectAsMap()

import json
with open("data/clusters/method2_clusters.json", "w") as f:
    json.dump({"zone_to_cluster": cluster_lookup}, f)
```

---

## 4. Performance Optimization & Benchmarking

### 4.1 Spark Configuration Tuning
```yaml
# config/spark_config.yaml
spark:
  shuffle_partitions: 200        # Default 200; tune based on data size
  broadcast_threshold: 10485760  # 10MB broadcast join threshold
  max_core_executors: 4          # Parallelism level
  driver_memory: 4g
  executor_memory: 4g
```

### 4.2 Execution Plan Analysis
```python
# Print query execution plan
mobility_scaled.explain(extended=True)

# Example output shows:
# - Scan operation (read Parquet)
# - GroupBy aggregation
# - Window function computation
# - Total parallelism: N partitions
```

**Optimization Checklist:**
- [ ] Broadcast small lookup tables (< 10MB)
- [ ] Avoid shuffle operations where possible
- [ ] Use columnar format (Parquet) for storage
- [ ] Partition data by hot key (PULocationID)
- [ ] Cache intermediate dataframes if reused

---

## 5. Database Storage (Optional, for Large Scale)

### 5.1 Parquet Files (Recommended for 2-month project)
```python
# Simple, fast columnar storage
df.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("data/processed/method2_features/")
```

### 5.2 PostgreSQL (If structured queries needed)
```python
# Write to PostgreSQL (requires JDBC)
df.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/taxi_db") \
    .option("dbtable", "method2_features") \
    .option("user", "postgres") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

---

## 6. Key Deliverables (Big Data Module)

### Week 2:
- [ ] Ingestion script (01_data_ingestion.py)
- [ ] Data validation report (ingestion_quality.json)
- [ ] Baseline features (baseline_features.parquet)

### Week 5:
- [ ] All 4 feature extraction scripts (04_method1, 05_method2, 06_method3)
- [ ] Feature datasets for all methods
- [ ] Clustering models (method1_kmeans_model, method2_kmeans_model, method3_kmeans_model)

### Week 6:
- [ ] Performance benchmarking report (time, memory, parallelism analysis)
- [ ] Feature statistics (distributions, correlations)

### Week 8:
- [ ] **Big Data Module Report** (20-30 pages):
  - ETL pipeline architecture & design
  - Feature engineering methodology (4 methods)
  - Clustering algorithm implementation & optimization
  - Performance analysis & scalability insights
  - CAP Theorem analysis for distributed system design

---

## 7. Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Out of Memory | Too many partitions | Reduce partition count |
| Shuffle timeout | Large shuffle operations | Increase executor memory or use broadcast join |
| Slow clustering | High-dimensional data | Use PCA for dimensionality reduction |
| Data skew | Hotspotting on few zones | Use salted keys for partitioning |

---

## 8. Prompt Templates for AI Assistance

> "I'm at Week 4 implementing Method 2 features. Write a PySpark script that:
> 1. Groups taxi trips by PULocationID
> 2. Computes 5 aggregations: trip_count, avg_distance, avg_fare, avg_passenger_count, trip_duration
> 3. Applies StandardScaler normalization
> 4. Handles null values gracefully
> Include performance considerations and explain the Spark execution plan."

> "For Method 3 (OD-Flow), explain how to compute inflow/outflow per zone efficiently in Spark. 
> Write code to prevent shuffles and use broadcast joins where appropriate."