# 🔄 Data Pipeline Specification

**Purpose:** Define detailed data flow, format specifications, and technical implementation for Taxi Demand Prediction project

---

## 1. End-to-End Data Flow

```
[NYC Taxi Raw Data (CSV/Parquet)]
    ↓
[1. Data Ingestion & Validation]
    ↓
[2. Data Cleaning & Outlier Detection]
    ↓
[3. Feature Engineering (by method)]
    ├─→ Baseline: Zone aggregation
    ├─→ Method 1: Demand clustering
    ├─→ Method 2: Mobility clustering  
    └─→ Method 3: OD-flow clustering
    ↓
[4. Temporal Aggregation (15min, 30min, 60min)]
    ↓
[5. Train/Val/Test Split]
    ↓
[6. Model Training (4 variants)]
    ↓
[7. Inference & Evaluation]
```

---

## 2. Input Data Specification

### 2.1 Raw NYC Taxi Data
**Source:** NYC TLC Taxi Trip Records  
**Format:** CSV/Parquet  
**Size:** ~5-10M records per month recommended for 2-month project

#### Schema:
```json
{
  "VendorID": "integer (1-2)",
  "tpep_pickup_datetime": "timestamp",
  "tpep_dropoff_datetime": "timestamp",
  "passenger_count": "integer (1-6)",
  "trip_distance": "float (0.1 to 200)",
  "PULocationID": "integer (1-263)",
  "DOLocationID": "integer (1-263)",
  "RateCodeID": "integer (1-6)",
  "store_and_fwd_flag": "string (Y/N)",
  "payment_type": "integer (1-6)",
  "fare_amount": "float (0 to 500)",
  "extra": "float (0 to 10)",
  "mta_tax": "float (0.5 or 0)",
  "improvement_surcharge": "float (0.3 or 0)",
  "tip_amount": "float (0 to 100)",
  "tolls_amount": "float (0 to 50)",
  "total_amount": "float (0 to 1000)"
}
```

**Required Fields:** `tpep_pickup_datetime`, `PULocationID`, `DOLocationID`, `passenger_count`, `trip_distance`, `fare_amount`

### 2.2 Taxi Zone Lookup
**Source:** taxi_zone_lookup.csv  
**Purpose:** Map LocationID ↔ Borough ↔ Zone

```json
{
  "LocationID": "integer",
  "Borough": "string (Manhattan, Bronx, Brooklyn, Queens, Staten Island, Unknown)",
  "Zone": "string",
  "service_zone": "string (Yellow, Green, Boro, Airports)"
}
```

---

## 3. Data Validation & Quality Checks

### 3.1 Stage 1: Schema Validation
```python
# In PySpark
required_cols = ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID', 'fare_amount']
assert all(col in df.columns for col in required_cols), "Missing required columns"
```

### 3.2 Stage 2: Outlier Detection (Z-score method)
```python
# For numeric features
features_to_check = ['fare_amount', 'trip_distance', 'passenger_count']
z_threshold = 3.0  # Remove records with |z| > 3

for col in features_to_check:
    mean = df.select(F.mean(col)).collect()[0][0]
    stddev = df.select(F.stddev(col)).collect()[0][0]
    df = df.filter((F.abs((F.col(col) - mean) / stddev) <= z_threshold))
```

### 3.3 Stage 3: Data Integrity Checks
| Check | Condition | Action |
|-------|-----------|--------|
| Null values | Any null in required fields | Drop row |
| Date range | Date outside [2019-01, 2024-12] | Drop row |
| Location IDs | LocationID not in [1, 263] | Drop row |
| Trip distance | < 0 or > 200 miles | Drop row |
| Passenger count | < 1 or > 10 | Drop row |
| Fare amount | < 0 or > 500 | Drop row |

**Quality Metric:** Retain ≥ 95% of original records after cleaning

---

## 4. Feature Engineering by Method

### 4.1 Baseline: Zone-Based Aggregation

**Temporal Granularity:** 15-minute intervals  
**Aggregation Key:** `(PULocationID, time_bucket_15min)`

**Features Extracted:**
```python
agg_features = {
    'demand_count': 'count(*)',                    # number of trips
    'avg_fare': 'avg(fare_amount)',                # average fare
    'avg_distance': 'avg(trip_distance)',          # avg trip distance
    'avg_passenger': 'avg(passenger_count)',       # avg passengers
    'sum_fare': 'sum(fare_amount)',                # total revenue
    'median_distance': 'percentile(trip_distance, 0.5)',
    'p95_distance': 'percentile(trip_distance, 0.95)'
}
```

**Output Schema:**
```json
{
  "PULocationID": "integer (zone)",
  "time_bucket": "timestamp (rounded to 15min)",
  "date_day": "integer (0-6, day of week)",
  "hour": "integer (0-23)",
  "demand_count": "integer",
  "avg_fare": "float",
  "avg_distance": "float",
  "avg_passenger": "float",
  ...
}
```

**Storage:** `baseline_features_{date_range}.parquet`

---

### 4.2 Method 1: Demand-Based Clustering

**Purpose:** Cluster zones by demand pattern similarity over time

**Aggregation:** 
1. For each zone, extract hourly demand time-series (24 hours per day)
2. Compute features: mean, std, peak_hour, trend

**Feature Matrix (before clustering):**
```python
demand_features = {
    'demand_mean': mean(demand_by_hour),           # avg hourly demand
    'demand_std': std(demand_by_hour),             # variability
    'demand_peak_hour': argmax(demand_by_hour),    # busiest hour
    'demand_night_ratio': sum(0-6)/sum(all),       # night demand fraction
    'demand_morning_peak': max(7-10 demand),       # morning rush
    'demand_evening_peak': max(17-20 demand)       # evening rush
}
```

**Clustering:**
- Algorithm: K-Means++
- Distance: Euclidean (after normalization)
- K selection: Silhouette score (try K=3 to 8)

**Output:** 
```
method1_clusters.pkl
{
  "zone_to_cluster": {LocationID: cluster_id},
  "silhouette_score": float,
  "optimal_k": integer,
  "cluster_centers": array(K, num_features)
}
```

---

### 4.3 Method 2: Mobility Pattern Clustering ⭐ (Primary)

**Purpose:** Cluster zones by movement patterns (trip distance, fare, passengers, duration)

**Feature Extraction:**
```python
mobility_features = {
    'trip_count': 'count(*)',                      # total trips from zone
    'avg_trip_distance': 'avg(trip_distance)',     # typical trip length
    'avg_fare': 'avg(fare_amount)',                # pricing pattern
    'avg_passenger_count': 'avg(passenger_count)', # typical party size
    'trip_duration': 'avg(dropoff_time - pickup_time)'  # typical duration
}
```

**Per-Zone Aggregation:**
```python
# Aggregate per PULocationID
method2_matrix = df.groupBy('PULocationID').agg({
    'trip_distance': 'avg',
    'fare_amount': 'avg', 
    'passenger_count': 'avg',
    'trip_id': 'count'  # as trip_count
})
```

**Feature Normalization:**
```python
# Standardization (zero mean, unit variance)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(method2_matrix)
```

**Clustering:**
- K-Means++ with Silhouette analysis
- Optimal K: typically 4-6 clusters (downtown, airport, residential, mixed, etc.)

**Cluster Semantics:**
| Cluster | Characteristics | Zones |
|---------|-----------------|-------|
| Downtown | High trip_count, high fare, short distance | Manhattan |
| Airport | High fare, long distance, specific zones | JFK, LaGuardia |
| Residential | Low fare, short trips | Outer boroughs |
| Mixed | Medium values across all features | Transitional areas |

**Output:**
```
method2_clusters.pkl + method2_semantics.json
{
  "zone_to_cluster": {LocationID: cluster_id},
  "cluster_semantics": {cluster_id: description},
  "silhouette_score": float
}
```

---

### 4.4 Method 3: OD-Flow Based Clustering

**Purpose:** Cluster zones by Origin-Destination connectivity patterns

**OD Matrix Construction:**
```python
# For each zone pair (PULocationID, DOLocationID), count trips
od_matrix = df.groupBy('PULocationID', 'DOLocationID').count()

# Pivot to get per-zone connections
inflow = df.groupBy('DOLocationID').count().rename('inflow')
outflow = df.groupBy('PULocationID').count().rename('outflow')

# Top destinations (for each zone, what are top 5 destinations?)
top_destinations = df.groupBy('PULocationID').agg(
    F.collect_list(F.col('DOLocationID')).as('top_dest')
)
```

**Feature Enrichment:**
Combine Method 2 features + OD features:
```python
method3_features = method2_features + {
    'inflow_count': inflow,
    'outflow_count': outflow,
    'inflow_diversity': num_unique_origins,        # how many zones feed into this zone
    'outflow_diversity': num_unique_destinations,  # how many zones this zone feeds to
    'connectivity_score': (inflow + outflow) / (zone_demand)  # connectivity ratio
}
```

**Clustering:**
- Same as Method 2 (K-Means++)
- Typically produces more nuanced clusters

**Output:**
```
method3_clusters.pkl + method3_od_matrix.parquet
```

---

## 5. Temporal Aggregation Specifications

**Three time buckets for multi-resolution evaluation:**

### 5.1 15-Minute Aggregation
```python
df['time_bucket'] = F.date_trunc('15 minutes', df['tpep_pickup_datetime'])
```

### 5.2 30-Minute Aggregation
```python
df['time_bucket'] = F.date_trunc('30 minutes', df['tpep_pickup_datetime'])
```

### 5.3 60-Minute Aggregation
```python
df['time_bucket'] = F.date_trunc('1 hour', df['tpep_pickup_datetime'])
```

**Data shape (per method, per time bucket):**
- Rows: num_clusters × num_time_steps
- Columns: 7+ features (demand_count, fare, distance, etc.)
- Time span: ~6-12 months (depends on dataset size)

---

## 6. Train/Validation/Test Split

**Strategy:** Time-based split (no information leakage)

```python
# Example with 12 months of data
train_end = '2023-11-01'      # First 11 months
val_end = '2023-12-01'        # Next 1 month
test_end = '2024-01-01'       # Last 1 month

train_data = df[df.date < train_end]     # 11 months
val_data = df[(df.date >= train_end) & (df.date < val_end)]   # 1 month
test_data = df[df.date >= val_end]       # 1 month
```

**Ratio:** 85% train, ~7-8% val, ~7-8% test

---

## 7. Output Data Formats

### 7.1 Feature Datasets
```
data/
├── baseline_features_2023.parquet
├── method1_features_2023.parquet
├── method2_features_2023.parquet
├── method3_features_2023.parquet
├── clusters/
│   ├── baseline_clusters.pkl
│   ├── method1_clusters.pkl
│   ├── method2_clusters.pkl
│   └── method3_clusters.pkl
└── train_val_test_split.json
```

### 7.2 Cluster Metadata
```json
{
  "method": "Method 2: Mobility",
  "num_clusters": 5,
  "optimal_k": 5,
  "silhouette_score": 0.68,
  "zones": [1, 2, 3, ...],
  "cluster_assignments": {1: 0, 2: 1, ...},
  "cluster_semantics": {0: "Downtown", 1: "Airport", ...}
}
```

### 7.3 Model Predictions
```json
{
  "method": "Method 2",
  "time_bucket": "15min",
  "predictions": [
    {"cluster": 0, "timestamp": "2024-01-01 00:15", "predicted_demand": 45.2},
    {"cluster": 1, "timestamp": "2024-01-01 00:15", "predicted_demand": 12.3},
    ...
  ],
  "metrics": {
    "MAE": 3.45,
    "RMSE": 5.67,
    "MAPE": 0.12
  }
}
```

---

## 8. Database Storage Recommendations

### Option A: Parquet Files (Simple, for < 50GB)
- **Pros:** Simple, fast queries with PySpark
- **Cons:** No ACID, limited concurrency
- **Recommendation:** Start here for 2-month project

### Option B: HBase/Cassandra (Distributed, scalable)
- **Key Design:** `{ClusterID}#{Timestamp}` (salted key to avoid hotspots)
- **Column Families:** 
  - `features`: demand, fare, distance, passengers
  - `metadata`: cluster_info, zone_mapping
- **TTL:** Keep for 6 months then archive

### Option C: PostgreSQL (Structured, small scale)
- **Schema:** 
  ```sql
  CREATE TABLE cluster_predictions (
    id BIGSERIAL PRIMARY KEY,
    method VARCHAR(50),
    cluster_id INT,
    time_bucket TIMESTAMP,
    demand_count INT,
    avg_fare FLOAT,
    avg_distance FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
  );
  ```
- **Indexes:** `(cluster_id, time_bucket)` for fast queries

**Recommendation for 2-month project:** Parquet + local PostgreSQL for metadata

---

## 9. Data Versioning & Tracking

**Tool:** DVC (Data Version Control) or MLflow

```yaml
# dvc.yaml
stages:
  data_ingestion:
    cmd: python scripts/01_ingest.py
    deps:
      - raw_data/
    outs:
      - data/ingested_data.parquet
    
  method2_clustering:
    cmd: python scripts/03_method2_cluster.py
    deps:
      - data/ingested_data.parquet
    outs:
      - data/method2_features.parquet
      - models/method2_clusters.pkl
```

---

## 10. Quality Assurance Checklist

- [ ] All 4 feature sets generated successfully
- [ ] Record counts match between methods (within ±2%)
- [ ] No null values in feature columns
- [ ] Silhouette scores > 0.5 for all clustering methods
- [ ] Train/val/test split verified (no date overlap)
- [ ] All cluster labels assigned (no cluster ID gaps)
- [ ] Feature distributions reasonable (no extreme outliers)
- [ ] Data reproducible (same seed → same results)
