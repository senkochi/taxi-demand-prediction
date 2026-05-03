# 🛠️ Development Setup & Repository Structure

**Purpose:** Define project structure, environment setup, dependencies, and development workflow

---

## 1. Repository Structure

```
taxi-demand-prediction/
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── Makefile                          # Quick commands
│
├── .github/
│   └── workflows/
│       ├── tests.yml                 # Unit tests CI
│       └── lint.yml                  # Code quality checks
│
├── config/
│   ├── config.yaml                   # Main config (paths, parameters)
│   ├── spark_config.yaml             # Spark cluster settings
│   └── logging_config.yaml           # Logging setup
│
├── data/
│   ├── raw/                          # Original NYC taxi data (gitignored, large)
│   ├── processed/
│   │   ├── baseline_features/        # Baseline method outputs
│   │   ├── method1_features/         # Demand clustering outputs
│   │   ├── method2_features/         # Mobility clustering outputs (main)
│   │   ├── method3_features/         # OD-flow clustering outputs
│   │   └── clusters/                 # Cluster pickle files
│   ├── external/                     # taxi_zone_lookup.csv, metadata
│   └── train_val_test/               # Split datasets
│
├── scripts/
│   ├── 00_env_setup.sh               # Setup Spark, Java, dependencies
│   ├── 01_data_ingestion.py          # Raw data → Parquet
│   ├── 02_data_validation.py         # Quality checks, outlier detection
│   ├── 03_method0_baseline.py        # Baseline feature extraction
│   ├── 04_method1_clustering.py      # Demand clustering
│   ├── 05_method2_clustering.py      # Mobility clustering (main)
│   ├── 06_method3_clustering.py      # OD-flow clustering
│   ├── 07_train_val_test_split.py   # Temporal split
│   ├── 08_model_training.py          # SSTZIP-GNN training loop
│   └── 09_evaluation.py              # Metrics computation & comparison
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Data loading utilities
│   │   ├── validators.py             # Data validation functions
│   │   └── preprocessor.py           # Preprocessing logic
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── baseline_features.py      # Baseline feature extraction
│   │   ├── demand_features.py        # Method 1 features
│   │   ├── mobility_features.py      # Method 2 features (main)
│   │   └── od_features.py            # Method 3 features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dgcn.py                   # Diffusion Graph Convolution layer
│   │   ├── tcn.py                    # Temporal Convolution Network layer
│   │   ├── zip_head.py               # Zero-Inflated Poisson head
│   │   ├── sstzip_gnn.py             # Full model architecture
│   │   └── losses.py                 # Custom loss functions
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # PyTorch Lightning Trainer wrapper
│   │   ├── callbacks.py              # Custom callbacks (logging, checkpointing)
│   │   └── optimizers.py             # Optimizer configurations
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # MAE, RMSE, Zero-Inflation Score
│   │   ├── visualization.py          # Plotting utilities
│   │   └── comparison.py             # Method comparison logic
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Logging setup
│       ├── config_loader.py          # Config file handling
│       └── constants.py              # Magic numbers, paths
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_data_validators.py   # Data validation tests
│   │   ├── test_feature_extraction.py # Feature extraction tests
│   │   └── test_models.py            # Model forward pass tests
│   │
│   ├── integration/
│   │   ├── test_pipeline.py          # End-to-end pipeline tests
│   │   └── test_model_training.py    # Training loop tests
│   │
│   └── fixtures/
│       ├── sample_data.py            # Small sample datasets
│       └── mock_configs.py           # Test configurations
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_method_comparison.ipynb    # Clustering comparison
│   ├── 03_results_analysis.ipynb     # Final results & visualizations
│   └── 04_thesis_figures.ipynb       # Publication-ready figures
│
├── reports/
│   ├── figures/                      # Generated plots
│   ├── tables/                       # Results tables (CSV/JSON)
│   ├── big_data_module.md            # Big Data report draft
│   ├── ml_module.md                  # ML/DL report draft
│   └── experiments_log.md            # Experiment tracking
│
├── docker/
│   ├── Dockerfile.spark              # Spark cluster Docker image
│   ├── Dockerfile.app                # Application Docker image
│   └── docker-compose.yml            # Multi-container setup
│
├── docs/
│   ├── API.md                        # API documentation
│   ├── ARCHITECTURE.md               # System design
│   └── TROUBLESHOOTING.md            # Common issues & fixes
│
└── .gitignore
    # Ignore data, models, logs, virtual envs
```

---

## 2. Environment Setup

### 2.1 Prerequisites
- **Python:** 3.10 or 3.11
- **Java:** JDK 11+ (for Spark)
- **Spark:** 3.3.0 or higher
- **Git:** For version control

### 2.2 Option A: Local Development (Recommended for 2-person team)

#### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/taxi-demand-prediction.git
cd taxi-demand-prediction
```

#### Step 2: Create Python Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda env create -f environment.yml
conda activate taxi-prediction
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Setup Spark
```bash
# Download Spark (if not installed)
./scripts/00_env_setup.sh

# Verify Spark
spark-submit --version
```

#### Step 5: Configure Paths
```bash
cp config/config.yaml.template config/config.yaml
# Edit config.yaml with your paths
```

### 2.3 Option B: Docker Setup (For reproducibility)

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up -d

# Execute script inside container
docker-compose exec app python scripts/01_data_ingestion.py
```

### 2.3.5 Cassandra Docker Setup (Local Development)

**Why Local Docker for Cassandra?**
- Eliminates AWS cost/complexity concerns
- Easy to simulate fault tolerance (kill/restart containers)
- 3-node local cluster sufficient for testing distributed system concepts
- Mirrors production topology locally

**Quick Start:**

```bash
# Start Cassandra 3-node cluster
docker-compose -f docker/docker-compose.yml up -d cassandra-1 cassandra-2 cassandra-3

# Wait for cluster stabilization (30 seconds)
sleep 30

# Verify cluster health
docker exec cassandra-node-1 nodetool status

# Expected output (all UN = Up Normal):
# UN  172.19.0.2   104.5 KB   256     33.3%
# UN  172.19.0.3   108.3 KB   256     33.3%
# UN  172.19.0.4   102.1 KB   256     33.3%

# Connect to CQL shell
docker exec -it cassandra-node-1 cqlsh

# Inside cqlsh:
# CREATE KEYSPACE taxi_db WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};
# USE taxi_db;
# [create tables from MODULE_DISTRIBUTED_DATABASE.md Section 3]
```

**Docker Compose Cassandra Section:**

```yaml
# docker/docker-compose.yml - Add this section

version: '3.9'

services:
  cassandra-1:
    image: cassandra:4.0
    container_name: cassandra-node-1
    environment:
      CASSANDRA_CLUSTER_NAME: "taxi-cluster"
      CASSANDRA_DC: "us-east-1"
      CASSANDRA_RACK: "rack1"
      CASSANDRA_SEEDS: "cassandra-1"
    ports:
      - "9042:9042"
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

**Cassandra Python Dependencies:**

Add to `requirements.txt`:
```txt
cassandra-driver==3.25.0
pyspark-cassandra==1.0.0
```

**Verify Cassandra Connection from Python:**

```python
# scripts/test_cassandra_connection.py
from cassandra.cluster import Cluster

def test_cassandra():
    cluster = Cluster(['localhost'])  # 9042 is default port
    session = cluster.connect()
    
    rows = session.execute("SELECT release_version FROM system.local")
    print(f"Cassandra version: {rows[0].release_version}")
    
    session.shutdown()
    cluster.shutdown()

if __name__ == "__main__":
    test_cassandra()
    print("✅ Cassandra connection successful!")
```

Run:
```bash
python scripts/test_cassandra_connection.py
```

### 2.4 Option C: Cloud Setup (AWS/GCP)

**AWS EMR Example:**
```bash
aws emr create-cluster \
  --name taxi-prediction \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --ec2-key-pair your-key-pair
```

---

## 3. Dependencies

### 3.1 Core Dependencies

```txt
# requirements.txt

# Data Processing
pyspark==3.3.2
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0

# Deep Learning
torch==2.0.0
pytorch-lightning==2.0.0
torch-geometric==2.3.0

# Utilities
pyyaml==6.0
python-dotenv==1.0.0
tqdm==4.65.0
joblib==1.2.0

# Evaluation & Visualization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.13.0
pandas-profiling==3.6.0

# Testing
pytest==7.3.0
pytest-cov==4.0.0

# Code Quality
black==23.3.0
flake8==6.0.0
isort==5.12.0
mypy==1.0.0

# Logging & Monitoring
mlflow==2.2.0
wandb==0.15.0
```

### 3.2 Conda Environment (for Spark compatibility)

```yaml
# environment.yml
name: taxi-prediction
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - openjdk=11
  - pyspark=3.3.2
  - pandas=2.0.0
  - numpy=1.24.0
  - scikit-learn=1.2.0
  - pytorch::pytorch::cpuonly
  - pytorch::pytorch-lightning::gpu
  - pytorch::torch-geometric
  - pip
  - pip:
    - -r requirements.txt
```

---

## 4. Configuration Management

### 4.1 Main Config (YAML)

```yaml
# config/config.yaml

project:
  name: "Taxi Demand Prediction"
  version: "1.0"
  description: "Multiresolution taxi demand prediction with clustering methods"

data:
  raw_path: "./data/raw/"
  processed_path: "./data/processed/"
  output_path: "./data/outputs/"
  external_path: "./data/external/"
  
  # Data parameters
  date_range: ["2022-01-01", "2023-12-31"]
  sample_size: null  # null = use all data
  
paths:
  scripts: "./scripts/"
  models: "./data/models/"
  logs: "./logs/"
  reports: "./reports/"

clustering:
  methods: ["baseline", "method1", "method2", "method3"]
  
  method1:
    name: "Demand-Based Clustering"
    k_range: [3, 4, 5, 6, 7, 8]
    
  method2:
    name: "Mobility Pattern Clustering"
    k_range: [4, 5, 6, 7, 8]
    features: ["trip_count", "avg_distance", "avg_fare", "avg_passenger", "trip_duration"]
    
  method3:
    name: "OD-Flow Clustering"
    k_range: [4, 5, 6, 7, 8]

temporal_aggregation:
  buckets: [15, 30, 60]  # minutes

train_val_test:
  train_ratio: 0.85
  val_ratio: 0.08
  test_ratio: 0.07

model:
  architecture: "SSTZIP-GNN"
  hidden_dim: 64
  num_layers: 3
  dropout: 0.2
  
  training:
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
    early_stopping_patience: 15
    device: "cpu"  # "cuda" if GPU available

evaluation:
  metrics: ["MAE", "RMSE", "MAPE", "zero_inflation_score"]
  time_buckets: [15, 30, 60]

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 4.2 Spark Configuration

```yaml
# config/spark_config.yaml
spark:
  app_name: "Taxi-Demand-Prediction"
  master: "local[4]"  # Use 4 cores locally; "spark://..." for cluster
  
  # Memory settings
  driver_memory: "4g"
  executor_memory: "4g"
  executor_cores: 2
  
  # SQL settings
  sql_shuffle_partitions: 200
  
  # Parquet optimization
  parquet_compression: "snappy"
```

---

## 5. Quick Commands (Makefile)

```makefile
# Makefile

.PHONY: setup install test run clean help

help:
	@echo "Available commands:"
	@echo "  make setup       - Setup environment"
	@echo "  make install     - Install dependencies"
	@echo "  make data        - Run data pipeline"
	@echo "  make cluster     - Run clustering methods"
	@echo "  make train       - Train all models"
	@echo "  make evaluate    - Evaluate and compare"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Code quality checks"
	@echo "  make clean       - Clean outputs"

setup:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	@echo "✅ Environment ready!"

install:
	pip install -r requirements.txt

data:
	python scripts/01_data_ingestion.py
	python scripts/02_data_validation.py

cluster:
	python scripts/03_method0_baseline.py
	python scripts/04_method1_clustering.py
	python scripts/05_method2_clustering.py
	python scripts/06_method3_clustering.py

train:
	python scripts/08_model_training.py

evaluate:
	python scripts/09_evaluation.py

test:
	pytest tests/ -v --cov=src

lint:
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf logs/*
```

---

## 6. Development Workflow

### 6.1 Feature Branch Workflow

```bash
# Create feature branch
git checkout -b feature/method2-optimization

# Make changes & commit
git add src/features/mobility_features.py
git commit -m "feat: optimize mobility features extraction"

# Push to remote
git push origin feature/method2-optimization

# Create Pull Request (PR)
# After code review: merge to main
```

### 6.2 Code Review Checklist

- [ ] Code follows PEP 8 style
- [ ] All tests pass (`make test`)
- [ ] No new linting errors (`make lint`)
- [ ] Docstrings added to new functions
- [ ] Changes documented in CHANGELOG.md
- [ ] No hardcoded paths (use config instead)
- [ ] Performance impact assessed

---

## 7. Running Scripts

### 7.1 Single Script Execution

```bash
# Run with default config
python scripts/05_method2_clustering.py

# Run with custom config
python scripts/05_method2_clustering.py --config config/custom_config.yaml

# Run with logging
python scripts/05_method2_clustering.py --log-level DEBUG
```

### 7.2 Full Pipeline

```bash
# Execute complete pipeline
make data
make cluster
make train
make evaluate

# Or use orchestration
python scripts/run_pipeline.py --methods all --time-buckets 15 30 60
```

### 7.3 Monitoring

```bash
# Watch logs in real-time
tail -f logs/pipeline_$(date +%Y%m%d).log

# Use MLflow UI
mlflow ui --host localhost --port 5000
# Then visit http://localhost:5000
```

---

## 8. Testing

### 8.1 Unit Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_data_validators.py::test_outlier_detection -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### 8.2 Integration Tests

```bash
# Test full pipeline with sample data
pytest tests/integration/test_pipeline.py -v -s
```

---

## 9. Version Control Best Practices

### 9.1 .gitignore

```
# Virtual environments
venv/
env/
*.egg-info/

# Data (large files)
data/raw/
data/processed/
*.parquet
*.csv

# Models & outputs
data/models/
*.pkl
*.pth

# Logs & cache
logs/
.pytest_cache/
__pycache__/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Config with secrets
config/config.local.yaml
.env
```

### 9.2 Commit Message Convention

```
feat: add mobility features extraction
fix: resolve null values in clustering
docs: update data pipeline specification
test: add unit tests for validators
perf: optimize spark groupby operations
refactor: rename variables for clarity
```

---

## 10. Documentation

### 10.1 Code Docstrings (Google Style)

```python
def extract_mobility_features(df: pd.DataFrame, zone_column: str = 'PULocationID') -> pd.DataFrame:
    """Extract mobility features from taxi data.
    
    Args:
        df: Input dataframe with taxi trips
        zone_column: Column name for zone IDs
        
    Returns:
        DataFrame with aggregated mobility features per zone
        
    Raises:
        ValueError: If required columns missing
        
    Example:
        >>> trips = pd.read_parquet('trips.parquet')
        >>> features = extract_mobility_features(trips)
    """
    pass
```

### 10.2 README.md

Include:
- Project overview
- Quick start guide
- File structure explanation
- Contributing guidelines
- Results summary
- Citation (for thesis)

---

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| Spark: "Exception in thread 'main'" | Verify Java 11+ installed: `java -version` |
| Memory error on local Spark | Increase `spark.driver.memory` in config |
| Module not found | Ensure `src/` added to PYTHONPATH |
| Tests fail after code changes | Run `make lint` and `make test` |

---

## 12. Resources & References

- **Spark Documentation:** https://spark.apache.org/docs/latest/
- **PyTorch Lightning:** https://pytorch-lightning.readthedocs.io/
- **Project README:** see `README.md`
- **Paper Reference:** Multiresolution Taxi Demand Prediction (provided)
