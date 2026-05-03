# 🚕 Taxi Demand Prediction

**Project Objective:** Compare 4 clustering methods for taxi demand prediction using deep learning (SSTZIP-GNN)

**Duration:** 8 weeks | **Team:** 2 people | **Scope:** NYC Taxi Dataset (5M+ records)

---

## 📋 Quick Start

### 1️⃣ Setup Development Environment

```bash
# Clone repo (already done)
cd taxi-demand-prediction

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Verify Installation

```bash
# Check Python
python --version  # Should be 3.11+

# Check Spark
spark-submit --version

# Test imports
python -c "import pyspark; import torch; import pytorch_lightning; print('✅ All dependencies OK!')"
```

### 3️⃣ Project Structure

```
taxi-demand-prediction/
├── config/                 # Configuration files (YAML)
├── data/                  # Data directory (raw → processed)
│   ├── raw/              # NYC taxi raw data
│   ├── processed/        # Processed features per method
│   └── train_val_test/   # Train/val/test splits
├── src/                  # Source code
│   ├── data/            # Data loading & validation
│   ├── features/        # Feature extraction (all 4 methods)
│   ├── models/          # Model architecture (SSTZIP-GNN)
│   ├── training/        # Training pipeline
│   ├── evaluation/      # Metrics & comparison
│   └── utils/           # Utilities
├── scripts/              # Executable scripts (Week 1-7 pipeline)
├── tests/                # Unit & integration tests
├── notebooks/            # Jupyter notebooks (EDA, analysis)
├── reports/              # Results & visualizations
└── logs/                 # Training & execution logs
```

---

## 🚀 Running the Pipeline

### Option A: Step-by-Step (Recommended for Week 1-2)

```bash
# Week 1-2: Data Setup
python scripts/01_data_ingestion.py       # Load NYC taxi data
python scripts/02_data_validation.py      # Data quality checks

# Week 3-5: Clustering Methods
python scripts/03_method0_baseline.py     # Baseline (zone-based)
python scripts/04_method1_clustering.py   # Method 1 (demand-based)
python scripts/05_method2_clustering.py   # Method 2 (mobility patterns) ⭐
python scripts/06_method3_clustering.py   # Method 3 (OD-flow)

# Week 5-6: Deep Learning
python scripts/08_model_training.py       # Train SSTZIP-GNN (all 4 methods)

# Week 7: Evaluation
python scripts/09_evaluation.py           # Compare results
```

### Option B: Using Makefile (Quick)

```bash
make install    # Setup environment
make data       # Run data pipeline
make cluster    # Run all clustering methods
make train      # Train models
make evaluate   # Compare results
```

---

## 📊 Configuration

Edit `config/config.yaml` to customize:
- Data date range
- Clustering parameters (k_range, features)
- Model hyperparameters
- Evaluation metrics

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_data_validators.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Documentation

- [PROJECT_ROADMAP.md](ai/PROJECT_ROADMAP.md) - Detailed timeline & milestones
- [DEVELOPMENT_SETUP.md](ai/DEVELOPMENT_SETUP.md) - Full setup guide
- [MODULE_BIG_DATA_ENGINEERING.md](ai/MODULE_BIG_DATA_ENGINEERING.md) - Data engineering details
- [MODULE_DEEP_LEARNING_MODELING.md](ai/MODULE_DEEP_LEARNING_MODELING.md) - Model architecture
- [MODULE_DISTRIBUTED_DATABASE.md](ai/MODULE_DISTRIBUTED_DATABASE.md) - Cassandra setup

---

## ⚠️ Week 1.1 Checklist

- [x] Git repo cloned
- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Folder structure ready
- [ ] Spark verified
- [ ] Ready for data ingestion (Week 1.2)

---

## 🤝 Team Workflow

- **Person A:** Data engineering, Spark ETL, infrastructure
- **Person B:** ML/Analytics, clustering, model architecture
- **Weekly Sync:** Mondays 10 AM
- **Code Review:** GitHub PRs required

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| Spark error | Verify Java 11+: `java -version` |
| Memory error | Increase `spark.driver.memory` in config |
| Module not found | Add src to PYTHONPATH |
| GPU not detected | Check PyTorch CUDA version in requirements |

---

## 📞 References

- [Spark Documentation](https://spark.apache.org/docs/latest/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- [NYC Taxi Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Status:** Week 1 Setup In Progress ⏳
