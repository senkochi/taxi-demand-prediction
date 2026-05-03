# 📊 Experiment Tracking & Comparison Framework

**Purpose:** Define how to systematically track, log, and compare experiments across 4 clustering methods

---

## 1. Experiment Tracking Overview

```
┌─────────────────────────────────────────────────┐
│ Experiment 1: Baseline + 15min bucket           │
│ ├─ Model Checkpoint                              │
│ ├─ Metrics (MAE, RMSE, etc.)                    │
│ ├─ Predictions (inference output)               │
│ ├─ Config (hyperparameters, paths)              │
│ └─ Metadata (date, duration, notes)             │
├─────────────────────────────────────────────────┤
│ Experiment 2: Method 1 + 30min bucket           │
│ ...                                             │
├─────────────────────────────────────────────────┤
│ Experiment 3: Method 2 + 60min bucket           │
│ ...                                             │
└─────────────────────────────────────────────────┘

Total Experiments: 4 methods × 3 time buckets = 12 base experiments
+ variations (hyperparameter sweeps, ensemble attempts, etc.)
```

---

## 2. MLflow Setup

### 2.1 MLflow Tracking Server

```bash
# Start MLflow server locally
mlflow server --host 0.0.0.0 --port 5000

# Access UI: http://localhost:5000
```

### 2.2 Python Code Integration

```python
import mlflow
from pathlib import Path

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create/use experiment
mlflow.set_experiment("Taxi-Demand-Prediction")

# Log parameters, metrics, artifacts
with mlflow.start_run(run_name="Method2_15min_v1"):
    # Parameters
    mlflow.log_params({
        "method": "Method 2: Mobility Clustering",
        "time_bucket": 15,
        "model": "SSTZIP-GNN",
        "hidden_dim": 64,
        "num_layers": 3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "k_clusters": 5
    })
    
    # Metrics (logged during training)
    mlflow.log_metric("train_loss", 0.45, step=1)
    mlflow.log_metric("val_loss", 0.52, step=1)
    ...
    
    # Final metrics
    mlflow.log_metrics({
        "MAE": 3.45,
        "RMSE": 5.67,
        "MAPE": 0.12,
        "Zero_Inflation_Score": 0.78
    })
    
    # Artifacts
    mlflow.log_artifact("reports/method2_results.json")
    mlflow.pytorch.log_model(model, "model")
```

### 2.3 MLflow Folder Structure

```
mlruns/
├── 0/                          # Default experiment
└── 1/                          # Taxi-Demand-Prediction
    ├── meta.yaml               # Experiment metadata
    ├── {run_id_1}/
    │   ├── meta.yaml           # Run metadata
    │   ├── metrics/
    │   │   ├── train_loss
    │   │   ├── val_loss
    │   │   ├── MAE
    │   │   ├── RMSE
    │   │   └── ...
    │   ├── params/
    │   │   ├── method
    │   │   ├── time_bucket
    │   │   └── ...
    │   ├── artifacts/
    │   │   ├── model/
    │   │   ├── results.json
    │   │   └── plots/
    │   └── tags/
    │       ├── status
    │       ├── team_member
    │       └── notes
    └── {run_id_2}/
        └── ...
```

---

## 3. Experiment Naming Convention

### 3.1 Standard Format

```
{METHOD}_{TIME_BUCKET}_{VARIANT}_{SEED}_{DATE}

Examples:
- baseline_15_v1_seed42_20240101
- method1_30_v2_seed42_20240102
- method2_60_main_seed42_20240105
- method3_15_ensemble_seed42_20240110
```

### 3.2 Run Tags (in MLflow)

```python
mlflow.set_tags({
    "method": "Method 2",
    "time_bucket": 15,
    "team_member": "Person A",
    "version": "v1",
    "status": "completed",  # or "in_progress", "failed"
    "notes": "Optimized feature normalization"
})
```

---

## 4. Metrics Collection & Logging

### 4.1 Core Metrics

**For each model × time_bucket combination:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Zero-Inflated specific metric
    # How well the model predicts "no demand" (zero) vs "demand" (non-zero)
    zero_mask = y_true == 0
    zero_accuracy = (y_pred[zero_mask] < 0.5).mean()  # Predict zeros correctly
    non_zero_accuracy = (y_pred[~zero_mask] >= 0.5).mean()  # Predict non-zeros correctly
    zero_inflation_score = (zero_accuracy + non_zero_accuracy) / 2
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Zero_Inflation_Score": zero_inflation_score,
        "Zero_Prediction_Accuracy": zero_accuracy,
        "Non_Zero_Prediction_Accuracy": non_zero_accuracy
    }

# Log to MLflow
metrics_dict = compute_metrics(y_test, y_pred_test)
mlflow.log_metrics(metrics_dict)
```

### 4.2 Per-Cluster Metrics

```python
def compute_per_cluster_metrics(y_true, y_pred, cluster_ids):
    """Compute metrics per cluster for granular analysis."""
    
    results = {}
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        y_true_c = y_true[mask]
        y_pred_c = y_pred[mask]
        
        results[f"cluster_{cluster_id}_MAE"] = mean_absolute_error(y_true_c, y_pred_c)
        results[f"cluster_{cluster_id}_RMSE"] = np.sqrt(mean_squared_error(y_true_c, y_pred_c))
        results[f"cluster_{cluster_id}_samples"] = len(y_true_c)
    
    mlflow.log_metrics(results)
    return results
```

### 4.3 Computational Efficiency Metrics

```python
import time

# Log training time
start_time = time.time()
# ... training loop ...
training_time = time.time() - start_time

mlflow.log_metrics({
    "training_time_seconds": training_time,
    "training_time_per_epoch": training_time / num_epochs,
    "inference_time_seconds": inference_time,
    "inference_time_per_sample": inference_time / len(test_data)
})
```

---

## 5. Experiment Tracking Template (Python Script)

```python
# scripts/08_model_training.py

import mlflow
import yaml
from pathlib import Path
from src.models import SSTZIP_GNN
from src.training import Trainer

def run_experiment(method: str, time_bucket: int, version: str = "v1"):
    """Run a single experiment with MLflow logging."""
    
    # Load configuration
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create experiment run
    experiment_name = "Taxi-Demand-Prediction"
    run_name = f"{method}_{time_bucket}_{version}"
    
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # 1. Log configuration
            mlflow.log_params({
                "method": method,
                "time_bucket": time_bucket,
                "model_architecture": config["model"]["architecture"],
                "hidden_dim": config["model"]["hidden_dim"],
                "learning_rate": config["model"]["training"]["learning_rate"],
                "batch_size": config["model"]["training"]["batch_size"],
                "epochs": config["model"]["training"]["epochs"],
                "k_clusters": config["clustering"][method.lower().replace(" ", "_")].get("k", "auto")
            })
            
            mlflow.set_tags({
                "team_member": "Person A",
                "status": "in_progress"
            })
            
            # 2. Load data
            print(f"Loading data for {method} (time_bucket={time_bucket}min)...")
            train_loader, val_loader, test_loader = load_data(method, time_bucket)
            
            # 3. Initialize model
            model = SSTZIP_GNN(
                in_channels=train_loader.dataset[0][0].shape[-1],
                hidden_dim=config["model"]["hidden_dim"],
                num_layers=config["model"]["num_layers"]
            )
            
            # 4. Train model
            trainer = Trainer(
                model=model,
                config=config,
                mlflow_run_id=run.info.run_id
            )
            
            trainer.train(train_loader, val_loader)
            
            # 5. Evaluate on test set
            metrics = trainer.evaluate(test_loader)
            mlflow.log_metrics(metrics)
            
            # 6. Per-cluster metrics
            per_cluster_metrics = trainer.evaluate_per_cluster(test_loader)
            for k, v in per_cluster_metrics.items():
                mlflow.log_metric(k, v)
            
            # 7. Save results
            results_file = f"reports/results_{method}_{time_bucket}.json"
            mlflow.log_artifact(results_file)
            
            # 8. Save model
            mlflow.pytorch.log_model(model, artifact_path="model")
            
            # Mark as complete
            mlflow.set_tag("status", "completed")
            
            print(f"✅ Experiment {run_name} completed!")
            
        except Exception as e:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(e))
            raise e

# Execute all 12 experiments
if __name__ == "__main__":
    methods = ["baseline", "method1", "method2", "method3"]
    time_buckets = [15, 30, 60]
    
    for method in methods:
        for time_bucket in time_buckets:
            run_experiment(method, time_bucket)
```

---

## 6. Comparison Framework

### 6.1 Results Aggregation

```python
import pandas as pd

def aggregate_results():
    """Aggregate all experiment results into comparison table."""
    
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name("Taxi-Demand-Prediction").experiment_id
    
    runs = client.search_runs(experiment_id)
    
    results = []
    for run in runs:
        if run.info.status == "FINISHED":
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "method": run.data.params.get("method"),
                "time_bucket": run.data.params.get("time_bucket"),
                "MAE": run.data.metrics.get("MAE"),
                "RMSE": run.data.metrics.get("RMSE"),
                "MAPE": run.data.metrics.get("MAPE"),
                "Zero_Inflation_Score": run.data.metrics.get("Zero_Inflation_Score"),
                "Training_Time_Sec": run.data.metrics.get("training_time_seconds"),
                "Start_Time": run.info.start_time,
            })
    
    df = pd.DataFrame(results)
    df.to_csv("reports/comparison_results.csv", index=False)
    return df
```

### 6.2 Comparison Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_method_comparison(df):
    """Create comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE by method & time bucket
    pivot_mae = df.pivot_table(values='MAE', index='method', columns='time_bucket')
    pivot_mae.plot(ax=axes[0, 0], marker='o')
    axes[0, 0].set_title('MAE Comparison')
    axes[0, 0].set_ylabel('MAE')
    
    # 2. RMSE by method & time bucket
    pivot_rmse = df.pivot_table(values='RMSE', index='method', columns='time_bucket')
    pivot_rmse.plot(ax=axes[0, 1], marker='o')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    
    # 3. Zero Inflation Score
    pivot_zis = df.pivot_table(values='Zero_Inflation_Score', index='method', columns='time_bucket')
    pivot_zis.plot(ax=axes[1, 0], marker='o')
    axes[1, 0].set_title('Zero-Inflation Score')
    axes[1, 0].set_ylabel('Score')
    
    # 4. Training efficiency (time vs MAE)
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        axes[1, 1].scatter(method_df['Training_Time_Sec'], method_df['MAE'], label=method, s=100)
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Efficiency: MAE vs Training Time')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/method_comparison.png", dpi=300)
    print("✅ Comparison plot saved!")

# Execute
df = aggregate_results()
plot_method_comparison(df)
```

### 6.3 Statistical Significance Testing

```python
from scipy import stats

def statistical_comparison(df):
    """Test if methods are significantly different."""
    
    methods = df['method'].unique()
    time_bucket = 60  # Compare at 60-min bucket for focus
    
    results_per_method = {}
    for method in methods:
        mask = (df['method'] == method) & (df['time_bucket'] == time_bucket)
        results_per_method[method] = df[mask]['MAE'].values
    
    # ANOVA test: are means significantly different?
    f_stat, p_value = stats.f_oneway(*results_per_method.values())
    
    print(f"ANOVA Test (H0: all methods have same MAE)")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant at α=0.05? {p_value < 0.05}")
    
    # Pairwise comparisons (t-tests)
    method_list = list(methods)
    print("\nPairwise Comparisons:")
    for i, m1 in enumerate(method_list):
        for m2 in method_list[i+1:]:
            t_stat, p = stats.ttest_ind(results_per_method[m1], results_per_method[m2])
            print(f"{m1} vs {m2}: p={p:.4f} {'***' if p < 0.05 else 'ns'}")
```

---

## 7. Experiment Log Template

### 7.1 Manual Log (Markdown)

```markdown
# Experiment Log

## Experiment 1: Baseline + 15min
- **Date:** 2024-01-05
- **Duration:** 45 min
- **Team Member:** Person A
- **Status:** ✅ Completed
- **Results:**
  - MAE: 3.82
  - RMSE: 5.45
  - MAPE: 0.14
  - Zero Inflation: 0.75
- **Notes:** Baseline performs as expected. Used all historical data.
- **MLflow Run ID:** 5a2e8c9d

## Experiment 2: Method 1 + 15min
- **Date:** 2024-01-06
- **Duration:** 1h 15min
- **Team Member:** Person B
- **Status:** ✅ Completed
- **Results:**
  - MAE: 3.45 ⬇️ (10% improvement)
  - RMSE: 5.12 ⬇️
  - Zero Inflation: 0.78 ⬆️
- **Notes:** Demand clustering helps reduce error. Interesting cluster semantics.
- **MLflow Run ID:** 7f3a1b2e

...
```

### 7.2 Auto-Generated Summary (from MLflow)

```python
def generate_experiment_summary():
    """Auto-generate summary from MLflow runs."""
    
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("Taxi-Demand-Prediction")
    runs = client.search_runs(exp.experiment_id)
    
    summary_md = "# Experiment Summary\n\n"
    summary_md += f"- **Total Runs:** {len(runs)}\n"
    summary_md += f"- **Completed:** {sum(1 for r in runs if r.info.status == 'FINISHED')}\n"
    summary_md += f"- **Failed:** {sum(1 for r in runs if r.info.status == 'FAILED')}\n\n"
    
    summary_md += "## Results by Method\n\n"
    for method in ["baseline", "method1", "method2", "method3"]:
        method_runs = [r for r in runs if r.data.params.get("method") == method]
        if method_runs:
            avg_mae = np.mean([r.data.metrics.get("MAE", 0) for r in method_runs])
            summary_md += f"- **{method}:** avg MAE = {avg_mae:.3f}\n"
    
    with open("reports/experiment_summary.md", "w") as f:
        f.write(summary_md)
    
    print("✅ Summary saved to reports/experiment_summary.md")

generate_experiment_summary()
```

---

## 8. Hyperparameter Sweep (Optional Advanced)

```python
from optuna import create_study

def hyperparameter_search(method: str, time_bucket: int):
    """Optuna-based hyperparameter optimization."""
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        
        # Train model
        model = SSTZIP_GNN(hidden_dim=hidden_dim, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # ... training loop ...
        val_mae = trainer.validate(val_loader)
        
        return val_mae
    
    study = create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation MAE: {study.best_value:.4f}")

# Only run if time permits (beyond MVP)
# hyperparameter_search("method2", 60)
```

---

## 9. Comparison Results Template

```markdown
# Experimental Results: Method Comparison

## Summary Table (60-min aggregation)

| Method | MAE ↓ | RMSE ↓ | MAPE ↓ | Zero-Inf ↑ | Training Time |
|--------|-------|--------|--------|-----------|----------------|
| Baseline (Zone) | 3.82 | 5.45 | 0.14 | 0.75 | 25 min |
| Method 1 (Demand) | 3.45 | 5.12 | 0.13 | 0.78 | 42 min |
| Method 2 (Mobility) ⭐ | **3.12** | **4.89** | **0.11** | **0.82** | 55 min |
| Method 3 (OD-Flow) | 3.18 | 4.95 | 0.12 | 0.80 | 58 min |

**Key Finding:** Method 2 (Mobility-based clustering) provides best performance with 18% MAE improvement over baseline.

## Per-Time-Bucket Performance

### 15-minute aggregation (fine-grained prediction)
- Best: Method 2 (MAE: 4.23)
- Worst: Baseline (MAE: 5.01)
- Observation: Finer granularity benefits from better spatial clustering

### 30-minute aggregation
- Best: Method 2 (MAE: 3.65)
- Observation: Balanced trade-off between detail and noise

### 60-minute aggregation (coarse-grained)
- Best: Method 2 (MAE: 3.12)
- Observation: All methods converge; spatial clustering still helps

## Statistical Significance

- ANOVA: p < 0.05 (methods significantly different)
- Pairwise: Method 2 vs Baseline: p < 0.01 ***

## Computational Efficiency

- **Fastest training:** Baseline (25 min)
- **Slowest training:** Method 3 (58 min)
- **Best MAE/time ratio:** Method 2 (0.057 MAE per minute)

## Recommendation

**Method 2 (Mobility-Based Clustering)** is recommended for thesis:
✅ Best predictive performance (18% improvement)
✅ Reasonable training time (55 min)
✅ Interpretable cluster semantics
✅ Consistent across all time buckets
```

---

## 10. Checklist Before Final Submission

- [ ] All 12 base experiments completed (4 methods × 3 time buckets)
- [ ] Results reproducible (same random seed)
- [ ] MLflow tracks all runs with proper tagging
- [ ] Comparison table generated
- [ ] Statistical significance tested
- [ ] Visualizations created (comparison plots)
- [ ] Per-cluster metrics computed
- [ ] Computational efficiency analyzed
- [ ] Results documented in thesis-ready format
