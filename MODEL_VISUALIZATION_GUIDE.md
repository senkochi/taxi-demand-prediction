# Model Visualization Guide

## Overview
The visualization module provides comprehensive tools to review your trained SSTZIP-GNN model. Four visualization types are available:

1. **Training History** - Loss curves over epochs
2. **Predictions Analysis** - Detailed prediction evaluation (actual vs predicted, residuals, distributions)
3. **Model Architecture** - Visual representation of the SSTZIP-GNN structure
4. **Metrics Summary** - Dashboard of all evaluation metrics

---

## Generated Visualizations

### 1. Training History (`training_history.png`)
**What it shows:** 
- Left: Combined train/validation loss curves
- Right: Individual loss per epoch

**How to interpret:**
- Downward trend = model improving
- Gap between train/val = potential overfitting
- Validation loss increasing while train decreases = overfitting

**File location:** `reports/figures/training_history.png`

---

### 2. Predictions Analysis (`predictions_analysis.png`)
**6-panel analysis:**

**Panel 1: Actual vs Predicted**
- X-axis: Ground truth demand
- Y-axis: Predicted demand
- Red dashed line: Perfect prediction
- Interpretation: Points on the line = perfect predictions; above = overestimation; below = underestimation

**Panel 2: Residuals Plot**
- X-axis: Predicted demand
- Y-axis: Prediction error (residual)
- Red dashed line: Zero error
- Interpretation: Points clustered near zero = good; wide spread = high error variance

**Panel 3: Error Distribution**
- Histogram of absolute errors
- Red line: Mean absolute error (MAE)
- Interpretation: Narrow distribution = consistent predictions; wide = inconsistent

**Panel 4: Zero Probability Distribution**
- Histogram of π (structural zero probability)
- Interpretation: Shows how confident model is about which zones have zero demand

**Panel 5: Poisson Intensity Distribution**
- Histogram of λ (Poisson intensity parameter)
- Interpretation: Distribution of expected demand values

**Panel 6: Q-Q Plot (Normality Check)**
- Compares residuals to theoretical normal distribution
- Points on red line = normally distributed residuals
- Interpretation: Validates if residuals follow normal distribution assumption

**File location:** `reports/figures/predictions_analysis.png`

---

### 3. Model Architecture (`model_architecture.png`)
**What it shows:**
- Complete SSTZIP-GNN pipeline from input to output
- 6 layers:
  1. **Input:** 261 zones × 20 features
  2. **DGCN:** Spatial layer with 139K parameters
  3. **TCN:** Temporal layer with exponential dilation
  4. **Global Pooling:** Aggregates time dimension
  5. **ZIP Head:** Generates π and λ
  6. **Output:** Predictions with metrics

**Key information displayed:**
- Model parameters and architecture details
- Loss function formula
- Data flow through network

**File location:** `reports/figures/model_architecture.png`

---

### 4. Metrics Summary (`metrics_summary.png`)
**What it shows:**
- Dashboard of all computed metrics
- Each metric in its own box with value
- Easy comparison of model performance

**Common metrics:**
- **MAE:** Mean Absolute Error (lower is better)
- **RMSE:** Root Mean Squared Error (lower is better)
- **MAPE:** Mean Absolute Percentage Error (lower is better)
- **Zero-Inflation Error:** How well model captures structural zeros (lower is better)

**File location:** `reports/figures/metrics_summary.png`

---

## How to Use the Visualization Module

### Quick Start
```python
from src.visualization import ModelVisualization
import numpy as np

# Initialize visualization
viz = ModelVisualization(output_dir="reports/figures")

# Plot training history
history = {
    'train_losses': [5.0, 4.2, 3.5, 2.8, ...],
    'val_losses': [5.1, 4.3, 3.7, 3.0, ...]
}
viz.plot_training_history(history)

# Plot model architecture
viz.plot_model_architecture()

# Plot predictions
y_true = np.array([10, 5, 8, 12, ...])
y_pred_mean = np.array([9.5, 5.2, 7.8, 13.1, ...])
y_pred_zero = np.array([0.2, 0.5, 0.3, 0.1, ...])
y_pred_lambda = np.array([10.1, 5.5, 8.2, 12.8, ...])

viz.plot_predictions_analysis(y_true, y_pred_mean, y_pred_zero, y_pred_lambda)

# Plot metrics
metrics = {
    'MAE': 1.234,
    'RMSE': 1.567,
    'MAPE': 15.234
}
viz.plot_metrics_summary(metrics)
```

### During Training
```python
# After each epoch, update history and plot
viz.plot_training_history(history)
```

### After Training
```python
# Generate all visualizations
viz.plot_training_history(training_history)
viz.plot_predictions_analysis(y_test, y_pred_mean, y_pred_zero, y_pred_lambda)
viz.plot_model_architecture()
viz.plot_metrics_summary(eval_metrics)
```

---

## Integration with Trainer

The visualization can be integrated into the training loop:

```python
from src.visualization import ModelVisualization
from src.training.trainer import SSTZIPGNNTrainer

# After training
trainer = SSTZIPGNNTrainer(model, config)
trainer.train(train_loader, val_loader)

# Load history
with open('checkpoints/training_history.json') as f:
    history = json.load(f)

# Visualize
viz = ModelVisualization()
viz.plot_training_history(history)
```

---

## Interpreting Results

### Good Model Indicators
✅ Training and validation loss both decreasing  
✅ Validation loss not significantly higher than training loss  
✅ Predictions cluster near the "perfect prediction" line  
✅ Residuals centered around zero  
✅ Error distribution relatively narrow  
✅ Residuals approximately normally distributed  

### Warning Signs
⚠️ Training loss decreasing but validation loss increasing (overfitting)  
⚠️ Predictions far from the diagonal line (systematic bias)  
⚠️ Wide spread of residuals (inconsistent predictions)  
⚠️ Large gap between MAE and other metrics (outlier issues)  

---

## File Locations

All visualizations save to `reports/figures/`:
- `training_history.png` - Training curves
- `predictions_analysis.png` - 6-panel prediction analysis
- `model_architecture.png` - Model structure
- `metrics_summary.png` - Metrics dashboard

---

## Customization

### Change output directory
```python
viz = ModelVisualization(output_dir="custom/path/")
```

### Specify save location
```python
viz.plot_training_history(history, save_path="my_custom_plot.png")
```

### Adjust figure sizes
Modify `plt.rcParams['figure.figsize']` before creating visualizations

---

## Technical Details

**Dependencies:**
- matplotlib
- seaborn
- numpy
- pandas

**Image format:** PNG (300 DPI)
**Output size:** Varies by visualization type

---

## Summary

Use these visualizations to:
1. **Monitor training progress** → `training_history.png`
2. **Evaluate prediction quality** → `predictions_analysis.png`
3. **Understand model architecture** → `model_architecture.png`
4. **Assess overall performance** → `metrics_summary.png`

All visualizations are automatically generated when using the `ModelVisualization` class.
