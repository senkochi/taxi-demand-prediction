"""
Example: Using Visualizations to Review Trained SSTZIP-GNN Model
Demonstrates all visualization capabilities
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from src.visualization import ModelVisualization


def review_trained_model_example():
    """
    Complete example showing how to review a trained model
    """
    
    print("=" * 80)
    print("SSTZIP-GNN MODEL REVIEW - VISUALIZATION EXAMPLE")
    print("=" * 80)
    
    # Initialize visualization
    viz = ModelVisualization(output_dir="reports/figures")
    print("\n✓ Visualization module initialized")
    
    # =========================================================================
    # 1. PLOT MODEL ARCHITECTURE (no training needed)
    # =========================================================================
    print("\n[1/4] Generating model architecture diagram...")
    viz.plot_model_architecture()
    print("✓ Saved to: reports/figures/model_architecture.png")
    
    # =========================================================================
    # 2. LOAD AND PLOT TRAINING HISTORY
    # =========================================================================
    print("\n[2/4] Generating training history plots...")
    
    # Example: Load from saved history
    history_file = Path("checkpoints/training_history.json")
    if history_file.exists():
        with open(history_file) as f:
            history_data = json.load(f)
            history = {
                'train_losses': history_data.get('train_losses', []),
                'val_losses': history_data.get('val_losses', [])
            }
        viz.plot_training_history(history)
        print(f"✓ Loaded history from {history_file}")
    else:
        # Simulate training history for demo
        print("  (Training history file not found, generating synthetic data for demo)")
        epochs = 50
        history = {
            'train_losses': (5.0 * np.exp(-np.linspace(0, 2, epochs)) + 
                            np.random.normal(0, 0.1, epochs)).tolist(),
            'val_losses': (5.2 * np.exp(-np.linspace(0, 1.8, epochs)) + 
                          np.random.normal(0, 0.15, epochs)).tolist()
        }
        viz.plot_training_history(history)
    
    print("✓ Saved to: reports/figures/training_history.png")
    
    # =========================================================================
    # 3. PLOT PREDICTIONS ANALYSIS
    # =========================================================================
    print("\n[3/4] Generating predictions analysis...")
    
    # Generate synthetic test predictions
    # In practice, these come from model.predict(test_set)
    np.random.seed(42)
    n_test = 1000
    
    # Ground truth
    y_true = np.random.poisson(15, n_test)
    
    # Model predictions
    y_pred_mean = y_true + np.random.normal(0, 2.5, n_test)
    y_pred_zero = np.clip(0.2 * (y_true == 0).astype(float) + 
                          np.random.uniform(-0.1, 0.1, n_test), 0, 1)
    y_pred_lambda = np.maximum(y_pred_mean + np.random.normal(0, 1.5, n_test), 0.5)
    
    viz.plot_predictions_analysis(y_true, y_pred_mean, y_pred_zero, y_pred_lambda)
    print("✓ Saved to: reports/figures/predictions_analysis.png")
    
    # =========================================================================
    # 4. PLOT METRICS SUMMARY
    # =========================================================================
    print("\n[4/4] Generating metrics summary dashboard...")
    
    # Compute or load metrics
    from src.evaluation.metrics import Metrics
    
    metrics = {
        'MAE': Metrics.mean_absolute_error(y_pred_mean, y_true),
        'RMSE': Metrics.root_mean_squared_error(y_pred_mean, y_true),
        'MAPE': Metrics.mean_absolute_percentage_error(y_pred_mean, y_true),
        'Zero-Inflation Error': Metrics.zero_inflation_error(y_pred_zero, y_true)
    }
    
    viz.plot_metrics_summary(metrics)
    print("✓ Saved to: reports/figures/metrics_summary.png")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  1. reports/figures/model_architecture.png - SSTZIP-GNN structure")
    print("  2. reports/figures/training_history.png - Loss curves over epochs")
    print("  3. reports/figures/predictions_analysis.png - Prediction evaluation")
    print("  4. reports/figures/metrics_summary.png - Performance metrics")
    
    print("\nMetrics Summary:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n💡 Tip: Open the PNG files to review your model's performance!")
    print("=" * 80)


def review_per_zone_performance():
    """
    Example: Review model performance for top/bottom zones
    """
    print("\n" + "=" * 80)
    print("ZONE-LEVEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    viz = ModelVisualization(output_dir="reports/figures")
    
    # Simulate zone-level predictions
    zones = list(range(261))
    y_true_by_zone = {}
    y_pred_by_zone = {}
    
    for zone in zones:
        n_samples = 100
        y_true = np.random.poisson(10, n_samples)
        y_pred = y_true + np.random.normal(0, 2, n_samples)
        
        y_true_by_zone[zone] = y_true
        y_pred_by_zone[zone] = y_pred
    
    # Plot zone performance
    viz.plot_zone_performance(
        zones, 
        y_true_by_zone, 
        y_pred_by_zone,
        metric_fn=lambda y_t, y_p: np.mean(np.abs(y_t - y_p))
    )
    
    print("✓ Saved to: reports/figures/zone_performance.png")
    print("\nShows:")
    print("  - Top 10 best performing zones (green)")
    print("  - Top 10 worst performing zones (red)")


if __name__ == "__main__":
    # Run main review
    review_trained_model_example()
    
    # Run zone-level analysis
    review_per_zone_performance()
    
    print("\n" + "=" * 80)
    print("✅ Model review complete! Check reports/figures/ for all visualizations")
    print("=" * 80)
