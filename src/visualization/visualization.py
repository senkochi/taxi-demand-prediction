"""
Visualization Module for SSTZIP-GNN Model Training and Evaluation
Generates comprehensive plots for model review including:
- Training history (loss curves, metrics)
- Prediction analysis (actual vs predicted)
- Model performance by zone
- Architecture visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class ModelVisualization:
    """Generate visualizations for model training and evaluation"""
    
    def __init__(self, output_dir: str = "reports/figures"):
        """Initialize visualization module"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: Optional[str] = None):
        """
        Plot training and validation loss curves
        
        Args:
            history: Dict with keys 'train_losses' and 'val_losses'
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Combined loss
        axes[0].plot(history.get('train_losses', []), label='Train Loss', linewidth=2)
        axes[0].plot(history.get('val_losses', []), label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training History - Loss Curves', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Individual losses
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        if len(train_losses) > 0:
            axes[1].plot(train_losses, label='Train', alpha=0.7, linewidth=2)
        if len(val_losses) > 0:
            axes[1].plot(val_losses, label='Validation', alpha=0.7, linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Loss per Epoch', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            save_path = self.output_dir / "training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.close()
    
    def plot_predictions_analysis(self, y_true: np.ndarray, 
                                 y_pred_mean: np.ndarray,
                                 y_pred_zero: np.ndarray,
                                 y_pred_lambda: np.ndarray,
                                 save_path: Optional[str] = None):
        """
        Plot prediction analysis with multiple views
        
        Args:
            y_true: Ground truth demands
            y_pred_mean: Predicted mean (E[y] = (1-π)*λ)
            y_pred_zero: Predicted zero probability (π)
            y_pred_lambda: Predicted Poisson intensity (λ)
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        sample_size = min(1000, len(y_true))
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        ax1.scatter(y_true[indices], y_pred_mean[indices], alpha=0.5, s=20)
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred_mean.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Demand', fontsize=11)
        ax1.set_ylabel('Predicted Demand', fontsize=11)
        ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_pred_mean - y_true
        ax2.scatter(y_pred_mean[indices], residuals[indices], alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Demand', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        errors = np.abs(y_pred_mean - y_true)
        ax3.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
        ax3.set_xlabel('Absolute Error', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Zero Probability Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(y_pred_zero, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('Predicted Zero Probability (π)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Zero-Inflation Parameter Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Poisson Intensity Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(y_pred_lambda, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax5.set_xlabel('Predicted Poisson Intensity (λ)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Poisson Intensity Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Quantile-Quantile Plot
        ax6 = fig.add_subplot(gs[1, 2])
        sorted_residuals = np.sort(residuals)
        theoretical_q = np.quantile(np.random.normal(residuals.mean(), residuals.std(), 10000), 
                                   np.linspace(0, 1, len(residuals)))
        ax6.scatter(theoretical_q, sorted_residuals, alpha=0.5, s=20)
        
        min_val = min(theoretical_q.min(), sorted_residuals.min())
        max_val = max(theoretical_q.max(), sorted_residuals.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax6.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax6.set_ylabel('Sample Quantiles', fontsize=11)
        ax6.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "predictions_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_zone_performance(self, zones: List[int], 
                             y_true_by_zone: Dict[int, np.ndarray],
                             y_pred_by_zone: Dict[int, np.ndarray],
                             metric_fn=None,
                             save_path: Optional[str] = None):
        """
        Plot model performance by zone (top/bottom performers)
        
        Args:
            zones: List of zone IDs
            y_true_by_zone: Dict mapping zone_id to ground truth
            y_pred_by_zone: Dict mapping zone_id to predictions
            metric_fn: Function to compute metric (MAE by default)
            save_path: Optional path to save figure
        """
        if metric_fn is None:
            metric_fn = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
        
        # Compute metrics per zone
        zone_metrics = {}
        for zone in zones:
            if zone in y_true_by_zone and zone in y_pred_by_zone:
                metric = metric_fn(y_true_by_zone[zone], y_pred_by_zone[zone])
                zone_metrics[zone] = metric
        
        # Sort and get top/bottom
        sorted_zones = sorted(zone_metrics.items(), key=lambda x: x[1])
        top_zones = sorted_zones[-10:]  # Best 10
        bottom_zones = sorted_zones[:10]  # Worst 10
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Best zones
        zones_best = [z for z, _ in top_zones]
        metrics_best = [m for _, m in top_zones]
        axes[0].barh(range(len(zones_best)), metrics_best, color='green', alpha=0.7)
        axes[0].set_yticks(range(len(zones_best)))
        axes[0].set_yticklabels([f"Zone {z}" for z in zones_best])
        axes[0].set_xlabel('Metric Value', fontsize=11)
        axes[0].set_title('Top 10 Best Performing Zones', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Worst zones
        zones_worst = [z for z, _ in bottom_zones]
        metrics_worst = [m for _, m in bottom_zones]
        axes[1].barh(range(len(zones_worst)), metrics_worst, color='red', alpha=0.7)
        axes[1].set_yticks(range(len(zones_worst)))
        axes[1].set_yticklabels([f"Zone {z}" for z in zones_worst])
        axes[1].set_xlabel('Metric Value', fontsize=11)
        axes[1].set_title('Top 10 Worst Performing Zones', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "zone_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_model_architecture(self, save_path: Optional[str] = None):
        """
        Create a visual representation of SSTZIP-GNN architecture
        
        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'SSTZIP-GNN Architecture', 
               fontsize=18, fontweight='bold', ha='center')
        
        # Color scheme
        colors = {
            'input': '#E8F4F8',
            'spatial': '#FFE5CC',
            'temporal': '#E5CCFF',
            'aggregation': '#CCFFE5',
            'output': '#FFE5E5'
        }
        
        # Layer definitions
        layers = [
            {'y': 8.5, 'label': 'Input Layer', 'desc': '261 zones × 20 features', 'color': colors['input']},
            {'y': 7.2, 'label': 'DGCN (Spatial)', 'desc': 'Multi-hop diffusion\n261 zones → 64 channels', 'color': colors['spatial']},
            {'y': 5.9, 'label': 'TCN (Temporal)', 'desc': 'Causal convolutions\n96 timesteps → 64 channels', 'color': colors['temporal']},
            {'y': 4.6, 'label': 'Global Pooling', 'desc': 'Mean aggregation\nOver time', 'color': colors['aggregation']},
            {'y': 3.3, 'label': 'ZIP Head', 'desc': 'π (Bernoulli) + λ (Poisson)', 'color': colors['output']},
            {'y': 2.0, 'label': 'Output', 'desc': 'Predictions: E[y], π, λ', 'color': colors['output']},
        ]
        
        # Draw boxes and arrows
        box_width = 3
        box_height = 0.8
        
        for i, layer in enumerate(layers):
            y = layer['y']
            
            # Draw box
            fancy_box = FancyBboxPatch((5 - box_width/2, y - box_height/2),
                                      box_width, box_height,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor=layer['color'],
                                      linewidth=2)
            ax.add_patch(fancy_box)
            
            # Add text
            ax.text(5, y + 0.15, layer['label'], 
                   fontsize=12, fontweight='bold', ha='center', va='center')
            ax.text(5, y - 0.3, layer['desc'], 
                   fontsize=9, ha='center', va='center', style='italic')
            
            # Draw arrows between layers
            if i < len(layers) - 1:
                next_y = layers[i + 1]['y']
                arrow = FancyArrowPatch((5, y - box_height/2 - 0.1),
                                       (5, next_y + box_height/2 + 0.1),
                                       arrowstyle='->', mutation_scale=20,
                                       color='black', linewidth=2)
                ax.add_patch(arrow)
        
        # Add side annotations
        ax.text(0.5, 8.5, 'Input: 261 zones\n20 features each', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 7.2, '139K params\nResidual\nconnections', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 5.9, 'Exponential\ndilation\n2^0, 2^1, 2^2', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 4.6, 'Mean over\ntime', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 3.3, 'Zero-inflated\nPoisson output', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 2.0, 'MAE, RMSE\nZero-Inflation', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend for loss function
        ax.text(7, 0.8, 'Loss Function:', fontsize=10, fontweight='bold')
        ax.text(7, 0.3, 'y=0:  -log[π + (1-π)e^(-λ)]\ny>0: -log[(1-π)Poisson(y|λ)]', 
               fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "model_architecture.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_metrics_summary(self, metrics: Dict[str, float], 
                            save_path: Optional[str] = None):
        """
        Create a summary dashboard of all metrics
        
        Args:
            metrics: Dict with metric names and values
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(14, 8))
        
        # Create grid for metric boxes
        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)
        
        # Arrange in grid
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
        
        for idx, (name, value) in enumerate(metrics.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Create text box
            ax.text(0.5, 0.6, name, fontsize=14, fontweight='bold', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.text(0.5, 0.25, f'{value:.4f}', fontsize=20, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes, color='darkblue')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Add border
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                                 transform=ax.transAxes, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        
        fig.suptitle('SSTZIP-GNN Model Metrics Summary', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path is None:
            save_path = self.output_dir / "metrics_summary.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Testing Visualization Module...")
    
    viz = ModelVisualization()
    
    # Test 1: Training history
    history = {
        'train_losses': np.random.uniform(1, 5, 50) * np.linspace(1, 0.1, 50),
        'val_losses': np.random.uniform(1.2, 5.5, 50) * np.linspace(1, 0.15, 50)
    }
    viz.plot_training_history(history)
    print("✓ Training history plot created")
    
    # Test 2: Predictions analysis
    y_true = np.random.poisson(10, 1000)
    y_pred_mean = y_true + np.random.normal(0, 2, 1000)
    y_pred_zero = np.random.uniform(0, 0.5, 1000)
    y_pred_lambda = y_pred_mean + np.random.normal(0, 1, 1000)
    
    viz.plot_predictions_analysis(y_true, y_pred_mean, y_pred_zero, y_pred_lambda)
    print("✓ Predictions analysis plot created")
    
    # Test 3: Architecture diagram
    viz.plot_model_architecture()
    print("✓ Model architecture diagram created")
    
    # Test 4: Metrics summary
    metrics = {
        'MAE': 1.234,
        'RMSE': 1.567,
        'MAPE': 15.234,
        'Zero-Inflation Error': 0.234
    }
    viz.plot_metrics_summary(metrics)
    print("✓ Metrics summary created")
    
    print("\n✅ All visualization tests passed!")
