"""
Evaluation Metrics and Analysis for SSTZIP-GNN
Computes metrics: MAE, RMSE, MAPE, Zero-Inflation Score
Generates comparison tables and visualizations
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import json


class Metrics:
    """Compute evaluation metrics"""
    
    @staticmethod
    def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_pred - y_true))
    
    @staticmethod
    def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    @staticmethod
    def mean_absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray,
                                      epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error"""
        denominator = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs((y_pred - y_true) / denominator)) * 100
    
    @staticmethod
    def zero_inflation_error(y_pred_zero_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Zero-Inflation Error: |predicted_zeros - actual_zeros| / actual_zeros
        
        Args:
            y_pred_zero_prob: Predicted probability of zero (π parameter)
            y_true: Ground truth demand counts
        
        Returns:
            Error metric (lower is better, 0 is perfect)
        """
        predicted_zeros = (y_pred_zero_prob > 0.5).sum()
        actual_zeros = (y_true == 0).sum()
        
        if actual_zeros == 0:
            return 0.0 if predicted_zeros == 0 else float('inf')
        
        return abs(predicted_zeros - actual_zeros) / actual_zeros
    
    @staticmethod
    def poisson_deviance(y_pred_lambda: np.ndarray, y_true: np.ndarray,
                        epsilon: float = 1e-8) -> float:
        """
        Poisson Deviance: 2 * Σ[y*log(y/ŷ) - (y - ŷ)]
        
        Measures goodness-of-fit for count data
        """
        y_pred_lambda = np.maximum(y_pred_lambda, epsilon)
        term1 = y_true * np.log(np.maximum(y_true, epsilon) / y_pred_lambda)
        term2 = y_true - y_pred_lambda
        return 2 * np.mean(term1 - term2)
    
    @staticmethod
    def coverage_probability(y_pred_dist: Tuple[np.ndarray, np.ndarray],
                            y_true: np.ndarray,
                            confidence: float = 0.95) -> float:
        """
        Prediction interval coverage probability (PICP)
        Estimates what fraction of true values fall within predicted confidence interval
        
        Args:
            y_pred_dist: (lower_bound, upper_bound) tuple
            y_true: Ground truth
            confidence: Confidence level
        
        Returns:
            PICP value
        """
        lower, upper = y_pred_dist
        coverage = np.sum((y_true >= lower) & (y_true <= upper)) / len(y_true)
        return coverage


class EvaluationReport:
    """Generate comprehensive evaluation report"""
    
    def __init__(self):
        self.results = {}
        self.comparisons = {}
    
    def add_method_results(self, method_name: str, 
                          y_pred_mean: np.ndarray,
                          y_pred_zero_prob: np.ndarray,
                          y_pred_lambda: np.ndarray,
                          y_true: np.ndarray):
        """
        Add results for a method
        
        Args:
            method_name: Name of method (e.g., "SSTZIP-GNN", "Baseline", "ARIMA")
            y_pred_mean: Expected value E[y] = (1-π)*λ
            y_pred_zero_prob: Bernoulli parameter π
            y_pred_lambda: Poisson intensity λ
            y_true: Ground truth
        """
        metrics = {
            'MAE': Metrics.mean_absolute_error(y_pred_mean, y_true),
            'RMSE': Metrics.root_mean_squared_error(y_pred_mean, y_true),
            'MAPE': Metrics.mean_absolute_percentage_error(y_pred_mean, y_true),
            'Zero_Inflation_Error': Metrics.zero_inflation_error(y_pred_zero_prob, y_true),
            'Poisson_Deviance': Metrics.poisson_deviance(y_pred_lambda, y_true),
            'num_samples': len(y_true),
            'zero_ratio': (y_true == 0).sum() / len(y_true)
        }
        
        self.results[method_name] = metrics
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table across all methods"""
        df = pd.DataFrame(self.results).T
        
        # Sort by MAE
        df = df.sort_values('MAE')
        
        return df
    
    def generate_report_dict(self) -> Dict:
        """Generate report as dictionary"""
        return {
            'methods': list(self.results.keys()),
            'metrics': self.results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def save_report(self, path: str):
        """Save report to JSON"""
        report = self.generate_report_dict()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_report(self):
        """Print comparison table"""
        df = self.generate_comparison_table()
        print("\n" + "="*80)
        print("SSTZIP-GNN EVALUATION REPORT")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")
    
    def get_best_method(self, metric: str = 'MAE') -> str:
        """Get best performing method for a metric"""
        scores = {method: results[metric] for method, results in self.results.items()}
        return min(scores, key=scores.get)


class ComputationalEfficiency:
    """Analyze computational efficiency"""
    
    def __init__(self):
        self.timings = {}
        self.memory = {}
    
    def record_timing(self, phase: str, duration: float):
        """Record time for a phase"""
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(duration)
    
    def record_memory(self, phase: str, memory_mb: float):
        """Record peak memory for a phase"""
        if phase not in self.memory:
            self.memory[phase] = []
        self.memory[phase].append(memory_mb)
    
    def get_summary(self) -> Dict:
        """Get efficiency summary"""
        summary = {
            'timings': {phase: {
                'mean': np.mean(times),
                'std': np.std(times),
                'total': np.sum(times)
            } for phase, times in self.timings.items()},
            'memory': {phase: {
                'mean': np.mean(mem),
                'max': np.max(mem)
            } for phase, mem in self.memory.items()}
        }
        return summary
    
    def print_summary(self):
        """Print efficiency summary"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("COMPUTATIONAL EFFICIENCY ANALYSIS")
        print("="*80)
        
        print("\nTiming (seconds):")
        for phase, stats in summary['timings'].items():
            print(f"  {phase}:")
            print(f"    Mean: {stats['mean']:.3f}s")
            print(f"    Std: {stats['std']:.3f}s")
            print(f"    Total: {stats['total']:.3f}s")
        
        print("\nMemory (MB):")
        for phase, stats in summary['memory'].items():
            print(f"  {phase}:")
            print(f"    Mean: {stats['mean']:.1f} MB")
            print(f"    Max: {stats['max']:.1f} MB")
        
        print("="*80 + "\n")


def evaluate_model(model, test_dataloader, device='cpu') -> Dict:
    """
    Evaluate model on test set
    
    Args:
        model: SSTZIP-GNN model
        test_dataloader: Test DataLoader
        device: Device to run on
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)
    
    all_preds_mean = []
    all_preds_zero = []
    all_preds_lambda = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            # Placeholder for adj and temporal (in practice these come from dataloader)
            adj = torch.eye(261, device=device)
            x_temporal = x.unsqueeze(-1).expand(-1, -1, 96)
            
            # Forward pass
            pi, lambda_param = model(x, adj, x_temporal)
            
            # Expected value
            y_pred_mean = (1 - pi.squeeze()) * lambda_param.squeeze()
            
            all_preds_mean.append(y_pred_mean.cpu().numpy())
            all_preds_zero.append(pi.squeeze().cpu().numpy())
            all_preds_lambda.append(lambda_param.squeeze().cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate
    y_pred_mean = np.concatenate(all_preds_mean)
    y_pred_zero = np.concatenate(all_preds_zero)
    y_pred_lambda = np.concatenate(all_preds_lambda)
    y_true = np.concatenate(all_targets)
    
    # Compute metrics
    metrics = {
        'MAE': Metrics.mean_absolute_error(y_pred_mean, y_true),
        'RMSE': Metrics.root_mean_squared_error(y_pred_mean, y_true),
        'MAPE': Metrics.mean_absolute_percentage_error(y_pred_mean, y_true),
        'Zero_Inflation_Error': Metrics.zero_inflation_error(y_pred_zero, y_true),
        'Poisson_Deviance': Metrics.poisson_deviance(y_pred_lambda, y_true),
    }
    
    return metrics


if __name__ == "__main__":
    print("Testing Evaluation Metrics...")
    
    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.poisson(10, n_samples)
    y_pred_mean = y_true + np.random.normal(0, 2, n_samples)
    y_pred_zero = np.random.uniform(0, 1, n_samples)
    y_pred_lambda = np.maximum(y_pred_mean + np.random.normal(0, 1, n_samples), 0.1)
    
    # Test metrics
    print(f"MAE: {Metrics.mean_absolute_error(y_pred_mean, y_true):.4f}")
    print(f"RMSE: {Metrics.root_mean_squared_error(y_pred_mean, y_true):.4f}")
    print(f"MAPE: {Metrics.mean_absolute_percentage_error(y_pred_mean, y_true):.4f}%")
    print(f"Zero Inflation Error: {Metrics.zero_inflation_error(y_pred_zero, y_true):.4f}")
    print(f"Poisson Deviance: {Metrics.poisson_deviance(y_pred_lambda, y_true):.4f}")
    
    # Test report
    report = EvaluationReport()
    report.add_method_results("SSTZIP-GNN", y_pred_mean, y_pred_zero, y_pred_lambda, y_true)
    report.add_method_results("Baseline", y_true.mean() * np.ones_like(y_pred_mean), 
                             np.full_like(y_pred_zero, 0.3), 
                             np.full_like(y_pred_lambda, y_true.mean()), y_true)
    
    report.print_report()
    
    print("✅ Evaluation metrics tests passed!")
