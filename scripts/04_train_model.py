"""
Training Script for SSTZIP-GNN Model - Phase 4
Trains SSTZIP-GNN on all 4 clustering methods
Supports: Baseline, Method 1 (Demand), Method 2 (Mobility), Method 3 (OD-Flow)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import pickle
from datetime import datetime
import yaml
import traceback

from src.models.sstzip_gnn import SSTZIPGNNModel
from src.training.trainer import SSTZIPGNNLightning
from src.evaluation.metrics import Metrics
from src.visualization import ModelVisualization
from src.data.data_loader import TaxiDemandDataModule


def run_experiment(method: str, config: dict, data_module: TaxiDemandDataModule,
                   adj_matrix: torch.Tensor) -> dict:
    """
    Run single training experiment for a clustering method
    
    Args:
        method: Clustering method (baseline, method1, method2, method3)
        config: Configuration dictionary
        data_module: Initialized TaxiDemandDataModule
        adj_matrix: Adjacency matrix [num_zones, num_zones]
    
    Returns:
        results dict with metrics and metadata
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING SSTZIP-GNN: {method.upper()}")
    print(f"{'='*80}\n")
    
    try:
        # =========================================================================
        # 1. CONFIGURATION
        # =========================================================================
        model_config = config.get('model', {})
        training_config = model_config.get('training', {})
        
        print(f"[Config] Sequence length: {data_module.sequence_length}")
        print(f"[Config] Batch size: {data_module.batch_size}")
        print(f"[Config] Learning rate: {training_config.get('learning_rate', 0.001)}")
        print(f"[Config] Epochs: {training_config.get('epochs', 50)}")
            
        # =========================================================================
        # 2. GET DATA LOADERS
        # =========================================================================
        print(f"[1/7] Getting data loaders for {method}...")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        print(f"✓ DataLoaders ready")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Adjacency matrix: {adj_matrix.shape}")
            
        # =========================================================================
        # 3. GET FEATURE DIMENSION
        # =========================================================================
        print(f"\n[2/7] Getting feature dimension from data...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get feature dimension from dataset
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (tuple, list)):
            x_sample = sample_batch[0]
        else:
            x_sample = sample_batch
        
        feature_dim = x_sample.shape[-1] if x_sample.dim() > 2 else x_sample.shape[1]
        print(f"✓ Feature dimension: {feature_dim}")
        
        # =========================================================================
        # 3. INITIALIZE MODEL
        # =========================================================================
        print(f"\n[3/7] Initializing SSTZIP-GNN model...")
        
        model = SSTZIPGNNModel(
            num_zones=adj_matrix.shape[0],
            feature_dim=feature_dim,
            spatial_dim=model_config.get('spatial_dim', 64),
            temporal_dim=model_config.get('temporal_dim', 64),
            num_spatial_layers=model_config.get('num_spatial_layers', 2),
            num_spatial_hops=model_config.get('num_spatial_hops', 3),
            num_temporal_layers=model_config.get('num_temporal_layers', 3),
            hidden_dim_zip=model_config.get('hidden_dim_zip', 128),
            dropout=model_config.get('dropout', 0.2)
        )
        
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized: {num_params:,} parameters")
        
        # Move adjacency matrix to device
        adj_matrix = adj_matrix.to(device)
            
        # =========================================================================
        # 4. CREATE LIGHTNING TRAINER
        # =========================================================================
        print(f"\n[4/7] Setting up PyTorch Lightning trainer...")
        
        trainer_module = SSTZIPGNNLightning(
            model=model,
            feature_dim=feature_dim,
            learning_rate=training_config.get('learning_rate', 0.001),
            weight_decay=1e-5,
            patience=training_config.get('early_stopping_patience', 10),
            accumulation_steps=1
        )
        
        # Store adjacency matrix in trainer module for use in forward pass
        trainer_module.adj_matrix = adj_matrix
        
        epochs = training_config.get('epochs', 50)
        
        pl_trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=training_config.get('early_stopping_patience', 10),
                    mode='min'
                ),
                pl.callbacks.ModelCheckpoint(
                    dirpath=f'data/models/sstzip_gnn/{method}',
                    filename='epoch-{epoch:02d}-val_loss-{val_loss:.3f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
            ]
        )
            
        # =========================================================================
        # 5. TRAIN MODEL
        # =========================================================================
        print(f"\n[5/7] Training model ({epochs} epochs)...")
        start_time = datetime.now()
        
        pl_trainer.fit(trainer_module, train_dataloaders=train_loader, 
                      val_dataloaders=val_loader)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Training complete in {training_time:.1f} seconds")
            
        # =========================================================================
        # 6. EVALUATE ON TEST SET
        # =========================================================================
        print(f"\n[6/7] Evaluating on test set...")
        
        model.eval()
        test_predictions = []
        test_targets = []
        test_zones = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if isinstance(batch, (tuple, list)):
                    x_batch = batch[0]
                    y_batch = batch[1] if len(batch) > 1 else None
                else:
                    x_batch = batch
                    y_batch = None
                
                x_batch = x_batch.to(device)
                
                # Transpose for temporal encoder: [batch, features, seq_len]
                if x_batch.dim() == 3:
                    x_batch = x_batch.transpose(1, 2)
                
                # Project features to spatial_dim
                x_proj = trainer_module.feature_projection(x_batch.transpose(1, 2))
                x_proj = x_proj.transpose(1, 2)
                
                # Forward through temporal encoder
                x_temporal = model.temporal_encoder(x_proj)
                x_agg = model.temporal_pooling(x_temporal)
                pi, lam = model.zip_head(x_agg)
                
                # Prediction: E[y] = (1-π)*λ
                pred = (1 - pi) * lam
                test_predictions.append(pred.cpu().numpy())
                
                if y_batch is not None:
                    test_targets.append(y_batch.cpu().numpy())
        
        y_pred = np.concatenate(test_predictions, axis=0).flatten()
        y_test = np.concatenate(test_targets, axis=0).flatten() if test_targets else np.zeros_like(y_pred)
            
        # =========================================================================
        # 7. COMPUTE METRICS
        # =========================================================================
        print(f"\n[7/7] Computing metrics...")
        
        mae = Metrics.mean_absolute_error(y_pred, y_test) if len(y_test) > 0 else 0.0
        rmse = Metrics.root_mean_squared_error(y_pred, y_test) if len(y_test) > 0 else 0.0
        mape = Metrics.mean_absolute_percentage_error(y_pred, y_test) if len(y_test) > 0 else 0.0
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'Training_Time_Sec': training_time,
            'Num_Parameters': num_params,
        }
        
        print(f"\n✓ Test Metrics:")
        print(f"  - MAE:  {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAPE: {mape:.4f}%")
        print(f"  - Training Time: {training_time:.1f}s")
            
        # =========================================================================
        # 8. SAVE RESULTS
        # =========================================================================
        print(f"\n[8/8] Saving results...")
        print(f"✓ Feature dimension used: {feature_dim}")
        
        results_dir = Path(f'checkpoints/{method}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump({
                'test_metrics': metrics,
                'method': method,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save model
        model_path = results_dir / 'model.pt'
        torch.save(model.state_dict(), model_path)
        
        print(f"✓ Results saved to {results_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"\n[ERROR] Error in experiment {method}: {str(e)}")
        traceback.print_exc()
        raise e


def main():
    """
    Main training orchestration for Phase 4: Deep Learning Modeling
    
    Trains SSTZIP-GNN on all 4 clustering methods
    - Baseline: Zone-based aggregation
    - Method 1: Demand-based clustering
    - Method 2: Mobility pattern clustering
    - Method 3: OD-flow based clustering
    """
    
    print("\n" + "="*80)
    print("PHASE 4: DEEP LEARNING MODELING - SSTZIP-GNN TRAINING")
    print("="*80)
    
    # =========================================================================
    # SETUP: Load configuration
    # =========================================================================
    print("\n[Setup] Loading configuration...")
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # DATA LOADING & ADJACENCY MATRIX
    # =========================================================================
    print("\n[Data] Initializing data module and loading adjacency matrices...")
    
    methods = ["baseline", "method1", "method2", "method3"]
    results_summary = {}
    
    for method in methods:
        print(f"\n{'-'*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'-'*80}")
        
        try:
            # Create data module for this method
            data_module = TaxiDemandDataModule(
                duckdb_path="data/processed/taxi_features.duckdb",
                clustering_method=method,
                sequence_length=config.get('model', {}).get('seq_length', 96),
                forecast_horizon=1,
                batch_size=config.get('model', {}).get('training', {}).get('batch_size', 32),
                num_workers=0,  # Set to 0 to avoid DataLoader issues
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            # Setup data module (load data and create datasets)
            print(f"Setting up data module for {method}...")
            data_module.setup()
            
            # Get adjacency matrix from clustering
            print(f"Creating adjacency matrix from {method} clustering...")
            adj_matrix = data_module.adjacency_matrix
            print(f"✓ Adjacency matrix: {adj_matrix.shape}")
            
            # =====================================================================
            # TRAIN EXPERIMENT
            # =====================================================================
            metrics = run_experiment(method, config, data_module, adj_matrix)
            results_summary[method] = metrics
            
            print(f"\n[OK] {method.upper()} training completed successfully!")
            
        except FileNotFoundError as e:
            print(f"\n[SKIP] Skipping {method}: {str(e)}")
            print("   Make sure clustering results exist in data/models/")
            continue
        except Exception as e:
            print(f"\n[ERROR] Error training {method}: {str(e)}")
            traceback.print_exc()
            continue
    
    # =========================================================================
    # COMPARISON & SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80 + "\n")
    
    if results_summary:
        # Create comparison DataFrame
        df_results = pd.DataFrame(results_summary).T
        print(df_results.to_string())
        
        # Save summary
        summary_path = Path("reports/training_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(summary_path)
        print(f"\n✓ Summary saved to {summary_path}")
        
        # Find best method
        best_method = df_results['MAE'].idxmin()
        best_mae = df_results.loc[best_method, 'MAE']
        print(f"\n🏆 Best Method: {best_method} (MAE: {best_mae:.4f})")
    else:
        print("❌ No experiments completed successfully!")
    
    print("\n" + "="*80)
    print("✅ PHASE 4 TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
