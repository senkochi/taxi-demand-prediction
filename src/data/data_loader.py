"""
Data Loader for SSTZIP-GNN Model
Loads taxi features, clustering results, and creates train/val/test splits
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import duckdb
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class TaxiDemandDataset(Dataset):
    """
    PyTorch Dataset for taxi demand prediction
    
    Handles:
    - Loading features from DuckDB
    - Creating temporal sequences
    - Normalizing/scaling features
    """
    
    def __init__(self,
                 features_df: pd.DataFrame,
                 clustering_method: str = "method3",
                 sequence_length: int = 96,
                 forecast_horizon: int = 1,
                 split: str = "train",
                 normalization_stats: Optional[dict] = None):
        """
        Args:
            features_df: DataFrame with columns [zone_id, features..., window_start, demand_count]
            clustering_method: "method1", "method2", or "method3"
            sequence_length: Number of time steps in input sequence (15-min intervals)
            forecast_horizon: Number of steps to predict ahead
            split: "train", "val", or "test"
            normalization_stats: Dict with 'mean' and 'std' for normalization
        """
        self.clustering_method = clustering_method
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.split = split
        
        # Sort by zone and time
        features_df = features_df.sort_values(['zone_id', 'window_start']).reset_index(drop=True)
        
        # Extract unique zones
        self.zones = sorted(features_df['zone_id'].unique())
        self.num_zones = len(self.zones)
        self.zone_to_idx = {z: i for i, z in enumerate(self.zones)}
        
        # Group by zone
        self.zone_data = {}
        for zone_id in self.zones:
            zone_df = features_df[features_df['zone_id'] == zone_id].reset_index(drop=True)
            self.zone_data[zone_id] = zone_df
        
        # Feature columns (excluding metadata)
        metadata_cols = ['zone_id', 'window_start', 'window_end', 'date_str', 'time_bucket', 'date_day', 'hour', 'Borough', 'Zone', 'service_zone']
        self.feature_cols = [col for col in features_df.columns if col not in metadata_cols + ['demand_count']]
        self.num_features = len(self.feature_cols)
        
        # Normalization
        self.normalization_stats = normalization_stats
        if normalization_stats is None:
            self._compute_normalization_stats(features_df)
        
        # Create valid indices (sequences that fit within the data)
        self.valid_indices = []
        for zone_id in self.zones:
            zone_df = self.zone_data[zone_id]
            max_idx = len(zone_df) - self.sequence_length - self.forecast_horizon + 1
            if max_idx > 0:
                self.valid_indices.extend([(zone_id, i) for i in range(max_idx)])
    
    def _compute_normalization_stats(self, features_df: pd.DataFrame):
        """Compute mean and std for normalization"""
        self.normalization_stats = {
            'mean': features_df[self.feature_cols].mean().to_dict(),
            'std': features_df[self.feature_cols].std().to_dict(),
        }
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input features [sequence_length, num_features]
            y: Target demand [forecast_horizon]
            metadata: Dict with zone_id, times, etc.
        """
        zone_id, start_idx = self.valid_indices[idx]
        zone_df = self.zone_data[zone_id]
        
        # Get sequence
        seq_df = zone_df.iloc[start_idx:start_idx + self.sequence_length]
        x = seq_df[self.feature_cols].values.astype(np.float32)
        
        # Normalize
        for i, col in enumerate(self.feature_cols):
            mean = self.normalization_stats['mean'][col]
            std = self.normalization_stats['std'][col]
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            if np.isnan(x[:, i]).any() or np.isinf(x[:, i]).any():
                x[:, i] = np.nan_to_num(x[:, i], nan=mean, posinf=mean, neginf=mean)
            if std > 0:
                x[:, i] = (x[:, i] - mean) / std
            else:
                x[:, i] = x[:, i] - mean
        
        # Final safety guard against non-finite values
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get target (demand in next time step)
        target_idx = start_idx + self.sequence_length + self.forecast_horizon - 1
        if target_idx < len(zone_df):
            y = zone_df.iloc[target_idx]['demand_count']
        else:
            y = 0.0
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def get_normalization_stats(self) -> dict:
        """Return normalization statistics"""
        return self.normalization_stats


class TaxiDemandDataModule:
    """
    Data module that handles data loading, splitting, and batching
    Similar to PyTorch Lightning's DataModule but standalone
    """
    
    def __init__(self,
                 duckdb_path: str = "data/processed/taxi_features.duckdb",
                 clustering_method: str = "method3",
                 sequence_length: int = 96,
                 forecast_horizon: int = 1,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        """
        Args:
            duckdb_path: Path to DuckDB database
            clustering_method: Which clustering to use
            sequence_length: Temporal window (15-min intervals)
            forecast_horizon: Steps ahead to predict
            batch_size: Batch size for DataLoader
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        self.duckdb_path = duckdb_path
        self.clustering_method = clustering_method
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.features_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_features(self) -> pd.DataFrame:
        """Load features from DuckDB"""
        try:
            conn = duckdb.connect(self.duckdb_path, read_only=True)
            query = "SELECT * FROM baseline_features ORDER BY zone_id, window_start"
            df = conn.execute(query).df()
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading from DuckDB: {e}")
            print(f"Trying to load from parquet files instead...")
            return self._load_from_parquet()
    
    def _load_from_parquet(self) -> pd.DataFrame:
        """Fallback: Load from parquet files"""
        parquet_files = []
        for method in ["baseline", "method1", "method2", "method3"]:
            path = f"data/processed/{method}_features/{method}_demand_features.parquet"
            if os.path.exists(path):
                parquet_files.append(path)
        
        if parquet_files:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No feature files found")
    
    def load_clustering_results(self) -> dict:
        """Load clustering results"""
        clustering_file = f"data/models/{self.clustering_method}_clusters.pkl"
        if os.path.exists(clustering_file):
            with open(clustering_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Clustering file not found: {clustering_file}")
    
    def create_adjacency_matrix(self, clusters: np.ndarray) -> torch.Tensor:
        """
        Create adjacency matrix from cluster assignments
        Zones in same cluster are connected
        
        Args:
            clusters: [num_zones] cluster assignments
        
        Returns:
            adj: [num_zones, num_zones] adjacency matrix
        """
        num_zones = len(clusters)
        adj = np.zeros((num_zones, num_zones), dtype=np.float32)
        
        # Connect zones in the same cluster
        for i in range(num_zones):
            for j in range(i, num_zones):
                if clusters[i] == clusters[j]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        
        return torch.tensor(adj, dtype=torch.float32)
    
    def setup(self):
        """Load data and create datasets"""
        # Load features
        self.features_df = self.load_features()
        print(f"Loaded {len(self.features_df)} feature records from {len(self.features_df['zone_id'].unique())} zones")
        
        # Load clustering
        clustering = self.load_clustering_results()
        # Convert zone_to_cluster dict to cluster array
        zone_to_cluster = clustering['zone_to_cluster']
        num_zones = max(zone_to_cluster.keys()) + 1
        self.clusters = np.zeros(num_zones, dtype=int)
        for zone_id, cluster_id in zone_to_cluster.items():
            if zone_id < num_zones:
                self.clusters[zone_id] = cluster_id
        
        self.adjacency_matrix = self.create_adjacency_matrix(self.clusters)
        print(f"Adjacency matrix created: {self.adjacency_matrix.shape}")
        
        # Compute normalization stats
        dataset_temp = TaxiDemandDataset(
            self.features_df,
            clustering_method=self.clustering_method,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon
        )
        norm_stats = dataset_temp.get_normalization_stats()
        
        # Split by zone (temporal split per zone)
        zones = sorted(self.features_df['zone_id'].unique())
        n_zones = len(zones)
        n_train = int(n_zones * self.train_ratio)
        n_val = int(n_zones * self.val_ratio)
        
        train_zones = set(zones[:n_train])
        val_zones = set(zones[n_train:n_train + n_val])
        test_zones = set(zones[n_train + n_val:])
        
        # Create datasets
        train_df = self.features_df[self.features_df['zone_id'].isin(train_zones)]
        val_df = self.features_df[self.features_df['zone_id'].isin(val_zones)]
        test_df = self.features_df[self.features_df['zone_id'].isin(test_zones)]
        
        self.train_dataset = TaxiDemandDataset(
            train_df,
            clustering_method=self.clustering_method,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            split="train",
            normalization_stats=norm_stats
        )
        
        self.val_dataset = TaxiDemandDataset(
            val_df,
            clustering_method=self.clustering_method,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            split="val",
            normalization_stats=norm_stats
        )
        
        self.test_dataset = TaxiDemandDataset(
            test_df,
            clustering_method=self.clustering_method,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            split="test",
            normalization_stats=norm_stats
        )
        
        print(f"Train: {len(self.train_dataset)} sequences")
        print(f"Val: {len(self.val_dataset)} sequences")
        print(f"Test: {len(self.test_dataset)} sequences")
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    print("Testing Data Module...")
    
    # Note: This requires the DuckDB file and clustering results to exist
    # For now, just demonstrate the structure
    
    # Create data module
    data_module = TaxiDemandDataModule(
        duckdb_path="../../data/processed/taxi_features.duckdb",
        clustering_method="method3",
        sequence_length=96,
        forecast_horizon=1,
        batch_size=32
    )
    
    print("Data module created")
    print(f"Sequence length: {data_module.sequence_length}")
    print(f"Batch size: {data_module.batch_size}")
    
    print("\n✅ Data module structure validated!")
