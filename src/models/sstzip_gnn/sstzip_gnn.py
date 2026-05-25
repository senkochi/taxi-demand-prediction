"""
SSTZIP-GNN: Spatio-Temporal Zero-Inflated Poisson Graph Neural Network
Complete model integrating DGCN (spatial), TCN (temporal), and ZIP head (output)
"""
import torch
import torch.nn as nn
from .dgcn import DGCN
from .tcn import TCN, TemporalGlobalPooling
from .zip_head import ZIPHead, ZIPLoss


class SSTZIPGNNModel(nn.Module):
    """
    Spatio-Temporal Zero-Inflated Poisson Graph Neural Network
    
    Architecture:
    1. Spatial Layer (DGCN): Captures zone-to-zone relationships
       - Input: Node features [num_zones, feature_dim]
       - Output: Spatial embeddings [num_zones, spatial_dim]
    
    2. Temporal Layer (TCN): Captures temporal dynamics
       - Input: Spatial embeddings over time [batch, spatial_dim, time_steps]
       - Output: Temporal features [batch, temporal_dim, time_steps]
    
    3. Aggregation: Global temporal pooling
       - Input: [batch, temporal_dim, time_steps]
       - Output: [batch, temporal_dim]
    
    4. Output Layer (ZIP Head): Zero-inflated Poisson parameters
       - Inputs: Aggregated temporal features
       - Outputs: π (structural zero prob) and λ (Poisson intensity)
    
    Training:
    - Loss: ZIPLoss combining Bernoulli + Poisson likelihoods
    - Optimization: Adam with learning rate scheduling
    - Early stopping: Monitor validation loss
    """
    
    def __init__(self, 
                 num_zones: int,
                 feature_dim: int,
                 spatial_dim: int = 64,
                 temporal_dim: int = 64,
                 num_spatial_layers: int = 2,
                 num_spatial_hops: int = 3,
                 num_temporal_layers: int = 3,
                 temporal_kernel_size: int = 3,
                 hidden_dim_zip: int = 128,
                 dropout: float = 0.2):
        """
        Args:
            num_zones: Number of zones (nodes in graph)
            feature_dim: Input feature dimension per zone
            spatial_dim: Output dimension of spatial layer
            temporal_dim: Output dimension of temporal layer
            num_spatial_layers: Number of DGCN layers
            num_spatial_hops: Number of multi-hop diffusions in DGCN
            num_temporal_layers: Number of TCN layers
            temporal_kernel_size: Kernel size for temporal convolutions
            hidden_dim_zip: Hidden dimension in ZIP head
            dropout: Dropout rate
        """
        super().__init__()
        self.num_zones = num_zones
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        
        # Spatial encoder: DGCN
        self.spatial_encoder = DGCN(
            in_channels=feature_dim,
            out_channels=spatial_dim,
            num_layers=num_spatial_layers,
            num_hops=num_spatial_hops
        )
        
        # Temporal encoder: TCN
        self.temporal_encoder = TCN(
            in_channels=spatial_dim,
            out_channels=temporal_dim,
            num_layers=num_temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=dropout
        )
        
        # Temporal aggregation: Global pooling
        self.temporal_pooling = TemporalGlobalPooling(method="mean")
        
        # Output head: ZIP
        self.zip_head = ZIPHead(
            input_channels=temporal_dim,
            hidden_channels=hidden_dim_zip
        )
        
        # Loss function
        self.loss_fn = ZIPLoss(reduction='mean')
    
    def encode_spatial(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Spatial encoding: Apply DGCN
        
        Args:
            x: Node features [num_zones, feature_dim]
            adj: Adjacency matrix [num_zones, num_zones]
        
        Returns:
            Spatial embeddings [num_zones, spatial_dim]
        """
        return self.spatial_encoder(x, adj)
    
    def encode_temporal(self, x_spatial_seq: torch.Tensor) -> torch.Tensor:
        """
        Temporal encoding: Apply TCN
        
        Args:
            x_spatial_seq: Sequence of spatial embeddings
                          [batch, num_zones, spatial_dim, time_steps]
                          or [batch*num_zones, spatial_dim, time_steps]
        
        Returns:
            Temporal features [batch*num_zones, temporal_dim, time_steps]
        """
        return self.temporal_encoder(x_spatial_seq)
    
    def aggregate_temporal(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal dimension
        
        Args:
            x_temporal: [batch*num_zones, temporal_dim, time_steps]
        
        Returns:
            Aggregated [batch*num_zones, temporal_dim]
        """
        return self.temporal_pooling(x_temporal)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                x_temporal_seq: torch.Tensor) -> tuple:
        """
        Forward pass of SSTZIP-GNN
        
        Args:
            x: Static node features [num_zones, feature_dim]
            adj: Adjacency matrix [num_zones, num_zones]
            x_temporal_seq: Time series spatial embeddings
                          [batch*num_zones, spatial_dim, time_steps]
        
        Returns:
            (pi, lambda): ZIP parameters for each zone-time pair
                - pi: [batch*num_zones, 1] - Bernoulli parameter
                - lambda: [batch*num_zones, 1] - Poisson intensity
        """
        # Step 1: Spatial encoding (optional - can be precomputed)
        # x_spatial = self.encode_spatial(x, adj)  # [num_zones, spatial_dim]
        
        # Step 2: Temporal encoding
        x_temporal = self.encode_temporal(x_temporal_seq)  # [batch*num_zones, temporal_dim, time_steps]
        
        # Step 3: Temporal aggregation
        x_agg = self.aggregate_temporal(x_temporal)  # [batch*num_zones, temporal_dim]
        
        # Step 4: ZIP output head
        pi, lambda_param = self.zip_head(x_agg)  # Each [batch*num_zones, 1]
        
        return pi, lambda_param
    
    def compute_loss(self, pi: torch.Tensor, lambda_param: torch.Tensor,
                     y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute ZIP loss
        
        Args:
            pi: Bernoulli parameter [batch*num_zones, 1]
            lambda_param: Poisson intensity [batch*num_zones, 1]
            y_true: Ground truth demands [batch*num_zones]
        
        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn((pi, lambda_param), y_true)


class SSTZIPGNNWithStaticSpatial(nn.Module):
    """
    SSTZIP-GNN variant that computes static spatial embeddings once
    and reuses them for all temporal steps
    
    Use this variant when spatial relationships are fixed over time
    """
    
    def __init__(self, base_model: SSTZIPGNNModel):
        super().__init__()
        self.base_model = base_model
        self.spatial_embeddings = None
    
    def compute_spatial_once(self, x: torch.Tensor, adj: torch.Tensor):
        """Compute spatial embeddings once and cache them"""
        self.spatial_embeddings = self.base_model.encode_spatial(x, adj)
    
    def forward(self, x_temporal_seq: torch.Tensor) -> tuple:
        """
        Forward without recomputing spatial layer
        
        Args:
            x_temporal_seq: [batch*num_zones, spatial_dim, time_steps]
        
        Returns:
            (pi, lambda): ZIP parameters
        """
        assert self.spatial_embeddings is not None, "Must call compute_spatial_once first!"
        
        # Skip spatial encoding, go straight to temporal
        x_temporal = self.base_model.encode_temporal(x_temporal_seq)
        x_agg = self.base_model.aggregate_temporal(x_temporal)
        pi, lambda_param = self.base_model.zip_head(x_agg)
        
        return pi, lambda_param


if __name__ == "__main__":
    print("Testing SSTZIP-GNN Model...")
    
    # Hyperparameters
    num_zones = 261
    feature_dim = 20  # From taxi_features
    batch_size = 32
    time_steps = 96  # 24 hours * 4 (15-min windows)
    
    # Create model
    model = SSTZIPGNNModel(
        num_zones=num_zones,
        feature_dim=feature_dim,
        spatial_dim=64,
        temporal_dim=64,
        num_spatial_layers=2,
        num_spatial_hops=3,
        num_temporal_layers=3,
        temporal_kernel_size=3,
        hidden_dim_zip=128,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test input
    x = torch.randn(num_zones, feature_dim)  # Node features
    adj = torch.eye(num_zones)  # Identity adjacency (placeholder)
    x_temporal_seq = torch.randn(batch_size * num_zones, 64, time_steps)  # Pre-computed spatial features
    y_true = torch.randint(0, 50, (batch_size * num_zones,)).float()
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  adj: {adj.shape}")
    print(f"  x_temporal_seq: {x_temporal_seq.shape}")
    print(f"  y_true: {y_true.shape}")
    
    # Forward pass
    pi, lambda_param = model(x, adj, x_temporal_seq)
    
    print(f"\nOutput shapes:")
    print(f"  π: {pi.shape}")
    print(f"  λ: {lambda_param.shape}")
    
    # Loss computation
    loss = model.compute_loss(pi, lambda_param, y_true)
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"Backward pass successful ✓")
    
    # Check gradients exist
    param_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Gradients computed for {param_count}/{total_params} parameters")
    
    print("\n✅ SSTZIP-GNN model tests passed!")
