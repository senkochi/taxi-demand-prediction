"""
Diffusion Graph Convolution Network (DGCN) - Spatial Layer
Captures zone relationships and spatial diffusion of demand
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNLayer(nn.Module):
    """
    Diffusion Graph Convolution with dual flow (inflow + outflow)
    
    Formula:
    H = Σ(k=0 to K) [w_out_k * (D_out^-1 * A)^k + w_in_k * (D_in^-1 * A^T)^k] * X
    
    Where:
    - A: Adjacency matrix (zone connectivity)
    - D_out, D_in: Out-degree and in-degree diagonal matrices
    - w_out_k, w_in_k: Learnable weights for k-hop diffusion
    - X: Node features (demand, fare, etc.)
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 3):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_hops: Number of diffusion hops (K in formula)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops
        
        # Learnable weights for each hop (outflow direction)
        self.w_out = nn.Parameter(torch.randn(num_hops, in_channels, out_channels))
        
        # Learnable weights for each hop (inflow direction)
        self.w_in = nn.Parameter(torch.randn(num_hops, in_channels, out_channels))
        
        # Linear transformation for initial features
        self.lin_init = nn.Linear(in_channels, out_channels, bias=True)
        
        # Bias terms
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Layer normalization
        self.ln = nn.LayerNorm(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.w_out)
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.lin_init.weight)
        nn.init.zeros_(self.bias)
    
    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^-1 * A
        
        Args:
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Normalized adjacency matrix
        """
        # Compute out-degree: D_out = sum over columns
        degree_out = adj.sum(dim=1, keepdim=True)
        degree_out = torch.clamp(degree_out, min=1e-7)  # Avoid division by zero
        
        # Normalize: D^-1 * A
        adj_normalized = adj / degree_out
        
        return adj_normalized
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DGCN
        
        Args:
            x: Node features [num_nodes, in_channels]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Normalize adjacency matrices
        adj_norm_out = self._normalize_adj(adj)  # D^-1 * A (outflow)
        adj_norm_in = self._normalize_adj(adj.t())  # D^-1 * A^T (inflow)
        
        # Initialize output with linear transformation
        out = self.lin_init(x)  # [num_nodes, out_channels]
        
        # Multi-hop diffusion
        x_out = x  # Current outflow features
        x_in = x   # Current inflow features
        
        for hop in range(self.num_hops):
            # Outflow diffusion: D^-1 * A applied hop times
            x_out = torch.mm(adj_norm_out, x_out)  # [num_nodes, in_channels]
            out_hop = torch.matmul(x_out, self.w_out[hop])  # [num_nodes, out_channels]
            out = out + out_hop
            
            # Inflow diffusion: D^-1 * A^T applied hop times
            x_in = torch.mm(adj_norm_in, x_in)  # [num_nodes, in_channels]
            in_hop = torch.matmul(x_in, self.w_in[hop])  # [num_nodes, out_channels]
            out = out + in_hop
        
        # Add bias
        out = out + self.bias
        
        # Layer normalization for stability
        out = self.ln(out)
        
        return out


class DGCN(nn.Module):
    """
    Stacked DGCN layers with residual connections
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, 
                 num_hops: int = 3, dropout: float = 0.1):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_layers: Number of stacked DGCN layers
            num_hops: Diffusion hops in each layer
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Build layers
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels
            self.layers.append(DGCNLayer(layer_in, out_channels, num_hops=num_hops))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Residual connection projection (if dimensions don't match)
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters for all layers"""
        for layer in self.layers:
            layer.reset_parameters()
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked DGCN layers
        
        Args:
            x: Node features [num_nodes, in_channels]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Features [num_nodes, out_channels]
        """
        # Residual connection: store input for later
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x)
        else:
            x_residual = x
        
        # Pass through DGCN layers
        x = self.layers[0](x, adj)
        x = self.dropouts[0](x)
        
        # Add residual connection
        x = x + x_residual
        x = F.relu(x)
        
        # Additional layers
        for i in range(1, self.num_layers):
            x_residual = x
            x = self.layers[i](x, adj)
            x = self.dropouts[i](x)
            x = x + x_residual
            x = F.relu(x)
        
        return x


if __name__ == "__main__":
    # Test DGCN layer
    print("Testing DGCN Layer...")
    
    num_nodes = 261
    in_channels = 6
    out_channels = 64
    
    # Create test inputs
    x = torch.randn(num_nodes, in_channels)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj > 0.5).float()  # Binary adjacency matrix
    
    # Create model
    dgcn_layer = DGCNLayer(in_channels, out_channels, num_hops=3)
    output = dgcn_layer(x, adj)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({num_nodes}, {out_channels})")
    assert output.shape == (num_nodes, out_channels), "Output shape mismatch!"
    
    # Test stacked DGCN
    print("\nTesting Stacked DGCN...")
    dgcn = DGCN(in_channels, out_channels, num_layers=2, num_hops=3)
    output = dgcn(x, adj)
    print(f"Stacked DGCN output shape: {output.shape}")
    assert output.shape == (num_nodes, out_channels), "Output shape mismatch!"
    
    print("\n✅ DGCN tests passed!")
