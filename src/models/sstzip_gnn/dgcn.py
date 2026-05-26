"""
DGCN: Diffusion Graph Convolutional Network
Implements spatial graph convolution with multi-hop diffusion kernels
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class DGCNLayer(nn.Module):
    """
    Single DGCN layer with multi-hop diffusion kernels
    
    Architecture:
    - Applies diffusion kernels: (D^-1 * A)^k and (D^-1 * A^T)^k
    - Supports multi-hop neighborhoods (k = 0, 1, 2, ..., num_hops)
    - Learnable weights for each hop order
    
    Formula:
    H = Σ_k [ w_out_k * (D_out^-1 * A)^k + w_in_k * (D_in^-1 * A^T)^k ] * X
    where:
    - D_out = out-degree matrix
    - D_in = in-degree matrix
    - w_out_k, w_in_k = learnable hop weights
    - A = adjacency matrix
    - X = input node features
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 3):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_hops: Number of hop orders to use (0 to num_hops-1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Learnable weights for outgoing and incoming diffusion kernels
        # out_weights: [num_hops] weights for (D_out^-1 * A)^k
        # in_weights: [num_hops] weights for (D_in^-1 * A^T)^k
        self.out_weights = nn.Parameter(torch.Tensor(num_hops))
        self.in_weights = nn.Parameter(torch.Tensor(num_hops))
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters
        init.xavier_uniform_(self.weight)
        init.ones_(self.out_weights)
        init.ones_(self.in_weights)
        init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DGCN layer
        
        Args:
            x: Node features [num_nodes, in_channels]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Compute degree matrices
        out_degree = torch.sum(adj, dim=1)  # Out-degree: sum over columns
        in_degree = torch.sum(adj, dim=0)   # In-degree: sum over rows
        
        # Avoid division by zero
        out_degree_inv = torch.where(
            out_degree > 0,
            1.0 / out_degree,
            torch.zeros_like(out_degree)
        )
        in_degree_inv = torch.where(
            in_degree > 0,
            1.0 / in_degree,
            torch.zeros_like(in_degree)
        )
        
        # Create diagonal matrices
        D_out_inv = torch.diag(out_degree_inv)      # [num_nodes, num_nodes]
        D_in_inv = torch.diag(in_degree_inv)        # [num_nodes, num_nodes]
        
        # Initialize aggregated output
        out = torch.zeros_like(x)
        
        # Apply diffusion kernels for each hop
        # K^0 = I (identity)
        # K^k = (D^-1 * A)^k for k > 0
        
        # Initialize K_out and K_in as identity
        K_out = torch.eye(num_nodes, device=device)  # (D_out^-1 * A)^0 = I
        K_in = torch.eye(num_nodes, device=device)   # (D_in^-1 * A^T)^0 = I
        
        # Apply outgoing kernel
        K_out_hop = D_out_inv @ adj  # (D_out^-1 * A)^1
        
        # Apply incoming kernel
        K_in_hop = D_in_inv @ adj.t()  # (D_in^-1 * A^T)^1
        
        # Accumulate contributions from each hop
        for hop in range(self.num_hops):
            if hop == 0:
                # Zero-hop: identity
                out = out + self.out_weights[hop] * x
                out = out + self.in_weights[hop] * x
            else:
                # Multi-hop: apply diffusion powers
                out = out + self.out_weights[hop] * (K_out_hop @ x)
                out = out + self.in_weights[hop] * (K_in_hop @ x)
                
                # Update for next hop
                K_out_hop = K_out_hop @ D_out_inv @ adj
                K_in_hop = K_in_hop @ D_in_inv @ adj.t()
        
        # Linear transformation
        out = out @ self.weight + self.bias
        
        return out


class DGCN(nn.Module):
    """
    Multi-layer Diffusion Graph Convolutional Network
    
    Stacks multiple DGCN layers with optional activation functions
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_layers: int = 2, num_hops: int = 3, 
                 activation: str = 'relu', dropout: float = 0.0):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_layers: Number of DGCN layers to stack
            num_hops: Number of hop orders in each layer
            activation: Activation function ('relu', 'tanh', or 'linear')
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_hops = num_hops
        
        # Build layers
        self.layers = nn.ModuleList()
        hidden_dim = max(out_channels // 2, 32)
        
        for i in range(num_layers):
            if i == 0:
                layer_in = in_channels
                layer_out = hidden_dim if num_layers > 1 else out_channels
            elif i == num_layers - 1:
                layer_in = hidden_dim
                layer_out = out_channels
            else:
                layer_in = hidden_dim
                layer_out = hidden_dim
            
            self.layers.append(DGCNLayer(layer_in, layer_out, num_hops))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all DGCN layers
        
        Args:
            x: Node features [num_nodes, in_channels]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output embeddings [num_nodes, out_channels]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            
            # Apply activation and dropout to intermediate layers
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x
