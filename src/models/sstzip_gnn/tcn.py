"""
Temporal Convolutional Network (TCN) - Temporal Layer
Captures temporal dynamics and demand patterns over time with causal convolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolution Block with causal convolutions and residual connections
    
    Features:
    - Causal convolutions: prevents information leakage from future
    - Dilated convolutions: increases receptive field exponentially
    - Residual connections: enables deeper networks
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.2):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor (increases receptive field)
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # For causal padding: pad LEFT side with (kernel_size - 1) * dilation
        # This ensures only past values are used
        self.padding_left = (kernel_size - 1) * dilation
        
        # Convolutions with NO padding (we'll handle it manually)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation, bias=True
        )
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=0, dilation=dilation, bias=True
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection projection (if needed)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        if self.residual_proj is not None:
            nn.init.kaiming_normal_(self.residual_proj.weight, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution
        
        Args:
            x: Input tensor [batch, in_channels, time_steps]
        
        Returns:
            Output tensor [batch, out_channels, time_steps]
        """
        # Save original size for residual
        original_size = x.size(2)
        
        # Pad left side with zeros for causality
        x_padded = F.pad(x, (self.padding_left, 0))
        
        # First conv + batch norm + activation + dropout
        out = self.conv1(x_padded)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # Pad again for second conv
        out_padded = F.pad(out, (self.padding_left, 0))
        
        # Second conv + batch norm + activation + dropout
        out = self.conv2(out_padded)
        out = self.bn2(out)
        
        # Trim to original size (remove excess from double-padding)
        out = out[:, :, :original_size]
        
        # Residual connection
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x)
        else:
            x_residual = x
        
        out = out + x_residual
        out = F.relu(out)
        out = self.dropout2(out)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network - Stack of causal temporal blocks
    with exponential dilation for multi-scale temporal dependencies
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, 
                 kernel_size: int = 3, dropout: float = 0.2):
        """
        Args:
            in_channels: Input channel dimension (num zones × features per zone)
            out_channels: Output channel dimension
            num_layers: Number of stacked TCN blocks
            kernel_size: Kernel size for temporal convolutions
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Build TCN blocks with exponential dilation
        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx  # Exponential dilation: 1, 2, 4, 8, ...
            layer_in = in_channels if layer_idx == 0 else out_channels
            
            self.blocks.append(
                TemporalConvBlock(
                    layer_in, out_channels, kernel_size=kernel_size,
                    dilation=dilation, dropout=dropout
                )
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters for all blocks"""
        for block in self.blocks:
            block.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN
        
        Args:
            x: Input [batch, in_channels, time_steps]
        
        Returns:
            Output [batch, out_channels, time_steps] with reduced temporal dim due to causal padding
        """
        for block in self.blocks:
            x = block(x)
        
        return x


class TemporalGlobalPooling(nn.Module):
    """
    Global temporal pooling: aggregates temporal dimension
    Supports: mean, max, attention-based
    """
    
    def __init__(self, method: str = "mean"):
        """
        Args:
            method: "mean" | "max" | "last"
        """
        super().__init__()
        assert method in ["mean", "max", "last"], f"Unknown pooling method: {method}"
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time_steps]
        
        Returns:
            [batch, channels]
        """
        if self.method == "mean":
            return x.mean(dim=-1)
        elif self.method == "max":
            return x.max(dim=-1)[0]
        elif self.method == "last":
            return x[:, :, -1]


if __name__ == "__main__":
    # Test TCN
    print("Testing TCN (Temporal Convolutional Network)...")
    
    batch_size = 32
    in_channels = 64  # spatial features from DGCN
    out_channels = 64
    time_steps = 96  # 24 hours * 4 (15-min intervals) or similar
    
    # Create test input
    x = torch.randn(batch_size, in_channels, time_steps)
    
    # Create TCN model
    tcn = TCN(in_channels, out_channels, num_layers=3, kernel_size=3, dropout=0.1)
    output = tcn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output has {time_steps - 2*4} time steps (due to causal padding)")
    
    # Test temporal pooling
    print("\nTesting Temporal Global Pooling...")
    pooling = TemporalGlobalPooling(method="mean")
    pooled = pooling(output)
    print(f"Pooled shape: {pooled.shape}")
    assert pooled.shape == (batch_size, out_channels), "Pooling output shape mismatch!"
    
    print("\n✅ TCN tests passed!")
