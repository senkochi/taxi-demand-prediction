"""
Zero-Inflated Poisson (ZIP) Head and Loss Function
Handles zero-inflated demand predictions with Bernoulli + Poisson components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZIPHead(nn.Module):
    """
    Zero-Inflated Poisson head for taxi demand prediction
    
    Outputs:
    - π (pi): Bernoulli probability of zero demand (excess zeros)
    - λ (lambda): Poisson intensity (expected count)
    
    Architecture:
    - Shared hidden layer for feature extraction
    - Separate output heads for π and λ
    - Sigmoid activation for π ∈ [0,1]
    - Softplus activation for λ > 0
    """
    
    def __init__(self, input_channels: int, hidden_channels: int = 128):
        """
        Args:
            input_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Bernoulli head (π): probability of structural zero
        # Higher π means more likely to have zero demand (not demand noise)
        self.pi_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Poisson head (λ): intensity/rate parameter
        # Higher λ means more demand expected
        self.lambda_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()  # Output > 0
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch, input_channels] or [batch*num_nodes, input_channels]
        
        Returns:
            pi: [batch, 1] - Bernoulli parameter (structural zero probability)
            lambda: [batch, 1] - Poisson intensity parameter
        """
        shared_features = self.shared(x)
        pi = self.pi_head(shared_features)
        lambda_param = self.lambda_head(shared_features)
        
        return pi, lambda_param


class ZIPLoss(nn.Module):
    """
    Zero-Inflated Poisson Loss Function
    
    Likelihood formulation:
    - For y=0: π + (1-π) * Poisson(0|λ) = π + (1-π) * exp(-λ)
    - For y>0: (1-π) * Poisson(y|λ)
    
    Loss = -log(likelihood) for numerical stability
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            eps: Small constant for numerical stability
        """
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, y_pred: tuple, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute ZIP loss
        
        Args:
            y_pred: (pi, lambda) tuple from ZIPHead
                - pi: [batch] or [batch, 1] - Bernoulli parameter
                - lambda: [batch] or [batch, 1] - Poisson intensity
            y_true: [batch] - Ground truth demand counts
        
        Returns:
            loss: Scalar or [batch] depending on reduction
        """
        pi, lambda_param = y_pred
        
        # Ensure correct shapes
        if pi.dim() > 1:
            pi = pi.squeeze(-1)
        if lambda_param.dim() > 1:
            lambda_param = lambda_param.squeeze(-1)
        y_true = y_true.float().squeeze(-1) if y_true.dim() > 1 else y_true.float()
        
        # Clamp for numerical stability
        pi = torch.clamp(pi, self.eps, 1 - self.eps)
        lambda_param = torch.clamp(lambda_param, self.eps, 1e6)
        
        # Create binary mask for zero vs non-zero observations
        is_zero = (y_true == 0).float()
        is_nonzero = 1.0 - is_zero
        
        # For y=0: log[π + (1-π) * exp(-λ)]
        prob_zero = pi + (1 - pi) * torch.exp(-lambda_param)
        loss_zero = -torch.log(prob_zero + self.eps)
        
        # For y>0: log[(1-π) * Poisson(y|λ)]
        # = log(1-π) + log(Poisson)
        # = log(1-π) + y*log(λ) - λ - log(y!)
        # NLL = -log[...] = -log(1-π) - y*log(λ) + λ + log(y!)
        log_lambda = torch.log(lambda_param + self.eps)
        log_factorial_y = torch.lgamma(y_true + 1)  # Compute log(y!)
        loss_nonzero = -torch.log(1 - pi + self.eps) - y_true * log_lambda + lambda_param + log_factorial_y
        
        # Combine losses
        loss = is_zero * loss_zero + is_nonzero * loss_nonzero
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ZIPMetrics(nn.Module):
    """
    Compute metrics for Zero-Inflated Poisson predictions
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def zero_inflation_score(y_pred: tuple, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute zero inflation score: ratio of predicted zeros to actual zeros
        
        Args:
            y_pred: (pi, lambda) tuple
            y_true: Ground truth counts
        
        Returns:
            score: Ratio of predicted to actual zeros (1.0 is perfect)
        """
        pi, lambda_param = y_pred
        
        # Predicted zero probability
        if pi.dim() > 1:
            pi = pi.squeeze(-1)
        if lambda_param.dim() > 1:
            lambda_param = lambda_param.squeeze(-1)
        
        predicted_zero_prob = pi + (1 - pi) * torch.exp(-lambda_param)
        predicted_zeros = (predicted_zero_prob > 0.5).float().sum()
        
        # Actual zeros
        actual_zeros = (y_true == 0).float().sum()
        
        # Avoid division by zero
        if actual_zeros == 0:
            return torch.tensor(0.0)
        
        return predicted_zeros / actual_zeros
    
    @staticmethod
    def mean_absolute_error(y_pred: tuple, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE using expected value: E[y] = (1-π) * λ
        """
        pi, lambda_param = y_pred
        
        if pi.dim() > 1:
            pi = pi.squeeze(-1)
        if lambda_param.dim() > 1:
            lambda_param = lambda_param.squeeze(-1)
        
        # Expected value
        y_pred_mean = (1 - pi) * lambda_param
        
        return torch.abs(y_pred_mean - y_true.float()).mean()


if __name__ == "__main__":
    # Test ZIP Head and Loss
    print("Testing Zero-Inflated Poisson Head...")
    
    batch_size = 32
    input_dim = 128
    
    # Create ZIP head
    zip_head = ZIPHead(input_channels=input_dim)
    
    # Test input
    x = torch.randn(batch_size, input_dim)
    pi, lambda_param = zip_head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"π shape: {pi.shape}, range: [{pi.min():.3f}, {pi.max():.3f}]")
    print(f"λ shape: {lambda_param.shape}, range: [{lambda_param.min():.3f}, {lambda_param.max():.3f}]")
    
    assert pi.shape == (batch_size, 1), "π shape mismatch!"
    assert lambda_param.shape == (batch_size, 1), "λ shape mismatch!"
    assert (pi >= 0).all() and (pi <= 1).all(), "π not in [0,1]!"
    assert (lambda_param > 0).all(), "λ not positive!"
    
    # Test loss
    print("\nTesting Zero-Inflated Poisson Loss...")
    
    zip_loss = ZIPLoss()
    
    # Create synthetic labels
    y_true = torch.randint(0, 50, (batch_size,)).float()
    
    loss = zip_loss((pi, lambda_param), y_true)
    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    
    # Test with batch loss
    loss_batch = zip_loss((pi, lambda_param), y_true)
    assert loss_batch.item() > 0, "Batch loss should be positive!"
    
    # Test metrics
    print("\nTesting Zero-Inflated Poisson Metrics...")
    
    metrics = ZIPMetrics()
    
    zi_score = metrics.zero_inflation_score((pi, lambda_param), y_true)
    print(f"Zero-Inflation Score: {zi_score.item():.4f}")
    
    mae = metrics.mean_absolute_error((pi, lambda_param), y_true)
    print(f"MAE: {mae.item():.4f}")
    
    print("\n✅ ZIP Head and Loss tests passed!")
