# 🧠 Module 2: Deep Learning Modeling - SSTZIP-GNN Implementation

**Owner:** Person B | **Duration:** Weeks 5-6 | **Deliverables:** Trained models, evaluation metrics, visualization

---

## 1. SSTZIP-GNN Architecture Overview

```
Input (Spatial-Temporal Graph)
    │
    ├─→ [Spatial Layer: DGCN]        Captures zone relationships
    │
    ├─→ [Temporal Layer: TCN]         Captures time-series dynamics
    │
    └─→ [Output Heads]                Dual prediction
         ├─ Head A (Sigmoid): π ∈ [0,1]  → Zero probability
         └─ Head B (Softplus): λ ∈ [0,∞) → Poisson mean
    │
    └─→ ZIP Loss Function
         = Binary Cross-Entropy(π) + Poisson Log-Likelihood(λ|y)
```

---

## 2. Implementation: DGCN Spatial Layer

### 2.1 Diffusion Graph Convolution

**Formula:** 
$$H = \sum_{k=0}^K \left( w_{k,1}(D_{out}^{-1}A)^k + w_{k,2}(D_{in}^{-1}A^T)^k \right) X$$

Where:
- $A$: Adjacency matrix (dynamic, based on clustering method)
- $D_{out}, D_{in}$: Out/in-degree matrices
- $w_{k,1}, w_{k,2}$: Learnable weights for outflow/inflow
- $X$: Node features (demand, fare, etc.)

```python
# src/models/dgcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class DGCNLayer(MessagePassing):
    """Diffusion Graph Convolution with dual flow"""
    
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Learnable weights for outflow & inflow
        self.w_out = torch.nn.Parameter(torch.randn(num_layers, out_channels))
        self.w_in = torch.nn.Parameter(torch.randn(num_layers, out_channels))
        
        # Linear transformation
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, adj_t):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            adj_t: Normalized adjacency matrix
        Returns:
            out: Updated features [num_nodes, out_channels]
        """
        out = self.lin(x)  # Linear transformation
        
        x_out = x  # For outflow
        x_in = x   # For inflow
        
        # Multi-hop diffusion
        for k in range(self.num_layers):
            # Outflow: D^-1 A
            x_out = torch.mm(adj_t, x_out)
            out = out + self.w_out[k] * x_out
            
            # Inflow: D^-1 A^T
            x_in = torch.mm(adj_t.t(), x_in)
            out = out + self.w_in[k] * x_in
        
        return out
```

### 2.2 Adjacency Matrix Construction

```python
# src/utils/graph_utils.py

def create_adjacency_matrix(cluster_labels, method="connectivity"):
    """Create graph adjacency matrix from cluster labels"""
    
    num_zones = len(cluster_labels)
    adj = torch.zeros(num_zones, num_zones)
    
    if method == "connectivity":
        # Connect zones in same cluster
        for i in range(num_zones):
            for j in range(num_zones):
                if cluster_labels[i] == cluster_labels[j]:
                    adj[i, j] = 1.0
    
    elif method == "od_flow":
        # Use actual OD-flow matrix (from Method 3)
        # adj[i, j] = flow(zone_i → zone_j)
        adj = load_od_matrix()  # Load from data
    
    # Normalize: D^-1 A (degree normalization)
    degree = adj.sum(dim=1, keepdim=True)
    degree[degree == 0] = 1  # Avoid division by zero
    adj_normalized = adj / degree
    
    return adj_normalized

# For each method, create appropriate adjacency matrix
adj_baseline = create_adjacency_matrix(baseline_clusters, "connectivity")
adj_method1 = create_adjacency_matrix(method1_clusters, "connectivity")
adj_method2 = create_adjacency_matrix(method2_clusters, "connectivity")
adj_method3 = create_adjacency_matrix(method3_clusters, "od_flow")
```

---

## 3. Implementation: TCN Temporal Layer

### 3.1 Temporal Convolution Network

```python
# src/models/tcn.py
import torch.nn as nn

class TemporalConvBlock(nn.Module):
    """Causal dilated 1D convolution with residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        
        # Causal padding: ensures no information leakage from future
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection (if channel mismatch)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time_steps]
        Returns:
            out: [batch, out_channels, time_steps]
        """
        # Remove causal padding (keep only valid future predictions)
        out = self.net(x)
        out = out[:, :, :-((len(self.net) - 1) * 0)]  # Adjust for causal padding
        
        # Residual connection
        return self.relu(out + self.residual(x))

class TCNEncoder(nn.Module):
    """Stack of TCN blocks"""
    
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, dropout=0.2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            layers.append(
                TemporalConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
```

---

## 4. Implementation: Zero-Inflated Poisson (ZIP) Head

### 4.1 ZIP Output Layer

```python
# src/models/zip_head.py

class ZIPHead(nn.Module):
    """Dual head for Zero-Inflated Poisson distribution"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Head A: Bernoulli (is this a structural zero?)
        self.mask_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_features),
            nn.Sigmoid()  # π ∈ [0, 1]
        )
        
        # Head B: Poisson (intensity if non-zero)
        self.poisson_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_features),
            nn.Softplus()  # λ ∈ [0, ∞)
        )
    
    def forward(self, x):
        """
        Args:
            x: Features [batch, in_features] or [batch, num_clusters]
        Returns:
            pi: Zero probability [batch, out_features]
            lambda: Poisson mean [batch, out_features]
        """
        pi = self.mask_head(x)
        lam = self.poisson_head(x)
        return pi, lam

class ZIPLoss(nn.Module):
    """Custom Zero-Inflated Poisson Loss"""
    
    def forward(self, y_true, pi, lam, eps=1e-7):
        """
        Args:
            y_true: Target demand [batch, num_clusters]
            pi: Zero probability [batch, num_clusters]
            lam: Poisson mean [batch, num_clusters]
            eps: Numerical stability factor
        Returns:
            loss: Scalar loss value
        """
        
        # Clamp to avoid numerical issues
        lam = torch.clamp(lam, min=eps)
        pi = torch.clamp(pi, min=eps, max=1-eps)
        
        # For y=0: -log[π + (1-π) * exp(-λ)]
        # For y>0: -log[(1-π) * Poisson(y;λ)]
        
        is_zero = (y_true == 0).float()
        is_nonzero = 1.0 - is_zero
        
        # Log-likelihood for structural zeros
        log_zero_prob = torch.log(pi + (1 - pi) * torch.exp(-lam) + eps)
        
        # Log-likelihood for Poisson counts (non-zero)
        # log[Poisson(y;λ)] = y*log(λ) - λ - log(y!)
        log_poisson = y_true * torch.log(lam + eps) - lam - torch.lgamma(y_true + 1)
        log_nonzero_prob = torch.log(1 - pi + eps) + log_poisson
        
        # Combine losses
        loss = -(is_zero * log_zero_prob + is_nonzero * log_nonzero_prob)
        
        return loss.mean()
```

---

## 5. Full SSTZIP-GNN Model

```python
# src/models/sstzip_gnn.py

class SSTZIP_GNN(nn.Module):
    """Spatiotemporal Zero-Inflated GNN"""
    
    def __init__(self, num_nodes, in_channels, hidden_dim=64, num_layers=3, 
                 output_dim=1, dropout=0.2):
        super().__init__()
        
        # Spatial layer: DGCN
        self.dgcn = DGCNLayer(in_channels, hidden_dim, num_layers=num_layers)
        
        # Temporal layer: TCN
        self.tcn = TCNEncoder(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        
        # Output heads: Zero-Inflated Poisson
        self.zip_head = ZIPHead(hidden_dim, output_dim)
        
        # Loss function
        self.zip_loss = ZIPLoss()
    
    def forward(self, x, adj_t, temporal_x=None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            adj_t: Normalized adjacency matrix [num_nodes, num_nodes]
            temporal_x: Temporal sequence [batch, num_nodes, time_steps, channels] (optional)
        Returns:
            pi, lam: Zero-probability and Poisson mean
        """
        
        # Step 1: Spatial convolution (DGCN)
        spatial_features = self.dgcn(x, adj_t)  # [num_nodes, hidden_dim]
        
        # Step 2: Temporal convolution (TCN)
        if temporal_x is not None:
            # Reshape for TCN: [batch*num_nodes, channels, time_steps]
            batch_size = temporal_x.size(0)
            temporal_x_reshaped = temporal_x.view(batch_size * self.num_nodes, -1, temporal_x.size(-1))
            temporal_features = self.tcn(temporal_x_reshaped)
            temporal_features = temporal_features.mean(dim=-1)  # Aggregate over time
        else:
            temporal_features = spatial_features
        
        # Step 3: Output heads (ZIP)
        pi, lam = self.zip_head(temporal_features)
        
        return pi, lam
    
    def compute_loss(self, y_true, pi, lam):
        """Compute ZIP loss"""
        return self.zip_loss(y_true, pi, lam)
```

---

## 6. Training Loop with PyTorch Lightning

```python
# src/training/trainer.py
import pytorch_lightning as pl

class SSTZIPGNNTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for SSTZIP-GNN"""
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = ZIPLoss()
    
    def forward(self, x, adj, temporal_x):
        return self.model(x, adj, temporal_x)
    
    def training_step(self, batch, batch_idx):
        x, adj, temporal_x, y = batch
        pi, lam = self(x, adj, temporal_x)
        
        loss = self.criterion(y, pi, lam)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, adj, temporal_x, y = batch
        pi, lam = self(x, adj, temporal_x)
        
        loss = self.criterion(y, pi, lam)
        self.log('val_loss', loss)
        
        # Log additional metrics
        mae = torch.abs(y - lam).mean()
        self.log('val_mae', mae)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['model']['training']['learning_rate']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Training script
def train_model(train_loader, val_loader, config, method_name):
    """Train SSTZIP-GNN for specific method"""
    
    model = SSTZIP_GNN(
        num_nodes=263,
        in_channels=5,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    trainer_module = SSTZIPGNNTrainer(model, config)
    
    trainer = pl.Trainer(
        max_epochs=config['model']['training']['epochs'],
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['model']['training']['early_stopping_patience']
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"data/models/{method_name}",
                monitor='val_loss',
                save_top_k=1
            )
        ],
        logger=pl.loggers.MLflowLogger(experiment_name="Taxi-Prediction")
    )
    
    trainer.fit(trainer_module, train_loader, val_loader)
    
    return trainer_module
```

---

## 7. Evaluation & Prediction

```python
# src/evaluation/metrics.py

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, adj, temporal_x, y = batch
            x, adj, temporal_x, y = x.to(device), adj.to(device), temporal_x.to(device), y.to(device)
            
            pi, lam = model(x, adj, temporal_x)
            
            # Predictions: use mean of ZIP distribution
            predictions.append(lam.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-7)))
    
    # Zero-Inflation Score
    zero_mask = targets == 0
    zero_acc = np.mean(predictions[zero_mask] < 0.5)
    nonzero_acc = np.mean(predictions[~zero_mask] >= 0.5)
    zip_score = (zero_acc + nonzero_acc) / 2
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Zero_Inflation_Score': zip_score
    }
```

---

## 8. Key Deliverables (ML Module)

### Week 5:
- [ ] DGCN layer implementation (dgcn.py)
- [ ] TCN encoder implementation (tcn.py)
- [ ] ZIP loss function (zip_head.py)
- [ ] Full SSTZIP-GNN model (sstzip_gnn.py)

### Week 6:
- [ ] Training loop (trainer.py with PyTorch Lightning)
- [ ] 4 trained models (method1, method2, method3, baseline)
- [ ] Model checkpoints saved

### Week 7:
- [ ] **ML/DL Module Report** (20-30 pages):
  - Model architecture design & justification
  - Loss function derivation & numerical stability
  - Training dynamics & convergence analysis
  - Hyperparameter tuning approach
  - Comparison of all 4 models

---

## 9. Prompt Templates for AI Assistance

> "I need to implement DGCN layer in PyTorch Geometric that handles directed graphs (inflow/outflow).
> The layer should compute H = Σ(w_k,1 * (D_out^-1 A)^k + w_k,2 * (D_in^-1 A^T)^k) * X.
> Make sure to handle numerical stability and gradient flow."

> "Write a custom PyTorch loss function for Zero-Inflated Poisson regression that:
> 1. Handles both structural zeros (π) and Poisson counts (λ)
> 2. Has numerical stability for large λ
> 3. Includes comments explaining the mathematical derivation"